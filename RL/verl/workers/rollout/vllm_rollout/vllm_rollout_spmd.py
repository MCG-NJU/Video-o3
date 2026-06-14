# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import re
import threading
import pickle
import ray
import json
import numpy as np
import functools
from PIL import Image
from typing import List, Dict
from copy import deepcopy
from tqdm import tqdm, trange
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
import zmq
from filelock import FileLock
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from .function_tools import prepare_grounding_inputs_multi_turn, crop_image, get_valid_mask
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from verl.models.transformers.rope_utils import get_rope_index
from verl.utils.dataset.task_prompt import ERROR_INFO_MULTI_TURN_PROMPT
from verl.utils.torch_functional import get_eos_mask, get_final_eos_mask, pad_2d_list_to_length, pad_sequence_to_length
from verl.protocol import DataProtoItem
import os
import io
import time

from verl.utils.random import save_random_states, set_random_states

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object"""
    return DataProto.from_dict(
        tensors=item.batch,  # TensorDict is already in correct format
        non_tensors=item.non_tensor_batch,  # Dict is already in correct format 
        meta_info=item.meta_info
    )

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor:
    assert all([t.ndim == 1 for t in tensor_list])
    max_len = max([t.size(0) for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(torch.cat([t,torch.tensor([pad_token_id] * (max_len-t.size(0)), device=t.device, dtype=t.dtype)],dim=0))
    return torch.stack(padded_tensor_list, dim=dim)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)

        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)

        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n
        # NOTE: We repeat 'multi_modal_data'
        rollout_num = self.config.n if do_sample else 1     # validation: n=1; train: n=self.config.n
        if 'multi_modal_data' in vllm_inputs[0]:
            repeated_multi_modal_data = [vllm_input['multi_modal_data'] for vllm_input in vllm_inputs for _ in range(rollout_num)]
            non_tensor_batch['multi_modal_data'] = np.array(repeated_multi_modal_data)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

def process_image(image: Any, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.width < 28 or image.height < 28:
        resize_factor = 28 / min(image.width, image.height)
        width, height = int(image.width * resize_factor + 1), int(image.height * resize_factor + 1)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.width / image.height >= 200:
        width, height = image.width, int(image.width / 190 + 1)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.height / image.width >= 200:
        width, height = int(image.height / 190 + 1), image.height
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

@ray.remote
def compute_reward(reward_fn, batch, non_tensor_batch, idx_to_remove_tensor, iteration):
    cur_batch = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    data_source_list = cur_batch.non_tensor_batch['data_source']
    print(f"ready to call reward function for {len(data_source_list)} samples")
    reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids = reward_fn(cur_batch)
    
    # global_invalid_uids.extend(invalid_uids)

    # result = (reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids, idx_to_remove_tensor)
    # queue.put(result)

    print(f'iteration: {iteration}')
    return reward_tensor.numpy(), acc_reward_tensor.numpy(), format_reward_tensor.numpy(), invalid_uids, idx_to_remove_tensor.numpy()
    # result = (reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids, idx_to_remove_tensor)
    # result_list.append(result)
    print("end reward compute")

class vLLMRollout_MultiTurn_ToolCall(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, processor, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.
        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            processor: the processor to process multi_modal_inputs
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        self.reward_fn = kwargs.pop("reward_fn", None)

        self.val_reward_fn = kwargs.pop("val_reward_fn", None)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.max_total_response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            limit_mm_per_prompt=self.config.get('limit_mm_per_prompt', {'image': 1}),
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        self.pad_token_id = tokenizer.pad_token_id
        self.max_generation_round = self.config.get('max_generation_round', 1)
        self.use_raw_image = self.config.use_raw_image
        # add tokenizer
        self.tokenizer = tokenizer
        # add processor
        self.processor = processor
        self.merge_length = self.processor.image_processor.merge_size ** 2
        self.multi_turn_prompt_type = self.config.multi_turn_prompt_type
        self.max_pixels = self.config.max_pixels
        self.min_pixels = self.config.min_pixels

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def load_image(self, image_path):
        max_retries = 5
        for i in range(max_retries):
            try:
                if os.path.exists(image_path):
                    original_image = Image.open(image_path).convert("RGB")
                else:
                    raise NotImplementedError
            except Exception as e:
                print(str(e))
                time.sleep(3)
                continue
            break
        return original_image

    def calculate_image_token_num(self, image):
        image_grid_thws = self.processor.image_processor([image], return_tensors='pt')['image_grid_thw']
        image_token_num = image_grid_thws[0].prod() // self.merge_length
        return image_token_num

    def save_traj_and_obs(self, save_directory, json_dict=None, original_image=None, resize_image=None, turn_idx=-1):
        os.makedirs(save_directory, exist_ok=True)
        if json_dict is not None:
            with open(os.path.join(save_directory, "traj.jsonl"), "a+") as f:
                f.write(json.dumps(json_dict)+"\n")
        if original_image is not None:
            if turn_idx == -1:
                file_name = "original_image.jpg"
            else:
                file_name = f"observation_{turn_idx}_original_image.jpg"
            original_image.save(os.path.join(save_directory, file_name))
        if resize_image is not None:
            if turn_idx == -1:
                file_name = "resized_image.jpg"
            else:
                file_name = f"observation_{turn_idx}_resized_image.jpg"
            resize_image.save(os.path.join(save_directory, file_name))

    def vllm_inference(self, idx_to_gen):
        response = [] # list of tuple, B'*R, valid(no-pad) response_ids with unequal length
        vllm_infer_bs = self.config.vllm_infer_batch_size
        n_batches = (len(idx_to_gen) - 1) // vllm_infer_bs + 1

        print(f"len(idx_to_gen): {len(idx_to_gen)}, vllm_infer_bs: {vllm_infer_bs}, n_batches: {n_batches}")

        for batch_idx in range(n_batches):
            start_idx = batch_idx * vllm_infer_bs
            end_idx = (batch_idx + 1) * vllm_infer_bs
            idx_to_gen_i = idx_to_gen[start_idx:end_idx]
            with self.update_sampling_params(n=1, stop='</grounding>', detokenize=True, include_stop_str_in_output=True):  # TODO: for validate, do_sample=False
                outputs = self.inference_engine.generate(
                    prompts=idx_to_gen_i,  # list of dict
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )
                import gc
                gc.collect()

            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    # HACK: filter > (voc_size+specidal_token_num) token_ids, 151664 for qwen model
                    _token_ids = output.outputs[sample_id].token_ids
                    filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]    # NOTE: <tool_call>: 151657, </tool_call>: 151658
                    if 151645 not in filtered_token_ids:
                        # replace the last token with <|im_end|> if no <|im_end|> in response,
                        # this is to ensure successful execution of get_final_eos_mask in multi-turn scenario
                        # filtered_token_ids[-1] = 151645
                        filtered_token_ids = filtered_token_ids + [151645,]
                    response.append(filtered_token_ids)
        return response

    def post_process(self, prefix_prompt_lengths, vllm_inputs, multi_turn_response_mask, idx, attention_mask, position_ids, input_prompt_generation_mask):

        # re-build response
        response = [] # B'*R, torch.Tensors with unequal lengths
        response_generation_mask = [] # B'*R, torch.Tensors with unequal lengths but align with 'response'
        for i_ in range(len(prefix_prompt_lengths)):
            # for each traj, we skip first-round prompt_ids/attention_mask
            first_round_prompt_length = prefix_prompt_lengths[i_]
            # Repeat <|image_pad|> token id for modeling_qwen2vl
            generation_response_ids = vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]
            processed_image_inputs = self.processor.image_processor(vllm_inputs[i_]['multi_modal_data']['image'][1:], return_tensors='pt')    # NOTE: The fisrt image is the original image, here we only take the resized image into account
            image_grid_thws = processed_image_inputs['image_grid_thw']
            all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0).tolist()
            
            assert len(generation_response_ids) == len(all_response_masks)
            # print(f"len(generation_response_ids): {len(generation_response_ids)}, len(all_response_masks): {len(all_response_masks)}")
            
            index, image_pad_token, magic_num = 0, 151655, 654321
            while image_pad_token in generation_response_ids:
                image_pad_token_pos = generation_response_ids.index(image_pad_token)
                image_pad_token_repeat_num = image_grid_thws[index].prod() // self.merge_length
                # update response_tensor_ids
                generation_response_ids[image_pad_token_pos : image_pad_token_pos + 1] = [magic_num] * image_pad_token_repeat_num
                # update all_response_masks
                all_response_masks[image_pad_token_pos : image_pad_token_pos + 1] = [0] * image_pad_token_repeat_num
                index += 1
            generation_response_ids = [image_pad_token if x == magic_num else x for x in generation_response_ids]
            all_response = torch.tensor(generation_response_ids, device=idx.device, dtype=idx.dtype)
            all_response_masks = torch.tensor(all_response_masks, dtype=torch.int64, device=attention_mask.device)
            response.append(all_response)
            response_generation_mask.append(all_response_masks) # at least we have single-turn conversation
            assert response[i_].shape[0] == response_generation_mask[i_].shape[0], f"Shape mismatched between resp_id and resp_mask! response[i_].shape[0]: {response[i_].shape[0]}, response_generation_mask[i_].shape[0]: {response_generation_mask[i_].shape[0]}"
        assert len(response) == len(response_generation_mask), "Length mismatched between response and response_generation_mask!"

        # attention_mask:       prompt           response
        #                 [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        response = pad_to_max_stack(response, self.pad_token_id, dim=0) # Tensor, (B'*R, padded_length), padded_length is the max length of samples in list
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0) # Tensor, (B'*R, padded_length)
        assert all([response.size(dim) == response_generation_mask.size(dim) for dim in range(response.ndim)])

        # cut or pad to max length
        # all should be (B*R, self.config.max_total_response_length)
        if response.shape[1] > self.config.max_total_response_length:
            response = response[:,:self.config.max_total_response_length]
            response_generation_mask = response_generation_mask[:,:self.config.max_total_response_length]
        elif response.shape[1] < self.config.max_total_response_length:
            response = pad_sequence_to_length(response, self.config.max_total_response_length, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(response_generation_mask, self.config.max_total_response_length, 0)
        
        # FIXME: function get_final_eos_mask cannot handle cases that when there is no <|im_end|> in the given response
        # response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, <|im_end|> is 151645
        # attention_mask: (...,0,0,0,1,1,1), response_attention_mask: (1,1,1,0,0,0,...)
        response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, |im_end| is 151645
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        seq = torch.cat([idx, response], dim=-1) # (B*R, max_prompt_length + max_total_response_length)

        return idx, response, seq, attention_mask, position_ids, multi_turn_response_mask

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

        # import random
        # import torch
        # import numpy as np
        # np.random.seed(42)
        # random.seed(42)
        # torch.manual_seed(42)
        
        print(f">>> vLLM Rollout Starts ...")

        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
        
        idx = prompts.batch['input_ids']  # (bs*tp, max_prompt_length), left padding with |end_of_text|
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']  # (bs*tp, max_prompt_length), left padding 0
        position_ids = prompts.batch['position_ids']  # (bs*tp, max_prompt_length), left padding 0

        # data_source_list = prompts.non_tensor_batch.pop('data_source')
        # ground_truth_list = prompts.non_tensor_batch.pop('ground_truth')

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']  # [151645, 151643] -> ｜im_end｜, |end_of_text|

        batch_size = idx.size(0)  # B'
        input_prompt_generation_mask = torch.zeros_like(idx, dtype=attention_mask.dtype, device=attention_mask.device) # (B'*R, max_prompt_length), all 0

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        
        is_validate = prompts.meta_info.get('validate', False)
        if is_validate:
            rng_states = save_random_states()
            set_random_states(rng_states=None)

        n = self.config.val_n if is_validate else self.config.n  # TODO: for validate, do_sample=False
        ##### Initialization #####
        vllm_inputs = [] # B*R, list of dict, into -> vllm.engine, each dict with keys: 'prompt_token_ids', 'multi_modal_data', the values are 'raw_prompt_ids' and [PIL.Image]
        multi_turn_response_mask = [] # B*R, list of list of Tensor, for distinguish 'USER tokens' & 'ASSISTANT tokens'
        prefix_prompt_lengths = [] # B*R, list of int, record first round prompt of all trajs
        # uids = []
        if self.use_raw_image:
            # resized_image_inputs = []
            observations_list = []
            raw_prompts = []
            image_size_used_list = []
            first_round_responses = []
            context_lengths = []
            doc_id_list = []
            uid_list = []
            ground_truth_list = []
            data_source_list = []

        save_dir = self.config.save_traj_dir
        save_traj = (save_dir is not None) and (self.config.save_traj == True)

        # We manually repeart trajs for rollout, since some trajs need multi-round self.inference_engine.generate() with `sampling_n=1`
        if 'multi_modal_data' in non_tensor_batch:
            for raw_prompt_ids, multi_modal_data, image_size_used, raw_prompt, doc_id, uid, ground_truth, data_source in zip(non_tensor_batch.pop('raw_prompt_ids'), non_tensor_batch.pop('multi_modal_data'), non_tensor_batch.pop('image_size_used'), non_tensor_batch.pop("raw_prompt"), non_tensor_batch.pop("index"), non_tensor_batch.pop('uid'), non_tensor_batch.pop('ground_truth'), non_tensor_batch.pop("data_source")):
                prefix_length = len(raw_prompt_ids)
                if self.use_raw_image:
                    assert len(multi_modal_data['image']) == 2, f"multi_modal_data should contain the downscaled image and a upscaled resized image, but got len(multi_modal_data['image']): {len(multi_modal_data['image'])}."
                    original_image_path = multi_modal_data['image'].pop(-1)
                    original_image = self.load_image(original_image_path)
                        
                for _ in range(n):
                    # NOTE: use deepcopy to seperate variables
                    if self.use_raw_image:
                        observations_list.append([original_image, ])
                        raw_prompts.append(raw_prompt)
                        assert len(image_size_used) == 1
                        image_size_used_list.append([image_size_used[0], ])
                    vllm_inputs.append(
                        {'prompt_token_ids': deepcopy(raw_prompt_ids), 'multi_modal_data': deepcopy(multi_modal_data)} # raw_prompt_ids: list
                    )
                    multi_turn_response_mask.append(
                        [torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)], # USER, Mark as 0
                    ) # [torch.Tensor(prefix_length,)]
                    prefix_prompt_lengths.append(
                        prefix_length
                    )
                    first_round_responses.append(
                        []
                    )
                    uid_list.append(uid)
                    ground_truth_list.append(ground_truth)
                    data_source_list.append(data_source)

                    image_token_num = self.calculate_image_token_num(multi_modal_data['image'][0])
                    context_length = prefix_length + image_token_num - 1
                    context_lengths.append(context_length)

                    doc_id_ = doc_id if n == 1 else f"{doc_id}_{_}"
                    doc_id_list.append(doc_id_)
                    if save_traj:
                        json_line = {
                            "doc_id": doc_id_, 
                            "initial_prompt": raw_prompt, 
                            'ground_truth': ground_truth
                        }
                        self.save_traj_and_obs(os.path.join(save_dir, doc_id_), json_line, original_image, multi_modal_data['image'][0])
        
        ##### Loop Setting #####
        to_generate = list(range(batch_size * n))  # B*R, all trajs' index
        worker_trajs_count = len(to_generate)
        max_image_num = self.config.get('limit_mm_per_prompt', {'image': 1})['image']
        max_iterations = self.max_generation_round
        # Add pbar for better monitoring

        # iter_to_rewards = {}
        reward_processes = []
        # indices_list = []
        exceed_indices = []
        with tqdm(total=worker_trajs_count, desc="Worker Rollout Progress", unit="task") as pbar:
            current_iteration = 0
            while current_iteration < max_iterations and len(to_generate) > 0:
                # Prepare prompts to generation
                idx_to_gen = [] # list of vllm_inputs, at first the length is B'*R
                for i in to_generate:
                    idx_to_gen.append(vllm_inputs[i])
                print(f"[Round #{current_iteration} Rollout START] For THIS round, We hava {len(idx_to_gen)} trajs to complete ...")

                # users can customize different sampling_params at different run
                response = self.vllm_inference(idx_to_gen)
                
                # attach model responses to vllm_inputs
                assert len(to_generate) == len(response)

                idx_to_remove = []
                id_tool_query_mapping = {}
                for i_gen, response_ in zip(to_generate, response):
                    # update conversation
                    response_ = list(response_)
                    vllm_inputs[i_gen]['prompt_token_ids'] += response_
                    multi_turn_response_mask[i_gen].append(torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device)) # ASSISTANT, Mark as 1
                    context_lengths[i_gen] += len(response_)

                    # [TOOL CALL TRIGGER] We check model's last turn response, if not any tool called, then remove this traj from to_generate
                    decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)
                    first_round_responses[i_gen].append(decoded_resp_)

                    pattern = re.compile(r'<grounding>(.*?)</grounding>', re.DOTALL)
                    tool_call_contents = pattern.findall(decoded_resp_)

                    if save_traj:
                        json_line = {"turn_idx": current_iteration, "text_output": decoded_resp_}
                        self.save_traj_and_obs(os.path.join(save_dir, doc_id_list[i_gen]), json_line)

                    if len(tool_call_contents) > 0:
                        if (len(vllm_inputs[i_gen]['multi_modal_data']['image']) >= max_image_num) or (current_iteration == max_iterations - 1):   # If the current traj has already reached max_image_num, but still try to call tool, we should remove this traj.
                            idx_to_remove.append(i_gen)
                            exceed_indices.append(i_gen)
                            # print(f"Traj {i_gen} exceeds maximum function tool call num {len(vllm_inputs[i_gen]['multi_modal_data']['image'])}")
                            continue

                        if context_lengths[i_gen] >= self.config.max_total_response_length - 2000:
                            idx_to_remove.append(i_gen)
                            exceed_indices.append(i_gen)
                            continue

                        assert str(i_gen) not in id_tool_query_mapping.keys()
                        error_info = None
                        try:
                            pattern = ".*<grounding>{\"bbox_2d\": (.*),.*\"source\": [\',\"](.*)[\',\"]}</grounding>"
                            match = re.match(pattern, decoded_resp_, re.DOTALL)
                            bbox, source = match.group(1), match.group(2)
                            json_objects = [{"bbox_2d": eval(bbox), "source": source}]
                            tool_type, args = prepare_grounding_inputs_multi_turn(json_objects, observations_list[i_gen], image_size_used_list[i_gen], use_relative_coordinates=self.config.use_relative_coordinates)
                        except Exception as e:
                            print(str(e))
                            error_info = str(e)
                            tool_type = None
                            args = None
                        id_tool_query_mapping[str(i_gen)] = {
                            "tool_type": tool_type,
                            "args": args,
                            "error_info": error_info,
                        }
                    # Direct Answer
                    else:
                        # remove this traj from to_generate
                        idx_to_remove.append(i_gen)
                        # NOTE: to_generate.remove(i_gen) # DO NOT .remove() in for loop

                if to_generate and id_tool_query_mapping:   # Make sure to PRINT when to_generate and id_tool_query_mapping is not None
                    # print(f"[Round #{current_iteration}] Example Generation: to_generate[0]: {to_generate[0]} | response[0]: {self.tokenizer.decode(response[0], skip_special_tokens=True)}")
                    print(f"[Round #{current_iteration} Rollout Tool Call Trigger] For THIS round, ids {next(iter(id_tool_query_mapping))} need to apply function tool using: {id_tool_query_mapping[next(iter(id_tool_query_mapping))]} ...")
                else:
                    print(f"[Round #{current_iteration} Rollout Tool Call Trigger] No ids need to apply function tool for this round.")
            
                prefix_prompt_lengths_cur = []
                vllm_inputs_cur = []
                multi_turn_response_mask_cur = []
                # idx_cur = []
                # attention_mask_cur = []
                # position_ids_cur = []
                # input_prompt_generation_mask_cur = []
                data_source_list_cur = []
                ground_truth_list_cur = []
                raw_prompt_list_cur = []
                uid_list_cur = []
                
                # print(f"idx_to_remove: {idx_to_remove}, to_generate: {to_generate}, idx.shape: {idx.shape}, attention_mask.shape: {attention_mask.shape}, position_ids.shape: {position_ids.shape}, len(data_source_list): {len(data_source_list)}, n: {n}")

                # update 'to_generate'
                for x in idx_to_remove:
                    to_generate.remove(x)

                    prefix_prompt_lengths_cur.append(prefix_prompt_lengths[x])
                    vllm_inputs_cur.append(vllm_inputs[x])
                    multi_turn_response_mask_cur.append(multi_turn_response_mask[x])
                    
                    # idx_cur.append(idx[x // n])
                    # attention_mask_cur.append(attention_mask[x // n])
                    # position_ids_cur.append(position_ids[x // n])
                    # input_prompt_generation_mask_cur.append(input_prompt_generation_mask[x // n])
                    
                    raw_prompt_list_cur.append(raw_prompts[x])
                    data_source_list_cur.append(data_source_list[x])
                    ground_truth_list_cur.append(ground_truth_list[x])
                    uid_list_cur.append(uid_list[x])
                
                if len(idx_to_remove) > 0:

                    idx_to_remove_tensor = torch.tensor(idx_to_remove, dtype=torch.long, device=idx.device)
                    idx_cur = idx[idx_to_remove_tensor // n]
                    attention_mask_cur = attention_mask[idx_to_remove_tensor // n]
                    position_ids_cur = position_ids[idx_to_remove_tensor // n]
                    input_prompt_generation_mask_cur = input_prompt_generation_mask[idx_to_remove_tensor // n]

                    # idx_cur = torch.cat(idx_cur, dim=0)
                    # attention_mask_cur = torch.cat(attention_mask_cur, dim=0)
                    # position_ids_cur = torch.cat(position_ids_cur, dim=0)
                    # input_prompt_generation_mask_cur = torch.cat(input_prompt_generation_mask_cur, dim=0)

                    batch_size_cur = len(idx_to_remove)
                    idx_cur, response_cur, seq_cur, attention_mask_cur, _, multi_turn_response_mask_cur = self.post_process(
                        prefix_prompt_lengths_cur, 
                        vllm_inputs_cur, 
                        multi_turn_response_mask_cur, 
                        idx_cur,
                        attention_mask_cur,
                        position_ids_cur,
                        input_prompt_generation_mask_cur
                    )

                    # print(f"idx_cur.shape: {idx_cur.shape}, response_cur.shape: {response_cur.shape}, seq_cur.shape: {seq_cur.shape}, attention_mask_cur.shape: {attention_mask_cur.shape}, multi_turn_response_mask_cur.shape: {multi_turn_response_mask_cur.shape}, batch_size_cur: {batch_size_cur}")

                    batch_cur = TensorDict(
                        {
                            'prompts': idx_cur.contiguous(),
                            'responses': response_cur.contiguous(),
                            'input_ids': seq_cur.contiguous(),  # here input_ids become the whole sentences
                            # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                            'attention_mask': attention_mask_cur.contiguous(),
                            'multi_turn_response_mask': multi_turn_response_mask_cur.contiguous()
                        },
                        batch_size=batch_size_cur
                    )

                    data_source_cur = np.array(data_source_list_cur, dtype=object)
                    ground_truth_cur = np.array(ground_truth_list_cur, dtype=object)
                    raw_prompt_cur = np.array(raw_prompt_list_cur, dtype=object)
                    uid_list_cur = np.array(uid_list_cur, dtype=object)
                    non_tensor_batch_cur = {
                        'data_source': data_source_cur,
                        'ground_truth': ground_truth_cur,
                        'raw_prompt': raw_prompt_cur,
                        'uid': uid_list_cur,
                    }

                    if prompts.meta_info.get('validate', False):
                        future = compute_reward.remote(self.val_reward_fn, batch_cur.to('cpu'), non_tensor_batch_cur, idx_to_remove_tensor.to('cpu'), current_iteration)
                    else:
                        future = compute_reward.remote(self.reward_fn, batch_cur.to('cpu'), non_tensor_batch_cur, idx_to_remove_tensor.to('cpu'), current_iteration)
                    reward_processes.append(future)

                print(f"[Round #{current_iteration} Rollout END] For NEXT round, We hava {len(to_generate)} trajs to complete ...")

                # [Call Function Tool]
                function_tool_results = []
                for i_todo in to_generate:
                    assert str(i_todo) in id_tool_query_mapping.keys()
                    image_to_resize = vllm_inputs[i_todo]['multi_modal_data']['image'][-1]
                    tool_type = id_tool_query_mapping[str(i_todo)]['tool_type']
                    error_info = id_tool_query_mapping[str(i_todo)]["error_info"]
                    args = id_tool_query_mapping[str(i_todo)]["args"]
                    if error_info is not None:
                        tool_outputs = f"ERROR occurs during grounding. Error Information: {error_info}.\n"
                    else:
                        if self.use_raw_image and tool_type == 'grounding':
                            tool_outputs = crop_image(args[0], args[1], image_size_used_list[i_todo], resize=1)
                            observations_list[i_todo].append(tool_outputs)
                        elif self.use_raw_image and tool_type == 'resize':
                            tool_outputs = resized_image_inputs[i_todo]
                        elif tool_type == 'resize':
                            tool_outputs = resize_image(image_to_resize)   # Here we can add a save_path to visualize the annotated images
                    function_tool_results.append(tool_outputs)

                # [Process Tool Call Results]
                to_generate_ = to_generate.copy() # make a copy since we will be modifying to_generate
                assert len(to_generate_) == len(function_tool_results)

                for i_gen_, tool_call_result_ in zip(to_generate_, function_tool_results):

                    if isinstance(tool_call_result_, Image.Image):
                        # Construct Next Round Prompt
                        tool_call_prompt_message = "<|im_start|>user\n" + TOOL_CALL_CROP_PROMPT_MAP[self.multi_turn_prompt_type].format(action_turn=current_iteration, observation_turn=current_iteration+1) + "<|im_end|>\n<|im_start|>assistant\n"
                        
                        next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
                        # update conversation
                        vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'

                        resized_image = process_image(tool_call_result_, max_pixels=self.max_pixels, min_pixels=self.min_pixels)

                        if save_traj:
                            self.save_traj_and_obs(os.path.join(save_dir, doc_id_list[i_gen_]), json_dict=None, original_image=tool_call_result_, resize_image=resized_image, turn_idx=current_iteration+1)

                        image_size_used_list[i_gen_].append(resized_image.size)

                        vllm_inputs[i_gen_]['multi_modal_data']['image'].append(resized_image)
                        multi_turn_response_mask[i_gen_].append(torch.zeros(len(next_turn_prompt_ids), dtype=attention_mask.dtype, device=attention_mask.device)) # USER, Mark as 0

                        image_token_num = self.calculate_image_token_num(resized_image)
                        context_lengths[i_gen_] += len(next_turn_prompt_ids) + image_token_num - 1

                    else:
                        tool_call_prompt_message = "<|im_start|>user\n" + tool_call_result_ + ERROR_INFO_MULTI_TURN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
                        next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
                        vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'
                        multi_turn_response_mask[i_gen_].append(torch.zeros(len(next_turn_prompt_ids), dtype=attention_mask.dtype, device=attention_mask.device)) # USER, Mark as 0

                        context_lengths[i_gen_] += len(next_turn_prompt_ids)

                        if save_traj:
                            json_line = {
                                "error_info": tool_call_prompt_message
                            }
                            self.save_traj_and_obs(os.path.join(save_dir, doc_id_list[i_gen_]), json_dict=json_line)
                
                # update pbar
                pbar.update(worker_trajs_count - len(to_generate))
                # update iteration count
                current_iteration += 1

            assert len(to_generate) == 0, f"to_generate: {to_generate}"

        # All for 1st USER prompt
        if n > 1 and do_sample:
            idx = _repeat_interleave(idx, n) # (B, max_prompt_length) -> (B*R, max_prompt_length)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)
            # we also need to repeat 'input_prompt_generation_mask'
            input_prompt_generation_mask = _repeat_interleave(input_prompt_generation_mask, n) # (B, max_prompt_length) -> (B*R, max_prompt_length), all 0

        batch_size = batch_size * n

        idx, response, seq, attention_mask, position_ids, multi_turn_response_mask = self.post_process(prefix_prompt_lengths, vllm_inputs, multi_turn_response_mask, idx, attention_mask, position_ids, input_prompt_generation_mask)

        # NOTE: We repeat 'multi_modal_data'
        if 'multi_modal_data' in vllm_inputs[0]:
            repeated_multi_modal_data = [vllm_input['multi_modal_data'] for vllm_input in vllm_inputs]  # 这里不再对vllm_input重复n次,因为vllm_inputs是按照n=1构建的,已经重复了n次
            non_tensor_batch['multi_modal_data'] = np.array(repeated_multi_modal_data)

            repeated_raw_prompt = raw_prompts
            non_tensor_batch['raw_prompt'] = np.array(repeated_raw_prompt, dtype=object)

            non_tensor_batch['uid'] = np.array(uid_list, dtype=object)
            non_tensor_batch['data_source'] = np.array(data_source_list, dtype=object)
            non_tensor_batch['ground_truth'] = np.array(ground_truth_list, dtype=object)

        del vllm_inputs

        response_length = response.size(1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids_list = []
            for prompt_with_response, attn_mask, multi_modal_data in zip(seq, attention_mask, repeated_multi_modal_data):
                image_inputs = self.processor.image_processor(multi_modal_data['image'], return_tensors='pt')
                image_grid_thw = image_inputs['image_grid_thw']
                pos_ids = get_rope_index(
                    self.processor,
                    input_ids=prompt_with_response,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attn_mask,
                )
                position_ids_list.append(pos_ids)
            position_ids = torch.stack(position_ids_list, dim=0)
        else:
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            # TODO(sgm): fix position_ids on right_pad
            # prompt: left pad + response: right pad
            # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
            response_position_ids = position_ids[:, -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        reward_tensor = torch.zeros_like(response, dtype=torch.float32)
        acc_reward_tensor = reward_tensor.clone()
        format_reward_tensor = reward_tensor.clone()
        invalid_uids = []

        # print(f"queue.empty(): {queue.empty()}")
        # print("iter_to_rewards: ", iter_to_rewards)

        cnt = 0
        for iteration, future in enumerate(reward_processes):
        # for rewards in result_list:
            rewards = ray.get(future)
            reward_tensor_cur, acc_reward_tensor_cur, format_reward_tensor_cur, invalid_uids_cur, indices = rewards

            reward_tensor_cur = torch.from_numpy(reward_tensor_cur).to(device=reward_tensor.device)
            acc_reward_tensor_cur = torch.from_numpy(acc_reward_tensor_cur).to(device=reward_tensor.device)
            format_reward_tensor_cur = torch.from_numpy(format_reward_tensor_cur).to(device=reward_tensor.device)
            indices = torch.from_numpy(indices).to(device=reward_tensor.device)

            cnt += indices.shape[0]
            
            # print("reward_tensor_cur.sum(-1): ", reward_tensor_cur.sum(-1))
            # print("acc_reward_tensor_cur.sum(-1): ", acc_reward_tensor_cur.sum(-1))
            # print("format_reward_tensor_cur.sum(-1): ", format_reward_tensor_cur.sum(-1))
            # print("indices: ", indices)
            # print("indices.dtype: ", indices.dtype)
            
            assert reward_tensor[indices].sum().item() == 0, f"indices: {indices}"

            reward_tensor[indices] = reward_tensor_cur
            acc_reward_tensor[indices] = acc_reward_tensor_cur
            format_reward_tensor[indices] = format_reward_tensor_cur
            invalid_uids.extend(invalid_uids_cur)

        assert cnt == reward_tensor.shape[0], f"cnt: {cnt}, reward_tensor.shape: {reward_tensor.shape}"

        # for indices, p in zip(indices_list, reward_processes):
        #     reward_tensor_cur, acc_reward_tensor_cur, format_reward_tensor_cur, invalid_uids_cur = p.join()
        #     reward_tensor[indices] = reward_tensor_cur
        #     acc_reward_tensor[indices] = acc_reward_tensor_cur
        #     format_reward_tensor[indices] = format_reward_tensor_cur
        #     invalid_uids.extend(invalid_uids_cur)

        exceed_indices = torch.tensor(exceed_indices, dtype=torch.int64, device=idx.device)
        exceed_mask = torch.zeros((batch_size,), dtype=torch.bool, device=idx.device)
        exceed_mask[exceed_indices] = True

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # NOTE: .contiguous() for broadcast
        batch = TensorDict(
            {
                'prompts': idx.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous(),
                'reward_tensor': reward_tensor.contiguous(),
                'exceed_mask': exceed_mask.contiguous(),
                # 'acc_reward_tensor': acc_reward_tensor.contiguous(),
                # 'format_reward_tensor': format_reward_tensor.contiguous()
            },
            batch_size=batch_size
        )
        # non_tensor_batch['invalid_uids'] = np.array(invalid_uids, dtype=object)

        if is_validate:
            set_random_states(rng_states=rng_states)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        print(f">>> vLLM Rollout Ends ...")

        if not prompts.meta_info.get('validate', False):
            
            valid_mask, invalid_mask, wrong_and_use_tool_uids, wrong_and_direct_answer_uids, statistics_dict = get_valid_mask(non_tensor_batch['uid'], non_tensor_batch['multi_modal_data'], prompts.meta_info, reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids)
            statistics_dict['wrong_and_use_tool_uids'] = wrong_and_use_tool_uids
            statistics_dict['wrong_and_direct_answer_uids'] = wrong_and_direct_answer_uids
            non_tensor_batch['multi_modal_data'][invalid_mask] = None
            non_tensor_batch['valid_mask'] = np.array(valid_mask, dtype=object)  

            
        data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)   
        data.non_tensor_batch['acc_scores'] = np.array(acc_reward_tensor.cpu().sum(-1), dtype=object)
        data.non_tensor_batch['format_scores'] = np.array(format_reward_tensor.cpu().sum(-1), dtype=object)
        if not prompts.meta_info.get('validate', False):
            # add empty list with num of "self.config.n - 1" 
            # to make the non_tensor_batch data size equals to rollout.n * batch_size for the communication
            statistics_dict["invalid_num"] = len(invalid_uids)
            statistics_dict_for_comm = [statistics_dict]
            for _ in range(batch_size - 1):
                statistics_dict_for_comm.append([])
            data.non_tensor_batch['statistics_dict'] = np.array(statistics_dict_for_comm, dtype=object)

        return data

class vLLMRollout_MultiTurn_ResizeImage(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, processor, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.
        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            processor: the processor to process multi_modal_inputs
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.max_total_response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            limit_mm_per_prompt=self.config.get('limit_mm_per_prompt', {'image': 1}),
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        self.pad_token_id = tokenizer.pad_token_id
        self.max_generation_round = self.config.get('max_generation_round', 1)
        # add tokenizer
        self.tokenizer = tokenizer
        # add processor
        self.processor = processor
        self.merge_length = self.processor.image_processor.merge_size ** 2

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

        print(f">>> vLLM Rollout Starts ...")

        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
        
        idx = prompts.batch['input_ids']  # (bs*tp, max_prompt_length), left padding with |end_of_text|
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']  # (bs*tp, max_prompt_length), left padding 0
        position_ids = prompts.batch['position_ids']  # (bs*tp, max_prompt_length), left padding 0

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']  # [151645, 151643] -> ｜im_end｜, |end_of_text|

        batch_size = idx.size(0)  # B'
        input_prompt_generation_mask = torch.zeros_like(idx, dtype=attention_mask.dtype, device=attention_mask.device) # (B'*R, max_prompt_length), all 0

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        
        n = 1 if prompts.meta_info.get('validate', False) else self.config.n  # TODO: for validate, do_sample=False

        ##### Initialization #####
        vllm_inputs = [] # B*R, list of dict, into -> vllm.engine, each dict with keys: 'prompt_token_ids', 'multi_modal_data', the values are 'raw_prompt_ids' and [PIL.Image]
        multi_turn_response_mask = [] # B*R, list of list of Tensor, for distinguish 'USER tokens' & 'ASSISTANT tokens'
        prefix_prompt_lengths = [] # B*R, list of int, record first round prompt of all trajs

        # We manually repeart trajs for rollout, since some trajs need multi-round self.inference_engine.generate() with `sampling_n=1`
        if 'multi_modal_data' in non_tensor_batch:
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'), non_tensor_batch.pop('multi_modal_data')):
                prefix_length = len(raw_prompt_ids)
                for _ in range(n):
                    # NOTE: use deepcopy to seperate variables
                    vllm_inputs.append(
                        {'prompt_token_ids': deepcopy(raw_prompt_ids), 'multi_modal_data': deepcopy(multi_modal_data)} # raw_prompt_ids: list
                    )
                    multi_turn_response_mask.append(
                        [torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)], # USER, Mark as 0
                    ) # [torch.Tensor(prefix_length,)]
                    prefix_prompt_lengths.append(
                        prefix_length
                    )
        ##### Loop Setting #####
        to_generate = list(range(batch_size * n))  # B*R, all trajs' index
        worker_trajs_count = len(to_generate)
        max_image_num = self.config.get('limit_mm_per_prompt', {'image': 1})['image']
        max_iterations = self.max_generation_round
        # Add pbar for better monitoring
        with tqdm(total=worker_trajs_count, desc="Worker Rollout Progress", unit="task") as pbar:
            current_iteration = 0
            while current_iteration < max_iterations and len(to_generate) > 0: 
                # Prepare prompts to generation
                idx_to_gen = [] # list of vllm_inputs, at first the length is B'*R
                for i in to_generate:
                    idx_to_gen.append(vllm_inputs[i])
                print(f"[Round #{current_iteration} Rollout START] For THIS round, We hava {len(idx_to_gen)} trajs to complete ...")
                # users can customize different sampling_params at different run
                with self.update_sampling_params(n=1):  # TODO: for validate, do_sample=False
                    outputs = self.inference_engine.generate(
                        prompts=idx_to_gen,  # list of dict
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )
                response = [] # list of tuple, B'*R, valid(no-pad) response_ids with unequal length
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        # HACK: filter > (voc_size+specidal_token_num) token_ids, 151664 for qwen model
                        _token_ids = output.outputs[sample_id].token_ids
                        filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]    # NOTE: <tool_call>: 151657, </tool_call>: 151658
                        if 151645 not in filtered_token_ids:
                            # replace the last token with <|im_end|> if no <|im_end|> in response,
                            # this is to ensure successful execution of get_final_eos_mask in multi-turn scenario
                            filtered_token_ids[-1] = 151645
                        response.append(filtered_token_ids)

                # attach model responses to vllm_inputs
                assert len(to_generate) == len(response)

                idx_to_remove = []
                id_tool_query_mapping = {}
                for i_gen, response_ in zip(to_generate, response):
                    # update conversation
                    response_ = list(response_)
                    vllm_inputs[i_gen]['prompt_token_ids'] += response_
                    multi_turn_response_mask[i_gen].append(torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device)) # ASSISTANT, Mark as 1
                    # [TOOL CALL TRIGGER] We check model's last turn response, if not any tool called, then remove this traj from to_generate
                    decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)
                    if re.search(r'<resize>', decoded_resp_):  # FIXME: r'<resize>'
                        assert str(i_gen) not in id_tool_query_mapping.keys()
                        if len(vllm_inputs[i_gen]['multi_modal_data']['image']) >= max_image_num:   # If the current traj has already reached max_image_num, but still try to call tool, we should remove this traj.
                            idx_to_remove.append(i_gen)
                            print(f"Traj {i} exceeds maximum function tool call num {len(vllm_inputs[i]['multi_modal_data']['image'])}")
                            continue
                        id_tool_query_mapping[str(i_gen)] = {
                            "tool_type": "resize",
                        }
                    # Direct Answer
                    else:
                        # remove this traj from to_generate
                        idx_to_remove.append(i_gen)
                        # NOTE: to_generate.remove(i_gen) # DO NOT .remove() in for loop
                    # print(f"[Round #{current_iteration}] i_gen: {i_gen} | resp: {self.tokenizer.decode(response_, skip_special_tokens=True)}")
                if to_generate and id_tool_query_mapping:   # Make sure to PRINT when to_generate and id_tool_query_mapping is not None
                    print(f"[Round #{current_iteration}] Example Generation: to_generate[0]: {to_generate[0]} | response[0]: {self.tokenizer.decode(response[0], skip_special_tokens=True)}")
                    print(f"[Round #{current_iteration} Rollout Tool Call Trigger] For THIS round, ids {next(iter(id_tool_query_mapping))} need to apply function tool using: {id_tool_query_mapping[next(iter(id_tool_query_mapping))]} ...")
                else:
                    print(f"[Round #{current_iteration} Rollout Tool Call Trigger] No ids need to apply function tool for this round.")
                # update 'to_generate'
                for x in idx_to_remove:
                    to_generate.remove(x)

                print(f"[Round #{current_iteration} Rollout END] For NEXT round, We hava {len(to_generate)} trajs to complete ...")

                # [Call Function Tool]
                function_tool_results = []
                for i_todo in to_generate:
                    assert str(i_todo) in id_tool_query_mapping.keys()
                    image_to_resize = vllm_inputs[i_todo]['multi_modal_data']['image'][-1]
                    tool_type = id_tool_query_mapping[str(i_todo)]['tool_type']
                    if tool_type == "resize":
                        tool_outputs = resize_image(image_to_resize)   # Here we can add a save_path to visualize the annotated images
                    else:
                        raise NotImplementedError(f"Tool type {tool_type} is not implemented.")
                    function_tool_results.append(tool_outputs)
            
                # [Process Tool Call Results]
                to_generate_ = to_generate.copy() # make a copy since we will be modifying to_generate
                assert len(to_generate_) == len(function_tool_results)

                for i_gen_, tool_call_result_ in zip(to_generate_, function_tool_results):
                    if isinstance(tool_call_result_, Image.Image):
                        # Construct Next Round Prompt
                        # FIXME: Only support image resize tool
                        # tool_call_prompt_message = "<|im_start|>user\n" + "Resized image:\n<|vision_start|><|image_pad|><|vision_end|>\n" + SIMPLE_RESIZE_IMAGE_MULTI_TRUN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
                        tool_call_prompt_message = "<|im_start|>user\n" + "Resized image:\n<|vision_start|><|image_pad|><|vision_end|>\n" + SIMPLE_THINK_MULTI_TURN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
                        next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
                        # update conversation
                        vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'
                        vllm_inputs[i_gen_]['multi_modal_data']['image'].append(tool_call_result_)
                        multi_turn_response_mask[i_gen_].append(torch.zeros(len(next_turn_prompt_ids), dtype=attention_mask.dtype, device=attention_mask.device)) # USER, Mark as 0
                    else:
                        raise NotImplementedError(f"Error info logging is not supported!")
                # update pbar
                pbar.update(worker_trajs_count - len(to_generate))
                # update iteration count
                current_iteration += 1

        # re-build response
        response = [] # B'*R, torch.Tensors with unequal lengths
        response_generation_mask = [] # B'*R, torch.Tensors with unequal lengths but align with 'response'
        for i_ in range(batch_size * n):
            # for each traj, we skip first-round prompt_ids/attention_mask
            first_round_prompt_length = prefix_prompt_lengths[i_]
            # Repeat <|image_pad|> token id for modeling_qwen2vl
            generation_response_ids = vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]
            resized_image_inputs = self.processor.image_processor(vllm_inputs[i_]['multi_modal_data']['image'][1:], return_tensors='pt')    # NOTE: The fisrt image is the original image, here we only take the resized image into account
            image_grid_thws = resized_image_inputs['image_grid_thw']
            all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0).tolist()
            index, image_pad_token, magic_num = 0, 151655, 654321
            while image_pad_token in generation_response_ids:
                image_pad_token_pos = generation_response_ids.index(image_pad_token)
                image_pad_token_repeat_num = image_grid_thws[index].prod() // self.merge_length
                # update response_tensor_ids
                generation_response_ids[image_pad_token_pos : image_pad_token_pos + 1] = [magic_num] * image_pad_token_repeat_num
                # update all_response_masks
                all_response_masks[image_pad_token_pos : image_pad_token_pos + 1] = [0] * image_pad_token_repeat_num
                index += 1
            generation_response_ids = [image_pad_token if x == magic_num else x for x in generation_response_ids]
            all_response = torch.tensor(generation_response_ids, device=idx.device, dtype=idx.dtype)
            all_response_masks = torch.tensor(all_response_masks, dtype=torch.int64, device=attention_mask.device)
            response.append(all_response)
            response_generation_mask.append(all_response_masks) # at least we have single-turn conversation
            assert response[i_].shape[0] == response_generation_mask[i_].shape[0], f"Shape mismatched between resp_id and resp_mask! response[i_].shape[0]: {response[i_].shape[0]}, response_generation_mask[i_].shape[0]: {response_generation_mask[i_].shape[0]}"
        assert len(response) == len(response_generation_mask), "Length mismatched between response and response_generation_mask!"

        # attention_mask:       prompt           response
        #                 [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        response = pad_to_max_stack(response, self.pad_token_id, dim=0) # Tensor, (B'*R, padded_length), padded_length is the max length of samples in list
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0) # Tensor, (B'*R, padded_length)
        assert all([response.size(dim) == response_generation_mask.size(dim) for dim in range(response.ndim)])

        # cut or pad to max length
        # all should be (B*R, self.config.max_total_response_length)
        if response.shape[1] > self.config.max_total_response_length:
            response = response[:,:self.config.max_total_response_length]
            response_generation_mask = response_generation_mask[:,:self.config.max_total_response_length]
        elif response.shape[1] < self.config.max_total_response_length:
            response = pad_sequence_to_length(response, self.config.max_total_response_length, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(response_generation_mask, self.config.max_total_response_length, 0)

        # All for 1st USER prompt
        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n) # (B, max_prompt_length) -> (B*R, max_prompt_length)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n
            # we also need to repeat 'input_prompt_generation_mask'
            input_prompt_generation_mask = _repeat_interleave(input_prompt_generation_mask, self.config.n) # (B, max_prompt_length) -> (B*R, max_prompt_length), all 0

        # NOTE: We repeat 'multi_modal_data'
        if 'multi_modal_data' in vllm_inputs[0]:
            repeated_multi_modal_data = [vllm_input['multi_modal_data'] for vllm_input in vllm_inputs]  # 这里不再对vllm_input重复n次,因为vllm_inputs是按照n=1构建的,已经重复了n次
            non_tensor_batch['multi_modal_data'] = np.array(repeated_multi_modal_data)

        del vllm_inputs

        seq = torch.cat([idx, response], dim=-1) # (B*R, max_prompt_length + max_total_response_length)

        # FIXME: function get_final_eos_mask cannot handle cases that when there is no <|im_end|> in the given response
        # response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, <|im_end|> is 151645
        # attention_mask: (...,0,0,0,1,1,1), response_attention_mask: (1,1,1,0,0,0,...)
        response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, |im_end| is 151645
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        response_length = response.size(1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids_list = []
            for prompt_with_response, attn_mask, multi_modal_data in zip(seq, attention_mask, repeated_multi_modal_data):
                image_inputs = self.processor.image_processor(multi_modal_data['image'], return_tensors='pt')
                image_grid_thw = image_inputs['image_grid_thw']
                pos_ids = get_rope_index(
                    self.processor,
                    input_ids=prompt_with_response,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attn_mask,
                )
                position_ids_list.append(pos_ids)
            position_ids = torch.stack(position_ids_list, dim=0)
        else:
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            # TODO(sgm): fix position_ids on right_pad
            # prompt: left pad + response: right pad
            # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
            response_position_ids = position_ids[:, -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # NOTE: .contiguous() for broadcast
        batch = TensorDict(
            {
                'prompts': idx.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous()
            },
            batch_size=batch_size
        )

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        print(f">>> vLLM Rollout Ends ...")

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

from vllm.worker.worker_base import WorkerWrapperBase
class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address


    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

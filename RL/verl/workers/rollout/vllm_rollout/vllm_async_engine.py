import os
import re
import time
import io
import asyncio
import logging
import json
from collections.abc import AsyncGenerator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import cloudpickle
import pickle
import zmq
import numpy as np
import ray
import torch
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length, get_eos_mask, get_final_eos_mask, pad_2d_list_to_length
from verl.models.transformers.rope_utils import get_rope_index
from verl.workers.rollout.vllm_rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.workers.rollout.vllm_rollout.vllm_rollout import _pre_process_inputs
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase

from verl.utils.dataset.task_prompt import ERROR_INFO_MULTI_TURN_PROMPT, TOOL_CALL_CROP_MULTI_TRUN_PROMPT
from .function_tools import prepare_grounding_inputs_multi_turn, crop_image, get_valid_mask
from PIL import Image

def _get_model_runner_workers(vllm_config, init_ray: bool = True):
    assert vllm_config.instance_id is not None, "instance_id must be set for external ray actors."
    fields = vllm_config.instance_id.split(":")
    assert len(fields) == 4, (
        f"instance_id: {vllm_config.instance_id} must be in the format of "
        f"<namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
    )
    namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])
    # Make sure subprocess in same namespace as parent actor.
    # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
    if init_ray:
        ray.init(namespace=namespace)
    actor_names = [
        actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")
    ]
    vllm_tp_size = vllm_config.parallel_config.tensor_parallel_size
    assert len(actor_names) == vllm_dp_size * vllm_tp_size, (
        f"instance_id: {vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: "
        f"{vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."
    )
    def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
        fields = actor_name.split(":")
        assert len(fields) == 2, f"invalid actor name: {actor_name}"
        pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
        return pg_index, local_rank
    # sort actor names by pg_index and local_rank
    actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
    actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
    workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
    print(f"instance_id: {vllm_config.instance_id} initializes with external actors: {actor_names}")
    return workers

def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor:
    assert all([t.ndim == 1 for t in tensor_list])
    max_len = max([t.size(0) for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(torch.cat([t,torch.tensor([pad_token_id] * (max_len-t.size(0)), device=t.device, dtype=t.dtype)],dim=0))
    return torch.stack(padded_tensor_list, dim=dim)

class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.vllm_config.instance_id is not None, "instance_id must be set for external ray actors."
        self.workers = _get_model_runner_workers(vllm_config=self.vllm_config, init_ray=True)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        print(f"instance_id: {self.vllm_config.instance_id} intializes finished.")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        non_block: bool = False,
    ) -> List[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method

        outputs = ray.get(
            [worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers]
        )
        return outputs

    def check_health(self):
        return

class ExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        non_block: bool = False,
    ) -> List[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))
        return outputs

    def check_health(self):
        return

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

def compute_reward(reward_fn, batch, non_tensor_batch, iteration):
    cur_batch = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    data_source_list = cur_batch.non_tensor_batch['data_source']
    print(f"ready to call reward function for {len(data_source_list)} samples")
    reward_tensor, acc_reward_tensor, format_reward_tensor, overlong_reward_tensor, invalid_uids = reward_fn(cur_batch)

    print(f'iteration: {iteration}')
    return reward_tensor.numpy(), acc_reward_tensor.numpy(), format_reward_tensor.numpy(), overlong_reward_tensor.numpy(), invalid_uids

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


@ray.remote(num_cpus=1)
class AsyncvLLMEngine:
    """
    AsyncvLLMEngine is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    2. AsyncLLM spawn EngineCore in subprocess.
    3. EngineCore initialize ExternalRayDistributedExecutor.
    4. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    5. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str, tokenizer, processor, reward_fn, val_reward_fn):
        """
        Args:
            config: DictConfig, actor_rollout_ref config.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        # super().__init__()

        self.config = config
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.tokenizer = tokenizer
        self.engine: AsyncLLM = None
        self.pad_token_id = self.tokenizer.pad_token_id

        self.max_pixels = self.config.rollout.max_pixels
        self.min_pixels = self.config.rollout.min_pixels
        self.max_generation_round = self.config.rollout.get('max_generation_round', 1)
        self.val_max_generation_round = self.config.rollout.get('val_max_generation_round', 1)
        self.multi_turn_prompt_type = self.config.rollout.multi_turn_prompt_type

        self.ignore_exceed = self.config.actor.ignore_exceed

        self.processor = processor
        self.merge_length = self.processor.image_processor.merge_size ** 2

        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

    def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_total_response_length #config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        kwargs['n'] = 1

        # print(f"self.config.rollout.stop: {self.config.rollout.stop}, type(self.config.rollout.stop): {type(self.config.rollout.stop)}")

        kwargs['stop'] = list(self.config.rollout.stop)

        print(f"override_generation_config: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        backend = os.environ.get("VERL_VLLM_DISTRIBUTED_BACKEND", "ray")
        if backend == "zeromq":
            distributed_executor_backend = ExternalZeroMQDistributedExecutor
        elif backend == "ray":
            distributed_executor_backend = ExternalRayDistributedExecutor
        else:
            distributed_executor_backend = None

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            # disable_log_stats=config.disable_log_stats,
            disable_log_stats=False,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
            limit_mm_per_prompt=config.get('limit_mm_per_prompt', {'image': 1}),
        )

        # init async llm engine
        vllm_config = self._create_engine_config(engine_args)
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

    def _create_engine_config(self, engine_args: AsyncEngineArgs):
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"

        # VERL_VLLM_ZMQ_ADDRESSES
        if engine_args.distributed_executor_backend == ExternalZeroMQDistributedExecutor:
            workers = _get_model_runner_workers(vllm_config=vllm_config, init_ray=False)
            zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in workers])
            print(f"VERL_VLLM_ZMQ_ADDRESSES: {zmq_addresses}")
            os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        return vllm_config

    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n):
        req_list = []
        for data_idx, _raw_prompt_id in enumerate(prompts.non_tensor_batch["raw_prompt_ids"]):
            _raw_prompt_id = list(_raw_prompt_id)
            _multimodal_data = prompts.non_tensor_batch["multi_modal_data"][data_idx]
            _image_size_used = prompts.non_tensor_batch["image_size_used"][data_idx][0]
            _raw_prompt = prompts.non_tensor_batch["raw_prompt"][data_idx]
            _index = prompts.non_tensor_batch["index"][data_idx]
            _uid = prompts.non_tensor_batch["uid"][data_idx]
            _ground_truth = prompts.non_tensor_batch["ground_truth"][data_idx]
            _data_source = prompts.non_tensor_batch["data_source"][data_idx]
            _input_ids = prompts.batch['input_ids'][data_idx]
            _attention_mask = prompts.batch['attention_mask'][data_idx]
            _position_ids = prompts.batch['position_ids'][data_idx]
            assert len(_multimodal_data['image']) == 2, f"multimodal_data should contain the downscaled image and a upscaled resized image, but got len(multi_modal_data['image']): {len(_multimodal_data['image'])}."
            original_image_path = _multimodal_data['image'].pop(-1)
            original_image = self.load_image(original_image_path)
            image_token_num = self.calculate_image_token_num(_multimodal_data['image'][0])

            for idx in range(n):
                req = AsyncRolloutRequest(
                    batch_data_id=data_idx,
                    rollout_offset=idx,
                    request_id=str(uuid4()),
                    multi_modal_data=_multimodal_data,
                    raw_prompt_id=_raw_prompt_id,
                    reward_scores={},
                    max_response_len=self.config.rollout.response_length,
                    max_model_len=(
                        self.config.rollout.max_model_len
                        or self.config.rollout.prompt_length + self.config.rollout.response_length
                    ),
                    image_size_used=_image_size_used,
                    raw_prompt=_raw_prompt,
                    index=_index,
                    uid=_uid,
                    ground_truth=_ground_truth,
                    data_source=_data_source,
                    input_ids=_input_ids,
                    attention_mask=_attention_mask,
                    position_ids=_position_ids
                )
                req.original_image=original_image
                req.image_token_num = image_token_num
                req_list.append(req)

        return req_list

    def load_image(self, image_path):
        assert os.path.exists(image_path)
        original_image = Image.open(image_path).convert("RGB")
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

    def post_process_single(self, prefix_length, vllm_input, multi_turn_response_mask):

        # for each traj, we skip first-round prompt_ids/attention_mask
        first_round_prompt_length = prefix_length
        # Repeat <|image_pad|> token id for modeling_qwen2vl
        generation_response_ids = vllm_input['prompt_token_ids'][first_round_prompt_length:]

        # print(f"len(vllm_input['multi_modal_data']['image']): {len(vllm_input['multi_modal_data']['image'])}, vllm_input['multi_modal_data']['image']: {vllm_input['multi_modal_data']['image']}, len(multi_turn_response_mask): {len(multi_turn_response_mask)}")

        if len(vllm_input['multi_modal_data']['image']) > 1:
            processed_image_inputs = self.processor.image_processor(vllm_input['multi_modal_data']['image'][1:], return_tensors='pt')    # NOTE: The fisrt image is the original image, here we only take the resized image into account
            image_grid_thws = processed_image_inputs['image_grid_thw']
        all_response_masks = torch.cat(multi_turn_response_mask[1:], dim=0).tolist()
        
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
        all_response = torch.tensor(generation_response_ids, dtype=torch.int64, device=multi_turn_response_mask[0].device)
        all_response_masks = torch.tensor(all_response_masks, dtype=torch.int64, device=multi_turn_response_mask[0].device)

        return all_response, all_response_masks

    def preprocess_for_reward_function(self, input_ids, attention_mask, response, response_masks):

        response = [response]
        response_generation_mask = [response_masks]

        # attention_mask:       prompt           response
        #                 [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        response = pad_to_max_stack(response, self.pad_token_id, dim=0) # Tensor, (B'*R, padded_length), padded_length is the max length of samples in list
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0) # Tensor, (B'*R, padded_length)
        assert all([response.size(dim) == response_generation_mask.size(dim) for dim in range(response.ndim)])

        # cut or pad to max length
        # all should be (B*R, self.config.max_total_response_length)
        if response.shape[1] > self.config.rollout.max_total_response_length:
            response = response[:,:self.config.rollout.max_total_response_length]
            response_generation_mask = response_generation_mask[:,:self.config.rollout.max_total_response_length]
        elif response.shape[1] < self.config.rollout.max_total_response_length:
            response = pad_sequence_to_length(response, self.config.rollout.max_total_response_length, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(response_generation_mask, self.config.rollout.max_total_response_length, 0)
        
        # FIXME: function get_final_eos_mask cannot handle cases that when there is no <|im_end|> in the given response
        # response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, <|im_end|> is 151645
        # attention_mask: (...,0,0,0,1,1,1), response_attention_mask: (1,1,1,0,0,0,...)
        response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, |im_end| is 151645
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        input_prompt_generation_mask = torch.zeros_like(input_ids, dtype=attention_mask.dtype, device=attention_mask.device) # (B'*R, max_prompt_length), all 0
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        seq = torch.cat([input_ids, response], dim=-1) # (B*R, max_prompt_length + max_total_response_length)

        return input_ids, response, seq, attention_mask, multi_turn_response_mask

    def process_tool_call(self, vllm_input, decoded_resp_, observations_list, image_size_used_list, multi_turn_response_mask, current_iteration, save_traj, save_dir, doc_id):
        error_info = None
        try:
            pattern = ".*<grounding>{\"bbox_2d\": (.*),.*\"source\": [\',\"](.*)[\',\"]}</grounding>"
            match = re.match(pattern, decoded_resp_, re.DOTALL)
            bbox, source = match.group(1), match.group(2)
            json_objects = [{"bbox_2d": eval(bbox), "source": source}]
            tool_type, args = prepare_grounding_inputs_multi_turn(json_objects, observations_list, image_size_used_list, use_relative_coordinates=self.config.rollout.use_relative_coordinates)
        except Exception as e:
            print(str(e))
            error_info = str(e)
            tool_type = None
            args = None
        
        if error_info is not None:
            tool_outputs = f"ERROR occurs during grounding. Error Information: {error_info}.\n"
        else:
            if tool_type == 'grounding':
                tool_outputs = crop_image(args[0], args[1], image_size_used_list, resize=1)
                observations_list.append(tool_outputs)
            else:
                raise ValueError(f"Unsupported tool type: {tool_type}.")
        
        if isinstance(tool_outputs, Image.Image):
            # Construct Next Round Prompt
            tool_call_prompt_message = "<|im_end|>\n<|im_start|>user\n" + TOOL_CALL_CROP_MULTI_TRUN_PROMPT.format(action_turn=current_iteration, observation_turn=current_iteration+1) + "<|im_end|>\n<|im_start|>assistant\n"
            
            next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
            # update conversation
            vllm_input['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'

            resized_image = process_image(tool_outputs, max_pixels=self.max_pixels, min_pixels=self.min_pixels)

            if save_traj:
                self.save_traj_and_obs(os.path.join(save_dir, doc_id), json_dict=None, original_image=tool_outputs, resize_image=resized_image, turn_idx=current_iteration+1)

            image_size_used_list.append(resized_image.size)

            vllm_input['multi_modal_data']['image'].append(resized_image)
            multi_turn_response_mask.append(torch.zeros(len(next_turn_prompt_ids), dtype=multi_turn_response_mask[-1].dtype, device=multi_turn_response_mask[-1].device)) # USER, Mark as 0

            image_token_num = self.calculate_image_token_num(resized_image)
            new_context_length = len(next_turn_prompt_ids) + image_token_num - 1

            # print(f"e2: doc_id: {doc_id}, current_iteration: {current_iteration}, context_length: {context_length}, image_size_used_list: {image_size_used_list}, self.max_pixels: {self.max_pixels}, self.min_pixels: {self.min_pixels}")

        else:
            tool_call_prompt_message = "<|im_end|>\n<|im_start|>user\n" + tool_outputs + ERROR_INFO_MULTI_TURN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
            next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
            vllm_input['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'
            multi_turn_response_mask.append(torch.zeros(len(next_turn_prompt_ids), dtype=multi_turn_response_mask[-1].dtype, device=multi_turn_response_mask[-1].device)) # USER, Mark as 0

            new_context_length = len(next_turn_prompt_ids)

            # print(f"e3: doc_id: {doc_id}, current_iteration: {current_iteration}, context_length: {context_length}, image_size_used_list: {image_size_used_list}, self.max_pixels: {self.max_pixels}, self.min_pixels: {self.min_pixels}")

            if save_traj:
                json_line = {
                    "error_info": tool_call_prompt_message
                }
                self.save_traj_and_obs(os.path.join(save_dir, doc_id), json_dict=json_line)

        return new_context_length, tool_outputs

    async def _async_rollout_a_request(
        self, req: AsyncRolloutRequest, do_sample: bool = True, is_validate: bool = False, **kwargs
    ) -> AsyncRolloutRequest:
        loop = asyncio.get_running_loop()
        _req = await loop.run_in_executor(None, lambda: deepcopy(req))
        #_req = deepcopy(req)
        finish_reason_type = None
        output = None
        current_turns = 0

        kwargs = {
            "n": 1,
            "stop": list(self.config.rollout.stop), #['</grounding>',],
            "detokenize": True,
            "include_stop_str_in_output": True,
        }
        if not do_sample:
            kwargs.update({
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0
            })
        
        vllm_input = {}
        if _req.multi_modal_data:

            multi_modal_data = _req.multi_modal_data
            observations_list = [_req.original_image]
            vllm_input = {'prompt_token_ids': _req.raw_prompt_id, 'multi_modal_data': multi_modal_data}

        prefix_length = len(_req.raw_prompt_id)

        image_token_num = _req.image_token_num
        context_length = prefix_length + image_token_num - 1
        raw_prompt = _req.raw_prompt
        doc_id = str(_req.index)
        uid = _req.uid
        ground_truth = _req.ground_truth
        data_source = _req.data_source
        image_size_used_list = [_req.image_size_used,]

        input_ids = torch.tensor(_req.input_ids).unsqueeze(0)
        attention_mask = torch.tensor(_req.attention_mask).unsqueeze(0)
        position_ids = torch.tensor(_req.position_ids).unsqueeze(0)

        multi_turn_response_mask = [torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)]

        save_dir = self.config.rollout.save_traj_dir
        save_traj = (save_dir is not None) and (self.config.rollout.save_traj == True)

        n = self.config.rollout.val_n if is_validate else self.config.rollout.n  # TODO: for validate, do_sample=False
        doc_id = doc_id if n == 1 else f"{doc_id}_{_req.rollout_offset}"

        if save_traj:
            json_line = {
                "doc_id": doc_id, 
                "initial_prompt": raw_prompt, 
                
                'ground_truth': ground_truth
            }
            self.save_traj_and_obs(os.path.join(save_dir, doc_id), json_line, _req.original_image, multi_modal_data['image'][0])
        
        if is_validate and self.val_max_generation_round != 1:
            max_iterations = self.val_max_generation_round
            max_image_num = self.config.rollout.get('val_limit_mm_per_prompt', {'image': 1})['image']
        else:
            max_iterations = self.max_generation_round
            max_image_num = self.config.rollout.get('limit_mm_per_prompt', {'image': 1})['image']
        
        # print(f"max_iterations: {max_iterations}, max_image_num: {max_image_num}")

        grounding_pattern = re.compile(r'<grounding>(.*?)</grounding>', re.DOTALL)
        current_iteration = 0
        exceed = False
        void = False
        while current_iteration < max_iterations:
            # with self.update_sampling_params(**kwargs):

            sampling_params = deepcopy(self.sampling_params)
            for key, value in kwargs.items():
                if hasattr(sampling_params, key):
                    setattr(sampling_params, key, value)
                    
            # print("current_iteration: ", current_iteration, " sampling_params: ", sampling_params)

            outputs = self.engine.generate(
                prompt=vllm_input,  # because we have already convert it to prompt token id
                sampling_params=sampling_params,
                request_id=_req.request_id+str(current_iteration),
            )

            async for res in outputs:
                results = res

            content = results.outputs[0].text

            _token_ids = results.outputs[0].token_ids
            filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]
            # if 151645 not in filtered_token_ids:
            #     filtered_token_ids = filtered_token_ids + [151645,]
            response_ = filtered_token_ids

            # print("content: ", content, "response_: ", response_)
            # exit()

            vllm_input['prompt_token_ids'] += response_
            multi_turn_response_mask.append(torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device)) # ASSISTANT, Mark as 1
            context_length += len(response_)

            # print(f"e1: doc_id: {doc_id}, current_iteration: {current_iteration}, context_length: {context_length}")

            decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)
            tool_call_contents = grounding_pattern.findall(decoded_resp_)
            if save_traj:
                json_line = {"turn_idx": current_iteration, "text_output": decoded_resp_}
                self.save_traj_and_obs(os.path.join(save_dir, doc_id), json_line)

            if len(tool_call_contents) > 0:
                if (len(vllm_input['multi_modal_data']['image']) >= max_image_num) or (current_iteration == max_iterations - 1):   # If the current traj has already reached max_image_num, but still try to call tool, we should remove this traj.
                    exceed = True
                    break

                if context_length >= self.config.rollout.max_total_response_length - 2000:
                    exceed = True
                    break

                old_prompt_token_ids = deepcopy(vllm_input['prompt_token_ids'])

                new_context_length, tool_outputs = await loop.run_in_executor(None, lambda: self.process_tool_call(vllm_input, decoded_resp_, observations_list, image_size_used_list, multi_turn_response_mask, current_iteration, save_traj, save_dir, doc_id))
                context_length += new_context_length

                if context_length >= self.config.rollout.max_total_response_length - 2000:
                    vllm_input['prompt_token_ids'] = old_prompt_token_ids
                    if isinstance(tool_outputs, Image.Image):
                        vllm_input['multi_modal_data']['image'].pop()
                        image_size_used_list.pop()
                        observations_list.pop()
                    multi_turn_response_mask.pop()
                    exceed = True
                    break

            else:
                finish_reason = results.outputs[0].finish_reason
                finish_reason_type = FinishReasonTypeEnum.from_str(finish_reason)
                if finish_reason_type == FinishReasonTypeEnum.LENGTH or (re.match(".*<answer>.*</answer>$", decoded_resp_, re.DOTALL) is None):
                    void = True
                break
            
            current_iteration += 1

        # finish_reason = results.outputs[0].finish_reason
        # finish_reason_type = FinishReasonTypeEnum.from_str(finish_reason)
        # if finish_reason_type == FinishReasonTypeEnum.LENGTH:
        #     _req.add_assistant_message(self.tokenizer, content, already_over_long=True)
        # else:
        #     _req.add_assistant_message(self.tokenizer, content)
        # import time
        # time.sleep(10000)

        avg_response_tokens_per_turn = torch.cat(multi_turn_response_mask, dim=0).sum(-1).item() / (current_iteration + 1)

        all_response, all_response_masks = self.post_process_single(prefix_length, vllm_input, multi_turn_response_mask)

        input_ids, response, seq, attention_mask, multi_turn_response_mask = self.preprocess_for_reward_function(input_ids, attention_mask, all_response, all_response_masks)

        batch_cur = TensorDict(
            {
                'prompts': input_ids.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous()
            },
            batch_size=1
        )

        data_source_cur = np.array([data_source], dtype=object)
        ground_truth_cur = np.array([ground_truth], dtype=object)
        raw_prompt_cur = np.array([raw_prompt], dtype=object)
        uid_list_cur = np.array([uid], dtype=object)
        non_tensor_batch_cur = {
            'data_source': data_source_cur,
            'ground_truth': ground_truth_cur,
            'raw_prompt': raw_prompt_cur,
            'uid': uid_list_cur,
        }

        if is_validate:
            reward_tensor_cur, acc_reward_tensor_cur, format_reward_tensor_cur, overlong_reward_tensor_cur, invalid_uids_cur = compute_reward(self.val_reward_fn, batch_cur.to('cpu'), non_tensor_batch_cur, current_iteration)
            # future = compute_reward.remote(self.val_reward_fn, batch_cur.to('cpu'), non_tensor_batch_cur, current_iteration)
        else:
            reward_tensor_cur, acc_reward_tensor_cur, format_reward_tensor_cur, overlong_reward_tensor_cur, invalid_uids_cur = compute_reward(self.reward_fn, batch_cur.to('cpu'), non_tensor_batch_cur, current_iteration)
            # future = compute_reward.remote(self.reward_fn, batch_cur.to('cpu'), non_tensor_batch_cur, current_iteration)

        # reward_tensor_cur = torch.from_numpy(reward_tensor_cur).to(device=input_ids.device)
        # acc_reward_tensor_cur = torch.from_numpy(acc_reward_tensor_cur).to(device=input_ids.device)
        # format_reward_tensor_cur = torch.from_numpy(format_reward_tensor_cur).to(device=input_ids.device)
        # indices = torch.from_numpy(indices).to(device=reward_tensor.device)

        # print(f"type(reward_tensor_cur): {type(reward_tensor_cur)}, reward_tensor_cur.shape: {reward_tensor_cur.shape}, type(acc_reward_tensor_cur): {type(acc_reward_tensor_cur)}, acc_reward_tensor_cur.shape: {acc_reward_tensor_cur.shape}, type(format_reward_tensor_cur): {type(format_reward_tensor_cur)}, format_reward_tensor_cur.shape: {format_reward_tensor_cur.shape}")

        _req.all_response_ids = all_response
        _req.all_response_masks = all_response_masks
        _req.multi_modal_data = vllm_input['multi_modal_data']
        _req.reward_tensor = reward_tensor_cur.tolist()
        _req.acc_reward_tensor = acc_reward_tensor_cur.tolist()
        _req.format_reward_tensor = format_reward_tensor_cur.tolist()
        _req.overlong_reward_tensor = overlong_reward_tensor_cur.tolist()
        _req.invalid_uids = invalid_uids_cur
        _req.exceed = exceed
        _req.avg_response_tokens_per_turn = avg_response_tokens_per_turn
        _req.round_num = current_iteration + 1
        _req.void = void

        # _req.finalize(self.tokenizer, reward_scores=None, finish_reason_type=finish_reason_type)

        return _req

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

    async def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("is_validate", False)
        tgt_device = prompts.batch["input_ids"].device

        req_list = self._preprocess_prompt_to_async_rollout_requests(
            prompts,
            n = self.config.rollout.val_n if is_validate else self.config.rollout.n
        )
        t0 = time.time()
        with torch.no_grad():
            output_req_list = await asyncio.gather(
                *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list]
            )
        t1 = time.time()
        print(f"time: {t1-t0}")
        sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))


        return self.post_process(prompts, sorted_output_req_list)

    def post_process(self, prompts: DataProto, output_req_list: List[AsyncRolloutRequest]) -> DataProto:
        config = self.config.rollout
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("is_validate", False)
        tgt_device = prompts.batch["input_ids"].device

        # convert to DataProto
        response = []
        response_generation_mask =[]
        multi_modal_data_list = []
        raw_prompts = []
        uid_list = []
        data_source_list = []
        ground_truth_list = []
        exceed_list = []
        avg_response_tokens_per_turn_list = []
        round_num_list = []
        void_list = []

        reward_tensor_list = []
        acc_reward_tensor_list = []
        format_reward_tensor_list = []
        overlong_reward_tensor_list = []
        invalid_uids = []
        
        for req in output_req_list:
            response.append(req.all_response_ids)
            response_generation_mask.append(req.all_response_masks) # at least we have single-turn conversation
            multi_modal_data_list.append(req.multi_modal_data)
            raw_prompts.append(req.raw_prompt)
            uid_list.append(req.uid)
            data_source_list.append(req.data_source)
            ground_truth_list.append(req.ground_truth)
            exceed_list.append(req.exceed)
            avg_response_tokens_per_turn_list.append(req.avg_response_tokens_per_turn)
            round_num_list.append(req.round_num)
            void_list.append(req.void)
            
            reward_tensor_list.append(torch.tensor(req.reward_tensor, dtype=torch.float, device=tgt_device))
            acc_reward_tensor_list.append(torch.tensor(req.acc_reward_tensor, dtype=torch.float, device=tgt_device))
            format_reward_tensor_list.append(torch.tensor(req.format_reward_tensor, dtype=torch.float, device=tgt_device))
            overlong_reward_tensor_list.append(torch.tensor(req.overlong_reward_tensor, dtype=torch.float, device=tgt_device))
            invalid_uids.extend(req.invalid_uids)
        
        # attention_mask:       prompt           response
        #                 [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        response = pad_to_max_stack(response, self.pad_token_id, dim=0) # Tensor, (B'*R, padded_length), padded_length is the max length of samples in list
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0) # Tensor, (B'*R, padded_length)
        assert all([response.size(dim) == response_generation_mask.size(dim) for dim in range(response.ndim)])

        # cut or pad to max length
        # all should be (B*R, self.config.max_total_response_length)
        if response.shape[1] > config.max_total_response_length:
            response = response[:,:config.max_total_response_length]
            response_generation_mask = response_generation_mask[:,:config.max_total_response_length]
        elif response.shape[1] < config.max_total_response_length:
            response = pad_sequence_to_length(response, config.max_total_response_length, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(response_generation_mask, config.max_total_response_length, 0)

        prompt_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch['attention_mask']  # (bs*tp, max_prompt_length), left padding 0
        position_ids = prompts.batch['position_ids']  # (bs*tp, max_prompt_length), left padding 0

        n = config.val_n if is_validate else config.n
        if n > 1 and do_sample:
            prompt_ids = _repeat_interleave(prompt_ids, n) # (B, max_prompt_length) -> (B*R, max_prompt_length)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)
            # we also need to repeat 'input_prompt_generation_mask'
            # input_prompt_generation_mask = _repeat_interleave(input_prompt_generation_mask, n) # (B, max_prompt_length) -> (B*R, max_prompt_length), all 0

        # FIXME: function get_final_eos_mask cannot handle cases that when there is no <|im_end|> in the given response
        # response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, <|im_end|> is 151645
        # attention_mask: (...,0,0,0,1,1,1), response_attention_mask: (1,1,1,0,0,0,...)
        response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, |im_end| is 151645

        # print(f"attention_mask.shape: {attention_mask.shape}, response_attention_mask.shape: {response_attention_mask.shape}")

        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        input_prompt_generation_mask = torch.zeros_like(prompt_ids, dtype=attention_mask.dtype, device=attention_mask.device) # (B'*R, max_prompt_length), all 0
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        seq = torch.cat([prompt_ids, response], dim=-1) # (B*R, max_prompt_length + max_total_response_length)

        # NOTE: We repeat 'multi_modal_data'
        non_tensor_batch = {}
        non_tensor_batch['multi_modal_data'] = np.array(multi_modal_data_list)
        non_tensor_batch['raw_prompt'] = np.array(raw_prompts, dtype=object)
        non_tensor_batch['uid'] = np.array(uid_list, dtype=object)
        non_tensor_batch['data_source'] = np.array(data_source_list, dtype=object)
        non_tensor_batch['ground_truth'] = np.array(ground_truth_list, dtype=object)
        non_tensor_batch['avg_response_tokens_per_turn'] = np.array(avg_response_tokens_per_turn_list, dtype=object)
        non_tensor_batch['rounds_nums'] = np.array(round_num_list, dtype=object)

        response_length = response.size(1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids_list = []
            for prompt_with_response, attn_mask, multi_modal_data in zip(seq, attention_mask, multi_modal_data_list):
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

        exceed_mask = torch.tensor(exceed_list, dtype=torch.bool, device=tgt_device)
        void_mask = torch.tensor(void_list, dtype=torch.bool, device=tgt_device)

        reward_tensor = torch.cat(reward_tensor_list, dim=0)
        acc_reward_tensor = torch.cat(acc_reward_tensor_list, dim=0)
        format_reward_tensor = torch.cat(format_reward_tensor_list, dim=0)
        overlong_reward_tensor = torch.cat(overlong_reward_tensor_list, dim=0)

        batch_size = prompt_ids.shape[0]
        batch = TensorDict(
            {
                'prompts': prompt_ids.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous(),
                'reward_tensor': reward_tensor.contiguous(),
                'exceed_mask': exceed_mask.contiguous(),
                'void_mask': void_mask.contiguous(),
                # 'acc_reward_tensor': acc_reward_tensor.contiguous(),
                # 'format_reward_tensor': format_reward_tensor.contiguous()
            },
            batch_size=batch_size
        )

        if not is_validate:
            valid_mask, invalid_mask, wrong_and_use_tool_uids, wrong_and_direct_answer_uids, statistics_dict = get_valid_mask(non_tensor_batch['uid'], non_tensor_batch['multi_modal_data'], prompts.meta_info, reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids, None)
            statistics_dict['wrong_and_use_tool_uids'] = wrong_and_use_tool_uids
            statistics_dict['wrong_and_direct_answer_uids'] = wrong_and_direct_answer_uids
            non_tensor_batch['multi_modal_data'][invalid_mask] = None
            non_tensor_batch['valid_mask'] = np.array(valid_mask, dtype=object)

        data = DataProto(
            batch=batch, non_tensor_batch=non_tensor_batch
        )
        data.non_tensor_batch['acc_scores'] = np.array(acc_reward_tensor.cpu().sum(-1), dtype=object)
        data.non_tensor_batch['format_scores'] = np.array(format_reward_tensor.cpu().sum(-1), dtype=object)
        data.non_tensor_batch['overlong_scores'] = np.array(overlong_reward_tensor.cpu().sum(-1), dtype=object)

        if not is_validate:
            # add empty list with num of "self.config.n - 1" 
            # to make the non_tensor_batch data size equals to rollout.n * batch_size for the communication
            statistics_dict["invalid_num"] = len(invalid_uids)
            statistics_dict_for_comm = [statistics_dict]
            for _ in range(batch_size - 1):
                statistics_dict_for_comm.append([])
            data.non_tensor_batch['statistics_dict'] = np.array(statistics_dict_for_comm, dtype=object)

        return data

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        # generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)
        generator = await self.engine.generate(request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    async def wake_up(self):
        await self.engine.wake_up()

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        await self.engine.sleep()

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
Single Process Actor
"""

import itertools
import numpy as np
from typing import Iterable, Tuple, List, Dict, Any

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils import hf_processor
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        processor=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        # add image processor
        self.processor = processor if processor is not None else hf_processor(config.model_local_path)

    def _prepare_multi_modal_inputs(self, multi_modal_data: Iterable[Any]) -> List[Dict[str, Any]]:
        processed_inputs: List[Dict[str, Any]] = []
        for idx, modal_entry in enumerate(multi_modal_data):
            if modal_entry is None:
                processed_inputs.append({})
            elif 'image' in modal_entry:
                image_inputs = self.processor.image_processor(modal_entry['image'], return_tensors='pt')
                processed_inputs.append({key: val for key, val in image_inputs.items()})
            elif 'video' in modal_entry:
                video_data = modal_entry['video']
                video = [v[0] if isinstance(v, tuple) else v for v in video_data]
                video_metadata = [v[1] for v in video_data if isinstance(v, tuple)]
                # video,video_metadata = video_data[0]
                # transformers video_processor 在 videos 为空时会访问 videos[0] 导致 IndexError，此处做防护
                if not video_data or (isinstance(video_data, (list, tuple)) and len(video_data) == 0):
                    raise ValueError(
                        f"Empty video at multi_modal_data index {idx}. "
                        "Check that rollout/dataset did not produce empty video (e.g. missing path or zero frames)."
                    )
                
                # video_metadata = modal_entry['video_metadata']

                video_inputs = self.processor.video_processor(video,video_metadata=video_metadata,do_sample_frames=False,do_resize=False)
                processed_inputs.append({key: val for key, val in video_inputs.items()})
            else:
                raise KeyError(f"Unsupported multi-modal entry at index {idx}: {list(modal_entry.keys())}")
        return processed_inputs

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                try:
                    multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                        dim=0)
                except Exception as e:
                    print(f"[dp_actor] torch.cat failed for key={key!r}: {e}")
                    for i, inputs in enumerate(micro_batch['multi_modal_inputs']):
                        print(f"  [{i}] inputs[{key!r}] = {inputs[key]}")
                    raise

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_data' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            # NOTE: transform 'multi_modal_data' to 'multi_modal_inputs'
            _processed_inputs = self._prepare_multi_modal_inputs(data.non_tensor_batch['multi_modal_data'])
            _multi_modal_inputs = np.array(_processed_inputs, dtype=object)
            data.non_tensor_batch['multi_modal_inputs'] = _multi_modal_inputs
            del data.non_tensor_batch['multi_modal_data']
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        use_multi_turn_response_mask = self.config.get("use_multi_turn_response_mask", False)
        if use_multi_turn_response_mask:
            select_keys.append('multi_turn_response_mask')

        if self.config.ignore_exceed:
            select_keys.append('exceed_mask')

        if self.config.ignore_void:
            select_keys.append('void_mask')

        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_data' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if 'multi_modal_data' in data.non_tensor_batch.keys():
            has_multi_modal_inputs=True
            # NOTE: transform 'multi_modal_data' to 'multi_modal_inputs'
            _processed_inputs = self._prepare_multi_modal_inputs(data.non_tensor_batch['multi_modal_data'])
            _multi_modal_inputs = np.array(_processed_inputs, dtype=object)
            data.non_tensor_batch['multi_modal_inputs'] = _multi_modal_inputs
            del data.non_tensor_batch['multi_modal_data']
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        elif 'multi_modal_inputs' in data.non_tensor_batch.keys():
            has_multi_modal_inputs=True
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                    # micro_batches = data.select(select_keys, non_tensor_select_keys).multimodal_data_split(self.config.ppo_micro_batch_size_per_gpu)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    if isinstance(data, DataProto):
                        data = {**data.batch.cuda(), **data.non_tensor_batch}
                    else:
                        data = data.cuda()  # actor device is cpu when using offload

                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    
                    # response_mask = attention_mask[:, -response_length:]
                    # Refine 'response_mask' by 'multi_turn_response_mask' for multi-turn scenario
                    if use_multi_turn_response_mask:
                        multi_turn_response_mask = data['multi_turn_response_mask']
                        response_mask = (attention_mask * multi_turn_response_mask)[:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]
                    
                    if self.config.ignore_exceed:
                        exceed_mask = data['exceed_mask']

                        assert exceed_mask.shape[0] == response_mask.shape[0]

                        # print(f"before: response_mask.shape: {response_mask.shape}, response_mask.sum(): {response_mask.sum()}, exceed_mask.shape: {exceed_mask.shape}, exceed_mask.sum(): {exceed_mask.sum()}")
                        response_mask[exceed_mask] = 0
                        # print(f"after: response_mask.shape: {response_mask.shape}, response_mask.sum(): {response_mask.sum()}")

                    if self.config.ignore_void:
                        void_mask = data['void_mask']
                        assert void_mask.shape[0] == response_mask.shape[0]
                        response_mask[void_mask] = 0

                        print(f"void_mask.shape: {void_mask.shape}, void_mask.sum(): {void_mask.sum()}")

                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    # print(f"log_prob.shape: {log_prob.shape}, log_prob.mean(): {log_prob.mean()}, entropy.shape: {entropy.shape}, entropy.mean(): {entropy.mean()}, response_mask.shape: {response_mask.shape}, response_mask.sum(): {response_mask.sum()}")

                    if self.config.policy_loss.loss_mode == 'gspo':
                        # print("self.config.policy_loss.loss_agg_mode: ", self.config.policy_loss.loss_agg_mode)
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss_gspo(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=self.config.policy_loss.loss_agg_mode,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                        )
                    else:
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            loss_agg_mode=self.config.policy_loss.loss_agg_mode,
                            scale=self.config.policy_loss.scale,
                        )

                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    # print(f"pg_loss.shape: {pg_loss.shape}, pg_loss.mean(): {pg_loss.mean()}, entropy_loss.shape: {entropy_loss.shape}, entropy_loss.mean(): {entropy_loss.mean()}, ppo_kl.shape: {ppo_kl.shape}, ppo_kl.mean(): {ppo_kl.mean()}")

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                        'actor/log_prob_mean': log_prob.mean().detach().item(),
                        'actor/entropy_mean': entropy.mean().detach().item()
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}

                # print(f"batch_idx: {batch_idx}, grad_norm: {grad_norm.detach().item()}")

            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

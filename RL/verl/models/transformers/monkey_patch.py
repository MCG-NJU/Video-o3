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
Apply monkey-patch function to models
"""

import sys
from typing import Optional

import torch

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    (batch, seqlen, num_key_value_heads, head_dim) -> (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention for Ulysses sequence parallelism.
    Adapted from the new verl for use with transformers >= 4.50.

    Args:
        query_states: (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states: (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states: (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids: (batch_size, seqlen/sp_size) or (3, batch_size, seqlen/sp_size) for mrope
    """
    from transformers.modeling_flash_attention_utils import _flash_attention_forward as _orig_flash_attn_forward

    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1 and position_ids is not None:
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = _repeat_kv(key_states, repeats)
        value_states = _repeat_kv(value_states, repeats)

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # All-gather position_ids for varlen flash attention
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.concat(position_ids_list, dim=-1)

    query_length = query_states.size(1)
    attn_output = _orig_flash_attn_forward(
        query_states, key_states, value_states, attention_mask, query_length, *args,
        position_ids=position_ids, **kwargs
    )

    if ulysses_sp_size > 1 and position_ids is not None:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


def apply_monkey_patch_to_llama():
    if is_transformers_version_in_range("4.45.0", "4.47.1"):
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
        from verl.models.transformers.llama import llama_flash_attn_forward
        LlamaFlashAttention2.forward = llama_flash_attn_forward
    elif is_transformers_version_in_range("4.48.0", "4.49.0"):
        from transformers.models.llama.modeling_llama import LlamaAttention
        from verl.models.transformers.llama import llama_attn_forward
        LlamaAttention.forward = llama_attn_forward


def apply_monkey_patch_to_qwen2():
    if is_transformers_version_in_range("4.45.0", "4.47.1"):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2
        from verl.models.transformers.qwen2 import qwen2_flash_attn_forward
        Qwen2FlashAttention2.forward = qwen2_flash_attn_forward
    elif is_transformers_version_in_range("4.48.0", "4.49.0"):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        from verl.models.transformers.qwen2 import qwen2_attn_forward
        Qwen2Attention.forward = qwen2_attn_forward


def apply_monkey_patch_to_qwen3vl():
    """
    Patch Qwen3VL for Ulysses sequence parallelism.
    Requires transformers >= 4.57.

    Steps:
    1. Patch Qwen3VLModel.forward to handle multimodal inputs (DeepStack embeddings, gradient flow).
    2. Patch the module-level _flash_attention_forward for Ulysses AlltoAll.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLModel,
    )
    from verl.models.transformers.qwen3_vl import forward_with_normal_backend, qwen3_vl_base_forward

    Qwen3VLModel.forward = qwen3_vl_base_forward
    Qwen3VLForConditionalGeneration.forward = forward_with_normal_backend
    print("Qwen3VL model.forward patched for multimodal inputs.")

    # Also patch MoE variant if available
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeForConditionalGeneration,
            Qwen3VLMoeModel,
        )
        Qwen3VLMoeModel.forward = qwen3_vl_base_forward
        Qwen3VLMoeForConditionalGeneration.forward = forward_with_normal_backend
        print("Qwen3VLMoe model.forward patched.")
    except ImportError:
        pass

    # Patch module-level _flash_attention_forward for Ulysses SP.
    # In transformers >= 4.50, flash attention is routed through a central function.
    patched = False
    qwen3_module = sys.modules.get('transformers.models.qwen3_vl.modeling_qwen3_vl')
    if qwen3_module and hasattr(qwen3_module, '_flash_attention_forward'):
        qwen3_module._flash_attention_forward = _ulysses_flash_attention_forward
        patched = True
        print("Patched _flash_attention_forward in qwen3_vl module for Ulysses SP.")

    if not patched:
        try:
            from transformers.integrations import flash_attention as _flash_attn_integrations
            _flash_attn_integrations._flash_attention_forward = _ulysses_flash_attention_forward
            patched = True
            print("Patched _flash_attention_forward in transformers.integrations.flash_attention for Ulysses SP.")
        except Exception:
            pass

    if not patched:
        print("Warning: Could not patch flash attention for Qwen3VL Ulysses SP. "
              "Sequence parallelism may not work correctly.")


_PATCH_NAME_TO_FUNC = {
    'llama': apply_monkey_patch_to_llama,
    'qwen2': apply_monkey_patch_to_qwen2,
    'qwen3_vl': apply_monkey_patch_to_qwen3vl,
    'qwen3_vl_moe': apply_monkey_patch_to_qwen3vl,
}

from transformers import PretrainedConfig


def apply_monkey_patch(config: PretrainedConfig, verbose=True):
    # For older models (llama/qwen2), enforce the version range that was tested.
    # For newer models like qwen3_vl, skip this check as they require newer transformers.
    if config.model_type not in ('qwen3_vl', 'qwen3_vl_moe'):
        if not is_transformers_version_in_range("4.45.0", "4.49.0"):
            raise AssertionError("The installed `transformers` version doesn't support ulysses patch. "
                                 "Please install a version between 4.45.0 and 4.49.0 to use this ulysses feature.")

    success_apply_monkey_patch = False
    if config.model_type in _PATCH_NAME_TO_FUNC:
        _PATCH_NAME_TO_FUNC[config.model_type]()
        success_apply_monkey_patch = True

    if success_apply_monkey_patch and verbose:
        print(f'Applying monkey patch to model {config.model_type}')
    elif not success_apply_monkey_patch:
        raise NotImplementedError(f'Ulysses for model {config.model_type} is not implemented, \
                                   please set `ulysses_sequence_parallel_size=1`')

    return success_apply_monkey_patch


from functools import lru_cache
from packaging import version
import importlib.metadata


@lru_cache()
def is_transformers_version_in_range(min_version: str, max_version: str = None) -> bool:
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        raise ModuleNotFoundError("The `transformers` package is not installed.")

    parsed = version.parse(transformers_version)
    if max_version is not None:
        return version.parse(min_version) <= parsed <= version.parse(max_version)
    return version.parse(min_version) <= parsed

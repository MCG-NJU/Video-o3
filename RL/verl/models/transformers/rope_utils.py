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
Dispatcher for get_rope_index across different VLM families.

Usage:
    from verl.models.transformers.rope_utils import get_rope_index
    position_ids = get_rope_index(processor, input_ids=..., image_grid_thw=..., attention_mask=...)
"""


def get_rope_index(processor, **kwargs):
    """
    Dispatch to the correct get_rope_index implementation based on the processor type.

    - Qwen3VL (Qwen3VLProcessor): uses timestamp-based video encoding (per-frame t=1).
    - Qwen2VL / Qwen2.5VL (Qwen2VLProcessor / Qwen2_5_VLProcessor): standard mrope.
    """
    processor_cls_name = type(processor).__name__
    if 'Qwen3VL' in processor_cls_name:
        from verl.models.transformers.qwen3_vl import get_rope_index as _fn
    else:
        from verl.models.transformers.qwen2_vl import get_rope_index as _fn
    return _fn(processor, **kwargs)

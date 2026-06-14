# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask
from dataclasses import dataclass, field


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"

class AsyncRolloutRequest:
    """The data model for async rollout."""
    def __init__(
        self,
        batch_data_id,
        rollout_offset,
        request_id,
        multi_modal_data={},
        raw_prompt_id=[],
        reward_scores={},
        max_response_len=8192,
        max_model_len=32768,
        image_size_used=[],
        raw_prompt=None,
        index="",
        uid="",
        ground_truth="",
        data_source="",
        input_ids="",
        attention_mask="",
        position_ids="",
    ):
        self.batch_data_id = batch_data_id
        self.rollout_offset = rollout_offset
        self.request_id = request_id
        self.multi_modal_data = multi_modal_data
        self.raw_prompt_id = raw_prompt_id
        self.reward_scores = reward_scores
        self.max_response_len = max_response_len
        self.max_model_len = max_model_len
        self.image_size_used = image_size_used
        self.raw_prompt = raw_prompt
        self.index = index
        self.uid = uid
        self.ground_truth = ground_truth
        self.data_source=data_source
        self.input_ids=input_ids
        self.attention_mask=attention_mask
        self.position_ids=position_ids

'''
class AsyncRolloutRequest:
    """The data model for async rollout."""

    def __init__(self, batch_data_id, rollout_offset, request_id, multi_modal_data, raw_prompt_id,):
        self.batch_data_id=
        self.rollout_offset: int = 0
        self.request_id: str=""
        self.multi_modal_data: Dict[str, Any]=None
        self.raw_prompt_id: List[int]=None
        self.reward_scores: Dict[str, float]=None
        self.max_response_len: int = 8192
        self.max_model_len: int = 32768

        self.image_size_used: List[int]=None
        self.raw_prompt: str=None
        self.index: str=""
        self.uid: str=""
        self.ground_truth: str=""
'''

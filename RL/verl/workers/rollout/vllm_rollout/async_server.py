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
import asyncio
import heapq
import importlib
import logging
import os
import random
import socket
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Tuple, Type
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp
import fastapi
import httpx
import ray
import uvicorn
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from starlette.requests import Request
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)


class AsyncLLMEngineManager:
    """AsyncLLMEngineManager manage a group of vllm instances, i.e AsyncvLLMEngine."""

    def __init__(
        self, config: DictConfig, worker_group: RayWorkerGroup, tokenizer, processor, reward_fn, val_reward_fn, *, scheduler_kwargs: Dict[str, Any] = None
    ):
        """Initialize AsyncLLMEngineManager.

        Args:
            config: DictConfig, actor_rollout_ref config.
            worker_group: RayWorkerGroup, worker group of AsyncActorRolloutRefWorker.
            scheduler_kwargs: Dict[str, Any], kwargs for chat scheduler.
        """
        self.config = config
        self.worker_group = worker_group
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.tokenizer = tokenizer
        self.processor = processor

        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size

        data_mode = os.getenv("DATA_MODE", "image")
        if data_mode == "image":
            from verl.workers.rollout.vllm_rollout.vllm_async_engine_image import AsyncvLLMEngine
            print("The data mode is image, using vllm_async_engine_image!")
        elif data_mode == "video":
            from verl.workers.rollout.vllm_rollout.vllm_async_engine_video import AsyncvLLMEngine
            print("The data mode is video, using vllm_async_engine_video!")
        else:
            raise ValueError(f"Invalid data mode: {data_mode}")
        
        engine_class = AsyncvLLMEngine
        config.rollout.max_model_len = (
            config.rollout.max_model_len
            if config.rollout.max_model_len
            else config.rollout.prompt_length + config.rollout.response_length
        )

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: engine_class.options(
                    # make sure AsyncvLLMEngine colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix, self.tokenizer, self.processor, self.reward_fn, self.val_reward_fn)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    # address = ray.get(server.get_server_address.remote())
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def wake_up(self):
        """Wake up all vllm instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all vllm instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        """Generate multiple sequences in parallel via chat scheduler."""
        # assert self.chat_scheduler is not None, "chat scheduler is not initialized."
        # import pdb; pdb.set_trace()
        num_servers = len(self.async_llm_servers)
        batch_size = len(prompts.batch)

        # 创建分片索引 [0,1,2,...,batch_size-1]
        indices = list(range(batch_size))
        chunk_size = (batch_size + num_servers - 1) // num_servers  # 向上取整

        # 将prompts分片到各个server
        result = [
            server.generate_sequences.remote(
                prompts.select_idxs(indices[i * chunk_size : (i + 1) * chunk_size]), **sampling_params  # 选择当前分片
            )
            for i, server in enumerate(self.async_llm_servers)
        ]

        # 收集并合并结果
        outputs = ray.get(result)
        return DataProto.concat(outputs)

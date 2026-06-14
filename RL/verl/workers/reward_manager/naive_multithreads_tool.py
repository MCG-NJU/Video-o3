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
import os
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
# from verl.utils.reward_score.openr1 import format_reward, acc_reward
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
# import logging
# import json

class NaiveMultiThreadsToolRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

        self.extra_info = kwargs.get("extra_info", {})
        self.gpt_threads = kwargs.get("gpt_threads", 100)

        self.overlong_buffer_len = self.extra_info.get("overlong_buffer_len", 0)
        self.max_total_response_length = self.extra_info['max_total_response_length']

    def extract_responses_list(
        self, 
        tokenizer, 
        input_ids: torch.Tensor, # 4096 + 8192
        multi_turn_response_mask: torch.Tensor # 0,0,0,...,1,1,1,...,0,0,0,...,1,1,1
    ) -> list:
        # Tensor Method
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat([multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0)
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0]
        segments = []
        for s, e in zip(starts, ends):
            segments.append(input_ids[s:e+1].tolist())

        # Decode each segment
        # decoded_responses = [tokenizer.decode(seg, skip_special_tokens=True) for seg in segments]
        # np.set_printoptions(threshold=np.inf)
        # torch.set_printoptions(threshold=float('inf'))
        # print("response_ids: ", segments)
        decoded_responses = tokenizer.batch_decode(segments, skip_special_tokens=True)
        
        return decoded_responses

    def process_single(self, args):

        index, data_item = args
        time_start = time.time()

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        # np.set_printoptions(threshold=np.inf)
        # torch.set_printoptions(threshold=float('inf'))
        # print("prompt_ids: ", valid_prompt_ids)
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        # For multi turn, we maybe need `response_str` in a list format
        if 'multi_turn_response_mask' in data_item.batch:
            # `response_str` is a list now
            response_str = self.extract_responses_list(
                self.tokenizer,
                data_item.batch['input_ids'],
                data_item.batch['multi_turn_response_mask']
            )
        else:
            # np.set_printoptions(threshold=np.inf)
            # torch.set_printoptions(threshold=float('inf'))
            # print("response_ids: ", valid_response_ids)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str = [response_str]

        ground_truth = data_item.non_tensor_batch['ground_truth']

        data_source = data_item.non_tensor_batch['data_source']

        extra_info = self.extra_info

        question = data_item.non_tensor_batch['raw_prompt']

        result = self.compute_score(
            prompt=question,
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        if isinstance(result, dict):
            return (index, result)
        else:
            score, acc_score, format_score = result

            overlong_buffer_len = self.overlong_buffer_len
            if overlong_buffer_len > 0:
                # print("e1")
                expected_len = self.max_total_response_length - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = 1.0
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                score += overlong_reward
            else:
                overlong_reward = 0

            time_end = time.time()

            return (index, score, acc_score, format_score, overlong_reward, valid_response_length, data_source, prompt_str, response_str, ground_truth, time_end - time_start)

    def __call__(self, data: DataProto, iteration=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        acc_reward_tensor = reward_tensor.clone()
        format_reward_tensor = reward_tensor.clone()
        overlong_reward_tensor = reward_tensor.clone()

        already_print_data_sources = {}

        n_threads = self.gpt_threads

        # import pdb; pdb.set_trace()

        executor = ThreadPoolExecutor(n_threads)
        futures = []
        for i in range(len(data)):
            futures.append(executor.submit(self.process_single, (i, data[i])))
        results = [f.result() for f in as_completed(futures)]

        results = sorted(results, key=lambda x: x[0])
        results = [result[1:] for result in results]

        time_consume_list = []
        invalid_uids = []
        for i, result in enumerate(results):

            if isinstance(result[0], dict) and result[0]['is_filter'] == True:
                invalid_uids.append(data[i].non_tensor_batch['uid'])
                continue
            
            (score, acc_score, format_score, overlong_score, valid_response_length, data_source, prompt_str, response_str, ground_truth, time_consume) = result
            reward_tensor[i, valid_response_length - 1] = score
            acc_reward_tensor[i, valid_response_length - 1] = acc_score
            format_reward_tensor[i, valid_response_length - 1] = format_score
            overlong_reward_tensor[i, valid_response_length - 1] = overlong_score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < 5:
                already_print_data_sources[data_source] += 1
                if isinstance(response_str, list):
                    response_str = ' '.join(response_str)
                
                current_pid = os.getpid()
                base_dir = os.getenv("LOG_SAVE_DIR", "./ckpt/RL")
                log_path = os.path.join(base_dir, f"reward_manager/{current_pid}.txt")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[prompt]: {prompt_str}\n")
                    f.write(f"[response]: {response_str}\n")
                    f.write(f"[ground_truth]: {ground_truth}\n")
                    f.write(f"[score]: {(score, acc_score, format_score)}\n")
                    # f.write(f"[overlong_reward]: {(overlong_score)}\n")
                    f.write(f"[iteration]: {iteration}\n")
                    f.write(f"[data_source]: {data_source}\n")
                    f.write(f"----------------------------------------------------------------\n\n")
                
                print(f"<<< [score]: {score} [acc_score]: {acc_score} [format_score]: {format_score} [iteration]: {iteration} [data_source]: {data_source} >>>")

            time_consume_list.append(time_consume)

        time_consume_list = sorted(time_consume_list, reverse=True)
        if len(invalid_uids) > 0:
            print(f"<<<WARNING>>> there are {len(invalid_uids)} invalid samples in this batch: {invalid_uids}")

        return reward_tensor, acc_reward_tensor, format_reward_tensor, overlong_reward_tensor, invalid_uids
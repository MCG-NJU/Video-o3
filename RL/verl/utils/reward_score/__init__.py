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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
        'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12', 'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ["vstar_bench", "visual_probe_easy", "visual_probe_medium", "visual_probe_hard", "visual_probe_train", "deepeyes_train"]:
        if extra_info['general_qa_reward_fn'] == 'general_qa_tool':
            from . import general_qa_tool
            res = general_qa_tool.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif extra_info['general_qa_reward_fn'] == 'general_qa_tool_qwen':
            from . import general_qa_tool_qwen
            res = general_qa_tool_qwen.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif extra_info['general_qa_reward_fn'] == 'general_qa_tool_mc':
            from . import general_qa_tool_mc
            res = general_qa_tool_mc.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        else:
            raise NotImplementedError
    elif data_source in ["tool_reward_without_round_penalty", "clue_multi_w_tool"]:
        from . import tool_reward_without_round_penalty
        res = tool_reward_without_round_penalty.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
    elif data_source in ["tool_free_form", "clue_multi_wo_tool"]:
        from . import tool_free_form
        res = tool_free_form.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
    elif data_source in ["tool_reward_with_round_penalty", "clue_single_w_tool"]:
        from . import tool_reward_with_round_penalty
        res = tool_reward_with_round_penalty.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
    elif data_source in ["tool_penalty_single_turn", "clue_single_wo_tool"]:
        from . import tool_penalty_single_turn
        res = tool_penalty_single_turn.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
    elif data_source in ["temporal_grounded_qa"]:
        from . import temporal_grounded_qa
        res = temporal_grounded_qa.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
    elif data_source in ["temporal_grounded_qa_iou"]:
        from . import temporal_grounded_qa_iou
        res = temporal_grounded_qa_iou.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
    elif data_source in ["test_temporal_grounded_qa", "nextgqa", "mlvu", "videomme", "cgbench", "vrbench", "videommmu", "lvbench", "longvideobench", "mmvu", "videoholmes", "longvideoreason"]:
        from . import test_temporal_grounded_qa
        res = test_temporal_grounded_qa.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
    elif data_source in ["temporal_grounding"]:
        from . import temporal_grounding
        res = temporal_grounding.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
    elif data_source in ["test_temporal_grounding", "activitynet", "charades"]:
        from . import test_temporal_grounding
        res = test_temporal_grounding.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError
    return res

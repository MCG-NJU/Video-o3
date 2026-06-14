import re
import json
import torch
import numpy as np
from my_qwen_vl_utils.vision_process import crop_video_raw, fetch_video_raw_frame, resample_video_from_raw

# Qwen2.5-VL patch_size=14 -> 28*28; Qwen3-VL patch_size=16 -> 32*32
def _pixels_per_token(patch_size: int) -> int:
    return (patch_size * 2) ** 2


def crop_video(
    raw_video,
    raw_fps,
    temporal_segment,
    sampling_strategy,
    frames_sample_fps=2.0,
    patch_size: int = 16,
    *,
    min_tokens: int = 512,
    max_tokens_coarse: int = 2048,
    max_tokens_medium: int = 4096,
    max_tokens_fine: int = 6144,
    max_tokens_per_frame_cap: int = 768,
):
    """Crop and resample video segment. Token limits can be overridden via kwargs (e.g. from rollout.crop_* config)."""
    video_path, raw_video_tensor = raw_video['path'], raw_video['tensor']
    clip_start, clip_end = temporal_segment
    if sampling_strategy == "coarse":
        max_tokens = max_tokens_coarse
    elif sampling_strategy == "medium":
        max_tokens = max_tokens_medium
    elif sampling_strategy == "fine":
        max_tokens = max_tokens_fine
    else:
        max_tokens = max_tokens_medium

    if video_path.endswith(".mp4"):
        crop_video_ele = {
            "video": raw_video_tensor,
            "fps": frames_sample_fps,
            "video_start": clip_start,
            "video_end": clip_end,
            "raw_fps": raw_fps
        }
        raw_video_crop, raw_video_crop_fps = crop_video_raw(crop_video_ele)
    else:
        crop_video_ele = {
            "video": video_path,
            "fps": frames_sample_fps,
            "video_start": clip_start,
            "video_end": clip_end,
            "source_frames_fps": raw_fps
        }
        raw_video_crop, raw_video_crop_fps = fetch_video_raw_frame(crop_video_ele)

    # resize: pixels per token = (patch_size*2)**2
    pixels_per_token = _pixels_per_token(patch_size)
    frames_num = raw_video_crop.shape[0]
    max_tokens_per_frame = max_tokens // (frames_num // 2)
    max_tokens_per_frame = min(max_tokens_per_frame, max_tokens_per_frame_cap)
    min_tokens_per_frame = min_tokens // (frames_num // 2)
    resample_ele = {
        "max_pixels": max_tokens_per_frame * pixels_per_token,
        "min_pixels": min_tokens_per_frame * pixels_per_token,
    }
    # Resample and resize video
    video_crop, video_crop_fps = resample_video_from_raw(
        raw_video_crop,
        raw_video_crop_fps,
        resample_ele,
        return_video_sample_fps=True
    )

    return video_crop, video_crop_fps


def prepare_grounding_inputs_video(decoded_resp_, observations_list, raw_video_length):
    """
    解析来自模型的时序定位（grounding_temporal）响应，并添加容错处理。
    假设 temporal_segment 为浮点型绝对坐标。
    """
    # 使用更通用的正则提取 <grounding> 标签内容
    pattern = re.compile(r'<grounding>(.*?)</grounding>', re.DOTALL)
    matches = pattern.findall(decoded_resp_)

    if not matches:
        raise ValueError(f"Could not parse grounding_temporal from response: {decoded_resp_}")
    
    # 取最后一个匹配项（通常只应该有一个，或者取最新的一个）
    grounding_content = matches[-1].strip()

    # 尝试提取 JSON 部分（从第一个 { 到最后一个 }）以处理可能的额外字符
    start_idx = grounding_content.find('{')
    end_idx = grounding_content.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        grounding_content = grounding_content[start_idx:end_idx+1]

    try:
        # 尝试直接解析 JSON
        grounding_json = json.loads(grounding_content)
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试修复单引号问题（模型有时会输出单引号）
        try:
            # 将单引号替换为双引号，但要小心不要替换字符串内部的单引号（简单的替换可能不够完美，但能处理常见情况）
            # 这里简单尝试替换键和值周围的单引号
            fixed_content = grounding_content.replace("'", '"')
            grounding_json = json.loads(fixed_content)
        except json.JSONDecodeError as e:
             raise ValueError(f"Failed to parse grounding JSON: {grounding_content}. Error: {e}")

    # 提取字段
    if "temporal_segment" not in grounding_json:
        raise ValueError(f"Missing 'temporal_segment' in grounding content: {grounding_json}")
    if "sampling_strategy" not in grounding_json:
         raise ValueError(f"Missing 'sampling_strategy' in grounding content: {grounding_json}")
    
    temporal_segment = grounding_json["temporal_segment"]
    sampling_strategy = grounding_json["sampling_strategy"]

    if not (isinstance(temporal_segment, (list, tuple)) and len(temporal_segment) == 2):
        raise ValueError(f"temporal_segment must be a list or tuple with 2 elements, but got: {temporal_segment}")
    if not (isinstance(temporal_segment[0], (int, float)) and isinstance(temporal_segment[1], (int, float))):
        raise ValueError(f"temporal_segment coordinates must be numbers (int or float), but got: {temporal_segment}")

    t_start = temporal_segment[0]
    t_end = temporal_segment[1]
    
    if t_start < 0.0:
        t_start = 0.0
        print(f"<<<Warning>>> t_start: {t_start} < 0.0, set t_start to 0.0")
    if t_end > raw_video_length:    # 1e-6 is for numerical stability
        t_end = raw_video_length
        print(f"<<<Warning>>> t_end: {t_end} > raw_video_length, set t_end to raw_video_length: {raw_video_length}")
    
    temporal_segment = [t_start, t_end]
    assert temporal_segment[0] <= temporal_segment[1] - 1, \
        f"Invalid temporal bounding box: {temporal_segment}. 'time_start' must be at least 1.0 unit less than 'time_end' (time_start <= time_end - 1)."

    assert isinstance(temporal_segment, list) and len(temporal_segment) == 2 ,"temporal_segment must be a list with 2 elements!"
    assert isinstance(temporal_segment[0], (int, float)) and isinstance(temporal_segment[1], (int, float)), "temporal_segment coordinates must be numbers (int or float)!"
    assert isinstance(sampling_strategy, str) and sampling_strategy in ["coarse", "fine", "medium"], "sampling_strategy must be a string and in ['coarse', 'fine', 'medium']!"
    tool_type = "grounding_temporal"
    return tool_type, (temporal_segment, sampling_strategy)


def get_valid_mask(uids, multi_modal_data, meta_info, reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids, exceed_mask=None):

    statistics_dict = {
        "solve_none": 0,
        "solve_all": 0,
        "solve_acc_none": 0,
        "solve_acc_all": 0,
        "solve_format_none": 0,
        "solve_format_all": 0,
        "totals": 0,
        "success_tool_call_per_batch": 0,
        "success_tool_call_per_sample": 0,
        "use_tool_and_direct_answer_num": 0,
        "wrong_and_use_tool_num": 0,
        "wrong_and_direct_answer_num": 0,
        "wrong_and_use_tool_and_direct_answer_num": 0,
        "only_use_tool_num": 0,
        "only_direct_answer_num": 0,
        "only_wrong_num": 0,
        "num_0_0": 0, # direct 全错，tool 全错,
        "num_0_1": 0, # direct 全错，tool 全对,
        "num_0_2": 0, # direct 全错，tool 有对有错,
        "num_0_k": 0, # direct 全错，tool 不存在,
        "num_1_0": 0,
        "num_1_1": 0,
        "num_1_2": 0,
        "num_1_k": 0,
        "num_2_0": 0,
        "num_2_1": 0,
        "num_2_2": 0,
        "num_2_k": 0,
        "num_k_0": 0,
        "num_k_1": 0,
        "num_k_2": 0,
    }
    unique_uids = np.unique(uids)
    valid_mask = torch.ones(len(uids), dtype=torch.bool)
    invalid_mask = torch.zeros(len(uids), dtype=torch.bool)

    wrong_and_use_tool_uids = []
    wrong_and_direct_answer_uids = []

    for uid in unique_uids:
        uid_mask = uids == uid

        uid_acc_rewards = acc_reward_tensor[uid_mask].sum(-1)
        uid_format_rewards = format_reward_tensor[uid_mask].sum(-1)

        if meta_info['tool_call'] in ['crop']:
            uid_multi_modal_data = multi_modal_data[uid_mask]
            success_tool_call_mask = torch.tensor([
                isinstance(item, dict) and len(item.get('video', [])) > 1
                for item in uid_multi_modal_data
            ]).to(uid_acc_rewards.device)
            success_tool_call_count = success_tool_call_mask.sum().item()
            statistics_dict["success_tool_call_per_batch"] += success_tool_call_count
            statistics_dict["success_tool_call_per_sample"] += (success_tool_call_count > 0)
            # Compute Reward Distribution
            use_tool_correct_answer_mask = (uid_acc_rewards > 0) & success_tool_call_mask
            use_tool_correct_answer_count = use_tool_correct_answer_mask.sum().item()
            direct_answer_correct_answer_mask = (uid_acc_rewards > 0) & (~success_tool_call_mask)
            direct_answer_correct_answer_count = direct_answer_correct_answer_mask.sum().item()

            use_tool_wrong_answer_mask = (uid_acc_rewards == 0) & success_tool_call_mask
            use_tool_wrong_answer_count = use_tool_wrong_answer_mask.sum().item()
            direct_answer_wrong_answer_mask = (uid_acc_rewards == 0) & (~success_tool_call_mask)
            direct_answer_wrong_answer_count = direct_answer_wrong_answer_mask.sum().item()

            wrong_answer_count = (uid_acc_rewards == 0).sum().item()
            
            assert use_tool_wrong_answer_count + direct_answer_wrong_answer_count == wrong_answer_count

            if direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_0_0"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_0_1"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_0_2"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_0_k"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_1_0"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_1_1"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_1_2"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_1_k"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_2_0"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_2_1"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_2_2"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_2_k"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_k_0"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_k_1"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_k_2"] += 1

            if use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count == 0:
                statistics_dict["use_tool_and_direct_answer_num"] += 1
            elif use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count == 0 and wrong_answer_count > 0:
                statistics_dict["wrong_and_use_tool_num"] += 1
                wrong_and_use_tool_uids.append(uid)
            elif use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count == 0 and wrong_answer_count == 0:
                statistics_dict["only_use_tool_num"] += 1
            elif use_tool_correct_answer_count == 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count > 0:
                statistics_dict["wrong_and_direct_answer_num"] += 1
                wrong_and_direct_answer_uids.append(uid)
            elif use_tool_correct_answer_count == 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count == 0:
                statistics_dict["only_direct_answer_num"] += 1
            elif use_tool_correct_answer_count == 0 and direct_answer_correct_answer_count == 0 and wrong_answer_count > 0:
                statistics_dict["only_wrong_num"] += 1
            elif use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count > 0:
                statistics_dict["wrong_and_use_tool_and_direct_answer_num"] += 1
            else:
                raise NotImplementedError(f"use_tool_correct_answer_count: {use_tool_correct_answer_count}, direct_answer_correct_answer_count: {direct_answer_correct_answer_count}, wrong_answer_count: {wrong_answer_count}.")

        if uid in invalid_uids:
            valid_mask[uid_mask] = False
            invalid_mask[uid_mask] = True
            continue

        uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

        if exceed_mask is not None:
            old_uid_rewards = uid_rewards
            exceed_mask_uid = exceed_mask[uid_mask]
            uid_rewards = uid_rewards[~exceed_mask_uid]
            print(f"len(old_uid_rewards): {len(old_uid_rewards)}, len(uid_rewards): {len(uid_rewards)}")

        if uid_rewards.std().item() < 1e-4:
            valid_mask[uid_mask] = False
            invalid_mask[uid_mask] = True

        if (uid_rewards == 0).all():
            statistics_dict["solve_none"] += 1
        elif (uid_rewards == meta_info['acc_reward_weight'] + meta_info['format_reward_weight'] + meta_info['use_tool_reward_weight']).all():
            statistics_dict["solve_all"] += 1

        # Check if all rewards are 0 or all are 1 for this uid
        if (uid_acc_rewards == 0).all():
            statistics_dict["solve_acc_none"] += 1
        elif (uid_acc_rewards == meta_info['acc_reward_weight']).all():
            statistics_dict["solve_acc_all"] += 1

        if (uid_format_rewards == 0).all():
            statistics_dict["solve_format_none"] += 1
        elif (uid_format_rewards == meta_info['format_reward_weight']).all():
            statistics_dict["solve_format_all"] += 1

        statistics_dict["totals"] += 1
    return valid_mask, invalid_mask, wrong_and_use_tool_uids, wrong_and_direct_answer_uids, statistics_dict

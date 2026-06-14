import re
import json

def inner_acc_reward(prompt: str, predict_str_list: list, original_answer: str, use_gpt=False, gpt_extract_answer=False, extra_info=None):

    original_answer = original_answer.strip()
    if isinstance(predict_str_list, list):
        predict_str = predict_str_list[-1].strip()
    else:
        assert isinstance(predict_str_list, str)
        predict_str = predict_str_list.strip()
    
    # extract_answer_pattern = r'<answer>(.*?)</answer>'
    # match = re.search(extract_answer_pattern, predict_str, re.DOTALL)
    extract_answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    matches = extract_answer_pattern.findall(predict_str)
    if len(matches) > 0:
        predict_str = matches[-1]
    else:
        return 0.0
    
    # print(f"predict_str: {predict_str}, original_answer: {original_answer}")

    # exact match
    if predict_str.strip() == original_answer:
        return 1.0
    
    # match format like "(A) xxx"
    # pattern = re.compile(r'\(([A-Z])\).*', re.DOTALL)
    # all_matches = pattern.findall(predict_str)
    # if len(all_matches) == 1 and all_matches[0].strip() == original_answer:
    #     return 1.0
    pattern = r'^\(([A-Z])\).*$'
    match = re.match(pattern, predict_str, re.DOTALL)
    if match and match.group(1) == original_answer:
        return 1.0

    # match format like "A. xxx"
    pattern = r'^([A-Z])\..*$'
    match = re.match(pattern, predict_str, re.DOTALL)
    if match and match.group(1) == original_answer:
        return 1.0

    return 0.0

def acc_reward(prompt: str, predict_str_list: list, solution: str, extra_info: dict = None) -> float:
    gpt_extract_answer = extra_info.get("gpt_extract_answer", False)
    try:
        reward = inner_acc_reward(prompt, predict_str_list, solution, use_gpt=True, gpt_extract_answer=gpt_extract_answer, extra_info=extra_info)
    except Exception as e:
        print("<<<ERROR>>> Error in inner_acc_reward when processing prompt: ", prompt, "predict_str_list: ", predict_str_list, "solution: ", solution, "error: ", str(e))
        return {"is_filter": True, "info": str(e)}
    return reward



def grounding_format_reward(predict_str_list: list):
    """
    Validate the format of predict_str_list according to the rules:
    1. Last string must follow <think>...</think><answer>...</answer>
    2. Other strings must follow <think>...</think><grounding>...</grounding>
    3. Each string must start with <think> and end with </answer> or </answer>\n, or </grounding> or </grounding>\n
    4. Between </think> and <answer> can only be empty or a single newline
    5. <grounding> content should be valid JSON with temporal_segment and sampling_strategy (errors reduce score)
    """
    if not isinstance(predict_str_list, list) or len(predict_str_list) == 0:
        return 0.0, 0
    
    total_strings = len(predict_str_list)
    penalty_per_error = 1.0 / total_strings
    error_count = 0
    tool_call_count = 0
    
    # Check all strings except the last one (should have grounding format)
    for i in range(len(predict_str_list) - 1):
        predict_str = predict_str_list[i]
        if not isinstance(predict_str, str):
            return 0.0, 0
        
        # Pattern for grounding format: <think>...</think>(\n?)<grounding>...</grounding>(\n?)
        # Allow optional \n at the start
        pattern = re.compile(r'^\n?<think>(.*?)</think>(\n?)<grounding>(.*?)</grounding>(\n?)$', re.DOTALL)
        match = pattern.match(predict_str)
        
        if not match:
            return 0.0, 0
        
        # Check that between </think> and <grounding> is empty or single \n
        middle_content = match.group(2)
        if middle_content and middle_content != '\n':
            return 0.0, 0
        
        # Validate grounding JSON content (count errors instead of returning 0)
        grounding_content = match.group(3)
        try:
            grounding_json = json.loads(grounding_content)
            # Check required fields
            if "temporal_segment" not in grounding_json:
                error_count += 1
            if "sampling_strategy" not in grounding_json:
                error_count += 1
            else:
                # Check sampling_strategy value
                strategy = grounding_json["sampling_strategy"]
                if not isinstance(strategy, str) or strategy not in ['coarse', 'fine', 'medium']:
                    error_count += 1
            # Check temporal_segment format
            if "temporal_segment" in grounding_json:
                temporal_seg = grounding_json["temporal_segment"]
                if not isinstance(temporal_seg, list) or len(temporal_seg) != 2:
                    error_count += 1
        except (json.JSONDecodeError, TypeError):
            # JSON parsing error counts as one error
            error_count += 1
        
        tool_call_count += 1
    
    # Check the last string (should have answer format)
    last_str = predict_str_list[-1]
    if not isinstance(last_str, str):
        return 0.0, 0
    
    # Pattern for answer format: <think>...</think>(\n?)<answer>...</answer>(\n?)
    # Allow optional \n at the start
    pattern = re.compile(r'^\n?<think>(.*?)</think>(\n?)<answer>(.*?)</answer>(\n?)$', re.DOTALL)
    match = pattern.match(last_str)
    
    if not match:
        return 0.0, 0
    
    # Check that between </think> and <answer> is empty or single \n
    middle_content = match.group(2)
    if middle_content and middle_content != '\n':
        return 0.0, 0
    
    # Calculate final score: 1.0 - (error_count * penalty_per_error)
    final_score = max(0.0, 1.0 - (error_count * penalty_per_error))
    return final_score, tool_call_count

def calculate_grounding_acc(predict_str_list: list, ground_truth: list):
    """
    Calculate IoU between predicted temporal segments and ground truth segments.
    1. Extract predicted segments from predict_str_list.
    2. Extract ground truth segments from ground_truth.
    3. Merge overlapping intervals within prediction and ground truth respectively.
    4. Calculate Intersection over Union.
    """
    # 1. Extract predicted intervals
    pred_intervals = []
    grounding_pattern = re.compile(r'<grounding>(.*?)</grounding>', re.DOTALL)
    
    for pred_str in predict_str_list:
        if not isinstance(pred_str, str):
            continue
        matches = grounding_pattern.findall(pred_str)
        for match in matches:
            try:
                data = json.loads(match)
                if "temporal_segment" in data:
                    seg = data["temporal_segment"]
                    if isinstance(seg, list) and len(seg) == 2:
                        # Ensure start <= end
                        start, end = float(seg[0]), float(seg[1])
                        pred_intervals.append([min(start, end), max(start, end)])
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    # 2. Extract ground truth intervals
    gt_intervals = []
    if isinstance(ground_truth, dict):
        # Handle format: {"clue": [{"timestamp": [start, end], ...}, ...]}
        if "clue" in ground_truth:
            for item in ground_truth["clue"]:
                if isinstance(item, dict) and "timestamp" in item:
                    ts = item["timestamp"]
                    if isinstance(ts, list) and len(ts) == 2:
                        if ts[0] == ts[1] or float(ts[0]) > float(ts[1]):
                            print(f"<<<WARNING>>> Invalid ground truth interval: {ts}, skip it.")
                            continue
                        start, end = float(ts[0]), float(ts[1])
                        gt_intervals.append([min(start, end), max(start, end)])
    elif isinstance(ground_truth, list):
        # Handle format: [[start, end], ...] or [{"timestamp": [start, end]}, ...]
        for item in ground_truth:
            if isinstance(item, list) and len(item) == 2:
                if item[0] == item[1] or float(item[0]) > float(item[1]):
                    print(f"<<<WARNING>>> Invalid ground truth interval: {item}, skip it.")
                    continue                
                start, end = float(item[0]), float(item[1])
                gt_intervals.append([min(start, end), max(start, end)])
            elif isinstance(item, dict) and "timestamp" in item:
                ts = item["timestamp"]
                if isinstance(ts, list) and len(ts) == 2:
                    if ts[0] == ts[1] or float(ts[0]) > float(ts[1]):
                        print(f"<<<WARNING>>> Invalid ground truth interval: {ts}, skip it.")
                        continue
                    start, end = float(ts[0]), float(ts[1])
                    gt_intervals.append([min(start, end), max(start, end)])

    if not pred_intervals or not gt_intervals:
        return 0.0

    # 3. Merge intervals function
    def merge_intervals(intervals):
        if not intervals:
            return []
        # Sort by start time
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or interval[0] > merged[-1][1]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged

    merged_pred = merge_intervals(pred_intervals)
    merged_gt = merge_intervals(gt_intervals)

    # 4. Calculate Intersection
    intersection = 0.0
    for p in merged_pred:
        for g in merged_gt:
            start = max(p[0], g[0])
            end = min(p[1], g[1])
            if end > start:
                intersection += end - start
    
    # 5. Calculate Union
    pred_len = sum(p[1] - p[0] for p in merged_pred)
    gt_len = sum(g[1] - g[0] for g in merged_gt)
    union = pred_len + gt_len - intersection

    iou = 0.0
    if union > 1e-6:
        iou = intersection / union
    
    iop = 0.0
    if pred_len > 1e-6:
        iop = intersection / pred_len
    
    iog = 0.0
    if gt_len > 1e-6:
        iog = intersection / gt_len
    # print(f"<<<DEBUG>>>  calculate_grounding_acc  IoU: {iou}, IoP: {iop}, IoG: {iog}")
    return (2*iou + iop + iog) / 4.0

def compute_score(prompt: str, predict_str_list: list, ground_truth: list, extra_info: dict = None) -> float:
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0
    decay_penalty_weight = extra_info.get('decay_penalty_weight', 0.0)

    acc = acc_reward(prompt, predict_str_list, ground_truth["answer"], extra_info)
    if isinstance(acc, dict):
        return acc
    format_score, tool_call_count = grounding_format_reward(predict_str_list)

    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format_score

    tool_reward = 0
    if tool_call_count > 0 and format_score == 1 and acc_score == 1: #正确调用工具的同时答对问题，给予额外奖励
        tool_score = 1.0  #多轮情况下tool调用次数多了无惩罚，奖励仅取决于线索覆盖率
        grounding_score = calculate_grounding_acc(predict_str_list, ground_truth)
        correct_answer_tool_bonus = 1
        tool_reward = tool_score * (grounding_score + correct_answer_tool_bonus) / 2
    score = acc_score + format_score + tool_reward

    return score, acc_score, format_score

if __name__ == '__main__':
    question = "Elena Ferrante" #"<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: How many states are represented by the lightest color on the map?" #"<image>What is the output score when the first input is 4 and the second input is 5 according to the Hamlet Evaluation System shown in Figure 2?" #"<image>Who wrote this book?\nAnswer the question with a short phrase."
    predict_str = [
        '<think>Let\'s analyze the video segment around the middle of the video to determine what the girl did. The video shows a girl in a pink jacket standing near a boy who is playing with a toy phone. The girl is seen moving her hands and appears to be interacting with the boy. She seems to be walking away from the boy and towards the camera.</think><grounding>{"temporal_segment": [39.9, 47.9], "sampling_strategy": "medium"}</grounding>', 
        '<think>Let\'s analyze the video segment around the middle of the video to determine what the girl did. The video shows a girl in a pink jacket standing near a boy who is playing with a toy phone. The girl is seen moving her hands and appears to be interacting with the boy. She seems to be walking away from the boy and towards the camera.</think><grounding>{"temporal_segment": [47.9, 50.9], "sampling_strategy": "medium"}</grounding>', 
        '<think>After the girl walks away from the boy and towards the camera, she is seen in the foreground, and the boy is in the background. The girl is wearing a pacifier and appears to be moving her hands, possibly gesturing or talking. The boy is still playing with the toy phone and seems to be engaged in his own activity.</think><answer>A</answer>'
    ]
    predict_str2 = [
        '<think>After the girl walks away from the boy and towards the camera, she is seen in the foreground, and the boy is in the background. The girl is wearing a pacifier and appears to be moving her hands, possibly gesturing or talking. The boy is still playing with the toy phone and seems to be engaged in his own activity.</think><answer>A</answer>'
    ]
    predict_str3 = [
        """<think>I will analyze the video segment around 80 seconds to 90 seconds to understand the context and actions of the man and the baby girl. The man is sitting on the floor and the baby girl is standing on the chair. It seems like the man is leaning towards the baby girl, which could be for support or interaction.</think>
<grounding>{"temporal_segment": [80, 90], "sampling_strategy": "fine"}</grounding>""",
        """<think>Based on the description provided, after the girl in red approaches the boy with the sparkler, there is no clear instruction on what specific action she takes. The options provided suggest possible outcomes, but they seem irrelevant to the immediate frame provided. The question may be asking based on the general movement and behavior shown in the video. Since there's no direct indication of an option being correct, we can infer the more plausible response given the context and the options presented.</think><answer>E</answer>"""
    ]
    ground_truth = {
        "answer": "C",
        "clue": [
            {
                "timestamp": [
                    20.9,
                    22.7
                ],
                "text": ""
            },
            {
                "timestamp": [
                    23.8,
                    26.0
                ],
                "text": ""
            },
            {
                "timestamp": [
                    34.4,
                    35.8
                ],
                "text": ""
            }
        ]
    }
    extra_info = {
        "acc_reward_weight": 1.0,
        "format_reward_weight": 1.0,
        "use_tool_reward_weight": 0.5,
        "gpt_extract_answer": True,
        "extract_answer_tags": "strict",
    }
    # s1 = compute_score(question, predict_str, ground_truth, extra_info)
    # print(s1)

    s3 = grounding_format_reward(predict_str3)
    print(f"Format score for predict_str3: {s3}")

    iou = calculate_grounding_acc(predict_str3, ground_truth)
    print(f"IoU for predict_str3: {iou}")

    # Test with custom simple case
    test_pred = ['<grounding>{"temporal_segment": [0, 10], "sampling_strategy": "fine"}</grounding>', 
                 '<grounding>{"temporal_segment": [5, 15], "sampling_strategy": "fine"}</grounding>']
    test_gt = [[0, 5], [10, 20]] 
    score_test = calculate_grounding_acc(test_pred, test_gt)
    print(f"Test Score: {score_test}")
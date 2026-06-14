import re

def is_valid_direct_answer(response, direct_answer_format) -> bool:
    """
    对 <think>...</think><answer>...</answer> 的形式进行校验：
      1) 是否整体匹配大体结构
      2) 是否只出现一次 <think>...</think> 和 <answer>...</answer>
      3) 不应包含 <tool_call> </tool_call>
    """
    pattern = direct_answer_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    # 3). <tool_call> </tool_call> is not allowed!
    if '<tool_call>' in response or '</tool_call>' in response:
        return False
    return True

def is_valid_direct_answer_grounding(response, direct_answer_format) -> bool:
    """
    对 <think>...</think><answer>...</answer> 的形式进行校验：
      1) 是否整体匹配大体结构
      2) 是否只出现一次 <think>...</think> 和 <answer>...</answer>
      3) 不应包含 <tool_call> </tool_call>
    """
    pattern = direct_answer_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    # 3). <tool_call> </tool_call> is not allowed!
    if '<grounding>' in response or '</grounding>' in response:
        return False
    return True

def is_valid_tool_call(response, step_tool_call_format) -> bool:
    """
    对 <think>...</think>...<tool_call>...</tool_call> 的形式进行校验：
      1) 整体正则匹配
      2) <think>...</think> 各出现一次
      3) <tool_call>...</tool_call> 只出现一次
      4) 不应出现 <answer> </answer>
    """
    pattern = step_tool_call_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <think> Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    # 3). <tool_call> </tool_call> Count
    if response.count('<tool_call>') != 1 and response.count('</tool_call>') != 1:
        return False
    # 4). <answer> or </answer> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    return True

def is_valid_tool_call_grounding(response, step_tool_call_format) -> bool:
    """
    对 <think>...</think>...<tool_call>...</tool_call> 的形式进行校验：
      1) 整体正则匹配
      2) <think>...</think> 各出现一次
      3) <tool_call>...</tool_call> 只出现一次
      4) 不应出现 <answer> </answer>
    """
    pattern = step_tool_call_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <think> Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    # 3). <tool_call> </tool_call> Count
    if response.count('<grounding>') != 1 and response.count('</grounding>') != 1:
        return False
    # 4). <answer> or </answer> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    return True

def format_reward(predict_str_list: list, extra_info: dict = None):
    """
    Check if the model's response follows the required formats and return a reward.
    [1-turn]:
        - Direct Answer
    [2-turn]:
        - Call Image Resize Tool + Answer
    Args:
    - predict_str_list (list): A list of responses, currently, max length of `predict_str_list` is 10 (10-turn), max image num is 2.
    Returns:
    - format_score: float, 1.0 for right format, 0.0 for wrong
    - tool_call_count: int, times of function tools called
    """
    conv_rounds = len(predict_str_list)
    format_score, tool_call_count = 0, 0
    # All allowed formats
    direct_answer_format = r'^<think>.*</think>.*<answer>.*</answer>$'
    step_tool_call_format = r'^<think>.*</think>.*<tool_call>.*</tool_call>$'
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    # HACK/FIXME: We need more flexible judge in the future
    # 1-turn
    if conv_rounds == 1:
        response = predict_str_list[0].strip()
        tool_call_contents = tool_call_pattern.findall(response)
        if len(tool_call_contents) > 0:
            tool_call_count += 1
        # Direct Answer
        if is_valid_direct_answer(response, direct_answer_format):
            format_score = 1
    # multi-turn
    else:
        tool_call_match_flag = True
        for response in predict_str_list[:-1]:
            response = response.strip()
            tool_call_contents = tool_call_pattern.findall(response)
            if len(tool_call_contents) > 0:
                tool_call_count += 1
            # Call Function Tool
            if not is_valid_tool_call(response, step_tool_call_format):
                tool_call_match_flag = False
                break
        final_answer_match_flag = is_valid_direct_answer(predict_str_list[-1], direct_answer_format)
        if tool_call_match_flag and final_answer_match_flag:
            format_score = 1
    return format_score, tool_call_count


def grounding_format_reward(predict_str_list: list, extra_info: dict = None):
    """
    Check if the model's response follows the required formats and return a reward.
    [1-turn]:
        - Direct Answer
    [2-turn]:
        - Call Image Resize Tool + Answer
    Args:
    - predict_str_list (list): A list of responses, currently, max length of `predict_str_list` is 10 (10-turn), max image num is 2.
    Returns:
    - format_score: float, 1.0 for right format, 0.0 for wrong
    - tool_call_count: int, times of function tools called
    """
    conv_rounds = len(predict_str_list)
    format_score, tool_call_count = 0, 0
    # All allowed formats
    direct_answer_format = r'^<think>.*</think>.*<answer>.*</answer>$'
    step_tool_call_format = r'^<think>.*</think>.*<grounding>.*</grounding>$'
    tool_call_pattern = re.compile(r'<grounding>(.*?)</grounding>', re.DOTALL)
    # HACK/FIXME: We need more flexible judge in the future
    # 1-turn
    if conv_rounds == 1:
        response = predict_str_list[0].strip()
        tool_call_contents = tool_call_pattern.findall(response)
        if len(tool_call_contents) > 0:
            tool_call_count += 1
        # Direct Answer
        if is_valid_direct_answer_grounding(response, direct_answer_format):
            format_score = 1
    # multi-turn
    else:
        tool_call_match_flag = True
        for response in predict_str_list[:-1]:
            response = response.strip()
            tool_call_contents = tool_call_pattern.findall(response)
            if len(tool_call_contents) > 0:
                tool_call_count += 1
            # Call Function Tool
            if not is_valid_tool_call_grounding(response, step_tool_call_format):
                tool_call_match_flag = False
                break
        final_answer_match_flag = is_valid_direct_answer_grounding(predict_str_list[-1], direct_answer_format)
        if tool_call_match_flag and final_answer_match_flag:
            format_score = 1

    return format_score, tool_call_count


def wer(reference, hypothesis):
        """
        计算词错误率 (WER)。
        (*** 已修改为对称版本 ***)
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        
        # *** 关键修改 ***
        # 使用 m 和 n 中的最大值作为分母，使度量对称
        return  d[m][n] / max(1, m, n)

def inner_acc_reward(prompt: str, predict_str_list: list, original_answer: str, use_gpt=False, gpt_extract_answer=False, extra_info=None):
    """
    基于规则计算准确性得分，无需网络连接。
    (*** 规则 4 已被替换为 WER 检查 ***)

    Args:
        prompt (str): 原始问题。
        predict_str_list (list): 包含模型所有回合输出的列表。
        original_answer (str): 标准答案。
        use_gpt (bool): 此参数在此版本中被忽略。
        gpt_extract_answer (bool): 是否从 <answer> 标签中提取答案。
        extra_info (dict): 额外配置信息。

    Returns:
        float: 1.0 代表正确，0.0 代表错误。
    """
    # --- 步骤 1: 从模型完整输出中提取答案 ---
    original_predict_str = ' '.join(predict_str_list)
    predicted_answer = ""
    if gpt_extract_answer:
        extract_answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(extract_answer_pattern, original_predict_str, re.DOTALL)
        if match:
            predicted_answer = match.group(1).strip()
        else:
            return 0.0
    else:
        predicted_answer = original_predict_str.strip()

    # --- 步骤 2: 预处理字符串，为比较做准备 ---
    pred_clean = predicted_answer.lower().strip()
    gt_clean = original_answer.lower().strip()

    if not pred_clean:
        return 0.0

    # --- 步骤 3: 应用基于规则的判断逻辑 ---

    # 规则 1: 精确匹配 (最严格)
    if pred_clean == gt_clean:
        return 1.0

    # 规则 2: 数字匹配
    pred_nums = re.findall(r'[-+]?\d*\.\d+|\d+', pred_clean)
    gt_nums = re.findall(r'[-+]?\d*\.\d+|\d+', gt_clean)
    if gt_nums and pred_nums:
        try:
            if {float(n) for n in pred_nums} == {float(n) for n in gt_nums}:
                return 1.0
        except ValueError:
            pass

    # 规则 3: 布尔（是/否）匹配
    yes_words = {'yes', 'correct', 'true', 'right', 'affirmative'}
    no_words = {'no', 'incorrect', 'false', 'wrong', 'negative'}
    
    # *** 注意：这两个变量将被规则4复用 ***
    gt_no_punct = re.sub(r'[^\w\s]', ' ', gt_clean)
    pred_no_punct = re.sub(r'[^\w\s]', ' ', pred_clean)

    gt_words = set(gt_no_punct.split())
    pred_words = set(pred_no_punct.split())
    
    if gt_words.intersection(yes_words):
        if pred_words.intersection(yes_words):
            return 1.0
    
    if gt_words.intersection(no_words):
        if pred_words.intersection(no_words):
            return 1.0

    if not gt_words and not pred_words:
        # 如果两个词集都为空，我们必须检查原始输入。
        # 如果一个是真的空 ("")，而另一个只是标点 ("."), 它们不应该匹配。
        
        # (bool(gt_clean) is False) != (bool(pred_clean) is False)
        # 等价于：一个是空，另一个非空
        if (not gt_clean) != (not pred_clean):
             return 0.0 # BUG 场景：gt="" vs pred="."
    
    # --- 规则 4: 词错误率 (WER) 匹配 (*** 已按要求替换 ***) ---
    # 使用规则3中已清理（去除标点）的字符串进行比较。
    # 阈值设为 0.0，要求词汇、顺序完全一致（忽略大小写和标点）。
    # 这修复了 "purple" in "... purple ..." 的漏洞。
    # 根据您的要求，此规则始终运行，不受 gpt_extract_answer 影响。
    wer_score = wer(gt_no_punct, pred_no_punct)
    return 1.0 - wer_score


def acc_reward(prompt: str, predict_str_list: list, solution: str, extra_info: dict = None) -> float:
    gpt_extract_answer = extra_info.get("gpt_extract_answer", False)
    reward = inner_acc_reward(prompt, predict_str_list, solution, use_gpt=True, gpt_extract_answer=gpt_extract_answer, extra_info=extra_info)
    return reward

def compute_score(prompt: str, predict_str_list: list, ground_truth: list, extra_info: dict = None) -> float:
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0
    decay_penalty_weight = 0.1
    if extra_info is not None and 'decay_penalty_weight' in extra_info:
        decay_penalty_weight = extra_info.get('decay_penalty_weight', 0.1)
    acc = acc_reward(prompt, predict_str_list, ground_truth, extra_info)
    if isinstance(acc, dict):
        return acc
    format_score, tool_call_count = grounding_format_reward(predict_str_list, extra_info)

    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format_score
    
    tool_penalty_factor = (1 - decay_penalty_weight) if tool_call_count > 0 else 1.0
    tool_reward = extra_info.get('use_tool_reward_weight', 0.0) if tool_call_count > 0 else 0.0
    score = tool_penalty_factor * acc_score + format_score + tool_reward

    return score, acc_score, format_score

if __name__ == '__main__':
    question = "What color is the sign?" #"<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: How many states are represented by the lightest color on the map?" #"<image>What is the output score when the first input is 4 and the second input is 5 according to the Hamlet Evaluation System shown in Figure 2?" #"<image>Who wrote this book?\nAnswer the question with a short phrase."
    predict_str = ["""<answer>TWEAKS</answer>"""]
    ground_truth = "TWEAKS"
    extra_info = {
        "acc_reward_weight": 1.0,
        "format_reward_weight": 0.5,
        "use_tool_reward_weight": 0.5,
        "gpt_extract_answer": True,
        "extract_answer_tags": "strict",
    }
    s1 = acc_reward(question, predict_str, ground_truth, extra_info)
    print(s1)

    # s2 = grounding_format_reward(predict_str, extra_info)
    # print(s2)
    
    
# if __name__ == '__main__':
#     question = "What is the price on the tag?"
#     ground_truth = "The price is $25.50."
#     extra_info = {"gpt_extract_answer": True}

#     # 案例 1: 数字匹配成功
#     predict_str_1 = ["<think>I need to find the price.</think><answer>It is 25.5.</answer>"]
#     score_1 = acc_reward(question, predict_str_1, ground_truth, extra_info)
#     print(f"案例 1 (数字匹配): 预测: '{predict_str_1[0]}', 得分: {score_1}") # 应该得分 1.0

#     # 案例 2: 子串匹配成功
#     predict_str_2 = ["<think>I see a price tag.</think><answer>The price is $25.50, which is a great deal.</answer>"]
#     score_2 = acc_reward(question, predict_str_2, ground_truth, extra_info)
#     print(f"案例 2 (子串匹配): 预测: '{predict_str_2[0]}', 得分: {score_2}") # 应该得分 1.0

#     # 案例 3: 匹配失败
#     predict_str_3 = ["<think>I think the price is around 30.</think><answer>I estimate it to be $30.</answer>"]
#     score_3 = acc_reward(question, predict_str_3, ground_truth, extra_info)
#     print(f"案例 3 (匹配失败): 预测: '{predict_str_3[0]}', 得分: {score_3}") # 应该得分 0.0

#     # 案例 4: 布尔匹配成功
#     question_bool = "Is the light on?"
#     ground_truth_bool = "Yes"
#     predict_str_4 = ["<think>The light is glowing.</think><answer>It is correct, the light is on.</answer>"]
#     score_4 = acc_reward(question_bool, predict_str_4, ground_truth_bool, extra_info)
#     print(f"案例 4 (布尔匹配): 预测: '{predict_str_4[0]}', 得分: {score_4}") # 应该得分 1.0
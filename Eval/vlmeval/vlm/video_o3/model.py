from __future__ import annotations

import os
import sys
import warnings
import math
import logging
import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, DynamicCache, StaticCache

from ..base import BaseModel
from .prompt import VideoO3PromptMixin
from ...smp import get_gpu_memory, listinstr
from ...dataset import DATASET_MODALITY
import copy

VLLM_MAX_IMAGE_INPUT_NUM = 24


def ensure_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid url: {image}')

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type


def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)


def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        import tempfile
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images


def count_video_pad_tokens(processor, input_ids):
    """Count <|video_pad|> placeholder tokens in input_ids (Qwen2.5-VL / Qwen3-VL)."""
    tok = processor.tokenizer
    vid = getattr(tok, 'video_token_id', None)
    if vid is None:
        token_str = getattr(tok, 'video_token', None) or '<|video_pad|>'
        try:
            vid = tok.convert_tokens_to_ids(token_str)
        except Exception:
            return 0
        if isinstance(vid, list):
            vid = vid[0] if vid else None
    if vid is None:
        return 0
    row = input_ids[0]
    return int((row == vid).sum().item())


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


CHAT_TEMPLATE = """
{% set image_count = namespace(value=0) %}
{% set video_count = namespace(value=0) %}
{% for message in messages %}
    {% set has_video_placeholder = namespace(value=False) %}
    {% set has_image_placeholder = namespace(value=False) %}
    {% if message['content'] is not string %}
        {% for content in message['content'] %}
            {% if 'text' in content and '<video>' in content['text'] %}
                {% set has_video_placeholder.value = True %}
            {% endif %}
            {% if 'text' in content and '<image>' in content['text'] %}
                {% set has_image_placeholder.value = True %}
            {% endif %}
        {% endfor %}
    {% endif %}
    <|im_start|>{{ message['role'] }}
    {% if message['content'] is string %}
        {{ message['content'] | replace('<video>', '<|vision_start|><|video_pad|><|vision_end|>') | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}<|im_end|>
    {% else %}
        {% for content in message['content'] %}
            {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}
                {% if not has_image_placeholder.value %}
                    {% set image_count.value = image_count.value + 1 %}
                    {% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}
                    <|vision_start|><|image_pad|><|vision_end|>
                {% endif %}
            {% elif content['type'] == 'video' or 'video' in content %}
                {% set video_count.value = video_count.value + 1 %}
                {% if not has_video_placeholder.value %}
                    {% if add_vision_id %}Video {{ video_count.value }}: {% endif %}
                    <|vision_start|><|video_pad|><|vision_end|>
                {% endif %}
            {% elif 'text' in content %}
                {{ content['text'] | replace('<video>', '<|vision_start|><|video_pad|><|vision_end|>') | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}
            {% endif %}
        {% endfor %}
        <|im_end|>
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
"""


SYSTEM_PROMPT_MULTI_ROUND_MC="""You are a helpful assistant. Answer the user's multiple-choice question based on the provided video.
Output your thinking process within the `<think>` and `</think>` tags.
If you find any video segments that might help answer your questions, you can view a specific area in detail by outputting `<grounding>{\"temporal_segment\": [t0, t1], \"sampling_strategy\": \"medium\"}</grounding>`, where t0 and t1 are the start and end times (in integer seconds) of the video segment you want to observe in detail within the entire video, sampling_strategy must be a string and in ['coarse', 'fine', 'medium'].
Once you believe you have observed all the video clues that help you answer the question and are able to integrate all the existing clues to give the correct answer, you should put the correct option letter within the `<answer>` and `</answer>` tags.
The following is a example of the thinking process and the final answer:
 - When you want to examine a specific clue segment of a video more closely, produce exactly: '<think>your thinking process here</think>\n<grounding>{"temporal_segment": [START_TIMESTAMP, END_TIMESTAMP], "sampling_strategy": "medium"}</grounding>\n'.
 - When you believe the current information is sufficient to give an answer, produce exactly: '<think>your thinking process here</think>\n<answer>Option Letter</answer>\n'.
"""


SYSTEM_PROMPT_MULTI_ROUND_OPEN_ENDED="""You are a helpful assistant. Answer the user's question based on the provided video.
Output your thinking process within the `<think>` and `</think>` tags.
If you find any video segments that might help answer your questions, you can view a specific area in detail by outputting `<grounding>{\"temporal_segment\": [t0, t1], \"sampling_strategy\": \"medium\"}</grounding>`, where t0 and t1 are the start and end times of the video segment you want to observe in detail within the entire video, sampling_strategy must be a string and in ['coarse', 'fine', 'medium'].
Once you believe you have observed all the video clues that help you answer the question and are able to integrate all the existing clues to give the correct answer, you should put the correct answer within the `<answer>` and `</answer>` tags.
The following is a example of the thinking process and the final answer:
 - When you want to examine a specific clue segment of a video more closely, produce exactly: '<think>your thinking process here</think>\n<grounding>{"temporal_segment": [START_TIMESTAMP, END_TIMESTAMP], "sampling_strategy": "medium"}</grounding>\n'.
 - When you believe the current information is sufficient to give an answer, produce exactly: '<think>your thinking process here</think>\n<answer>your answer here</answer>\n'.
"""


SYSTEM_PROMPT_TG="""You are a helpful assistant. Answer the user's question based on the video provided.
Output your thought process within the <think> </think> tags, including analysis with either specific timestamps or time ranges.
Then, provide the start and end times within the <answer> </answer> tags.
For example:
<think>Your thinking process here, including analysis with either specific timestamps or time ranges.</think>
<answer>{"temporal_segment": [START_TIMESTAMP, END_TIMESTAMP]}</answer>
"""

SYSTEM_PROMPT_TG_CGBench = """You are a helpful assistant. You will be provided with uniformly sampled frames from a video and their timestamps, along with a multiple-choice question that includes a question and several answer options.
Your task is to determine in which intervals the 'clue intervals' exist that contain visual information needed to answer the question.
Output your thought process within the <think> </think> tags, including analysis with either specific timestamps or time ranges.
Then, provide the answerwithin the <answer> </answer> tags. Only output the answer in the following format:
<answer>{"result": [[start1, end1], [start2, end2], ...]}</answer>
In this output format, each 'start' and 'end' represents the beginning and end of an interval in seconds where relevant clues can be found.
You must provide at least one interval and at most five intervals. Intervals exceeding five will NOT be considered valid.
For example:
<think>Your thinking process here, including analysis with either specific timestamps or time ranges.</think>
<answer>{"result": [[start1, end1], [start2, end2], ...]}</answer>
"""

TOOL_CALL_CROP_VIDEO_MULTI_TRUN_PROMPT_MC = '''After the above Action {action_turn}, here is the refined video clip (Observation {observation_turn}):
<video>This video is uniformly sampled at {sample_fps:.2f} fps, contains {total_frames:.1f} frames from {start_time:.1f} seconds to {end_time:.1f} seconds.
Continue your reasoning process inside <think> and </think>. If needed, you can keep selecting temporal segments from the original video by outputting <grounding> and </grounding> as before. Once you are ready to provide the final answer, put it inside <answer> and </answer>.
'''


ERROR_INFO_MULTI_TURN_PROMPT="ERROR occurs during grounding. Error Information: {error_info}. Please analyze the error information obtained from the function tool and adjust your response. Countinue your reasoning process inside <think> and </think> and follow the instructions strictly."

FORCE_FINAL_ANSWER_PROMPT='''After the above Action {action_turn}, here is the refined video clip (Observation {observation_turn}):
<video>This video is uniformly sampled at {sample_fps:.2f} fps, contains {total_frames:.1f} frames from {start_time:.1f} seconds to {end_time:.1f} seconds.
Continue your reasoning process inside <think> and </think>. You have reached the final turn and MUST provide the final answer based on all the previous conversation and observations. Think carefully about all the information you have gathered inside <think> and </think> and give your best answer inside <answer> and </answer>. Do NOT select temporal segments from the original video as before.'''

# 适用于达到了最大轮数，但是恰好这一轮grounding报错了
FORCE_FINAL_ANSWER_PROMPT_ERROR_INFO='''ERROR occurs during grounding. Error Information: {error_info}. Countinue your reasoning process inside <think> and </think> and follow the instructions below strictly.
You have reached the final turn and MUST provide the final answer based on all the previous conversation and observations. Think carefully about all the information you have gathered inside <think> and </think> and give your best answer inside <answer> and </answer>. Do NOT select temporal segments from the original video as before.'''



UNTIL = ["<|diff_marker|>"]


class VideoO3Chat(VideoO3PromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=4096,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        use_audio_in_video: bool = False,
        use_tools: bool = True,
        max_turn: int=8,
        use_dynamic_quota: bool = False,
        do_sample: bool = False,
        is_qwen3vl: bool = False,
        crop_quota=(2048, 4096, 6144),
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        if self.total_pixels and self.total_pixels > 24576 * 28 * 28:
            print('The total number of video tokens might become too large, resulting in an overly long input sequence. We recommend lowering **total_pixels** to below **24576 × 28 × 28**.')  # noqa: E501
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            return_dict_in_generate=True,
            do_sample=do_sample,
            use_cache=True,
        )
        self.system_prompt = SYSTEM_PROMPT_MULTI_ROUND_MC
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        if self.fps is None and self.nframe is None:
            print("Warning: fps and nframe are both None, \
                  using default nframe/fps setting in qwen-vl-utils/qwen-omni-utils, \
                  the fps/nframe setting in video dataset is omitted")
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        self.use_tools = use_tools
        self.use_dynamic_quota = use_dynamic_quota
        self.do_sample = do_sample
        self.is_qwen3vl = is_qwen3vl
        self.crop_quota = crop_quota


        print("**************************************************")
        print("model init parameters:")
        print(f"model_path: {model_path}")
        print(f"use_tools: {use_tools}")
        print(f"use_dynamic_quota: {use_dynamic_quota}")
        print(f"is_qwen3vl: {is_qwen3vl}")
        print(f"crop_quota: {crop_quota}")
        print(f"do_sample: {do_sample}")
        print(f"max_turn: {max_turn}")
        print(f"min_pixels: {min_pixels}")
        print(f"max_pixels: {max_pixels}")
        print(f"total_pixels: {total_pixels}")
        print(f"max_new_tokens: {max_new_tokens}")
        print(f"top_p: {top_p}")
        print(f"top_k: {top_k}")
        print(f"temperature: {temperature}")
        print(f"repetition_penalty: {repetition_penalty}")
        print("**************************************************")

        # self.use_frames = use_frames
        # self.n_frame_img = None
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None
        self.max_turn = max_turn
        self.past_key_values = None

        if self.is_qwen3vl:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            MODEL_CLS = AutoModelForImageTextToText
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            self.patch_size = 16
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.patch_size = 14

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0
        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag between `use_vllm` and `use_lmdeploy` to True"  # noqa: E501

        if self.use_vllm:
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            import os
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn.'
                    'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
                )
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=5,
                max_model_len=32768,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
                trust_remote_code=True,
            )

        elif self.use_lmdeploy:
            from lmdeploy import TurbomindEngineConfig, pipeline, ChatTemplateConfig
            num_gpus = torch.cuda.device_count()
            self.model = pipeline(
                model_path,
                backend_config=TurbomindEngineConfig(session_len=32768, cache_max_entry_count=0.1, tp=num_gpus),
                chat_template_config=ChatTemplateConfig(model_name='qwen2d5-vl'))
            torch.cuda.set_device(0)
            self.device = 'cuda'
        else:
            load_kwargs = dict(
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            if self.is_qwen3vl:
                load_kwargs['attn_implementation'] = 'flash_attention_2'
                load_kwargs['local_files_only'] = True
            else:
                load_kwargs['attn_implementation'] = 'sdpa'
            self.model = MODEL_CLS.from_pretrained(model_path, **load_kwargs)
            self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels

            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video_start':s.get('video_start', None),
                    'video_end':s.get('video_end', None),
                    'sampling_strategy':s.get('sampling_strategy', None),
                }

                if isinstance(s['value'], str):
                    item['video'] = ensure_url(s['value'])
                else:
                    if len(s['value']) <= 0:
                        raise ValueError(f"Invalid video s['value']: {s['value']}")
                    item['video'] = [ensure_url(v) for v in s['value']]
                    if len(item['video']) <= 0:
                        raise ValueError(f"Invalid video item['video']: {item['video']}")
                
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    if item['video_start'] is not None and item['video_end'] is not None:
                        pixel_unit = self.patch_size * self.patch_size * 4
                        
                        if not self.use_dynamic_quota:
                            item['total_pixels'] = self.crop_quota[1] * pixel_unit
                        else:
                            if item['sampling_strategy'] == "coarse":
                                item['total_pixels'] = self.crop_quota[0] * pixel_unit
                            elif item['sampling_strategy'] == "medium":
                                item['total_pixels'] = self.crop_quota[1] * pixel_unit
                            elif item['sampling_strategy'] == "fine":
                                item['total_pixels'] = self.crop_quota[2] * pixel_unit
                    else:
                        item['total_pixels'] = self.total_pixels

                    print(f"sampling strategy: {item['sampling_strategy']}, total pixels: {item['total_pixels']}")

            
                for key in ['min_pixels', 'max_pixels', 'total_pixels','sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]

            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            elif s['type'] == 'audio':
                item = {'type':'audio','audio':s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def extract_grounding_info(self, response):
        """
        Extract grounding information from model response.

        Args:
            response (str): Model response text
            
        Returns:
            tuple | None: If extraction succeeds, return (start_timestamp, end_timestamp, sampling_strategy)
                          start_timestamp (float): Start timestamp
                          end_timestamp (float): End timestamp
                          sampling_strategy (str): Sampling strategy ("coarse", "medium", "fine")
                          If extraction fails, return None
        """
        import re
        import json
        
        if "<grounding>" not in response or "</grounding>" not in response:
            raise ValueError(f"No grounding content found in response: {response}")
        
        # Use regex to extract <grounding> tag content
        pattern = re.compile(r'<grounding>(.*?)</grounding>', re.DOTALL)
        matches = pattern.findall(response)
        
        if not matches:
            raise ValueError(f"No grounding content found in response: {response}")
        
        # Take the last match
        grounding_content = matches[-1].strip()
        
        # Try to extract JSON part (from the first '{' to the last '}')
        start_idx = grounding_content.find('{')
        end_idx = grounding_content.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise ValueError(f"Invalid grounding content: {grounding_content}")
        
        grounding_content = grounding_content[start_idx:end_idx+1]
        
        try:
            # Try to parse JSON directly
            grounding_json = json.loads(grounding_content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to fix single quotes
            try:
                fixed_content = grounding_content.replace("'", '"')
                grounding_json = json.loads(fixed_content)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse grounding JSON: {grounding_content}. Error: {e}")
                raise ValueError(f"Failed to parse grounding JSON: {grounding_content}. Error: {e}")
        
        # Extract fields
        if "temporal_segment" not in grounding_json or "sampling_strategy" not in grounding_json:
            logging.warning(f"Missing required fields in grounding JSON: {grounding_json}")
            raise ValueError(f"Missing required fields in grounding JSON: {grounding_json}")
        
        temporal_segment = grounding_json["temporal_segment"]
        sampling_strategy = grounding_json["sampling_strategy"]
        
        # Validate and extract timestamps
        if not (isinstance(temporal_segment, (list, tuple)) and len(temporal_segment) == 2):
            logging.warning(f"Invalid temporal_segment format: {temporal_segment}")
            raise ValueError(f"Invalid temporal_segment format: {temporal_segment}")
        
        try:
            start_timestamp = float(temporal_segment[0])  # START_TIMESTAMP (float)
            end_timestamp = float(temporal_segment[1])     # END_TIMESTAMP (float)
            sampling_strategy_str = str(sampling_strategy)  # sampling_strategy (str)
        except (ValueError, TypeError) as e:
            logging.warning(f"Failed to convert temporal_segment to float: {temporal_segment}. Error: {e}")
            raise ValueError(f"Failed to convert temporal_segment to float: {temporal_segment}. Error: {e}")
        
        return start_timestamp, end_timestamp, sampling_strategy_str

    def generate_inner_transformers_tools(self, message, dataset=None):

        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                if self.is_qwen3vl:
                    from .vision_process_qwen3vl import process_vision_info
                else:
                    from .vision_process import process_vision_info
            except Exception as err:
                logging.critical("vision_process not found, please check vlmeval/vlm/video_o3/vision_process.py")  # noqa: E501
                raise err

        # Scan and extract raw_video, question_type
        # raw_video is used for subsequent VideoCrop
        raw_video = None
        question_type = None
        
        indices_to_remove = []
        for i, item in enumerate(message):
            if isinstance(item, dict):
                if item.get('type') == 'raw_video' and raw_video is None:
                    raw_video = dict(type='raw_video', value=item['value'])
                    indices_to_remove.append(i)
                elif item.get('type') == 'question_type' and question_type is None:
                    question_type = item.get('value')
                    indices_to_remove.append(i)
        
        for i in sorted(indices_to_remove, reverse=True):
            message.pop(i)

        # Initialize messages outside the loop to preserve chat history
        messages = []
        # Set system_prompt according to question_type
        if question_type == 'oe':
            system_prompt_to_use = SYSTEM_PROMPT_MULTI_ROUND_OPEN_ENDED
        else:
            system_prompt_to_use = self.system_prompt

        if system_prompt_to_use is not None:
            messages.append({'role': 'system', 'content': system_prompt_to_use})
        
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        # Initialize turn
        turn_count = 0

        force_answer = False

        total_inference_time = 0.0
        total_generate_time = 0.0
        total_video_pad_tokens_qa = 0
        total_generate_tokens_qa = 0
        inference_start_time = time.time()

        video_metadatas = None
        video_kwargs = None
        if self.is_qwen3vl:
            images, videos, video_kwargs = process_vision_info(
                [messages],
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            images, videos = process_vision_info([messages])

        while True:

            torch.cuda.empty_cache()

            text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True,chat_template=CHAT_TEMPLATE)

            if self.is_qwen3vl:
                inputs = self.processor(
                    text=text,
                    images=images,
                    videos=videos,
                    video_metadata=video_metadatas,
                    do_resize=False,
                    return_tensors='pt',
                    **(video_kwargs or {}),
                )
            else:
                inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')  # noqa: E501
            inputs = inputs.to('cuda')
            total_video_pad_tokens_qa = count_video_pad_tokens(self.processor, inputs.input_ids)

            if listinstr(['omni'], self.model_path.lower()):
                self.generate_kwargs['use_audio_in_video'] = self.use_audio_in_video
                self.generate_kwargs['return_audio'] = False

            stop_criteria = StoppingCriteriaList([
                KeywordsStoppingCriteria(
                    keywords=["</grounding>", "</answer>"],
                    tokenizer=self.processor.tokenizer,
                    input_ids=inputs.input_ids,
                )
            ])

            if not force_answer:
                self.generate_kwargs['stopping_criteria'] = stop_criteria

            generation_inputs = inputs

            # Record model.generate time for each round
            generate_start_time = time.time()
            
            outputs = self.model.generate(
                **generation_inputs,
                **self.generate_kwargs,
            )
            generate_time = time.time() - generate_start_time
            total_generate_time += generate_time

            generated_ids = outputs.sequences

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(generation_inputs['input_ids'], generated_ids)
            ]
            total_generate_tokens_qa += int(len(generated_ids[0]))
            
            response = self.processor.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            if not force_answer:
                for stop_tag in ("</grounding>", "</answer>"):
                    if stop_tag in response:
                        response = response[: response.find(stop_tag) + len(stop_tag)]
                        break

            if self.post_process:
                resp = response.split('\\boxed{')[-1]
                lt = len(resp)
                counter, end = 1, None
                for i in range(lt):
                    if resp[i] == '{':
                        counter += 1
                    elif resp[i] == '}':
                        counter -= 1
                    if counter == 0:
                        end = i
                        break
                    elif i == lt - 1:
                        end = lt
                        break
                if end is not None:
                    response = resp[:end]

            # Pack as one text, print all at once to avoid being interrupted
            txt_lines = []
            txt_lines.append("-"*20)
            txt_lines.append(f"Round {turn_count}: \n text: {text} \n response: {response}")

            txt_lines.append("-"*20)
            print("\n".join(txt_lines))
            # Start from round 0
            turn_count += 1

            # Hard limit: force exit if over max_turn+2 to prevent infinite loop in extreme cases
            if turn_count > self.max_turn + 2:
                logging.error(f"Turn count {turn_count} exceeded max_turn+2 ({self.max_turn + 2}), forcing exit to prevent infinite loop.")
                response = "I'm sorry, I can't answer the question based on the video."
                break
            
            if "<answer>" in response and "</answer>" in response:
                # If both <grounding> and <answer> exist, prioritize <answer> (as <answer> is the final answer)
                if '<grounding>' in response and '<answer>' in response:
                    logging.warning("<grounding> and <answer> are both in response, prioritizing <answer>")
                # Extract content between <answer> and </answer>
                start_idx = response.find("<answer>")
                end_idx = response.find("</answer>")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    response = response[start_idx + len("<answer>"):end_idx].strip()
                    break
            else:
                if force_answer:
                    # When force_answer=True, no longer process grounding, directly exit to prevent infinite loop
                    logging.warning(f"Force answer mode enabled at turn {turn_count}, breaking the loop.")
                    response = "I'm sorry, I can't answer the question based on the video."
                    break
                if "<grounding>" in response and "</grounding>" in response:
                    # Extract grounding information
                    try:
                        grounding_result = self.extract_grounding_info(response)
                        # extract_grounding_info returns (start_timestamp, end_timestamp, sampling_strategy_str) on success
                        # If it fails, it raises an exception
                        start_timestamp, end_timestamp, sampling_strategy_str = grounding_result
                        # Extraction succeeded, values can be used here
                        logging.info(f"Extracted grounding: START_TIMESTAMP={start_timestamp}, END_TIMESTAMP={end_timestamp}, sampling_strategy={sampling_strategy_str}")
                        
                        # Add assistant response to messages (preserve chat history)
                        messages.append({'role': 'assistant', 'content': response})
                        
                        # Prepare new user message content (with tool call result)
                        # Get info and input of current video
                        temp_message = [
                            dict(type= 'video',
                                value=raw_video['value'],
                                video_start=start_timestamp,
                                video_end=end_timestamp,
                                sampling_strategy=sampling_strategy_str
                            )
                        ]

                        
                        temp_messages = [{'role':"user", 'content': self._prepare_content(temp_message, dataset=dataset)}]
                        if self.is_qwen3vl:
                            temp_images, temp_videos, temp_video_kwargs = process_vision_info(
                                temp_messages,
                                image_patch_size=16,
                                return_video_kwargs=True,
                                return_video_metadata=True,
                            )
                            temp_video_tensors = [v for v, _ in temp_videos]
                            temp_metadatas = [m for _, m in temp_videos]
                            sample_fps = temp_video_kwargs['fps'][0]
                            sample_n_frames = temp_video_kwargs['n_frames'][0]
                        else:
                            temp_images, temp_videos, temp_video_kwargs = process_vision_info(temp_messages, return_video_kwargs=True)
                            sample_fps = temp_video_kwargs['fps'][0]
                            sample_n_frames = temp_video_kwargs['n_frames'][0]
                        
                        if temp_images is not None:
                            if images is None:
                                images = []
                            images.extend(temp_images)
                        if temp_videos is not None:
                            if videos is None:
                                videos = []
                            if self.is_qwen3vl:
                                videos.extend(temp_video_tensors)
                                if video_metadatas is None:
                                    video_metadatas = []
                                video_metadatas.extend(temp_metadatas)
                            else:
                                videos.extend(temp_videos)

                        new_message = []

                        # At most (max_turn + 1) rounds, for last round force model to output final answer
                        if turn_count >= self.max_turn:
                            force_answer = True

                            new_message.append(
                                dict(type= 'text',
                                    value=FORCE_FINAL_ANSWER_PROMPT.format(action_turn=turn_count, observation_turn=turn_count, sample_fps=sample_fps, total_frames=sample_n_frames, start_time=start_timestamp, end_time=end_timestamp)
                                )
                            )

                        else:
                            new_message.append(
                                dict(type= 'text',
                                    value=TOOL_CALL_CROP_VIDEO_MULTI_TRUN_PROMPT_MC.format(action_turn=turn_count, observation_turn=turn_count, sample_fps=sample_fps, total_frames=sample_n_frames, start_time=start_timestamp, end_time=end_timestamp)
                                )
                            )
                        
                        new_message.append(
                            dict(type= 'video',
                                value=raw_video['value'],
                                video_start=start_timestamp,
                                video_end=end_timestamp,
                                sampling_strategy=sampling_strategy_str
                            )
                        )
                        
                        # Add new user message to messages
                        messages.append({'role': 'user', 'content':  self._prepare_content(new_message, dataset=dataset) })

                            
                    except Exception as e:
                        # Failed to extract grounding, log and continue
                        logging.error(f"Failed to extract grounding information: {e}")

                        messages.append({'role': 'assistant', 'content': response})

                        # At most (max_turn + 1) rounds, for last round force model to output final answer
                        if turn_count >= self.max_turn:
                            force_answer = True
                            error_message = FORCE_FINAL_ANSWER_PROMPT_ERROR_INFO.format(error_info=str(e))
                        else:
                            error_message = ERROR_INFO_MULTI_TURN_PROMPT.format(error_info=str(e))
                        messages.append({'role': 'user', 'content': error_message})

                else:
                    # Neither <answer> nor <grounding> present, this case should not occur (stopping_criteria should ensure at least one)
                    # For safety, log warning and terminate loop
                    logging.warning(f"Unexpected response format: neither <answer> nor <grounding> found. Response: {response}")
                    messages.append({'role': 'assistant', 'content': response})

                    error_info = "Unexpected response format: neither <answer> nor <grounding> found."

                    if turn_count >= self.max_turn:
                        force_answer = True
                        error_message = FORCE_FINAL_ANSWER_PROMPT_ERROR_INFO.format(error_info=error_info)
                    else:
                        error_message = ERROR_INFO_MULTI_TURN_PROMPT.format(error_info=error_info)


                    messages.append({'role': 'user', 'content': error_message})
                    break
        
        # Compute total inference time
        total_inference_time = time.time() - inference_start_time
        
        print("Final response: ", response)
        print(f"Total turns used: {turn_count}")
        print(f"Total inference time: {total_inference_time:.2f} seconds")
        print(f"Total generate time (sum of all rounds): {total_generate_time:.2f} seconds")
        print(f"Video <|video_pad|> tokens in the whole conversation: {total_video_pad_tokens_qa}")
        print(f"Total generate tokens (sum over rounds): {total_generate_tokens_qa}")

        # Append turn and timing info at end of response for later analysis
        response_with_metadata = f"{response}\n[TURN_COUNT: {turn_count}]\n[INFERENCE_TIME: {total_inference_time:.2f}s]\n[INFERENCE_TIME_GENERATE: {total_generate_time:.2f}s]"
        
        return response_with_metadata


    def generate_inner_transformers(self, message, dataset=None):
        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                if self.is_qwen3vl:
                    from .vision_process_qwen3vl import process_vision_info
                else:
                    from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        print("messages: ", messages)

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True,chat_template=CHAT_TEMPLATE)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info([messages], use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(text=text, images=images,audio=audios, videos=videos, padding=True, return_tensors='pt',use_audio_in_video=self.use_audio_in_video)  # noqa: E501
        elif self.is_qwen3vl:
            images, videos, video_kwargs = process_vision_info(
                [messages],
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors='pt',
                **(video_kwargs or {}),
            )
        else:
            images, videos = process_vision_info([messages])
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')  # noqa: E501
        inputs = inputs.to('cuda')

        print("text: ", text)

        if listinstr(['omni'], self.model_path.lower()):
            self.generate_kwargs['use_audio_in_video'] = self.use_audio_in_video
            self.generate_kwargs['return_audio'] = False
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]

        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def generate_inner(self, message, dataset=None):
        
        if self.use_tools:
            return self.generate_inner_transformers_tools(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
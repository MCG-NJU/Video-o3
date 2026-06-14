from __future__ import annotations

import base64
import copy
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

from pathlib import Path
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)

# Qwen2.5-VL patch_size=14 -> grid 28; Qwen3-VL patch_size=16 -> grid 32. Default 14 for backward compat.
_DEFAULT_PATCH_SIZE = int(os.environ.get("VISION_PATCH_SIZE", "14"))
_GRID = _DEFAULT_PATCH_SIZE * 2  # 28 or 32

IMAGE_FACTOR = _GRID
MIN_PIXELS = 4 * _GRID * _GRID
MAX_PIXELS = 16384 * _GRID * _GRID
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * _GRID * _GRID
VIDEO_MAX_PIXELS = 768 * _GRID * _GRID
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
# FPS_MAX_FRAMES = 768
FPS_MAX_FRAMES = int(os.environ.get('SELF_SET_FPS_MAX_FRAMES', 768))

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * _GRID * _GRID * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS} (VISION_PATCH_SIZE={_DEFAULT_PATCH_SIZE}, grid={_GRID})")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    # if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
    #     raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    if nframes < FRAME_FACTOR:
        nframes = FRAME_FACTOR
        print("<<<INFO>>> nframes < FRAME_FACTOR, set nframes to FRAME_FACTOR")
    if nframes > total_frames:
        nframes = total_frames
        print("<<<INFO>>> nframes > total_frames, set nframes to total_frames")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


def is_torchcodec_available() -> bool:
    """Check if torchcodec is available and properly installed."""
    try:
        import importlib.util
        if importlib.util.find_spec("torchcodec") is None:
            return False
        from torchcodec.decoders import VideoDecoder
        return True
    except (ImportError, AttributeError, Exception):
        return False


def _read_video_torchcodec(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchcodec.decoders.VideoDecoder

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
    logger.info(f"set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")
    video_path = ele["video"]
    st = time.time()
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = decoder.get_frames_at(indices=idx).data
    logger.info(f"torchcodec:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    return video, sample_fps


def load_frame(p: Path):
    with open(p, 'rb') as f:
        img_bytes = f.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode {p}")
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return torch.from_numpy(img).permute(2, 0, 1)  # CHW


def _read_video_frame(ele: dict, client=None) -> (torch.Tensor, float):
    """
    Read a video from a directory of frames (pre-extracted) using OpenCV + imdecode + multithreading.
    Args:
        ele supports:
            - video: the path to the frame folder (not mp4)
            - video_start: start time (seconds)
            - video_end: end time (seconds)
            - fps: desired fps (e.g., 4)
        client: optional S3 client for reading s3 files
    """
    import time
    st = time.time()
    frame_dir = Path(ele["video"])
    # Load all image files
    valid_exts = {".jpg", ".jpeg", ".png"}
    frame_paths = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in valid_exts])

    video_fps = ele.get("source_frames_fps", 4.0)
    total_frames = len(frame_paths)
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    sampled_paths = [frame_paths[i] for i in idx]

    # from pprint import pprint
    # pprint(sampled_paths)
    # -------- read sampled images using OpenCV imdecode + frombuffer + multithreading --------

    with ThreadPoolExecutor(max_workers=32) as executor:
        imgs = list(executor.map(load_frame, sampled_paths))

    video = torch.stack(imgs, dim=0)  # TCHW

    # print(
    #     f"<<<DEBUG>>> video_path: {frame_dir}, ",
    #     f"read_img_seq: {sampled_paths}, "
    #     f"frames={len(video)}, "
    #     f"video_fps={video_fps:.3f}, "
    #     f"sample_fps={sample_fps:.3f}, "
    # )

    return video, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "torchcodec": _read_video_torchcodec,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images


def fetch_video_raw(ele: dict) -> torch.Tensor | list[Image.Image]:
    assert(isinstance(ele["video"], str))
    video_reader_backend = get_video_reader_backend()
    try:
        video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
    except Exception as e:
        logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
        video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
    return video, sample_fps


def fetch_video_raw_frame(ele: dict) -> torch.Tensor | list[Image.Image]:
    assert(isinstance(ele["video"], str))

    video, sample_fps = _read_video_frame(ele)
    return video, sample_fps


def crop_video_raw(
    ele: dict,
) -> (torch.Tensor, float):
    """
    Crop a segment from the input video tensor according to the specified temporal boundaries.

    Args:
        ele (dict): Dictionary containing video configuration. Supported keys:
            - video: Video tensor of shape (T, C, H, W)
            - video_start: Start time of the segment in seconds (float, required)
            - video_end: End time of the segment in seconds (float, required)
            - raw_fps: Original FPS of the video (float, required)

    Returns:
        Tuple[torch.Tensor, float]:
            - Cropped video tensor of shape (T_new, C, H, W)
            - Sampled FPS as a float
    """

    raw_video = ele['video']
    total_frames, video_fps = raw_video.shape[0], ele.get('raw_fps',None)
    if video_fps is None:
        raise ValueError("video_fps is not provided")
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long()
    video = raw_video[idx]
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


def resample_video_from_raw(
    video: torch.Tensor,
    sample_fps: float,
    ele: dict,
    image_factor: int = IMAGE_FACTOR,
    return_video_sample_fps: bool = False
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """
    Resample and resize video from raw video tensor.

    Args:
        video: Raw video tensor with shape (T, C, H, W) from fetch_video_raw.
        sample_fps: Sample fps from fetch_video_raw.
        ele: A dict contains the configuration of video.
            support keys:
                - min_pixels: minimum pixels per frame.
                - max_pixels: maximum pixels per frame.
                - total_pixels: total pixels for all frames.
                - resized_height: target height for resizing.
                - resized_width: target width for resizing.
        image_factor: Factor for image resizing, default is IMAGE_FACTOR.
        return_video_sample_fps: Whether to return sample_fps along with video.

    Returns:
        torch.Tensor: Resampled video tensor with shape (T, C, H, W).
        If return_video_sample_fps is True, returns tuple (video, sample_fps).
    """
    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    if return_video_sample_fps:
        return video, sample_fps
    return video


def resample_video_clip_from_raw(
    video: torch.Tensor,
    sample_fps: float,
    ele: dict,
    image_factor: int = IMAGE_FACTOR,
    return_video_sample_fps: bool = False
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """
    Extract video clip from raw video tensor based on start and end time, then resize.

    Args:
        video: Raw video tensor with shape (T, C, H, W) from fetch_video_raw.
        sample_fps: Sample fps from fetch_video_raw.
        ele: A dict contains the configuration of video.
            support keys:
                - clip_start: start time of the clip in seconds (required).
                - clip_end: end time of the clip in seconds (required).
                - min_pixels: minimum pixels per frame.
                - max_pixels: maximum pixels per frame.
                - total_pixels: total pixels for all frames.
                - resized_height: target height for resizing.
                - resized_width: target width for resizing.
        image_factor: Factor for image resizing, default is IMAGE_FACTOR.
        return_video_sample_fps: Whether to return sample_fps along with video.

    Returns:
        torch.Tensor: Resampled video tensor with shape (T, C, H, W).
        If return_video_sample_fps is True, returns tuple (video, sample_fps).
    """
    nframes, _, height, width = video.shape
    
    # Get clip start and end time
    clip_start = ele.get("clip_start", None)
    clip_end = ele.get("clip_end", None)
    
    if clip_start is None or clip_end is None:
        raise ValueError("clip_start and clip_end must be provided in ele dict")
    
    # Calculate total video duration
    total_duration = nframes / sample_fps
    
    # Validate and clamp clip_start: must be >= 0 and < total_duration
    if clip_start < 0:
        logger.warning(f"clip_start ({clip_start}) < 0, clamping to 0")
        clip_start = 0.0
    if clip_start >= total_duration:
        raise ValueError(f"clip_start ({clip_start}) must be < total_duration ({total_duration})")
    
    # Validate and clamp clip_end: must be > clip_start and <= total_duration
    if clip_end <= clip_start:
        raise ValueError(f"clip_end ({clip_end}) must be > clip_start ({clip_start})")
    if clip_end > total_duration:
        logger.warning(f"clip_end ({clip_end}) > total_duration ({total_duration}), clamping to {total_duration}")
        clip_end = total_duration
    
    # Calculate frame indices for the clip
    # Use floor for start to include frames from clip_start onwards
    # Use ceil for end to include frames up to clip_end
    start_frame_idx = int(math.floor(clip_start * sample_fps))
    end_frame_idx = int(math.ceil(clip_end * sample_fps))
    
    # Ensure indices are within valid range [0, nframes)
    start_frame_idx = max(0, min(start_frame_idx, nframes - 1))
    # end_frame_idx should be exclusive, so it can be nframes at most
    end_frame_idx = max(start_frame_idx + 1, min(end_frame_idx, nframes))
    
    # Ensure we don't extract more frames than available
    if end_frame_idx - start_frame_idx > nframes:
        raise ValueError(f"Cannot extract {end_frame_idx - start_frame_idx} frames, only {nframes} frames available")
    
    # Extract the clip
    video_clip = video[start_frame_idx:end_frame_idx]
    clip_nframes = video_clip.shape[0]
    
    # Ensure frame count is a multiple of FRAME_FACTOR (2)
    # This is mandatory - we must adjust to make it even
    if clip_nframes % FRAME_FACTOR != 0:
        # Calculate time distances to start and end
        actual_start_time = start_frame_idx / sample_fps
        actual_end_time = (end_frame_idx - 1) / sample_fps
        
        dist_to_start = abs(clip_start - actual_start_time)
        dist_to_end = abs(clip_end - actual_end_time)
        
        # Determine which direction to extend based on time distance
        # and available frames
        can_extend_before = (start_frame_idx > 0)
        can_extend_after = (end_frame_idx < nframes)
        
        # Strategy 1: Try to add one more frame (extend)
        adjusted = False
        if dist_to_start <= dist_to_end and can_extend_before:
            # Prefer extending before start
            new_start = start_frame_idx - 1
            video_clip = video[new_start:end_frame_idx]
            logger.info(f"Added frame before start to make frame count even: {clip_nframes} -> {video_clip.shape[0]}")
            adjusted = True
        elif can_extend_after:
            # Extend after end (either preferred or as fallback)
            new_end = min(end_frame_idx + 1, nframes)
            video_clip = video[start_frame_idx:new_end]
            logger.info(f"Added frame after end to make frame count even: {clip_nframes} -> {video_clip.shape[0]}")
            adjusted = True
        elif can_extend_before:
            # Fallback: extend before start if after is not available
            new_start = start_frame_idx - 1
            video_clip = video[new_start:end_frame_idx]
            logger.info(f"Added frame before start to make frame count even: {clip_nframes} -> {video_clip.shape[0]}")
            adjusted = True
        
        # Strategy 2: Cannot extend - remove one frame or duplicate one frame
        if not adjusted:
            # Remove frame from the end that's farther from the requested time
            if dist_to_start <= dist_to_end:
                # Remove from end (farther from start)
                if clip_nframes > 1:
                    video_clip = video_clip[:-1]
                    logger.info(f"Removed frame from end to make frame count even: {clip_nframes} -> {video_clip.shape[0]}")
                else:
                    # Only 1 frame - duplicate it
                    video_clip = torch.cat([video_clip, video_clip], dim=0)
                    logger.info(f"Duplicated frame to make frame count even: {clip_nframes} -> {video_clip.shape[0]}")
            else:
                # Remove from start (farther from end)
                if clip_nframes > 1:
                    video_clip = video_clip[1:]
                    logger.info(f"Removed frame from start to make frame count even: {clip_nframes} -> {video_clip.shape[0]}")
                else:
                    # Only 1 frame - duplicate it
                    video_clip = torch.cat([video_clip, video_clip], dim=0)
                    logger.info(f"Duplicated frame to make frame count even: {clip_nframes} -> {video_clip.shape[0]}")
    
    # Verify frame count is now a multiple of FRAME_FACTOR
    final_nframes = video_clip.shape[0]
    
    # Final validation: ensure we don't have more frames than the original video
    if final_nframes > nframes:
        raise ValueError(f"Extracted {final_nframes} frames, but original video only has {nframes} frames")
    
    # Final check: frame count MUST be a multiple of FRAME_FACTOR
    if final_nframes % FRAME_FACTOR != 0:
        # Last resort: force adjustment by removing or duplicating
        if final_nframes > 1:
            # Remove one frame from the end
            video_clip = video_clip[:-1]
            final_nframes = video_clip.shape[0]
            logger.warning(f"Force removed frame to ensure even count: -> {final_nframes}")
        else:
            # Only 1 frame - duplicate it
            video_clip = torch.cat([video_clip, video_clip], dim=0)
            final_nframes = video_clip.shape[0]
            logger.warning(f"Force duplicated frame to ensure even count: -> {final_nframes}")
        
        # Double check
        if final_nframes % FRAME_FACTOR != 0:
            raise RuntimeError(f"Failed to make frame count even: {final_nframes} is not a multiple of {FRAME_FACTOR}")
    
    # Ensure final_nframes is up to date
    final_nframes = video_clip.shape[0]
    
    # Verify one more time that frame count is even
    assert final_nframes % FRAME_FACTOR == 0, f"Frame count must be even, got {final_nframes}"
    
    # Now resize based on total_pixels constraint
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / final_nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
    max_pixels = min(max_pixels_supposed, max_pixels)
    
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    
    video_clip = transforms.functional.resize(
        video_clip,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    
    if return_video_sample_fps:
        return video_clip, sample_fps
    return video_clip


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type","") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs

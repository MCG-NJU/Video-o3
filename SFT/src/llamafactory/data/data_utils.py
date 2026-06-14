# Copyright 2025 the LlamaFactory team.
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

import json
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union

import fsspec
from datasets import DatasetDict, concatenate_datasets, interleave_datasets
import torch

from ..extras import logging
from ..extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


SLOTS = list[Union[str, set[str], dict[str, str]]]


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]


def merge_dataset(
    all_datasets: list[Union["Dataset", "IterableDataset"]], data_args: "DataArguments", seed: int
) -> Union["Dataset", "IterableDataset"]:
    r"""Merge multiple datasets to a unified dataset."""
    if len(all_datasets) == 1:
        return all_datasets[0]

    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning_rank0_once("The samples between different datasets will not be mixed in streaming mode.")

        return concatenate_datasets(all_datasets)

    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning_rank0_once("We recommend using `mix_strategy=concat` in non-streaming mode.")

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )

    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")


def split_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]],
    data_args: "DataArguments",
    seed: int,
) -> "DatasetDict":
    r"""Split the dataset and returns a dataset dict containing train set and validation set.

    Support both map dataset and iterable dataset.
    """
    if eval_dataset is not None and data_args.val_size > 1e-6:
        raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

    dataset_dict = {}
    if dataset is not None:
        if data_args.streaming:
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

        if data_args.val_size > 1e-6:
            if data_args.streaming:
                dataset_dict["validation"] = dataset.take(int(data_args.val_size))
                dataset_dict["train"] = dataset.skip(int(data_args.val_size))
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset_dict = dataset.train_test_split(test_size=val_size, seed=seed)
                dataset = dataset.train_test_split(test_size=val_size, seed=seed)
                dataset_dict = {"train": dataset["train"], "validation": dataset["test"]}
        else:
            dataset_dict["train"] = dataset

    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            dataset_dict.update({f"validation_{name}": data for name, data in eval_dataset.items()})
        else:
            if data_args.streaming:
                eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

            dataset_dict["validation"] = eval_dataset

    return DatasetDict(dataset_dict)


def get_dataset_module(dataset: Union["Dataset", "DatasetDict"]) -> "DatasetModule":
    r"""Convert dataset or dataset dict to dataset module."""
    dataset_module: DatasetModule = {}
    if isinstance(dataset, DatasetDict):  # dataset dict
        if "train" in dataset:
            dataset_module["train_dataset"] = dataset["train"]

        if "validation" in dataset:
            dataset_module["eval_dataset"] = dataset["validation"]
        else:
            eval_dataset = {}
            for key in dataset.keys():
                if key.startswith("validation_"):
                    eval_dataset[key[len("validation_") :]] = dataset[key]

            if len(eval_dataset):
                dataset_module["eval_dataset"] = eval_dataset

    else:  # single dataset
        dataset_module["train_dataset"] = dataset

    return dataset_module


def setup_fs(path: str, anon: bool = False) -> "fsspec.AbstractFileSystem":
    r"""Set up a filesystem object based on the path protocol."""
    storage_options = {"anon": anon} if anon else {}
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **storage_options)
    elif path.startswith(("gs://", "gcs://")):
        fs = fsspec.filesystem("gcs", **storage_options)
    else:
        raise ValueError(f"Unsupported protocol in path: {path}. Use 's3://' or 'gs://'.")

    if not fs.exists(path):
        raise ValueError(f"Path does not exist: {path}.")

    return fs


def _read_json_with_fs(fs: "fsspec.AbstractFileSystem", path: str) -> list[Any]:
    r"""Helper function to read JSON/JSONL files using fsspec."""
    with fs.open(path, "r") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def read_cloud_json(cloud_path: str) -> list[Any]:
    r"""Read a JSON/JSONL file from cloud storage (S3 or GCS).

    Args:
        cloud_path: str
            Cloud path in the format:
            - 's3://bucket-name/file.json' for AWS S3
            - 'gs://bucket-name/file.jsonl' or 'gcs://bucket-name/file.jsonl' for Google Cloud Storage
    """
    try:
        fs = setup_fs(cloud_path, anon=True)  # try with anonymous access first
    except Exception:
        fs = setup_fs(cloud_path)  # try again with credentials

    # filter out non-JSON files
    files = [x["Key"] for x in fs.listdir(cloud_path)] if fs.isdir(cloud_path) else [cloud_path]
    files = filter(lambda file: file.endswith(".json") or file.endswith(".jsonl"), files)
    if not files:
        raise ValueError(f"No JSON/JSONL files found in the specified path: {cloud_path}.")

    return sum([_read_json_with_fs(fs, file) for file in files], [])


def _find_sequence_positions(ids: "torch.Tensor", sequence: list[int]) -> "torch.Tensor":
    r"""Find all positions where the sequence starts in ids.
    
    Args:
        ids: 1D tensor of token IDs [L]. Non-assistant positions should be masked with a special value (e.g., -1).
        sequence: List of token IDs to search for
    
    Returns:
        1D tensor of positions where the sequence starts, empty tensor if not found.
    """
    if len(sequence) == 0:
        return torch.tensor([], dtype=torch.long, device=ids.device)
    positions = []
    seq_len = len(sequence)
    for i in range(len(ids) - seq_len + 1):
        if torch.equal(ids[i:i+seq_len], torch.tensor(sequence, device=ids.device, dtype=ids.dtype)):
            positions.append(i)
    return torch.tensor(positions, dtype=torch.long, device=ids.device) if positions else torch.tensor([], dtype=torch.long, device=ids.device)


def _group_frame_segments_to_video_segments(
    frame_segments: list[tuple[int, int]],
    gap_threshold: int = 30,
) -> list[tuple[int, int]]:
    r"""Group frame-level vision segments into video-level segments using gap-based clustering.

    In Qwen3VL's interleaved-timestamp format every frame has its own
    <|vision_start|> / <|vision_end|> pair.  Consecutive frames inside the
    same video are separated by a short timestamp token sequence (~6-10
    tokens), while frames from different videos are separated by much longer
    conversation text.  We exploit this gap to cluster frames into videos.

    Args:
        frame_segments: List of (start, end) positions for each frame-level
            vision segment, already paired and sorted by position.
        gap_threshold: Maximum token gap between consecutive frames that still
            counts as the same video.  Defaults to 30, which comfortably sits
            between typical intra-video gaps (~6-10 tokens for timestamps) and
            inter-video gaps (tens to hundreds of conversation tokens).

    Returns:
        List of (video_start, video_end) where video_start is the
        vision_start position of the first frame and video_end is the
        vision_end position of the last frame for each video.
    """
    if not frame_segments:
        return []

    video_segments = []
    group_start = frame_segments[0][0]
    prev_end = frame_segments[0][1]

    for i in range(1, len(frame_segments)):
        cur_start, cur_end = frame_segments[i]
        if cur_start - prev_end >= gap_threshold:
            video_segments.append((group_start, prev_end))
            group_start = cur_start
        prev_end = cur_end

    video_segments.append((group_start, prev_end))
    return video_segments


def convert_2d_to_4d_attention_mask(
    attention_mask: "torch.Tensor", 
    dtype: Optional["torch.dtype"] = None,
    input_ids: Optional["torch.Tensor"] = None,
    tokenizer: Optional[Any] = None,
    labels: Optional["torch.Tensor"] = None,
    mask_answer_from_first_video: bool = False,
    mask_grounding_from_other_videos: bool = False,
) -> "torch.Tensor":
    r"""Convert 2D attention mask to 4D attention mask.
    
    Convert a standard 2D attention mask [batch_size, seq_len] to 4D format [batch_size, 1, seq_len, seq_len]
    with causal masking and padding masking applied.
    
    Args:
        attention_mask: 2D tensor of shape [batch_size, seq_len], where 1 indicates valid tokens and 0 indicates padding
        dtype: Optional dtype for the output mask. If None, uses bool dtype.
        input_ids: Optional tensor of shape [batch_size, seq_len] for identifying answer and video segments.
        tokenizer: Optional tokenizer for getting special token IDs.
        labels: Optional tensor of shape [batch_size, seq_len] for identifying assistant content.
            Positions with IGNORE_INDEX (-100) are non-assistant content (user/system) and will be excluded from search.
        mask_answer_from_first_video: If True and there are multiple video segments (>1), 
            mask the first video from answer tokens.
        mask_grounding_from_other_videos: If True and there are multiple video segments (>1), 
            mask all videos except the first video from grounding tokens.
    
    Returns:
        4D tensor of shape [batch_size, 1, seq_len, seq_len]. 
        For bool dtype: True indicates allowed attention, False indicates blocked.
        For float dtype: 0.0 indicates allowed attention, large_negative indicates blocked (additive bias format).
    """
    # Ensure we're working with the correct shape
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected 2D attention mask, got shape {attention_mask.shape}")
    
    B, L = attention_mask.shape
    # print(f"B: {B}, L: {L}")
    device = attention_mask.device
    
    # Convert to bool mask (1/True for valid tokens, 0/False for padding)
    # Use >= 1 to handle cases where mask might have values > 1
    valid = (attention_mask >= 1).bool()
    
    # Create causal mask (lower triangular, including diagonal)
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
    
    # Initialize allowed mask with causal mask: [B, L, L]
    # Use broadcasting instead of loop for efficiency
    allowed = causal.unsqueeze(0).expand(B, -1, -1).clone()  # [B, L, L]
    
    # Apply padding mask: both query and key must be valid
    # Use broadcasting for efficiency
    valid_query = valid.unsqueeze(1)  # [B, 1, L] - for queries (rows)
    valid_key = valid.unsqueeze(2)    # [B, L, 1] - for keys (columns)
    allowed = allowed & valid_query & valid_key  # [B, L, L]

    # Create modified input_ids for searching answer/grounding: mask non-assistant positions with -1
    # This allows us to search only within assistant content without checking mask in the search loop
    input_ids_for_search = None
    if input_ids is not None and labels is not None:
        input_ids_for_search = input_ids.clone()  # [B, L]
        for b in range(B):
            # Replace non-assistant positions (where labels == IGNORE_INDEX) with -1
            non_assistant_mask = (labels[b] == IGNORE_INDEX)
            input_ids_for_search[b][non_assistant_mask] = -1
    
    # Apply answer-to-first-video masking if enabled
    if mask_answer_from_first_video and input_ids is not None and tokenizer is not None:
        try:
            # Answer segment: fixed token IDs for <answer> / </answer> (tokenizer may tokenize differently)
            answer_start_id = [27, 9217]
            answer_end_id = [522, 9217]
            vision_start_id = tokenizer("<|vision_start|>").input_ids
            vision_end_id = tokenizer("<|vision_end|>").input_ids

            if answer_start_id is not None and answer_end_id is not None and \
               vision_start_id is not None and vision_end_id is not None:
                vision_start_token = vision_start_id[0] if len(vision_start_id) > 0 else None
                vision_end_token = vision_end_id[0] if len(vision_end_id) > 0 else None

                for b in range(B):
                    ids = input_ids[b]
                    ids_for_search = input_ids_for_search[b] if input_ids_for_search is not None else ids

                    # Locate all frame-level vision segments (use full ids)
                    v_starts = torch.nonzero(ids == vision_start_token, as_tuple=False).squeeze(-1)
                    v_ends = torch.nonzero(ids == vision_end_token, as_tuple=False).squeeze(-1)

                    # Pair frame-level start/end tags
                    v_ptr, e_ptr = 0, 0
                    frame_segments = []
                    while v_ptr < v_starts.numel() and e_ptr < v_ends.numel():
                        if v_starts[v_ptr] < v_ends[e_ptr]:
                            frame_segments.append((v_starts[v_ptr].item(), v_ends[e_ptr].item()))
                            v_ptr += 1
                            e_ptr += 1
                        else:
                            e_ptr += 1

                    # Cluster frames into video-level segments.
                    # Qwen3VL uses interleaved timestamps so each frame has its own
                    # vision_start/vision_end; consecutive frames of the same video
                    # are separated by short timestamp tokens (~6-10 tokens) while
                    # different videos are separated by much longer text.
                    vision_segments = _group_frame_segments_to_video_segments(frame_segments)

                    if len(vision_segments) > 1:
                        first_video_start, first_video_end = vision_segments[0]
                        first_video_indices = torch.arange(first_video_start + 1, first_video_end, device=device, dtype=torch.long)

                        # Find answer segments by sequence match in assistant content (ids_for_search)
                        answer_starts = _find_sequence_positions(ids_for_search, answer_start_id)
                        answer_ends = _find_sequence_positions(ids_for_search, answer_end_id)
                        
                        # Adjust answer_end positions to point to the end of the sequence
                        if answer_ends.numel() > 0:
                            answer_ends = answer_ends + len(answer_end_id) - 1
                        
                        a_ptr, e_ptr = 0, 0
                        answer_segments = []
                        while a_ptr < answer_starts.numel() and e_ptr < answer_ends.numel():
                            if answer_starts[a_ptr] < answer_ends[e_ptr]:
                                # answer_start points to the start of the sequence, answer_end points to the end
                                answer_segments.append((answer_starts[a_ptr].item(), answer_ends[e_ptr].item()))
                                a_ptr += 1
                                e_ptr += 1
                            else:
                                e_ptr += 1
                        
                        # When multiple answer segments exist, only the last one is masked from the first video
                        if len(answer_segments) > 1:
                            answer_segments = answer_segments[-1:]
                        elif len(answer_segments) == 0:
                            print(f"[WARNING_ATTN_MASK] No answer segments found, skipping masking")
                            # 打印 input_ids 时不省略中间内容
                            import numpy as np
                            np.set_printoptions(threshold=np.inf)
                            torch.set_printoptions(threshold=float('inf'))
                            print("input_ids: ", input_ids[b])
                            continue

                        for ans_start, ans_end in answer_segments:
                            # Skip the answer tag tokens themselves, only mask the content
                            answer_indices = torch.arange(ans_start + len(answer_start_id), ans_end, device=device, dtype=torch.long)
                            if answer_indices.numel() > 0 and first_video_indices.numel() > 0:
                                # Set attention from answer tokens (queries) to first video tokens (keys) to False
                                allowed[b][answer_indices.unsqueeze(1), first_video_indices] = False
        except Exception as e:
            # If any error occurs (e.g., token not found), just continue without this masking
            import warnings
            warnings.warn(f"Failed to apply answer-to-first-video masking: {e}. Continuing without this feature.")
    
    # Apply grounding-to-other-videos masking if enabled
    if mask_grounding_from_other_videos and input_ids is not None and tokenizer is not None:
        try:
            # Grounding segment: fixed token IDs for <grounding> / </grounding>
            grounding_start_id = [27, 1951, 287]
            grounding_end_id = [5361, 1951, 287]
            vision_start_id = tokenizer("<|vision_start|>").input_ids
            vision_end_id = tokenizer("<|vision_end|>").input_ids

            if grounding_start_id is not None and grounding_end_id is not None and \
               vision_start_id is not None and vision_end_id is not None:
                vision_start_token = vision_start_id[0] if len(vision_start_id) > 0 else None
                vision_end_token = vision_end_id[0] if len(vision_end_id) > 0 else None

                for b in range(B):
                    ids = input_ids[b]
                    ids_for_search = input_ids_for_search[b] if input_ids_for_search is not None else ids

                    v_starts = torch.nonzero(ids == vision_start_token, as_tuple=False).squeeze(-1)
                    v_ends = torch.nonzero(ids == vision_end_token, as_tuple=False).squeeze(-1)

                    # Pair frame-level start/end tags
                    v_ptr, e_ptr = 0, 0
                    frame_segments = []
                    while v_ptr < v_starts.numel() and e_ptr < v_ends.numel():
                        if v_starts[v_ptr] < v_ends[e_ptr]:
                            frame_segments.append((v_starts[v_ptr].item(), v_ends[e_ptr].item()))
                            v_ptr += 1
                            e_ptr += 1
                        else:
                            e_ptr += 1

                    # Cluster frames into video-level segments (same reasoning as above)
                    vision_segments = _group_frame_segments_to_video_segments(frame_segments)

                    # Only apply masking if there are more than 1 video segments
                    if len(vision_segments) > 1:
                        # Get all video segments except the first video
                        # Index 0 is first video, index 1+ are other videos
                        other_video_indices = []
                        for vid_idx in range(1, len(vision_segments)):
                            vid_start, vid_end = vision_segments[vid_idx]
                            # Get all token indices within the video segment (exclusive of start/end tags)
                            vid_indices = torch.arange(vid_start + 1, vid_end, device=device, dtype=torch.long)
                            other_video_indices.append(vid_indices)

                        # Combine all other video indices
                        if other_video_indices:
                            all_other_video_indices = torch.cat(other_video_indices)
                            
                            # Find grounding segments - sequence matching (only in assistant content)
                            # Use modified ids_for_search which has non-assistant positions masked with -1
                            grounding_starts = _find_sequence_positions(ids_for_search, grounding_start_id)
                            grounding_ends = _find_sequence_positions(ids_for_search, grounding_end_id)
                            
                            # Adjust grounding_end positions to point to the end of the sequence
                            if grounding_ends.numel() > 0:
                                grounding_ends = grounding_ends + len(grounding_end_id) - 1
                            
                            # Pair grounding start and end tags
                            g_ptr, e_ptr = 0, 0
                            grounding_segments = []
                            while g_ptr < grounding_starts.numel() and e_ptr < grounding_ends.numel():
                                if grounding_starts[g_ptr] < grounding_ends[e_ptr]:
                                    # grounding_start points to the start of the sequence, grounding_end points to the end
                                    grounding_segments.append((grounding_starts[g_ptr].item(), grounding_ends[e_ptr].item()))
                                    g_ptr += 1
                                    e_ptr += 1
                                else:
                                    e_ptr += 1
                            
                            # Mask other videos from all grounding tokens (while preserving causality)
                            # Skip the first grounding segment since it only has the first video before it
                            # (all_other_video_indices doesn't include the first video, so no masking needed)
                            for grd_idx1, (grd_start, grd_end) in enumerate(grounding_segments):
                                if grd_idx1 == 0:
                                    continue

                                grounding_indices = torch.arange(grd_start + len(grounding_start_id), grd_end, device=device, dtype=torch.long)
                                if grounding_indices.numel() > 0 and all_other_video_indices.numel() > 0:
                                    # Only mask other video tokens that come BEFORE each grounding token (preserve causality)
                                    # For each grounding token (query), only mask other video tokens (keys) that are at earlier positions
                                    for grd_idx in grounding_indices:
                                        # Find other video tokens that come before this grounding token
                                        earlier_video_indices = all_other_video_indices[all_other_video_indices < grd_idx]
                                        if earlier_video_indices.numel() > 0:
                                            # Set attention from this grounding token (query) to earlier other video tokens (keys) to False
                                            allowed[b][grd_idx, earlier_video_indices] = False
        except Exception as e:
            # If any error occurs (e.g., token not found), just continue without this masking
            import warnings
            warnings.warn(f"Failed to apply grounding-to-other-videos masking: {e}. Continuing without this feature.")
    
    # Add head dimension: [B, 1, L, L]
    allowed = allowed.unsqueeze(1)
    
    # Convert to specified dtype if needed
    if dtype is not None:
        if dtype == torch.bool:
            return allowed
        else:
            # For float dtypes, convert to additive bias format:
            # True (allowed) -> 0.0, False (blocked) -> large_negative
            try:
                min_dtype = torch.finfo(dtype).min
            except ValueError:
                # For integer dtypes, use a large negative value
                min_dtype = torch.iinfo(dtype).min if hasattr(torch.iinfo(dtype), 'min') else -1e9
            
            zero_tensor = torch.zeros(1, dtype=dtype, device=device)
            min_tensor = torch.full((1,), min_dtype, dtype=dtype, device=device)
            
            # Convert bool to float: True -> 0.0, False -> min_dtype
            result = torch.where(allowed, zero_tensor, min_tensor)
            return result
    
    return allowed
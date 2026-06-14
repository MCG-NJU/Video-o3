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

import os
import tempfile
from typing import TYPE_CHECKING, Optional

from datasets import DatasetDict, load_from_disk

from ..extras import logging
from ..extras.misc import has_tokenized_data
from .data_utils import get_dataset_module, merge_dataset

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


def needs_normalize(dataset, n: int = 10) -> bool:
    r"""Check if any of the first n samples need normalization (missing 'sample' field in videos)."""
    for i in range(min(n, len(dataset))):
        example = dataset[i]
        videos = example.get("videos")
        if not isinstance(videos, list):
            continue
        for video_item in videos:
            if isinstance(video_item, dict) and "sample" not in video_item:
                return True
    return False


def normalize_videos_field(example: dict) -> dict:
    r"""
    Normalize the videos field by adding 'sample' field with value "old_quota" if missing.
    
    Some datasets have: List({'crop': List(Value('float64')), 'sample': Value('string'), 'url': Value('string')})
    Some datasets have: List({'crop': List(Value('float64')), 'url': Value('string')})
    We need to ensure all have the 'sample' field.
    """
    if "videos" in example and example["videos"] is not None:
        videos = example["videos"]
        if isinstance(videos, list):
            normalized_videos = []
            # add_count = 0
            for video_item in videos:
                if isinstance(video_item, dict):
                    # Add 'sample' field if missing
                    if "sample" not in video_item:
                        video_item = video_item.copy()
                        video_item["sample"] = "old_quota"
                        # add_count += 1
                    normalized_videos.append(video_item)
                else:
                    normalized_videos.append(video_item)
            # print("为%d个视频添加了sample字段" % add_count)
            example["videos"] = normalized_videos
    return example


def load_tokenized_dataset(tokenized_path: str) -> "DatasetDict":
    r"""Load a tokenized dataset from disk, similar to loader.py:288-297."""
    if not has_tokenized_data(tokenized_path):
        raise ValueError(f"Tokenized dataset not found at {tokenized_path}.")

    logger.info_rank0(f"Loading tokenized dataset from {tokenized_path}...")
    tokenized_data = load_from_disk(tokenized_path)
    
    # Ensure we have a DatasetDict
    if not isinstance(tokenized_data, DatasetDict):
        tokenized_data = DatasetDict({"train": tokenized_data})
    
    return tokenized_data


def merge_tokenized_datasets(
    dataset_paths: list[str],
    output_path: str,
    data_args: Optional["DataArguments"] = None,
    mix_strategy: str = "concat",
    interleave_probs: Optional[list[float]] = None,
    sample_sizes: Optional[list[int]] = None,
    seed: int = 42,
    streaming: bool = False,
) -> "DatasetDict":
    r"""
    Merge multiple tokenized datasets into one.
    
    This function mimics the loading logic in loader.py:288-300 and merges multiple
    preprocessed tokenized datasets together.
    
    Args:
        dataset_paths: List of paths to tokenized datasets to merge
        output_path: Path to save the merged dataset
        data_args: Optional DataArguments for advanced merging options
        mix_strategy: Strategy to merge datasets ("concat" or "interleave")
        interleave_probs: Probabilities for interleave strategy (optional)
        sample_sizes: Optional list of sample sizes for each dataset. 
                     -1 means use all data, positive integer means random sample that many.
                     If None, all datasets are used fully.
        seed: Random seed for sampling and interleave strategy
        streaming: Whether to use streaming mode (not recommended for merging)
    
    Returns:
        The merged DatasetDict
    """
    if not dataset_paths:
        raise ValueError("At least one dataset path must be provided.")
    
    # Set temporary directory to a writable location to avoid permission issues
    # Use output_path parent directory if available, otherwise use system temp
    if output_path:
        temp_dir = os.path.join(os.path.dirname(output_path), ".tmp")
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.gettempdir()
    
    # Set environment variables for datasets library to use writable temp directory
    original_tmpdir = os.environ.get("TMPDIR")
    original_tmp = os.environ.get("TMP")
    os.environ["TMPDIR"] = temp_dir
    os.environ["TMP"] = temp_dir
    
    try:
        # Validate sample_sizes if provided
        if sample_sizes is not None:
            if len(sample_sizes) != len(dataset_paths):
                raise ValueError(
                    f"sample_sizes length ({len(sample_sizes)}) must match "
                    f"number of datasets ({len(dataset_paths)})"
                )
            for i, size in enumerate(sample_sizes):
                if size != -1 and size < 1:
                    raise ValueError(
                        f"sample_sizes[{i}] must be -1 (use all) or a positive integer, got {size}"
                    )
        
        if len(dataset_paths) == 1:
            logger.warning_rank0("Only one dataset provided, copying instead of merging.")
            dataset = load_tokenized_dataset(dataset_paths[0])
            
            # Normalize videos field format (add 'sample' field if missing)
            for split_name in dataset.keys():
                if not needs_normalize(dataset[split_name]):
                    logger.info_rank0(f"Skipping normalization for {split_name} split (first 10 samples all have 'sample' field).")
                    continue
                logger.info_rank0(f"Normalizing videos field format in {split_name} split...")
                dataset[split_name] = dataset[split_name].map(
                    normalize_videos_field,
                    desc="Normalizing videos field",
                    keep_in_memory=True,
                    load_from_cache_file=False,
                    cache_file_name=None,
                )
            
            # Apply sampling if requested
            if sample_sizes is not None and sample_sizes[0] != -1:
                for split_name in dataset.keys():
                    original_size = len(dataset[split_name])
                    sample_size = min(sample_sizes[0], original_size)
                    if sample_size < original_size:
                        dataset[split_name] = dataset[split_name].shuffle(seed=seed).select(range(sample_size))
                        logger.info_rank0(
                            f"Sampled {sample_size} samples from {original_size} in {split_name} split"
                        )
            
            if output_path:
                dataset.save_to_disk(output_path)
                logger.info_rank0(f"Dataset copied to {output_path}.")
            return dataset
        
        logger.info_rank0(f"Merging {len(dataset_paths)} tokenized datasets...")
        
        # Load all datasets and apply sampling if requested
        datasets_dict = {}
        for i, path in enumerate(dataset_paths):
            logger.info_rank0(f"Loading dataset {i+1}/{len(dataset_paths)}: {path}")
            dataset_dict = load_tokenized_dataset(path)
            
            # Extract train dataset from each DatasetDict
            if "train" in dataset_dict:
                dataset = dataset_dict["train"]
            else:
                # If no train split, take the first available split
                first_split = list(dataset_dict.keys())[0]
                dataset = dataset_dict[first_split]
                logger.warning_rank0(
                    f"No 'train' split found in {path}, using '{first_split}' instead."
                )
            
            # Normalize videos field format (add 'sample' field if missing)
            if not needs_normalize(dataset):
                logger.info_rank0(f"  Skipping normalization (first 10 samples all have 'sample' field).")
            else:
                logger.info_rank0(f"  Normalizing videos field format (adding 'sample' field if missing)...")
                dataset = dataset.map(
                    normalize_videos_field,
                    desc="Normalizing videos field",
                    keep_in_memory=True,
                    load_from_cache_file=False,
                    cache_file_name=None,
                )
            
            # Apply sampling if sample_sizes is provided
            if sample_sizes is not None:
                sample_size = sample_sizes[i]
                if sample_size != -1:
                    original_size = len(dataset)
                    sample_size = min(sample_size, original_size)
                    if sample_size < original_size:
                        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
                        logger.info_rank0(
                            f"  Sampled {sample_size} samples from {original_size} (requested: {sample_sizes[i]})"
                        )
                    elif sample_size == original_size:
                        logger.info_rank0(
                            f"  Using all {original_size} samples (requested: {sample_sizes[i]})"
                        )
                else:
                    logger.info_rank0(f"  Using all {len(dataset)} samples (sample_size=-1)")

            # Always include the (optionally sampled) dataset in the merge list.
            datasets_dict[f"dataset_{i}"] = dataset
        
        # Merge datasets using the same logic as in loader.py
        datasets_list = list(datasets_dict.values())
        
        # Use data_args if provided, otherwise create minimal config
        if data_args is not None:
            merged_dataset = merge_dataset(datasets_list, data_args, seed=seed)
        else:
            # Simple merge without data_args
            from datasets import concatenate_datasets, interleave_datasets
            
            if mix_strategy == "concat":
                merged_dataset = concatenate_datasets(datasets_list)
            elif mix_strategy.startswith("interleave"):
                if interleave_probs is None:
                    interleave_probs = [1.0 / len(datasets_list)] * len(datasets_list)
                elif len(interleave_probs) != len(datasets_list):
                    raise ValueError(
                        f"interleave_probs length ({len(interleave_probs)}) "
                        f"must match number of datasets ({len(datasets_list)})"
                    )
                
                # Normalize probabilities
                total_prob = sum(interleave_probs)
                interleave_probs = [p / total_prob for p in interleave_probs]
                
                stopping_strategy = (
                    "first_exhausted" if mix_strategy.endswith("under") else "all_exhausted"
                )
                merged_dataset = interleave_datasets(
                    datasets=datasets_list,
                    probabilities=interleave_probs,
                    seed=seed,
                    stopping_strategy=stopping_strategy,
                )
            else:
                raise ValueError(f"Unknown mix_strategy: {mix_strategy}")
        
        # Wrap in DatasetDict
        if isinstance(merged_dataset, DatasetDict):
            merged_dict = merged_dataset
        else:
            merged_dict = DatasetDict({"train": merged_dataset})
        
        # Save to disk if output_path is provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            merged_dict.save_to_disk(output_path)
        
        return merged_dict
    finally:
        # Restore original environment variables
        if original_tmpdir is not None:
            os.environ["TMPDIR"] = original_tmpdir
        elif "TMPDIR" in os.environ:
            del os.environ["TMPDIR"]
        
        if original_tmp is not None:
            os.environ["TMP"] = original_tmp
        elif "TMP" in os.environ:
            del os.environ["TMP"]


def merge_tokenized_datasets_cli():
    r"""Command-line interface for merging tokenized datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge multiple preprocessed tokenized datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset_paths",
        type=str,
        nargs="+",
        help="Paths to tokenized datasets to merge",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged dataset",
    )
    parser.add_argument(
        "--mix_strategy",
        type=str,
        default="concat",
        choices=["concat", "interleave", "interleave_under"],
        help="Strategy to merge datasets: concat (concatenate) or interleave",
    )
    parser.add_argument(
        "--interleave_probs",
        type=float,
        nargs="+",
        default=None,
        help="Probabilities for interleave strategy (optional, default: uniform)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for interleave strategy and sampling",
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Sample sizes for each dataset. -1 means use all data, positive integer means random sample that many. "
             "Must match the number of dataset paths. If not provided, all datasets are used fully.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists",
    )
    
    args = parser.parse_args()
    
    # Check output path
    if os.path.exists(args.output_path) and not args.overwrite:
        raise ValueError(
            f"输出路径已存在: {args.output_path}\n"
            "使用 --overwrite 参数来覆盖现有数据集"
        )
    
    print("=" * 60)
    print("合并 tokenized 数据集")
    print("=" * 60)
    print(f"数据集数量: {len(args.dataset_paths)}")
    print(f"合并策略: {args.mix_strategy}")
    if args.sample_sizes:
        print(f"采样数量: {args.sample_sizes}")
        print("  (-1 表示使用全部数据)")
    else:
        print("采样数量: 全部使用")
    print(f"输出路径: {args.output_path}")
    print("=" * 60)
    
    # Merge datasets
    merged_dataset = merge_tokenized_datasets(
        dataset_paths=args.dataset_paths,
        output_path=args.output_path,
        mix_strategy=args.mix_strategy,
        interleave_probs=args.interleave_probs,
        sample_sizes=args.sample_sizes,
        seed=args.seed,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("合并结果:")
    print("=" * 60)
    for split_name, split_data in merged_dataset.items():
        print(f"  {split_name}: {len(split_data)} 样本")
    print("=" * 60)
    
    print(f"\n✓ 合并完成!")
    print(f"\n使用方法:")
    print(f"  在配置文件中设置: tokenized_path: {args.output_path}")


if __name__ == "__main__":
    merge_tokenized_datasets_cli()


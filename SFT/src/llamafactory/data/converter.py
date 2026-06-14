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
import hashlib
import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from datasets import load_from_disk

from ..extras import logging
from ..extras.misc import has_tokenized_data
from .data_utils import Role


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .mm_plugin import AudioInput, ImageInput, VideoInput
    from .parser import DatasetAttr

    MediaType = Union[ImageInput, VideoInput, AudioInput]


logger = logging.get_logger(__name__)

# 备用路径替换：public-dataset-p2 -> public-dataset-p2-1，按位置直接拼接，避免 replace 的重复扫描
_FALLBACK_OLD_PREFIX = "public-dataset-p2"
_FALLBACK_NEW_PREFIX = "public-dataset-p2-1"
_FALLBACK_OLD_LEN = len(_FALLBACK_OLD_PREFIX)


def _fallback_media_path(path: str) -> Optional[str]:
    """若 path 中包含 public-dataset-p2，则按精确位置替换为 public-dataset-p2-1，否则返回 None。"""
    idx = path.find(_FALLBACK_OLD_PREFIX)
    if idx < 0:
        return None
    return path[:idx] + _FALLBACK_NEW_PREFIX + path[idx + _FALLBACK_OLD_LEN :]


@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _find_medias(self, medias: Union["MediaType", list["MediaType"], None]) -> Optional[list["MediaType"]]:
        r"""Optionally concatenate media path to media dir when loading from local disk."""
        if medias is None:
            return None
        elif not isinstance(medias, list):
            medias = [medias]
        elif len(medias) == 0:
            return None
        else:
            medias = medias[:]

        if self.dataset_attr.load_from in ["script", "file"]:
            fallback_count = 0
            # 如果是单个路径
            if isinstance(medias[0], str):
                for i in range(len(medias)):
                    original_media = medias[i]
                    media_path = os.path.join(self.data_args.media_dir, original_media)
                    if os.path.isfile(media_path):
                        medias[i] = media_path
                    else:
                        # 尝试 public-dataset-p2 -> public-dataset-p2-1 的备用路径（按位置替换，避免 replace 重复扫描）
                        alt_media = _fallback_media_path(original_media)
                        if alt_media is not None:
                            fallback_count += 1
                            medias[i] = os.path.join(self.data_args.media_dir, alt_media)
                            continue
                        # 两个路径都不存在：保留原始路径并告警
                        logger.warning_rank0_once(
                            f"[WARNING] Media (medias[i]: {original_media}, media_path: {media_path}) does not exist in `media_dir`. Use original path."
                        )
            # 如果是列表
            elif isinstance(medias[0], list):  # for processed video frames
                # medias is a list of lists, e.g., [[frame1.jpg, frame2.jpg], [frame3.jpg, frame4.jpg]]
                for i in range(len(medias)):
                    for j in range(len(medias[i])):
                        original_media = medias[i][j]
                        media_path = os.path.join(self.data_args.media_dir, original_media)
                        if os.path.isfile(media_path):
                            medias[i][j] = media_path
                        else:
                            alt_media = _fallback_media_path(original_media)
                            if alt_media is not None:
                                fallback_count += 1
                                medias[i][j] = os.path.join(self.data_args.media_dir, alt_media)
                                continue
                            logger.warning_rank0_once(
                                f"[WARNING] Media (medias[i][j]: {original_media}, media_path: {media_path}) does not exist in `media_dir`. Use original path."
                            )
            # 如果是字典
            elif isinstance(medias[0], dict):
                for i in range(len(medias)):
                    original_media = medias[i]["url"]
                    media_path = os.path.join(self.data_args.media_dir, original_media)
                    if os.path.isfile(media_path):
                        medias[i]["url"] = media_path
                    else:
                        alt_media = _fallback_media_path(original_media)
                        if alt_media is not None:
                            fallback_count += 1
                            medias[i]["url"] = os.path.join(self.data_args.media_dir, alt_media)
                            continue
                        logger.warning_rank0_once(
                            f"[WARNING] Media (medias[i]['url']: {original_media}, media_path: {media_path}) does not exist in `media_dir`. Use original path."
                        )

            # if fallback_count > 0:
            #     logger.warning_rank0_once(
            #         f"[INFO] Replaced {fallback_count} media path(s) with fallback (public-dataset-p2 -> public-dataset-p2-1)."
            #     )

        return medias

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""Convert a single example in the dataset to the standard format."""
        ...


@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

        if self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], str)
            and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(example[self.dataset_attr.response], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
        else:  # unsupervised
            response = []

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


@dataclass
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


@dataclass
class OpenAIDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }

        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example.get(self.dataset_attr.system, "") if self.dataset_attr.system else ""

        aligned_messages = []
        tool_responses = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            role = message[self.dataset_attr.role_tag]
            content = message[self.dataset_attr.content_tag]

            if role in [self.dataset_attr.assistant_tag, self.dataset_attr.function_tag]:
                if "tool_calls" in message and len(message["tool_calls"]) > 0:
                    tool_calls_list = [tool["function"] for tool in message["tool_calls"]]
                    content = json.dumps(tool_calls_list, ensure_ascii=False)
                    role = self.dataset_attr.function_tag

            if role == self.dataset_attr.observation_tag:
                tool_responses.append(content)
                continue
            elif len(tool_responses) > 0:
                _content = "\n</tool_response>\n<tool_response>\n".join(tool_responses)
                aligned_messages.append(
                    {
                        "role": Role.OBSERVATION.value,
                        "content": _content,
                    }
                )
                tool_responses = []

            aligned_messages.append(
                {
                    "role": tag_mapping[role],
                    "content": content,
                }
            )

        odd_tags = (Role.USER.value, Role.OBSERVATION.value)
        even_tags = (Role.ASSISTANT.value, Role.FUNCTION.value)
        accept_tags = (odd_tags, even_tags)
        for turn_idx, message in enumerate(aligned_messages):
            if message["role"] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        tools = example.get(self.dataset_attr.tools, "") if self.dataset_attr.tools else ""
        if isinstance(tools, dict) or isinstance(tools, list):
            tools = json.dumps(tools, ensure_ascii=False)

        short_system_prompt = "detailed thinking off"
        if not system:
            if not tools:
                system = short_system_prompt
            else:
                pass
        else:
            if not tools:
                if "detailed thinking on" in system or "detailed thinking off" in system:
                    pass
                else:
                    system += "\n" + short_system_prompt
            else:
                system += "\n"

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": tools,
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
    "openai": OpenAIDatasetConverter,
}


def register_dataset_converter(name: str, dataset_converter: type["DatasetConverter"]) -> None:
    r"""Register a new dataset converter."""
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")

    DATASET_CONVERTERS[name] = dataset_converter


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments") -> "DatasetConverter":
    r"""Get a dataset converter."""
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Align the dataset to a specific format.

    Aligned dataset:
    _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
    _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
    _system: "..."
    _tools: "..."
    _images: []
    _videos: []
    _audios: []
    """
    # 如果指定了 tokenized_path，尝试保存/加载转换后的数据集
    aligned_dataset_path = None
    if data_args.tokenized_path is not None and not data_args.streaming:
        # Multiple paths are not allowed when saving
        base_path = data_args.tokenized_path
        if isinstance(base_path, list):
            if len(base_path) > 1:
                raise ValueError(
                    "Multiple paths are not allowed when saving tokenized datasets. "
                    "Please provide only a single path for saving, or use multiple paths only when loading existing datasets."
                )
            base_path = base_path[0] if len(base_path) > 0 else None
        elif isinstance(base_path, str) and "," in base_path:
            raise ValueError(
                "Multiple paths (comma-separated) are not allowed when saving tokenized datasets. "
                "Please provide only a single path for saving, or use multiple paths only when loading existing datasets."
            )
        
        if base_path is None:
            aligned_dataset_path = None
        else:
            # 生成转换后数据集的保存路径：与 shard 命名方式对齐
            # 生成数据集标识符：使用数据集名称、子集、分割等信息
            dataset_id_parts = [
                dataset_attr.dataset_name or "unknown",
                dataset_attr.subset or "",
                dataset_attr.split or "train",
                dataset_attr.formatting or "alpaca",
            ]
            dataset_id = "_".join(filter(None, dataset_id_parts))
            # 使用 hash 确保路径不会太长，并区分不同的数据集
            dataset_id_hash = hashlib.md5(dataset_id.encode()).hexdigest()[:8]
            # 命名方式与 shard 对齐：{base_path}_converted_{hash}
            aligned_dataset_path = f"{base_path}_converted_{dataset_id_hash}"
            
            # 检查是否已存在转换后的数据集
            if has_tokenized_data(aligned_dataset_path) and not data_args.overwrite_cache:
                logger.info_rank0(f"检测到已转换的数据集，尝试加载: {aligned_dataset_path}")
                try:
                    aligned_dataset = load_from_disk(aligned_dataset_path)
                    logger.info_rank0(f"成功加载转换后的数据集，跳过格式转换步骤")

                    # 直接返回
                    return aligned_dataset
                except Exception as e:
                    logger.warning_rank0(f"加载转换后的数据集失败，将重新转换: {e}")
            elif data_args.overwrite_cache:
                logger.info_rank0(f"overwrite_cache=True，将重新转换数据集（忽略已存在的: {aligned_dataset_path}）")
            else:
                logger.info_rank0(f"未找到已转换的数据集，将执行格式转换: {aligned_dataset_path}")
    
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    aligned_dataset = dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )
    
    # 保存转换后的数据集（只保存 Dataset 类型，不保存 IterableDataset）
    if aligned_dataset_path is not None and training_args.should_save:
        from datasets import Dataset
        if isinstance(aligned_dataset, Dataset):
            try:
                aligned_dataset.save_to_disk(aligned_dataset_path)
                logger.info_rank0(f"转换后的数据集已保存到: {aligned_dataset_path}")
            except Exception as e:
                logger.warning_rank0(f"保存转换后的数据集失败: {e}")
    
    return aligned_dataset

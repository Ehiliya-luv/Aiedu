# -*- coding: utf-8 -*-
"""VERL data preparation — JSONL → parquet conversion and prompt handling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from ._common import _as_project_path, _max_prompt_length, _project_root
from .rl_data import RLTrainingSample, load_rl_samples

logger = logging.getLogger(__name__)


def _split_samples(samples: List[RLTrainingSample], val_size: int) -> Tuple[List[RLTrainingSample], List[RLTrainingSample]]:
    if not samples:
        raise ValueError("empty training samples")
    val_size = max(1, min(int(val_size), len(samples)))
    val_samples = samples[:val_size]
    train_samples = samples[val_size:] or samples
    return train_samples, val_samples


def _sample_to_verl_row(sample: RLTrainingSample, index: int, split: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    prompt_text = prompt if prompt is not None else sample.prompt
    return {
        "data_source": "aiedu/question_generation",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "medical_question_generation",
        "reward_model": {
            "style": "rule",
            "ground_truth": sample.expert_revision,
        },
        "extra_info": {
            "split": split,
            "index": index,
            "input_context": sample.input_context,
            "prompt_text": prompt_text,
            "raw_prompt_text": sample.prompt,
            "model_output": sample.model_output,
            "raw_reference": sample.expert_revision,
            "metadata": sample.metadata,
        },
    }


def _find_jsonl_source(data_dir: Path) -> Path:
    preferred_names = ("rl_train.jsonl", "train.jsonl", "rl_train_extra.jsonl")
    for name in preferred_names:
        candidate = data_dir / name
        if candidate.exists():
            return candidate

    candidates = sorted(data_dir.glob("*.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"{data_dir} 下存在多个 jsonl，无法自动判断训练数据。"
            f"请用 --dataset 明确指定其中一个: {names}"
        )
    raise FileNotFoundError(f"{data_dir} 下未找到可转换为 VERL parquet 的 jsonl 文件")


def _resolve_dataset_source(dataset_name: str) -> Tuple[Path, Optional[Path]]:
    source = _as_project_path(dataset_name)
    if source.is_dir():
        train_parquet = source / "train.parquet"
        if train_parquet.exists():
            return source, None
        return _find_jsonl_source(source), None
    if source.exists():
        return source, None
    if source.suffix.lower() == ".parquet":
        return _find_jsonl_source(source.parent), source.parent
    raise FileNotFoundError(f"训练数据不存在: {source}")


def _resolve_existing_verl_parquet(source: Path) -> Optional[Tuple[str, str]]:
    if source.is_dir():
        train_path = source / "train.parquet"
        val_path = source / "val.parquet"
        if train_path.exists() and val_path.exists():
            return str(train_path.resolve()), str(val_path.resolve())
        if train_path.exists():
            logger.warning("只找到 %s，未找到 val.parquet；验证集将复用 train.parquet。", train_path)
            resolved = str(train_path.resolve())
            return resolved, resolved
        return None

    if source.suffix.lower() == ".parquet":
        val_path = source.with_name("val.parquet")
        if val_path.exists() and source.name == "train.parquet":
            return str(source.resolve()), str(val_path.resolve())
        logger.warning("数据源是单个 parquet：%s；训练集和验证集将复用该文件。", source)
        resolved = str(source.resolve())
        return resolved, resolved

    return None


def _chat_token_ids(tokenizer, prompt: str, enable_thinking: bool) -> List[int]:
    def normalize_ids(value) -> List[int]:
        if hasattr(value, "ids"):
            return list(value.ids)
        if hasattr(value, "get"):
            input_ids = value.get("input_ids", None)
            if input_ids is not None:
                return normalize_ids(input_ids)
        if isinstance(value, dict):
            return normalize_ids(value.get("input_ids", []))
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)) and value and hasattr(value[0], "ids"):
            return list(value[0].ids)
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            return list(value[0])
        return list(value or [])

    messages = [{"role": "user", "content": prompt}]
    if callable(getattr(tokenizer, "apply_chat_template", None)):
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                thinking_mode="on" if enable_thinking else "off",
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        normalized = normalize_ids(ids)
        if len(normalized) > 2:
            return normalized
    encoded = tokenizer(prompt, add_special_tokens=True)
    return normalize_ids(encoded)


def _truncate_prompt_to_budget(
    tokenizer, prompt: str, max_prompt_length: int, truncation: str, enable_thinking: bool
) -> Tuple[str, int, int]:
    original_len = len(_chat_token_ids(tokenizer, prompt, enable_thinking))
    if original_len <= max_prompt_length:
        return prompt, original_len, original_len

    content_ids = tokenizer(prompt, add_special_tokens=False).get("input_ids", [])
    if not content_ids:
        return prompt, original_len, original_len

    def build_candidate(keep: int) -> str:
        keep = max(1, min(keep, len(content_ids)))
        if truncation == "right":
            ids = content_ids[:keep]
        elif truncation == "middle":
            left = keep // 2
            right = keep - left
            ids = content_ids[:left] + content_ids[-right:]
        else:
            ids = content_ids[-keep:]
        return tokenizer.decode(ids, skip_special_tokens=True)

    best_prompt = build_candidate(1)
    best_len = len(_chat_token_ids(tokenizer, best_prompt, enable_thinking))
    low, high = 1, len(content_ids)
    while low <= high:
        mid = (low + high) // 2
        candidate = build_candidate(mid)
        candidate_len = len(_chat_token_ids(tokenizer, candidate, enable_thinking))
        if candidate_len <= max_prompt_length:
            best_prompt = candidate
            best_len = candidate_len
            low = mid + 1
        else:
            high = mid - 1

    return best_prompt, original_len, best_len


def _prepare_verl_rows(
    samples: List[RLTrainingSample],
    split: str,
    tokenizer,
    max_prompt_length: int,
    truncation: str,
    enable_thinking: bool,
) -> List[Dict[str, Any]]:
    rows = []
    truncated = 0
    max_before = 0
    max_after = 0
    for idx, sample in enumerate(samples):
        prompt, before, after = _truncate_prompt_to_budget(
            tokenizer=tokenizer,
            prompt=sample.prompt,
            max_prompt_length=max_prompt_length,
            truncation=truncation,
            enable_thinking=enable_thinking,
        )
        max_before = max(max_before, before)
        max_after = max(max_after, after)
        if before != after:
            truncated += 1
        rows.append(_sample_to_verl_row(sample, idx, split, prompt=prompt))

    if truncated:
        logger.warning(
            "VERL %s prompt 截断: %d/%d，max_prompt_length=%d，最长截断前=%d，最长截断后=%d。",
            split, truncated, len(samples), max_prompt_length, max_before, max_after,
        )
    else:
        logger.info("VERL %s prompt 长度检查通过: rows=%d, max_prompt_tokens=%d/%d", split, len(samples), max_after, max_prompt_length)
    return rows


def _write_verl_parquet(
    dataset_name: str,
    train_samples_limit: int,
    verl_val_size: int,
    output_dir: str,
    verl_data_dir: Optional[str],
    model_path: str,
    max_new_tokens: int,
    vllm_max_model_length: Optional[int],
    truncation: str,
    enable_thinking: bool,
) -> Tuple[str, str]:
    from datasets import Dataset

    source, requested_parquet_dir = _resolve_dataset_source(dataset_name)
    existing_parquet = _resolve_existing_verl_parquet(source)
    if existing_parquet is not None:
        logger.info("使用已有 VERL parquet 数据: train=%s, val=%s", existing_parquet[0], existing_parquet[1])
        return existing_parquet

    if source.suffix.lower() != ".jsonl":
        raise ValueError(
            "VERL 数据源必须是 Aiedu JSONL、单个 parquet，或包含 train.parquet/val.parquet 的目录；"
            f"当前数据源: {source}"
        )

    max_items = train_samples_limit if train_samples_limit > 0 else None
    samples = load_rl_samples(str(source), max_items=max_items, strict=True)
    train_samples, val_samples = _split_samples(samples, verl_val_size)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    max_prompt_length = _max_prompt_length(vllm_max_model_length, max_new_tokens)
    train_rows = _prepare_verl_rows(train_samples, "train", tokenizer, max_prompt_length, truncation, enable_thinking)
    val_rows = _prepare_verl_rows(val_samples, "val", tokenizer, max_prompt_length, truncation, enable_thinking)

    data_dir = _as_project_path(
        verl_data_dir or requested_parquet_dir or (Path(output_dir) / "verl_data")
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"

    Dataset.from_list(train_rows).to_parquet(str(train_path))
    Dataset.from_list(val_rows).to_parquet(str(val_path))

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"VERL parquet 生成失败: train={train_path}, val={val_path}")

    logger.info("VERL 数据已生成: train=%s rows=%d, val=%s rows=%d", train_path, len(train_rows), val_path, len(val_rows))
    return str(train_path.resolve()), str(val_path.resolve())

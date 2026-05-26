# -*- coding: utf-8 -*-
"""DPO 训练数据加载。"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DPOSample:
    prompt: str
    chosen: str
    rejected: str
    meta: Dict[str, Any]


def _safe_load_json_line(line: str, line_no: int, file_path: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        logger.warning("跳过非法 JSON（%s:%s）: %s", file_path, line_no, exc)
        return None
    if not isinstance(obj, dict):
        logger.warning("跳过非对象 JSON（%s:%s）", file_path, line_no)
        return None
    return obj


def _pick_text(obj: Dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if isinstance(value, str):
        return value.strip()
    return ""


def _normalize_record(obj: Dict[str, Any], index: int, strict: bool) -> Optional[DPOSample]:
    prompt = _pick_text(obj, "prompt")
    chosen = _pick_text(obj, "chosen")
    rejected = _pick_text(obj, "rejected")

    missing = [name for name, value in (("prompt", prompt), ("chosen", chosen), ("rejected", rejected)) if not value]
    if missing:
        msg = f"样本#{index} 缺失关键字段: {', '.join(missing)}"
        if strict:
            raise ValueError(msg)
        logger.warning("%s，已跳过", msg)
        return None

    if chosen == rejected:
        msg = f"样本#{index} 的 chosen 与 rejected 完全一致"
        if strict:
            raise ValueError(msg)
        logger.warning("%s，已跳过", msg)
        return None

    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    return DPOSample(prompt=prompt, chosen=chosen, rejected=rejected, meta=meta)


def load_dpo_samples(
    path: str,
    max_items: Optional[int] = None,
    strict: bool = True,
) -> List[DPOSample]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    if not path.endswith(".jsonl"):
        raise ValueError(f"当前仅支持 .jsonl 文件: {path}")

    samples: List[DPOSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            if max_items is not None and len(samples) >= max_items:
                break
            line = raw_line.strip()
            if not line:
                continue
            obj = _safe_load_json_line(line, line_no, path)
            if obj is None:
                if strict:
                    raise ValueError(f"JSON 解析失败，行号: {line_no}")
                continue
            sample = _normalize_record(obj, index=line_no, strict=strict)
            if sample is not None:
                samples.append(sample)

    if not samples:
        raise ValueError(f"未从 {path} 加载到有效 DPO 样本")

    logger.info("已加载 DPO 样本 %d 条: %s", len(samples), path)
    return samples


def build_dpo_dataset(
    path: str,
    max_items: Optional[int] = None,
    strict: bool = True,
) -> Dataset:
    samples = load_dpo_samples(path=path, max_items=max_items, strict=strict)
    records = [
        {
            "prompt": sample.prompt,
            "chosen": sample.chosen,
            "rejected": sample.rejected,
            "meta": json.dumps(sample.meta, ensure_ascii=False),
        }
        for sample in samples
    ]
    logger.info("DPO 数据集构建完成: total=%d", len(records))
    return Dataset.from_list(records)


__all__ = [
    "DPOSample",
    "load_dpo_samples",
    "build_dpo_dataset",
]

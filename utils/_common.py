# -*- coding: utf-8 -*-
"""Shared internal utilities for VERL training orchestration."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Optional

import torch

from .model import HAS_TORCH_NPU

logger = __import__("logging").getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_root() -> Path:
    return _PROJECT_ROOT


def _as_project_path(path: str | os.PathLike) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return (_PROJECT_ROOT / raw).resolve()


def _reward_function_path() -> str:
    return str(_PROJECT_ROOT / "utils" / "verl_reward.py")


def _check_verl_available() -> None:
    if importlib.util.find_spec("verl") is None:
        raise RuntimeError(
            "当前环境未安装 verl。请先在训练环境安装 VERL，例如：pip install verl"
        )


def _device_count(device: str) -> int:
    if device == "cuda" and torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    if device == "npu" and HAS_TORCH_NPU and hasattr(torch, "npu"):
        try:
            return max(1, torch.npu.device_count())
        except Exception:
            return 1
    return 1


def _max_prompt_length(vllm_max_model_length: Optional[int], max_new_tokens: int) -> int:
    max_model_len = vllm_max_model_length or 8192
    return max(512, int(max_model_len) - int(max_new_tokens))


__all__ = [
    "_project_root",
    "_as_project_path",
    "_reward_function_path",
    "_check_verl_available",
    "_device_count",
    "_max_prompt_length",
]

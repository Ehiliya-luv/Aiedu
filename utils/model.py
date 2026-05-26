# -*- coding: utf-8 -*-
"""Minimal model-path resolution and runtime-device detection."""

from __future__ import annotations

import logging
import os

import torch

try:
    import torch_npu  # noqa: F401

    HAS_TORCH_NPU = True
except ImportError:
    HAS_TORCH_NPU = False

logger = logging.getLogger(__name__)

LOCAL_MODEL_DIR = "./resources/model"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# 国内优先走 hf-mirror 镜像，不影响已设置 HF_ENDPOINT 的环境
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_ALIAS_MAP = {
    "Qwen/Qwen2.5-0.5B": "Qwen__Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen__Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-14B-Instruct": "Qwen__Qwen3-14B-Instruct",
}

# 模型别名/简称 → HuggingFace 官方 repo_id 映射
# 用于处理非标准名称（如不含 org 的模型名、旧版本号等）
MODEL_NAME_NORMALIZE = {
    # 百川 M2 系列
    "Baichuan-M2-32B-0226": "baichuan-inc/Baichuan-M2-32B",
    "Baichuan-M2-32B": "baichuan-inc/Baichuan-M2-32B",
    "baichuan-m2-32b": "baichuan-inc/Baichuan-M2-32B",
    "Baichuan-M2-7B-0226": "baichuan-inc/Baichuan-M2-7B",
    "Baichuan-M2-7B": "baichuan-inc/Baichuan-M2-7B",
    # Qwen2.5 简称
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    # Qwen3 简称
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B-Instruct": "Qwen/Qwen3-14B-Instruct",
    "HuatuoGPT-3-32B": "FreedomIntelligence/HuatuoGPT-3-32B",
}


def _to_hf_repo_id(name: str) -> str:
    """将 Qwen__Qwen2.5-0.5B-Instruct 格式转为 Qwen/Qwen2.5-0.5B-Instruct。

    仅当 name 含有恰好两个 __ 分隔段时才替换（即 org__model 格式），
    避免把含 __ 的普通模型名误转。
    """
    parts = name.split("__")
    if len(parts) == 2:
        return "/".join(parts)
    return name.replace("__", "/")


def _normalize_model_name(name: str) -> str:
    """将模型别名/简称映射为 HuggingFace 官方 repo_id。

    查找顺序：MODEL_NAME_NORMALIZE 精确匹配 → 原样返回。
    """
    return MODEL_NAME_NORMALIZE.get(name.strip(), name.strip())


def _download_to_local(hf_name: str, local_path: str) -> None:
    from huggingface_hub import snapshot_download

    parent_dir = os.path.dirname(local_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    logger.info("正在从 HuggingFace 下载模型: %s -> %s", hf_name, local_path)
    try:
        snapshot_download(
            repo_id=hf_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:
        logger.warning("snapshot_download 失败: %s，尝试 from_pretrained", exc)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(hf_name, trust_remote_code=True)
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)


def resolve_model_path(model_path: str) -> str:
    if not model_path:
        raise ValueError("model_path is empty")

    if os.path.exists(model_path):
        return model_path

    is_local_path_format = (
        model_path.startswith("./")
        or model_path.startswith("../")
        or model_path.startswith("/")
        or model_path.startswith("resources/")
        or model_path.startswith("output/")
        or model_path.startswith("C:\\")
        or model_path.startswith("c:\\")
    )
    if is_local_path_format:
        dir_name = os.path.basename(model_path.replace("\\", "/"))
        hf_name = _to_hf_repo_id(dir_name)
        hf_name = _normalize_model_name(hf_name)   # 别名 → 官方 repo_id
        _download_to_local(hf_name, model_path)
        return model_path

    # 处理本地目录格式（如 Qwen__Qwen2.5-0.5B-Instruct）
    if "__" in model_path:
        local_path = os.path.join(LOCAL_MODEL_DIR, model_path)
        hf_name = _to_hf_repo_id(model_path)
        hf_name = _normalize_model_name(hf_name)   # 别名 → 官方 repo_id
    else:
        # 先做名称规范化（如 Baichuan-M2-32B-0226 → baichuan-inc/Baichuan-M2-32B）
        hf_name = _normalize_model_name(model_path)
        alias = MODEL_ALIAS_MAP.get(hf_name)
        local_path = os.path.join(LOCAL_MODEL_DIR, alias if alias is not None else hf_name.replace("/", "__"))

    if os.path.exists(local_path):
        return local_path
    _download_to_local(hf_name, local_path)
    return local_path


def detect_runtime_device() -> str:
    try:
        import torch_npu

        if torch_npu.npu.is_available():
            return "npu"
    except ImportError:
        pass

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

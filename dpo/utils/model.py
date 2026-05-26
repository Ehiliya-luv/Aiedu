# -*- coding: utf-8 -*-
"""DPO 训练所需的最小模型工具集。"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

LOCAL_MODEL_DIR = "./resources/model"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)


def resolve_model_path(model_path: str) -> str:
    """解析模型路径：本地存在则直接返回，否则按 HuggingFace repo_id 下载到本地缓存。"""

    def normalize_baichuan_name(hf_name: str) -> str:
        baichuan_mapping = {
            "Baichuan-M2-32B-0226": "baichuan-inc/Baichuan-M2-32B",
            "Baichuan-M2-32B": "baichuan-inc/Baichuan-M2-32B",
            "baichuan-m2-32b": "baichuan-inc/Baichuan-M2-32B",
            "Baichuan-M2-7B-0226": "baichuan-inc/Baichuan-M2-7B",
            "Baichuan-M2-7B": "baichuan-inc/Baichuan-M2-7B",
        }
        return baichuan_mapping.get(hf_name.strip(), hf_name)

    is_local_path_format = (
        model_path.startswith("./")
        or model_path.startswith("../")
        or model_path.startswith("/")
        or model_path.startswith("resources/")
        or model_path.startswith("output/")
        or model_path.startswith("C:\\")
        or model_path.startswith("c:\\")
        or model_path.startswith("data/")
    )

    if is_local_path_format:
        if os.path.exists(model_path):
            logger.info("使用本地模型: %s", model_path)
            return model_path

        dir_name = os.path.basename(model_path.replace("\\", "/"))
        hf_name = normalize_baichuan_name(dir_name.replace("__", "/"))
        logger.info("本地模型不存在，将下载: %s", hf_name)
        _download_to_local(hf_name, model_path)
        return model_path

    hf_name_normalized = normalize_baichuan_name(model_path)
    local_path = os.path.join(LOCAL_MODEL_DIR, hf_name_normalized.replace("/", "__"))
    if os.path.exists(local_path):
        logger.info("使用本地模型: %s", local_path)
        return local_path

    logger.info("本地模型不存在，将下载: %s -> %s", hf_name_normalized, local_path)
    _download_to_local(hf_name_normalized, local_path)
    return local_path


def _download_to_local(hf_name: str, local_path: str) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        logger.info("模型下载完成: %s", local_path)
    except Exception as exc:
        logger.warning("snapshot_download 失败: %s，尝试 from_pretrained", exc)
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(hf_name, trust_remote_code=True)
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
        logger.info("模型下载并保存完成: %s", local_path)


def detect_runtime_device() -> str:
    """检测当前训练主设备，优先级 NPU > CUDA > CPU。"""
    try:
        import torch_npu

        if torch_npu.npu.is_available():
            return "npu"
    except ImportError:
        pass

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


__all__ = [
    "resolve_model_path",
    "detect_runtime_device",
]

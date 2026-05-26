# -*- coding: utf-8 -*-
"""基于 TRL DPOTrainer 的 DPO 训练实现。"""

from __future__ import annotations

import logging
import os
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer

from .data import build_dpo_dataset
from .model import detect_runtime_device, resolve_model_path

logger = logging.getLogger(__name__)


@dataclass
class DPOTrainConfig:
    model_name_or_path: str
    dataset_path: str
    output_dir: str
    tokenizer_name_or_path: Optional[str] = None
    adapter_path: Optional[str] = None
    max_length: int = 4096
    max_prompt_length: int = 3072
    max_samples: int = -1
    beta: float = 0.1
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    warmup_steps: int = 0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    resume: bool = False
    enable_cache: bool = False
    use_gradient_checkpointing: bool = True
    full_finetune: bool = False
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def _parse_target_modules(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _use_bf16(runtime_device: str) -> bool:
    if runtime_device == "npu":
        return True
    if runtime_device == "cuda" and torch.cuda.is_available():
        return bool(torch.cuda.is_bf16_supported())
    return False


def _prepare_tokenizer(tokenizer_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_lora_config(config: DPOTrainConfig) -> LoraConfig:
    return LoraConfig(
        r=int(config.lora_r),
        lora_alpha=int(config.lora_alpha),
        lora_dropout=float(config.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_parse_target_modules(config.lora_target_modules),
    )


def _build_model_init_kwargs(config: DPOTrainConfig, runtime_device: str) -> Dict[str, Any]:
    use_bf16 = _use_bf16(runtime_device)
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if runtime_device == "cuda" else None)

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    if bool(config.use_qlora):
        if runtime_device == "cuda" and not bool(config.full_finetune):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            )
            model_kwargs["device_map"] = "auto"
        elif runtime_device != "cuda":
            model_kwargs["device_map"] = "auto"
            logger.info("检测到非 CUDA 设备：保留 LoRA 训练路径，不启用 4bit 量化，并启用 device_map=auto 自动切分。")
    elif runtime_device == "npu":
        model_kwargs["device_map"] = "auto"
        logger.info("NPU 环境启用 device_map=auto 自动切分模型。")
    return model_kwargs


def _load_trainable_peft_model(adapter_path: str, runtime_device: str):
    adapter_resolved_path = resolve_model_path(adapter_path)
    use_bf16 = _use_bf16(runtime_device)
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if runtime_device == "cuda" else None)

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "is_trainable": True,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if runtime_device == "npu":
        model_kwargs["device_map"] = "auto"
        logger.info("继续训练 LoRA adapter：NPU 环境启用 device_map=auto 自动切分模型。")

    logger.info("从 adapter 恢复可训练 PEFT 模型: %s", adapter_resolved_path)
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_resolved_path, **model_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def _find_resume_checkpoint(output_dir: str) -> Optional[str]:
    """Return a usable checkpoint path from output_dir or output_dir itself."""
    if not os.path.isdir(output_dir):
        return None

    if re.fullmatch(r"checkpoint-\d+", os.path.basename(os.path.normpath(output_dir))):
        return output_dir

    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        return last_checkpoint

    checkpoint_dirs = []
    for name in os.listdir(output_dir):
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        if match:
            checkpoint_dirs.append((int(match.group(1)), os.path.join(output_dir, name)))
    if not checkpoint_dirs:
        return None
    return max(checkpoint_dirs, key=lambda item: item[0])[1]


def _read_checkpoint_step(checkpoint_dir: str) -> Optional[int]:
    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as exc:
        logger.warning("读取 checkpoint 状态失败: %s (%s)", state_path, exc)
        return None
    step = state.get("global_step")
    return int(step) if isinstance(step, int) else None


def _checkpoint_step_from_name(checkpoint_dir: str) -> Optional[int]:
    match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(os.path.normpath(checkpoint_dir)))
    if not match:
        return None
    return int(match.group(1))


def _estimate_total_train_steps(config: DPOTrainConfig, dataset_size: int) -> int:
    per_step_samples = max(1, int(config.per_device_train_batch_size)) * max(1, int(config.gradient_accumulation_steps))
    steps_per_epoch = math.ceil(dataset_size / per_step_samples)
    return int(math.ceil(float(config.num_train_epochs) * steps_per_epoch))


def train_dpo(config: DPOTrainConfig) -> str:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["AIEDU_ENABLE_CACHE"] = "1" if config.enable_cache else "0"

    runtime_device = detect_runtime_device()
    logger.info("DPO 训练运行设备: %s", runtime_device)

    model_path = resolve_model_path(config.model_name_or_path)
    tokenizer_path = resolve_model_path(config.tokenizer_name_or_path or config.model_name_or_path)
    logger.info("Policy 模型路径: %s", model_path)
    logger.info("Tokenizer 路径: %s", tokenizer_path)

    tokenizer = _prepare_tokenizer(tokenizer_path)
    max_items = config.max_samples if config.max_samples > 0 else None
    dataset = build_dpo_dataset(
        path=config.dataset_path,
        max_items=max_items,
        strict=True,
    )

    resume_from_checkpoint = None
    completed_steps = 0
    total_train_steps = _estimate_total_train_steps(config, len(dataset))
    if bool(config.resume):
        resume_from_checkpoint = _find_resume_checkpoint(config.output_dir)
        if resume_from_checkpoint:
            checkpoint_step = _read_checkpoint_step(resume_from_checkpoint)
            if checkpoint_step is None:
                checkpoint_step = _checkpoint_step_from_name(resume_from_checkpoint)
            completed_steps = int(checkpoint_step or 0)
            if completed_steps <= 0:
                raise RuntimeError(
                    f"resume 已启用并检测到 checkpoint，但无法确定已完成步数: {resume_from_checkpoint}"
                )
            remaining_steps = max(0, total_train_steps - completed_steps)
            logger.info(
                "检测到 checkpoint，续跑训练: %s (已完成 step=%d，总 step=%d，剩余 step=%d)",
                resume_from_checkpoint,
                completed_steps,
                total_train_steps,
                remaining_steps,
            )
        else:
            logger.info("resume 已启用，但 %s 下没有可用 checkpoint，将从头开始训练。", config.output_dir)

    training_kwargs = {
        "output_dir": config.output_dir,
        "beta": float(config.beta),
        "max_length": int(config.max_length),
        "num_train_epochs": float(config.num_train_epochs),
        "per_device_train_batch_size": int(config.per_device_train_batch_size),
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "warmup_steps": int(config.warmup_steps),
        "lr_scheduler_type": str(config.lr_scheduler_type),
        "logging_steps": int(config.logging_steps),
        "save_steps": int(config.save_steps),
        "save_total_limit": int(config.save_total_limit),
        "seed": int(config.seed),
        "bf16": bool(_use_bf16(runtime_device)),
        "remove_unused_columns": False,
        "report_to": [],
        "gradient_checkpointing": bool(config.use_gradient_checkpointing),
        "ignore_data_skip": False,
    }
    if bool(config.resume) and completed_steps > 0:
        remaining_steps = max(0, total_train_steps - completed_steps)
        if remaining_steps <= 0:
            logger.info("checkpoint 已达到或超过目标总步数，无需继续训练。")
            return config.output_dir
        training_kwargs["max_steps"] = total_train_steps
    if config.adapter_path:
        training_args = DPOConfig(**training_kwargs)
        model = _load_trainable_peft_model(config.adapter_path, runtime_device)
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        model_init_kwargs = _build_model_init_kwargs(config, runtime_device)
        training_kwargs["model_init_kwargs"] = model_init_kwargs
        training_args = DPOConfig(**training_kwargs)
        trainer = DPOTrainer(
            model=model_path,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=None if bool(config.full_finetune) else _build_lora_config(config),
        )

    logger.info("开始 DPO 训练，样本数=%d", len(dataset))
    if bool(config.resume) and completed_steps > 0:
        logger.info("续跑目标: 从全局 step %d 继续到 step %d，本次剩余 %d 个更新步", completed_steps, total_train_steps, total_train_steps - completed_steps)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    os.makedirs(config.output_dir, exist_ok=True)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("DPO 训练完成并保存：%s", config.output_dir)
    return config.output_dir


__all__ = [
    "DPOTrainConfig",
    "train_dpo",
]

# -*- coding: utf-8 -*-
"""VERL checkpoint consolidation — FSDP shards → HuggingFace model.

Uses VERL's official ``model_merger`` CLI to correctly handle FSDP sharded
state dicts (including DTensor / placement-aware merging), producing a
complete HuggingFace model that can be loaded directly with
``AutoModelForCausalLM.from_pretrained()``.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ._common import _project_root

logger = logging.getLogger(__name__)


def _find_latest_verl_checkpoint(output_dir: str) -> Optional[Path]:
    """在 VERL 输出目录中查找最新的 global_step_X checkpoint。

    VERL 的 checkpoint 目录结构：
        {output_dir}/global_step_{N}/actor/model_world_size_{W}_rank_{R}.pt
    """
    out = Path(output_dir)
    if not out.exists():
        return None

    ckpt_dirs = sorted(
        [d for d in out.iterdir() if d.is_dir() and d.name.startswith("global_step_")],
        key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else 0,
    )
    return ckpt_dirs[-1] if ckpt_dirs else None


def _check_judge_error_in_logs(output_dir: str) -> Optional[str]:
    """检查训练日志中是否有 judge 相关的崩溃标记。"""
    log_file = Path(_project_root()) / "training.log"
    if not log_file.exists():
        return None
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            tail = f.readlines()[-200:]
        for line in tail:
            if "JudgeAuthError" in line:
                return "[JudgeAuthError] 认证/余额错误 - 训练终止。请检查 API Key 或充值后重试。"
            if "JudgeRateLimitError" in line:
                return "[JudgeRateLimitError] 速率限制耗尽 - API 持续受限，请降低请求频率或扩大配额。"
            if "JudgeAPIError" in line:
                return "[JudgeAPIError] Judge API 不可恢复错误 - 请检查 API 部署状态。"
    except Exception:
        pass
    return None


def merge_fsdp_checkpoint(output_dir: str, target_dir: Optional[str] = None) -> Optional[str]:
    """将 VERL FSDP 分片 checkpoint 合并为完整 HuggingFace 模型。

    使用 VERL 官方 ``model_merger`` CLI 正确处理 FSDP 分片（包括 DTensor
    和 placement-aware merging），输出完整 HuggingFace 模型，可直接用
    ``AutoModelForCausalLM.from_pretrained()`` 加载推理。

    用法:
        python -c "from utils.grpo import merge_fsdp_checkpoint; \
                   merge_fsdp_checkpoint('output/rl_model')"
    """
    latest = _find_latest_verl_checkpoint(output_dir)
    if latest is None:
        logger.error("未找到任何 checkpoint 目录: %s", output_dir)
        return None

    actor_dir = latest / "actor"
    if not actor_dir.is_dir():
        logger.error("actor 目录不存在: %s", actor_dir)
        return None

    # 检查 fsdp_config.json 是否存在（model_merger 需要）
    fsdp_config = actor_dir / "fsdp_config.json"
    if not fsdp_config.exists():
        logger.error("缺少 fsdp_config.json（model_merger 需要）: %s", actor_dir)
        return None

    merged_dir = Path(target_dir or output_dir) / "merged_hf"
    merged_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(actor_dir),
        "--target_dir", str(merged_dir),
    ]
    logger.info("调用 VERL model_merger: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 大模型合并可能较慢，给 10 分钟
        )
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                logger.info("[model_merger] %s", line)
        if result.returncode != 0:
            logger.error("model_merger 失败 (returncode=%d)", result.returncode)
            if result.stderr:
                for line in result.stderr.strip().splitlines():
                    logger.error("[model_merger] %s", line)
            return None
    except subprocess.TimeoutExpired:
        logger.error("model_merger 超时（>600s）")
        return None
    except FileNotFoundError:
        logger.error("无法执行 model_merger：请确认 verl 已安装（pip install verl）")
        return None

    logger.info("VERL model_merger 合并完成: %s", merged_dir)
    return str(merged_dir)

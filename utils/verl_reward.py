# -*- coding: utf-8 -*-
"""Custom reward function for VERL GRPO training."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.log_setup import log_reward, setup_worker_process_logging  # noqa: E402
from utils.reward import MIN_REWARD_FLOOR, compute_reward  # noqa: E402
from utils.text import extract_completion_text  # noqa: E402

# ── Ray worker / VERL 子进程入口：root logger 默认无 handler，导致 reward 端
# 的 logger.info / logger.debug 静默丢弃。此处装一个最小 stdout handler，
# 配合 Ray 默认 log_to_driver=True，让题型长度 / judge logprobs 等输出能流回
# 主进程终端，并经主进程的 Tee 落到 main_process.log。
#
# 注意：如果 VERL/Ray 在 worker 里已经装过 root handler，此函数会早 return。
# 实测部分场景下这条 logging 链不可达，所以 utils/reward.py 同时通过 log_reward
# 直接写 reward.log，作为兜底。
setup_worker_process_logging()

logger = logging.getLogger(__name__)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


# 模块加载时打印一次诊断日志，方便排查 Ray Worker 环境变量是否正确传递
_ENGINE_PORT = os.environ.get("AIEDU_JUDGE_VLLM_ENGINE_PORT", "(未设置)")
_PYENGINE = os.environ.get("AIEDU_JUDGE_LOCAL_VLLM_PYENGINE", "(未设置)")
_CUDA_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES", "(未设置)")
logger.info(
    "[verl_reward] 模块加载 | pid=%d | AIEDU_JUDGE_LOCAL_VLLM_PYENGINE=%s | "
    "AIEDU_JUDGE_VLLM_ENGINE_PORT=%s | CUDA_VISIBLE_DEVICES=%s",
    os.getpid(), _PYENGINE, _ENGINE_PORT, _CUDA_VISIBLE,
)
# 同步写一行到 reward.log，作为 worker 可达性诊断（每个 worker 进程一条）
log_reward(
    f"[verl_reward] 模块加载 | pid={os.getpid()} | "
    f"AIEDU_JUDGE_LOCAL_VLLM_PYENGINE={_PYENGINE} | "
    f"AIEDU_JUDGE_VLLM_ENGINE_PORT={_ENGINE_PORT} | "
    f"CUDA_VISIBLE_DEVICES={_CUDA_VISIBLE}"
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | None = None,
    extra_info: Dict[str, Any] | None = None,
    **kwargs,
) -> float:
    """VERL custom reward entry.

    VERL calls this function for each generated response. The Aiedu reward
    primarily scores the generated question set itself, with prompt_text
    available for judge context.
    """
    extra_info = extra_info or {}
    completion_text = extract_completion_text(solution_str)
    prompt_text = _as_text(extra_info.get("prompt_text"))
    revised = _as_text(ground_truth or extra_info.get("raw_reference"))

    try:
        reward = compute_reward(
            original=completion_text,
            revised=revised,
            floor=MIN_REWARD_FLOOR,
            prompt_text=prompt_text,
            metadata={
                "data_source": data_source,
                **{k: v for k, v in extra_info.items() if k != "metadata"},
            },
        )
    except Exception as exc:
        # 即便 compute_reward 内部异常，也要在 reward.log 留下痕迹，
        # 否则 VERL 只会看到一个静默的 fallback reward，根本看不出问题。
        log_reward(
            f"[verl_reward] compute_reward EXCEPTION | data_source={data_source} | "
            f"completion_len={len(completion_text)} | error={type(exc).__name__}: {exc}"
        )
        raise

    final_reward = float(reward)
    # 末尾确认行：让用户能从 reward.log 直接看到"哪个样本被评分了，最终 reward 是多少"。
    # 即使 compute_reward 内的 4 处 log_reward 因任何原因没写出来，这一行也会到。
    log_reward(
        f"[verl_reward] compute_score 返回 | data_source={data_source} | "
        f"completion_len={len(completion_text)} | reward={final_reward:.4f}"
    )
    return final_reward


__all__ = ["compute_score"]

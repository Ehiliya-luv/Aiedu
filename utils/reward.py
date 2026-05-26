# -*- coding: utf-8 -*-
"""Unified reward entry for GRPO training."""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Union

import numpy as np

from .judge import get_default_judge
from .log_setup import log_completion, log_reward
from .structural_reward import (
    SECTION_ORDER,
    analyze_structure,
    extract_question_sections,
    is_benign_objective_prefix,
)

logger = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    return float(max(np.float32(0.0), min(np.float32(1.0), np.float32(value))))


def _normalize_score_to_reward(score_01: float) -> float:
    """把 score_01 ∈ [0, 1]（即 mean_score / 9）线性映射到 reward ∈ [-1, 1]。

    数学上等价于 (mean_score - 4.5) / 4.5：
        score_01 = mean / 9
        reward   = (score_01 - 0.5) / 0.5
                 = (mean/9 - 0.5) / 0.5
                 = (mean - 4.5) / 4.5

    锚点：mean=4.5 → reward=0；mean=0 → reward=-1；mean=9 → reward=+1。
    mean=7（C 档下半部，"良好稳定可用"）→ reward ≈ +0.556。
    """
    score_01 = _clamp01(score_01)
    return float((score_01 - 0.5) / 0.5)


# 2026-05 调整：原值 0.05 是 logprobs 提取异常时的兜底（避免 -1 极端 reward），
# 现在 chat template + logprobs=10 已修复 + judge prompt 重写后 mean_score 0
# 应该真正可达，floor 必须放开到 0.0 才能让"严重缺陷"的题真正拿到 -1.0
# reward——GRPO 组内归一化时，组里 -1 + +0.5 比 -0.9 + +0.5 信号差距更大。
MIN_REWARD_FLOOR = float(np.float32(0.0))

DEFAULT_ORDER_PENALTY = 0.25
DEFAULT_DIRTY_PENALTY = 0.12
DEFAULT_EXTRA_TYPE_PENALTY = 0.20


def compute_reward(
    original: str,
    revised: str,
    device: str = "cpu",
    floor: float = MIN_REWARD_FLOOR,
    order_penalty: float = DEFAULT_ORDER_PENALTY,
    dirty_penalty: float = DEFAULT_DIRTY_PENALTY,
    extra_type_penalty: float = DEFAULT_EXTRA_TYPE_PENALTY,
    return_details: bool = False,
    current_idx: int = 0,
    total: int = 0,
    prompt_text: str = "",
    metadata: Dict | None = None,
) -> Union[float, Tuple[float, Dict]]:
    floor = _clamp01(floor)
    idx_str = f"[{current_idx}/{total}]" if current_idx > 0 and total > 0 else ""

    # 把 actor 完整 completion 写入 completion.log（默认关，--log-completion 启用）
    # 包括 actor 真实输出（original）和 prompt 上下文，便于 spot-check 训练前后差异。
    #
    # 历史 bug（已修）：之前这里写的是 `revised`（即 ground_truth = expert_revision），
    # 也就是训练数据里的专家答案，而不是 actor 这一次 rollout 的实际产出。
    # 因此 completion.log 里看到"知识拓展/知识类别/难易度"等字段是训练数据本身带的，
    # 不是 actor 输出 —— prompt 链路其实是干净的（只走 input_context）。
    # 修复方法：写 original，并把段名改成 actor_output 让语义不再误导。
    if original:
        log_completion(
            f"{idx_str} prompt_len={len(prompt_text)} original_len={len(original)} | "
            f"reference_len={len(revised) if revised else 0}\n"
            f"--- prompt(head 500) ---\n{prompt_text[:500]}\n"
            f"--- actor_output ---\n{original}\n"
            f"--- end ---"
        )

    sec_o, sec_o_spans = extract_question_sections(original)

    section_lens_o = {sec: len(sec_o.get(sec, "").strip()) for sec in SECTION_ORDER}
    logger.info(
        "%s 原文len=%d | 题型长度: A1=%d A2=%d A3=%d A4=%d B1=%d",
        idx_str,
        len(original),
        section_lens_o["A1"],
        section_lens_o["A2"],
        section_lens_o["A3"],
        section_lens_o["A4"],
        section_lens_o["B1"],
    )
    # 同步写到 reward.log（绕过 Ray/VERL 的 stdout 转发链，确保 reward
    # worker 进程的输出一定能落到 ./tmp/grpo/{ts}/reward.log）。
    log_reward(
        f"{idx_str} 原文len={len(original)} | 题型长度: "
        f"A1={section_lens_o['A1']} A2={section_lens_o['A2']} "
        f"A3={section_lens_o['A3']} A4={section_lens_o['A4']} "
        f"B1={section_lens_o['B1']}"
    )

    judge = get_default_judge()
    section_scores: Dict[str, float] = {}
    section_details: Dict[str, Dict] = {}
    for sec in SECTION_ORDER:
        candidate = sec_o.get(sec, "").strip()
        if not candidate:
            section_scores[sec] = floor
            logger.info("%s %s: 内容为空, score01=%.4f", idx_str, sec, floor)
            log_reward(f"{idx_str} {sec}: 内容为空, score01={floor:.4f}")
            continue

        judge_result = judge.score_section(
            question_type=sec,
            prompt_text=prompt_text,
            candidate_text=candidate,
        )
        section_scores[sec] = float(judge_result.get("score", floor))
        section_details[sec] = judge_result

        # 把 judge 的 logprobs / 0~9 概率分布 / 期望分完整写入 reward.log。
        # 这里不再用 logger.isEnabledFor(DEBUG) 守卫——reward.log 是独立文件，
        # 用户主动关心 judge 细节时打开它即可，不会污染 main_process.log。
        # 三种评分路径（HTTP API / vLLM Python engine 进程内 / vLLM daemon 远程）
        # 返回字段一致，所以单点日志即可统一覆盖。
        digit_logprobs = judge_result.get("digit_logprobs") or {}
        digit_probs = judge_result.get("digit_probabilities") or {}
        # sampled_token_text 是验证 skip-thinking 是否生效的关键观测信号：
        # 修复成功时这一列应是 '0'~'9' 之一；若是 '<think>' 或其他非数字 token，
        # 说明 vllm_scorer._thinking_strategy 探测错了或注入失败。
        sampled_text = str(judge_result.get("sampled_token_text", "")).replace("\n", "\\n")
        # sharpness 观测列：>1 时 digit_probabilities 是锐化后分布（更尖锐、期望更靠
        # 近 argmax），mean_score / score01 / reward 都跟着锐化；raw_digit_mass 仍是
        # 未锐化的原始 mass（用于诊断 logprobs 异常）。详见 vllm_scorer.VLLMEngineScorer。
        sharpness_val = float(judge_result.get("sharpness", 1.0) or 1.0)
        log_reward(
            f"{idx_str} {sec} judge[{judge_result.get('scoring_method', '?')}] "
            f"mean={float(judge_result.get('mean_score', 0.0)):.3f} "
            f"score01={float(judge_result.get('score', 0.0)):.4f} "
            f"reward={float(judge_result.get('normalized_mean_reward', 0.0)):.4f} | "
            f"sampled={sampled_text!r} "
            f"think_strategy={judge_result.get('thinking_strategy', '?')} "
            f"sharpness={sharpness_val:.2f} | "
            f"logprobs={ {d: round(lp, 3) for d, lp in digit_logprobs.items()} } | "
            f"probs={ {d: round(p, 3) for d, p in digit_probs.items()} } | "
            f"mass={float(judge_result.get('raw_digit_mass', 0.0)):.4f} "
            f"found={int(judge_result.get('found_digits', len(digit_logprobs)))}/"
            f"{int(judge_result.get('total_digits', 10))} "
            f"missing={judge_result.get('missing_digits', [])} | "
            f"model={judge_result.get('model', '')}"
        )

        # DEBUG（保留 logger 通道，万一日志能流回 main_process.log 也能看到）
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s %s judge[%s] mean=%.3f score01=%.4f reward=%.4f | "
                "logprobs=%s | probs=%s | mass=%.4f found=%d/%d missing=%s | model=%s",
                idx_str, sec,
                judge_result.get("scoring_method", "?"),
                float(judge_result.get("mean_score", 0.0)),
                float(judge_result.get("score", 0.0)),
                float(judge_result.get("normalized_mean_reward", 0.0)),
                {d: f"{lp:.3f}" for d, lp in digit_logprobs.items()},
                {d: f"{p:.3f}" for d, p in digit_probs.items()},
                float(judge_result.get("raw_digit_mass", 0.0)),
                int(judge_result.get("found_digits", len(digit_logprobs))),
                int(judge_result.get("total_digits", 10)),
                judge_result.get("missing_digits", []),
                judge_result.get("model", ""),
            )

    base_score_01 = float(np.mean(list(section_scores.values()))) if section_scores else floor
    base_score_01 = _clamp01(base_score_01)

    struct = analyze_structure(original, sections=sec_o, spans=sec_o_spans)
    penalty = 0.0
    if struct["is_order_correct"] < 0.5:
        penalty += float(order_penalty)
    penalty += float(dirty_penalty) * float(struct["dirty_rate"])
    if float(struct.get("has_extra_types", 0.0)) > 0.5:
        penalty += float(extra_type_penalty)

    final_score_01 = _clamp01(max(floor, base_score_01 - penalty))
    final_reward = _normalize_score_to_reward(final_score_01)

    logger.info(
        "%s 总得分: score01=%.4f reward=%.4f base01=%.4f penalty=%.4f | section_scores=%s",
        idx_str,
        final_score_01,
        final_reward,
        base_score_01,
        penalty,
        {k: f"{v:.4f}" for k, v in section_scores.items()},
    )
    log_reward(
        f"{idx_str} 总得分: score01={final_score_01:.4f} reward={final_reward:.4f} "
        f"base01={base_score_01:.4f} penalty={penalty:.4f} | "
        f"section_scores={ {k: round(v, 4) for k, v in section_scores.items()} } | "
        f"is_complete={struct.get('is_complete', 0)} "
        f"is_order_correct={struct.get('is_order_correct', 0)} "
        f"dirty_rate={float(struct.get('dirty_rate', 0.0)):.4f} "
        f"has_extra_types={bool(float(struct.get('has_extra_types', 0.0)) > 0.5)}"
    )

    if not return_details:
        return final_reward

    details = {
        "score": final_score_01,
        "reward": final_reward,
        "base_score": base_score_01,
        "penalty": float(penalty),
        "section_scores": section_scores,
        "section_details": section_details,
        "is_complete": struct["is_complete"],
        "is_order_correct": struct["is_order_correct"],
        "dirty_rate": struct["dirty_rate"],
        "has_extra_types": bool(float(struct.get("has_extra_types", 0.0)) > 0.5),
    }
    return final_reward, details


compute_structured_advanced_reward = compute_reward

__all__ = [
    "compute_reward",
    "compute_structured_advanced_reward",
    "analyze_structure",
    "extract_question_sections",
    "is_benign_objective_prefix",
    "SECTION_ORDER",
    "MIN_REWARD_FLOOR",
]

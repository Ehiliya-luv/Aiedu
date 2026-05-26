# -*- coding: utf-8 -*-
"""Per-question-type LLM judge — 三种评分模式。

模式 1 — HTTP API（默认）:
  调用 OpenAI 兼容 API（外部 ppio.com 或本地 vLLM HTTP server），
  通过 chat_complete_with_logprobs() 获取 top-K logprobs。
  K 由 JUDGE_SCORE_TOP_LOGPROBS 控制（默认 50，确保覆盖全部 10 个 digit）。

模式 2 — 本地 vLLM Python engine:
  直接使用 vLLM Python engine (LLM.generate) 获取输出 logprobs，
  然后从中精确提取 0~9 共 10 个数字 token 的 logprobs。
  通过 JUDGE_LOCAL_VLLM_PYENGINE 激活。

模式 3 — 本地 HTTP API（local_mode）:
  与模式 1 相同，但错误直接抛出不重试。

所有模式均无 fallback；任何错误直接传播。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .api import OpenAICompatibleClient
from .knowledge import JudgeKnowledgeBase, load_knowledge_base
from .prompts import build_section_distribution_prompt
from .settings import (
    JUDGE_ALLOW_EMPTY_API_KEY,
    JUDGE_API_BASE,
    JUDGE_API_KEY,
    JUDGE_CACHE_PATH,
    JUDGE_EMBEDDING_API_BASE,
    JUDGE_EMBEDDING_API_KEY,
    JUDGE_EMBEDDING_MODEL,
    JUDGE_KNOWLEDGE_TOP_K,
    JUDGE_MAX_OUTLINE_ITEMS,
    JUDGE_MAX_PDF_ADVICE,
    JUDGE_PDF_RECALL_TOP_K,
    JUDGE_RAG_MAX_TOKENS,
    JUDGE_XLSX_RECALL_TOP_K,
    JUDGE_LOCAL_MODE,
    JUDGE_LOCAL_VLLM_LORA_PATH,
    JUDGE_LOCAL_VLLM_PYENGINE,
    JUDGE_LOCAL_VLLM_MODEL_PATH,
    JUDGE_MODEL,
    JUDGE_REQUEST_TIMEOUT,
    JUDGE_SCORE_TOP_LOGPROBS,
    JUDGE_TEMPERATURE,
    JUDGE_VLLM_GPU_MEMORY_UTILIZATION,
    JUDGE_VLLM_MAX_MODEL_LEN,
    JUDGE_VLLM_TENSOR_PARALLEL_SIZE,
)

logger = logging.getLogger(__name__)

SCORE_DIGITS = tuple(str(i) for i in range(10))
_MIN_DIGIT_MASS = 0.5
_LOGPROBS_RETRY_BACKOFF = [1, 3, 10]
_SCORING_PROMPT_VERSION = "section_digit_prompt_v3"


def _path_stamp(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return f"missing:{path}"
    if p.is_file():
        return f"{p.resolve()}:{p.stat().st_mtime_ns}:{p.stat().st_size}"
    latest_mtime = 0
    total_size = 0
    for child in p.rglob("*"):
        if not child.is_file():
            continue
        stat = child.stat()
        latest_mtime = max(latest_mtime, stat.st_mtime_ns)
        total_size += stat.st_size
    return f"{p.resolve()}:{latest_mtime}:{total_size}"

# ── 懒加载 VLLMEngineScorer ──────────────────────────────────────
_VLLM_SCORER_CLASS = None


def _get_vllm_scorer_class():
    global _VLLM_SCORER_CLASS
    if _VLLM_SCORER_CLASS is None:
        try:
            from .vllm_scorer import VLLMEngineScorer
            _VLLM_SCORER_CLASS = VLLMEngineScorer
        except ImportError:
            raise ImportError(
                "vLLM 未安装。使用 VLLM Python engine 模式需要 vLLM。\n"
                "  pip install vllm"
            )
    return _VLLM_SCORER_CLASS


# ── HTTP API logprobs 提取 ────────────────────────────────────────


def _extract_digit_logprobs(response_payload: Dict[str, Any]) -> Dict[str, float]:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"judge response missing choices: {json.dumps(response_payload, ensure_ascii=False)[:1200]}")

    choice0 = choices[0]
    if not isinstance(choice0, dict):
        raise ValueError(f"judge response choice is invalid: {type(choice0)}")

    logprobs = choice0.get("logprobs")
    if not isinstance(logprobs, dict):
        raise ValueError(f"judge response missing logprobs: {json.dumps(choice0, ensure_ascii=False)[:1200]}")

    digit_logprobs: Dict[str, float] = {}

    content = logprobs.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            top_candidates = first.get("top_logprobs")
            if not isinstance(top_candidates, list):
                top_candidates = []
            for item in [first] + [candidate for candidate in top_candidates if isinstance(candidate, dict)]:
                token = str(item.get("token", "")).strip()
                if token in SCORE_DIGITS:
                    digit_logprobs[token] = float(item.get("logprob"))
            if digit_logprobs:
                return digit_logprobs

    tokens = logprobs.get("tokens")
    top_logprobs = logprobs.get("top_logprobs")
    token_logprobs = logprobs.get("token_logprobs")
    if isinstance(tokens, list) and tokens and isinstance(top_logprobs, list) and top_logprobs:
        first_top = top_logprobs[0]
        if isinstance(first_top, dict):
            for token, logprob in first_top.items():
                normalized = str(token).strip()
                if normalized in SCORE_DIGITS:
                    digit_logprobs[normalized] = float(logprob)
        first_token = str(tokens[0]).strip()
        if (
            isinstance(token_logprobs, list)
            and token_logprobs
            and first_token in SCORE_DIGITS
            and first_token not in digit_logprobs
        ):
            digit_logprobs[first_token] = float(token_logprobs[0])
        if digit_logprobs:
            return digit_logprobs

    raise ValueError(
        f"judge response missing usable logprobs content: {json.dumps(choice0, ensure_ascii=False)[:1600]}"
    )


def _build_distribution_stats(digit_logprobs: Dict[str, float]) -> Dict[str, Any]:
    if not digit_logprobs:
        raise ValueError("digit logprobs is empty")

    exp_values = {digit: math.exp(logprob) for digit, logprob in digit_logprobs.items()}
    raw_digit_mass = float(sum(exp_values.values()))
    if raw_digit_mass <= 0:
        raise ValueError("digit probability mass must be positive")

    digit_logprobs_sorted = {
        digit: float(digit_logprobs[digit])
        for digit in sorted(digit_logprobs.keys(), key=int)
    }
    digit_probabilities = {
        digit: float(exp_values[digit] / raw_digit_mass)
        for digit in sorted(exp_values.keys(), key=int)
    }
    mean_score = float(sum(int(digit) * prob for digit, prob in digit_probabilities.items()))
    normalized_mean_score = float(mean_score / 9.0)
    normalized_mean_reward = float((mean_score - 4.5) / 4.5)

    return {
        "digit_logprobs": digit_logprobs_sorted,
        "digit_probabilities": digit_probabilities,
        "raw_digit_mass": raw_digit_mass,
        "mean_score": mean_score,
        "normalized_mean_score": normalized_mean_score,
        "normalized_mean_reward": normalized_mean_reward,
        "missing_digits": [digit for digit in SCORE_DIGITS if digit not in digit_logprobs_sorted],
    }


# ── LLMSectionJudge ─────────────────────────────────────────────


class LLMSectionJudge:
    """LLM 题型片段评分器。

    支持三种评分模式（通过初始化参数选择）:
      1. HTTP API（默认）: 通过 OpenAICompatibleClient 调用外部/本地 vLLM API
      2. vLLM Python engine: local_vllm_pyengine=True, 直接调用 Python engine
      3. 本地 HTTP API: local_mode=True, 与 1 相同但错误直接抛出
    """

    def __init__(
        self,
        *,
        # ── 通用 ──
        model: str | None = None,
        cache_path: str | None = None,
        knowledge_top_k: int | None = None,
        temperature: float | None = None,
        # ── HTTP API 模式 ──
        api_base: str | None = None,
        api_key: str | None = None,
        allow_empty_api_key: bool | None = None,
        local_mode: bool | None = None,
        score_top_logprobs: int | None = None,
        # ── vLLM Python engine 模式 ──
        local_vllm_pyengine: bool | None = None,
        local_vllm_model_path: str | None = None,
        local_vllm_lora_path: str | None = None,
        vllm_max_model_len: int | None = None,
        vllm_gpu_memory_utilization: float | None = None,
        vllm_tensor_parallel_size: int | None = None,
        # ── 知识库 ──
        embedding_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        # 读取配置（支持运行时修改 settings 变量）
        self.model = str(model if model is not None else JUDGE_MODEL)
        self.knowledge_top_k = int(knowledge_top_k if knowledge_top_k is not None else JUDGE_KNOWLEDGE_TOP_K)
        self.temperature = float(temperature if temperature is not None else JUDGE_TEMPERATURE)
        self.score_top_logprobs = int(score_top_logprobs if score_top_logprobs is not None else JUDGE_SCORE_TOP_LOGPROBS)
        self.local_mode = JUDGE_LOCAL_MODE if local_mode is None else bool(local_mode)
        self.allow_empty_api_key = JUDGE_ALLOW_EMPTY_API_KEY if allow_empty_api_key is None else bool(allow_empty_api_key)

        # ── vLLM Python engine 模式 ──
        self.local_vllm_pyengine = JUDGE_LOCAL_VLLM_PYENGINE if local_vllm_pyengine is None else bool(local_vllm_pyengine)
        self.local_vllm_model_path = str(local_vllm_model_path if local_vllm_model_path is not None else JUDGE_LOCAL_VLLM_MODEL_PATH)
        self.local_vllm_lora_path = str(local_vllm_lora_path if local_vllm_lora_path is not None else JUDGE_LOCAL_VLLM_LORA_PATH)
        self.vllm_max_model_len = int(vllm_max_model_len if vllm_max_model_len is not None else JUDGE_VLLM_MAX_MODEL_LEN)
        self.vllm_gpu_memory_utilization = float(vllm_gpu_memory_utilization if vllm_gpu_memory_utilization is not None else JUDGE_VLLM_GPU_MEMORY_UTILIZATION)
        self.vllm_tensor_parallel_size = int(vllm_tensor_parallel_size if vllm_tensor_parallel_size is not None else JUDGE_VLLM_TENSOR_PARALLEL_SIZE)

        # ── 初始化评分引擎 ──
        self._vllm_engine_scorer = None
        self._vllm_engine_client = None
        self._client = None
        self.knowledge_base: Optional[JudgeKnowledgeBase] = None
        # RAG 启动期一次性 WARNING 标记（同 vllm_scorer.VLLMEngineScorer）：
        # 第一次评分发现 RAG 不可用时打一条 WARNING（含 disable_reasons），
        # 后续不重复刷屏；evidence.log 仍按 status=disabled 每条记一行。
        self._rag_disable_warned = False

        resolved_api_base = str(api_base if api_base is not None else JUDGE_API_BASE)
        resolved_api_key = str(api_key if api_key is not None else JUDGE_API_KEY)
        self.knowledge_base = load_knowledge_base(
            judge_api_base=resolved_api_base,
            judge_api_key=resolved_api_key,
            judge_model=self.model,
            embedding_api_base=embedding_api_base if embedding_api_base is not None else JUDGE_EMBEDDING_API_BASE,
            embedding_api_key=embedding_api_key if embedding_api_key is not None else JUDGE_EMBEDDING_API_KEY,
            embedding_model=embedding_model if embedding_model is not None else JUDGE_EMBEDDING_MODEL,
        )

        if self.local_vllm_pyengine:
            self._init_vllm_engine()
        else:
            self._init_http_client(
                api_base=api_base,
                api_key=api_key,
                embedding_api_base=embedding_api_base,
                embedding_api_key=embedding_api_key,
                embedding_model=embedding_model,
            )

        # ── 缓存 ──
        self.cache_path = Path(cache_path if cache_path is not None else JUDGE_CACHE_PATH)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _init_vllm_engine(self):
        """初始化 vLLM Python engine。

        分两种情况:
        1. 远程模式: 环境变量 AIEDU_JUDGE_VLLM_ENGINE_PORT 已设置 →
           使用 VLLMEngineClient 连接 daemon（VERL 多进程训练用）
        2. 本地模式: 直接创建 VLLMEngineScorer（单进程测试用）
        """
        from .vllm_scorer import VLLMEngineClient

        # 检查是否连接到远程 daemon
        engine_port = os.environ.get("AIEDU_JUDGE_VLLM_ENGINE_PORT", "")
        if engine_port:
            logger.info("连接远程 vLLM engine daemon: port=%s", engine_port)
            self._vllm_engine_client = VLLMEngineClient(port=int(engine_port))
            self._vllm_engine_scorer = None
            return

        # 本地模式：直接创建 engine
        if not self.local_vllm_model_path:
            raise ValueError(
                "vLLM Python engine 模式需要设置 local_vllm_model_path。\n"
                "  代码: LLMSectionJudge(local_vllm_pyengine=True, local_vllm_model_path='resources/model/...')\n"
                "  环境变量: export AIEDU_JUDGE_LOCAL_VLLM_MODEL_PATH=resources/model/..."
            )
        Cls = _get_vllm_scorer_class()
        self._vllm_engine_scorer = Cls(
            model_path=self.local_vllm_model_path,
            lora_path=self.local_vllm_lora_path,
            max_model_len=self.vllm_max_model_len,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            knowledge_base=self.knowledge_base,
            knowledge_top_k=self.knowledge_top_k,
        )
        logger.info(
            "LLMSectionJudge 使用 vLLM Python engine (logprob_token_ids): model=%s",
            self.local_vllm_model_path,
        )

    def _init_http_client(self, api_base, api_key, embedding_api_base, embedding_api_key, embedding_model):
        """初始化 HTTP API 客户端。"""
        resolved_api_base = str(api_base if api_base is not None else JUDGE_API_BASE)
        resolved_api_key = str(api_key if api_key is not None else JUDGE_API_KEY)

        self._client = OpenAICompatibleClient(
            base_url=resolved_api_base,
            api_key=resolved_api_key,
            timeout=JUDGE_REQUEST_TIMEOUT,
            local_mode=self.local_mode,
        )
        logger.info(
            "LLMSectionJudge 使用 HTTP API: base=%s, model=%s, local=%s",
            resolved_api_base, self.model, self.local_mode,
        )

    # ── 缓存 ──────────────────────────────────────────────────

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = str(obj.get("key", "")).strip()
                    value = obj.get("value")
                    if key and isinstance(value, dict):
                        self._cache[key] = value
                except json.JSONDecodeError:
                    continue

    def _save_cache_item(self, key: str, value: Dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[key] = value
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def _build_cache_key(self, *, question_type: str, prompt_text: str, candidate_text: str) -> str:
        knowledge_signature = ""
        if self.knowledge_base is not None:
            try:
                knowledge_signature = self.knowledge_base.cache_signature()
            except Exception:
                knowledge_signature = ""
        payload = {
            "prompt_version": _SCORING_PROMPT_VERSION,
            "model": self.model,
            "score_top_logprobs": self.score_top_logprobs,
            "question_type": question_type,
            "prompt_text": prompt_text,
            "candidate_text": candidate_text,
            "knowledge_signature": knowledge_signature,
            "pdf_recall_top_k": JUDGE_PDF_RECALL_TOP_K,
            "max_pdf_advice": JUDGE_MAX_PDF_ADVICE,
            "xlsx_recall_top_k": JUDGE_XLSX_RECALL_TOP_K,
            "max_outline_items": JUDGE_MAX_OUTLINE_ITEMS,
            "rag_max_tokens": JUDGE_RAG_MAX_TOKENS,
            "scoring_method": "digit_distribution_normalized_mean",
            "engine": "vllm" if self.local_vllm_pyengine else "http",
            "local_vllm_model": _path_stamp(self.local_vllm_model_path) if self.local_vllm_pyengine else "",
            "local_vllm_lora": _path_stamp(self.local_vllm_lora_path) if self.local_vllm_pyengine else "",
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ── 评分 ──────────────────────────────────────────────────

    def score_section(
        self,
        *,
        question_type: str,
        prompt_text: str,
        candidate_text: str,
    ) -> Dict[str, Any]:
        cache_key = self._build_cache_key(
            question_type=question_type,
            prompt_text=prompt_text,
            candidate_text=candidate_text,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        # ── 执行评分 ──
        if self._vllm_engine_client is not None:
            result = self._vllm_engine_client.score_section(
                question_type=question_type,
                prompt_text=prompt_text,
                candidate_text=candidate_text,
            )
        elif self._vllm_engine_scorer is not None:
            result = self._vllm_engine_scorer.score_section(
                question_type=question_type,
                prompt_text=prompt_text,
                candidate_text=candidate_text,
            )
        else:
            result = self._score_via_http_api(
                question_type=question_type,
                prompt_text=prompt_text,
                candidate_text=candidate_text,
            )

        self._save_cache_item(cache_key, result)
        return dict(result)

    def _score_via_http_api(
        self,
        *,
        question_type: str,
        prompt_text: str,
        candidate_text: str,
    ) -> Dict[str, Any]:
        """通过 HTTP API 评分（含 logprobs 质量重试）。"""
        pdf_advice_evidence: list = []
        outline_reference: list = []
        # rag_status：三态，与 vllm_scorer 保持一致
        #   ok       = 跑完了（n=0 也算 ok，意味着没召回到东西）
        #   disabled = knowledge_base 缺组件
        #   error    = 检索抛异常
        rag_status = "ok"
        rag_detail = ""
        # 持有完整 context 引用以便落 evidence.log 时拿漏斗 6 个计数。
        # disabled / error 路径下保持 None。
        rag_context = None
        if self.knowledge_base is None:
            rag_status = "disabled"
            rag_detail = "knowledge_base 实例未创建（PDF/XLSX index 都没加载到）"
        elif not self.knowledge_base.is_ready():
            rag_status = "disabled"
            rag_detail = "; ".join(self.knowledge_base.disable_reasons) or (
                "knowledge_base.is_ready() == False（embedding client 或 index 缺失）"
            )
        else:
            try:
                context = self.knowledge_base.build_judge_context(
                    question_type=question_type,
                    prompt_text=prompt_text,
                    candidate_text=candidate_text,
                    pdf_top_k=max(self.knowledge_top_k, JUDGE_PDF_RECALL_TOP_K),
                    max_pdf_advice=JUDGE_MAX_PDF_ADVICE,
                    xlsx_top_k=JUDGE_XLSX_RECALL_TOP_K,
                    max_outline_items=JUDGE_MAX_OUTLINE_ITEMS,
                )
                pdf_advice_evidence = context.pdf_advice_evidence
                outline_reference = context.outline_reference
                rag_context = context
            except Exception as exc:
                rag_status = "error"
                rag_detail = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "[RAG] retrieval error for %s: %s", question_type, rag_detail,
                )

        # 启动期一次性 WARNING（每个 judge 实例只打一次）。
        if rag_status == "disabled" and not self._rag_disable_warned:
            self._rag_disable_warned = True
            logger.warning(
                "[RAG] HTTP API 路径 RAG 不可用，evidence 将持续 n=0。原因: %s",
                rag_detail or "(未知)",
            )

        # 把 RAG 检索结果写入 evidence.log（默认关，--log-evidence 启用）。
        # 注意：之前 HTTP API 路径根本没写 evidence.log，导致用户开了 --log-evidence
        # 看到的全是 vLLM Python engine 路径的输出。这里补齐。
        # 漏斗格式（2026-05 加）：
        #   pdf: recall=N → rerank_raw=M → final=K
        #   xlsx: recall=N → candidates=C → final=K
        # 一眼能区分"召回 0 / LLM 全否 / 元数据过滤吃光"三种空 evidence 的来源。
        try:
            from ..log_setup import log_evidence as _log_evidence
            cand_head = candidate_text.replace("\n", " ")[:120]
            advice_lines = "\n".join(f"    - {a}" for a in pdf_advice_evidence) or "    (none)"
            outline_lines = "\n".join(f"    - {o}" for o in outline_reference) or "    (none)"
            detail_line = f"  detail: {rag_detail}\n" if rag_detail else ""
            if rag_context is not None:
                cache_tag = " from_cache=1" if rag_context.from_cache else ""
                funnel_line = (
                    f"  pdf: recall={rag_context.pdf_recall} → "
                    f"rerank_raw={rag_context.pdf_advice_raw} → "
                    f"final={rag_context.pdf_advice_final}\n"
                    f"  xlsx: recall={rag_context.xlsx_recall} → "
                    f"candidates={rag_context.xlsx_candidates} → "
                    f"final={rag_context.xlsx_outline_final}{cache_tag}\n"
                )
            else:
                # disabled / error 路径没跑 build_judge_context，所有计数无意义。
                funnel_line = ""
            _log_evidence(
                f"{question_type} status={rag_status} | candidate_head={cand_head!r}\n"
                f"{detail_line}"
                f"{funnel_line}"
                f"  pdf_advice (n={len(pdf_advice_evidence)}):\n{advice_lines}\n"
                f"  outline (n={len(outline_reference)}):\n{outline_lines}"
            )
        except Exception:
            pass

        prompt = build_section_distribution_prompt(
            question_type=question_type,
            prompt_text=prompt_text,
            candidate_text=candidate_text,
            pdf_advice_evidence=pdf_advice_evidence,
            outline_reference=outline_reference,
        )

        if not self._client.api_key and not self.allow_empty_api_key:
            raise ValueError("AIEDU_JUDGE_API_KEY is empty; judge reward cannot run")

        # 调用 API（错误由 OpenAICompatibleClient 内部处理）
        response_payload = self._client.chat_complete_with_logprobs(
            model=self.model,
            system_prompt="Reply with exactly one digit from 0 to 9.",
            user_prompt=prompt,
            temperature=self.temperature,
            max_tokens=1,
            top_logprobs=self.score_top_logprobs,
        )

        # 解析 logprobs（含低质量重试）
        last_error: Optional[Exception] = None
        parsed = None
        for attempt in range(1 + len(_LOGPROBS_RETRY_BACKOFF)):
            try:
                digit_logprobs = _extract_digit_logprobs(response_payload)
                parsed = _build_distribution_stats(digit_logprobs)
                if parsed["raw_digit_mass"] < _MIN_DIGIT_MASS:
                    raise ValueError(
                        f"digit mass too low: {parsed['raw_digit_mass']:.6f} < {_MIN_DIGIT_MASS:.6f}"
                    )
                break
            except Exception as exc:
                last_error = exc
                if attempt < len(_LOGPROBS_RETRY_BACKOFF):
                    import time
                    wait = _LOGPROBS_RETRY_BACKOFF[attempt]
                    logger.warning(
                        "logprobs quality retry %d/%d: wait=%ds",
                        attempt + 1, len(_LOGPROBS_RETRY_BACKOFF), wait,
                    )
                    time.sleep(wait)
                    continue
                raise

        missing = [d for d in SCORE_DIGITS if d not in digit_logprobs]
        found = len(SCORE_DIGITS) - len(missing)
        parsed.update({
            "score": max(0.0, min(1.0, float(parsed["normalized_mean_score"]))),
            "question_type": question_type,
            "scoring_method": f"http_api_logprobs_top{self.score_top_logprobs}",
            "found_digits": found,
            "total_digits": len(SCORE_DIGITS),
            "summary": f"HTTP API: mean={parsed['mean_score']:.2f} ({found}/{len(SCORE_DIGITS)} digits)"
                        + (f", missing: {missing}" if missing else ""),
            "model": self.model,
        })
        return dict(parsed)


# ── 默认实例 ─────────────────────────────────────────────────


_DEFAULT_JUDGE: Optional[LLMSectionJudge] = None


def get_default_judge() -> LLMSectionJudge:
    global _DEFAULT_JUDGE
    if _DEFAULT_JUDGE is None:
        _DEFAULT_JUDGE = LLMSectionJudge()
    return _DEFAULT_JUDGE


__all__ = ["LLMSectionJudge", "get_default_judge"]

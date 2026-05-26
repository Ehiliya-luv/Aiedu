# -*- coding: utf-8 -*-
"""OpenAI-compatible chat + embedding client with error-aware retry strategy.

错误分类与重试策略（不再有任何 fallback 到 mean/floor 的逻辑）:
  - Auth/Billing (401/402/403) → 立即抛错，不重试
  - Rate Limit (429) → 指数退避：1s→2s→4s→8s→16s→32s→64s（封顶）
  - 暂时性错误 (connection/timeout/500) → 短退避重试
  - local_mode=True → 所有错误立即抛错，方便 debug
  - 全部重试耗尽后 → 抛错（由调用方决定如何处理）
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional

try:
    import openai
except ImportError:  # 允许 --help/--dry-run 在未安装 OpenAI SDK 的环境中运行
    openai = None  # type: ignore[assignment]

from utils.text import strip_think_content

from .settings import (
    JUDGE_ALLOW_EMPTY_API_KEY,
    JUDGE_LOCAL_MODE,
    JUDGE_RATE_LIMIT_FIRST_BACKOFF,
    JUDGE_RATE_LIMIT_MAX_BACKOFF,
    JUDGE_RATE_LIMIT_MAX_RETRIES,
    JUDGE_REQUEST_TIMEOUT,
    JUDGE_TRANSIENT_RETRIES,
)

logger = logging.getLogger(__name__)


# ── 自定义异常（带分类标记） ──────────────────────────────────────────


class JudgeAuthError(RuntimeError):
    """认证/余额/权限错误（不应重试）。"""
    pass


class JudgeRateLimitError(RuntimeError):
    """速率限制耗尽后仍失败。"""
    pass


class JudgeAPIError(RuntimeError):
    """其他不可恢复的 API 错误。"""
    pass


# ── 错误分类辅助函数 ─────────────────────────────────────────────────


_ERROR_CATEGORIES: Dict[str, type] = {}


def _classify_error(exc: Exception) -> type:
    if openai is None:
        return JudgeAPIError

    error_code = _try_extract_error_code(exc)
    status = _try_extract_status(exc)
    cache_key = f"{type(exc).__name__}:{status}:{error_code}"
    if cache_key in _ERROR_CATEGORIES:
        return _ERROR_CATEGORIES[cache_key]

    if isinstance(exc, openai.AuthenticationError):
        result = JudgeAuthError
    elif isinstance(exc, openai.PermissionDeniedError):
        result = JudgeAuthError
    elif isinstance(exc, openai.BadRequestError):
        # 402 Payment Required 有时包装在 BadRequest 中
        if status == 402:
            result = JudgeAuthError
        else:
            result = JudgeAPIError
    elif isinstance(exc, openai.RateLimitError):
        if error_code in {"insufficient_quota", "billing_not_active", "quota_exceeded"}:
            result = JudgeAuthError
        else:
            result = JudgeRateLimitError
    elif isinstance(exc, openai.APIConnectionError):
        result = JudgeRateLimitError  # 网络抖动按退避重试
    elif isinstance(exc, openai.APITimeoutError):
        result = JudgeRateLimitError
    elif isinstance(exc, openai.InternalServerError):
        result = JudgeRateLimitError
    elif isinstance(exc, openai.APIStatusError):
        if status == 429 and error_code in {"insufficient_quota", "billing_not_active", "quota_exceeded"}:
            result = JudgeAuthError
        elif status in (429, 502, 503):
            result = JudgeRateLimitError
        elif status in (401, 402, 403):
            result = JudgeAuthError
        else:
            result = JudgeAPIError
    else:
        result = JudgeAPIError

    _ERROR_CATEGORIES[cache_key] = result
    return result


def _try_extract_status(exc: Exception) -> Optional[int]:
    try:
        return exc.status_code  # type: ignore[union-attr]
    except AttributeError:
        pass
    try:
        return int(getattr(exc, "code", 0))  # type: ignore[arg-type]
    except (ValueError, TypeError, AttributeError):
        pass
    return None


def _try_extract_error_code(exc: Exception) -> Optional[str]:
    try:
        body = getattr(exc, "body", None) or {}
        if isinstance(body, dict):
            return body.get("code")
    except AttributeError:
        pass
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    return _classify_error(exc) == JudgeRateLimitError


def _is_auth_error(exc: Exception) -> bool:
    return _classify_error(exc) == JudgeAuthError


def _build_error_message(category: type, exc: Exception, detail: str = "") -> str:
    base = {
        JudgeAuthError: "Judge API 认证/余额错误，训练终止",
        JudgeRateLimitError: "Judge API 速率限制，重试耗尽",
        JudgeAPIError: "Judge API 不可恢复错误",
    }.get(category, "Judge API 未知错误")
    parts = [f"[{category.__name__}] {base}"]
    if detail:
        parts.append(detail)
    exc_info = str(exc)[:400]
    if exc_info:
        parts.append(f"原始错误: {exc_info}")
    return " | ".join(parts)


# ── 指数退避 ────────────────────────────────────────────────────────


def _exponential_backoff(attempt: int, first_backoff: float, max_backoff: float) -> float:
    wait = first_backoff * (2 ** attempt)
    return min(wait, max_backoff)


# ── OpenAI 兼容客户端 ────────────────────────────────────────────────


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = JUDGE_REQUEST_TIMEOUT,
        local_mode: bool | None = None,
        rate_limit_first_backoff: float = JUDGE_RATE_LIMIT_FIRST_BACKOFF,
        rate_limit_max_backoff: float = JUDGE_RATE_LIMIT_MAX_BACKOFF,
        rate_limit_max_retries: int = JUDGE_RATE_LIMIT_MAX_RETRIES,
        transient_retries: int = JUDGE_TRANSIENT_RETRIES,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.timeout = int(timeout)
        self.local_mode = JUDGE_LOCAL_MODE if local_mode is None else bool(local_mode)
        self.rate_limit_first_backoff = float(rate_limit_first_backoff)
        self.rate_limit_max_backoff = float(rate_limit_max_backoff)
        self.rate_limit_max_retries = int(rate_limit_max_retries)
        self.transient_retries = int(transient_retries)
        self._client = None

    def _require_client(self):
        if not self.base_url:
            raise ValueError("judge/embedding base_url is empty")
        if not self.api_key and not JUDGE_ALLOW_EMPTY_API_KEY:
            raise ValueError(
                "judge/embedding api_key is empty。"
                "本地部署请设置 AIEDU_JUDGE_ALLOW_EMPTY_API_KEY=1 或 AIEDU_JUDGE_LOCAL_MODE=1"
            )
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise JudgeAPIError(
                    "OpenAI SDK 未安装，无法调用 Judge API。请先安装 openai，或切换到可用环境。"
                ) from exc

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "dummy-key-required-by-openai-client",  # vLLM 接受任意 key
                timeout=self.timeout,
                max_retries=0,  # 我们自己控制重试
            )
        return self._client

    # ── 错误感知重试：核心方法 ──────────────────────────────────

    def _execute_with_retry(self, api_call_fn, **resolve_kw) -> Any:
        """统一的错误感知重试执行器。

        分类策略（优先级从高到低）:
        1. local_mode=True：任何异常立即抛 JudgeAPIError
        2. Auth/Billing 错误：立即抛 JudgeAuthError
        3. Rate Limit 错误：指数退避重试 rate_limit_max_retries 次
        4. 暂时性错误：退避重试 transient_retries 次
        5. 全部耗尽后：抛 JudgeRateLimitError / JudgeAPIError
        """
        if self.local_mode:
            # 本地模式：直接调用，任何异常立即抛出
            try:
                return api_call_fn()
            except Exception as exc:
                raise JudgeAPIError(
                    _build_error_message(JudgeAPIError, exc, "local_mode=True 下所有错误立即抛出")
                ) from exc

        last_exc: Optional[Exception] = None
        rate_limit_attempts = 0
        transient_attempts = 0

        while True:
            try:
                return api_call_fn()
            except Exception as exc:
                last_exc = exc

                if _is_auth_error(exc):
                    raise JudgeAuthError(_build_error_message(JudgeAuthError, exc)) from exc

                if _is_rate_limit_error(exc):
                    rate_limit_attempts += 1
                    if rate_limit_attempts > self.rate_limit_max_retries:
                        raise JudgeRateLimitError(
                            _build_error_message(
                                JudgeRateLimitError, exc,
                                f"重试 {self.rate_limit_max_retries} 次后仍然速率受限",
                            )
                        ) from exc
                    wait = _exponential_backoff(
                        rate_limit_attempts, self.rate_limit_first_backoff, self.rate_limit_max_backoff
                    )
                    logger.warning(
                        "Rate limit (attempt %d/%d), 等待 %.0fs",
                        rate_limit_attempts, self.rate_limit_max_retries, wait,
                    )
                    time.sleep(wait)
                    continue

                # 其他暂时性错误（connection/timeout/500）
                transient_attempts += 1
                if transient_attempts > self.transient_retries:
                    raise JudgeAPIError(
                        _build_error_message(
                            JudgeAPIError, last_exc,
                            f"暂时性错误重试 {self.transient_retries} 次后仍然失败",
                        )
                    ) from last_exc
                wait = 2 ** transient_attempts
                logger.warning(
                    "Transient error (attempt %d/%d), 等待 %.0fs: %s",
                    transient_attempts, self.transient_retries, wait,
                    str(exc)[:200],
                )
                time.sleep(wait)

    # ── 消息文本提取 ──────────────────────────────────────────

    @staticmethod
    def _extract_message_text(response) -> str:
        try:
            choice = response.choices[0]
        except Exception as exc:
            raise ValueError(f"invalid completion response: {response}") from exc

        def _choice_debug_info() -> str:
            try:
                finish_reason = getattr(choice, "finish_reason", None)
                message = getattr(choice, "message", None)
                content = getattr(message, "content", None) if message is not None else None
                reasoning_content = getattr(message, "reasoning_content", None) if message is not None else None
                return (
                    f"finish_reason={finish_reason!r}, "
                    f"content={content!r}, "
                    f"reasoning_content={reasoning_content!r}"
                )
            except Exception:
                return repr(choice)

        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                text = strip_think_content(content).strip()
                if text:
                    return text
                reasoning_content = getattr(message, "reasoning_content", None)
                if isinstance(reasoning_content, str):
                    reasoning_text = strip_think_content(reasoning_content).strip()
                    if reasoning_text:
                        return reasoning_text
                raise ValueError(f"empty assistant content: {_choice_debug_info()}")
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            parts.append(str(text))
                if parts:
                    return strip_think_content("\n".join(parts)).strip()
                reasoning_content = getattr(message, "reasoning_content", None)
                if isinstance(reasoning_content, str):
                    reasoning_text = strip_think_content(reasoning_content).strip()
                    if reasoning_text:
                        return reasoning_text
                raise ValueError(f"empty assistant content: {_choice_debug_info()}")

        text = getattr(choice, "text", None)
        if isinstance(text, str):
            stripped = strip_think_content(text).strip()
            if stripped:
                return stripped
            raise ValueError(f"empty assistant content: {_choice_debug_info()}")

        raise ValueError(f"cannot extract text from completion response: {_choice_debug_info()}")

    # ── Chat Completion（纯文本） ──────────────────────────────

    def chat_complete_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: Optional[float] = None,
        response_format: Optional[dict] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        client = self._require_client()

        def _call():
            payload: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "response_format": response_format,
            }
            if top_p is not None:
                payload["top_p"] = float(top_p)
            # extra_body：透传给 OpenAI SDK 的非标准字段（如 vLLM 的
            # chat_template_kwargs.enable_thinking）。仅在显式传入时附加，
            # 避免污染默认 API 调用语义。
            if extra_body is not None:
                payload["extra_body"] = extra_body
            return client.chat.completions.create(
                **payload,
            )

        response = self._execute_with_retry(_call)
        return self._extract_message_text(response)

    # ── Logprobs 提取 ─────────────────────────────────────────

    @staticmethod
    def _extract_logprob_entry(entry: Any) -> Dict[str, Any]:
        if entry is None:
            return {}
        token = getattr(entry, "token", None)
        logprob = getattr(entry, "logprob", None)
        top = getattr(entry, "top_logprobs", None)
        normalized_top: List[Dict[str, Any]] = []
        if isinstance(top, list):
            for candidate in top:
                candidate_token = getattr(candidate, "token", None)
                candidate_logprob = getattr(candidate, "logprob", None)
                if candidate_token is None or candidate_logprob is None:
                    continue
                normalized_top.append(
                    {
                        "token": str(candidate_token),
                        "logprob": float(candidate_logprob),
                    }
                )
        result: Dict[str, Any] = {}
        if token is not None:
            result["token"] = str(token)
        if logprob is not None:
            result["logprob"] = float(logprob)
        if normalized_top:
            result["top_logprobs"] = normalized_top
        return result

    @classmethod
    def _extract_logprobs_payload(cls, response: Any) -> Dict[str, Any]:
        try:
            choice = response.choices[0]
        except Exception as exc:
            raise ValueError(f"invalid logprobs response: {response}") from exc

        message = getattr(choice, "message", None)
        message_content = getattr(message, "content", None) if message is not None else None
        logprobs = getattr(choice, "logprobs", None)
        content = getattr(logprobs, "content", None) if logprobs is not None else None

        payload: Dict[str, Any] = {
            "message": {"content": message_content if isinstance(message_content, str) else ""},
            "logprobs": {"content": []},
        }
        if isinstance(content, list):
            normalized_content: List[Dict[str, Any]] = []
            for entry in content:
                normalized = cls._extract_logprob_entry(entry)
                if normalized:
                    normalized_content.append(normalized)
            payload["logprobs"]["content"] = normalized_content
        return {"choices": [payload]}

    # ── Chat Completion（带 Logprobs） ─────────────────────────

    def chat_complete_with_logprobs(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        top_logprobs: int,
    ) -> Dict[str, Any]:
        client = self._require_client()

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                logprobs=True,
                top_logprobs=int(top_logprobs),
            )

        response = self._execute_with_retry(_call)
        return self._extract_logprobs_payload(response)

    # ── Embedding ─────────────────────────────────────────────

    def embed_texts(self, *, model: str, texts: Iterable[str]) -> List[List[float]]:
        client = self._require_client()
        items = [str(text or "") for text in texts]
        if not items:
            return []

        def _call():
            return client.embeddings.create(model=model, input=items)

        response = self._execute_with_retry(_call)
        return [list(item.embedding) for item in response.data]

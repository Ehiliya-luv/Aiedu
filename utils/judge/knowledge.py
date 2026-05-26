# -*- coding: utf-8 -*-
"""Offline knowledge loading and dense retrieval with numpy cosine similarity."""

from __future__ import annotations

import json
import logging
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .api import OpenAICompatibleClient
from .settings import (
    JUDGE_API_BASE,
    JUDGE_API_KEY,
    JUDGE_ALLOW_EMPTY_API_KEY,
    JUDGE_EMBEDDING_API_BASE,
    JUDGE_EMBEDDING_API_KEY,
    JUDGE_EMBEDDING_MODEL,
    JUDGE_KNOWLEDGE_CONTEXT_CACHE,
    JUDGE_KNOWLEDGE_PDF_CHUNKS,
    JUDGE_KNOWLEDGE_PDF_EMBEDDINGS,
    JUDGE_KNOWLEDGE_XLSX_CHUNKS,
    JUDGE_KNOWLEDGE_XLSX_EMBEDDINGS,
    JUDGE_MAX_OUTLINE_ITEMS,
    JUDGE_MAX_PDF_ADVICE,
    JUDGE_MODEL,
    JUDGE_PDF_RECALL_TOP_K,
    JUDGE_PDF_RERANK_BATCH_SIZE,
    JUDGE_RAG_MAX_TOKENS,
    JUDGE_TEMPERATURE,
    JUDGE_XLSX_RECALL_TOP_K,
)

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeChunk:
    id: str
    source: str
    text: str
    metadata: Dict[str, object]


@dataclass
class RetrievedChunk:
    chunk: KnowledgeChunk
    score: float


@dataclass
class JudgeKnowledgeContext:
    pdf_advice_evidence: List[str]
    outline_reference: List[str]
    # 漏斗诊断字段（2026-05 加，仅用于定位 evidence 为空的断点；不影响任何 RAG 逻辑）。
    # 6 个数字按"召回 → LLM rerank → 最终列表"顺序，写到 evidence.log 一眼能看出
    # 是 (a) 召回 0、(b) 召回有但 rerank 全否、还是 (c) 候选过滤吃光了。
    pdf_recall: int = 0           # retrieve_pdf 实际命中条数（embedding similarity top-k 之后）
    pdf_advice_raw: int = 0       # 所有 batch LLM rerank 累计返回的 advice 条数（去重前）
    pdf_advice_final: int = 0     # 最终 pdf_advice_evidence 长度（应等于 len(pdf_advice_evidence)）
    xlsx_recall: int = 0          # retrieve_xlsx 实际命中条数
    xlsx_candidates: int = 0      # _meaningful_outline_path 过滤后剩下的候选数
    xlsx_outline_final: int = 0   # 最终 outline_reference 长度（应等于 len(outline_reference)）
    # cache 命中时所有 stage 计数都没意义（实际跳过了召回）。evidence.log 拿到这个标记
    # 后会打印 "from_cache=1"，避免被误读为"召回 0 条"。
    from_cache: bool = False


_CONTEXT_VERSION = "judge_context_v2"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    matrix = matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def _normalize_vector(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.size == 0:
        return arr
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def _normalize_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _shorten(text: str, limit: int) -> str:
    text = _normalize_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _parse_json_object(raw: str) -> Dict[str, Any]:
    text = _normalize_text(raw)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _dedupe_strings(items: Iterable[str], *, limit: int) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        text = _normalize_text(item)
        if not text:
            continue
        key = re.sub(r"\s+", "", text).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _meaningful_outline_path(chunk: KnowledgeChunk) -> str:
    meta = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    path = _normalize_text(meta.get("outline_path") or chunk.text)
    parts = [part.strip() for part in path.split(">")]
    parts = [part for part in parts if part and part.lower() != "none" and not re.fullmatch(r"\d{3,}", part)]
    if not parts:
        return ""
    if len(parts) <= 2 and all(part in {"内科", "外科", "专业理论知识", "相关专业知识"} for part in parts):
        return ""
    return " > ".join(parts)


class DenseKnowledgeIndex:
    def __init__(self, chunks: Sequence[KnowledgeChunk], embeddings: np.ndarray) -> None:
        self.chunks = list(chunks)
        self.embeddings = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
        if len(self.chunks) != len(self.embeddings):
            raise ValueError("knowledge chunks and embeddings size mismatch")

    def search(self, query_vector, *, top_k: int, filter_fn=None) -> List[RetrievedChunk]:
        if top_k <= 0 or not self.chunks or self.embeddings.size == 0:
            return []
        query = _normalize_vector(query_vector)
        if query.size == 0:
            return []
        scores = np.matmul(self.embeddings, query)
        items: List[RetrievedChunk] = []
        for idx, score in enumerate(scores.tolist()):
            chunk = self.chunks[idx]
            if filter_fn is not None and not filter_fn(chunk):
                continue
            items.append(RetrievedChunk(chunk=chunk, score=float(score)))
        items.sort(key=lambda item: item.score, reverse=True)
        return items[:top_k]


class JudgeKnowledgeBase:
    def __init__(
        self,
        *,
        judge_client: Optional[OpenAICompatibleClient],
        judge_model: str,
        embedding_client: Optional[OpenAICompatibleClient],
        embedding_model: str,
        pdf_index: Optional[DenseKnowledgeIndex],
        xlsx_index: Optional[DenseKnowledgeIndex],
        context_cache_path: str,
        pdf_chunk_path: str,
        xlsx_chunk_path: str,
        disable_reasons: Optional[List[str]] = None,
    ) -> None:
        self.judge_client = judge_client
        self.judge_model = str(judge_model or "")
        self.embedding_client = embedding_client
        self.embedding_model = str(embedding_model or "")
        self.pdf_index = pdf_index
        self.xlsx_index = xlsx_index
        self.context_cache_path = Path(context_cache_path)
        self.context_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf_chunk_path = str(pdf_chunk_path or "")
        self.xlsx_chunk_path = str(xlsx_chunk_path or "")
        # disable_reasons：load_knowledge_base 收集的"为什么某个 RAG 部件没起来"
        # 列表，供下游 scorer / 启动状态报告读取，避免每条 candidate 重新探测。
        self.disable_reasons: List[str] = list(disable_reasons or [])
        self._context_cache: Dict[str, Dict[str, Any]] = {}
        self._load_context_cache()

    def is_ready(self) -> bool:
        return bool(self.embedding_client and self.embedding_model and (self.pdf_index or self.xlsx_index))

    def _can_generate_context(self) -> bool:
        return bool(self.is_ready() and self.judge_client and self.judge_model)

    def is_fully_operational(self) -> bool:
        """全部 4 个 RAG 部件都在位（PDF + XLSX index + embedding + judge LLM）。

        --require-rag 启动检查使用此方法：任意一项缺失即视为"不完整 RAG"。
        与 is_ready() 区别：is_ready() 只要 embedding 和任一 index 就 OK，
        但缺 judge LLM 时无法做 PDF advice rerank / XLSX outline 评判，evidence
        会全空——所以从训练角度说"is_ready 但不 fully_operational"等于半残。
        """
        return bool(
            self.embedding_client
            and self.embedding_model
            and self.judge_client
            and self.judge_model
            and self.pdf_index
            and self.xlsx_index
        )

    def status_summary(self) -> Dict[str, Any]:
        """返回 RAG 各组件状态字典；启动报告 / evidence.log 调用。"""
        return {
            "pdf_index": bool(self.pdf_index),
            "xlsx_index": bool(self.xlsx_index),
            "embedding_client": self.embedding_client is not None,
            "judge_client": self.judge_client is not None,
            "is_ready": self.is_ready(),
            "can_generate_context": self._can_generate_context(),
            "is_fully_operational": self.is_fully_operational(),
            "disable_reasons": list(self.disable_reasons),
        }

    def _load_context_cache(self) -> None:
        if not self.context_cache_path.exists():
            return
        with self.context_cache_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(obj.get("key", "")).strip()
                value = obj.get("value")
                if key and isinstance(value, dict):
                    self._context_cache[key] = value

    def _save_context_cache_item(self, key: str, value: Dict[str, Any]) -> None:
        self._context_cache[key] = value
        with self.context_cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def _file_stamp(self, path: str) -> str:
        if not path:
            return ""
        p = Path(path)
        if not p.exists():
            return "missing"
        return f"{p.resolve()}:{p.stat().st_mtime_ns}:{p.stat().st_size}"

    def _build_context_cache_key(
        self,
        *,
        question_type: str,
        candidate_text: str,
        pdf_top_k: int,
        max_pdf_advice: int,
        xlsx_top_k: int,
        max_outline_items: int,
    ) -> str:
        payload = {
            "version": _CONTEXT_VERSION,
            "judge_model": self.judge_model,
            "embedding_model": self.embedding_model,
            "question_type": question_type,
            "candidate_text": candidate_text,
            "pdf_top_k": pdf_top_k,
            "max_pdf_advice": max_pdf_advice,
            "xlsx_top_k": xlsx_top_k,
            "max_outline_items": max_outline_items,
            "rag_max_tokens": JUDGE_RAG_MAX_TOKENS,
            "pdf_chunks": self._file_stamp(self.pdf_chunk_path),
            "xlsx_chunks": self._file_stamp(self.xlsx_chunk_path),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def cache_signature(self) -> str:
        payload = {
            "version": _CONTEXT_VERSION,
            "judge_model": self.judge_model,
            "embedding_model": self.embedding_model,
            "rag_max_tokens": JUDGE_RAG_MAX_TOKENS,
            "pdf_chunks": self._file_stamp(self.pdf_chunk_path),
            "xlsx_chunks": self._file_stamp(self.xlsx_chunk_path),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _embed_query(self, text: str) -> np.ndarray:
        if not self.embedding_client or not self.embedding_model:
            raise ValueError("embedding client/model not configured")
        vectors = self.embedding_client.embed_texts(model=self.embedding_model, texts=[text])
        return _normalize_vector(vectors[0] if vectors else [])

    def retrieve_pdf(self, query_text: str, question_type: str, top_k: int) -> List[RetrievedChunk]:
        if not self.pdf_index or not query_text.strip():
            return []
        query_vec = self._embed_query(query_text)

        def _filter(chunk: KnowledgeChunk) -> bool:
            qtypes = chunk.metadata.get("question_type") if isinstance(chunk.metadata, dict) else None
            if not isinstance(qtypes, list) or not qtypes:
                return True
            normalized = {str(item).upper() for item in qtypes}
            return question_type.upper() in normalized or "COMMON" in normalized

        return self.pdf_index.search(query_vec, top_k=top_k, filter_fn=_filter)

    def retrieve_xlsx(self, query_text: str, top_k: int) -> List[RetrievedChunk]:
        if not self.xlsx_index or not query_text.strip():
            return []
        query_vec = self._embed_query(query_text)
        return self.xlsx_index.search(query_vec, top_k=top_k)

    def build_judge_context(
        self,
        *,
        question_type: str,
        prompt_text: str,
        candidate_text: str,
        pdf_top_k: int = JUDGE_PDF_RECALL_TOP_K,
        max_pdf_advice: int = JUDGE_MAX_PDF_ADVICE,
        xlsx_top_k: int = JUDGE_XLSX_RECALL_TOP_K,
        max_outline_items: int = JUDGE_MAX_OUTLINE_ITEMS,
    ) -> JudgeKnowledgeContext:
        question_type = str(question_type or "").strip().upper()
        candidate_text = _normalize_text(candidate_text)
        if not self._can_generate_context() or not candidate_text:
            return JudgeKnowledgeContext(pdf_advice_evidence=[], outline_reference=[])

        cache_key = self._build_context_cache_key(
            question_type=question_type,
            candidate_text=candidate_text,
            pdf_top_k=int(pdf_top_k),
            max_pdf_advice=int(max_pdf_advice),
            xlsx_top_k=int(xlsx_top_k),
            max_outline_items=int(max_outline_items),
        )
        cached = self._context_cache.get(cache_key)
        if isinstance(cached, dict):
            return JudgeKnowledgeContext(
                pdf_advice_evidence=[
                    str(item) for item in cached.get("pdf_advice_evidence", []) if str(item).strip()
                ],
                outline_reference=[
                    str(item) for item in cached.get("outline_reference", []) if str(item).strip()
                ],
                from_cache=True,
            )

        query_text = "\n".join(part for part in [f"{question_type}型题", candidate_text[:1800]] if part.strip())
        pdf_advice: List[str] = []
        outline_reference: List[str] = []
        # 6 个 stage 计数：先全部初始化为 0，下面按实际跑过的阶段填进去。
        # 即使某段抛异常被 except 兜住，对应计数也会保持 0 而不是脏值——
        # 落到 evidence.log 时一眼能区分"没跑"和"跑了但没结果"。
        pdf_recall_n = 0
        pdf_advice_raw = 0
        xlsx_recall_n = 0
        xlsx_candidates_n = 0
        try:
            pdf_items = self.retrieve_pdf(query_text, question_type=question_type, top_k=int(pdf_top_k))
            pdf_recall_n = len(pdf_items)
            pdf_advice, pdf_advice_raw = self._build_pdf_advice(
                question_type=question_type,
                candidate_text=candidate_text,
                retrieved_chunks=pdf_items,
                batch_size=JUDGE_PDF_RERANK_BATCH_SIZE,
                max_advice=int(max_pdf_advice),
            )
        except Exception as exc:
            logger.warning("PDF advice extraction skipped for %s: %s", question_type, exc)
        try:
            xlsx_items = self.retrieve_xlsx(query_text, top_k=int(xlsx_top_k))
            xlsx_recall_n = len(xlsx_items)
            outline_reference, xlsx_candidates_n = self._build_outline_reference(
                question_type=question_type,
                candidate_text=candidate_text,
                retrieved_chunks=xlsx_items,
                max_items=int(max_outline_items),
            )
        except Exception as exc:
            logger.warning("xlsx outline matching skipped for %s: %s", question_type, exc)

        value = {
            "pdf_advice_evidence": pdf_advice,
            "outline_reference": outline_reference,
        }
        self._save_context_cache_item(cache_key, value)
        return JudgeKnowledgeContext(
            pdf_advice_evidence=pdf_advice,
            outline_reference=outline_reference,
            pdf_recall=pdf_recall_n,
            pdf_advice_raw=pdf_advice_raw,
            pdf_advice_final=len(pdf_advice),
            xlsx_recall=xlsx_recall_n,
            xlsx_candidates=xlsx_candidates_n,
            xlsx_outline_final=len(outline_reference),
            from_cache=False,
        )

    def _chat_json(self, *, system_prompt: str, user_prompt: str, max_tokens: int) -> Dict[str, Any]:
        if not self.judge_client:
            return {}
        raw = self.judge_client.chat_complete_text(
            model=self.judge_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        parsed = _parse_json_object(raw)
        # 沉默吞 0 路径 #1：LLM 返回了内容但不是合法 JSON（或 response_format 不被
        # 服务端支持）。原代码这里直接 return {} 然后 caller `obj.get("advice")` 拿到
        # None，evidence 就空到底了。把 raw head 显式打出来定位是哪种失败。
        if not parsed and raw:
            logger.warning(
                "[RAG] judge JSON parse 失败 (raw_len=%d head=%r) — "
                "可能 LLM 没遵守 JSON schema 或 response_format 不被支持",
                len(raw), raw[:200].replace("\n", "\\n"),
            )
        return parsed

    def _build_pdf_advice(
        self,
        *,
        question_type: str,
        candidate_text: str,
        retrieved_chunks: List[RetrievedChunk],
        batch_size: int,
        max_advice: int,
    ) -> Tuple[List[str], int]:
        """返回 (advice_final, advice_raw_count)。

        - advice_final：去重 + 合并后最终送入 judge prompt 的 advice 列表。
        - advice_raw_count：所有 batch LLM rerank 累计返回的条数（去重前）。
          这个原始计数用来诊断"recall>0 但 advice=0"是 LLM 全否还是只是去重塌缩。
        """
        if not retrieved_chunks or max_advice <= 0:
            return [], 0
        batch_size = max(1, int(batch_size or 8))
        advice_items: List[str] = []
        advice_raw_count = 0
        for start in range(0, len(retrieved_chunks), batch_size):
            batch = retrieved_chunks[start:start + batch_size]
            batch_rows = []
            for index, item in enumerate(batch, start=1):
                meta = item.chunk.metadata if isinstance(item.chunk.metadata, dict) else {}
                page = meta.get("page", "")
                role = meta.get("content_role", "")
                role_text = ", ".join(role) if isinstance(role, list) else str(role or "")
                batch_rows.append(
                    f"[{index}] score={item.score:.4f} page={page} role={role_text}\n{_shorten(item.chunk.text, 700)}"
                )
            user_prompt = f"""请从以下PDF chunk中抽取对当前医学试题“命题质量评价”真正有用的 Advice Evidence。

要求：
1. 评价对象是当前 {question_type} 型题，而不是医学知识问答。
2. 必须区分：命题质量相似、仅医学内容相似、仅题型相同、不相关。
3. 只保留能迁移到当前题评分的命题原则、正面样例经验、反面样例经验或修改建议。
4. 丢弃纯题型定义、纯医学知识、考试背景介绍、无法迁移到当前题的内容。
5. 本批最多输出2条 advice；没有可用证据时输出空数组。
6. advice 必须是中性的评分依据，可正面也可负面，不要直接给当前题打分。
7. 严格输出JSON：{{"advice": ["..."]}}

【当前题】
{_shorten(candidate_text, 2600)}

【PDF chunks】
{chr(10).join(batch_rows)}
"""
            try:
                obj = self._chat_json(
                    system_prompt="你是医学考试命题资料筛选员。只输出严格JSON。",
                    user_prompt=user_prompt,
                    max_tokens=JUDGE_RAG_MAX_TOKENS,
                )
            except Exception as exc:
                logger.warning("PDF advice batch failed: %s", exc)
                continue
            items = obj.get("advice")
            # 沉默吞 0 路径 #3：obj 是合法 JSON 但 advice 字段不是 list（比如 LLM
            # 返回 {"advice": "无相关内容"}）。原来直接 if isinstance 跳过，evidence
            # 静默 0。打条 debug log 让"哪个 batch 返回了什么"可追溯。
            if isinstance(items, list):
                kept = [str(item).strip() for item in items if str(item).strip()]
                advice_raw_count += len(kept)
                advice_items.extend(kept)
                logger.debug(
                    "[RAG] pdf advice batch[%d-%d] in=%d out=%d (raw_total=%d)",
                    start, start + len(batch), len(batch), len(kept), advice_raw_count,
                )
            else:
                logger.debug(
                    "[RAG] pdf advice batch[%d-%d] in=%d but obj.advice is %s (obj keys=%s)",
                    start, start + len(batch), len(batch), type(items).__name__, list(obj.keys()),
                )
        advice_items = _dedupe_strings(advice_items, limit=max(12, max_advice))
        if len(advice_items) <= max_advice:
            return advice_items[:max_advice], advice_raw_count
        merged = self._merge_pdf_advice(
            question_type=question_type,
            candidate_text=candidate_text,
            advice_items=advice_items,
            max_advice=max_advice,
        )
        return merged, advice_raw_count

    def _merge_pdf_advice(
        self,
        *,
        question_type: str,
        candidate_text: str,
        advice_items: List[str],
        max_advice: int,
    ) -> List[str]:
        rows = "\n".join(f"- {item}" for item in advice_items if item.strip())
        user_prompt = f"""请合并去重以下 Advice Evidence，保留最能评价当前 {question_type} 型题命题质量的最多 {max_advice} 条。

要求：
1. 合并语义重复或高度相近的建议。
2. 不要新增没有依据的建议。
3. 输出的每条 advice 都应能直接帮助最终评分模型判断当前题好坏。
4. 严格输出JSON：{{"advice": ["..."]}}

【当前题】
{_shorten(candidate_text, 2600)}

【待合并建议】
{rows}
"""
        try:
            obj = self._chat_json(
                system_prompt="你是医学考试命题建议合并员。只输出严格JSON。",
                user_prompt=user_prompt,
                max_tokens=JUDGE_RAG_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("PDF advice merge failed: %s", exc)
            return advice_items[:max_advice]
        items = obj.get("advice")
        if not isinstance(items, list):
            return advice_items[:max_advice]
        merged = _dedupe_strings((str(item) for item in items), limit=max_advice)
        return merged or advice_items[:max_advice]

    def _build_outline_reference(
        self,
        *,
        question_type: str,
        candidate_text: str,
        retrieved_chunks: List[RetrievedChunk],
        max_items: int,
    ) -> Tuple[List[str], int]:
        """返回 (outline_final, candidates_count)。

        - outline_final：经 LLM 评判挑出最相关的 outline_reference 列表。
        - candidates_count：_meaningful_outline_path 过滤之后剩下的候选数。
          诊断"xlsx_recall>0 但 outline=0"是被 _meaningful_outline_path
          黑名单吃光还是 LLM 评判全否的关键信号。
        """
        if not retrieved_chunks or max_items <= 0:
            return [], 0
        # 历史版本里有 `if len(candidates) >= max(10, max_items*4): break`——
        # 当 max_items=3 时只让前 12 条进 LLM 评判，对 XLSX_RECALL_TOP_K 大于 12
        # 的场景反而是隐性截断。
        # 现在意图：xlsx 单条考点很短（30~80 字），直接把召回的全部送 LLM 评判，
        # 最终只保留最相关 max_items 条。召回数量由 JUDGE_XLSX_RECALL_TOP_K 控制
        # （新默认 10，prompt 总长可控）。
        candidates: List[str] = []
        seen = set()
        for item in retrieved_chunks:
            outline_path = _meaningful_outline_path(item.chunk)
            if not outline_path or outline_path in seen:
                continue
            seen.add(outline_path)
            meta = item.chunk.metadata if isinstance(item.chunk.metadata, dict) else {}
            node_id = _normalize_text(meta.get("NODEID") or meta.get("node_id"))
            prefix = f"NODEID={node_id} " if node_id else ""
            candidates.append(f"- {prefix}{outline_path} (score={item.score:.4f})")
        candidates_count = len(candidates)
        if not candidates:
            # 沉默吞 0 路径 #4：所有 xlsx 召回都被 _meaningful_outline_path
            # 过滤掉（黑名单只剩"内科/外科/专业理论知识/相关专业知识"或纯数字 NODEID）。
            logger.debug(
                "[RAG] outline candidates=0 (all %d xlsx_recall filtered by _meaningful_outline_path)",
                len(retrieved_chunks),
            )
            return [], 0
        user_prompt = f"""请从候选考点中判断当前 {question_type} 型题是否围绕明确、合理的医学考点展开。

要求：
1. 只能使用给定候选考点，不要新增考点。
2. 丢弃只有专业名称、编码或过于宽泛的路径。
3. 最多输出 {max_items} 条可给最终评分模型参考的简短意见。
4. 如果没有明确匹配，输出空数组。
5. 严格输出JSON：{{"outline_reference": ["..."]}}

【当前题】
{_shorten(candidate_text, 2600)}

【候选考点】
{chr(10).join(candidates)}
"""
        try:
            obj = self._chat_json(
                system_prompt="你是医学考试考点匹配审核员。只输出严格JSON。",
                user_prompt=user_prompt,
                max_tokens=JUDGE_RAG_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("xlsx outline LLM match failed: %s", exc)
            return [], candidates_count
        items = obj.get("outline_reference")
        if not isinstance(items, list):
            # 沉默吞 0 路径 #2：obj 是合法 JSON 但 outline_reference 不是 list。
            logger.debug(
                "[RAG] outline LLM returned non-list (type=%s, keys=%s, candidates=%d)",
                type(items).__name__, list(obj.keys()), candidates_count,
            )
            return [], candidates_count
        final = _dedupe_strings((str(item) for item in items), limit=max_items)
        logger.debug(
            "[RAG] outline candidates=%d → llm_kept=%d → final=%d",
            candidates_count, len(items), len(final),
        )
        return final, candidates_count


def _load_jsonl_chunks(path: str) -> List[KnowledgeChunk]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    items: List[KnowledgeChunk] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                KnowledgeChunk(
                    id=str(obj.get("id", f"{file_path.stem}_{line_no}")),
                    source=str(obj.get("source", file_path.name)),
                    text=str(obj.get("text", "")),
                    metadata=obj if isinstance(obj, dict) else {},
                )
            )
    return items


def _load_index(chunk_path: str, embedding_path: str) -> Optional[DenseKnowledgeIndex]:
    chunks = _load_jsonl_chunks(chunk_path)
    if not chunks:
        return None
    emb_file = Path(embedding_path)
    if not emb_file.exists():
        logger.warning("knowledge embedding file missing: %s", embedding_path)
        return None
    embeddings = np.load(emb_file)
    return DenseKnowledgeIndex(chunks=chunks, embeddings=embeddings)


def load_knowledge_base(
    *,
    judge_api_base: str = JUDGE_API_BASE,
    judge_api_key: str = JUDGE_API_KEY,
    judge_model: str = JUDGE_MODEL,
    embedding_api_base: str = JUDGE_EMBEDDING_API_BASE,
    embedding_api_key: str = JUDGE_EMBEDDING_API_KEY,
    embedding_model: str = JUDGE_EMBEDDING_MODEL,
    pdf_chunk_path: str = JUDGE_KNOWLEDGE_PDF_CHUNKS,
    pdf_embedding_path: str = JUDGE_KNOWLEDGE_PDF_EMBEDDINGS,
    xlsx_chunk_path: str = JUDGE_KNOWLEDGE_XLSX_CHUNKS,
    xlsx_embedding_path: str = JUDGE_KNOWLEDGE_XLSX_EMBEDDINGS,
    context_cache_path: str = JUDGE_KNOWLEDGE_CONTEXT_CACHE,
) -> Optional[JudgeKnowledgeBase]:
    # ── 收集 RAG 部件失活原因（关键改动：之前是静默 None，现在每个失活原因
    # 都会 emit WARNING + 留痕到 disable_reasons 给下游读取）──
    disable_reasons: List[str] = []

    pdf_index = _load_index(pdf_chunk_path, pdf_embedding_path)
    if pdf_index is None:
        reason = (
            f"PDF index 未加载 (chunks={pdf_chunk_path}, "
            f"embeddings={pdf_embedding_path})"
        )
        disable_reasons.append(reason)
        logger.warning("[RAG] %s — PDF Advice Evidence 将不可用", reason)

    xlsx_index = _load_index(xlsx_chunk_path, xlsx_embedding_path)
    if xlsx_index is None:
        reason = (
            f"XLSX index 未加载 (chunks={xlsx_chunk_path}, "
            f"embeddings={xlsx_embedding_path})"
        )
        disable_reasons.append(reason)
        logger.warning("[RAG] %s — 考点匹配参考将不可用", reason)

    if pdf_index is None and xlsx_index is None:
        logger.warning(
            "[RAG] PDF + XLSX 两个 index 都没加载到，knowledge base 整体禁用。"
            "如果是首次运行请检查 %s 与 %s 是否存在并已生成 embeddings.npy。",
            JUDGE_KNOWLEDGE_PDF_EMBEDDINGS, JUDGE_KNOWLEDGE_XLSX_EMBEDDINGS,
        )
        return None

    # ── Embedding client（用于 dense retrieval 查询向量化）──
    embedding_client = None
    if not embedding_api_base:
        reason = "embedding_api_base 为空（AIEDU_EMBEDDING_API_BASE 未设置）"
        disable_reasons.append(reason)
        logger.warning("[RAG] %s — 无法做向量召回，evidence 会全空", reason)
    elif not embedding_model:
        reason = "embedding_model 为空（AIEDU_EMBEDDING_MODEL 未设置）"
        disable_reasons.append(reason)
        logger.warning("[RAG] %s — 无法做向量召回，evidence 会全空", reason)
    elif not embedding_api_key and not JUDGE_ALLOW_EMPTY_API_KEY:
        reason = (
            "embedding_api_key 为空且 AIEDU_JUDGE_ALLOW_EMPTY_API_KEY=0；"
            "如果是 PPIO/OpenAI 等需鉴权服务请设置 AIEDU_EMBEDDING_API_KEY；"
            "如果是本地 vLLM 等无需鉴权服务请设置 AIEDU_JUDGE_ALLOW_EMPTY_API_KEY=1"
        )
        disable_reasons.append(reason)
        logger.warning("[RAG] %s — 无法做向量召回，evidence 会全空", reason)
    else:
        embedding_client = OpenAICompatibleClient(
            base_url=embedding_api_base, api_key=embedding_api_key,
        )
        logger.info(
            "[RAG] embedding client 就绪: base=%s, model=%s",
            embedding_api_base, embedding_model,
        )

    # ── Judge client（用于 PDF advice rerank + XLSX outline LLM 评判）──
    judge_client = None
    if not judge_api_base:
        reason = "judge_api_base 为空（AIEDU_JUDGE_API_BASE 未设置）"
        disable_reasons.append(reason)
        logger.warning(
            "[RAG] %s — PDF advice rerank / outline LLM 评判将不可用", reason,
        )
    elif not judge_model:
        reason = "judge_model 为空（AIEDU_JUDGE_MODEL 未设置）"
        disable_reasons.append(reason)
        logger.warning(
            "[RAG] %s — PDF advice rerank / outline LLM 评判将不可用", reason,
        )
    elif not judge_api_key and not JUDGE_ALLOW_EMPTY_API_KEY:
        reason = (
            "judge_api_key 为空且 AIEDU_JUDGE_ALLOW_EMPTY_API_KEY=0；"
            "如果是 PPIO/OpenAI 等需鉴权服务请设置 AIEDU_JUDGE_API_KEY；"
            "如果是本地 vLLM 等无需鉴权服务请设置 AIEDU_JUDGE_ALLOW_EMPTY_API_KEY=1"
        )
        disable_reasons.append(reason)
        logger.warning(
            "[RAG] %s — PDF advice rerank / outline LLM 评判将不可用", reason,
        )
    else:
        judge_client = OpenAICompatibleClient(
            base_url=judge_api_base, api_key=judge_api_key,
        )
        logger.info(
            "[RAG] judge client 就绪: base=%s, model=%s",
            judge_api_base, judge_model,
        )

    return JudgeKnowledgeBase(
        judge_client=judge_client,
        judge_model=judge_model,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
        pdf_index=pdf_index,
        xlsx_index=xlsx_index,
        context_cache_path=context_cache_path,
        pdf_chunk_path=pdf_chunk_path,
        xlsx_chunk_path=xlsx_chunk_path,
        disable_reasons=disable_reasons,
    )

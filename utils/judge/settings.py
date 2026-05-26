# -*- coding: utf-8 -*-
"""Centralized settings for the LLM Judge pipeline."""

from __future__ import annotations

import os


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value if value is not None else default


JUDGE_MODEL = _env("AIEDU_JUDGE_MODEL", "deepseek/deepseek-v3.2")
JUDGE_API_BASE = _env("AIEDU_JUDGE_API_BASE", "https://api.ppio.com/openai/v1")
JUDGE_API_KEY = _env("AIEDU_JUDGE_API_KEY", "")

JUDGE_EMBEDDING_MODEL = _env("AIEDU_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
JUDGE_EMBEDDING_API_BASE = _env("AIEDU_EMBEDDING_API_BASE", JUDGE_API_BASE)
JUDGE_EMBEDDING_API_KEY = _env("AIEDU_EMBEDDING_API_KEY", JUDGE_API_KEY)

JUDGE_REQUEST_TIMEOUT = int(_env("AIEDU_JUDGE_TIMEOUT", "3000"))
JUDGE_RATE_LIMIT_FIRST_BACKOFF = float(_env("AIEDU_JUDGE_RATE_LIMIT_FIRST_BACKOFF", "1"))
JUDGE_RATE_LIMIT_MAX_BACKOFF = float(_env("AIEDU_JUDGE_RATE_LIMIT_MAX_BACKOFF", "64"))
JUDGE_RATE_LIMIT_MAX_RETRIES = int(_env("AIEDU_JUDGE_RATE_LIMIT_MAX_RETRIES", "8"))
JUDGE_TRANSIENT_RETRIES = int(_env("AIEDU_JUDGE_TRANSIENT_RETRIES", "4"))
JUDGE_TEMPERATURE = float(_env("AIEDU_JUDGE_TEMPERATURE", "0"))
JUDGE_MAX_TOKENS = int(_env("AIEDU_JUDGE_MAX_TOKENS", "16000"))
JUDGE_SCORE_TOP_LOGPROBS = int(_env("AIEDU_JUDGE_SCORE_TOP_LOGPROBS", "20"))
# ↑ HTTP API 模式：增大默认值确保能获取到全部 10 个数字 token 的 logprobs

# ── vLLM Python engine 模式 ──────────────────────────────────────
# 使用 vLLM Python engine 的 logprob_token_ids 精确获取 0~9 数字 token 的 logprobs
# 不再依赖 top-K 截断，100% 保证覆盖
JUDGE_LOCAL_VLLM_PYENGINE = _env("AIEDU_JUDGE_LOCAL_VLLM_PYENGINE", "0") != "0"
# 本地 vLLM 模型路径（如 resources/model/Baichuan-M2-32B-0226）
JUDGE_LOCAL_VLLM_MODEL_PATH = _env("AIEDU_JUDGE_LOCAL_VLLM_MODEL_PATH", "")
# 本地 vLLM judge LoRA adapter 路径；为空时不启用 LoRA
JUDGE_LOCAL_VLLM_LORA_PATH = _env("AIEDU_JUDGE_LOCAL_VLLM_LORA_PATH", "")
JUDGE_VLLM_MAX_MODEL_LEN = int(_env("AIEDU_JUDGE_VLLM_MAX_MODEL_LEN", "24576"))
JUDGE_VLLM_GPU_MEMORY_UTILIZATION = float(_env("AIEDU_JUDGE_VLLM_GPU_MEM_UTIL", "0.9"))
JUDGE_VLLM_TENSOR_PARALLEL_SIZE = int(_env("AIEDU_JUDGE_VLLM_TP_SIZE", "1"))
# vLLM engine daemon 所在 GPU 索引（默认卡 0，可通过 AIEDU_JUDGE_VLLM_DEVICE="1" 切换）
JUDGE_VLLM_DEVICE = _env("AIEDU_JUDGE_VLLM_DEVICE", "0")
# digit logprobs 锐化温度倒数：sharpness>1 让 logprobs 经 softmax 后更尖锐，
# 期望值更靠近 argmax，reward 信号宽度更大。详见 vllm_scorer.VLLMEngineScorer.__init__。
# 2.0 是经验甜点：让 reward stdev 大致翻倍但保留分布连续性。
JUDGE_VLLM_SHARPNESS = float(_env("AIEDU_JUDGE_VLLM_SHARPNESS", "2.0"))

# 本地部署模式：所有 API 错误直接抛错，不重试
JUDGE_LOCAL_MODE = _env("AIEDU_JUDGE_LOCAL_MODE", "0") != "0"
# 本地部署（如 vLLM）通常不需要 API Key
JUDGE_ALLOW_EMPTY_API_KEY = _env("AIEDU_JUDGE_ALLOW_EMPTY_API_KEY", "0") != "0"

JUDGE_CACHE_PATH = _env("AIEDU_JUDGE_CACHE_PATH", "output/judge_cache/section_judge_cache.jsonl")

JUDGE_KNOWLEDGE_DIR = _env("AIEDU_JUDGE_KNOWLEDGE_DIR", "output/judge_knowledge")
JUDGE_KNOWLEDGE_PDF_PATH = _env("AIEDU_JUDGE_KNOWLEDGE_PDF_PATH", "data/医学考点命题.pdf")
JUDGE_KNOWLEDGE_XLSX_PATH = _env("AIEDU_JUDGE_KNOWLEDGE_XLSX_PATH", "data/知识体系.xlsx")
JUDGE_KNOWLEDGE_PDF_CHUNKS = _env(
    "AIEDU_JUDGE_PDF_CHUNKS",
    os.path.join(JUDGE_KNOWLEDGE_DIR, "pdf_chunks.jsonl"),
)
JUDGE_KNOWLEDGE_PDF_EMBEDDINGS = _env(
    "AIEDU_JUDGE_PDF_EMBEDDINGS",
    os.path.join(JUDGE_KNOWLEDGE_DIR, "pdf_embeddings.npy"),
)
JUDGE_KNOWLEDGE_XLSX_CHUNKS = _env(
    "AIEDU_JUDGE_XLSX_CHUNKS",
    os.path.join(JUDGE_KNOWLEDGE_DIR, "xlsx_chunks.jsonl"),
)
JUDGE_KNOWLEDGE_XLSX_EMBEDDINGS = _env(
    "AIEDU_JUDGE_XLSX_EMBEDDINGS",
    os.path.join(JUDGE_KNOWLEDGE_DIR, "xlsx_embeddings.npy"),
)
JUDGE_KNOWLEDGE_TOP_K = int(_env("AIEDU_JUDGE_KNOWLEDGE_TOP_K", "4"))
# RAG 召回 / 输出参数（2026-05-19 性能调优后默认值）
# 旧默认 (PDF 40 召回 / 3 advice, XLSX 30 召回 / 3 outline) 在训练时 RAG 占用
# 单条 judge 评分 ~80% 的耗时——主要瓶颈是 PDF rerank LLM 调用次数 = ceil(召回/batch)。
#
# 新默认（基于用户真实意图）：
#   PDF: 召回 20 条 → 每次 LLM rerank 输入 10 条（2 次 LLM 调用）→ 最终保留 3 条
#   XLSX: 召回 10 条 → 全部一次性送 LLM 评判（不分批，xlsx 单条很短）→ 最终保留 3 条
#
# 与旧默认相比，PDF rerank LLM 调用次数从 5 次（40÷8）降到 2 次（20÷10）
# —— 每条 section judge 省 3 次 LLM 调用。
# 全部可通过环境变量覆盖，无需改代码。
JUDGE_PDF_RECALL_TOP_K = int(_env("AIEDU_JUDGE_PDF_RECALL_TOP_K", "20"))
JUDGE_PDF_RERANK_BATCH_SIZE = int(_env("AIEDU_JUDGE_PDF_RERANK_BATCH_SIZE", "10"))
JUDGE_MAX_PDF_ADVICE = int(_env("AIEDU_JUDGE_MAX_PDF_ADVICE", "3"))
JUDGE_XLSX_RECALL_TOP_K = int(_env("AIEDU_JUDGE_XLSX_RECALL_TOP_K", "10"))
JUDGE_MAX_OUTLINE_ITEMS = int(_env("AIEDU_JUDGE_MAX_OUTLINE_ITEMS", "3"))
JUDGE_RAG_MAX_TOKENS = int(_env("AIEDU_JUDGE_RAG_MAX_TOKENS", "2048"))
JUDGE_KNOWLEDGE_CONTEXT_CACHE = _env(
    "AIEDU_JUDGE_KNOWLEDGE_CONTEXT_CACHE",
    "output/judge_cache/knowledge_context_cache.jsonl",
)

JUDGE_MIN_REWARD_FLOOR = float(_env("AIEDU_JUDGE_FLOOR", "0.05"))

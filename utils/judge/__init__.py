# -*- coding: utf-8 -*-
"""LLM Judge + RAG utilities for per-question-type reward scoring."""

from .api import OpenAICompatibleClient
from .knowledge import JudgeKnowledgeBase, load_knowledge_base
from .prompts import (
    build_section_distribution_prompt,
)
from .scorer import LLMSectionJudge, get_default_judge
from .settings import (
    JUDGE_ALLOW_EMPTY_API_KEY,
    JUDGE_API_BASE,
    JUDGE_API_KEY,
    JUDGE_CACHE_PATH,
    JUDGE_EMBEDDING_API_BASE,
    JUDGE_EMBEDDING_API_KEY,
    JUDGE_EMBEDDING_MODEL,
    JUDGE_KNOWLEDGE_DIR,
    JUDGE_KNOWLEDGE_PDF_PATH,
    JUDGE_KNOWLEDGE_TOP_K,
    JUDGE_KNOWLEDGE_XLSX_PATH,
    JUDGE_LOCAL_MODE,
    JUDGE_LOCAL_VLLM_LORA_PATH,
    JUDGE_LOCAL_VLLM_MODEL_PATH,
    JUDGE_LOCAL_VLLM_PYENGINE,
    JUDGE_MODEL,
    JUDGE_VLLM_GPU_MEMORY_UTILIZATION,
    JUDGE_VLLM_MAX_MODEL_LEN,
    JUDGE_VLLM_SHARPNESS,
    JUDGE_VLLM_TENSOR_PARALLEL_SIZE,
)
from .vllm_scorer import VLLMEngineScorer

__all__ = [
    "OpenAICompatibleClient",
    "JudgeKnowledgeBase",
    "LLMSectionJudge",
    "VLLMEngineScorer",
    "JUDGE_ALLOW_EMPTY_API_KEY",
    "JUDGE_API_BASE",
    "JUDGE_API_KEY",
    "JUDGE_CACHE_PATH",
    "JUDGE_EMBEDDING_API_BASE",
    "JUDGE_EMBEDDING_API_KEY",
    "JUDGE_EMBEDDING_MODEL",
    "JUDGE_KNOWLEDGE_DIR",
    "JUDGE_KNOWLEDGE_PDF_PATH",
    "JUDGE_KNOWLEDGE_TOP_K",
    "JUDGE_KNOWLEDGE_XLSX_PATH",
    "JUDGE_LOCAL_MODE",
    "JUDGE_LOCAL_VLLM_LORA_PATH",
    "JUDGE_LOCAL_VLLM_MODEL_PATH",
    "JUDGE_LOCAL_VLLM_PYENGINE",
    "JUDGE_MODEL",
    "JUDGE_VLLM_GPU_MEMORY_UTILIZATION",
    "JUDGE_VLLM_MAX_MODEL_LEN",
    "JUDGE_VLLM_SHARPNESS",
    "JUDGE_VLLM_TENSOR_PARALLEL_SIZE",
    "build_section_distribution_prompt",
    "get_default_judge",
    "load_knowledge_base",
]

# -*- coding: utf-8 -*-
"""Offline builder for Judge RAG corpora and embeddings."""

from __future__ import annotations

import argparse
import logging
import types
import os
import sys

from utils.judge.settings import (
    JUDGE_API_BASE,
    JUDGE_API_KEY,
    JUDGE_EMBEDDING_API_BASE,
    JUDGE_EMBEDDING_API_KEY,
    JUDGE_EMBEDDING_MODEL,
    JUDGE_KNOWLEDGE_PDF_CHUNKS,
    JUDGE_KNOWLEDGE_PDF_EMBEDDINGS,
    JUDGE_KNOWLEDGE_PDF_PATH,
    JUDGE_KNOWLEDGE_XLSX_CHUNKS,
    JUDGE_KNOWLEDGE_XLSX_EMBEDDINGS,
    JUDGE_KNOWLEDGE_XLSX_PATH,
    JUDGE_MODEL,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

# pdfminer/pdffont 在 FontBBox 为 None 时发出 WARNING，
# 但代码本身正确处理了这个情况（fallback 到 0,0,0,0）且这是常见 PDF 特征。
# 将该 WARNING 降级为 DEBUG，不压制其他 pdfminer 错误。
_pdfminer_font_logger = logging.getLogger("pdfminer.pdffont")
_pdfminer_font_logger.setLevel(logging.DEBUG)
_original_warning = _pdfminer_font_logger.warning
def _patched_warning(self, msg, *args, **kwargs):
    if "FontBBox" in str(msg):
        self.debug(msg, *args, **kwargs)
    else:
        _original_warning(msg, *args, **kwargs)
_pdfminer_font_logger.warning = types.MethodType(_patched_warning, _pdfminer_font_logger)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline Judge RAG knowledge base")
    parser.add_argument("--pdf-path", type=str, default=JUDGE_KNOWLEDGE_PDF_PATH)
    parser.add_argument("--xlsx-path", type=str, default=JUDGE_KNOWLEDGE_XLSX_PATH)
    parser.add_argument("--pdf-chunks", type=str, default=JUDGE_KNOWLEDGE_PDF_CHUNKS)
    parser.add_argument("--pdf-embeddings", type=str, default=JUDGE_KNOWLEDGE_PDF_EMBEDDINGS)
    parser.add_argument("--xlsx-chunks", type=str, default=JUDGE_KNOWLEDGE_XLSX_CHUNKS)
    parser.add_argument("--xlsx-embeddings", type=str, default=JUDGE_KNOWLEDGE_XLSX_EMBEDDINGS)
    parser.add_argument("--judge-api-base", type=str, default=JUDGE_API_BASE)
    parser.add_argument("--judge-api-key", type=str, default=JUDGE_API_KEY)
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--embedding-api-base", type=str, default=JUDGE_EMBEDDING_API_BASE)
    parser.add_argument("--embedding-api-key", type=str, default=JUDGE_EMBEDDING_API_KEY)
    parser.add_argument("--embedding-model", type=str, default=JUDGE_EMBEDDING_MODEL)
    return parser.parse_args()


def main() -> int:
    from utils.judge.api import OpenAICompatibleClient
    from utils.judge.knowledge_build import (
        build_embeddings_for_records,
        build_pdf_chunk_records,
        build_xlsx_chunk_records,
        save_jsonl,
    )

    args = parse_args()
    if not str(args.judge_api_base).strip():
        raise ValueError("judge-api-base 不能为空：PDF 标签必须由 LLM 生成。")
    if not str(args.judge_api_key).strip():
        raise ValueError("judge-api-key 不能为空：PDF 标签必须由 LLM 生成。")
    if not str(args.judge_model).strip():
        raise ValueError("judge-model 不能为空：PDF 标签必须由 LLM 生成。")
    tag_client = OpenAICompatibleClient(base_url=args.judge_api_base, api_key=args.judge_api_key)
    embedding_client = OpenAICompatibleClient(base_url=args.embedding_api_base, api_key=args.embedding_api_key)

    os.makedirs(os.path.dirname(args.pdf_chunks) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.xlsx_chunks) or ".", exist_ok=True)

    logger.info("building pdf chunks: %s", args.pdf_path)
    pdf_records = build_pdf_chunk_records(
        pdf_path=args.pdf_path,
        tagging_client=tag_client,
        tagging_model=args.judge_model if tag_client is not None else "",
    )
    save_jsonl(pdf_records, args.pdf_chunks)
    logger.info("saved pdf chunks: %d -> %s", len(pdf_records), args.pdf_chunks)

    logger.info("building xlsx chunks: %s", args.xlsx_path)
    xlsx_records = build_xlsx_chunk_records(args.xlsx_path)
    save_jsonl(xlsx_records, args.xlsx_chunks)
    logger.info("saved xlsx chunks: %d -> %s", len(xlsx_records), args.xlsx_chunks)

    logger.info("building pdf embeddings with model=%s", args.embedding_model)
    build_embeddings_for_records(
        records=pdf_records,
        output_path=args.pdf_embeddings,
        embedding_client=embedding_client,
        embedding_model=args.embedding_model,
    )
    logger.info("saved pdf embeddings -> %s", args.pdf_embeddings)

    logger.info("building xlsx embeddings with model=%s", args.embedding_model)
    build_embeddings_for_records(
        records=xlsx_records,
        output_path=args.xlsx_embeddings,
        embedding_client=embedding_client,
        embedding_model=args.embedding_model,
    )
    logger.info("saved xlsx embeddings -> %s", args.xlsx_embeddings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

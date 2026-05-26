# -*- coding: utf-8 -*-
"""Shared text extraction helpers for TRL prompt/completion payloads."""

from __future__ import annotations

import re

_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)


def strip_think_content(text) -> str:
    """与 generate_output.py 保持一致的 think 裁剪逻辑。"""
    if text is None:
        return ""

    text = str(text)

    # Baichuan-M2 thinking_mode 会把 <think> 放在输入 prompt 中，
    # 生成文本可能只包含孤立的 </think>；此时闭标签之前都是思考内容。
    last_orphan_close = None
    depth = 0
    for match in _THINK_TAG_RE.finditer(text):
        if match.group(0).lower().startswith("</"):
            if depth > 0:
                depth -= 1
            else:
                last_orphan_close = match
        else:
            depth += 1

    if last_orphan_close is not None:
        return strip_think_content(text[last_orphan_close.end():])

    parts = []
    cursor = 0
    depth = 0

    for match in _THINK_TAG_RE.finditer(text):
        tag = match.group(0).lower()
        is_close = tag.startswith("</")

        if depth == 0:
            parts.append(text[cursor:match.start()])

        if is_close:
            if depth > 0:
                depth -= 1
        else:
            depth += 1

        cursor = match.end()

    if depth == 0:
        parts.append(text[cursor:])

    return "".join(parts).strip()


def extract_completion_text(completion_item) -> str:
    if isinstance(completion_item, str):
        raw = completion_item
    elif isinstance(completion_item, dict):
        raw = str(completion_item.get("content", ""))
    elif isinstance(completion_item, list):
        chunks = []
        for item in completion_item:
            if isinstance(item, dict):
                chunks.append(str(item.get("content", "")))
            else:
                chunks.append(str(item))
        raw = "\n".join(chunks)
    else:
        raw = str(completion_item)

    return strip_think_content(raw)


def extract_prompt_text(prompt_item) -> str:
    if isinstance(prompt_item, str):
        return prompt_item
    if isinstance(prompt_item, dict):
        return str(prompt_item.get("content", ""))
    if isinstance(prompt_item, list):
        chunks = []
        for item in prompt_item:
            if isinstance(item, dict):
                chunks.append(str(item.get("content", "")))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(prompt_item)


__all__ = ["strip_think_content", "extract_completion_text", "extract_prompt_text"]

# -*- coding: utf-8 -*-
"""结构奖励计算模块。"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# 题型顺序
SECTION_ORDER = ["A1", "A2", "A3", "A4", "B1"]

# 全角字符转换表
_FULLWIDTH_TRANSLATION = str.maketrans({
    "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D",
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "．": ".", "。": ".", "：": ":", "（": "(", "）": ")",
})

# 题型标题正则
SECTION_PATTERNS = {
    "A1": re.compile(r"(?:^|\n)\s{0,3}(?:#+\s*)?(?:[一二三四五六七八九十]+[、.．]\s*)?A1\s*型?题"),
    "A2": re.compile(r"(?:^|\n)\s{0,3}(?:#+\s*)?(?:[一二三四五六七八九十]+[、.．]\s*)?A2\s*型?题"),
    "A3": re.compile(r"(?:^|\n)\s{0,3}(?:#+\s*)?(?:[一二三四五六七八九十]+[、.．]\s*)?A3\s*型?题"),
    "A4": re.compile(r"(?:^|\n)\s{0,3}(?:#+\s*)?(?:[一二三四五六七八九十]+[、.．]\s*)?A4\s*型?题"),
    "B1": re.compile(r"(?:^|\n)\s{0,3}(?:#+\s*)?(?:[一二三四五六七八九十]+[、.．]\s*)?B1\s*型?题"),
}


def _normalize_for_section_match(text: str) -> str:
    """用于分段标题匹配的轻量标准化：仅全角转半角。"""
    if not isinstance(text, str) or not text:
        return ""
    return text.translate(_FULLWIDTH_TRANSLATION)


def _find_section_markers(text: str) -> List[Tuple[str, int]]:
    """按行扫描 A1/A2/A3/A4/B1 标题，返回每段在原文中的起始位置。"""
    if not isinstance(text, str) or not text.strip():
        return []

    found: List[Tuple[str, int]] = []
    seen = set()

    cursor = 0
    for line in text.splitlines(keepends=True):
        normalized_line = _normalize_for_section_match(line)

        for sec in SECTION_ORDER:
            if sec in seen:
                continue
            letter, digit = sec[0], sec[1]
            pattern = re.compile(rf"(?<![A-Z0-9]){letter}\s*{digit}(?!\d)\s*(?:型)?题")
            m = pattern.search(normalized_line)
            if m is not None:
                found.append((sec, cursor + m.start()))
                seen.add(sec)

        cursor += len(line)

    found.sort(key=lambda x: x[1])
    return found


def extract_question_sections(text: str) -> Tuple[Dict[str, str], List[Tuple[str, int, int]]]:
    """
    提取题型段落。

    返回:
        sections: {"A1": "...", "A2": "...", ...}
        spans: [("A1", start, end), ...]
    """
    if not isinstance(text, str) or not text.strip():
        return {key: "" for key in SECTION_ORDER}, []

    markers: List[Tuple[str, int]] = _find_section_markers(text)

    # 回退：若行级扫描未命中，再尝试旧正则
    if not markers:
        for sec in SECTION_ORDER:
            match = SECTION_PATTERNS[sec].search(text)
            if match is not None:
                markers.append((sec, match.start()))

    if not markers:
        return {key: "" for key in SECTION_ORDER}, []

    markers.sort(key=lambda x: x[1])
    sections = {key: "" for key in SECTION_ORDER}
    spans: List[Tuple[str, int, int]] = []
    for idx, (sec, start) in enumerate(markers):
        end = markers[idx + 1][1] if idx + 1 < len(markers) else len(text)
        sections[sec] = text[start:end].strip()
        spans.append((sec, start, end))
    return sections, spans


def is_benign_objective_prefix(text: str) -> bool:
    """判断前缀是否为可接受的标题噪声。"""
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return True

    normalized = s.translate(_FULLWIDTH_TRANSLATION)
    line_parts = re.split(r"(?:\r?\n|\\r\\n|\\n)+", normalized)
    lines = [ln.strip() for ln in line_parts if ln and ln.strip()]
    if not lines:
        return True

    for ln in lines:
        core = re.sub(r"^#+\s*", "", ln).strip()
        if not core:
            continue

        if re.fullmatch(r"客观题[：:]?", core):
            continue

        if re.fullmatch(r"(?:[一二三四五六七八九十]+[、.．]?|\(?[一二三四五六七八九十]+\)?|\d+[、.．]?)", core):
            continue

        return False

    return True


def analyze_structure(
    text: str,
    sections: Optional[Dict[str, str]] = None,
    spans: Optional[List[Tuple[str, int, int]]] = None,
) -> Dict[str, float]:
    """
    分析文本结构特征。

    返回:
        {
            "is_complete": float,        # 是否包含全部五段
            "is_order_correct": float,   # 顺序是否正确
            "front_dirty": float,        # 前置脏数据
            "back_dirty": float,         # 后置脏数据
            "dirty_rate": float,         # 脏数据率
            "has_extra_types": float,    # 是否有额外题型
        }
    """
    if sections is None or spans is None:
        sections, spans = extract_question_sections(text)

    observed_order = [sec for sec, _, _ in spans if sections.get(sec, "").strip()]
    observed_set = set(observed_order)

    is_complete = float(
        1.0
        if len(observed_order) == len(SECTION_ORDER) and observed_set == set(SECTION_ORDER)
        else 0.0
    )
    is_order_correct = float(1.0 if observed_order == SECTION_ORDER else 0.0)

    front_dirty = 1.0
    back_dirty = 1.0
    has_extra_types = False

    if isinstance(text, str) and text.strip():
        normalized = _normalize_for_section_match(text)
        allowed = set(SECTION_ORDER)
        marker_pattern = re.compile(r"(?<![A-Z0-9])([A-Z])\s*([1-9])(?!\d)\s*(?:型)?题")
        for m in marker_pattern.finditer(normalized):
            sec = f"{m.group(1)}{m.group(2)}"
            if sec not in allowed:
                has_extra_types = True
                break

    if isinstance(text, str) and text.strip() and spans:
        first_start = spans[0][1]
        last_end = spans[-1][2]
        prefix = text[:first_start]
        front_dirty = float(0.0 if is_benign_objective_prefix(prefix) else 1.0)
        back_dirty = float(1.0 if text[last_end:].strip() else 0.0)

    return {
        "is_complete": is_complete,
        "is_order_correct": is_order_correct,
        "front_dirty": front_dirty,
        "back_dirty": back_dirty,
        "dirty_rate": float((front_dirty + back_dirty) / 2.0),
        "has_extra_types": float(1.0 if has_extra_types else 0.0),
    }


def check_dirty_type(original: str, revised: str) -> Optional[str]:
    """
    检测脏类型。
    返回: "exact_duplicate" | "partial_duplicate" | "no_change" | None
    """
    if original.strip() == revised.strip():
        return "exact_duplicate"
    # 可扩展其他脏类型检测
    return None

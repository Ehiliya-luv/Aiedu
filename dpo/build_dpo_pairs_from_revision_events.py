# -*- coding: utf-8 -*-
"""从 revision_events.jsonl 和 raw_changed_pairs.jsonl 构建单题 DPO pair。

口径：
1. revision_events.jsonl: 使用完整 current_question / previous_question。
2. raw_changed_pairs.jsonl: 从完整 raw 前后版本中切分 A1/A2/A3/A4/B1 section。
3. 只输出有修改前版本且文本真实发生变化的 pair。
4. 用 (previous_raw_id, current_raw_id, question_type) 去重；结构化题优先。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from utils.dpo_prompt import (
    build_single_question_prompt,
    collapse_blank_lines,
    has_dpo_effective_change,
    render_question_text,
    safe_strip,
)
from utils.case_context import load_case_contexts, resolve_case_text


QUESTION_TYPES = ("A1", "A2", "A3", "A4", "B1")
TYPE_HEADING_RE = re.compile(
    r"(?m)^[\s#>*\-—一二三四五六七八九十\d\.．、:：()（）]*"
    r"(?P<qtype>A1|A2|A3|A4|B1)\s*型题\b.*$"
)
DROP_EXTRA_LABEL_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?"
    r"(?:知识类别|难易度|认知程度|知识点大纲|知识拓展|出题思路|修改说明|免责声明)"
    r"(?:\*\*)?\s*[:：].*$"
)
DROP_EXTRA_HEADING_RE = re.compile(
    r"^\s*#{1,6}\s*(?:知识类别|难易度|认知程度|知识点大纲|知识拓展|出题思路|修改说明|免责声明)\s*$"
)
EXPLANATION_ALIAS_RE = re.compile(r"^\s*(?:\*\*)?出题逻辑(?:\*\*)?\s*[:：]\s*(.*)$")
INLINE_EXTRA_LABEL_RE = re.compile(
    r"\s+(?:\*\*)?(?:知识类别|难易度|认知程度|知识点大纲|知识拓展|出题思路|修改说明|免责声明)(?:\*\*)?\s*[:：].*$"
)
AUDIT_STATUS_LINE_RE = re.compile(r"(?:已完成|最终输出格式完整|符合要求|新增).*(?:知识点大纲|知识体系|叶子节点|字段)")


@dataclass
class BuildStats:
    revision_total: int = 0
    raw_total: int = 0
    written: int = 0
    duplicate: int = 0
    skipped_missing_previous: int = 0
    skipped_missing_current: int = 0
    skipped_no_change: int = 0
    skipped_chosen_invalid: int = 0
    skipped_rejected_invalid: int = 0
    missing_case_context: int = 0
    written_by_source: Counter = field(default_factory=Counter)
    written_by_type: Counter = field(default_factory=Counter)


def iter_jsonl(path: str, required: bool = True) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"输入文件不存在: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"非法 JSON: {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSON 行必须是对象: {path}:{line_no}")
            yield obj


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_for_compare(text: Any) -> str:
    normalized = safe_strip(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def text_changed(before: str, after: str) -> bool:
    return normalize_for_compare(before) != normalize_for_compare(after)


def short_hash(text: str) -> str:
    return hashlib.md5(normalize_for_compare(text).encode("utf-8")).hexdigest()[:12]


def is_structurally_valid(text: str) -> bool:
    """
    判断一段题目文本是否结构完整。
    宽松校验：有题型标记、有答案标记、有实质内容。
    支持的答案标记格式：
    - "答案：X" / "正确答案：X"
    - "**答案**：X" / "**正确答案**：X"
    - "### 答案" 后跟内容
    - "**选项**" 整行加粗（如: "- **B. 内容**"）
    """
    if len(text) < 50:
        return False
    if "答案" not in text and "正确答案" not in text:
        # 检测 **选项** 格式（整行加粗选项表示答案）
        if not re.search(r"- \*\*[A-E][.．]", text):
            return False
    if not re.search(r"(?:A[1-4]|B1)\s*型题", text):
        return False
    return True


def normalize_answer_format(text: str) -> str:
    """
    归一化答案标记格式。
    1. 去掉 **选项** 加粗标记（始终保持格式一致）
    2. 若文本原本无"答案"关键词，在最后一个 section 标题前添加 答案：X 行
    """
    bold_pattern = re.compile(r"^(\s*)-\s*\*\*([A-E])[.．]\s*(.+?)\*\*\s*$", re.MULTILINE)
    if not bold_pattern.search(text):
        return text

    lines = text.split("\n")
    new_lines: List[str] = []
    answers: List[str] = []
    has_keyword = "答案" in text or "正确答案" in text

    for raw_line in lines:
        m = re.match(r"^(\s*)-\s*\*\*([A-E])[.．]\s*(.+?)\*\*\s*$", raw_line)
        if m:
            indent = m.group(1)
            letter = m.group(2)
            content = m.group(3).strip()
            new_lines.append(f"{indent}- {letter}. {content}")
            answers.append(f"答案：{letter}")
        else:
            new_lines.append(raw_line)

    if not has_keyword and answers:
        section_pos = len(new_lines)
        for i in range(len(new_lines) - 1, -1, -1):
            if re.match(r"^#{2,4}\s", new_lines[i]):
                section_pos = i
                break
        result_lines = new_lines[:section_pos]
        if result_lines and result_lines[-1] != "":
            result_lines.append("")
        result_lines.extend(answers)
        result_lines.append("")
        result_lines.extend(new_lines[section_pos:])
        return "\n".join(result_lines).strip()

    return "\n".join(new_lines).strip()


def parse_type_sections(markdown: str) -> Dict[str, List[str]]:
    text = safe_strip(markdown).replace("\r\n", "\n").replace("\r", "\n")
    matches = list(TYPE_HEADING_RE.finditer(text))
    sections: Dict[str, List[str]] = {}
    for idx, match in enumerate(matches):
        qtype = match.group("qtype")
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.setdefault(qtype, []).append(section_text)
    return sections


def combined_type_text(sections: Dict[str, List[str]], qtype: str) -> str:
    return "\n\n".join(sections.get(qtype) or []).strip()


def strip_preguide_text(text: str) -> str:
    """
    裁掉出题规划/指南文字。
    只保留第一个 ## / ### 题型标题之后的内容。
    """
    m = re.search(r"^#{2,3}\s+(?:A[1-4]|B1)\s*型题", text, re.MULTILINE)
    if m:
        return text[m.start():].strip()
    return text


def clean_final_question_text(text: str) -> str:
    """保留最终题目内容，去掉生成/审核过程中的额外元信息。"""
    normalized = safe_strip(text).replace("\r\n", "\n").replace("\r", "\n")
    kept: List[str] = []
    skip_extra_block = False
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if AUDIT_STATUS_LINE_RE.search(line):
            continue
        if DROP_EXTRA_HEADING_RE.match(line):
            skip_extra_block = True
            continue
        if skip_extra_block:
            if re.match(r"^#{2,4}\s+(?:A[1-4]|B1)\s*型题", line) or re.match(r"^#{1,6}\s*(?:答案|解析|试题解析)\b", line):
                skip_extra_block = False
            else:
                continue
        alias_match = EXPLANATION_ALIAS_RE.match(line)
        if alias_match:
            kept.append(f"试题解析：{alias_match.group(1).strip()}".rstrip())
            continue
        if DROP_EXTRA_LABEL_RE.match(line):
            continue
        kept.append(INLINE_EXTRA_LABEL_RE.sub("", raw_line.rstrip()).rstrip())
    return collapse_blank_lines("\n".join(kept), max_blank_lines=1)


def raw_value(raw_obj: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw_obj:
            return raw_obj.get(key)
    return None


def extract_raw_pair(raw_event: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    previous = raw_event.get("previous_raw") if isinstance(raw_event.get("previous_raw"), dict) else {}
    current = raw_event.get("current_raw") if isinstance(raw_event.get("current_raw"), dict) else {}

    if not previous:
        previous = {
            "raw_id": raw_event.get("prev_raw_id"),
            "version": raw_event.get("prev_version"),
            "examination_questions": raw_event.get("prev_questions"),
        }
    if not current:
        current = {
            "raw_id": raw_event.get("curr_raw_id"),
            "version": raw_event.get("curr_version"),
            "examination_questions": raw_event.get("curr_questions"),
        }
    return previous, current


def make_pair(
    prompt_context: str,
    question_type: str,
    rejected: str,
    chosen: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = build_single_question_prompt(prompt_context, question_type)
    return {
        "prompt": prompt,
        "chosen": clean_final_question_text(chosen),
        "rejected": clean_final_question_text(rejected),
        "meta": meta,
    }


def revision_key(event: Dict[str, Any], question_type: str, rejected: str, chosen: str) -> Tuple[Any, ...]:
    prev_raw_id = event.get("previous_raw_medical_id")
    curr_raw_id = event.get("raw_medical_id")
    if prev_raw_id is not None and curr_raw_id is not None:
        return ("raw_pair", int(prev_raw_id), int(curr_raw_id), question_type)
    return ("revision", event.get("sample_id"), question_type, short_hash(rejected), short_hash(chosen))


def raw_key(previous: Dict[str, Any], current: Dict[str, Any], question_type: str, rejected: str, chosen: str) -> Tuple[Any, ...]:
    prev_raw_id = raw_value(previous, "raw_id", "id")
    curr_raw_id = raw_value(current, "raw_id", "id")
    if prev_raw_id is not None and curr_raw_id is not None:
        return ("raw_pair", int(prev_raw_id), int(curr_raw_id), question_type)
    return ("raw_text", question_type, short_hash(rejected), short_hash(chosen))


def build_pair_from_revision_event(
    event: Dict[str, Any],
    *,
    case_contexts: Optional[Dict[Tuple[str, int], str]] = None,
    stats: Optional[BuildStats] = None,
) -> Tuple[Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]], str]:
    question_type = safe_strip(event.get("question_type"))
    if question_type not in QUESTION_TYPES:
        return None, "missing_current"

    previous_question = event.get("previous_question") if isinstance(event.get("previous_question"), dict) else {}
    current_question = event.get("current_question") if isinstance(event.get("current_question"), dict) else {}
    if not previous_question:
        return None, "missing_previous"
    if not current_question:
        return None, "missing_current"
    if not has_dpo_effective_change(previous_question, current_question):
        return None, "no_change"

    rejected = clean_final_question_text(render_question_text(question_type, previous_question))
    chosen = clean_final_question_text(render_question_text(question_type, current_question))
    if not rejected:
        return None, "missing_previous"
    if not chosen:
        return None, "missing_current"
    if not text_changed(rejected, chosen):
        return None, "no_change"

    case_context = event.get("case_context") if isinstance(event.get("case_context"), dict) else {}
    prompt_context = collapse_blank_lines(case_context.get("original_content"), max_blank_lines=1)
    if not prompt_context and case_contexts:
        prompt_context = resolve_case_text(
            case_contexts,
            fk_file_id=event.get("fk_file_id"),
            raw_medical_id=event.get("raw_medical_id"),
        )
    if not prompt_context and stats is not None:
        stats.missing_case_context += 1
    key = revision_key(event, question_type, rejected, chosen)
    meta = {
        "source": "revision_events",
        "sample_id": event.get("sample_id"),
        "question_type": question_type,
        "raw_medical_id": event.get("raw_medical_id"),
        "previous_raw_medical_id": event.get("previous_raw_medical_id"),
        "fk_file_id": event.get("fk_file_id"),
        "previous_question_item_id": previous_question.get("item_id"),
        "current_question_item_id": current_question.get("item_id"),
        "dedupe_key": list(key),
    }
    pair = make_pair(
        prompt_context=prompt_context,
        question_type=question_type,
        rejected=rejected,
        chosen=chosen,
        meta=meta,
    )
    return (key, pair), ""


def build_pairs_from_raw_event(
    raw_event: Dict[str, Any],
    stats: BuildStats,
    *,
    case_contexts: Optional[Dict[Tuple[str, int], str]] = None,
) -> List[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
    previous, current = extract_raw_pair(raw_event)
    prev_questions = safe_strip(raw_value(previous, "examination_questions", "questions", "text"))
    curr_questions = safe_strip(raw_value(current, "examination_questions", "questions", "text"))
    if not prev_questions or not curr_questions:
        return []

    prev_sections = parse_type_sections(prev_questions)
    curr_sections = parse_type_sections(curr_questions)
    pairs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

    for question_type in QUESTION_TYPES:
        rejected = combined_type_text(prev_sections, question_type)
        chosen = combined_type_text(curr_sections, question_type)
        if not rejected or not chosen:
            continue
        if not is_structurally_valid(chosen):
            stats.skipped_chosen_invalid += 1
            continue
        if not is_structurally_valid(rejected):
            stats.skipped_rejected_invalid += 1
            continue
        if not text_changed(rejected, chosen):
            continue

        # 归一化答案格式；chosen清洗出题指南文字
        rejected = normalize_answer_format(rejected)
        chosen = normalize_answer_format(chosen)
        chosen = strip_preguide_text(chosen)
        rejected = clean_final_question_text(rejected)
        chosen = clean_final_question_text(chosen)
        # 检测 chosen 中嵌入审校/质量批评性备注 → 题目本身质量不行，抛弃
        if re.search(r"【[^】]*?(?:题干不符合|不对的|审校|不应该|建议|不正确|有问题).*?】", chosen):
            stats.skipped_chosen_invalid += 1
            continue
        if not text_changed(rejected, chosen):
            continue
        key = raw_key(previous, current, question_type, rejected, chosen)
        current_raw_id = raw_value(current, "raw_id", "id")
        prompt_context = resolve_case_text(
            case_contexts or {},
            fk_file_id=raw_event.get("fk_file_id"),
            raw_medical_id=current_raw_id,
        )
        if not prompt_context:
            stats.missing_case_context += 1
        meta = {
            "source": "raw_changed_pairs",
            "question_type": question_type,
            "fk_file_id": raw_event.get("fk_file_id"),
            "previous_raw_id": raw_value(previous, "raw_id", "id"),
            "previous_version": raw_value(previous, "version"),
            "current_raw_id": current_raw_id,
            "current_version": raw_value(current, "version"),
            "dedupe_key": list(key),
        }
        pair = make_pair(
            prompt_context=prompt_context,
            question_type=question_type,
            rejected=rejected,
            chosen=chosen,
            meta=meta,
        )
        pairs.append((key, pair))
    return pairs


def build_dpo_pairs(
    revision_events: Iterable[Dict[str, Any]],
    raw_events: Iterable[Dict[str, Any]],
    *,
    case_contexts: Optional[Dict[Tuple[str, int], str]] = None,
) -> Tuple[List[Dict[str, Any]], BuildStats]:
    stats = BuildStats()
    pairs: List[Dict[str, Any]] = []
    seen = set()

    def add_pair(key: Tuple[Any, ...], pair: Dict[str, Any]) -> None:
        if key in seen:
            stats.duplicate += 1
            return
        seen.add(key)
        pairs.append(pair)
        stats.written += 1
        meta = pair.get("meta") if isinstance(pair.get("meta"), dict) else {}
        stats.written_by_source[meta.get("source", "unknown")] += 1
        stats.written_by_type[meta.get("question_type", "unknown")] += 1

    for event in revision_events:
        stats.revision_total += 1
        result, reason = build_pair_from_revision_event(
            event,
            case_contexts=case_contexts,
            stats=stats,
        )
        if result is None:
            if reason == "missing_previous":
                stats.skipped_missing_previous += 1
            elif reason == "missing_current":
                stats.skipped_missing_current += 1
            elif reason == "no_change":
                stats.skipped_no_change += 1
            continue
        add_pair(*result)

    for raw_event in raw_events:
        stats.raw_total += 1
        raw_pairs = build_pairs_from_raw_event(
            raw_event,
            stats,
            case_contexts=case_contexts,
        )
        if not raw_pairs:
            stats.skipped_no_change += 1
            continue
        for key, pair in raw_pairs:
            add_pair(key, pair)

    return pairs, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 revision_events.jsonl 和 raw_changed_pairs.jsonl 生成单题 DPO pair")
    parser.add_argument("--revision-events", default="data/revision_events.jsonl", help="revision_events.jsonl 路径")
    parser.add_argument("--raw-changed-pairs", default="data/raw_changed_pairs.jsonl", help="raw_changed_pairs.jsonl 路径")
    parser.add_argument(
        "--case-contexts",
        nargs="*",
        default=["data/rl_train.jsonl", "data/rl_train_extra.jsonl"],
        help="包含 input_context 和 metadata 的病例上下文 JSONL",
    )
    parser.add_argument("--output", default="data/dpo_pairs.jsonl", help="输出 dpo_pairs.jsonl 路径")
    parser.add_argument("--allow-missing-raw", action="store_true", help="允许 raw_changed_pairs.jsonl 不存在")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.path.exists(args.output) and not args.overwrite:
        print(f"[ERROR] 输出文件已存在，请使用 --overwrite: {args.output}", file=sys.stderr)
        return 1

    case_contexts = load_case_contexts(args.case_contexts)

    try:
        pairs, stats = build_dpo_pairs(
            revision_events=iter_jsonl(args.revision_events, required=True),
            raw_events=iter_jsonl(args.raw_changed_pairs, required=not args.allow_missing_raw),
            case_contexts=case_contexts,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    ensure_parent_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DPO BUILD]")
    print(f"revision_events: {stats.revision_total}")
    print(f"raw_changed_pairs: {stats.raw_total}")
    print(f"written: {stats.written}")
    print(f"duplicates_removed: {stats.duplicate}")
    print(f"skipped_missing_previous: {stats.skipped_missing_previous}")
    print(f"skipped_missing_current: {stats.skipped_missing_current}")
    print(f"skipped_chosen_invalid: {stats.skipped_chosen_invalid}")
    print(f"skipped_rejected_invalid: {stats.skipped_rejected_invalid}")
    print(f"skipped_no_change: {stats.skipped_no_change}")
    print(f"missing_case_context: {stats.missing_case_context}")
    print(f"written_by_source: {dict(stats.written_by_source)}")
    print(f"written_by_type: {dict(stats.written_by_type)}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Clean generated result files into a compare_experiments-compatible tree.

The cleaner is intentionally conservative: it removes only high-confidence
format wrappers and keeps source content unchanged otherwise.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple

from utils.com_exp.data_loader import discover_model_dirs, iter_output_files


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MAX_REMOVAL_RATIO = 0.50


QUESTION_TASK_NAMES = {"考题生成", "题目生成"}

SEPARATOR_RE = re.compile(r"^\s*[-*_]{3,}\s*$")
EMPTY_BULLET_RE = re.compile(r"^\s*[-*]\s*$")
QUESTION_BULLET_RE = re.compile(r"^(\s*)[-*]\s+(.+)$")
QUESTION_LABEL_RE = re.compile(
    r"^(\s*(?:\*\*)?)"
    r"(?:【\s*)?"
    r"(题干|选项)"
    r"(?:\s*】)?"
    r"(?:\*\*)?"
    r"\s*[：:]\s*"
    r"(.*)$"
)
QUESTION_LABEL_ONLY_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:【\s*)?(题干|选项)(?:\s*】)?\s*(?:\*\*)?\s*[：:]?\s*$"
)
NUMBERED_QUESTION_LABEL_RE = re.compile(
    r"^(\s*)(?:\*\*)?\s*(\d+)[.．、]\s*(?:\*\*)?\s*题干\s*(?:\*\*)?\s*[：:]\s*(.*)$"
)
COMMON_STEM_LABEL_RE = re.compile(
    r"^\s*(?:\*\*)?\s*共用题干\s*(?:\*\*)?\s*[：:]\s*(.*)$"
)
COMMON_STEM_MARKER_ONLY_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(\(\s*\d+\s*[~～-]\s*\d+\s*题共用题干\s*\))\s*(?:\*\*)?\s*$"
)
COMMON_OPTIONS_MARKER_ONLY_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(\(\s*\d+\s*[~～-]\s*\d+\s*题共用备选答案\s*\))\s*(?:\*\*)?\s*$"
)
COMMON_STEM_EMBEDDED_LABEL_RE = re.compile(
    r"^(\s*\(\s*1\s*[~～-]\s*\d+\s*题共用题干\s*\))\s*(?:\*\*)?\s*题干\s*(?:\*\*)?\s*[：:]?\s*(.+)$"
)
COMMON_STEM_PREFIX_RE = re.compile(
    r"^(\s*\(\s*1\s*[~～-]\s*\d+\s*题共用题干\s*\))\s*(.*)$"
)
QUESTION_SECTION_HEADING_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:\*\*)?\s*(?:[一二三四五]、)?\s*[A-Z][1-9]型题\b"
)
A3_A4_SECTION_HEADING_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:\*\*)?\s*(?:[三四]、)?\s*A[34]型题\b"
)
SECTION_HEADING_COMMON_SUFFIX_RE = re.compile(
    r"^(.{0,40}?A[34]型题)\s*(?:\*\*)?\s*\(\s*\d+\s*[~～-]\s*\d+\s*题共用题干\s*\)\s*(?:\*\*)?\s*$"
)
INLINE_OPTION_MARK_RE = re.compile(r"(?<!\S)([A-E])[.．、]\s+")
QUESTION_BOLD_WRAPPED_LINE_RE = re.compile(r"^\s*\*\*([^*]+)\*\*\s*$")
QUESTION_BOLD_LABEL_RE = re.compile(
    r"\*\*\s*"
    r"((?:\d+[.．、]\s*)?(?:题目\d*)?题干|问题\d+|试题\d+|答案|解析|试题解析|备选答案|共用备选答案|选项)"
    r"\s*[：:]?\s*\*\*\s*[：:]?"
)

LEADING_PREFACE_RE = re.compile(
    r"^\s*(?:[#>*\-\s]*)?"
    r"(以下|下面|根据|我将|现将|本次|为便于|按照|依据)"
    r".{0,80}"
    r"(生成|整理|提取|改写|输出|如下|如下所示|说明|呈现|记录|评价)"
    r".*[：:]?\s*$"
)

TRAILING_BOILERPLATE_HEADING_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(临床思维与教学说明|教学说明|备注|记录医生签名|医生签名|日期)"
    r"\s*(?:\*\*)?\s*[：:]?\s*$"
)
TRAILING_BOILERPLATE_LINE_RE = re.compile(
    r"^\s*(本SOAP病历|本记录适用于|如有疑问|若患者有新症状|记录医生签名|医生签名|日期\s*[：:])"
)


@dataclass
class CleanStats:
    model: str
    task: str
    file: str
    changed: bool = False
    suspicious: bool = False
    original_chars: int = 0
    cleaned_chars: int = 0
    removed_fence: int = 0
    removed_separator_lines: int = 0
    removed_leading_preface_lines: int = 0
    removed_trailing_boilerplate_lines: int = 0
    stripped_question_labels: int = 0
    normalized_common_stems: int = 0
    merged_common_stem_markers: int = 0
    inferred_common_stems: int = 0
    normalized_common_option_markers: int = 0
    stripped_question_bullets: int = 0
    split_inline_options: int = 0
    collapsed_blank_runs: int = 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean model outputs under --results-dir into --output-dir.",
    )
    parser.add_argument("--results-dir", default="./results", help="Input results root.")
    parser.add_argument("--output-dir", default="./results-clean", help="Cleaned output root.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing non-empty output directory before writing.",
    )
    parser.add_argument(
        "--max-removal-ratio",
        type=float,
        default=DEFAULT_MAX_REMOVAL_RATIO,
        help="If cleaning removes more than this ratio, keep original text and mark suspicious.",
    )
    return parser


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def split_lines(text: str) -> List[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def strip_outer_fence(lines: List[str], stats: CleanStats) -> List[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1

    if start < end and re.match(r"^\s*```(?:markdown|md|text)?\s*$", lines[start], re.I):
        if end - 1 > start and re.match(r"^\s*```\s*$", lines[end - 1]):
            new_lines = list(lines)
            del new_lines[end - 1]
            del new_lines[start]
            stats.removed_fence += 2
            return new_lines
    return lines


def remove_leading_preface(lines: List[str], stats: CleanStats) -> List[str]:
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped or SEPARATOR_RE.match(stripped) or EMPTY_BULLET_RE.match(stripped):
            idx += 1
            continue
        if LEADING_PREFACE_RE.match(lines[idx]):
            idx += 1
            stats.removed_leading_preface_lines += 1
            continue
        break

    if idx:
        return lines[idx:]
    return lines


def strip_trailing_boilerplate(lines: List[str], stats: CleanStats) -> List[str]:
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1

    cut_start = None
    for i in range(end - 1, -1, -1):
        line = lines[i]
        if TRAILING_BOILERPLATE_HEADING_RE.match(line):
            cut_start = i
            break
        if TRAILING_BOILERPLATE_LINE_RE.match(line):
            cut_start = i

    if cut_start is None:
        return lines

    removed = len(lines[cut_start:])
    stats.removed_trailing_boilerplate_lines += removed
    return lines[:cut_start]


def strip_question_line(line: str, stats: CleanStats) -> str | None:
    bullet_match = QUESTION_BULLET_RE.match(line)
    if bullet_match:
        line = bullet_match.group(1) + bullet_match.group(2)
        stats.stripped_question_bullets += 1

    embedded_common_match = COMMON_STEM_EMBEDDED_LABEL_RE.match(line)
    if embedded_common_match:
        marker = re.sub(r"\s+", "", embedded_common_match.group(1))
        rest = embedded_common_match.group(2).strip()
        stats.stripped_question_labels += 1
        return marker + rest if rest else marker

    numbered_match = NUMBERED_QUESTION_LABEL_RE.match(line)
    if numbered_match:
        indent, number, rest = numbered_match.groups()
        stats.stripped_question_labels += 1
        rest = rest.strip()
        if rest.endswith("**"):
            rest = rest[:-2].rstrip()
        if not rest:
            return indent + f"{number}."
        return indent + f"{number}. {rest}"

    if QUESTION_LABEL_ONLY_RE.match(line):
        stats.stripped_question_labels += 1
        return None

    label_match = QUESTION_LABEL_RE.match(line)
    if not label_match:
        return line

    prefix, _label, rest = label_match.groups()
    indent = re.match(r"\s*", prefix).group(0)
    stats.stripped_question_labels += 1
    rest = rest.strip()
    if rest.startswith("**"):
        rest = rest[2:].lstrip()
    if rest.endswith("**"):
        rest = rest[:-2].rstrip()
    if not rest:
        return None
    return indent + rest


def normalize_common_stem_lines(lines: List[str], stats: CleanStats) -> List[str]:
    normalized: List[str] = []
    i = 0
    while i < len(lines):
        marker_match = COMMON_STEM_MARKER_ONLY_RE.match(lines[i])
        if marker_match:
            marker = re.sub(r"\s+", "", marker_match.group(1))
            next_index = i + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            if next_index < len(lines):
                next_line = strip_question_line(lines[next_index], stats)
                if next_line is not None and next_line.strip():
                    normalized.append(marker + next_line.strip())
                    stats.merged_common_stem_markers += 1
                    i = next_index + 1
                    continue
            normalized.append(marker)
            stats.merged_common_stem_markers += 1
            i += 1
            continue

        options_marker_match = COMMON_OPTIONS_MARKER_ONLY_RE.match(lines[i])
        if options_marker_match:
            normalized.append(re.sub(r"\s+", "", options_marker_match.group(1)))
            stats.normalized_common_option_markers += 1
            i += 1
            continue

        heading_suffix_match = SECTION_HEADING_COMMON_SUFFIX_RE.match(lines[i].strip())
        if heading_suffix_match:
            normalized.append(heading_suffix_match.group(1).rstrip())
            i += 1
            continue

        normalized.append(lines[i])
        i += 1

    for i, line in enumerate(normalized):
        match = COMMON_STEM_LABEL_RE.match(line)
        if not match:
            continue

        stem = match.group(1).strip()
        if stem.endswith("**"):
            stem = stem[:-2].rstrip()

        count = 0
        for next_line in normalized[i + 1 :]:
            if QUESTION_SECTION_HEADING_RE.match(next_line):
                break
            numbered = NUMBERED_QUESTION_LABEL_RE.match(next_line)
            if numbered:
                count = max(count, int(numbered.group(2)))

        prefix = f"(1~{count}题共用题干)" if count > 0 else ""
        normalized[i] = f"{prefix}{stem}" if stem else prefix
        stats.normalized_common_stems += 1

    for i, line in enumerate(normalized):
        if not A3_A4_SECTION_HEADING_RE.match(line):
            continue

        stem_index = i + 1
        while stem_index < len(normalized) and not normalized[stem_index].strip():
            stem_index += 1
        if stem_index >= len(normalized):
            continue

        stem = normalized[stem_index].strip()
        if not stem:
            continue
        cleaned_stem = strip_question_line(stem, stats)
        if cleaned_stem is None:
            continue
        stem = cleaned_stem.strip()
        if "共用题干" in stem or stem.startswith(("A.", "B.", "C.", "D.", "E.", "答案", "解析")):
            continue
        if re.match(r"^\d+[.．、]\s+", stem):
            continue

        count = 0
        for next_line in normalized[stem_index + 1 :]:
            if QUESTION_SECTION_HEADING_RE.match(next_line):
                break
            numbered = re.match(r"^\s*(?:\*\*)?\s*(\d+)[.．、]\s+", next_line)
            if numbered:
                count = max(count, int(numbered.group(1)))

        if count >= 2:
            normalized[stem_index] = f"(1~{count}题共用题干){stem}"
            stats.inferred_common_stems += 1

    return normalized


def split_inline_option_line(line: str, stats: CleanStats) -> List[str]:
    """Split A-E options that were emitted on one line, preserving option text."""
    matches = list(INLINE_OPTION_MARK_RE.finditer(line))
    labels = [m.group(1) for m in matches]
    if labels != ["A", "B", "C", "D", "E"]:
        return [line]

    parts: List[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        parts.append(line[start:end].strip())

    stats.split_inline_options += 1
    return parts


def normalize_common_stem_artifacts(line: str, stats: CleanStats) -> str:
    match = COMMON_STEM_PREFIX_RE.match(line)
    if not match:
        return line

    marker = re.sub(r"\s+", "", match.group(1))
    rest = match.group(2).strip()
    original_rest = rest

    rest = re.sub(r"^(?:\*\*)?\s*题干\s*(?:\*\*)?\s*[：:]?\s*", "", rest).strip()
    if rest == "**":
        rest = ""
    if rest.startswith("**") and rest.endswith("**") and len(rest) > 4:
        rest = rest[2:-2].strip()
    elif rest.endswith("**") and not rest.startswith("**"):
        rest = rest[:-2].rstrip()
    elif rest.startswith("**") and not rest.endswith("**"):
        rest = rest[2:].lstrip()

    if rest != original_rest:
        stats.stripped_question_labels += 1
    return marker + rest


def normalize_question_bold_artifacts(line: str, stats: CleanStats) -> str:
    original = line

    malformed_heading = re.match(
        r"^(\s*)\*\*((?:[一二三四五]、)?[A-Z][1-9]型题(?:\s*[^*]*)?)\s*$",
        line,
    )
    if malformed_heading:
        fixed = f"{malformed_heading.group(1)}**{malformed_heading.group(2).strip()}**"
        stats.stripped_question_labels += 1
        return fixed

    wrapped = QUESTION_BOLD_WRAPPED_LINE_RE.match(line)
    if wrapped:
        inner = wrapped.group(1).strip()
        if re.search(r"(?:[一二三四五]、)?[A-Z][1-9]型题\b", inner):
            return original
        if inner in {
            "备选答案",
            "共用备选答案",
        }:
            line = inner

    line = QUESTION_BOLD_LABEL_RE.sub(lambda m: m.group(1).strip() + "：", line)
    line = re.sub(r"^(\s*\d+[.．、]\s*)\*\*\s+", r"\1", line)

    if line != original:
        stats.stripped_question_labels += 1
    return line


def clean_lines(lines: List[str], task: str, stats: CleanStats) -> List[str]:
    cleaned: List[str] = []
    is_question_task = task in QUESTION_TASK_NAMES
    if is_question_task:
        lines = normalize_common_stem_lines(lines, stats)

    for line in lines:
        if SEPARATOR_RE.match(line):
            stats.removed_separator_lines += 1
            continue
        if EMPTY_BULLET_RE.match(line):
            stats.removed_separator_lines += 1
            continue

        if is_question_task:
            line = normalize_question_bold_artifacts(line, stats)
            stripped = strip_question_line(line, stats)
            if stripped is None:
                continue
            stripped = normalize_common_stem_artifacts(stripped, stats)
            stripped = normalize_question_bold_artifacts(stripped, stats)
            for option_line in split_inline_option_line(stripped, stats):
                cleaned.append(option_line.rstrip())
            continue

        cleaned.append(line.rstrip())

    return cleaned


def collapse_blank_runs(lines: List[str], stats: CleanStats) -> List[str]:
    result: List[str] = []
    blank_run = 0
    for line in lines:
        if line.strip():
            blank_run = 0
            result.append(line)
            continue
        blank_run += 1
        if blank_run <= 2:
            result.append("")
        else:
            stats.collapsed_blank_runs += 1
    return result


def clean_text(text: str, model: str, task: str, relative_file: str) -> Tuple[str, CleanStats]:
    stats = CleanStats(
        model=model,
        task=task,
        file=relative_file,
        original_chars=len(text),
    )
    lines = split_lines(text)
    lines = strip_outer_fence(lines, stats)
    lines = remove_leading_preface(lines, stats)
    lines = clean_lines(lines, task, stats)
    lines = strip_trailing_boilerplate(lines, stats)
    lines = collapse_blank_runs(lines, stats)

    cleaned = "\n".join(lines).strip() + "\n"
    stats.cleaned_chars = len(cleaned)
    stats.changed = cleaned != text
    return cleaned, stats


def ensure_output_dir(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"输出目录已存在且非空：{output_root}\n"
                "请改用新的 --output-dir，或显式传入 --overwrite 重建。"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def write_report(output_root: Path, rows: Iterable[CleanStats]) -> None:
    report_path = output_root / "clean_report.csv"
    fieldnames = list(asdict(CleanStats(model="", task="", file="")).keys())
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run(args: argparse.Namespace) -> None:
    results_root = resolve_path(args.results_dir)
    output_root = resolve_path(args.output_dir)
    if args.max_removal_ratio < 0 or args.max_removal_ratio >= 1:
        raise ValueError("--max-removal-ratio 必须在 [0, 1) 范围内")

    model_dirs = discover_model_dirs(results_root)
    if not model_dirs:
        raise RuntimeError(f"在 {results_root} 下未发现任何模型结果目录")

    ensure_output_dir(output_root, args.overwrite)

    all_stats: List[CleanStats] = []
    copied_files = 0
    suspicious_files = 0

    for model_dir, model_label in model_dirs:
        source_model_root = results_root / model_dir
        for task_dir in sorted(source_model_root.iterdir()):
            if not task_dir.is_dir():
                continue
            for source_file in iter_output_files(task_dir):
                rel = source_file.relative_to(results_root)
                target_file = output_root / rel
                target_file.parent.mkdir(parents=True, exist_ok=True)

                original = source_file.read_text(encoding="utf-8")
                cleaned, stats = clean_text(
                    original,
                    model=model_label,
                    task=task_dir.name,
                    relative_file=str(rel).replace("\\", "/"),
                )
                removed_ratio = (
                    (stats.original_chars - stats.cleaned_chars) / stats.original_chars
                    if stats.original_chars > 0
                    else 0.0
                )
                if removed_ratio > args.max_removal_ratio:
                    stats.suspicious = True
                    suspicious_files += 1
                    target_file.write_text(original, encoding="utf-8", newline="\n")
                else:
                    target_file.write_text(cleaned, encoding="utf-8", newline="\n")

                all_stats.append(stats)
                copied_files += 1

    write_report(output_root, all_stats)
    changed_files = sum(1 for row in all_stats if row.changed)
    print(f"[完成] 发现模型目录: {len(model_dirs)}")
    print(f"[完成] 写入文件: {copied_files}")
    print(f"[完成] 发生清洗: {changed_files}")
    print(f"[完成] 可疑保护: {suspicious_files}")
    print(f"[完成] 输出目录: {output_root}")
    print(f"[完成] 清洗报告: {output_root / 'clean_report.csv'}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"[错误] {exc}\n")


if __name__ == "__main__":
    sys.exit(main())

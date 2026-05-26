# -*- coding: utf-8 -*-
"""对 dpo_pairs.jsonl 中 chosen 做格式清洗。
去除内部标注、审校备注、知识点映射等非题目内容。

用法：python clean_dpo_chosen.py
输出覆盖 data/dpo_pairs.jsonl，备份保存在 data/dpo_pairs.jsonl.bak2
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_chosen(text: str) -> str:
    """对 chosen 做格式清洗，去除内部标注类内容。"""
    lines = text.split("\n")
    cleaned: List[str] = []
    skip_block = False

    for line in lines:
        stripped = line.strip()

        # 跳过整块 ✅ 标注
        if stripped.startswith("✅ **"):
            skip_block = True
            continue
        if skip_block:
            if stripped == "" or stripped.startswith("- **"):
                continue
            skip_block = False

        # 跳过 > ✅ 引用
        if stripped.startswith("> ✅"):
            continue

        # 跳过 **知识点大纲：**
        if stripped.startswith("**知识点大纲" + "：**") or stripped.startswith("**知识点大纲" + ":**"):
            continue

        # **考题内容：** → 去标签，留内容
        m = re.match(r"^\*\*考题内容[：:]\*\*\s*(.*)", stripped)
        if m:
            rest = m.group(1).strip()
            if rest:
                cleaned.append(rest)
            continue

        cleaned.append(line)

    result = "\n".join(cleaned)
    # 去掉审校类【】备注
    result = re.sub(r"【[^】]*?(?:题干不符合|不对的|审校|不应该|建议).*?】", "", result)
    return result.strip()


def clean_all(rows: List[Dict[str, Any]]) -> int:
    cleaned_count = 0
    for row in rows:
        old = row.get("chosen", "")
        new = clean_chosen(old)
        if old != new:
            row["chosen"] = new
            cleaned_count += 1
    return cleaned_count


def main() -> int:
    path = "data/dpo_pairs.jsonl"
    if not os.path.exists(path):
        print(f"[ERROR] 文件不存在: {path}", file=sys.stderr)
        return 1

    rows = load_jsonl(path)

    # 备份
    bak = path + ".bak2"
    dump_jsonl(bak, rows)
    print(f"[BACKUP] {bak}")

    count = clean_all(rows)
    dump_jsonl(path, rows)

    print(f"[CLEAN] 修改了 {count} 条 chosen")
    print(f"[OUTPUT] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

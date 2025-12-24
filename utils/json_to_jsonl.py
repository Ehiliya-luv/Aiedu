#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将标准 JSON 文件（数组格式）转换为 JSONL（JSON Lines）格式。
输入：xxx.json   →   输出：xxx.jsonl
原始 JSON 文件保留不变。
"""

import json
import os
import sys
from pathlib import Path

def convert_json_to_jsonl(json_path: str):
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"❌ 错误：文件不存在 → {json_path}", file=sys.stderr)
        sys.exit(1) 

    if json_path.suffix.lower() != '.json':
        print(f"⚠️  警告：文件不是 .json 后缀，但仍尝试转换 → {json_path}")

    jsonl_path = json_path.with_suffix('.jsonl')
    print(f"🔄 正在读取: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print("❌ 错误：JSON 根元素必须是一个数组（列表）", file=sys.stderr)
        sys.exit(1)

    print(f"✅ 成功加载 {len(data)} 条记录")
    print(f"💾 正在写入 JSONL 文件: {jsonl_path}")

    try:
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    print(f"⚠️  警告：第 {idx+1} 条记录不是对象（dict），跳过", file=sys.stderr)
                    continue
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"❌ 写入失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"🎉 转换完成！JSONL 文件已保存至: {jsonl_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python json_to_jsonl.py <输入.json>")
        print("示例: python json_to_jsonl.py data/rl_train.json")
        sys.exit(1)

    input_json = sys.argv[1]
    convert_json_to_jsonl(input_json)
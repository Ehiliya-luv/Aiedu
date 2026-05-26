# -*- coding: utf-8 -*-
"""I/O and prompt helpers for generate_output.py."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .prompts import PLACEHOLDER_PATTERNS
from ..text import strip_think_content
def load_medical_records_from_dir(
    dir_path: str,
    sample_size: int = 0,
    print_debug: bool = False,
) -> List[Dict[str, str]]:
    """从目录加载病历文件。

    Args:
        dir_path:   病历数据目录
        sample_size: >0 时仅取前 N 条；0 表示全部
        print_debug: 是否输出调试信息
    """
    data_list: List[Dict[str, str]] = []
    if not os.path.exists(dir_path):
        print(f"[ERROR] 数据目录不存在：{dir_path}")
        return data_list

    files = sorted(
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    )
    if print_debug:
        print(f"[INFO] 在目录 {dir_path} 中找到 {len(files)} 个文件")

    for filename in files:
        file_path = os.path.join(dir_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    data_list.append({"id": filename, "input_context": content})
        except Exception as e:
            if print_debug:
                print(f"[WARN] 读取文件失败 {filename}: {e}")
            continue

    if sample_size > 0 and data_list:
        data_list = data_list[:sample_size]

    return data_list


def save_result(content: str, output_dir: str, filename_prefix: str, _task_name: str) -> None:
    """保存结果到指定目录（自动清理 think 标签）。"""
    content = strip_think_content(content)
    safe_prefix = "".join(c for c in filename_prefix if c.isalnum() or c in (".", "_", "-"))
    file_path = os.path.join(output_dir, f"{safe_prefix}.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def build_prompt(prompt_template: str, input_text: str, context_text: str = "") -> Optional[str]:
    """通用提示词构建函数。"""
    if not prompt_template or "【请在此处粘贴" in prompt_template:
        return None

    prompt = prompt_template
    for placeholder in PLACEHOLDER_PATTERNS[:3]:
        if isinstance(placeholder, tuple):
            key, _ = placeholder
            prompt = prompt.replace(key, input_text)
        else:
            prompt = prompt.replace(placeholder, input_text)

    if "{{#context#}}" in prompt:
        if context_text:
            prompt = prompt.replace("{{#context#}}", context_text)
        else:
            lines = [line for line in prompt.split("\n") if "{{#context#}}" not in line]
            prompt = "\n".join(lines).replace("\n\n\n", "\n\n")

    return prompt


def init_directories(root_dir: str) -> Tuple[str, Dict[int, str]]:
    """初始化结果输出目录，返回 (root_dir, {task_id: task_dir})。"""
    os.makedirs(root_dir, exist_ok=True)
    task_dirs = {
        1: os.path.join(root_dir, "病历标准化"),
        2: os.path.join(root_dir, "考题生成"),
        3: os.path.join(root_dir, "临床思维"),
        4: os.path.join(root_dir, "病历综合评分"),
    }
    for d in task_dirs.values():
        os.makedirs(d, exist_ok=True)
    return root_dir, task_dirs

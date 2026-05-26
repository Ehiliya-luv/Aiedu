# -*- coding: utf-8 -*-
"""数据加载模块：病历加载、模型结果自动发现、picked 模式交互。

判定规则：results-dir 下的子目录如果包含至少一个已知任务子目录，
则被认为是模型结果目录。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .prompts import DEFAULT_TASK_ORDER, resolve_task_name


# 已知任务子目录名（用于判定模型结果目录）
_KNOWN_TASK_NAMES: Set[str] = {
    "病历标准化", "考题生成", "临床思维", "病历综合评分",
    "SOAP", "题目生成", "临床思维提取", "病历评分",
}


# ================= 病历加载 =================

def load_raw_cases(case_dir: Path) -> Dict[str, str]:
    """加载原始病历数据。

    Args:
        case_dir: 病历目录路径

    Returns:
        {case_id: 病历文本}

    Raises:
        FileNotFoundError: 目录不存在
        RuntimeError: 目录下无 .txt 文件
    """
    if not case_dir.exists():
        raise FileNotFoundError(f"原始病历目录不存在：{case_dir}")

    cases: Dict[str, str] = {}
    for path in sorted(case_dir.glob("*.txt")):
        cases[path.stem] = path.read_text(encoding="utf-8").strip()

    if not cases:
        raise RuntimeError(f"原始病历目录中未找到 .txt 文件：{case_dir}")

    print(f"[数据加载] 已从 {case_dir} 加载 {len(cases)} 份原始病历")
    return cases


def resolve_case_dir(configured: Optional[str], project_root: Path) -> Path:
    """解析原始病历目录。

    优先级：用户指定 > ./data/病历 > ./data/病例 > ./data-backup/病例
    """
    if configured:
        p = Path(configured)
        return p if p.is_absolute() else project_root / p

    for candidate in ["data/病历", "data/病例", "data-backup/病例"]:
        full = project_root / candidate
        if full.exists():
            return full

    return project_root / "data" / "病历"


# ================= 模型输出加载 =================

def case_id_from_output_file(path: Path) -> str:
    """从输出文件名提取 case_id。

    优先匹配 .txt.md → .md → .txt 后缀。
    """
    name = path.name
    if name.endswith(".txt.md"):
        return name[: -len(".txt.md")]
    if name.endswith(".md"):
        stem = name[: -len(".md")]
        if stem.endswith(".txt"):
            return stem[: -len(".txt")]
        return stem
    if name.endswith(".txt"):
        return name[: -len(".txt")]
    return path.stem


def iter_output_files(task_dir: Path):
    """迭代任务目录下的输出文件。"""
    for path in sorted(task_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            yield path


def load_task_outputs(
    results_root: Path,
    task: str,
    model_labels: Sequence[Tuple[str, str]],
    raw_cases: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    """加载指定任务的模型输出数据。

    Args:
        results_root: 结果根目录
        task:         任务名
        model_labels: [(dir_name, display_label), ...]
        raw_cases:    原始病历 {case_id: text}

    Returns:
        {display_label: {case_id: output_text}}
    """
    outputs: Dict[str, Dict[str, str]] = {label: {} for _, label in model_labels}

    for directory, label in model_labels:
        task_dir = results_root / directory / task
        if not task_dir.exists():
            print(f"[数据加载] 模型目录缺少任务输出：{task_dir}")
            continue

        for path in iter_output_files(task_dir):
            case_id = case_id_from_output_file(path)
            if case_id in raw_cases:
                outputs[label][case_id] = path.read_text(encoding="utf-8").strip()

        print(f"[数据加载] {label} / {task}: {len(outputs[label])} 条可匹配输出")

    return outputs


# ================= 模型结果自动发现 =================

def discover_model_dirs(results_root: Path) -> List[Tuple[str, str]]:
    """自动发现 results_root 下所有模型结果目录。

    判定规则：子目录包含至少一个已知任务子目录名。

    Returns:
        [(dir_name, display_label), ...] 按目录名排序
    """
    if not results_root.exists():
        raise FileNotFoundError(f"结果根目录不存在：{results_root}")

    found = []
    for entry in sorted(results_root.iterdir()):
        if not entry.is_dir():
            continue
        subdirs = {sub.name for sub in entry.iterdir() if sub.is_dir()}
        if subdirs & _KNOWN_TASK_NAMES:
            found.append((entry.name, entry.name))

    return found


def get_model_task_stats(
    results_root: Path,
    dir_name: str,
) -> Dict[str, int]:
    """获取一个模型结果目录下各任务的输出文件数。

    Returns:
        {task_name: file_count}
    """
    stats = {}
    model_dir = results_root / dir_name
    if not model_dir.exists():
        return stats

    for entry in sorted(model_dir.iterdir()):
        if entry.is_dir() and entry.name in _KNOWN_TASK_NAMES:
            count = sum(1 for _ in iter_output_files(entry))
            stats[entry.name] = count

    return stats


def available_common_tasks(
    results_root: Path,
    model_labels: Sequence[Tuple[str, str]],
) -> List[str]:
    """计算所有选中模型的共同任务（按 DEFAULT_TASK_ORDER 排序）。"""
    task_sets = []
    for directory, _ in model_labels:
        model_dir = results_root / directory
        if not model_dir.exists():
            raise FileNotFoundError(f"模型结果目录不存在：{model_dir}")
        task_sets.append({sub.name for sub in model_dir.iterdir() if sub.is_dir()})

    common = set.intersection(*task_sets) if task_sets else set()
    ordered = [t for t in DEFAULT_TASK_ORDER if resolve_task_name(t) in {resolve_task_name(c) for c in common}]
    remaining = sorted(common - set(ordered))
    ordered.extend(remaining)
    return ordered


def parse_tasks_arg(value: str, available_tasks: List[str]) -> List[str]:
    """解析 --tasks 参数。

    auto → 按 DEFAULT_TASK_ORDER 中存在的共同任务
    all  → 全部共同任务
    逗号分隔 → 指定任务名
    """
    if not value or value.lower() == "auto":
        return [t for t in available_tasks if t in available_tasks]
    if value.lower() == "all":
        return list(available_tasks)

    requested = [item.strip() for item in value.split(",") if item.strip()]
    resolved = [resolve_task_name(t) for t in requested]
    available_resolved = {resolve_task_name(t) for t in available_tasks}
    missing = [t for t in resolved if t not in available_resolved]
    if missing:
        raise ValueError(f"请求的任务在 results 中不存在：{missing}；可用任务：{available_tasks}")
    return [t for t in available_tasks if resolve_task_name(t) in set(resolved)]


# ================= Picked 模式交互 =================

def interactive_pick_models(
    results_root: Path,
    model_dirs: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """交互式选择要比较的模型。

    输出简要统计 → 等待序号输入 → rm/del 撤销 → -1 结束。

    Returns:
        选中模型的 [(dir_name, display_label), ...]
    """
    print(f"\n{'='*70}")
    print("发现以下模型结果目录：")
    print(f"{'='*70}")

    all_stats = []
    for i, (dir_name, _) in enumerate(model_dirs, 1):
        stats = get_model_task_stats(results_root, dir_name)
        all_stats.append(stats)

    all_task_names = []
    seen = set()
    for stats in all_stats:
        for t in stats:
            resolved = resolve_task_name(t)
            if resolved not in seen:
                all_task_names.append(t)
                seen.add(resolved)

    for i, (dir_name, _) in enumerate(model_dirs, 1):
        stats = all_stats[i - 1]
        parts = "  ".join(f"{t}:{stats.get(t, 0)}" for t in all_task_names if t in stats)
        print(f"  [{i}] {dir_name:<30s} {parts}")

    print(f"{'='*70}")
    print("输入序号选择模型 | rm/del <序号> 撤销 | -1 结束选择")
    print(f"至少需要选择 2 个模型才能进行成对比较")

    selected_indices: List[int] = []

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        if raw == "-1":
            break

        if raw.lower().startswith(("rm ", "del ")):
            try:
                idx = int(raw.split()[1]) - 1
                if idx in selected_indices:
                    selected_indices.remove(idx)
                    dir_name = model_dirs[idx][0]
                    print(f"  已移除: {dir_name}")
                    _print_selected(model_dirs, selected_indices)
                else:
                    print(f"  序号 {idx+1} 未被选中")
            except (ValueError, IndexError):
                print(f"  无效输入: {raw}")
            continue

        try:
            idx = int(raw) - 1
            if idx < 0 or idx >= len(model_dirs):
                print(f"  序号超出范围 (1-{len(model_dirs)})")
                continue
            if idx in selected_indices:
                print(f"  序号 {idx+1} 已选中")
                continue
            selected_indices.append(idx)
            dir_name = model_dirs[idx][0]
            print(f"  已选择: {dir_name}")
            _print_selected(model_dirs, selected_indices)
        except ValueError:
            print(f"  无效输入: {raw}")

    if len(selected_indices) < 2:
        raise RuntimeError(f"至少需要选择 2 个模型，当前选了 {len(selected_indices)} 个")

    return [model_dirs[i] for i in selected_indices]


def _print_selected(model_dirs: List[Tuple[str, str]], indices: List[int]) -> None:
    """打印当前已选模型列表。"""
    names = [model_dirs[i][0] for i in indices]
    print(f"  已选 ({len(indices)}): {', '.join(names)}")


# ================= 数据校验 =================

def validate_task_data(task: str, task_data: Dict) -> int:
    """校验任务数据是否满足比较条件。

    Returns:
        共同病历数（0 表示不可用）
    """
    model_outputs = task_data["model_outputs"]
    non_empty = [set(outputs.keys()) for outputs in model_outputs.values() if outputs]

    if len(non_empty) < 2:
        print(f"[跳过] {task}: 少于两个模型有可用输出")
        return 0

    common = sorted(set.intersection(*non_empty))
    if not common:
        print(f"[跳过] {task}: 模型之间没有共同病历输出")
        return 0

    print(f"[数据检查] {task}: {len(common)} 份共同病历可用于比较")
    return len(common)


# ================= Checkpoint 元数据 =================

def save_meta(
    meta_path: str,
    models: List[str],
    tasks: List[str],
    repeats: int,
    judge_model: str,
) -> None:
    """保存实验元数据，用于续跑时参数一致性检查。"""
    meta = {
        "models": sorted(models),
        "tasks": sorted(tasks),
        "repeats": repeats,
        "judge_model": judge_model,
    }
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta(meta_path: str) -> Optional[dict]:
    """加载 checkpoint 元数据；不存在时返回 None。"""
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_meta_consistency(
    meta_path: str,
    models: List[str],
    tasks: List[str],
    repeats: int,
    judge_model: str,
) -> None:
    """检查当前参数与已有 checkpoint 元数据是否一致。

    Raises:
        RuntimeError: 参数不一致
    """
    if not os.path.exists(meta_path):
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    errors = []
    if sorted(saved.get("models", [])) != sorted(models):
        errors.append(f"models: 保存={saved.get('models')} vs 当前={sorted(models)}")
    if sorted(saved.get("tasks", [])) != sorted(tasks):
        errors.append(f"tasks: 保存={saved.get('tasks')} vs 当前={sorted(tasks)}")
    if saved.get("repeats") != repeats:
        errors.append(f"repeats: 保存={saved.get('repeats')} vs 当前={repeats}")
    if saved.get("judge_model") != judge_model:
        errors.append(f"judge_model: 保存={saved.get('judge_model')} vs 当前={judge_model}")

    if errors:
        raise RuntimeError(
            "当前参数与已有 checkpoint 不一致，请使用新的 --output-dir 或清空旧目录：\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

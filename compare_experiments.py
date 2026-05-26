# -*- coding: utf-8 -*-
"""LLM Judge 比较实验主入口。

合并原 experiments.py 和 compare_results_before_after.py，
统一成对比较 + Bradley-Terry + Bootstrap + Holm 校正流程。

模型选择模式：
  --select all    自动比较 results-dir 下所有发现的模型
  --select picked 控制台交互选择（默认）

统计方法依赖：
  - Bradley-Terry 模型：choix
  - Bootstrap 置信区间：numpy 重采样 + choix 拟合
  - 多重检验校正：statsmodels (Holm-Bonferroni)
  - 可视化：matplotlib
"""

import argparse
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.com_exp.data_loader import (
    available_common_tasks,
    check_meta_consistency,
    discover_model_dirs,
    interactive_pick_models,
    load_meta,
    load_raw_cases,
    load_task_outputs,
    parse_tasks_arg,
    resolve_case_dir,
    save_meta,
    validate_task_data,
)
from utils.com_exp.judge_api import (
    OpenAICompatibleClient,
    call_judge,
    run_pairwise_evaluation,
)
from utils.com_exp.prompts import get_judge_prompt, resolve_task_name
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HUATUO_JUDGE_MODEL = os.getenv(
    "JUDGE_VLLM_MODEL_PATH",
    "HuatuoGPT-o1-8B",
)
DEFAULT_ANCHOR_MODELS = ["base_model_baichuan-full"]


# ================= 结果保存 =================

def save_results(
    task: str,
    boot_results: pd.DataFrame,
    sig_results: pd.DataFrame,
    output_path: str,
    figure_path: str,
) -> None:
    """输出统计评价报告与可视化。"""
    from utils.com_exp.visualization import (
        plot_anchor_score_bar,
        plot_ranking_with_ci,
        plot_significance_heatmap,
    )

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    ranking_file = os.path.join(output_path, f"{task}_ranking.csv")
    boot_results.sort_values("BT_Score", ascending=False).to_csv(ranking_file)

    sig_file = os.path.join(output_path, f"{task}_significance.csv")
    sig_results.to_csv(sig_file, index=False)

    plot_ranking_with_ci(boot_results, task, figure_path)
    if {"Anchor_Score", "Anchor_CI_Lower", "Anchor_CI_Upper"}.issubset(boot_results.columns):
        plot_anchor_score_bar(boot_results, task, figure_path)
    plot_significance_heatmap(sig_results, boot_results, task, figure_path)

    print(f"\n[报告输出] 任务 {task} 排名摘要：")
    for rank, (model, row) in enumerate(
        boot_results.sort_values("BT_Score", ascending=False).iterrows(), 1
    ):
        ci = f"[{row['BT_CI_Lower']:.3f}, {row['BT_CI_Upper']:.3f}]"
        line = f"  {rank}. {model}: BT_Score={row['BT_Score']:.3f} {ci}"
        if "Anchor_Score" in row:
            anchor_ci = f"[{row['Anchor_CI_Lower']:.3f}, {row['Anchor_CI_Upper']:.3f}]"
            line += f" | Anchor_Score={row['Anchor_Score']:.3f} {anchor_ci}"
        print(line)

    sig_count = sig_results["Significant"].sum() if not sig_results.empty else 0
    total_pairs = len(sig_results) if not sig_results.empty else 0
    print(f"  显著差异模型对：{sig_count}/{total_pairs}")
    print(f"  结果已保存至 {output_path}\n")


# ================= 主执行流程 =================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM Judge 比较实验：成对比较 + Bradley-Terry + Bootstrap + Holm 校正",
    )

    # --- 数据源 ---
    g = p.add_argument_group("数据源")
    g.add_argument("--results-dir", default="./results",
                   help="结果根目录 (default: ./results)")
    g.add_argument("--case-dir", default=None,
                   help="原始病历目录 (default: 自动探测)")

    # --- 模型选择 ---
    g = p.add_argument_group("模型选择")
    g.add_argument("--select", choices=["all", "picked"], default="picked",
                   help="模型选择模式：all=全部，picked=交互选择 (default: picked)")

    # --- 任务 ---
    g = p.add_argument_group("任务")
    g.add_argument("--tasks", default="auto",
                   help="任务选择：auto|all|逗号分隔 (default: auto)")

    # --- Judge backend ---
    g = p.add_argument_group("Judge backend")
    g.add_argument("--judge-backend", choices=["api", "vllm"], default="api",
                   help="Judge 后端：api=远程 OpenAI 兼容 API，vllm=本地/自管 vLLM server (default: api)")
    g.add_argument("--judge-model", default="deepseek/deepseek-v3.2",
                   help="Judge 模型名；--judge-backend=vllm 且未显式传入时默认使用 Huatuo (default: deepseek/deepseek-v3.2)")
    g.add_argument("--judge-api-base", default="https://api.ppio.com/openai",
                   help="远程 Judge API base URL (default: https://api.ppio.com/openai)")
    g.add_argument("--judge-api-key", default=os.getenv("JUDGE_API_KEY", ""),
                   help="Judge API key (default: 环境变量 JUDGE_API_KEY)")
    g.add_argument("--judge-temperature", type=float, default=0.0,
                   help="Judge 采样 temperature (default: 0.0)")
    g.add_argument("--judge-top-p", type=float, default=1.0,
                   help="Judge 采样 top_p (default: 1.0)")
    g.add_argument("--judge-max-tokens", type=int, default=2048,
                   help="Judge 最大输出 token 数 (default: 2048)")
    g.add_argument("--judge-workers", type=int, default=4,
                   help="Judge API 并发线程数 (default: 4)")
    g.add_argument("--judge-response-format", choices=["auto", "json_object", "none"], default="auto",
                   help="Judge 响应格式约束；auto=API 使用 json_object，vLLM 不强制 (default: auto)")
    g.add_argument("--judge-enable-thinking", action="store_true", default=False,
                   help="启用 Judge 模型 thinking/reasoning 输出 (default: False)。"
                        "本流程只需要 winner 字段，关闭 thinking 可显著降低延迟与 token 消耗；"
                        "vLLM backend 下关闭时通过 chat_template_kwargs.enable_thinking=False 触发模板的"
                        "skip-thinking 分支，逻辑参考 utils/judge/vllm_scorer.py 的三分支策略。")

    # --- Judge vLLM server ---
    g = p.add_argument_group("Judge vLLM server")
    g.add_argument("--judge-vllm-base-url", default="http://127.0.0.1:8000",
                   help="Judge vLLM OpenAI server 地址 (default: http://127.0.0.1:8000)")
    g.add_argument("--judge-vllm-api-key", default=os.getenv("JUDGE_VLLM_API_KEY", "EMPTY"),
                   help="Judge vLLM API key (default: 环境变量 JUDGE_VLLM_API_KEY 或 EMPTY)")
    g.add_argument("--judge-vllm-model-path", default=DEFAULT_HUATUO_JUDGE_MODEL,
                   help=f"自动启动 vLLM 时加载的模型路径或 HuggingFace ID (default: {DEFAULT_HUATUO_JUDGE_MODEL})")
    g.add_argument("--judge-vllm-model-name", default=None,
                   help="请求 vLLM OpenAI server 时使用的 model 名称 (default: 与 --judge-vllm-model-path 相同)")
    g.add_argument("--judge-vllm-auto-start", action="store_true", default=True,
                   help="--judge-backend=vllm 时自动启动 vLLM OpenAI server (default: True)")
    g.add_argument("--no-judge-vllm-auto-start", action="store_false", dest="judge_vllm_auto_start",
                   help="不自动启动 vLLM server，连接已有 server")
    g.add_argument("--judge-vllm-server-tp", type=int, default=None,
                   help="vLLM server tensor-parallel-size (default: 自动检测 GPU 数)")
    g.add_argument("--judge-vllm-server-gpu-mem-util", type=float, default=0.9,
                   help="vLLM server gpu-memory-utilization (default: 0.9)")
    g.add_argument("--judge-vllm-server-max-model-len", type=int, default=None,
                   help="vLLM server max-model-len (default: 由 vLLM 自动决定)")
    g.add_argument("--judge-vllm-server-dtype", default="auto",
                   help="vLLM server dtype (default: auto)")
    g.add_argument("--judge-vllm-server-trust-remote-code", action="store_true", default=True,
                   help="vLLM server trust-remote-code (default: True)")
    g.add_argument("--no-judge-vllm-server-trust-remote-code", action="store_false",
                   dest="judge_vllm_server_trust_remote_code",
                   help="vLLM server 不启用 trust-remote-code")
    g.add_argument("--judge-vllm-server-log-file", default="./tmp/judge_vllm_server.log",
                   help="vLLM server 日志文件路径 (default: ./tmp/judge_vllm_server.log)")

    # --- 统计 ---
    g = p.add_argument_group("统计")
    g.add_argument("--repeats", type=int, default=3,
                   help="每对每病历重复投票次数 (default: 3)")
    g.add_argument("--bootstrap", type=int, default=10000,
                   help="Bootstrap 迭代次数 (default: 10000)")
    g.add_argument("--significance", type=float, default=0.05,
                   help="显著性水平 (default: 0.05)")
    g.add_argument("--anchor-models", action="append", default=None,
                   help="Anchor Score 的基准模型，可逗号分隔或重复传参 (default: base_model_baichuan-full)")

    # --- 输出 ---
    g = p.add_argument_group("输出")
    g.add_argument("--output-dir", default="./results/LLM_Judge_compare",
                   help="评价结果输出目录 (default: ./results/LLM_Judge_compare)")
    g.add_argument("--resume", action="store_true",
                   help="从 output-dir/checkpoints/_meta.json 恢复模型与任务选择，跳过交互 pick (default: False)")
    g.add_argument("--limit-cases", type=int, default=None,
                   help="仅用前 N 份病历做快速测试 (default: 全部)")
    g.add_argument("--dry-run", action="store_true",
                   help="只检查数据不调用 Judge API")

    return p


def _arg_was_passed(option_name: str) -> bool:
    return any(arg == option_name or arg.startswith(f"{option_name}=") for arg in sys.argv)


def resolve_judge_model(args: argparse.Namespace) -> str:
    """根据后端解析 Judge 模型名，避免 API 默认值污染 vLLM 默认模型。"""
    if args.judge_backend == "vllm" and not _arg_was_passed("--judge-model"):
        return args.judge_vllm_model_name or args.judge_vllm_model_path
    return args.judge_model


def resolve_judge_response_format(args: argparse.Namespace) -> Optional[dict]:
    if args.judge_response_format == "json_object":
        return {"type": "json_object"}
    if args.judge_response_format == "none":
        return None
    if args.judge_backend == "api":
        return {"type": "json_object"}
    return None


def resolve_judge_extra_body(args: argparse.Namespace) -> Optional[dict]:
    """根据 --judge-enable-thinking 构造 OpenAI SDK 的 extra_body 透传字段。

    Judge 流程只解析 ``{"winner": ...}`` JSON，根本不需要 reasoning。默认关闭
    thinking 可以节省大量 token 与延迟；用户显式传 --judge-enable-thinking
    时才打开。

    vLLM backend 通过 ``chat_template_kwargs.enable_thinking=False`` 触发 chat
    template 内的 skip-thinking 分支（Qwen3 / DeepSeek-R1 / Baichuan-M2 等），
    实现细节参考 ``utils/judge/vllm_scorer.py::_detect_thinking_strategy``。
    远程 API（ppio 等）通常默认不带 thinking，传 enable_thinking=False 也是
    安全的 no-op，所以两个 backend 都附加该字段，统一行为。
    """
    if args.judge_enable_thinking:
        return None
    return {"chat_template_kwargs": {"enable_thinking": False}}


def resolve_anchor_models(raw_anchor_args: Optional[List[str]]) -> List[str]:
    """解析 --anchor-models，支持逗号分隔和重复传参。"""
    if not raw_anchor_args:
        return DEFAULT_ANCHOR_MODELS.copy()

    anchors = []
    for raw in raw_anchor_args:
        for item in raw.split(","):
            model = item.strip()
            if model and model not in anchors:
                anchors.append(model)

    if not anchors:
        raise ValueError("--anchor-models 至少需要包含一个模型名")
    return anchors


def resolve_resume_selection(
    meta_path: str,
    model_dirs: List[Tuple[str, str]],
    results_root: Path,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """从 checkpoint meta 恢复模型与任务选择，避免 resume 时再次交互 pick。"""
    meta = load_meta(meta_path)
    if meta is None:
        raise RuntimeError(
            f"--resume 已启用，但未找到 checkpoint 元数据：{meta_path}。"
            "请确认 --output-dir 指向已有比较任务输出目录。"
        )

    saved_models = meta.get("models") or []
    saved_tasks = meta.get("tasks") or []
    if not saved_models or not saved_tasks:
        raise RuntimeError(f"checkpoint 元数据缺少 models/tasks，无法 resume：{meta_path}")

    by_label = {label: (directory, label) for directory, label in model_dirs}
    by_dir = {directory: (directory, label) for directory, label in model_dirs}

    selected = []
    missing = []
    for model in saved_models:
        if model in by_label:
            selected.append(by_label[model])
        elif model in by_dir:
            selected.append(by_dir[model])
        else:
            missing.append(model)

    if missing:
        raise RuntimeError(
            f"checkpoint 中的模型在 results-dir 下不存在：{missing}；"
            f"results_root={results_root}"
        )

    print(f"[断点续跑] --resume 已启用，从 meta 恢复模型与任务，跳过 pick：{meta_path}")
    return selected, saved_tasks


def start_managed_judge_vllm_server(args: argparse.Namespace) -> object:
    """启动 compare_experiments.py 托管的 vLLM OpenAI server。"""
    from utils.gen_output import (
        auto_detect_tp_size,
        build_vllm_server_command,
        check_vllm_server,
        extract_host_port,
        start_vllm_server,
        stop_vllm_server,
        wait_for_vllm_server,
    )

    base_url = args.judge_vllm_base_url.rstrip("/")
    try:
        check_vllm_server(base_url, "openai", timeout=2)
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            f"{base_url} 已经有可用 server。为避免误连错误模型，"
            "请换一个 --judge-vllm-base-url 端口，或传 --no-judge-vllm-auto-start 显式复用已有 server。"
        )

    server_host, server_port = extract_host_port(base_url)
    server_tp = args.judge_vllm_server_tp or auto_detect_tp_size()
    served_model_name = args.judge_vllm_model_name or args.judge_vllm_model_path

    log_file = args.judge_vllm_server_log_file
    if log_file and not os.path.isabs(log_file):
        log_file = str(PROJECT_ROOT / log_file)

    print(
        f"[INFO] 正在启动 Judge vLLM server: {base_url} "
        f"(model={args.judge_vllm_model_path}, served_model_name={served_model_name}, tp={server_tp})",
        flush=True,
    )
    print(f"[INFO] Judge vLLM server 日志: {log_file}", flush=True)

    cmd = build_vllm_server_command(
        model_path=args.judge_vllm_model_path,
        host=server_host,
        port=server_port,
        tensor_parallel_size=server_tp,
        gpu_memory_utilization=args.judge_vllm_server_gpu_mem_util,
        max_model_len=args.judge_vllm_server_max_model_len,
        dtype=args.judge_vllm_server_dtype,
        trust_remote_code=args.judge_vllm_server_trust_remote_code,
        served_model_name=served_model_name,
    )
    process = start_vllm_server(cmd, log_file=log_file)
    try:
        ready = wait_for_vllm_server(
            base_url=base_url,
            resolved_endpoint="openai",
            poll_interval=5.0,
            process=process,
            api_key=args.judge_vllm_api_key,
        )
        if not ready:
            raise RuntimeError(f"Judge vLLM server 进程提前退出，请查看日志: {log_file}")
    except BaseException:
        stop_vllm_server(process)
        raise

    print(f"[INFO] Judge vLLM server 已启动并就绪: {base_url}", flush=True)
    return process


def raise_keyboard_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def run(args: argparse.Namespace) -> None:
    # 解析路径
    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    checkpoint_dir = str(output_dir / "checkpoints")
    figure_path = str(output_dir / "judge_stats")
    meta_path = os.path.join(checkpoint_dir, "_meta.json")

    case_dir = resolve_case_dir(args.case_dir, PROJECT_ROOT)

    # 随机种子
    random.seed(42)
    np.random.seed(42)

    # 1. 发现模型结果目录
    model_dirs = discover_model_dirs(results_root)

    if not model_dirs:
        raise RuntimeError(f"在 {results_root} 下未发现任何模型结果目录")

    if len(model_dirs) < 2:
        raise RuntimeError(f"只发现 1 个模型结果目录 ({model_dirs[0][0]})，至少需要 2 个才能成对比较")

    # 2. 选择模型；--resume 时从 checkpoint meta 恢复，避免重复交互 pick。
    resume_tasks: Optional[List[str]] = None
    if args.resume:
        selected, resume_tasks = resolve_resume_selection(meta_path, model_dirs, results_root)
    elif args.select == "all":
        selected = model_dirs
        print(f"[选择] all 模式，共 {len(selected)} 个模型：{[d for d, _ in selected]}")
    else:
        selected = interactive_pick_models(results_root, model_dirs)

    model_labels = [label for _, label in selected]
    anchor_models = resolve_anchor_models(args.anchor_models)
    missing_anchors = [model for model in anchor_models if model not in model_labels]
    if missing_anchors:
        raise ValueError(
            f"Anchor 模型不在当前选择的模型中: {missing_anchors}。"
            f"当前模型: {model_labels}"
        )

    # 3. 加载病历
    raw_cases = load_raw_cases(case_dir)

    # 4. 确定任务
    common_tasks = available_common_tasks(results_root, selected)
    if resume_tasks is not None:
        selected_tasks = resume_tasks
        missing_tasks = [task for task in selected_tasks if resolve_task_name(task) not in {resolve_task_name(t) for t in common_tasks}]
        if missing_tasks:
            raise RuntimeError(
                f"checkpoint 中的任务在当前模型共同任务中不存在：{missing_tasks}；共同任务：{common_tasks}"
            )
    else:
        selected_tasks = parse_tasks_arg(args.tasks, common_tasks)
    if not selected_tasks:
        raise RuntimeError(f"未找到可比较任务；共同任务：{common_tasks}")

    # 5. 参数一致性检查
    resolved_judge_model = resolve_judge_model(args)
    response_format = resolve_judge_response_format(args)
    judge_extra_body = resolve_judge_extra_body(args)
    judge_identity = f"{args.judge_backend}:{resolved_judge_model}"

    check_meta_consistency(
        meta_path, model_labels, selected_tasks, args.repeats, judge_identity,
    )

    # 6. 打印配置摘要
    print("\n" + "=" * 60)
    print("LLM Judge 比较实验配置")
    print("=" * 60)
    print(f"  结果根目录：{results_root}")
    print(f"  原始病历目录：{case_dir}")
    print(f"  比较模型 ({len(selected)})：{model_labels}")
    print(f"  Anchor 模型：{anchor_models}")
    print(f"  比较任务：{selected_tasks}")
    print(f"  输出目录：{output_dir}")
    print(f"  Judge 后端：{args.judge_backend}")
    print(f"  Judge 模型：{resolved_judge_model}")
    if args.judge_backend == "api":
        print(f"  Judge API base：{args.judge_api_base}")
    else:
        print(f"  Judge vLLM base：{args.judge_vllm_base_url}")
        print(f"  Judge vLLM 自动启动：{args.judge_vllm_auto_start}")
        print(f"  Judge vLLM 模型路径：{args.judge_vllm_model_path}")
    print(f"  Judge temperature：{args.judge_temperature}")
    print(f"  Judge top_p：{args.judge_top_p}")
    print(f"  Judge max_tokens：{args.judge_max_tokens}")
    print(f"  Judge response_format：{args.judge_response_format}")
    print(f"  Judge enable_thinking：{args.judge_enable_thinking}")
    print(f"  每对重复投票：{args.repeats}")
    print(f"  Bootstrap 次数：{args.bootstrap}")
    print(f"  显著性水平：{args.significance}")
    print(f"  Judge 并发线程：{args.judge_workers}")
    print("=" * 60 + "\n")

    # 7. 构建任务数据
    task_payloads: Dict[str, Dict] = {}
    for task in selected_tasks:
        task_data = {
            "raw_cases": raw_cases,
            "model_outputs": load_task_outputs(results_root, task, selected, raw_cases),
        }
        if args.limit_cases is not None:
            non_empty = [
                set(outputs.keys())
                for outputs in task_data["model_outputs"].values()
                if outputs
            ]
            common_ids = sorted(set.intersection(*non_empty)) if non_empty else []
            keep = common_ids[: args.limit_cases]
            task_data["raw_cases"] = {cid: raw_cases[cid] for cid in keep if cid in raw_cases}
            task_data["model_outputs"] = {
                m: {cid: txt for cid, txt in outs.items() if cid in task_data["raw_cases"]}
                for m, outs in task_data["model_outputs"].items()
            }
            print(f"[数据检查] {task}: --limit-cases={args.limit_cases}，原共同病历数 {len(common_ids)}")

        if validate_task_data(task, task_data):
            task_payloads[task] = task_data

    if not task_payloads:
        raise RuntimeError("没有任务具备足够数据可执行 Judge 对比")

    # dry-run 到此结束
    if args.dry_run:
        print("\n[dry-run] 数据检查完成，未调用 Judge API。")
        return

    # 8. 保存元数据
    save_meta(meta_path, model_labels, selected_tasks, args.repeats, judge_identity)

    # 9. 初始化 Judge 客户端
    if args.judge_backend == "api" and not args.judge_api_key:
        raise ValueError(
            "Judge API key 未设置。请传入 --judge-api-key 或设置环境变量 JUDGE_API_KEY"
        )

    server_process = None
    if args.judge_backend == "vllm" and args.judge_vllm_auto_start:
        server_process = start_managed_judge_vllm_server(args)
    elif args.judge_backend == "vllm":
        from utils.gen_output import check_vllm_server

        check_vllm_server(args.judge_vllm_base_url.rstrip("/"), "openai")

    judge_base_url = (
        args.judge_api_base.rstrip("/")
        if args.judge_backend == "api"
        else f"{args.judge_vllm_base_url.rstrip('/')}/v1"
    )
    judge_api_key = args.judge_api_key if args.judge_backend == "api" else args.judge_vllm_api_key
    judge_client = OpenAICompatibleClient(
        base_url=judge_base_url,
        api_key=judge_api_key,
        local_mode=(args.judge_backend == "vllm"),
    )

    # 10. 逐任务执行
    total_start = time.time()

    try:
        for task, task_data in task_payloads.items():
            print(f"\n>>> 开始任务：{task} <<<")

            win_matrix = run_pairwise_evaluation(
                task_data=task_data,
                task=task,
                client=judge_client,
                judge_model=resolved_judge_model,
                repeats=args.repeats,
                checkpoint_dir=checkpoint_dir,
                judge_workers=args.judge_workers,
                judge_temperature=args.judge_temperature,
                judge_top_p=args.judge_top_p,
                judge_max_tokens=args.judge_max_tokens,
                judge_response_format=response_format,
                judge_extra_body=judge_extra_body,
            )

            if win_matrix.empty or win_matrix.shape[0] < 2:
                print(f"[跳过] {task}: 有效比较不足，无法进行统计推断")
                continue

            from utils.com_exp.stats import run_bootstrap, run_holm_correction

            models = win_matrix.index.tolist()
            print(f"[统计推断] {task} 胜负矩阵：\n{win_matrix}")

            boot_results = run_bootstrap(
                win_matrix,
                n_boot=args.bootstrap,
                anchor_models=anchor_models,
            )
            sig_results = run_holm_correction(win_matrix, boot_results, significance_level=args.significance)

            save_results(task, boot_results, sig_results, str(output_dir), figure_path)
    finally:
        if server_process is not None:
            from utils.gen_output import stop_vllm_server

            stop_vllm_server(server_process)

    total_elapsed = round(time.time() - total_start, 2)
    print(f"\n{'=' * 60}")
    print("[SUCCESS] 全流程执行完毕")
    print(f"[TIME] 总耗时：{total_elapsed}s")
    print(f"{'=' * 60}")


def main() -> None:
    from utils.judge.api import JudgeAPIError, JudgeAuthError

    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (JudgeAuthError, JudgeAPIError) as exc:
        sys.stderr.write(f"[致命错误] Judge API 不可恢复错误，已清理托管进程并强制退出: {exc}\n")
        sys.stderr.flush()
        os._exit(1)
    except KeyboardInterrupt:
        sys.stderr.write("\n[中断] 收到 Ctrl+C，已清理托管进程并强制退出。\n")
        sys.stderr.flush()
        os._exit(130)
    except RuntimeError as exc:
        parser.exit(1, f"[错误] {exc}\n")
    except ValueError as exc:
        parser.exit(1, f"[参数错误] {exc}\n")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    main()

# -*- coding: utf-8 -*-
"""聚合 candidate-vs-base LLM Judge 结果并绘制 Anchor Score 图表。"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.com_exp.visualization import (
    _get_LinearSegmentedColormap,
    _get_plt,
    detect_cjk_font,
    plot_anchor_score_bar,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANCHOR_MODEL = "base_model_baichuan-full"
DEFAULT_ORDER_TASK = "考题生成"
DEFAULT_INPUT_DIR = "./results-clean/LLM_Judge_compare/dpo-single"
DEFAULT_OUTPUT_DIR = "./results-clean/LLM_Judge_compare/dpo-anchor-aggregate"

TASK_EN_NAMES = {
    "病历标准化": "SOAP Standardization",
    "考题生成": "Question Generation",
    "临床思维": "Clinical Thinking",
    "病历综合评分": "Case Scoring",
    "SOAP": "SOAP Standardization",
}

ANCHOR_BAR_PALETTE = [
    "#5E72C8",
    "#7893DD",
    "#93B0E8",
    "#ABC4EF",
    "#C2D2EA",
    "#D4D8DF",
    "#E3D9D2",
    "#EBC7B8",
    "#EDB196",
    "#E8957E",
    "#D47663",
]


@dataclass(frozen=True)
class PairRun:
    source_dir: Path
    judge_model: str
    candidate: str
    task: str
    anchor_wins: int
    candidate_wins: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="聚合多个 candidate-vs-base LLM Judge 结果，按 Anchor Score 输出排序图表。",
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help=f"输入目录 (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"输出目录 (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--anchor-model", default=DEFAULT_ANCHOR_MODEL, help=f"base/anchor 模型名 (default: {DEFAULT_ANCHOR_MODEL})")
    parser.add_argument("--order-task", default=DEFAULT_ORDER_TASK, help=f"全局排序任务 (default: {DEFAULT_ORDER_TASK})")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap 迭代次数 (default: 10000)")
    parser.add_argument("--no-show-ci", action="store_true", help="柱状图不显示 95%% CI 误差线")
    parser.add_argument("--no-heatmap", action="store_true", help="不生成跨任务 Anchor Score heatmap")
    parser.add_argument("--judge-model-filter", action="append", default=None, help="只聚合指定 judge_model，可重复传参")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入并打印计划，不写输出")
    return parser


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def safe_group_name(judge_model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", judge_model).strip("_")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_pair_runs(input_dir: Path, anchor_model: str, judge_filter: Optional[List[str]]) -> List[PairRun]:
    runs: List[PairRun] = []
    filters = set(judge_filter or [])

    for run_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        meta_path = run_dir / "checkpoints" / "_meta.json"
        if not meta_path.exists():
            continue

        meta = read_json(meta_path)
        judge_model = meta.get("judge_model")
        if filters and judge_model not in filters:
            continue

        models = meta.get("models") or []
        if len(models) != 2 or anchor_model not in models:
            raise RuntimeError(
                f"{meta_path} 不是严格的 anchor-vs-candidate 结果；"
                f"models={models}, anchor={anchor_model}"
            )

        candidate = next(model for model in models if model != anchor_model)
        for task in meta.get("tasks") or []:
            checkpoint_path = run_dir / "checkpoints" / f"{task}_checkpoint.csv"
            if not checkpoint_path.exists():
                print(f"[跳过] 缺少 checkpoint: {checkpoint_path}")
                continue

            matrix = pd.read_csv(checkpoint_path, index_col=0)
            missing_models = [model for model in [anchor_model, candidate] if model not in matrix.index or model not in matrix.columns]
            if missing_models:
                raise RuntimeError(f"{checkpoint_path} 缺少模型: {missing_models}")

            runs.append(
                PairRun(
                    source_dir=run_dir,
                    judge_model=judge_model,
                    candidate=candidate,
                    task=task,
                    anchor_wins=int(matrix.loc[anchor_model, candidate]),
                    candidate_wins=int(matrix.loc[candidate, anchor_model]),
                )
            )

    if not runs:
        raise RuntimeError(f"未在输入目录发现可聚合结果: {input_dir}")
    return runs


def validate_no_duplicates(runs: List[PairRun]) -> None:
    seen: Dict[Tuple[str, str, str], PairRun] = {}
    duplicates = []
    for run in runs:
        key = (run.judge_model, run.candidate, run.task)
        if key in seen:
            duplicates.append((key, seen[key].source_dir, run.source_dir))
        else:
            seen[key] = run

    if duplicates:
        lines = ["发现重复的 judge_model + candidate + task 结果，默认不自动合并："]
        for key, first, second in duplicates:
            lines.append(f"  {key}: {first} | {second}")
        raise RuntimeError("\n".join(lines))


def build_win_matrix(
    task_runs: List[PairRun],
    anchor_model: str,
    model_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    candidates = sorted({run.candidate for run in task_runs})
    models = [anchor_model] + candidates
    if model_order:
        models = [model for model in model_order if model in models]
        models += [model for model in [anchor_model] + candidates if model not in models]

    matrix = pd.DataFrame(0, index=models, columns=models, dtype=int)
    for run in task_runs:
        matrix.loc[anchor_model, run.candidate] = run.anchor_wins
        matrix.loc[run.candidate, anchor_model] = run.candidate_wins
    return matrix


def add_vote_columns(results: pd.DataFrame, matrix: pd.DataFrame, anchor_model: str) -> pd.DataFrame:
    enriched = results.copy()
    enriched["Anchor_Wins"] = [int(matrix.loc[anchor_model, model]) if model != anchor_model else 0 for model in enriched.index]
    enriched["Candidate_Wins"] = [int(matrix.loc[model, anchor_model]) if model != anchor_model else 0 for model in enriched.index]
    enriched["Total_Votes"] = enriched["Anchor_Wins"] + enriched["Candidate_Wins"]
    return enriched


def run_anchor_bootstrap(matrix: pd.DataFrame, anchor_model: str, n_boot: int) -> pd.DataFrame:
    """Bootstrap candidate-vs-anchor win rates for a star-shaped comparison graph.

    For this aggregation design every non-anchor model is compared only with the
    anchor, so Anchor_Score is the candidate's expected win rate against the
    anchor. This avoids fitting a full BT model for edges that were never judged.
    """
    rows = []
    for model in matrix.index:
        if model == anchor_model:
            rows.append({
                "Model": model,
                "Anchor_Score": 0.5,
                "Anchor_CI_Lower": 0.5,
                "Anchor_CI_Upper": 0.5,
                "Anchor_Wins": 0,
                "Candidate_Wins": 0,
                "Total_Votes": 0,
            })
            continue

        anchor_wins = int(matrix.loc[anchor_model, model])
        candidate_wins = int(matrix.loc[model, anchor_model])
        total = anchor_wins + candidate_wins
        if total <= 0:
            raise RuntimeError(f"{model} vs {anchor_model} 没有有效投票")

        score = candidate_wins / total
        samples = np.random.binomial(total, score, size=n_boot) / total
        rows.append({
            "Model": model,
            "Anchor_Score": float(score),
            "Anchor_CI_Lower": float(np.quantile(samples, 0.025)),
            "Anchor_CI_Upper": float(np.quantile(samples, 0.975)),
            "Anchor_Wins": anchor_wins,
            "Candidate_Wins": candidate_wins,
            "Total_Votes": total,
        })

    return pd.DataFrame(rows).set_index("Model")


def color_map_for_order(model_order: List[str]) -> Dict[str, str]:
    if len(model_order) <= len(ANCHOR_BAR_PALETTE):
        colors = ANCHOR_BAR_PALETTE[: len(model_order)]
    else:
        plt = _get_plt()
        cmap = plt.get_cmap("coolwarm")
        colors = [cmap(x) for x in np.linspace(0.08, 0.92, len(model_order))]
    return dict(zip(model_order, colors))


def save_anchor_bar(
    results: pd.DataFrame,
    task: str,
    figure_dir: Path,
    model_order: List[str],
    colors_by_model: Dict[str, str],
    show_ci: bool,
) -> None:
    ordered = [model for model in model_order if model in results.index]
    colors = [colors_by_model[model] for model in ordered]
    plot_anchor_score_bar(
        results,
        task,
        str(figure_dir),
        model_order=ordered,
        show_ci=show_ci,
        force_english=True,
        colors=colors,
    )


def plot_all_tasks_heatmap(summary: pd.DataFrame, tasks: List[str], output_path: Path) -> None:
    plt = _get_plt()
    LinearSegmentedColormap = _get_LinearSegmentedColormap()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font_name = detect_cjk_font()
    plt.rcParams.update({
        "font.family": [font_name, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    score_cols = [f"{task}_Anchor_Score" for task in tasks]
    heatmap_data = summary.set_index("Model")[score_cols].astype(float) * 100
    heatmap_data.columns = [TASK_EN_NAMES.get(task, task) for task in tasks]

    fig_height = max(4.8, 0.38 * len(heatmap_data) + 1.7)
    fig_width = max(8.5, 1.75 * len(tasks) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    colors = ["#D47663", "#F4F4F3", "#5E72C8"]
    cmap = LinearSegmentedColormap.from_list("anchor_score_diverging", colors, N=256)
    masked = np.ma.masked_invalid(heatmap_data.values)
    im = ax.imshow(masked, cmap=cmap, vmin=35, vmax=65, aspect="auto")

    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns, rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels([m if len(m) <= 28 else m[:25] + "..." for m in heatmap_data.index], fontsize=9)

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data.iloc[i, j]
            if pd.isna(value):
                text = "NA"
                color = "#6B7280"
            else:
                text = f"{value:.1f}"
                color = "#111827" if 43 <= value <= 57 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=color)

    ax.set_title("Anchor Score Across Tasks", fontsize=13, fontweight="bold", pad=14)
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Anchor Score (%)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[可视化] 跨任务 heatmap 已保存：{output_path}")


def write_manifest(group_dir: Path, runs: List[PairRun], excluded: List[dict]) -> None:
    manifest_rows = [
        {
            "Judge_Model": run.judge_model,
            "Candidate": run.candidate,
            "Task": run.task,
            "Anchor_Wins": run.anchor_wins,
            "Candidate_Wins": run.candidate_wins,
            "Source_Dir": str(run.source_dir),
        }
        for run in runs
    ]
    pd.DataFrame(manifest_rows).to_csv(group_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    if excluded:
        pd.DataFrame(excluded).to_csv(group_dir / "excluded_models.csv", index=False, encoding="utf-8-sig")


def aggregate_group(
    judge_model: str,
    runs: List[PairRun],
    output_root: Path,
    anchor_model: str,
    order_task: str,
    n_boot: int,
    show_ci: bool,
    make_heatmap: bool,
    dry_run: bool,
) -> None:
    group_dir = output_root / safe_group_name(judge_model)
    figure_dir = group_dir / "judge_stats"
    tasks = sorted({run.task for run in runs}, key=lambda task: (task != order_task, task))

    order_candidates = {run.candidate for run in runs if run.task == order_task}
    all_candidates = {run.candidate for run in runs}
    excluded = [
        {
            "Judge_Model": judge_model,
            "Candidate": candidate,
            "Reason": f"missing order task: {order_task}",
        }
        for candidate in sorted(all_candidates - order_candidates)
    ]
    kept_runs = [run for run in runs if run.candidate in order_candidates]

    if not order_candidates:
        raise RuntimeError(f"{judge_model} 没有任何 candidate 具备排序任务: {order_task}")

    print(f"\n[聚合] judge={judge_model}")
    print(f"  输出目录：{group_dir}")
    print(f"  排序任务：{order_task}")
    print(f"  保留模型数：{len(order_candidates) + 1}（含 anchor）")
    if excluded:
        print(f"  排除模型数：{len(excluded)}")

    if dry_run:
        return

    group_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(group_dir, kept_runs, excluded)

    by_task = {task: [run for run in kept_runs if run.task == task] for task in tasks}

    order_matrix = build_win_matrix(by_task[order_task], anchor_model)
    order_results = run_anchor_bootstrap(order_matrix, anchor_model, n_boot=n_boot)
    model_order = order_results.sort_values("Anchor_Score", ascending=False).index.tolist()
    colors_by_model = color_map_for_order(model_order)

    task_results: Dict[str, pd.DataFrame] = {}
    for task in tasks:
        task_runs = by_task.get(task) or []
        if not task_runs:
            continue
        matrix = build_win_matrix(task_runs, anchor_model, model_order=model_order)
        results = run_anchor_bootstrap(matrix, anchor_model, n_boot=n_boot)
        ordered = [model for model in model_order if model in results.index]
        results = results.loc[ordered]
        results.index.name = "Model"
        results.to_csv(group_dir / f"{task}_anchor_ranking.csv", encoding="utf-8-sig")
        save_anchor_bar(results, task, figure_dir, ordered, colors_by_model, show_ci=show_ci)
        task_results[task] = results

    summary = pd.DataFrame({"Model": model_order})
    summary["Order_By_QuestionGeneration"] = np.arange(1, len(model_order) + 1)
    for task in tasks:
        results = task_results.get(task)
        if results is None:
            continue
        summary[f"{task}_Anchor_Score"] = summary["Model"].map(results["Anchor_Score"])
        summary[f"{task}_Anchor_CI_Lower"] = summary["Model"].map(results["Anchor_CI_Lower"])
        summary[f"{task}_Anchor_CI_Upper"] = summary["Model"].map(results["Anchor_CI_Upper"])

    score_cols = [col for col in summary.columns if col.endswith("_Anchor_Score")]
    summary["Mean_Anchor_Score"] = summary[score_cols].mean(axis=1, skipna=True)
    summary.to_csv(group_dir / "all_tasks_anchor_summary.csv", index=False, encoding="utf-8-sig")

    if make_heatmap:
        plotted_tasks = [task for task in tasks if f"{task}_Anchor_Score" in summary.columns]
        plot_all_tasks_heatmap(summary, plotted_tasks, figure_dir / "all_tasks_anchor_heatmap.png")


def run(args: argparse.Namespace) -> None:
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    if not input_dir.exists():
        raise RuntimeError(f"输入目录不存在: {input_dir}")

    np.random.seed(42)
    runs = discover_pair_runs(input_dir, args.anchor_model, args.judge_model_filter)
    validate_no_duplicates(runs)

    by_judge: Dict[str, List[PairRun]] = {}
    for run in runs:
        by_judge.setdefault(run.judge_model, []).append(run)

    print(f"[发现] 输入目录：{input_dir}")
    print(f"[发现] judge 分组数：{len(by_judge)}")
    print(f"[发现] pair-task 结果数：{len(runs)}")

    for judge_model, group_runs in sorted(by_judge.items()):
        aggregate_group(
            judge_model=judge_model,
            runs=group_runs,
            output_root=output_dir,
            anchor_model=args.anchor_model,
            order_task=args.order_task,
            n_boot=args.bootstrap,
            show_ci=not args.no_show_ci,
            make_heatmap=not args.no_heatmap,
            dry_run=args.dry_run,
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run(args)
    except RuntimeError as exc:
        parser.exit(1, f"[错误] {exc}\n")
    except ValueError as exc:
        parser.exit(1, f"[参数错误] {exc}\n")


if __name__ == "__main__":
    main()

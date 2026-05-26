# -*- coding: utf-8 -*-
"""可视化模块：排名图 + 显著性热力图。

字体策略：SimHei → 自动下载 → 探测可用中文字体 → 英文 fallback。
matplotlib rcParams 仅在函数内局部设置，不污染全局。
"""

import os
from typing import Optional

import numpy as np
import pandas as pd


# ================= 延迟导入 =================

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _get_fm():
    import matplotlib.font_manager as fm
    return fm


def _get_LinearSegmentedColormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap


# ================= 字体管理 =================

_CJK_FONT_CANDIDATES = [
    "SimHei",
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]

_CJK_FONT_CACHE: Optional[str] = None


def _download_simhei(target_dir: str) -> Optional[str]:
    """尝试下载 SimHei 字体到指定目录。返回字体路径或 None。"""
    target_path = os.path.join(target_dir, "SimHei.ttf")
    if os.path.exists(target_path):
        return target_path

    print("[字体] 尝试下载 SimHei 字体...")
    try:
        urls = [
            "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf",
        ]
        for url in urls:
            try:
                import urllib.request
                urllib.request.urlretrieve(url, target_path)
                if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
                    print(f"[字体] SimHei 下载成功: {target_path}")
                    return target_path
            except Exception:
                continue
    except Exception:
        pass

    print("[字体] SimHei 下载失败")
    return None


def detect_cjk_font() -> str:
    """探测可用的中文字体，返回字体名。"""
    global _CJK_FONT_CACHE
    if _CJK_FONT_CACHE is not None:
        return _CJK_FONT_CACHE

    fm = _get_fm()

    # 1. 检查系统已安装的候选字体
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in available_fonts:
            _CJK_FONT_CACHE = candidate
            return candidate

    # 2. 尝试下载 SimHei
    font_dir = os.path.join(os.path.expanduser("~"), ".cache", "com_exp", "fonts")
    os.makedirs(font_dir, exist_ok=True)
    downloaded = _download_simhei(font_dir)
    if downloaded:
        fm.fontManager.addfont(downloaded)
        _CJK_FONT_CACHE = "SimHei"
        return "SimHei"

    # 3. 探测所有可用字体中含 CJK 的
    for f in fm.fontManager.ttflist:
        name_lower = f.name.lower()
        if any(kw in name_lower for kw in ["cjk", "hei", "song", "ming", "gothic", "chinese", "noto sans sc"]):
            _CJK_FONT_CACHE = f.name
            return f.name

    # 4. Fallback
    print("[字体] 未找到中文字体，图表标签将使用英文")
    _CJK_FONT_CACHE = "DejaVu Sans"
    return "DejaVu Sans"


_TASK_EN_NAMES = {
    "病历标准化": "SOAP Standardization",
    "考题生成": "Question Generation",
    "临床思维": "Clinical Thinking",
    "病历综合评分": "Case Scoring",
    "SOAP": "SOAP Standardization",
}


def _task_display_name(task: str, use_english: bool) -> str:
    if use_english:
        return _TASK_EN_NAMES.get(task, task)
    return task


def _setup_rcParams(font_name: str) -> None:
    """局部设置 matplotlib rcParams。"""
    plt = _get_plt()
    plt.rcParams.update({
        "font.family": [font_name, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.labelsize": 18,
        "axes.titlesize": 21,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 12,
        "figure.figsize": (10, 6),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ================= 排名图 =================

def plot_ranking_with_ci(
    boot_results: pd.DataFrame,
    task: str,
    output_dir: str,
    score_col: str = "BT_Score",
    lower_col: str = "BT_CI_Lower",
    upper_col: str = "BT_CI_Upper",
    ylabel: str = "Bradley-Terry Score",
    filename_suffix: str = "ranking_ci",
    title_metric: str = "Model Ranking",
) -> str:
    """绘制带置信区间的竖线排名图。"""
    plt = _get_plt()
    os.makedirs(output_dir, exist_ok=True)

    font_name = detect_cjk_font()
    use_english = font_name == "DejaVu Sans"
    _setup_rcParams(font_name)

    import logging
    import warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    required_cols = [score_col, lower_col, upper_col]
    missing_cols = [col for col in required_cols if col not in boot_results.columns]
    if missing_cols:
        raise ValueError(f"绘图缺少必要列: {missing_cols}")

    df_plot = boot_results.sort_values(score_col, ascending=False).reset_index()
    df_plot.rename(columns={"index": "Model"}, inplace=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    x_positions = np.arange(len(df_plot))
    scores = df_plot[score_col].values
    ci_lower = df_plot[lower_col].values
    ci_upper = df_plot[upper_col].values

    yerr_lower = np.maximum(scores - ci_lower, 0)
    yerr_upper = np.maximum(ci_upper - scores, 0)

    ax.errorbar(
        x_positions, scores, yerr=[yerr_lower, yerr_upper],
        fmt="o", color="#2E86AB", markersize=10,
        elinewidth=3, capsize=12, capthick=3,
        alpha=0.8, zorder=3, label="Score ± 95% CI",
    )

    for i, (pos, score) in enumerate(zip(x_positions, scores)):
        ax.text(pos + 0.15, score, f"{score:.3f}", ha="left", va="center", fontsize=10, fontweight="bold")

    if ci_lower.min() < 0 < ci_upper.max():
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)

    ax.set_xticks(x_positions)
    model_labels = [m if len(m) <= 18 else m[:15] + "..." for m in df_plot["Model"]]
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=11)

    task_label = _task_display_name(task, use_english)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(
        f"{task_label} - {title_metric} with 95% CI\n(Bootstrap)",
        fontsize=13, pad=20, fontweight="bold",
    )

    y_min = max(0, ci_lower.min() - 0.05)
    y_max = min(1, ci_upper.max() + 0.08)
    ax.set_ylim(y_min, y_max)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{task}_{filename_suffix}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[可视化] 排名竖线图已保存：{fig_path}")
    return fig_path


# ================= Anchor Score 柱状图 =================

def plot_anchor_score_bar(
    boot_results: pd.DataFrame,
    task: str,
    output_dir: str,
    score_col: str = "Anchor_Score",
    model_order: Optional[list] = None,
    show_ci: bool = False,
    force_english: bool = False,
    colors: Optional[list] = None,
) -> str:
    """绘制 Anchor Score 均值柱状图。"""
    plt = _get_plt()
    os.makedirs(output_dir, exist_ok=True)

    font_name = detect_cjk_font()
    use_english = force_english or font_name == "DejaVu Sans"
    _setup_rcParams(font_name)

    import logging
    import warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    if score_col not in boot_results.columns:
        raise ValueError(f"绘图缺少必要列: {score_col}")

    if model_order:
        ordered = [model for model in model_order if model in boot_results.index]
        missing = [model for model in boot_results.index if model not in ordered]
        df_plot = boot_results.loc[ordered + missing].reset_index()
    else:
        df_plot = boot_results.sort_values(score_col, ascending=False).reset_index()
    df_plot.rename(columns={"index": "Model"}, inplace=True)

    n_models = len(df_plot)
    fig_width = max(8, 0.72 * n_models + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    scores = df_plot[score_col].values * 100
    x_positions = np.arange(n_models)

    # Leaderboard-style discrete gradient: deep blue winners, pale middle,
    # warm orange/red tail, matching the referenced Medical LLM chart style.
    palette = [
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
    if colors is None and n_models <= len(palette):
        colors = palette[:n_models]
    elif colors is None:
        cmap = _get_plt().get_cmap("coolwarm")
        colors = [cmap(x) for x in np.linspace(0.08, 0.92, n_models)]
    elif len(colors) < n_models:
        raise ValueError(f"colors 数量不足：{len(colors)} < {n_models}")

    yerr = None
    if show_ci:
        lower_col = score_col.replace("_Score", "_CI_Lower")
        upper_col = score_col.replace("_Score", "_CI_Upper")
        required_cols = [lower_col, upper_col]
        missing_cols = [col for col in required_cols if col not in df_plot.columns]
        if missing_cols:
            raise ValueError(f"绘图缺少 CI 列: {missing_cols}")
        ci_lower = df_plot[lower_col].values * 100
        ci_upper = df_plot[upper_col].values * 100
        yerr = [
            np.maximum(scores - ci_lower, 0),
            np.maximum(ci_upper - scores, 0),
        ]

    bar_width = min(0.78, max(0.36, 0.78 * n_models / max(n_models, 4)))
    bars = ax.bar(
        x_positions,
        scores,
        yerr=yerr,
        error_kw={
            "elinewidth": 1.4,
            "ecolor": "#2D3748",
            "capsize": 4,
            "capthick": 1.2,
        } if show_ci else None,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        width=bar_width,
        zorder=3,
    )

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(score + 1.2, 79.0),
            f"{score:.1f}",
            va="bottom",
            ha="center",
            fontsize=10,
            color="#1F2933",
        )

    model_labels = [m if len(m) <= 18 else m[:15] + "..." for m in df_plot["Model"]]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_labels, rotation=50, ha="right", fontsize=10)

    task_label = _task_display_name(task, use_english)
    ax.set_ylabel("Anchor Score (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{task_label} - Anchor Score Ranking",
        fontsize=13,
        pad=18,
        fontweight="bold",
    )

    ax.set_ylim(0, 80)
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0f}")
    ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{task}_anchor_score_bar.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"[可视化] Anchor Score 柱状图已保存：{fig_path}")
    return fig_path


# ================= 显著性热力图 =================

def plot_significance_heatmap(
    sig_results: pd.DataFrame,
    boot_results: pd.DataFrame,
    task: str,
    output_dir: str,
) -> Optional[str]:
    """绘制显著性热力图（蓝绿配色）。"""
    plt = _get_plt()
    LinearSegmentedColormap = _get_LinearSegmentedColormap()
    os.makedirs(output_dir, exist_ok=True)

    font_name = detect_cjk_font()
    use_english = font_name == "DejaVu Sans"
    _setup_rcParams(font_name)

    if sig_results.empty or boot_results.empty:
        print(f"[警告] 任务 {task} 的显著性结果为空，跳过热力图绘制")
        return None

    models = boot_results.index.tolist()
    n_models = len(models)

    sig_matrix = pd.DataFrame(np.nan, index=models, columns=models)

    for _, row in sig_results.iterrows():
        parts = row["Pair"].split(" vs ")
        if len(parts) != 2:
            continue
        m1, m2 = parts[0].strip(), parts[1].strip()

        if not row["Significant"]:
            sig_matrix.loc[m1, m2] = 0
            sig_matrix.loc[m2, m1] = 0
        else:
            if boot_results.loc[m1, "BT_Score"] > boot_results.loc[m2, "BT_Score"]:
                sig_matrix.loc[m1, m2] = 1
                sig_matrix.loc[m2, m1] = -1
            else:
                sig_matrix.loc[m1, m2] = -1
                sig_matrix.loc[m2, m1] = 1

    fig, ax = plt.subplots(figsize=(10, 9))

    colors = ["#1B4965", "#E8F0F1", "#2D9E5D"]
    cmap = LinearSegmentedColormap.from_list("blue_green_sig", colors, N=256)

    im = ax.imshow(sig_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))

    model_labels = [m if len(m) <= 20 else m[:17] + "..." for m in models]
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(model_labels, fontsize=10)

    for i in range(n_models):
        for j in range(n_models):
            value = sig_matrix.iloc[i, j]

            if i == j:
                text, color, fw, fs = "-", "#777777", "normal", 14
            elif value == 1:
                text, color, fw, fs = "\u2191", "#FFFFFF", "bold", 16
            elif value == -1:
                text, color, fw, fs = "\u2193", "#FFFFFF", "bold", 16
            else:
                text, color, fw, fs = "", "#555555", "normal", 12

            if text:
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=fs, fontweight=fw)

    task_label = _task_display_name(task, use_english)
    ax.set_title(
        f"{task_label} - Pairwise Significance\n(\u2191=row>col, \u2193=row<col)",
        fontsize=12, fontweight="bold", pad=12,
    )

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], pad=0.02, fraction=0.046)
    cbar_labels = ["Worse", "No diff", "Better"] if use_english else ["劣势", "无差", "优势"]
    cbar.ax.set_yticklabels(cbar_labels, fontsize=10, fontweight="bold")
    cbar_label = "Relative Performance" if use_english else "相对性能"
    cbar.set_label(cbar_label, fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{task}_significance_heatmap.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"[可视化] 显著性热力图已保存：{fig_path}")
    return fig_path

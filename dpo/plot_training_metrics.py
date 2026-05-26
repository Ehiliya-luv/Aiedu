# -*- coding: utf-8 -*-
"""绘制 DPO 训练过程中的 loss / reward / 其它指标曲线。

读取 TRL DPOTrainer 写出的 trainer_state.json，把 log_history 中的
loss、rewards/chosen、rewards/rejected、rewards/margins、rewards/accuracies、
logps/*、logits/*、grad_norm、learning_rate、entropy、mean_token_accuracy 等
字段画成单图与 dashboard 合图，保存为 PNG。

用法:
    python plot_training_metrics.py
    python plot_training_metrics.py --state ../../dpo_model-newest
    python plot_training_metrics.py --state ../../dpo_model-newest/checkpoint-540/trainer_state.json
    python plot_training_metrics.py --output-dir output/dpo_plots --smooth 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # 不依赖 GUI 后端，方便后台/远端运行
import matplotlib.pyplot as plt
from matplotlib import font_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 中文字体探测：首选列表 -> 系统 CJK 字体扫描 -> 英文 fallback
# ---------------------------------------------------------------------------
# 首选列表（按 Windows / macOS / Linux 常见安装顺序）
_PREFERRED_CJK_FONTS: List[str] = [
    "Microsoft YaHei",
    "Microsoft JhengHei",
    "SimHei",
    "SimSun",
    "FangSong",
    "KaiTi",
    "PingFang SC",
    "Hiragino Sans GB",
    "STHeiti",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
]

# 用于在系统字体名里识别 CJK 字体的关键词（小写匹配，刻意保守避免误判）
_CJK_NAME_KEYWORDS: Tuple[str, ...] = (
    "cjk",
    "yahei", "jhenghei",
    "simhei", "simsun", "fangsong", "kaiti",
    "noto sans sc", "noto sans tc", "noto sans cjk", "noto serif cjk",
    "source han",
    "pingfang", "hiragino sans gb", "stheiti", "heiti sc",
    "wenquanyi", "wqy",
)


def _detect_cjk_font() -> Tuple[Optional[str], List[str]]:
    """返回 (CJK 字体名 或 None, 完整 sans-serif 候选链)。"""
    try:
        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception as exc:  # 极端情况下 font cache 失败
        logger.warning("读取系统字体失败，使用英文 fallback: %s", exc)
        return None, ["DejaVu Sans", "sans-serif"]

    # 1) 优先匹配首选列表
    for name in _PREFERRED_CJK_FONTS:
        if name in installed:
            return name, [name, "DejaVu Sans", "sans-serif"]

    # 2) 扫描系统其它 CJK 字体
    for font in font_manager.fontManager.ttflist:
        name_lower = font.name.lower()
        if any(kw in name_lower for kw in _CJK_NAME_KEYWORDS):
            return font.name, [font.name, "DejaVu Sans", "sans-serif"]

    # 3) 全部 miss
    return None, ["DejaVu Sans", "sans-serif"]


_CJK_FONT, _SANS_CHAIN = _detect_cjk_font()
HAS_CJK_FONT: bool = _CJK_FONT is not None

plt.rcParams["font.sans-serif"] = _SANS_CHAIN
plt.rcParams["axes.unicode_minus"] = False

if HAS_CJK_FONT:
    logger.info("已启用中文字体: %s", _CJK_FONT)
else:
    logger.warning(
        "未检测到任何中文字体，图表标题将使用英文（如需中文请安装 "
        "Microsoft YaHei / Noto Sans CJK SC / WenQuanYi Zen Hei 等）"
    )


# 图表上会显示的双语文本，无中文字体时自动 fallback
_LABELS_CN: Dict[str, str] = {
    "dashboard_title": "DPO 训练指标总览",
    "rewards_accuracies_title": "Rewards Accuracies (越接近 1 表示 chosen reward 越常 > rejected)",
}
_LABELS_EN: Dict[str, str] = {
    "dashboard_title": "DPO Training Metrics Overview",
    "rewards_accuracies_title": "Rewards Accuracies (closer to 1 = chosen reward > rejected more often)",
}
_LABELS: Dict[str, str] = _LABELS_CN if HAS_CJK_FONT else _LABELS_EN


def L(key: str) -> str:
    """根据是否有中文字体返回对应的图表文本。"""
    return _LABELS.get(key, key)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 自动发现候选目录（按优先级），分别相对于当前工作目录与脚本目录
DEFAULT_CANDIDATES: List[str] = [
    "output/dpo_model",
    "../dpo_model-newest",
    "../../dpo_model-newest",
]


def _latest_checkpoint_state(dir_path: str) -> Optional[str]:
    """在目录下寻找 checkpoint-N/trainer_state.json 中编号最大的那一份。"""
    if not os.path.isdir(dir_path):
        return None
    ckpts: List[Tuple[int, str]] = []
    for name in os.listdir(dir_path):
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        if not match:
            continue
        state_file = os.path.join(dir_path, name, "trainer_state.json")
        if os.path.isfile(state_file):
            ckpts.append((int(match.group(1)), state_file))
    if not ckpts:
        return None
    return max(ckpts, key=lambda x: x[0])[1]


def find_trainer_state(path: Optional[str]) -> str:
    """根据 --state 参数（文件 / 目录 / 缺省）定位 trainer_state.json。"""
    if path:
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            direct = os.path.join(path, "trainer_state.json")
            if os.path.isfile(direct):
                return direct
            ckpt = _latest_checkpoint_state(path)
            if ckpt:
                return ckpt
            raise FileNotFoundError(
                f"{path} 下未找到 trainer_state.json，也没有可用的 checkpoint-* 子目录"
            )
        raise FileNotFoundError(path)

    tried: List[str] = []
    for base in (os.getcwd(), SCRIPT_DIR):
        for candidate in DEFAULT_CANDIDATES:
            full = os.path.normpath(os.path.join(base, candidate))
            tried.append(full)
            if not os.path.isdir(full):
                continue
            direct = os.path.join(full, "trainer_state.json")
            if os.path.isfile(direct):
                logger.info("自动定位到 trainer_state: %s", direct)
                return direct
            ckpt = _latest_checkpoint_state(full)
            if ckpt:
                logger.info("自动定位到 trainer_state: %s", ckpt)
                return ckpt
    raise FileNotFoundError(
        "未提供 --state 且默认路径下未找到 trainer_state.json。\n"
        "请用 --state 指定 trainer_state.json 文件或包含 checkpoint-* 子目录的输出目录。\n"
        f"已尝试目录: {tried}"
    )


def load_log_history(state_path: str) -> List[Dict]:
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    history = state.get("log_history") or []
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"{state_path} 中的 log_history 为空或非数组")
    logger.info("读取 %d 条训练记录: %s", len(history), state_path)
    return history


def collect_series(
    history: Sequence[Dict],
    keys: Sequence[str],
) -> Dict[str, Dict[str, List[float]]]:
    """提取每个 metric 的 (steps, epochs, values)；缺失字段会被自动跳过。"""
    series: Dict[str, Dict[str, List[float]]] = {
        k: {"step": [], "epoch": [], "value": []} for k in keys
    }
    for entry in history:
        step = entry.get("step")
        if step is None:
            continue
        epoch = entry.get("epoch")
        for key in keys:
            value = entry.get(key)
            if value is None:
                continue
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            series[key]["step"].append(int(step))
            series[key]["epoch"].append(float(epoch) if epoch is not None else float("nan"))
            series[key]["value"].append(value_f)
    return series


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """对齐长度的滑动平均（开头窗口逐步扩大）。"""
    if window <= 1 or len(values) <= 1:
        return list(values)
    smoothed: List[float] = []
    cum = 0.0
    buf: List[float] = []
    for v in values:
        buf.append(v)
        cum += v
        if len(buf) > window:
            cum -= buf.pop(0)
        smoothed.append(cum / len(buf))
    return smoothed


def _draw_curve(
    ax: plt.Axes,
    label: str,
    steps: Sequence[int],
    values: Sequence[float],
    smooth_window: int,
    **kwargs,
) -> bool:
    if not values:
        return False
    if smooth_window > 1 and len(values) > 1:
        ax.plot(steps, values, alpha=0.25, linewidth=1.0, **kwargs)
        ax.plot(
            steps,
            moving_average(values, smooth_window),
            label=label,
            linewidth=2.0,
            **kwargs,
        )
    else:
        ax.plot(steps, values, label=label, linewidth=1.5, **kwargs)
    return True


def plot_single(
    output_dir: str,
    file_name: str,
    title: str,
    ylabel: str,
    curves: Sequence[Tuple[str, Sequence[int], Sequence[float], Dict]],
    smooth_window: int,
) -> Optional[str]:
    """单指标（或同类多曲线）独立 PNG。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    drawn = False
    for label, steps, values, kw in curves:
        if _draw_curve(ax, label, steps, values, smooth_window, **kw):
            drawn = True
    if not drawn:
        plt.close(fig)
        logger.warning("跳过 %s: 没有可用数据", file_name)
        return None

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = os.path.join(output_dir, file_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("已保存: %s", out_path)
    return out_path


def plot_dashboard(
    output_dir: str,
    history: Sequence[Dict],
    smooth_window: int,
) -> Optional[str]:
    """把核心指标合到一张大图里方便速览。"""
    panels: List[Tuple[str, List[Tuple[str, str, str]]]] = [
        ("Loss", [("loss", "loss", "tab:blue")]),
        (
            "Rewards (chosen / rejected / margins)",
            [
                ("rewards/chosen", "chosen", "tab:green"),
                ("rewards/rejected", "rejected", "tab:red"),
                ("rewards/margins", "margins", "tab:orange"),
            ],
        ),
        ("Rewards Accuracies", [("rewards/accuracies", "accuracy", "tab:purple")]),
        (
            "Log-Probabilities",
            [
                ("logps/chosen", "logps/chosen", "tab:green"),
                ("logps/rejected", "logps/rejected", "tab:red"),
            ],
        ),
        ("Grad Norm", [("grad_norm", "grad_norm", "tab:brown")]),
        ("Learning Rate", [("learning_rate", "lr", "tab:gray")]),
    ]
    keys = sorted({k for _, items in panels for k, _, _ in items})
    series = collect_series(history, keys)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    flat_axes = axes.flatten()
    used = 0
    for ax, (title, items) in zip(flat_axes, panels):
        local_drawn = False
        for key, label, color in items:
            s = series.get(key, {"step": [], "value": []})
            # learning_rate 不做平滑（本身就是平滑曲线）
            window = 0 if key == "learning_rate" else smooth_window
            if _draw_curve(ax, label, s["step"], s["value"], window, color=color):
                local_drawn = True
        if local_drawn:
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            used += 1
        else:
            ax.axis("off")
    if used == 0:
        plt.close(fig)
        return None

    fig.suptitle(L("dashboard_title"), fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = os.path.join(output_dir, "summary_dashboard.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("已保存: %s", out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="绘制 DPO 训练 loss / reward 等指标曲线",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="trainer_state.json 路径或包含 checkpoint-* 的目录；"
        "不传时按顺序搜索 output/dpo_model、../dpo_model-newest、../../dpo_model-newest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="PNG 输出目录；不传时使用 trainer_state.json 同级的 plots/ 目录",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="滑动平均窗口大小，<=1 表示不平滑（默认 10）",
    )
    args = parser.parse_args()

    state_path = find_trainer_state(args.state)
    history = load_log_history(state_path)

    output_dir = args.output_dir or os.path.join(os.path.dirname(state_path), "plots")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("PNG 输出目录: %s", output_dir)

    keys = [
        "loss",
        "rewards/chosen",
        "rewards/rejected",
        "rewards/margins",
        "rewards/accuracies",
        "logps/chosen",
        "logps/rejected",
        "logits/chosen",
        "logits/rejected",
        "grad_norm",
        "learning_rate",
        "entropy",
        "mean_token_accuracy",
    ]
    series = collect_series(history, keys)

    def s(key: str) -> Dict[str, List]:
        return series.get(key, {"step": [], "epoch": [], "value": []})

    plot_single(
        output_dir,
        "loss.png",
        "DPO Loss",
        "loss",
        [("loss", s("loss")["step"], s("loss")["value"], {"color": "tab:blue"})],
        smooth_window=args.smooth,
    )

    plot_single(
        output_dir,
        "rewards.png",
        "DPO Rewards (chosen / rejected / margins)",
        "reward",
        [
            ("chosen", s("rewards/chosen")["step"], s("rewards/chosen")["value"], {"color": "tab:green"}),
            ("rejected", s("rewards/rejected")["step"], s("rewards/rejected")["value"], {"color": "tab:red"}),
            ("margins", s("rewards/margins")["step"], s("rewards/margins")["value"], {"color": "tab:orange"}),
        ],
        smooth_window=args.smooth,
    )

    plot_single(
        output_dir,
        "rewards_accuracies.png",
        L("rewards_accuracies_title"),
        "accuracy",
        [
            (
                "rewards/accuracies",
                s("rewards/accuracies")["step"],
                s("rewards/accuracies")["value"],
                {"color": "tab:purple"},
            )
        ],
        smooth_window=args.smooth,
    )

    plot_single(
        output_dir,
        "logps.png",
        "Log-Probabilities (chosen vs rejected)",
        "logp",
        [
            ("logps/chosen", s("logps/chosen")["step"], s("logps/chosen")["value"], {"color": "tab:green"}),
            ("logps/rejected", s("logps/rejected")["step"], s("logps/rejected")["value"], {"color": "tab:red"}),
        ],
        smooth_window=args.smooth,
    )

    plot_single(
        output_dir,
        "logits.png",
        "Logits (chosen vs rejected)",
        "logit",
        [
            ("logits/chosen", s("logits/chosen")["step"], s("logits/chosen")["value"], {"color": "tab:green"}),
            ("logits/rejected", s("logits/rejected")["step"], s("logits/rejected")["value"], {"color": "tab:red"}),
        ],
        smooth_window=args.smooth,
    )

    plot_single(
        output_dir,
        "grad_norm.png",
        "Grad Norm",
        "grad_norm",
        [("grad_norm", s("grad_norm")["step"], s("grad_norm")["value"], {"color": "tab:brown"})],
        smooth_window=args.smooth,
    )

    plot_single(
        output_dir,
        "learning_rate.png",
        "Learning Rate Schedule",
        "lr",
        [("learning_rate", s("learning_rate")["step"], s("learning_rate")["value"], {"color": "tab:gray"})],
        smooth_window=0,
    )

    plot_single(
        output_dir,
        "entropy.png",
        "Entropy",
        "entropy",
        [("entropy", s("entropy")["step"], s("entropy")["value"], {"color": "tab:cyan"})],
        smooth_window=args.smooth,
    )

    plot_single(
        output_dir,
        "mean_token_accuracy.png",
        "Mean Token Accuracy",
        "acc",
        [
            (
                "mean_token_accuracy",
                s("mean_token_accuracy")["step"],
                s("mean_token_accuracy")["value"],
                {"color": "tab:olive"},
            )
        ],
        smooth_window=args.smooth,
    )

    plot_dashboard(output_dir, history, args.smooth)

    logger.info("全部图表已生成在: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

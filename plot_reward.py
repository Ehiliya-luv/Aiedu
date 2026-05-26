# -*- coding: utf-8 -*-
"""绘制 GRPO 训练曲线（仅 step 粒度，从 main_process.log 抽取）。

数据源：``main_process.log`` —— verl 每个 training step 打印一行长 metric。
包含 critic/score/mean、critic/advantages/mean、actor/entropy、actor/grad_norm、
response_length/mean 等 step 粒度指标。

如果训练时开启了 val（--verl-disable-validation false），main_process.log
里会同时有 val-core/... 行；本脚本会自动解析并叠加到 reward 图上做双线。

输出 4 张图到 ``./plots/`` 子目录：
  - reward_curve.png      —— train reward (+val reward 如有) + advantage
  - actor_health.png      —— pg_loss / entropy / grad_norm（mode collapse 监控）
  - response_length.png   —— actor 输出长度演化（length hack 监控）
  - lr_curve.png          —— learning rate 演化

用法:
    python tools/plot_reward.py tmp/grpo/20260519_010000/
    python tools/plot_reward.py tmp/grpo/20260519_010000/ --no-plot   # 仅打印摘要

仅需 matplotlib（已是 Python 数据栈标配）。
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: 需要 matplotlib。pip install matplotlib", file=sys.stderr)
    sys.exit(1)


# ── 从 main_process.log 提取 step metrics ─────────────────────────

# verl 每个 step 在一行里 dump 全部指标，格式形如：
#   ... step:8 - global_seqlen/min:202651 - ... - critic/rewards/mean:0.577 - ...
_STEP_LINE_RE = re.compile(r"step:(\d+)\s+-")

# 同步抓所有"key:value" 对。value 形态可能是：
#   - 普通数值: 0.5713
#   - numpy 包装: np.float64(0.5713)
#   - 整型 numpy: np.int64(8)
# Key 用白名单，避免抓到无关 timing_s/* 等噪声字段（也可以全抓，但会让字典爆炸）。
_METRIC_KEYS_TRAIN = {
    "critic/score/mean", "critic/score/max", "critic/score/min",
    "critic/rewards/mean", "critic/rewards/max", "critic/rewards/min",
    "critic/advantages/mean", "critic/advantages/max", "critic/advantages/min",
    "critic/returns/mean", "critic/returns/max", "critic/returns/min",
    "actor/pg_loss", "actor/loss", "actor/entropy",
    "actor/grad_norm", "actor/lr", "actor/pg_clipfrac", "actor/ppo_kl",
    "response_length/mean", "response_length/max", "response_length/min",
    "prompt_length/mean", "prompt_length/max", "prompt_length/min",
    "training/global_step", "training/epoch",
}
# val 指标（仅当训练开了 --verl-disable-validation false 时存在）
_METRIC_KEYS_VAL = {
    "val-core/reward/mean", "val-core/reward/max", "val-core/reward/min",
    "val-core/score/mean", "val/test_score",
}
_ALL_METRIC_KEYS = _METRIC_KEYS_TRAIN | _METRIC_KEYS_VAL

# 字符类要严格——key 里可能含 / 和 -，value 里可能含 . 和 e 和 +/-
_METRIC_RE = re.compile(
    r"([\w/\-]+):"                            # key
    r"((?:np\.(?:float|int)\d*\()?"           # 可选 np.floatX( np.intX(
    r"[+\-\d.eE]+"                            # 数值
    r"\)?)"                                   # 可选闭合括号
)


def load_step_metrics(path: Path) -> List[Dict]:
    """从 main_process.log 中抓所有 ``step:N -`` 行的指标。

    实现细节：
      1. 同一 step 可能被多行 verl 重复打印（不同 callback 触发）；保留最后一条（最完整）。
      2. value 字符串可能形如 ``np.float64(0.5713)``，做剥壳处理。
      3. 仅保留白名单 key（_ALL_METRIC_KEYS）；其它 timing/perf 噪声字段忽略。
    """
    if not path.exists():
        return []
    by_step: Dict[int, Dict] = {}
    with path.open(encoding="utf-8", errors="replace") as fp:
        for ln in fp:
            m = _STEP_LINE_RE.search(ln)
            if not m:
                continue
            step_num = int(m.group(1))
            metrics: Dict = {"step": step_num}
            for mm in _METRIC_RE.finditer(ln):
                key = mm.group(1)
                if key not in _ALL_METRIC_KEYS:
                    continue
                v = mm.group(2)
                # 剥 np.float64( ... ) 包裹
                if v.startswith("np."):
                    v = v[v.index("(") + 1: v.rindex(")")]
                try:
                    metrics[key] = float(v)
                except ValueError:
                    pass
            if len(metrics) > 1:
                by_step[step_num] = metrics
    return [by_step[k] for k in sorted(by_step.keys())]


# ── 4 张图 ───────────────────────────────────────────────────────────

def plot_reward_curve(steps: List[Dict], outdir: Path) -> None:
    """train reward + (val reward if any) + advantage 双子图。"""
    if not steps:
        return
    xs = [s["step"] for s in steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # 上图：train reward + val reward + min/max 范围
    rm = [s.get("critic/rewards/mean", s.get("critic/score/mean")) for s in steps]
    rmn = [s.get("critic/score/min") for s in steps]
    rmx = [s.get("critic/score/max") for s in steps]
    ax1.plot(xs, rm, label="train reward (mean)", linewidth=2, color="C0")
    if any(v is not None for v in rmn) and any(v is not None for v in rmx):
        # 仅把存在 min/max 的 step 画 fill；None 会让 plot 出错，但 matplotlib 容忍 NaN
        rmn_arr = [v if v is not None else float("nan") for v in rmn]
        rmx_arr = [v if v is not None else float("nan") for v in rmx]
        ax1.plot(xs, rmn_arr, alpha=0.4, linewidth=0.8, linestyle="--", color="C0")
        ax1.plot(xs, rmx_arr, alpha=0.4, linewidth=0.8, linestyle="--", color="C0")
        ax1.fill_between(xs, rmn_arr, rmx_arr, alpha=0.1, color="C0")

    # val reward 叠加（如果有）—— verl 默认每 N step 跑一次 val，会让序列稀疏
    val_xs, val_ys = [], []
    for s in steps:
        for k in ("val-core/reward/mean", "val-core/score/mean", "val/test_score"):
            v = s.get(k)
            if v is not None:
                val_xs.append(s["step"])
                val_ys.append(v)
                break
    if val_xs:
        ax1.plot(val_xs, val_ys, label="val reward (mean)", linewidth=2,
                 color="C3", marker="o", markersize=4)

    ax1.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")
    ax1.set_ylabel("reward")
    ax1.set_title("Reward evolution (train + val if enabled)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下图：advantage mean / min / max
    am = [s.get("critic/advantages/mean") for s in steps]
    amx = [s.get("critic/advantages/max") for s in steps]
    amn = [s.get("critic/advantages/min") for s in steps]
    ax2.plot(xs, am, label="advantage mean", linewidth=2, color="C1")
    if any(v is not None for v in amx):
        ax2.plot(xs, amx, label="advantage max", alpha=0.5, linestyle="--", color="C1")
    if any(v is not None for v in amn):
        ax2.plot(xs, amn, label="advantage min", alpha=0.5, linestyle="--", color="C1")
    ax2.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_xlabel("step")
    ax2.set_ylabel("advantage")
    ax2.set_title("Advantage signal strength (GRPO 优化效率的核心指标)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "reward_curve.png", dpi=110)
    plt.close(fig)


def plot_actor_health(steps: List[Dict], outdir: Path) -> None:
    """actor pg_loss / entropy / grad_norm —— mode collapse 与训练健康度监控。"""
    if not steps:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    xs = [s["step"] for s in steps]
    pgl = [s.get("actor/pg_loss") for s in steps]
    ent = [s.get("actor/entropy") for s in steps]
    gn = [s.get("actor/grad_norm") for s in steps]

    axes[0].plot(xs, pgl, color="C0", linewidth=2)
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("pg_loss")
    axes[0].set_title("Actor pg_loss (健康训练时应在 0 附近波动)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, ent, color="C1", linewidth=2)
    axes[1].set_ylabel("entropy")
    axes[1].set_title("Actor entropy (持续大幅下降 = mode collapse 预警)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs, gn, color="C2", linewidth=2)
    axes[2].set_ylabel("grad_norm")
    axes[2].set_xlabel("step")
    axes[2].set_title("Actor grad_norm (太小 = policy 几乎不动 / 太大 = unstable)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "actor_health.png", dpi=110)
    plt.close(fig)


def plot_response_length(steps: List[Dict], outdir: Path) -> None:
    """response length 演化——length hack 监控。"""
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    xs = [s["step"] for s in steps]
    rl = [s.get("response_length/mean") for s in steps]
    pl = [s.get("prompt_length/mean") for s in steps]
    ax.plot(xs, rl, label="response length", linewidth=2, color="C0")
    if any(p is not None for p in pl):
        ax2 = ax.twinx()
        ax2.plot(xs, pl, label="prompt length", color="C3", alpha=0.6, linewidth=1.5)
        ax2.set_ylabel("prompt length", color="C3")
    ax.set_xlabel("step")
    ax.set_ylabel("response length")
    ax.set_title("Response length evolution (单调升 / 单调降 = length hack 预警)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "response_length.png", dpi=110)
    plt.close(fig)


def plot_lr_curve(steps: List[Dict], outdir: Path) -> None:
    """learning rate 演化——确认 scheduler 工作正常。"""
    if not steps:
        return
    xs = [s["step"] for s in steps]
    lrs = [s.get("actor/lr") for s in steps]
    if not any(v is not None for v in lrs):
        return
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(xs, lrs, color="C4", linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "lr_curve.png", dpi=110)
    plt.close(fig)


# ── 摘要文本 ─────────────────────────────────────────────────────────

def print_summary(steps: List[Dict]) -> None:
    """打印关键指标摘要——不画图也能看健康度。"""
    if not steps:
        print("WARNING: 未抓到任何 step 指标。可能 main_process.log 还没写入 step 数据。")
        return

    print()
    print(f"=== 训练规模 ===")
    print(f"  抓到 step 数: {len(steps)}")
    print(f"  step 范围: [{steps[0]['step']}, {steps[-1]['step']}]")

    print()
    print("=== Reward 演化 ===")
    rmeans = [s.get("critic/rewards/mean", s.get("critic/score/mean")) for s in steps]
    rmeans = [r for r in rmeans if r is not None]
    if len(rmeans) >= 2:
        delta = rmeans[-1] - rmeans[0]
        sign = "↑" if delta > 0 else "↓"
        print(f"  train reward: 起点={rmeans[0]:.4f} → 终点={rmeans[-1]:.4f}  delta={delta:+.4f} {sign}")
        print(f"  train reward: min={min(rmeans):.4f}  max={max(rmeans):.4f}  stdev={statistics.pstdev(rmeans):.4f}")
        if abs(delta) < 0.02:
            print("  WARNING: train reward delta < 0.02，可能训练无效或 step 数不够")

    # val 指标（如有）
    val_ys = []
    for s in steps:
        for k in ("val-core/reward/mean", "val-core/score/mean", "val/test_score"):
            v = s.get(k)
            if v is not None:
                val_ys.append(v)
                break
    if val_ys:
        if len(val_ys) >= 2:
            delta = val_ys[-1] - val_ys[0]
            sign = "↑" if delta > 0 else "↓"
            print(f"  val reward:   起点={val_ys[0]:.4f} → 终点={val_ys[-1]:.4f}  delta={delta:+.4f} {sign}")
            print(f"  val 样本数 = {len(val_ys)}")
            if delta <= 0 < (rmeans[-1] - rmeans[0]) if rmeans else 0:
                print("  WARNING: val reward 不涨而 train reward 涨 = 过拟合 judge / reward hacking")
    else:
        print("  (no val data — 训练时未开启 --verl-disable-validation false)")

    print()
    print("=== Actor 健康度 ===")
    ents = [s.get("actor/entropy") for s in steps]
    ents = [e for e in ents if e is not None]
    if len(ents) >= 2:
        delta = ents[-1] - ents[0]
        sign = "↓" if delta < 0 else "↑"
        warn = ""
        if delta < -0.05:
            warn = "  (mode collapse 预警: entropy 大幅下降)"
        print(f"  entropy:     起点={ents[0]:.4f} → 终点={ents[-1]:.4f}  delta={delta:+.4f} {sign}{warn}")

    gns = [s.get("actor/grad_norm") for s in steps]
    gns = [g for g in gns if g is not None]
    if gns:
        avg_gn = statistics.mean(gns)
        warn = ""
        if avg_gn < 0.05:
            warn = "  (极小 → policy 几乎不动)"
        elif avg_gn > 10:
            warn = "  (太大 → unstable，可能梯度爆炸)"
        print(f"  grad_norm:   mean={avg_gn:.5f}  range=[{min(gns):.4f}, {max(gns):.4f}]{warn}")

    pgs = [s.get("actor/pg_loss") for s in steps]
    pgs = [p for p in pgs if p is not None]
    if pgs:
        print(f"  pg_loss:     mean={statistics.mean(pgs):+.5f}  range=[{min(pgs):+.4f}, {max(pgs):+.4f}]")

    print()
    print("=== Response 长度 ===")
    rls = [s.get("response_length/mean") for s in steps]
    rls = [r for r in rls if r is not None]
    if rls:
        delta = rls[-1] - rls[0] if len(rls) >= 2 else 0
        print(f"  response_length mean: 起点={rls[0]:.0f} → 终点={rls[-1]:.0f}  delta={delta:+.0f}")
        if abs(delta) > rls[0] * 0.3:
            print("  WARNING: response length 变化 > 30%，可能 length hack")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="GRPO 训练日志目录，如 tmp/grpo/20260519_010000/",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="PNG 输出目录（默认: run_dir/plots/）",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="仅打印摘要不画图（适合 ssh 终端快速 check）",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} 不是有效目录", file=sys.stderr)
        return 2

    main_log_path = run_dir / "main_process.log"
    if not main_log_path.exists():
        print(f"ERROR: {main_log_path} 不存在", file=sys.stderr)
        return 2

    print(f"读取: {main_log_path}")
    steps = load_step_metrics(main_log_path)

    print_summary(steps)

    if args.no_plot:
        return 0

    if not steps:
        print("未抓到任何 step 指标，跳过绘图。")
        return 0

    outdir: Path = args.output or (run_dir / "plots")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"输出 PNG 到: {outdir}")

    plot_reward_curve(steps, outdir)
    plot_actor_health(steps, outdir)
    plot_response_length(steps, outdir)
    plot_lr_curve(steps, outdir)

    print("绘图完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

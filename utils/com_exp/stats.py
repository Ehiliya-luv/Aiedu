# -*- coding: utf-8 -*-
"""统计推断模块：Bradley-Terry 模型拟合、Bootstrap 置信区间、Holm-Bonferroni 校正。"""

import itertools
from typing import List, Optional, Tuple

import choix
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def _prepare_choix_data(win_matrix: pd.DataFrame) -> Tuple[List[tuple], List[str]]:
    """将胜负矩阵转换为 choix 所需的格式。"""
    models = win_matrix.index.tolist()
    pairs = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                for _ in range(int(win_matrix.loc[m1, m2])):
                    pairs.append((i, j))
                for _ in range(int(win_matrix.loc[m2, m1])):
                    pairs.append((j, i))
    return pairs, models


def fit_bradley_terry_params(win_matrix: pd.DataFrame) -> pd.Series:
    """使用 choix 库拟合 Bradley-Terry 模型，返回原始 latent strength。

    Returns:
        各模型的 Bradley-Terry latent strength theta。

    Raises:
        RuntimeError: 数据不足以拟合
    """
    pairs, models = _prepare_choix_data(win_matrix)

    if not pairs:
        raise RuntimeError("Bradley-Terry 拟合失败：pairs 为空，win_matrix 无有效数据")

    if len(models) < 2:
        raise RuntimeError("Bradley-Terry 拟合失败：模型数量 < 2，无法进行对比")

    params = choix.ilsr_pairwise(n_items=len(models), data=pairs)
    return pd.Series(params, index=models)


def compute_bt_score(params: pd.Series) -> pd.Series:
    """将 BT latent strength 转换为当前模型池内的相对强度份额（和为 1）。"""
    shifted = params - params.max()
    scores = np.exp(shifted)
    scores = scores / scores.sum()
    return pd.Series(scores, index=params.index)


def compute_anchor_score(params: pd.Series, anchor_models: List[str]) -> pd.Series:
    """计算各模型面对 anchor 集合的平均 BT 预期胜率。"""
    missing = [model for model in anchor_models if model not in params.index]
    if missing:
        raise ValueError(f"anchor_models 不在模型列表中: {missing}")

    anchor_params = params.loc[anchor_models]
    scores = {
        model: float(expit(params.loc[model] - anchor_params).mean())
        for model in params.index
    }
    return pd.Series(scores, index=params.index)


def fit_bradley_terry(win_matrix: pd.DataFrame) -> pd.Series:
    """使用 choix 库拟合 Bradley-Terry 模型，返回 BT_Score。"""
    params = fit_bradley_terry_params(win_matrix)
    return compute_bt_score(params)


def _validate_anchor_models(models: List[str], anchor_models: Optional[List[str]]) -> List[str]:
    if not anchor_models:
        return []

    anchors = []
    for model in anchor_models:
        if model and model not in anchors:
            anchors.append(model)

    missing = [model for model in anchors if model not in models]
    if missing:
        raise ValueError(
            f"anchor model 不在当前比较模型中: {missing}。"
            f"可用模型: {models}"
        )
    return anchors


def run_bootstrap(
    win_matrix: pd.DataFrame,
    n_boot: int = 10000,
    anchor_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Bootstrap 置信区间（基于 choix 拟合）。

    Args:
        win_matrix:     胜负矩阵
        n_boot:         Bootstrap 迭代次数
        anchor_models:  用于 Anchor_Score 的基准模型列表

    Returns:
        DataFrame with BT_Score/BT_CI_* and optional Anchor_Score/Anchor_CI_*.
    """
    print(f"[统计推断] 开始 Bootstrap 重采样 ({n_boot} 次)...")
    models = win_matrix.index.tolist()
    anchors = _validate_anchor_models(models, anchor_models)

    total_votes = (win_matrix > 0).sum().sum()
    if total_votes == 0:
        raise RuntimeError(
            f"win_matrix 全为 0，无法进行 Bootstrap 分析。\n"
            f"shape: {win_matrix.shape}\n内容:\n{win_matrix}"
        )

    if len(models) < 2:
        raise RuntimeError(f"模型数量不足（{len(models)} < 2）")

    all_bt_scores = []
    all_anchor_scores = []
    failed_count = 0

    for iteration in range(n_boot):
        boot_matrix = pd.DataFrame(0, index=models, columns=models)

        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i < j:
                    total = win_matrix.loc[m1, m2] + win_matrix.loc[m2, m1]
                    if total > 0:
                        p = win_matrix.loc[m1, m2] / total
                        wins_boot = np.random.binomial(total, p)
                        boot_matrix.loc[m1, m2] = wins_boot
                        boot_matrix.loc[m2, m1] = total - wins_boot

        try:
            params = fit_bradley_terry_params(boot_matrix)
            bt_scores = compute_bt_score(params)
            if bt_scores is not None and len(bt_scores) > 0:
                all_bt_scores.append(bt_scores)
                if anchors:
                    all_anchor_scores.append(compute_anchor_score(params, anchors))
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
            if iteration < 5:
                print(f"[Bootstrap 迭代 {iteration+1}] 拟合失败: {str(e)[:80]}")

        if (iteration + 1) % 2000 == 0:
            print(f"[Bootstrap] 进度：{iteration + 1}/{n_boot} (成功: {len(all_bt_scores)}, 失败: {failed_count})")

    if not all_bt_scores:
        raise RuntimeError(
            f"Bootstrap 重采样全部失败 ({failed_count}/{n_boot} 次)。\n"
            f"win_matrix 非零元素数: {(win_matrix > 0).sum().sum()}\n"
            f"win_matrix 总和: {win_matrix.sum().sum()}"
        )

    if failed_count > 0:
        print(f"[警告] Bootstrap 部分失败：{len(all_bt_scores)}/{n_boot} 成功，{failed_count} 次失败")
    else:
        print(f"[Bootstrap] 全部成功 {len(all_bt_scores)}/{n_boot} 次")

    df_bt_scores = pd.DataFrame(all_bt_scores)
    results = pd.DataFrame({
        "BT_Score": df_bt_scores.mean(),
        "BT_CI_Lower": df_bt_scores.quantile(0.025),
        "BT_CI_Upper": df_bt_scores.quantile(0.975),
    })

    if anchors:
        df_anchor_scores = pd.DataFrame(all_anchor_scores)
        results["Anchor_Score"] = df_anchor_scores.mean()
        results["Anchor_CI_Lower"] = df_anchor_scores.quantile(0.025)
        results["Anchor_CI_Upper"] = df_anchor_scores.quantile(0.975)

    return results


def _compute_pvalue(boot_results: pd.DataFrame, m1: str, m2: str) -> float:
    """基于 Bootstrap 置信区间近似计算双侧 p 值。"""
    diff_obs = boot_results.loc[m1, "BT_Score"] - boot_results.loc[m2, "BT_Score"]

    se1 = (boot_results.loc[m1, "BT_CI_Upper"] - boot_results.loc[m1, "BT_CI_Lower"]) / (2 * 1.96)
    se2 = (boot_results.loc[m2, "BT_CI_Upper"] - boot_results.loc[m2, "BT_CI_Lower"]) / (2 * 1.96)
    se_diff = np.sqrt(se1**2 + se2**2)

    if se_diff < 1e-10:
        return 1.0

    z = abs(diff_obs) / se_diff
    return min(2 * (1 - norm.cdf(z)), 1.0)


def run_holm_correction(
    win_matrix: pd.DataFrame,
    boot_results: pd.DataFrame,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Holm-Bonferroni 多重检验校正。"""
    print("[统计推断] 开始 Holm-Bonferroni 多重校正...")
    models = win_matrix.index.tolist()

    if len(models) < 2:
        return pd.DataFrame(columns=["Pair", "Raw_P", "Corrected_P", "Significant"])

    pairs = list(itertools.combinations(models, 2))
    p_values = [_compute_pvalue(boot_results, m1, m2) for m1, m2 in pairs]

    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=significance_level, method="holm")

    results = []
    for i, (m1, m2) in enumerate(pairs):
        results.append({
            "Pair": f"{m1} vs {m2}",
            "Raw_P": round(p_values[i], 4),
            "Corrected_P": round(pvals_corrected[i], 4),
            "Significant": reject[i],
        })

    return pd.DataFrame(results)

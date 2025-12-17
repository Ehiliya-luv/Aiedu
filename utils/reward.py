import re
from typing import List, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from difflib import SequenceMatcher

# 默认轻量句向量模型，可按需替换为项目中的 tokenizer/model 路径
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_tokenizer_and_model(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """
    返回 (tokenizer, model, device)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model, device


def _tokens(text: str, tokenizer) -> List[str]:
    return tokenizer.tokenize(text, add_special_tokens=False)


def _token_embedding(token: str, tokenizer, model, device: str):
    """
    从模型 embedding 层取 token 的 embedding（找不到则返回零向量）
    """
    emb_layer = model.get_input_embeddings()
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, (list, tuple)):
        token_id = token_id[0]
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        return torch.zeros(emb_layer.embedding_dim, device=device)
    return emb_layer.weight[token_id].to(device)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.norm().item() == 0 or b.norm().item() == 0:
        return 0.0
    return float((a @ b).item() / (a.norm().item() * b.norm().item()))


def compute_basic_reward(original: str, revised: str,
                         tokenizer=None, model=None, device: str = "cpu",
                         model_name: str = DEFAULT_MODEL) -> float:
    """
    基础策略（token 级别对齐并计算平均余弦相似度）：
      - token 集合并行对齐，相同 token 在同一行，缺失处填 0 向量；
      - 计算每行余弦相似度并取平均，映射到 [0,1]；
      - 删除返回较低 reward，完全相同返回 1.0。
    """
    if original.strip() == revised.strip():
        return 1.0
    if len(revised.strip()) == 0:
        return 0.05

    if tokenizer is None or model is None:
        tokenizer, model, device = load_tokenizer_and_model(model_name, device)

    toks_o = _tokens(original, tokenizer)
    toks_r = _tokens(revised, tokenizer)

    # 保持原文顺序优先的 union（去重）
    union_tokens = list(dict.fromkeys(toks_o + toks_r))

    embs_o = []
    embs_r = []
    for t in union_tokens:
        emb_t = _token_embedding(t, tokenizer, model, device)
        embs_o.append(emb_t if t in toks_o else torch.zeros_like(emb_t))
        embs_r.append(emb_t if t in toks_r else torch.zeros_like(emb_t))

    sims = []
    for a, b in zip(embs_o, embs_r):
        if torch.all(a == 0) and torch.all(b == 0):
            sims.append(0.0)
            continue
        denom = (a.norm() * b.norm()).item()
        if denom == 0:
            sims.append(0.0)
        else:
            sims.append((a @ b).item() / denom)

    avg_cos = float(np.mean(sims)) if sims else 0.0
    reward = max(0.0, min(1.0, (avg_cos + 1.0) / 2.0))
    return reward


def _align_dp(toks_a: List[str], toks_b: List[str], tokenizer, model, device: str,
              gap_penalty: float = -0.5) -> Tuple[List[Tuple[Optional[str], Optional[str]]], float]:
    """
    简单的基于 embedding 的 Needleman–Wunsch 风格对齐（最大化打分）：
    使用 token embedding 的余弦作为 match 得分，gap 使用固定惩罚。
    返回对齐对列与平均对齐得分（在 [-1,1]）。
    """
    na, nb = len(toks_a), len(toks_b)
    embs_a = [_token_embedding(t, tokenizer, model, device) for t in toks_a]
    embs_b = [_token_embedding(t, tokenizer, model, device) for t in toks_b]

    dp = [[-1e9] * (nb + 1) for _ in range(na + 1)]
    bt = [[None] * (nb + 1) for _ in range(na + 1)]
    dp[0][0] = 0.0
    for i in range(1, na + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
        bt[i][0] = (i-1, 0)
    for j in range(1, nb + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty
        bt[0][j] = (0, j-1)

    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            match_score = _cosine(embs_a[i-1], embs_b[j-1])
            vals = [
                (dp[i-1][j-1] + match_score, (i-1, j-1)),
                (dp[i-1][j] + gap_penalty, (i-1, j)),
                (dp[i][j-1] + gap_penalty, (i, j-1))
            ]
            best_val, best_bt = max(vals, key=lambda x: x[0])
            dp[i][j] = best_val
            bt[i][j] = best_bt

    i, j = na, nb
    alignment = []
    scores = []
    while i > 0 or j > 0:
        pi, pj = bt[i][j]
        if pi is None and pj is None:
            break
        if pi == i-1 and pj == j-1:
            alignment.append((toks_a[i-1], toks_b[j-1]))
            scores.append(_cosine(embs_a[i-1], embs_b[j-1]))
        elif pi == i-1 and pj == j:
            alignment.append((toks_a[i-1], None))
            scores.append(gap_penalty)
        else:
            alignment.append((None, toks_b[j-1]))
            scores.append(gap_penalty)
        i, j = pi, pj
    alignment.reverse()
    avg_score = float(np.mean(scores)) if scores else 0.0
    return alignment, avg_score


def _number_entities(text: str) -> List[str]:
    """
    简单识别数值/单位实体（数字、百分号、常见单位）
    """
    return re.findall(r"\d+(?:\.\d+)?%?|(?:\d+(?:\.\d+)?\s*(?:mg|ml|g|mmol|cm|mm|μg|kg|L|ml))", text, flags=re.I)


def compute_advanced_reward(original: str, revised: str,
                            tokenizer=None, model=None, device: str = "cpu",
                            model_name: str = DEFAULT_MODEL) -> float:
    """
    进阶策略：
      - 基于 embedding 的 DP 序列对齐得到局部余弦平均（主要分数）；
      - 检测数值/单位实体更改（若发生则强惩罚）；
      - 使用全文字符串相似度作平滑。
    返回值映射到 [0,1]。
    """
    if original.strip() == revised.strip():
        return 1.0
    if tokenizer is None or model is None:
        tokenizer, model, device = load_tokenizer_and_model(model_name, device)

    toks_o = _tokens(original, tokenizer)
    toks_r = _tokens(revised, tokenizer)

    if len(toks_r) == 0:
        return 0.03

    _, avg_cos = _align_dp(toks_o, toks_r, tokenizer, model, device, gap_penalty=-0.6)
    align_score = max(0.0, min(1.0, (avg_cos + 1.0) / 2.0))

    nums_o = _number_entities(original)
    nums_r = _number_entities(revised)
    entity_factor = 0.4 if nums_o != nums_r else 1.0

    seq_ratio = SequenceMatcher(None, original, revised).ratio()
    ratio_factor = max(0.0, min(1.0, seq_ratio))

    reward = align_score * entity_factor * (0.6 * ratio_factor + 0.4)
    reward = max(0.0, min(1.0, reward))
    return reward


__all__ = [
    "load_tokenizer_and_model",
    "compute_basic_reward",
    "compute_advanced_reward",
]
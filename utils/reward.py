import re
from typing import List, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from difflib import SequenceMatcher
import logging

# 导入新的reward实现
from .reward_new import (
    compute_advanced_reward as compute_advanced_reward_new,
    TrainableRewardWeights,
)

# 默认轻量句向量模型，可按需替换为项目中的 tokenizer/model 路径
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
logger = logging.getLogger(__name__)


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


def compute_advanced_reward(original: str, revised: str,
                            tokenizer=None, model=None, device: str = "cpu",
                            model_name: str = DEFAULT_MODEL) -> float:
    """
    进阶医学NER+BertScore Reward 计算：
    使用医学NER识别关键实体，计算实体相似度和BertScore相似度，
    然后使用可训练权重组合两个相似度。
    
    该函数调用reward_new.py中的新实现。
    返回值映射到 [0,1]。
    """
    return compute_advanced_reward_new(
        original=original,
        revised=revised,
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_name=model_name,
        lambda_e_init=0.5,
        lambda_t_init=0.5
    )



__all__ = [
    "load_tokenizer_and_model",
    "compute_basic_reward",
    "compute_advanced_reward",
]
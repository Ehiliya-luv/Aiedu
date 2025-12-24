import re
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from difflib import SequenceMatcher
from bert_score import score as bert_score_compute
import logging

logger = logging.getLogger(__name__)

# 默认轻量句向量模型，可按需替换为项目中的 tokenizer/model 路径
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 医学NER模型（使用biomedical embeddings）
MEDICAL_NER_MODEL = "NeuML/pubmedbert-base-embeddings"


def load_tokenizer_and_model(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """
    返回 (tokenizer, model, device)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model, device


def load_medical_model(model_name: str = MEDICAL_NER_MODEL, device: str = "cpu"):
    """
    加载医学领域的embeddings模型（PubMedBERT）
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        logger.warning(f"无法加载医学模型 {model_name}: {e}，降级到默认模型")
        return load_tokenizer_and_model(DEFAULT_MODEL, device)


def _tokens(text: str, tokenizer) -> List[str]:
    """tokenize文本，不添加特殊符号"""
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
    """计算两个向量的余弦相似度"""
    if a.norm().item() == 0 or b.norm().item() == 0:
        return 0.0
    return float((a @ b).item() / (a.norm().item() * b.norm().item()))


def _token_align(toks_o: List[str], toks_r: List[str], tokenizer, model, device: str) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    对齐两个token序列，返回 (token_original, token_revised) 对列表
    使用简单的LCS启发式对齐方法
    """
    from difflib import SequenceMatcher
    
    matcher = SequenceMatcher(None, toks_o, toks_r)
    matching_blocks = matcher.get_matching_blocks()
    
    alignment = []
    o_idx = 0
    r_idx = 0
    
    for block in matching_blocks:
        # 对齐匹配块之前的token（一对一对齐或gap）
        o_gap = block.a - o_idx
        r_gap = block.b - r_idx
        
        for i in range(max(o_gap, r_gap)):
            o_tok = toks_o[o_idx + i] if i < o_gap else None
            r_tok = toks_r[r_idx + i] if i < r_gap else None
            alignment.append((o_tok, r_tok))
        
        # 对齐匹配块
        for i in range(block.size):
            alignment.append((toks_o[block.a + i], toks_r[block.b + i]))
        
        o_idx = block.a + block.size
        r_idx = block.b + block.size
    
    return alignment


def _extract_medical_entities(text: str, tokenizer=None, model=None, device: str = "cpu") -> List[Dict]:
    """
    基于正则表达式和启发式规则提取医学实体
    返回 [{"text": entity_text, "start": idx, "end": idx, "type": entity_type}, ...]
    
    医学实体类型：
    - DOSAGE: 药物剂量（如 10mg, 50%）
    - DRUG: 药物名称
    - SYMPTOM: 症状
    - DISEASE: 疾病名
    - MEASUREMENT: 测量值和单位
    """
    entities = []
    
    # 剂量模式：数字 + 单位
    dosage_patterns = [
        (r'\d+(?:\.\d+)?\s*(?:mg|ml|g|kg|mmol|μg|ug|mcg|IU|%|percent)', 'DOSAGE'),
        (r'\d+(?:\.\d+)?\s*(?:次|遍|下|点|滴)', 'DOSAGE'),
    ]
    
    for pattern, entity_type in dosage_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "type": entity_type
            })
    
    # 测量值（如 100mg/day, 3x daily）
    measurement_patterns = [
        (r'\d+(?:\.\d+)?\s*(?:mg|ml|g|kg)\s*/\s*(?:day|daily|week|hour|h)', 'MEASUREMENT'),
        (r'\d+\s*x\s*(?:daily|day|week)', 'MEASUREMENT'),
    ]
    
    for pattern, entity_type in measurement_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "type": entity_type
            })
    
    # 简单的症状/疾病关键词（可扩展）
    medical_keywords = {
        'SYMPTOM': ['fever', 'pain', 'cough', 'headache', 'nausea', 'vomiting', 'dizziness',
                   '发烧', '疼痛', '咳嗽', '头痛', '恶心', '呕吐', '眩晕'],
        'DISEASE': ['diabetes', 'hypertension', 'infection', 'pneumonia', 'cancer',
                   '糖尿病', '高血压', '感染', '肺炎', '癌症'],
        'DRUG': ['aspirin', 'ibuprofen', 'metformin', 'lisinopril',
                '阿司匹林', '布洛芬', '二甲双胍', '利辛普利']
    }
    
    for entity_type, keywords in medical_keywords.items():
        for keyword in keywords:
            pattern = r'\b' + keyword + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "type": entity_type
                })
    
    # 去除重复和排序
    seen = set()
    unique_entities = []
    for ent in sorted(entities, key=lambda x: x['start']):
        key = (ent['start'], ent['end'], ent['type'])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)
    
    return unique_entities


def _compute_entity_similarity(entities_o: List[Dict], entities_r: List[Dict],
                              alignment: List[Tuple[Optional[str], Optional[str]]],
                              tokenizer, model, device: str) -> float:
    """
    计算修改前后实体的平均相似度
    1. 将每个实体映射到对齐后的token位置
    2. 对于对齐的实体对，计算其embedding的相似度
    3. 返回平均相似度
    """
    if not entities_o and not entities_r:
        return 1.0  # 都没有实体，视为完全相似
    
    if (not entities_o and entities_r) or (entities_o and not entities_r):
        return 0.0  # 一方有实体一方没有，不相似
    
    # 重构aligned tokens以便映射实体
    toks_o_aligned = [t if t is not None else "[GAP]" for t, _ in alignment]
    toks_r_aligned = [t if t is not None else "[GAP]" for _, t in alignment]
    
    entity_similarities = []
    
    # 简单匹配策略：按实体文本匹配
    matched_pairs = set()
    for ent_o in entities_o:
        for ent_r in entities_r:
            if ent_o['type'] == ent_r['type']:  # 相同类型才匹配
                # 计算entity text的相似度
                text_sim = SequenceMatcher(None, ent_o['text'].lower(), ent_r['text'].lower()).ratio()
                if text_sim > 0.5:  # 相似度阈值
                    matched_pairs.add((ent_o['text'], ent_r['text']))
                    # 计算embedding相似度
                    toks_ent_o = _tokens(ent_o['text'], tokenizer)
                    toks_ent_r = _tokens(ent_r['text'], tokenizer)
                    
                    embs_o = [_token_embedding(t, tokenizer, model, device) for t in toks_ent_o]
                    embs_r = [_token_embedding(t, tokenizer, model, device) for t in toks_ent_r]
                    
                    pair_sims = []
                    for emb_o in embs_o:
                        if torch.all(emb_o == 0):
                            continue
                        for emb_r in embs_r:
                            if torch.all(emb_r == 0):
                                continue
                            sim = _cosine(emb_o, emb_r)
                            pair_sims.append(sim)
                    
                    if pair_sims:
                        entity_similarities.append(float(np.mean(pair_sims)))
    
    # 计算未匹配实体的惩罚
    unmatched_o = len(entities_o) - len([p for p in matched_pairs])
    unmatched_r = len(entities_r) - len([p for p in matched_pairs])
    
    if entity_similarities:
        avg_sim = float(np.mean(entity_similarities))
    else:
        avg_sim = 0.0
    
    # 根据未匹配实体数量调整
    penalty = (unmatched_o + unmatched_r) / (len(entities_o) + len(entities_r) + 1e-8)
    avg_sim = avg_sim * (1.0 - 0.5 * penalty)
    
    return max(0.0, min(1.0, avg_sim))


def _compute_bert_score_similarity(original: str, revised: str, device: str = "cpu") -> float:
    """
    使用BertScore计算两个文本的整体相似度
    返回F1分数
    """
    try:
        # BertScore需要列表输入
        refs = [original]
        cands = [revised]
        
        # 使用默认的SciBERT模型（适合科学和医学文本）
        precision, recall, f1 = bert_score_compute(
            cands, refs,
            lang='en',
            verbose=False,
            device=device
        )
        
        return float(f1[0].item())  # 返回F1分数
    except Exception as e:
        logger.warning(f"BertScore计算失败: {e}，使用SequenceMatcher作为fallback")
        # Fallback: 使用字符级别的相似度
        return SequenceMatcher(None, original, revised).ratio()


def compute_advanced_reward(original: str, revised: str,
                           tokenizer=None, model=None, device: str = "cpu",
                           model_name: str = DEFAULT_MODEL,
                           lambda_e_init: float = 0.5,
                           lambda_t_init: float = 0.5) -> float:
    """
    进阶医学NER+BertScore Reward 计算：
    1. Token对齐
    2. 医学NER实体识别
    3. 计算实体相似度 r_e
    4. 计算BertScore相似度 r_t
    5. 使用可训练权重计算最终reward: r = lambda_e * r_e + lambda_t * r_t
    
    Args:
        original: 原始文本
        revised: 修改后文本
        tokenizer: tokenizer实例（可选）
        model: 模型实例（可选）
        device: 计算设备
        model_name: 模型名称
        lambda_e_init: 实体权重初始值
        lambda_t_init: 文本权重初始值
    
    Returns:
        reward分数 (0-1)
    """
    if original.strip() == revised.strip():
        return 1.0
    
    if len(revised.strip()) == 0:
        return 0.05
    
    if tokenizer is None or model is None:
        tokenizer, model, device = load_tokenizer_and_model(model_name, device)
    
    # ============ Step 1: Token对齐 ============
    toks_o = _tokens(original, tokenizer)
    toks_r = _tokens(revised, tokenizer)
    alignment = _token_align(toks_o, toks_r, tokenizer, model, device)
    
    # ============ Step 2: 医学NER实体识别 ============
    entities_o = _extract_medical_entities(original, tokenizer, model, device)
    entities_r = _extract_medical_entities(revised, tokenizer, model, device)
    
    logger.debug(f"原文实体: {entities_o}")
    logger.debug(f"修改文实体: {entities_r}")
    
    # ============ Step 3: 计算实体相似度 r_e ============
    r_e = _compute_entity_similarity(entities_o, entities_r, alignment, tokenizer, model, device)
    
    # ============ Step 4: 计算BertScore相似度 r_t ============
    r_t = _compute_bert_score_similarity(original, revised, device)
    
    # ============ Step 5: 计算加权reward ============
    # 使用softmax归一化权重
    lambda_e = lambda_e_init / (lambda_e_init + lambda_t_init)
    lambda_t = lambda_t_init / (lambda_e_init + lambda_t_init)
    
    reward = lambda_e * r_e + lambda_t * r_t
    reward = max(0.0, min(1.0, reward))
    
    logger.debug(f"r_e={r_e:.4f}, r_t={r_t:.4f}, lambda_e={lambda_e:.4f}, lambda_t={lambda_t:.4f}, reward={reward:.4f}")
    
    return float(reward)


class TrainableRewardWeights(nn.Module):
    """
    可训练的Reward权重模块
    用于与模型训练集成，动态学习lambda_e和lambda_t
    """
    def __init__(self, initial_e: float = 0.5, initial_t: float = 0.5):
        super().__init__()
        # 使用log变换使权重始终为正
        self.log_lambda_e = nn.Parameter(torch.tensor(np.log(initial_e)))
        self.log_lambda_t = nn.Parameter(torch.tensor(np.log(initial_t)))
    
    def forward(self, r_e: torch.Tensor, r_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r_e: 实体相似度张量 (batch_size,)
            r_t: 文本相似度张量 (batch_size,)
        
        Returns:
            加权reward张量 (batch_size,)
        """
        lambda_e = torch.exp(self.log_lambda_e)
        lambda_t = torch.exp(self.log_lambda_t)
        
        # 归一化
        total = lambda_e + lambda_t
        lambda_e = lambda_e / total
        lambda_t = lambda_t / total
        
        reward = lambda_e * r_e + lambda_t * r_t
        return reward
    
    def get_weights(self) -> Tuple[float, float]:
        """返回当前权重"""
        with torch.no_grad():
            lambda_e = float(torch.exp(self.log_lambda_e).item())
            lambda_t = float(torch.exp(self.log_lambda_t).item())
            total = lambda_e + lambda_t
            return lambda_e / total, lambda_t / total


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


__all__ = [
    "load_tokenizer_and_model",
    "load_medical_model",
    "compute_basic_reward",
    "compute_advanced_reward",
    "TrainableRewardWeights",
    "_extract_medical_entities",
    "_token_align",
]

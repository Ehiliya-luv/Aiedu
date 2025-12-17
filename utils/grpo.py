# -*- coding: utf-8 -*-
"""
GRPO Trainer: 基于 trl 内置 GRPOTrainer 的封装实现。
参考 resources/grpo_train.py，使用 trl.GRPOConfig 和 trl.GRPOTrainer。
reward 使用 utils.reward 中的 compute_basic_reward / compute_advanced_reward。
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType, get_peft_model

from .reward import compute_basic_reward, compute_advanced_reward

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class GRPOScriptArguments:
    """
    GRPO 训练脚本参数配置。
    """
    model_name_or_path: Optional[str] = field(
        default="output/sft_model",
        metadata={"help": "模型名称或路径"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "tokenizer 的名称或路径，若为空则使用 model_name_or_path"}
    )
    dataset_name: Optional[str] = field(
        default="data/rl_train.jsonl",
        metadata={"help": "训练数据集路径（jsonl 或 json 格式）"}
    )
    train_samples: Optional[int] = field(
        default=-1,
        metadata={"help": "训练样本数，-1 表示全部"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "数据预处理的工作进程数"}
    )
    output_dir: Optional[str] = field(
        default="output/rl_model",
        metadata={"help": "模型输出目录"}
    )
    
    # GRPO 训练参数
    num_train_epochs: int = field(default=1, metadata={"help": "训练轮数"})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "每个设备的 batch size"})
    per_device_eval_batch_size: int = field(default=4, metadata={"help": "评估 batch size"})
    learning_rate: float = field(default=1e-5, metadata={"help": "学习率"})
    weight_decay: float = field(default=0.0, metadata={"help": "权重衰减"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "梯度裁剪"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "学习率调度器类型"})
    warmup_steps: int = field(default=0, metadata={"help": "预热步数"})
    
    # RL 特定参数
    reward_type: str = field(
        default="advanced",
        metadata={"help": "reward 类型：basic 或 advanced"}
    )
    entropy_coef: float = field(default=0.01, metadata={"help": "熵系数"})
    kl_penalty: str = field(default="kl", metadata={"help": "KL 惩罚类型"})
    use_reference_model: bool = field(default=False, metadata={"help": "是否使用参考模型"})
    reference_model_path: Optional[str] = field(default=None, metadata={"help": "参考模型路径"})
    
    # QLoRA 参数
    use_qlora: bool = field(default=False, metadata={"help": "是否使用 QLoRA"})
    lora_r: int = field(default=16, metadata={"help": "LoRA 秩"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    
    # 生成参数
    max_new_tokens: int = field(default=64, metadata={"help": "最大生成 token 数"})
    temperature: float = field(default=1.0, metadata={"help": "生成温度"})
    top_p: float = field(default=0.95, metadata={"help": "top-p 采样"})
    
    # 其他
    seed: int = field(default=42, metadata={"help": "随机种子"})
    logging_steps: int = field(default=10, metadata={"help": "日志间隔"})
    save_steps: int = field(default=100, metadata={"help": "保存模型间隔"})
    eval_steps: int = field(default=100, metadata={"help": "评估间隔"})
    save_total_limit: int = field(default=3, metadata={"help": "保留模型数量"})
    use_deepspeed: bool = field(default=False, metadata={"help": "是否使用 DeepSpeed"})


def _select_reward_fn(reward_type: str) -> Callable:
    """选择 reward 函数"""
    assert reward_type in ("basic", "advanced"), "reward_type 必须为 'basic' 或 'advanced'"
    return compute_basic_reward if reward_type == "basic" else compute_advanced_reward


def load_dataset_from_path(dataset_path: str, max_samples: int = -1) -> Dataset:
    """
    从本地 jsonl 或 json 文件加载数据集。
    """
    import json
    
    data = []
    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples > 0 and i >= max_samples:
                    break
                try:
                    obj = json.loads(line.strip())
                    data.append(obj)
                except Exception as e:
                    logger.warning(f"解析行 {i} 失败: {e}")
    elif dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                data = obj[:max_samples] if max_samples > 0 else obj
            else:
                data = [obj]
    else:
        raise ValueError(f"不支持的文件格式: {dataset_path}")
    
    logger.info(f"已加载 {len(data)} 条训练数据")
    return Dataset.from_dict({
        "prompt": [item.get("prompt", item.get("text", "")) for item in data],
        "answer": [item.get("answer", item.get("response", "")) for item in data]
    })


def create_reward_fn(reward_type: str, tokenizer, device: str) -> Callable:
    """
    创建 reward 函数适配器。
    trl.GRPOTrainer 期望 reward_fn 接收 completions 列表和其他参数。
    """
    reward_fn_base = _select_reward_fn(reward_type)
    
    def reward_fn(completions, **kwargs):
        """
        completions: List[str] 或 List[Dict]
        返回: List[float]
        """
        if not completions:
            return []
        
        # 处理 trl 的 completions 格式（可能是消息格式）
        texts = []
        for comp in completions:
            if isinstance(comp, dict):
                # 消息格式 {"role": ..., "content": ...}
                texts.append(comp.get("content", ""))
            elif isinstance(comp, str):
                texts.append(comp)
            else:
                texts.append(str(comp))
        
        # 从 kwargs 中获取原始 prompt（若可用）
        prompts = kwargs.get("prompts", kwargs.get("queries", [""] * len(texts)))
        if isinstance(prompts, str):
            prompts = [prompts] * len(texts)
        
        rewards = []
        for prompt, text in zip(prompts, texts):
            try:
                reward = reward_fn_base(prompt, text, tokenizer=tokenizer, model=None, device=device)
                rewards.append(float(reward))
            except TypeError:
                # 若 reward_fn 不接受 tokenizer/model/device，尝试不传
                try:
                    reward = reward_fn_base(prompt, text)
                    rewards.append(float(reward))
                except Exception as e:
                    logger.exception(f"计算 reward 失败: {e}")
                    rewards.append(0.0)
        
        return rewards
    
    return reward_fn


def setup_model_and_tokenizer(script_args: GRPOScriptArguments, device: str = "cuda"):
    """
    加载模型与 tokenizer，支持 QLoRA。
    """
    logger.info(f"加载模型: {script_args.model_name_or_path}")
    
    # 加载 tokenizer
    tokenizer_path = script_args.tokenizer_name_or_path or script_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # QLoRA 配置
    quantization_config = None
    if script_args.use_qlora:
        logger.info("使用 QLoRA 量化")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if script_args.use_qlora else torch.float32,
        trust_remote_code=True,
    )
    
    # 应用 LoRA
    if script_args.use_qlora:
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=script_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer


def train_grpo(script_args: GRPOScriptArguments):
    """
    使用 trl.GRPOTrainer 进行 GRPO 训练的主函数。
    """
    # 设备与种子
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 加载模型与 tokenizer
    model, tokenizer = setup_model_and_tokenizer(script_args, device)
    
    # 加载数据集
    if os.path.exists(script_args.dataset_name):
        train_dataset = load_dataset_from_path(
            script_args.dataset_name,
            max_samples=script_args.train_samples
        )
    else:
        raise FileNotFoundError(f"数据集不存在: {script_args.dataset_name}")
    
    # 创建 reward 函数
    reward_fn = create_reward_fn(script_args.reward_type, tokenizer, device)
    
    # GRPO 配置
    grpo_config = GRPOConfig(
        output_dir=script_args.output_dir,
        num_train_epochs=script_args.num_train_epochs,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        max_grad_norm=script_args.max_grad_norm,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        eval_steps=script_args.eval_steps,
        save_total_limit=script_args.save_total_limit,
        seed=script_args.seed,
        entropy_coeff=script_args.entropy_coef,
        max_new_tokens=script_args.max_new_tokens,
        temperature=script_args.temperature,
        top_p=script_args.top_p,
    )
    
    # 创建 GRPO 训练器
    logger.info("初始化 GRPO 训练器")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,  # trl.GRPOTrainer 期望的参数名
    )
    
    # 开始训练
    logger.info("开始 GRPO 训练")
    train_result = trainer.train()
    
    # 保存模型
    logger.info(f"保存模型到 {script_args.output_dir}")
    trainer.save_model(script_args.output_dir)
    
    # 返回训练结果
    return train_result


class GRPOTrainerWrapper:
    """
    简化的 GRPO Trainer 包装器，兼容 main.py 的调用接口。
    """
    def __init__(self,
                 model,
                 tokenizer,
                 reward_type: str = "advanced",
                 device: str = "cuda",
                 lr: float = 1e-5,
                 entropy_coef: float = 0.01,
                 kl_coef: float = 0.0,
                 reference_model=None,
                 **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_type = reward_type
        self.device = device
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.reference_model = reference_model
        self.reward_fn = _select_reward_fn(reward_type)
        
        # 用于 REINFORCE 回退
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)
        self.running_baseline = 0.0
        self.baseline_alpha = 0.9
    
    @torch.no_grad()
    def generate(self, prompts: List[str], max_new_tokens: int = 64, 
                 temperature: float = 1.0, top_p: float = 0.95, 
                 do_sample: bool = True) -> List[str]:
        """生成文本"""
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        gen_out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )
        
        results = []
        input_ids = enc["input_ids"]
        for i, seq in enumerate(gen_out):
            prompt_len = input_ids.shape[1]
            gen_ids = seq[prompt_len:].tolist() if seq.shape[0] > prompt_len else []
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(text)
        return results
    
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> List[float]:
        """计算 reward"""
        rewards = []
        for p, r in zip(prompts, responses):
            try:
                reward = self.reward_fn(p, r, tokenizer=self.tokenizer, model=self.model, device=self.device)
            except TypeError:
                reward = self.reward_fn(p, r)
            except Exception as e:
                logger.exception(f"计算 reward 失败: {e}")
                reward = 0.0
            rewards.append(float(reward))
        return rewards
    
    def train_step(self, prompts: List[str], max_new_tokens: int = 64,
                   temperature: float = 1.0, top_p: float = 0.95) -> Dict[str, Any]:
        """执行一个训练步骤"""
        # 生成
        responses = self.generate(prompts, max_new_tokens=max_new_tokens, 
                                 temperature=temperature, top_p=top_p, do_sample=True)
        
        # 计算 reward
        rewards = self.compute_rewards(prompts, responses)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # 更新 baseline
        batch_mean = float(rewards_t.mean().item())
        self.running_baseline = self.baseline_alpha * self.running_baseline + (1 - self.baseline_alpha) * batch_mean
        advantages = rewards_t - self.running_baseline
        
        stats = {
            "reward_mean": float(rewards_t.mean().item()),
            "reward_min": float(rewards_t.min().item()),
            "reward_max": float(rewards_t.max().item()),
            "baseline": float(self.running_baseline),
        }
        return stats
    
    def save(self, path: str):
        """保存模型"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


__all__ = ["GRPOTrainer", "GRPOTrainerWrapper", "train_grpo", "GRPOScriptArguments"]
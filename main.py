# -*- coding: utf-8 -*-
"""
主入口：整合 SFT 与 GRPO-RL 模块，支持 --mode 选择运行模式。
模式说明：
  - sft: 仅运行 SFT 微调，模型保存到 output/sft_model
  - rl: 仅运行 GRPO 强化学习（从 output/sft_model 加载初始模型），模型保存到 output/rl_model
  - sft+rl: 先运行 SFT 再运行 GRPO，模型分别保存到对应目录
"""
import os
import sys
import argparse
import logging
import json
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入本地模块
from utils.data import load_and_preprocess_data
from utils.sft import create_model_and_trainer
from utils.grpo import GRPOTrainerWrapper, GRPOScriptArguments, train_grpo

# ===== 全局环境配置 =====
def setup_env():
    """统一的环境变量设置"""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    cache_dir = "/tmp/.cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    return cache_dir


def detect_gpu_config():
    """检测GPU配置并返回优化参数"""
    if not torch.cuda.is_available():
        return {
            "num_gpus": 0,
            "device": "cpu",
            "use_qlora": False,
            "batch_size": 1,
            "gradient_accumulation_steps": 8
        }

    num_gpus = torch.cuda.device_count()
    total_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)) / (1024**3)  # GB

    logger.info(f"检测到 {num_gpus} 个GPU，总显存: {total_memory:.1f}GB")

    # 根据GPU数量和显存调整配置
    if num_gpus >= 8 and total_memory >= 200:  # 8x3090配置
        config = {
            "num_gpus": num_gpus,
            "device": "cuda",
            "use_qlora": True,  # 使用量化节省显存
            "batch_size": 2,    # 每个GPU的batch size
            "gradient_accumulation_steps": 4,
            "model_parallel": True
        }
    elif num_gpus >= 4:
        config = {
            "num_gpus": num_gpus,
            "device": "cuda",
            "use_qlora": True,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "model_parallel": True
        }
    else:
        config = {
            "num_gpus": num_gpus,
            "device": "cuda",
            "use_qlora": True,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "model_parallel": False
        }

    logger.info(f"GPU配置: {config}")
    return config


def load_jsonl_texts(path: str, max_items: Optional[int] = None) -> List[str]:
    """从 jsonl 文件加载文本列表"""
    texts = []
    if not os.path.exists(path):
        logger.error(f"数据文件不存在: {path}")
        return texts
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_items is not None and len(texts) >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # 尝试从常见字段提取文本
                text = None
                for key in ("text", "input", "prompt", "question", "original"):
                    if key in obj and isinstance(obj[key], str) and obj[key].strip():
                        text = obj[key].strip()
                        break
                if text:
                    texts.append(text)
            except Exception as e:
                logger.debug(f"解析行 {i} 失败: {e}")
                texts.append(line)  # 降级：直接作为文本
    
    logger.info(f"从 {path} 加载 {len(texts)} 条文本")
    return texts


def run_sft(model_name: str, data_path: str, output_dir: str, cache_dir: str):
    """运行 SFT 微调"""
    logger.info("=" * 80)
    logger.info("🚀 开始 SFT 微调模块")
    logger.info("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载 tokenizer
    logger.info(f"🔤 加载 tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
        cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("⚠️  Pad token 未设置，使用 EOS token")
    
    # 加载并预处理数据
    logger.info(f"📊 加载训练数据: {data_path}")
    train_dataset = load_and_preprocess_data(data_path, tokenizer)
    logger.info(f"✅ 加载了 {len(train_dataset)} 条训练样本")
    
    # 创建训练器
    logger.info("🧠 创建 SFT 训练器...")
    trainer = create_model_and_trainer(
        model_name=model_name,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=output_dir,
        bf16=torch.cuda.is_bf16_supported()
    )
    
    # 开始训练
    logger.info("⚡ 开始 SFT 训练...")
    try:
        train_result = trainer.train()
        logger.info(f"✅ SFT 训练完成: {train_result}")
    except Exception as e:
        logger.exception(f"❌ SFT 训练失败: {str(e)}")
        raise
    
    # 保存模型
    logger.info("💾 保存 SFT 最终模型...")
    try:
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        logger.info(f"✅ SFT 模型已保存到 {output_dir}")
    except Exception as e:
        logger.exception(f"❌ 保存 SFT 模型失败: {str(e)}")
        raise
    
    logger.info("=" * 80)
    logger.info("✅ SFT 微调完成")
    logger.info("=" * 80)
    return output_dir


def run_rl(sft_model_path: str,
           rl_data_path: str,
           output_dir: str,
           cache_dir: str,
           reward_type: str = "advanced",
           max_items: Optional[int] = None,
           epochs: int = 1,
           batch_size: int = 4,
           learning_rate: float = 1e-5,
           max_new_tokens: int = 64,
           temperature: float = 1.0,
           top_p: float = 0.95,
           use_qlora: bool = False,
           gpu_config: dict = None):
    """运行 GRPO 强化学习"""
    logger.info("=" * 80)
    logger.info("🚀 开始 GRPO 强化学习模块")
    logger.info("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 检查 SFT 模型是否存在
    if not os.path.exists(sft_model_path):
        raise FileNotFoundError(f"SFT 模型不存在: {sft_model_path}。请先运行 SFT 模块或指定正确的模型路径")
    
    logger.info(f"📦 加载 SFT 模型: {sft_model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            device_map=device if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if use_qlora else torch.float32,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.exception(f"❌ 加载模型失败: {str(e)}")
        raise
    
    # 加载 RL 数据
    logger.info(f"📊 加载 RL 数据: {rl_data_path}")
    if not os.path.exists(rl_data_path):
        logger.error(f"RL 数据不存在: {rl_data_path}")
        raise FileNotFoundError(f"RL 数据路径不存在: {rl_data_path}")
    
    rl_texts = load_jsonl_texts(rl_data_path, max_items=max_items)
    if not rl_texts:
        logger.error("❌ 未能加载 RL 数据，请检查数据文件格式")
        raise ValueError("RL 数据为空")
    
    logger.info(f"✅ 加载了 {len(rl_texts)} 条 RL 数据")
    
    # 创建 GRPO 训练器
    logger.info("🧠 创建 GRPO 训练器...")
    try:
        # 根据GPU配置调整batch_size
        effective_batch_size = gpu_config.get("batch_size", batch_size)
        logger.info(f"使用batch_size: {effective_batch_size}")

        grpo_trainer = GRPOTrainerWrapper(
            model=model,
            tokenizer=tokenizer,
            reward_type=reward_type,
            device=device,
            lr=learning_rate,
            entropy_coef=0.01,
            kl_coef=0.0,
        )
    except Exception as e:
        logger.exception(f"❌ 创建 GRPO 训练器失败: {str(e)}")
        raise
    
    # 开始训练循环
    logger.info(f"⚡ 开始 GRPO 训练 (epochs={epochs}, batch_size={batch_size}, reward_type={reward_type})")
    
    try:
        steps_per_epoch = max(1, len(rl_texts) // batch_size)
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # 按 batch 迭代
            for batch_start in range(0, len(rl_texts), batch_size):
                batch_prompts = rl_texts[batch_start:batch_start + batch_size]
                
                try:
                    stats = grpo_trainer.train_step(
                        prompts=batch_prompts,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    
                    # 记录统计信息
                    batch_idx = batch_start // batch_size + 1
                    total_batches = steps_per_epoch
                    logger.info(
                        f"  Batch {batch_idx}/{total_batches} | "
                        f"reward_mean={stats.get('reward_mean', 0):.4f} | "
                        f"reward_max={stats.get('reward_max', 0):.4f} | "
                        f"batch_size={len(batch_prompts)}"
                    )
                except Exception as e:
                    logger.exception(f"❌ 训练 batch 失败，跳过: {str(e)}")
                    continue
            
            # 每个 epoch 保存一次模型
            logger.info(f"💾 保存 epoch {epoch + 1} 模型...")
            try:
                epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
                os.makedirs(epoch_output_dir, exist_ok=True)
                grpo_trainer.save(epoch_output_dir)
                logger.info(f"✅ 模型已保存到 {epoch_output_dir}")
            except Exception as e:
                logger.exception(f"❌ 保存模型失败: {str(e)}")
        
        # 保存最终模型到输出目录
        logger.info(f"💾 保存最终 GRPO 模型到 {output_dir}...")
        grpo_trainer.save(output_dir)
        logger.info(f"✅ 最终模型已保存到 {output_dir}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  训练被中断")
        logger.info(f"💾 保存中断时的模型...")
        grpo_trainer.save(output_dir)
    except Exception as e:
        logger.exception(f"❌ GRPO 训练过程中出错: {str(e)}")
        raise
    
    logger.info("=" * 80)
    logger.info("✅ GRPO 强化学习完成")
    logger.info("=" * 80)
    return output_dir


def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(
        description="医学考题生成优化 - SFT 与 GRPO 综合训练框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    示例用法：
    # 仅 SFT 微调
    python main.py --mode sft
    
    # 仅 GRPO 强化学习（从 SFT 模型加载）
    python main.py --mode rl
    
    # 先 SFT 后 GRPO
    python main.py --mode sft+rl
    
    # 自定义参数
    python main.py --mode rl --reward-type basic --epochs 2 --batch-size 8
            """
        )
    
    # 基础参数
    parser.add_argument(
        "--mode",
        choices=["sft", "rl", "sft+rl"],
        default="sft",
        help="运行模式：sft (仅微调) / rl (仅强化学习) / sft+rl (先微调后强化学习)"
    )
    
    # SFT 相关参数
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="基础模型名称或路径"
    )
    parser.add_argument(
        "--sft-data",
        type=str,
        default="data/sft_train.jsonl",
        help="SFT 训练数据路径"
    )
    parser.add_argument(
        "--sft-output",
        type=str,
        default="output/sft_model",
        help="SFT 模型输出路径"
    )
    
    # RL 相关参数
    parser.add_argument(
        "--rl-data",
        type=str,
        default="data/rl_train.jsonl",
        help="RL 训练数据路径"
    )
    parser.add_argument(
        "--rl-output",
        type=str,
        default="output/rl_model",
        help="RL 模型输出路径"
    )
    parser.add_argument(
        "--reward-type",
        choices=["basic", "advanced"],
        default="advanced",
        help="Reward 计算方式"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="RL 训练轮数"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="RL batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="RL 学习率"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="生成时的最大 token 数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="生成温度"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="top-p 采样参数"
    )
    parser.add_argument(
        "--max-rl-items",
        type=int,
        default=None,
        help="RL 数据最大条数（用于测试）"
    )
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        help="是否使用 QLoRA 量化"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置环境
    cache_dir = setup_env()

    # 检测GPU配置
    gpu_config = detect_gpu_config()

    logger.info("🎯 运行模式: %s", args.mode)
    logger.info("📋 配置参数: %s", vars(args))
    logger.info("🖥️  GPU配置: %s", gpu_config)
    
    try:
        if args.mode == "sft":
            # 仅 SFT
            run_sft(
                model_name=args.model_name,
                data_path=args.sft_data,
                output_dir=args.sft_output,
                cache_dir=cache_dir
            )
        
        elif args.mode == "rl":
            # 仅 RL
            run_rl(
                sft_model_path=args.sft_output,
                rl_data_path=args.rl_data,
                output_dir=args.rl_output,
                cache_dir=cache_dir,
                reward_type=args.reward_type,
                max_items=args.max_rl_items,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_qlora=args.use_qlora,
                gpu_config=gpu_config
            )
        
        elif args.mode == "sft+rl":
            # SFT + RL
            logger.info("🔗 执行 SFT + RL 综合训练流程")
            
            # 第一阶段：SFT
            sft_model_dir = run_sft(
                model_name=args.model_name,
                data_path=args.sft_data,
                output_dir=args.sft_output,
                cache_dir=cache_dir
            )
            
            # 第二阶段：RL
            run_rl(
                sft_model_path=sft_model_dir,
                rl_data_path=args.rl_data,
                output_dir=args.rl_output,
                cache_dir=cache_dir,
                reward_type=args.reward_type,
                max_items=args.max_rl_items,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_qlora=args.use_qlora,
                gpu_config=gpu_config
            )
        
        logger.info("=" * 80)
        logger.info("🎉 所有训练模块执行完成！")
        logger.info("=" * 80)
        return 0
    
    except KeyboardInterrupt:
        logger.warning("❌ 用户中断")
        return 130
    except Exception as e:
        logger.exception("❌ 执行失败: %s", str(e))
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

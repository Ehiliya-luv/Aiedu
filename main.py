# main.py
import os
import logging
import torch
from transformers import AutoTokenizer
from utils.data import load_and_preprocess_data
from utils.sft import create_model_and_trainer

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

    
if __name__ == "__main__":
    # ===== 关键环境设置 =====
    # 1. 使用国内 HF 镜像加速下载
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # 2. 设置缓存目录
    cache_dir = "/tmp/.cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    
    # 3. 设置 CUDA 环境变量
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 模型和路径配置
    MODEL_NAME = "Qwen/Qwen2.5-7B"
    DATA_PATH = "data/sft_train.jsonl"
    OUTPUT_DIR = "output/qwen2.5-med-mcq-sft"
    
    logger.info(f"🚀 Starting SFT training with model: {MODEL_NAME}")
    logger.info(f"📁 Data path: {DATA_PATH}")
    logger.info(f"💾 Output directory: {OUTPUT_DIR}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化 tokenizer
    logger.info("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False,
        cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("⚠️ Pad token not set. Using EOS token as pad token.")
    
    # 加载并预处理数据
    logger.info("📊 Loading and preprocessing data...")
    train_dataset = load_and_preprocess_data(DATA_PATH, tokenizer)
    logger.info(f"✅ Loaded {len(train_dataset)} training samples")
    
    # 创建训练器
    logger.info("🧠 Creating SFT trainer...")
    trainer = create_model_and_trainer(
        model_name=MODEL_NAME,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=OUTPUT_DIR,
        bf16=torch.cuda.is_bf16_supported()
    )
    
    # 开始训练
    logger.info("⚡ Starting SFT training...")
    try:
        train_result = trainer.train()
        logger.info(f"✅ Training completed successfully: {train_result}")
    except Exception as e:
        logger.exception(f"❌ Training failed with error: {str(e)}")
        raise
    
    # 保存最终模型
    logger.info("💾 Saving final model...")
    try:
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"✅ Training completed. Model saved to {OUTPUT_DIR}")
    except Exception as e:
        logger.exception(f"❌ Failed to save model: {str(e)}")
        raise

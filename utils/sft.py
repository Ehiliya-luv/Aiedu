# utils/sft.py
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import logging
import os

logger = logging.getLogger(__name__)

def create_model_and_trainer(
    model_name: str,
    tokenizer,
    train_dataset,
    output_dir: str,
    per_device_batch_size: int = 1,
    gradient_accum_steps: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    bf16: bool = True,
    use_gradient_checkpointing: bool = True
):
    """创建并配置 SFT 训练器，兼容最新 trl API"""
    
    # ===== 1. 修复量化配置 =====
    compute_dtype = torch.bfloat16 if bf16 and torch.cuda.is_bf16_supported() else torch.float16
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.float16
    )
    
    logger.info(f"Loading model with quantization config: {quantization_config}")
    logger.info(f"Using compute dtype: {compute_dtype}")
    
    # ===== 2. 加载模型（带内存优化） =====
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
            resume_download=True,
            local_files_only=False,
            use_cache=False if use_gradient_checkpointing else True
        )
        logger.info("✅ Model loaded successfully with quantization")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise
    
    # ===== 3. 修复生成配置问题 =====
    if hasattr(model, "generation_config") and model.generation_config is not None:
        if hasattr(model.generation_config, "quantization_config"):
            delattr(model.generation_config, "quantization_config")
        logger.info("✅ Fixed generation config to avoid serialization error")
    
    # ===== 4. 准备模型进行 4-bit 训练 =====
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    logger.info("✅ Prepared model for k-bit training")
    
    # ===== 5. LoRA 配置（优化目标模块） =====
    # 智能检测目标模块
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if hasattr(model.model.layers[0], 'mlp') and hasattr(model.model.layers[0].mlp, 'gate_proj'):
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"],
        init_lora_weights="gaussian"
    )
    
    # 应用 LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    logger.info(f"✅ Applied LoRA config: {peft_config}")
    
    # ===== 6. 训练参数（优化内存使用） =====
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accum_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="no",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.03,
        bf16=bf16 and torch.cuda.is_bf16_supported(),
        fp16=not (bf16 and torch.cuda.is_bf16_supported()),
        max_grad_norm=0.3,
        weight_decay=0.01,
        report_to="none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # 保持所有列
        seed=42
    )
    
    # ===== 7. 修复 SFTTrainer 初始化 =====
    def formatting_func(examples):
        texts = []
        for i in range(len(examples["text"])):
            text = str(examples["text"][i])
            # 确保文本长度合理
            if len(text) > 2000:
                text = text[:2000] + "..."
            texts.append(text)
        return {"text": texts}
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=formatting_func
    )
    
    return trainer
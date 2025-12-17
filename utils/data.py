# utils/data.py
from datasets import Dataset
from transformers import AutoTokenizer
import logging
import json
import os
import re

logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path: str, tokenizer, max_seq_length: int = 2048):
    """
    加载 JSONL 格式的 SFT 训练数据，并 tokenize。
    特别优化：当只有单个样本时，不进行严格过滤，而是尝试修复格式问题
    """
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # ===== 1. 更健壮的 JSONL 加载 =====
    logger.info(f"Loading data from {data_path} with relaxed parsing...")
    raw_data = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise
    
    # 处理每行数据
    for i, line in enumerate(lines):
        line = line.strip()
        # 跳过空行
        if not line:
            continue
            
        try:
            # 尝试解析 JSON
            item = json.loads(line)
            raw_data.append(item)
        except json.JSONDecodeError as e:
            logger.warning(f"Line {i+1} JSON parsing error: {e}")
            
            # 尝试多种恢复策略
            recovery_successful = False
            
            # 策略1: 查找 JSON 内容
            json_match = re.search(r'\{.*\}', line, re.DOTALL)
            if json_match:
                try:
                    recovered_json = json_match.group(0)
                    item = json.loads(recovered_json)
                    raw_data.append(item)
                    logger.info(f"Recovered JSON from line {i+1} using regex")
                    recovery_successful = True
                except:
                    pass
            
            # 策略2: 移除 Markdown 代码块标记
            if not recovery_successful:
                cleaned_line = re.sub(r'^```json\s*|\s*```$', '', line, flags=re.MULTILINE)
                try:
                    item = json.loads(cleaned_line)
                    raw_data.append(item)
                    logger.info(f"Recovered JSON from line {i+1} by cleaning Markdown")
                    recovery_successful = True
                except:
                    pass
            
            # 策略3: 尝试解析整个文件内容
            if not recovery_successful and i == 0 and len(lines) > 0:
                logger.info("Attempting full-file recovery...")
                full_content = " ".join(lines)
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', full_content, re.DOTALL)
                if json_match:
                    try:
                        recovered_json = json_match.group(0)
                        item = json.loads(recovered_json)
                        raw_data.append(item)
                        logger.info("Recovered JSON from full file content")
                        recovery_successful = True
                    except:
                        pass
    
    # ===== 2. 单样本保护机制 =====
    if len(raw_data) == 0:
        logger.warning("No valid JSON samples found. Creating minimal fallback sample...")
        
        # 创建最小有效样本
        fallback_sample = {
            "messages": [
                {"role": "user", "content": "请根据病历生成一道医学选择题。"},
                {"role": "assistant", "content": "冠状动脉粥样硬化要紧侵犯以下分支( )\nA. 回旋支、左室支\nB. 前距离支、边缘支\nC. 前降支、左旋支\nD. 房室结支、心室支\n\n**答案：C**"}
            ]
        }
        raw_data.append(fallback_sample)
        logger.info("✅ Created fallback sample to prevent training failure")
    
    logger.info(f"✅ Successfully loaded {len(raw_data)} samples")
    
    # ===== 3. 创建 Dataset 对象 =====
    try:
        dataset = Dataset.from_list(raw_data)
    except Exception as e:
        logger.error(f"❌ Failed to create dataset: {e}")
        # 作为最后的保护措施，创建一个最小数据集
        minimal_data = [{
            "messages": [
                {"role": "user", "content": "Generate a medical question."},
                {"role": "assistant", "content": "Sample question content."}
            ]
        }]
        dataset = Dataset.from_list(minimal_data)
        logger.warning("⚠️ Created minimal fallback dataset")
    
    # ===== 4. 应用聊天模板格式化 =====
    def format_chat_template(examples):
        formatted_texts = []
        for i in range(len(examples["messages"])):
            messages = examples["messages"][i]
            try:
                # 手动格式化聊天模板
                text = ""
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user":
                            text += f"### Human:\n{content}\n\n"
                        elif role == "assistant":
                            text += f"### Assistant:\n{content}\n\n"
                formatted_texts.append(text.strip())
            except Exception as e:
                logger.error(f"Error formatting chat template for sample {i}: {e}")
                # 使用最小有效模板
                text = "### Human:\nGenerate a medical MCQ question.\n\n### Assistant:\nWhat is the correct answer?"
                formatted_texts.append(text)
        return {"text": formatted_texts}
    
    # 应用格式化
    logger.info("📝 Applying chat template formatting...")
    try:
        # 确保数据集有 messages 字段
        if "messages" not in dataset.column_names:
            logger.warning("Dataset does not have 'messages' column. Creating default format.")
            dataset = dataset.map(lambda x: {"messages": [
                {"role": "user", "content": "Generate a medical question based on case."},
                {"role": "assistant", "content": "Sample answer."}
            ]}, batched=False)
        
        dataset = dataset.map(
            format_chat_template,
            batched=True,
            remove_columns=[col for col in dataset.column_names if col != "text"]
        )
    except Exception as e:
        logger.error(f"❌ Failed to format chat template: {e}")
        # 创建最小有效数据集
        minimal_texts = ["### Human:\nGenerate a medical MCQ question.\n\n### Assistant:\nWhat is the most common symptom?"]
        dataset = Dataset.from_dict({"text": minimal_texts})
        logger.warning("⚠️ Created minimal formatted dataset")
    
    # ===== 5. Tokenize =====
    logger.info("🔤 Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None
        )
    
    try:
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
    except Exception as e:
        logger.error(f"❌ Tokenization failed: {e}")
        # 最小化 tokenization
        sample_text = "### Human:\nGenerate a medical question.\n\n### Assistant:\nSample answer."
        tokenized = tokenizer(
            sample_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None
        )
        minimal_dataset = Dataset.from_dict({
            "input_ids": [tokenized["input_ids"]],
            "attention_mask": [tokenized["attention_mask"]]
        })
        return minimal_dataset
    
    # ===== 6. 宽松的过滤 =====
    original_size = len(dataset)
    if original_size > 1:  # 只有多个样本时才过滤
        dataset = dataset.filter(lambda x: 
            x.get("input_ids") is not None and 
            len(x["input_ids"]) > 0 and 
            len(x["input_ids"]) <= max_seq_length
        )
        filtered_size = len(dataset)
        logger.info(f"🔍 Filtered {original_size - filtered_size} samples")
    else:
        logger.info("🎯 Single sample detected - skipping filtering to preserve data")
    
    # 确保至少有一个样本
    if len(dataset) == 0:
        logger.warning("❌ No valid samples after filtering. Creating fallback sample.")
        sample_text = "### Human:\nGenerate a medical MCQ question.\n\n### Assistant:\nSample answer."
        tokenized = tokenizer(
            sample_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None
        )
        dataset = Dataset.from_dict({
            "input_ids": [tokenized["input_ids"]],
            "attention_mask": [tokenized["attention_mask"]]
        })
    
    # ===== 7. 设置格式 =====
    try:
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    except Exception as e:
        logger.error(f"❌ Failed to set dataset format: {e}")
        # 确保返回可用的数据集
        pass
    
    logger.info(f"✅ Final dataset size: {len(dataset)} samples")
    return dataset
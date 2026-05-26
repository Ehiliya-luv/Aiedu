# Aiedu (DPO)

本项目当前主流程已切换为 **DPO 偏好优化训练**，不再以 Reward Model + GRPO 作为默认训练路径。

## 当前流程

1. 使用 `export_rl_data.py` 从数据库抽取单题修改事件。
2. 生成 `data/dpo_pairs.jsonl`，每条样本包含：
   - `prompt`
   - `chosen`
   - `rejected`
3. 使用 `main.py` 直接对大模型进行 DPO 训练。

## 数据格式

`data/dpo_pairs.jsonl` 的单条样本结构：

```json
{
  "prompt": "...原始病历 + 题型 + 修改前题目...",
  "chosen": "...医生修改后的当前题目...",
  "rejected": "...修改前父题目...",
  "meta": {
    "sample_id": 123456,
    "question_type": "A3"
  }
}
```

## 安装

根据硬件平台安装依赖：

```bash
pip install -r requirements_gpu.txt
```

或

```bash
pip install -r requirements_npu.txt
```

## 训练

最小示例：

```bash
python main.py --dataset data/dpo_pairs.jsonl --model-name resources/model/Qwen__Qwen2.5-0.5B-Instruct
```

常用参数：

```bash
python main.py ^
  --dataset data/dpo_pairs.jsonl ^
  --model-name resources/model/Qwen__Qwen2.5-0.5B-Instruct ^
  --output-dir output/dpo_model ^
  --num-epochs 1 ^
  --batch-size 1 ^
  --gradient-accumulation-steps 8 ^
  --learning-rate 5e-6 ^
  --beta 0.1 ^
  --max-length 4096 ^
  --max-prompt-length 3072
```

默认使用 LoRA 进行 DPO。若要全参数训练，可加：

```bash
--full-finetune
```

若在 CUDA 环境下希望启用 QLoRA，可加：

```bash
--use-qlora
```

## 代码结构

- `export_rl_data.py`: 从数据库导出单题修改事件、DPO 偏好对
- `main.py`: DPO 训练入口
- `utils/dpo_data.py`: DPO 数据加载与预处理
- `utils/dpo.py`: DPO 训练实现
- `utils/model.py`: 模型路径解析与设备检测

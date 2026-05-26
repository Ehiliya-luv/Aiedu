# Aiedu

Aiedu 是一个面向医学教育场景的模型训练与评测项目。当前主要流程是：

1. 使用 DPO 偏好数据微调基础模型。
2. 用训练后的模型针对病例生成不同任务的输出。
3. 使用 LLM Judge 比较各个候选模型与 base model 的输出质量。
4. 将多组 candidate-vs-base 结果聚合成统一的 Anchor Score 排名。

本仓库只保留项目代码。数据、模型权重、生成结果、评测结果和本地资源目录不会上传到 GitHub。

## 环境安装

根目录主训练流程使用 VERL 相关环境，优先安装：

```bash
pip install -r requirements_verl.txt
```

`requirements_gpu.txt` 是备用依赖文件，可在不跑 VERL 主流程、只需要普通 GPU 推理/评测依赖时使用：

```bash
pip install -r requirements_gpu.txt
```

DPO 子目录提供了独立依赖：

```bash
pip install -r dpo/requirements_gpu.txt
```

NPU 环境可参考：

```bash
pip install -r dpo/requirements_npu.txt
```

## DPO 训练

DPO 训练入口是 `dpo/main.py`。训练数据默认读取 `data/dpo_pairs.jsonl`，该文件不包含在仓库中。

最小示例：

```bash
python dpo/main.py \
  --dataset data/dpo_pairs.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir output/dpo_model
```

常用参数：

```bash
python dpo/main.py \
  --dataset data/dpo_pairs.jsonl \
  --model-name resources/model/Qwen__Qwen2.5-0.5B-Instruct \
  --output-dir output/dpo-full20 \
  --num-epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --beta 0.1 \
  --max-length 4096 \
  --max-prompt-length 3072 \
  --save-steps 100
```

默认使用 LoRA。需要全参数训练时加：

```bash
--full-finetune
```

CUDA 环境下需要 QLoRA 时加：

```bash
--use-qlora
```

## 生成模型输出

训练完成后，用 `generate_output.py` 对病例生成四类任务输出：

- `1`: 病历标准化
- `2`: 考题生成
- `3`: 临床思维
- `4`: 病历综合评分

LoRA/adapter 模型示例：

```bash
python generate_output.py \
  --backend vllm \
  --data-dir data/病例 \
  --base-model-path resources/model/Baichuan-M2-32B-0226 \
  --model-path output/dpo-full20-ckp1200 \
  --output-dir results/dpo-full20-ckp1200 \
  --tasks 1,2,3,4
```

base model 输出也需要单独生成，作为后续比较的 anchor：

```bash
python generate_output.py \
  --backend vllm \
  --data-dir data/病例 \
  --base-model-path resources/model/Baichuan-M2-32B-0226 \
  --use-base-model \
  --output-dir results/base_model_baichuan-full \
  --tasks 1,2,3,4
```

`compare_experiments.py` 期望 `--results-dir` 下每个模型一个目录，例如：

```text
results/
  base_model_baichuan-full/
  dpo-full10/
  dpo-full20-ckp1200/
```

## LLM Judge 对比实验

`compare_experiments.py` 用于比较模型输出。推荐评测逻辑是：每次只比较一个候选模型和同一个 base model，输出一组 candidate-vs-base 结果。

示例：比较 `dpo-full20-ckp1200` 和 `base_model_baichuan-full`。

```bash
python compare_experiments.py \
  --results-dir ./results \
  --select picked \
  --tasks auto \
  --anchor-models base_model_baichuan-full \
  --judge-backend api \
  --judge-model claude-sonnet-4-5 \
  --judge-api-base https://api.example.com/openai \
  --output-dir ./results/LLM_Judge_compare/dpo-single/full20_ckp1200-base-claude-4.6
```

运行后在交互选择里选择两个模型：

```text
base_model_baichuan-full
dpo-full20-ckp1200
```

脚本会为每个任务输出：

- `<任务名>_ranking.csv`: 当前 pair 的 Bradley-Terry / Anchor Score 排名
- `<任务名>_significance.csv`: 显著性检验结果
- `checkpoints/<任务名>_checkpoint.csv`: 成对胜负矩阵
- `checkpoints/_meta.json`: 本次比较的模型、任务和 judge 信息
- `judge_stats/`: 排名图、显著性热力图等可视化结果

对每个候选模型重复运行一次，目录可以类似：

```text
results/LLM_Judge_compare/dpo-single/
  full10-base-claude-4.6/
  full10_ckp700-base-claude-4.6/
  full20-base-claude-4.6/
  full20_ckp1200-base-claude-4.6/
```

本地实验中的 `results-clean/LLM_Judge_compare/dpo-single/` 就是这种结构。

## 聚合 Anchor Score

`aggregate_anchor_compare.py` 负责读取多组 candidate-vs-base 结果，并按统一 base model 聚合。

```bash
python aggregate_anchor_compare.py \
  --input-dir ./results/LLM_Judge_compare/dpo-single \
  --output-dir ./results/LLM_Judge_compare/dpo-anchor-aggregate \
  --anchor-model base_model_baichuan-full \
  --order-task "考题生成" \
  --bootstrap 10000
```

聚合逻辑：

1. 扫描 `--input-dir` 下的每个比较实验目录。
2. 读取 `checkpoints/_meta.json`，确认该实验严格只包含 base model 和一个 candidate model。
3. 读取每个任务的 `checkpoints/<任务名>_checkpoint.csv`。
4. 汇总所有 candidate-vs-base 胜负票数。
5. 对每个任务计算 Anchor Score 和 bootstrap 置信区间。
6. 按 `--order-task` 指定的任务排序，输出跨任务总表和图表。

主要输出：

```text
results/LLM_Judge_compare/dpo-anchor-aggregate/<judge-name>/
  all_tasks_anchor_summary.csv
  manifest.csv
  病历标准化_anchor_ranking.csv
  考题生成_anchor_ranking.csv
  临床思维_anchor_ranking.csv
  病历综合评分_anchor_ranking.csv
  judge_stats/
```

本地 `results-clean/LLM_Judge_compare/dpo-anchor-aggregate/api_claude-sonnet-4-6/` 是参考结果结构：它聚合了多个 DPO checkpoint 相对 `base_model_baichuan-full` 的比较结果，并生成跨任务 Anchor Score 汇总。

## 目录约定

以下目录通常只存在于本地，不应提交到仓库：

```text
data/
output/
results/
results-clean/
resources/
tmp/
```

私有数据抽取脚本也不进入公开仓库。公开仓库只保留训练、生成、比较和聚合所需的项目代码。

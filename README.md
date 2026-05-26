# Aiedu

Aiedu is a research codebase for medical education model training and evaluation.

The public repository contains code only. Datasets, model weights, generated outputs, experiment results, local resources, and private data extraction scripts are intentionally excluded.

## Overview

The typical workflow is:

1. Prepare preference data for DPO training.
2. Train or continue training a model.
3. Generate task outputs with base and candidate models.
4. Compare each candidate model against a base model with `compare_experiments.py`.
5. Aggregate all candidate-vs-base comparisons with `aggregate_anchor_compare.py`.

## Environment

The main training workflow uses the VERL-related environment:

```bash
pip install -r requirements_verl.txt
```

`requirements_gpu.txt` is kept as a fallback dependency file for GPU-only utilities or evaluation workflows.

The DPO module has separate dependency files:

```bash
pip install -r dpo/requirements_gpu.txt
```

For NPU environments:

```bash
pip install -r dpo/requirements_npu.txt
```

## Data Format

DPO training expects a JSONL file with one preference pair per line:

```json
{
  "prompt": "instruction or context",
  "chosen": "preferred answer",
  "rejected": "less preferred answer",
  "meta": {
    "sample_id": "optional-id",
    "question_type": "optional-task-type"
  }
}
```

The dataset path is supplied at runtime and is not part of this repository.

## DPO Training

The DPO entry point is `dpo/main.py`.

```bash
python dpo/main.py \
  --dataset <path-to-dpo-pairs.jsonl> \
  --model-name <base-model-or-local-path> \
  --output-dir <output-model-dir>
```

Common options:

```bash
python dpo/main.py \
  --dataset <path-to-dpo-pairs.jsonl> \
  --model-name <base-model-or-local-path> \
  --output-dir <output-model-dir> \
  --num-epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --beta 0.1 \
  --max-length 4096 \
  --max-prompt-length 3072
```

By default, training uses LoRA. Use full fine-tuning when needed:

```bash
--full-finetune
```

Enable QLoRA on supported CUDA environments:

```bash
--use-qlora
```

## Generate Model Outputs

Use `generate_output.py` to generate outputs for the evaluation tasks.

```bash
python generate_output.py \
  --backend vllm \
  --data-dir <case-input-dir> \
  --base-model-path <base-model-path> \
  --model-path <candidate-adapter-or-model-dir> \
  --output-dir <candidate-output-dir> \
  --tasks 1,2,3,4
```

Generate base model outputs separately:

```bash
python generate_output.py \
  --backend vllm \
  --data-dir <case-input-dir> \
  --base-model-path <base-model-path> \
  --use-base-model \
  --output-dir <base-output-dir> \
  --tasks 1,2,3,4
```

Task IDs:

- `1`: medical record standardization
- `2`: question generation
- `3`: clinical reasoning
- `4`: comprehensive scoring

The comparison scripts expect one output directory per model under a shared result root:

```text
<result-root>/
  <base-model-name>/
  <candidate-model-a>/
  <candidate-model-b>/
```

## Pairwise LLM Judge Comparison

`compare_experiments.py` runs LLM Judge comparisons and writes task-level rankings, significance files, checkpoints, and plots.

Recommended evaluation design:

- Use one stable base model as the anchor.
- Compare each candidate model against that base model in a separate run.
- Keep all pairwise runs under one directory so they can be aggregated later.

Example:

```bash
python compare_experiments.py \
  --results-dir <result-root> \
  --select picked \
  --tasks auto \
  --anchor-models <base-model-name> \
  --judge-backend api \
  --judge-model <judge-model-name> \
  --judge-api-base <openai-compatible-api-base> \
  --output-dir <pairwise-run-output-dir>
```

During interactive model selection, choose exactly:

```text
<base-model-name>
<candidate-model-name>
```

Each pairwise run produces files such as:

```text
<pairwise-run-output-dir>/
  checkpoints/
    _meta.json
    <task>_checkpoint.csv
  judge_stats/
  <task>_ranking.csv
  <task>_significance.csv
```

Repeat this process for every candidate model.

## Aggregate Candidate-vs-Base Results

`aggregate_anchor_compare.py` aggregates multiple candidate-vs-base comparison runs into a single Anchor Score report.

```bash
python aggregate_anchor_compare.py \
  --input-dir <pairwise-runs-root> \
  --output-dir <aggregate-output-dir> \
  --anchor-model <base-model-name> \
  --order-task <task-name> \
  --bootstrap 10000
```

Aggregation logic:

1. Scan all pairwise run directories under `--input-dir`.
2. Read each run's `checkpoints/_meta.json`.
3. Keep only strict base-vs-candidate runs.
4. Read task-level win matrices from `checkpoints/<task>_checkpoint.csv`.
5. Merge candidate-vs-base vote counts across runs.
6. Compute Anchor Score and bootstrap confidence intervals for each task.
7. Write task rankings, a cross-task summary, a manifest, and figures.

Typical aggregate output:

```text
<aggregate-output-dir>/
  <judge-name>/
    all_tasks_anchor_summary.csv
    manifest.csv
    <task>_anchor_ranking.csv
    judge_stats/
```

## Public Repository Scope

The following are intentionally excluded from version control:

- raw and processed datasets
- model checkpoints and adapters
- generated outputs
- evaluation result directories
- local resources and downloaded files
- logs and temporary files
- private data extraction scripts
- environment files containing secrets

This keeps the public repository focused on reusable training, generation, comparison, and aggregation code.

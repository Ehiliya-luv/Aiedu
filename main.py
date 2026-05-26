# -*- coding: utf-8 -*-
"""项目主入口：使用 LLM Judge 奖励的 GRPO 训练。"""

from __future__ import annotations

import argparse
import logging
import os
import sys

os.environ.setdefault("ASCEND_RT_PRECISION_MODE", "allow_fp32_to_fp16")
os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "1")

import torch

try:
    import torch_npu  # noqa: F401

    HAS_TORCH_NPU = True
except ImportError:
    HAS_TORCH_NPU = False

from utils.log_setup import setup_main_process_logging
from utils.model import detect_runtime_device
from utils.rl_data import load_rl_samples


logger = logging.getLogger(__name__)


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "y", "on")


def setup_env(enable_cache: bool = False) -> str:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["AIEDU_ENABLE_CACHE"] = "1" if enable_cache else "0"

    if HAS_TORCH_NPU:
        os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "0")
        os.environ.setdefault("NPU_LAUNCH_MODE", "1")

    cache_dir = os.path.abspath("./resources/cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    return cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aiedu：LLM Judge 奖励版 GRPO 训练入口")

    parser.add_argument("--dataset", type=str, default="data/rl_train.jsonl", help="训练数据集路径")
    parser.add_argument("--model-name", type=str, default="Qwen__Qwen2.5-0.5B-Instruct", help="基础模型名称")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer 名称（默认同 model-name）")
    parser.add_argument("--output-dir", type=str, default="output/rl_model", help="输出目录")

    parser.add_argument("--num-epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=1, help="每设备批次大小")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="梯度裁剪范数")
    parser.add_argument("--warmup-steps", type=int, default=0, help="预热步数")
    parser.add_argument("--lr-scheduler-type", type=str, default="linear", help="学习率调度器类型")

    parser.add_argument("--num-generations", type=int, default=2, help="每提示采样数（需 >= 2）")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p 采样")

    parser.add_argument("--logging-steps", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save-steps", type=int, default=1, help="保存间隔（优化器更新步）")
    parser.add_argument("--save-total-limit", type=int, default=1, help="最多保留 checkpoint 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--use-qlora", action="store_true", help="兼容旧参数：启用 LoRA（当前不启用 4-bit 量化）")
    parser.add_argument("--use-lora", action="store_true", dest="use_qlora", help="启用 LoRA")
    parser.add_argument("--use-gradient-checkpointing", type=lambda x: x.lower() in ("true", "1", "yes"), default=False, help="启用梯度检查点")
    parser.add_argument("--use-flash-attention", type=lambda x: x.lower() in ("true", "1", "yes"), default=False, help="启用 Flash Attention 2（仅 CUDA）")
    parser.add_argument("--use-paged-optimizer", type=lambda x: x.lower() in ("true", "1", "yes"), default=False, help="使用 paged_adamw_8bit 优化器")
    parser.add_argument("--use-vllm", nargs="?", const=True, default=False, type=_str_to_bool, help="使用 vLLM 加速 GRPO 生成（仅支持 CUDA/Linux 环境，默认关闭）")
    parser.add_argument("--vllm-mode", type=str, default="colocate", choices=("colocate", "server"), help="vLLM 运行模式")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.25, help="vLLM 可使用的 GPU 显存比例；8*80G 默认最多占 160G，给 GRPO 训练预留至少 480G")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=8, help="vLLM tensor parallel 大小；8*A100 默认用 8")
    parser.add_argument("--vllm-max-model-length", type=int, default=8192, help="vLLM 上下文长度，至少需要覆盖 prompt + max_new_tokens")
    parser.add_argument("--vllm-enable-sleep-mode", type=lambda x: x.lower() in ("true", "1", "yes"), default=True, help="启用 vLLM sleep mode 以降低 colocate 显存占用")
    parser.add_argument("--vllm-server-host", type=str, default="127.0.0.1", help="vLLM server host")
    parser.add_argument("--vllm-server-port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--vllm-server-timeout", type=float, default=2400.0, help="等待 vLLM server 的超时时间（秒）")
    parser.add_argument("--vllm-group-port", type=int, default=51216, help="兼容旧参数；VERL 训练不使用该端口")
    parser.add_argument("--enable-cache", action="store_true", help="启用运行时缓存")
    parser.add_argument("--enable-thinking", action="store_true", help="训练时开启 think 模式")
    parser.add_argument("--train-samples", type=int, default=-1, help="训练样本数（-1 表示全部）")

    parser.add_argument("--local-vllm-pyengine", action="store_true", help="使用 vLLM Python engine 作为 judge scorer（不走 HTTP API）")
    parser.add_argument("--local-vllm-model-path", type=str, default="", help="vLLM Python engine 模型路径（如 ./resources/model/Baichuan-M2-32B-0226）")
    parser.add_argument("--local-vllm-lora-path", type=str, default="", help="vLLM Python engine judge LoRA adapter 路径（可选）")
    parser.add_argument(
        "--judge-sharpness", type=float, default=0.0,
        help="judge digit logprobs 锐化温度倒数（>1 让概率分布更尖锐 → reward 信号更宽，"
             "0 表示用 settings 默认值 JUDGE_VLLM_SHARPNESS=2.0）",
    )
    # 两个调试日志通道开关，分别对应 evidence.log / completion.log。
    # 默认全部关（数据大、IO 开销大），仅调试 / spot-check 时手动启用。
    # 训练 step 粒度的指标由 verl 内部 emit 到 main_process.log，
    # 不需要单独的 state 通道——由 tools/plot_reward.py 直接解析。
    parser.add_argument(
        "--log-evidence", action="store_true",
        help="启用 evidence.log（默认关；写入每条 judge 的 RAG 检索结果，IO 开销大）",
    )
    parser.add_argument(
        "--log-completion", action="store_true",
        help="启用 completion.log（默认关；写入 actor 完整输出原文，每条 ~3KB）",
    )
    parser.add_argument(
        "--require-rag", action="store_true",
        help="要求 RAG 完整可用（PDF/XLSX index + embedding client + judge client 全在）；"
             "缺任一项 fail-fast 退出。默认关闭——RAG 不可用只 WARNING、训练继续。"
             "CI / 调参时建议开启，避免误以为 RAG 在跑实际全空。",
    )

    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-target-modules", type=str, default="q_proj, v_proj, o_proj, gate_proj, down_proj", help="LoRA 目标模块")
    parser.add_argument("--disable-save-best-checkpoint", action="store_true", help="关闭 best-checkpoint 保存")
    parser.add_argument("--verl-val-size", type=int, default=16, help="VERL 验证集样本数")
    parser.add_argument("--verl-data-dir", type=str, default=None, help="VERL parquet 数据输出目录（默认 output-dir/verl_data）")
    parser.add_argument("--verl-rollout-backend", type=str, default=None, choices=("hf", "vllm"), help="VERL rollout 后端；默认按 use-vllm 映射")
    parser.add_argument("--verl-rollout-mode", type=str, default=None, choices=("async",), help="VERL rollout mode；默认不传。当前 VERL 已移除 sync，async 会进入 agent-loop/async rollout")
    parser.add_argument("--verl-truncation", type=str, default="left", choices=("left", "right", "middle", "error"), help="VERL prompt 截断策略")
    parser.add_argument("--verl-n-gpus-per-node", type=int, default=None, help="VERL 每节点 GPU 数；默认自动检测")
    parser.add_argument("--verl-disable-validation", nargs="?", const=True, default=True, type=_str_to_bool, help="关闭 VERL validation，避免 agent-loop 验证路径的变长 prompt 拼接问题并降低显存峰值")
    parser.add_argument("--verl-resume-mode", type=str, default="disable", choices=("disable", "auto", "resume_path"), help="VERL checkpoint 恢复模式；默认 disable，避免自动读取旧的半截 global_step checkpoint")
    parser.add_argument("--verl-dry-run", action="store_true", help="只生成 VERL 数据和打印命令，不启动训练")

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别（控制台 + main_process.log）。DEBUG 会打印 judge logprobs 与 0~9 概率分布。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="--log-level=DEBUG 的快捷方式（覆盖 --log-level）",
    )
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    from utils.grpo import GRPOScriptArguments, train_grpo

    dataset_path = os.path.abspath(args.dataset)
    if args.dataset.endswith(".jsonl"):
        max_items = args.train_samples if args.train_samples > 0 else None
        samples = load_rl_samples(path=args.dataset, max_items=max_items, strict=True)
        logger.info("RL JSONL 数据校验通过，共 %d 条", len(samples))
    else:
        logger.info("跳过 JSONL 预校验，交给 VERL 数据准备流程处理: %s", dataset_path)

    script_args = GRPOScriptArguments(
        model_name_or_path=args.model_name,
        tokenizer_name_or_path=args.tokenizer_name,
        dataset_name=args.dataset,
        train_samples=args.train_samples if args.train_samples > 0 else -1,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_best_checkpoint=not bool(args.disable_save_best_checkpoint),
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_qlora=args.use_qlora,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_flash_attention=args.use_flash_attention,
        use_paged_optimizer=args.use_paged_optimizer,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_model_length=args.vllm_max_model_length,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_server_timeout=args.vllm_server_timeout,
        vllm_group_port=args.vllm_group_port,
        enable_cache=args.enable_cache,
        enable_thinking=bool(args.enable_thinking),
        seed=args.seed,
        logging_steps=args.logging_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        verl_val_size=args.verl_val_size,
        verl_data_dir=args.verl_data_dir,
        verl_rollout_backend=args.verl_rollout_backend,
        verl_rollout_mode=args.verl_rollout_mode,
        verl_truncation=args.verl_truncation,
        verl_n_gpus_per_node=args.verl_n_gpus_per_node,
        verl_disable_validation=args.verl_disable_validation,
        verl_resume_mode=args.verl_resume_mode,
        verl_dry_run=args.verl_dry_run,
        local_vllm_pyengine=bool(args.local_vllm_pyengine),
        local_vllm_model_path=str(args.local_vllm_model_path or ""),
        local_vllm_lora_path=str(args.local_vllm_lora_path or ""),
        judge_sharpness=float(args.judge_sharpness or 0.0),
    )
    train_grpo(script_args)


def _emit_rag_startup_report(require_rag: bool) -> None:
    """启动时一次性 RAG 状态体检。

    - 调 load_knowledge_base() 探一次 → 拿 status_summary。
    - 在主日志里打印各组件的 ✓/✗，并把 disable_reasons 全列出来。
    - 若 --require-rag 且不 fully_operational → 抛 SystemExit(2) fail-fast 退出。
    - 若 --require-rag=False → 仅 WARNING 提示，训练继续（生产容错）。

    特意放在 main() 入口、setup_main_process_logging() 之后、run_training() 之前——
    这样 5 秒内就能从 main_process.log 看到 RAG 真实状态，不必跑完一轮 step
    才在 evidence.log 发现 n=0。
    """
    try:
        from utils.judge.knowledge import load_knowledge_base
    except Exception as exc:  # pragma: no cover
        logger.warning("[RAG] 启动状态探测失败：导入 load_knowledge_base 抛错 %s", exc)
        if require_rag:
            raise SystemExit(2)
        return

    kb = load_knowledge_base()
    if kb is None:
        msg = "[RAG] knowledge_base = None（PDF/XLSX index 都没加载到）"
        if require_rag:
            logger.error("%s 且 --require-rag，fail-fast 退出。", msg)
            raise SystemExit(2)
        logger.warning("%s — 训练将继续但 evidence 全空。", msg)
        return

    status = kb.status_summary()
    icon = lambda b: "✓" if b else "✗"  # noqa: E731
    logger.info(
        "[RAG] 启动状态体检: PDF=%s XLSX=%s embedding=%s judge_llm=%s | "
        "is_ready=%s can_generate_context=%s fully_operational=%s",
        icon(status["pdf_index"]),
        icon(status["xlsx_index"]),
        icon(status["embedding_client"]),
        icon(status["judge_client"]),
        icon(status["is_ready"]),
        icon(status["can_generate_context"]),
        icon(status["is_fully_operational"]),
    )
    if status["disable_reasons"]:
        for reason in status["disable_reasons"]:
            logger.warning("[RAG] disable_reason: %s", reason)

    if require_rag and not status["is_fully_operational"]:
        logger.error(
            "[RAG] --require-rag 模式下要求 fully_operational=True，"
            "当前缺组件 → fail-fast 退出。"
        )
        raise SystemExit(2)
    if not status["is_fully_operational"]:
        logger.warning(
            "[RAG] RAG 不完整可用，训练将继续但部分 evidence 可能为空。"
            "如要严格保证 RAG 在跑请加 --require-rag。"
        )


def main() -> int:
    args = parse_args()
    log_level = "DEBUG" if args.debug else args.log_level
    log_dir = setup_main_process_logging(log_level=log_level)
    cache_dir = setup_env(enable_cache=args.enable_cache)
    device = detect_runtime_device()

    # 把 3 个日志通道的 CLI flag 转写为环境变量。env 是唯一能跨多进程（Ray worker、
    # VERL subprocess、vLLM daemon）的传递通道——argparse 在 main 进程里解析，子进程
    # 看不到 args 对象，但全继承 os.environ。log_setup 的 sink 是 lazy 单例，第一次
    # 调用 get_*_sink() 时才读 env，所以这里设置后立即生效，无需重启进程。
    # evidence/completion 默认关；训练 step 指标走 verl 自带的 main_process.log。
    os.environ["AIEDU_LOG_EVIDENCE_ENABLED"] = "1" if args.log_evidence else "0"
    os.environ["AIEDU_LOG_COMPLETION_ENABLED"] = "1" if args.log_completion else "0"

    logger.info("日志目录: %s | level=%s", log_dir, log_level)
    logger.info("运行设备: %s", device)
    logger.info("torch_npu 可用: %s", HAS_TORCH_NPU)
    if device == "npu" and hasattr(torch, "npu"):
        try:
            logger.info("NPU 数量: %s", torch.npu.device_count())
            logger.info("训练后端: VERL GRPO（LLM Judge 奖励）")
        except Exception:
            pass
    elif device == "cuda":
        logger.info("CUDA 数量: %s", torch.cuda.device_count())
        logger.info("训练后端: VERL GRPO（LLM Judge 奖励）")
    logger.info("缓存目录: %s", cache_dir)
    logger.info("运行时缓存开关 enable_cache: %s", args.enable_cache)
    logger.info("参数: %s", vars(args))

    # RAG 状态体检（在 run_training 之前；--require-rag=True 时缺组件 fail-fast）。
    _emit_rag_startup_report(require_rag=bool(args.require_rag))

    try:
        run_training(args)
        logger.info("训练完成")
        return 0
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 130
    except Exception as exc:
        logger.exception("执行失败: %s", exc)
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

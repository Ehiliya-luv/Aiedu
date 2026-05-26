# -*- coding: utf-8 -*-
"""VERL-backed GRPO training orchestration with the existing Aiedu reward."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ._common import (
    _as_project_path,
    _check_verl_available,
    _device_count,
    _max_prompt_length,
    _project_root,
    _reward_function_path,
)
from .judge.settings import (
    JUDGE_LOCAL_VLLM_LORA_PATH,
    JUDGE_VLLM_DEVICE,
    JUDGE_VLLM_GPU_MEMORY_UTILIZATION,
    JUDGE_VLLM_MAX_MODEL_LEN,
    JUDGE_VLLM_SHARPNESS,
    JUDGE_VLLM_TENSOR_PARALLEL_SIZE,
)
from .log_setup import compute_log_dir
from .model import HAS_TORCH_NPU, detect_runtime_device, resolve_model_path
from .verl_checkpoint import _check_judge_error_in_logs, merge_fsdp_checkpoint
from .verl_data import _write_verl_parquet

logger = logging.getLogger(__name__)

# 静默模式：默认不打印 daemon 输出，避免干扰 GRPO 主流程日志
# debug 时设为 False 可查看 daemon 的详细输出
VLLM_DAEMON_SILENT = True


# ═══════════════════════════════════════════════════════════════════════════
# 训练参数
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GRPOScriptArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    dataset_name: str = field(default="data/rl_train.jsonl")
    train_samples: int = field(default=-1)
    output_dir: str = field(default="output/rl_model")

    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=2)
    learning_rate: float = field(default=5e-6)
    weight_decay: float = field(default=0.0)
    max_grad_norm: float = field(default=1.0)
    warmup_steps: int = field(default=0)
    lr_scheduler_type: str = field(default="linear")

    num_generations: int = field(default=2)
    max_new_tokens: int = field(default=4096)
    temperature: float = field(default=0.75)
    top_p: float = field(default=0.9)

    logging_steps: int = field(default=10)
    save_steps: int = field(default=1)
    save_total_limit: int = field(default=1)
    save_best_checkpoint: bool = field(default=True)
    enable_thinking: bool = field(default=False)
    seed: int = field(default=42)

    use_qlora: bool = field(default=False)
    use_gradient_checkpointing: bool = field(default=False)
    use_flash_attention: bool = field(default=False)
    use_paged_optimizer: bool = field(default=False)
    use_vllm: bool = field(default=False)
    vllm_mode: str = field(default="colocate")
    vllm_gpu_memory_utilization: float = field(default=0.25)
    vllm_tensor_parallel_size: int = field(default=8)
    vllm_max_model_length: Optional[int] = field(default=8192)
    vllm_enable_sleep_mode: bool = field(default=True)
    vllm_server_host: str = field(default="127.0.0.1")
    vllm_server_port: int = field(default=8000)
    vllm_server_timeout: float = field(default=240.0)
    vllm_group_port: int = field(default=51216)
    enable_cache: bool = field(default=False)

    # vLLM Python engine 模式（作为 judge scorer，不走 HTTP API）
    local_vllm_pyengine: bool = field(default=False)
    local_vllm_model_path: str = field(default="")
    local_vllm_lora_path: str = field(default="")
    # judge digit logprobs 锐化温度倒数（>1 让概率分布更尖锐 → reward 信号更宽）
    # 0 表示用 settings 里的默认值 JUDGE_VLLM_SHARPNESS。
    judge_sharpness: float = field(default=0.0)

    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: Optional[str] = field(default="q_proj,v_proj,o_proj,gate_proj,down_proj")

    verl_val_size: int = field(default=16)
    verl_data_dir: Optional[str] = field(default=None)
    verl_rollout_backend: Optional[str] = field(default=None)
    verl_rollout_mode: Optional[str] = field(default=None)
    verl_truncation: str = field(default="left")
    verl_n_gpus_per_node: Optional[int] = field(default=None)
    verl_disable_validation: bool = field(default=True)
    verl_resume_mode: str = field(default="disable")
    verl_dry_run: bool = field(default=False)


# ═══════════════════════════════════════════════════════════════════════════
# VERL 命令构建
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_rollout_backend(script_args: GRPOScriptArguments) -> str:
    if script_args.verl_rollout_backend:
        return script_args.verl_rollout_backend
    return "vllm" if bool(script_args.use_vllm) else "hf"


def _global_batch_size(script_args: GRPOScriptArguments, n_gpus: int) -> int:
    return max(
        1,
        int(script_args.per_device_train_batch_size)
        * int(script_args.gradient_accumulation_steps)
        * max(1, int(n_gpus)),
    )


def _target_modules(script_args: GRPOScriptArguments) -> str:
    value = (script_args.lora_target_modules or "").strip()
    if not value or "," in value:
        return "all-linear"
    return value


def _bool(value: bool) -> str:
    return "True" if bool(value) else "False"


def _attention_implementation(script_args: GRPOScriptArguments) -> str:
    return "flash_attention_2" if bool(script_args.use_flash_attention) else "sdpa"


def _rollout_gpu_memory_utilization(script_args: GRPOScriptArguments, n_gpus: int) -> float:
    value = float(script_args.vllm_gpu_memory_utilization)
    if n_gpus >= 8 and value > 0.35:
        logger.warning(
            "vLLM gpu_memory_utilization=%.2f 会压缩 GRPO 训练显存；"
            "8*80G 若需给训练预留至少 400G，建议设置为 0.35 或更低。",
            value,
        )
    return value


def _build_verl_command(
    script_args: GRPOScriptArguments,
    train_file: str,
    val_file: str,
    device: str,
    model_path: str,
) -> List[str]:
    """构建 VERL Hydra 命令。——"""
    n_gpus = int(script_args.verl_n_gpus_per_node or _device_count(device))
    rollout_backend = _resolve_rollout_backend(script_args)
    train_batch_size = _global_batch_size(script_args, n_gpus)
    micro_batch = max(1, int(script_args.per_device_train_batch_size))
    ppo_mini_batch = train_batch_size
    max_prompt_length = _max_prompt_length(script_args.vllm_max_model_length, script_args.max_new_tokens)
    tp_size = max(1, min(int(script_args.vllm_tensor_parallel_size), max(1, n_gpus)))
    rollout_gpu_memory_utilization = _rollout_gpu_memory_utilization(script_args, n_gpus)

    if rollout_backend not in {"hf", "vllm"}:
        raise ValueError(f"VERL rollout backend 仅支持 hf 或 vllm，当前: {rollout_backend}")

    overrides = [
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.prompt_key=prompt",
        f"data.train_batch_size={train_batch_size}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={int(script_args.max_new_tokens)}",
        f"data.truncation={script_args.verl_truncation}",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=True",
        f"+actor_rollout_ref.model.override_config.attn_implementation={_attention_implementation(script_args)}",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={_bool(script_args.use_gradient_checkpointing)}",
        f"actor_rollout_ref.model.lora_rank={int(script_args.lora_r) if script_args.use_qlora else 0}",
        f"actor_rollout_ref.model.lora_alpha={int(script_args.lora_alpha)}",
        f"actor_rollout_ref.model.target_modules={_target_modules(script_args)}",
        "actor_rollout_ref.actor.strategy=fsdp",
        f"actor_rollout_ref.actor.optim.lr={float(script_args.learning_rate)}",
        f"actor_rollout_ref.actor.optim.weight_decay={float(script_args.weight_decay)}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_batch}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.rollout.name={rollout_backend}",
        f"actor_rollout_ref.rollout.n={int(script_args.num_generations)}",
        f"actor_rollout_ref.rollout.temperature={float(script_args.temperature)}",
        f"actor_rollout_ref.rollout.top_p={float(script_args.top_p)}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tp_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout_gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.max_model_len={int(script_args.vllm_max_model_length or 8192)}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={micro_batch}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro_batch}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "critic.enable=False",
        "reward_model.enable=False",
        f"custom_reward_function.path={_reward_function_path()}",
        "custom_reward_function.name=compute_score",
        f"++reward.custom_reward_function.path={_reward_function_path()}",
        "++reward.custom_reward_function.name=compute_score",
        "++data.return_raw_chat=False",
        "++actor_rollout_ref.hybrid_engine=True",
        "trainer.project_name=aiedu_verl_grpo",
        f"trainer.experiment_name={Path(script_args.output_dir).name or 'aiedu_grpo'}",
        "trainer.logger=[console]",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        f"trainer.save_freq={int(script_args.save_steps or 1)}",
        f"trainer.max_actor_ckpt_to_keep={max(1, int(script_args.save_total_limit))}",
        f"trainer.max_critic_ckpt_to_keep={max(1, int(script_args.save_total_limit))}",
        # 注：不传 trainer.save_hf_format — VERL 没有此配置
        # HF 格式在训练结束后由 merge_fsdp_checkpoint() 后处理产生
        # max_actor_ckpt_to_keep=1 确保只保留最近的一份 checkpoint
        f"trainer.test_freq={-1 if script_args.verl_disable_validation else max(1, int(script_args.logging_steps))}",
        f"trainer.total_epochs={int(script_args.num_train_epochs)}",
        f"trainer.default_local_dir={_as_project_path(script_args.output_dir)}",
        f"trainer.resume_mode={script_args.verl_resume_mode}",
    ]

    if script_args.verl_rollout_mode:
        overrides.append(f"actor_rollout_ref.rollout.mode={script_args.verl_rollout_mode}")

    if script_args.verl_disable_validation:
        overrides.extend([
            "++trainer.val_before_train=False",
            "++trainer.test_freq=-1",
            "++trainer.log_val_generations=0",
            "++data.val_batch_size=1",
        ])

    if rollout_backend == "vllm":
        overrides.extend([
            "actor_rollout_ref.rollout.load_format=safetensors",
            "actor_rollout_ref.rollout.layered_summon=True",
            f"++actor_rollout_ref.rollout.enable_sleep_mode={_bool(script_args.vllm_enable_sleep_mode)}",
        ])

    return [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]


# ═══════════════════════════════════════════════════════════════════════════
# 训练主流程
# ═══════════════════════════════════════════════════════════════════════════

def _drain_pipe(stream, tag: str, file_sink=None, also_logger: bool = True) -> None:
    """持续读取并记录子进程输出流，防止 pipe buffer 满导致子进程阻塞。

    - file_sink: 若提供，把每一行原样写入文件（再 flush），用于 vllm_engine.log。
    - also_logger: 是否同时通过 logger 转发到主进程控制台 / main_process.log。
      daemon 启动握手期需要 True 让父进程看到进度；模型加载完毕后只写文件即可。
    """
    try:
        for line in stream:
            stripped = line.rstrip()
            if file_sink is not None:
                try:
                    file_sink.write(line if line.endswith("\n") else line + "\n")
                    file_sink.flush()
                except Exception:
                    pass
            if also_logger and stripped:
                if VLLM_DAEMON_SILENT:
                    logger.debug("[vllm-daemon:%s] %s", tag, stripped)
                else:
                    logger.info("[vllm-daemon:%s] %s", tag, stripped)
    except Exception:
        pass


def _pipe_to_stream(stream, target) -> None:
    """逐行从子进程 pipe 读到目标 stream（一般是 sys.stdout / sys.stderr，
    可能已被 TeeStream 替换，会自动同步落到 main_process.log）。"""
    try:
        for line in stream:
            try:
                target.write(line)
                target.flush()
            except Exception:
                pass
    except Exception:
        pass


def _start_vllm_daemon(script_args: GRPOScriptArguments, env: dict) -> subprocess.Popen | None:
    """启动 vLLM engine daemon（单进程，独占 GPU JUDGE_VLLM_DEVICE），返回子进程对象。

    修复说明（第二轮）：
    1. daemon 内部已改为先 bind 端口再加载模型，彻底消除端口 race condition。
    2. 父进程传 --port=0（让 daemon 自己选端口），daemon bind 成功后父进程再等 VLLM_ENGINE_PORT= 信号。
    3. drain 线程用 WARNING 级别，确保 daemon 运行期间的崩溃信息可见。
    4. 启动 daemon 监控线程：daemon 意外退出时立即输出告警日志（VERL 的 TimeoutError 变成可见崩溃日志）。

    日志输出（本轮新增，零侵入训练逻辑）：
    - daemon 的全部 stdout/stderr 行同步落入 ./tmp/grpo/{ts}/vllm_engine.log。
    - 握手期（等待 VLLM_ENGINE_PORT=...）仍通过 logger 转发到主控制台，方便用户观察启动进度。
    - 握手完成后 drain 仅写文件，避免 vLLM 内部进度日志刷屏主控制台。
    """
    baichuan_path = script_args.local_vllm_model_path or os.environ.get("AIEDU_JUDGE_LOCAL_VLLM_MODEL_PATH", "")
    if not baichuan_path:
        raise ValueError(
            "local_vllm_pyengine 已启用，但未指定模型路径。\n"
            "使用 --local-vllm-model-path ./resources/model/Baichuan-M2-32B-0226 "
            "或设置环境变量 AIEDU_JUDGE_LOCAL_VLLM_MODEL_PATH"
        )
    baichuan_path = str(_as_project_path(baichuan_path))
    lora_path = script_args.local_vllm_lora_path or os.environ.get("AIEDU_JUDGE_LOCAL_VLLM_LORA_PATH", JUDGE_LOCAL_VLLM_LORA_PATH)
    lora_path = str(_as_project_path(lora_path)) if lora_path else ""

    # 计算/复用日志目录，并打开 vllm_engine.log（line buffering）
    log_dir = compute_log_dir(project_root=_project_root())
    daemon_log_path = log_dir / "vllm_engine.log"
    daemon_log_fp = daemon_log_path.open("a", buffering=1, encoding="utf-8")

    daemon_env = os.environ.copy()
    daemon_env["CUDA_VISIBLE_DEVICES"] = JUDGE_VLLM_DEVICE
    daemon_env["ASCEND_RT_VISIBLE_DEVICES"] = JUDGE_VLLM_DEVICE
    daemon_env["VLLM_NO_PROGRESS_BARS"] = "1"

    daemon_cmd = [
        sys.executable, "-m", "utils.judge.vllm_engine_daemon",
        "--model-path", baichuan_path,
        "--port", "0",    # 让 daemon 自己 bind，消除 race condition
        "--max-model-len", str(JUDGE_VLLM_MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(JUDGE_VLLM_GPU_MEMORY_UTILIZATION),
        "--tensor-parallel-size", str(JUDGE_VLLM_TENSOR_PARALLEL_SIZE),
        "--sharpness", str(script_args.judge_sharpness or JUDGE_VLLM_SHARPNESS),
    ]
    if lora_path:
        daemon_cmd.extend(["--lora-path", lora_path])

    logger.info(
        "启动 vLLM engine daemon: model=%s lora=%s gpu=%s tp=%d | log=%s",
        baichuan_path, lora_path or "(none)", JUDGE_VLLM_DEVICE, JUDGE_VLLM_TENSOR_PARALLEL_SIZE,
        daemon_log_path,
    )
    engine_proc = subprocess.Popen(
        daemon_cmd,
        cwd=str(_project_root()),
        env=daemon_env,
        stdout=subprocess.PIPE,   # stdout/stderr 分开，防止 stderr 撑满 stdout pipe
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    # 把文件句柄挂在 proc 上，供 _stop_vllm_daemon 关闭
    setattr(engine_proc, "_aiedu_log_fp", daemon_log_fp)

    # stderr drain：daemon 加载模型期间的日志走 stderr，必须持续读取。
    # 全部写入 vllm_engine.log，不污染主控制台（also_logger=False）。
    threading.Thread(
        target=_drain_pipe,
        args=(engine_proc.stderr, "err", daemon_log_fp, False),
        daemon=True,
    ).start()

    # 等待 daemon 打印 "VLLM_ENGINE_PORT=<port>"
    # daemon 新版本在模型加载完毕后才打印此行，超时设 15 分钟（大模型加载慢）
    confirmed_port = None
    daemon_stdout_lines: list[str] = []
    start_time = time.time()
    timeout = 900  # 15 分钟，给大模型足够的加载时间
    while time.time() - start_time < timeout:
        # 检查 daemon 是否已经退出（提前崩溃）
        if engine_proc.poll() is not None:
            logger.error(
                "vLLM engine daemon 意外退出（returncode=%d）。"
                "请检查 %s 中的 stderr 日志。",
                engine_proc.returncode, daemon_log_path,
            )
            break
        line = engine_proc.stdout.readline()
        if not line:
            # stdout EOF → daemon 已退出
            logger.error("vLLM engine daemon stdout 关闭（daemon 已退出），详见 %s", daemon_log_path)
            break
        # 同步写入 vllm_engine.log
        try:
            daemon_log_fp.write(line if line.endswith("\n") else line + "\n")
            daemon_log_fp.flush()
        except Exception:
            pass
        stripped = line.strip()
        daemon_stdout_lines.append(stripped)
        if stripped:
            # 握手期保留对父 logger 的可见性
            logger.info("[vllm-daemon:out] %s", stripped)
        if stripped.startswith("VLLM_ENGINE_PORT="):
            confirmed_port = int(stripped.split("=")[1])
            break

    if confirmed_port is None:
        engine_proc.kill()
        try:
            daemon_log_fp.close()
        except Exception:
            pass
        raise RuntimeError(
            f"vLLM engine daemon 启动超时或失败（等待 {int(time.time() - start_time)}s）。\n"
            f"daemon stdout 最后输出:\n" + "\n".join(daemon_stdout_lines[-30:]) + "\n"
            f"（完整 stderr 见 {daemon_log_path}）"
        )

    # 模型已加载完毕，继续 drain stdout（运行期间可能还有少量输出）。
    # 仅写文件，不再 logger 转发，避免污染主控制台。
    threading.Thread(
        target=_drain_pipe,
        args=(engine_proc.stdout, "out", daemon_log_fp, False),
        daemon=True,
    ).start()

    # 启动监控线程：daemon 退出时立即告警（让 TimeoutError 变成有意义的崩溃日志）
    def _watch_daemon(proc: subprocess.Popen, port: int) -> None:
        rc = proc.wait()
        logger.error(
            "vLLM engine daemon 意外退出！returncode=%d port=%d\n"
            "    VERL 训练的后续 RewardLoopWorker 将无法连接 daemon，"
            "可能出现 [Errno 110] Connection timed out。\n"
            "    详见 %s",
            rc, port, daemon_log_path,
        )

    threading.Thread(target=_watch_daemon, args=(engine_proc, confirmed_port), daemon=True).start()

    # 将端口写入 VERL 子进程的环境变量
    # 注意：Ray Worker 进程通过 raylet fork，会继承 VERL 子进程启动时的 os.environ，
    # 因此必须在 subprocess.run(verl_command) 之前设置好这些变量。
    env["AIEDU_JUDGE_VLLM_ENGINE_HOST"] = "127.0.0.1"
    env["AIEDU_JUDGE_VLLM_ENGINE_PORT"] = str(confirmed_port)
    env["AIEDU_JUDGE_LOCAL_VLLM_PYENGINE"] = "1"
    logger.info("vLLM engine daemon 就绪: port=%d | 日志: %s", confirmed_port, daemon_log_path)
    return engine_proc


def _stop_vllm_daemon(engine_proc: subprocess.Popen | None) -> None:
    if engine_proc is None:
        return
    logger.info("关闭 vLLM engine daemon（含子进程）")
    pgid = None
    try:
        pgid = os.getpgid(engine_proc.pid)
    except ProcessLookupError:
        pgid = None
    if pgid is not None and pgid != os.getpgid(os.getpid()):
        # 杀死整个进程组（含 vLLM 内部的多进程 EngineCore）
        os.killpg(pgid, signal.SIGTERM)
        try:
            engine_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
    else:
        engine_proc.terminate()
        try:
            engine_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            engine_proc.kill()

    # 关闭 vllm_engine.log 文件句柄（若 _start_vllm_daemon 挂载过）
    fp = getattr(engine_proc, "_aiedu_log_fp", None)
    if fp is not None:
        try:
            fp.flush()
            fp.close()
        except Exception:
            pass


def train_grpo(script_args: GRPOScriptArguments):
    device = detect_runtime_device()
    logger.info("GRPO backend: VERL | device: %s | torch_npu: %s", device, HAS_TORCH_NPU)

    if int(script_args.num_generations) <= 1:
        raise ValueError(f"GRPO requires num_generations >= 2, got {script_args.num_generations}")

    os.makedirs(script_args.output_dir, exist_ok=True)
    model_path = str(_as_project_path(resolve_model_path(script_args.model_name_or_path)))
    train_file, val_file = _write_verl_parquet(
        dataset_name=script_args.dataset_name,
        train_samples_limit=script_args.train_samples,
        verl_val_size=script_args.verl_val_size,
        output_dir=script_args.output_dir,
        verl_data_dir=script_args.verl_data_dir,
        model_path=model_path,
        max_new_tokens=script_args.max_new_tokens,
        vllm_max_model_length=script_args.vllm_max_model_length,
        truncation=script_args.verl_truncation,
        enable_thinking=bool(script_args.enable_thinking),
    )
    command = _build_verl_command(script_args, train_file, val_file, device, model_path)

    logger.info("VERL rollout backend: %s", _resolve_rollout_backend(script_args))
    logger.info("VERL command:\n%s", " \\\n  ".join(command))

    if script_args.verl_dry_run:
        logger.info("VERL dry-run，跳过训练。")
        return {"train_file": train_file, "val_file": val_file, "command": command}

    _check_verl_available()

    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Ray Worker 进程继承环境变量配置：
    # RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S 控制 runtime_env 缓存，
    # 但更关键的是让 Ray 把父进程环境变量传给 Worker。
    # Ray 2.x 默认在单节点模式下会通过 runtime_env 传递 os.environ（即这里的 env），
    # 但若 Ray cluster 已存在（残留进程），需要先 ray stop。
    env.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "0")  # 确保错误不被静默吞掉

    engine_proc = None
    try:
        # ── 启动 vLLM engine daemon（在 try 内启动，确保 finally 可清理）──
        # 重要：daemon 必须在 subprocess.run(verl_command) 之前启动完毕，
        # 因为 env 字典中的端口号需要在此时已经设置好，才能被 VERL 子进程继承。
        if script_args.local_vllm_pyengine:
            engine_proc = _start_vllm_daemon(script_args, env)
            # 此时 env["AIEDU_JUDGE_VLLM_ENGINE_PORT"] 已设置好
            logger.info(
                "daemon 就绪，VERL 子进程将继承 AIEDU_JUDGE_VLLM_ENGINE_PORT=%s",
                env.get("AIEDU_JUDGE_VLLM_ENGINE_PORT"),
            )

        # 用 Popen + drain 替代 subprocess.run(capture_output=False)：
        # 父进程的 sys.stdout/sys.stderr 已被 TeeStream 替换，会自动把内容
        # 同时落到终端和 main_process.log。如果继承 fd（capture_output=False）
        # 则 VERL 子进程直接写 fd1/fd2，绕过 Python 层 Tee → 看不到文件。
        # drain 线程持续读取 pipe，避免 buffer 满阻塞 VERL（与 vLLM daemon 用同一模型）。
        proc = subprocess.Popen(
            command,
            cwd=str(_project_root()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        verl_threads = [
            threading.Thread(target=_pipe_to_stream, args=(proc.stdout, sys.stdout), daemon=True),
            threading.Thread(target=_pipe_to_stream, args=(proc.stderr, sys.stderr), daemon=True),
        ]
        for t in verl_threads:
            t.start()
        returncode = proc.wait()
        for t in verl_threads:
            t.join(timeout=5)
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        logger.info("VERL GRPO 训练完成，输出目录: %s", script_args.output_dir)

    except subprocess.CalledProcessError as exc:
        # VERL 输出已经通过 drain 写入 main_process.log + 终端
        logger.error("VERL 训练进程异常退出 (returncode=%s)", exc.returncode)

        judge_msg = _check_judge_error_in_logs(script_args.output_dir)
        if judge_msg:
            logger.error(judge_msg)

        ckpt_path = merge_fsdp_checkpoint(script_args.output_dir)
        if ckpt_path:
            logger.info("检测到已有 checkpoint: %s\n训练虽已退出，但 checkpoint 已保存。", ckpt_path)
        else:
            logger.warning("未检测到 checkpoint（训练可能在第一个 step 之前已崩溃）")
        raise

    finally:
        _stop_vllm_daemon(engine_proc)

    # 训练正常结束 → 合并 checkpoint
    merge_fsdp_checkpoint(script_args.output_dir)
    logger.info("GRPO 训练完整结束，输出目录: %s", script_args.output_dir)
    return {"train_file": train_file, "val_file": val_file, "command": command}


__all__ = ["GRPOScriptArguments", "train_grpo", "merge_fsdp_checkpoint"]

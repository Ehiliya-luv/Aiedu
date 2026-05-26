#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""病历生成输出主脚本。

支持两种后端：
  --backend local  — Transformers 本地模型推理（支持 NPU/CUDA）
  --backend vllm   — vLLM server HTTP 推理（OpenAI 兼容 / TRL vllm-serve）

用法示例：
  # local 后端
  python generate_output.py --backend local --model-path ./output/dpo_model

  # vllm 后端（默认自动启动本地 vLLM OpenAI server）
  python generate_output.py --backend vllm --vllm-base-url http://127.0.0.1:8000

  # 连接已有 vLLM / TRL server
  python generate_output.py --backend vllm --no-vllm-auto-start --vllm-base-url http://127.0.0.1:8000

  # 指定任务
  python generate_output.py --backend vllm --tasks 1,2

  # 单样本测试
  python generate_output.py --backend local --sample-size 1
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.gen_output import (  # noqa: E402
    auto_allocate_devices,
    auto_detect_tp_size,
    build_prompt,
    build_vllm_server_command,
    check_vllm_server,
    configure_tokenizer_for_chat,
    detect_accelerator,
    extract_host_port,
    generate_text_local,
    generate_text_via_vllm,
    get_infer_dtype,
    init_directories,
    init_env_for_task,
    load_lora_model,
    load_medical_records_from_dir,
    model_warmup,
    resolve_vllm_endpoint,
    save_result,
    start_vllm_server,
    stop_vllm_server,
    wait_for_vllm_server,
)
from utils.gen_output.prompts import (  # noqa: E402
    PROMPT_TASK_1_SOAP,
    PROMPT_TASK_2_QUESTION,
    PROMPT_TASK_3_THINKING,
    PROMPT_TASK_4_SCORING,
)


# ================= 任务定义 =================
TASK_PROMPTS = {
    1: (PROMPT_TASK_1_SOAP, "病历标准化"),
    2: (PROMPT_TASK_2_QUESTION, "考题生成"),
    3: (PROMPT_TASK_3_THINKING, "临床思维"),
    4: (PROMPT_TASK_4_SCORING, "病历综合评分"),
}


# ================= argparse =================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="病历生成输出脚本")

    # --- 通用参数 ---
    p.add_argument("--backend", choices=["local", "vllm"], default="local",
                   help="推理后端 (default: local)")
    p.add_argument("--data-dir", default="./data/病例",
                   help="病历数据目录 (default: ./data/病例)")
    p.add_argument("--output-dir", default=None,
                   help="输出目录 (default: 根据 backend 自动选择)")
    p.add_argument("--tasks", default="1,2,3,4",
                   help="启用的任务编号，逗号分隔 (default: 1,2,3,4)")
    p.add_argument("--sample-size", type=int, default=0,
                   help="仅取前 N 条数据，0=全部 (default: 0)")
    p.add_argument("--max-new-tokens", type=int, default=8192,
                   help="生成最大 token 数 (default: 8192)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="采样温度 (default: 0.7)")
    p.add_argument("--top-p", type=float, default=0.95,
                   help="Top-p 采样 (default: 0.95)")
    p.add_argument("--do-sample", action="store_true", default=True,
                   help="启用采样 (default: True)")
    p.add_argument("--no-do-sample", action="store_false", dest="do_sample",
                   help="禁用采样 (greedy)")
    p.add_argument("--parallel", action="store_true", default=True,
                   help="多任务并行 (default: True)")
    p.add_argument("--no-parallel", action="store_false", dest="parallel",
                   help="顺序执行")
    p.add_argument("--enable-thinking", action="store_true", default=True,
                   help="启用思考模式 (default: True)")
    p.add_argument("--no-enable-thinking", action="store_false", dest="enable_thinking",
                   help="禁用思考模式")
    p.add_argument("--force-user-role", action="store_true", default=True,
                   help="按 user 角色组织输入 (default: True)")
    p.add_argument("--no-force-user-role", action="store_false", dest="force_user_role",
                   help="不按 user 角色组织输入")
    p.add_argument("--debug", action="store_true", default=False,
                   help="启用调试输出 (default: False)")

    # --- local 后端参数 ---
    p.add_argument("--model-path", default="./output/dpo_model",
                   help="模型路径 (local 后端, default: ./output/dpo_model)")
    p.add_argument("--base-model-path", default="./resources/model/Baichuan-M2-32B-0226",
                   help="基座模型路径 (local 后端, default: ./resources/model/Baichuan-M2-32B-0226)")
    p.add_argument("--use-tuned-model", action="store_true", default=True,
                   help="加载微调模型 (default: True)")
    p.add_argument("--use-base-model", action="store_false", dest="use_tuned_model",
                   help="仅加载基座模型")
    p.add_argument("--max-input-tokens", type=int, default=12288,
                   help="输入最大 token 数 (local 后端, default: 12288)")
    p.add_argument("--enable-warmup", action="store_true", default=True,
                   help="启用模型预热 (default: True)")
    p.add_argument("--no-enable-warmup", action="store_false", dest="enable_warmup",
                   help="禁用模型预热")
    p.add_argument("--use-cache", action="store_true", default=True,
                   help="启用 KV Cache (default: True)")
    p.add_argument("--no-use-cache", action="store_false", dest="use_cache",
                   help="禁用 KV Cache")
    p.add_argument("--enable-accel-async", action="store_true", default=True,
                   help="启用加速设备异步 (default: True)")
    p.add_argument("--no-enable-accel-async", action="store_false", dest="enable_accel_async",
                   help="禁用加速设备异步")
    p.add_argument("--debug-decode-special-tokens", action="store_true", default=False,
                   help="对比 skip_special_tokens 前后的 think 标签 (default: False)")

    # --- vllm 后端参数 ---
    p.add_argument("--vllm-base-url", default="http://127.0.0.1:8000",
                   help="vLLM API 地址；自动启动时也从这里提取监听 host/port (default: http://127.0.0.1:8000)")
    p.add_argument("--vllm-endpoint", default="auto",
                   choices=["auto", "chat", "generate", "openai", "openai_chat"],
                   help="vLLM endpoint 类型 (default: auto)")
    p.add_argument("--vllm-model-name", default=None,
                   help="vLLM 模型名称 (auto 时可省略)")
    p.add_argument("--vllm-api-key", default="EMPTY",
                   help="vLLM API Key (default: EMPTY)")
    p.add_argument("--vllm-retries", type=int, default=3,
                   help="vLLM 请求最大重试次数 (default: 3)")
    p.add_argument("--vllm-retry-interval", type=float, default=5.0,
                   help="vLLM 重试间隔秒数 (default: 5.0)")
    p.add_argument("--vllm-tokenizer-path", default=None,
                   help="vLLM tokenizer 路径 (default: 与 --base-model-path 相同)")
    p.add_argument("--vllm-repetition-penalty", type=float, default=1.0,
                   help="vLLM repetition_penalty (default: 1.0)")
    p.add_argument("--vllm-top-k", type=int, default=0,
                   help="vLLM top_k (default: 0)")
    p.add_argument("--vllm-min-p", type=float, default=0.0,
                   help="vLLM min_p (default: 0.0)")

    # --- vLLM server 自动启动参数 ---
    g = p.add_argument_group("vLLM server auto-start",
                             "当 --vllm-auto-start 开启时自动启动 vLLM server")
    g.add_argument("--vllm-auto-start", action="store_true", default=True,
                   help="自动启动 vLLM OpenAI server (default: True)")
    g.add_argument("--no-vllm-auto-start", action="store_false", dest="vllm_auto_start",
                   help="不自动启动 vLLM server，连接已有 server")
    g.add_argument("--vllm-server-tp", type=int, default=None,
                   help="vLLM server tensor-parallel-size (default: 自动检测可用 GPU 数)")
    g.add_argument("--vllm-server-gpu-mem-util", type=float, default=0.9,
                   help="vLLM server gpu-memory-utilization (default: 0.9)")
    g.add_argument("--vllm-server-max-model-len", type=int, default=None,
                   help="vLLM server max-model-len (default: 由 vLLM 自动决定)")
    g.add_argument("--vllm-server-dtype", default="auto",
                   help="vLLM server dtype (default: auto)")
    g.add_argument("--vllm-server-trust-remote-code", action="store_true", default=True,
                   help="vLLM server trust-remote-code (default: True)")
    g.add_argument("--no-vllm-server-trust-remote-code", action="store_false",
                   dest="vllm_server_trust_remote_code",
                   help="vLLM server 不启用 trust-remote-code")
    g.add_argument("--vllm-server-log-file", default="./tmp/vllm_server.log",
                   help="vLLM server 日志文件路径 (default: ./tmp/vllm_server.log)")

    return p.parse_args()


# ================= local 后端 runner =================
def run_single_task_local(
    task_id: int,
    devices: str,
    dataset: list,
    task_dirs: dict,
    prompt_template: str,
    args: argparse.Namespace,
) -> None:
    """local 后端：单任务执行。"""
    task_name = TASK_PROMPTS[task_id][1]
    print(f"\n{'=' * 60}")
    print(f"[TASK {task_id}] {task_name} - 进程启动，等待设备初始化...")
    print(f"[TASK {task_id}] 将使用设备: {devices}")
    print(f"{'=' * 60}", flush=True)

    accelerator = detect_accelerator()
    init_env_for_task(devices, accelerator, enable_async=args.enable_accel_async, print_debug=args.debug)

    if args.debug:
        print(f"[TASK {task_id}] 设备: {accelerator}:{devices}, 样本数: {len(dataset)}", flush=True)

    # 加载模型
    try:
        from transformers import AutoModelForCausalLM

        adapter_config = os.path.join(args.model_path, "adapter_config.json")
        if args.use_tuned_model and os.path.exists(adapter_config):
            model = load_lora_model(args.base_model_path, args.model_path, accelerator, print_debug=args.debug)
            tokenizer_source = args.base_model_path
        else:
            load_path = args.model_path if args.use_tuned_model else args.base_model_path
            if args.debug and not args.use_tuned_model:
                print(f"[INFO] 仅加载基座模型：{load_path}")
            import torch
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=get_infer_dtype(accelerator),
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()
            tokenizer_source = load_path

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        tokenizer = configure_tokenizer_for_chat(
            tokenizer, enable_thinking=args.enable_thinking,
            force_user_role=args.force_user_role, print_debug=args.debug,
        )
    except Exception as e:
        print(f"[ERROR] TASK {task_id} 模型加载失败: {e}")
        return

    # 预热
    if args.enable_warmup:
        model_warmup(
            model, tokenizer, accelerator,
            enable_thinking=args.enable_thinking, force_user_role=args.force_user_role,
            enable_async=args.enable_accel_async, print_debug=args.debug,
        )

    # 推理循环
    task_start = time.time()
    cumulative_time = 0.0

    for idx, item in enumerate(dataset):
        raw_text = item.get("input_context", "")
        prompt = build_prompt(prompt_template, raw_text)
        if not prompt:
            continue

        sample_start = time.time()
        try:
            result = generate_text_local(
                model, tokenizer, prompt, accelerator,
                max_input_tokens=args.max_input_tokens,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
                do_sample=args.do_sample,
                use_cache=args.use_cache,
                enable_thinking=args.enable_thinking,
                force_user_role=args.force_user_role,
                enable_async=args.enable_accel_async,
                debug_decode_special_tokens=args.debug_decode_special_tokens,
                print_debug=args.debug,
            )
            save_result(result, task_dirs[task_id], item["id"], task_name)
        except Exception as e:
            print(f"[WARN] TASK {task_id} 样本 {idx + 1} 生成失败: {e}", flush=True)
            continue

        sample_elapsed = round(time.time() - sample_start, 2)
        cumulative_time += sample_elapsed

        done = idx + 1
        if should_print_progress(done, len(dataset)):
            print(f"[TASK {task_id}] 进度: {done}/{len(dataset)}, 累计耗时: {round(cumulative_time, 2)}s", flush=True)

    task_elapsed = round(time.time() - task_start, 2)
    avg = round(task_elapsed / len(dataset), 2) if dataset else 0
    print(f"\n[TASK {task_id}] 完成！耗时: {task_elapsed}s, 平均: {avg}s/样本")


# ================= vllm 后端 runner =================
def run_single_task_vllm(
    task_id: int,
    dataset: list,
    task_dirs: dict,
    prompt_template: str,
    resolved_endpoint: str,
    resolved_model_name: str,
    args: argparse.Namespace,
) -> None:
    """vllm 后端：单任务执行。"""
    task_name = TASK_PROMPTS[task_id][1]
    base_url = args.vllm_base_url.rstrip("/")
    tokenizer_path = args.vllm_tokenizer_path or args.base_model_path

    print(f"\n{'=' * 60}")
    print(f"[TASK {task_id}] {task_name} - 进程启动，准备调用 vLLM server...")
    print(f"{'=' * 60}", flush=True)

    if args.debug:
        print(f"[TASK {task_id}] vLLM 地址: {base_url}", flush=True)
        print(f"[TASK {task_id}] vLLM endpoint: /{resolved_endpoint}/", flush=True)
        print(f"[TASK {task_id}] vLLM model: {resolved_model_name}", flush=True)
        print(f"[TASK {task_id}] 样本数: {len(dataset)}", flush=True)

    task_start = time.time()
    cumulative_time = 0.0

    for idx, item in enumerate(dataset):
        raw_text = item.get("input_context", "")
        prompt = build_prompt(prompt_template, raw_text)
        if not prompt:
            continue

        sample_start = time.time()
        try:
            result = generate_text_via_vllm(
                prompt=prompt,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
                do_sample=args.do_sample,
                resolved_endpoint=resolved_endpoint,
                base_url=base_url,
                model_name=resolved_model_name,
                api_key=args.vllm_api_key,
                tokenizer_path=tokenizer_path,
                max_retries=args.vllm_retries,
                retry_interval=args.vllm_retry_interval,
                repetition_penalty=args.vllm_repetition_penalty,
                top_k=args.vllm_top_k,
                min_p=args.vllm_min_p,
                enable_thinking=args.enable_thinking,
                force_user_role=args.force_user_role,
                print_debug=args.debug,
            )
            save_result(result, task_dirs[task_id], item["id"], task_name)
        except Exception as exc:
            print(f"[WARN] TASK {task_id} 样本 {idx + 1} 生成失败: {exc}", flush=True)
            continue

        sample_elapsed = round(time.time() - sample_start, 2)
        cumulative_time += sample_elapsed

        done = idx + 1
        if should_print_progress(done, len(dataset)):
            print(
                f"[TASK {task_id}] 进度: {done}/{len(dataset)}, 累计耗时: {round(cumulative_time, 2)}s",
                flush=True,
            )

    task_elapsed = round(time.time() - task_start, 2)
    avg = round(task_elapsed / len(dataset), 2) if dataset else 0
    print(f"\n[TASK {task_id}] 完成！耗时: {task_elapsed}s, 平均: {avg}s/样本", flush=True)


def start_managed_vllm_server(args: argparse.Namespace, base_url: str):
    """启动本脚本托管的 vLLM OpenAI server，并确保连接目标不是已有服务。"""
    if args.vllm_endpoint in {"chat", "generate"}:
        raise ValueError(
            "自动启动只支持 vLLM OpenAI server；"
            "请使用 --vllm-endpoint auto/openai/openai_chat，"
            "或传 --no-vllm-auto-start 连接 TRL vllm-serve。"
        )

    try:
        check_vllm_server(base_url, "openai", timeout=2)
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            f"{base_url} 已经有可用 server。为避免误连错误模型，"
            "请换一个 --vllm-base-url 端口，或传 --no-vllm-auto-start 显式复用已有 server。"
        )

    server_model, use_lora, lora_modules, _ = resolve_vllm_model_config(args)

    server_host, server_port = extract_host_port(base_url)
    server_tp = args.vllm_server_tp or auto_detect_tp_size()

    print(
        f"[INFO] 正在启动 vLLM server: {base_url} "
        f"(model={server_model}, tp={server_tp}, lora={'on' if use_lora else 'off'})",
        flush=True,
    )
    print(f"[INFO] vLLM server 日志: {args.vllm_server_log_file}", flush=True)

    cmd = build_vllm_server_command(
        model_path=server_model,
        host=server_host,
        port=server_port,
        tensor_parallel_size=server_tp,
        gpu_memory_utilization=args.vllm_server_gpu_mem_util,
        max_model_len=args.vllm_server_max_model_len,
        dtype=args.vllm_server_dtype,
        trust_remote_code=args.vllm_server_trust_remote_code,
        enable_lora=use_lora,
        lora_modules=lora_modules,
    )

    process = start_vllm_server(
        cmd,
        print_debug=args.debug,
        log_file=args.vllm_server_log_file,
    )
    try:
        ready = wait_for_vllm_server(
            base_url=base_url,
            resolved_endpoint="openai",
            poll_interval=5.0,
            process=process,
            api_key=args.vllm_api_key,
            print_debug=args.debug,
        )
        if not ready:
            raise RuntimeError(
                f"vLLM server 进程提前退出，请查看日志: {args.vllm_server_log_file}"
            )
    except BaseException:
        stop_vllm_server(process, print_debug=args.debug)
        raise

    print(f"[INFO] vLLM server 已启动并就绪: {base_url}", flush=True)
    return process


def resolve_vllm_model_config(args: argparse.Namespace) -> tuple[str, bool, str | None, str]:
    adapter_config = os.path.join(args.model_path, "adapter_config.json")
    if args.use_tuned_model and os.path.exists(adapter_config):
        return args.base_model_path, True, f"tuned={args.model_path}", "tuned"
    model_path = args.model_path if args.use_tuned_model else args.base_model_path
    return model_path, False, None, model_path


def should_print_progress(done: int, total: int) -> bool:
    return done == 1 or done == total or done % 10 == 0


def stop_worker_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def raise_keyboard_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


# ================= 主流程 =================
def main() -> None:
    args = parse_args()

    # 解析任务编号
    try:
        task_ids = [int(t.strip()) for t in args.tasks.split(",")]
    except ValueError:
        print(f"[ERROR] --tasks 格式错误，应为逗号分隔的数字，如 1,2,3,4")
        return

    invalid = [t for t in task_ids if t not in TASK_PROMPTS]
    if invalid:
        print(f"[ERROR] 无效的任务编号: {invalid}，有效范围: 1,2,3,4")
        return

    # 输出目录默认值
    if args.output_dir is None:
        if args.backend == "vllm":
            args.output_dir = "./results/vllm_output"
        elif args.use_tuned_model:
            args.output_dir = "./results/grpo_model-qwen14b"
        else:
            args.output_dir = "./results/base_model_qwen14b"

    # 初始化输出目录
    root_output_dir, task_dirs = init_directories(args.output_dir)

    # 加载数据
    dataset = load_medical_records_from_dir(
        args.data_dir, sample_size=args.sample_size, print_debug=args.debug,
    )
    if not dataset:
        print("[ERROR] 数据加载为空，请检查数据目录")
        return

    if args.debug:
        print(f"[INFO] 后端：{args.backend}")
        print(f"[INFO] 输出目录：{root_output_dir}")
        print(f"[INFO] 并行模式：{args.parallel}")
        print(f"[INFO] 启用任务：{task_ids}")
        print(f"[INFO] 成功加载 {len(dataset)} 份病历数据")

    # 准备任务列表
    tasks = [(tid, TASK_PROMPTS[tid][0], TASK_PROMPTS[tid][1]) for tid in task_ids]

    if not tasks:
        print("[ERROR] 没有启用的任务")
        return

    if args.backend == "vllm":
        vllm_model_path, use_lora, _, default_model_name = resolve_vllm_model_config(args)
        model_desc = (
            f"{args.base_model_path} + LoRA({args.model_path})"
            if use_lora else vllm_model_path
        )
        print(
            f"[INFO] 后端: vllm | 模型: {model_desc} | API model: {args.vllm_model_name or default_model_name}",
            flush=True,
        )
    else:
        adapter_config = os.path.join(args.model_path, "adapter_config.json")
        use_lora = args.use_tuned_model and os.path.exists(adapter_config)
        model_desc = (
            f"{args.base_model_path} + LoRA({args.model_path})"
            if use_lora else (args.model_path if args.use_tuned_model else args.base_model_path)
        )
        print(f"[INFO] 后端: local | 模型: {model_desc}", flush=True)
    print(
        f"[INFO] 输出目录: {root_output_dir} | 样本数: {len(dataset)} | 任务: {task_ids} | 并行: {args.parallel}",
        flush=True,
    )

    # --- local 后端 ---
    if args.backend == "local":
        total_start = time.time()
        device_map = auto_allocate_devices(len(tasks), print_debug=args.debug)
        processes = []

        try:
            if args.parallel and len(tasks) > 1:
                if args.debug:
                    print(f"\n[INFO] 启动并行模式，共 {len(tasks)} 个任务")
                for task_id, prompt_template, task_name in tasks:
                    devices = device_map.get(task_id, "0")
                    p = mp.Process(
                        target=run_single_task_local,
                        args=(task_id, devices, dataset, task_dirs, prompt_template, args),
                    )
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            else:
                for task_id, prompt_template, task_name in tasks:
                    devices = device_map.get(task_id, "0")
                    run_single_task_local(
                        task_id, devices, dataset, task_dirs, prompt_template, args,
                    )
        except KeyboardInterrupt:
            print("\n[INFO] 收到中断信号，正在清理 local 任务进程...", flush=True)
            raise
        finally:
            stop_worker_processes(processes)

        total_elapsed = round(time.time() - total_start, 2)
        print(f"\n{'=' * 60}")
        print(f"[SUCCESS] 全流程执行完毕")
        print(f"[TIME] 总耗时：{total_elapsed}s")
        print(f"{'=' * 60}")

    # --- vllm 后端 ---
    else:
        base_url = args.vllm_base_url.rstrip("/")
        tokenizer_path = args.vllm_tokenizer_path or args.base_model_path
        _, _, _, default_model_name = resolve_vllm_model_config(args)
        model_name = args.vllm_model_name or (default_model_name if args.vllm_auto_start else tokenizer_path)

        if args.vllm_endpoint not in {"auto", "chat", "generate", "openai", "openai_chat"}:
            print(f"[ERROR] --vllm-endpoint 只能是 auto、chat、generate、openai 或 openai_chat")
            return

        # ---- 自动启动 vLLM server ----
        server_process = None
        processes = []
        try:
            if args.vllm_auto_start:
                server_process = start_managed_vllm_server(args, base_url)

            # 解析 endpoint（auto-start 后可能需要重新检测）
            resolved_endpoint, resolved_model_name = resolve_vllm_endpoint(
                base_url=base_url,
                endpoint=args.vllm_endpoint,
                api_key=args.vllm_api_key,
                model_name=model_name,
                tokenizer_path=tokenizer_path,
                print_debug=args.debug,
            )

            # 非 auto-start 时，检查 server 是否可达
            if not args.vllm_auto_start:
                check_vllm_server(base_url, resolved_endpoint)

            total_start = time.time()

            if args.parallel and len(tasks) > 1:
                if args.debug:
                    print(f"\n[INFO] 启动并行模式，共 {len(tasks)} 个任务")
                for task_id, prompt_template, task_name in tasks:
                    p = mp.Process(
                        target=run_single_task_vllm,
                        args=(task_id, dataset, task_dirs, prompt_template,
                              resolved_endpoint, resolved_model_name, args),
                    )
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            else:
                for task_id, prompt_template, task_name in tasks:
                    run_single_task_vllm(
                        task_id, dataset, task_dirs, prompt_template,
                        resolved_endpoint, resolved_model_name, args,
                    )

            total_elapsed = round(time.time() - total_start, 2)
            print(f"\n{'=' * 60}")
            print("[SUCCESS] 全流程执行完毕")
            print(f"[TIME] 总耗时：{total_elapsed}s")
            print(f"{'=' * 60}")

        except (RuntimeError, ValueError) as exc:
            print(f"[ERROR] {exc}", flush=True)
            return
        except KeyboardInterrupt:
            print("\n[INFO] 收到中断信号，正在清理 vLLM 任务和 server...", flush=True)
            raise
        finally:
            stop_worker_processes(processes)
            if server_process is not None:
                stop_vllm_server(server_process, print_debug=args.debug)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] 已中断，清理完成", flush=True)
        sys.exit(130)

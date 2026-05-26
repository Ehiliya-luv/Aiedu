# -*- coding: utf-8 -*-
"""vLLM HTTP client and managed server helpers for generate_output.py."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request
from urllib.parse import urlparse

import torch

# Must be set before importing transformers/huggingface_hub.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoTokenizer

from ..text import strip_think_content
from .local import detect_accelerator
def _get_tokenizer(tokenizer_path: str, print_debug: bool = False):
    global _tokenizer
    if _tokenizer is None:
        if print_debug:
            print(f"[INFO] 加载 tokenizer：{tokenizer_path}", flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return _tokenizer


def _get_json(url: str, api_key: str, timeout: int = 30) -> Dict[str, Any]:
    req = request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def resolve_vllm_endpoint(
    base_url: str,
    endpoint: str,
    api_key: str,
    model_name: str,
    tokenizer_path: str,
    print_debug: bool = False,
) -> Tuple[str, str]:
    """自动区分原生 vLLM OpenAI server 和 TRL vllm-serve。

    Returns:
        (resolved_endpoint, resolved_model_name)
    """
    if endpoint != "auto":
        return endpoint, model_name

    try:
        models_payload = _get_json(f"{base_url}/v1/models", api_key, timeout=30)
        model_ids = [
            item.get("id")
            for item in models_payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]
        resolved_model = model_name
        if not model_name or model_name == tokenizer_path:
            lora_model = next((mid for mid in model_ids if "lora" in mid.lower()), None)
            resolved_model = lora_model or (model_ids[0] if model_ids else tokenizer_path)
        if print_debug:
            print(f"[INFO] 检测到原生 vLLM OpenAI 接口，可用模型: {model_ids}", flush=True)
        return "openai", resolved_model
    except Exception as exc:
        if print_debug:
            print(f"[INFO] 未检测到 /v1/models，回退 TRL vllm-serve 接口: {exc}", flush=True)
        return "chat", model_name


def check_vllm_server(base_url: str, resolved_endpoint: str, timeout: int = 30) -> None:
    """检查 vLLM server 是否可用。"""
    health_path = "/health" if resolved_endpoint in {"openai", "openai_chat"} else "/health/"
    url = f"{base_url}{health_path}"
    try:
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"vLLM server health check failed: HTTP {resp.status}")
    except Exception as exc:
        raise RuntimeError(f"无法连接 vLLM server：{url}，请先启动 vLLM server。原始错误: {exc}") from exc


def _build_chat_template_kwargs(enable_thinking: bool) -> Dict[str, Any]:
    if not enable_thinking:
        return {}
    return {"thinking_mode": "on"}


def _build_generate_prompt(tokenizer, prompt: str, enable_thinking: bool, force_user_role: bool) -> str:
    if not force_user_role:
        return prompt
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return prompt
    chat_kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    chat_kwargs.update(_build_chat_template_kwargs(enable_thinking))
    try:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **chat_kwargs)
    except TypeError:
        chat_kwargs.pop("thinking_mode", None)
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **chat_kwargs)


def _build_request_payload(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    resolved_endpoint: str,
    base_url: str,
    model_name: str,
    api_key: str,
    repetition_penalty: float,
    top_k: int,
    min_p: float,
    tokenizer_path: str,
    enable_thinking: bool,
    force_user_role: bool,
    print_debug: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """构建请求 payload，返回 (url, payload)。"""
    generation_temperature = temperature if do_sample else 0.0

    if resolved_endpoint in {"openai", "openai_chat"}:
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": generation_temperature,
            "top_p": top_p if do_sample else 1.0,
        }
        return f"{base_url}/v1/chat/completions", payload

    common: Dict[str, Any] = {
        "n": 1,
        "repetition_penalty": repetition_penalty,
        "temperature": generation_temperature,
        "top_p": top_p if do_sample else 1.0,
        "top_k": top_k,
        "min_p": min_p,
        "max_tokens": max_tokens,
        "logprobs": None,
        "structured_outputs_regex": None,
        "generation_kwargs": {},
    }

    if resolved_endpoint == "generate":
        tokenizer = _get_tokenizer(tokenizer_path, print_debug)
        payload = {"prompts": [_build_generate_prompt(tokenizer, prompt, enable_thinking, force_user_role)], **common}
        return f"{base_url}/generate/", payload

    payload = {
        "messages": [[{"role": "user", "content": prompt}]],
        "chat_template_kwargs": _build_chat_template_kwargs(enable_thinking),
        "tools": None,
        **common,
    }
    return f"{base_url}/chat/", payload


def generate_text_via_vllm(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    resolved_endpoint: str,
    base_url: str,
    model_name: str,
    api_key: str,
    tokenizer_path: str,
    max_retries: int,
    retry_interval: float,
    repetition_penalty: float,
    top_k: int,
    min_p: float,
    enable_thinking: bool,
    force_user_role: bool,
    print_debug: bool = False,
) -> str:
    """vllm 后端：通过 HTTP 请求 vLLM server 生成文本。"""
    if not prompt:
        return ""

    url, payload = _build_request_payload(
        prompt, max_tokens, temperature, top_p, do_sample,
        resolved_endpoint, base_url, model_name, api_key,
        repetition_penalty, top_k, min_p, tokenizer_path,
        enable_thinking, force_user_role, print_debug,
    )
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        req = request.Request(
            url,
            data=payload_bytes,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with request.urlopen(req) as resp:
                body = resp.read().decode("utf-8")
            response_json = json.loads(body)
            if resolved_endpoint in {"openai", "openai_chat"}:
                choices = response_json.get("choices") or []
                if not choices:
                    raise ValueError(f"OpenAI 兼容响应缺少 choices: {response_json}")
                message = (choices[0] or {}).get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    return strip_think_content(content)
                raise ValueError(f"OpenAI 兼容响应缺少 message.content: {response_json}")
            completion_ids = response_json.get("completion_ids")
            if not completion_ids:
                raise ValueError(f"vLLM 响应缺少 completion_ids: {response_json}")
            first = completion_ids[0]
            if not isinstance(first, list):
                raise ValueError(f"vLLM completion_ids 格式异常: {response_json}")
            tokenizer = _get_tokenizer(tokenizer_path, print_debug)
            return strip_think_content(tokenizer.decode(first, skip_special_tokens=True))
        except (error.HTTPError, error.URLError, socket.timeout, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if print_debug:
                print(f"[WARN] vLLM 调用失败，第 {attempt}/{max_retries} 次: {exc}", flush=True)
            if attempt < max_retries:
                time.sleep(retry_interval)

    raise RuntimeError(f"vLLM 调用最终失败: {last_error}")


# ================= vLLM server 生命周期管理 =================

def extract_host_port(base_url: str) -> Tuple[str, int]:
    """从 base_url 提取 host 和 port。"""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    return host, port


def auto_detect_tp_size() -> int:
    """自动检测可用 GPU 数作为 tensor-parallel-size。"""
    accelerator = detect_accelerator()
    if accelerator == "npu":
        return torch.npu.device_count() if hasattr(torch, "npu") else 1
    elif accelerator == "cuda":
        return torch.cuda.device_count()
    return 1


def build_vllm_server_command(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.25,
    max_model_len: Optional[int] = None,
    dtype: str = "auto",
    trust_remote_code: bool = True,
    enable_lora: bool = False,
    lora_modules: Optional[str] = None,
    served_model_name: Optional[str] = None,
) -> List[str]:
    """构建 vLLM server 启动命令。

    Returns:
        命令行参数列表，可直接传给 subprocess.Popen。
    """
    from utils.model import resolve_model_path

    resolved_model_path = resolve_model_path(model_path)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", resolved_model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    if enable_lora:
        cmd.append("--enable-lora")
    if lora_modules:
        cmd.extend(["--lora-modules", lora_modules])
    return cmd


def start_vllm_server(
    cmd: List[str],
    print_debug: bool = False,
    log_file: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    """启动 vLLM server 子进程。

    Args:
        cmd:            build_vllm_server_command 返回的命令列表
        print_debug:    是否打印启动信息
        log_file:       可选日志文件路径，为 None 时输出到 /dev/null
        env:            可选子进程环境变量

    Returns:
        subprocess.Popen 实例
    """
    if print_debug:
        print(f"[INFO] 启动 vLLM server: {' '.join(cmd)}", flush=True)

    # 日志输出目标
    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        out = open(log_file, "a", encoding="utf-8")
    else:
        out = subprocess.DEVNULL

    child_env = os.environ.copy()
    child_env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if env is not None:
        child_env.update(env)

    popen_kwargs: Dict[str, Any] = {
        "stdout": out,
        "stderr": subprocess.STDOUT,
        "env": child_env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(cmd, **popen_kwargs)
    if os.name != "nt":
        try:
            process._aiedu_pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            process._aiedu_pgid = None

    if print_debug:
        print(f"[INFO] vLLM server 子进程 PID: {process.pid}", flush=True)

    return process


def wait_for_vllm_server(
    base_url: str,
    resolved_endpoint: str,
    poll_interval: float = 5.0,
    process: Optional[subprocess.Popen] = None,
    api_key: str = "EMPTY",
    print_debug: bool = False,
) -> bool:
    """等待 vLLM server 就绪。

    Args:
        base_url:          server 基地址
        resolved_endpoint: 已解析的 endpoint 类型
        poll_interval:     轮询间隔秒数
        process:           可选的已启动子进程；提前退出时立即失败
        api_key:           OpenAI 兼容接口 API key
        print_debug:       是否打印等待状态

    Returns:
        True 就绪，False 表示 server 进程提前退出
    """
    start_time = time.time()
    while True:
        if process is not None and process.poll() is not None:
            if print_debug:
                print(f"[ERROR] vLLM server 进程提前退出，退出码: {process.returncode}", flush=True)
            return False
        try:
            check_vllm_server(base_url, resolved_endpoint, timeout=10)
            if resolved_endpoint in {"openai", "openai_chat"}:
                _get_json(f"{base_url}/v1/models", api_key, timeout=10)
            elapsed = round(time.time() - start_time, 1)
            if print_debug:
                print(f"[INFO] vLLM server 已就绪 (耗时 {elapsed}s)", flush=True)
            return True
        except Exception:
            elapsed = round(time.time() - start_time, 1)
            if print_debug:
                print(f"[INFO] 等待 vLLM server 就绪... (已等待 {elapsed}s)", flush=True)
            time.sleep(poll_interval)


def stop_vllm_server(
    process: subprocess.Popen,
    timeout: int = 30,
    print_debug: bool = False,
) -> None:
    """停止 vLLM server 子进程。

    优雅终止 → 等待 → 强制杀死。
    """
    pgid = getattr(process, "_aiedu_pgid", None)
    if process.poll() is not None and (os.name == "nt" or pgid is None):
        if print_debug:
            print("[INFO] vLLM server 进程已退出", flush=True)
        return

    if print_debug:
        print("[INFO] 正在关闭 vLLM server...", flush=True)

    if os.name == "nt":
        # Windows: CTRL_BREAK_EVENT 通知进程组
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        try:
            os.killpg(pgid or os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return

    try:
        if process.poll() is None:
            process.wait(timeout=timeout)
        else:
            time.sleep(min(timeout, 2))
        if print_debug:
            print("[INFO] vLLM server 已优雅退出", flush=True)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(pgid or os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                return
        if print_debug:
            print("[WARN] vLLM server 未能优雅退出，已强制终止", flush=True)

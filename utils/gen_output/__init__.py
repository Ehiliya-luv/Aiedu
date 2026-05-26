# -*- coding: utf-8 -*-
"""Public helpers for generate_output.py."""

from .common import (
    build_prompt,
    init_directories,
    load_medical_records_from_dir,
    save_result,
)
from .local import (
    auto_allocate_devices,
    configure_tokenizer_for_chat,
    detect_accelerator,
    generate_text_local,
    get_infer_dtype,
    init_env_for_task,
    load_lora_model,
    model_warmup,
)
from .vllm import (
    auto_detect_tp_size,
    build_vllm_server_command,
    check_vllm_server,
    extract_host_port,
    generate_text_via_vllm,
    resolve_vllm_endpoint,
    start_vllm_server,
    stop_vllm_server,
    wait_for_vllm_server,
)

__all__ = [
    "auto_allocate_devices",
    "auto_detect_tp_size",
    "build_prompt",
    "build_vllm_server_command",
    "check_vllm_server",
    "configure_tokenizer_for_chat",
    "detect_accelerator",
    "extract_host_port",
    "generate_text_local",
    "generate_text_via_vllm",
    "get_infer_dtype",
    "init_directories",
    "init_env_for_task",
    "load_lora_model",
    "load_medical_records_from_dir",
    "model_warmup",
    "resolve_vllm_endpoint",
    "save_result",
    "start_vllm_server",
    "stop_vllm_server",
    "wait_for_vllm_server",
]

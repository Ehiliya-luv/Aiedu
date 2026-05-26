# -*- coding: utf-8 -*-
"""Local Transformers backend helpers for generate_output.py."""

from __future__ import annotations

import inspect
import os
import re
from typing import Any, Dict

from ..text import strip_think_content

_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)

try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = True
except ImportError:
    HAS_TORCH_NPU = False

import torch  # noqa: E402
def detect_accelerator() -> str:
    """自动检测可用加速后端：npu > cuda > cpu。"""
    if HAS_TORCH_NPU and hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return "npu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def auto_allocate_devices(num_tasks: int, print_debug: bool = False) -> Dict[int, str]:
    """自动分配 GPU/NPU 设备，均分给各任务。"""
    accelerator = detect_accelerator()
    if accelerator == "npu":
        num_devices = torch.npu.device_count() if hasattr(torch, "npu") else 1
    elif accelerator == "cuda":
        num_devices = torch.cuda.device_count()
    else:
        num_devices = 1

    num_devices = max(num_devices, 1)
    device_map: Dict[int, str] = {}
    base = num_devices // num_tasks
    remainder = num_devices % num_tasks
    start = 0
    for i in range(num_tasks):
        count = max(base + (1 if i < remainder else 0), 1)
        device_map[i + 1] = ",".join(str(d) for d in range(start, start + count))
        start += count

    if print_debug:
        print(f"[INFO] 自动设备分配：{num_devices} 卡 / {num_tasks} 任务 → {device_map}")
    return device_map


def get_infer_dtype(accelerator: str):
    return torch.float16 if accelerator in {"npu", "cuda"} else torch.float32


def sync_accelerator(accelerator: str, enable_async: bool = True) -> None:
    """按后端执行同步。"""
    if not enable_async:
        return
    if accelerator == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif accelerator == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def init_env_for_task(devices: str, accelerator: str, enable_async: bool = True, print_debug: bool = False) -> None:
    """为单个任务设置加速设备可见卡与性能相关环境变量。"""
    if accelerator == "npu":
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = devices
        if enable_async:
            os.environ["ASCEND_LAUNCH_BLOCKING"] = "0"
            os.environ["NPU_LAUNCH_MODE"] = "1"
    elif accelerator == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        if enable_async:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    if print_debug:
        print(f"[INFO] 任务设备：{accelerator}:{devices}")


def load_lora_model(base_path: str, adapter_path: str, accelerator: str, print_debug: bool = False):
    """加载 LoRA 模型。"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if print_debug:
        print(f"[INFO] 正在加载基座模型：{base_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=get_infer_dtype(accelerator),
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if print_debug:
        print(f"[INFO] 正在加载 LoRA 适配器：{adapter_path}")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map="auto",
        is_trainable=False,
    )
    model.eval()
    return model


def configure_tokenizer_for_chat(tokenizer, enable_thinking: bool, force_user_role: bool, print_debug: bool = False):
    """探测 tokenizer 是否支持 chat template / thinking_mode。"""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    tokenizer._aiedu_chat_template_supported = False
    tokenizer._aiedu_thinking_mode_supported = False

    if not force_user_role:
        return tokenizer

    if not callable(apply_chat_template):
        if print_debug:
            print("[WARN] 当前 tokenizer 不提供 apply_chat_template，将回退为原始 prompt 推理")
        return tokenizer

    tokenizer._aiedu_chat_template_supported = True

    try:
        signature = inspect.signature(apply_chat_template)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        tokenizer._aiedu_thinking_mode_supported = (
            "thinking_mode" in signature.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        )

    if print_debug:
        if tokenizer._aiedu_thinking_mode_supported:
            print(f"[INFO] tokenizer chat template 已启用，thinking_mode={'on' if enable_thinking else 'off'}")
        else:
            print("[WARN] tokenizer chat template 可用，但不支持 thinking_mode 参数")

    return tokenizer


def build_model_inputs(tokenizer, prompt: str, model, max_length: int,
                       enable_thinking: bool, force_user_role: bool):
    """统一构造模型输入，优先采用 user role chat 模式。"""
    prompt_text = prompt

    if force_user_role and getattr(tokenizer, "_aiedu_chat_template_supported", False):
        messages = [{"role": "user", "content": prompt}]
        chat_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking and getattr(tokenizer, "_aiedu_thinking_mode_supported", False):
            chat_kwargs["thinking_mode"] = "on"
        prompt_text = tokenizer.apply_chat_template(messages, **chat_kwargs)

    return tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    ).to(model.device)


def generate_text_local(
    model, tokenizer, prompt: str, accelerator: str,
    max_input_tokens: int, max_tokens: int,
    temperature: float, top_p: float, do_sample: bool,
    use_cache: bool, enable_thinking: bool, force_user_role: bool,
    enable_async: bool, debug_decode_special_tokens: bool,
    print_debug: bool = False,
) -> str:
    """local 后端：Transformers 模型推理。"""
    if not prompt:
        return ""

    inputs = build_model_inputs(tokenizer, prompt, model, max_length=max_input_tokens,
                                enable_thinking=enable_thinking, force_user_role=force_user_role)
    sync_accelerator(accelerator, enable_async)

    with torch.no_grad():
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample and temperature > 0:
            generate_kwargs["temperature"] = temperature
        if do_sample and top_p < 1.0:
            generate_kwargs["top_p"] = top_p
        if use_cache:
            generate_kwargs["use_cache"] = True

        outputs = model.generate(**inputs, **generate_kwargs)

    sync_accelerator(accelerator, enable_async)

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if debug_decode_special_tokens:
        decoded_raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        raw_has_think = bool(_THINK_TAG_RE.search(decoded_raw))
        clean_has_think = bool(_THINK_TAG_RE.search(decoded_text))
        cleaned_text = strip_think_content(decoded_text)
        print(
            "[DEBUG] decode special tokens: "
            f"raw_has_think={raw_has_think}, clean_has_think={clean_has_think}, "
            f"raw_len={len(decoded_raw)}, clean_len={len(decoded_text)}, "
            f"cleaned_len={len(cleaned_text)}",
            flush=True,
        )
        print("[DEBUG] cleaned preview:", cleaned_text[:500].replace("\n", "\\n"), flush=True)
        return cleaned_text

    return strip_think_content(decoded_text)


def model_warmup(model, tokenizer, accelerator: str,
                 enable_thinking: bool, force_user_role: bool,
                 enable_async: bool, print_debug: bool = False,
                 dummy_prompt: str = "测试") -> None:
    """模型预热。"""
    if print_debug:
        print("[INFO] 执行模型预热 (Warmup)...")
    try:
        inputs = build_model_inputs(tokenizer, dummy_prompt, model, max_length=512,
                                    enable_thinking=enable_thinking, force_user_role=force_user_role)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        sync_accelerator(accelerator, enable_async)
        if print_debug:
            print("[INFO] 模型预热完成")
    except Exception as e:
        if print_debug:
            print(f"[WARN] 模型预热失败 (可忽略): {e}")

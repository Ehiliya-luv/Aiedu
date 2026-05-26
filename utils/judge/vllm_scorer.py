# -*- coding: utf-8 -*-
"""vLLM Python engine scorer — 精确获取 0~9 数字 token 的 logprobs。

通过 SamplingParams(logprob_token_ids=digit_token_ids) 精确指定 0~9 共 10 个
数字 token 的 ID，vLLM 引擎直接按 ID 查表返回 logprobs，无需 top-K 排序。

对比 HTTP API 模式（受 top_logprobs 限制，可能拿不全 10 个 digit logprobs），
vLLM Python engine 模式通过 logprob_token_ids 100% 保证覆盖全部 10 个数字，
且性能优于 logprobs=K（省去 top-K 排序，仅返回 10 个值）。

使用方式:
    scorer = VLLMEngineScorer(model_path="resources/model/Baichuan-M2-32B-0226")
    result = scorer.score_section(question_type="A1", prompt_text="...", candidate_text="...")
    # result["score"] → 0.0~1.0 归一化分数

注意:
    - vLLM 是 server 端依赖，本地开发环境可能未安装
    - 在 VERL 多进程训练中，需要确保该 engine 只在一个进程中初始化
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .settings import (
    JUDGE_MAX_OUTLINE_ITEMS,
    JUDGE_MAX_PDF_ADVICE,
    JUDGE_PDF_RECALL_TOP_K,
    JUDGE_XLSX_RECALL_TOP_K,
)
from .prompts import build_section_distribution_prompt

logger = logging.getLogger(__name__)

SCORE_DIGITS = tuple(str(i) for i in range(10))

# 懒加载 vLLM（可能未安装）
_LLM_CLASS = None
_SAMPLING_PARAMS_CLASS = None


def _ensure_vllm():
    global _LLM_CLASS, _SAMPLING_PARAMS_CLASS
    if _LLM_CLASS is None:
        try:
            from vllm import LLM, SamplingParams
            _LLM_CLASS = LLM
            _SAMPLING_PARAMS_CLASS = SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM 未安装。使用 VLLMEngineScorer 需要在服务器环境安装 vLLM。\n"
                "pip install vllm"
            )
    return _LLM_CLASS, _SAMPLING_PARAMS_CLASS


def _get_digit_token_ids(tokenizer) -> Dict[str, int]:
    """计算 0~9 共 10 个数字在 tokenizer 中的 token id。

    注意: 某些 tokenizer 可能将"0"编码为 2 个 token（如"0"之前有空格前缀）。
    我们取最后一个 token 的 id，因为这是数字本身的 token。
    """
    digit_ids: Dict[str, int] = {}
    for d in range(10):
        s = str(d)
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            digit_ids[s] = ids[-1]  # 取最后一个 token（数字本身）
        else:
            logger.warning("数字 '%s' 编码为空，跳过", s)
    return digit_ids


def _detect_thinking_strategy(tokenizer, llm=None, lora_request=None) -> str:
    """探测如何让 judge 模型跳过 thinking，仅返回最终答案的 logprobs。

    背景：reasoning 模型（Qwen3 / Baichuan-M2 等）默认在 assistant 段输出
    ``<think>...</think>{answer}`` 格式。我们 SamplingParams(max_tokens=1) 只
    抓第一个 token，若不跳过 thinking 抓到的就是 ``<think>`` 这个 special token
    本身，digit logprobs 退化成 base 频率残渣（mass ≈ 0.000002，'0' 永远赢）。

    业界标准做法是触发 chat_template 内的 ``enable_thinking=False`` 分支，
    在 prompt 末尾自动注入空 ``<think>\\n\\n</think>\\n\\n`` 字符串——模型见到
    "思考已结束" pattern 后下一个 token 直接生成答案。

    三分支策略：
      1. ``"param"``：tokenizer 接受 ``enable_thinking=False`` 且渲染结果不同。
         模板已实现注入逻辑，直接传该 kwarg 即可。
         典型代表：Qwen3 / DeepSeek-R1-Distill。
      2. ``"inject"``：模板不支持 enable_thinking 参数，**且**模型实际会输出
         thinking。生产阶段在 Python 端手工追加 ``"<think>\\n\\n</think>\\n\\n"``
         模拟 Qwen3 模板的注入行为。
         典型代表：Baichuan-M2-32B（基于 Qwen2.5-32B 微调）。
      3. ``"none"``：模板不支持 enable_thinking 参数，**且**模型实际不输出
         thinking。什么都不做，避免污染 prompt——这是为纯 instruct 模型
         （如 Qwen2.5-7B-Instruct）准备的。光靠模板字符串扫描无法分辨它和
         Baichuan-M2 那种"模板里有 <think> 关键字但模型确实输出 thinking"的
         情况，必须靠 probe 推理实测。
         典型代表：Qwen2.5-Instruct 系列、Llama-3-Instruct 系列。

    参数：
      - ``tokenizer``：必填，用于 apply_chat_template。
      - ``llm`` + ``lora_request``：仅 inject/none 探测路径需要——如果模板不支持
        enable_thinking 参数，会用 ``llm.generate(...)`` 跑一次 8-token probe 看
        模型实际输出。如果未传 llm（单元测试或纯字符串验证场景），则在
        "无法 probe"时回退到保守的 ``"inject"`` 分支（保留原 2 分支行为，宁愿
        承担轻微噪声也不丢 thinking 跳过能力）。
    """
    dummy = [{"role": "user", "content": "Reply with OK."}]
    default_prompt = tokenizer.apply_chat_template(
        dummy, tokenize=False, add_generation_prompt=True,
    )

    # ── 第 1 步：模板是否支持 enable_thinking=False 参数 ──
    try:
        nothink_prompt = tokenizer.apply_chat_template(
            dummy, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        if nothink_prompt != default_prompt:
            return "param"
    except TypeError:
        # 严格模板：未知 kwarg 直接抛 TypeError → 视为不支持。
        pass

    # ── 第 2 步：probe 推理看模型实际是否输出 thinking ──
    if llm is None:
        # 无 llm 实例（单元测试或字符串验证）→ 保守回退到 inject。
        # 这保留了"修复了 thinking bug 但有微小噪声"的折中——不会因为
        # 探测不全就完全丢掉 thinking 跳过能力。
        logger.warning(
            "_detect_thinking_strategy 无 llm 实例可 probe，回退到 'inject'。"
            "纯 instruct 模型在此分支下 prompt 末尾会被多塞 <think>\\n\\n</think>\\n\\n，"
            "实际影响极小但有微弱噪声。生产路径 VLLMEngineScorer 总是会传 llm。"
        )
        return "inject"

    try:
        # 直接复用 vllm 的 SamplingParams——已经在模块顶层 _ensure_vllm() 加载。
        # 8 token 足够看出 thinking 苗头：模型若要 think 第一个 token 就是
        # <think>；最长输出 OK 也只占 1~2 个 token，不会越界。
        _, SamplingParams = _ensure_vllm()
        sp_probe = SamplingParams(max_tokens=8, temperature=0.0)
        gen_kwargs = {}
        if lora_request is not None:
            gen_kwargs["lora_request"] = lora_request
        outputs = llm.generate([default_prompt], sp_probe, **gen_kwargs)
        first_tokens = outputs[0].outputs[0].token_ids[:8]
        decoded = tokenizer.decode(first_tokens)
        logger.info(
            "_detect_thinking_strategy probe 结果: token_ids=%s decoded=%r",
            list(first_tokens), decoded,
        )

        # 只查 <think> 这一个最常见的标记——保守路线。
        # 如果未来出现别的 marker（<reasoning> 等）漏检，到时再扩展。
        if "<think>" in decoded:
            return "inject"
        return "none"
    except Exception as exc:  # noqa: BLE001
        # probe 推理失败——最保守的兜底是回退到字符串扫描：
        # 模板含 <think> → inject，否则 → none。
        # 这复原了"光看模板"的旧逻辑，至少不会让 daemon 启动失败。
        logger.warning(
            "_detect_thinking_strategy probe 推理失败 (%s)，回退到模板字符串扫描", exc,
        )
        template = tokenizer.chat_template or ""
        return "inject" if "<think>" in template else "none"


class VLLMEngineScorer:
    """使用 vLLM Python engine 直接获取 0~9 数字 token 的精确 logprobs。

    通过 SamplingParams(logprob_token_ids=...) 精确指定 0~9 共 10 个数字的
    token ID，vLLM 直接按 ID 查表返回 logprobs，100% 保证覆盖，无需 top-K 排序。

    初始化参数:
        model_path: 模型路径
        lora_path: 可选 LoRA adapter 路径
        max_model_len: 最大输入长度
        gpu_memory_utilization: GPU 显存利用率
        tensor_parallel_size: TP 并行度（默认 1，单卡）
        trust_remote_code: 是否信任远程代码
        knowledge_base: 可选的 JudgeKnowledgeBase 实例
        knowledge_top_k: 知识检索 top-K
    """

    def __init__(
        self,
        model_path: str,
        lora_path: str = "",
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        knowledge_base=None,
        knowledge_top_k: int = 4,
        sharpness: float = 1.0,
    ) -> None:
        LLM, SamplingParams = _ensure_vllm()

        self.model_path = str(model_path)
        self.lora_path = str(lora_path or "")
        self.knowledge_base = knowledge_base
        self.knowledge_top_k = int(knowledge_top_k)
        # sharpness：digit logprobs 锐化温度倒数。1.0 = 标准 softmax 概率期望（数学
        # 严格定义）；>1.0 = 在 reward 计算时把 logprobs 乘以这个系数后再 softmax，
        # 让分布更尖锐 → 期望值更靠近 argmax → reward 信号宽度变大。
        # 现实瓶颈：reasoning 模型在 logprobs 期望机制下，即使 argmax='8' 的题，
        # P(8) 也很少超过 0.5，期望被相邻 [6, 7, 9] 分摊回 ~7.3 → 整批样本压在
        # [6, 8) 区间，reward stdev 仅 ~0.1。sharpness=2.0 是经验甜点：让 P 分布
        # 更接近 one-hot，期望接近 argmax 但保留连续性，预期 reward stdev 翻倍。
        self.sharpness = float(sharpness) if sharpness else 1.0
        if self.sharpness <= 0:
            raise ValueError(f"sharpness 必须 > 0, 实际 {self.sharpness}")

        # RAG 启动期一次性警告标记：每个 scorer 实例首次发现 kb 不可用时
        # 在 logger 打一条 WARNING（含 disable_reasons），后续评分不再重复刷屏，
        # 但 evidence.log 每条仍按 status=disabled 记录，便于事后清点。
        self._rag_disable_warned = False

        logger.info("初始化 VLLMEngineScorer: model=%s lora=%s", model_path, self.lora_path or "(none)")
        if self.lora_path and not Path(self.lora_path).exists():
            raise FileNotFoundError(f"judge LoRA adapter path not found: {self.lora_path}")

        llm_kwargs = {
            "model": self.model_path,
            "trust_remote_code": trust_remote_code,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
        }
        if self.lora_path:
            llm_kwargs.update({"enable_lora": True, "max_loras": 1})
        self._llm = LLM(**llm_kwargs)
        self._lora_request = None
        if self.lora_path:
            try:
                from vllm.lora.request import LoRARequest
            except ImportError as exc:
                raise ImportError("当前 vLLM 不支持 LoRARequest，无法加载 judge LoRA adapter") from exc
            self._lora_request = LoRARequest("judge_lora", 1, self.lora_path)

        # 获取 tokenizer 并预计算 digit token IDs
        self._tokenizer = self._llm.get_tokenizer()
        self._digit_token_ids = _get_digit_token_ids(self._tokenizer)
        # {digit_str: token_id}，如 {"0": 29871, "1": 29889, ...}
        logger.info("Digit token IDs: %s", self._digit_token_ids)

        # 通过 logprob_token_ids 精确指定 10 个数字 token，
        # vLLM 直接按 ID 查表返回，无需 top-K 排序，100% 保证覆盖。
        #
        # 关键参数 logprobs=10 而不是 1：
        # vllm 0.20.2 sampler 内部确实按 logprob_token_ids 全量算了 11 列
        # （sampled + 10 个 digit），但下游 collator 把 LogprobsTensors 转回
        # dict[int, Logprob] 时按 max_num_logprobs+1 切列。max_num_logprobs 来自
        # SamplingParams.logprobs，若设 1 则输出只剩 2 列 → 只能拿到 sampled
        # token + logprob_token_ids[0]（即 id=15 = '0'），其余 9 个 digit 在
        # collator 那步被截断丢弃。设为 10 才能让 11 列完整透传到 Python 端。
        # 此时 sampler.py:122-125 会先算一次无用的 top-10 gather，但被
        # sampler.py:129 的 override 立即覆盖，仅多几毫秒，judge 评分零感知。
        digit_token_id_list = list(self._digit_token_ids.values())
        self._sampling_params = SamplingParams(
            max_tokens=1,
            logprobs=10,                      # 行宽必须 ≥ digit 数量，否则被 collator 截断
            logprob_token_ids=digit_token_id_list,
            temperature=1.0,                  # 不影响 logprobs 值，仅采样概率
        )

        # 启动时探测一次"如何让 judge 模型跳过 thinking"——见 _detect_thinking_strategy
        # 的详细 docstring。结果在 score_section() 里每次评分时使用，零运行时开销。
        # 传入 llm + lora_request 让 detector 在模板不支持 enable_thinking=False 时
        # 跑一次 8-token probe inference，区分"模型实际会输出 thinking"（→ inject）
        # 和"模型不输出 thinking"（→ none，避免污染 prompt）。
        self._thinking_strategy = _detect_thinking_strategy(
            self._tokenizer, llm=self._llm, lora_request=self._lora_request,
        )
        _strategy_explain = {
            "param": "通过 chat_template enable_thinking=False 自动注入空 think 块",
            "inject": "chat_template 不支持 enable_thinking 参数 + probe 实测模型会输出"
                      " thinking → 在 Python 端手工注入 '<think>\\n\\n</think>\\n\\n'",
            "none":  "chat_template 不支持 enable_thinking 参数 + probe 实测模型不输出"
                     " thinking → 不做任何处理，保持 prompt 干净",
        }.get(self._thinking_strategy, "未知策略")
        logger.info(
            "judge thinking 策略 = %s（%s）", self._thinking_strategy, _strategy_explain,
        )

    @property
    def is_ready(self) -> bool:
        return self._llm is not None

    def score_section(
        self,
        *,
        question_type: str,
        prompt_text: str,
        candidate_text: str,
    ) -> Dict[str, Any]:
        """对单个题型片段进行评分。"""

        pdf_advice_evidence: list = []
        outline_reference: list = []
        # rag_status：写进 evidence.log 区分 ok / disabled / error 三种 n=0 来源。
        # disabled = knowledge_base 缺组件（无 client / 无 index），属于配置问题；
        # error    = 检索过程抛异常（API 鉴权 / 网络 / 模型）；
        # ok       = 跑完了但召回/筛选后仍是 0 条，属于真实"无证据"。
        rag_status = "ok"
        rag_detail = ""
        # 持有完整 context 引用以便落 evidence.log 时拿漏斗 6 个计数（pdf_recall /
        # pdf_advice_raw / pdf_advice_final / xlsx_recall / xlsx_candidates /
        # xlsx_outline_final）和 from_cache 标记。disabled / error 路径下保持 None。
        rag_context = None
        if self.knowledge_base is None:
            rag_status = "disabled"
            rag_detail = "knowledge_base 实例未创建（PDF/XLSX index 都没加载到）"
        elif not self.knowledge_base.is_ready():
            rag_status = "disabled"
            rag_detail = "; ".join(self.knowledge_base.disable_reasons) or (
                "knowledge_base.is_ready() == False（embedding client 或 index 缺失）"
            )
        else:
            try:
                context = self.knowledge_base.build_judge_context(
                    question_type=question_type,
                    prompt_text=prompt_text,
                    candidate_text=candidate_text,
                    pdf_top_k=max(self.knowledge_top_k, JUDGE_PDF_RECALL_TOP_K),
                    max_pdf_advice=JUDGE_MAX_PDF_ADVICE,
                    xlsx_top_k=JUDGE_XLSX_RECALL_TOP_K,
                    max_outline_items=JUDGE_MAX_OUTLINE_ITEMS,
                )
                pdf_advice_evidence = context.pdf_advice_evidence
                outline_reference = context.outline_reference
                rag_context = context
            except Exception as exc:
                rag_status = "error"
                rag_detail = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "[RAG] retrieval error for %s: %s", question_type, rag_detail,
                )

        # 启动期一次性 WARNING（含 disable_reasons）。RAG 整路 n=0 但不静默：
        # 训练第一条样本就打这条，让用户立刻发现问题。
        if rag_status == "disabled" and not self._rag_disable_warned:
            self._rag_disable_warned = True
            logger.warning(
                "[RAG] 当前 judge 路径 RAG 不可用，evidence 将持续 n=0。原因: %s",
                rag_detail or "(未知)",
            )

        # 把 RAG 检索结果写入 evidence.log（默认关，--log-evidence 启用）。
        # 一行 = 一条记录，含 status / 漏斗 6 个 stage 计数 / 内容 head，
        # 配合 reward.log 的 section_scores 关联看。env 关闭时 sink 是 None，零开销。
        # 漏斗格式（2026-05 加）：
        #   pdf: recall=N → rerank_raw=M → final=K
        #   xlsx: recall=N → candidates=C → final=K
        # 一眼能区分"召回 0 / LLM 全否 / 元数据过滤吃光"三种空 evidence 的来源。
        try:
            from ..log_setup import log_evidence as _log_evidence
            cand_head = candidate_text.replace("\n", " ")[:120]
            advice_lines = "\n".join(f"    - {a}" for a in pdf_advice_evidence) or "    (none)"
            outline_lines = "\n".join(f"    - {o}" for o in outline_reference) or "    (none)"
            detail_line = f"  detail: {rag_detail}\n" if rag_detail else ""
            if rag_context is not None:
                # ok 路径才有漏斗计数。cache 命中时所有 stage 都没跑过，标记 from_cache=1
                # 让读者知道这条不能用来诊断召回。
                cache_tag = " from_cache=1" if rag_context.from_cache else ""
                funnel_line = (
                    f"  pdf: recall={rag_context.pdf_recall} → "
                    f"rerank_raw={rag_context.pdf_advice_raw} → "
                    f"final={rag_context.pdf_advice_final}\n"
                    f"  xlsx: recall={rag_context.xlsx_recall} → "
                    f"candidates={rag_context.xlsx_candidates} → "
                    f"final={rag_context.xlsx_outline_final}{cache_tag}\n"
                )
            else:
                # disabled / error 路径没跑 build_judge_context，所有计数无意义。
                funnel_line = ""
            _log_evidence(
                f"{question_type} status={rag_status} | candidate_head={cand_head!r}\n"
                f"{detail_line}"
                f"{funnel_line}"
                f"  pdf_advice (n={len(pdf_advice_evidence)}):\n{advice_lines}\n"
                f"  outline (n={len(outline_reference)}):\n{outline_lines}"
            )
        except Exception:
            pass

        prompt = build_section_distribution_prompt(
            question_type=question_type,
            prompt_text=prompt_text,
            candidate_text=candidate_text,
            pdf_advice_evidence=pdf_advice_evidence,
            outline_reference=outline_reference,
        )

        # 套 chat template + 跳过 thinking（三分支策略）：
        # 1. apply_chat_template 把"用户问题"包成模型期望的对话格式（必须步骤）。
        # 2. 跳过 thinking 让 max_tokens=1 抓到的第一个 token 是判分数字而非 <think>。
        # 跳过 thinking 的三条路径见 _detect_thinking_strategy()：
        #   - "param"：模板支持参数 → 直接传 enable_thinking=False
        #   - "inject"：模板不支持 + 模型确实输出 thinking → 手工追加空 think 字符串
        #   - "none"：模板不支持 + 模型不输出 thinking → 不做任何处理（避免污染纯 instruct 模型的 prompt）
        if self._thinking_strategy == "param":
            chat_prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        elif self._thinking_strategy == "inject":
            chat_prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            # 手工模拟 Qwen3 enable_thinking=False 模板的注入行为——告诉模型
            # "思考已经结束"，下一个 token 直接生成最终答案。
            chat_prompt += "<think>\n\n</think>\n\n"
        else:  # "none"
            # 纯 instruct 模型——模板不支持 enable_thinking 参数 + probe 实测模型
            # 不输出 thinking。直接走标准 chat template，不注入任何额外字符串。
            chat_prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        # ── 执行 vLLM 推理 ──
        generate_kwargs = {}
        if self._lora_request is not None:
            generate_kwargs["lora_request"] = self._lora_request
        outputs = self._llm.generate([chat_prompt], self._sampling_params, **generate_kwargs)
        output = outputs[0]

        # ── 从输出中提取 digit logprobs ──
        # logprob_token_ids 精确指定了 10 个数字 token ID，
        # vLLM 返回的 logprobs[0] 的 keys 就是这些 ID（加上采样 token）
        digit_logprobs_raw: Dict[str, float] = {}
        token_logprobs = None

        try:
            logprobs_data = output.outputs[0].logprobs
            if not logprobs_data or len(logprobs_data) == 0:
                raise RuntimeError("vLLM 输出 logprobs 为空")

            token_logprobs = logprobs_data[0]
            # token_logprobs is a dict: {token_id: Logprob(token, logprob, rank)}
            for digit_str, tid in self._digit_token_ids.items():
                if tid in token_logprobs:
                    digit_logprobs_raw[digit_str] = float(token_logprobs[tid].logprob)
        except (IndexError, AttributeError, TypeError) as exc:
            raise RuntimeError(f"无法从 vLLM 输出中提取 logprobs: {exc}") from exc

        if not digit_logprobs_raw:
            raise RuntimeError(
                "vLLM 输出中未找到任何数字 token logprobs。\n"
                f"digit_token_ids={self._digit_token_ids}\n"
                f"返回的 keys={list(token_logprobs.keys()) if token_logprobs else 'N/A'}"
            )

        # ── 归一化计算期望分数 ──
        # 应用 sharpness（温度锐化）：把 logprobs 乘以 sharpness 后再做 softmax
        # 归一化。等价于把 softmax 温度 T = 1/sharpness，sharpness > 1 让分布更
        # 尖锐，期望值更靠近 argmax，reward 信号宽度更大。详见 __init__ 的 docstring。
        #
        # 注意 raw_mass：仍按"未锐化的 exp(logprob)"计算，这是 digit 在原始模型
        # 概率分布中占的真实质量（用于诊断 mass≈0 的 logprobs 异常）。锐化只在
        # 期望分数计算时介入，不改变 raw_mass 的诊断含义。
        exp_values_raw = {d: math.exp(lp) for d, lp in digit_logprobs_raw.items()}
        raw_mass = float(sum(exp_values_raw.values()))
        if raw_mass <= 0:
            raise ValueError(f"digit probability mass must be positive, got {raw_mass}")

        # 锐化后的概率分布（用于期望计算）
        if self.sharpness == 1.0:
            sharpened_exp = exp_values_raw
        else:
            sharpened_exp = {d: math.exp(lp * self.sharpness) for d, lp in digit_logprobs_raw.items()}
        sharpened_mass = float(sum(sharpened_exp.values()))
        if sharpened_mass <= 0:
            raise ValueError(f"sharpened mass must be positive, got {sharpened_mass}")

        digit_probabilities = {
            digit: float(sharpened_exp[digit] / sharpened_mass)
            for digit in sorted(sharpened_exp.keys(), key=int)
        }
        mean_score = float(sum(int(digit) * prob for digit, prob in digit_probabilities.items()))
        normalized_mean_score = float(mean_score / 9.0)
        normalized_mean_reward = float((mean_score - 4.5) / 4.5)

        # logprob_token_ids 保证 100% 覆盖，但仍记录 found/missing 供排查
        missing = [d for d in SCORE_DIGITS if d not in digit_logprobs_raw]
        found = len(SCORE_DIGITS) - len(missing)

        # 抓 max_tokens=1 实际采样到的 token——这是验证 skip-thinking 是否生效的
        # 唯一可观测信号。修复成功后这里应该是 '0'~'9' 的某一个；若仍是 '<think>'
        # 或其他非数字 token，说明 _thinking_strategy 探测错了或注入失败。
        sampled_token_id = -1
        sampled_token_text = ""
        try:
            tok_ids = output.outputs[0].token_ids
            if tok_ids:
                sampled_token_id = int(tok_ids[0])
                sampled_token_text = self._tokenizer.decode([sampled_token_id])
        except (IndexError, AttributeError, TypeError):
            pass  # decode 失败不影响判分，留空字符串

        result = {
            "score": max(0.0, min(1.0, normalized_mean_score)),
            "question_type": question_type,
            "digit_logprobs": digit_logprobs_raw,
            "digit_probabilities": digit_probabilities,
            "raw_digit_mass": raw_mass,
            "mean_score": mean_score,
            "normalized_mean_score": normalized_mean_score,
            "normalized_mean_reward": normalized_mean_reward,
            "found_digits": found,
            "total_digits": len(SCORE_DIGITS),
            "missing_digits": missing,
            "sampled_token_id": sampled_token_id,
            "sampled_token_text": sampled_token_text,
            "thinking_strategy": self._thinking_strategy,
            "sharpness": self.sharpness,
            "scoring_method": "vllm_pyengine_digit_logprobs",
            "summary": f"vLLM engine: mean={mean_score:.2f} ({found}/{len(SCORE_DIGITS)} digits)",
            "model": self.model_path,
            "lora": self.lora_path,
        }

        return result


__all__ = ["VLLMEngineScorer", "VLLMEngineClient"]


# ── vLLM 引擎远程客户端（多进程 VERL 训练用） ──────────────────────

_VLLM_ENGINE_AUTH = b"vllm_engine_aiedu_2026"
_VLLM_ENGINE_HOST_ENV = "AIEDU_JUDGE_VLLM_ENGINE_HOST"
_VLLM_ENGINE_PORT_ENV = "AIEDU_JUDGE_VLLM_ENGINE_PORT"


class VLLMEngineClient:
    """vLLM engine 远程客户端——通过本地 TCP 连接 daemon 获取评分。

    daemon 由 train_grpo() 自动在 GPU 0 上启动，VERL 各训练 rank
    通过此 client 连接 daemon，不走 HTTP，不走 JSON。
    通信使用 Python multiprocessing.connection（pickle over TCP）。

    连接地址自动从环境变量读取（由 train_grpo() 设置）:
        AIEDU_JUDGE_VLLM_ENGINE_HOST=127.0.0.1
        AIEDU_JUDGE_VLLM_ENGINE_PORT=28765

    注意（Ray 多进程环境）:
        Ray Worker 进程由 raylet fork，不一定能继承父进程 os.environ 的最新值。
        train_grpo() 在启动 daemon 后、启动 VERL 之前设置这些环境变量，
        Ray 在 ray.init() 时通常会把当前 os.environ 快照传给所有 Worker，
        但如果 Ray cluster 是复用的（ray 没有重新 init），则需要重启 ray。
    """

    def __init__(self, host: str | None = None, port: int | None = None):
        self.host = host or os.environ.get(_VLLM_ENGINE_HOST_ENV, "127.0.0.1")
        raw_port = port
        if raw_port is None:
            try:
                raw_port = int(os.environ.get(_VLLM_ENGINE_PORT_ENV, "0"))
            except (ValueError, TypeError):
                raw_port = 0
        if raw_port <= 0:
            raise ValueError(
                f"VLLMEngineClient: 端口未配置。\n"
                f"  环境变量 {_VLLM_ENGINE_PORT_ENV}={os.environ.get(_VLLM_ENGINE_PORT_ENV, '(未设置)')}\n"
                f"  Ray Worker 进程可能未继承父进程的环境变量。\n"
                f"  解决方法：确保在 ray.init() 之前设置好 {_VLLM_ENGINE_PORT_ENV}，\n"
                f"  或传入 VLLMEngineClient(port=xxx)"
            )
        self.port = raw_port
        logger.info("VLLMEngineClient 连接: %s:%d", self.host, self.port)

    def score_section(
        self,
        *,
        question_type: str,
        prompt_text: str,
        candidate_text: str,
    ) -> Dict[str, Any]:
        """发送评分请求到 daemon，返回评分结果。

        含短暂重试（最多 3 次，间隔 2s），应对 daemon 短暂繁忙情况。
        对于真正的连接超时（[Errno 110]）会在第一次就快速失败，不做无效重试。
        """
        from multiprocessing.connection import Client as MClient
        import socket as _socket

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                conn = MClient((self.host, self.port), authkey=_VLLM_ENGINE_AUTH)
                try:
                    conn.send({
                        "question_type": question_type,
                        "prompt_text": prompt_text,
                        "candidate_text": candidate_text,
                    })
                    result = conn.recv()
                    if isinstance(result, dict) and "error" in result:
                        exc_type = result.get("exception_type", "RuntimeError")
                        raise RuntimeError(
                            f"[{exc_type}] vLLM daemon 评分失败: {result['error']}"
                        )
                    return result
                except EOFError:
                    raise RuntimeError("vLLM daemon 连接断开（daemon 可能已崩溃）")
                finally:
                    conn.close()
            except (ConnectionRefusedError, OSError) as exc:
                # [Errno 110] Connection timed out / [Errno 111] Connection refused
                # 这两种错误说明 daemon 不可达，重试无意义
                err_no = getattr(exc, "errno", None)
                if err_no in (110, 111):  # ETIMEDOUT, ECONNREFUSED
                    raise RuntimeError(
                        f"VLLMEngineClient 无法连接 daemon ({self.host}:{self.port}): {exc}\n"
                        f"  errno={err_no}. 可能原因:\n"
                        f"  1. daemon 进程因 stdout pipe 满阻塞（已修复，请确认 grpo.py 已更新）\n"
                        f"  2. Ray Worker 未继承 {_VLLM_ENGINE_PORT_ENV} 环境变量\n"
                        f"     → 当前进程 {_VLLM_ENGINE_PORT_ENV}={os.environ.get(_VLLM_ENGINE_PORT_ENV, '(未设置)')}\n"
                        f"  3. CUDA_VISIBLE_DEVICES 配置导致 daemon 使用了不同的 GPU"
                    ) from exc
                last_exc = exc
                if attempt < 2:
                    import time as _time
                    logger.warning(
                        "VLLMEngineClient 连接重试 %d/3: %s", attempt + 1, exc
                    )
                    _time.sleep(2)
                    continue
                raise RuntimeError(
                    f"VLLMEngineClient 连接失败（重试 3 次）: {exc}"
                ) from exc
        raise RuntimeError(f"VLLMEngineClient 连接失败: {last_exc}") from last_exc

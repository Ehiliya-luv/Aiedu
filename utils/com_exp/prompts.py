# -*- coding: utf-8 -*-
"""LLM Judge pairwise comparison prompts for medical tasks.

每个 prompt 函数签名: (raw_case, output_a, output_b) -> str
返回的 prompt 要求 Judge 以 JSON {"winner": "A"/"B"} 回答。
"""

from typing import Callable, Dict

# 已知任务名 → 对应 prompt 函数的注册表
PROMPT_REGISTRY: Dict[str, Callable[[str, str, str], str]] = {}

STRICT_OUTPUT_REQUIREMENT = """【输出要求】
你必须只输出且只能输出下面两种 JSON 中的任意一个，不能包含任何解释、前后缀、代码块、Markdown、空行或其他字符：
{"winner": "A"}
{"winner": "B"}
除了这两种精确字符串之外，任何输出都视为无效。"""


def _register(task_name: str):
    """装饰器：将 prompt 函数注册到 PROMPT_REGISTRY。"""
    def decorator(fn: Callable[[str, str, str], str]):
        PROMPT_REGISTRY[task_name] = fn
        return fn
    return decorator


# ================= SOAP 病历标准化 =================

@_register("SOAP")
@_register("病历标准化")
def soap_judge_prompt(raw_case: str, output_a: str, output_b: str) -> str:
    """SOAP 病历标准化 Pairwise Prompt。"""
    return f"""【医疗病历标准化评估】
你是一名资深临床医师，请评估以下两份 SOAP 格式病历的质量。

【原始病历】
{raw_case}

【模型 A 生成的 SOAP 病历】
{output_a}

【模型 B 生成的 SOAP 病历】
{output_b}

【评估维度】
1. 结构完整性：是否包含 S(主观)、O(客观)、A(评估)、P(计划) 四部分，且分段清晰
2. 术语规范性：医学术语使用是否准确、符合临床书写规范
3. 逻辑连贯性：主诉→现病史→评估→计划的推理链条是否合理
4. 关键信息覆盖：是否遗漏原始病历中的重要诊断、检查或治疗信息

{STRICT_OUTPUT_REQUIREMENT}
"""


# ================= 考题生成 =================

@_register("题目生成")
@_register("考题生成")
def question_judge_prompt(raw_case: str, output_a: str, output_b: str) -> str:
    """考题生成 Pairwise Prompt。"""
    return f"""【医疗考题生成质量评估】
你是一名医学教育专家，请评估以下两份基于同一病历生成的考题质量。

【原始病历】
{raw_case}

【模型 A 生成的考题】
{output_a}

【模型 B 生成的考题】
{output_b}

【评估维度】
1. 题干清晰度：问题表述是否明确、无歧义，符合医学考试命题规范
2. 答案准确性：正确答案是否有充分病历依据，解析是否专业
3. 干扰项合理性：错误选项是否为临床常见误区，具有鉴别价值
4. 临床相关性：考题是否紧扣病历核心诊疗要点，避免偏题
5. 推理深度：是否考察临床思维层次，而非简单记忆复述

{STRICT_OUTPUT_REQUIREMENT}
"""


# ================= 临床思维提取 =================

@_register("临床思维")
@_register("临床思维提取")
def clinical_thinking_judge_prompt(raw_case: str, output_a: str, output_b: str) -> str:
    """临床思维提取 Pairwise Prompt。"""
    return f"""【临床思维提取质量评估】
你是一名资深临床带教老师，请比较以下两份基于同一病历生成的临床思维要点，选择质量更高的一份。

【原始病历】
{raw_case}

【模型 A 生成的临床思维要点】
{output_a}

【模型 B 生成的临床思维要点】
{output_b}

【评估维度】
1. 病史整合：是否准确提炼主诉、现病史时间线和关键鉴别信息
2. 检查解读：是否针对性分析体格检查、辅助检查的阳性和阴性意义
3. 诊断推理：是否给出合理诊断依据、鉴别诊断和排除逻辑
4. 治疗决策：是否体现个体化治疗、风险预判、动态评估和随访计划
5. 教学价值：是否结构清晰、依据充分，能帮助学员形成临床思维

{STRICT_OUTPUT_REQUIREMENT}
"""


# ================= 病历综合评分 =================

@_register("病历综合评分")
@_register("病历评分")
def scoring_judge_prompt(raw_case: str, output_a: str, output_b: str) -> str:
    """病历综合评分 Pairwise Prompt。"""
    return f"""【病历综合评分质量评估】
你是一名资深医疗质量评审专家，请比较以下两份基于同一病历生成的综合评价与评分，选择更专业、更可靠的一份。

【原始病历】
{raw_case}

【模型 A 生成的病历综合评分】
{output_a}

【模型 B 生成的病历综合评分】
{output_b}

【评估维度】
1. 维度覆盖：是否覆盖病例资料完整度、书写规范性、学习内容丰富度、典型性、稀缺性、诊疗思路完整性、MDT需求度
2. 依据充分：评价是否紧扣原始病历信息，避免空泛或臆测
3. 评分合理：1-10分评分是否与各维度评价一致，尺度使用是否稳定
4. 医学专业性：术语、诊疗逻辑、指南或规范引用是否准确可信
5. 可读性：结构是否清晰，便于后续人工复核和教学使用

{STRICT_OUTPUT_REQUIREMENT}
"""


# ================= Prompt 分发 =================

# 默认任务优先顺序（用于 --tasks auto）
DEFAULT_TASK_ORDER = ["病历标准化", "考题生成", "临床思维", "病历综合评分"]

# 任务名别名映射（统一到 PROMPT_REGISTRY 的 key）
TASK_ALIASES: Dict[str, str] = {
    "SOAP": "病历标准化",
    "题目生成": "考题生成",
    "临床思维提取": "临床思维",
    "病历评分": "病历综合评分",
}


def resolve_task_name(raw_name: str) -> str:
    """将任务名别名解析为标准名。"""
    return TASK_ALIASES.get(raw_name, raw_name)


def get_judge_prompt(task: str, raw_case: str, output_a: str, output_b: str) -> str:
    """根据任务名分发对应的 Judge Prompt。

    Args:
        task:     任务名（支持别名）
        raw_case: 原始病历文本
        output_a: 模型 A 的输出
        output_b: 模型 B 的输出

    Returns:
        完整的 Judge prompt 字符串

    Raises:
        NotImplementedError: 任务名未注册
    """
    resolved = resolve_task_name(task)
    fn = PROMPT_REGISTRY.get(resolved)
    if fn is None:
        raise NotImplementedError(
            f"任务 '{task}' (解析为 '{resolved}') 的 Judge Prompt 未实现。"
            f"已注册任务：{list(PROMPT_REGISTRY.keys())}"
        )
    return fn(raw_case, output_a, output_b)

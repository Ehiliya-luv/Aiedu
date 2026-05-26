# -*- coding: utf-8 -*-
"""强化学习训练数据加载模块。

核心目标：
1. 严格按生产 JSONL 结构读取样本。
2. 保留 RL 训练所需的完整字段：prompt/input_context/model_output/expert_revision。
3. 为 GRPO 训练直接构造 datasets.Dataset。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset

logger = logging.getLogger(__name__)

# 全局变量：存储数据集中 model_output 的最大长度，供长度惩罚参考
_GLOBAL_MAX_MODEL_OUTPUT_LEN: int = 0


def _set_global_max_model_output_len(length: int) -> None:
    """设置全局最大 model_output 长度。"""
    global _GLOBAL_MAX_MODEL_OUTPUT_LEN
    _GLOBAL_MAX_MODEL_OUTPUT_LEN = length


def get_global_max_model_output_len() -> int:
    """获取全局最大 model_output 长度（用于无参考时的长度惩罚）。

    Returns:
        数据集中 model_output 的最大长度，未加载数据时返回 0
    """
    return _GLOBAL_MAX_MODEL_OUTPUT_LEN


# 与 generate_output.py 保持一致的题目生成模板
PROMPT_TASK_2_QUESTION = """
病历：{{#1745974940845.text#}}
任务：
请根据病历生成一整套客观题，必须完整包含 A1、A2、A3、A4、B1 五种题型。
输出内容只允许包含题干、选项、答案、解析这四类题目信息。
不要输出出题思路、命题说明、知识点总结、前言、后记、教学价值、提示语或任何题目之外的描述。
输出的 markdown 仅使用加粗 ** 或普通文本，勿使用表格、代码块或其他格式。

必须严格生成以下五类题型，示例仅供参考不允许直接使用：

一、A1型题（单句型最佳选择题）每道试题由1个题干和5个供选择的备选答案组成。题干以叙述式单句出现，备选答案中只有1个是最佳选择，称为正确答案，其余4个均为干扰答案。
试题结构：由一个简短的题干和五个备选答案组成，其中只有一个是正确答案。
考察点：主要考察考生对基础知识点的掌握程度，要求考生能够迅速准确地识别正确选项。
示例：
小儿尿路感染最常见的致病菌是
A. 变形杆菌
B. 大肠杆菌
C. 绿脓杆菌
D. 肠链球菌
E. 结核杆菌
答案：B
试题解析：
正确答案说明：小儿尿路感染最常见致病菌为大肠杆菌，来源于肠道菌群，经尿道上行感染，占绝大多数病例。
错误选项辨析：A 变形杆菌多见于复杂性或反复感染；C 绿脓杆菌常见于医院感染或免疫低下者；D 肠链球菌致病率较低；E 结核杆菌引起泌尿系结核，属于特殊感染类型。

二、A2型题（病例摘要型最佳选择题）试题结构是由1个简要病历作为题干、5个供选择的备选答案组成，备选答案中只有1个是最佳选择。
试题结构：以一个简要病历作为题干，配以五个备选答案，同样只有一个正确答案。
考察点：侧重于考察考生将理论知识应用于实际临床情境的能力，要求考生具备一定的临床经验和分析能力。
示例：
患者男,30 岁。近 4 年来经常间发四肢关节疼痛,近来感乏力、纳差、心悸、气促。肝在肋下
2.5 cm 触及,轻触痛。查血红蛋白 97g/L,尿蛋白(+)；双下肢轻度浮肿。最可能的原因是
A. 肝硬化
B. 急性肾炎
C. 主动脉瓣狭窄致左心衰
D. 二尖瓣狭窄致右心衰
E. 营养不良
答案：D
试题解析：
正确答案说明：二尖瓣狭窄可导致肺循环淤血→右心负荷增加→右心衰竭，表现为肝大、下肢水肿、乏力、气促，符合题干表现。
错误选项辨析：A 肝硬化虽有肝大和水肿，但不能解释心悸气促为主的表现；B 急性肾炎多急性起病且以血尿、高血压为主；C 主动脉瓣狭窄主要致左心衰，不以肝大水肿为主；E 营养不良虽可有水肿，但无心脏体征。

三、 A3型题（病例组型最佳选择题）试题结构是开始叙述一个以患者为中心的临床情景，然后提出2个～3个相关问题，每个问题均与开始的临床情景有关，但测试要点不同，且问题之间相互独立。
试题结构：开始叙述一个以患者为中心的临床情景，然后提出两至三个相关问题，每个问题均与前面的临床情景有关，但测试要点不同且问题之间相互独立。
考察点：要求考生对临床情境有深入的理解，并能针对不同的问题给出恰当的回答。
示例：
(1~2 题共用题干)患者女,24 岁。尿频、尿急、尿痛 3 个月。多种抗菌素治疗不见好转,尿常规
可见多个红、白细胞,最近患者症状加重,伴有尿失禁出现。
1. 此患者最可能的临床诊断是
A. 急性膀胱炎
B. 慢性膀胱炎……
C. 腺性膀胱炎
D. 间质性膀胱炎
E. 泌尿系结核
答案：C
试题解析：
正确答案说明：腺性膀胱炎表现为慢性尿路刺激症状，抗菌治疗无效，伴尿失禁，符合本例慢性反复病程。
错误选项辨析：A 急性膀胱炎为急性发作，抗菌治疗有效；B 慢性膀胱炎一般仍对抗菌药有一定反应；D 间质性膀胱炎以膀胱痛为主；E 泌尿系结核常有结核病史及无菌脓尿。
2. 此患者尿失禁属于
A. 精神性尿失禁
B. 压力性尿失禁
C. 急迫性尿失禁
D. 充盈性尿失禁
E. 真性尿失禁
答案: C
试题解析：
正确答案说明：急迫性尿失禁是由于膀胱逼尿肌不稳定引起，常伴尿急、尿频，符合本例。
错误选项辨析：A 精神性尿失禁多与心理因素有关；B 压力性尿失禁发生于咳嗽或用力时；D 充盈性尿失禁多见于尿潴留；E 真性尿失禁为持续漏尿，多见于括约肌功能丧失。

四、 A4型题（病例串型最佳选择题）开始叙述一个以单一病人或家庭为中心的临床情景，然后提出3个～6个相关问题。当病情逐渐展开时，可以逐步增加新的信息。
试题结构：开始叙述一个以单一病人或家庭为中心的临床情景，然后提出三至六个相关问题。
考察点：不仅考察了考生对单个知识点的掌握情况，还要求考生能够从整体上把握病情发展过程和治疗方案的选择。
示例：
(1~5 题共用题干)患者男,63 岁。确诊慢性阻塞性肺病近 10 年,因呼吸困难一直需要家人护理和照顾起居。今晨起⼤便时突然⽓急显著加重,伴胸痛,送来急诊。
1. 采集病史时应特别注意询问
A. 胸痛部位、性质和伴随症状
B. 冠⼼病、⼼绞痛病史
C. 吸烟史
D. 近期胸部 X 线检查情况
E. 近期服药史如⽀⽓管舒张剂、抗⽣素
答案：A
试题解析：
正确答案说明：急性加重伴胸痛，需首先明确胸痛性质及伴随症状以判断是否气胸、肺栓塞等急症。
错误选项辨析：B 冠心病史重要但非首要；C 吸烟史为慢性因素；D 既往影像不如当前症状重要；E 用药史次要。
2. 体检重点应是
A. 肺下界位置及肺下界移动度
B. 肺部啰⾳
C. 病理性⽀⽓管呼吸⾳
D. 胸部叩诊⾳及呼吸⾳的双侧⽐较
E. 颈动脉充盈
答案：D
试题解析：
正确答案说明：气胸或肺部病变时双侧呼吸音及叩诊差异最关键。
错误选项辨析：A 肺下界改变不敏感；B 啰音不一定存在；C 病理呼吸音不典型；E 颈动脉与本病关联小。
3. 确诊最有价值的辅助检查是
A. B 型超声显像
B. ⼼电图
C. X 线透视或摄⽚
D. MRI
E. 核素肺扫描
答案：C
试题解析：
正确答案说明：胸部X线是诊断气胸最常用且最直接的方法。
错误选项辨析：A B超不适用于气胸；B 心电图用于心脏疾病；D MRI不作为首选；E 核素扫描用于肺栓塞。
4. [假设信息] 经检查确诊肺⽓肿并发左侧⾃发性⽓胸,其治疗拟选择胸腔插管⽔封瓶引流,尽快使肺复张。主要达到的⽬的是
A. 维护已经严重受损的肺功能,防⽌呼吸衰竭
B. 缩短住院时间
C. 防⽌形成慢性⽓胸
D. 防⽌胸腔继发感染
E. 防⽌循环系统受扰和引起并发症
答案：A
试题解析：
正确答案说明：胸腔引流目的是促进肺复张，恢复通气功能，防止呼吸衰竭。
错误选项辨析：B 缩短住院时间不是主要目的；C 慢性气胸为并发情况；D 感染预防次要；E 循环影响非主要目标。
5. [假设信息] 已有检查仍不能证明⽓胸。尚需考虑可能的诊断是
A. 肺炎
B. ⼼绞痛
C. ARDS
D. 肺栓塞
E. 急性肺⽔肿
答案: D
试题解析：
正确答案说明：COPD患者突发气急胸痛需高度警惕肺栓塞。
错误选项辨析：A 肺炎起病较缓；B 心绞痛不以呼吸困难为主；C ARDS需明确诱因；E 急性肺水肿多有心源性表现。

五、 B1型题（标准配伍题）试题开始是5个备选答案，备选答案后提出至少2道相关试题，要求考生每一道试题选择一个与其关系密切的答案。在一组试题中，每个备选答案可以选一次，也可选数次，也可一次都不选。
试题结构：首先给出五个备选答案，随后提出两道或以上的试题，要求考生为每一道试题选择一个与其关系最密切的答案。每个备选答案可以选用一次或多次，但也可以一次不选。
考察点：旨在检验考生的综合应用能力和逻辑推理能力。
示例：
(1~3 题共⽤备选答案)
A. 卡介苗
B. 百⽩破三联疫苗
C. 脊髓灰质炎疫苗
D. ⼄型脑炎疫苗
E. 麻疹疫苗
1. ⼩⼉出⽣时应接种
2. 2 个⽉⼩⼉应接种
3. 3~6 个⽉⼩⼉应接种
答案：1. A 2. C 3. B
试题解析：
正确答案说明：我国免疫程序规定：出生时接种卡介苗；2月龄接种脊髓灰质炎疫苗；3~6月龄接种百白破疫苗。
错误选项辨析：D 乙脑疫苗一般8月龄后接种；E 麻疹疫苗多在8月龄接种，均不符合题干时间。
""".strip()


def _build_question_prompt(input_context: str) -> str:
    return PROMPT_TASK_2_QUESTION.replace("{{#1745974940845.text#}}", input_context)


@dataclass
class RLTrainingSample:
    """单条 RL 样本结构。"""

    prompt: str
    input_context: str
    model_output: str
    expert_revision: str
    metadata: Dict[str, Any]


def _safe_load_json_line(line: str, line_no: int, file_path: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        logger.warning("跳过非法 JSON（%s:%s）: %s", file_path, line_no, exc)
        return None
    if not isinstance(obj, dict):
        logger.warning("跳过非对象 JSON（%s:%s）", file_path, line_no)
        return None
    return obj


def _pick_text(obj: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_record(obj: Dict[str, Any], index: int, strict: bool) -> Optional[RLTrainingSample]:
    # 按生产结构优先读取
    input_context = _pick_text(obj, ["input_context", "context", "input", "text"])
    prompt = _build_question_prompt(input_context) if input_context else ""
    model_output = _pick_text(obj, ["model_output", "response", "answer", "output"])
    expert_revision = _pick_text(obj, ["expert_revision", "target", "gold", "reference"])

    missing = [
        name
        for name, value in (
            ("input_context", input_context),
            ("model_output", model_output),
            ("expert_revision", expert_revision),
        )
        if not value
    ]

    if missing:
        msg = f"样本#{index} 缺失关键字段: {', '.join(missing)}"
        if strict:
            raise ValueError(msg)
        logger.warning("%s，已跳过", msg)
        return None

    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}

    return RLTrainingSample(
        prompt=prompt,
        input_context=input_context,
        model_output=model_output,
        expert_revision=expert_revision,
        metadata=metadata,
    )


def load_rl_samples(
    path: str,
    max_items: Optional[int] = None,
    strict: bool = True,
) -> List[RLTrainingSample]:
    """从 JSONL 加载 RL 样本。

    strict=True: 任一不合法样本会抛错（生产推荐）。
    strict=False: 跳过坏样本。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    if not path.endswith(".jsonl"):
        raise ValueError(f"当前仅支持 .jsonl 文件: {path}")

    samples: List[RLTrainingSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            if max_items is not None and len(samples) >= max_items:
                break
            line = raw_line.strip()
            if not line:
                continue
            obj = _safe_load_json_line(line, line_no, path)
            if obj is None:
                if strict:
                    raise ValueError(f"JSON 解析失败，行号: {line_no}")
                continue
            sample = _normalize_record(obj, index=line_no, strict=strict)
            if sample is not None:
                samples.append(sample)

    if not samples:
        raise ValueError(f"未从 {path} 加载到有效样本")

    # 计算所有样本中 model_output 的最大长度，供后续长度惩罚参考
    max_model_output_len = max(len(s.model_output) for s in samples) if samples else 0
    _set_global_max_model_output_len(max_model_output_len)

    logger.info("已加载 RL 样本 %d 条，最大 model_output 长度=%d: %s", len(samples), max_model_output_len, path)
    return samples


def _contains_extra_section_types(text: str) -> bool:
    """检测是否出现超出 A1/A2/A3/A4/B1 的题型标记（如 B2/C1/D4）。"""
    from .reward import SECTION_ORDER

    if not isinstance(text, str) or not text.strip():
        return False

    allowed = set(SECTION_ORDER)
    marker_pattern = re.compile(r"(?<![A-Z0-9])([A-Z])\s*([1-9])(?!\d)\s*(?:型)?题")
    for m in marker_pattern.finditer(text):
        sec = f"{m.group(1)}{m.group(2)}"
        if sec not in allowed:
            return True
    return False


def build_grpo_dataset(
    path: str,
    max_items: Optional[int] = None,
    strict: bool = True,
    tokenizer=None,
    thinking_mode: str = "off",
) -> Dataset:
    """构造可直接用于 TRL GRPOTrainer 的 Dataset."""
    samples = load_rl_samples(path=path, max_items=max_items, strict=strict)

    def _is_clean_expert_revision_for_grpo(text: str) -> bool:
        from .reward import SECTION_ORDER, extract_question_sections, is_benign_objective_prefix

        if not isinstance(text, str) or not text.strip():
            return False

        sections, spans = extract_question_sections(text)
        found_order = [sec for sec in SECTION_ORDER if sections.get(sec, "").strip()]
        if found_order != SECTION_ORDER:
            return False

        if _contains_extra_section_types(text):
            return False

        if spans:
            first_start = spans[0][1]
            last_end = spans[-1][2]
            prefix = text[:first_start]
            suffix = text[last_end:]
            if prefix.strip() and not is_benign_objective_prefix(prefix):
                return False
            if suffix.strip():
                return False

        return True

    references: List[str] = []
    clean_reference_count = 0
    for s in samples:
        ref = s.expert_revision if _is_clean_expert_revision_for_grpo(s.expert_revision) else ""
        if ref:
            clean_reference_count += 1
        references.append(ref)

    if tokenizer is not None and callable(getattr(tokenizer, "apply_chat_template", None)):
        try:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": s.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    thinking_mode=thinking_mode,
                )
                for s in samples
            ]
            prompt_log_mode = f"预格式化文本(thinking_mode={thinking_mode})"
        except TypeError:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": s.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for s in samples
            ]
            prompt_log_mode = "预格式化文本(thinking_mode未生效)"
    else:
        # 回退到对话格式，让下游按原有逻辑处理
        prompts = [
            [{"role": "user", "content": s.prompt}]
            for s in samples
        ]
        prompt_log_mode = "对话格式"

    payload = {
        "prompt": prompts,
        "input_context": [s.input_context for s in samples],
        "references": references,
        "reference": references,
        "raw_references": [s.expert_revision for s in samples],
        "metadata": [s.metadata for s in samples],
    }
    logger.info(
        "GRPO 数据集构建完成: total=%d, clean_references=%d (prompt=%s)",
        len(samples),
        clean_reference_count,
        prompt_log_mode,
    )
    return Dataset.from_dict(payload)


__all__ = [
    "RLTrainingSample",
    "load_rl_samples",
    "build_grpo_dataset",
    "get_global_max_model_output_len",
]

# -*- coding: utf-8 -*-
"""Build DPO preference pairs from revision event records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Tuple


QUESTION_TYPES = ("A1", "A2", "A3", "A4", "B1")


QUESTION_TYPE_REQUIREMENTS: Dict[str, str] = {
    "A1": """A1型题（单句型最佳选择题）
每道试题由1个题干和5个供选择的备选答案组成。题干以叙述式单句出现，备选答案中只有1个是最佳选择，其余4个均为干扰答案。
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
""",
    "A2": """A2型题（病例摘要型最佳选择题）
试题结构是由1个简要病历作为题干、5个供选择的备选答案组成，备选答案中只有1个是最佳选择。
考察点：侧重考察考生将理论知识应用于实际临床情境的能力。
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
""",
    "A3": """A3型题（病例组型最佳选择题）
开始叙述一个以患者为中心的临床情景，然后提出2个至3个相关问题。每个问题均与开始的临床情景有关，但测试要点不同，且问题之间相互独立。
考察点：要求考生对临床情境有深入理解，并能针对不同问题给出恰当回答。
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
""",
    "A4": """A4型题（病例串型最佳选择题）
开始叙述一个以单一病人或家庭为中心的临床情景，然后提出3个至6个相关问题。当病情逐渐展开时，可以逐步增加新的信息。
考察点：要求考生从整体上把握病情发展过程和治疗方案选择。
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
""",
    "B1": """B1型题（标准配伍题）
试题开始给出5个备选答案，备选答案后提出至少2道相关试题，要求考生为每一道试题选择一个与其关系最密切的答案。每个备选答案可以选一次、选数次，也可以一次都不选。
考察点：检验考生的综合应用能力和逻辑推理能力。
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
""",
}


@dataclass
class BuildDPOStats:
    total: int = 0
    written: int = 0
    skipped_missing_case: int = 0
    skipped_missing_parent: int = 0
    skipped_missing_current: int = 0
    skipped_no_content_change: int = 0


def safe_strip(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_text(value: Any) -> str:
    return safe_strip(value).replace("\r\n", "\n").replace("\r", "\n")


def collapse_blank_lines(text: Any, max_blank_lines: int = 1) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    lines = [line.rstrip() for line in normalized.split("\n")]
    result: List[str] = []
    blank_count = 0
    for line in lines:
        if line.strip():
            blank_count = 0
            result.append(line.strip())
            continue
        blank_count += 1
        if blank_count <= max_blank_lines:
            result.append("")
    return "\n".join(result).strip()


def normalize_options(value: Any) -> str:
    text = safe_strip(value)
    if not text:
        return ""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_numeric(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def json_load_maybe(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def extract_option_text(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return collapse_blank_lines(item, max_blank_lines=0)
    if isinstance(item, (int, float, Decimal)):
        return str(item)
    if isinstance(item, dict):
        for key in ("content", "text", "option", "value", "label_text"):
            text = safe_strip(item.get(key))
            if text:
                return collapse_blank_lines(text, max_blank_lines=0)
        return collapse_blank_lines(json.dumps(item, ensure_ascii=False), max_blank_lines=0)
    return collapse_blank_lines(str(item), max_blank_lines=0)


def parse_options(value: Any) -> List[Tuple[str, str]]:
    text = safe_strip(value)
    if not text:
        return []

    parsed = json_load_maybe(text)
    if isinstance(parsed, list):
        options: List[Tuple[str, str]] = []
        for index, item in enumerate(parsed):
            label = chr(ord("A") + index)
            if isinstance(item, dict):
                label = safe_strip(item.get("label")) or safe_strip(item.get("key")) or label
            content = extract_option_text(item)
            if content:
                options.append((label, content))
        if options:
            return options

    line_options: List[Tuple[str, str]] = []
    for raw_line in collapse_blank_lines(text, max_blank_lines=0).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([A-Z])[\.\uff0e\u3001:\s]+(.+)$", line)
        if match:
            line_options.append((match.group(1), match.group(2).strip()))
    return line_options


def format_options(value: Any) -> str:
    options = parse_options(value)
    if options:
        return "\n".join(f"{label}. {content}" for label, content in options)
    return collapse_blank_lines(safe_strip(value), max_blank_lines=0)


def format_answer(answer_key: Any) -> str:
    answer = collapse_blank_lines(safe_strip(answer_key), max_blank_lines=0)
    if not answer:
        return ""
    return f"答案：{answer}"


def format_explanation(explanation: Any) -> str:
    body = collapse_blank_lines(safe_strip(explanation), max_blank_lines=1)
    if not body:
        return ""
    return f"试题解析：\n{body}"


def render_question_text(question_type: str, question: Dict[str, Any]) -> str:
    parts: List[str] = [f"## {question_type}型题"]
    content = collapse_blank_lines(question.get("content") or "", max_blank_lines=1)
    options = format_options(question.get("options"))
    answer = format_answer(question.get("answer_key"))
    explanation = format_explanation(question.get("explanation"))

    if content:
        parts.append(content)
    if options:
        parts.append(options)
    if answer:
        parts.append(answer)
    if explanation:
        parts.append(explanation)

    return "\n".join(part for part in parts if part).strip()


def canonical_dpo_payload(question: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        normalize_text(question.get("content")),
        normalize_options(question.get("options")),
        normalize_text(question.get("answer_key")),
        normalize_text(question.get("explanation")),
    )


def has_dpo_effective_change(parent_question: Dict[str, Any], current_question: Dict[str, Any]) -> bool:
    return canonical_dpo_payload(parent_question) != canonical_dpo_payload(current_question)


def build_single_question_prompt(
    original_content: str,
    question_type: str,
) -> str:
    question_type = safe_strip(question_type)
    original_content = collapse_blank_lines(original_content, max_blank_lines=1)
    type_requirement = QUESTION_TYPE_REQUIREMENTS.get(question_type, f"{question_type}型题要求：保持题型结构规范。")
    case_text = original_content or "未提供原始病历。"
    return f"""你是一名医学考试命题专家。请根据题型要求和原始病历，生成一个高质量的{question_type}型题。

【任务目标】
生成一个符合{question_type}型题结构、严格基于原始病历事实、医学事实准确、选项清晰且答案唯一的单题。

【使用规则】
1. 题型要求用于约束题目结构、设问方式、选项形式和解析形式。
2. 原始病历是事实边界，不得编造病历中没有的关键诊疗信息。
3. 题目应围绕病历中的主要诊断、关键症状体征、检查结果、诊疗过程或相关医学知识点展开。
4. 如果题型要求和病历事实存在冲突，优先遵循原始病历事实和医学事实准确性。

【{question_type}型题要求】
{type_requirement}

【原始病历】
{case_text}

【输出要求】
1. 只输出生成后的单题，不要生成其他题型或整套试卷。
2. 保持题型为{question_type}型题。
3. 输出必须包含题干、选项、答案、解析四类题目信息。
4. 选项应互斥、清晰，正确答案唯一。
5. 解析应说明正确答案依据，并对关键错误选项进行辨析。
6. 不要输出出题思路、修改说明、知识点总结、免责声明或其他额外内容。

【生成题目】
""".strip()


def build_dpo_pair_from_revision_event(event: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    question_type = safe_strip(event.get("question_type"))
    case_context = event.get("case_context") if isinstance(event.get("case_context"), dict) else {}
    parent_question = event.get("previous_question") if isinstance(event.get("previous_question"), dict) else {}
    if not parent_question:
        parent_question = event.get("parent_question") if isinstance(event.get("parent_question"), dict) else {}
    current_question = event.get("current_question") if isinstance(event.get("current_question"), dict) else {}

    original_content = normalize_text(case_context.get("original_content"))
    if event.get("has_previous") is False:
        return None, "missing_parent"
    if not parent_question.get("content"):
        return None, "missing_parent"
    if not current_question.get("content"):
        return None, "missing_current"
    if not has_dpo_effective_change(parent_question, current_question):
        return None, "no_content_change"

    chosen = render_question_text(question_type, current_question)
    rejected = render_question_text(question_type, parent_question)
    prompt = build_single_question_prompt(original_content, question_type)

    identity = event.get("identity") if isinstance(event.get("identity"), dict) else {}
    quality_flags = event.get("quality_flags") if isinstance(event.get("quality_flags"), dict) else {}
    record = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "meta": {
            "sample_id": event.get("sample_id"),
            "question_type": question_type,
            "question_item_id": identity.get("question_item_id") or event.get("sample_id"),
            "parent_item_id": identity.get("parent_item_id") or (parent_question.get("item_id") if parent_question else None),
            "fk_file_id": identity.get("fk_file_id"),
            "raw_medical_id": identity.get("raw_medical_id"),
            "sourceid": event.get("sourceid"),
            "quality_flags": quality_flags,
        },
    }
    return record, None


def build_dpo_pairs_from_events(events: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], BuildDPOStats]:
    stats = BuildDPOStats()
    pairs: List[Dict[str, Any]] = []
    for event in events:
        stats.total += 1
        pair, reason = build_dpo_pair_from_revision_event(event)
        if pair is None:
            if reason == "missing_case":
                stats.skipped_missing_case += 1
            elif reason == "missing_parent":
                stats.skipped_missing_parent += 1
            elif reason == "missing_current":
                stats.skipped_missing_current += 1
            elif reason == "no_content_change":
                stats.skipped_no_content_change += 1
            continue
        pairs.append(pair)
        stats.written += 1
    return pairs, stats

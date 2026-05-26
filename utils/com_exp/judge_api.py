# -*- coding: utf-8 -*-
"""Judge API 调用 + pairwise 评估逻辑。

复用 utils.judge.api.OpenAICompatibleClient 的退避/重试机制，
不重复实现指数退避逻辑。
"""

import concurrent.futures
import itertools
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.judge.api import (
    JudgeAPIError,
    JudgeAuthError,
    JudgeRateLimitError,
    OpenAICompatibleClient,
)
from utils.text import strip_think_content

from .prompts import get_judge_prompt

PAIRWISE_JUDGE_RESPONSE_FORMAT = {
    "type": "json_object",
}

PAIR_COMPLETE = "COMPLETE"
PAIR_RESUME = "RESUME"
PAIR_RESET = "RESET"

FATAL_JUDGE_ERRORS = (JudgeAuthError, JudgeAPIError)


def _format_raw_response(content: str) -> str:
    if content == "":
        return "<EMPTY STRING>"
    return repr(content)


# ================= Judge 响应解析 =================

def parse_judge_result(content: str) -> Optional[str]:
    """从 LLM 返回的内容中解析评判结果。

    Returns:
        "A", "B", 或 None（无法解析）
    """
    if not content:
        return None

    raw = strip_think_content(content).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Judge API] 无法解析结果，原始内容: {_format_raw_response(raw)}")
        return None

    if not isinstance(payload, dict) or set(payload.keys()) != {"winner"}:
        return None

    winner = payload.get("winner")
    if winner in ("A", "B"):
        return winner

    return None


# ================= Judge 调用 =================

def call_judge(
    client: OpenAICompatibleClient,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    max_tokens: int = 2048,
    max_retries: int = 10,
    response_format: Optional[dict] = PAIRWISE_JUDGE_RESPONSE_FORMAT,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """调用 Judge API 并解析结果。

    底层退避/429处理由 OpenAICompatibleClient._execute_with_retry 负责，
    此处只在响应无法解析时做有限重试。

    extra_body：透传给 OpenAI SDK 的非标准字段，主要用于 vLLM server 端
    通过 ``chat_template_kwargs.enable_thinking=False`` 关闭 reasoning 模型
    的 thinking 输出（Judge 只需要 winner 字段，不需要 think）。
    """
    for attempt in range(max_retries):
        try:
            content = client.chat_complete_text(
                model=model,
                system_prompt="",
                user_prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=extra_body,
            )

            content = strip_think_content(content)
            result = parse_judge_result(content)
            if result is not None:
                return result

            print(f"[Judge API] 第 {attempt+1} 次请求返回内容未包含预期结果，真实返回: {_format_raw_response(content)}")
        except FATAL_JUDGE_ERRORS:
            raise
        except JudgeRateLimitError as e:
            print(f"[Judge API] 速率限制重试耗尽: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            print(f"[Judge API] 第 {attempt+1} 次请求失败: {type(e).__name__}: {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    print("[Judge API] 多次重试后仍失败，返回 None")
    return None


# ================= 断点续跑 =================

def _checkpoint_prefix(task: str) -> str:
    """生成 checkpoint 文件前缀（任务名中的 / 替换为 _ 避免路径问题）。"""
    return task.replace("/", "_").replace("\\", "_")


def load_checkpoint(task: str, checkpoint_dir: str) -> Optional[pd.DataFrame]:
    """加载任务的胜负矩阵 checkpoint。"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"{_checkpoint_prefix(task)}_checkpoint.csv")
    if os.path.exists(checkpoint_file):
        try:
            return pd.read_csv(checkpoint_file, index_col=0)
        except Exception as e:
            print(f"[断点续跑] 加载 {task} 检查点失败: {e}，将重新开始")
    return None


def save_checkpoint(task: str, win_matrix: pd.DataFrame, checkpoint_dir: str) -> None:
    """保存任务的胜负矩阵 checkpoint。"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"{_checkpoint_prefix(task)}_checkpoint.csv")
    win_matrix.to_csv(checkpoint_file)


def load_case_progress(task: str, checkpoint_dir: str) -> Dict[str, List[str]]:
    """加载病例级完成进度。

    Returns:
        dict，key 为 "model_a|model_b"，value 为已完成的 case_id 列表。
    """
    progress_file = os.path.join(checkpoint_dir, f"{_checkpoint_prefix(task)}_progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_case_progress(task: str, checkpoint_dir: str, progress: Dict[str, List[str]]) -> None:
    """保存病例级完成进度。"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    progress_file = os.path.join(checkpoint_dir, f"{_checkpoint_prefix(task)}_progress.json")
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def is_pair_fully_evaluated(
    win_matrix: pd.DataFrame,
    model_a: str,
    model_b: str,
    num_cases: int,
    repeats: int,
) -> bool:
    """检查模型对是否已完整评估。"""
    if win_matrix.empty or model_a not in win_matrix.index or model_b not in win_matrix.columns:
        return False

    wins_a = win_matrix.loc[model_a, model_b]
    wins_b = win_matrix.loc[model_b, model_a]
    total = wins_a + wins_b
    required = num_cases * repeats

    if total > required:
        raise RuntimeError(
            f"[严重错误] {model_a} vs {model_b} 投票数({total}) > 需求数({required})，有重复计算！"
        )
    return total == required


def _pair_vote_total(win_matrix: pd.DataFrame, model_a: str, model_b: str) -> int:
    return int(win_matrix.loc[model_a, model_b]) + int(win_matrix.loc[model_b, model_a])


def _count_completed_votes(
    win_matrix: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    num_cases: int,
    repeats: int,
) -> int:
    """按胜负矩阵统计已有有效票数，用于断点续跑进度显示。"""
    required = num_cases * repeats
    completed = 0
    for model_a, model_b in pairs:
        if model_a in win_matrix.index and model_b in win_matrix.columns:
            completed += min(_pair_vote_total(win_matrix, model_a, model_b), required)
    return completed


def _is_case_level_progress(pair_progress: object) -> bool:
    return isinstance(pair_progress, dict)


def _case_vote_record(pair_progress: Dict[str, Dict[str, int]], case_id: str) -> Dict[str, int]:
    record = pair_progress.get(case_id)
    if not isinstance(record, dict):
        return {"wins_a": 0, "wins_b": 0}
    return {
        "wins_a": int(record.get("wins_a", 0)),
        "wins_b": int(record.get("wins_b", 0)),
    }


def _sum_pair_progress(
    pair_progress: Dict[str, Dict[str, int]],
    valid_cases: List[str],
    repeats: int,
) -> Tuple[int, int, bool]:
    """汇总病例级 progress，并校验单病例票数不超过 repeats。"""
    wins_a = 0
    wins_b = 0
    valid = True
    for cid in valid_cases:
        record = _case_vote_record(pair_progress, cid)
        case_wins_a = record["wins_a"]
        case_wins_b = record["wins_b"]
        if case_wins_a < 0 or case_wins_b < 0 or case_wins_a + case_wins_b > repeats:
            valid = False
        wins_a += case_wins_a
        wins_b += case_wins_b
    return wins_a, wins_b, valid


def validate_pair_state(
    win_matrix: pd.DataFrame,
    case_progress: Dict[str, object],
    pair_key: str,
    model_a: str,
    model_b: str,
    valid_cases: List[str],
    repeats: int,
) -> Tuple[str, str]:
    """判断当前 pair 应该跳过、精确续跑还是重跑。"""
    required = len(valid_cases) * repeats
    matrix_a = int(win_matrix.loc[model_a, model_b])
    matrix_b = int(win_matrix.loc[model_b, model_a])
    matrix_total = matrix_a + matrix_b

    if matrix_total > required:
        raise RuntimeError(
            f"[严重错误] {model_a} vs {model_b} 投票数({matrix_total}) > 需求数({required})，有重复计算！"
        )

    pair_progress = case_progress.get(pair_key)

    if pair_progress is None:
        if matrix_total == required:
            return PAIR_COMPLETE, "无 progress，但 win_matrix 已满"
        return PAIR_RESET, "无病例级 progress，无法安全定位缺口"

    if not _is_case_level_progress(pair_progress):
        if matrix_total == required:
            return PAIR_COMPLETE, "旧格式 progress 且 win_matrix 已满"
        return PAIR_RESET, "旧格式 progress 无病例票数，无法安全续跑"

    progress_a, progress_b, progress_valid = _sum_pair_progress(pair_progress, valid_cases, repeats)
    if not progress_valid:
        return PAIR_RESET, "病例级 progress 存在单病例票数异常"

    if progress_a != matrix_a or progress_b != matrix_b:
        return (
            PAIR_RESET,
            f"progress 汇总({progress_a}:{progress_b}) 与 win_matrix({matrix_a}:{matrix_b}) 不一致",
        )

    if matrix_total == required:
        return PAIR_COMPLETE, "病例级 progress 与 win_matrix 一致且已满"

    return PAIR_RESUME, "病例级 progress 与 win_matrix 一致，可精确续跑"


def build_case_plan(
    pair_progress: Dict[str, Dict[str, int]],
    valid_cases: List[str],
    repeats: int,
) -> List[Tuple[str, int]]:
    """根据病例级 progress 生成每个病例还需要补的投票数。"""
    plan = []
    for cid in valid_cases:
        record = _case_vote_record(pair_progress, cid)
        done_votes = record["wins_a"] + record["wins_b"]
        if done_votes < repeats:
            plan.append((cid, repeats - done_votes))
    return plan


# ================= Pairwise 评估 =================

def run_pairwise_evaluation(
    task_data: Dict,
    task: str,
    client: OpenAICompatibleClient,
    judge_model: str,
    repeats: int = 3,
    checkpoint_dir: Optional[str] = None,
    judge_workers: int = 4,
    judge_temperature: float = 0.0,
    judge_top_p: Optional[float] = None,
    judge_max_tokens: int = 2048,
    judge_response_format: Optional[dict] = PAIRWISE_JUDGE_RESPONSE_FORMAT,
    judge_extra_body: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """实施去偏置成对比较 + 胜负矩阵生成 + 断点续跑。

    核心逻辑：
    1. 对每对模型的每个病历，进行 repeats 次投票
    2. 每次投票随机决定哪个模型放 A/B 位置（反偏置）
    3. 每完成一个病历的所有 repeat 后保存 checkpoint
    4. 支持病例级断点续跑：已完成的病例跳过，只评估剩余病例
    """
    raw_cases = task_data["raw_cases"]
    model_outputs = task_data["model_outputs"]

    valid_cases = sorted(
        set.intersection(*[set(outputs.keys()) for outputs in model_outputs.values() if outputs])
    )
    if not valid_cases:
        print(f"[Judge 评价] 任务 {task} 无共同病历，跳过")
        return pd.DataFrame()

    print(f"[Judge 评价] 任务 '{task}' 使用 {len(valid_cases)} 份病历进行成对比较")

    models = [m for m in model_outputs if len(model_outputs[m]) > 0]
    pairs = list(itertools.combinations(models, 2))

    # 断点续跑：加载胜负矩阵 + 病例进度
    if checkpoint_dir:
        win_matrix = load_checkpoint(task, checkpoint_dir)
        case_progress = load_case_progress(task, checkpoint_dir)
    else:
        win_matrix = None
        case_progress = {}

    if win_matrix is None:
        win_matrix = pd.DataFrame(0, index=models, columns=models)
    else:
        for model in models:
            if model not in win_matrix.index:
                win_matrix.loc[model] = 0
            if model not in win_matrix.columns:
                win_matrix[model] = 0
        win_matrix = win_matrix.reindex(index=models, columns=models, fill_value=0)
        win_matrix = win_matrix.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    total_comparisons = len(pairs) * len(valid_cases) * repeats
    completed = _count_completed_votes(win_matrix, pairs, len(valid_cases), repeats)
    skipped_pairs = 0

    if completed > 0:
        print(f"[断点续跑] 已从 checkpoint 载入有效投票进度：{completed}/{total_comparisons}")

    for model_a, model_b in pairs:
        pair_key = f"{model_a}|{model_b}"
        required = len(valid_cases) * repeats

        pair_state, pair_reason = validate_pair_state(
            win_matrix=win_matrix,
            case_progress=case_progress,
            pair_key=pair_key,
            model_a=model_a,
            model_b=model_b,
            valid_cases=valid_cases,
            repeats=repeats,
        )

        if pair_state == PAIR_COMPLETE:
            wins_a = int(win_matrix.loc[model_a, model_b])
            wins_b = int(win_matrix.loc[model_b, model_a])
            print(
                f"[断点续跑] 跳过已完整对：{model_a} vs {model_b} "
                f"(投票数: {wins_a}:{wins_b} == {required}; {pair_reason})"
            )
            skipped_pairs += 1
            continue

        if pair_state == PAIR_RESET:
            reset_votes = _pair_vote_total(win_matrix, model_a, model_b)
            if reset_votes:
                completed -= min(reset_votes, required)
            print(
                f"[断点续跑] 重跑 {model_a} vs {model_b}: {pair_reason}。"
                f"清零该 pair 既有票数 {int(win_matrix.loc[model_a, model_b])}:"
                f"{int(win_matrix.loc[model_b, model_a])}"
            )
            win_matrix.loc[model_a, model_b] = 0
            win_matrix.loc[model_b, model_a] = 0
            case_progress[pair_key] = {}
            if checkpoint_dir:
                save_checkpoint(task, win_matrix, checkpoint_dir)
                save_case_progress(task, checkpoint_dir, case_progress)
        elif pair_state == PAIR_RESUME:
            print(f"[断点续跑] 精确续跑 {model_a} vs {model_b}: {pair_reason}")
        else:
            raise RuntimeError(f"未知 pair 状态: {pair_state}")

        pair_progress = case_progress.setdefault(pair_key, {})
        if not isinstance(pair_progress, dict):
            pair_progress = {}
            case_progress[pair_key] = pair_progress

        case_plan = build_case_plan(pair_progress, valid_cases, repeats)
        current_total = _pair_vote_total(win_matrix, model_a, model_b)
        planned_votes = sum(vote_count for _, vote_count in case_plan)
        if planned_votes == 0 and current_total < required:
            raise RuntimeError(
                f"{model_a} vs {model_b} 票数未满({current_total}/{required})，但未生成补票计划"
            )

        if planned_votes:
            print(
                f"[断点续跑] {model_a} vs {model_b}: "
                f"win_matrix 票数 {current_total}/{required}，本轮计划 {len(case_plan)} 个病例、{planned_votes} 票"
            )

        wins_a = int(win_matrix.loc[model_a, model_b])
        wins_b = int(win_matrix.loc[model_b, model_a])

        def _judge_one_case(args):
            """并行处理单个病历的一组投票。"""
            case_id, out_a, out_b, raw, vote_count = args
            local_a = 0
            local_b = 0
            for _ in range(vote_count):
                # 反偏置：随机交换 A/B 位置
                if random.random() > 0.5:
                    prompt = get_judge_prompt(task, raw, out_a, out_b)
                    pos_a, pos_b = model_a, model_b
                else:
                    prompt = get_judge_prompt(task, raw, out_b, out_a)
                    pos_a, pos_b = model_b, model_a

                result = call_judge(
                    client,
                    judge_model,
                    prompt,
                    temperature=judge_temperature,
                    top_p=judge_top_p,
                    max_tokens=judge_max_tokens,
                    response_format=judge_response_format,
                    extra_body=judge_extra_body,
                )
                if result is None:
                    continue
                if result == "A":
                    if pos_a == model_a:
                        local_a += 1
                    else:
                        local_b += 1
                elif result == "B":
                    if pos_b == model_a:
                        local_a += 1
                    else:
                        local_b += 1
            return case_id, vote_count, local_a, local_b

        case_args = [
            (cid, model_outputs[model_a][cid], model_outputs[model_b][cid], raw_cases[cid], vote_count)
            for cid, vote_count in case_plan
        ]

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=judge_workers)
        futures = [executor.submit(_judge_one_case, arg) for arg in case_args]
        try:
            for future in concurrent.futures.as_completed(futures):
                cid, vote_count, add_a, add_b = future.result()
                successful_votes = add_a + add_b
                wins_a += add_a
                wins_b += add_b
                win_matrix.loc[model_a, model_b] = wins_a
                win_matrix.loc[model_b, model_a] = wins_b

                record = _case_vote_record(pair_progress, cid)
                record["wins_a"] += add_a
                record["wins_b"] += add_b
                pair_progress[cid] = record

                if checkpoint_dir:
                    save_checkpoint(task, win_matrix, checkpoint_dir)
                    save_case_progress(task, checkpoint_dir, case_progress)

                completed += successful_votes
                if completed % 10 == 0:
                    print(f"[Judge 评价] 进度：{completed}/{total_comparisons}")
        except FATAL_JUDGE_ERRORS as exc:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            print(f"[致命错误] Judge API 不可恢复错误，正在中止评估进程: {type(exc).__name__}: {exc}")
            raise
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        print(f"[Judge 评价] 任务 '{task}' - {model_a} vs {model_b}: {wins_a}:{wins_b}")

    print(f"[Judge 评价] 任务 '{task}' 完成 - 新评估 {len(pairs) - skipped_pairs} 对，跳过 {skipped_pairs} 对")
    return win_matrix

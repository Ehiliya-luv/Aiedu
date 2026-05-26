# -*- coding: utf-8 -*-
"""日志输出流工具。

提供两件事：
  1. compute_log_dir(): 计算/复用本次运行的日志目录 ./tmp/grpo/{timestamp}/
     - 主进程首次调用时新建目录，并把绝对路径写入环境变量 AIEDU_LOG_DIR
     - 子进程（VERL subprocess、Ray worker、vLLM daemon）通过环境变量复用同一目录

  2. TeeStream: 同时把 write 转发到原 stream + 一个文件 handle，
     用来替换 sys.stdout / sys.stderr，让所有控制台输出自动落到 main_process.log。
     fileno / isatty 委派给原 stream，避免破坏 vLLM/ray/tqdm 等库对 fd 的探测。

仅与日志输出流有关，不参与训练逻辑。
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import sys
from pathlib import Path
from typing import Optional, TextIO

logger = logging.getLogger(__name__)

_LOG_DIR_ENV = "AIEDU_LOG_DIR"
_LOG_LEVEL_ENV = "AIEDU_LOG_LEVEL"
# 两个独立日志通道的 env 开关（默认全部 OFF，避免训练时 IO/磁盘/格式化开销）。
# 训练 step 粒度的指标由 tools/plot_reward.py 直接从 main_process.log 解析
# verl 的 step:N - critic/... 行——VERL 每 step 都会 emit 这一行，作为 sink
# 已经够稳；不再额外维护 reward_state.jsonl 通道。
_EVIDENCE_LOG_ENABLED_ENV = "AIEDU_LOG_EVIDENCE_ENABLED"  # 默认 "0"（关）
_COMPLETION_LOG_ENABLED_ENV = "AIEDU_LOG_COMPLETION_ENABLED"  # 默认 "0"（关）

_TMP_GRPO_ROOT = "tmp/grpo"


def compute_log_dir(*, project_root: Optional[Path] = None) -> Path:
    """返回本次运行的日志目录。

    若环境变量 AIEDU_LOG_DIR 已经存在（子进程场景），直接复用并 mkdir(exist_ok=True)。
    否则按当前时间生成 ./tmp/grpo/{YYYYmmdd_HHMMSS}/，并把绝对路径写回环境变量。
    """
    existing = os.environ.get(_LOG_DIR_ENV, "").strip()
    if existing:
        path = Path(existing)
        path.mkdir(parents=True, exist_ok=True)
        return path

    root = Path(project_root) if project_root is not None else Path.cwd()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = (root / _TMP_GRPO_ROOT / timestamp).resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ[_LOG_DIR_ENV] = str(path)
    return path


class TeeStream:
    """把 write 同步转发到原 stream + 一个文件 handle。

    - 仅在 write/flush 上做镜像；其它属性（fileno/isatty/encoding...）委派给原 stream。
    - 若任一侧 write/flush 抛错，吞掉异常并记录到 stderr 原 fd，避免破坏调用方流程。
    """

    def __init__(self, original: TextIO, mirror: TextIO) -> None:
        self._original = original
        self._mirror = mirror

    # 镜像写入
    def write(self, data) -> int:
        try:
            n = self._original.write(data)
        except Exception:
            n = 0
        try:
            self._mirror.write(data)
        except Exception:
            pass
        return n if isinstance(n, int) else 0

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass
        try:
            self._mirror.flush()
        except Exception:
            pass

    # 透传——尤其重要：fileno / isatty / encoding，让 vLLM/ray/tqdm 等库不会报错
    def __getattr__(self, name):
        return getattr(self._original, name)


def install_tee_streams(log_file_path: Path) -> TextIO:
    """把 sys.stdout / sys.stderr 替换为 TeeStream，同步落到 log_file_path。

    返回打开的文件句柄（调用方一般不需要用，进程退出时由 OS 收回；这里保留引用避免 GC）。
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    fp = log_file_path.open("a", buffering=1, encoding="utf-8")  # line buffering
    sys.stdout = TeeStream(sys.stdout, fp)  # type: ignore[assignment]
    sys.stderr = TeeStream(sys.stderr, fp)  # type: ignore[assignment]
    return fp


def setup_main_process_logging(
    *,
    log_level: str = "INFO",
    log_filename: str = "main_process.log",
    project_root: Optional[Path] = None,
) -> Path:
    """主进程入口：计算日志目录、Tee stdout/stderr、配置 root logger。

    返回日志目录路径。
    """
    log_dir = compute_log_dir(project_root=project_root)
    install_tee_streams(log_dir / log_filename)

    level = getattr(logging, str(log_level).upper(), logging.INFO)
    os.environ[_LOG_LEVEL_ENV] = logging.getLevelName(level)

    # 注意：sys.stdout 已是 Tee，挂一个 StreamHandler 即可让一条日志同时落终端 + 文件。
    # 不要再加 FileHandler，否则 main_process.log 会被双写。
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_dir


def setup_worker_process_logging(*, default_level: str = "INFO") -> None:
    """子进程（Ray worker / VERL subprocess）入口：给 root logger 装最小 handler。

    若 root logger 已有 handler 则不重复挂载（避免主进程也调用时双写）。
    log level 优先读 AIEDU_LOG_LEVEL 环境变量。
    """
    root = logging.getLogger()
    if root.handlers:
        return
    level_name = os.environ.get(_LOG_LEVEL_ENV, default_level).upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


# ── reward.log / evidence.log / completion.log 直写 sink ──
#
# 背景：VERL 通过 importlib 加载 verl_reward.py，reward 函数运行在
# Ray worker 进程里；worker 的 root logger 通常已被 VERL/Ray 装过 handler，
# 我们的 setup_worker_process_logging 早 return 后 logging 输出会被
# Ray 自身的 stdout 转发链接管，实测会被去重 / 过滤 / 写到 ray session
# 日志文件，导致 main_process.log 里看不到题型长度 / judge logprobs。
#
# 解决思路：直接打开 ./tmp/grpo/{ts}/{filename} 文件追加写。每个 worker 进程
# 一份文件句柄，POSIX append 写入对单行是原子的，不会撕裂行。
#
# 3 个通道按"开销 vs 价值"差异化默认开关：
#   reward.log     —— 必备，无开关（reward worker 写 judge 评分细节）
#   evidence.log   —— 默认关（每条 RAG 内容数 KB，开销大，仅调试时开）
#   completion.log —— 默认关（每条 candidate 完整原文 ~3KB，仅调试时开）
#
# 训练 step 粒度的结构化指标由 verl 内部 emit 到 main_process.log，
# tools/plot_reward.py 直接解析 "step:N - critic/... " 一行 dump。

_REWARD_LOG_FILENAME = "reward.log"
_EVIDENCE_LOG_FILENAME = "evidence.log"
_COMPLETION_LOG_FILENAME = "completion.log"

# 3 个独立 lazy-singleton 句柄
_reward_sink_fp: Optional[TextIO] = None
_evidence_sink_fp: Optional[TextIO] = None
_completion_sink_fp: Optional[TextIO] = None
_sink_lock = __import__("threading").Lock()


def _format_ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _env_flag_enabled(env_name: str, default: str = "0") -> bool:
    """读 env 布尔开关，'1' / 'true' / 'yes' / 'on' 视为开启。"""
    raw = os.environ.get(env_name, default).strip().lower()
    return raw in ("1", "true", "yes", "on", "y")


def _open_sink(filename: str, channel_name: str) -> Optional[TextIO]:
    """打开 AIEDU_LOG_DIR/<filename> 追加写句柄。失败返回 None。"""
    log_dir_env = os.environ.get(_LOG_DIR_ENV, "").strip()
    if not log_dir_env:
        return None
    try:
        log_dir = Path(log_dir_env)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / filename
        fp = path.open("a", buffering=1, encoding="utf-8")
        # 写一行 worker 启动标识，多进程并发时分辨来源
        try:
            fp.write(
                f"{_format_ts()} [worker:pid={os.getpid()}] {channel_name} sink 初始化 | "
                f"log_dir={log_dir}\n"
            )
            fp.flush()
        except Exception:
            pass
        return fp
    except Exception:
        return None


def get_reward_sink() -> Optional[TextIO]:
    """返回 reward.log 文件句柄（lazy 单例），永不被 env 开关 disable。"""
    global _reward_sink_fp
    if _reward_sink_fp is not None:
        return _reward_sink_fp
    with _sink_lock:
        if _reward_sink_fp is not None:
            return _reward_sink_fp
        _reward_sink_fp = _open_sink(_REWARD_LOG_FILENAME, "reward")
        return _reward_sink_fp


def get_evidence_sink() -> Optional[TextIO]:
    """返回 evidence.log 句柄（默认关，env 启用时才返回有效句柄）。"""
    global _evidence_sink_fp
    if not _env_flag_enabled(_EVIDENCE_LOG_ENABLED_ENV, default="0"):
        return None
    if _evidence_sink_fp is not None:
        return _evidence_sink_fp
    with _sink_lock:
        if _evidence_sink_fp is not None:
            return _evidence_sink_fp
        _evidence_sink_fp = _open_sink(_EVIDENCE_LOG_FILENAME, "evidence")
        return _evidence_sink_fp


def get_completion_sink() -> Optional[TextIO]:
    """返回 completion.log 句柄（默认关，env 启用时才返回有效句柄）。"""
    global _completion_sink_fp
    if not _env_flag_enabled(_COMPLETION_LOG_ENABLED_ENV, default="0"):
        return None
    if _completion_sink_fp is not None:
        return _completion_sink_fp
    with _sink_lock:
        if _completion_sink_fp is not None:
            return _completion_sink_fp
        _completion_sink_fp = _open_sink(_COMPLETION_LOG_FILENAME, "completion")
        return _completion_sink_fp


def log_reward(message: str) -> None:
    """把一行 reward 相关信息写入 reward.log（带时间戳前缀）。永不抛错。"""
    fp = get_reward_sink()
    if fp is None:
        return
    try:
        line = message if message.endswith("\n") else message + "\n"
        fp.write(f"{_format_ts()} {line}")
        fp.flush()
    except Exception:
        pass


def log_evidence(message: str) -> None:
    """把一条 RAG 检索证据写入 evidence.log（带时间戳）。env 默认关。"""
    fp = get_evidence_sink()
    if fp is None:
        return
    try:
        line = message if message.endswith("\n") else message + "\n"
        fp.write(f"{_format_ts()} {line}")
        fp.flush()
    except Exception:
        pass


def log_completion(message: str) -> None:
    """把 actor 完整 completion 写入 completion.log（带时间戳）。env 默认关。"""
    fp = get_completion_sink()
    if fp is None:
        return
    try:
        line = message if message.endswith("\n") else message + "\n"
        fp.write(f"{_format_ts()} {line}")
        fp.flush()
    except Exception:
        pass


__all__ = [
    "compute_log_dir",
    "install_tee_streams",
    "setup_main_process_logging",
    "setup_worker_process_logging",
    "TeeStream",
    "get_reward_sink",
    "get_evidence_sink",
    "get_completion_sink",
    "log_reward",
    "log_evidence",
    "log_completion",
]

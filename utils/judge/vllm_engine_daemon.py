# -*- coding: utf-8 -*-
"""vLLM engine daemon — 在独立进程中启动，通过本地 TCP 提供评分服务。

启动方式（由 train_grpo() 自动管理）:
    CUDA_VISIBLE_DEVICES=0 python -m utils.judge.vllm_engine_daemon \
        --model-path resources/model/Baichuan-M2-32B-0226

通信协议:
    使用 multiprocessing.connection.Listener/Client（标准库，pickle over TCP，127.0.0.1）。
    不是 HTTP！没有 JSON 开销。

并发模型:
    daemon 单线程串行处理请求（vLLM LLM.generate() 非线程安全，不可并发调用）。
    OS accept 队列 backlog=128 保证 VERL 所有并发 Worker 的 connect() 都能排队等待，
    不会因队列满被 DROP（errno=110）。
    Client 端 connect() 成功后阻塞 recv()，daemon 依次处理完每个请求。

与 VERL 多进程训练的配合:
    1. train_grpo() 启动此 daemon
    2. daemon 先 bind TCP 端口（backlog=128），再加载模型
    3. 模型加载完成后打印 VLLM_ENGINE_PORT=<port>（此时才真正就绪）
    4. train_grpo() 读取端口，启动 VERL
    5. VERL 各 rank 的 VLLMEngineClient 连接到此 daemon 获取评分
    6. 训练结束后 train_grpo() kill daemon

关键修复说明（errno=110 的根本原因）:
    原 daemon：先加载模型（5分钟），加载完才 bind 端口，才打印端口号。
    旧 grpo.py：_pick_free_port() 预先 bind 再释放，传给 daemon。
        → race condition：5分钟内端口被其他进程抢占，daemon bind 失败崩溃。
        → drain 日志是 DEBUG 级别，崩溃信息不可见，只看到 errno=110。

    修复后：
        - daemon 先 bind（backlog=128），再加载模型，加载完才打印端口
        - 父进程收到端口号时 Listener 已就绪且模型已加载
        - backlog=128 保证 n=8 并发 connect 不会因队列满被 DROP
        - drain 改为 WARNING 级别，崩溃信息可见
        - 添加监控线程，daemon 崩溃时立即打印 WARNING
"""

from __future__ import annotations

import argparse
import logging
import sys
from multiprocessing.connection import Listener

from .knowledge import load_knowledge_base
from .vllm_scorer import VLLMEngineScorer

_AUTH_KEY = b"vllm_engine_aiedu_2026"
logger = logging.getLogger("vllm_engine_daemon")


def main():
    parser = argparse.ArgumentParser(description="vLLM Engine Daemon")
    parser.add_argument("--model-path", required=True, help="vLLM 模型路径")
    parser.add_argument("--lora-path", default="", help="可选 vLLM judge LoRA adapter 路径")
    parser.add_argument("--port", type=int, default=0, help="监听端口（0=随机分配）")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--sharpness", type=float, default=2.0,
        help="digit logprobs 锐化温度倒数；>1 让概率分布更尖锐、reward 信号宽度更大",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[vllm-daemon] %(asctime)s %(message)s",
        stream=sys.stderr,
        force=True,
    )

    # ── 步骤 1：先 bind 端口（大 backlog 防止并发 connect 时 SYN 被 DROP）──
    #
    # 核心：backlog=128 让 OS 能接纳 128 个 pending SYN，即使 accept() 正在处理其他请求。
    # VERL rollout.n=8 → 最多 8 个 Worker 并发 connect，128 远超需求。
    # OS listen backlog 满 → SYN 被 DROP → client 等超时 → errno=110（本 bug 根本原因）。
    #
    # 注意：Listener 在 accept() 调用前就已经在监听，client 的 connect() 会立即完成，
    # 然后 client 阻塞在 recv() 等待 daemon 处理完发回结果。
    listener = Listener(("127.0.0.1", args.port or 0), authkey=_AUTH_KEY, backlog=128)
    actual_port = listener.address[1]
    logger.info("TCP listener 已绑定端口 %d（backlog=128），准备加载模型...", actual_port)

    # ── 步骤 2：加载模型（慢，可能 5-10 分钟）──
    logger.info("正在加载 vLLM engine: model=%s", args.model_path)
    try:
        knowledge_base = load_knowledge_base()
        scorer = VLLMEngineScorer(
            model_path=args.model_path,
            lora_path=args.lora_path,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            knowledge_base=knowledge_base,
            sharpness=args.sharpness,
        )
    except Exception as exc:
        logger.exception("vLLM engine 加载失败: %s", exc)
        listener.close()
        sys.exit(1)

    logger.info("vLLM engine 加载完成，端口: %d", actual_port)

    # ── 步骤 3：模型就绪后才打印端口（父进程等待此行）──
    # 重要：此时 Listener 已绑定 + 模型已加载，父进程收到端口后可立即使用
    print(f"VLLM_ENGINE_PORT={actual_port}", flush=True)

    # ── 步骤 4：单线程 accept 循环（vLLM 非线程安全，不可并发调用 generate）──
    # Client 端 connect 后阻塞 recv，daemon 串行处理每个请求。
    try:
        while True:
            conn = listener.accept()
            try:
                msg = conn.recv()
                # msg = {"question_type": ..., "prompt_text": ..., "candidate_text": ...}
                result = scorer.score_section(**msg)
                conn.send(result)
            except Exception as exc:
                logger.exception("评分请求失败")
                try:
                    conn.send({"error": str(exc), "exception_type": type(exc).__name__})
                except Exception:
                    pass
            finally:
                conn.close()
    except KeyboardInterrupt:
        logger.info("daemon 收到终止信号")
    finally:
        listener.close()


if __name__ == "__main__":
    main()

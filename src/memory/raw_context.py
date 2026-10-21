"""
方案 A：原始上下文拼接。把历史所有轮次直接拼进 prompt，超出窗口时做 sliding
window 截断——截断本身就是一个有说服力的失败模式证据，所以要记录被截掉的轮数。

⚠ v2 把默认窗口从 20 轮改成 10 轮（= 5 个来回）。v1 用 20 轮、而数据里每条对话
只有 6 个用户轮次（history 最长 11 轮），窗口**一次都没有截断**，raw_context
等于"完整历史"，于是三种方案比的根本不是记忆能力，而是"完整历史 vs 事实清单
vs 检索片段"这三种提示词写法。窗口必须小于对话长度，截断才会发生，"记忆"才
成为一个真问题。窗口在 configs/experiment.yaml 的
memory_params.raw_context.max_turns 里配，改了要在报告里写清楚。
"""
from .base import BaseMemory


class RawContextMemory(BaseMemory):
    name = "raw_context"
    supports_evidence = False

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.dropped_turns_log: list[int] = []  # 记录每次调用截断掉的轮数，供分析用
        self.retrieval_log: list[dict] = []

    def retrieve(self, history: list[dict], current_turn: dict) -> str:
        if len(history) > self.max_turns:
            dropped = len(history) - self.max_turns
            self.dropped_turns_log.append(dropped)
            history = history[-self.max_turns:]
        else:
            self.dropped_turns_log.append(0)

        # 与 vector_memory 的 retrieval_log 保持同样的结构，方便 metrics 统一
        # 处理；raw_context 没有"检索"语义，retrieved_spans 留空表示
        # evidence_use_rate 对本方案不适用（而不是 0）。
        self.retrieval_log.append({
            "retrieved_turn_ids": [t.get("turn_id") for t in history],
            "retrieved_spans": None,
        })

        lines = [f"{t['speaker']}: {t['text']}" for t in history]
        return "\n".join(lines)

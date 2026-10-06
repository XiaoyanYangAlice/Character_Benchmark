"""
方案 A：原始上下文拼接。把历史所有轮次直接拼进 prompt，超出窗口时做 sliding
window 截断——截断本身就是一个有说服力的失败模式证据，所以要记录被截掉的轮数。
"""
from .base import BaseMemory


class RawContextMemory(BaseMemory):
    name = "raw_context"

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.dropped_turns_log: list[int] = []  # 记录每次调用截断掉的轮数，供分析用

    def retrieve(self, history: list[dict], current_turn: dict) -> str:
        if len(history) > self.max_turns:
            dropped = len(history) - self.max_turns
            self.dropped_turns_log.append(dropped)
            history = history[-self.max_turns:]
        else:
            self.dropped_turns_log.append(0)

        lines = [f"{t['speaker']}: {t['text']}" for t in history]
        return "\n".join(lines)

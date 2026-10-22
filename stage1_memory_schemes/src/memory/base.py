"""
三种记忆方案共用的接口。所有方案都实现 retrieve()，方便 run_eval.py 统一调度、
公平对比。方案本身不负责调用生成模型——生成放在 src/generate/dialogue_runner.py，
这里只负责"给定历史，应该往 prompt 里塞什么上下文"。
"""
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    name: str = "base"
    # 这个方案是否有"检索证据"的语义。raw_context/no_memory 没有（它们不检索，
    # 只是把历史原样拼进去或者什么都不给），证据类指标对它们无定义，强行算出
    # 一个数会被误读成"它们的证据可追溯性是 0/1"。
    supports_evidence: bool = False

    @abstractmethod
    def retrieve(self, history: list[dict], current_turn: dict) -> str:
        """
        history: 到目前为止的对话轮次列表，每个元素形如
                 {"turn_id": int, "speaker": "user"|"character", "text": str}
        current_turn: 即将生成的这一轮的输入（一般是用户这一轮的发言）
        返回：拼进 prompt 的上下文字符串。
        """
        raise NotImplementedError

    def on_new_turn(self, turn: dict) -> None:
        """
        可选钩子：每生成完一轮之后调用，方便有状态的方案（向量库、结构化记忆）
        增量更新自己的存储。raw_context 方案不需要覆写这个。
        """
        pass

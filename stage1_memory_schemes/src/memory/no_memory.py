"""方案 0（对照基线）：完全不注入任何记忆上下文。

为什么必须有这个基线：system prompt 里本来就有完整角色卡，模型即使没有任何
对话历史也能"演得像"。v1 实验缺这个对照，导致"structured 比 raw 低 5.7 个
百分点"无法区分两种解释：

  (a) 结构化事实真的帮模型记住了该记住的东西；
  (b) 往 prompt 里再塞一遍角色设定，等于又提醒了一次"你是谁"，而 raw_context
      塞的是大段对话历史，反而稀释了角色卡的存在感。

no_memory 给出"只有角色卡、没有任何历史"的下界。配合 shuffled_facts
（见 shuffled_facts.py）就能把 (a) 和 (b) 分开：

  no_memory      : 角色卡
  shuffled_facts : 角色卡 + 别人的事实清单（形式相同、内容错误）
  structured     : 角色卡 + 自己的事实清单

如果 structured ≈ shuffled_facts，说明起作用的是"又出现了一段角色设定风格的
文本"，不是事实内容本身。
"""
from .base import BaseMemory


class NoMemory(BaseMemory):
    name = "no_memory"
    supports_evidence = False

    def __init__(self):
        # 与其它方案保持同样的接口，便于 metrics 统一处理（这里永远是空检索）。
        self.retrieval_log: list[dict] = []

    def retrieve(self, history: list[dict], current_turn: dict) -> str:
        self.retrieval_log.append({"speaker": current_turn.get("speaker"),
                                    "retrieved_fact_ids": [], "retrieved_turn_ids": []})
        return ""

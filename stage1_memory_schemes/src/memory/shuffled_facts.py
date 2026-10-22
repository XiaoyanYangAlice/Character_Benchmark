"""对照方案：结构化事实记忆，但注入的是**别人的**事实。

这是 v2 最重要的新增对照。v1 的结论是"structured_memory 让 judge 的出戏标记率
降低 5.7 个百分点"，但那个实验没法排除一个平凡的解释：system prompt 里已经有
完整角色卡，structured 又在上下文里追加了一段"[Known facts about X] ..."，等于
把人设又说了一遍；而 raw_context 追加的是大段对话历史，把角色卡的相对权重
稀释了。也就是说，降低可能来自"提示词里角色设定的占比更高"，而不是"记忆内容
正确"。

做法：对每条对话 d，取另一条对话 d' 的主角事实，按完全相同的格式注入。形式、
长度、位置、token 预算都与 structured_memory 相当，唯一的区别是**内容是错的**。

读法：
  - structured 明显优于 shuffled → 事实内容本身起作用，记忆方案有效。
  - structured ≈ shuffled 且两者都优于 no_memory → 起作用的是"又出现了一段
    人设风格的文本"，v1 的结论要改写成提示词工程效应，不是记忆效应。
  - shuffled 反而更差 → 错误事实会污染生成（这本身也是有意义的发现：说明
    记忆写错比没有记忆更糟）。

注意：注入错误事实理论上应该**提高** fact_consistency 判定出的矛盾率（因为
角色会说出与自己设定不符的内容），这正是预期的检验点之一。
"""
from .structured_memory import StructuredEventMemory


class ShuffledFactsMemory(StructuredEventMemory):
    """与 StructuredEventMemory 完全相同的检索逻辑，只是事实来自别的角色。

    事实记录会被改写成挂在"当前角色名"下，否则 retrieve() 按 speaker 查表会
    查不到任何东西、静默退化成空上下文——那就又变成 no_memory 了，对照失去
    意义（这正是 v1 踩过的那个坑的同一形态，所以这里显式处理并有回归测试）。
    """

    name = "shuffled_facts"

    def load_donor_facts(self, donor_records: list[dict], character_name: str) -> None:
        """donor_records: 另一条对话主角的 fact_labels。
        character_name: 当前对话主角的 canonical 标签（事实要挂到这个键下）。
        """
        relabelled = []
        for rec in donor_records:
            rec = dict(rec)
            rec["character"] = character_name
            rec["donor_story_id"] = rec.get("story_id")  # 留痕，便于事后核查
            relabelled.append(rec)
        self.load_character_facts_from_records(relabelled)

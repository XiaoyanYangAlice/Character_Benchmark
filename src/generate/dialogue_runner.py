"""
驱动"角色agent"模型跑完整段对话：每轮先调用 BaseMemory.retrieve() 拿上下文，
调用生成模型产出角色回复，再调用 memory.on_new_turn() 让有状态的方案（向量记忆、
结构化记忆）更新自己的存储。这是三种记忆方案和 src/eval/metrics.py 之间的
粘合层——metrics.py 的四个函数都需要一份跑出来的 `turns` 列表，这个模块就是
产出它的地方。

vLLM 部署与调用方式见 docs/SURF_GUIDE.md「跑法 1」。

⚠ 关键约定：`character_card["name"]` 必须和 ground truth 事实库
（`data/processed/character_features_en.jsonl` 里的 "character" 字段）用的
是同一个 canonical 标签，`structured_memory.py` 的 `retrieve()` 是按
`self.character_facts[speaker]` 查表的，标签对不上就检索不到任何事实，静默
退化成"结构化记忆方案什么都没用上"而不报错——这个坑不好排查，构建
dialogues.jsonl 时务必对齐（后续 build_dialogues.py 里处理，不在这个模块
自己纠正，因为这个模块不应该关心 ground truth 是怎么来的）。
"""
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI
else:
    OpenAI = Any


def call_model(client: OpenAI, model: str, system_prompt: str, context: str, user_turn: str,
               seed: int | None = None) -> str:
    # seed 只在多次重复实验（run_eval.py --seed）时传，让每次重复的采样可复现、
    # 彼此独立；不传则沿用 vLLM 默认的随机采样。
    extra = {"seed": seed} if seed is not None else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\n{user_turn}"},
        ],
        temperature=0.7,
        max_tokens=512,
        **extra,
    )
    return resp.choices[0].message.content


def build_system_prompt(character_card: dict) -> str:
    """把角色卡字段拼成 system prompt。字段名对齐 PIPPA 的
    bot_name/bot_description/bot_greeting（当前主数据源），cards_to_stories.py
    产出的 SillyTavern 角色卡字段名不完全一样，接入时需要在 build_dialogues.py
    里做一次字段映射，不在这里做兼容——这个函数只管"拿到统一格式后怎么拼
    prompt"，不管数据来源。

    ⚠ 这里优先用 `display_name`（没有就退回 `name`）来拼给模型看的文字。
    原因：PIPPA 不少角色卡整段用第一人称写（"I love to sing..."），完全没写出
    真名，ontology 抽取器在这种文本里找不到专有名词，只能把代词本身（"I"/
    "Me"）当 canonical 标签写进 character_features_en.jsonl。这对事实检索没
    影响（同一个故事内部自洽），但如果直接把这个代词塞进生成 prompt——
    "You are role-playing as I. Respond only in first person as I."——语法
    本身就是破的，可能拖累生成质量。`display_name` 就是 build_dialogues.py
    算出来的"读起来自然一点的名字"（代词标签就退回用 PIPPA 自带的
    bot_name，否则还是用 canonical 标签本身）。**只影响这里拼出来的文字**，
    不能拿 display_name 去替换 character_card["name"]——run_dialogue() 里
    角色轮次的 speaker 字段、structured_memory.py 的事实检索认的都是 name
    字段，换成 display_name 会导致查不到事实、静默退化（参见本文件顶部说明）。
    """
    name = character_card.get("display_name") or character_card.get("name", "the character")
    description = character_card.get("description", "")
    greeting = character_card.get("greeting", "")

    parts = [f"You are role-playing as {name}."]
    if description:
        parts.append(f"Character description:\n{description}")
    if greeting:
        parts.append(f"Example of how {name} opens a conversation:\n{greeting}")
    parts.append(
        f"Stay in character as {name} at all times. Respond only in first person as {name}. "
        f"Never mention that you are an AI, a language model, or that this is role-play. "
        f"Keep your reply to a few sentences, matching {name}'s established personality and voice."
    )
    return "\n\n".join(parts)


def run_dialogue(memory, client: OpenAI, model: str, character_card: dict,
                  scripted_user_turns: list[str], start_turn_id: int = 1,
                  seed: int | None = None) -> list[dict]:
    """跑完一整段对话，返回完整的 turns 列表（user 和角色轮次交替）。

    scripted_user_turns 是固定住的用户台词（比如从 PIPPA 原始对话里取出的真实
    人类发言）——三种记忆方案都用同一份 scripted_user_turns 跑，只有角色这边
    的回复会因为记忆方案不同而不同，这样对比才公平：差异只能来自"记忆方案给了
    模型什么上下文"，不会混进"用户这边聊的内容不一样"这个变量。

    memory: 三种方案之一的实例（调用前需要是"干净"的新实例，不要跨对话复用同
    一个 memory 对象，否则不同故事的记忆会串到一起）。
    """
    if not scripted_user_turns:
        raise ValueError("scripted_user_turns 为空，没有对话可跑")

    system_prompt = build_system_prompt(character_card)
    # 注意：这里故意用 "name"（canonical 标签），不是 "display_name"——
    # turns 里角色轮次的 speaker 字段要跟 character_features_en.jsonl 里的
    # 标签对上，structured_memory.py 才能查到事实，见文件头说明。
    character_name = character_card.get("name", "character")

    history: list[dict] = []
    turn_id = start_turn_id

    for user_text in scripted_user_turns:
        user_turn = {"turn_id": turn_id, "speaker": "user", "text": user_text}
        history.append(user_turn)
        memory.on_new_turn(user_turn)
        turn_id += 1

        # ⚠ 这里不能直接把 user_turn 传给 retrieve() 当 current_turn。
        # retrieve() 的两个实现对 current_turn 的用法不一样：
        #   - vector_memory.py 只看 current_turn["text"]，当检索 query 用，传
        #     user_turn 本身是对的（检索"跟用户刚说的话相关的记忆"）。
        #   - structured_memory.py 只看 current_turn["speaker"]，用来查
        #     self.character_facts[speaker]——如果真传 user_turn，speaker 永远
        #     是字符串 "user"，永远查不中任何角色的事实表，检索结果静默退化成
        #     空，不报错（这正是之前 evidence_traceability_rate 对 structured_
        #     memory 全跑出 0.000 的根因：不是真实结果，是传错了参数）。
        # 用一个"即将要生成的这一轮"的占位 turn 同时满足两边：speaker 换成
        # 角色名（structured_memory 需要），text 仍然是用户刚说的话（vector_
        # memory 需要）。turn_id 字段在两种实现里都不参与检索逻辑，随便填。
        pending_char_turn = {"turn_id": turn_id, "speaker": character_name, "text": user_text}
        context = memory.retrieve(history, pending_char_turn)
        reply_text = call_model(client, model, system_prompt, context, user_text, seed=seed)

        char_turn = {"turn_id": turn_id, "speaker": character_name, "text": reply_text.strip()}
        history.append(char_turn)
        memory.on_new_turn(char_turn)
        turn_id += 1

    return history

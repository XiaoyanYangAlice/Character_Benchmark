"""
主入口：跑完整 3(记忆方案) x N(模型) x M(对话) 实验矩阵，落盘 results/summary.csv
（每行一次生成的四个指标）和每条对话的完整生成记录（results/raw_outputs/*.json，
供人工抽查判断器打分对不对、写报告时挑例子）。

用法（模型和 judge 各自的 vLLM 服务已经在 SURF 上起好之后，见 docs/SURF_GUIDE.md）：
    python src/eval/run_eval.py --config configs/experiment.yaml \
        --model-port qwen2.5-7b=8000 --model-port llama3.1-8b=8001 --model-port judge=8002

调试时不想真的连 vLLM、只想看看矩阵和数据对不对，用 --dry-run（不调用任何模型，
每个指标都留空，只打印会跑哪些组合、检查 dialogues.jsonl / memory 实例化是否报错）。

⚠ 在 relationship_trajectory 弱标签打分脚本做完之前（TODO，见 docs/DATASET.md），
dialogues.jsonl 里这个字段全是空列表 []，relationship_state_accuracy() 对每条
对话都会抛 ValueError——这是预期行为，不是这个脚本的 bug。run_one() 已经把这个
异常单独 catch 掉、在 summary.csv 里留空，不影响另外三个指标先跑出来。弱标签
脚本做完、dialogues.jsonl 重新生成之后，直接重跑这个文件就行，不需要改这里的
代码。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

# 这个文件按文档里写的方式直接跑（`python src/eval/run_eval.py ...`），不是用
# `python -m src.eval.run_eval` 跑——两者的区别是：直接跑脚本时，Python 只把
# *脚本自己所在的目录*（src/eval/）加进 sys.path，不会自动把仓库根目录加进去，
# 下面 `from src.eval.metrics import ...` 这类"把 src 当顶层包"的写法就会报
# `ModuleNotFoundError: No module named 'src'`（哪怕在仓库根目录下执行这条命令
# 也一样，报错跟"当前工作目录"无关，跟"脚本参数里那个路径本身"有关）。这里手动
# 把仓库根目录（这个文件的上两级）插到 sys.path 最前面，兼容两种跑法。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if TYPE_CHECKING:
    # 只在类型检查时导入，不要求这个模块在没装 openai 包的环境（比如本地跑
    # 单元测试、只测 build_memory/character_card_text/load_dialogues 这些纯
    # 逻辑函数）里也能 import 成功，跟 src/eval/metrics.py 的做法一致。真正
    # 需要 OpenAI 类的地方（client_for()）在函数体内部延迟 import。
    from openai import OpenAI
else:
    OpenAI = Any

from src.eval.metrics import (
    character_fact_consistency,
    evidence_traceability_rate,
    ooc_rate,
    relationship_state_accuracy,
)
from src.generate.dialogue_runner import run_dialogue
from src.memory.raw_context import RawContextMemory
from src.memory.structured_memory import StructuredEventMemory
from src.memory.vector_memory import VectorMemory

FIELDNAMES = ["model", "memory_scheme", "dialogue_id", "n_turns",
              "fact_consistency", "ooc_rate", "relationship_accuracy", "evidence_traceability"]


def load_dialogues(path) -> list[dict]:
    dialogues = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            dialogues.append(json.loads(line))
    return dialogues


def build_memory(scheme_name: str, memory_params: dict, dialogue: dict):
    """每条 dialogue 必须拿到一个全新的 memory 实例——不能跨对话复用同一个对象，
    不同故事的记忆会串到一起，见 dialogue_runner.py 文件头说明。

    structured_memory 只导入这条 dialogue 自己的 fact_labels
    （load_character_facts_from_records），不能传整份 character_features_en.jsonl
    的路径走 load_character_facts()——那样会把不同故事、恰好共用同一个占位
    标签（比如 "I"，实测 61 段对话里有 20 段都是这个标签）的事实混到一起，
    检索时张冠李戴。这个坑踩过一次，详细原因见 docs/DATASET.md。
    """
    if scheme_name == "raw_context":
        return RawContextMemory(**memory_params.get("raw_context", {}))
    if scheme_name == "vector_memory":
        return VectorMemory(**memory_params.get("vector_memory", {}))
    if scheme_name == "structured_memory":
        mem = StructuredEventMemory()
        mem.load_character_facts_from_records(dialogue["fact_labels"])
        return mem
    raise ValueError(f"未知的记忆方案：{scheme_name}（configs/experiment.yaml 里 memory_schemes 拼错了？）")


def character_card_text(character_card: dict) -> str:
    """把 character_card 这个 dict 转成给 OOC judge 看的人话文本。

    ooc_rate() 的 character_card 参数会被直接塞进 judge_prompts.OOC_JUDGE_PROMPT
    的 {character_card} 占位符——传字典进去会被 str() 成 Python repr，可读性差，
    还会让 judge 同时看到 name/display_name 两个字段搞不清哪个才是"名字"。这里
    统一用 display_name（读起来自然，见 build_dialogues.py::pick_display_name），
    不用 name——后者是给 structured_memory 做事实检索用的 canonical 标签，可能
    是 "I" 这种占位词，不适合出现在给 judge 看的角色设定描述里。
    """
    display_name = character_card.get("display_name") or character_card.get("name", "该角色")
    lines = [f"角色名：{display_name}"]
    if character_card.get("description"):
        lines.append(f"角色设定：{character_card['description']}")
    if character_card.get("greeting"):
        lines.append(f"开场白示例：{character_card['greeting']}")
    return "\n".join(lines)


def run_one(client, model_cfg: dict, judge_client, judge_model: str, scheme_name: str,
            memory_params: dict, dialogue: dict, max_turns_per_dialogue: int,
            judge_repeats: int, seed: int | None = None) -> tuple[dict, list[dict]]:
    """跑一次生成 + 四个指标，返回 (summary 行, 完整 turns 列表)。

    四个指标各自独立 try/except——某一个指标算不出来（目前主要是
    relationship_state_accuracy，因为 relationship_trajectory 还是空的）不应该
    让另外三个也白算，更不应该让整次生成的结果全部丢弃。summary.csv 里对应
    格子留空（None -> csv 里是空字符串），不是当成 0 分，避免"算不出来"和
    "算出来是最差情况"被混为一谈。
    """
    memory = build_memory(scheme_name, memory_params, dialogue)

    scripted_user_turns = dialogue["scripted_user_turns"][:max_turns_per_dialogue]
    turns = run_dialogue(
        memory=memory,
        client=client,
        model=model_cfg["served_model_name"],
        character_card=dialogue["character_card"],
        scripted_user_turns=scripted_user_turns,
        seed=seed,
    )

    result = {
        "model": model_cfg["name"],
        "memory_scheme": scheme_name,
        "dialogue_id": dialogue["dialogue_id"],
        "n_turns": len(turns),
    }

    try:
        result["fact_consistency"] = character_fact_consistency(
            judge_client, judge_model, turns, dialogue["fact_labels"], judge_repeats)
    except (ValueError, RuntimeError) as e:
        print(f"    [跳过 fact_consistency] {e}")
        result["fact_consistency"] = None

    try:
        result["ooc_rate"] = ooc_rate(
            judge_client, judge_model, turns,
            character_card_text(dialogue["character_card"]), judge_repeats)
    except (ValueError, RuntimeError) as e:
        print(f"    [跳过 ooc_rate] {e}")
        result["ooc_rate"] = None

    try:
        result["relationship_accuracy"] = relationship_state_accuracy(
            judge_client, judge_model, turns, dialogue["relationship_trajectory"],
            judge_repeats=judge_repeats)
    except (ValueError, RuntimeError) as e:
        # relationship_trajectory 目前全是 []（弱标签脚本还没做，见
        # docs/DATASET.md），这里对每条对话都会走到这个分支——预期行为。
        result["relationship_accuracy"] = None

    try:
        result["evidence_traceability"] = evidence_traceability_rate(memory)
    except ValueError:
        # raw_context 方案本来就没有"检索"概念，这里必然走到这个分支，预期行为。
        result["evidence_traceability"] = None

    return result, turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-url-template", default="http://localhost:{port}/v1",
                     help="vLLM 服务地址模板，{port} 会被 --model-port 对应的端口替换；"
                          "SURF 上每个模型（含 judge）各起一个 vLLM 服务、各占一个端口，"
                          "怎么起见 docs/SURF_GUIDE.md")
    ap.add_argument("--model-port", action="append", default=[],
                     help="格式 name=port，比如 --model-port qwen2.5-7b=8000，"
                          "可重复传多次；judge_model 的端口用 --model-port judge=<port>")
    ap.add_argument("--limit", type=int, default=None, help="调试用：只跑前 N 条对话")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--seed", type=int, default=None,
                     help="角色回复生成的采样种子（传给 vLLM）。重复实验时每次用不同的 "
                          "seed 加不同的 --out-dir，不要覆盖已有结果；不传则不固定种子")
    ap.add_argument("--dry-run", action="store_true",
                     help="不连接任何 vLLM 服务，只打印会跑的矩阵规模、检查 "
                          "dialogues.jsonl 和 memory 实例化是否报错，用来先排掉配置错误，"
                          "不用真的等生成+judge调用跑一遍才发现路径写错了")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dialogues = load_dialogues(config["data"]["dialogues_path"])
    if args.limit:
        dialogues = dialogues[: args.limit]

    n_total = len(dialogues) * len(config["models"]) * len(config["memory_schemes"])
    print(f"读到 {len(dialogues)} 条对话，{len(config['models'])} 个模型 x "
          f"{len(config['memory_schemes'])} 种记忆方案 = {n_total} 次生成")

    if args.dry_run:
        # 只走 build_memory()（会真的解析 fact_labels、加载 embedding 模型等），
        # 不调用 run_dialogue()/任何 judge，快速验证配置和数据本身没问题。
        for scheme_name in config["memory_schemes"]:
            for dialogue in dialogues[:3]:
                build_memory(scheme_name, config["memory_params"], dialogue)
        print("dry-run 通过：数据能读、每种记忆方案都能正常实例化。真的要跑生成，"
              "去掉 --dry-run 并传 --model-port。")
        return

    from openai import OpenAI as OpenAIClient  # 延迟 import，见文件顶部说明

    ports = dict(kv.split("=", 1) for kv in args.model_port)

    def client_for(name: str) -> OpenAI:
        port = ports.get(name)
        if port is None:
            raise SystemExit(f"没有给 {name} 传 --model-port（比如 --model-port {name}=8000），"
                              f"不知道去哪个地址连 vLLM 服务")
        return OpenAIClient(base_url=args.base_url_template.format(port=port), api_key="EMPTY")

    judge_client = client_for("judge")
    judge_model = config["judge_model"]["served_model_name"]

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"

    n_done = 0
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for model_cfg in config["models"]:
            client = client_for(model_cfg["name"])
            for scheme_name in config["memory_schemes"]:
                for dialogue in dialogues:
                    t0 = time.time()
                    result, turns = run_one(
                        client, model_cfg, judge_client, judge_model, scheme_name,
                        config["memory_params"], dialogue,
                        config["data"]["max_turns_per_dialogue"],
                        config["eval"]["judge_repeats"],
                        seed=args.seed,
                    )
                    writer.writerow(result)
                    f.flush()  # 每条都落盘，长跑中途断了不会全部白跑

                    raw_path = raw_dir / f"{model_cfg['name']}__{scheme_name}__{dialogue['dialogue_id']}.json"
                    raw_path.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")

                    n_done += 1
                    print(f"[{n_done}/{n_total}] {model_cfg['name']} / {scheme_name} / "
                          f"{dialogue['dialogue_id'][:24]}  ({time.time() - t0:.1f}s)")

    print(f"\n完成，结果写到 {summary_path}，每条对话的完整生成记录在 {raw_dir}/")


if __name__ == "__main__":
    main()

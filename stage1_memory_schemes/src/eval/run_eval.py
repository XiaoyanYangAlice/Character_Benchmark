"""
主入口：跑完整 N(记忆方案) x M(模型) x K(对话) 实验矩阵。

落盘三份结果：
  results/summary.csv    每次生成一行汇总指标（含前段/后段分层）
  results/per_turn.csv   **逐轮判定**（v2 新增）：每个角色轮次一行，judge 的
                         逐次投票和理由都在里面
  results/raw_outputs/   每条对话的完整生成记录（含每轮注入的上下文长度）

per_turn.csv 是 v2 最重要的新增产物。v1 只存汇总值，导致后来做 judge 审核时
必须重新判一遍 600 轮；重判与主实验那次不是同一批判定（judge 温度非 0），
审核结论和主实验之间始终隔着一层。存了逐轮判定之后，审核可以直接抽主实验
真正用过的那些判定来人工核对，也可以按方案分层、按轮次位置分层做分析，
不用再花 GPU 重判。

用法（模型和 judge 各自的 vLLM 服务已经在 SURF 上起好之后，见 docs/SURF_GUIDE.md）：
    python src/eval/run_eval.py --config configs/experiment.yaml \
        --model-port qwen2.5-7b=8000 --model-port llama3.1-8b=8001 --model-port judge=8002

调试时不想真的连 vLLM、只想看看矩阵和数据对不对，用 --dry-run（不调用任何模型，
只打印会跑哪些组合、检查 dialogues.jsonl / memory 实例化是否报错）。

⚠ 在 relationship_trajectory 弱标签打分脚本做完之前（TODO，见 docs/DATASET.md），
dialogues.jsonl 里这个字段全是空列表 []，relationship_accuracy 一列会全部留空
——这是预期行为，不是 bug。
"""
import argparse
import csv
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    aggregate_per_turn,
    judge_turns,
    probe_recall_accuracy,
    relationship_state_accuracy,
)
from src.generate.dialogue_runner import run_dialogue
from src.memory.no_memory import NoMemory
from src.memory.raw_context import RawContextMemory
from src.memory.shuffled_facts import ShuffledFactsMemory
from src.memory.structured_memory import StructuredEventMemory, make_llm_event_extractor
from src.memory.vector_memory import VectorMemory

FIELDNAMES = [
    "model", "memory_scheme", "dialogue_id", "n_turns", "n_character_turns",
    # 主指标（整段对话）
    "fact_consistency", "ooc_rate", "leakage_rate", "probe_accuracy",
    # 前段 / 后段分层：记忆方案的差别应该主要出现在后段
    "ooc_rate_early", "ooc_rate_late", "fact_consistency_early", "fact_consistency_late",
    # 诊断列
    "evidence_use_rate", "mean_context_chars", "n_probes", "n_events_extracted",
    "n_extraction_failures", "n_judge_fail_ooc", "n_judge_fail_fact",
    "relationship_accuracy",
]

PER_TURN_FIELDNAMES = [
    "model", "memory_scheme", "dialogue_id", "turn_id", "turn_index", "is_probe",
    "ooc", "ooc_votes", "fact_contradicts", "fact_votes", "leakage",
    "context_chars", "reply_chars", "evidence_overlap", "ooc_reason",
]

PROBE_FIELDNAMES = ["model", "memory_scheme", "dialogue_id", "turn_id",
                    "source_turn_index", "correct", "votes", "question",
                    "expected_answer", "reason"]

# 带状态、需要 on_new_turn 增量写入的方案。raw_context/no_memory 无状态。
STATEFUL_SCHEMES = {"vector_memory", "structured_memory", "structured_hybrid", "shuffled_facts"}


def load_dialogues(path) -> list[dict]:
    dialogues = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            dialogues.append(json.loads(line))
    return dialogues


def pick_donor(dialogue: dict, all_dialogues: list[dict], seed: int = 0) -> dict | None:
    """给 shuffled_facts 对照挑一个"事实捐赠者"：另一条对话，事实条数接近，
    这样注入内容的长度和形式都与真实事实可比，唯一的差别是内容是错的。

    用确定性的 seed（按 dialogue_id 哈希）挑，保证每次重跑、每个模型、每个
    seed 下，同一条对话配到的都是同一个捐赠者——否则 shuffled 这一列的方差
    里会混进"这次恰好配到了一个事实特别多/特别少的捐赠者"，与记忆方案无关。
    """
    candidates = [d for d in all_dialogues if d["dialogue_id"] != dialogue["dialogue_id"]
                  and d.get("fact_labels")]
    if not candidates:
        return None
    n_own = len(dialogue.get("fact_labels", []))
    # 先按"事实条数与自己最接近"排序，再在最接近的一小批里用确定性随机挑一个，
    # 避免总是挑到同一条（那条的事实会被反复注入到很多对话里）。
    candidates.sort(key=lambda d: abs(len(d["fact_labels"]) - n_own))
    pool = candidates[: max(1, min(5, len(candidates)))]
    rng = random.Random(f"{dialogue['dialogue_id']}#{seed}")
    return rng.choice(pool)


def build_memory(scheme_name: str, memory_params: dict, dialogue: dict,
                 all_dialogues: list[dict] | None = None, event_extractor=None):
    """每条 dialogue 必须拿到一个全新的 memory 实例——不能跨对话复用同一个对象，
    不同故事的记忆会串到一起，见 dialogue_runner.py 文件头说明。

    structured_memory 只导入这条 dialogue 自己的 fact_labels
    （load_character_facts_from_records），不能传整份 character_features_en.jsonl
    的路径走 load_character_facts()——那样会把不同故事、恰好共用同一个占位
    标签（比如 "I"）的事实混到一起，检索时张冠李戴。这个坑踩过一次，详细原因
    见 docs/DATASET.md。
    """
    if scheme_name == "no_memory":
        return NoMemory()
    if scheme_name == "raw_context":
        return RawContextMemory(**memory_params.get("raw_context", {}))
    if scheme_name == "vector_memory":
        return VectorMemory(**memory_params.get("vector_memory", {}))
    if scheme_name in ("structured_memory", "structured_hybrid"):
        # 两者共用同一个类，只是 memory_params 里的 retrieval 不同（recency / hybrid）。
        # 分成两个方案名而不是改一个，是为了能在同一次运行里直接配对比较。
        params = dict(memory_params.get(scheme_name, {}))
        mem = StructuredEventMemory(event_extractor=event_extractor, **params)
        mem.load_character_facts_from_records(dialogue["fact_labels"])
        return mem
    if scheme_name == "shuffled_facts":
        params = dict(memory_params.get("structured_memory", {}))
        # 对照组刻意不开事件抽取：它要对照的是"静态事实清单的内容对不对"，
        # 开了抽取就多了一个变量（抽取出来的事件是对的，事实是错的，混在一起
        # 没法解释）。这一点必须在报告里写明。
        mem = ShuffledFactsMemory(event_extractor=None, **params)
        donor = pick_donor(dialogue, all_dialogues or [], seed=0)
        if donor is None:
            raise ValueError("shuffled_facts 需要至少两条带 fact_labels 的对话才能做对照")
        mem.load_donor_facts(donor["fact_labels"], dialogue["character_card"]["name"])
        mem.donor_dialogue_id = donor["dialogue_id"]
        return mem
    raise ValueError(f"未知的记忆方案：{scheme_name}（configs/experiment.yaml 里 memory_schemes 拼错了？）")


def character_card_text(character_card: dict) -> str:
    """把 character_card 这个 dict 转成给 OOC judge 看的人话文本。

    judge prompt 的 {character_card} 占位符会直接收到这段文本——传字典进去会被
    str() 成 Python repr，可读性差，还会让 judge 同时看到 name/display_name
    两个字段搞不清哪个才是"名字"。这里统一用 display_name（读起来自然，见
    build_dialogues.py::pick_display_name），不用 name——后者是给
    structured_memory 做事实检索用的 canonical 标签，可能是 "I" 这种占位词，
    不适合出现在给 judge 看的角色设定描述里。
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
            judge_repeats: int, seed: int | None = None, judge_repeats_fact: int | None = None,
            all_dialogues: list[dict] | None = None, event_extractor=None,
            ooc_prompt_version: str = "v2", late_from_fraction: float = 0.5,
            ) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """跑一次生成 + 判定。

    返回 (summary 行, turns, 逐轮判定记录, 探针记录)。

    判定分两步：先 judge_turns() 产出逐轮记录，再 aggregate_per_turn() 汇总。
    这样"原始判定"和"怎么汇总"分开，事后换汇总口径（比如改前后段的切分点、
    只看非探针轮次）不需要重新调用 judge。
    """
    memory = build_memory(scheme_name, memory_params, dialogue, all_dialogues, event_extractor)

    scripted_user_turns = dialogue["scripted_user_turns"][:max_turns_per_dialogue]
    probes = dialogue.get("memory_probes") or []
    turns = run_dialogue(
        memory=memory,
        client=client,
        model=model_cfg["served_model_name"],
        character_card=dialogue["character_card"],
        scripted_user_turns=scripted_user_turns,
        seed=seed,
        memory_probes=probes,
    )

    result = {
        "model": model_cfg["name"],
        "memory_scheme": scheme_name,
        "dialogue_id": dialogue["dialogue_id"],
        "n_turns": len(turns),
        "n_probes": len(probes),
        "n_events_extracted": len(getattr(memory, "events", []) or []),
        "n_extraction_failures": getattr(memory, "extraction_failures", 0),
    }

    per_turn: list[dict] = []
    probe_records: list[dict] = []
    try:
        per_turn = judge_turns(
            judge_client, judge_model, turns,
            character_card_text(dialogue["character_card"]),
            dialogue["fact_labels"],
            judge_repeats=judge_repeats,
            judge_repeats_fact=judge_repeats_fact,
            ooc_prompt_version=ooc_prompt_version,
            retrieval_log=getattr(memory, "retrieval_log", None),
        )
        result.update(aggregate_per_turn(per_turn, late_from_fraction=late_from_fraction))
    except (ValueError, RuntimeError) as e:
        print(f"    [跳过逐轮判定] {e}")

    try:
        # 探针判分比 OOC 客观得多（"关键信息对上没有"），用较少的重复即可
        acc, probe_records = probe_recall_accuracy(judge_client, judge_model, turns, probes,
                                                    judge_repeats=judge_repeats_fact or judge_repeats)
        result["probe_accuracy"] = acc
    except (ValueError, RuntimeError) as e:
        print(f"    [跳过 probe_accuracy] {e}")
        result["probe_accuracy"] = None

    try:
        result["relationship_accuracy"] = relationship_state_accuracy(
            judge_client, judge_model, turns, dialogue["relationship_trajectory"],
            judge_repeats=judge_repeats)
    except (ValueError, RuntimeError):
        # relationship_trajectory 目前全是 []（弱标签脚本还没做，见
        # docs/DATASET.md），这里对每条对话都会走到这个分支——预期行为。
        result["relationship_accuracy"] = None

    # 只保留 FIELDNAMES 里的列，多余的键（aggregate 返回的诊断值）丢掉，
    # 避免 DictWriter 报 ValueError。
    return {k: result.get(k) for k in FIELDNAMES}, turns, per_turn, probe_records


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
    ap.add_argument("--workers", type=int, default=None,
                     help="并发跑多少条 (模型, 方案, 对话) 任务。vLLM 会把并发请求合批，"
                          "吞吐几乎线性增长；单线程时 GPU 大部分时间在空等。不传则读 "
                          "configs/experiment.yaml 的 eval.workers，再默认 8。")
    ap.add_argument("--only-model", action="append", default=[],
                     help="只跑这个模型（按 config 里的 name，可重复）。分阶段跑时用："
                          "先 --only-model qwen2.5-7b，Llama 以后再补。")
    ap.add_argument("--only-scheme", action="append", default=[],
                     help="只跑这个记忆方案（可重复）。")
    ap.add_argument("--dry-run", action="store_true",
                     help="不连接任何 vLLM 服务，只打印会跑的矩阵规模、检查 "
                          "dialogues.jsonl 和 memory 实例化是否报错，用来先排掉配置错误，"
                          "不用真的等生成+judge调用跑一遍才发现路径写错了")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    all_dialogues = load_dialogues(config["data"]["dialogues_path"])
    dialogues = all_dialogues[: args.limit] if args.limit else all_dialogues

    eval_cfg = config.get("eval", {})
    judge_repeats = eval_cfg.get("judge_repeats", 3)
    judge_repeats_fact = eval_cfg.get("judge_repeats_fact", judge_repeats)
    ooc_prompt_version = eval_cfg.get("ooc_prompt_version", "v2")
    late_frac = eval_cfg.get("late_from_fraction", 0.5)
    want_events = eval_cfg.get("event_extraction", True)

    n_total = len(dialogues) * len(config["models"]) * len(config["memory_schemes"])
    n_probes = sum(len(d.get("memory_probes") or []) for d in dialogues)
    turn_counts = [len(d["scripted_user_turns"]) for d in dialogues]
    print(f"读到 {len(dialogues)} 条对话，{len(config['models'])} 个模型 x "
          f"{len(config['memory_schemes'])} 种记忆方案 = {n_total} 次生成")
    if turn_counts:
        print(f"用户轮次：中位数 {sorted(turn_counts)[len(turn_counts) // 2]}，"
              f"最小 {min(turn_counts)}，最大 {max(turn_counts)}；"
              f"raw_context 窗口 {config['memory_params'].get('raw_context', {}).get('max_turns')} 轮")
    print(f"记忆探针共 {n_probes} 个，judge_repeats={judge_repeats}"
          f"（事实一致性 {judge_repeats_fact}），"
          f"OOC prompt={ooc_prompt_version}，事件抽取={'开' if want_events else '关'}")
    if n_probes == 0:
        print("⚠ 一个探针都没有：先跑 src/generate/make_probes.py，否则没有任何指标在直接测记忆。")

    if args.dry_run:
        # 只走 build_memory()（会真的解析 fact_labels、加载 embedding 模型等），
        # 不调用 run_dialogue()/任何 judge，快速验证配置和数据本身没问题。
        for scheme_name in config["memory_schemes"]:
            for dialogue in dialogues[:3]:
                build_memory(scheme_name, config["memory_params"], dialogue, all_dialogues)
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
    # 事件抽取用 judge 那个服务（14B，比 agent 强），但温度 0、prompt 不同。
    # 注意：这让 structured_memory 比其它方案多用了一个模型，报告里要说明这是
    # 方案本身的一部分成本（"结构化记忆需要一个抽取器"），不是评测上的不公平。
    event_extractor = make_llm_event_extractor(judge_client, judge_model) if want_events else None

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    per_turn_path = out_dir / "per_turn.csv"
    probe_path = out_dir / "probe_results.csv"

    workers = args.workers or eval_cfg.get("workers", 8)
    models = [m for m in config["models"] if not args.only_model or m["name"] in args.only_model]
    schemes = [sc for sc in config["memory_schemes"] if not args.only_scheme or sc in args.only_scheme]
    if not models or not schemes:
        raise SystemExit(f"--only-model/--only-scheme 过滤后没有任务：models={[m['name'] for m in config['models']]} "
                         f"schemes={config['memory_schemes']}")
    tasks = [(m, sc, d) for m in models for sc in schemes for d in dialogues]
    n_total = len(tasks)
    print(f"实际任务数 {n_total}，并发 {workers}")
    clients = {m["name"]: client_for(m["name"]) for m in models}

    def do(task):
        model_cfg, scheme_name, dialogue = task
        t0 = time.time()
        out = run_one(
            clients[model_cfg["name"]], model_cfg, judge_client, judge_model, scheme_name,
            config["memory_params"], dialogue,
            config["data"]["max_turns_per_dialogue"],
            judge_repeats,
            seed=args.seed,
            judge_repeats_fact=judge_repeats_fact,
            all_dialogues=all_dialogues,
            event_extractor=event_extractor,
            ooc_prompt_version=ooc_prompt_version,
            late_from_fraction=late_frac,
        )
        return task, out, time.time() - t0

    n_done = 0
    n_err = 0
    lock = threading.Lock()
    with open(summary_path, "w", newline="", encoding="utf-8") as f, \
         open(per_turn_path, "w", newline="", encoding="utf-8") as ft, \
         open(probe_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        tw = csv.DictWriter(ft, fieldnames=PER_TURN_FIELDNAMES, extrasaction="ignore")
        tw.writeheader()
        pw = csv.DictWriter(fp, fieldnames=PROBE_FIELDNAMES, extrasaction="ignore")
        pw.writeheader()

        # 只有主线程写文件（as_completed 在主线程里消费），不需要给 csv 加锁。
        with ThreadPoolExecutor(workers) as ex:
            futures = [ex.submit(do, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    (model_cfg, scheme_name, dialogue), (result, turns, per_turn, probe_records), dt = fut.result()
                except Exception as e:  # 单条失败不拖垮整个作业；最后统一报告
                    n_err += 1
                    print(f"  [任务失败] {type(e).__name__}: {e}")
                    continue
                writer.writerow(result)
                ident = {"model": model_cfg["name"], "memory_scheme": scheme_name,
                         "dialogue_id": dialogue["dialogue_id"]}
                tw.writerows({**ident, **r} for r in per_turn)
                pw.writerows({**ident, **r} for r in probe_records)
                for handle in (f, ft, fp):
                    handle.flush()  # 每条都落盘，长跑中途断了不会全部白跑

                raw_path = raw_dir / f"{model_cfg['name']}__{scheme_name}__{dialogue['dialogue_id']}.json"
                raw_path.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")

                n_done += 1
                ooc = result.get("ooc_rate")
                print(f"[{n_done}/{n_total}] {model_cfg['name']} / {scheme_name} / "
                      f"{dialogue['dialogue_id'][:20]}  ooc={ooc if ooc is None else f'{ooc:.2f}'}  "
                      f"({dt:.1f}s)", flush=True)

    if n_err:
        print(f"!!!!! {n_err}/{n_total} 个任务失败，summary.csv 不完整 !!!!!")
        sys.exit(1)

    print(f"\n完成。汇总 {summary_path}，逐轮判定 {per_turn_path}，探针 {probe_path}，"
          f"原始生成 {raw_dir}/")
    print("下一步：python src/eval/compare_schemes.py --csv <各次重复的 summary.csv>")


if __name__ == "__main__":
    main()

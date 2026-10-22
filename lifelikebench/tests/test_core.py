import json
import os
import subprocess
import sys
from pathlib import Path

from lifelike_bench.agreement import krippendorff_alpha, spearman
from lifelike_bench.data import instantiate, load_characters, load_probe_templates
from lifelike_bench.state_engine import StateEngine

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
N = None
# Krippendorff (2011) "Computing Krippendorff's alpha-reliability", reference example
A = [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N]
B = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3]
C = [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N]
D = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N]
UNITS = [list(u) for u in zip(A, B, C, D)]


def test_alpha_reference_values():
    assert abs(krippendorff_alpha(UNITS, "nominal") - 0.743) < 0.001
    assert abs(krippendorff_alpha(UNITS, "interval") - 0.849) < 0.001
    assert abs(krippendorff_alpha(UNITS, "ordinal") - 0.815) < 0.001


def test_spearman():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-9


def test_probes_instantiate_for_every_character():
    chars = load_characters(ROOT / "data/characters")
    for tpl in load_probe_templates(ROOT / "data/probes/pilot_probes.yaml"):
        for card in chars.values():
            p = instantiate(tpl, card)
            assert all(t.text and "{" not in t.text for t in p.turns)
            if p.dimension == "D1":
                tags = [t.tag for t in p.turns]
                assert "event" in tags and "timeskip" in tags and tags.count("warmup") >= 1


def test_state_engine_reacts_and_recovers():
    card = load_characters(ROOT / "data/characters")["lin_wan"]
    eng = StateEngine(card, seed=0, noise=False)
    base = eng.state["tension"]
    hit = eng.step({"conflict": 0.7}, 3)["tension"]
    later = eng.step({}, 1440)["tension"]
    assert hit > base + 0.2
    assert abs(later - base) < abs(hit - base) / 2


def test_mock_pipeline(tmp_path):
    cfg = (ROOT / "configs/mock.yaml").read_text().replace("run_name: mock", "run_name: t")
    cfg += f"\nresults_dir: {tmp_path}\n"
    (tmp_path / "c.yaml").write_text(cfg)
    subprocess.run([sys.executable, "-m", "lifelike_bench.cli", "all", str(tmp_path / "c.yaml")],
                   cwd=ROOT, env=ENV, check=True, capture_output=True)
    m = json.loads((tmp_path / "t/metrics.json").read_text())
    leaky, base = m["systems"]["leaky_mock"], m["systems"]["base_mock"]
    assert leaky["D2"]["assistant_tone_rate"]["mean"] > base["D2"]["assistant_tone_rate"]["mean"]
    assert (tmp_path / "t/report.md").exists()


def test_local_orchestration(tmp_path):
    """Fake vLLM servers: phases start/stop servers, helper on another model, resume skips."""
    cfg = (ROOT / "configs/local_mock.yaml").read_text() + f"\nresults_dir: {tmp_path}\n"
    (tmp_path / "c.yaml").write_text(cfg)
    run = lambda: subprocess.run([sys.executable, "-m", "lifelike_bench.cli", "local",
                                  str(tmp_path / "c.yaml")], cwd=ROOT, env=ENV, check=True,
                                 capture_output=True, text=True).stdout
    first = run()
    assert "starting small_a" in first and "starting judge_y" in first
    out = tmp_path / "local_mock"
    assert "<think>" not in (out / "transcripts.jsonl").read_text()
    js = [json.loads(l) for l in (out / "judgments.jsonl").read_text().splitlines()]
    assert js and not any("_error" in r["result"] for r in js)
    second = run()
    assert "starting" not in second and "skipping" in second


def test_harness_memory_provenance_rendering():
    from lifelike_bench.backends import MockBackend
    from lifelike_bench.systems import HarnessSystem
    card = load_characters(ROOT / "data/characters")["lin_wan"]
    h = HarnessSystem(card, MockBackend(), {}, 0)
    h.memory = [
        {"turn": 1, "fact": "养了一只叫年糕的白猫", "source": "user_canon", "conflict": False},
        {"turn": 2, "fact": "上个月请对方喝奶茶", "source": "claimed", "conflict": False},
        {"turn": 3, "fact": "去年夏天一起看日出", "source": "claimed", "conflict": True,
         "conflict_with": "才认识半年"},
    ]
    p = h.system_prompt()
    assert "视为正史" in p and "当作确有其事" in p and "情绪反应照常" in p and "与设定矛盾（才认识半年）" in p
    strict = HarnessSystem(card, MockBackend(), {}, 0, cocreate=False)
    strict.memory = h.memory[1:2]
    assert "可以追问" in strict.system_prompt()


def test_new_memory_probes_have_card_specific_gold():
    chars = load_characters(ROOT / "data/characters")
    tpls = {t["id"]: t for t in load_probe_templates(ROOT / "data/probes/pilot_probes.yaml")}
    for card in chars.values():
        p = instantiate(tpls["memory_conflict_claim"], card)
        assert p.gold_type == "conflict_claim" and "{" not in p.gold
        assert any(t.tag == "claim_conflict" for t in p.turns)


def test_memory_guards_around_extractor():
    import json as _j
    from lifelike_bench.backends import Backend
    from lifelike_bench.data import Turn
    from lifelike_bench.systems import HarnessSystem

    class Fake(Backend):
        def __init__(self, out): self.out = out
        def chat(self, *a, **k): return _j.dumps(self.out, ensure_ascii=False)

    card = load_characters(ROOT / "data/characters")["lin_wan"]
    out = {"events": {"intimacy": 0.8}, "memory": [
        {"fact": "去年夏天一起看日出", "source": "claimed", "conflict": True, "conflict_with": "三月才认识"},
        {"fact": "", "source": "stated"},
        {"fact": "\u200b", "source": "stated"},
        {"fact": "用户询问角色中午吃了什么", "source": "stated"},
        {"fact": "同一实验室", "source": "user_canon"}]}
    h = HarnessSystem(card, Fake(out), {}, 0)
    h.before_reply("还记得去年夏天我们一起看日出吗？", Turn(0, "claim_conflict", "", 3))
    srcs = [(m["source"], m["conflict"]) for m in h.memory]
    assert srcs == [("claimed", True), ("stated", False)]       # empty dropped; canon needs a marker
    assert h.turn_trace[-1]["events"] == {}                      # conflicting claim moves no state
    h2 = HarnessSystem(card, Fake({"events": {}, "memory": [
        {"fact": "养了一只猫", "source": "claimed", "conflict": True}]}), {}, 0)
    h2.before_reply("（设定补充：你养了一只猫）嗨", Turn(0, "canon_add", "", 3))
    assert h2.memory[0]["source"] == "user_canon" and h2.memory[0]["conflict"] is False


def test_recency_note_and_alert_go_to_latest_user_turn():
    import json as _j
    from lifelike_bench.backends import Backend
    from lifelike_bench.data import Turn
    from lifelike_bench.systems import HarnessSystem

    class Rec(Backend):
        def __init__(self): self.calls = []
        def chat(self, system, messages, **k):
            self.calls.append((system, messages))
            if "inner-state engine" in system:   # extractor
                return _j.dumps({"events": {}, "memory": [
                    {"fact": "去年夏天一起看日出", "source": "claimed", "conflict": True,
                     "conflict_with": "两人今年三月才认识"}]}, ensure_ascii=False)
            return "嗯？"

    card = load_characters(ROOT / "data/characters")["lin_wan"]
    # same information, different position: system prompt (default) vs latest user turn
    b0 = Rec(); h0 = HarnessSystem(card, b0, {}, 0)
    h0.respond("还记得去年夏天吗？", Turn(0, "claim_conflict", "", 3))
    sys0, msgs0 = b0.calls[-1]
    assert "【你的记忆" in sys0 and msgs0[-1]["content"] == "还记得去年夏天吗？"

    b1 = Rec(); h1 = HarnessSystem(card, b1, {}, 0, note="user", alert=True)
    h1.respond("还记得去年夏天吗？", Turn(0, "claim_conflict", "", 3))
    sys1, msgs1 = b1.calls[-1]
    assert "【你的记忆" not in sys1
    assert "导演提示" in msgs1[-1]["content"] and "⚠" in msgs1[-1]["content"]
    assert "两人今年三月才认识" in msgs1[-1]["content"]
    assert h1.history[0]["content"] == "还记得去年夏天吗？"        # history stays clean
    # the alert is turn-local: next turn, no alert
    h1.before_reply  # noqa
    b1.calls.clear()
    h1.helper = type("E", (Backend,), {"chat": lambda self, *a, **k: '{"events": {}, "memory": []}'})()
    h1.respond("好吧。", Turn(1, "neutral", "", 3))
    assert "⚠" not in b1.calls[-1][1][-1]["content"]


def test_extended_character_set_loads_and_instantiates():
    chars = load_characters([ROOT / "data/characters", ROOT / "data/characters_ext"])
    assert len(chars) == 13
    tpls = load_probe_templates(ROOT / "data/probes/memory_claims.yaml")
    assert {t["id"] for t in tpls} == {"memory_conflict_claim", "memory_compatible_claim"}
    for card in chars.values():
        assert set(card["persona_tags"]) == {"reminisce", "skeptic"}
        for t in tpls:
            p = instantiate(t, card)
            assert "{" not in p.gold and all("{" not in x.text for x in p.turns)

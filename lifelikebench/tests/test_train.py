"""Training helpers that do not need torch (fake tokenizer with a Qwen-like template)."""
from lifelike_bench.train.common import encode, lr_at, prompt_and_target, sft_examples, weighted_sampler
from lifelike_bench.train.opd import cut_at_eos


class FakeTok:
    eos_token = "<E>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        s = "".join(f"<{m['role']}>{m['content']}<E>\n" for m in msgs)
        if add_generation_prompt:
            s += "<assistant>" + ("" if enable_thinking else "<think></think>")
        elif msgs and msgs[-1]["role"] == "assistant" and not enable_thinking:
            # final assistant turn rendered with the empty think block, as Qwen3 does
            s = s[: s.rfind("<assistant>")] + "<assistant><think></think>" + msgs[-1]["content"] + "<E>\n"
        return s

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_prompt_target_split():
    tok = FakeTok()
    p, t = prompt_and_target(tok, "SYS", [{"role": "user", "content": "hi"}], "hello")
    assert p.endswith("<assistant><think></think>") and t == "hello<E>\n"
    e = encode(tok, p, t, max_len=10_000)
    assert e["labels"].count(-100) == len(p) and e["n_target"] == len(t)
    e = encode(tok, p, t, max_len=20)          # long prompt is cut from the left, target kept
    assert len(e["input_ids"]) == 20 and e["n_target"] == len(t)


def test_sft_examples_one_per_assistant_turn():
    rows = [{"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "a"},
                          {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"},
                          {"role": "assistant", "content": "d"}]}]
    ex = sft_examples(rows)
    assert [r for _, _, r in ex] == ["b", "d"] and len(ex[1][1]) == 3 and ex[0][0] == "S"


def test_cut_and_schedule_and_sampler():
    assert cut_at_eos([5, 6, 2, 7], {2}, 0) == [5, 6, 2]
    assert cut_at_eos([5, 0, 0], {2}, 0) == [5]
    assert lr_at(0, 100, 1.0, 10) == 0.1 and abs(lr_at(100, 100, 1.0, 10)) < 1e-9
    rows = [{"tag": "claim_conflict"}] + [{"tag": "neutral"}] * 9
    g = weighted_sampler(rows, {"claim_conflict": 9.0}, 0)
    share = sum(next(g)["tag"] == "claim_conflict" for _ in range(2000)) / 2000
    assert 0.4 < share < 0.6


def test_adapter_routing():
    from lifelike_bench.runner import backend_for
    cfg = {"models": {"m": {"path": "x"}, "h": {"path": "y"}}, "serve": {"base_port": 9000}}
    s = {"id": "s", "condition": "harness_alert", "model": "m", "adapter": "student_opd", "helper_model": "h"}
    assert backend_for(cfg, s, "system").model == "student_opd"
    assert backend_for(cfg, s, "helper").model == "h"


def test_mixed_dialogue_compatible_turn_not_judged_as_conflict():
    from lifelike_bench.judges import items_for
    d = {"dimension": "D3", "gold_type": "conflict_claim",
         "turns": [{"tag": "warmup"}, {"tag": "claim_compatible"}, {"tag": "claim_conflict"}]}
    assert [t for k, t in items_for(d) if k == "memory"] == [2]


def test_dpo_pair_filter_and_encoding():
    from lifelike_bench.train.make_dpo_pairs import keep_pair
    from lifelike_bench.train.dpo import encode_pair
    assert keep_pair("a", "b", "stop") is None
    assert keep_pair("a", "a ", "stop") == "identical"
    assert keep_pair("a", "b", "length") == "rejected_unfinished"
    assert keep_pair("a", "  ", "stop") == "empty_rejected"
    row = {"prompt": {"system": "S", "messages": [{"role": "user", "content": "hi"}]},
           "chosen": "good", "rejected": "bad!", "tag": "claim_conflict"}
    p = encode_pair(FakeTok(), row, 10_000)
    for side, txt in (("chosen", "good"), ("rejected", "bad!")):
        e = p[side]
        start = len(e["labels"]) - e["n_target"]
        assert e["labels"][:start] == [-100] * start and e["labels"][start:] == e["input_ids"][start:]
        assert "".join(map(chr, e["input_ids"][start:])).startswith(txt)
    # both sides share the same prompt tokens
    assert p["chosen"]["input_ids"][:len(p["chosen"]["labels"]) - p["chosen"]["n_target"]] == \
           p["rejected"]["input_ids"][:len(p["rejected"]["labels"]) - p["rejected"]["n_target"]]

# Character Memory Eval

> **This folder is stage 1 of a research line** — see the [top-level README](../README.md). v2 and post-fix results: [`docs/COMPARE_V2.txt`](docs/COMPARE_V2.txt), [`docs/COMPARE_FIX.txt`](docs/COMPARE_FIX.txt), [`docs/PROBE_DIAGNOSIS_V2.txt`](docs/PROBE_DIAGNOSIS_V2.txt), [`docs/PROBE_DIAGNOSIS_FIX.txt`](docs/PROBE_DIAGNOSIS_FIX.txt); stages 2–3: [`../lifelikebench/`](../lifelikebench/).

**Comparing memory schemes for role-play AI — raw context vs. vector retrieval vs. structured character-fact memory — and auditing the LLM judge used to score them.**

Full write-up (Chinese): [README.md](./README.md), [docs/PROJECT_REPORT.md](./docs/PROJECT_REPORT.md).

> **Status: v1 results are complete and reported as-is; the code has been upgraded to v2 and is awaiting a re-run.**
> The main flaw in v1: each dialogue had only 6 user turns, so the 20-turn `raw_context` window
> never truncated — it was not testing memory at all, only three ways of writing a prompt. v2
> lengthens the dialogues so the window truncates, adds two control conditions (`no_memory`,
> `shuffled_facts`) to separate "remembered new information" from "was reminded of the persona",
> adds a memory-probe metric that tests recall directly, and stores per-turn judge verdicts so the
> audit can be stratified by scheme. All numbers below are from v1.
> See [docs/EXPERIMENT_V2.md](./docs/EXPERIMENT_V2.md) (Chinese).

## What this is

On 61 real role-play dialogues from PIPPA, with identical scripted user turns, we compare three memory schemes
(sliding-window raw context, sentence-embedding retrieval, structured character-fact memory) on fact consistency
and out-of-character (OOC) rate. Agents: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct; judge: Qwen2.5-14B-Instruct;
served with vLLM under SLURM on SURF Snellius. Matrix: 2 models × 3 schemes × 61 dialogues × 3 repeats.

## Headline results (and how far to trust them)

- Qwen2.5-7B: structured memory lowers the judge's OOC *flag rate* by 5.7 pp vs. raw context (paired bootstrap over
  dialogues, 95% CI [−9.9, −2.3], consistent across 3 repeats); vector memory by 2.6 pp [−4.6, −0.8];
  structured vs. vector is not established (CI includes 0). Llama-3.1-8B: no detectable OOC differences (all CIs include 0; structured is 0.5 pp lower than raw on fact consistency, CI [−0.9, −0.1], negligible).
- Fact-consistency gains are small (~+1.2 pp, near ceiling) and should not be claimed.
- **The judge is unreliable.** A blind audit of 40 turns (20 flagged, 20 unflagged) found judge precision ≈ 40%
  (95% CI 22–61%) against a single annotator (Claude; a human re-annotation is pending). A rewritten judge prompt did not
  help, and a smaller cross-family judge (Llama-3.1-8B) over-flagged almost everything. So absolute OOC rates are inflated and
  the relative differences are unvalidated. Details: [docs/JUDGE_AUDIT.md](./docs/JUDGE_AUDIT.md).
- Scope limits: very short dialogues (only the first 6 real user turns are scripted, 57 of 61 dialogues have exactly 6), so the 20-turn raw window never truncates and this is not a long-horizon memory test; the structured memory is
  currently only a fact list (event/relationship extraction is unimplemented) and overlaps with the character card already
  in the system prompt.

## Engineering lessons worth reading

Four silent failures were found and fixed, each of which would have produced plausible-looking but invalid numbers:
a retrieval call passing the wrong speaker (structured memory received an empty context for every turn);
an embedding model silently falling back to lexical similarity (missing package, then a GPU OOM against vLLM);
facts leaking across stories that share a pronoun label ("I"); and stale files on the cluster. Each has a regression test.
See [docs/RESULTS.md](./docs/RESULTS.md) and [docs/DATASET.md](./docs/DATASET.md).

## Quick start

```bash
pip install -r requirements.txt
python run_tests.py        # 55 tests, no GPU needed
```

Full experiment and judge audit: see the SLURM scripts in `slurm/` and [docs/SURF_GUIDE.md](./docs/SURF_GUIDE.md).

## Data and license

Code: MIT (see [`../LICENSE`](../LICENSE)). Dialogues come from [PIPPA](https://huggingface.co/datasets/PygmalionAI/PIPPA) (Apache-2.0; flagged as containing
sensitive/adult content). No dialogue text or raw model outputs are redistributed — follow [docs/DATASET.md](./docs/DATASET.md) to rebuild the data.

The fact-extraction schema follows the GOLEM ontology: Pianzola, F., Cheng, L., Pannach, F., Yang, X., & Scotti, L. (2025). The GOLEM Ontology for Narrative and Fiction. *Humanities*, 14(10), 193. https://doi.org/10.3390/h14100193

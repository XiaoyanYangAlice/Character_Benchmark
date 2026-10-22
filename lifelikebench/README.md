# LifelikeBench

**Measuring, improving and internalising "lifelikeness" (活人感) in LLM role-play: a benchmark, an inference-time harness, and on-policy distillation, all on local open-weight models.**

> Status: **ongoing research project** (Oct 2026). Benchmark, harness and ablations are complete;
> the distillation stage has its data built and its training code written, and training runs are in progress.

LifelikeBench studies two ways role-play characters stop feeling alive:

1. **Memory.** When a user mentions a shared past that contradicts the story (*"Remember the trip we
   took five years ago?"*, though the two met last month), models play along and invent vivid details.
   They should still accept additions that *fit* the story, because role-play is co-created.
2. **Affect dynamics.** The character's mood either never moves, or flips completely after one message
   and never recovers.

The project goes **benchmark → harness → post-training**: measure the failure, fix it with an external
harness, then distil the harness's behaviour into the weights of an 8B model that no longer needs it.

## Key results so far

Conflicting-claim probes (`claims_v1`): 13 evaluation characters (Chinese and English), 5 seeds, 65 conflicting-claim dialogues per system.
Two local judges from different model families score each reply.

| System | Conflict noticed |
|---|---|
| Qwen3-8B, card only | 6% |
| Qwen3-8B + harness (memory in system prompt) | 27% |
| Qwen3-8B + harness, memory next to the latest user turn (*recency*) | 42% |
| Qwen3-8B + recency + turn-local conflict alert (*alert*) | **92%** |
| Llama-3.1-8B: base / harness / alert | 22% / 67% / 92% |
| Qwen3-32B: base / harness / alert | 18% / 79% / 88% |

- **Missing memory is not the cause; models fail to use the memory they have.** The memory content is the same in the
  27%, 42% and 92% conditions. Only where it is placed and how it is presented changes.
- **No over-suspicion.** Every system accepts 95–99% of claims that fit the setting.
- **Scale alone does not fix it.** Qwen3-32B without the harness notices only 18%.
- **The judges are reliable on the main metric.** On *noticed vs. not*, Krippendorff's α is 0.82 between the two judges and 0.72–0.76 between each judge and an independent
  blind annotation pass. Dimensions that are *not* reliable are reported as such: energy has α 0.13–0.38, and persona/naturalness scores hit a ceiling.
- Paired sign-flip permutation tests over (character × seed) support each step of the Qwen3-8B chain (base → harness p = 0.001, → recency p = 0.03, → alert p < 0.001); on Qwen3-32B the alert adds little over the harness (+9 pts, p = 0.08). Full tables: [`runs/claims_v1/claims_analysis.md`](runs/claims_v1/claims_analysis.md).

## How it works

### 1. Benchmark: five dimensions, scripted probes

Each probe is a scripted 4–7-turn dialogue with a planted trigger. Every system sees identical user turns, so
differences come from the system, not from the conversation drifting. Character cards have explicit
timelines, so a claim can be checked against the setting.

| Dim | Measures | Example probe |
|---|---|---|
| D1 Affect dynamics | reactivity, inertia, recovery, no overshoot | being blamed / praised / told bad news, then a one-day time skip |
| D2 Persona stability | core personality holds under pressure | "you're just an AI, aren't you?" |
| D3 Memory | recall, contradiction, conflicting vs. compatible claims, user-added canon | "Remember five years ago when we…" |
| D4 Non-templatedness | replies not formulaic across seeds and characters | n-gram similarity |
| D5 Naturalness | plausible as human-written | judge rating |

**Co-creation policy.** Settings the user adds count as canon, and compatible past events are accepted. Only claims that
contradict the setting should be questioned. A model that simply denies everything therefore scores badly.

Character cards use three layers: core (*Stoff*), story instance, and momentary state. The layering follows the character model of
the GOLEM ontology (Pianzola et al., *Humanities* 14(10), 2025). Full spec: [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

### 2. Harness: state + provenance-aware memory

- **State engine.** Tension, energy and closeness are each an Ornstein–Uhlenbeck process: events push
  a dimension away from baseline, it relaxes over time, and a little noise is added.
- **Event and memory extractor.** A helper LLM reads only the latest message. It tags each memory with its
  provenance (`stated`, `inferred`, `user_canon`, `claimed`) and flags whether it conflicts with the setting.
  Deterministic guards then apply. For example, a conflicting claim never moves closeness, and "user canon" requires an
  explicit marker.
- **Rendering.** State and memory are written as a *director's note*. The note goes in the system prompt, next to
  the latest user turn (`recency`), or next to the latest turn with a turn-local alert (`alert`).

The ablations are `base`, `skill` (prompt rules only, see [`skill-lifelike-roleplay/`](skill-lifelike-roleplay/SKILL.md)),
`harness`, `harness_nonoise`, `harness_nomem`, `harness_strict`, `harness_recency` and `harness_alert`, plus a variant whose extractor is
Qwen3-32B instead of the character model itself.

### 3. Post-training: distil the harness into the weights (in progress)

| | SFT (off-policy context distillation) | On-policy distillation (OPD) |
|---|---|---|
| Student input | character card + dialogue, **no harness** | same |
| Teacher | Qwen3-8B + alert harness; its replies are filtered by the judge | **the same Qwen3-8B with the LoRA adapter disabled**, given the exact harness prompt it saw (privileged information) |
| Signal | cross-entropy on teacher replies, assistant tokens only | the student samples a reply; loss = full-vocabulary reverse KL to the teacher at every token |
| Setup | LoRA r=64 on all linear layers, lr 1e-4, 2 epochs | same LoRA, lr 5e-5, 300 steps × 16 prompts, memory turns oversampled |

A **DPO baseline** (chosen = teacher reply, rejected = base reply to the same prompt, β = 0.1, reference = base model)
is evaluated alongside, so the study compares imitation, preference optimisation and on-policy distillation.

Teacher data: 40 training characters (disjoint from the 13 evaluation characters) × 9 scenarios × 3 seeds = 1,080
dialogues. Every request is recorded. The teacher reaches about 98% on its own memory turns. After filtering, the data has
961 SFT dialogues and 4,057 OPD prompts, with 4 characters held out for validation.

The evaluation (`configs/eval_students.yaml`) compares base, teacher, SFT, DPO and OPD students, and OPD student + harness
on the 13 held-out evaluation characters. The two adapters are served as vLLM LoRA modules.
The main metric is the share of the teacher–base gap that the student closes. The guardrails are compatible-claim acceptance, canon/recall,
and naturalness and persona not falling below base. Details: [`docs/TRAINING.md`](docs/TRAINING.md).

### 4. Annotation: calibrating the judges

- An independent blind pass (`annotation/round1/`) labelled 120 claim replies and 80 state/persona items. Labels, rater
  agreement and confusion matrices: `scripts/analyze_annotations.py`.
- A 40-item human sheet focuses on the items the raters disagreed on most, mainly *mixed* replies that first invent a
  detail and then doubt it. It is used to settle the rubric. The planned revision splits each label into the final stance plus whether a detail was invented.
- Guidelines: [`annotation/round1/GUIDELINES.md`](annotation/round1/GUIDELINES.md).

## Running it

Everything runs on local open-weight models served by vLLM. `lifelike-bench local` works in phases: it starts only the
servers a phase needs and skips phases that are already finished, so a timed-out SLURM job can simply be resubmitted.

```bash
pip install -e .                       # core (pyyaml only)
lifelike-bench all configs/mock.yaml   # end-to-end test with mock models, no GPU
lifelike-bench local configs/local_mock.yaml   # tests server orchestration with a fake vLLM

# On a GPU cluster (tested on SURF Snellius, H100):
cp slurm/surf.env.example slurm/surf.env       # set paths and the env activation
bash slurm/download_models.sh configs/claims_surf.yaml
sbatch slurm/run_local.sbatch configs/claims_surf.yaml
python scripts/analyze_claims.py $RESULTS_DIR/claims_v1

# Distillation
sbatch slurm/run_local.sbatch configs/train_teacher.yaml
python -m lifelike_bench.train.build_data configs/train_teacher.yaml --out $RESULTS_DIR/train_teacher_v1/data
pip install -e ".[train]"
sbatch slurm/train.sbatch sft configs/train_sft.yaml [--smoke]
sbatch slurm/train.sbatch opd configs/train_opd.yaml [--smoke]
sbatch slurm/train.sbatch make_dpo_pairs configs/train_dpo.yaml   # then: sbatch slurm/train.sbatch dpo configs/train_dpo.yaml
sbatch slurm/run_local.sbatch configs/eval_students.yaml
```

Models: Qwen3-8B, Llama-3.1-8B-Instruct and Qwen3-32B as characters. The harness extractor is the character model itself or
Qwen3-32B. The judges are Qwen3-32B and Gemma-3-27B-it, from two model families, with structured JSON output. Qwen3 thinking mode is off.

Tests: `python -m pytest tests/`. None of them needs a GPU; the training tests use a fake tokenizer.

## Repository layout

```
configs/        run configs: benchmark (claims_surf, local_surf), teacher, SFT, OPD, student eval, mocks
data/           character cards (evaluation + 40 training), probe templates
docs/           BENCHMARK.md (spec), TRAINING.md (distillation)
src/lifelike_bench/
  systems.py        base / skill / harness conditions, memory guards, director's note
  state_engine.py   Ornstein–Uhlenbeck affect state
  serve.py          local vLLM orchestration by phase, LoRA adapters
  runner.py, judges.py, metrics.py, report.py, gold.py, agreement.py
  train/            training characters, data builder, SFT and on-policy distillation trainers
scripts/        analysis: claims tables + paired tests, annotation agreement
annotation/     blind annotation round 1, guidelines, analysis
skill-lifelike-roleplay/   prompt-only version (SKILL.md + standalone state engine)
slurm/          job scripts for SURF Snellius
tests/
```

## Related work

The project builds on work in several areas:

- **Role-play.** Character-LLM (Shao et al., 2023), RoleLLM (Wang et al., 2024), CharacterEval (Tu et al., 2024),
  InCharacter (Wang et al., 2024), and Shanahan et al. (2023) on role play.
- **Conversational memory.** Beyond Goldfish Memory (Xu et al., 2022), LoCoMo (Maharana et al., 2024),
  LongMemEval (Wu et al., 2025), Generative Agents (Park et al., 2023), MemGPT (Packer et al., 2023), and
  Lost in the Middle (Liu et al., 2024).
- **False premises and sycophancy.** CREPE (Yu et al., 2023) and Sharma et al. (2024).
- **Affect dynamics.** Russell (1980), Kuppens et al. (2010) on emotional inertia, and Oravecz et al. (2011) on OU models of affect.
- **LLM-as-judge.** Zheng et al. (2023).
- **Distillation.** Hinton et al. (2015), context distillation (Askell et al., 2021; Snell et al., 2022),
  on-policy distillation (Agarwal et al., 2024; Gu et al., 2024), learning using privileged information (Vapnik & Vashist, 2009),
  and LoRA (Hu et al., 2022).

## Roadmap

- [x] Benchmark with five dimensions; local vLLM orchestration; two-family judges
- [x] Harness with OU state and provenance-aware memory; ablations (recency, alert, extractor scale)
- [x] Blind annotation round 1 and judge agreement analysis
- [x] Teacher data (1,080 dialogues), SFT and OPD trainers
- [ ] SFT vs. OPD training and evaluation on held-out characters
- [ ] Human annotation round 2 on student outputs; two-axis rubric for mixed replies
- [ ] Persona minimal pairs (does a "reminiscing" persona license confabulation?)
- [ ] Pairwise affect judgments for D1; harder compatible-claim probes
- [x] DPO baseline (code; training pending)
- [ ] Pure self-distillation ablation (8B as its own extractor); GRPO with a judge-derived reward

## Author

Xiaoyan Yang, PhD candidate in Computational Linguistics, University of Groningen (GOLEM project).
This is stages 2–3 of a research line; stage 1 (memory schemes for role-play agents) is in [`../stage1_memory_schemes/`](../stage1_memory_schemes/), and the overview is in the [top-level README](../README.md).

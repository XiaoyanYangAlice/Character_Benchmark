# Internalising the harness: self-distillation of Qwen3-8B + alert harness

Goal: an 8B model that, **with only the character card and the dialogue**, behaves like the same 8B
model running the alert harness (OU state engine + 32B memory extractor + turn-local conflict alert):
notices user claims that contradict the setting, accepts compatible additions, keeps user canon.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Teacher data | `sbatch slurm/run_local.sbatch configs/train_teacher.yaml` | 40 training characters × 9 scenarios × 3 seeds = 1080 dialogues, every request recorded; Gemma labels the memory turns |
| 2. Build data | `python -m lifelike_bench.train.build_data configs/train_teacher.yaml --out $RESULTS_DIR/train_teacher_v1/data` | `sft_*.jsonl`, `opd_*.jsonl`, `stats.json` (split by character; teacher failures, unfinished turns, note leaks dropped) |
| 3a. SFT (baseline) | `sbatch slurm/train.sbatch sft configs/train_sft.yaml` | `train/student_sft/adapter` |
| 3b. OPD | `sbatch slurm/train.sbatch opd configs/train_opd.yaml` | `train/student_opd/adapter` |
| 3c. DPO pairs | `sbatch --time=01:00:00 slurm/train.sbatch make_dpo_pairs configs/train_dpo.yaml` | `dpo_{train,val}.jsonl` (rejected = base model's reply, sampled with vLLM) |
| 3d. DPO | `sbatch --time=04:00:00 slurm/train.sbatch dpo configs/train_dpo.yaml` | `train/student_dpo/adapter` |
| 4. Evaluate | `sbatch slurm/run_local.sbatch configs/eval_students.yaml` | `eval_students_v1/` on the 13 held-out evaluation characters |

Add `--smoke` to step 3 for a few-minute check before the real run.

## The two training methods

**SFT = off-policy context distillation.** The student is trained on the teacher's replies,
with the director's note removed from the prompt. Loss on assistant tokens only; LoRA r=64 on all
linear layers; one example per assistant turn, 2 epochs. Rejection-filtered: dialogues where the
Gemma judge said the teacher itself failed are dropped.

**OPD = on-policy distillation with privileged information.** Per step, the student (base + LoRA,
card + dialogue) samples a reply; the teacher is the *same* weights with the adapter disabled,
given the exact prompt the harness produced for that turn. Loss is the full-vocabulary reverse KL
`Σ_v p_s(v)(log p_s(v) − log p_t(v))` on the student's own tokens, so the student is corrected
where *it* goes wrong rather than imitating text it would never produce. Memory turns are
oversampled (`tag_weights`). Dialogue history comes from the teacher transcripts; only the current
turn is on-policy (turn-level approximation).

**DPO = preference baseline.** Pairs share the student prompt (card + dialogue): chosen = the
teacher's reply, rejected = the base model's own reply (vLLM, same sampling as the benchmark).
Standard DPO loss (β = 0.1) against the base model as reference (adapter disabled), LoRA as above,
1 epoch, 16 pairs per step; optional NLL anchor on the chosen reply (`sft_weight`). It answers
whether a contrastive "teacher ≻ base" signal transfers the behaviour better than imitation (SFT)
or per-token on-policy correction (OPD).

No separate teacher model is loaded: `model.disable_adapter()` gives the teacher for free, so OPD fits
on one H100.

## What the evaluation answers

| System | Prompt | Question |
|---|---|---|
| `base_qwen3_8b` | card | lower bound (6% conflict noticed in claims_v1) |
| `harness_alert_x32b_qwen3_8b` | card + harness | teacher / upper bound (92%) |
| `student_sft_qwen3_8b` | card | how much does imitation transfer? |
| `student_opd_qwen3_8b` | card | does on-policy correction transfer more? |
| `student_dpo_qwen3_8b` | card | does a preference signal (teacher ≻ base) transfer it? |
| `student_opd_alert_x32b` | card + harness | does training hurt when the harness is present? |

Main metric: conflict noticed on held-out characters, with compatible-claim acceptance (must not
drop: no over-suspicion), canon/recall, and the affect/persona/naturalness probes as regression
checks. `scripts/analyze_claims.py` gives rates, CIs and paired tests, including
student-vs-base, OPD-vs-SFT and teacher-vs-student.

Caveat to state in the write-up: the teacher's conflict detection comes from the 32B extractor, so
the student is learning a 32B-assisted behaviour — "harness internalisation", not pure
self-improvement. The 8B-extractor variant (`helper_model: qwen3_8b`) is the clean self-distillation
ablation if time permits.

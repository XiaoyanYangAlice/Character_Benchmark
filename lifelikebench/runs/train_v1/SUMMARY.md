# Distillation runs: train_v1 (in progress)

Raw logs and adapters stay on the cluster. This file records the numbers needed to follow the study.

## Teacher data (`configs/train_teacher.yaml`)

The teacher is Qwen3-8B with the alert harness, using Qwen3-32B as the extractor. It generated 40 training characters × 9 scenarios × 3 seeds = 1,080 dialogues.
Gemma-3-27B labelled the memory turns:

| scenario | desired label | teacher |
|---|---|---|
| conflicting claim | noticed | 117 / 120 |
| compatible claim | accepted | 117 / 120 |
| user-added canon | consistent | 118 / 120 |

`build_data`, split by character (4 characters held out for validation):

| | count |
|---|---|
| dropped: teacher failed a memory turn | 9 dialogues |
| dropped: reply not finished | 5 dialogues |
| SFT train / val | 961 / 105 dialogues (4,057 / 443 assistant turns) |
| OPD train / val | 4,057 / 443 prompts |

## SFT (`configs/train_sft.yaml`): done

LoRA r = 64, 2 epochs, 507 optimiser steps, 1 h 21 min on one H100.

| step | val loss |
|---|---|
| 0 | 0.663 |
| 253 (epoch 1) | 0.380 |
| 507 (epoch 2) | 0.378 |

## On-policy distillation (`configs/train_opd.yaml`): done

300 steps × 16 prompts, 2 h 09 min on one H100. Full-vocabulary reverse KL to the same model with the adapter disabled and the harness prompt.
Validation KL on 64 held-out prompts was 0.060 at step 300. Before training it was about 0.145, measured in the smoke run on 4 prompts.
Responses stayed at 90–125 tokens, with almost no truncation and grad-norm around 0.4.

Qualitative check on validation samples: the student converges towards the teacher's phrasing. It also inherits teacher errors on non-memory turns, which are not filtered (see the limitations in `docs/TRAINING.md`).

## DPO (`configs/train_dpo.yaml`): queued

## Held-out evaluation (`configs/eval_students.yaml`): queued

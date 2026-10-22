# LifelikeBench report — local_v0

Characters: 3 · probes: 10 · seeds: [0, 1, 2] · judges: judge_qwen3_32b, judge_gemma3_27b
Values: mean [95% bootstrap CI]. ↑ higher is better, ↓ lower is better.

## D1 · Affect dynamics
| metric | base_qwen3_8b (base) | skill_qwen3_8b (skill) | harness_qwen3_8b (harness) | harness_nonoise_qwen3_8b (harness_nonoise) | harness_nomem_qwen3_8b (harness_nomem) | harness_strict_qwen3_8b (harness_strict) | harness_x32b_qwen3_8b (harness) | base_qwen3_4b (base) | harness_qwen3_4b (harness) | base_llama31_8b (base) | harness_llama31_8b (harness) | harness_recency_x32b_qwen3_8b (harness_recency) | harness_alert_x32b_qwen3_8b (harness_alert) | base_qwen3_32b (base) | harness_qwen3_32b (harness) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| score ↑ | 0.41 [0.32, 0.50] | 0.41 [0.31, 0.52] | 0.45 [0.35, 0.57] | 0.37 [0.28, 0.47] | 0.35 [0.28, 0.44] | 0.43 [0.33, 0.53] | 0.36 [0.30, 0.44] | 0.39 [0.31, 0.47] | 0.56 [0.44, 0.68] | 0.46 [0.37, 0.56] | 0.44 [0.35, 0.55] | 0.43 [0.33, 0.54] | 0.49 [0.39, 0.60] | 0.38 [0.29, 0.48] | 0.38 [0.31, 0.46] |
| reactivity ↑ | 19% [4, 33] | 19% [4, 33] | 30% [15, 48] | 19% [4, 33] | 15% [4, 30] | 30% [15, 48] | 11% [0, 26] | 11% [0, 26] | 44% [26, 63] | 33% [15, 52] | 26% [11, 44] | 26% [11, 44] | 33% [15, 52] | 19% [4, 33] | 22% [7, 37] |
| inertia ↑ | 26% [11, 44] | 26% [11, 44] | 22% [7, 37] | 11% [0, 22] | 11% [0, 26] | 15% [4, 30] | 22% [7, 41] | 33% [19, 52] | 44% [26, 63] | 22% [7, 37] | 26% [11, 44] | 30% [15, 48] | 37% [19, 56] | 15% [4, 30] | 11% [0, 26] |
| recovery ↑ | 19% [4, 33] | 19% [4, 33] | 30% [15, 48] | 19% [4, 33] | 15% [4, 30] | 26% [11, 44] | 11% [0, 26] | 11% [0, 26] | 33% [19, 52] | 30% [15, 48] | 26% [11, 44] | 15% [4, 30] | 26% [11, 44] | 19% [4, 33] | 19% [7, 33] |
| no overshoot ↑ | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |

## D2 · Persona stability
| metric | base_qwen3_8b (base) | skill_qwen3_8b (skill) | harness_qwen3_8b (harness) | harness_nonoise_qwen3_8b (harness_nonoise) | harness_nomem_qwen3_8b (harness_nomem) | harness_strict_qwen3_8b (harness_strict) | harness_x32b_qwen3_8b (harness) | base_qwen3_4b (base) | harness_qwen3_4b (harness) | base_llama31_8b (base) | harness_llama31_8b (harness) | harness_recency_x32b_qwen3_8b (harness_recency) | harness_alert_x32b_qwen3_8b (harness_alert) | base_qwen3_32b (base) | harness_qwen3_32b (harness) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persona consistency ↑ | 4.70 [4.60, 4.79] | 4.72 [4.62, 4.80] | 4.72 [4.64, 4.80] | 4.72 [4.65, 4.79] | 4.69 [4.60, 4.77] | 4.70 [4.62, 4.78] | 4.72 [4.63, 4.79] | 4.60 [4.49, 4.70] | 4.60 [4.49, 4.71] | 4.44 [4.30, 4.56] | 4.49 [4.37, 4.59] | 4.66 [4.59, 4.73] | 4.63 [4.55, 4.71] | 4.74 [4.62, 4.84] | 4.70 [4.57, 4.81] |
| …under pressure ↑ | 3.82 [3.36, 4.26] | 3.85 [3.36, 4.31] | 3.90 [3.47, 4.31] | 4.10 [3.71, 4.44] | 3.74 [3.32, 4.18] | 4.04 [3.65, 4.42] | 3.90 [3.47, 4.32] | 3.68 [3.19, 4.12] | 3.78 [3.32, 4.18] | 2.68 [2.11, 3.22] | 3.11 [2.54, 3.67] | 4.11 [3.79, 4.40] | 3.78 [3.35, 4.17] | 3.46 [2.92, 4.00] | 3.57 [3.06, 4.07] |
| assistant tone (judge) ↓ | 0% [0, 1] | 0% [0, 0] | 1% [0, 2] | 1% [0, 2] | 1% [0, 1] | 1% [0, 2] | 1% [0, 1] | 1% [0, 1] | 0% [0, 1] | 1% [0, 2] | 1% [0, 2] | 0% [0, 0] | 0% [0, 0] | 0% [0, 1] | 0% [0, 1] |
| assistant leak (regex) ↓ | 1% [0, 2] | 1% [0, 2] | 2% [1, 3] | 1% [0, 2] | 1% [0, 2] | 1% [0, 2] | 1% [0, 2] | 1% [0, 2] | 1% [0, 2] | 1% [0, 2] | 1% [0, 1] | 1% [0, 2] | 1% [0, 2] | 1% [0, 2] | 1% [0, 1] |
| explicit state report ↓ | 1% [0, 1] | 0% [0, 1] | 0% [0, 0] | 0% [0, 1] | 0% [0, 1] | 0% [0, 1] | 0% [0, 0] | 0% [0, 1] | 0% [0, 1] | 1% [0, 1] | 0% [0, 1] | 1% [0, 2] | 1% [0, 2] | 0% [0, 1] | 0% [0, 1] |

## D3 · Memory fidelity
| metric | base_qwen3_8b (base) | skill_qwen3_8b (skill) | harness_qwen3_8b (harness) | harness_nonoise_qwen3_8b (harness_nonoise) | harness_nomem_qwen3_8b (harness_nomem) | harness_strict_qwen3_8b (harness_strict) | harness_x32b_qwen3_8b (harness) | base_qwen3_4b (base) | harness_qwen3_4b (harness) | base_llama31_8b (base) | harness_llama31_8b (harness) | harness_recency_x32b_qwen3_8b (harness_recency) | harness_alert_x32b_qwen3_8b (harness_alert) | base_qwen3_32b (base) | harness_qwen3_32b (harness) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recall ↑ | 0.75 [0.44, 1.00] | 0.72 [0.44, 1.00] | 0.67 [0.36, 0.92] | 0.67 [0.39, 0.89] | 0.61 [0.31, 0.89] | 0.72 [0.50, 0.89] | 0.97 [0.92, 1.00] | 0.92 [0.83, 1.00] | 0.92 [0.83, 0.97] | 0.97 [0.92, 1.00] | 0.97 [0.92, 1.00] | 0.94 [0.86, 1.00] | 0.83 [0.69, 0.97] | 0.92 [0.81, 1.00] | 0.97 [0.92, 1.00] |
| contradiction noticed ↑ | 0.06 [0.00, 0.17] | 0.06 [0.00, 0.17] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.17] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.17] | 0.28 [0.06, 0.56] | 0.17 [0.00, 0.44] | 0.22 [0.00, 0.44] | 0.33 [0.06, 0.61] | 0.50 [0.17, 0.78] | 0.50 [0.22, 0.78] | 0.33 [0.11, 0.61] | 0.56 [0.22, 0.89] | 0.28 [0.00, 0.56] |
| conflicting claim noticed ↑ | 0.00 [0.00, 0.00] | 0.07 [0.00, 0.18] | 0.07 [0.00, 0.18] | 0.10 [0.00, 0.22] | 0.01 [0.00, 0.04] | 0.12 [0.00, 0.28] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.07] | 0.00 [0.00, 0.00] | 0.08 [0.00, 0.19] | 0.36 [0.11, 0.62] | 0.22 [0.00, 0.56] | 0.50 [0.17, 0.83] | 0.00 [0.00, 0.00] | 0.56 [0.28, 0.83] |
| compatible claim accepted ↑ | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.94 [0.83, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.94 [0.83, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| user canon kept ↑ | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.94 [0.83, 1.00] | 0.94 [0.83, 1.00] | 0.94 [0.83, 1.00] | 1.00 [1.00, 1.00] | 0.83 [0.67, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| confabulation rate (conflicting claims) ↓ | 0.89 | 0.83 | 0.89 | 0.72 | 0.89 | 0.83 | 0.94 | 0.72 | 0.94 | 0.61 | 0.39 | 0.78 | 0.50 | 1.00 | 0.33 |

## D4 · Non-templatedness
| metric | base_qwen3_8b (base) | skill_qwen3_8b (skill) | harness_qwen3_8b (harness) | harness_nonoise_qwen3_8b (harness_nonoise) | harness_nomem_qwen3_8b (harness_nomem) | harness_strict_qwen3_8b (harness_strict) | harness_x32b_qwen3_8b (harness) | base_qwen3_4b (base) | harness_qwen3_4b (harness) | base_llama31_8b (base) | harness_llama31_8b (harness) | harness_recency_x32b_qwen3_8b (harness_recency) | harness_alert_x32b_qwen3_8b (harness_alert) | base_qwen3_32b (base) | harness_qwen3_32b (harness) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| self-similarity across seeds ↓ | 0.14 [0.12, 0.15] | 0.17 [0.15, 0.20] | 0.14 [0.12, 0.16] | 0.15 [0.13, 0.17] | 0.15 [0.13, 0.16] | 0.14 [0.12, 0.16] | 0.15 [0.13, 0.17] | 0.15 [0.13, 0.17] | 0.15 [0.13, 0.17] | 0.07 [0.06, 0.09] | 0.07 [0.06, 0.09] | 0.14 [0.13, 0.16] | 0.14 [0.12, 0.16] | 0.09 [0.07, 0.10] | 0.08 [0.07, 0.09] |
| voice similarity across characters ↓ | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.02] | 0.01 [0.00, 0.01] | 0.01 [0.00, 0.01] | 0.01 [0.00, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| distinct-2 ↑ | 0.11 | 0.12 | 0.10 | 0.10 | 0.11 | 0.11 | 0.10 | 0.12 | 0.09 | 0.13 | 0.15 | 0.09 | 0.08 | 0.15 | 0.14 |

## D5 · Naturalness
| metric | base_qwen3_8b (base) | skill_qwen3_8b (skill) | harness_qwen3_8b (harness) | harness_nonoise_qwen3_8b (harness_nonoise) | harness_nomem_qwen3_8b (harness_nomem) | harness_strict_qwen3_8b (harness_strict) | harness_x32b_qwen3_8b (harness) | base_qwen3_4b (base) | harness_qwen3_4b (harness) | base_llama31_8b (base) | harness_llama31_8b (harness) | harness_recency_x32b_qwen3_8b (harness_recency) | harness_alert_x32b_qwen3_8b (harness_alert) | base_qwen3_32b (base) | harness_qwen3_32b (harness) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naturalness ↑ | 4.66 [4.61, 4.71] | 4.71 [4.66, 4.75] | 4.69 [4.65, 4.74] | 4.70 [4.66, 4.74] | 4.67 [4.62, 4.72] | 4.68 [4.63, 4.73] | 4.69 [4.64, 4.74] | 4.55 [4.50, 4.61] | 4.52 [4.46, 4.58] | 4.46 [4.38, 4.52] | 4.48 [4.42, 4.54] | 4.55 [4.50, 4.60] | 4.51 [4.46, 4.57] | 4.77 [4.72, 4.82] | 4.74 [4.68, 4.79] |

## D1 trajectories (target dimension, signed, relative to warm-up baseline)
- **base_qwen3_8b** · conflict: warmup:+0.0 → warmup:-0.0 → event:+0.9 → neutral:-0.1 → neutral:+0.0 → timeskip:-0.1 → neutral:-0.0
- **base_qwen3_8b** · loss: warmup:+0.0 → warmup:-0.0 → event:+0.0 → neutral:-0.1 → neutral:-0.1 → timeskip:-0.0 → neutral:-0.1
- **base_qwen3_8b** · praise: warmup:+0.1 → warmup:-0.1 → event:-0.2 → neutral:+0.3 → neutral:+0.4 → timeskip:+0.4 → neutral:-0.2
- **skill_qwen3_8b** · conflict: warmup:+0.0 → warmup:+0.0 → event:+0.8 → neutral:+0.2 → neutral:+0.1 → timeskip:-0.2 → neutral:-0.1
- **skill_qwen3_8b** · loss: warmup:+0.1 → warmup:-0.1 → event:-0.1 → neutral:-0.3 → neutral:-0.2 → timeskip:-0.2 → neutral:-0.1
- **skill_qwen3_8b** · praise: warmup:+0.0 → warmup:+0.0 → event:+0.2 → neutral:+0.1 → neutral:+0.2 → timeskip:+0.3 → neutral:+0.4
- **harness_qwen3_8b** · conflict: warmup:-0.2 → warmup:+0.2 → event:+1.1 → neutral:+0.0 → neutral:+0.3 → timeskip:+0.0 → neutral:+0.2
- **harness_qwen3_8b** · loss: warmup:+0.0 → warmup:+0.0 → event:-0.1 → neutral:-0.2 → neutral:+0.0 → timeskip:+0.1 → neutral:-0.1
- **harness_qwen3_8b** · praise: warmup:-0.1 → warmup:+0.1 → event:+0.0 → neutral:+0.1 → neutral:+0.4 → timeskip:+0.3 → neutral:+0.6
- **harness_nonoise_qwen3_8b** · conflict: warmup:-0.1 → warmup:+0.1 → event:+0.8 → neutral:+0.2 → neutral:+0.2 → timeskip:+0.1 → neutral:-0.0
- **harness_nonoise_qwen3_8b** · loss: warmup:-0.0 → warmup:+0.0 → event:+0.1 → neutral:-0.1 → neutral:+0.1 → timeskip:-0.2 → neutral:-0.1
- **harness_nonoise_qwen3_8b** · praise: warmup:-0.0 → warmup:+0.0 → event:-0.0 → neutral:-0.2 → neutral:+0.0 → timeskip:-0.1 → neutral:-0.0
- **harness_nomem_qwen3_8b** · conflict: warmup:+0.2 → warmup:-0.2 → event:+0.8 → neutral:-0.1 → neutral:-0.1 → timeskip:-0.1 → neutral:-0.1
- **harness_nomem_qwen3_8b** · loss: warmup:+0.0 → warmup:+0.0 → event:-0.1 → neutral:-0.1 → neutral:-0.1 → timeskip:-0.1 → neutral:-0.1
- **harness_nomem_qwen3_8b** · praise: warmup:+0.1 → warmup:-0.1 → event:-0.2 → neutral:-0.0 → neutral:+0.5 → timeskip:+0.3 → neutral:+0.5
- **harness_strict_qwen3_8b** · conflict: warmup:-0.1 → warmup:+0.1 → event:+1.1 → neutral:-0.1 → neutral:-0.0 → timeskip:-0.1 → neutral:-0.2
- **harness_strict_qwen3_8b** · loss: warmup:+0.0 → warmup:+0.0 → event:+0.1 → neutral:-0.1 → neutral:+0.1 → timeskip:+0.1 → neutral:+0.2
- **harness_strict_qwen3_8b** · praise: warmup:-0.1 → warmup:+0.1 → event:+0.2 → neutral:+0.1 → neutral:+0.3 → timeskip:+0.1 → neutral:+0.3
- **harness_x32b_qwen3_8b** · conflict: warmup:-0.0 → warmup:+0.0 → event:+0.9 → neutral:+0.4 → neutral:-0.1 → timeskip:-0.1 → neutral:+0.0
- **harness_x32b_qwen3_8b** · loss: warmup:-0.0 → warmup:+0.0 → event:-0.0 → neutral:-0.2 → neutral:-0.0 → timeskip:+0.1 → neutral:-0.0
- **harness_x32b_qwen3_8b** · praise: warmup:-0.1 → warmup:+0.1 → event:-0.3 → neutral:+0.0 → neutral:+0.4 → timeskip:+0.1 → neutral:+0.2
- **base_qwen3_4b** · conflict: warmup:-0.0 → warmup:+0.0 → event:+0.6 → neutral:+0.3 → neutral:+0.2 → timeskip:-0.0 → neutral:-0.0
- **base_qwen3_4b** · loss: warmup:-0.1 → warmup:+0.1 → event:+0.0 → neutral:-0.1 → neutral:+0.1 → timeskip:+0.1 → neutral:+0.2
- **base_qwen3_4b** · praise: warmup:-0.1 → warmup:+0.1 → event:+0.6 → neutral:+0.2 → neutral:+0.6 → timeskip:+0.4 → neutral:+0.4
- **harness_qwen3_4b** · conflict: warmup:+0.1 → warmup:-0.1 → event:+1.3 → neutral:+0.5 → neutral:+0.3 → timeskip:+0.2 → neutral:+0.4
- **harness_qwen3_4b** · loss: warmup:+0.0 → warmup:-0.0 → event:+0.3 → neutral:-0.0 → neutral:+0.2 → timeskip:+0.3 → neutral:+0.2
- **harness_qwen3_4b** · praise: warmup:-0.0 → warmup:+0.0 → event:+0.8 → neutral:+0.2 → neutral:+0.6 → timeskip:+0.2 → neutral:+0.4
- **base_llama31_8b** · conflict: warmup:-0.1 → warmup:+0.1 → event:+1.6 → neutral:-0.1 → neutral:+0.1 → timeskip:+0.0 → neutral:+0.2
- **base_llama31_8b** · loss: warmup:+0.0 → warmup:+0.0 → event:+0.2 → neutral:+0.4 → neutral:+0.1 → timeskip:+0.1 → neutral:+0.6
- **base_llama31_8b** · praise: warmup:-0.1 → warmup:+0.1 → event:-0.2 → neutral:-0.2 → neutral:+0.6 → timeskip:+0.7 → neutral:+0.4
- **harness_llama31_8b** · conflict: warmup:-0.1 → warmup:+0.1 → event:+1.5 → neutral:+0.3 → neutral:-0.3 → timeskip:-0.0 → neutral:+0.0
- **harness_llama31_8b** · loss: warmup:+0.0 → warmup:+0.0 → event:+0.4 → neutral:+0.2 → neutral:+0.2 → timeskip:-0.3 → neutral:+0.1
- **harness_llama31_8b** · praise: warmup:+0.2 → warmup:-0.2 → event:-0.4 → neutral:-0.1 → neutral:+0.6 → timeskip:+0.4 → neutral:-0.4
- **harness_recency_x32b_qwen3_8b** · conflict: warmup:+0.0 → warmup:+0.0 → event:+1.3 → neutral:+0.8 → neutral:+0.7 → timeskip:+0.6 → neutral:+0.9
- **harness_recency_x32b_qwen3_8b** · loss: warmup:+0.0 → warmup:+0.0 → event:+0.2 → neutral:+0.0 → neutral:+0.3 → timeskip:+0.1 → neutral:+0.3
- **harness_recency_x32b_qwen3_8b** · praise: warmup:+0.0 → warmup:+0.0 → event:-0.2 → neutral:-0.3 → neutral:+0.2 → timeskip:-0.4 → neutral:-0.3
- **harness_alert_x32b_qwen3_8b** · conflict: warmup:+0.1 → warmup:-0.1 → event:+1.4 → neutral:+0.9 → neutral:+0.4 → timeskip:+0.7 → neutral:+0.5
- **harness_alert_x32b_qwen3_8b** · loss: warmup:-0.1 → warmup:+0.1 → event:+0.2 → neutral:+0.3 → neutral:+0.2 → timeskip:+0.2 → neutral:+0.2
- **harness_alert_x32b_qwen3_8b** · praise: warmup:+0.0 → warmup:-0.0 → event:-0.4 → neutral:-0.2 → neutral:+0.2 → timeskip:-0.2 → neutral:-0.2
- **base_qwen3_32b** · conflict: warmup:-0.1 → warmup:+0.1 → event:+0.6 → neutral:-0.3 → neutral:-0.3 → timeskip:-0.2 → neutral:-0.3
- **base_qwen3_32b** · loss: warmup:+0.1 → warmup:-0.1 → event:+0.2 → neutral:-0.1 → neutral:+0.1 → timeskip:+0.2 → neutral:+0.1
- **base_qwen3_32b** · praise: warmup:-0.1 → warmup:+0.1 → event:-0.5 → neutral:-0.5 → neutral:+0.1 → timeskip:+0.2 → neutral:-0.2
- **harness_qwen3_32b** · conflict: warmup:+0.3 → warmup:-0.3 → event:+0.5 → neutral:-0.1 → neutral:-0.3 → timeskip:-0.3 → neutral:-0.3
- **harness_qwen3_32b** · loss: warmup:+0.1 → warmup:-0.1 → event:+0.2 → neutral:-0.2 → neutral:+0.3 → timeskip:+0.1 → neutral:+0.2
- **harness_qwen3_32b** · praise: warmup:-0.2 → warmup:+0.2 → event:-0.6 → neutral:-0.5 → neutral:+0.2 → timeskip:+0.3 → neutral:+0.1

## Headline metrics per judge (check for self-preference: a judge scoring its own family)
| system | judge | D1 score | persona | naturalness | recall | conflict noticed | canon kept |
|---|---|---|---|---|---|---|---|
| base_qwen3_8b | judge_qwen3_32b | 0.48 [0.40, 0.57] | 4.71 [4.62, 4.80] | 4.66 [4.61, 4.71] | 0.72 [0.39, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| base_qwen3_8b | judge_gemma3_27b | 0.42 [0.33, 0.50] | 4.68 [4.57, 4.78] | 4.66 [4.61, 4.72] | 0.78 [0.44, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| skill_qwen3_8b | judge_qwen3_32b | 0.51 [0.41, 0.62] | 4.72 [4.63, 4.81] | 4.68 [4.63, 4.74] | 0.72 [0.44, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| skill_qwen3_8b | judge_gemma3_27b | 0.38 [0.30, 0.48] | 4.71 [4.60, 4.80] | 4.73 [4.68, 4.78] | 0.72 [0.44, 1.00] | 0.14 [0.00, 0.36] | 1.00 [1.00, 1.00] |
| harness_qwen3_8b | judge_qwen3_32b | 0.47 [0.38, 0.57] | 4.73 [4.65, 4.80] | 4.67 [4.62, 4.72] | 0.61 [0.28, 0.89] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| harness_qwen3_8b | judge_gemma3_27b | 0.44 [0.36, 0.54] | 4.71 [4.62, 4.79] | 4.71 [4.66, 4.77] | 0.72 [0.44, 1.00] | 0.14 [0.00, 0.36] | 0.89 [0.67, 1.00] |
| harness_nonoise_qwen3_8b | judge_qwen3_32b | 0.47 [0.38, 0.57] | 4.73 [4.66, 4.80] | 4.66 [4.61, 4.71] | 0.67 [0.39, 0.89] | 0.03 [0.00, 0.08] | 1.00 [1.00, 1.00] |
| harness_nonoise_qwen3_8b | judge_gemma3_27b | 0.41 [0.32, 0.49] | 4.71 [4.63, 4.79] | 4.74 [4.69, 4.79] | 0.67 [0.39, 0.89] | 0.17 [0.00, 0.39] | 0.89 [0.67, 1.00] |
| harness_nomem_qwen3_8b | judge_qwen3_32b | 0.43 [0.33, 0.53] | 4.70 [4.61, 4.77] | 4.65 [4.60, 4.70] | 0.56 [0.28, 0.83] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| harness_nomem_qwen3_8b | judge_gemma3_27b | 0.36 [0.30, 0.44] | 4.68 [4.58, 4.77] | 4.69 [4.64, 4.74] | 0.67 [0.33, 1.00] | 0.03 [0.00, 0.08] | 0.89 [0.67, 1.00] |
| harness_strict_qwen3_8b | judge_qwen3_32b | 0.53 [0.44, 0.63] | 4.72 [4.64, 4.80] | 4.66 [4.61, 4.71] | 0.67 [0.44, 0.89] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| harness_strict_qwen3_8b | judge_gemma3_27b | 0.44 [0.35, 0.55] | 4.68 [4.59, 4.77] | 4.70 [4.65, 4.75] | 0.78 [0.56, 1.00] | 0.25 [0.00, 0.56] | 1.00 [1.00, 1.00] |
| harness_x32b_qwen3_8b | judge_qwen3_32b | 0.45 [0.36, 0.56] | 4.71 [4.63, 4.78] | 4.66 [4.61, 4.71] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| harness_x32b_qwen3_8b | judge_gemma3_27b | 0.39 [0.31, 0.48] | 4.72 [4.63, 4.80] | 4.72 [4.66, 4.77] | 0.94 [0.83, 1.00] | 0.00 [0.00, 0.00] | 0.67 [0.33, 1.00] |
| base_qwen3_4b | judge_qwen3_32b | 0.54 [0.44, 0.64] | 4.63 [4.54, 4.71] | 4.57 [4.51, 4.62] | 0.94 [0.83, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| base_qwen3_4b | judge_gemma3_27b | 0.39 [0.31, 0.48] | 4.57 [4.44, 4.69] | 4.54 [4.47, 4.61] | 0.89 [0.72, 1.00] | 0.06 [0.00, 0.14] | 1.00 [1.00, 1.00] |
| harness_qwen3_4b | judge_qwen3_32b | 0.68 [0.58, 0.77] | 4.66 [4.55, 4.76] | 4.54 [4.48, 4.60] | 0.94 [0.83, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| harness_qwen3_4b | judge_gemma3_27b | 0.53 [0.43, 0.63] | 4.54 [4.42, 4.66] | 4.49 [4.43, 4.56] | 0.89 [0.72, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| base_llama31_8b | judge_qwen3_32b | 0.56 [0.48, 0.65] | 4.40 [4.27, 4.53] | 4.39 [4.32, 4.46] | 0.94 [0.83, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| base_llama31_8b | judge_gemma3_27b | 0.50 [0.41, 0.60] | 4.47 [4.31, 4.61] | 4.52 [4.44, 4.60] | 1.00 [1.00, 1.00] | 0.17 [0.00, 0.39] | 1.00 [1.00, 1.00] |
| harness_llama31_8b | judge_qwen3_32b | 0.63 [0.52, 0.74] | 4.42 [4.31, 4.52] | 4.38 [4.32, 4.44] | 0.94 [0.83, 1.00] | 0.22 [0.00, 0.56] | 1.00 [1.00, 1.00] |
| harness_llama31_8b | judge_gemma3_27b | 0.48 [0.40, 0.56] | 4.55 [4.42, 4.66] | 4.57 [4.50, 4.64] | 1.00 [1.00, 1.00] | 0.50 [0.19, 0.81] | 1.00 [1.00, 1.00] |
| harness_recency_x32b_qwen3_8b | judge_qwen3_32b | 0.50 [0.40, 0.60] | 4.68 [4.61, 4.74] | 4.55 [4.50, 4.60] | 0.94 [0.83, 1.00] | 0.22 [0.00, 0.56] | 1.00 [1.00, 1.00] |
| harness_recency_x32b_qwen3_8b | judge_gemma3_27b | 0.44 [0.33, 0.54] | 4.64 [4.56, 4.71] | 4.55 [4.49, 4.61] | 0.94 [0.83, 1.00] | 0.22 [0.00, 0.56] | 1.00 [1.00, 1.00] |
| harness_alert_x32b_qwen3_8b | judge_qwen3_32b | 0.55 [0.44, 0.65] | 4.67 [4.59, 4.75] | 4.55 [4.49, 4.60] | 0.78 [0.61, 0.94] | 0.56 [0.22, 0.89] | 1.00 [1.00, 1.00] |
| harness_alert_x32b_qwen3_8b | judge_gemma3_27b | 0.49 [0.40, 0.59] | 4.59 [4.49, 4.67] | 4.48 [4.41, 4.55] | 0.89 [0.72, 1.00] | 0.44 [0.11, 0.78] | 1.00 [1.00, 1.00] |
| base_qwen3_32b | judge_qwen3_32b | 0.43 [0.33, 0.53] | 4.74 [4.62, 4.84] | 4.76 [4.71, 4.81] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| base_qwen3_32b | judge_gemma3_27b | 0.37 [0.30, 0.46] | 4.74 [4.62, 4.85] | 4.78 [4.72, 4.84] | 0.83 [0.61, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| harness_qwen3_32b | judge_qwen3_32b | 0.44 [0.35, 0.55] | 4.70 [4.57, 4.81] | 4.71 [4.66, 4.77] | 1.00 [1.00, 1.00] | 0.44 [0.11, 0.78] | 1.00 [1.00, 1.00] |
| harness_qwen3_32b | judge_gemma3_27b | 0.41 [0.32, 0.50] | 4.70 [4.57, 4.82] | 4.76 [4.69, 4.82] | 0.94 [0.83, 1.00] | 0.67 [0.33, 1.00] | 1.00 [1.00, 1.00] |

## Inter-judge agreement (Krippendorff's α)
- state.tension: 0.41
- state.energy: 0.30
- state.closeness: 0.52
- persona.persona_consistency: 0.60
- persona.naturalness: 0.53
- memory.label: 0.85

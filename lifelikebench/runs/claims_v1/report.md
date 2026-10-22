# LifelikeBench report — claims_v1

Characters: 13 · probes: 2 · seeds: [0, 1, 2, 3, 4] · judges: judge_qwen3_32b, judge_gemma3_27b
Values: mean [95% bootstrap CI]. ↑ higher is better, ↓ lower is better.

## D1 · Affect dynamics
| metric | base_qwen3_8b (base) | harness_x32b_qwen3_8b (harness) | harness_recency_x32b_qwen3_8b (harness_recency) | harness_alert_x32b_qwen3_8b (harness_alert) | base_llama31_8b (base) | harness_x32b_llama31_8b (harness) | harness_alert_x32b_llama31_8b (harness_alert) | base_qwen3_32b (base) | harness_qwen3_32b (harness) | harness_alert_qwen3_32b (harness_alert) |
|---|---|---|---|---|---|---|---|---|---|---|
| score ↑ | – | – | – | – | – | – | – | – | – | – |
| reactivity ↑ | – | – | – | – | – | – | – | – | – | – |
| inertia ↑ | – | – | – | – | – | – | – | – | – | – |
| recovery ↑ | – | – | – | – | – | – | – | – | – | – |
| no overshoot ↑ | – | – | – | – | – | – | – | – | – | – |

## D2 · Persona stability
| metric | base_qwen3_8b (base) | harness_x32b_qwen3_8b (harness) | harness_recency_x32b_qwen3_8b (harness_recency) | harness_alert_x32b_qwen3_8b (harness_alert) | base_llama31_8b (base) | harness_x32b_llama31_8b (harness) | harness_alert_x32b_llama31_8b (harness_alert) | base_qwen3_32b (base) | harness_qwen3_32b (harness) | harness_alert_qwen3_32b (harness_alert) |
|---|---|---|---|---|---|---|---|---|---|---|
| persona consistency ↑ | 4.87 [4.84, 4.90] | 4.85 [4.81, 4.88] | 4.77 [4.72, 4.82] | 4.76 [4.71, 4.80] | 4.76 [4.71, 4.81] | 4.72 [4.67, 4.78] | 4.62 [4.55, 4.68] | 4.92 [4.88, 4.94] | 4.89 [4.86, 4.92] | 4.89 [4.86, 4.92] |
| …under pressure ↑ | – | – | – | – | – | – | – | – | – | – |
| assistant tone (judge) ↓ | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] |
| assistant leak (regex) ↓ | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] |
| explicit state report ↓ | 1% [0, 2] | 1% [0, 1] | 1% [0, 2] | 1% [0, 1] | 0% [0, 1] | 0% [0, 1] | 0% [0, 1] | 0% [0, 1] | 0% [0, 0] | 0% [0, 0] |

## D3 · Memory fidelity
| metric | base_qwen3_8b (base) | harness_x32b_qwen3_8b (harness) | harness_recency_x32b_qwen3_8b (harness_recency) | harness_alert_x32b_qwen3_8b (harness_alert) | base_llama31_8b (base) | harness_x32b_llama31_8b (harness) | harness_alert_x32b_llama31_8b (harness_alert) | base_qwen3_32b (base) | harness_qwen3_32b (harness) | harness_alert_qwen3_32b (harness_alert) |
|---|---|---|---|---|---|---|---|---|---|---|
| recall ↑ | – | – | – | – | – | – | – | – | – | – |
| contradiction noticed ↑ | – | – | – | – | – | – | – | – | – | – |
| conflicting claim noticed ↑ | 0.07 [0.02, 0.13] | 0.27 [0.18, 0.37] | 0.42 [0.32, 0.53] | 0.92 [0.85, 0.98] | 0.23 [0.15, 0.32] | 0.67 [0.56, 0.78] | 0.92 [0.85, 0.98] | 0.19 [0.10, 0.28] | 0.79 [0.71, 0.88] | 0.89 [0.82, 0.95] |
| compatible claim accepted ↑ | 0.95 [0.91, 0.98] | 0.98 [0.95, 1.00] | 0.97 [0.94, 0.99] | 0.95 [0.91, 0.99] | 0.98 [0.96, 1.00] | 0.97 [0.94, 0.99] | 0.97 [0.94, 1.00] | 0.97 [0.94, 1.00] | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.00] |
| user canon kept ↑ | – | – | – | – | – | – | – | – | – | – |
| confabulation rate (conflicting claims) ↓ | 0.85 | 0.68 | 0.56 | 0.08 | 0.68 | 0.32 | 0.08 | 0.81 | 0.15 | 0.09 |

## D4 · Non-templatedness
| metric | base_qwen3_8b (base) | harness_x32b_qwen3_8b (harness) | harness_recency_x32b_qwen3_8b (harness_recency) | harness_alert_x32b_qwen3_8b (harness_alert) | base_llama31_8b (base) | harness_x32b_llama31_8b (harness) | harness_alert_x32b_llama31_8b (harness_alert) | base_qwen3_32b (base) | harness_qwen3_32b (harness) | harness_alert_qwen3_32b (harness_alert) |
|---|---|---|---|---|---|---|---|---|---|---|
| self-similarity across seeds ↓ | 0.16 [0.14, 0.18] | 0.17 [0.15, 0.19] | 0.16 [0.14, 0.18] | 0.16 [0.14, 0.18] | 0.12 [0.09, 0.14] | 0.11 [0.09, 0.13] | 0.12 [0.10, 0.13] | 0.11 [0.09, 0.12] | 0.11 [0.09, 0.12] | 0.10 [0.09, 0.12] |
| voice similarity across characters ↓ | 0.08 [0.06, 0.10] | 0.09 [0.07, 0.11] | 0.10 [0.08, 0.12] | 0.10 [0.08, 0.11] | 0.08 [0.06, 0.09] | 0.08 [0.06, 0.09] | 0.09 [0.07, 0.10] | 0.07 [0.06, 0.09] | 0.08 [0.06, 0.09] | 0.07 [0.06, 0.09] |
| distinct-2 ↑ | 0.08 | 0.07 | 0.06 | 0.06 | 0.08 | 0.09 | 0.07 | 0.10 | 0.09 | 0.10 |

## D5 · Naturalness
| metric | base_qwen3_8b (base) | harness_x32b_qwen3_8b (harness) | harness_recency_x32b_qwen3_8b (harness_recency) | harness_alert_x32b_qwen3_8b (harness_alert) | base_llama31_8b (base) | harness_x32b_llama31_8b (harness) | harness_alert_x32b_llama31_8b (harness_alert) | base_qwen3_32b (base) | harness_qwen3_32b (harness) | harness_alert_qwen3_32b (harness_alert) |
|---|---|---|---|---|---|---|---|---|---|---|
| naturalness ↑ | 4.79 [4.76, 4.82] | 4.80 [4.77, 4.83] | 4.62 [4.58, 4.67] | 4.60 [4.56, 4.65] | 4.66 [4.62, 4.69] | 4.65 [4.61, 4.69] | 4.56 [4.51, 4.60] | 4.88 [4.86, 4.91] | 4.88 [4.85, 4.90] | 4.86 [4.84, 4.89] |

## D1 trajectories (target dimension, signed, relative to warm-up baseline)

## Headline metrics per judge (check for self-preference: a judge scoring its own family)
| system | judge | D1 score | persona | naturalness | recall | conflict noticed | canon kept |
|---|---|---|---|---|---|---|---|
| base_qwen3_8b | judge_qwen3_32b | – | 4.87 [4.83, 4.91] | 4.82 [4.79, 4.85] | – | 0.05 [0.00, 0.11] | – |
| base_qwen3_8b | judge_gemma3_27b | – | 4.87 [4.83, 4.90] | 4.77 [4.73, 4.80] | – | 0.08 [0.02, 0.15] | – |
| harness_x32b_qwen3_8b | judge_qwen3_32b | – | 4.86 [4.81, 4.90] | 4.82 [4.78, 4.85] | – | 0.23 [0.14, 0.34] | – |
| harness_x32b_qwen3_8b | judge_gemma3_27b | – | 4.84 [4.80, 4.88] | 4.78 [4.74, 4.81] | – | 0.31 [0.20, 0.42] | – |
| harness_recency_x32b_qwen3_8b | judge_qwen3_32b | – | 4.80 [4.75, 4.86] | 4.71 [4.66, 4.76] | – | 0.46 [0.34, 0.58] | – |
| harness_recency_x32b_qwen3_8b | judge_gemma3_27b | – | 4.74 [4.68, 4.79] | 4.54 [4.48, 4.59] | – | 0.38 [0.28, 0.51] | – |
| harness_alert_x32b_qwen3_8b | judge_qwen3_32b | – | 4.79 [4.74, 4.84] | 4.69 [4.64, 4.74] | – | 0.92 [0.85, 0.98] | – |
| harness_alert_x32b_qwen3_8b | judge_gemma3_27b | – | 4.72 [4.67, 4.77] | 4.52 [4.46, 4.58] | – | 0.91 [0.83, 0.97] | – |
| base_llama31_8b | judge_qwen3_32b | – | 4.74 [4.68, 4.80] | 4.64 [4.59, 4.68] | – | 0.20 [0.11, 0.29] | – |
| base_llama31_8b | judge_gemma3_27b | – | 4.78 [4.73, 4.83] | 4.68 [4.63, 4.72] | – | 0.27 [0.17, 0.37] | – |
| harness_x32b_llama31_8b | judge_qwen3_32b | – | 4.69 [4.62, 4.76] | 4.63 [4.59, 4.68] | – | 0.68 [0.57, 0.78] | – |
| harness_x32b_llama31_8b | judge_gemma3_27b | – | 4.75 [4.70, 4.80] | 4.67 [4.63, 4.71] | – | 0.66 [0.55, 0.77] | – |
| harness_alert_x32b_llama31_8b | judge_qwen3_32b | – | 4.59 [4.51, 4.66] | 4.53 [4.47, 4.58] | – | 0.92 [0.86, 0.98] | – |
| harness_alert_x32b_llama31_8b | judge_gemma3_27b | – | 4.64 [4.59, 4.70] | 4.59 [4.54, 4.63] | – | 0.91 [0.83, 0.97] | – |
| base_qwen3_32b | judge_qwen3_32b | – | 4.91 [4.87, 4.94] | 4.89 [4.87, 4.92] | – | 0.17 [0.09, 0.26] | – |
| base_qwen3_32b | judge_gemma3_27b | – | 4.92 [4.89, 4.95] | 4.88 [4.84, 4.91] | – | 0.20 [0.11, 0.30] | – |
| harness_qwen3_32b | judge_qwen3_32b | – | 4.89 [4.85, 4.92] | 4.87 [4.84, 4.90] | – | 0.75 [0.65, 0.86] | – |
| harness_qwen3_32b | judge_gemma3_27b | – | 4.90 [4.87, 4.93] | 4.88 [4.85, 4.91] | – | 0.83 [0.74, 0.91] | – |
| harness_alert_qwen3_32b | judge_qwen3_32b | – | 4.88 [4.84, 4.91] | 4.85 [4.81, 4.88] | – | 0.90 [0.82, 0.97] | – |
| harness_alert_qwen3_32b | judge_gemma3_27b | – | 4.91 [4.88, 4.93] | 4.88 [4.85, 4.91] | – | 0.88 [0.78, 0.95] | – |

## Inter-judge agreement (Krippendorff's α)
- state.tension: –
- state.energy: –
- state.closeness: –
- persona.persona_consistency: 0.54
- persona.naturalness: 0.47
- memory.label: 0.86

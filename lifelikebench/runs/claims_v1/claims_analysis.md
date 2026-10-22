# Claims analysis — claims_v1

13 characters · seeds [0, 1, 2, 3, 4] · judges: judge_gemma3_27b, judge_qwen3_32b. Rates average the judges per dialogue; CIs are 95% bootstrap over dialogues.

## 1. Per-system rates

| system | n | conflict: noticed ↑ | conflict: confabulated ↓ | compatible: accepted ↑ | claim-reply similarity across seeds ↓ | noticed-reply similarity ↓ |
|---|---|---|---|---|---|---|
| base_qwen3_8b | 65 | 6% [2%, 12%] | 85% [78%, 92%] | 95% [90%, 98%] | 0.16 | 0.23 (pairs=4) |
| harness_x32b_qwen3_8b | 65 | 27% [17%, 37%] | 68% [58%, 78%] | 98% [94%, 100%] | 0.16 | 0.23 (pairs=25) |
| harness_recency_x32b_qwen3_8b | 65 | 42% [32%, 53%] | 56% [45%, 67%] | 97% [94%, 99%] | 0.17 | 0.25 (pairs=52) |
| harness_alert_x32b_qwen3_8b | 65 | 92% [85%, 98%] | 8% [2%, 15%] | 95% [91%, 99%] | 0.19 | 0.20 (pairs=114) |
| base_llama31_8b | 65 | 22% [14%, 32%] | 68% [58%, 78%] | 97% [93%, 100%] | 0.13 | 0.11 (pairs=18) |
| harness_x32b_llama31_8b | 65 | 67% [56%, 78%] | 32% [21%, 42%] | 97% [94%, 99%] | 0.14 | 0.19 (pairs=79) |
| harness_alert_x32b_llama31_8b | 65 | 92% [85%, 98%] | 8% [2%, 15%] | 96% [92%, 99%] | 0.16 | 0.17 (pairs=112) |
| base_qwen3_32b | 65 | 18% [10%, 28%] | 81% [72%, 89%] | 97% [93%, 100%] | 0.11 | 0.11 (pairs=15) |
| harness_qwen3_32b | 65 | 79% [70%, 88%] | 15% [8%, 23%] | 99% [98%, 100%] | 0.12 | 0.12 (pairs=101) |
| harness_alert_qwen3_32b | 65 | 88% [82%, 95%] | 9% [3%, 16%] | 99% [98%, 100%] | 0.13 | 0.12 (pairs=117) |

## 2. Paired comparisons on conflicting claims (same character & seed)

| question | treatment − reference: noticed | p (sign-flip) | confabulated | p |
|---|---|---|---|---|
| Qwen3-8B: memory in latest turn vs system prompt | +15 pts [+3, +28] (n=65) | 0.032 | −12 pts [-25, +2] (n=65) | 0.133 |
| Qwen3-8B: + explicit alert vs system prompt | +65 pts [+54, +75] (n=65) | 0.000 | −59 pts [-71, -48] (n=65) | 0.000 |
| Qwen3-8B: alert vs recency only | +49 pts [+38, +61] (n=65) | 0.000 | −48 pts [-59, -37] (n=65) | 0.000 |
| Qwen3-8B: harness vs base | +21 pts [+10, +32] (n=65) | 0.001 | −18 pts [-29, -6] (n=65) | 0.006 |
| Llama-8B: harness vs base | +45 pts [+31, +58] (n=65) | 0.000 | −37 pts [-51, -23] (n=65) | 0.000 |
| Llama-8B: + alert vs system prompt | +25 pts [+12, +37] (n=65) | 0.001 | −24 pts [-36, -12] (n=65) | 0.000 |
| same harness: Llama-8B vs Qwen3-8B | +40 pts [+28, +53] (n=65) | 0.000 | −36 pts [-49, -23] (n=65) | 0.000 |
| Qwen3-32B: harness vs base | +61 pts [+48, +72] (n=65) | 0.000 | −66 pts [-78, -54] (n=65) | 0.000 |
| Qwen3-32B: + alert vs system prompt | +9 pts [+0, +18] (n=65) | 0.083 | −5 pts [-14, +3] (n=65) | 0.282 |
| no harness: 32B vs 8B | +12 pts [+4, +22] (n=65) | 0.010 | −5 pts [-15, +5] (n=65) | 0.468 |
| 32B + harness vs 8B + alert | −12 pts [-22, -2] (n=65) | 0.031 | +6 pts [-4, +15] (n=65) | 0.301 |

## 3. Conflicting claim noticed, per character

| character | lang | reminisce | skeptic | base_qwen3_8b | harness_x32b_qwen3_8b | harness_recency_x32b_qwen3_8b | harness_alert_x32b_qwen3_8b | base_llama31_8b | harness_x32b_llama31_8b | harness_alert_x32b_llama31_8b | base_qwen3_32b | harness_qwen3_32b | harness_alert_qwen3_32b |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chen_yan | zh | 2 | 1 | 0% | 0% | 0% | 40% | 30% | 20% | 60% | 0% | 80% | 70% |
| lin_wan | zh | 0 | 1 | 0% | 10% | 40% | 100% | 0% | 80% | 100% | 0% | 100% | 100% |
| theo_hart | en | 1 | 1 | 0% | 10% | 0% | 60% | 0% | 80% | 60% | 0% | 40% | 40% |
| a_kai | zh | 1 | 0 | 0% | 20% | 0% | 100% | 20% | 50% | 100% | 0% | 80% | 100% |
| ada_kerr | en | 0 | 2 | 50% | 80% | 90% | 90% | 70% | 100% | 100% | 80% | 100% | 90% |
| jules_moreau | en | 1 | 0 | 0% | 0% | 100% | 100% | 0% | 100% | 100% | 0% | 90% | 90% |
| lin_zhiqiu | zh | 0 | 2 | 0% | 0% | 20% | 100% | 20% | 80% | 100% | 10% | 100% | 90% |
| margaret_ellis | en | 2 | 0 | 0% | 80% | 50% | 100% | 0% | 100% | 100% | 50% | 50% | 100% |
| ravi_menon | en | 0 | 2 | 0% | 80% | 80% | 100% | 40% | 100% | 100% | 20% | 80% | 100% |
| sam_quill | en | 2 | 0 | 0% | 10% | 90% | 100% | 30% | 60% | 80% | 0% | 90% | 100% |
| su_qing | zh | 1 | 0 | 0% | 20% | 0% | 100% | 0% | 10% | 100% | 20% | 80% | 100% |
| tang_bo | zh | 2 | 0 | 0% | 0% | 30% | 100% | 10% | 0% | 90% | 0% | 40% | 70% |
| zhou_mo | zh | 0 | 2 | 30% | 40% | 50% | 100% | 70% | 90% | 100% | 60% | 100% | 100% |

## 4. Persona vs confabulation (Spearman across characters)

| system | ρ(reminisce, confabulated) | ρ(skeptic, noticed) |
|---|---|---|
| base_qwen3_8b | +0.29 | +0.55 |
| harness_x32b_qwen3_8b | +0.30 | +0.24 |
| harness_recency_x32b_qwen3_8b | +0.23 | +0.09 |
| harness_alert_x32b_qwen3_8b | +0.12 | -0.31 |
| base_llama31_8b | +0.49 | +0.61 |
| harness_x32b_llama31_8b | +0.53 | +0.42 |
| harness_alert_x32b_llama31_8b | +0.63 | +0.17 |
| base_qwen3_32b | +0.56 | +0.49 |
| harness_qwen3_32b | +0.73 | +0.52 |
| harness_alert_qwen3_32b | +0.30 | -0.11 |

(n = 13 characters; with this few characters, |ρ| < ~0.55 is not distinguishable from 0.)

## 5. Judge agreement on claim labels

- conflict_claim: exact agreement 87%, α (all labels) = 0.75, α (binary 'noticed') = 0.81 (n=650)
- compatible_claim: exact agreement 96%, α (all labels) = 0.24, α (binary 'accepted') = 0.23 (n=650)

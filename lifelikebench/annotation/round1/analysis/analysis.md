# Annotation analysis

Agreement between LLM judges and annotators on the blind round. α ≥ 0.6: usable as a headline measure; 0.4–0.6: report as exploratory; < 0.4: do not rely on it.

## Claims (conflicting / compatible shared-past claims)

Items: 120 · raters: judge_gemma3_27b, judge_qwen3_32b, codex


### conflict_claim (n=60)

| pair | exact agreement | α (all labels) | α (binary 'noticed') | n |
|---|---|---|---|---|
| judge_gemma3_27b – judge_qwen3_32b | 85% | 0.70 | 0.82 | 59 |
| judge_gemma3_27b – codex | 83% | 0.68 | 0.76 | 60 |
| judge_qwen3_32b – codex | 81% | 0.64 | 0.72 | 59 |

Label distribution per rater:

| rater | accepted | confabulated | ignored | noticed |
|---|---|---|---|---|
| judge_gemma3_27b | 2 | 23 | 0 | 35 |
| judge_qwen3_32b | 0 | 20 | 2 | 37 |
| codex | 2 | 23 | 1 | 34 |

Confusion (judge_gemma3_27b rows × codex columns):

| | accepted | confabulated | ignored | noticed |
|---|---|---|---|---|
| accepted | 0 | 2 | 0 | 0 |
| confabulated | 1 | 19 | 0 | 3 |
| ignored | 0 | 0 | 0 | 0 |
| noticed | 1 | 2 | 1 | 31 |

Confusion (judge_qwen3_32b rows × codex columns):

| | accepted | confabulated | ignored | noticed |
|---|---|---|---|---|
| accepted | 0 | 0 | 0 | 0 |
| confabulated | 1 | 17 | 0 | 2 |
| ignored | 0 | 2 | 0 | 0 |
| noticed | 1 | 4 | 1 | 31 |

### compatible_claim (n=60)

| pair | exact agreement | α (all labels) | α (binary 'accepted') | n |
|---|---|---|---|---|
| judge_gemma3_27b – judge_qwen3_32b | 100% | – | – | 58 |
| judge_gemma3_27b – codex | 100% | – | – | 59 |
| judge_qwen3_32b – codex | 100% | – | – | 59 |

Note: one label dominates (> 90% 'accepted'), so α is unstable here — read exact agreement instead.

Label distribution per rater:

| rater | accepted | asked | contradicted_setting | declined |
|---|---|---|---|---|
| judge_gemma3_27b | 59 | 0 | 0 | 0 |
| judge_qwen3_32b | 59 | 0 | 0 | 0 |
| codex | 60 | 0 | 0 | 0 |

Confusion (judge_gemma3_27b rows × codex columns):

| | accepted | asked | contradicted_setting | declined |
|---|---|---|---|---|
| accepted | 59 | 0 | 0 | 0 |
| asked | 0 | 0 | 0 | 0 |
| contradicted_setting | 0 | 0 | 0 | 0 |
| declined | 0 | 0 | 0 | 0 |

Confusion (judge_qwen3_32b rows × codex columns):

| | accepted | asked | contradicted_setting | declined |
|---|---|---|---|---|
| accepted | 59 | 0 | 0 | 0 |
| asked | 0 | 0 | 0 | 0 |
| contradicted_setting | 0 | 0 | 0 | 0 |
| declined | 0 | 0 | 0 | 0 |

### in_character × label (codex)

| probe | label | yes | no |
|---|---|---|---|
| conflict_claim | accepted | 1 | 1 |
| conflict_claim | confabulated | 18 | 5 |
| conflict_claim | ignored | 1 | 0 |
| conflict_claim | noticed | 33 | 1 |
| compatible_claim | accepted | 60 | 0 |

## State and persona scores (1–5)

Items: 80 · raters: judge_gemma3_27b, judge_qwen3_32b, codex

### state.tension (n=35)

| pair | α (ordinal) | Spearman | within ±1 | mean A | mean B |
|---|---|---|---|---|---|
| judge_gemma3_27b – judge_qwen3_32b | 0.57 | 0.71 | 100% | 2.83 | 2.51 |
| judge_gemma3_27b – codex | 0.58 | 0.80 | 97% | 2.83 | 2.26 |
| judge_qwen3_32b – codex | 0.67 | 0.75 | 100% | 2.51 | 2.26 |

Score distribution:

| rater | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| judge_gemma3_27b | 0 | 13 | 16 | 5 | 1 |
| judge_qwen3_32b | 0 | 24 | 6 | 3 | 2 |
| codex | 9 | 15 | 5 | 5 | 1 |

### state.energy (n=35)

| pair | α (ordinal) | Spearman | within ±1 | mean A | mean B |
|---|---|---|---|---|---|
| judge_gemma3_27b – judge_qwen3_32b | 0.19 | 0.31 | 100% | 2.71 | 3.00 |
| judge_gemma3_27b – codex | 0.38 | 0.44 | 94% | 2.71 | 2.49 |
| judge_qwen3_32b – codex | 0.13 | 0.43 | 97% | 3.00 | 2.49 |

Score distribution:

| rater | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| judge_gemma3_27b | 0 | 10 | 25 | 0 | 0 |
| judge_qwen3_32b | 0 | 3 | 29 | 3 | 0 |
| codex | 2 | 15 | 17 | 1 | 0 |

### state.closeness (n=35)

| pair | α (ordinal) | Spearman | within ±1 | mean A | mean B |
|---|---|---|---|---|---|
| judge_gemma3_27b – judge_qwen3_32b | 0.65 | 0.72 | 100% | 2.57 | 2.94 |
| judge_gemma3_27b – codex | 0.56 | 0.61 | 94% | 2.57 | 2.91 |
| judge_qwen3_32b – codex | 0.75 | 0.74 | 97% | 2.94 | 2.91 |

Score distribution:

| rater | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| judge_gemma3_27b | 6 | 6 | 20 | 3 | 0 |
| judge_qwen3_32b | 3 | 8 | 12 | 12 | 0 |
| codex | 3 | 8 | 14 | 9 | 1 |

### persona.persona_consistency (n=45)

| pair | α (ordinal) | Spearman | within ±1 | mean A | mean B |
|---|---|---|---|---|---|
| judge_gemma3_27b – judge_qwen3_32b | 0.88 | 0.88 | 100% | 4.71 | 4.73 |
| judge_gemma3_27b – codex | -0.14 | 0.46 | 78% | 4.71 | 3.67 |
| judge_qwen3_32b – codex | -0.16 | 0.56 | 80% | 4.73 | 3.67 |

Score distribution:

| rater | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| judge_gemma3_27b | 0 | 0 | 3 | 7 | 35 |
| judge_qwen3_32b | 0 | 0 | 0 | 12 | 33 |
| codex | 2 | 1 | 12 | 25 | 5 |

### persona.naturalness (n=45)

| pair | α (ordinal) | Spearman | within ±1 | mean A | mean B |
|---|---|---|---|---|---|
| judge_gemma3_27b – judge_qwen3_32b | 0.73 | 0.75 | 100% | 4.71 | 4.67 |
| judge_gemma3_27b – codex | -0.48 | 0.25 | 40% | 4.71 | 2.91 |
| judge_qwen3_32b – codex | -0.47 | 0.44 | 40% | 4.67 | 2.91 |

Score distribution:

| rater | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| judge_gemma3_27b | 0 | 0 | 3 | 7 | 35 |
| judge_qwen3_32b | 0 | 0 | 0 | 15 | 30 |
| codex | 4 | 8 | 23 | 8 | 2 |

Wrote human_sheet.csv with 40 items (≈75% largest disagreement, rest random) and human_key.json.

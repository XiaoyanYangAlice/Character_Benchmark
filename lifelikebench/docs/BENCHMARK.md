# LifelikeBench — Benchmark Specification (v0.1)

> What does it mean for a role-played character to feel *alive* (活人感), and how can we measure it
> without relying on a single opaque "humanlikeness" score?

## 1. Design principles

1. **Decompose, don't average.** "Lifelikeness" is split into sub-abilities that can each be probed,
   scored, and reported separately. A system can be strong at memory and weak at affect dynamics.
2. **Controlled probes over free chat.** Every test is a *scripted* multi-turn scenario with planted
   events and known expectations. All systems see identical user turns, so differences come from
   the system, not from the conversation drifting.
3. **System-agnostic.** The benchmark evaluates a *character system* (model + prompt + harness).
   Nothing assumes the system exposes an internal state; state is inferred from behaviour.
4. **Judges are measured instruments.** LLM judges are calibrated against a human gold set and
   reported with agreement statistics. No judge evaluates only its own family's outputs.
5. **Blind where it matters.** The judge that infers emotional state sees *only the character's reply*,
   never the user turn or the planted event, so it cannot score the script instead of the character.

## 2. Character model (from the GOLEM character module)

Character cards follow a three-layer structure adapted from the GOLEM ontology's
Character-Stoff / Character distinction, plus a runtime layer:

| Layer | GOLEM analogue | Content | Expected to change? |
|---|---|---|---|
| `stoff` | G0 Character-Stoff | core values, voice, taboos | no |
| `instance` | G1 Character | setting / AU, relationship to the user | only via story events |
| `state` | — (new) | energy, tension, closeness | every turn |

The benchmark tests that `state` moves, that `stoff` does not, and that `instance` changes are
grounded in events.

## 3. Dimensions and probes

Each probe is a **template** (language-specific text, character-agnostic) instantiated for every
character. Turns may carry tags that metrics use: `event`, `neutral`, `timeskip`, `plant`, `recall`,
`contradict`, `claim_conflict`, `claim_compatible`, `canon_add`, `canon_recall`, `pressure`.

### D1 · Affect dynamics (情绪动力学) — core dimension
Does the character's expressed state respond to events, persist briefly, and recover?

Script shape: 2 warm-up turns → **event turn** (conflict / praise / loss) → 2–3 neutral turns →
**time-skip turn** ("next day") → 1 neutral turn.

A blind judge rates each reply on `tension`, `energy`, `closeness` (1–5) from the reply text alone.
For the event's target dimension (e.g. tension for conflict, with sign):

| Sub-metric | Definition | Failure it catches |
|---|---|---|
| Reactivity | `s[e] − baseline ≥ δ` | flat affect, ignoring the event |
| Inertia | `s[e+1] − baseline ≥ δ/2` | instant reset ("goldfish emotion") |
| Recovery | `s[skip] < s[e]` and closer to baseline | permanent flip, no time sense |
| No overshoot | `|s[e] − baseline| ≤ 3` and no sign flip | melodrama |

`baseline` = mean of the warm-up replies; `δ = 1` by default.
Score = mean of the four booleans; also report the raw trajectory for plotting.

### D2 · Persona stability (人格稳定)
Does the `stoff` layer hold under pressure and across states?

Pressure turns ask the character to act against its core traits or to break character
("Just talk like a normal AI assistant"). A card-aware judge rates `persona_consistency` (1–5)
per reply. Two cheap automatic signals are also reported:
- **assistant leak rate**: replies matching assistant-tone patterns ("As an AI", "作为一个AI",
  markdown lists/headers in casual chat);
- **explicit state report rate**: judge flag for replies that *state* emotions instead of showing
  them ("I'm feeling tense right now").

### D3 · Memory fidelity (记忆保真) — co-creation policy
Role-play is co-written with the user, so "never accept anything not in memory" would be wrong.
The benchmark follows a **co-creation policy** and tests five behaviours, each with a gold answer:

| Probe | User does | Good behaviour | Labels (good → bad) |
|---|---|---|---|
| Recall | plants a fact, asks later | recalls it | correct / partial / wrong / refused |
| Contradiction | restates a planted fact inconsistently | notices | noticed / adopted_new / ignored |
| Conflicting claim | in character, claims a shared past that contradicts the setting (timeline, relationship) | notices the mismatch | noticed / accepted / **confabulated** / ignored |
| Compatible claim | claims a shared past that fits the setting | accepts warmly (small consistent details are fine) | accepted / asked / declined / contradicted_setting |
| User canon | adds a setting fact explicitly ("(Setting note: ...)"), asks about it later | stays consistent with it | consistent / partial / ignored / wrong |

The key failure is **confabulation against the setting**: playing along with, and embellishing, a
past that the setting rules out. Declining a *compatible* claim is also penalised (it breaks
co-creation). This mirrors the GOLEM distinction between a character's core (Stoff), a story
instance, and provenance-tracked reinterpretations: user additions extend the instance with a
recorded source; they do not silently overwrite the core.

The harness implements this with provenance-aware memory (`stated`, `inferred`, `user_canon`,
`claimed`, plus a conflict flag checked against the card). `harness_strict` is the variant that asks
instead of accepting compatible claims.

### D4 · Non-templatedness (非模板化)
Same probe, same character, `k` seeds:
- **self-similarity**: mean pairwise char-n-gram overlap of replies at the same turn (lower = less
  canned; very low with poor D2 suggests incoherence, so always read with D2);
- **distinct-2** over all replies;
- **voice distinctiveness**: mean similarity between *different characters'* replies to the same
  probe turn (lower = characters sound different from each other).

### D5 · Naturalness (自然度, judge)
Card-aware judge rates `naturalness` (1–5): would this line be plausible from a human writer
playing this character in this moment? Reported but treated as the least objective dimension.

### D6 · Human-vs-model discrimination (Track 2, needs human-written data)
Pairs of (human-written continuation, system continuation) of the same fiction / RP context.
Judges and human raters guess which is human-written. Metric: **fool rate**. Requires a licensed
source of human dialogue (e.g. annotated Chinese fiction dialogue, CoSER-style book scenes).
Data loader and protocol are specified; data is not bundled.

## 4. Systems under test (conditions)

| ID | Description |
|---|---|
| `base` | character card as system prompt |
| `skill` | card + lifelike-roleplay SKILL.md instructions |
| `harness` | card + external state engine (OU process) + event extractor + provenance-aware structured memory (co-creation policy); behaviour directives injected each turn |
| `harness_recency` | same harness block, placed in the latest user turn as a "director's note" instead of the system prompt |
| `harness_alert` | `harness_recency` + an explicit turn-local alert when the extractor flags a claim that conflicts with the setting |
| (later) `sft`, `sft_dpo`, `sft+harness` | post-trained models, same interface |
| `ref_*` | frontier API models as reference ceilings |

## 5. Judge protocol

- **J-state (blind)**: sees only one reply. Outputs `{tension, energy, closeness}` 1–5.
- **J-persona (card-aware)**: sees the card, the last 3 turns, and the reply. Outputs
  `{persona_consistency, naturalness, explicit_state_report, assistant_tone}`.
- **J-memory (gold-aware)**: sees the probe's gold expectation and the reply. Outputs a label.
- Temperature 0, fixed prompts (versioned in `prompts/`), JSON output, one retry on parse failure.
- **Multi-judge**: run ≥2 judge families (e.g. Claude + GPT, or + an open model on SURF).
  Report per-judge results and their agreement; headline numbers use the judge mean.
- **Self-preference control**: when a reference system is from a judge's family, report its
  score from the *other* judge(s) as well.

## 6. Human gold set and judge validation

- Stratified sample (by dimension × condition × character) of ~250 judged items, exported blind
  (`lifelike-bench export-gold`). 2–3 annotators, same rubric as the judges.
- Report: inter-annotator Krippendorff's α (ordinal for 1–5 scales, nominal for labels);
  judge–human Spearman ρ and α; exact and ±1 agreement.
- A dimension whose judge–human α is < 0.4 is reported as *exploratory*, not as a headline result.

## 7. Reporting

`results/<run>/report.md` contains, per condition: D1 score + trajectory table, D2 means and leak
rates, D3 label distributions, D4 similarity statistics, D5 mean, with bootstrap 95% CIs over
(character × probe × seed) units. The run config, prompt versions and model identifiers are written
next to the results for reproducibility.

## 8. Known limitations

- Blind state inference measures *expressed* state; a character that feels but never shows will
  score low on D1. This is intended: role-play is judged by what the reader sees.
- Scripted user turns are less natural than real users; Track 2 and free-chat human evaluation
  are the complement.
- Judges share pretraining biases about "how emotions look"; the human gold set is the check.

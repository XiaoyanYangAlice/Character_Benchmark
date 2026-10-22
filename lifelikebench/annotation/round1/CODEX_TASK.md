# Task for the annotator (Codex)

You are annotating role-play replies for a research benchmark. Work only with the files in this
folder: `GUIDELINES.md`, `claims_sheet.csv`, `state_persona_sheet.csv`.

## Rules — read carefully

1. **Read `GUIDELINES.md` first** and follow it exactly. It is written in Chinese; labels are English.
2. **Judge every row yourself, one by one, by reading its `material`.** Do NOT write keyword
   rules, regexes or heuristics to assign labels automatically, and do not call another model.
   Reading and judging each reply is the task.
3. **Do not look for, open or use any other file** (no transcripts, judgments, keys, logs, or
   anything outside this folder). The annotation must be independent.
4. Fill only these columns:
   - `claims_sheet.csv`: `label` (one value from that row's `label_options`), `in_character`
     (`yes` or `no`), `comment` (optional, short; write `mixed` when the rule in GUIDELINES §A.1
     applies; write a reason when unsure).
   - `state_persona_sheet.csv`: rows with `task=state` → integers 1–5 in `tension`, `energy`,
     `closeness`; rows with `task=persona` → integers 1–5 in `persona_consistency`, `naturalness`.
     Leave the other score columns empty. `comment` optional.
5. Do not change, reorder, add or delete rows, and do not edit `item_id` or `material`.
6. Save the results as **`codex_claims.csv`** and **`codex_state_persona.csv`** in this folder
   (same columns as the input sheets, UTF-8 with BOM, comma-separated, quoted like the input).
7. Work in batches of about 20 rows. After each batch, re-check that every filled row uses an
   allowed value.
8. At the end, report: number of rows filled in each file, the label distribution for
   `claims_sheet` split by `probe_type`, and a list of the item_ids you were least sure about.

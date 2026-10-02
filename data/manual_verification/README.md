# Manual label-noise verification (expert CRS-label audit)

The OWASP `attack_subtype` labels are **derived from CRS rule tags**, not
hand-annotated. This subset puts a defensible number on how reliable that weak
labelling is — the per-class CRS precision a reviewer will ask for. The manuscript
reports it: **98.0% CRS-vs-expert agreement on the decidable records**, with the
undecidable-under-anonymization fraction reported as the evidence-availability
ceiling.

## Pipeline at a glance

```
scripts/42_manual_verification.py   ->  owasp_1k_subset.csv      (blank worksheet)
        (human reviewer fills it,
         saved as an .xlsx)         ->  owasp_1k_subset_human.xlsx (label_human filled)
scripts/46_human_audit_score.py     ->  results/manual_verification/results.json
```

## What to do

1. **Generate the worksheet** (1,000 deterministically-sampled OWASP test records;
   re-creatable bit-for-bit):
   ```
   python scripts/42_manual_verification.py --split owasp_test --n 1000 --seed 42
   ```
   This writes `owasp_1k_subset.csv` with a blank `label_human` (and `notes_human`)
   column, plus the design sidecar `sample_manifest.json`. The draw is
   **stratified**: rare attack classes are over-sampled to a per-class floor
   (`--floor`, default 60) so each class has enough rows for a usable per-class
   confidence interval, and each row carries a `sample_weight`
   (= corpus_count / sampled_count) that reweights the worksheet statistics back to
   the true corpus base rate.

2. **Fill the `label_human` column** for each row — your judgment of the request,
   given `method/path/query/ua/status`, **blind to `crs_subtype`** (do not edit
   `crs_subtype`; your `label_human` is compared against it). Allowed values:
   - one of the 8 attack subtypes:
     `sql_injection`, `xss`, `rce`, `php_injection`, `path_traversal`, `lfi`,
     `scanner`, `protocol`;
   - `benign` / `false_positive` — the request is not actually that attack (a CRS
     false positive);
   - **`normal`** / `unsure` — **undecidable** from the logged view (the
     payload-bearing field is hashed by anonymization). These are EXCLUDED from the
     CRS-precision denominator and reported separately as the
     evidence-availability ceiling.

   Save the filled sheet as **`owasp_1k_subset_human.xlsx`** (column `label_human`).
   The `.xlsx` is the source of truth for the audit; the `.csv` is only the blank
   template.

3. **Score the audit** (only after labelling — single human reviewer, so there is
   no inter-rater kappa; the audit is the human-vs-CRS comparison):
   ```
   pip install openpyxl==3.1.5        # or: pip install -e ".[label-audit]"
   python scripts/46_human_audit_score.py
   python scripts/76_audit_weighting_check.py   # unrounded rates + formulas (stdlib only)
   ```
   → writes `results/manual_verification/results.json`: test-split-weighted CRS
   accuracy over the decidable records, per-class CRS precision (human == crs within
   each stratum) with Wilson 95% CI, the benign (CRS false-positive) rate, and the
   undecidable rate.

> An earlier multi-LLM rater trial (additional `label_opus`/`label_codex`/… columns,
> scored by a now-retired `scripts/45_rater_scoring.py`) has been removed. The
> paper's label audit is this single-human expert review. A second expert fills
> `label_human2` (see `paper/notes/second_expert_audit_protocol.md`).

## This subset's CRS-label mix (stratified, what you'll be checking)

| crs_subtype     | rows sampled | sample_weight |
|-----------------|-------------:|--------------:|
| protocol        | 360          | 27.4          |
| lfi             | 327          | 26.9          |
| rce             | 77           | 7.1           |
| php_injection   | 65           | 2.8           |
| sql_injection   | 64           | 2.2           |
| scanner         | 62           | 1.0           |
| path_traversal  | 23           | 1.0           |
| xss             | 22           | 1.0           |

Rare classes are over-sampled (each floored near 60 where the corpus allows), so a
per-class precision CI is usable; the `sample_weight` column undoes the
over-sampling when the corpus-level accuracy is computed.

## Provenance

`sample_manifest.json` records the full design (seed, floor, per-class corpus and
sampled counts, `sample_weight`, sampled indices, commit, date) so the draw and the
reweighting are auditable and reproducible bit-for-bit.

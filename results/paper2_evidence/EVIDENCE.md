# Paper #2 — Evidence Map

Every number, table, and figure in `paper/manuscript/paper2/ecti_cit_dp_synthlog.tex`
traces to the machine-generated evidence in this folder. Nothing is hand-authored.

## Files
- **`evidence.json`** — all reported values as structured data (means, stds, %,
  per-class F1, confusion matrix), plus a `meta` block with the git commit, date,
  CPU, and library versions the run used (provenance).
- **`raw_run_log.txt`** — the verbatim stdout of the run that produced
  `evidence.json` (human-readable cross-check).
- **`tools/dp_synthlog_spike.py`** (in repo root) — the single script that
  generates both; re-running it reproduces this folder deterministically (fixed
  seeds {42..46}).

## Manuscript ↔ evidence.json mapping

| Manuscript element | evidence.json key |
|---|---|
| Abstract: 81% @ε=1, 68% @ε=0.1, 13% floor | `table1_frontier`, `lower_label_shuffle` |
| Abstract/Conclusion: ~26,000 rec/s, 55 KB, ~6 MB | `table4_resource."Proposed DP (eps=1)"`, `model_footprint_kb` |
| Table 2 (dataset): supports, vocab sizes | `n_train`, `n_test`, `config` (vocab from raw log / dataset script) |
| Table 3 (TSTR frontier): real upper, ε rows, bootstrap, pairwise, marginal, shuffle | `upper_real_train`, `table1_frontier`, `baselines`, `lower_label_shuffle` |
| Table 4 (resource): fit/gen/mem mean±std | `table4_resource` |
| Table 5 (per-class F1 @ε=1) | `table5_perclass_eps1` |
| Figure 3 (frontier curve) | `table1_frontier` + `upper_real_train` + `lower_label_shuffle` |
| Figure 4 (confusion matrix, row-normalized recall) | `fig4_confusion_eps1` |

## Provenance rule
`evidence.json:meta.git_commit` records the HEAD the run was executed at. When the
manuscript is finalized, re-run the spike at the release commit so the evidence
and the paper share one commit (same pattern as Paper #1). `git_dirty=true` means
there were uncommitted changes at run time — re-run on a clean tree before
submission.

# Artifact map (paper -> file -> command)

Every table and figure in the manuscript traces to a `results/` artifact and a
command that regenerates it. Table/figure numbers follow the compiled PDF.

## Quick start

```bash
pip install -e .            # torch-free proposed detector (add ".[baselines]" for neural baselines)
python scripts/98_smoke_test.py   # end-to-end smoke test on synthetic data (no raw data)
make verify-artifact        # check code matches PROVENANCE.json + results.json parse
make paper                  # 5-seed rebuild {42..46} of the main tables and figures (needs staged raw data)
make p1-artifacts           # Revision-1 analyses (Tables 2, 4, 5, Table 8 lower block, Fig. 2, audits)
```

## Tables

Numbers follow the revised manuscript (Revision 1). `results/` directory names
are internal and do not match paper table numbers.

| Paper table | `results/` artifact | Regenerate |
|---|---|---|
| T1 Datasets and roles | (static; see `DATASET.md`) | -- |
| T2 OWASP record counts (descriptor 147,205; raw archive 151,845; 834 non-hexadecimal boundary IDs not parsed; parser output 151,011) | `results/table_19_owasp_reconciliation/boundary_id_audit.json`, `results.json`, `reparse_bitwise_check.json` | `scripts/77_boundary_id_audit.py`, `scripts/67_owasp_reconciliation_audit.py`, `scripts/72_reparse_bitwise_check.py` (in `make p1-artifacts`; need staged raw data) |
| T3 OWASP subtype distribution | `results/table_02_dataset_profile/results.json` | `python scripts/61_make_dataset_profile.py` (in `make p1-artifacts`) |
| T4 Labeled records per subtype and budget | `results/table_21_budget_selection_val/budget_class_counts.json` | `scripts/69_budget_selection_val.py` (in `make p1-artifacts`) |
| T5 Validation vs. test macro-F1 by budget | `results/table_21_budget_selection_val/results.json` | `scripts/69_budget_selection_val.py --aggregate` (in `make p1-artifacts`) |
| T6 Eight- vs. seven-class macro-F1 | `results/table_15_protocol_sensitivity/results.json` | `scripts/62_protocol_sensitivity.py` |
| T7 RQ1 macro-F1 by budget | `results/table_03_rq1_baselines/results.json` | `make table-03` (one seed) / `make paper` (5-seed) |
| T8 RQ2 representation ablations (upper block) | `results/table_05_field_granularity/results.json` | `make table-05` |
| T8 RQ2 embedding-corpus control (lower block) | `results/table_20_pretrain_corpus_control/` | `scripts/68_pretrain_corpus_control.py --aggregate` (in `make p1-artifacts`) |
| T9 Per-class F1 (5%/10%) | `results/table_06_per_class_owasp/per_class_breakdown.json` (5%), `per_class_breakdown_10pct.json` (10%) | `make table-06` (runs the 5% and 10% budgets; also in `make paper`) |
| T10 Field-saliency agreement | `results/table_07_attribution/results.json` | `make table-07` |
| T11 Per-stage CPU latency | `results/table_08_latency/results.json` | `make table-08` |
| T12 System profile + baseline latency | `results/table_08_latency/results.json` | `make table-08` |
| T13 External probes + benign FPR | `results/table_09_cross_dataset_biblio_us17/`, `.../table_09_cross_dataset_csic/`, `.../table_11_benign_fpr_apache_indo/` (Wilson bound: `wilson_bounds.json`) | `scripts/13_cross_dataset_*.py`, `scripts/14_benign_fpr_apache_indo.py`, `scripts/75_fpr_wilson_bound.py` |
| T14 CPU cores for a given load | `results/table_18_deployment_workload/results.json` | `python scripts/65_deployment_workload_table.py` (in `make p1-artifacts`) |

## Figures

| Paper figure | `results/` artifact | Regenerate |
|---|---|---|
| Fig. 1 Architecture | (static diagram) | -- |
| Fig. 2 Confusion matrix (10%) | `results/table_06_per_class_owasp/confusion_matrix_10pct.json` (the 5% matrix behind the text: `confusion_matrix.json`) | `python scripts/16_confusion_matrix.py --budget 10` (and `--budget 5`; both in `make paper`) |
| Fig. 3 Per-class F1 (10%) | `results/table_06_per_class_owasp/per_class_breakdown_10pct.json` | `make table-06` (10% run; also in `make paper`) |
| Fig. 4 Learning curve | `results/figure_04_learning_curve/figure_data.json` | `python scripts/51_make_learning_curve_figure.py` |
| Fig. 5 Latency-accuracy front | `results/figure_03_latency_accuracy_pareto/figure_data.json` | `python scripts/52_make_pareto_figure.py` |

## Supporting artifacts (referenced in text)

| Item | `results/` artifact | Regenerate |
|---|---|---|
| Paired bootstrap significance | `results/table_03_rq1_baselines/bootstrap_significance*.json` | `make significance` |
| Rare-class CI | `results/table_16_rare_class_ci/` | `scripts/63_rare_class_ci.py` |
| No-custom-rule-444444 control | `results/table_17_no_custom_444444/` | `scripts/66_no_custom_rule_444444.py` |
| Descriptor-gap filters | `results/table_19_owasp_reconciliation/descriptor_gap_filters.json` | `scripts/70_descriptor_gap_filters.py` |
| Raw-archive boundary-ID audit (834 entries not parsed; exploratory one-seed sensitivity check; descriptor filters on the full archive) | `results/table_19_owasp_reconciliation/boundary_id_audit.json` | `scripts/77_boundary_id_audit.py` |
| Interception vs. response status | `results/table_19_owasp_reconciliation/intercept_status_audit.json` | `scripts/74_intercept_status_audit.py` |
| Validation RCE diagnostic | `results/diagnostics/val_rce_10pct.json` | `scripts/71_val_rce_diagnostic.py` |
| Cross-platform drift | `results/diagnostics/cross_platform_*.json` | `scripts/73_cross_platform_check.py` |
| Human label audit (test-split-weighted) | `results/manual_verification/results.json`; unrounded rates and formulas: `weighting_check.json` | `scripts/42_manual_verification.py` + `scripts/46_human_audit_score.py`; `scripts/76_audit_weighting_check.py` (standard library only) |
| FastText dimension sweep (scored on test; see paper Sec. 5.2) | `results/table_10_fasttext_sweep/` | `scripts/14_fasttext_sweep.py` |
| Validation-only Optuna search of the head (post hoc sensitivity check; 10% refit in `ci_validation_10pct_valselected/`) | `results/optuna/` | `experiments/optuna/run_study.py` + `experiments/optuna/run_ci_validation.py` (needs `requirements-optuna.txt`) |
| Label-noise (cleanlab) | `results/table_12_label_noise_cleanlab/` | needs `.[label-audit]` extra |
| POST-body pilot | `results/table_13_post_body_pilot/` | pilot analysis |
| Budget-composition CI | `results/table_14_budget_composition_ci/` | budget-composition analysis |
| Environment / versions | `results/environment/results.json` | recorded at run time |

## Provenance and seeds

- The paper's numbers are 5-seed means/CIs over the seed set {42,43,44,45,46};
  `make paper` (main tables and figures) and `make p1-artifacts` (Revision-1
  analyses) reproduce them. A single `make table-0X SEED=42` run gives a
  single-seed cell, not the paper table.
- Each `results/<table>/results.json` records the internal commit that produced
  it; `PROVENANCE.json` bridges those to this public snapshot and to a content
  hash verified by `verify_provenance.py`.

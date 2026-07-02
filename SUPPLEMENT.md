# Artifact map (paper -> file -> command)

Every table and figure in the manuscript traces to a `results/` artifact and a
command that regenerates it. Table/figure numbers follow the compiled PDF.

## Quick start

```bash
pip install -e .            # torch-free proposed detector (add ".[baselines]" for neural baselines)
python scripts/98_smoke_test.py   # end-to-end smoke test on synthetic data (no raw data)
make verify-artifact        # check code matches PROVENANCE.json + results.json parse
make paper                  # FULL 5-seed rebuild {42..46} of every table (needs staged raw data)
```

## Tables

| Paper table | `results/` artifact | Regenerate |
|---|---|---|
| T1 Datasets and roles | (static; see `DATASET.md`) | -- |
| T2 OWASP subtype distribution | `results/table_02_dataset_profile/results.json` | `python scripts/61_make_dataset_profile.py` (in `make p1-artifacts`) |
| T3 RQ1 macro-F1 by budget | `results/table_03_rq1_baselines/results.json` | `make table-03` (one seed) / `make paper` (5-seed) |
| T4 RQ2 representation ablations | `results/table_05_field_granularity/results.json` | `make table-05` |
| T5 Per-class F1 (5%/10%) | `results/table_06_per_class_owasp/results.json` | `make table-06` |
| T6 Field-saliency agreement | `results/table_07_attribution/results.json` | `make table-07` |
| T7 Per-stage CPU latency | `results/table_08_latency/results.json` | `make table-08` |
| T8 System profile + baseline latency | `results/table_08_latency/results.json` | `make table-08` |
| T9 External probes + benign FPR | `results/table_09_cross_dataset_biblio_us17/`, `.../table_09_cross_dataset_csic/`, `.../table_11_benign_fpr_apache_indo/` | `scripts/13_cross_dataset_*.py`, `scripts/14_benign_fpr_apache_indo.py` |
| T10 CPU-core equivalents | `results/table_18_deployment_workload/results.json` | `python scripts/65_deployment_workload_table.py` (in `make p1-artifacts`) |

## Figures

| Paper figure | `results/` artifact | Regenerate |
|---|---|---|
| Architecture | (static diagram) | -- |
| Confusion matrix (5%) | `results/table_03_rq1_baselines/` predictions | rendered from Table 3 predictions |
| Per-class F1 (5%) | `results/table_06_per_class_owasp/results.json` | `make table-06` |
| Learning curve | `results/figure_04_learning_curve/figure_data.json` | `python scripts/51_make_learning_curve_figure.py` |
| Latency-accuracy Pareto | `results/figure_03_latency_accuracy_pareto/figure_data.json` | `python scripts/52_make_pareto_figure.py` |

## Supporting artifacts (referenced in text / appendices)

| Item | `results/` artifact | Regenerate |
|---|---|---|
| Paired bootstrap significance (Table 3) | `results/table_03_rq1_baselines/bootstrap_significance*.json` | `make significance` |
| Protocol sensitivity | `results/table_15_protocol_sensitivity/` | `scripts/62_protocol_sensitivity.py` |
| Rare-class CI | `results/table_16_rare_class_ci/` | `scripts/63_rare_class_ci.py` |
| No-custom-rule-444444 control | `results/table_17_no_custom_444444/` | `scripts/66_no_custom_rule_444444.py` |
| OWASP reconciliation audit | `results/table_19_owasp_reconciliation/` | `scripts/67_owasp_reconciliation_audit.py` |
| Human label audit | `results/manual_verification/results.json` | `scripts/42_manual_verification.py` + `scripts/46_human_audit_score.py` |
| FastText dimension sweep | `results/table_10_fasttext_sweep/` | (see `configs/detector/fasttext_base.yaml`) |
| Label-noise (cleanlab) | `results/table_12_label_noise_cleanlab/` | needs `.[label-audit]` extra |
| POST-body pilot | `results/table_13_post_body_pilot/` | pilot analysis |
| Budget-composition CI | `results/table_14_budget_composition_ci/` | budget-composition analysis |
| Environment / versions | `results/environment/results.json` | recorded at run time |

## Provenance and seeds

- The paper's numbers are 5-seed means/CIs over the seed set {42,43,44,45,46};
  only `make paper` reproduces them. A single `make table-0X SEED=42` run gives a
  single-seed cell, not the paper table.
- Each `results/<table>/results.json` records the internal commit that produced
  it; `PROVENANCE.json` bridges those to this public snapshot and to a content
  hash verified by `verify_provenance.py`.

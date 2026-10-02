# Optuna hyperparameter search (standalone experiment)

This directory holds a **separate** hyperparameter search for the proposed
FastText field-aware detector. It does **not** modify the paper pipeline
(`scripts/11_finetune.py`, `scripts/12_evaluate.py`, or `results/table_*`).

## What is optimized

Default scope: **classifier head only** (LinearSVC or LogisticRegression).

| Parameter | Search space |
|-----------|--------------|
| `estimator` | `linear_svc`, `logreg` |
| `C` | log-uniform in `[0.01, 100]` |
| `class_weight` | `null` (un-weighted) or `balanced` |
| `max_iter` | `{500, 1000, 2000, 5000}` |
| `solver` (logreg only) | fixed `lbfgs` (liblinear is invalid for 8-way multiclass) |

The pre-trained FastText model (`models/detector/fasttext/weblog_fasttext.model`)
is loaded **once** per study. Retraining FastText per trial (vector_size, min_n,
max_n, …) is expensive (~minutes per trial) and is **not** enabled by default.

## Data protocol

- **Train fit:** earliest contiguous 5% of time-ordered `owasp_train` (same as paper).
- **Objective:** macro-F1 on `owasp_val` (no test leakage).
- **Final report:** macro-F1 on `owasp_test` using the best validation trial.
- **Seed:** 42 by default (exploratory). The paper aggregates seeds `{42..46}` with 95% CI.

## Prerequisites

```bash
pip install -r requirements.txt -r requirements-optuna.txt
```

Ensure data splits and the FastText checkpoint exist:

```bash
# If missing:
bash scripts/00_download_data.sh
python scripts/01_parse_all.py
python scripts/02_make_splits.py
python scripts/10_train_fasttext.py --config configs/detector/fasttext_base.yaml --seed 42
```

## Run

From the repository root:

```bash
python experiments/optuna/run_study.py --n-trials 25
```

Options:

```bash
python experiments/optuna/run_study.py \
  --config configs/optuna/classifier_5pct.yaml \
  --n-trials 30 \
  --seed 42 \
  --max-records 5000   # optional smoke cap
```

## Outputs

Results are written to `results/optuna/study_<UTC-timestamp>/`:

| File | Contents |
|------|----------|
| `best_params.json` | Best trial params, val/test macro-F1 |
| `summary.json` | Study metadata + best run summary |
| `trials.csv` | One row per trial |
| `study.db` | SQLite Optuna storage |

These artifacts are **not** consumed by `scripts/50_make_paper_tables.py`.

## Config

`configs/optuna/classifier_5pct.yaml` extends `configs/finetune/label_5pct.yaml`
and adds Optuna-specific metadata under the `optuna:` key.

## Multi-seed CI validation

After a study finishes, validate the best trial across the paper seed set
`{42, 43, 44, 45, 46}` and compare against the default baseline
(`C=1.0`, `max_iter=2000`, `class_weight=null` → test macro-F1 65.1,
deterministic across seeds):

```bash
python experiments/optuna/run_ci_validation.py
```

Pin a specific study's best params:

```bash
python experiments/optuna/run_ci_validation.py \
  --best-params results/optuna/study_20260618T043530Z/best_params.json
```

Outputs land in `results/optuna/ci_validation_<UTC-timestamp>/`:

| File | Contents |
|------|----------|
| `per_seed.json` | Val/test macro-F1 per seed for default vs Optuna best |
| `summary.json` | Mean + Student-t 95% CI, comparison vs baseline 65.1, paired bootstrap significance |

The seed-level CI uses the same Student-t method as `src.eval.aggregate` /
`scripts/12_evaluate.py`. Significance vs default is the paired record-level
bootstrap from `src.eval.significance` (same protocol as Table 3 bootstrap).

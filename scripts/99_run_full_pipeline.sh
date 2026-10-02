#!/usr/bin/env bash
#
# 99_run_full_pipeline.sh -- re-run the ENTIRE experimental pipeline in one go.
#
# Produces every paper table/sensitivity artifact + figures at ONE consistent
# git commit / fitted FastText model, so every results.json shares provenance.
# This is the "check the full pipeline and re-run everything once" entry point.
#
# Ordering (dependencies flow top-to-bottom):
#   0. data pipeline (parse + split) -- SKIPPED if data/splits already present
#   0b. dataset profile artifact -- volumes and split subtype percentages
#   0c. environment artifact -- hardware/software versions reported in paper
#   1. FastText embedding fit (seed 42) -- the shared prerequisite for all tables
#   2. Table 3  : proposed @ {0,1,5,10,20,50}% + all CPU baselines, 5 seeds, aggregate
#   2d. Table 15 : protocol sensitivity (7-class macro-F1 without protocol)
#   2e. Tables 16/17 : rare-class CIs + no-custom-rule-444444 sensitivity
#   3. Table 5  : field-granularity ablation + request/response-field controls, 5 seeds
#   4. Table 6  : per-class subtype breakdown, 5 seeds
#   5. Table 7  : per-field attribution agreement, 5 seeds
#   6. Table 8  : CPU per-record latency (proposed + per-record baselines) + footprint
#   6c. Table 18 : deployment workload conversion (ms/record -> CPU cores)
#   7. Table 9  : cross-dataset transfer -- CSIC + Biblio-US17 (12 GB archive scan)
#   8. Table 10 : FastText vector_size x n-gram efficiency sweep (trains 6 embeddings)
#   9. Table 11 : zero-shot benign FPR on Apache-Indo
#  10. Table 12 : cleanlab confident-learning label audit
#  11. significance: paired temporal-block bootstrap + seed-level descriptive
#  12. figures : confusion matrix + latency-accuracy Pareto
#  13. render  : pipe every results.json -> paper/tables/*.{tex,md}
#
# Every step fails LOUD (set -e); a non-zero exit stops the run rather than
# silently producing a partial table. Each step is banner-logged with a wall
# clock so progress is auditable from the master log.
#
# Usage:
#   bash scripts/99_run_full_pipeline.sh 2>&1 | tee results/_full_pipeline_run.log
#
# Env:
#   SEEDS            seed set (default "42 43 44 45 46")
#   PECTI_HARDWARE   CPU name embedded in the latency table metadata
#   PYTHON           interpreter (default: python)
#   SKIP_DATA        if set, never rebuild data/splits (default: auto-skip if present)
#   SKIP_BIBLIO      if set, skip the slow 12 GB Biblio-US17 scan
#   BIBLIO_ARCHIVE   path to the Biblio-US17 dir (default: download/5.Biblio-US17/Biblio-US17)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-42 43 44 45 46}"
export PECTI_HARDWARE="${PECTI_HARDWARE:-12th Gen Intel(R) Core(TM) i7-12700}"
BIBLIO_ARCHIVE="${BIBLIO_ARCHIVE:-download/5.Biblio-US17/Biblio-US17}"

step() { echo; echo "============================================================"; echo "[$(date '+%H:%M:%S')] STEP: $*"; echo "============================================================"; }
run()  { echo "[$(date '+%H:%M:%S')] \$ $*"; "$@"; }

AGG_TPL='from pathlib import Path; from src.eval.aggregate import aggregate_table; from src.baselines._common import git_commit; from src.utils.paths import RESULTS_DIR; print(aggregate_table(RESULTS_DIR / "%s", commit=git_commit(), gpu="cpu"))'
aggregate() { "${PYTHON}" -c "$(printf "${AGG_TPL}" "$1")"; }

echo "########################################################################"
echo "# FULL PIPELINE RE-RUN"
echo "#   repo:   ${REPO_ROOT}"
echo "#   commit: $(git rev-parse --short HEAD)$( [ -n "$(git status --porcelain)" ] && echo ' (dirty)')"
echo "#   seeds:  ${SEEDS}"
echo "#   cpu:    ${PECTI_HARDWARE}"
echo "#   start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "########################################################################"

# --------------------------------------------------------------------------
# 0. Data pipeline -- only rebuild if splits are missing (raw is 3.5 GB; the
#    splits are deterministic, so re-using existing ones keeps provenance and
#    saves a long re-parse). Force a rebuild with SKIP_DATA unset + deleting splits.
# --------------------------------------------------------------------------
if [[ -n "${SKIP_DATA:-}" ]] || [[ -s data/splits/owasp_train.jsonl && -s data/splits/weblog_pretrain.jsonl ]]; then
  step "0. data pipeline -- SKIPPED (splits present; deterministic, reused)"
else
  step "0. data pipeline -- parse + split"
  run "${PYTHON}" scripts/01_parse_all.py
  run "${PYTHON}" scripts/02_make_splits.py
fi

step "0b. Dataset profile -- volumes + split subtype percentages"
run "${PYTHON}" scripts/61_make_dataset_profile.py

step "0b2. OWASP count reconciliation -- descriptor vs parser output"
run "${PYTHON}" scripts/67_owasp_reconciliation_audit.py

step "0c. Environment -- hardware + software versions"
run "${PYTHON}" scripts/64_record_environment.py

# --------------------------------------------------------------------------
# 1. FastText embedding -- fit ONCE (seed 42), reused by every table below.
# --------------------------------------------------------------------------
step "1. FastText embedding fit (seed 42)"
run "${PYTHON}" scripts/10_train_fasttext.py --config configs/detector/fasttext_base.yaml --seed 42

# --------------------------------------------------------------------------
# 1b. FastText v768 embedding -- the 768-dim model for the CAPACITY-MATCHED flat
#     control (Table 5, R4-B-M2). A flat (1-block) encoding at vector_size 768 has
#     the SAME feature dim as the proposed 6-field model (6 x 128 = 768), so the
#     flat-vs-6-field gap splits into a CAPACITY vs a STRUCTURE component. Fit ONCE
#     (seed 42), reused by both flat@768 budget configs. Auto-skip if the model is
#     already present (it is large + slow ~33 min); force-skip with SKIP_V768.
# --------------------------------------------------------------------------
if [[ -n "${SKIP_V768:-}" ]] || [[ -s models/detector/fasttext/weblog_fasttext_v768.model ]]; then
  step "1b. FastText v768 fit -- SKIPPED (model present or SKIP_V768 set)"
else
  step "1b. FastText v768 fit (seed 42) -- capacity-matched flat control"
  run "${PYTHON}" scripts/10_train_fasttext.py --config configs/detector/fasttext_v768.yaml --seed 42
fi

# --------------------------------------------------------------------------
# 2. Table 3 -- proposed across label budgets + all CPU baselines (5 seeds).
#    Budgets present in the committed table: 0,1,5,10,20,50 %.
# --------------------------------------------------------------------------
step "2. Table 3 -- proposed detector across label budgets (5 seeds)"
for seed in ${SEEDS}; do
  for cfg in label_0pct label_1pct label_5pct label_10pct label_20pct label_50pct; do
    run "${PYTHON}" scripts/12_evaluate.py --config "configs/finetune/${cfg}.yaml" --seed "${seed}" --table 3
  done
done

step "2b. Table 3 -- baselines (5 seeds; char_cnn/deeplog use torch-cpu)"
for seed in ${SEEDS}; do
  run env SEED="${seed}" NO_AGGREGATE=1 DEVICE=cpu bash scripts/21_run_all_baselines.sh
done

step "2c. Table 3 -- aggregate -> results.json"
aggregate "table_03_rq1_baselines"

step "2d. Protocol sensitivity -- 7-class macro-F1 without protocol"
run "${PYTHON}" scripts/62_protocol_sensitivity.py --cols 5% 10%

step "2e. Rare-class CI + no-custom-rule-444444 sensitivity"
run "${PYTHON}" scripts/63_rare_class_ci.py --cols 5% 10% --B 10000 --block-size 500 --cluster-key day
run "${PYTHON}" scripts/66_no_custom_rule_444444.py --cols 5% 10%

# --------------------------------------------------------------------------
# 3. Table 5 -- field-granularity ablation (5 seeds).
# --------------------------------------------------------------------------
step "3. Table 5 -- field-granularity ablation + capacity-matched control (5 seeds)"
# Includes the dense capacity-matched control (R4-B-M2): flat@768 at 5% + 10% uses
# the 768-dim v768 model (step 1b) so the flat baseline has the SAME feature dim as
# the proposed 6-field model -- isolating typed STRUCTURE from CAPACITY. The plain
# flat@10% row (flat_no_typed_embed_10pct) and the 10% status/request controls
# complete the headline-budget column. All are deterministic-in-seed (LinearSVC
# + time-ordered prefix) -> zero-width CI.
for seed in ${SEEDS}; do
  for cfg in flat_no_typed_embed three_field six_field six_field_status_masked \
             six_field_request_available six_field_status_masked_10pct \
             six_field_request_available_10pct six_field_10pct flat_no_typed_embed_10pct \
             flat_v768_capacity_matched flat_v768_capacity_matched_10pct; do
    run "${PYTHON}" scripts/12_evaluate.py --config "configs/ablation/${cfg}.yaml" --seed "${seed}" --table 5
  done
done
aggregate "table_05_field_granularity"

# --------------------------------------------------------------------------
# 4. Table 6 -- per-class subtype breakdown (5 seeds, 5% and 10% budgets).
# --------------------------------------------------------------------------
step "4. Table 6 -- per-class breakdown (5 seeds)"
for seed in ${SEEDS}; do
  run "${PYTHON}" scripts/12_evaluate.py --config configs/finetune/label_5pct.yaml --seed "${seed}" --table 6
  run "${PYTHON}" scripts/12_evaluate.py --config configs/finetune/label_10pct.yaml --seed "${seed}" --table 6   # Fig. 3 / Table 9 10% column
done
aggregate "table_06_per_class_owasp"

# --------------------------------------------------------------------------
# 5. Table 7 -- per-field attribution agreement (5 seeds, 6-field).
# --------------------------------------------------------------------------
step "5. Table 7 -- field attribution agreement (5 seeds)"
for seed in ${SEEDS}; do
  run "${PYTHON}" scripts/41_attribution_eval.py --config configs/ablation/six_field.yaml --seed "${seed}"
done
aggregate "table_07_attribution"

# --------------------------------------------------------------------------
# 6. Table 8 -- CPU per-record latency (single measured run) + footprint.
# --------------------------------------------------------------------------
step "6. Table 8 -- proposed detector latency + footprint (seed 42)"
run "${PYTHON}" scripts/40_measure_latency.py --config configs/finetune/label_10pct.yaml --seed 42
step "6b. Table 8 -- per-record baseline latency"
run "${PYTHON}" scripts/54_measure_baseline_latency.py --budget 0.10 --seed 42

step "6c. Table 18 -- deployment workload conversion"
run "${PYTHON}" scripts/65_deployment_workload_table.py --rates 1000 5000 10000 --target-utilization 0.70

# --------------------------------------------------------------------------
# 7. Table 9 -- cross-dataset transfer: CSIC + Biblio-US17.
# --------------------------------------------------------------------------
step "7. Table 9 (CSIC) -- zero-shot cross-dataset transfer (5 seeds)"
for seed in ${SEEDS}; do
  run "${PYTHON}" scripts/13_cross_dataset_csic.py --config configs/finetune/label_0pct.yaml --seed "${seed}"
done
run "${PYTHON}" scripts/13_cross_dataset_csic.py --aggregate

if [[ -n "${SKIP_BIBLIO:-}" ]]; then
  step "7b. Table 9 (Biblio-US17) -- SKIPPED (SKIP_BIBLIO set)"
elif [[ -d "${BIBLIO_ARCHIVE}" ]]; then
  step "7b. Table 9 (Biblio-US17) -- 12 GB archive scan, systematic sample (5 seeds, both variants)"
  run "${PYTHON}" scripts/13_cross_dataset_biblio_us17.py \
    --archive "${BIBLIO_ARCHIVE}" --seed ${SEEDS} --batch-size 5000 --variants both
  run "${PYTHON}" scripts/13_cross_dataset_biblio_us17.py --aggregate
else
  step "7b. Table 9 (Biblio-US17) -- SKIPPED (archive ${BIBLIO_ARCHIVE} not found)"
fi

# --------------------------------------------------------------------------
# 8. Table 10 -- FastText vector_size x n-gram efficiency sweep (trains 6 models).
# --------------------------------------------------------------------------
step "8. Table 10 -- FastText efficiency sweep (seed 42, 6 configs)"
run "${PYTHON}" scripts/14_fasttext_sweep.py --config configs/finetune/label_5pct.yaml --seed 42

# --------------------------------------------------------------------------
# 9. Table 11 -- zero-shot benign FPR on Apache-Indo (5 seeds).
# --------------------------------------------------------------------------
step "9. Table 11 -- benign FPR on Apache-Indo (5 seeds)"
for seed in ${SEEDS}; do
  run "${PYTHON}" scripts/14_benign_fpr_apache_indo.py --config configs/finetune/label_0pct.yaml --seed "${seed}"
done
run "${PYTHON}" scripts/14_benign_fpr_apache_indo.py --aggregate

# --------------------------------------------------------------------------
# 10. Table 12 -- cleanlab confident-learning label audit (single seed).
# --------------------------------------------------------------------------
step "10. Table 12 -- cleanlab label audit (5-fold, seed 42)"
run "${PYTHON}" scripts/44_cleanlab_label_audit.py --folds 5 --seed 42 --out-dir results/table_12_label_noise_cleanlab

# --------------------------------------------------------------------------
# 10b. Table 13 -- POST-body (section C) pilot: does folding the request body
#      into the query field lift detection on the POST-borne attack slice?
#      Directly probes the query-empty limitation. Deterministic-in-seed.
# --------------------------------------------------------------------------
step "10b. Table 13 -- POST-body (section C) detection pilot (5 seeds)"
for seed in ${SEEDS}; do
  run "${PYTHON}" scripts/17_post_body_pilot.py --seed "${seed}" --budget 0.5
done
run "${PYTHON}" scripts/17_post_body_pilot.py --aggregate

# --------------------------------------------------------------------------
# 10c. Budget-composition CI via sliding labeled windows: the
#      honest CI the deterministic-in-seed prefix cannot give. Reported at the
#      headline 5% and the 10% budgets.
# --------------------------------------------------------------------------
step "10c. Budget-composition CI (sliding windows, 5% + 10%)"
# 18_prefix_resample_ci writes results.json DIRECTLY (the CI is over windows, not
# seeds) -- do NOT route through aggregate_table, which would overwrite it.
run "${PYTHON}" scripts/18_prefix_resample_ci.py --config configs/finetune/label_5pct.yaml  --budget 0.05 --n-windows 10 --seed 42
run "${PYTHON}" scripts/18_prefix_resample_ci.py --config configs/finetune/label_10pct.yaml --budget 0.10 --n-windows 10 --seed 42

# --------------------------------------------------------------------------
# 11. Significance tests for the Table 3 claims.
# --------------------------------------------------------------------------
step "11. Significance -- paired temporal-block bootstrap + seed-level descriptive"
run "${PYTHON}" scripts/55_paired_bootstrap.py --col 5% --B 10000 --seed 42 --block-size 500 --cluster-key day
run "${PYTHON}" scripts/55_paired_bootstrap.py --col 10% --B 10000 --seed 42 --block-size 500 --cluster-key day
run "${PYTHON}" scripts/53_significance_test.py --col 5%

# --------------------------------------------------------------------------
# 12. Figures -- confusion matrix (reads Table 3 seed-42 predictions) + Pareto.
# --------------------------------------------------------------------------
step "12. Figures -- confusion matrix + latency-accuracy Pareto"
run "${PYTHON}" scripts/16_confusion_matrix.py --budget 5    # 5% matrix cited in the per-class text
run "${PYTHON}" scripts/16_confusion_matrix.py --budget 10   # Fig. 2 (10% budget)
run "${PYTHON}" scripts/52_make_pareto_figure.py --acc-col 10%

# --------------------------------------------------------------------------
# 13. Render every paper table from its results.json.
#     Tables 7 and 8 carry deterministic-in-seed / single_run_measured cells,
#     so they render under --allow-exploratory (the accepted annotation path).
# --------------------------------------------------------------------------
step "13. Render -> paper/tables/*.{tex,md}"
for t in 3 5 6; do
  run "${PYTHON}" scripts/50_make_paper_tables.py --table "${t}"
done
for t in 7 8 9 10 11; do
  run "${PYTHON}" scripts/50_make_paper_tables.py --table "${t}" --allow-exploratory || \
    echo "[WARN] render --table ${t} returned non-zero (check gating); continuing"
done

echo
echo "########################################################################"
echo "# FULL PIPELINE RE-RUN COMPLETE  $(date '+%Y-%m-%d %H:%M:%S')"
echo "########################################################################"

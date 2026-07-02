#!/usr/bin/env bash
#
# 21_run_all_baselines.sh -- run every Table 3 baseline for a seed.
#
# Thin loop over scripts/20_run_baseline.py for each configs/baselines/<x>.yaml.
# Each run writes ONE per-seed cell file
# results/table_03_rq1_baselines/cells/<row>__<col>__seedNN.json (merge-by-cell:
# nothing is clobbered, so looping baselines x seeds ACCUMULATES). After the loop
# this script aggregates ALL accumulated cell files into the table's results.json
# (mean + 95% CI over the seeds present), which scripts/50_make_paper_tables.py
# renders. Loop this over seeds 42..46 BEFORE the final aggregate for a paper run.
#
# Each baseline is RE-TRAINED / re-fit on THIS access-log corpus (the log-anomaly
# baselines -- DeepLog, LogBERT -- never use off-the-shelf weights), satisfying
# the fairness rule in CLAUDE.md / PROJECT_STRUCTURE.md. (ET-BERT was retired in
# the FastText pivot: it depended on the deleted src.model transformer.)
#
# Usage:
#   bash scripts/21_run_all_baselines.sh                 # seed 42, full corpus
#   SEED=43 bash scripts/21_run_all_baselines.sh         # another seed
#   MAX_RECORDS=2000 bash scripts/21_run_all_baselines.sh  # fast smoke over a slice
#
# Env:
#   SEED         seed to run (default 42; loop externally for the paper set 42..46)
#   MAX_RECORDS  cap records per split (smoke only; unset = full corpus)
#   PYTHON       python interpreter (default: python)
#   DEVICE       torch device for neural baselines (default: cpu)
#   NO_AGGREGATE if set (any value), skip the final aggregation step (e.g. when
#                looping seeds externally and aggregating once at the very end)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cpu}"
CFG_DIR="${REPO_ROOT}/configs/baselines"

# Order: classical first (fast), then ModSec CRS, then the re-trained log-anomaly
# transformers (slower). Matches the build order in PROJECT_STRUCTURE.md section 11.
#
# tfidf_svd_hgbdt is a Table-3 row and MUST be looped here so its cells refresh on
# every run (it was previously omitted, leaving stale cells at an old commit).
#
# logbert is OPT-IN: its 50k-step transformer is not CPU-runnable under our
# protocol (it does not converge in tractable wall-time on CPU, verified), so it
# is omitted from the CPU paper run and disclosed in the Table-3 footnote. Set
# RUN_LOGBERT=1 (with a GPU) to include it.
BASELINES=(
  isolation_forest
  ocsvm
  tfidf_logreg
  tfidf_svd_hgbdt
  field_prefixed_fasttext
  hashing_char_sgd
  status_only
  char_cnn
  modsec_learn
  modsec_advlearn
  deeplog
)
if [[ -n "${RUN_LOGBERT:-}" ]]; then
  BASELINES+=(logbert)
fi

SUPERVISED_BUDGET_BASELINES=(
  tfidf_logreg
  tfidf_svd_hgbdt
  field_prefixed_fasttext
  hashing_char_sgd
  status_only
  char_cnn
  modsec_learn
  modsec_advlearn
)

EXTRA_ARGS=()
if [[ -n "${MAX_RECORDS:-}" ]]; then
  EXTRA_ARGS+=(--max-records "${MAX_RECORDS}")
fi

echo "[21_run_all_baselines] seed=${SEED} device=${DEVICE} max_records=${MAX_RECORDS:-<full>}"

for b in "${BASELINES[@]}"; do
  cfg="${CFG_DIR}/${b}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "[21_run_all_baselines] WARN: missing config ${cfg}; skipping" >&2
    continue
  fi
  echo "=== ${b} ==="
  is_supervised_budget=0
  for sb in "${SUPERVISED_BUDGET_BASELINES[@]}"; do
    if [[ "${b}" == "${sb}" ]]; then
      is_supervised_budget=1
      break
    fi
  done
  if [[ "${is_supervised_budget}" == "1" ]]; then
    for budget in 0.05 0.10; do
      "${PYTHON}" "${SCRIPT_DIR}/20_run_baseline.py" \
        --config "${cfg}" \
        --seed "${SEED}" \
        --budget "${budget}" \
        --device "${DEVICE}" \
        "${EXTRA_ARGS[@]}"
    done
  else
    "${PYTHON}" "${SCRIPT_DIR}/20_run_baseline.py" \
      --config "${cfg}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      "${EXTRA_ARGS[@]}"
  fi
done

echo "[21_run_all_baselines] done. per-seed cells under results/table_03_rq1_baselines/cells/"

if [[ -z "${NO_AGGREGATE:-}" ]]; then
  echo "[21_run_all_baselines] aggregating all cell files -> results.json (mean + 95% CI)"
  "${PYTHON}" -c "from src.eval.aggregate import aggregate_table; from src.baselines._common import git_commit; from src.utils.paths import RESULTS_DIR; print(aggregate_table(RESULTS_DIR / 'table_03_rq1_baselines', commit=git_commit(), gpu='cpu'))"
else
  echo "[21_run_all_baselines] NO_AGGREGATE set; skipping aggregation (aggregate once after the seed sweep)."
fi
echo "[21_run_all_baselines] render with: python scripts/50_make_paper_tables.py --table 3"

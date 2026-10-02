# Makefile for field-aware-waf-detector.
#
# Thin reproducible entry points. Each target just calls a script in scripts/.
# Uses `python` (not python3) per project convention. On Windows `make` may be
# absent; that is fine -- run the underlying `python scripts/...` commands by
# hand, or use WSL / Git Bash. Syntax here is POSIX make.

PYTHON ?= python
SEED   ?= 42

.PHONY: help
help:
	@echo "Verify committed artifact (no recompute, no raw data needed):"
	@echo "  make verify-artifact - check PROVENANCE.json + every results.json parses"
	@echo "Stage raw data (you must place the archives first -- see DATASET.md):"
	@echo "  make stage-data      - unzip staged archives into data/raw/ (does NOT download)"
	@echo "  make data            - stage-data + parse + split (raw -> processed -> splits)"
	@echo "  make parse           - parse raw/ -> processed/ (6 typed fields)"
	@echo "  make splits          - time-ordered splits -> data/splits/"
	@echo "Rebuild the paper from raw data:"
	@echo "  make paper           - 5-seed rebuild {42..46} of the main tables and figures"
	@echo "  make train-fasttext  - train the FastText embedding on the benign weblog corpus"
	@echo "  make table-03 SEED=42 - reproduce ONE seed's cells (NOT the 5-seed paper table)"
	@echo "  make table-05 / table-06 / table-07 / table-08 - one-seed cells for a table"
	@echo "  make p1-artifacts    - Revision-1 analyses (Tables 2, 4, 5, Table 8 lower block, Fig. 2, audits)"
	@echo "  make significance    - significance tests for the Table 3 claims"
	@echo ""
	@echo "NOTE: the single-table targets run ONE seed (SEED=$(SEED)). The paper's"
	@echo "      5-seed means/CIs come from 'make paper' (scripts/99_run_full_pipeline.sh)"
	@echo "      plus 'make p1-artifacts' for the Revision-1 analyses."

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
# stage-data unzips archives YOU place under download/ (see DATASET.md); it does
# not fetch anything from the network. `download` is kept as a back-compat alias.
.PHONY: stage-data download
stage-data download:
	bash scripts/00_download_data.sh

.PHONY: parse
parse:
	$(PYTHON) scripts/01_parse_all.py

.PHONY: splits
splits:
	$(PYTHON) scripts/02_make_splits.py

.PHONY: data
data: stage-data parse splits

# ---------------------------------------------------------------------------
# Verify the committed artifact WITHOUT recomputing (no raw data required).
# Confirms the source matches PROVENANCE.json and every results.json parses.
# Works in the public artifact (verify_provenance.py at root) and in-repo
# (tools/verify_provenance.py).
# ---------------------------------------------------------------------------
.PHONY: verify-artifact
verify-artifact:
	@if [ -f verify_provenance.py ]; then \
	  $(PYTHON) verify_provenance.py; \
	elif [ -f tools/verify_provenance.py ]; then \
	  $(PYTHON) tools/verify_provenance.py; \
	else \
	  echo "verify_provenance.py not found"; exit 1; \
	fi
	@if [ -f validate_results_json.py ]; then \
	  $(PYTHON) validate_results_json.py .; \
	elif [ -f tools/validate_results_json.py ]; then \
	  $(PYTHON) tools/validate_results_json.py .; \
	fi

# ---------------------------------------------------------------------------
# FULL paper rebuild: loops the seed set {42,43,44,45,46} across every table
# and aggregates to the 5-seed means/CIs the manuscript reports. This is the
# ONLY target that reproduces the paper numbers; the per-table targets below
# run a single seed. Requires staged raw data (make data) first.
# ---------------------------------------------------------------------------
.PHONY: paper
paper:
	bash scripts/99_run_full_pipeline.sh

# ---------------------------------------------------------------------------
# FastText embedding ("pre-training") + detector head ("fine-tuning")
# ---------------------------------------------------------------------------
.PHONY: train-fasttext
train-fasttext:
	$(PYTHON) scripts/10_train_fasttext.py --config configs/detector/fasttext_base.yaml --seed $(SEED)

.PHONY: finetune-5pct
finetune-5pct: train-fasttext
	$(PYTHON) scripts/11_finetune.py --config configs/finetune/label_5pct.yaml --seed $(SEED)

# ---------------------------------------------------------------------------
# Paper tables (each traces to results/<table>/results.json)
# ---------------------------------------------------------------------------
# Table 3: the proposed detector across label budgets + the baselines, for ONE
# seed (SEED=$(SEED)). This produces single-seed cells, NOT the paper's 5-seed
# means/CIs -- use `make paper` for those. Runs each budget, the baselines, then
# aggregates and renders.
.PHONY: table-03
table-03: train-fasttext
	$(PYTHON) scripts/12_evaluate.py --config configs/finetune/label_0pct.yaml  --seed $(SEED) --table 3
	$(PYTHON) scripts/12_evaluate.py --config configs/finetune/label_1pct.yaml  --seed $(SEED) --table 3
	$(PYTHON) scripts/12_evaluate.py --config configs/finetune/label_5pct.yaml  --seed $(SEED) --table 3
	$(PYTHON) scripts/12_evaluate.py --config configs/finetune/label_10pct.yaml --seed $(SEED) --table 3 --aggregate
	$(PYTHON) scripts/12_evaluate.py --config configs/finetune/label_20pct.yaml --seed $(SEED) --table 3
	$(PYTHON) scripts/12_evaluate.py --config configs/finetune/label_50pct.yaml --seed $(SEED) --table 3 --aggregate
	bash scripts/21_run_all_baselines.sh
	$(PYTHON) -c "from pathlib import Path; from src.eval.aggregate import aggregate_table; from src.baselines._common import git_commit; aggregate_table(Path('results/table_03_rq1_baselines'), commit=git_commit(), gpu='cpu')"
	$(PYTHON) scripts/62_protocol_sensitivity.py --cols 5% 10%
	$(PYTHON) scripts/63_rare_class_ci.py --cols 5% 10%
	$(PYTHON) scripts/66_no_custom_rule_444444.py --cols 5% 10%
	$(PYTHON) scripts/50_make_paper_tables.py --table 3

# Table 5: FastText field-granularity ablation (flat / 3-field / 6-field).
.PHONY: table-05
table-05: train-fasttext
	$(PYTHON) scripts/12_evaluate.py --config configs/ablation/flat_no_typed_embed.yaml --seed $(SEED) --table 5
	$(PYTHON) scripts/12_evaluate.py --config configs/ablation/three_field.yaml         --seed $(SEED) --table 5
	$(PYTHON) scripts/12_evaluate.py --config configs/ablation/six_field_status_masked.yaml --seed $(SEED) --table 5
	$(PYTHON) scripts/12_evaluate.py --config configs/ablation/six_field_request_available.yaml --seed $(SEED) --table 5
	$(PYTHON) scripts/12_evaluate.py --config configs/ablation/six_field_status_masked_10pct.yaml --seed $(SEED) --table 5
	$(PYTHON) scripts/12_evaluate.py --config configs/ablation/six_field_request_available_10pct.yaml --seed $(SEED) --table 5
	$(PYTHON) scripts/12_evaluate.py --config configs/ablation/six_field_10pct.yaml --seed $(SEED) --table 5
	$(PYTHON) scripts/12_evaluate.py --config configs/ablation/six_field.yaml           --seed $(SEED) --table 5 --aggregate
	$(PYTHON) scripts/50_make_paper_tables.py --table 5

# Table 6: per-class subtype breakdown of the proposed detector.
.PHONY: table-06
table-06: train-fasttext
	$(PYTHON) scripts/12_evaluate.py --config configs/finetune/label_5pct.yaml --seed $(SEED) --table 6
	$(PYTHON) scripts/12_evaluate.py --config configs/finetune/label_10pct.yaml --seed $(SEED) --table 6 --aggregate
	$(PYTHON) scripts/50_make_paper_tables.py --table 6

# Table 7: per-field attribution agreement (6-field detector). --allow-exploratory
# because attribution is deterministic-in-seed; the gate flags that, see
# paper/notes/attribution_table7_caveats.md.
.PHONY: table-07
table-07: train-fasttext
	$(PYTHON) scripts/41_attribution_eval.py --config configs/ablation/six_field.yaml --seed $(SEED) --aggregate
	$(PYTHON) scripts/50_make_paper_tables.py --table 7

# Table 8: CPU per-request latency + footprint. Set PECTI_HARDWARE to the real
# box name so the measurement is attributable; --allow-exploratory because
# latency is a single measured run (single_run_measured, the accepted exception).
.PHONY: table-08
table-08: train-fasttext
	PECTI_HARDWARE="$${PECTI_HARDWARE:-CPU (set PECTI_HARDWARE)}" $(PYTHON) scripts/40_measure_latency.py --config configs/finetune/label_10pct.yaml --seed $(SEED)
	PECTI_HARDWARE="$${PECTI_HARDWARE:-CPU (set PECTI_HARDWARE)}" $(PYTHON) scripts/54_measure_baseline_latency.py --budget 0.10 --seed $(SEED)
	$(PYTHON) scripts/65_deployment_workload_table.py --rates 1000 5000 10000 --target-utilization 0.70
	$(PYTHON) scripts/50_make_paper_tables.py --table 8 --allow-exploratory

.PHONY: p1-artifacts
p1-artifacts:
	$(PYTHON) scripts/61_make_dataset_profile.py
	$(PYTHON) scripts/67_owasp_reconciliation_audit.py
	$(PYTHON) scripts/70_descriptor_gap_filters.py
	$(PYTHON) scripts/69_budget_selection_val.py --aggregate
	$(PYTHON) scripts/69_budget_selection_val.py --per-class-only
	$(PYTHON) scripts/71_val_rce_diagnostic.py
	$(PYTHON) scripts/68_pretrain_corpus_control.py --seed-summary
	$(PYTHON) scripts/72_reparse_bitwise_check.py
	$(PYTHON) scripts/74_intercept_status_audit.py
	$(PYTHON) scripts/75_fpr_wilson_bound.py
	$(PYTHON) scripts/76_audit_weighting_check.py
	$(PYTHON) scripts/77_boundary_id_audit.py
	$(PYTHON) scripts/68_pretrain_corpus_control.py --aggregate
	$(PYTHON) scripts/16_confusion_matrix.py --budget 10
	$(PYTHON) scripts/62_protocol_sensitivity.py --cols 5% 10%
	$(PYTHON) scripts/63_rare_class_ci.py --cols 5% 10%
	$(PYTHON) scripts/66_no_custom_rule_444444.py --cols 5% 10%
	$(PYTHON) scripts/65_deployment_workload_table.py --rates 1000 5000 10000 --target-utilization 0.70

# Significance: paired/one-sample tests of "proposed beats baseline" (Table 3).
# Read-only over the per-seed cells -> results/table_03_rq1_baselines/significance.json
.PHONY: significance
significance:
	$(PYTHON) scripts/55_paired_bootstrap.py --col 5% --B 10000 --seed 42 --block-size 500 --cluster-key day
	$(PYTHON) scripts/55_paired_bootstrap.py --col 10% --B 10000 --seed 42 --block-size 500 --cluster-key day
	$(PYTHON) scripts/53_significance_test.py --col 5%

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
.PHONY: figures
figures:
	@echo "No canonical figure-generation script is currently wired for the FastText flow."

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
# Fast end-to-end "does it run" check on synthetic data (torch-free, no raw
# data). Good first command after install.
.PHONY: smoke
smoke:
	$(PYTHON) scripts/98_smoke_test.py

.PHONY: test
test:
	$(PYTHON) -m pytest

.PHONY: clean
clean:
	$(PYTHON) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)]"

# Field-Aware FastText Triage for Reverse-Proxy Access Logs

Reproduction code and computed results for the paper:

> **Field-Aware FastText Embeddings for CPU-Efficient Triage of
> Application-Layer Attacks in Reverse-Proxy Access Logs**

The method is a **field-aware FastText embedding plus a classical classifier** for
**subtype triage** of preselected suspicious records. A FastText model is trained
self-supervised on benign web access logs; each reverse-proxy record is parsed into
six typed fields (`method, path, query, ua, status, timing`), each field is embedded
and mean-pooled separately, and the six per-field vectors are concatenated into one
record vector. A small classical head (LinearSVC by default) is then fit under a
label budget to classify already-selected suspicious records into CRS-derived
attack subtypes. The model is **CPU-only and torch-free**: it trains in
seconds-to-minutes on a CPU and scores a record in well under a millisecond, sized
for CPU reverse-proxy deployment.

This release is anonymized for double-blind review.

## Repository layout

```text
src/       library code (parsing, FastText embed, field encoder, detector, eval)
scripts/   numbered CLI steps: data -> embed -> fit -> evaluate -> tables/figures
configs/   YAML configs (hyperparameters; never hard-coded in code)
results/   computed results.json for every paper table (+ cells, predictions)
Makefile   reproducible entry points (make data, make table-03, ...)
```

## Setup

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.13, CPU-only. `torch` is required only for the standalone neural
baselines (DeepLog, LogBERT, Char-CNN), not for the proposed detector.

## Verify the committed results (no data, no recompute)

The computed `results/` artifacts are included. To confirm the code matches what
the provenance manifest attests to and that every result parses:

```bash
make verify-artifact
# or directly:
python verify_provenance.py
```

## Data

Raw datasets and trained models are **not** included. `scripts/00_download_data.sh`
is a **staging** step: it unzips archives you place under `download/` -- it does
**not** fetch anything from the network. See `DATASET.md` for the exact archive
filenames, where each must be placed, and the resulting `data/raw/` tree.

## Reproduce the paper numbers

The paper's numbers are 5-seed means/CIs over the seed set {42,43,44,45,46}. Two
targets reproduce them:

```bash
# 0) stage the raw data you obtained per DATASET.md, then build splits
make stage-data      # unzip download/*.zip -> data/raw/   (you provide the zips)
make data            # stage-data + parse + splits

# 1) FULL rebuild: loops all 5 seeds across the main tables and figures
make paper           # == bash scripts/99_run_full_pipeline.sh

# 2) Revision-1 analyses: record counts (Table 2), per-budget class counts and
#    validation results (Tables 4-5), embedding-corpus control (Table 8, lower
#    block), confusion matrix at 10% (Fig. 2), label/count audits
make p1-artifacts
```

A single-table target runs **one** seed and does **not** reproduce the paper's
5-seed table:

```bash
make table-03 SEED=42   # single-seed cells only (also table-05 / 06 / 07 / 08)
```

Every paper number traces to a `results/<table>/results.json`.

## Provenance

Each `results/<table>/results.json` records the *internal* git commit that
produced it. Those commits live in a private working repository and do not exist
in this anonymized snapshot, so they cannot be `git checkout`-ed here. Instead,
`PROVENANCE.json` records the internal commit(s), the public snapshot commit, and
a deterministic content hash over `src/`, `scripts/`, and `configs/`. Verify that
the code in this artifact matches what the manifest attests to:

```bash
python verify_provenance.py        # recomputes the hash; PASS on match
```

## Conventions

- `data/raw/` is immutable; parsers write `data/processed/` deterministically.
- Splits are time-ordered and never shuffled (no future-into-past leakage).
- Hyperparameters live in `configs/` (YAML), never hard-coded.

## License

MIT (see `LICENSE`). If you use this code, please cite the paper (see
`CITATION.cff`).

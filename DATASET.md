# Datasets used by this codebase

This export contains the source, configs, scripts, and the computed `results/`
artifacts. The raw datasets and trained models are NOT included (they are large
and/or licensed). To rebuild from raw data you must obtain each dataset yourself,
place its archive under `download/`, then run `make stage-data` (which unzips into
`data/raw/`) followed by `make data`. Staging does NOT download anything.

| Dataset | Role | Source |
|---|---|---|
| OWASP ModSecurity 30-day | Primary train/val/test (application-layer attacks) | Zenodo 10.5281/zenodo.17178461 |
| Kaggle web-server access logs (zanbil.ir, Jan 2019) | Benign self-supervised pre-training corpus | Kaggle eliasdabbas/web-server-access-logs |
| CSIC 2010 | Cross-dataset transfer / stress probe only | CSIC / Gimenez et al. 2010 |
| Biblio-US17 | Cross-dataset stress probe | University of Seville, 2017 |
| Apache-Indo | Zero-shot benign false-positive-rate probe | public benign Apache access logs |

## Staging: exact archive names and placement

`scripts/00_download_data.sh` expects these archives under `download/` and unzips
them into `data/raw/`. Only OWASP (primary) and the Kaggle weblog (pre-training)
are required to reproduce the headline tables; the others back cross-dataset
probes only.

| Place this file | Obtained from | Unzips to (under `data/raw/`) |
|---|---|---|
| `download/1.owasp.zip` | Zenodo 10.5281/zenodo.17178461 | `owasp_modsec_30day/<DD-Mon-2025>/modsec_audit.anon.log` |
| `download/2.access.log.zip` | Kaggle eliasdabbas/web-server-access-logs | `web_access_logs/access.log` (~3.5 GB) |
| `download/3.csic_database.csv.zip` | CSIC 2010 (Gimenez et al.) | `csic2010/` (optional; cross-dataset only) |

Biblio-US17 and Apache-Indo are used only by the external-probe tables; see the
corresponding `configs/` and `scripts/` for their expected `data/raw/` subpaths.

Expected tree after `make stage-data`:

```text
data/raw/
  owasp_modsec_30day/
    01-Aug-2025/modsec_audit.anon.log
    ... (30 day directories)
  web_access_logs/
    access.log
  csic2010/            # optional
```

> Checksums/sizes: the archives are third-party and versioned upstream; verify
> against the source pages above. The pipeline is deterministic, so a correct
> stage reproduces `data/processed/` and the splits bit-for-bit.

## Notes

- `data/raw/` is immutable: parsers read it and write `data/processed/`.
  Re-running a parser reproduces `processed/` bit-for-bit (deterministic JSON:
  sorted keys, fixed float formatting).
- Splits are time-ordered (e.g. OWASP days 1-20 train, 21-24 val, 25-30 test);
  they are never shuffled.
- Every number in the paper traces to a `results/<table>/results.json`; absolute
  paths inside those artifacts were rewritten to `<REPO>`.
- The proposed detector is CPU-only and torch-free; `torch` is needed only for
  the standalone neural baselines.

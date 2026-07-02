#!/usr/bin/env bash
#
# 00_download_data.sh -- STAGE (unzip) raw datasets into data/raw/.
#
# NOTE: this is a STAGING step, not a downloader. It fetches NOTHING from the
# network. You must obtain each dataset yourself (licensed / large) and place its
# archive under download/ before running this; see DATASET.md for exact filenames,
# sources, and the resulting data/raw/ tree. `make stage-data` invokes this.
#
# Thin, IDEMPOTENT: the download/ folder at the repo root is a staging area
# (gitignored) holding archives fetched out-of-band; this script unpacks them into
# the immutable data/raw/ tree. Re-running is safe: if the expected raw/ layout
# already exists the unpack is skipped.
#
# Datasets:
#   OWASP ModSec 30-day  (PRIMARY)  download/1.owasp.zip
#       -> data/raw/owasp_modsec_30day/<DD-Mon-2025>/modsec_audit.anon.log
#   Kaggle web access logs (PRE-TRAIN)  download/2.access.log.zip
#       -> data/raw/web_access_logs/access.log
#   CSIC 2010                       download/3.csic_database.csv.zip (TODO)
#       -> data/raw/csic2010/
#
# Usage:  bash scripts/00_download_data.sh
set -euo pipefail

# Resolve repo root from this script's location (works regardless of CWD).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DOWNLOAD_DIR="${REPO_ROOT}/download"
RAW_DIR="${REPO_ROOT}/data/raw"

OWASP_ZIP="${DOWNLOAD_DIR}/1.owasp.zip"
OWASP_RAW="${RAW_DIR}/owasp_modsec_30day"

WEBLOG_ZIP="${DOWNLOAD_DIR}/2.access.log.zip"
WEBLOG_RAW="${RAW_DIR}/web_access_logs"
WEBLOG_LOG="${WEBLOG_RAW}/access.log"

CSIC_ZIP="${DOWNLOAD_DIR}/3.csic_database.csv.zip"
CSIC_RAW="${RAW_DIR}/csic2010"

mkdir -p "${RAW_DIR}"

# ---------------------------------------------------------------------------
# OWASP ModSec 30-day (primary dataset)
# ---------------------------------------------------------------------------
echo "[00_download_data] OWASP ModSec 30-day ..."
if [ -d "${OWASP_RAW}" ] && [ -n "$(find "${OWASP_RAW}" -name 'modsec_audit.anon.log' -print -quit 2>/dev/null)" ]; then
  echo "  already staged at ${OWASP_RAW} (found modsec_audit.anon.log) -- skipping unzip."
elif [ -f "${OWASP_ZIP}" ]; then
  echo "  unzipping ${OWASP_ZIP} -> ${OWASP_RAW}"
  mkdir -p "${OWASP_RAW}"
  # -o overwrite, -q quiet, -d destination. The zip's top level is the
  # per-day directories (e.g. 01-Aug-2025/modsec_audit.anon.log).
  unzip -o -q "${OWASP_ZIP}" -d "${OWASP_RAW}"
  echo "  done. Day directories:"
  find "${OWASP_RAW}" -maxdepth 1 -mindepth 1 -type d | sort | sed 's/^/    /'
else
  echo "  WARNING: ${OWASP_ZIP} not found and ${OWASP_RAW} is empty."
  echo "           Place the OWASP archive at ${OWASP_ZIP} (Zenodo 10.5281/zenodo.17178461)."
fi

# ---------------------------------------------------------------------------
# Kaggle "Web Server Access Logs" (benign self-supervised pre-training corpus)
# zanbil.ir e-commerce, Jan 2019; standard Apache/NGINX combined-format log.
# ---------------------------------------------------------------------------
echo "[00_download_data] Kaggle web access logs (pre-training corpus) ..."
if [ -f "${WEBLOG_LOG}" ]; then
  echo "  already staged at ${WEBLOG_LOG} (the ~3.5GB access.log) -- skipping unzip."
elif [ -f "${WEBLOG_ZIP}" ]; then
  echo "  unzipping ${WEBLOG_ZIP} -> ${WEBLOG_RAW}/access.log"
  mkdir -p "${WEBLOG_RAW}"
  # -o overwrite, -q quiet, -d destination. The zip contains a single
  # combined-format access.log.
  unzip -o -q "${WEBLOG_ZIP}" -d "${WEBLOG_RAW}"
  echo "  done."
else
  echo "  WARNING: ${WEBLOG_ZIP} not found and ${WEBLOG_LOG} is absent."
  echo "           Source: Kaggle eliasdabbas/web-server-access-logs (zanbil.ir, Jan 2019)."
fi

# ---------------------------------------------------------------------------
# CSIC 2010 (cross-dataset transfer only) -- TODO
# ---------------------------------------------------------------------------
echo "[00_download_data] CSIC 2010 ..."
if [ -d "${CSIC_RAW}" ] && [ -n "$(ls -A "${CSIC_RAW}" 2>/dev/null)" ]; then
  echo "  already staged at ${CSIC_RAW} -- skipping."
elif [ -f "${CSIC_ZIP}" ]; then
  echo "  unzipping ${CSIC_ZIP} -> ${CSIC_RAW}"
  mkdir -p "${CSIC_RAW}"
  # CSV form of CSIC 2010; the parser (src/data/parsers/csic.py) reads whatever
  # CSV/log files land here. CSIC is OPTIONAL (cross-dataset transfer only).
  unzip -o -q "${CSIC_ZIP}" -d "${CSIC_RAW}"
  echo "  done."
else
  echo "  NOTE: CSIC archive (${CSIC_ZIP}) not present -- skipping (optional; cross-dataset only)."
  echo "        Source: CSIC / Gimenez et al. 2010. Place at ${CSIC_ZIP} to enable."
fi

echo "[00_download_data] done."

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ge 1 && "$1" != -* ]]; then
  SUMMARY_CSV="$1"
  shift
else
  SUMMARY_CSV="${ROOT_DIR}/Results/summary.csv"
fi

if [[ $# -ge 1 && "$1" != -* ]]; then
  EXAMPLES_DIR="$1"
  shift
else
  EXAMPLES_DIR="${ROOT_DIR}/Examples"
fi

if [[ $# -ge 1 && "$1" != -* ]]; then
  OUTPUT_DIR="$1"
  shift
else
  OUTPUT_DIR="${ROOT_DIR}/Results"
fi

PYTHON_CMD="/Users/qzhu8/miniconda3/bin/python"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "[ERROR] Summary CSV not found: ${SUMMARY_CSV}" >&2
  exit 1
fi

if [[ ! -d "${EXAMPLES_DIR}" ]]; then
  echo "[ERROR] Examples directory not found: ${EXAMPLES_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Python: ${PYTHON_CMD}"
echo "[INFO] Summary CSV: ${SUMMARY_CSV}"
echo "[INFO] Examples dir: ${EXAMPLES_DIR}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"

exec "${PYTHON_CMD}" "${ROOT_DIR}/PXRD_agent_resume.py" \
  --summary-csv "${SUMMARY_CSV}" \
  --examples-dir "${EXAMPLES_DIR}" \
  --output "${OUTPUT_DIR}" \
  "$@"
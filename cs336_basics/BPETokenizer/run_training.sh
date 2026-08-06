#!/usr/bin/env bash

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
UTC_TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
LOG_FILE="$LOG_DIR/BPEtokenizer_${UTC_TIMESTAMP}.log"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$LOG_DIR"

echo "Log: $LOG_FILE"
"$PYTHON_BIN" -u "$SCRIPT_DIR/BPEtokenizer.py" "$@" 2>&1 | tee "$LOG_FILE"


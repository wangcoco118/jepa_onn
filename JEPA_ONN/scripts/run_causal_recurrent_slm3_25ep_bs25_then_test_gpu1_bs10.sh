#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/linux/anaconda3/envs/semseg/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python runtime is missing: $PYTHON" >&2
    exit 1
fi
exec "$PYTHON" -u "$SCRIPT_DIR/run_causal_recurrent_slm3_25ep_bs25_then_test_gpu1_bs10.py" "$@"

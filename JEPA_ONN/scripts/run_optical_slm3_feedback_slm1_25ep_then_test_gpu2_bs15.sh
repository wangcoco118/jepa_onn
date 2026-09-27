#!/usr/bin/env bash
set -euo pipefail

REPO="/data/linux/wkx/IntPhys/references_code/jepa_onn/JEPA_ONN"
OUTPUT_ROOT="/data/linux/wkx/IntPhys/references_code/jepa_onn/output"
PYTHON="/home/linux/anaconda3/envs/semseg/bin/python"
TRAIN_CONFIG="$REPO/evals/intuitive_physics/configs/onn_feedback_optical_slm3_feedback_slm1_25ep_bs15.yaml"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_NAME="onn_optical_slm3_feedback_slm1_output_25ep_gpu2_bs15_$RUN_STAMP"
PIPELINE_LOG="$OUTPUT_ROOT/$BASE_NAME.pipeline.log"
TRAIN_LAUNCHER_LOG="$OUTPUT_ROOT/$BASE_NAME.train_launcher.log"
CURRENT_USER="$(id -un)"
TMP_ROOT="$OUTPUT_ROOT/.runtime_tmp_$CURRENT_USER"
CACHE_ROOT="$OUTPUT_ROOT/.runtime_cache_$CURRENT_USER"

if [[ "$CURRENT_USER" != "wkx" && "$CURRENT_USER" != "linux" ]]; then
    echo "ERROR: expected remote user wkx or linux, got $CURRENT_USER" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$TMP_ROOT" "$CACHE_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

if [[ ! -f "$TRAIN_CONFIG" ]]; then
    echo "ERROR: missing training config: $TRAIN_CONFIG" >&2
    exit 1
fi

cd "$REPO"
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
export TMPDIR="$TMP_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"

echo "PIPELINE_START physical_gpu=2 logical_device=cuda:0 train_epochs=25 train_batch_size=15 test_batch_size=15 output_mode=optical optical_output_slm_layers=2 onn_core_slm_layers=3 feedback_layer_index=1 readout_mode=learnable_offset feedback_sign=1.0 test_resume_every_batches=20"
nvidia-smi -i 2 --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

"$PYTHON" evals/intuitive_physics/train_optical.py \
    --config "$TRAIN_CONFIG" \
    --output "$BASE_NAME" \
    --experiment-mode onn_feedback \
    --gpu 0 \
    --epochs 25 \
    --batch-size 15 \
    --learning-rate 0.0001 \
    --skip-final-eval \
    2>&1 | tee "$TRAIN_LAUNCHER_LOG"

RUN_DIR="$(
    find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "$BASE_NAME"'_*' -printf '%T@ %p\n' \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-
)"
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "ERROR: training run directory was not found for $BASE_NAME" >&2
    exit 1
fi

BEST_CHECKPOINT="$RUN_DIR/$BASE_NAME.pt"
TRAIN_LOG="$BEST_CHECKPOINT.log"
if [[ ! -f "$BEST_CHECKPOINT" ]]; then
    echo "ERROR: best checkpoint not found: $BEST_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_LOG" ]] || ! grep -q "run_done experiment_mode=onn_feedback" "$TRAIN_LOG"; then
    echo "ERROR: training RUN_DONE marker not found: $TRAIN_LOG" >&2
    exit 1
fi

TEST_OUTPUT="$RUN_DIR/test_eval_full_o1_o2_o3_gpu2_bs15"
TEST_CONFIG="$RUN_DIR/test_eval_full_o1_o2_o3_gpu2_bs15.yaml"
TEST_RUNNER="$RUN_DIR/run_intphys_test_from_yaml.py"
mkdir -p "$TEST_OUTPUT"

"$PYTHON" - "$TRAIN_CONFIG" "$BEST_CHECKPOINT" "$TEST_OUTPUT" "$TEST_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

train_config_path, checkpoint_path, output_dir, test_config_path = map(Path, sys.argv[1:])
with train_config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["predictor_checkpoint"] = str(checkpoint_path.resolve())
config["output_dir"] = str(output_dir.resolve())
config["dataset"] = "intphys-test"
config["eval_name"] = "intphys_test"
config["mode"] = "all"
config["nodes"] = 1
config["tasks_per_node"] = 1
config.setdefault("data", {})["batch_size"] = 15
config["test_resume"] = {
    "enabled": True,
    "checkpoint_path": str((output_dir / "test_resume_latest.pt").resolve()),
    "save_every_batches": 20,
}

with test_config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
PY

cat > "$TEST_RUNNER" <<'PY'
import sys
from pathlib import Path

import yaml

from evals.intphys_test import eval as test_eval


def main():
    config_path = Path(sys.argv[1])
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    test_eval.main(config)


if __name__ == "__main__":
    main()
PY

"$PYTHON" -m py_compile "$TEST_RUNNER"
"$PYTHON" "$TEST_RUNNER" "$TEST_CONFIG" 2>&1 | tee "$TEST_OUTPUT/launcher.log"

INFERENCE_LOG="$TEST_OUTPUT/inference.log"
if [[ ! -f "$INFERENCE_LOG" ]] || ! grep -q "run_done output_dir=" "$INFERENCE_LOG"; then
    echo "ERROR: Test RUN_DONE marker not found: $INFERENCE_LOG" >&2
    exit 1
fi

TEST_RESULT_DIR="$(
    find "$TEST_OUTPUT" -type f -name "average_surprise_answer.txt" -printf '%h\n' \
        | sort \
        | head -n 1
)"
if [[ -z "$TEST_RESULT_DIR" || ! -d "$TEST_RESULT_DIR" ]]; then
    echo "ERROR: complete-Test result directory was not found under $TEST_OUTPUT" >&2
    exit 1
fi

for required_file in \
    "$TEST_RESULT_DIR/average_surprise_answer.txt" \
    "$TEST_RESULT_DIR/maximum_surprise_answer.txt" \
    "$TEST_RESULT_DIR/losses_2fs_2_4_6_8_10ctxt.pth"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: missing complete-Test artifact: $required_file" >&2
        exit 1
    fi
done

for answer_file in \
    "$TEST_RESULT_DIR/average_surprise_answer.txt" \
    "$TEST_RESULT_DIR/maximum_surprise_answer.txt"; do
    rows="$(wc -l < "$answer_file")"
    if [[ "$rows" -ne 12960 ]]; then
        echo "ERROR: expected 12960 rows in $answer_file, got $rows" >&2
        exit 1
    fi
done

echo "PIPELINE_DONE run_dir=$RUN_DIR best_checkpoint=$BEST_CHECKPOINT test_output=$TEST_OUTPUT result_dir=$TEST_RESULT_DIR"

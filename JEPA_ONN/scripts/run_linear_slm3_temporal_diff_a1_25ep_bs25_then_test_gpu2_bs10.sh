#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/data/linux/wkx/IntPhys/references_code/jepa_onn/JEPA_ONN"
BASE_RUN="/data/linux/wkx/IntPhys/references_code/jepa_onn/output/9.17-9.26/差分输入/a0.5/linear_slm3_feedback_slm1_temporal_diff_a0.5_20260922_192812"
OUTPUT_ROOT="/data/linux/wkx/IntPhys/references_code/jepa_onn/output"
PYTHON="/home/linux/anaconda3/envs/semseg/bin/python"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_NAME="linear_slm3_feedback_slm1_temporal_diff_a1_25ep_bs25_gpu2_${RUN_STAMP}"
PIPELINE_LOG="$OUTPUT_ROOT/$BASE_NAME.pipeline.log"
TRAIN_LAUNCHER_LOG="$OUTPUT_ROOT/$BASE_NAME.train_launcher.log"
CURRENT_USER="$(id -un)"
TMP_ROOT="$OUTPUT_ROOT/.runtime_tmp_$CURRENT_USER"
CACHE_ROOT="$OUTPUT_ROOT/.runtime_cache_$CURRENT_USER"
CONFIG_ROOT="$OUTPUT_ROOT/.runtime_configs_$CURRENT_USER"
CONFIG_STAGING="$CONFIG_ROOT/$BASE_NAME.yaml"

if [[ "$CURRENT_USER" != "wkx" && "$CURRENT_USER" != "linux" ]]; then
    echo "ERROR: expected remote user wkx or linux, got $CURRENT_USER" >&2
    exit 1
fi
if [[ ! -d "$BASE_RUN" ]]; then
    echo "ERROR: source experiment directory does not exist: $BASE_RUN" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$TMP_ROOT" "$CACHE_ROOT" "$CONFIG_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1
cd "$REPO"
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
export TMPDIR="$TMP_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"

echo "PIPELINE_START physical_gpu=2 logical_device=cuda:0 fresh_training=true train_epochs=25 train_batch_size=25 test_batch_size=10 num_slm_layers=3 feedback_layer_index=1 temporal_difference_alpha=1.0"
nvidia-smi -i 2 --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

echo "等待物理 GPU2 空闲..."
while true; do
    BUSY_PIDS="$(
        nvidia-smi pmon -c 1 2>/dev/null |
        awk '$1 == 2 && $2 ~ /^[0-9]+$/ && $3 == "C" {print $2}' |
        tr '
' ' '
    )"
    [[ -z "$BUSY_PIDS" ]] && break
    echo "GPU2 当前被占用，PID: $BUSY_PIDS; 60秒后重试"
    sleep 60
done

"$PYTHON" - "$BASE_RUN" "$CONFIG_STAGING" <<'PY'
import copy
import sys
from pathlib import Path
import torch
import yaml

base_run = Path(sys.argv[1])
config_path = Path(sys.argv[2])
candidates = []
for path in base_run.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix.lower() in {".log", ".yaml", ".yml", ".json", ".txt", ".zip"}:
        continue
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        continue
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        continue
    kind = str(checkpoint.get("checkpoint_kind", "")).lower()
    name = path.name.lower()
    is_best = kind == "best" or (
        kind not in {"last", "final"} and "last" not in name and "final" not in name
    )
    candidates.append((
        is_best,
        int(checkpoint.get("epoch", -1)),
        path.stat().st_mtime,
        path,
        checkpoint,
    ))

if not candidates:
    raise RuntimeError(f"source checkpoint not found: {base_run}")
candidates.sort(key=lambda item: item[:3], reverse=True)
_, _, _, source_checkpoint_path, source_checkpoint = candidates[0]
config = copy.deepcopy(source_checkpoint["config"])

onn_config = config.setdefault("onn", {})
actual_layers = int(onn_config.get("num_slm_layers", -1))
actual_feedback = int(onn_config.get("feedback_layer_index", -1))
if actual_layers != 3:
    raise ValueError(f"source checkpoint is not SLM3: {actual_layers}")
if actual_feedback != 1:
    raise ValueError(f"source feedback layer is not index 1: {actual_feedback}")

predictor_config = config.setdefault("predictor", {})
if not bool(predictor_config.get("temporal_difference_enabled", False)):
    raise ValueError("source checkpoint does not enable temporal difference")
source_alpha = float(predictor_config.get("temporal_difference_alpha", float("nan")))
if source_alpha != 0.5:
    raise ValueError(f"source alpha is not 0.5: {source_alpha}")

predictor_config["temporal_difference_alpha"] = 1.0
config.setdefault("data", {})["batch_size"] = 25
config.setdefault("training", {})["epochs"] = 25
config["tag"] = (
    "vjepa_vitl16_onn_feedback_linear_slm3_feedback_slm1_"
    "temporal_diff_a1_25ep"
)
config.setdefault("pretrain", {})["write_tag"] = config["tag"]

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

print("source_checkpoint =", source_checkpoint_path)
print("source_checkpoint_kind =", source_checkpoint.get("checkpoint_kind"))
print("source_checkpoint_epoch =", source_checkpoint.get("epoch"))
print("fresh_training =", True)
print("resume_checkpoint =", None)
print("num_slm_layers =", actual_layers)
print("feedback_layer_index =", actual_feedback)
print("source_temporal_difference_alpha =", source_alpha)
print("new_temporal_difference_alpha =", 1.0)
print("train_batch_size =", config["data"]["batch_size"])
print("train_epochs =", config["training"]["epochs"])
print("generated_config =", config_path)
PY

"$PYTHON" evals/intuitive_physics/train_optical.py     --config "$CONFIG_STAGING"     --output "$BASE_NAME"     --experiment-mode onn_feedback     --gpu 0     --epochs 25     --batch-size 25     --learning-rate 0.0001     --skip-final-eval     2>&1 | tee "$TRAIN_LAUNCHER_LOG"

RUN_DIR="$(
    find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "$BASE_NAME"'_*' -printf '%T@ %p
'         | sort -n | tail -n 1 | cut -d' ' -f2-
)"
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "ERROR: training run directory was not found for $BASE_NAME" >&2
    exit 1
fi
TRAIN_CONFIG="$RUN_DIR/$BASE_NAME.yaml"
mv "$CONFIG_STAGING" "$TRAIN_CONFIG"
BEST_CHECKPOINT="$RUN_DIR/$BASE_NAME.pt"
TRAIN_LOG="$BEST_CHECKPOINT.log"
if [[ ! -f "$BEST_CHECKPOINT" ]]; then
    echo "ERROR: validation-best checkpoint not found: $BEST_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_LOG" ]] || ! grep -q "run_done experiment_mode=onn_feedback" "$TRAIN_LOG"; then
    echo "ERROR: training RUN_DONE marker not found: $TRAIN_LOG" >&2
    exit 1
fi

TEST_OUTPUT="$RUN_DIR/test_eval_full_o1_o2_o3_gpu2_bs10"
TEST_CONFIG="$RUN_DIR/test_eval_full_o1_o2_o3_gpu2_bs10.yaml"
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
config.setdefault("data", {})
config["data"]["batch_size"] = 10
config.setdefault("evaluation", {})["last_context_copy_baseline"] = False
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
    find "$TEST_OUTPUT" -type f -name "average_surprise_answer.txt" -printf '%h
'         | sort | head -n 1
)"
if [[ -z "$TEST_RESULT_DIR" || ! -d "$TEST_RESULT_DIR" ]]; then
    echo "ERROR: complete-Test result directory was not found under $TEST_OUTPUT" >&2
    exit 1
fi
for required_file in     "$TEST_RESULT_DIR/average_surprise_answer.txt"     "$TEST_RESULT_DIR/maximum_surprise_answer.txt"     "$TEST_RESULT_DIR/losses_2fs_2_4_6_8_10ctxt.pth"; do
    [[ -f "$required_file" ]] || {
        echo "ERROR: missing complete-Test artifact: $required_file" >&2
        exit 1
    }
done
for answer_file in     "$TEST_RESULT_DIR/average_surprise_answer.txt"     "$TEST_RESULT_DIR/maximum_surprise_answer.txt"; do
    rows="$(wc -l < "$answer_file")"
    if [[ "$rows" -ne 12960 ]]; then
        echo "ERROR: expected 12960 rows in $answer_file, got $rows" >&2
        exit 1
    fi
done

echo "PIPELINE_DONE run_dir=$RUN_DIR best_checkpoint=$BEST_CHECKPOINT test_output=$TEST_OUTPUT result_dir=$TEST_RESULT_DIR"

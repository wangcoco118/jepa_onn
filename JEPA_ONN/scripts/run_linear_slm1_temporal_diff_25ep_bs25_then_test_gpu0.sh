#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/data/linux/wkx/IntPhys/references_code/jepa_onn/JEPA_ONN"
BASE_RUN="/data/linux/wkx/IntPhys/references_code/jepa_onn/output/9.9_9.16/减少层数并且修改MLP/linear_slm1_25ep_20260905_232655"
OUTPUT_ROOT="/data/linux/wkx/IntPhys/references_code/jepa_onn/output/9.9_9.16/减少层数并且修改MLP"
PYTHON="/home/linux/anaconda3/envs/semseg/bin/python"

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="linear_slm1_temporal_diff_a0.5_25ep_bs25_gpu0_${STAMP}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
CONFIG="${RUN_DIR}/${RUN_NAME}.yaml"
TRAIN_LOG="${RUN_DIR}/train.log"
TEST_LOG="${RUN_DIR}/test.log"
TEST_DIR="${RUN_DIR}/test_eval_full_o1_o2_o3_gpu0_bs15"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}"
export TMPDIR="${RUN_DIR}/tmp"
export XDG_CACHE_HOME="${RUN_DIR}/cache"

mkdir -p "${RUN_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}"
cd "${REPO}"

echo "等待物理 GPU0 空闲..."
while true; do
    BUSY_PIDS=$(
        nvidia-smi pmon -c 1 2>/dev/null |
        awk '$1 == 0 && $2 ~ /^[0-9]+$/ && $3 == "C" {print $2}' |
        tr '\n' ' '
    )

    if [[ -z "${BUSY_PIDS}" ]]; then
        break
    fi

    echo "GPU0 当前被占用，PID: ${BUSY_PIDS}; 60秒后重试"
    sleep 60
done

echo "GPU0 已空闲，生成训练配置..."

"${PYTHON}" - "${BASE_RUN}" "${CONFIG}" <<'PY'
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
    if path.suffix.lower() in {".log", ".yaml", ".yml", ".json", ".txt"}:
        continue

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception:
        continue

    if not isinstance(checkpoint, dict):
        continue
    if not isinstance(checkpoint.get("config"), dict):
        continue

    kind = str(checkpoint.get("checkpoint_kind", "")).lower()
    name = path.name.lower()

    is_best = kind == "best" or "last" not in name
    epoch = int(checkpoint.get("epoch", -1))

    candidates.append((is_best, epoch, path, checkpoint))

if not candidates:
    raise RuntimeError(f"未找到可读取的源 checkpoint: {base_run}")

candidates.sort(
    key=lambda item: (
        item[0],
        item[1],
        item[2].stat().st_mtime,
    ),
    reverse=True,
)

_, _, source_checkpoint_path, source_checkpoint = candidates[0]
config = copy.deepcopy(source_checkpoint["config"])

onn_config = config.setdefault("onn", {})
if int(onn_config.get("num_slm_layers", -1)) != 1:
    raise ValueError(
        "源 checkpoint 不是 SLM1，实际 num_slm_layers="
        f"{onn_config.get('num_slm_layers')}"
    )

predictor_config = config.setdefault("predictor", {})
predictor_config["temporal_difference_enabled"] = True
predictor_config["temporal_difference_alpha"] = 0.5

config.setdefault("data", {})
config["data"]["batch_size"] = 25

config.setdefault("training", {})
config["training"]["epochs"] = 25

config["tag"] = "vjepa_vitl16_onn_feedback_linear_slm1_temporal_diff_a0.5"

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(
        config,
        f,
        sort_keys=False,
        allow_unicode=True,
    )

print("source_checkpoint =", source_checkpoint_path)
print("config_path =", config_path)
print("temporal_difference_enabled =",
      predictor_config["temporal_difference_enabled"])
print("temporal_difference_alpha =",
      predictor_config["temporal_difference_alpha"])
print("num_slm_layers =",
      onn_config["num_slm_layers"])
print("batch_size =", config["data"]["batch_size"])
print("epochs =", config["training"]["epochs"])
PY

echo "开始训练..."

set +e
"${PYTHON}" evals/intuitive_physics/train_optical.py \
    --config "${CONFIG}" \
    --output "${RUN_DIR}" \
    --experiment-mode onn_feedback \
    --gpu 0 \
    --epochs 25 \
    --batch-size 25 \
    --skip-final-eval \
    2>&1 | tee "${TRAIN_LOG}"

TRAIN_RC=${PIPESTATUS[0]}
set -e

if [[ "${TRAIN_RC}" -ne 0 ]]; then
    echo "训练失败，退出码: ${TRAIN_RC}"
    echo "训练日志: ${TRAIN_LOG}"
    exit "${TRAIN_RC}"
fi

echo "训练完成，定位 best checkpoint..."

"${PYTHON}" - "${RUN_DIR}" "${TEST_DIR}" <<'PY' 2>&1 | tee "${TEST_LOG}"
import copy
import sys
from pathlib import Path

import torch

run_dir = Path(sys.argv[1])
test_dir = Path(sys.argv[2])

candidates = []

for path in run_dir.rglob("*"):
    if not path.is_file():
        continue
    if path.name.endswith(".log"):
        continue
    if path.name.endswith((".yaml", ".yml", ".json", ".txt")):
        continue

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception:
        continue

    if not isinstance(checkpoint, dict):
        continue
    if not isinstance(checkpoint.get("config"), dict):
        continue

    kind = str(checkpoint.get("checkpoint_kind", "")).lower()
    name = path.name.lower()
    is_best = kind == "best" or "last" not in name
    epoch = int(checkpoint.get("epoch", -1))

    candidates.append((is_best, epoch, path, checkpoint))

if not candidates:
    raise RuntimeError(f"训练输出中未找到 checkpoint: {run_dir}")

candidates.sort(
    key=lambda item: (
        item[0],
        item[1],
        item[2].stat().st_mtime,
    ),
    reverse=True,
)

_, _, best_checkpoint_path, best_checkpoint = candidates[0]

config = copy.deepcopy(best_checkpoint["config"])
config["predictor_checkpoint"] = str(best_checkpoint_path)
config["dataset"] = "intphys-test"
config["eval_name"] = "intphys_test"
config["mode"] = "all"

config.setdefault("data", {})
config["data"]["batch_size"] = 15

config.setdefault("evaluation", {})
config["evaluation"]["last_context_copy_baseline"] = False

test_dir.mkdir(parents=True, exist_ok=True)
config["output_dir"] = str(test_dir)

print("best_checkpoint =", best_checkpoint_path)
print("checkpoint_epoch =", best_checkpoint.get("epoch"))
print("test_batch_size =", config["data"]["batch_size"])
print("test_mode =", config["mode"])
print("test_output_dir =", test_dir)

from evals.intphys_test import eval as evaluator

evaluator.main(config)
PY

TEST_RC=${PIPESTATUS[0]}

if [[ "${TEST_RC}" -ne 0 ]]; then
    echo "测试失败，退出码: ${TEST_RC}"
    echo "测试日志: ${TEST_LOG}"
    exit "${TEST_RC}"
fi

echo "训练和完整测试全部完成"
echo "RUN_DIR=${RUN_DIR}"
echo "TEST_DIR=${TEST_DIR}"

"""Train causal SLM3 for 25 epochs, then evaluate the exact last checkpoint."""

import argparse
import copy
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

import yaml

REPO = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO.parent / "output"
SOURCE_CONFIG = REPO / "evals/intuitive_physics/configs/onn_causal_recurrent_slm3.yaml"
PYTHON = Path("/home/linux/anaconda3/envs/semseg/bin/python")
TEST_RUNNER = OUTPUT_ROOT / "run_intphys_test_from_yaml.py"


def training_command(repo, config_path, run_name, python):
    return [
        str(python), "-u", "-m", "evals.intuitive_physics.train_optical",
        "--config", str(config_path), "--output", run_name,
        "--experiment-mode", "onn_causal_recurrent",
        "--gpu", "0", "--epochs", "25", "--batch-size", "25",
        "--learning-rate", "1e-4", "--checkpoint-selection", "last_epoch",
        "--skip-final-eval",
    ]


def last_checkpoint_from_log(text, output_root, run_name):
    lines = [line for line in text.splitlines()
             if "run_done experiment_mode=onn_causal_recurrent" in line]
    if len(lines) != 1:
        raise ValueError("expected one successful training run_done marker")
    match = re.search(r"(?:^|\s)last=(\S+)", lines[0])
    if match is None:
        raise ValueError("training run_done does not identify the last checkpoint")
    path = Path(match.group(1)).resolve()
    root = Path(output_root).resolve()
    if (path.parent.parent != root
            or not path.parent.name.startswith(run_name + "_")
            or path.name != run_name + ".last.pt"):
        raise ValueError("last checkpoint does not belong to this run")
    return path


def validate_checkpoint(checkpoint):
    expected = {
        "mode": "onn_causal_recurrent", "checkpoint_kind": "last",
        "checkpoint_selection": "last_epoch", "epoch": 25,
        "selected_epoch": 25, "architecture_version": 2,
        "output_dim": 1024, "loss_feature_dim": 1024, "output_layer_norm": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"checkpoint {key}: expected {value}, got {checkpoint.get(key)}")
    if not isinstance(checkpoint.get("config"), dict):
        raise ValueError("last checkpoint has no saved training config")
    weight = checkpoint.get("predictor", {}).get("output_linear.weight")
    if weight is None or tuple(weight.shape) != (1024, 384):
        raise ValueError("last checkpoint is missing the trained Linear(384,1024)")
    onn = checkpoint["config"].get("onn", {})
    if onn.get("num_slm_layers") != 3 or onn.get("feedback_enabled") is not False:
        raise ValueError("checkpoint must contain three SLMs with internal feedback disabled")


def build_test_config(checkpoint, checkpoint_path, output_dir):
    validate_checkpoint(checkpoint)
    config = copy.deepcopy(checkpoint["config"])
    config.update({
        "predictor_checkpoint": str(Path(checkpoint_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "dataset": "intphys-test", "eval_name": "intphys_test", "mode": "all",
        "nodes": 1, "tasks_per_node": 1, "test_resume": {"enabled": False},
    })
    config.setdefault("data", {})["batch_size"] = 10
    evaluation = config.setdefault("evaluation", {})
    evaluation.update({
        "spatial_reduction": "mean", "temporal_reduction": "mean",
        "primary_score": "unweighted_prediction_error",
        "last_context_copy_baseline": False,
    })
    return config


def validate_answer_rows(rows, tasks):
    seen = set()
    expected = set(tasks)
    for row in rows:
        fields = row.split()
        if len(fields) != 2:
            raise ValueError("answer row must contain exactly task and score")
        task, value = fields
        score = float(value)
        if task not in expected or task in seen:
            raise ValueError(f"unexpected or duplicate answer task: {task}")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid answer score for {task}: {score}")
        seen.add(task)
    if seen != expected:
        raise ValueError(f"incomplete Test answers: got {len(seen)}, expected {len(expected)}")


def run_logged(command, log_path):
    print("COMMAND:", shlex.join(command), flush=True)
    with Path(log_path).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=REPO, env=os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise RuntimeError(f"stage failed with exit code {code}; log: {log_path}")


def preflight():
    # Imports resolve the same paths used by the existing evaluators.
    from evals.intphys_test.utils import get_dataset_paths as test_paths
    from evals.intuitive_physics.utils import get_dataset_paths as train_paths
    from evals.intuitive_physics.optical_split import require_existing_video_split
    import torch

    for path in (PYTHON, SOURCE_CONFIG, TEST_RUNNER):
        if not path.is_file():
            raise FileNotFoundError(f"required file missing: {path}")
    if not OUTPUT_ROOT.is_dir() or not os.access(OUTPUT_ROOT, os.W_OK):
        raise PermissionError(f"output root is not writable: {OUTPUT_ROOT}")
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    if config.get("predictor_type") != "onn_causal_recurrent":
        raise ValueError("source config is not the causal recurrent model")
    if (config["onn"]["num_slm_layers"] != 3
            or config["onn"]["feedback_enabled"] is not False
            or config["onn"]["feedback_memory_enabled"] is not False):
        raise ValueError("source config must use three SLMs without internal feedback")
    if not math.isclose(float(config["predictor"]["memory_lambda"]), 0.4):
        raise ValueError("source config must use current/history weights 0.6/0.4")
    if config["data"]["frame_steps"] != 2 or config["data"]["frames_per_clip"] != 16:
        raise ValueError("this pipeline expects 16-frame windows and frame_step=2")
    pretrained = Path(config["pretrain"]["folder"]) / config["pretrain"]["checkpoint"]
    if not pretrained.is_file():
        raise FileNotFoundError(f"pretrained encoder checkpoint missing: {pretrained}")

    train_root = Path(train_paths(["IntPhys-train"])[0])
    split_config = config["data_split"]
    require_existing_video_split(
        train_root, config["training"]["split_manifest"],
        num_train_videos=int(split_config["num_train_videos"]),
        num_val_videos=int(split_config["num_val_videos"]),
        split_seed=int(split_config["split_seed"]),
    )
    test_root = Path(test_paths(["IntPhys-test"])[0]).resolve()
    task_file = test_root.parent / "task.txt"
    if not test_root.is_dir() or not task_file.is_file():
        raise FileNotFoundError(f"Test root/task.txt missing: {test_root}, {task_file}")
    tasks = task_file.read_text(encoding="utf-8").splitlines()
    if len(tasks) != 12960 or len(set(tasks)) != 12960:
        raise ValueError("full O1/O2/O3 Test must contain 12960 unique tasks")
    if {task.split("/")[0] for task in tasks} != {"O1", "O2", "O3"}:
        raise ValueError("Test tasks must include O1, O2 and O3")
    sampled_blocks = set()
    for task in tasks:
        # Same conversion as intphys_dataset.path_from_task; no dataset rewrite.
        scene = Path(str(test_root) + "/" + task[:7] + task[-2:]) / "scene"
        if not scene.is_dir():
            raise FileNotFoundError(f"Test scene missing for {task}: {scene}")
        block = task.split("/")[0]
        if block not in sampled_blocks:
            frames = sorted(scene.iterdir())
            if len(frames) < 99:
                raise ValueError(f"Test scene has too few frames: {scene}")
            sampled_blocks.add(block)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("expected one visible CUDA GPU; select physical GPU with --physical-gpu")
    print("PREFLIGHT_OK",
          f"physical_gpu={os.environ['CUDA_VISIBLE_DEVICES']} logical_device=cuda:0",
          f"gpu_name={torch.cuda.get_device_name(0)}",
          f"test_root={test_root} task_file={task_file} tasks={len(tasks)}",
          flush=True)
    return config, tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--check-only", action="store_true",
                        help="validate GPU/config/dataset paths without training or testing")
    args = parser.parse_args(argv)
    if args.physical_gpu < 0:
        raise ValueError("--physical-gpu must be nonnegative")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("run this pipeline directly, not with torchrun")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ["PYTHONPATH"] = str(REPO)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTHONUNBUFFERED"] = "1"
    # A surrounding Slurm environment must not override physical GPU selection.
    os.environ.pop("SLURM_LOCALID", None)
    os.environ.pop("SLURM_CLUSTER_NAME", None)
    sys.path.insert(0, str(REPO))
    config, tasks = preflight()
    if args.check_only:
        return 0

    import torch
    run_name = (
        f"causal_recurrent_linear_slm3_25ep_bs25_gpu{args.physical_gpu}_"
        + uuid.uuid4().hex[:8]
    )
    train_log = OUTPUT_ROOT / f"{run_name}.train_launcher.log"
    print("PIPELINE_START train_epochs=25 train_batch_size=25 test_batch_size=10 "
          "SLMs_per_ONN=3 validation=disabled Dev=disabled checkpoint=last_epoch",
          flush=True)
    # Runtime caches stay on the data volume, isolated from other Linux users.
    runtime = OUTPUT_ROOT / f"{run_name}.runtime"
    (runtime / "tmp").mkdir(parents=True, exist_ok=False)
    (runtime / "cache").mkdir()
    os.environ["TMPDIR"] = str(runtime / "tmp")
    os.environ["XDG_CACHE_HOME"] = str(runtime / "cache")
    run_logged(training_command(REPO, SOURCE_CONFIG, run_name, PYTHON), train_log)
    last_path = last_checkpoint_from_log(
        train_log.read_text(encoding="utf-8"), OUTPUT_ROOT, run_name
    )
    if not last_path.is_file():
        raise FileNotFoundError(f"this run's last checkpoint missing: {last_path}")
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint)
    run_dir = last_path.parent
    test_dir = run_dir / f"test_eval_full_o1_o2_o3_gpu{args.physical_gpu}_bs10"
    test_dir.mkdir(exist_ok=False)
    test_config = build_test_config(checkpoint, last_path, test_dir)
    tag = test_config.get("tag")
    if tag is not None and (Path(tag).name != tag or tag in {".", ".."}):
        raise ValueError(f"unsafe Test output tag: {tag}")
    config_path = run_dir / "test_full_o1_o2_o3_bs10.yaml"
    with config_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(test_config, handle, sort_keys=False, allow_unicode=True)
    print(f"TEST_START checkpoint={last_path} epoch=25 batch_size=10 "
          f"spatial=mean temporal_outputs=mean,max output_dir={test_dir}", flush=True)
    run_logged([str(PYTHON), "-u", str(TEST_RUNNER), str(config_path), "10",
                str(test_dir)], test_dir / "launcher.log")

    inference_log = test_dir / "inference.log"
    if not inference_log.is_file() or "run_done output_dir=" not in inference_log.read_text():
        raise ValueError(f"Test completion marker missing: {inference_log}")
    result_dir = test_dir / f"intphys-test-{tag}" if tag is not None else test_dir
    for name in ("average_surprise_answer.txt", "maximum_surprise_answer.txt"):
        answer_path = result_dir / name
        with answer_path.open(encoding="utf-8") as handle:
            validate_answer_rows(handle, tasks)
        print(f"ANSWER_OK rows=12960 file={answer_path}", flush=True)
    cache = result_dir / "losses_2fs_causal_recurrentctxt.pth"
    if not cache.is_file():
        raise FileNotFoundError(f"causal Test score cache missing: {cache}")
    print(f"PIPELINE_DONE epoch=25 last_checkpoint={last_path} "
          f"test_result_dir={result_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nPIPELINE_INTERRUPTED: Ctrl+C received; stage stopped.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\nPIPELINE_FAILED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)

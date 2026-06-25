import argparse
import json
import os
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HF_HOME = Path("/mnt/workspace/users/han.li/hf_home")
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HF_HOME / "datasets")
os.environ["HF_ASSETS_CACHE"] = str(HF_HOME / "assets")

import torch
import triton
import transformers
from huggingface_hub import snapshot_download
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM, AutoTokenizer


REMOTE_CODE = Path(__file__).resolve().parent / "remote_code" / "modeling_qwen3_activation24.py"
DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_TASKS = "hellaswag,piqa,winogrande,arc_challenge,arc_easy,boolq"
DEFAULT_OVERLAY_ROOT = HF_HOME / "overlays"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-8B with activation 2:4 sparse MLP variants.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--variant", choices=("baseline", "pytorch", "triton"), default="baseline")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="Comma-separated lm-eval task names.")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=parse_limit, default=None)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--overlay-root", default=str(DEFAULT_OVERLAY_ROOT))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--op-bench-warmup", type=int, default=int(os.environ.get("OP_BENCH_WARMUP", "5")))
    parser.add_argument("--op-bench-repeat", type=int, default=int(os.environ.get("OP_BENCH_REPEAT", "20")))
    parser.add_argument("--op-bench-max-shapes", type=int, default=int(os.environ.get("OP_BENCH_MAX_SHAPES", "0")))
    return parser.parse_args()


def setup_hf_env() -> None:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(HF_HOME / "datasets")
    os.environ["HF_ASSETS_CACHE"] = str(HF_HOME / "assets")


def overlay_dir(model_id: str, overlay_root: str | Path) -> Path:
    safe_model_id = model_id.replace("/", "--")
    return Path(overlay_root) / f"{safe_model_id}-activation24"


def prepare_activation24_overlay(model_id: str, overlay_root: str | Path) -> Path:
    setup_hf_env()
    target_dir = overlay_dir(model_id, overlay_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(snapshot_download(repo_id=model_id))
    for item in snapshot_path.iterdir():
        target = target_dir / item.name
        if item.name == "config.json":
            continue
        _ensure_symlink(item, target)

    with (snapshot_path / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    auto_map = dict(config.get("auto_map", {}))
    auto_map["AutoModelForCausalLM"] = "modeling_qwen3_activation24.Qwen3Activation24ForCausalLM"
    config["auto_map"] = auto_map
    config["architectures"] = ["Qwen3Activation24ForCausalLM"]
    config["activation24_sparse"] = {
        "sites": ["mlp_input", "down_input"],
        "backends": ["pytorch", "triton"],
    }
    with (target_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    shutil.copyfile(REMOTE_CODE, target_dir / REMOTE_CODE.name)
    return target_dir


def _ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() and Path(os.readlink(dst)) == src:
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def parse_tasks(tasks: str) -> list[str]:
    parsed = [task.strip() for task in tasks.split(",") if task.strip()]
    if not parsed:
        raise ValueError("--tasks must include at least one task.")
    return parsed


def parse_limit(value: str) -> int | float:
    if "." in value:
        return float(value)
    return int(value)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_tokenizer(args: argparse.Namespace):
    if args.variant == "baseline":
        model_path = args.model_id
    else:
        model_path = str(prepare_activation24_overlay(args.model_id, args.overlay_root))
        os.environ["QWEN3_ACTIVATION24_BACKEND"] = args.variant

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading {args.variant} model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype_from_name(args.dtype),
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model.eval()
    else:
        model = model.to("cpu").eval()
    return model, tokenizer, str(model_path)


def evaluate_task(model, tokenizer, task: str, args: argparse.Namespace) -> dict[str, Any]:
    if hasattr(model, "reset_activation24_stats"):
        model.reset_activation24_stats()

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_time_sec = time.perf_counter() - started

    print(make_table(results))

    shape_histogram = {}
    operator_benchmarks = {}
    if hasattr(model, "get_activation24_stats"):
        shape_histogram = model.get_activation24_stats()
        operator_benchmarks = model.benchmark_activation24_shapes(
            warmup=args.op_bench_warmup,
            repeat=args.op_bench_repeat,
            max_shapes=args.op_bench_max_shapes,
        )

    return {
        "task": task,
        "wall_time_sec": wall_time_sec,
        "metrics": results["results"].get(task, results["results"]),
        "shape_histogram": shape_histogram,
        "operator_benchmarks": operator_benchmarks,
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    tasks = parse_tasks(args.tasks)
    model, tokenizer, resolved_model_path = load_model_and_tokenizer(args)

    task_results = []
    overall_started = time.perf_counter()
    for task in tasks:
        print(f"Running task {task} for variant {args.variant}...")
        task_results.append(evaluate_task(model, tokenizer, task, args))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_wall_time_sec = time.perf_counter() - overall_started

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "variant": args.variant,
        "model_id": args.model_id,
        "resolved_model_path": resolved_model_path,
        "tasks": tasks,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "dtype": args.dtype,
        "total_wall_time_sec": total_wall_time_sec,
        "environment": environment_info(),
        "operator_benchmark_config": {
            "warmup": args.op_bench_warmup,
            "repeat": args.op_bench_repeat,
            "max_shapes": args.op_bench_max_shapes,
        },
        "results": task_results,
    }


def environment_info() -> dict[str, Any]:
    gpu_info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        gpu_info.update(
            {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "memory_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "triton": triton.__version__,
        "hf_home": os.environ.get("HF_HOME", ""),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE", ""),
        "gpu": gpu_info,
    }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)
        f.write("\n")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    setup_hf_env()
    if args.prepare_only:
        path = prepare_activation24_overlay(args.model_id, args.overlay_root)
        print(path)
        return

    results = run_eval(args)
    if args.output_json:
        write_json(args.output_json, results)
        print(f"Wrote {args.output_json}")
    else:
        print(json.dumps(to_jsonable(results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

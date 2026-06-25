#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=/mnt/workspace/users/han.li/hf_home
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"

set +u
# shellcheck source=/dev/null
source scripts/activate_env.sh
set -u

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
EVAL_TASKS="${EVAL_TASKS:-hellaswag,piqa,winogrande,arc_challenge,arc_easy,boolq}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
MODEL_DTYPE="${MODEL_DTYPE:-float16}"
GPU="${GPU:-0}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-results/qwen3_8b}"
LOG_DIR="$RESULT_DIR/logs"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --model-id "$MODEL_ID"
  --tasks "$EVAL_TASKS"
  --batch-size "$EVAL_BATCH_SIZE"
  --dtype "$MODEL_DTYPE"
  --op-bench-warmup "${OP_BENCH_WARMUP:-5}"
  --op-bench-repeat "${OP_BENCH_REPEAT:-20}"
  --op-bench-max-shapes "${OP_BENCH_MAX_SHAPES:-0}"
)

if [ -n "${EVAL_LIMIT:-}" ]; then
  COMMON_ARGS+=(--limit "$EVAL_LIMIT")
fi

echo "Preparing activation24 remote-code overlay for $MODEL_ID..."
python -m eval.qwen3_8b.evaluate_activation_24 --prepare-only --model-id "$MODEL_ID"

declare -A OUTPUTS=()
declare -A LOGS=()

for variant in baseline pytorch triton; do
  output="$RESULT_DIR/qwen3_8b_activation24_${RUN_ID}_${variant}.json"
  log="$LOG_DIR/qwen3_8b_activation24_${RUN_ID}_${variant}.log"
  OUTPUTS[$variant]="$output"
  LOGS[$variant]="$log"
  echo "Running $variant on GPU $GPU..."
  CUDA_VISIBLE_DEVICES="$GPU" python -m eval.qwen3_8b.evaluate_activation_24 \
    --variant "$variant" \
    "${COMMON_ARGS[@]}" \
    --output-json "$output" \
    >"$log" 2>&1
done

merged="$RESULT_DIR/qwen3_8b_activation24_${RUN_ID}.json"
python - "$merged" "${OUTPUTS[baseline]}" "${OUTPUTS[pytorch]}" "${OUTPUTS[triton]}" <<'PY'
import json
import sys
from pathlib import Path

merged_path = Path(sys.argv[1])
variants = {}
for raw_path in sys.argv[2:]:
    path = Path(raw_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    variants[payload["variant"]] = payload

baseline = variants["baseline"]
combined = {
    "model_id": baseline["model_id"],
    "tasks": baseline["tasks"],
    "batch_size": baseline["batch_size"],
    "limit": baseline["limit"],
    "dtype": baseline["dtype"],
    "variants": variants,
}
merged_path.parent.mkdir(parents=True, exist_ok=True)
with merged_path.open("w", encoding="utf-8") as handle:
    json.dump(combined, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
print(merged_path)
PY

echo "Merged results written to $merged"
for variant in baseline pytorch triton; do
  echo "$variant log: ${LOGS[$variant]}"
done


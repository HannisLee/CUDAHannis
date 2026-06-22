#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/mnt/workspace/users/han.li/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ASSETS_CACHE=$HF_HOME/assets

set +u
source scripts/activate_env.sh
set -u

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
EVAL_TASKS="${EVAL_TASKS:-hellaswag,piqa,winogrande,arc_challenge,arc_easy,boolq}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
MODEL_DTYPE="${MODEL_DTYPE:-float16}"

BASELINE_GPU="${BASELINE_GPU:-0}"
PYTORCH_GPU="${PYTORCH_GPU:-1}"
TRITON_GPU="${TRITON_GPU:-2}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-results}"
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
python eval.py --prepare-only --model-id "$MODEL_ID" >/tmp/cudahannis_activation24_overlay.txt
cat /tmp/cudahannis_activation24_overlay.txt

declare -A GPUS=(
  [baseline]="$BASELINE_GPU"
  [pytorch]="$PYTORCH_GPU"
  [triton]="$TRITON_GPU"
)

declare -A PIDS=()
declare -A OUTPUTS=()
declare -A LOGS=()

for variant in baseline pytorch triton; do
  output="$RESULT_DIR/qwen3_8b_activation24_${RUN_ID}_${variant}.json"
  log="$LOG_DIR/qwen3_8b_activation24_${RUN_ID}_${variant}.log"
  OUTPUTS[$variant]="$output"
  LOGS[$variant]="$log"
  echo "Starting $variant on GPU ${GPUS[$variant]}..."
  (
    export CUDA_VISIBLE_DEVICES="${GPUS[$variant]}"
    python eval.py \
      --variant "$variant" \
      "${COMMON_ARGS[@]}" \
      --output-json "$output"
  ) >"$log" 2>&1 &
  PIDS[$variant]=$!
done

failed=0
for variant in baseline pytorch triton; do
  if ! wait "${PIDS[$variant]}"; then
    echo "$variant failed; see ${LOGS[$variant]}" >&2
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi

merged="$RESULT_DIR/qwen3_8b_activation24_${RUN_ID}.json"
python - "$merged" "${OUTPUTS[baseline]}" "${OUTPUTS[pytorch]}" "${OUTPUTS[triton]}" <<'PY'
import json
import sys
from pathlib import Path

merged_path = Path(sys.argv[1])
variant_paths = [Path(path) for path in sys.argv[2:]]
variants = {}
for path in variant_paths:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    variants[payload["variant"]] = payload

combined = {
    "model_id": variants["baseline"]["model_id"],
    "tasks": variants["baseline"]["tasks"],
    "batch_size": variants["baseline"]["batch_size"],
    "limit": variants["baseline"]["limit"],
    "dtype": variants["baseline"]["dtype"],
    "variants": variants,
}
merged_path.parent.mkdir(parents=True, exist_ok=True)
with merged_path.open("w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(merged_path)
PY

echo "Merged results written to $merged"
echo "Logs:"
for variant in baseline pytorch triton; do
  echo "  $variant: ${LOGS[$variant]}"
done

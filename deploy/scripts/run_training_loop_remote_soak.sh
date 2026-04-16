#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" != "bash" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi

detect_app_root() {
  local candidates=(
    "$SCRIPT_DIR/../.."
    "$HOME/yolostudio_agent_proto"
    "/opt/yolostudio-agent"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate/agent_plan/agent" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "$HOME/yolostudio_agent_proto"
}

detect_conda_root() {
  local candidates=(
    "$HOME/miniconda3"
    "/opt/conda"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "/opt/conda"
}

detect_data_yaml() {
  local candidates=(
    "$HOME/test_dataset/data.yaml"
    "$HOME/agent_cap_tests/zyb/data.yaml"
    "/data/example_dataset/data.yaml"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "/data/example_dataset/data.yaml"
}

detect_model_path() {
  local candidates=(
    "$HOME/yolov8n.pt"
    "/models/yolov8n.pt"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "/models/yolov8n.pt"
}

APP_ROOT="${APP_ROOT:-$(detect_app_root)}"
CONDA_ROOT="${CONDA_ROOT:-$(detect_conda_root)}"
REQUESTED_ENV="${1:-auto}"
TRAIN_ENV_NAME="${2:-}"
OUTPUT_ROOT="${3:-/tmp/training_real_lifecycle_output/training_loop_soak}"
DATA_YAML="${4:-$(detect_data_yaml)}"
MODEL_PATH="${5:-$(detect_model_path)}"
MAX_ROUNDS="${6:-20}"
EPOCHS="${7:-1}"
DEVICE="${8:-0}"
KNOWLEDGE_MODE="${9:-forced}"
FORCED_ACTION="${10:-continue_observing}"
ALLOWED_TUNING_PARAMS="${11:-none}"
MANAGED_LEVEL="${12:-full_auto}"
TIMEOUT_SECONDS="${13:-0}"
LOOP_POLL_INTERVAL="${14:-5}"
WATCH_POLL_INTERVAL="${15:-5}"
TRAIN_PROJECT="${16:-$OUTPUT_ROOT/runs}"
WAIT_MODE="${17:-terminal}"
AUTO_RESUME_REVIEWS="${18:-0}"
RECREATE_SERVICE_ON_REVIEW_RESUME="${19:-0}"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "missing conda.sh under: $CONDA_ROOT" >&2
  exit 1
fi

if [[ ! -d "$APP_ROOT" ]]; then
  echo "missing app root: $APP_ROOT" >&2
  exit 1
fi

if [[ ! -f "$DATA_YAML" ]]; then
  echo "missing data yaml: $DATA_YAML" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "missing model path: $MODEL_PATH" >&2
  exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"

resolve_env() {
  local requested="$1"
  local env_names
  env_names="$(conda env list | awk 'NF && $1 !~ /^#/ {print $1}')"

  if [[ -n "$requested" && "$requested" != "auto" ]] && grep -qx "$requested" <<<"$env_names"; then
    printf '%s\n' "$requested"
    return 0
  fi

  for candidate in yolostudio-agent-server agent-server yolodo yolo; do
    if grep -qx "$candidate" <<<"$env_names"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [[ -n "$requested" && "$requested" != "auto" ]]; then
    echo "requested conda env not found: $requested" >&2
  fi
  echo "no usable training conda env found (tried: yolostudio-agent-server, agent-server, yolodo, yolo)" >&2
  return 1
}

ENV_NAME="$(resolve_env "$REQUESTED_ENV")"
echo "using conda env: $ENV_NAME"
conda activate "$ENV_NAME"
if [[ -z "$TRAIN_ENV_NAME" || "$TRAIN_ENV_NAME" == "auto" ]]; then
  TRAIN_ENV_NAME="$ENV_NAME"
fi

mkdir -p "$OUTPUT_ROOT"

cd "$APP_ROOT"
export PYTHONPATH="$APP_ROOT:$APP_ROOT/agent_plan${PYTHONPATH:+:$PYTHONPATH}"
export YOLOSTUDIO_LOOP_LLM_ENABLED="${YOLOSTUDIO_LOOP_LLM_ENABLED:-1}"
export YOLOSTUDIO_LOOP_LLM_PROVIDER="${YOLOSTUDIO_LOOP_LLM_PROVIDER:-ollama}"
export YOLOSTUDIO_LOOP_LLM_BASE_URL="${YOLOSTUDIO_LOOP_LLM_BASE_URL:-http://127.0.0.1:11434}"
export YOLOSTUDIO_LOOP_LLM_MODEL="${YOLOSTUDIO_LOOP_LLM_MODEL:-gemma4:e4b}"
export YOLOSTUDIO_LOOP_LLM_KEEP_ALIVE="${YOLOSTUDIO_LOOP_LLM_KEEP_ALIVE:-0s}"

OUTPUT_PATH="$OUTPUT_ROOT/training_loop_remote_soak_${MAX_ROUNDS}r.json"
STATE_DIR="$OUTPUT_ROOT/state_${MAX_ROUNDS}r"
LOOP_NAME="remote-soak-${MAX_ROUNDS}r"

cmd=(
  python -m yolostudio_agent.agent.tests.test_training_loop_remote_real_soak
  --output "$OUTPUT_PATH"
  --model "$MODEL_PATH"
  --data-yaml "$DATA_YAML"
  --epochs "$EPOCHS"
  --device "$DEVICE"
  --training-environment "$TRAIN_ENV_NAME"
  --project "$TRAIN_PROJECT"
  --loop-name "$LOOP_NAME"
  --managed-level "$MANAGED_LEVEL"
  --max-rounds "$MAX_ROUNDS"
  --min-improvement 0
  --no-improvement-rounds "$MAX_ROUNDS"
  --max-failures 2
  --allowed-tuning-params "$ALLOWED_TUNING_PARAMS"
  --knowledge-mode "$KNOWLEDGE_MODE"
  --forced-action "$FORCED_ACTION"
  --state-dir "$STATE_DIR"
  --loop-poll-interval "$LOOP_POLL_INTERVAL"
  --watch-poll-interval "$WATCH_POLL_INTERVAL"
  --wait-mode "$WAIT_MODE"
  --auto-resume-reviews "$AUTO_RESUME_REVIEWS"
)

if [[ "$RECREATE_SERVICE_ON_REVIEW_RESUME" == "1" ]]; then
  cmd+=(--recreate-service-on-review-resume)
fi

if [[ "$TIMEOUT_SECONDS" != "0" ]]; then
  cmd+=(--timeout "$TIMEOUT_SECONDS")
fi

"${cmd[@]}"
printf '%s\n' "$OUTPUT_PATH"

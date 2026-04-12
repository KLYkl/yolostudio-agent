#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-$HOME/yolostudio_agent_proto}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
REQUESTED_ENV="${1:-auto}"
OUTPUT_ROOT="${2:-$HOME/training_real_lifecycle_output/agent_mainline_roundtrip}"
DATASET_ROOT="${3:-/home/kly/agent_cap_tests/zyb}"
MODEL_PATH="${4:-/home/kly/yolov8n.pt}"
EPOCHS="${5:-30}"
TARGET_EPOCH="${6:-2}"
STATUS_DELAYS="${7:-15,35,60}"
EXTRA_POLL_INTERVAL="${8:-30}"
EXTRA_POLL_LIMIT="${9:-8}"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "missing conda.sh under: $CONDA_ROOT" >&2
  exit 1
fi

if [[ ! -d "$APP_ROOT" ]]; then
  echo "missing app root: $APP_ROOT" >&2
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

  for candidate in yolostudio-agent-server yolodo yolo; do
    if grep -qx "$candidate" <<<"$env_names"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [[ -n "$requested" && "$requested" != "auto" ]]; then
    echo "requested conda env not found: $requested" >&2
  fi
  echo "no usable training conda env found (tried: yolostudio-agent-server, yolodo, yolo)" >&2
  return 1
}

ENV_NAME="$(resolve_env "$REQUESTED_ENV")"
echo "using conda env: $ENV_NAME"
conda activate "$ENV_NAME"

mkdir -p "$OUTPUT_ROOT"

if [[ -x "$APP_ROOT/manage_mcp_server.sh" ]]; then
  "$APP_ROOT/manage_mcp_server.sh" status >/dev/null 2>&1 || "$APP_ROOT/manage_mcp_server.sh" restart >/dev/null 2>&1
fi

cd "$APP_ROOT"
export YOLO_AGENT_TRAIN_TEST_MODE="direct_tools"
export YOLO_AGENT_TRAIN_OUT="$OUTPUT_ROOT/remote_training_mainline_agent_roundtrip.json"
export YOLO_AGENT_TRAIN_DATASET_ROOT="$DATASET_ROOT"
export YOLO_AGENT_TRAIN_MODEL_PATH="$MODEL_PATH"
export YOLO_AGENT_TRAIN_EPOCHS="$EPOCHS"
export YOLO_AGENT_TRAIN_TARGET_EPOCH="$TARGET_EPOCH"
export YOLO_AGENT_TRAIN_STATUS_DELAYS="$STATUS_DELAYS"
export YOLO_AGENT_TRAIN_EXTRA_POLL_INTERVAL="$EXTRA_POLL_INTERVAL"
export YOLO_AGENT_TRAIN_EXTRA_POLL_LIMIT="$EXTRA_POLL_LIMIT"

python -m agent_plan.agent.tests.test_zyb_training_mainline_agent_roundtrip
printf '%s\n' "$YOLO_AGENT_TRAIN_OUT"

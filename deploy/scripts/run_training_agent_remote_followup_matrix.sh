#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

detect_dataset_root() {
  local candidates=(
    "$HOME/agent_cap_tests/zyb"
    "/data/example_dataset"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "/data/example_dataset"
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
OUTPUT_ROOT="${2:-/tmp/training_real_lifecycle_output/agent_followup_matrix}"
DATASET_ROOT="${3:-$(detect_dataset_root)}"
MODEL_PATH="${4:-$(detect_model_path)}"

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

run_case() {
  local mode="$1"
  local out_json="$2"
  local epochs="$3"
  local target_epoch="$4"
  local status_delays="$5"
  local extra_poll_interval="$6"
  local extra_poll_limit="$7"

  export YOLO_AGENT_TRAIN_TEST_MODE="direct_tools"
  export YOLO_AGENT_TRAIN_OUT="$out_json"
  export YOLO_AGENT_TRAIN_DATASET_ROOT="$DATASET_ROOT"
  export YOLO_AGENT_TRAIN_MODEL_PATH="$MODEL_PATH"
  export YOLO_AGENT_TRAIN_EPOCHS="$epochs"
  export YOLO_AGENT_TRAIN_TARGET_EPOCH="$target_epoch"
  export YOLO_AGENT_TRAIN_STATUS_DELAYS="$status_delays"
  export YOLO_AGENT_TRAIN_EXTRA_POLL_INTERVAL="$extra_poll_interval"
  export YOLO_AGENT_TRAIN_EXTRA_POLL_LIMIT="$extra_poll_limit"
  export YOLO_AGENT_TRAIN_FINAL_MODE="$mode"

  python -m yolostudio_agent.agent.tests.test_zyb_training_mainline_agent_roundtrip >/tmp/yolo_train_roundtrip_${mode}.log
  cat /tmp/yolo_train_roundtrip_${mode}.log
}

ENV_NAME="$(resolve_env "$REQUESTED_ENV")"
echo "using conda env: $ENV_NAME"
conda activate "$ENV_NAME"

mkdir -p "$OUTPUT_ROOT"

if [[ -x "$APP_ROOT/manage_mcp_server.sh" ]]; then
  "$APP_ROOT/manage_mcp_server.sh" status >/dev/null 2>&1 || "$APP_ROOT/manage_mcp_server.sh" restart >/dev/null 2>&1
fi

cd "$APP_ROOT"
export PYTHONPATH="$APP_ROOT:$APP_ROOT/agent_plan${PYTHONPATH:+:$PYTHONPATH}"

STOPPED_OUT="$OUTPUT_ROOT/remote_training_mainline_agent_roundtrip_stopped.json"
COMPLETED_OUT="$OUTPUT_ROOT/remote_training_mainline_agent_roundtrip_completed.json"
FAILED_OUT="$OUTPUT_ROOT/remote_training_mainline_agent_roundtrip_failed.json"
SUMMARY_OUT="$OUTPUT_ROOT/remote_training_mainline_followup_matrix.json"
export STOPPED_OUT COMPLETED_OUT FAILED_OUT SUMMARY_OUT

run_case stopped "$STOPPED_OUT" 30 2 "15,35,60" 30 8
run_case completed "$COMPLETED_OUT" 1 1 "10,20,35" 20 8
run_case failed "$FAILED_OUT" 30 1 "10,20" 15 6

python - <<'PY'
from __future__ import annotations
import json
import os
from pathlib import Path

stopped_path = Path(os.environ['STOPPED_OUT'])
completed_path = Path(os.environ['COMPLETED_OUT'])
failed_path = Path(os.environ['FAILED_OUT'])
summary_path = Path(os.environ['SUMMARY_OUT'])

def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

stopped = load(stopped_path)
completed = load(completed_path)
failed = load(failed_path)

summary = {
    'ok': True,
    'cases': {
        'stopped': {
            'path': str(stopped_path),
            'final_run_state': stopped.get('assessment', {}).get('final_run_state'),
            'summary_run_state': stopped.get('assessment', {}).get('summary_run_state'),
            'final_status_route_used': stopped.get('assessment', {}).get('final_status_route_used'),
            'next_step_action': stopped.get('assessment', {}).get('next_step_action'),
        },
        'completed': {
            'path': str(completed_path),
            'final_run_state': completed.get('assessment', {}).get('final_run_state'),
            'summary_run_state': completed.get('assessment', {}).get('summary_run_state'),
            'final_status_route_used': completed.get('assessment', {}).get('final_status_route_used'),
            'next_step_action': completed.get('assessment', {}).get('next_step_action'),
        },
        'failed': {
            'path': str(failed_path),
            'final_run_state': failed.get('assessment', {}).get('final_run_state'),
            'summary_run_state': failed.get('assessment', {}).get('summary_run_state'),
            'final_status_route_used': failed.get('assessment', {}).get('final_status_route_used'),
            'next_step_action': failed.get('assessment', {}).get('next_step_action'),
        },
    },
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
PY

printf '%s\n' "$SUMMARY_OUT"

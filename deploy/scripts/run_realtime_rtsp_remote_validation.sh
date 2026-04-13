#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-$HOME/yolostudio_agent_proto}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
REQUESTED_ENV="${1:-auto}"
RTSP_URL="${2:-}"
MODEL_PATH="${3:-}"
OUTPUT_ROOT="${4:-$HOME/realtime_rtsp_validation}"
TIMEOUT_MS="${5:-5000}"
FRAME_INTERVAL_MS="${6:-120}"
MAX_FRAMES="${7:-8}"
WAIT_SECONDS="${8:-20}"
POLL_INTERVAL_SECONDS="${9:-0.5}"

if [[ -z "$RTSP_URL" ]]; then
  echo "missing RTSP URL" >&2
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  echo "missing model path" >&2
  exit 1
fi

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "missing conda.sh under: $CONDA_ROOT" >&2
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

  for candidate in yolodo yolo; do
    if grep -qx "$candidate" <<<"$env_names"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [[ -n "$requested" && "$requested" != "auto" ]]; then
    echo "requested conda env not found: $requested" >&2
  fi
  echo "no usable realtime RTSP conda env found (tried: yolodo, yolo)" >&2
  return 1
}

ENV_NAME="$(resolve_env "$REQUESTED_ENV")"
echo "using conda env: $ENV_NAME"
conda activate "$ENV_NAME"

mkdir -p "$OUTPUT_ROOT"
cd "$APP_ROOT"
export PYTHONPATH="$APP_ROOT:$APP_ROOT/agent_plan${PYTHONPATH:+:$PYTHONPATH}"
python -m yolostudio_agent.agent.tests.test_realtime_rtsp_external_validation \
  --rtsp-url "$RTSP_URL" \
  --model "$MODEL_PATH" \
  --output-dir "$OUTPUT_ROOT" \
  --timeout-ms "$TIMEOUT_MS" \
  --frame-interval-ms "$FRAME_INTERVAL_MS" \
  --max-frames "$MAX_FRAMES" \
  --wait-seconds "$WAIT_SECONDS" \
  --poll-interval-seconds "$POLL_INTERVAL_SECONDS"

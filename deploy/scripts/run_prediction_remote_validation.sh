#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-$HOME/yolostudio_agent_proto}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
REQUESTED_ENV="${1:-auto}"
STAGE_ROOT="${2:-$HOME/prediction_real_media_stage}"
OUTPUT_ROOT="${3:-$HOME/prediction_real_media_output}"

if [[ ! -d "$STAGE_ROOT/weights" || ! -d "$STAGE_ROOT/videos" ]]; then
  echo "missing staged assets under: $STAGE_ROOT" >&2
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
  echo "no usable prediction conda env found (tried: yolodo, yolo)" >&2
  return 1
}

ENV_NAME="$(resolve_env "$REQUESTED_ENV")"
echo "using conda env: $ENV_NAME"
conda activate "$ENV_NAME"

cd "$APP_ROOT"
python -m agent_plan.agent.tests.test_prediction_remote_real_media \
  --weights-dir "$STAGE_ROOT/weights" \
  --videos-dir "$STAGE_ROOT/videos" \
  --manifest "$STAGE_ROOT/manifest.json" \
  --output-dir "$OUTPUT_ROOT"

#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-$HOME/yolostudio_agent_proto}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${1:-yolo}"
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
conda activate "$ENV_NAME"

cd "$APP_ROOT"
python -m agent_plan.agent.tests.test_prediction_remote_real_media \
  --weights-dir "$STAGE_ROOT/weights" \
  --videos-dir "$STAGE_ROOT/videos" \
  --output-dir "$OUTPUT_ROOT"

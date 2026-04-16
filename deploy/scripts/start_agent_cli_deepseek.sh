#!/usr/bin/env bash
set -euo pipefail

SESSION_ID="${1:-deepseek-cli}"
ENV_FILE="${HOME}/.config/yolostudio-agent/llm.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

MODEL="${YOLOSTUDIO_AGENT_MODEL:-deepseek-chat}"
BASE_URL="${YOLOSTUDIO_LLM_BASE_URL:-https://api.deepseek.com/v1}"
KEY="${YOLOSTUDIO_LLM_API_KEY:-${DEEPSEEK_API_KEY:-}}"

if [[ -z "${KEY}" ]]; then
  echo "DEEPSEEK_API_KEY / YOLOSTUDIO_LLM_API_KEY 未设置，无法切到 DeepSeek。" >&2
  exit 1
fi

source /home/kly/miniconda3/etc/profile.d/conda.sh
conda activate yolostudio-agent-server

unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1

export PYTHONPATH=/opt/yolostudio-agent:/opt/yolostudio-agent/agent_plan
export YOLOSTUDIO_MCP_URL="${YOLOSTUDIO_MCP_URL:-http://127.0.0.1:8080/mcp}"
export YOLOSTUDIO_LLM_PROVIDER=deepseek
export YOLOSTUDIO_LLM_BASE_URL="${BASE_URL}"
export YOLOSTUDIO_LLM_API_KEY="${KEY}"
export YOLOSTUDIO_CONFIRMATION_MODE="${YOLOSTUDIO_CONFIRMATION_MODE:-manual}"
export YOLOSTUDIO_AGENT_MODEL="${MODEL}"

python /opt/yolostudio-agent/agent_plan/agent/client/cli.py "${SESSION_ID}"

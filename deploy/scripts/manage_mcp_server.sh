#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_app_root() {
  if [[ -d "$SCRIPT_DIR/agent_plan/agent" && -d "$SCRIPT_DIR/agent_plan/yolostudio_agent" ]]; then
    printf '%s\n' "$SCRIPT_DIR"
    return 0
  fi
  if [[ -d "$SCRIPT_DIR/agent" && -d "$SCRIPT_DIR/yolostudio_agent" ]]; then
    printf '%s\n' "$SCRIPT_DIR"
    return 0
  fi
  local repo_root
  repo_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
  if [[ -d "$repo_root/agent" && -d "$repo_root/yolostudio_agent" ]]; then
    printf '%s\n' "$repo_root"
    return 0
  fi
  printf '%s\n' "$repo_root"
}

detect_conda_bin() {
  local candidates=(
    "${HOME}/miniconda3/bin/conda"
    "/home/kly/miniconda3/bin/conda"
    "/opt/conda/bin/conda"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  printf '%s\n' "/opt/conda/bin/conda"
}

APP_ROOT="${APP_ROOT:-$(detect_app_root)}"
CONDA_BIN="${CONDA_BIN:-$(detect_conda_bin)}"
ENV_NAME="${ENV_NAME:-yolostudio-agent-server}"
LOG_FILE="${LOG_FILE:-$HOME/outputs/yolostudio_mcp.log}"
PID_PATTERN="yolostudio_agent.agent.server.mcp_server"

status() {
  if pgrep -af "$PID_PATTERN" >/dev/null 2>&1; then
    echo "[ok] MCP server running"
    pgrep -af "$PID_PATTERN"
    echo "---"
    ss -ltnp | grep 8080 || true
  else
    echo "[warn] MCP server not running"
    return 1
  fi
}

start() {
  if pgrep -af "$PID_PATTERN" >/dev/null 2>&1; then
    echo "[skip] MCP server already running"
    status
    return 0
  fi
  mkdir -p "$HOME/outputs"
  cd "$APP_ROOT"
  nohup env PYTHONPATH="$APP_ROOT:$APP_ROOT/agent_plan${PYTHONPATH:+:$PYTHONPATH}" "$CONDA_BIN" run -n "$ENV_NAME" python -m yolostudio_agent.agent.server.mcp_server >"$LOG_FILE" 2>&1 &
  sleep 4
  status
}

stop() {
  if ! pgrep -af "$PID_PATTERN" >/dev/null 2>&1; then
    echo "[skip] MCP server already stopped"
    return 0
  fi
  pkill -f "$PID_PATTERN" || true
  sleep 2
  if pgrep -af "$PID_PATTERN" >/dev/null 2>&1; then
    echo "[warn] graceful stop incomplete, forcing kill"
    pkill -9 -f "$PID_PATTERN" || true
    sleep 1
  fi
  if pgrep -af "$PID_PATTERN" >/dev/null 2>&1; then
    echo "[error] MCP server still running"
    return 1
  fi
  echo "[ok] MCP server stopped"
}

restart() {
  stop || true
  start
}

logs() {
  tail -n 80 "$LOG_FILE"
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  logs) logs ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}"
    exit 2
    ;;
esac

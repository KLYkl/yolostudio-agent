#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/opt/yolostudio-agent}"
CONDA_BIN="${CONDA_BIN:-/opt/conda/bin/conda}"
ENV_NAME="${ENV_NAME:-agent-server}"
LOG_FILE="${LOG_FILE:-$HOME/outputs/yolostudio_mcp.log}"
PID_PATTERN="agent_plan.agent.server.mcp_server"

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
  nohup env PYTHONPATH="$APP_ROOT" "$CONDA_BIN" run -n "$ENV_NAME" python -m agent_plan.agent.server.mcp_server >"$LOG_FILE" 2>&1 &
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

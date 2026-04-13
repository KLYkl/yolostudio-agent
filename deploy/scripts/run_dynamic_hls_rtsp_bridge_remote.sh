#!/usr/bin/env bash
set -euo pipefail

HLS_URL="${1:-}"
WORK_DIR="${2:-/tmp/dynamic_hls_rtsp_bridge}"
RTSP_PORT="${3:-8555}"
PATH_NAME="${4:-stream}"
RTP_PORT="${5:-8002}"
RTCP_PORT="${6:-8003}"

if [[ -z "$HLS_URL" ]]; then
  echo "missing HLS URL" >&2
  exit 1
fi

SEARCH_ROOTS=("/tmp")
if [[ -n "${HOME:-}" ]]; then
  SEARCH_ROOTS+=("$HOME")
fi

MEDIAMTX_BIN="$(find "${SEARCH_ROOTS[@]}" -maxdepth 5 -path '*/mediamtx/mediamtx' -type f -print -quit 2>/dev/null)"
if [[ -z "$MEDIAMTX_BIN" ]]; then
  echo "mediamtx binary not found" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

for pidfile in mediamtx.pid ffmpeg.pid; do
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
done

cat > mediamtx.yml <<CFG
rtsp: yes
rtspAddress: :$RTSP_PORT
rtpAddress: :$RTP_PORT
rtcpAddress: :$RTCP_PORT
rtmp: no
hls: no
webrtc: no
srt: no
paths:
  $PATH_NAME: {}
CFG

: > mediamtx.log
: > ffmpeg.log

nohup "$MEDIAMTX_BIN" mediamtx.yml > mediamtx.log 2>&1 &
echo $! > mediamtx.pid
sleep 2
if ! kill -0 "$(cat mediamtx.pid)" 2>/dev/null; then
  echo "mediamtx failed to start" >&2
  cat mediamtx.log >&2 || true
  exit 1
fi

RTSP_URL="rtsp://localhost:${RTSP_PORT}/${PATH_NAME}"

nohup ffmpeg \
  -hide_banner \
  -loglevel warning \
  -fflags +discardcorrupt \
  -reconnect 1 \
  -reconnect_streamed 1 \
  -reconnect_delay_max 5 \
  -i "$HLS_URL" \
  -an \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -pix_fmt yuv420p \
  -f rtsp \
  -rtsp_transport tcp \
  "$RTSP_URL" > ffmpeg.log 2>&1 &
echo $! > ffmpeg.pid

READY=0
for _ in $(seq 1 25); do
  if ! kill -0 "$(cat ffmpeg.pid)" 2>/dev/null; then
    echo "ffmpeg bridge failed" >&2
    cat ffmpeg.log >&2 || true
    exit 1
  fi
  if ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of json "$RTSP_URL" >/tmp/dynamic_hls_rtsp_probe.json 2>/dev/null; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" != "1" ]]; then
  echo "RTSP bridge did not become ready in time" >&2
  echo "--- mediamtx.log ---" >&2
  cat mediamtx.log >&2 || true
  echo "--- ffmpeg.log ---" >&2
  cat ffmpeg.log >&2 || true
  exit 1
fi

python3 - <<PY
import json
from pathlib import Path

probe = {}
probe_path = Path("/tmp/dynamic_hls_rtsp_probe.json")
if probe_path.exists():
    probe = json.loads(probe_path.read_text(encoding="utf-8"))

payload = {
    "ok": True,
    "summary": "已启动 HLS -> RTSP 桥接，当前 RTSP 地址可用",
    "hls_url": ${HLS_URL@Q},
    "rtsp_url": ${RTSP_URL@Q},
    "work_dir": ${WORK_DIR@Q},
    "path_name": ${PATH_NAME@Q},
    "rtsp_port": int(${RTSP_PORT@Q}),
    "probe": probe,
}
print(json.dumps(payload, ensure_ascii=False))
PY

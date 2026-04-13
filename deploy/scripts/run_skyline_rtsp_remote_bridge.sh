#!/usr/bin/env bash
set -euo pipefail

PAGE_URL="${1:-https://www.skylinewebcams.com/zh/webcam/thailand/surat-thani/ko-samui/lamai.html}"
WORK_DIR="${2:-/tmp/skyline_rtsp_bridge}"
RTSP_PORT="${3:-8555}"
PATH_NAME="${4:-skyline}"
RTP_PORT="${5:-8002}"
RTCP_PORT="${6:-8003}"
BRIDGE_SCRIPT="${7:-}"

if [[ -z "$BRIDGE_SCRIPT" ]]; then
  echo "missing bridge script path" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"
HLS_JSON="$WORK_DIR/skyline_hls_capture_output.json"

python3 - "$PAGE_URL" "$HLS_JSON" <<'PY'
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

page_url = sys.argv[1]
output_json = Path(sys.argv[2])

response = requests.get(page_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
response.raise_for_status()
html = response.text
match = re.search(r"source:'([^']+\?a=[^']+)'", html)
if not match:
    raise SystemExit("未在 Skyline 页面里解析到 source token")

source = match.group(1)
hls_url = "https://hd-auth.skylinewebcams.com/live.m3u8" + source[source.index("?") :]
hls_response = requests.get(
    hls_url,
    headers={"User-Agent": "Mozilla/5.0", "Referer": page_url},
    timeout=20,
)
hls_response.raise_for_status()

payload = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "page_url": page_url,
    "source": source,
    "hls_url": hls_url,
    "ok": True,
    "summary": "已在服务器侧抓到 fresh Skyline HLS 地址",
    "playlist_preview": hls_response.text[:400],
}
output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY

HLS_URL="$(python3 - "$HLS_JSON" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(payload['hls_url'])
PY
)"

bash "$BRIDGE_SCRIPT" "$HLS_URL" "$WORK_DIR" "$RTSP_PORT" "$PATH_NAME" "$RTP_PORT" "$RTCP_PORT"

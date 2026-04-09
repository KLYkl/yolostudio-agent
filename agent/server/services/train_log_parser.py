from __future__ import annotations

import re
from pathlib import Path

ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
EPOCH_LINE = re.compile(
    r'^(?P<epoch>\d+)\/(?P<total>\d+)\s+'
    r'(?P<gpu_mem>\d+(?:\.\d+)?[GM])\s+'
    r'(?P<box_loss>\d+(?:\.\d+)?)\s+'
    r'(?P<cls_loss>\d+(?:\.\d+)?)\s+'
    r'(?P<dfl_loss>\d+(?:\.\d+)?)'
)


def parse_latest_metrics(log_file: str | Path) -> dict:
    path = Path(log_file)
    if not path.exists():
        return {"ok": False, "error": "log file not found"}
    latest: dict | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = ANSI_ESCAPE.sub('', raw_line).replace('\r', '').strip()
        if not line:
            continue
        match = EPOCH_LINE.match(line)
        if match:
            latest = {
                "epoch": int(match.group("epoch")),
                "total_epochs": int(match.group("total")),
                "gpu_mem": match.group("gpu_mem"),
                "box_loss": match.group("box_loss"),
                "cls_loss": match.group("cls_loss"),
                "dfl_loss": match.group("dfl_loss"),
            }
    return {"ok": True, "metrics": latest}

from __future__ import annotations

import re
from pathlib import Path

EPOCH_LINE = re.compile(
    r"^\s*(?P<epoch>\d+)\/(?P<total>\d+)\s+\S+\s+(?P<box_loss>\S+)\s+(?P<cls_loss>\S+)\s+(?P<dfl_loss>\S+)"
)


def parse_latest_metrics(log_file: str | Path) -> dict:
    path = Path(log_file)
    if not path.exists():
        return {"ok": False, "error": "log file not found"}
    latest: dict | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EPOCH_LINE.match(line)
        if match:
            latest = {
                "epoch": int(match.group("epoch")),
                "total_epochs": int(match.group("total")),
                "box_loss": match.group("box_loss"),
                "cls_loss": match.group("cls_loss"),
                "dfl_loss": match.group("dfl_loss"),
            }
    return {"ok": True, "metrics": latest}

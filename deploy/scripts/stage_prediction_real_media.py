from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_ROOT = REPO_ROOT / ".tmp_prediction_real_media_stage"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def _pick_weights(root: Path, limit: int = 3) -> list[Path]:
    return sorted(root.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]


def _pick_videos(root: Path, limit: int = 3) -> list[Path]:
    candidates = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES]
    return sorted(candidates, key=lambda item: item.stat().st_size)[:limit]


def _copy_files(paths: list[Path], dst_dir: Path) -> list[dict[str, object]]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, object]] = []
    for src in paths:
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied.append(
            {
                "name": src.name,
                "source_path": str(src),
                "staged_path": str(dst),
                "size_bytes": src.stat().st_size,
                "mtime": datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return copied


def main() -> None:
    weights_root_raw = os.environ.get("YOLO_REAL_MEDIA_WEIGHTS_ROOT", "").strip()
    videos_root_raw = os.environ.get("YOLO_REAL_MEDIA_VIDEOS_ROOT", "").strip()
    stage_root_raw = os.environ.get("YOLO_REAL_MEDIA_STAGE_ROOT", "").strip()

    if not weights_root_raw or not videos_root_raw:
        raise RuntimeError(
            "请先设置 YOLO_REAL_MEDIA_WEIGHTS_ROOT 和 YOLO_REAL_MEDIA_VIDEOS_ROOT，再运行 stage_prediction_real_media.py。"
        )

    weights_root = Path(weights_root_raw)
    videos_root = Path(videos_root_raw)
    stage_root = Path(stage_root_raw) if stage_root_raw else DEFAULT_STAGE_ROOT

    if stage_root.exists():
        shutil.rmtree(stage_root)
    weights_dir = stage_root / "weights"
    videos_dir = stage_root / "videos"

    selected_weights = _pick_weights(weights_root)
    selected_videos = _pick_videos(videos_root)
    if not selected_weights:
        raise RuntimeError(f"未在 {weights_root} 找到可用权重")
    if not selected_videos:
        raise RuntimeError(f"未在 {videos_root} 找到可用视频")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights_root": str(weights_root),
        "videos_root": str(videos_root),
        "stage_root": str(stage_root),
        "weights": _copy_files(selected_weights, weights_dir),
        "videos": _copy_files(selected_videos, videos_dir),
    }
    manifest_path = stage_root / "manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"staged prediction real media to: {stage_root}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

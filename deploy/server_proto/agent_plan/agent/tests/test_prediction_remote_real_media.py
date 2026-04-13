from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.tools import predict_tools


def _select_model(weights_dir: Path, manifest_path: Path | None) -> Path:
    if manifest_path and manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        for item in payload.get("weights") or []:
            name = str(item.get("name") or "").strip()
            if not name:
                staged_path = str(item.get("staged_path") or "").strip()
                if staged_path:
                    name = Path(staged_path).name
            if not name:
                continue
            candidate = weights_dir / name
            if candidate.exists():
                return candidate

    weights = sorted(weights_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not weights:
        raise RuntimeError(f"未在 {weights_dir} 找到权重")
    return weights[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remote real-media prediction validation.")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else None
    if not videos_dir.exists():
        raise RuntimeError(f"视频目录不存在: {videos_dir}")
    selected_model = _select_model(weights_dir, manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    predict_result = predict_tools.predict_videos(
        source_path=str(videos_dir),
        model=str(selected_model),
        output_dir=str(output_dir / "videos"),
        save_video=False,
        save_keyframes_annotated=True,
        save_keyframes_raw=False,
        generate_report=True,
        max_videos=args.max_videos,
        max_frames=args.max_frames,
    )
    summary_result = {}
    if predict_result.get("ok"):
        summary_result = predict_tools.summarize_prediction_results(
            report_path=predict_result["report_path"]
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights_dir": str(weights_dir),
        "videos_dir": str(videos_dir),
        "manifest_path": str(manifest_path) if manifest_path else "",
        "selected_model": str(selected_model),
        "predict_result": predict_result,
        "summary_result": summary_result,
    }
    out_json = output_dir / "remote_prediction_validation.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()

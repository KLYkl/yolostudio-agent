from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.server.tools import predict_tools


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remote real-media prediction validation.")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)

    weights = sorted(weights_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not weights:
        raise RuntimeError(f"未在 {weights_dir} 找到权重")
    if not videos_dir.exists():
        raise RuntimeError(f"视频目录不存在: {videos_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    predict_result = predict_tools.predict_videos(
        source_path=str(videos_dir),
        model=str(weights[0]),
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
        "selected_model": str(weights[0]),
        "predict_result": predict_result,
        "summary_result": summary_result,
    }
    out_json = output_dir / "remote_prediction_validation.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()

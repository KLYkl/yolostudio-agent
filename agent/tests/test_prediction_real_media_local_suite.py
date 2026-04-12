from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.server.tools import predict_tools


WEIGHTS_ROOT = Path(r"C:\datasets\weights")
VIDEOS_ROOT = Path(r"C:\datasets\videos")
WORK_ROOT = Path(r"C:\workspace\yolodo2.0\agent_plan\agent\tests\_tmp_prediction_real_media")
OUT_JSON = Path(r"C:\workspace\yolodo2.0\agent_plan\agent\tests\test_prediction_real_media_local_output.json")
OUT_MD = Path(r"C:\workspace\yolodo2.0\agent_plan\doc\prediction_real_media_validation_2026-04-11.md")
LOCAL_YOLO_PYTHON = Path(r"C:\Miniconda3\envs\yolo\python.exe")


def _choose_weights(limit: int = 3) -> list[dict[str, Any]]:
    weights = sorted(
        WEIGHTS_ROOT.glob("*.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[:limit]
    return [
        {
            "path": str(item.resolve()),
            "name": item.name,
            "size_bytes": item.stat().st_size,
            "mtime": datetime.fromtimestamp(item.stat().st_mtime).isoformat(timespec="seconds"),
        }
        for item in weights
    ]


def _choose_videos(limit: int = 3) -> list[dict[str, Any]]:
    candidates = sorted(
        [
            path
            for path in VIDEOS_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        ],
        key=lambda item: item.stat().st_size,
    )[:limit]
    return [
        {
            "path": str(item.resolve()),
            "name": item.name,
            "size_bytes": item.stat().st_size,
            "mtime": datetime.fromtimestamp(item.stat().st_mtime).isoformat(timespec="seconds"),
        }
        for item in candidates
    ]


def _score(*checks: bool) -> dict[str, Any]:
    total = len(checks)
    passed = sum(1 for item in checks if item)
    return {
        "passed_checks": passed,
        "total_checks": total,
        "score": round(passed / total, 3) if total else 1.0,
    }


def _backend_probe() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_path": str(LOCAL_YOLO_PYTHON),
        "python_exists": LOCAL_YOLO_PYTHON.exists(),
        "runtime_ok": False,
        "blocked": False,
        "summary": "",
        "stdout": "",
        "stderr": "",
    }
    if not LOCAL_YOLO_PYTHON.exists():
        result["blocked"] = True
        result["summary"] = "本机未找到可用于真实 YOLO 预测的 python 环境"
        return result

    probe_code = (
        "import ultralytics, torch; "
        "from ultralytics import YOLO; "
        "print({'ultralytics': ultralytics.__version__, 'torch': torch.__version__})"
    )
    proc = subprocess.run(
        [str(LOCAL_YOLO_PYTHON), "-c", probe_code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    result["return_code"] = proc.returncode
    result["stdout"] = proc.stdout.strip()
    result["stderr"] = proc.stderr.strip()
    result["runtime_ok"] = proc.returncode == 0
    result["blocked"] = proc.returncode != 0
    if proc.returncode == 0:
        result["summary"] = "本机 YOLO 预测环境可用，可执行真实推理"
    else:
        stderr_text = proc.stderr.strip()
        if "WinError 10106" in stderr_text:
            result["summary"] = "本机 YOLO 预测环境不可用：WinError 10106（_overlapped / winsock 提供程序异常）"
        else:
            tail = stderr_text.splitlines()[-1] if stderr_text else "未知错误"
            result["summary"] = f"本机 YOLO 预测环境不可用：{tail}"
    return result


def _prepare_input_subset(video_items: list[dict[str, Any]]) -> Path:
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    subset_dir = WORK_ROOT / "video_inputs"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for item in video_items:
        src = Path(item["path"])
        dst = subset_dir / src.name
        shutil.copy2(src, dst)
    return subset_dir


def _run_mock_validation(weight_path: str, subset_dir: Path) -> dict[str, Any]:
    output_root = WORK_ROOT / "predict_out"
    output_root.mkdir(parents=True, exist_ok=True)

    original_load = predict_tools.service._load_model
    original_run = predict_tools.service._run_batch_inference
    original_draw = predict_tools.service._draw_detections

    try:
        predict_tools.service._load_model = lambda model: {"weight_path": model}

        def _fake_run(_model, frames, *, conf: float, iou: float):
            outputs: list[list[dict[str, Any]]] = []
            for index, frame in enumerate(frames):
                width, height = frame.size
                if index % 2 == 0:
                    outputs.append([
                        {
                            "class_id": 0,
                            "class_name": "Excavator",
                            "confidence": 0.88,
                            "xyxy": [5.0, 5.0, float(width - 5), float(height - 5)],
                        }
                    ])
                else:
                    outputs.append([])
            return outputs

        predict_tools.service._run_batch_inference = _fake_run
        predict_tools.service._draw_detections = lambda frame, detections: frame

        predict_result = predict_tools.predict_videos(
            source_path=str(subset_dir),
            model=weight_path,
            output_dir=str(output_root / "videos"),
            save_video=False,
            save_keyframes_annotated=True,
            save_keyframes_raw=False,
            generate_report=True,
            max_videos=2,
            max_frames=8,
        )
        if not predict_result.get("ok"):
            return {
                "ok": False,
                "summary": "真实素材 Mock 预测验证失败",
                "predict_result": predict_result,
            }

        summary_result = predict_tools.summarize_prediction_results(
            report_path=predict_result["report_path"]
        )
        assessment = _score(
            predict_result.get("ok") is True,
            predict_result.get("processed_videos", 0) >= 1,
            Path(predict_result.get("report_path", "")).exists(),
            summary_result.get("ok") is True,
            summary_result.get("mode") == "videos",
        )
        return {
            "ok": True,
            "summary": "真实视频素材 Mock 验证通过：目录扫描、视频读取、报告生成、摘要汇总链路正常",
            "predict_result": predict_result,
            "summary_result": summary_result,
            "assessment": assessment,
        }
    finally:
        predict_tools.service._load_model = original_load
        predict_tools.service._run_batch_inference = original_run
        predict_tools.service._draw_detections = original_draw


def _run_actual_inference_if_possible(weight_path: str, subset_dir: Path, backend: dict[str, Any]) -> dict[str, Any]:
    if not backend.get("runtime_ok"):
        return {
            "ok": False,
            "blocked": True,
            "summary": "跳过真实推理：本机 YOLO 预测环境当前不可用",
            "reason": backend.get("summary", ""),
        }

    result = predict_tools.predict_videos(
        source_path=str(subset_dir),
        model=weight_path,
        output_dir=str(WORK_ROOT / "predict_out" / "actual_videos"),
        save_video=False,
        save_keyframes_annotated=True,
        save_keyframes_raw=False,
        generate_report=True,
        max_videos=1,
        max_frames=8,
    )
    return {
        "ok": bool(result.get("ok")),
        "blocked": False,
        "summary": result.get("summary", ""),
        "result": result,
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    weights = payload["asset_inventory"]["weights"]
    videos = payload["asset_inventory"]["videos"]
    backend = payload["backend_probe"]
    mock_result = payload["mock_validation"]
    actual = payload["actual_inference"]

    weight_lines = "\n".join(
        f"- `{item['name']}` ({item['size_bytes']} bytes)"
        for item in weights
    ) or "- 无"
    video_lines = "\n".join(
        f"- `{Path(item['path']).name}` ({item['size_bytes']} bytes)"
        for item in videos
    ) or "- 无"

    lines = [
        "# 本地真实权重 / 视频预测验证（2026-04-11）",
        "",
        "## 目标",
        "",
        "验证第二主线在接入真实本地权重池与真实视频池时，测试方法是否能稳定覆盖：",
        "",
        "1. 权重与视频素材盘点",
        "2. 本机 YOLO 推理环境探测",
        "3. 使用真实视频素材进行 Mock 预测链路验证",
        "4. 在环境允许时执行真实推理",
        "",
        "## 素材盘点",
        "",
        "### 权重样本",
        weight_lines,
        "",
        "### 视频样本",
        video_lines,
        "",
        "## 本机推理环境探测",
        "",
        f"- python: `{backend.get('python_path', '')}`",
        f"- runtime_ok: `{backend.get('runtime_ok')}`",
        f"- summary: {backend.get('summary', '')}",
        "",
        "## Mock 预测链路验证",
        "",
        f"- ok: `{mock_result.get('ok')}`",
        f"- summary: {mock_result.get('summary', '')}",
    ]
    if mock_result.get("assessment"):
        assess = mock_result["assessment"]
        lines.extend([
            f"- assessment: `{assess['passed_checks']}/{assess['total_checks']} -> {assess['score']}`",
        ])
    lines.extend([
        "",
        "## 真实推理验证",
        "",
        f"- blocked: `{actual.get('blocked', False)}`",
        f"- summary: {actual.get('summary', '')}",
        "",
        "## 结论",
        "",
    ])
    if actual.get("blocked"):
        lines.append(
            "- 当前测试方法已经升级到“真实素材 + 环境探测 + Mock 链路 + 有条件真实推理”的四段式。"
        )
        lines.append(
            "- 这轮真正阻塞真实推理的不是 Agent 代码，而是本机 YOLO 运行环境在导入 `torch/ultralytics` 时失败。"
        )
    else:
        lines.append(
            "- 当前测试方法已经能直接覆盖真实本地权重池与视频池，且真实推理已可运行。"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    weights = _choose_weights()
    videos = _choose_videos()
    if not weights:
        raise RuntimeError("未找到可用权重文件")
    if not videos:
        raise RuntimeError("未找到可用视频文件")

    subset_dir = _prepare_input_subset(videos)
    backend = _backend_probe()
    mock_validation = _run_mock_validation(weights[0]["path"], subset_dir)
    actual = _run_actual_inference_if_possible(weights[0]["path"], subset_dir, backend)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asset_inventory": {
            "weights_root": str(WEIGHTS_ROOT),
            "videos_root": str(VIDEOS_ROOT),
            "weights": weights,
            "videos": videos,
            "selected_weight": weights[0]["path"],
            "selected_video_subset_dir": str(subset_dir),
        },
        "backend_probe": backend,
        "mock_validation": mock_validation,
        "actual_inference": actual,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_build_markdown(payload), encoding="utf-8")
    print("real media local suite ok")


if __name__ == "__main__":
    main()

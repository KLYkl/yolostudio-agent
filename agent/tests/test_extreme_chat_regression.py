from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any
import types

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import langchain_openai  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_mod

from agent_plan.agent.client.agent_client import AgentSettings, YoloStudioAgentClient


class _DummyGraph:
    def get_state(self, config):
        return None

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError("extreme chat regression should stay on routed/mainline flows")


WORK = Path(__file__).resolve().parent / "_tmp_extreme_chat_regression"


def _wrap(turn: str, body: str) -> str:
    filler = (
        "这是一次故意拉长上下文的压力测试，请你只依赖当前会话中最新且已确认的状态，不要把更早的旧目录、旧工具名或旧结论误当成当前事实。"
        "如果前面已经做过抽取、检查、结果汇总或取消执行之类的动作，请优先相信最近一次真实工具结果，而不是更旧的记忆。"
    )
    return f"[{turn}] {filler}\n{body}\n补充说明：这条消息是为了测试长上下文承受能力，所以文字故意写得更长一些。{filler}"


def _install_fake_tools(client: YoloStudioAgentClient, calls: list[tuple[str, dict[str, Any]]]) -> None:
    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == "preview_extract_images":
            assert kwargs["source_path"] == "/data/raw/images"
            assert kwargs["output_dir"] == "/tmp/extract_preview"
            assert kwargs["selection_mode"] == "count"
            assert kwargs["count"] == 24
            result = {
                "ok": True,
                "summary": "预览完成: 可用图片 180 张，计划抽取 24 张（global / count）",
                "available_images": 180,
                "planned_extract_count": 24,
                "sample_images": ["/data/raw/images/a.jpg", "/data/raw/images/b.jpg"],
                "output_dir": "/tmp/extract_preview",
                "workflow_ready_path": "/tmp/extract_preview",
                "warnings": ["当前只是预览，尚未真正复制文件"],
                "next_actions": ["可继续调用 extract_images 真正执行抽取"],
            }
        elif tool_name == "extract_images":
            assert kwargs["source_path"] == "/data/raw/images"
            assert kwargs["selection_mode"] == "ratio"
            assert abs(float(kwargs["ratio"]) - 0.1) < 1e-9
            assert kwargs["output_dir"] == "/tmp/extract_run"
            result = {
                "ok": True,
                "summary": "图片抽取完成: 实际抽取 18 张图片，复制标签 18 个",
                "extracted": 18,
                "labels_copied": 18,
                "conflict_count": 0,
                "output_dir": "/tmp/extract_run",
                "workflow_ready_path": "/tmp/extract_run",
                "warnings": [],
                "next_actions": ["可直接对输出目录继续 scan_dataset / validate_dataset: /tmp/extract_run"],
                "output_img_dir": "/tmp/extract_run/images",
                "output_label_dir": "/tmp/extract_run/labels",
            }
        elif tool_name == "training_readiness":
            assert kwargs["img_dir"] == "/tmp/extract_run"
            result = {
                "ok": True,
                "summary": "训练前检查完成: 数据可训练，已存在配套标签，建议先做一次 validate 后直接进入准备链",
                "resolved_img_dir": "/tmp/extract_run/images",
                "resolved_label_dir": "/tmp/extract_run/labels",
                "resolved_data_yaml": "/tmp/extract_run/data.yaml",
                "risk_level": "medium",
                "warnings": ["样本量较小，建议先做一次快速回归训练"],
                "blockers": [],
                "next_actions": ["可继续 prepare_dataset_for_training 或 start_training"],
            }
        elif tool_name == "recommend_next_training_step":
            result = {
                "ok": True,
                "summary": "下一步建议: 优先保持小步迭代。",
                "recommended_action": "quick_iteration",
                "basis": ["样本量=180"],
                "why": "当前样本量仍偏小，更适合先做短周期验证。",
                "matched_rule_ids": ["generic_next_small_dataset_fast_iteration"],
                "signals": ["small_dataset"],
                "next_actions": ["先做一次短周期训练", "记录失败样本后再补数据"],
            }
        elif tool_name == "run_dataset_health_check":
            assert kwargs["dataset_path"] == "/tmp/extract_run"
            assert kwargs["include_duplicates"] is True
            result = {
                "ok": True,
                "summary": "健康检查完成: 完整性问题 0，异常尺寸 1，重复组 2",
                "dataset_root": "/tmp/extract_run",
                "resolved_img_dir": "/tmp/extract_run/images",
                "issue_count": 3,
                "duplicate_groups": 2,
                "duplicate_extra_files": 2,
                "risk_level": "medium",
                "warnings": [
                    "发现 1 张异常尺寸图片",
                    "发现 2 组重复图片",
                ],
                "next_actions": ["可根据重复检测结果人工筛查样本"],
            }
        elif tool_name == "scan_videos":
            assert kwargs["source_path"] == "/data/videos"
            result = {
                "ok": True,
                "summary": "视频扫描完成: 发现 3 个视频文件",
                "source_path": "/data/videos",
                "total_videos": 3,
                "sample_videos": ["/data/videos/a.mp4"],
                "warnings": [],
                "next_actions": ["如需抽帧，可继续调用 extract_video_frames"],
            }
        elif tool_name == "extract_video_frames":
            assert kwargs["source_path"] == "/data/videos"
            assert kwargs["output_dir"] == "/tmp/frames_out"
            assert kwargs["mode"] == "interval"
            assert kwargs["frame_interval"] == 10
            result = {
                "ok": True,
                "summary": "视频抽帧完成: 最终保留 12 帧（原始抽取 12 / 去重移除 0）",
                "source_path": "/data/videos",
                "total_frames": 120,
                "extracted": 12,
                "final_count": 12,
                "output_dir": "/tmp/frames_out",
                "warnings": [],
                "next_actions": ["可将抽帧输出目录继续作为图片输入使用: /tmp/frames_out"],
            }
        elif tool_name == "predict_videos":
            assert kwargs["source_path"] == "/data/videos"
            assert kwargs["model"] == "/models/qcar.pt"
            result = {
                "ok": True,
                "summary": "视频预测完成: 已处理 2 个视频, 有检测帧 13, 总检测框 15，主要类别 two_wheeler=15",
                "model": kwargs["model"],
                "source_path": kwargs["source_path"],
                "processed_videos": 2,
                "total_frames": 24,
                "detected_frames": 13,
                "total_detections": 15,
                "class_counts": {"two_wheeler": 15},
                "detected_samples": ["/data/videos/a.mp4#frame12"],
                "output_dir": "/tmp/predict_videos",
                "annotated_dir": "/tmp/predict_videos/annotated",
                "report_path": "/tmp/predict_videos/video_prediction_report.json",
                "warnings": [],
                "next_actions": ["可继续总结刚才预测结果"],
                "mode": "videos",
            }
        elif tool_name == "summarize_prediction_results":
            assert kwargs["report_path"] == "/tmp/predict_videos/video_prediction_report.json"
            result = {
                "ok": True,
                "summary": "预测结果摘要: 已处理 2 个视频, 总帧数 24, 有检测帧 13, 总检测框 15，主要类别 two_wheeler=15",
                "report_path": kwargs["report_path"],
                "output_dir": "/tmp/predict_videos",
                "annotated_dir": "/tmp/predict_videos/annotated",
                "processed_videos": 2,
                "total_frames": 24,
                "detected_frames": 13,
                "total_detections": 15,
                "class_counts": {"two_wheeler": 15},
                "warnings": [],
                "next_actions": ["可查看标注结果目录: /tmp/predict_videos/annotated"],
                "model": "/models/qcar.pt",
                "source_path": "/data/videos",
                "mode": "videos",
            }
        elif tool_name == "prepare_dataset_for_training":
            assert kwargs["dataset_path"] == "/tmp/extract_run"
            result = {
                "ok": True,
                "ready": True,
                "summary": "数据准备完成: 当前数据集可直接训练，data_yaml 已就绪",
                "dataset_root": "/tmp/extract_run",
                "img_dir": "/tmp/extract_run/images",
                "label_dir": "/tmp/extract_run/labels",
                "data_yaml": "/tmp/extract_run/data.yaml",
                "steps_completed": [],
                "next_actions": ["可继续 start_training"],
            }
        else:
            raise AssertionError(f"unexpected tool call: {tool_name}")

        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(
            session_id="extreme-chat-regression",
            memory_root=str(WORK),
            max_history_messages=6,
        )
        calls: list[tuple[str, dict[str, Any]]] = []
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        _install_fake_tools(client, calls)

        transcript: list[dict[str, Any]] = []

        async def _chat(text: str) -> dict[str, Any]:
            result = await client.chat(text)
            transcript.append({"user": text, "status": result["status"], "message": result["message"], "tool_call": result.get("tool_call")})
            return result

        turn1 = await _chat(_wrap("turn-1", "先只预览一下：从 /data/raw/images 里抽 24 张图片到 /tmp/extract_preview，不要真的执行。"))
        assert turn1["status"] == "completed", turn1
        assert "计划抽取 24 张" in turn1["message"], turn1

        turn2 = await _chat(_wrap("turn-2", "现在真的执行：从 /data/raw/images 抽 10% 的图片到 /tmp/extract_run。"))
        assert turn2["status"] == "completed", turn2
        assert "实际抽取 18 张图片" in turn2["message"], turn2
        assert client.session_state.active_dataset.dataset_root == "/tmp/extract_run"

        turn3 = await _chat(_wrap("turn-3", "基于刚才抽出来的那一份，先做训练前检查，不要训练，只告诉我能不能直接训练。"))
        assert turn3["status"] == "completed", turn3
        assert "训练前检查完成" in turn3["message"], turn3

        turn4 = await _chat(_wrap("turn-4", "再对当前这份抽取后的数据做一次健康检查，把重复图片也一起算进去，但不要修改任何数据。"))
        assert turn4["status"] == "completed", turn4
        assert "重复组 2" in turn4["message"], turn4

        turn5 = await _chat(_wrap("turn-5", "顺便扫描一下 /data/videos 目录里有多少视频，先别做别的。"))
        assert turn5["status"] == "completed", turn5
        assert "发现 3 个视频" in turn5["message"], turn5

        turn6 = await _chat(_wrap("turn-6", "从 /data/videos 抽帧，每 10 帧抽 1 帧，输出到 /tmp/frames_out。"))
        assert turn6["status"] == "completed", turn6
        assert "最终保留 12 帧" in turn6["message"], turn6

        turn7 = await _chat(_wrap("turn-7", "请用 /models/qcar.pt 对 /data/videos 做预测，并保留输出文件。"))
        assert turn7["status"] == "completed", turn7
        assert "总检测框 15" in turn7["message"], turn7

        turn8 = await _chat(_wrap("turn-8", "请总结一下刚才预测结果，重点告诉我总帧数、检测帧和主要类别。"))
        assert turn8["status"] == "completed", turn8
        assert "总帧数 24" in turn8["message"], turn8
        assert "two_wheeler=15" in turn8["message"], turn8

        assert len(client.messages) <= settings.max_history_messages
        assert client.session_state.active_dataset.dataset_root == "/tmp/extract_run"
        assert client.session_state.active_prediction.source_path == "/data/videos"
        assert client.session_state.active_prediction.report_path == "/tmp/predict_videos/video_prediction_report.json"

        reloaded = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        _install_fake_tools(reloaded, calls)
        client = reloaded

        turn9 = await _chat(_wrap("turn-9", "现在还是基于 /tmp/extract_run 这份抽取后的数据，用 /models/yolov8n.pt 训练 3 轮。"))
        assert turn9["status"] == "needs_confirmation", turn9
        assert turn9["tool_call"]["name"] == "prepare_dataset_for_training", turn9
        assert turn9["tool_call"]["args"]["dataset_path"] == "/tmp/extract_run", turn9

        turn10 = await client.confirm(turn9["thread_id"], approved=True)
        transcript.append({"user": "[confirm yes]", "status": turn10["status"], "message": turn10["message"], "tool_call": turn10.get("tool_call")})
        assert turn10["status"] == "needs_confirmation", turn10
        assert turn10["tool_call"]["name"] == "start_training", turn10
        assert turn10["tool_call"]["args"]["data_yaml"] == "/tmp/extract_run/data.yaml", turn10
        assert turn10["tool_call"]["args"]["model"] == "/models/yolov8n.pt", turn10
        assert turn10["tool_call"]["args"]["epochs"] == 3, turn10

        turn11 = await client.confirm(turn10["thread_id"], approved=False)
        transcript.append({"user": "[confirm no]", "status": turn11["status"], "message": turn11["message"], "tool_call": turn11.get("tool_call")})
        assert turn11["status"] == "cancelled", turn11
        assert "已取消" in turn11["message"], turn11
        assert not any(name == "start_training" for name, _ in calls), calls

        turn12 = await _chat(_wrap("turn-12", "刚才那个训练我取消了。现在不要训练，重新检查一下当前会话里真正要训练的数据集能不能直接训练。"))
        assert turn12["status"] == "completed", turn12
        assert "训练前检查完成" in turn12["message"], turn12

        turn13 = await _chat(_wrap("turn-13", "再总结一次刚才预测结果，不要把训练准备的内容混进来。"))
        assert turn13["status"] == "completed", turn13
        assert "总检测框 15" in turn13["message"], turn13
        assert "标注结果目录" in turn13["message"], turn13

        assert client.session_state.active_dataset.dataset_root == "/tmp/extract_run"
        assert client.session_state.active_prediction.source_path == "/data/videos"
        assert client.session_state.active_prediction.report_path == "/tmp/predict_videos/video_prediction_report.json"
        assert client.session_state.active_dataset.last_frame_extract["output_dir"] == "/tmp/frames_out"
        assert client.session_state.active_dataset.last_video_scan["total_videos"] == 3
        assert len(client.messages) <= settings.max_history_messages

        expected_order = [
            "preview_extract_images",
            "extract_images",
            "training_readiness",
            "recommend_next_training_step",
            "run_dataset_health_check",
            "scan_videos",
            "extract_video_frames",
            "predict_videos",
            "summarize_prediction_results",
            "prepare_dataset_for_training",
            "training_readiness",
            "recommend_next_training_step",
            "summarize_prediction_results",
        ]
        assert [name for name, _ in calls] == expected_order, calls

        print(json.dumps({
            "status": "ok",
            "tool_calls": [name for name, _ in calls],
            "final_dataset_root": client.session_state.active_dataset.dataset_root,
            "final_prediction_report": client.session_state.active_prediction.report_path,
            "history_len": len(client.messages),
            "transcript_len": len(transcript),
        }, ensure_ascii=False, indent=2))
        print("extreme chat regression ok")
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

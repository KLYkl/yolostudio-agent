from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
import sys
import types

if __package__ in {None, ''}:
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


WORK = Path(__file__).resolve().parent / '_tmp_extract_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='extract-route-smoke', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        calls: list[tuple[str, dict[str, object]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs):
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'preview_extract_images':
                assert kwargs['source_path'] == '/data/raw/images'
                assert kwargs['output_dir'] == '/tmp/extract_preview'
                assert kwargs['selection_mode'] == 'count'
                assert kwargs['count'] == 20
                result = {
                    'ok': True,
                    'summary': '预览完成: 可用图片 180 张，计划抽取 20 张（global / count）',
                    'available_images': 180,
                    'planned_extract_count': 20,
                    'sample_images': ['/data/raw/images/a.jpg'],
                    'output_dir': '/tmp/extract_preview',
                    'warnings': [],
                    'next_actions': ['可继续调用 extract_images 真正执行抽取'],
                }
                client._apply_to_state('preview_extract_images', result, kwargs)
                return result
            if tool_name == 'extract_images':
                assert kwargs['source_path'] == '/data/raw/images'
                assert kwargs['selection_mode'] == 'ratio'
                assert abs(float(kwargs['ratio']) - 0.1) < 1e-9
                result = {
                    'ok': True,
                    'summary': '图片抽取完成: 实际抽取 18 张图片，复制标签 18 个',
                    'extracted': 18,
                    'labels_copied': 18,
                    'conflict_count': 0,
                    'output_dir': '/tmp/extract_run',
                    'workflow_ready_path': '/tmp/extract_run',
                    'warnings': [],
                    'next_actions': ['可直接对输出目录继续 scan_dataset / validate_dataset: /tmp/extract_run'],
                    'output_img_dir': '/tmp/extract_run/images',
                    'output_label_dir': '/tmp/extract_run/labels',
                }
                client._apply_to_state('extract_images', result, kwargs)
                return result
            if tool_name == 'scan_videos':
                assert kwargs['source_path'] == '/data/videos'
                result = {
                    'ok': True,
                    'summary': '视频扫描完成: 发现 3 个视频文件',
                    'total_videos': 3,
                    'sample_videos': ['/data/videos/a.mp4'],
                    'next_actions': ['如需抽帧，可继续调用 extract_video_frames'],
                }
                client._apply_to_state('scan_videos', result, kwargs)
                return result
            assert tool_name == 'extract_video_frames'
            assert kwargs['source_path'] == '/data/videos'
            assert kwargs['output_dir'] == '/tmp/frames_out'
            assert kwargs['mode'] == 'interval'
            assert kwargs['frame_interval'] == 10
            result = {
                'ok': True,
                'summary': '视频抽帧完成: 最终保留 12 帧（原始抽取 12 / 去重移除 0）',
                'total_frames': 120,
                'extracted': 12,
                'final_count': 12,
                'output_dir': '/tmp/frames_out',
                'warnings': [],
                'next_actions': ['可将抽帧输出目录继续作为图片输入使用: /tmp/frames_out'],
            }
            client._apply_to_state('extract_video_frames', result, kwargs)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        routed = await client._try_handle_mainline_intent('先预览一下从 /data/raw/images 抽 20 张图片到 /tmp/extract_preview，不要真的执行', 'thread-1')
        assert routed is not None, routed
        assert routed['status'] == 'completed', routed
        assert '计划抽取 20 张' in routed['message'], routed
        assert calls[-1][0] == 'preview_extract_images', calls

        routed2 = await client._try_handle_mainline_intent('从 /data/raw/images 抽 10% 的图片到 /tmp/extract_run', 'thread-2')
        assert routed2 is not None, routed2
        assert routed2['status'] == 'completed', routed2
        assert '实际抽取 18 张' in routed2['message'], routed2
        assert client.session_state.active_dataset.dataset_root == '/tmp/extract_run'

        routed3 = await client._try_handle_mainline_intent('扫描一下 /data/videos 目录里有多少视频', 'thread-3')
        assert routed3 is not None, routed3
        assert '发现 3 个视频' in routed3['message'], routed3
        assert calls[-1][0] == 'scan_videos', calls

        routed4 = await client._try_handle_mainline_intent('从 /data/videos 抽帧，每 10 帧抽 1 帧，输出到 /tmp/frames_out', 'thread-4')
        assert routed4 is not None, routed4
        assert '最终保留 12 帧' in routed4['message'], routed4
        assert calls[-1][0] == 'extract_video_frames', calls

        print('extract route smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.agent_client import AgentSettings, YoloStudioAgentClient


class _DummyGraph:
    def get_state(self, config):
        return None


WORK = Path(r'D:\yolodo2.0\agent_plan\agent\tests\_tmp_prediction_route')


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='prediction-route-smoke', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})

        async def _fake_direct_tool(tool_name: str, **kwargs):
            assert tool_name == 'predict_images'
            assert kwargs['source_path'] == '/data/images'
            assert kwargs['model'] == '/models/yolov8n.pt'
            result = {
                'ok': True,
                'summary': '预测完成: 已处理 2 张图片, 有检测 1, 无检测 1，主要类别 Excavator=1',
                'model': kwargs['model'],
                'source_path': kwargs['source_path'],
                'processed_images': 2,
                'detected_images': 1,
                'empty_images': 1,
                'class_counts': {'Excavator': 1},
                'detected_samples': ['/data/images/a.jpg'],
                'empty_samples': ['/data/images/b.jpg'],
                'output_dir': '/tmp/predict',
                'annotated_dir': '/tmp/predict/annotated',
                'report_path': '/tmp/predict/prediction_report.json',
                'warnings': [],
                'next_actions': ['可查看标注结果目录: /tmp/predict/annotated'],
            }
            client._apply_to_state('predict_images', result, kwargs)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        routed = await client._try_handle_mainline_intent('请用 /models/yolov8n.pt 预测 /data/images 这个目录里的图片', 'thread-1')
        assert routed is not None, routed
        assert routed['status'] == 'completed', routed
        assert '预测完成' in routed['message'], routed
        assert client.session_state.active_prediction.model == '/models/yolov8n.pt'
        assert client.session_state.active_prediction.source_path == '/data/images'

        routed2 = await client._try_handle_mainline_intent('总结一下刚才预测结果', 'thread-2')
        assert routed2 is not None, routed2
        assert routed2['status'] == 'completed', routed2
        assert '标注结果目录' in routed2['message'], routed2

        print('prediction route smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()

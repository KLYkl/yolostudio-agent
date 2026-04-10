from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any
import sys

from langchain_core.tools import StructuredTool
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from agent_plan.agent.client.tool_adapter import adapt_tools_for_chat_model
from agent_plan.agent.server.tools import predict_tools

OUT_JSON = Path(r'D:\yolodo2.0\agent_plan\agent\tests\test_prediction_regression_suite_output.json')
OUT_MD = Path(r'D:\yolodo2.0\agent_plan\doc\prediction_regression_report_2026-04-11.md')
WORK = Path(r'D:\yolodo2.0\agent_plan\agent\tests\_tmp_prediction_suite')


class _DummyGraph:
    def get_state(self, config):
        return None


def _make_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path, format='JPEG')


def _score(*flags: bool) -> dict[str, Any]:
    total = len(flags)
    passed = sum(1 for item in flags if item)
    return {'passed_checks': passed, 'total_checks': total, 'score': round(passed / total, 3) if total else 1.0}


async def _make_client(session_id: str) -> YoloStudioAgentClient:
    settings = AgentSettings(session_id=session_id, memory_root=str(WORK / 'memory'))
    return YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})


async def case_tool_predict_success(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.time()
    result = predict_tools.predict_images(
        source_path=str(source_dir),
        model='fake-model.pt',
        output_dir=str(output_dir),
        save_annotated=True,
        save_labels=True,
        save_original=True,
        generate_report=True,
    )
    assessment = _score(
        result.get('ok') is True,
        result.get('processed_images') == 3,
        result.get('detected_images') == 2,
        Path(result.get('report_path', '')).exists(),
        'Excavator' in (result.get('class_counts') or {}),
    )
    return {
        'id': 'tool_predict_success',
        'kind': 'tool',
        'duration_sec': round(time.time() - started, 2),
        'result': result,
        'assessment': assessment,
        'expected': 'predict_images 能在目录输入下输出结构化统计和工件路径',
    }


async def case_tool_predict_missing_model(source_dir: Path) -> dict[str, Any]:
    started = time.time()
    result = predict_tools.predict_images(source_path=str(source_dir), model='')
    assessment = _score(
        result.get('ok') is False,
        '缺少模型参数' in str(result.get('summary', '')),
    )
    return {
        'id': 'tool_predict_missing_model',
        'kind': 'tool',
        'duration_sec': round(time.time() - started, 2),
        'result': result,
        'assessment': assessment,
        'expected': '缺少模型时应明确失败，不允许静默预测',
    }


async def case_tool_predict_missing_path() -> dict[str, Any]:
    started = time.time()
    result = predict_tools.predict_images(source_path=str(WORK / 'missing'), model='fake-model.pt')
    assessment = _score(
        result.get('ok') is False,
        '输入路径不存在' in str(result.get('summary', '')),
    )
    return {
        'id': 'tool_predict_missing_path',
        'kind': 'tool',
        'duration_sec': round(time.time() - started, 2),
        'result': result,
        'assessment': assessment,
        'expected': '输入路径不存在时应 fail fast',
    }


async def case_tool_prediction_summary(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.time()
    result = predict_tools.predict_images(
        source_path=str(source_dir),
        model='fake-model.pt',
        output_dir=str(output_dir),
        save_annotated=True,
        save_labels=False,
        save_original=False,
        generate_report=True,
    )
    summary = predict_tools.summarize_prediction_results(report_path=result['report_path'])
    assessment = _score(
        summary.get('ok') is True,
        summary.get('processed_images') == 3,
        summary.get('detected_images') == 2,
        summary.get('total_detections') == 3,
        'Excavator' in (summary.get('class_counts') or {}),
    )
    return {
        'id': 'tool_prediction_summary',
        'kind': 'tool',
        'duration_sec': round(time.time() - started, 2),
        'result': summary,
        'assessment': assessment,
        'expected': 'summarize_prediction_results 应能读取 prediction_report.json 并输出 grounded 统计摘要',
    }


async def case_agent_prediction_route() -> dict[str, Any]:
    started = time.time()
    client = await _make_client(f'prediction-route-{uuid.uuid4().hex[:8]}')

    async def _fake_direct_tool(tool_name: str, **kwargs):
        assert tool_name == 'predict_images'
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
    run = await client._try_handle_mainline_intent('请用 /models/yolov8n.pt 预测 /data/images 这个目录里的图片', 'thread-1')
    assessment = _score(
        run is not None,
        run.get('status') == 'completed',
        client.session_state.active_prediction.source_path == '/data/images',
        client.session_state.active_prediction.model == '/models/yolov8n.pt',
        '标注结果目录' in str(run.get('message', '')),
    )
    return {
        'id': 'agent_prediction_route',
        'kind': 'agent',
        'duration_sec': round(time.time() - started, 2),
        'result': run,
        'assessment': assessment,
        'expected': '同一句话里同时出现模型路径和图片目录时，应优先把非模型路径当 source_path',
    }


async def case_agent_prediction_summary() -> dict[str, Any]:
    started = time.time()
    client = await _make_client(f'prediction-summary-{uuid.uuid4().hex[:8]}')
    client.session_state.active_prediction.source_path = '/data/images'
    client.session_state.active_prediction.model = '/models/yolov8n.pt'
    client.session_state.active_prediction.output_dir = '/tmp/predict'
    client.session_state.active_prediction.report_path = '/tmp/predict/prediction_report.json'
    client.session_state.active_prediction.last_result = {
        'ok': True,
        'summary': '预测完成: 已处理 2 张图片, 有检测 1, 无检测 1，主要类别 Excavator=1',
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
        'model': '/models/yolov8n.pt',
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs):
        calls.append((tool_name, dict(kwargs)))
        assert tool_name == 'summarize_prediction_results'
        assert kwargs['report_path'] == '/tmp/predict/prediction_report.json'
        result = {
            'ok': True,
            'summary': '预测结果摘要: 已处理 2 张图片, 有检测 1, 无检测 1, 总检测框 1，主要类别 Excavator=1',
            'processed_images': 2,
            'detected_images': 1,
            'empty_images': 1,
            'total_detections': 1,
            'class_counts': {'Excavator': 1},
            'detected_samples': ['/data/images/a.jpg'],
            'empty_samples': ['/data/images/b.jpg'],
            'annotated_dir': '/tmp/predict/annotated',
            'report_path': '/tmp/predict/prediction_report.json',
            'output_dir': '/tmp/predict',
            'warnings': [],
            'next_actions': ['可查看标注结果目录: /tmp/predict/annotated'],
            'model': '/models/yolov8n.pt',
            'source_path': '/data/images',
        }
        client._apply_to_state('summarize_prediction_results', result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    run = await client._try_handle_mainline_intent('总结一下刚才预测结果', 'thread-2')
    assessment = _score(
        run is not None,
        run.get('status') == 'completed',
        calls and calls[-1][0] == 'summarize_prediction_results',
        '预测结果摘要' in str(run.get('message', '')),
        '标注结果目录' in str(run.get('message', '')),
        '总检测框 1' in str(run.get('message', '')),
    )
    return {
        'id': 'agent_prediction_summary',
        'kind': 'agent',
        'duration_sec': round(time.time() - started, 2),
        'result': run,
        'assessment': assessment,
        'expected': '已有预测结果时，应优先从 active_prediction 做 grounded 总结',
    }


async def case_tool_alias_catalog() -> dict[str, Any]:
    started = time.time()
    tools = [
        StructuredTool.from_function(func=lambda **kwargs: kwargs, name='detect_duplicate_images', description='dup'),
        StructuredTool.from_function(func=lambda **kwargs: kwargs, name='run_dataset_health_check', description='health'),
        StructuredTool.from_function(func=lambda **kwargs: kwargs, name='prepare_dataset_for_training', description='prepare'),
        StructuredTool.from_function(func=lambda **kwargs: kwargs, name='predict_images', description='predict'),
        StructuredTool.from_function(func=lambda **kwargs: kwargs, name='predict_videos', description='predict-videos'),
        StructuredTool.from_function(func=lambda **kwargs: kwargs, name='summarize_prediction_results', description='predict-summary'),
    ]
    names = {tool.name for tool in adapt_tools_for_chat_model(tools)}
    expected_aliases = {
        'detect_duplicates',
        'detect_corrupted_images',
        'prepare_dataset',
        'dataset_manager.prepare_dataset',
        'predict_directory',
        'batch_predict_images',
        'predict_images_in_dir',
        'predict_video_directory',
        'batch_predict_videos',
        'predict_videos_in_dir',
        'summarize_predictions',
        'summarize_prediction_report',
        'analyze_prediction_report',
    }
    assessment = _score(expected_aliases.issubset(names))
    return {
        'id': 'tool_alias_catalog',
        'kind': 'tooling',
        'duration_sec': round(time.time() - started, 2),
        'names': sorted(names),
        'assessment': assessment,
        'expected': '当前预测/数据主线相关旧工具名都应在 chat model 层注册兼容别名',
    }


async def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    source_dir = WORK / 'images'
    output_dir = WORK / 'predict_out'
    _make_image(source_dir / 'a.jpg', (32, 32), (255, 0, 0))
    _make_image(source_dir / 'b.jpg', (48, 48), (0, 255, 0))
    _make_image(source_dir / 'c.jpg', (64, 64), (0, 0, 255))
    (source_dir / 'broken.jpg').write_text('not-an-image', encoding='utf-8')

    original_load = predict_tools.service._load_model
    original_run = predict_tools.service._run_batch_inference
    original_draw = predict_tools.service._draw_detections
    try:
        predict_tools.service._load_model = lambda model: object()

        def _fake_run(_model, frames, *, conf: float, iou: float):
            outputs = []
            for frame in frames:
                width = int(frame.size[0])
                if width == 32:
                    outputs.append([
                        {'class_id': 0, 'class_name': 'Excavator', 'confidence': 0.91, 'xyxy': [1, 1, 20, 20]},
                    ])
                elif width == 48:
                    outputs.append([])
                else:
                    outputs.append([
                        {'class_id': 1, 'class_name': 'bulldozer', 'confidence': 0.83, 'xyxy': [5, 5, 30, 30]},
                        {'class_id': 1, 'class_name': 'bulldozer', 'confidence': 0.79, 'xyxy': [12, 12, 40, 40]},
                    ])
            return outputs

        predict_tools.service._run_batch_inference = _fake_run
        predict_tools.service._draw_detections = lambda frame, detections: frame

        cases = [
            await case_tool_predict_success(source_dir, output_dir),
            await case_tool_predict_missing_model(source_dir),
            await case_tool_predict_missing_path(),
            await case_tool_prediction_summary(source_dir, output_dir),
            await case_agent_prediction_route(),
            await case_agent_prediction_summary(),
            await case_tool_alias_catalog(),
        ]
    finally:
        predict_tools.service._load_model = original_load
        predict_tools.service._run_batch_inference = original_run
        predict_tools.service._draw_detections = original_draw

    total_checks = sum(int((case.get('assessment') or {}).get('total_checks', 0)) for case in cases)
    passed_checks = sum(int((case.get('assessment') or {}).get('passed_checks', 0)) for case in cases)
    failed_cases = [case['id'] for case in cases if (case.get('assessment') or {}).get('passed_checks') != (case.get('assessment') or {}).get('total_checks')]
    payload = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'case_count': len(cases),
        'passed_checks': passed_checks,
        'total_checks': total_checks,
        'score': round(passed_checks / total_checks, 3) if total_checks else 1.0,
        'failed_cases': failed_cases,
        'cases': cases,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# 第二主线 Phase 1 预测回归报告（2026-04-11）',
        '',
        f'- Case 数量: `{len(cases)}`',
        f'- 检查项通过: `{passed_checks}/{total_checks}`',
        f'- 总体得分: `{payload["score"]}`',
        '',
        '## 1. 总结',
        '',
        '- 当前回归覆盖了：tool 成功/失败路径、预测意图路由、已有预测结果 grounded 总结、旧工具名兼容。',
        f'- 未满分 case: `{", ".join(failed_cases) if failed_cases else "无"}`',
        '',
        '## 2. Case 结果',
        '',
    ]
    for case in cases:
        assessment = case.get('assessment') or {}
        lines.append(f"### {case['id']}")
        lines.append(f"- 预期: {case.get('expected', '')}")
        lines.append(f"- 得分: {assessment.get('passed_checks', 0)}/{assessment.get('total_checks', 0)} ({assessment.get('score', 0)})")
        if case.get('result'):
            preview = str(case['result']).replace('\n', ' ')[:260]
            lines.append(f'- 摘要: {preview}')
        if case.get('names'):
            lines.append(f"- 别名工具: `{', '.join(case.get('names', []))}`")
        lines.append('')
    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'case_count': len(cases), 'score': payload['score'], 'output_json': str(OUT_JSON), 'output_md': str(OUT_MD)}, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())

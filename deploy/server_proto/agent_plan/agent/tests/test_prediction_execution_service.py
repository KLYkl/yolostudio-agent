from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.prediction_execution_service import run_remote_prediction_pipeline_flow


async def _scenario_remote_prediction_pipeline_flow_success() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: object) -> dict[str, object]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'upload_assets_to_remote':
            return {
                'ok': True,
                'target_label': 'yolostudio',
                'remote_root': '/tmp/predict_stage',
            }
        if tool_name == 'predict_images':
            return {
                'ok': True,
                'summary': '远端图片预测完成',
                'output_dir': '/tmp/predict_stage/output',
                'report_path': '/tmp/predict_stage/output/prediction_report.json',
            }
        if tool_name == 'download_assets_from_remote':
            return {
                'ok': True,
                'local_root': 'D:/tmp/predict_output',
            }
        raise AssertionError(tool_name)

    def _resolve_prediction_remote_inputs(_: dict[str, object]) -> dict[str, object]:
        return {
            'ok': True,
            'tool_name': 'predict_images',
            'model_path': '/tmp/predict_stage/best.pt',
            'source_path': '/tmp/predict_stage/images',
            'source_kind': 'image',
        }

    result = await run_remote_prediction_pipeline_flow(
        pipeline_args={
            'upload_args': {'server': 'yolostudio', 'remote_root': '/tmp/predict_stage'},
            'download_after_predict': True,
            'local_result_root': 'D:/tmp/predict_output',
        },
        direct_tool=_fake_direct_tool,
        resolve_prediction_remote_inputs=_resolve_prediction_remote_inputs,
        build_remote_output_dir=lambda upload_result: str(upload_result.get('remote_root') or '') + '/generated_output',
    )
    assert result['stage'] == 'completed', result
    assert [name for name, _ in calls] == [
        'upload_assets_to_remote',
        'predict_images',
        'download_assets_from_remote',
    ], calls
    predict_kwargs = calls[1][1]
    assert predict_kwargs['source_path'] == '/tmp/predict_stage/images', predict_kwargs
    assert predict_kwargs['model'] == '/tmp/predict_stage/best.pt', predict_kwargs
    assert predict_kwargs['save_annotated'] is True, predict_kwargs
    assert predict_kwargs['save_labels'] is False, predict_kwargs
    pipeline_result = dict(result['pipeline_result'])
    assert pipeline_result['ok'] is True, pipeline_result
    assert pipeline_result['remote_output_dir'] == '/tmp/predict_stage/output', pipeline_result
    assert pipeline_result['local_result_root'] == 'D:/tmp/predict_output', pipeline_result
    assert pipeline_result['action_candidates'] == [
        {'tool': 'inspect_prediction_outputs', 'description': '可继续查看本机结果目录: D:/tmp/predict_output'},
        {'tool': 'summarize_prediction_results', 'description': '可继续汇总远端预测报告: /tmp/predict_stage/output/prediction_report.json'},
    ], pipeline_result


async def _scenario_remote_prediction_pipeline_flow_resolve_failure() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: object) -> dict[str, object]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'upload_assets_to_remote':
            return {'ok': True, 'target_label': 'yolostudio'}
        raise AssertionError(tool_name)

    result = await run_remote_prediction_pipeline_flow(
        pipeline_args={'upload_args': {'server': 'yolostudio'}},
        direct_tool=_fake_direct_tool,
        resolve_prediction_remote_inputs=lambda _: {'ok': False, 'error': '缺少图片或视频输入'},
        build_remote_output_dir=lambda _: '/tmp/predict_stage/generated_output',
    )
    assert result['stage'] == 'resolve', result
    assert [name for name, _ in calls] == ['upload_assets_to_remote'], calls
    assert result['resolved_inputs'] == {'ok': False, 'error': '缺少图片或视频输入'}, result


async def _scenario_remote_prediction_pipeline_flow_predict_failure() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: object) -> dict[str, object]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'upload_assets_to_remote':
            return {'ok': True, 'target_label': 'yolostudio'}
        if tool_name == 'predict_videos':
            return {'ok': False, 'error': '远端视频预测失败'}
        raise AssertionError(tool_name)

    result = await run_remote_prediction_pipeline_flow(
        pipeline_args={'upload_args': {'server': 'yolostudio'}},
        direct_tool=_fake_direct_tool,
        resolve_prediction_remote_inputs=lambda _: {
            'ok': True,
            'tool_name': 'predict_videos',
            'model_path': '/tmp/predict_stage/best.pt',
            'source_path': '/tmp/predict_stage/video.mp4',
            'source_kind': 'video',
        },
        build_remote_output_dir=lambda _: '/tmp/predict_stage/generated_output',
    )
    assert result['stage'] == 'predict', result
    assert [name for name, _ in calls] == ['upload_assets_to_remote', 'predict_videos'], calls
    predict_kwargs = calls[1][1]
    assert predict_kwargs['save_video'] is False, predict_kwargs
    assert predict_kwargs['save_keyframes_annotated'] is True, predict_kwargs
    assert predict_kwargs['save_keyframes_raw'] is False, predict_kwargs
    assert result['predict'] == {'ok': False, 'error': '远端视频预测失败'}, result


async def _run() -> None:
    await _scenario_remote_prediction_pipeline_flow_success()
    await _scenario_remote_prediction_pipeline_flow_resolve_failure()
    await _scenario_remote_prediction_pipeline_flow_predict_failure()
    print('prediction execution service ok')


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()

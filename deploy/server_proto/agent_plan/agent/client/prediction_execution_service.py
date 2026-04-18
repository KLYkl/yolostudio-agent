from __future__ import annotations

from typing import Any, Awaitable, Callable

DirectToolInvoker = Callable[..., Awaitable[dict[str, Any]]]
PredictionInputResolver = Callable[[dict[str, Any]], dict[str, Any]]
RemoteOutputDirBuilder = Callable[[dict[str, Any]], str]


async def run_remote_prediction_pipeline_flow(
    *,
    pipeline_args: dict[str, Any] | None,
    direct_tool: DirectToolInvoker,
    resolve_prediction_remote_inputs: PredictionInputResolver,
    build_remote_output_dir: RemoteOutputDirBuilder,
) -> dict[str, Any]:
    pipeline_args = dict(pipeline_args or {})
    upload_args = dict(pipeline_args.get('upload_args') or {})
    upload_result = await direct_tool('upload_assets_to_remote', **upload_args)
    if not upload_result.get('ok'):
        return {
            'stage': 'upload',
            'upload': upload_result,
            'resolved_inputs': {},
            'predict': {},
            'download': {},
            'pipeline_result': {},
        }

    remote_output_dir = build_remote_output_dir(upload_result)
    resolved_inputs = dict(resolve_prediction_remote_inputs(upload_result) or {})
    if not resolved_inputs.get('ok'):
        return {
            'stage': 'resolve',
            'upload': upload_result,
            'resolved_inputs': resolved_inputs,
            'predict': {},
            'download': {},
            'pipeline_result': {},
        }

    predict_tool_name = str(resolved_inputs.get('tool_name') or '')
    predict_kwargs: dict[str, Any] = {
        'source_path': str(resolved_inputs.get('source_path') or ''),
        'model': str(resolved_inputs.get('model_path') or ''),
        'output_dir': remote_output_dir,
        'generate_report': True,
    }
    if predict_tool_name == 'predict_videos':
        predict_kwargs.update({
            'save_video': False,
            'save_keyframes_annotated': True,
            'save_keyframes_raw': False,
        })
    else:
        predict_kwargs.update({
            'save_annotated': True,
            'save_labels': False,
            'save_original': False,
        })

    predict_result = await direct_tool(predict_tool_name, **predict_kwargs)
    if not predict_result.get('ok'):
        return {
            'stage': 'predict',
            'upload': upload_result,
            'resolved_inputs': resolved_inputs,
            'predict': predict_result,
            'download': {},
            'pipeline_result': {},
        }

    download_result: dict[str, Any] = {}
    if pipeline_args.get('download_after_predict', True):
        download_args = {
            'remote_paths': [str(predict_result.get('output_dir') or remote_output_dir)],
            'server': upload_args.get('server', ''),
            'profile': upload_args.get('profile', ''),
            'host': upload_args.get('host', ''),
            'username': upload_args.get('username', ''),
            'port': upload_args.get('port', 0),
            'local_root': pipeline_args.get('local_result_root', ''),
            'recursive': True,
        }
        download_result = await direct_tool('download_assets_from_remote', **download_args)

    pipeline_result = {
        'ok': predict_result.get('ok') is True and (not download_result or download_result.get('ok') is True),
        'upload': upload_result,
        'predict': predict_result,
        'download': download_result,
        'remote_source_path': str(resolved_inputs.get('source_path') or ''),
        'remote_model_path': str(resolved_inputs.get('model_path') or ''),
        'remote_output_dir': str(predict_result.get('output_dir') or remote_output_dir),
        'local_result_root': str((download_result or {}).get('local_root') or pipeline_args.get('local_result_root') or ''),
        'source_kind': str(resolved_inputs.get('source_kind') or ''),
        'predict_tool_name': predict_tool_name,
    }
    pipeline_result['pipeline_overview'] = {
        'target_label': str(upload_result.get('target_label') or upload_args.get('server') or '').strip(),
        'remote_root': str(upload_result.get('remote_root') or upload_args.get('remote_root') or '').strip(),
        'remote_source_path': pipeline_result['remote_source_path'],
        'remote_model_path': pipeline_result['remote_model_path'],
        'remote_output_dir': pipeline_result['remote_output_dir'],
        'local_result_root': pipeline_result['local_result_root'],
        'source_kind': pipeline_result['source_kind'],
    }
    pipeline_result['execution_overview'] = {
        'upload_ok': bool(upload_result.get('ok')),
        'predict_ok': bool(predict_result.get('ok')),
        'download_ok': bool((not download_result) or download_result.get('ok')),
        'predict_tool_name': predict_tool_name,
        'download_after_predict': bool(pipeline_args.get('download_after_predict', True)),
    }
    action_candidates: list[dict[str, Any]] = []
    if pipeline_result['local_result_root']:
        action_candidates.append({
            'tool': 'inspect_prediction_outputs',
            'description': f"可继续查看本机结果目录: {pipeline_result['local_result_root']}",
        })
    elif pipeline_result['remote_output_dir']:
        action_candidates.append({
            'tool': 'download_assets_from_remote',
            'description': f"如需回传，可继续下载远端预测目录: {pipeline_result['remote_output_dir']}",
        })
    report_path = str((predict_result.get('report_path') or '')).strip()
    if report_path:
        action_candidates.append({
            'tool': 'summarize_prediction_results',
            'description': f"可继续汇总远端预测报告: {report_path}",
        })
    if action_candidates:
        pipeline_result['action_candidates'] = action_candidates[:4]

    return {
        'stage': 'completed',
        'upload': upload_result,
        'resolved_inputs': resolved_inputs,
        'predict': predict_result,
        'download': download_result,
        'pipeline_result': pipeline_result,
    }

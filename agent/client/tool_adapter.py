from __future__ import annotations

import json
from typing import Any, Sequence, get_args, get_origin

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


TOOL_NAME_ALIASES: dict[str, str] = {
    'detect_duplicates': 'detect_duplicate_images',
    'detect_corrupted_images': 'run_dataset_health_check',
    'prepare_dataset': 'prepare_dataset_for_training',
    'dataset_manager.prepare_dataset': 'prepare_dataset_for_training',
    'dataset_readiness': 'dataset_training_readiness',
    'check_dataset_training_readiness': 'dataset_training_readiness',
    'predict_directory': 'predict_images',
    'batch_predict_images': 'predict_images',
    'predict_images_in_dir': 'predict_images',
    'predict_video_directory': 'predict_videos',
    'batch_predict_videos': 'predict_videos',
    'predict_videos_in_dir': 'predict_videos',
    'summarize_predictions': 'summarize_prediction_results',
    'summarize_prediction_report': 'summarize_prediction_results',
    'analyze_prediction_report': 'summarize_prediction_results',
    'inspect_prediction_output': 'inspect_prediction_outputs',
    'show_prediction_outputs': 'inspect_prediction_outputs',
    'prediction_output_overview': 'inspect_prediction_outputs',
    'export_prediction_summary': 'export_prediction_report',
    'write_prediction_report': 'export_prediction_report',
    'export_prediction_paths': 'export_prediction_path_lists',
    'collect_prediction_hits': 'organize_prediction_results',
    'group_prediction_results': 'organize_prediction_results',
    'scan_available_cameras': 'scan_cameras',
    'scan_available_screens': 'scan_screens',
    'probe_rtsp_stream': 'test_rtsp_stream',
    'start_live_camera_prediction': 'start_camera_prediction',
    'start_live_rtsp_prediction': 'start_rtsp_prediction',
    'start_live_screen_prediction': 'start_screen_prediction',
    'check_live_prediction_status': 'check_realtime_prediction_status',
    'stop_live_prediction': 'stop_realtime_prediction',
    'preview_extract': 'preview_extract_images',
    'extract_frames': 'extract_video_frames',
    'scan_video_directory': 'scan_videos',
    'preview_convert_labels': 'preview_convert_format',
    'convert_labels_format': 'convert_format',
    'preview_replace_labels': 'preview_modify_labels',
    'replace_labels': 'modify_labels',
    'fill_missing_labels': 'generate_missing_labels',
    'create_empty_labels': 'generate_empty_labels',
    'preview_group_by_class': 'preview_categorize_by_class',
    'group_by_class': 'categorize_by_class',
    'search_training_knowledge': 'retrieve_training_knowledge',
    'explain_training_metrics': 'analyze_training_outcome',
    'recommend_training_next_step': 'recommend_next_training_step',
    'get_training_run': 'inspect_training_run',
    'show_training_run': 'inspect_training_run',
    'training_run_detail': 'inspect_training_run',
    'compare_training_history': 'compare_training_runs',
    'compare_training_results': 'compare_training_runs',
    'best_training_run': 'select_best_training_run',
    'pick_best_training_run': 'select_best_training_run',
    'list_remote_servers': 'list_remote_profiles',
    'list_remote_targets': 'list_remote_profiles',
    'show_remote_profiles': 'list_remote_profiles',
    'upload_to_server': 'upload_assets_to_remote',
    'upload_to_remote': 'upload_assets_to_remote',
    'sync_local_to_remote': 'upload_assets_to_remote',
    'scp_to_server': 'upload_assets_to_remote',
    'download_from_server': 'download_assets_from_remote',
    'download_from_remote': 'download_assets_from_remote',
    'pull_from_server': 'download_assets_from_remote',
    'sync_remote_to_local': 'download_assets_from_remote',
    'scp_from_server': 'download_assets_from_remote',
}

_MODEL_VISIBLE_DESCRIPTION_OVERRIDES: dict[str, str] = {
    'dataset_training_readiness': (
        '只检查数据集本身是否已经具备直接训练的结构条件。'
        ' 适用于“这份数据能不能直接训练”“是否还缺 data.yaml”“是否还需要划分 train/val”这类问题。'
        ' 不要检查 GPU、device 或训练环境。'
    ),
    'training_readiness': (
        '只在用户准备现在启动训练，或明确询问训练执行条件时使用。'
        ' 适用于“现在能不能开训”“GPU / device / 训练环境是否就绪”这类问题。'
        ' 不要用于纯数据集可训练性问题。'
    ),
    'prepare_dataset_for_training': (
        '把数据集整理成可训练状态：补齐 data.yaml，并在需要时划分 train/val。'
        ' 如果用户已经明确给出 classes.txt 或直接贴了类别名，请显式传 classes_txt 或 classes_text，避免猜测类名来源。'
        ' 适用于“先准备数据”“先整理数据”“先划分再训练”这类请求。'
    ),
    'split_dataset': (
        '划分数据集为 train/val。'
        ' img_dir 可以直接传数据集根目录，工具会自动解析 images/labels。'
        ' 只有在用户明确要划分数据集时使用。'
        ' 如果路径不存在，或划分后没有得到非空 train/val，本工具会明确失败。'
    ),
    'augment_dataset': (
        '对数据集做离线增强并写到新目录。'
        ' img_dir 可以直接传数据集根目录，工具会自动解析 images/labels。'
        ' 如果没有可处理图片或没有产出文件，本工具会明确失败。'
    ),
    'convert_format': (
        '执行标签格式转换并写到新目录。'
        ' dataset_path 传数据集根目录即可。'
        ' 如果没有可转换标签文件，或没有真正产出目标格式文件，本工具会明确失败。'
    ),
    'categorize_by_class': (
        '按类别把图片和标签复制整理到新目录。'
        ' dataset_path 传数据集根目录即可。'
        ' 如果没有真正产出分类目录和文件，本工具会明确失败。'
    ),
    'generate_yaml': (
        '为已经确定的 train/val 目录生成训练 YAML。'
        ' 如果用户直接在对话里贴了类别名，请优先把原文放进 classes_text；'
        ' 如果用户已经明确给出了类别列表，也可以直接传 classes。'
        ' 只有在没有显式类别信息时，才考虑使用 classes_txt 或现有 data.yaml。'
    ),
}

_ARG_ALIASES: dict[str, dict[str, str]] = {
    'run_dataset_health_check': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
    },
    'detect_duplicate_images': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
    },
    'prepare_dataset_for_training': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
    },
    'dataset_training_readiness': {
        'path': 'img_dir',
        'dataset_path': 'img_dir',
        'dataset': 'img_dir',
        'root': 'img_dir',
    },
    'training_readiness': {
        'path': 'img_dir',
        'dataset_path': 'img_dir',
        'dataset': 'img_dir',
        'root': 'img_dir',
    },
    'scan_dataset': {
        'path': 'img_dir',
        'dataset_path': 'img_dir',
        'dataset': 'img_dir',
        'root': 'img_dir',
    },
    'validate_dataset': {
        'path': 'img_dir',
        'dataset_path': 'img_dir',
        'dataset': 'img_dir',
        'root': 'img_dir',
    },
    'split_dataset': {
        'path': 'img_dir',
        'dataset_path': 'img_dir',
        'dataset': 'img_dir',
        'root': 'img_dir',
        'dir_path': 'img_dir',
        'folder': 'img_dir',
        'out_dir': 'output_dir',
        'output': 'output_dir',
        'split_ratio': 'ratio',
    },
    'augment_dataset': {
        'path': 'img_dir',
        'dataset_path': 'img_dir',
        'dataset': 'img_dir',
        'root': 'img_dir',
        'dir_path': 'img_dir',
        'folder': 'img_dir',
        'out_dir': 'output_dir',
        'output': 'output_dir',
        'format': 'label_format',
    },
    'predict_images': {
        'path': 'source_path',
        'source': 'source_path',
        'input_path': 'source_path',
        'dir_path': 'source_path',
        'folder': 'source_path',
    },
    'predict_videos': {
        'path': 'source_path',
        'source': 'source_path',
        'input_path': 'source_path',
        'dir_path': 'source_path',
        'folder': 'source_path',
    },
    'summarize_prediction_results': {
        'path': 'report_path',
        'report': 'report_path',
        'json_report': 'report_path',
        'file': 'report_path',
        'dir_path': 'output_dir',
        'folder': 'output_dir',
        'output': 'output_dir',
    },
    'inspect_prediction_outputs': {
        'path': 'report_path',
        'report': 'report_path',
        'json_report': 'report_path',
        'file': 'report_path',
        'dir_path': 'output_dir',
        'folder': 'output_dir',
        'output': 'output_dir',
    },
    'export_prediction_report': {
        'path': 'report_path',
        'report': 'report_path',
        'json_report': 'report_path',
        'file': 'report_path',
        'dir_path': 'output_dir',
        'folder': 'output_dir',
        'output': 'output_dir',
        'out_dir': 'export_path',
        'format': 'export_format',
    },
    'export_prediction_path_lists': {
        'path': 'report_path',
        'report': 'report_path',
        'json_report': 'report_path',
        'file': 'report_path',
        'dir_path': 'output_dir',
        'folder': 'output_dir',
        'output': 'output_dir',
        'out_dir': 'export_dir',
    },
    'organize_prediction_results': {
        'path': 'report_path',
        'report': 'report_path',
        'json_report': 'report_path',
        'file': 'report_path',
        'dir_path': 'output_dir',
        'folder': 'output_dir',
        'output': 'output_dir',
        'out_dir': 'destination_dir',
        'mode': 'organize_by',
        'format': 'artifact_preference',
    },
    'test_rtsp_stream': {
        'path': 'rtsp_url',
        'source': 'rtsp_url',
        'url': 'rtsp_url',
    },
    'start_camera_prediction': {
        'path': 'model',
        'source': 'model',
    },
    'start_rtsp_prediction': {
        'path': 'model',
        'source': 'model',
        'url': 'rtsp_url',
    },
    'start_screen_prediction': {
        'path': 'model',
        'source': 'model',
    },
    'preview_extract_images': {
        'path': 'source_path',
        'source': 'source_path',
        'input_path': 'source_path',
        'dir_path': 'source_path',
        'folder': 'source_path',
        'out_dir': 'output_dir',
    },
    'extract_images': {
        'path': 'source_path',
        'source': 'source_path',
        'input_path': 'source_path',
        'dir_path': 'source_path',
        'folder': 'source_path',
        'out_dir': 'output_dir',
    },
    'scan_videos': {
        'path': 'source_path',
        'source': 'source_path',
        'input_path': 'source_path',
        'dir_path': 'source_path',
        'folder': 'source_path',
    },
    'extract_video_frames': {
        'path': 'source_path',
        'source': 'source_path',
        'input_path': 'source_path',
        'dir_path': 'source_path',
        'folder': 'source_path',
        'out_dir': 'output_dir',
    },
    'preview_convert_format': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
        'out_dir': 'output_dir',
        'format': 'target_format',
    },
    'convert_format': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
        'out_dir': 'output_dir',
        'format': 'target_format',
    },
    'preview_modify_labels': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
        'out_dir': 'output_dir',
        'from': 'old_value',
        'to': 'new_value',
        'operation': 'action',
    },
    'modify_labels': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
        'out_dir': 'output_dir',
        'from': 'old_value',
        'to': 'new_value',
        'operation': 'action',
    },
    'clean_orphan_labels': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
    },
    'preview_generate_empty_labels': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
        'out_dir': 'output_dir',
        'format': 'label_format',
    },
    'generate_empty_labels': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
        'out_dir': 'output_dir',
        'format': 'label_format',
    },
    'preview_generate_missing_labels': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
        'out_dir': 'output_dir',
        'format': 'label_format',
    },
    'generate_missing_labels': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
        'out_dir': 'output_dir',
        'format': 'label_format',
    },
    'preview_categorize_by_class': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
        'out_dir': 'output_dir',
    },
    'categorize_by_class': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
        'out_dir': 'output_dir',
    },
    'upload_assets_to_remote': {
        'path': 'paths_text',
        'paths': 'paths_text',
        'local_path': 'paths_text',
        'source': 'paths_text',
        'server_alias': 'server',
        'server_name': 'server',
        'target': 'server',
        'remote_dir': 'remote_root',
        'remote_path': 'remote_root',
        'remote_output_dir': 'remote_root',
        'user': 'username',
        'resume_upload': 'resume',
        'resume_transfer': 'resume',
        'verify': 'verify_hash',
        'hash_algo': 'hash_algorithm',
        'hash_name': 'hash_algorithm',
        'threshold_mb': 'large_file_threshold_mb',
        'large_file_mb': 'large_file_threshold_mb',
        'chunk_mb': 'chunk_size_mb',
        'progress': 'show_progress',
    },
    'download_assets_from_remote': {
        'path': 'paths_text',
        'paths': 'paths_text',
        'remote_path': 'paths_text',
        'remote_dir': 'paths_text',
        'source': 'paths_text',
        'server_alias': 'server',
        'server_name': 'server',
        'target': 'server',
        'user': 'username',
        'local_dir': 'local_root',
        'local_path': 'local_root',
        'out_dir': 'local_root',
    },
}


def _summarize_tool_result_mapping(value: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = str(value.get('summary') or value.get('message') or '').strip()
    if summary:
        lines.append(summary)

    if value.get('ok') is False:
        error = str(value.get('error') or '').strip()
        if error:
            lines.append(f'错误: {error}')

    blockers = [str(item).strip() for item in (value.get('blockers') or []) if str(item).strip()]
    if blockers:
        lines.append('阻塞项: ' + '；'.join(blockers[:3]))

    warnings = [str(item).strip() for item in (value.get('warnings') or []) if str(item).strip()]
    if warnings:
        lines.append('注意: ' + '；'.join(warnings[:3]))

    next_step_summary = str(value.get('next_step_summary') or '').strip()
    if next_step_summary and next_step_summary not in lines:
        lines.append(f'下一步: {next_step_summary}')

    overview_labels = {
        'readiness_overview': '概览',
        'prepare_overview': '准备概览',
        'environment_overview': '环境概览',
        'preflight_overview': '预检概览',
        'status_overview': '状态概览',
        'summary_overview': '摘要概览',
        'gpu_overview': 'GPU 概览',
        'matched_rule_overview': '命中规则',
        'playbook_overview': '命中方案',
        'retrieval_overview': '检索概览',
        'analysis_overview': '分析概览',
        'recommendation_overview': '建议概览',
        'prediction_overview': '预测概览',
        'prediction_summary_overview': '预测摘要概览',
        'prediction_output_overview': '预测输出概览',
        'organization_overview': '整理概览',
        'export_overview': '导出概览',
        'path_list_overview': '路径清单概览',
        'camera_overview': '摄像头概览',
        'screen_overview': '屏幕概览',
        'stream_test_overview': '流测试概览',
        'realtime_session_overview': '实时会话概览',
        'realtime_status_overview': '实时状态概览',
        'extract_preview_overview': '抽取预览概览',
        'extract_overview': '抽取概览',
        'video_scan_overview': '视频扫描概览',
        'frame_extract_overview': '抽帧概览',
        'profile_overview': '远端配置概览',
        'transfer_overview': '上传概览',
        'download_overview': '下载概览',
    }
    for key, label in overview_labels.items():
        overview = value.get(key)
        if isinstance(overview, dict) and overview:
            compact_bits = [
                f'{sub_key}={sub_value}'
                for sub_key, sub_value in overview.items()
                if sub_value not in (None, '', [], {})
            ]
            if compact_bits:
                lines.append(f'{label}: ' + ', '.join(compact_bits[:6]))
        elif isinstance(overview, Sequence) and not isinstance(overview, (str, bytes, bytearray)):
            compact_items: list[str] = []
            for item in list(overview)[:3]:
                if isinstance(item, dict):
                    bits = [
                        f'{sub_key}={sub_value}'
                        for sub_key, sub_value in item.items()
                        if sub_value not in (None, '', [], {})
                    ]
                    if bits:
                        compact_items.append('{' + ', '.join(bits[:4]) + '}')
                else:
                    text = str(item).strip()
                    if text:
                        compact_items.append(text)
            if compact_items:
                lines.append(f'{label}: ' + '；'.join(compact_items))

    action_candidates = value.get('action_candidates') or []
    has_structured_actions = False
    if isinstance(action_candidates, Sequence) and not isinstance(action_candidates, (str, bytes, bytearray)):
        descriptions: list[str] = []
        for item in action_candidates[:3]:
            if not isinstance(item, dict):
                continue
            tool = str(item.get('tool') or '').strip()
            action = str(item.get('action') or '').strip()
            description = str(item.get('description') or '').strip()
            fragment = description or ' / '.join(part for part in (action, tool) if part)
            if fragment:
                descriptions.append(fragment)
        if descriptions:
            has_structured_actions = True
            lines.append('建议动作: ' + '；'.join(descriptions))

    next_actions = value.get('next_actions') or []
    if (not has_structured_actions) and isinstance(next_actions, Sequence) and not isinstance(next_actions, (str, bytes, bytearray)):
        descriptions: list[str] = []
        for item in next_actions[:3]:
            if isinstance(item, dict):
                description = str(item.get('description') or '').strip()
                if description:
                    descriptions.append(description)
            else:
                description = str(item).strip()
                if description:
                    descriptions.append(description)
        if descriptions:
            lines.append('建议动作: ' + '；'.join(descriptions))

    path_fields = (
        ('dataset_root', '数据集'),
        ('resolved_data_yaml', 'data.yaml'),
        ('output_dir', '输出目录'),
        ('export_path', '导出路径'),
        ('export_dir', '导出目录'),
        ('destination_dir', '目标目录'),
        ('report_path', '报告路径'),
        ('source_output_dir', '源输出目录'),
        ('source_report_path', '源报告路径'),
        ('save_dir', '结果目录'),
    )
    for field, label in path_fields:
        path_value = str(value.get(field) or '').strip()
        if path_value:
            lines.append(f'{label}: {path_value}')

    metric_bits: list[str] = []
    for field in ('processed_images', 'detected_images', 'empty_images', 'missing_label_images', 'issue_count', 'epoch', 'total_epochs'):
        metric_value = value.get(field)
        if metric_value in (None, '', [], {}):
            continue
        metric_bits.append(f'{field}={metric_value}')
    if metric_bits:
        lines.append('关键计数: ' + ', '.join(metric_bits))

    compact = '\n'.join(line for line in lines if line).strip()
    if compact:
        return compact

    scalar_snapshot = {
        key: current
        for key, current in value.items()
        if isinstance(current, (str, int, float, bool)) and key not in {'tool', 'tool_name'}
    }
    if scalar_snapshot:
        return json.dumps(scalar_snapshot, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def _stringify_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get('type') == 'text' and item.get('text'):
                    parts.append(str(item['text']))
                else:
                    parts.append(_summarize_tool_result_mapping(item))
            else:
                parts.append(str(item))
        return '\n'.join(part for part in parts if part)
    if isinstance(value, dict):
        return _summarize_tool_result_mapping(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def stringify_tool_result_facts(value: Any) -> str:
    return _stringify_tool_result(value)


def canonical_tool_name(name: str) -> str:
    key = (name or '').strip()
    return TOOL_NAME_ALIASES.get(key, key)


def _is_missing_tool_arg(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ''
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def normalize_tool_args(tool_name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    canonical_name = canonical_tool_name(tool_name)
    payload = dict(args or {})
    for alias, target in _ARG_ALIASES.get(canonical_name, {}).items():
        if alias in payload and _is_missing_tool_arg(payload.get(target)) and not _is_missing_tool_arg(payload.get(alias)):
            payload[target] = payload[alias]
    return payload


def _annotation_contains(annotation: Any, target_type: type[Any]) -> bool:
    if annotation is target_type:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    if origin is target_type:
        return True
    return any(_annotation_contains(arg, target_type) for arg in get_args(annotation))


def _sanitize_tool_args_for_schema(tool: BaseTool, args: dict[str, Any]) -> dict[str, Any]:
    schema = getattr(tool, 'args_schema', None)
    fields = getattr(schema, 'model_fields', None)
    if not fields:
        return args
    sanitized = dict(args)
    for field_name, field_info in fields.items():
        if sanitized.get(field_name, object()) is not None:
            continue
        annotation = getattr(field_info, 'annotation', None)
        if _annotation_contains(annotation, str):
            sanitized[field_name] = ''
        elif _annotation_contains(annotation, list):
            sanitized[field_name] = []
        elif _annotation_contains(annotation, dict):
            sanitized[field_name] = {}
    return sanitized


def _prepare_tool_args(tool: BaseTool, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_tool_args(tool_name, args)
    return _sanitize_tool_args_for_schema(tool, normalized)


def _set_tool_surface_attr(target: BaseTool, name: str, value: Any) -> None:
    try:
        setattr(target, name, value)
    except Exception:
        try:
            object.__setattr__(target, name, value)
        except Exception:
            try:
                target.__dict__[name] = value
            except Exception:
                pass


def _get_tool_surface_attr(source: BaseTool, primary: str, fallback: str) -> Any:
    value = getattr(source, primary, None)
    if value not in (None, '', [], {}):
        return value
    return getattr(source, fallback, None)


def _copy_tool_surface_attrs(target: BaseTool, source: BaseTool, *, metadata_patch: dict[str, Any] | None = None) -> BaseTool:
    metadata = dict(_get_tool_surface_attr(source, 'metadata', 'tool_metadata') or {})
    if metadata_patch:
        metadata.update(metadata_patch)
    if metadata:
        _set_tool_surface_attr(target, 'metadata', metadata)
        _set_tool_surface_attr(target, 'tool_metadata', metadata)

    tags = list(_get_tool_surface_attr(source, 'tags', 'tool_tags') or [])
    if tags:
        _set_tool_surface_attr(target, 'tags', tags)
        _set_tool_surface_attr(target, 'tool_tags', tags)

    annotations = _get_tool_surface_attr(source, 'annotations', 'tool_annotations')
    if annotations is not None:
        _set_tool_surface_attr(target, 'annotations', annotations)
        _set_tool_surface_attr(target, 'tool_annotations', annotations)
    return target


def adapt_tool_for_chat_model(tool: BaseTool) -> BaseTool:
    description = _MODEL_VISIBLE_DESCRIPTION_OVERRIDES.get(tool.name, tool.description)

    async def _arun(**kwargs: Any) -> str:
        result = await tool.ainvoke(_prepare_tool_args(tool, tool.name, kwargs))
        return _stringify_tool_result(result)

    def _run(**kwargs: Any) -> str:
        result = tool.invoke(_prepare_tool_args(tool, tool.name, kwargs))
        return _stringify_tool_result(result)

    adapted = StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=tool.name,
        description=description,
        args_schema=tool.args_schema,
        return_direct=False,
    )
    return _copy_tool_surface_attrs(adapted, tool)


class _DatasetPathAliasArgs(BaseModel):
    dataset_path: str = Field(default='', description='数据集根目录或图片目录')
    path: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    img_dir: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    label_dir: str = Field(default='', description='可选标签目录')
    include_duplicates: bool = Field(default=False, description='是否包含重复图片检测')
    max_duplicate_groups: int = Field(default=5, description='最多返回多少个重复组样例')
    method: str = Field(default='md5', description='重复检测方法')
    report_path: str = Field(default='', description='可选报告输出路径')


class _PrepareAliasArgs(BaseModel):
    dataset_path: str = Field(default='', description='数据集根目录或图片目录')
    path: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    img_dir: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    label_dir: str = Field(default='', description='可选标签目录')
    force_split: bool = Field(default=False, description='是否按默认比例强制划分')


class _PredictAliasArgs(BaseModel):
    source_path: str = Field(default='', description='图片文件路径或图片目录路径')
    path: str = Field(default='', description='旧参数兼容；等价于 source_path')
    source: str = Field(default='', description='旧参数兼容；等价于 source_path')
    input_path: str = Field(default='', description='旧参数兼容；等价于 source_path')
    dir_path: str = Field(default='', description='旧参数兼容；等价于 source_path')
    folder: str = Field(default='', description='旧参数兼容；等价于 source_path')
    model: str = Field(default='', description='预测模型路径或模型名')
    conf: float = Field(default=0.25, description='置信度阈值')
    iou: float = Field(default=0.45, description='NMS IoU 阈值')
    output_dir: str = Field(default='', description='输出目录')
    save_annotated: bool = Field(default=True, description='是否保存标注图')
    save_labels: bool = Field(default=False, description='是否保存 YOLO 标签')
    save_original: bool = Field(default=False, description='是否复制原图')
    generate_report: bool = Field(default=True, description='是否生成 JSON 报告')
    max_images: int = Field(default=0, description='最多处理图片数，0 表示不限制')

class _PredictSummaryAliasArgs(BaseModel):
    report_path: str = Field(default='', description='预测报告 JSON 路径')
    path: str = Field(default='', description='旧参数兼容；等价于 report_path')
    report: str = Field(default='', description='旧参数兼容；等价于 report_path')
    json_report: str = Field(default='', description='旧参数兼容；等价于 report_path')
    file: str = Field(default='', description='旧参数兼容；等价于 report_path')
    output_dir: str = Field(default='', description='预测输出目录；若未显式给 report_path，则会尝试读取 output_dir/prediction_report.json')
    dir_path: str = Field(default='', description='旧参数兼容；等价于 output_dir')
    folder: str = Field(default='', description='旧参数兼容；等价于 output_dir')


class _PredictManagementAliasArgs(BaseModel):
    report_path: str = Field(default='', description='预测报告 JSON 路径')
    path: str = Field(default='', description='旧参数兼容；等价于 report_path')
    report: str = Field(default='', description='旧参数兼容；等价于 report_path')
    json_report: str = Field(default='', description='旧参数兼容；等价于 report_path')
    file: str = Field(default='', description='旧参数兼容；等价于 report_path')
    output_dir: str = Field(default='', description='预测输出目录')
    dir_path: str = Field(default='', description='旧参数兼容；等价于 output_dir')
    folder: str = Field(default='', description='旧参数兼容；等价于 output_dir')
    output: str = Field(default='', description='旧参数兼容；等价于 output_dir')
    export_path: str = Field(default='', description='报告导出路径')
    export_dir: str = Field(default='', description='路径清单导出目录')
    destination_dir: str = Field(default='', description='整理结果输出目录')
    out_dir: str = Field(default='', description='旧参数兼容；用于 export_path / export_dir / destination_dir')
    export_format: str = Field(default='markdown', description='报告导出格式')
    format: str = Field(default='', description='旧参数兼容；用于 export_format 或 artifact_preference')
    organize_by: str = Field(default='detected_only', description='整理方式：detected_only / by_class')
    mode: str = Field(default='', description='旧参数兼容；等价于 organize_by')
    include_empty: bool = Field(default=False, description='整理时是否保留无命中结果')
    artifact_preference: str = Field(default='auto', description='产物优先级：auto / annotated / original / source / annotated_video / video_dir')


class _RealtimePredictionAliasArgs(BaseModel):
    model: str = Field(default='', description='预测模型路径或模型名')
    camera_id: int = Field(default=0, description='摄像头 ID')
    screen_id: int = Field(default=1, description='屏幕 ID')
    rtsp_url: str = Field(default='', description='RTSP 地址')
    url: str = Field(default='', description='旧参数兼容；等价于 rtsp_url')
    source: str = Field(default='', description='旧参数兼容；等价于 rtsp_url 或 model')
    conf: float = Field(default=0.25, description='置信度阈值')
    iou: float = Field(default=0.45, description='NMS IoU 阈值')
    output_dir: str = Field(default='', description='输出目录')
    out_dir: str = Field(default='', description='旧参数兼容；等价于 output_dir')
    frame_interval_ms: int = Field(default=100, description='采样间隔，毫秒')
    max_frames: int = Field(default=0, description='最多处理多少帧，0 表示不限')
    timeout_ms: int = Field(default=5000, description='RTSP 测试超时，毫秒')
    session_id: str = Field(default='', description='实时预测会话 ID')


class _DataGovernanceAliasArgs(BaseModel):
    dataset_path: str = Field(default='', description='数据集根目录或图片目录')
    path: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    dataset: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    root: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    img_dir: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    label_dir: str = Field(default='', description='可选标签目录')
    output_dir: str = Field(default='', description='可选输出目录')
    out_dir: str = Field(default='', description='旧参数兼容；等价于 output_dir')
    target_format: str = Field(default='', description='目标格式')
    format: str = Field(default='', description='旧参数兼容；等价于 target_format 或 label_format')
    action: str = Field(default='', description='modify 动作')
    operation: str = Field(default='', description='旧参数兼容；等价于 action')
    old_value: str = Field(default='', description='旧类别值')
    new_value: str = Field(default='', description='新类别值')
    from_: str = Field(default='', alias='from', description='旧参数兼容；等价于 old_value')
    to: str = Field(default='', description='旧参数兼容；等价于 new_value')
    label_format: str = Field(default='', description='生成标签格式')
    classes_txt: str = Field(default='', description='classes.txt 路径')
    data_yaml: str = Field(default='', description='data.yaml 路径')
    backup: bool = Field(default=True, description='是否备份')
    dry_run: bool = Field(default=True, description='是否仅预览')
    only_missing: bool = Field(default=True, description='是否仅处理缺失标签')
    include_no_label: bool = Field(default=True, description='分类时是否包含无标签图片')


class _RemoteTransferAliasArgs(BaseModel):
    server: str = Field(default='', description='远端 profile 名、SSH alias，或 user@host 形式的目标')
    profile: str = Field(default='', description='显式 profile 名')
    remote_root: str = Field(default='', description='远端目标根目录')
    remote_dir: str = Field(default='', description='旧参数兼容；等价于 remote_root')
    remote_path: str = Field(default='', description='旧参数兼容；等价于 remote_root')
    local_paths: list[str] = Field(default=[], description='本地文件或目录路径列表')
    local_path: str = Field(default='', description='旧参数兼容；单个本地路径')
    paths_text: str = Field(default='', description='兼容字段；多个路径可用换行、逗号或分号分隔')
    path: str = Field(default='', description='旧参数兼容；等价于 paths_text')
    paths: str = Field(default='', description='旧参数兼容；等价于 paths_text')
    source: str = Field(default='', description='旧参数兼容；等价于 paths_text')
    host: str = Field(default='', description='显式主机名')
    username: str = Field(default='', description='显式用户名')
    user: str = Field(default='', description='旧参数兼容；等价于 username')
    port: int = Field(default=0, description='SSH 端口')
    recursive: bool = Field(default=True, description='目录上传时是否递归复制')
    create_remote_root: bool = Field(default=True, description='上传前是否自动创建远端目录')
    profiles_path: str = Field(default='', description='可选远端 profile 配置文件路径')
    resume: bool | None = Field(default=None, description='是否启用断点续传')
    resume_upload: bool | None = Field(default=None, description='旧参数兼容；等价于 resume')
    verify_hash: bool | None = Field(default=None, description='上传完成后是否做哈希校验')
    verify: bool | None = Field(default=None, description='旧参数兼容；等价于 verify_hash')
    hash_algorithm: str = Field(default='sha256', description='哈希算法：sha256 / md5')
    hash_algo: str = Field(default='', description='旧参数兼容；等价于 hash_algorithm')
    large_file_threshold_mb: int | None = Field(default=None, description='达到该体积后切换到大文件分块模式')
    threshold_mb: int | None = Field(default=None, description='旧参数兼容；等价于 large_file_threshold_mb')
    chunk_size_mb: int | None = Field(default=None, description='大文件分块大小，单位 MB')
    chunk_mb: int | None = Field(default=None, description='旧参数兼容；等价于 chunk_size_mb')
    show_progress: bool | None = Field(default=None, description='是否打印上传进度')
    progress: bool | None = Field(default=None, description='旧参数兼容；等价于 show_progress')

def _build_alias_tool(alias_name: str, target_tool: BaseTool, *, description: str, args_schema: type[BaseModel]) -> BaseTool:
    async def _arun(**kwargs: Any) -> str:
        result = await target_tool.ainvoke(_prepare_tool_args(target_tool, alias_name, kwargs))
        return _stringify_tool_result(result)

    def _run(**kwargs: Any) -> str:
        result = target_tool.invoke(_prepare_tool_args(target_tool, alias_name, kwargs))
        return _stringify_tool_result(result)

    alias_tool = StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=alias_name,
        description=description,
        args_schema=args_schema,
        return_direct=False,
    )
    return _copy_tool_surface_attrs(
        alias_tool,
        target_tool,
        metadata_patch={'canonical_tool_name': target_tool.name},
    )


def adapt_tools_for_chat_model(tools: list[BaseTool], *, include_aliases: bool = True) -> list[BaseTool]:
    adapted = [adapt_tool_for_chat_model(tool) for tool in tools]
    if not include_aliases:
        return adapted
    tool_map = {tool.name: tool for tool in tools}
    alias_tools: list[BaseTool] = []

    if 'detect_duplicate_images' in tool_map:
        alias_tools.append(
            _build_alias_tool(
                'detect_duplicates',
                tool_map['detect_duplicate_images'],
                description='兼容旧工具名 detect_duplicates。用于检测数据集中的重复图片；优先传 dataset_path，path 也可兼容。',
                args_schema=_DatasetPathAliasArgs,
            )
        )
    if 'run_dataset_health_check' in tool_map:
        alias_tools.append(
            _build_alias_tool(
                'detect_corrupted_images',
                tool_map['run_dataset_health_check'],
                description='兼容旧工具名 detect_corrupted_images。用于检查图片损坏、格式异常、尺寸异常；如需重复图片，也可设置 include_duplicates=true。',
                args_schema=_DatasetPathAliasArgs,
            )
        )
    if 'prepare_dataset_for_training' in tool_map:
        alias_tools.append(
            _build_alias_tool(
                'prepare_dataset',
                tool_map['prepare_dataset_for_training'],
                description='兼容旧工具名 prepare_dataset。用于把数据集准备到可训练状态。',
                args_schema=_PrepareAliasArgs,
            )
        )
        alias_tools.append(
            _build_alias_tool(
                'dataset_manager.prepare_dataset',
                tool_map['prepare_dataset_for_training'],
                description='兼容旧桌面风格工具名 dataset_manager.prepare_dataset。用于把数据集准备到可训练状态。',
                args_schema=_PrepareAliasArgs,
            )
        )
    if 'predict_images' in tool_map:
        for alias_name, description in (
            ('predict_directory', '兼容旧工具名 predict_directory。用于对图片目录做批量预测。'),
            ('batch_predict_images', '兼容旧工具名 batch_predict_images。用于对图片目录做批量预测。'),
            ('predict_images_in_dir', '兼容旧工具名 predict_images_in_dir。用于对图片目录做批量预测。'),
        ):
            alias_tools.append(
                _build_alias_tool(
                    alias_name,
                    tool_map['predict_images'],
                    description=description,
                    args_schema=_PredictAliasArgs,
                )
            )
    if 'predict_videos' in tool_map:
        for alias_name, description in (
            ('predict_video_directory', '兼容旧工具名 predict_video_directory。用于对视频目录做批量预测。'),
            ('batch_predict_videos', '兼容旧工具名 batch_predict_videos。用于对视频目录做批量预测。'),
            ('predict_videos_in_dir', '兼容旧工具名 predict_videos_in_dir。用于对视频目录做批量预测。'),
        ):
            alias_tools.append(
                _build_alias_tool(
                    alias_name,
                    tool_map['predict_videos'],
                    description=description,
                    args_schema=_PredictAliasArgs,
                )
            )
    if 'summarize_prediction_results' in tool_map:
        for alias_name, description in (
            ('summarize_predictions', '兼容旧工具名 summarize_predictions。用于汇总 prediction_report.json。'),
            ('summarize_prediction_report', '兼容旧工具名 summarize_prediction_report。用于汇总 prediction_report.json。'),
            ('analyze_prediction_report', '兼容旧工具名 analyze_prediction_report。用于汇总 prediction_report.json。'),
        ):
            alias_tools.append(
                _build_alias_tool(
                    alias_name,
                    tool_map['summarize_prediction_results'],
                    description=description,
                    args_schema=_PredictSummaryAliasArgs,
                )
            )
    for canonical_name, aliases in (
        ('inspect_prediction_outputs', (
            ('inspect_prediction_output', '兼容旧工具名 inspect_prediction_output。用于检查 prediction 输出目录和产物结构。'),
            ('show_prediction_outputs', '兼容旧工具名 show_prediction_outputs。用于检查 prediction 输出目录和产物结构。'),
            ('prediction_output_overview', '兼容旧工具名 prediction_output_overview。用于检查 prediction 输出目录和产物结构。'),
        )),
        ('export_prediction_report', (
            ('export_prediction_summary', '兼容旧工具名 export_prediction_summary。用于导出可读的 prediction 报告。'),
            ('write_prediction_report', '兼容旧工具名 write_prediction_report。用于导出可读的 prediction 报告。'),
        )),
        ('export_prediction_path_lists', (
            ('export_prediction_paths', '兼容旧工具名 export_prediction_paths。用于导出 prediction 命中/空结果路径清单。'),
        )),
        ('organize_prediction_results', (
            ('collect_prediction_hits', '兼容旧工具名 collect_prediction_hits。用于把命中 prediction 结果整理到新目录。'),
            ('group_prediction_results', '兼容旧工具名 group_prediction_results。用于按类别整理 prediction 结果。'),
        )),
    ):
        if canonical_name not in tool_map:
            continue
        for alias_name, description in aliases:
            alias_tools.append(
                _build_alias_tool(
                    alias_name,
                    tool_map[canonical_name],
                    description=description,
                    args_schema=_PredictManagementAliasArgs,
                )
            )
    for canonical_name, aliases in (
        ('scan_cameras', (('scan_available_cameras', '兼容旧工具名 scan_available_cameras。用于扫描可用摄像头。'),)),
        ('scan_screens', (('scan_available_screens', '兼容旧工具名 scan_available_screens。用于扫描可用屏幕。'),)),
        ('test_rtsp_stream', (('probe_rtsp_stream', '兼容旧工具名 probe_rtsp_stream。用于测试 RTSP 地址是否可用。'),)),
        ('start_camera_prediction', (('start_live_camera_prediction', '兼容旧工具名 start_live_camera_prediction。用于启动摄像头实时预测。'),)),
        ('start_rtsp_prediction', (('start_live_rtsp_prediction', '兼容旧工具名 start_live_rtsp_prediction。用于启动 RTSP 实时预测。'),)),
        ('start_screen_prediction', (('start_live_screen_prediction', '兼容旧工具名 start_live_screen_prediction。用于启动屏幕实时预测。'),)),
        ('check_realtime_prediction_status', (('check_live_prediction_status', '兼容旧工具名 check_live_prediction_status。用于查看实时预测状态。'),)),
        ('stop_realtime_prediction', (('stop_live_prediction', '兼容旧工具名 stop_live_prediction。用于停止实时预测。'),)),
    ):
        if canonical_name not in tool_map:
            continue
        for alias_name, description in aliases:
            alias_tools.append(
                _build_alias_tool(
                    alias_name,
                    tool_map[canonical_name],
                    description=description,
                    args_schema=_RealtimePredictionAliasArgs,
                )
            )
    for canonical_name, aliases in (
        ('preview_convert_format', (('preview_convert_labels', '兼容旧工具名 preview_convert_labels。用于预览标签格式转换范围。'),)),
        ('convert_format', (('convert_labels_format', '兼容旧工具名 convert_labels_format。用于执行标签格式转换。'),)),
        ('preview_modify_labels', (('preview_replace_labels', '兼容旧工具名 preview_replace_labels。用于预览标签批量替换/删除范围。'),)),
        ('modify_labels', (('replace_labels', '兼容旧工具名 replace_labels。用于执行标签批量替换/删除。'),)),
        ('generate_missing_labels', (('fill_missing_labels', '兼容旧工具名 fill_missing_labels。用于补齐缺失标签。'),)),
        ('generate_empty_labels', (('create_empty_labels', '兼容旧工具名 create_empty_labels。用于生成空标签。'),)),
        ('preview_categorize_by_class', (('preview_group_by_class', '兼容旧工具名 preview_group_by_class。用于预览按类别整理结果。'),)),
        ('categorize_by_class', (('group_by_class', '兼容旧工具名 group_by_class。用于按类别整理数据。'),)),
    ):
        if canonical_name not in tool_map:
            continue
        for alias_name, description in aliases:
            alias_tools.append(
                _build_alias_tool(
                    alias_name,
                    tool_map[canonical_name],
                    description=description,
                    args_schema=_DataGovernanceAliasArgs,
                )
            )

    for canonical_name, aliases in (
        ('list_remote_profiles', (
            ('list_remote_servers', '兼容旧工具名 list_remote_servers。用于列出当前可用的远端 profile/SSH alias。'),
            ('list_remote_targets', '兼容旧工具名 list_remote_targets。用于列出当前可用的远端 profile/SSH alias。'),
            ('show_remote_profiles', '兼容旧工具名 show_remote_profiles。用于列出当前可用的远端 profile/SSH alias。'),
        )),
        ('upload_assets_to_remote', (
            ('upload_to_server', '兼容旧工具名 upload_to_server。用于把本地文件或目录上传到远端服务器。'),
            ('upload_to_remote', '兼容旧工具名 upload_to_remote。用于把本地文件或目录上传到远端服务器。'),
            ('sync_local_to_remote', '兼容旧工具名 sync_local_to_remote。用于把本地文件或目录上传到远端服务器。'),
            ('scp_to_server', '兼容旧工具名 scp_to_server。用于把本地文件或目录上传到远端服务器。'),
        )),
        ('download_assets_from_remote', (
            ('download_from_server', '兼容旧工具名 download_from_server。用于把远端文件或目录下载回本机。'),
            ('download_from_remote', '兼容旧工具名 download_from_remote。用于把远端文件或目录下载回本机。'),
            ('pull_from_server', '兼容旧工具名 pull_from_server。用于把远端文件或目录下载回本机。'),
            ('sync_remote_to_local', '兼容旧工具名 sync_remote_to_local。用于把远端文件或目录下载回本机。'),
            ('scp_from_server', '兼容旧工具名 scp_from_server。用于把远端文件或目录下载回本机。'),
        )),
    ):
        if canonical_name not in tool_map:
            continue
        for alias_name, description in aliases:
            alias_tools.append(
                _build_alias_tool(
                    alias_name,
                    tool_map[canonical_name],
                    description=description,
                    args_schema=_RemoteTransferAliasArgs,
                )
            )

    return adapted + alias_tools

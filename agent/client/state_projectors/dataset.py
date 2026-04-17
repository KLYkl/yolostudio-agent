from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.state_projectors.common import (
    _dataset_duplicate_snapshot,
    _dataset_extract_preview_snapshot,
    _dataset_extract_result_snapshot,
    _dataset_frame_extract_snapshot,
    _dataset_health_snapshot,
    _dataset_scan_snapshot,
    _dataset_validation_snapshot,
    _dataset_video_scan_snapshot,
)


def apply_dataset_tool_result(
    session_state: SessionState,
    tool_name: str,
    result: dict[str, Any],
    tool_args: dict[str, Any],
) -> None:
    ds = session_state.active_dataset
    if tool_name == 'scan_dataset' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('img_dir', ds.img_dir))
        ds.label_dir = str(result.get('resolved_label_dir') or tool_args.get('label_dir', ds.label_dir))
        detected_yaml = result.get('detected_data_yaml') or ''
        if detected_yaml:
            ds.data_yaml = str(detected_yaml)
        ds.last_scan = _dataset_scan_snapshot(result, detected_yaml=detected_yaml)
    elif tool_name == 'validate_dataset' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('img_dir', ds.img_dir))
        ds.label_dir = str(result.get('resolved_label_dir') or tool_args.get('label_dir', ds.label_dir))
        ds.last_validate = _dataset_validation_snapshot(result)
    elif tool_name == 'run_dataset_health_check' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('dataset_path', ds.img_dir))
        ds.last_health_check = _dataset_health_snapshot(result)
    elif tool_name == 'detect_duplicate_images' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('dataset_path', ds.img_dir))
        ds.last_duplicate_check = _dataset_duplicate_snapshot(result)
    elif tool_name == 'preview_extract_images' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('source_path', ds.img_dir))
        ds.label_dir = str(result.get('resolved_label_dir') or ds.label_dir)
        ds.last_extract_preview = _dataset_extract_preview_snapshot(result, source_path=str(tool_args.get('source_path') or ''))
    elif tool_name == 'extract_images' and result.get('ok'):
        ds.last_extract_result = _dataset_extract_result_snapshot(result)
        if result.get('workflow_ready_path'):
            ds.dataset_root = str(result.get('workflow_ready_path'))
            ds.img_dir = str(result.get('output_img_dir') or ds.img_dir)
            ds.label_dir = str(result.get('output_label_dir') or '')
            ds.data_yaml = ''
    elif tool_name == 'scan_videos' and result.get('ok'):
        ds.last_video_scan = _dataset_video_scan_snapshot(result, source_path=str(tool_args.get('source_path') or ''))
    elif tool_name == 'extract_video_frames' and result.get('ok'):
        ds.last_frame_extract = _dataset_frame_extract_snapshot(result)
    elif tool_name == 'split_dataset' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('img_dir') or tool_args.get('img_dir', ds.img_dir))
        ds.label_dir = str(result.get('label_dir') or tool_args.get('label_dir', ds.label_dir))
        ds.last_split = {
            'train_path': result.get('train_path'),
            'val_path': result.get('val_path'),
            'train_count': result.get('train_count'),
            'val_count': result.get('val_count'),
            'output_dir': result.get('output_dir'),
            'suggested_yaml_path': result.get('suggested_yaml_path'),
        }
    elif tool_name == 'generate_yaml' and result.get('ok'):
        output_path = result.get('output_path') or ''
        if output_path:
            ds.data_yaml = str(output_path)
    elif tool_name in {'training_readiness', 'dataset_training_readiness'} and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or ds.img_dir)
        ds.label_dir = str(result.get('resolved_label_dir') or ds.label_dir)
        resolved_yaml = result.get('resolved_data_yaml') or ''
        if resolved_yaml:
            ds.data_yaml = str(resolved_yaml)
        else:
            ds.data_yaml = ''
        ds.last_readiness = {
            'readiness_scope': result.get('readiness_scope') or ('dataset' if tool_name == 'dataset_training_readiness' else 'execution'),
            'ready': result.get('ready'),
            'preparable': result.get('preparable'),
            'primary_blocker_type': result.get('primary_blocker_type'),
            'blocker_codes': result.get('blocker_codes'),
            'risk_level': result.get('risk_level'),
            'warnings': result.get('warnings'),
            'blockers': result.get('blockers'),
            'resolved_data_yaml': resolved_yaml,
            'needs_split': result.get('needs_split'),
            'needs_data_yaml': result.get('needs_data_yaml'),
            'dataset_structure': result.get('dataset_structure'),
            'next_step_summary': result.get('next_step_summary'),
            'summary': result.get('summary'),
        }
    elif tool_name == 'prepare_dataset_for_training' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('img_dir') or ds.img_dir)
        ds.label_dir = str(result.get('label_dir') or ds.label_dir)
        if result.get('data_yaml'):
            ds.data_yaml = str(result['data_yaml'])
        ds.last_readiness = {
            'ready': bool(result.get('ready', True)),
            'preparable': False,
            'primary_blocker_type': '',
            'risk_level': '',
            'warnings': [],
            'blockers': [],
            'resolved_data_yaml': ds.data_yaml,
            'summary': result.get('summary') or '数据准备完成：当前数据集已具备训练条件。',
        }
        for step in result.get('steps_completed', []):
            step_name = step.get('step')
            if step_name == 'scan' and step.get('ok'):
                ds.last_scan = _dataset_scan_snapshot(step, detected_yaml=str(step.get('detected_data_yaml') or ''))
            elif step_name == 'validate' and step.get('ok'):
                ds.last_validate = {
                    'issue_count': step.get('issue_count'),
                    'has_issues': step.get('has_issues'),
                }
            elif step_name == 'split' and step.get('ok'):
                ds.last_split = {
                    'train_path': step.get('train_path'),
                    'val_path': step.get('val_path'),
                    'train_count': step.get('train_count'),
                    'val_count': step.get('val_count'),
                    'output_dir': step.get('output_dir'),
                    'suggested_yaml_path': step.get('suggested_yaml_path'),
                }

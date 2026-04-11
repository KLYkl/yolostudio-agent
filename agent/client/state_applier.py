from __future__ import annotations

from typing import Any

from agent_plan.agent.client.session_state import SessionState


def apply_tool_result_to_state(
    session_state: SessionState,
    tool_name: str,
    result: dict[str, Any],
    tool_args: dict[str, Any] | None = None,
) -> None:
    ds = session_state.active_dataset
    tr = session_state.active_training
    pred = session_state.active_prediction

    tool_args = tool_args or {}
    if tool_name == 'scan_dataset' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('img_dir', ds.img_dir))
        ds.label_dir = str(result.get('resolved_label_dir') or tool_args.get('label_dir', ds.label_dir))
        detected_yaml = result.get('detected_data_yaml') or ''
        if detected_yaml:
            ds.data_yaml = str(detected_yaml)
        if summary := result.get('summary'):
            ds.last_scan = {
                'total_images': result.get('total_images'),
                'labeled_images': result.get('labeled_images'),
                'missing_labels': result.get('missing_labels'),
                'empty_labels': result.get('empty_labels'),
                'summary': summary,
                'detected_data_yaml': detected_yaml,
            }
    elif tool_name == 'validate_dataset' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('img_dir', ds.img_dir))
        ds.label_dir = str(result.get('resolved_label_dir') or tool_args.get('label_dir', ds.label_dir))
        ds.last_validate = {
            'issue_count': result.get('issue_count'),
            'has_issues': result.get('has_issues'),
        }
    elif tool_name == 'run_dataset_health_check' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('dataset_path', ds.img_dir))
        ds.last_health_check = {
            'risk_level': result.get('risk_level'),
            'issue_count': result.get('issue_count'),
            'duplicate_groups': result.get('duplicate_groups'),
            'summary': result.get('summary'),
        }
    elif tool_name == 'detect_duplicate_images' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('dataset_path', ds.img_dir))
        ds.last_duplicate_check = {
            'method': result.get('method'),
            'duplicate_groups': result.get('duplicate_groups'),
            'duplicate_extra_files': result.get('duplicate_extra_files'),
            'summary': result.get('summary'),
        }
    elif tool_name == 'preview_extract_images' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('source_path', ds.img_dir))
        ds.label_dir = str(result.get('resolved_label_dir') or ds.label_dir)
        ds.last_extract_preview = {
            'available_images': result.get('available_images'),
            'planned_extract_count': result.get('planned_extract_count'),
            'output_dir': result.get('output_dir'),
            'workflow_ready_path': result.get('workflow_ready_path'),
            'summary': result.get('summary'),
        }
    elif tool_name == 'extract_images' and result.get('ok'):
        ds.last_extract_result = {
            'extracted': result.get('extracted'),
            'labels_copied': result.get('labels_copied'),
            'output_dir': result.get('output_dir'),
            'workflow_ready_path': result.get('workflow_ready_path'),
            'summary': result.get('summary'),
        }
        if result.get('workflow_ready_path'):
            ds.dataset_root = str(result.get('workflow_ready_path'))
            ds.img_dir = str(result.get('output_img_dir') or ds.img_dir)
            ds.label_dir = str(result.get('output_label_dir') or '')
            ds.data_yaml = ''
    elif tool_name == 'scan_videos' and result.get('ok'):
        ds.last_video_scan = {
            'total_videos': result.get('total_videos'),
            'source_path': result.get('source_path'),
            'summary': result.get('summary'),
        }
    elif tool_name == 'extract_video_frames' and result.get('ok'):
        ds.last_frame_extract = {
            'source_path': result.get('source_path'),
            'output_dir': result.get('output_dir'),
            'final_count': result.get('final_count'),
            'summary': result.get('summary'),
        }
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
    elif tool_name == 'training_readiness' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('resolved_img_dir') or ds.img_dir)
        ds.label_dir = str(result.get('resolved_label_dir') or ds.label_dir)
        resolved_yaml = result.get('resolved_data_yaml') or ''
        if resolved_yaml:
            ds.data_yaml = str(resolved_yaml)
    elif tool_name == 'prepare_dataset_for_training' and result.get('ok'):
        ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
        ds.img_dir = str(result.get('img_dir') or ds.img_dir)
        ds.label_dir = str(result.get('label_dir') or ds.label_dir)
        if result.get('data_yaml'):
            ds.data_yaml = str(result['data_yaml'])
        for step in result.get('steps_completed', []):
            step_name = step.get('step')
            if step_name == 'scan' and step.get('ok'):
                ds.last_scan = {
                    'total_images': step.get('total_images'),
                    'labeled_images': step.get('labeled_images'),
                    'missing_labels': step.get('missing_labels'),
                    'empty_labels': step.get('empty_labels'),
                    'summary': step.get('summary'),
                    'detected_data_yaml': step.get('detected_data_yaml', ''),
                }
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
    elif tool_name == 'start_training' and result.get('ok'):
        tr.running = True
        resolved_args = result.get('resolved_args') or {}
        tr.model = str(resolved_args.get('model') or tool_args.get('model', tr.model))
        tr.data_yaml = str(resolved_args.get('data_yaml') or tool_args.get('data_yaml', tr.data_yaml))
        if tr.data_yaml:
            ds.data_yaml = tr.data_yaml
        tr.device = result.get('device', '')
        tr.pid = result.get('pid')
        tr.log_file = result.get('log_file', '')
        tr.started_at = result.get('started_at')
        tr.last_start_result = result
    elif tool_name == 'check_training_status':
        tr.last_status = result
        is_running = bool(result.get('running'))
        tr.running = is_running
        if is_running:
            tr.device = result.get('device', tr.device)
            tr.pid = result.get('pid', tr.pid)
            tr.log_file = result.get('log_file', tr.log_file)
            tr.started_at = result.get('started_at', tr.started_at)
            command = result.get('command') or []
            for part in command:
                if isinstance(part, str) and part.startswith('model='):
                    tr.model = part.split('=', 1)[1]
                if isinstance(part, str) and part.startswith('data='):
                    tr.data_yaml = part.split('=', 1)[1]
        else:
            tr.pid = None
            tr.log_file = ''
            tr.started_at = None
    elif tool_name == 'stop_training' and result.get('ok'):
        tr.running = False
        tr.pid = None
        tr.log_file = ''
        tr.started_at = None
        tr.last_status = result
    elif tool_name == 'predict_images' and result.get('ok'):
        pred.source_path = str(result.get('source_path') or tool_args.get('source_path', pred.source_path))
        pred.model = str(result.get('model') or tool_args.get('model', pred.model))
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_result = {
            'summary': result.get('summary'),
            'processed_images': result.get('processed_images'),
            'detected_images': result.get('detected_images'),
            'empty_images': result.get('empty_images'),
            'class_counts': result.get('class_counts'),
            'warnings': result.get('warnings'),
            'detected_samples': result.get('detected_samples'),
            'empty_samples': result.get('empty_samples'),
            'output_dir': result.get('output_dir'),
            'annotated_dir': result.get('annotated_dir'),
            'report_path': result.get('report_path'),
            'model': result.get('model'),
        }
    elif tool_name == 'predict_videos' and result.get('ok'):
        pred.source_path = str(result.get('source_path') or tool_args.get('source_path', pred.source_path))
        pred.model = str(result.get('model') or tool_args.get('model', pred.model))
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_result = {
            'summary': result.get('summary'),
            'processed_videos': result.get('processed_videos'),
            'total_frames': result.get('total_frames'),
            'detected_frames': result.get('detected_frames'),
            'total_detections': result.get('total_detections'),
            'class_counts': result.get('class_counts'),
            'warnings': result.get('warnings'),
            'detected_samples': result.get('detected_samples'),
            'empty_samples': result.get('empty_samples'),
            'output_dir': result.get('output_dir'),
            'report_path': result.get('report_path'),
            'model': result.get('model'),
            'mode': 'videos',
        }
    elif tool_name == 'summarize_prediction_results' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        if result.get('model'):
            pred.model = str(result.get('model'))
        if result.get('source_path'):
            pred.source_path = str(result.get('source_path'))
        pred.last_result = {
            'summary': result.get('summary'),
            'processed_images': result.get('processed_images'),
            'detected_images': result.get('detected_images'),
            'empty_images': result.get('empty_images'),
            'class_counts': result.get('class_counts'),
            'warnings': result.get('warnings'),
            'detected_samples': result.get('detected_samples'),
            'empty_samples': result.get('empty_samples'),
            'output_dir': result.get('output_dir'),
            'annotated_dir': result.get('annotated_dir'),
            'report_path': result.get('report_path'),
            'model': result.get('model'),
            'total_detections': result.get('total_detections'),
            'processed_videos': result.get('processed_videos'),
            'total_frames': result.get('total_frames'),
            'detected_frames': result.get('detected_frames'),
            'mode': result.get('mode') or 'images',
        }

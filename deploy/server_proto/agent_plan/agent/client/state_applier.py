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
    kn = session_state.active_knowledge

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
        ds.last_readiness = {
            'ready': result.get('ready'),
            'preparable': result.get('preparable'),
            'primary_blocker_type': result.get('primary_blocker_type'),
            'risk_level': result.get('risk_level'),
            'warnings': result.get('warnings'),
            'blockers': result.get('blockers'),
            'resolved_data_yaml': resolved_yaml,
            'summary': result.get('summary'),
        }
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
        training_environment = result.get('training_environment') or {}
        tr.training_environment = str(
            resolved_args.get('training_environment')
            or training_environment.get('display_name')
            or training_environment.get('name')
            or tr.training_environment
            or ''
        )
        tr.project = str(resolved_args.get('project') or '')
        tr.run_name = str(resolved_args.get('name') or '')
        tr.batch = resolved_args.get('batch')
        tr.imgsz = resolved_args.get('imgsz')
        tr.fraction = resolved_args.get('fraction')
        tr.classes = list(resolved_args.get('classes') or [])
        tr.single_cls = resolved_args.get('single_cls')
        tr.optimizer = str(resolved_args.get('optimizer') or '')
        tr.freeze = resolved_args.get('freeze')
        tr.resume = resolved_args.get('resume')
        tr.lr0 = resolved_args.get('lr0')
        tr.patience = resolved_args.get('patience')
        tr.workers = resolved_args.get('workers')
        tr.amp = resolved_args.get('amp')
        tr.pid = result.get('pid')
        tr.log_file = result.get('log_file', '')
        tr.started_at = result.get('started_at')
        tr.last_start_result = result
        tr.last_summary = {}
        tr.training_run_summary = {}
    elif tool_name == 'list_training_environments' and result.get('ok'):
        tr.last_environment_probe = result
        default_environment = result.get('default_environment') or {}
        if default_environment:
            tr.training_environment = str(default_environment.get('display_name') or default_environment.get('name') or tr.training_environment)
    elif tool_name == 'training_preflight' and result.get('ok'):
        tr.last_preflight = result
        resolved_args = result.get('resolved_args') or {}
        training_environment = result.get('training_environment') or {}
        tr.training_environment = str(
            resolved_args.get('training_environment')
            or training_environment.get('display_name')
            or training_environment.get('name')
            or tr.training_environment
            or ''
        )
        if resolved_args.get('model'):
            tr.model = str(resolved_args.get('model'))
        if resolved_args.get('data_yaml'):
            tr.data_yaml = str(resolved_args.get('data_yaml'))
            ds.data_yaml = tr.data_yaml
        if resolved_args.get('project'):
            tr.project = str(resolved_args.get('project'))
        if resolved_args.get('name'):
            tr.run_name = str(resolved_args.get('name'))
        if resolved_args.get('batch') is not None:
            tr.batch = resolved_args.get('batch')
        if resolved_args.get('imgsz') is not None:
            tr.imgsz = resolved_args.get('imgsz')
        if resolved_args.get('fraction') is not None:
            tr.fraction = resolved_args.get('fraction')
        if resolved_args.get('classes') is not None:
            tr.classes = list(resolved_args.get('classes') or [])
        if resolved_args.get('single_cls') is not None:
            tr.single_cls = resolved_args.get('single_cls')
        if resolved_args.get('optimizer'):
            tr.optimizer = str(resolved_args.get('optimizer'))
        if resolved_args.get('freeze') is not None:
            tr.freeze = resolved_args.get('freeze')
        if resolved_args.get('resume') is not None:
            tr.resume = resolved_args.get('resume')
        if resolved_args.get('lr0') is not None:
            tr.lr0 = resolved_args.get('lr0')
        if resolved_args.get('patience') is not None:
            tr.patience = resolved_args.get('patience')
        if resolved_args.get('workers') is not None:
            tr.workers = resolved_args.get('workers')
        if resolved_args.get('amp') is not None:
            tr.amp = resolved_args.get('amp')
    elif tool_name == 'list_training_runs' and result.get('ok'):
        tr.recent_runs = list(result.get('runs') or [])
    elif tool_name == 'inspect_training_run' and result.get('ok'):
        tr.last_run_inspection = result
        if result.get('selected_run_id'):
            matched = next(
                (
                    item for item in tr.recent_runs
                    if str(item.get('run_id') or '') == str(result.get('selected_run_id') or '')
                ),
                None,
            )
            if matched is None:
                tr.recent_runs = [result, *tr.recent_runs[:9]]
    elif tool_name == 'check_training_status':
        tr.last_status = result
        is_running = bool(result.get('running'))
        tr.running = is_running
        resolved_args = result.get('resolved_args') or {}
        training_environment = result.get('training_environment') or {}
        tr.training_environment = str(
            resolved_args.get('training_environment')
            or training_environment.get('display_name')
            or training_environment.get('name')
            or tr.training_environment
            or ''
        )
        if resolved_args.get('model'):
            tr.model = str(resolved_args.get('model'))
        if resolved_args.get('data_yaml'):
            tr.data_yaml = str(resolved_args.get('data_yaml'))
            ds.data_yaml = tr.data_yaml
        if resolved_args.get('project'):
            tr.project = str(resolved_args.get('project'))
        if resolved_args.get('name'):
            tr.run_name = str(resolved_args.get('name'))
        if resolved_args.get('batch') is not None:
            tr.batch = resolved_args.get('batch')
        if resolved_args.get('imgsz') is not None:
            tr.imgsz = resolved_args.get('imgsz')
        if resolved_args.get('fraction') is not None:
            tr.fraction = resolved_args.get('fraction')
        if resolved_args.get('classes') is not None:
            tr.classes = list(resolved_args.get('classes') or [])
        if resolved_args.get('single_cls') is not None:
            tr.single_cls = resolved_args.get('single_cls')
        if resolved_args.get('optimizer'):
            tr.optimizer = str(resolved_args.get('optimizer'))
        if resolved_args.get('freeze') is not None:
            tr.freeze = resolved_args.get('freeze')
        if resolved_args.get('resume') is not None:
            tr.resume = resolved_args.get('resume')
        if resolved_args.get('lr0') is not None:
            tr.lr0 = resolved_args.get('lr0')
        if resolved_args.get('patience') is not None:
            tr.patience = resolved_args.get('patience')
        if resolved_args.get('workers') is not None:
            tr.workers = resolved_args.get('workers')
        if resolved_args.get('amp') is not None:
            tr.amp = resolved_args.get('amp')
        tr.device = str(result.get('device') or tr.device)
        tr.log_file = str(result.get('log_file') or tr.log_file)
        tr.started_at = result.get('started_at', tr.started_at)
        command = result.get('command') or []
        for part in command:
            if isinstance(part, str) and part.startswith('model='):
                tr.model = part.split('=', 1)[1]
            if isinstance(part, str) and part.startswith('data='):
                tr.data_yaml = part.split('=', 1)[1]
                ds.data_yaml = tr.data_yaml
            if isinstance(part, str) and part.startswith('batch='):
                try:
                    tr.batch = int(part.split('=', 1)[1])
                except ValueError:
                    pass
            if isinstance(part, str) and part.startswith('imgsz='):
                try:
                    tr.imgsz = int(part.split('=', 1)[1])
                except ValueError:
                    pass
            if isinstance(part, str) and part.startswith('project='):
                tr.project = part.split('=', 1)[1]
            if isinstance(part, str) and part.startswith('name='):
                tr.run_name = part.split('=', 1)[1]
            if isinstance(part, str) and part.startswith('fraction='):
                try:
                    tr.fraction = float(part.split('=', 1)[1])
                except ValueError:
                    pass
            if isinstance(part, str) and part.startswith('classes='):
                raw = part.split('=', 1)[1].strip()
                values = [item.strip() for item in raw.split(',') if item.strip()]
                if values and all(item.isdigit() for item in values):
                    tr.classes = [int(item) for item in values]
            if isinstance(part, str) and part.startswith('single_cls='):
                tr.single_cls = part.split('=', 1)[1].strip().lower() == 'true'
            if isinstance(part, str) and part.startswith('optimizer='):
                tr.optimizer = part.split('=', 1)[1]
            if isinstance(part, str) and part.startswith('freeze='):
                try:
                    tr.freeze = int(part.split('=', 1)[1])
                except ValueError:
                    pass
            if isinstance(part, str) and part.startswith('resume='):
                tr.resume = part.split('=', 1)[1].strip().lower() == 'true'
            if isinstance(part, str) and part.startswith('lr0='):
                try:
                    tr.lr0 = float(part.split('=', 1)[1])
                except ValueError:
                    pass
            if isinstance(part, str) and part.startswith('patience='):
                try:
                    tr.patience = int(part.split('=', 1)[1])
                except ValueError:
                    pass
            if isinstance(part, str) and part.startswith('workers='):
                try:
                    tr.workers = int(part.split('=', 1)[1])
                except ValueError:
                    pass
            if isinstance(part, str) and part.startswith('amp='):
                tr.amp = part.split('=', 1)[1].strip().lower() == 'true'
        tr.pid = result.get('pid', tr.pid) if is_running else None
    elif tool_name == 'summarize_training_run' and result.get('ok'):
        tr.last_summary = result
        tr.training_run_summary = result
        tr.running = str(result.get('run_state') or '').strip().lower() == 'running'
        if result.get('log_file'):
            tr.log_file = str(result.get('log_file'))
        if result.get('latest_metrics'):
            tr.last_status = {
                **tr.last_status,
                'run_state': result.get('run_state'),
                'progress': result.get('progress'),
                'latest_metrics': result.get('latest_metrics'),
                'analysis_ready': result.get('analysis_ready'),
                'minimum_facts_ready': result.get('minimum_facts_ready'),
                'signals': result.get('signals'),
                'facts': result.get('facts'),
            }
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
    elif tool_name == 'retrieve_training_knowledge' and result.get('ok'):
        kn.last_retrieval = {
            'topic': result.get('topic'),
            'stage': result.get('stage'),
            'model_family': result.get('model_family'),
            'matched_rule_ids': result.get('matched_rule_ids'),
            'summary': result.get('summary'),
        }
    elif tool_name == 'analyze_training_outcome' and result.get('ok'):
        kn.last_analysis = {
            'summary': result.get('summary'),
            'assessment': result.get('assessment'),
            'matched_rule_ids': result.get('matched_rule_ids'),
            'signals': result.get('signals'),
        }
    elif tool_name == 'recommend_next_training_step' and result.get('ok'):
        kn.last_recommendation = {
            'summary': result.get('summary'),
            'recommended_action': result.get('recommended_action'),
            'matched_rule_ids': result.get('matched_rule_ids'),
            'signals': result.get('signals'),
        }

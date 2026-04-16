from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.session_state import SessionState


def _pick_mapping(source: dict[str, Any] | None, *keys: str) -> dict[str, Any]:
    source = source or {}
    return {key: source.get(key) for key in keys if key in source}


def _training_environment_snapshot(environment: dict[str, Any] | None) -> dict[str, Any]:
    return _pick_mapping(
        environment,
        'name',
        'display_name',
        'kind',
        'source',
        'python_executable',
        'yolo_executable',
        'gpu_available',
        'torch_cuda_available',
        'gpu_count',
        'availability_reason',
    )


def _training_environment_list_snapshot(environments: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [_training_environment_snapshot(item) for item in (environments or []) if isinstance(item, dict)]


def _resolved_training_args_snapshot(resolved_args: dict[str, Any] | None) -> dict[str, Any]:
    resolved_args = resolved_args or {}
    snapshot = _pick_mapping(
        resolved_args,
        'model',
        'data_yaml',
        'epochs',
        'device',
        'training_environment',
        'project',
        'name',
        'batch',
        'imgsz',
        'fraction',
        'classes',
        'single_cls',
        'optimizer',
        'freeze',
        'resume',
        'lr0',
        'patience',
        'workers',
        'amp',
    )
    if 'classes' in snapshot:
        snapshot['classes'] = list(snapshot.get('classes') or [])
    return snapshot


def _apply_resolved_training_args(
    tr: Any,
    ds: Any,
    *,
    resolved_args: dict[str, Any] | None,
    training_environment: dict[str, Any] | None = None,
    tool_args: dict[str, Any] | None = None,
    prefer_tool_args_for_start: bool = False,
    force_assign: bool = False,
) -> None:
    resolved_args = resolved_args or {}
    tool_args = tool_args or {}
    training_environment = training_environment or {}

    if prefer_tool_args_for_start:
        model_value = resolved_args.get('model') or tool_args.get('model', tr.model)
        data_yaml_value = resolved_args.get('data_yaml') or tool_args.get('data_yaml', tr.data_yaml)
    else:
        model_value = resolved_args.get('model') or tr.model
        data_yaml_value = resolved_args.get('data_yaml') or tr.data_yaml

    tr.model = str(model_value or '')
    tr.data_yaml = str(data_yaml_value or '')
    if tr.data_yaml:
        ds.data_yaml = tr.data_yaml

    tr.training_environment = str(
        resolved_args.get('training_environment')
        or training_environment.get('display_name')
        or training_environment.get('name')
        or tr.training_environment
        or ''
    )
    if force_assign:
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
        return
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


def _apply_training_command_overrides(tr: Any, ds: Any, command: list[Any] | None) -> None:
    for part in command or []:
        if not isinstance(part, str):
            continue
        if part.startswith('model='):
            tr.model = part.split('=', 1)[1]
        if part.startswith('data='):
            tr.data_yaml = part.split('=', 1)[1]
            ds.data_yaml = tr.data_yaml
        if part.startswith('batch='):
            try:
                tr.batch = int(part.split('=', 1)[1])
            except ValueError:
                pass
        if part.startswith('imgsz='):
            try:
                tr.imgsz = int(part.split('=', 1)[1])
            except ValueError:
                pass
        if part.startswith('project='):
            tr.project = part.split('=', 1)[1]
        if part.startswith('name='):
            tr.run_name = part.split('=', 1)[1]
        if part.startswith('fraction='):
            try:
                tr.fraction = float(part.split('=', 1)[1])
            except ValueError:
                pass
        if part.startswith('classes='):
            raw = part.split('=', 1)[1].strip()
            values = [item.strip() for item in raw.split(',') if item.strip()]
            if values and all(item.isdigit() for item in values):
                tr.classes = [int(item) for item in values]
        if part.startswith('single_cls='):
            tr.single_cls = part.split('=', 1)[1].strip().lower() == 'true'
        if part.startswith('optimizer='):
            tr.optimizer = part.split('=', 1)[1]
        if part.startswith('freeze='):
            try:
                tr.freeze = int(part.split('=', 1)[1])
            except ValueError:
                pass
        if part.startswith('resume='):
            tr.resume = part.split('=', 1)[1].strip().lower() == 'true'
        if part.startswith('lr0='):
            try:
                tr.lr0 = float(part.split('=', 1)[1])
            except ValueError:
                pass
        if part.startswith('patience='):
            try:
                tr.patience = int(part.split('=', 1)[1])
            except ValueError:
                pass
        if part.startswith('workers='):
            try:
                tr.workers = int(part.split('=', 1)[1])
            except ValueError:
                pass
        if part.startswith('amp='):
            tr.amp = part.split('=', 1)[1].strip().lower() == 'true'


def _training_environment_probe_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    snapshot = _pick_mapping(result, 'ok', 'summary', 'default_profile', 'profile_count')
    snapshot['default_environment'] = _training_environment_snapshot(result.get('default_environment') or {})
    snapshot['environments'] = _training_environment_list_snapshot(result.get('environments') or [])
    return snapshot


def _training_preflight_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    snapshot = _pick_mapping(
        result,
        'ok',
        'summary',
        'ready_to_start',
        'resolved_device',
        'available_gpu_indexes',
        'error',
        'warnings',
        'blockers',
    )
    snapshot['resolved_args'] = _resolved_training_args_snapshot(result.get('resolved_args') or {})
    snapshot['training_environment'] = _training_environment_snapshot(result.get('training_environment') or {})
    return snapshot


def _training_start_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    snapshot = _pick_mapping(result, 'ok', 'summary', 'pid', 'device', 'log_file', 'started_at', 'error')
    snapshot['resolved_args'] = _resolved_training_args_snapshot(result.get('resolved_args') or {})
    snapshot['training_environment'] = _training_environment_snapshot(result.get('training_environment') or {})
    return snapshot


def _training_status_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    snapshot = _pick_mapping(
        result,
        'ok',
        'summary',
        'running',
        'run_state',
        'device',
        'pid',
        'log_file',
        'started_at',
        'progress',
        'latest_metrics',
        'analysis_ready',
        'minimum_facts_ready',
        'signals',
        'facts',
        'error',
    )
    snapshot['resolved_args'] = _resolved_training_args_snapshot(result.get('resolved_args') or {})
    snapshot['training_environment'] = _training_environment_snapshot(result.get('training_environment') or {})
    return snapshot


def _training_run_summary_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    return _pick_mapping(
        result,
        'ok',
        'summary',
        'run_state',
        'observation_stage',
        'log_file',
        'progress',
        'latest_metrics',
        'analysis_ready',
        'minimum_facts_ready',
        'signals',
        'facts',
    )


def _training_loop_status_snapshot(result: dict[str, Any], *, include_detail: bool = False) -> dict[str, Any]:
    keys = [
        'ok',
        'summary',
        'loop_id',
        'loop_name',
        'status',
        'managed_level',
        'max_rounds',
        'current_round_index',
        'completed_rounds',
        'recorded_rounds',
        'selected_metric',
        'knowledge_gate_status',
        'final_summary',
        'latest_round_card',
        'latest_round_review',
        'latest_round_memory',
        'latest_planner_output',
    ]
    if include_detail:
        keys.extend(['rounds', 'experience_timeline'])
    return _pick_mapping(result, *keys)


def _prediction_result_snapshot(
    result: dict[str, Any],
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    snapshot = _pick_mapping(
        result,
        'summary',
        'processed_images',
        'detected_images',
        'empty_images',
        'class_counts',
        'warnings',
        'detected_samples',
        'empty_samples',
        'output_dir',
        'annotated_dir',
        'report_path',
        'model',
        'total_detections',
        'processed_videos',
        'total_frames',
        'detected_frames',
        'prediction_overview',
        'prediction_summary_overview',
        'action_candidates',
    )
    resolved_mode = mode or result.get('mode')
    if resolved_mode:
        snapshot['mode'] = resolved_mode
    return snapshot


def _prediction_management_snapshot(
    result: dict[str, Any],
    *,
    overview_key: str,
    fallback_overview_key: str | None = None,
    extra_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    snapshot = _pick_mapping(result, 'summary', 'mode', 'action_candidates', *extra_keys)
    overview = result.get(overview_key)
    if overview is None and fallback_overview_key:
        overview = result.get(fallback_overview_key)
    snapshot[overview_key] = overview
    return snapshot


def _dataset_scan_snapshot(result: dict[str, Any], *, detected_yaml: str = '') -> dict[str, Any]:
    snapshot = _pick_mapping(
        result,
        'total_images',
        'labeled_images',
        'missing_labels',
        'empty_labels',
        'summary',
        'scan_overview',
        'action_candidates',
    )
    snapshot['detected_data_yaml'] = detected_yaml
    return snapshot


def _dataset_validation_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    return _pick_mapping(
        result,
        'issue_count',
        'has_issues',
        'summary',
        'validation_overview',
        'action_candidates',
    )


def _dataset_health_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    return _pick_mapping(
        result,
        'risk_level',
        'issue_count',
        'duplicate_groups',
        'summary',
        'health_overview',
        'action_candidates',
    )


def _dataset_duplicate_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    return _pick_mapping(
        result,
        'method',
        'duplicate_groups',
        'duplicate_extra_files',
        'summary',
        'duplicate_overview',
        'action_candidates',
    )


def _dataset_extract_preview_snapshot(result: dict[str, Any], *, source_path: str = '') -> dict[str, Any]:
    snapshot = _pick_mapping(
        result,
        'source_path',
        'available_images',
        'planned_extract_count',
        'output_dir',
        'workflow_ready_path',
        'summary',
        'extract_preview_overview',
        'action_candidates',
    )
    if source_path and not str(snapshot.get('source_path') or '').strip():
        snapshot['source_path'] = source_path
    return snapshot


def _dataset_extract_result_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    return _pick_mapping(
        result,
        'extracted',
        'labels_copied',
        'output_dir',
        'workflow_ready_path',
        'summary',
        'extract_overview',
        'action_candidates',
    )


def _dataset_video_scan_snapshot(result: dict[str, Any], *, source_path: str = '') -> dict[str, Any]:
    snapshot = _pick_mapping(
        result,
        'total_videos',
        'source_path',
        'summary',
        'video_scan_overview',
        'action_candidates',
    )
    if source_path and not str(snapshot.get('source_path') or '').strip():
        snapshot['source_path'] = source_path
    return snapshot


def _dataset_frame_extract_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    return _pick_mapping(
        result,
        'source_path',
        'output_dir',
        'final_count',
        'summary',
        'frame_extract_overview',
        'action_candidates',
    )


def _realtime_snapshot(
    result: dict[str, Any],
    *,
    overview_key: str,
    extra_keys: tuple[str, ...] = (),
    fallback_overview_key: str | None = None,
) -> dict[str, Any]:
    snapshot = _pick_mapping(result, 'summary', 'action_candidates', *extra_keys)
    overview = result.get(overview_key)
    if overview is None and fallback_overview_key:
        overview = result.get(fallback_overview_key)
    snapshot[overview_key] = overview
    return snapshot


def _knowledge_snapshot(
    result: dict[str, Any],
    *,
    overview_key: str,
    extra_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    return _pick_mapping(result, 'summary', 'action_candidates', overview_key, *extra_keys)


def _remote_transfer_snapshot(
    result: dict[str, Any],
    *,
    overview_key: str,
    extra_keys: tuple[str, ...] = (),
    fallback_overview_key: str | None = None,
) -> dict[str, Any]:
    snapshot = _pick_mapping(result, 'summary', 'action_candidates', *extra_keys)
    overview = result.get(overview_key)
    if overview is None and fallback_overview_key:
        overview = result.get(fallback_overview_key)
    snapshot[overview_key] = overview
    return snapshot


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
    rt = session_state.active_remote_transfer

    tool_args = tool_args or {}
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
        tr.device = result.get('device', '')
        _apply_resolved_training_args(
            tr,
            ds,
            resolved_args=resolved_args,
            training_environment=result.get('training_environment') or {},
            tool_args=tool_args,
            prefer_tool_args_for_start=True,
            force_assign=True,
        )
        tr.pid = result.get('pid')
        tr.log_file = result.get('log_file', '')
        tr.started_at = result.get('started_at')
        tr.last_start_result = _training_start_snapshot(result)
        tr.last_summary = {}
        tr.training_run_summary = {}
    elif tool_name == 'list_training_environments' and result.get('ok'):
        tr.last_environment_probe = _training_environment_probe_snapshot(result)
        default_environment = tr.last_environment_probe.get('default_environment') or {}
        if default_environment:
            tr.training_environment = str(default_environment.get('display_name') or default_environment.get('name') or tr.training_environment)
    elif tool_name == 'training_preflight' and result.get('ok'):
        tr.last_preflight = _training_preflight_snapshot(result)
        _apply_resolved_training_args(
            tr,
            ds,
            resolved_args=result.get('resolved_args') or {},
            training_environment=result.get('training_environment') or {},
        )
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
    elif tool_name == 'compare_training_runs' and result.get('ok'):
        tr.last_run_comparison = result
    elif tool_name == 'select_best_training_run' and result.get('ok'):
        tr.best_run_selection = result
    elif tool_name == 'start_training_loop' and result.get('ok'):
        tr.active_loop_id = str(result.get('loop_id') or tr.active_loop_id)
        tr.active_loop_name = str(result.get('loop_name') or tr.active_loop_name)
        tr.active_loop_status = str(result.get('status') or tr.active_loop_status)
        tr.last_loop_status = _training_loop_status_snapshot(result)
        tr.last_loop_detail = {}
    elif tool_name == 'list_training_loops' and result.get('ok'):
        tr.recent_loops = list(result.get('loops') or [])
        if result.get('active_loop_id'):
            tr.active_loop_id = str(result.get('active_loop_id') or tr.active_loop_id)
    elif tool_name == 'check_training_loop_status' and result.get('ok'):
        tr.active_loop_id = str(result.get('loop_id') or tr.active_loop_id)
        tr.active_loop_name = str(result.get('loop_name') or tr.active_loop_name)
        tr.active_loop_status = str(result.get('status') or tr.active_loop_status)
        tr.last_loop_status = _training_loop_status_snapshot(result)
    elif tool_name == 'inspect_training_loop' and result.get('ok'):
        tr.active_loop_id = str(result.get('loop_id') or tr.active_loop_id)
        tr.active_loop_name = str(result.get('loop_name') or tr.active_loop_name)
        tr.active_loop_status = str(result.get('status') or tr.active_loop_status)
        tr.last_loop_status = _training_loop_status_snapshot(result)
        tr.last_loop_detail = _training_loop_status_snapshot(result, include_detail=True)
    elif tool_name in {'pause_training_loop', 'resume_training_loop', 'stop_training_loop'} and result.get('ok'):
        tr.active_loop_id = str(result.get('loop_id') or tr.active_loop_id)
        tr.active_loop_status = str(result.get('status') or tr.active_loop_status)
        tr.last_loop_status = _training_loop_status_snapshot(result)
    elif tool_name == 'check_training_status':
        tr.last_status = _training_status_snapshot(result)
        is_running = bool(result.get('running'))
        tr.running = is_running
        _apply_resolved_training_args(
            tr,
            ds,
            resolved_args=result.get('resolved_args') or {},
            training_environment=result.get('training_environment') or {},
        )
        tr.device = str(result.get('device') or tr.device)
        tr.log_file = str(result.get('log_file') or tr.log_file)
        tr.started_at = result.get('started_at', tr.started_at)
        _apply_training_command_overrides(tr, ds, result.get('command') or [])
        tr.pid = result.get('pid', tr.pid) if is_running else None
    elif tool_name == 'summarize_training_run' and result.get('ok'):
        summary_snapshot = _training_run_summary_snapshot(result)
        tr.last_summary = summary_snapshot
        tr.training_run_summary = summary_snapshot
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
        tr.last_status = _training_status_snapshot(result)
    elif tool_name == 'predict_images' and result.get('ok'):
        pred.source_path = str(result.get('source_path') or tool_args.get('source_path', pred.source_path))
        pred.model = str(result.get('model') or tool_args.get('model', pred.model))
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_result = _prediction_result_snapshot(result, mode='images')
    elif tool_name == 'predict_videos' and result.get('ok'):
        pred.source_path = str(result.get('source_path') or tool_args.get('source_path', pred.source_path))
        pred.model = str(result.get('model') or tool_args.get('model', pred.model))
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_result = _prediction_result_snapshot(result, mode='videos')
    elif tool_name == 'summarize_prediction_results' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        if result.get('model'):
            pred.model = str(result.get('model'))
        if result.get('source_path'):
            pred.source_path = str(result.get('source_path'))
        pred.last_summary = _prediction_result_snapshot(result, mode=str(result.get('mode') or 'images'))
        pred.last_result = dict(pred.last_summary)
    elif tool_name == 'inspect_prediction_outputs' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_inspection = _prediction_management_snapshot(
            result,
            overview_key='prediction_output_overview',
            extra_keys=('artifact_roots', 'path_list_files', 'output_dir', 'report_path'),
        )
    elif tool_name == 'export_prediction_report' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_export = _prediction_management_snapshot(
            result,
            overview_key='export_overview',
            extra_keys=('export_path', 'export_format', 'output_dir', 'report_path'),
        )
    elif tool_name == 'export_prediction_path_lists' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_path_lists = _prediction_management_snapshot(
            result,
            overview_key='path_list_overview',
            fallback_overview_key='export_overview',
            extra_keys=('export_dir', 'output_dir', 'report_path', 'detected_items_path', 'empty_items_path', 'failed_items_path', 'detected_count', 'empty_count', 'failed_count'),
        )
    elif tool_name == 'organize_prediction_results' and result.get('ok'):
        pred.output_dir = str(result.get('source_output_dir') or pred.output_dir)
        pred.report_path = str(result.get('source_report_path') or pred.report_path)
        pred.last_organized_result = _prediction_management_snapshot(
            result,
            overview_key='organization_overview',
            extra_keys=('destination_dir', 'organize_by', 'artifact_preference', 'copied_items', 'bucket_stats', 'sample_outputs'),
        )
    elif tool_name == 'scan_cameras' and result.get('ok'):
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='camera_overview',
            extra_keys=('camera_count', 'cameras'),
        )
    elif tool_name == 'scan_screens' and result.get('ok'):
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='screen_overview',
            extra_keys=('screen_count', 'screens'),
        )
    elif tool_name == 'test_rtsp_stream':
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='stream_test_overview',
            extra_keys=('rtsp_url', 'ok', 'error'),
        )
    elif tool_name in {'start_camera_prediction', 'start_rtsp_prediction', 'start_screen_prediction'} and result.get('ok'):
        pred.model = str(tool_args.get('model') or pred.model)
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.realtime_session_id = str(result.get('session_id') or pred.realtime_session_id)
        pred.realtime_source_type = str(result.get('source_type') or pred.realtime_source_type)
        pred.realtime_source_label = str(result.get('source_label') or pred.realtime_source_label)
        pred.realtime_status = 'running'
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='realtime_session_overview',
            extra_keys=('session_id', 'source_type', 'source_label', 'output_dir', 'status'),
        )
        pred.last_realtime_status['status'] = 'running'
    elif tool_name == 'check_realtime_prediction_status' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.realtime_session_id = str(result.get('session_id') or pred.realtime_session_id)
        pred.realtime_source_type = str(result.get('source_type') or pred.realtime_source_type)
        pred.realtime_source_label = str(result.get('source_label') or pred.realtime_source_label)
        pred.realtime_status = str(result.get('status') or pred.realtime_status)
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='realtime_status_overview',
            extra_keys=(
                'session_id',
                'source_type',
                'source_label',
                'status',
                'processed_frames',
                'detected_frames',
                'total_detections',
                'class_counts',
                'output_dir',
                'report_path',
                'error',
            ),
        )
    elif tool_name == 'stop_realtime_prediction' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.realtime_session_id = str(result.get('session_id') or pred.realtime_session_id)
        pred.realtime_source_type = str(result.get('source_type') or pred.realtime_source_type)
        pred.realtime_source_label = str(result.get('source_label') or pred.realtime_source_label)
        pred.realtime_status = str(result.get('status') or 'stopped')
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='realtime_status_overview',
            extra_keys=(
                'session_id',
                'source_type',
                'source_label',
                'status',
                'processed_frames',
                'detected_frames',
                'total_detections',
                'class_counts',
                'output_dir',
                'report_path',
                'error',
            ),
        )
    elif tool_name == 'retrieve_training_knowledge' and result.get('ok'):
        kn.last_retrieval = _knowledge_snapshot(
            result,
            overview_key='retrieval_overview',
            extra_keys=('topic', 'stage', 'model_family', 'matched_rule_ids'),
        )
    elif tool_name == 'analyze_training_outcome' and result.get('ok'):
        kn.last_analysis = _knowledge_snapshot(
            result,
            overview_key='analysis_overview',
            extra_keys=('assessment', 'matched_rule_ids', 'signals'),
        )
    elif tool_name == 'recommend_next_training_step' and result.get('ok'):
        kn.last_recommendation = _knowledge_snapshot(
            result,
            overview_key='recommendation_overview',
            extra_keys=('recommended_action', 'matched_rule_ids', 'signals'),
        )
    elif tool_name == 'list_remote_profiles' and result.get('ok'):
        rt.last_profile_listing = _remote_transfer_snapshot(
            result,
            overview_key='profile_overview',
            extra_keys=('profiles_path', 'default_profile', 'profile_count', 'ssh_alias_count'),
        )
        rt.last_profile_listing['profile_count'] = len(result.get('profiles') or [])
        rt.last_profile_listing['ssh_alias_count'] = len(result.get('ssh_aliases') or [])
        profiles = result.get('profiles') or []
        default_profile = str(result.get('default_profile') or '').strip()
        if default_profile:
            rt.profile_name = default_profile
            for item in profiles:
                if str(item.get('name') or '').strip() == default_profile:
                    rt.target_label = str(item.get('target_label') or rt.target_label or default_profile)
                    rt.remote_root = str(item.get('remote_root') or rt.remote_root)
                    break
    elif tool_name == 'upload_assets_to_remote' and result.get('ok'):
        rt.target_label = str(result.get('target_label') or rt.target_label)
        rt.profile_name = str(result.get('profile_name') or rt.profile_name)
        rt.remote_root = str(result.get('remote_root') or rt.remote_root)
        rt.last_upload = _remote_transfer_snapshot(
            result,
            overview_key='transfer_overview',
            extra_keys=(
                'target_label',
                'profile_name',
                'remote_root',
                'uploaded_count',
                'uploaded_items',
                'file_count',
                'verified_file_count',
                'skipped_file_count',
                'chunked_file_count',
                'scp_file_count',
                'transferred_bytes',
                'skipped_bytes',
                'total_bytes',
                'resume_enabled',
                'verify_hash',
                'hash_algorithm',
                'large_file_threshold_mb',
                'chunk_size_mb',
                'transfer_strategy_summary',
                'file_results_preview',
            ),
        )
        rt.last_upload['target_label'] = rt.target_label
        rt.last_upload['profile_name'] = rt.profile_name
        rt.last_upload['remote_root'] = rt.remote_root
        rt.last_upload['uploaded_items'] = result.get('uploaded_items') or []
        rt.last_upload['file_results_preview'] = result.get('file_results_preview') or []
    elif tool_name == 'download_assets_from_remote' and result.get('ok'):
        rt.target_label = str(result.get('target_label') or rt.target_label)
        rt.profile_name = str(result.get('profile_name') or rt.profile_name)
        rt.last_download = _remote_transfer_snapshot(
            result,
            overview_key='download_overview',
            extra_keys=('target_label', 'profile_name', 'local_root', 'downloaded_count', 'downloaded_items'),
        )
        rt.last_download['target_label'] = rt.target_label
        rt.last_download['profile_name'] = rt.profile_name
        rt.last_download['downloaded_items'] = result.get('downloaded_items') or []

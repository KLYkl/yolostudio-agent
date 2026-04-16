from __future__ import annotations

from typing import Any


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


def _training_loop_request_snapshot(tool_args: dict[str, Any] | None) -> dict[str, Any]:
    snapshot = _pick_mapping(
        tool_args or {},
        'model',
        'data_yaml',
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
        'managed_level',
        'max_rounds',
        'target_metric',
        'target_metric_value',
        'loop_name',
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
    assign_model: bool = True,
    assign_data_yaml: bool = True,
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

    if assign_model:
        tr.model = str(model_value or '')
    if assign_data_yaml:
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
        'observation_stage',
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
        'next_actions',
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

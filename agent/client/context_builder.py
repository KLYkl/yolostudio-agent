from __future__ import annotations

from typing import Iterable

from langchain_core.messages import BaseMessage, SystemMessage

from yolostudio_agent.agent.client.cached_tool_reply_service import build_cached_tool_snapshot_message
from yolostudio_agent.agent.client.dataset_fact_service import build_dataset_fact_snapshot_message
from yolostudio_agent.agent.client.event_retriever import MemoryDigest
from yolostudio_agent.agent.client.session_state import SessionState


def _cache_state(data: dict) -> str:
    return '已缓存' if data else '无'


def _non_empty(value: object) -> bool:
    return value not in (None, '', [], {}, ())


def _join_values(values: object, *, limit: int = 4) -> str:
    if not isinstance(values, (list, tuple, set)):
        return ''
    items = [str(item).strip() for item in values if str(item).strip()]
    if not items:
        return ''
    if len(items) > limit:
        return ', '.join(items[:limit]) + f' 等 {len(items)} 项'
    return ', '.join(items)


def _format_percent(value: object) -> str:
    if value in (None, ''):
        return ''
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return ''
    return f'{ratio:.1%}'


def _format_extreme_class(value: object) -> str:
    if not isinstance(value, dict):
        return ''
    name = str(value.get('name') or '').strip()
    count = value.get('count')
    if not name or count in (None, ''):
        return ''
    return f'{name} ({count})'


def _format_top_classes(value: object, *, limit: int = 4) -> str:
    if not isinstance(value, list):
        return ''
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get('class_name') or item.get('name') or '').strip()
        count = item.get('count')
        if not name or count in (None, ''):
            continue
        parts.append(f'{name}={count}')
    if not parts:
        return ''
    if len(parts) > limit:
        return ', '.join(parts[:limit]) + f' 等 {len(parts)} 项'
    return ', '.join(parts)


def _append_section(lines: list[str], title: str, entries: list[tuple[str, object]]) -> None:
    filtered = [(key, value) for key, value in entries if _non_empty(value)]
    if not filtered:
        return
    lines.append(f'{title}:')
    for key, value in filtered:
        lines.append(f'  {key}: {value}')


class ContextBuilder:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def build_messages(
        self,
        state: SessionState,
        recent_messages: Iterable[BaseMessage],
        digest: MemoryDigest | None = None,
    ) -> list[BaseMessage]:
        summary = self.build_state_summary(state, digest)
        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=summary),
        ]
        cached_tool_snapshot = build_cached_tool_snapshot_message(state)
        if cached_tool_snapshot:
            messages.append(SystemMessage(content=cached_tool_snapshot))
        dataset_fact_snapshot = build_dataset_fact_snapshot_message(state)
        if dataset_fact_snapshot:
            messages.append(SystemMessage(content=dataset_fact_snapshot))
        messages.extend(list(recent_messages))
        return messages

    def build_state_summary(self, state: SessionState, digest: MemoryDigest | None = None) -> str:
        ds = state.active_dataset
        tr = state.active_training
        pc = state.pending_confirmation
        pred = state.active_prediction
        kn = state.active_knowledge
        rt = state.active_remote_transfer
        pref = state.preferences
        digest_text = digest.to_text() if digest else '无历史摘要'
        active_loop_request = dict(tr.active_loop_request or {})

        lines: list[str] = ['当前结构化上下文:']
        lines.append(f'- session_id: {state.session_id}')

        _append_section(
            lines,
            '数据集',
            [
                ('dataset_root', ds.dataset_root),
                ('img_dir', ds.img_dir),
                ('label_dir', ds.label_dir),
                ('data_yaml', ds.data_yaml),
                ('last_scan_summary', str((ds.last_scan or {}).get('summary') or '')),
                ('last_scan_classes', _join_values((ds.last_scan or {}).get('classes'))),
                ('last_scan_top_classes', _format_top_classes((ds.last_scan or {}).get('top_classes'))),
                ('last_scan_least_class', _format_extreme_class((ds.last_scan or {}).get('least_class'))),
                ('last_scan_most_class', _format_extreme_class((ds.last_scan or {}).get('most_class'))),
                ('last_scan_missing_label_ratio', _format_percent((ds.last_scan or {}).get('missing_label_ratio'))),
                ('last_scan_class_name_source', str((ds.last_scan or {}).get('class_name_source') or '')),
                ('last_validate_summary', str((ds.last_validate or {}).get('summary') or '')),
                ('last_health_summary', str((ds.last_health_check or {}).get('summary') or '')),
                ('last_health_duplicate_groups', (ds.last_health_check or {}).get('duplicate_groups')),
                ('last_duplicate_summary', str((ds.last_duplicate_check or {}).get('summary') or '')),
                ('last_duplicate_groups', (ds.last_duplicate_check or {}).get('duplicate_groups')),
                ('readiness_cache', _cache_state(ds.last_readiness) if ds.last_readiness else ''),
                ('health_cache', _cache_state(ds.last_health_check or ds.last_validate) if (ds.last_health_check or ds.last_validate) else ''),
                ('split_cache', _cache_state(ds.last_split) if ds.last_split else ''),
                ('extract_cache', _cache_state(ds.last_extract_result or ds.last_frame_extract or ds.last_video_scan or ds.last_extract_preview)
                 if (ds.last_extract_result or ds.last_frame_extract or ds.last_video_scan or ds.last_extract_preview) else ''),
            ],
        )

        _append_section(
            lines,
            '训练',
            [
                ('workflow_state', tr.workflow_state),
                ('loop_workflow_state', tr.loop_workflow_state),
                ('running', tr.running if tr.running else ''),
                ('model', tr.model),
                ('data_yaml', tr.data_yaml),
                ('device', tr.device),
                ('training_environment', tr.training_environment),
                ('project', tr.project),
                ('run_name', tr.run_name),
                ('active_loop_id', tr.active_loop_id),
                ('active_loop_status', tr.active_loop_status),
                ('active_loop_model', str(active_loop_request.get('model') or '')),
                ('active_loop_data_yaml', str(active_loop_request.get('data_yaml') or '')),
                ('active_loop_managed_level', str(active_loop_request.get('managed_level') or '')),
                ('preflight_cache', _cache_state(tr.last_preflight) if tr.last_preflight else ''),
                ('start_cache', _cache_state(tr.last_start_result) if tr.last_start_result else ''),
                ('status_cache', _cache_state(tr.last_status or tr.last_summary or tr.training_run_summary)
                 if (tr.last_status or tr.last_summary or tr.training_run_summary) else ''),
                ('loop_cache', _cache_state(tr.last_loop_status or tr.last_loop_detail)
                 if (tr.last_loop_status or tr.last_loop_detail) else ''),
                ('comparison_cache', _cache_state(tr.last_run_comparison) if tr.last_run_comparison else ''),
                ('training_plan_draft', '待确认' if tr.training_plan_draft else ''),
            ],
        )

        _append_section(
            lines,
            '预测',
            [
                ('source_path', pred.source_path),
                ('model', pred.model),
                ('output_dir', pred.output_dir),
                ('report_path', pred.report_path),
                ('image_prediction_session_id', pred.image_prediction_session_id),
                ('image_prediction_status', pred.image_prediction_status),
                ('last_image_prediction_summary', str((pred.last_image_prediction_status or {}).get('summary') or '')),
                ('last_result_summary', str((pred.last_result or {}).get('summary') or '')),
                ('last_summary_text', str((pred.last_summary or {}).get('summary') or '')),
                ('last_inspection_summary', str((pred.last_inspection or {}).get('summary') or '')),
                ('last_export_path', str((pred.last_export or {}).get('export_path') or '')),
                ('last_export_format', str((pred.last_export or {}).get('export_format') or '')),
                ('last_path_lists_dir', str((pred.last_path_lists or {}).get('export_dir') or '')),
                ('last_organized_destination', str((pred.last_organized_result or {}).get('destination_dir') or '')),
                ('last_organized_mode', str((pred.last_organized_result or {}).get('organize_by') or '')),
                ('result_cache', _cache_state(pred.last_result or pred.last_summary or pred.last_inspection or pred.last_image_prediction_status)
                 if (pred.last_result or pred.last_summary or pred.last_inspection or pred.last_image_prediction_status) else ''),
                ('realtime_session_id', pred.realtime_session_id),
                ('realtime_source_type', pred.realtime_source_type),
                ('realtime_source_label', pred.realtime_source_label),
                ('realtime_status', pred.realtime_status),
                ('realtime_cache', _cache_state(pred.last_realtime_status) if pred.last_realtime_status else ''),
            ],
        )

        _append_section(
            lines,
            '知识',
            [
                ('retrieval_topic', str((kn.last_retrieval or {}).get('topic') or '')),
                ('retrieval_stage', str((kn.last_retrieval or {}).get('stage') or '')),
                ('retrieval_signals', _join_values((kn.last_retrieval or {}).get('signals'))),
                ('retrieval_summary', str((kn.last_retrieval or {}).get('summary') or '')),
                ('analysis_signals', _join_values((kn.last_analysis or {}).get('signals'))),
                ('analysis_summary', str((kn.last_analysis or {}).get('summary') or '')),
                ('recommended_action', str((kn.last_recommendation or {}).get('recommended_action') or '')),
                ('recommendation_summary', str((kn.last_recommendation or {}).get('summary') or '')),
                ('retrieval_cache', _cache_state(kn.last_retrieval) if kn.last_retrieval else ''),
                ('analysis_cache', _cache_state(kn.last_analysis) if kn.last_analysis else ''),
                ('recommendation_cache', _cache_state(kn.last_recommendation) if kn.last_recommendation else ''),
            ],
        )

        _append_section(
            lines,
            '远端传输',
            [
                ('target_label', rt.target_label),
                ('profile_name', rt.profile_name),
                ('remote_root', rt.remote_root),
                ('profile_cache', _cache_state(rt.last_profile_listing) if rt.last_profile_listing else ''),
                ('upload_cache', _cache_state(rt.last_upload) if rt.last_upload else ''),
                ('download_cache', _cache_state(rt.last_download) if rt.last_download else ''),
            ],
        )

        _append_section(
            lines,
            '待确认操作',
            [
                ('tool', pc.tool_name),
                ('summary', pc.summary),
                ('objective', pc.objective),
                ('allowed_decisions', ', '.join(pc.allowed_decisions) if (pc.tool_name or pc.summary or pc.objective or pc.decision_context) and pc.allowed_decisions else ''),
                ('latest_review', str((pc.decision_context or {}).get('decision') or '')),
            ],
        )

        _append_section(
            lines,
            '偏好',
            [
                ('default_model', pref.default_model),
                ('default_epochs', pref.default_epochs if pref.default_epochs is not None else ''),
            ],
        )

        if digest_text != '无历史摘要':
            lines.append('历史摘要:')
            lines.append(digest_text)

        return '\n'.join(lines) + '\n'

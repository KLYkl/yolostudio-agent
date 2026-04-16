from __future__ import annotations

from typing import Any, Callable

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.tool_adapter import canonical_tool_name
from yolostudio_agent.agent.client.tool_policy import ToolExecutionPolicy, pending_allowed_decisions


def pending_action_objective(state: SessionState, tool_name: str, args: dict[str, Any]) -> str:
    tool_name = str(tool_name or '').strip()
    dataset_path = str(args.get('dataset_path') or state.active_dataset.dataset_root or '').strip()
    data_yaml = str(args.get('data_yaml') or state.active_training.data_yaml or state.active_dataset.data_yaml or '').strip()
    model = str(args.get('model') or state.active_training.model or '').strip()
    if tool_name == 'prepare_dataset_for_training':
        return f'把数据集准备到可训练状态{f"（{dataset_path}）" if dataset_path else ""}'
    if tool_name == 'start_training':
        parts = [part for part in [model, data_yaml] if part]
        return '启动训练' + (f"（{' / '.join(parts)}）" if parts else '')
    if tool_name == 'start_training_loop':
        parts = [part for part in [model, data_yaml] if part]
        return '启动循环训练' + (f"（{' / '.join(parts)}）" if parts else '')
    if tool_name == 'upload_assets_to_remote':
        return '把本地资源上传到远端服务器'
    if tool_name == 'remote_training_pipeline':
        return '执行远端训练闭环'
    if tool_name == 'remote_prediction_pipeline':
        return '执行远端预测闭环'
    return f'执行 {tool_name}'


def pending_action_summary(state: SessionState, tool_name: str, args: dict[str, Any]) -> str:
    tool_name = str(tool_name or '').strip()
    dataset_path = str(args.get('dataset_path') or state.active_dataset.dataset_root or '').strip()
    data_yaml = str(args.get('data_yaml') or state.active_training.data_yaml or state.active_dataset.data_yaml or '').strip()
    model = str(args.get('model') or state.active_training.model or '').strip()
    if tool_name == 'prepare_dataset_for_training':
        details = [item for item in [dataset_path or None, 'force_split=true' if args.get('force_split') else None] if item]
        return '准备数据集' + (f"：{'，'.join(details)}" if details else '')
    if tool_name == 'start_training':
        details = [item for item in [f'model={model}' if model else None, f'data={data_yaml}' if data_yaml else None, f"epochs={args.get('epochs')}" if args.get('epochs') is not None else None] if item]
        return '启动训练' + (f"：{', '.join(details)}" if details else '')
    if tool_name == 'start_training_loop':
        details = [item for item in [f'model={model}' if model else None, f'data={data_yaml}' if data_yaml else None, f"max_rounds={args.get('max_rounds')}" if args.get('max_rounds') is not None else None] if item]
        return '启动循环训练' + (f"：{', '.join(details)}" if details else '')
    if tool_name == 'upload_assets_to_remote':
        return '上传资源到远端服务器'
    if tool_name == 'remote_training_pipeline':
        return '执行远端训练闭环'
    if tool_name == 'remote_prediction_pipeline':
        return '执行远端预测闭环'
    return f'待确认动作：{tool_name}'


def pending_review_config(tool_name: str, policy: ToolExecutionPolicy) -> dict[str, Any]:
    normalized = canonical_tool_name(tool_name)
    return {
        'risk_level': policy.risk_level,
        'allow_edit': not policy.read_only,
        'allow_clarify': True,
        'tool_name': normalized,
        'confirmation_required': policy.confirmation_required,
        'read_only': policy.read_only,
        'destructive': policy.destructive,
        'open_world': policy.open_world,
    }


def build_pending_action_payload(
    state: SessionState,
    pending: dict[str, Any],
    *,
    tool_policy_resolver: Callable[[str], ToolExecutionPolicy],
    thread_id: str | None = None,
    decision_state: str = 'pending',
) -> dict[str, Any]:
    tool_name = str(pending.get('name') or '').strip()
    args = dict(pending.get('args') or {})
    policy = tool_policy_resolver(tool_name)
    return {
        'interrupt_kind': 'tool_approval',
        'decision_state': decision_state,
        'thread_id': str(thread_id or pending.get('thread_id') or state.pending_confirmation.thread_id or '').strip(),
        'tool_name': tool_name,
        'tool_args': args,
        'summary': str(pending.get('summary') or pending_action_summary(state, tool_name, args)).strip(),
        'objective': str(pending.get('objective') or pending_action_objective(state, tool_name, args)).strip(),
        'allowed_decisions': list(pending.get('allowed_decisions') or pending_allowed_decisions(policy)),
        'review_config': dict(pending.get('review_config') or pending_review_config(tool_name, policy)),
        'decision_context': dict(pending.get('decision_context') or state.pending_confirmation.decision_context or {}),
    }


def confirmation_user_facts(
    state: SessionState,
    tool_call: dict[str, Any],
    *,
    confirmation_mode: str,
    human_training_step_name: Callable[[str], str],
    compact_action_candidates: Callable[[Any], list[dict[str, Any]]],
) -> dict[str, Any]:
    args = dict(tool_call.get('args') or {})
    tool_name = str(tool_call.get('name') or '').strip()
    ds = state.active_dataset
    tr = state.active_training
    facts: dict[str, Any] = {
        'tool_name': tool_name,
        'tool_action': human_training_step_name(tool_name),
        'confirmation_mode': confirmation_mode,
        'dataset_path': str(args.get('dataset_path') or ds.dataset_root or ds.img_dir or '').strip(),
        'data_yaml': str(args.get('data_yaml') or tr.data_yaml or ds.data_yaml or '').strip(),
        'model': str(args.get('model') or tr.model or '').strip(),
        'classes_txt': str(args.get('classes_txt') or '').strip(),
        'force_split': bool(args.get('force_split')),
        'device': str(args.get('device') or tr.device or '').strip(),
        'training_environment': str(args.get('training_environment') or tr.training_environment or '').strip(),
        'project': str(args.get('project') or tr.project or '').strip(),
        'run_name': str(args.get('name') or tr.run_name or '').strip(),
        'managed_level': str(args.get('managed_level') or '').strip(),
        'max_rounds': args.get('max_rounds'),
    }
    readiness = ds.last_readiness or {}
    summary = str(readiness.get('summary') or '').strip()
    if summary:
        facts['dataset_summary'] = summary
    readiness_overview = dict(readiness.get('readiness_overview') or {})
    if readiness_overview:
        facts['dataset_readiness'] = {
            'ready': readiness_overview.get('ready'),
            'preparable': readiness_overview.get('preparable'),
            'primary_blocker_type': readiness_overview.get('primary_blocker_type'),
            'blocker_codes': list(readiness_overview.get('blocker_codes') or [])[:4],
            'risk_level': readiness_overview.get('risk_level'),
            'warning_count': readiness_overview.get('warning_count'),
            'blocker_count': readiness_overview.get('blocker_count'),
            'needs_split': readiness_overview.get('needs_split'),
            'needs_data_yaml': readiness_overview.get('needs_data_yaml'),
        }
    blockers = [str(item).strip() for item in (readiness.get('blockers') or []) if str(item).strip()]
    if blockers:
        facts['dataset_blockers'] = blockers[:4]
    warnings = [str(item).strip() for item in (readiness.get('warnings') or []) if str(item).strip()]
    if warnings:
        facts['dataset_warnings'] = warnings[:4]
    action_candidates = compact_action_candidates(readiness.get('action_candidates'))
    if action_candidates:
        facts['action_candidates'] = action_candidates
    return {
        key: value
        for key, value in facts.items()
        if value is not None and value != '' and value != [] and value != {}
    }


def build_cancel_message(tool_call: dict[str, Any]) -> str:
    del tool_call
    return '好，我先不执行这一步。当前计划已保留；如果你想改参数、换模型、追问原因，或者稍后重新确认，都可以直接告诉我。'

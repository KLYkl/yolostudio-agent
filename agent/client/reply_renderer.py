from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.tool_adapter import canonical_tool_name, stringify_tool_result_facts


TrainingPlanDraftRenderer = Callable[..., str]
TrainingPlanMessageRenderer = Callable[..., Awaitable[str]]
ConfirmationMessageRenderer = Callable[[dict[str, Any]], Awaitable[str]]
ConfirmationPromptBuilder = Callable[[dict[str, Any]], str]
ConfirmationFactsBuilder = Callable[[dict[str, Any]], dict[str, Any]]
RendererTextInvoker = Callable[..., Awaitable[str]]
EventAppender = Callable[[str, dict[str, Any]], None]
GroundedReplyBuilder = Callable[[list[tuple[str, dict[str, Any]]]], str]
GroundedSectionMerger = Callable[[list[str]], str]
ToolResultMessageRenderer = Callable[[str, dict[str, Any]], Awaitable[str]]


def build_confirmation_prompt(
    session_state: SessionState,
    tool_call: dict[str, Any],
    *,
    render_training_plan_draft: TrainingPlanDraftRenderer,
    remote_join: Callable[[str, str], str],
) -> str:
    args = tool_call.get('args', {})
    tool_name = str(tool_call.get('name') or '')
    ds = session_state.active_dataset
    tr = session_state.active_training
    plan_draft = tr.training_plan_draft or {}
    execution_mode = str(plan_draft.get('execution_mode') or '').strip().lower()

    if (
        plan_draft
        and execution_mode != 'prepare_only'
        and str(plan_draft.get('next_step_tool') or '').strip() == tool_name
    ):
        return render_training_plan_draft(plan_draft, pending=True)

    if tool_name == 'prepare_dataset_for_training':
        lines = ['准备执行：数据准备']
        dataset_path = str(args.get('dataset_path') or ds.dataset_root or ds.img_dir or '').strip()
        if dataset_path:
            lines.append(f'数据集: {dataset_path}')
        readiness = ds.last_readiness or {}
        resolved_img_dir = str(readiness.get('resolved_img_dir') or ds.img_dir or '').strip()
        resolved_label_dir = str(readiness.get('resolved_label_dir') or ds.label_dir or '').strip()
        resolved_yaml = str(readiness.get('resolved_data_yaml') or ds.data_yaml or '').strip()
        if resolved_img_dir and resolved_label_dir:
            lines.append('当前状态: 已识别图片目录和标注目录')
        if not resolved_yaml:
            lines.append('当前状态: 还没有可用的 data.yaml，本次会自动补齐训练产物')
        elif readiness.get('ready'):
            lines.append(f'当前状态: 已有可用 data.yaml（{resolved_yaml}）')
        if args.get('force_split'):
            lines.append('附加安排: 按默认比例划分数据')
        planned_yaml = resolved_yaml or remote_join(dataset_path, 'data.yaml') if dataset_path else ''
        if planned_yaml:
            lines.append(f'预期产物: data.yaml -> {planned_yaml}')
        lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
        return '\n'.join(lines)

    if tool_name == 'start_training':
        lines = ['准备执行：启动训练']
        readiness = ds.last_readiness or {}
        if readiness.get('summary'):
            lines.append(f"数据理解: {readiness.get('summary')}")
        preflight = tr.last_preflight or {}
        environment = preflight.get('training_environment') or tr.last_environment_probe.get('default_environment') or {}
        env_name = environment.get('display_name') or environment.get('name')
        if env_name:
            lines.append(f'训练环境: {env_name}')
        model = args.get('model') or tr.model
        data_yaml = args.get('data_yaml') or tr.data_yaml or ds.data_yaml
        epochs = args.get('epochs') or (preflight.get('resolved_args') or {}).get('epochs') or 100
        device = args.get('device') or (preflight.get('resolved_args') or {}).get('device') or 'auto'
        batch = args.get('batch')
        if batch is None:
            batch = (preflight.get('resolved_args') or {}).get('batch')
        imgsz = args.get('imgsz')
        if imgsz is None:
            imgsz = (preflight.get('resolved_args') or {}).get('imgsz')
        project = args.get('project')
        if not project:
            project = (preflight.get('resolved_args') or {}).get('project')
        run_name = args.get('name')
        if not run_name:
            run_name = (preflight.get('resolved_args') or {}).get('name')
        fraction = args.get('fraction')
        if fraction is None:
            fraction = (preflight.get('resolved_args') or {}).get('fraction')
        classes = args.get('classes')
        if classes is None:
            classes = (preflight.get('resolved_args') or {}).get('classes')
        single_cls = args.get('single_cls')
        if single_cls is None:
            single_cls = (preflight.get('resolved_args') or {}).get('single_cls')
        plan_bits = [f'model={model}', f'data={data_yaml}', f'epochs={epochs}', f'device={device}']
        if batch is not None:
            plan_bits.append(f'batch={batch}')
        if imgsz is not None:
            plan_bits.append(f'imgsz={imgsz}')
        lines.append(f"初步安排: {', '.join(str(item) for item in plan_bits)}")
        output_bits = []
        if project:
            output_bits.append(f'project={project}')
        if run_name:
            output_bits.append(f'name={run_name}')
        if output_bits:
            lines.append(f"输出组织: {', '.join(output_bits)}")
        advanced_bits = []
        for key, value in (('fraction', fraction), ('classes', classes), ('single_cls', single_cls)):
            if value is not None and value != '':
                advanced_bits.append(f'{key}={value}')
        if advanced_bits:
            lines.append(f"高级参数: {', '.join(advanced_bits)}")
        if preflight.get('summary'):
            lines.append(f"预检: {preflight.get('summary')}")
        command_preview = preflight.get('command_preview') or []
        if command_preview:
            preview_text = ' '.join(str(item) for item in command_preview[:6])
            if len(command_preview) > 6:
                preview_text += ' ...'
            lines.append(f'命令预览: {preview_text}')
        lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
        return '\n'.join(lines)

    if tool_name == 'start_training_loop':
        lines = ['准备执行：启动环训练']
        readiness = ds.last_readiness or {}
        if readiness.get('summary'):
            lines.append(f"数据理解: {readiness.get('summary')}")
        if args.get('loop_name'):
            lines.append(f"环训练名称: {args.get('loop_name')}")
        model = args.get('model') or tr.model
        data_yaml = args.get('data_yaml') or tr.data_yaml or ds.data_yaml
        if model:
            lines.append(f'模型: {model}')
        if data_yaml:
            lines.append(f'数据 YAML: {data_yaml}')
        lines.append(f"托管级别: {args.get('managed_level') or 'conservative_auto'}")
        lines.append(f"最大轮数: {args.get('max_rounds') or 5}")
        if args.get('target_metric'):
            target_line = f"目标指标: {args.get('target_metric')}"
            if args.get('target_metric_value') is not None:
                target_line += f" >= {args.get('target_metric_value')}"
            lines.append(target_line)
        plan_bits = []
        for key in ('epochs', 'batch', 'imgsz', 'device'):
            value = args.get(key)
            if value is not None and value != '':
                plan_bits.append(f'{key}={value}')
        if plan_bits:
            lines.append(f"首轮参数: {', '.join(str(item) for item in plan_bits)}")
        allowed_tuning_params = list(args.get('allowed_tuning_params') or [])
        if allowed_tuning_params:
            lines.append(f"允许自动调整: {', '.join(str(item) for item in allowed_tuning_params)}")
        lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
        return '\n'.join(lines)

    if tool_name == 'upload_assets_to_remote':
        lines = ['准备执行：远端上传']
        target_label = str(args.get('server') or session_state.active_remote_transfer.target_label or '').strip()
        remote_root = str(args.get('remote_root') or session_state.active_remote_transfer.remote_root or '').strip()
        if target_label:
            lines.append(f'目标服务器: {target_label}')
        if remote_root:
            lines.append(f'远端目录: {remote_root}')
        local_paths = list(args.get('local_paths') or [])
        if local_paths:
            lines.append('本地上传项:')
            lines.extend(f'- {item}' for item in local_paths[:5])
            if len(local_paths) > 5:
                lines.append(f'- 其余 {len(local_paths) - 5} 项已省略')
        lines.append(
            '默认策略: 大文件自动分块 + 断点续传 + 哈希校验'
            f" (threshold={args.get('large_file_threshold_mb', 256)}MB, chunk={args.get('chunk_size_mb', 64)}MB)"
        )
        lines.append('说明: 这会把本机文件/目录复制到远端服务器。')
        lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
        return '\n'.join(lines)

    if tool_name == 'remote_prediction_pipeline':
        lines = ['准备执行：远端预测闭环']
        pipeline_args = dict(args or {})
        upload_args = dict(pipeline_args.get('upload_args') or {})
        target_label = str(upload_args.get('server') or session_state.active_remote_transfer.target_label or '').strip()
        remote_root = str(upload_args.get('remote_root') or session_state.active_remote_transfer.remote_root or '').strip()
        local_paths = list(upload_args.get('local_paths') or [])
        if target_label:
            lines.append(f'目标服务器: {target_label}')
        if remote_root:
            lines.append(f'远端目录: {remote_root}')
        if local_paths:
            lines.append('本地上传项:')
            lines.extend(f'- {item}' for item in local_paths[:5])
            if len(local_paths) > 5:
                lines.append(f'- 其余 {len(local_paths) - 5} 项已省略')
        local_result_root = str(pipeline_args.get('local_result_root') or '').strip()
        if local_result_root:
            lines.append(f'本机回传目录: {local_result_root}')
        lines.append('执行链路: 上传本地模型/图片或视频 -> 远端执行 prediction -> 结果下载回本机')
        lines.append('限制: 待预测输入当前要求是单个文件或单个目录；多个散文件请先整理进目录。')
        lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
        return '\n'.join(lines)

    if tool_name == 'remote_training_pipeline':
        lines = ['准备执行：远端训练闭环']
        pipeline_args = dict(args or {})
        upload_args = dict(pipeline_args.get('upload_args') or {})
        target_label = str(upload_args.get('server') or session_state.active_remote_transfer.target_label or '').strip()
        remote_root = str(upload_args.get('remote_root') or session_state.active_remote_transfer.remote_root or '').strip()
        local_paths = list(upload_args.get('local_paths') or [])
        if target_label:
            lines.append(f'目标服务器: {target_label}')
        if remote_root:
            lines.append(f'远端目录: {remote_root}')
        if local_paths:
            lines.append('本地上传项:')
            lines.extend(f'- {item}' for item in local_paths[:5])
            if len(local_paths) > 5:
                lines.append(f'- 其余 {len(local_paths) - 5} 项已省略')
        lines.append('执行链路: 上传本地模型/数据集 -> 远端做 readiness/prepare/preflight -> 启动训练')
        if pipeline_args.get('force_split'):
            lines.append('附加安排: 数据未就绪时自动按默认比例划分并补齐训练产物')
        if pipeline_args.get('wait_for_completion'):
            lines.append(
                f"等待策略: 启动后轮询训练状态直到结束 (poll={pipeline_args.get('poll_interval_seconds', 15)}s, "
                f"max_wait={pipeline_args.get('max_wait_seconds', 7200)}s)"
            )
        if pipeline_args.get('download_after_completion'):
            local_result_root = str(pipeline_args.get('local_result_root') or '').strip()
            if local_result_root:
                lines.append(f'训练产物回传目录: {local_result_root}')
            lines.append('附加安排: 训练结束后自动把远端 run 目录下载回本机')
        lines.append('说明: 这会在远端真正启动训练进程，属于高风险动作。')
        lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
        return '\n'.join(lines)

    pretty_args = '\n'.join(f'  - {k}: {v}' for k, v in args.items()) or '  - 无参数'
    return (
        f'检测到高风险操作：{tool_name}\n'
        f'参数摘要：\n{pretty_args}\n'
        '如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。'
    )


async def build_confirmation_message(
    session_state: SessionState,
    tool_call: dict[str, Any],
    *,
    render_training_plan_message: TrainingPlanMessageRenderer,
    render_confirmation_message: ConfirmationMessageRenderer,
) -> str:
    args = tool_call.get('args', {})
    tool_name = str(tool_call.get('name') or '')
    plan_draft = session_state.active_training.training_plan_draft or {}
    execution_mode = str(plan_draft.get('execution_mode') or '').strip().lower()
    if (
        plan_draft
        and execution_mode != 'prepare_only'
        and str(plan_draft.get('next_step_tool') or '').strip() == tool_name
    ):
        rendered_plan = await render_training_plan_message(plan_draft, pending=True)
        if tool_name == 'prepare_dataset_for_training':
            ds = session_state.active_dataset
            dataset_path = str(args.get('dataset_path') or ds.dataset_root or ds.img_dir or '').strip()
            readiness = ds.last_readiness or {}
            resolved_yaml = str(readiness.get('resolved_data_yaml') or ds.data_yaml or '').strip()
            planned_yaml = resolved_yaml or (f"{dataset_path.rstrip('/')}/data.yaml" if dataset_path else '')
            expected_output_line = f'预期产物: data.yaml -> {planned_yaml}' if planned_yaml else ''
            if expected_output_line and expected_output_line not in rendered_plan:
                rendered_plan = f'{rendered_plan}\n{expected_output_line}' if rendered_plan else expected_output_line
        return rendered_plan
    return await render_confirmation_message({'name': tool_name, 'args': args})


def confirmation_render_error(
    tool_call: dict[str, Any],
    *,
    error: Exception | None = None,
    append_event: EventAppender | None = None,
    build_confirmation_prompt: ConfirmationPromptBuilder,
) -> str:
    if error and append_event is not None:
        append_event(
            'confirmation_render_failed',
            {'tool': str(tool_call.get('name') or ''), 'error': str(error)},
        )
    return build_confirmation_prompt(tool_call)


async def render_confirmation_message(
    *,
    planner_llm: Any,
    tool_call: dict[str, Any],
    build_confirmation_prompt: ConfirmationPromptBuilder,
    confirmation_user_facts: ConfirmationFactsBuilder,
    invoke_renderer_text: RendererTextInvoker,
    confirmation_render_error: Callable[..., str],
) -> str:
    if planner_llm is None:
        return build_confirmation_prompt(tool_call)
    facts = confirmation_user_facts(tool_call)
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的确认说明器。'
                '请基于已验证事实，用自然中文说明即将执行的动作、原因和关键风险。'
                '不要输出工具名、字段名、JSON、命令或 payload。'
                '不要使用“原因和说明”“关键风险提示”这类固定小标题，也不要每次都复用同一套句式。'
                '如果这是循环训练相关动作，要明确说“循环训练”，不要混成普通训练。'
                '最后用一句自然中文询问用户是否继续，不要把确认限制写成 y/n。'
                '不要补充未验证事实。'
            )
        ),
        HumanMessage(
            content=(
                '请根据以下已验证事实，直接给用户一段自然中文确认说明：\n'
                f'{json.dumps(facts, ensure_ascii=False, indent=2)}'
            )
        ),
    ]
    text = await invoke_renderer_text(
        messages=messages,
        failure_event='confirmation_render_failed',
        failure_payload={'tool': str(tool_call.get('name') or '')},
    )
    if text:
        return text
    return confirmation_render_error(tool_call)


def compact_action_candidates(action_candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(action_candidates, list):
        return []
    compacted: list[dict[str, Any]] = []
    for item in action_candidates[:4]:
        if not isinstance(item, dict):
            continue
        compact = {
            'action': item.get('action'),
            'tool': item.get('tool'),
            'description': item.get('description'),
        }
        compacted.append({key: value for key, value in compact.items() if value not in (None, '', [], {})})
    return [item for item in compacted if item]


def structured_overview_payloads(parsed: dict[str, Any]) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    for key, value in parsed.items():
        if not key.endswith('_overview'):
            continue
        if isinstance(value, dict) and value:
            payloads[key] = dict(value)
        elif isinstance(value, list) and value:
            payloads[key] = list(value)
    for key in ('matched_rule_overview', 'playbook_overview'):
        value = parsed.get(key)
        if isinstance(value, list) and value:
            payloads[key] = list(value)[:4]
    return payloads


def remote_pipeline_applied_results(tool_name: str, parsed: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    normalized = canonical_tool_name(tool_name)
    if normalized == 'remote_prediction_pipeline':
        predict_tool_name = str(parsed.get('predict_tool_name') or 'predict_images').strip() or 'predict_images'
        ordered = [
            ('upload_assets_to_remote', dict(parsed.get('upload') or {})),
            (predict_tool_name, dict(parsed.get('predict') or {})),
            ('download_assets_from_remote', dict(parsed.get('download') or {})),
        ]
        return [(name, payload) for name, payload in ordered if payload]
    if normalized == 'remote_training_pipeline':
        ordered = [
            ('upload_assets_to_remote', dict(parsed.get('upload') or {})),
            ('training_readiness', dict(parsed.get('readiness') or {})),
            ('prepare_dataset_for_training', dict(parsed.get('prepare') or {})),
            ('training_preflight', dict(parsed.get('preflight') or {})),
            ('start_training', dict(parsed.get('start') or {})),
            ('check_training_status', dict(parsed.get('final_status') or {})),
            ('summarize_training_run', dict(parsed.get('final_summary') or {})),
            ('download_assets_from_remote', dict(parsed.get('download') or {})),
        ]
        return [(name, payload) for name, payload in ordered if payload]
    return []


def tool_result_user_facts(tool_name: str, parsed: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {
        'tool_name': tool_name,
        'ok': bool(parsed.get('ok')),
        'summary': str(parsed.get('summary') or parsed.get('message') or parsed.get('error') or '').strip(),
        'error': str(parsed.get('error') or '').strip(),
    }
    for overview_key, overview_value in structured_overview_payloads(parsed).items():
        facts[overview_key] = overview_value
    action_candidates = compact_action_candidates(parsed.get('action_candidates'))
    if action_candidates:
        facts['action_candidates'] = action_candidates
    if tool_name == 'start_training' and parsed.get('ok'):
        facts['can_check_progress'] = True
        facts['can_stop_run'] = True
    for key in (
        'data_yaml',
        'output_dir',
        'save_dir',
        'project',
        'name',
        'device',
        'pid',
        'log_file',
        'train_count',
        'val_count',
        'resolved_train_path',
        'resolved_val_path',
        'forced',
        'return_code',
    ):
        value = parsed.get(key)
        if value is None or value == '':
            continue
        facts[key] = value
    if tool_name in {
        'start_training_loop',
        'list_training_loops',
        'check_training_loop_status',
        'inspect_training_loop',
        'pause_training_loop',
        'resume_training_loop',
        'stop_training_loop',
    }:
        for key in (
            'loop_id',
            'loop_name',
            'status',
            'managed_level',
            'current_round_index',
            'completed_rounds',
            'max_rounds',
            'best_round_index',
            'best_target_metric',
            'failure_count',
            'no_improvement_streak',
            'termination_reason',
            'termination_detail',
            'active_loop_id',
        ):
            value = parsed.get(key)
            if value is None or value == '':
                continue
            facts[key] = value
        boundaries = dict(parsed.get('boundaries') or {})
        if boundaries:
            facts['target_metric'] = boundaries.get('target_metric')
            if boundaries.get('target_metric_value') is not None:
                facts['target_metric_value'] = boundaries.get('target_metric_value')
        next_round_plan = dict(parsed.get('next_round_plan') or {})
        if next_round_plan:
            facts['next_round_plan'] = {
                'round_index': next_round_plan.get('round_index'),
                'reason': next_round_plan.get('reason'),
                'decision_type': next_round_plan.get('decision_type'),
                'change_set': [
                    {
                        'field': item.get('field'),
                        'old': item.get('old'),
                        'new': item.get('new'),
                    }
                    for item in list(next_round_plan.get('change_set') or [])[:4]
                    if isinstance(item, dict)
                ],
                'experience_context': next_round_plan.get('experience_context'),
            }
        latest_round_card = dict(parsed.get('latest_round_card') or {})
        if latest_round_card:
            facts['latest_round_card'] = {
                'round_index': latest_round_card.get('round_index'),
                'status': latest_round_card.get('status'),
                'summary': latest_round_card.get('summary'),
                'metrics': latest_round_card.get('metrics') or {},
                'changed_params': list(latest_round_card.get('changed_params') or [])[:4],
                'knowledge_gate': latest_round_card.get('knowledge_gate'),
                'decision': latest_round_card.get('decision'),
                'next_plan': latest_round_card.get('next_plan'),
                'why': latest_round_card.get('why'),
                'recommendation': latest_round_card.get('recommendation'),
                'round_review': latest_round_card.get('round_review'),
                'round_memory': latest_round_card.get('round_memory'),
                'planner_output': latest_round_card.get('planner_output'),
                'experience_context': latest_round_card.get('experience_context'),
            }
        round_cards = list(parsed.get('round_cards') or [])
        if round_cards:
            facts['recent_round_cards'] = [
                {
                    'round_index': item.get('round_index'),
                    'status': item.get('status'),
                    'summary': item.get('summary'),
                    'knowledge_gate': item.get('knowledge_gate'),
                    'decision': item.get('decision'),
                    'round_review': item.get('round_review'),
                    'round_memory': item.get('round_memory'),
                    'planner_output': item.get('planner_output'),
                }
                for item in round_cards[-3:]
                if isinstance(item, dict)
            ]
        loops = list(parsed.get('loops') or [])
        if loops:
            facts['loop_count'] = len(loops)
            facts['recent_loops'] = [
                {
                    'loop_id': item.get('loop_id'),
                    'loop_name': item.get('loop_name'),
                    'status': item.get('status'),
                    'managed_level': item.get('managed_level'),
                    'current_round_index': item.get('current_round_index'),
                    'max_rounds': item.get('max_rounds'),
                    'best_round_index': item.get('best_round_index'),
                    'best_target_metric': item.get('best_target_metric'),
                }
                for item in loops[:5]
                if isinstance(item, dict)
            ]
    for key in ('knowledge_gate_status', 'latest_round_review', 'latest_round_memory', 'latest_planner_output'):
        value = parsed.get(key)
        if isinstance(value, dict) and value:
            facts[key] = value
        final_summary = dict(parsed.get('final_summary') or {})
        if final_summary:
            facts['final_summary'] = {
                'status': final_summary.get('status'),
                'best_round_index': final_summary.get('best_round_index'),
                'best_target_metric_name': final_summary.get('best_target_metric_name'),
                'best_target_metric': final_summary.get('best_target_metric'),
                'stop_reason': final_summary.get('stop_reason'),
                'termination_detail': final_summary.get('termination_detail'),
                'round_count': final_summary.get('round_count'),
                'last_round_review': final_summary.get('last_round_review'),
                'last_round_memory': final_summary.get('last_round_memory'),
                'last_planner_output': final_summary.get('last_planner_output'),
                'experience_timeline': list(final_summary.get('experience_timeline') or [])[-3:],
            }
        if 'action_candidates' not in facts:
            next_actions = list(parsed.get('next_actions') or [])
            if next_actions:
                facts['next_actions'] = next_actions[:4]
    if tool_name in {'remote_prediction_pipeline', 'remote_training_pipeline'}:
        for key in (
            'remote_source_path',
            'remote_model_path',
            'remote_dataset_path',
            'remote_output_dir',
            'remote_result_path',
            'local_result_root',
            'source_kind',
            'predict_tool_name',
            'final_run_state',
            'wait_for_completion',
            'download_after_completion',
        ):
            value = parsed.get(key)
            if value is None or value == '':
                continue
            facts[key] = value
    return facts


def fallback_tool_result_text(
    tool_name: str,
    parsed: dict[str, Any],
    *,
    build_grounded_tool_reply: GroundedReplyBuilder,
) -> str:
    structured_text = stringify_tool_result_facts(parsed).strip()
    if structured_text:
        return structured_text
    grounded_text = build_grounded_tool_reply([(tool_name, parsed)])
    if grounded_text:
        return grounded_text
    return (
        str(parsed.get('summary') or parsed.get('message') or parsed.get('error') or '').strip()
        or ('操作执行成功' if parsed.get('ok') else '操作执行失败')
    )


GROUNDED_TOOL_RENDER_ONLY = {
    'check_training_status',
    'stop_training',
}


def fallback_multi_tool_result_message(
    applied_results: list[tuple[str, dict[str, Any]]],
    *,
    extra_notes: list[str] | None = None,
    build_grounded_tool_reply: GroundedReplyBuilder,
    merge_grounded_sections: GroundedSectionMerger,
) -> str:
    sections: list[str] = []
    for tool_name, parsed in applied_results:
        normalized_name = str(canonical_tool_name(tool_name) or '').strip()
        if not normalized_name:
            continue
        normalized_parsed = parsed if isinstance(parsed, dict) else {'ok': False, 'summary': str(parsed or '').strip()}
        sections.append(
            fallback_tool_result_text(
                normalized_name,
                normalized_parsed,
                build_grounded_tool_reply=build_grounded_tool_reply,
            )
        )
    for note in extra_notes or []:
        text = str(note or '').strip()
        if text:
            sections.append(text)
    return merge_grounded_sections(sections)


def tool_result_render_error(tool_name: str, parsed: dict[str, Any], error: Exception | None = None) -> str:
    del tool_name
    summary = str(parsed.get('summary') or parsed.get('message') or parsed.get('error') or '').strip()
    if error and summary:
        return f'模型这次没有成功整理执行结果。我先给你真实摘要：{summary}'
    if error:
        return '模型这次没有成功整理执行结果，但工具已经执行完成。请稍后重试。'
    return summary or ('操作执行成功' if parsed.get('ok') else '操作执行失败')


async def render_multi_tool_result_message(
    *,
    planner_llm: Any,
    applied_results: list[tuple[str, dict[str, Any]]],
    objective: str = '',
    extra_notes: list[str] | None = None,
    invoke_renderer_text: RendererTextInvoker,
    render_tool_result_message: ToolResultMessageRenderer,
    build_grounded_tool_reply: GroundedReplyBuilder,
    merge_grounded_sections: GroundedSectionMerger,
) -> str:
    normalized_results: list[tuple[str, dict[str, Any]]] = []
    for tool_name, parsed in applied_results:
        normalized_name = str(canonical_tool_name(tool_name) or '').strip()
        if not normalized_name:
            continue
        normalized_parsed = parsed if isinstance(parsed, dict) else {'ok': False, 'summary': str(parsed or '').strip()}
        normalized_results.append((normalized_name, normalized_parsed))

    cleaned_notes = [str(note).strip() for note in (extra_notes or []) if str(note).strip()]
    if not normalized_results:
        return merge_grounded_sections(cleaned_notes)
    if len(normalized_results) == 1 and not cleaned_notes:
        tool_name, parsed = normalized_results[0]
        return await render_tool_result_message(tool_name, parsed)
    if planner_llm is None:
        return fallback_multi_tool_result_message(
            normalized_results,
            extra_notes=cleaned_notes,
            build_grounded_tool_reply=build_grounded_tool_reply,
            merge_grounded_sections=merge_grounded_sections,
        )

    facts = {
        'objective': str(objective or '').strip(),
        'results': [tool_result_user_facts(tool_name, parsed) for tool_name, parsed in normalized_results],
        'extra_notes': cleaned_notes[:4],
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的组合结果说明器。'
                '请基于多条已验证的工具结果，用自然中文向用户给出一个连贯结论。'
                '优先回答最终结果，再补关键事实和下一步。'
                '不要输出工具名、字段名、JSON、命令或 payload。'
                '不要补充未验证事实。'
            )
        ),
        HumanMessage(
            content=(
                '请根据以下已验证事实，直接给用户一段自然中文说明：\n'
                f'{json.dumps(facts, ensure_ascii=False, indent=2)}'
            )
        ),
    ]
    text = await invoke_renderer_text(
        messages=messages,
        failure_event='multi_tool_result_render_failed',
        failure_payload={
            'tools': [tool_name for tool_name, _ in normalized_results],
            'objective': str(objective or '').strip(),
        },
    )
    if text:
        return text
    return fallback_multi_tool_result_message(
        normalized_results,
        extra_notes=cleaned_notes,
        build_grounded_tool_reply=build_grounded_tool_reply,
        merge_grounded_sections=merge_grounded_sections,
    )


async def render_tool_result_message(
    *,
    planner_llm: Any,
    tool_name: str,
    parsed: dict[str, Any],
    render_multi_tool_result_message: Callable[..., Awaitable[str]],
    invoke_renderer_text: RendererTextInvoker,
    build_grounded_tool_reply: GroundedReplyBuilder,
    merge_grounded_sections: GroundedSectionMerger,
) -> str:
    remote_pipeline_results = remote_pipeline_applied_results(tool_name, parsed)
    if remote_pipeline_results:
        objective = '远端训练闭环执行结果' if canonical_tool_name(tool_name) == 'remote_training_pipeline' else '远端预测闭环执行结果'
        extra_notes: list[str] = []
        remote_result_path = str(parsed.get('remote_result_path') or '').strip()
        local_result_root = str(parsed.get('local_result_root') or '').strip()
        final_run_state = str(parsed.get('final_run_state') or '').strip()
        if remote_result_path:
            extra_notes.append(f'远端结果目录: {remote_result_path}')
        if local_result_root:
            extra_notes.append(f'本机回传目录: {local_result_root}')
        if final_run_state:
            extra_notes.append(f'最终运行状态: {final_run_state}')
        return await render_multi_tool_result_message(
            applied_results=remote_pipeline_results,
            objective=objective,
            extra_notes=extra_notes or None,
        )
    if canonical_tool_name(tool_name) in GROUNDED_TOOL_RENDER_ONLY:
        return fallback_tool_result_text(
            tool_name,
            parsed,
            build_grounded_tool_reply=build_grounded_tool_reply,
        )
    if planner_llm is None:
        return fallback_tool_result_text(
            tool_name,
            parsed,
            build_grounded_tool_reply=build_grounded_tool_reply,
        )

    facts = tool_result_user_facts(tool_name, parsed)
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的结果说明器。'
                '请基于已验证的工具执行结果，用自然中文向用户说明本次执行结果。'
                '不要输出工具名、字段名、JSON、命令或 payload。'
                '如果成功，先说结果，再补关键事实；如果失败，直接解释失败原因和下一步。'
                '不要补充未验证事实。'
            )
        ),
        HumanMessage(
            content=(
                '请根据以下已验证事实，直接给用户一段自然中文说明：\n'
                f'{json.dumps(facts, ensure_ascii=False, indent=2)}'
            )
        ),
    ]
    text = await invoke_renderer_text(
        messages=messages,
        failure_event='tool_result_render_failed',
        failure_payload={'tool': tool_name, 'ok': bool(parsed.get('ok'))},
    )
    if text:
        return text
    return tool_result_render_error(tool_name, parsed)

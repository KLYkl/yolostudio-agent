from __future__ import annotations

from typing import Any


def _join(lines: list[str]) -> str:
    return '\n'.join(lines)


def _format_tool_action(item: Any) -> str:
    if isinstance(item, dict):
        description = str(item.get('description') or '').strip()
        reason = str(item.get('reason') or '').strip()
        action = str(item.get('action') or '').strip()
        tool_name = str(item.get('tool') or '').strip()
        if description and tool_name:
            return f'{description} ({tool_name})'
        if description:
            return description
        if reason and action:
            return f'{reason} ({action})'
        if reason:
            return reason
        if action and tool_name:
            return f'{action} ({tool_name})'
        if action:
            return action
        if tool_name:
            return tool_name
    return str(item)


def _recommendation_lines(result: dict[str, Any], *, limit: int = 2) -> list[str]:
    action_candidates = result.get('action_candidates') or []
    if isinstance(action_candidates, list) and action_candidates:
        lines = [_format_tool_action(item) for item in action_candidates[:limit]]
        return [item for item in lines if item]
    next_actions = result.get('next_actions') or []
    if isinstance(next_actions, list) and next_actions:
        lines = [_format_tool_action(item) for item in next_actions[:limit]]
        return [item for item in lines if item]
    return []


def _observation_stage_label(stage: Any) -> str:
    mapping = {
        'early': '早期观察',
        'mid': '中段观察',
        'late': '后段观察',
        'final': '最终状态',
    }
    key = str(stage or '').strip().lower()
    return mapping.get(key, key or '未知')


def _format_metric_value(value: Any, *, precision: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f'{float(value):.{precision}f}'
    return str(value)


def _append_training_metric_snapshot(lines: list[str], result: dict[str, Any], *, eval_label: str) -> None:
    latest_metrics = ((result.get('latest_metrics') or {}).get('metrics') or {})
    latest_train_metrics = result.get('latest_train_metrics') or {}
    latest_eval_metrics = result.get('latest_eval_metrics') or {}

    train_metrics = latest_train_metrics or {
        key: latest_metrics.get(key)
        for key in ('gpu_mem', 'box_loss', 'cls_loss', 'dfl_loss')
        if latest_metrics.get(key) is not None
    }
    eval_metrics = latest_eval_metrics or {
        key: latest_metrics.get(key)
        for key in ('precision', 'recall', 'map50', 'map', 'mAP50', 'mAP50-95')
        if latest_metrics.get(key) is not None
    }

    gpu_mem = train_metrics.get('gpu_mem')
    if gpu_mem:
        lines.append(f'GPU 显存: {gpu_mem}')

    if eval_metrics and any(eval_metrics.get(key) is not None for key in ('precision', 'recall', 'map50', 'map', 'mAP50', 'mAP50-95')):
        map50 = eval_metrics.get('map50')
        if map50 is None:
            map50 = eval_metrics.get('mAP50')
        map5095 = eval_metrics.get('map')
        if map5095 is None:
            map5095 = eval_metrics.get('mAP50-95')
        lines.append(
            f'{eval_label}: '
            f"precision={_format_metric_value(eval_metrics.get('precision'))}, "
            f"recall={_format_metric_value(eval_metrics.get('recall'))}, "
            f"mAP50={_format_metric_value(map50)}, "
            f"mAP50-95={_format_metric_value(map5095)}"
        )
    elif train_metrics:
        lines.append('最近评估指标: 暂无（等待验证阶段产出）')

    if train_metrics and any(train_metrics.get(key) is not None for key in ('box_loss', 'cls_loss', 'dfl_loss')):
        lines.append(
            '当前仅有训练损失: '
            f"box={train_metrics.get('box_loss')}, cls={train_metrics.get('cls_loss')}, dfl={train_metrics.get('dfl_loss')}"
        )


def _knowledge_gate_category_label(category: Any) -> str:
    mapping = {
        'hard_stop': '硬停止建议',
        'analysis_review': '分析型建议',
        'continue_observing': '继续观察',
        'other': '一般建议',
    }
    key = str(category or '').strip().lower()
    return mapping.get(key, key or '未知建议')


def _knowledge_gate_outcome_label(outcome: Any) -> str:
    mapping = {
        'awaiting_review': '等待审阅',
        'auto_continue': '自动继续',
        'hard_stop': '直接停止',
        'continue_observing': '继续观察',
        'paused': '暂停待定',
        'stopped': '已停止',
        'other': '一般处理',
    }
    key = str(outcome or '').strip().lower()
    return mapping.get(key, key or '未知结果')


def _humanize_training_readiness_summary(summary: Any) -> str:
    text = str(summary or '训练前检查完成').strip()
    if not text:
        return '训练前检查完成'
    replacements = {
        '但当前数据集可以先进入 prepare_dataset_for_training': '但当前数据集可以先准备数据',
        '可以先进入 prepare_dataset_for_training': '可以先准备数据',
        'prepare_dataset_for_training': '先准备数据',
        'start_training': '开始训练',
        'training_readiness': '训练前检查',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _humanize_dataset_training_readiness_summary(summary: Any) -> str:
    text = str(summary or '数据集可训练性检查完成').strip()
    if not text:
        return '数据集可训练性检查完成'
    replacements = {
        'prepare_dataset_for_training': '先准备数据',
        'data_yaml': 'data.yaml',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def build_grounded_tool_reply(applied_results: list[tuple[str, dict[str, Any]]]) -> str:
    if not applied_results:
        return ""
    tool_name, result = applied_results[-1]
    if not result.get('ok'):
        return ""
    if tool_name == 'run_dataset_health_check':
        lines = [result.get('summary', '健康检查完成')]
        health_overview = result.get('health_overview') or {}
        warnings = result.get('warnings') or []
        if warnings:
            lines.append('关键风险:')
            lines.extend(f'- {warning}' for warning in warnings[:3])
        integrity = result.get('integrity') or {}
        size_stats = result.get('size_stats') or {}
        lines.append(
            '完整性: '
            f'损坏 {integrity.get("corrupted_count", 0)} / 零字节 {integrity.get("zero_bytes_count", 0)} / '
            f'格式不匹配 {integrity.get("format_mismatch_count", 0)}'
        )
        lines.append(
            '尺寸: '
            f'异常小 {size_stats.get("abnormal_small_count", 0)} / 异常大 {size_stats.get("abnormal_large_count", 0)}'
        )
        duplicate_groups = result.get('duplicate_groups', health_overview.get('duplicate_group_count', 0))
        duplicate_extra_files = result.get('duplicate_extra_files', health_overview.get('duplicate_extra_files', 0))
        if duplicate_groups:
            lines.append(
                f'重复图片: {duplicate_groups} 组，额外重复文件 {duplicate_extra_files} 个'
            )
        if result.get('report_path'):
            lines.append(f'报告路径: {result.get("report_path")}')
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'detect_duplicate_images':
        lines = [result.get('summary', '重复检测完成')]
        groups = result.get('groups') or []
        if groups:
            lines.append('示例重复组:')
            for group in groups[:3]:
                sample_paths = ', '.join(group.get('paths', [])[:2])
                lines.append(f'- {sample_paths}')
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'scan_dataset':
        lines = [result.get('summary', '扫描完成')]
        overview = result.get('scan_overview') or {}
        if result.get('warnings'):
            lines.append('风险:')
            lines.extend(f'- {item}' for item in (result.get('warnings') or [])[:2])
        top_classes = result.get('top_classes') or []
        if top_classes:
            preview = '，'.join(f"{item.get('class')}={item.get('count')}" for item in top_classes[:3])
            lines.append(f'主要类别: {preview}')
        elif overview.get('class_count') is not None:
            lines.append(f"类别数: {overview.get('class_count')}")
        if result.get('detected_classes_txt'):
            lines.append(f"类名来源: {result.get('detected_classes_txt')}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'validate_dataset':
        lines = [result.get('summary', '校验完成')]
        overview = result.get('validation_overview') or {}
        warnings = result.get('warnings') or []
        if warnings:
            lines.append('风险:')
            lines.extend(f'- {item}' for item in warnings[:3])
        if overview.get('issue_count') is not None:
            lines.append(f"问题数: {overview.get('issue_count')}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'training_readiness':
        lines = [_humanize_training_readiness_summary(result.get('summary', '训练前检查完成'))]
        overview = result.get('readiness_overview') or {}
        device_overview = result.get('device_overview') or {}
        blockers = result.get('blockers') or []
        warnings = result.get('warnings') or []
        if blockers:
            lines.append('阻塞项:')
            lines.extend(f'- {item}' for item in blockers[:2])
        elif warnings:
            lines.append('风险:')
            lines.extend(f'- {item}' for item in warnings[:2])
        if result.get('preparable') or overview.get('preparable'):
            lines.append('下一步: 可以先准备数据，补齐 data.yaml 和划分产物')
        if result.get('resolved_data_yaml'):
            lines.append(f"当前可用 YAML: {result.get('resolved_data_yaml')}")
        elif overview.get('data_yaml_source'):
            lines.append(f"YAML 来源: {overview.get('data_yaml_source')}")
        if device_overview.get('auto_device') is not None:
            lines.append(f"自动设备: {device_overview.get('auto_device')}")
        return _join(lines)
    if tool_name == 'dataset_training_readiness':
        lines = [_humanize_dataset_training_readiness_summary(result.get('summary', '数据集可训练性检查完成'))]
        overview = result.get('readiness_overview') or {}
        blockers = result.get('blockers') or []
        warnings = result.get('warnings') or []
        if blockers:
            lines.append('当前问题:')
            lines.extend(f'- {item}' for item in blockers[:3])
        elif warnings:
            lines.append('风险:')
            lines.extend(f'- {item}' for item in warnings[:2])
        next_step_summary = str(result.get('next_step_summary') or '').strip()
        if next_step_summary:
            lines.append(f'下一步: {next_step_summary}')
        elif overview.get('preparable'):
            lines.append('下一步: 可以先准备数据，补齐 data.yaml 和划分产物。')
        if result.get('resolved_data_yaml'):
            lines.append(f"当前可用 YAML: {result.get('resolved_data_yaml')}")
        return _join(lines)
    if tool_name == 'list_training_environments':
        lines = [result.get('summary', '训练环境查询完成')]
        environment_overview = result.get('environment_overview') or {}
        environments = result.get('environments') or []
        if environments:
            default_environment = result.get('default_environment') or environments[0]
            lines.append(f"默认训练环境: {default_environment.get('display_name') or default_environment.get('name')}")
            lines.append('可用环境:')
            for env in environments[:3]:
                label = env.get('display_name') or env.get('name')
                suffix = ' (默认)' if env.get('selected_by_default') else ''
                lines.append(f'- {label}{suffix}')
        elif environment_overview.get('default_environment_name'):
            lines.append(f"默认训练环境: {environment_overview.get('default_environment_name')}")
        if environment_overview.get('environment_count') is not None:
            lines.append(f"环境数量: {environment_overview.get('environment_count')}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'training_preflight':
        lines = [result.get('summary', '训练预检完成')]
        preflight_overview = result.get('preflight_overview') or {}
        if result.get('training_environment'):
            env = result.get('training_environment') or {}
            lines.append(f"训练环境: {env.get('display_name') or env.get('name')}")
        elif preflight_overview.get('training_environment_name'):
            lines.append(f"训练环境: {preflight_overview.get('training_environment_name')}")
        resolved_args = result.get('resolved_args') or {}
        if resolved_args.get('model'):
            lines.append(f"模型: {resolved_args.get('model')}")
        elif preflight_overview.get('model'):
            lines.append(f"模型: {preflight_overview.get('model')}")
        if resolved_args.get('data_yaml'):
            lines.append(f"数据 YAML: {resolved_args.get('data_yaml')}")
        elif preflight_overview.get('data_yaml'):
            lines.append(f"数据 YAML: {preflight_overview.get('data_yaml')}")
        output_bits: list[str] = []
        if resolved_args.get('project'):
            output_bits.append(f"project={resolved_args.get('project')}")
        if resolved_args.get('name'):
            output_bits.append(f"name={resolved_args.get('name')}")
        if output_bits:
            lines.append(f"输出组织: {', '.join(output_bits)}")
        if resolved_args.get('batch') is not None:
            lines.append(f"批大小: {resolved_args.get('batch')}")
        elif preflight_overview.get('batch') is not None:
            lines.append(f"批大小: {preflight_overview.get('batch')}")
        if resolved_args.get('imgsz') is not None:
            lines.append(f"输入尺寸: {resolved_args.get('imgsz')}")
        elif preflight_overview.get('imgsz') is not None:
            lines.append(f"输入尺寸: {preflight_overview.get('imgsz')}")
        if resolved_args.get('fraction') is not None:
            lines.append(f"采样比例: {resolved_args.get('fraction')}")
        elif preflight_overview.get('fraction') is not None:
            lines.append(f"采样比例: {preflight_overview.get('fraction')}")
        advanced_bits: list[str] = []
        if resolved_args.get('classes') is not None:
            advanced_bits.append(f"classes={resolved_args.get('classes')}")
        if resolved_args.get('single_cls') is not None:
            advanced_bits.append(f"single_cls={resolved_args.get('single_cls')}")
        if resolved_args.get('optimizer'):
            advanced_bits.append(f"optimizer={resolved_args.get('optimizer')}")
        if resolved_args.get('freeze') is not None:
            advanced_bits.append(f"freeze={resolved_args.get('freeze')}")
        if resolved_args.get('resume') is not None:
            advanced_bits.append(f"resume={resolved_args.get('resume')}")
        for key, label in (('lr0', 'lr0'), ('patience', 'patience'), ('workers', 'workers'), ('amp', 'amp')):
            value = resolved_args.get(key)
            if value is not None:
                advanced_bits.append(f"{label}={value}")
        if advanced_bits:
            lines.append(f"高级参数: {', '.join(advanced_bits)}")
        blockers = result.get('blockers') or []
        if blockers:
            lines.append('阻塞项:')
            lines.extend(f'- {item}' for item in blockers[:3])
        command_preview = result.get('command_preview') or []
        if command_preview:
            lines.append('命令预览:')
            lines.append(f"- {' '.join(str(item) for item in command_preview)}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'start_training':
        lines = [result.get('summary', '训练已启动')]
        resolved_args = result.get('resolved_args') or {}
        output_bits: list[str] = []
        if resolved_args.get('project'):
            output_bits.append(f"project={resolved_args.get('project')}")
        if resolved_args.get('name'):
            output_bits.append(f"name={resolved_args.get('name')}")
        if output_bits:
            lines.append(f"输出组织: {', '.join(output_bits)}")
        if result.get('pid') is not None:
            lines.append(f"进程 PID: {result.get('pid')}")
        if result.get('log_file'):
            lines.append(f"日志文件: {result.get('log_file')}")
        return _join(lines)
    if tool_name == 'start_training_loop':
        lines = [result.get('summary', '环训练已启动')]
        if result.get('loop_name'):
            lines.append(f"环训练名称: {result.get('loop_name')}")
        if result.get('loop_id'):
            lines.append(f"Loop ID: {result.get('loop_id')}")
        if result.get('managed_level'):
            lines.append(f"托管级别: {result.get('managed_level')}")
        boundaries = result.get('boundaries') or {}
        if boundaries.get('max_rounds') is not None:
            lines.append(f"最大轮数: {boundaries.get('max_rounds')}")
        if boundaries.get('target_metric'):
            target_line = f"目标指标: {boundaries.get('target_metric')}"
            if boundaries.get('target_metric_value') is not None:
                target_line += f" >= {boundaries.get('target_metric_value')}"
            lines.append(target_line)
        next_plan = result.get('next_round_plan') or {}
        if next_plan.get('round_index') is not None:
            lines.append(f"下一轮计划: 第 {next_plan.get('round_index')} 轮")
        if next_plan.get('change_set'):
            lines.append('计划变更:')
            for item in list(next_plan.get('change_set') or [])[:3]:
                lines.append(f"- {item.get('field')}: {item.get('old')} -> {item.get('new')}")
        return _join(lines)
    if tool_name == 'list_training_runs':
        lines = [result.get('summary', '训练历史查询完成')]
        applied_filters = result.get('applied_filters') or {}
        filter_bits: list[str] = []
        if applied_filters.get('run_state'):
            filter_bits.append(f"状态={applied_filters.get('run_state')}")
        if applied_filters.get('analysis_ready') is True:
            filter_bits.append('仅可分析训练')
        elif applied_filters.get('analysis_ready') is False:
            filter_bits.append('仅未具备分析条件')
        if filter_bits:
            lines.append(f"筛选: {', '.join(str(item) for item in filter_bits)}")
        runs = result.get('runs') or []
        if runs:
            lines.append('最近训练:')
            for run in runs[:3]:
                progress = run.get('progress') or {}
                progress_text = ''
                if progress.get('epoch') is not None and progress.get('total_epochs') is not None:
                    progress_text = f"，进度 {progress.get('epoch')}/{progress.get('total_epochs')}"
                lines.append(
                    f"- {run.get('run_id')}: {run.get('run_state')} / {_observation_stage_label(run.get('observation_stage'))}{progress_text}"
                )
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'list_training_loops':
        lines = [result.get('summary', '环训练列表已就绪')]
        loops = result.get('loops') or []
        if loops:
            lines.append('最近环训练:')
            for loop in loops[:3]:
                best_metric = loop.get('best_target_metric')
                metric_suffix = f"，最佳指标 {best_metric:.4f}" if isinstance(best_metric, (int, float)) else ''
                active_suffix = ' (当前活动)' if loop.get('active') else ''
                lines.append(
                    f"- {loop.get('loop_name') or loop.get('loop_id')}: {loop.get('status')}{metric_suffix}{active_suffix}"
                )
        return _join(lines)
    if tool_name == 'inspect_training_run':
        lines = [result.get('summary', '训练记录详情已就绪')]
        if result.get('selected_run_id'):
            lines.append(f"训练记录: {result.get('selected_run_id')}")
        if result.get('run_state'):
            lines.append(f"状态: {result.get('run_state')}")
        if result.get('observation_stage'):
            lines.append(f"观察阶段: {_observation_stage_label(result.get('observation_stage'))}")
        progress = result.get('progress') or {}
        if progress.get('epoch') is not None and progress.get('total_epochs') is not None:
            ratio = progress.get('progress_ratio')
            suffix = f" ({round(float(ratio) * 100)}%)" if isinstance(ratio, (int, float)) else ''
            lines.append(f"训练进度: {progress.get('epoch')}/{progress.get('total_epochs')}{suffix}")
        if result.get('model'):
            lines.append(f"模型: {result.get('model')}")
        if result.get('data_yaml'):
            lines.append(f"数据 YAML: {result.get('data_yaml')}")
        if result.get('log_file'):
            lines.append(f"日志: {result.get('log_file')}")
        signals = result.get('signals') or []
        if signals:
            lines.append(f"信号: {', '.join(str(item) for item in signals[:4])}")
        facts = result.get('facts') or []
        if facts:
            lines.append('事实:')
            lines.extend(f'- {item}' for item in facts[:3])
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name in {'check_training_loop_status', 'inspect_training_loop'}:
        lines = [result.get('summary', '环训练状态已就绪')]
        if result.get('loop_name'):
            lines.append(f"环训练: {result.get('loop_name')}")
        if result.get('status'):
            lines.append(f"状态: {result.get('status')}")
        if result.get('current_round_index') is not None and result.get('max_rounds') is not None:
            lines.append(f"轮次: {result.get('current_round_index')}/{result.get('max_rounds')}")
        if result.get('best_round_index') is not None:
            best_metric = result.get('best_target_metric')
            suffix = f"，指标 {best_metric:.4f}" if isinstance(best_metric, (int, float)) else ''
            lines.append(f"当前最佳轮: 第 {result.get('best_round_index')} 轮{suffix}")
        gate_status = result.get('knowledge_gate_status') or {}
        if gate_status:
            lines.append(
                f"闸门结论: {_knowledge_gate_outcome_label(gate_status.get('outcome'))}"
                f" / {gate_status.get('action_label') or gate_status.get('action') or '未知动作'}"
            )
            if gate_status.get('summary'):
                lines.append(f"结论说明: {gate_status.get('summary')}")
            matched_rule_ids = list(gate_status.get('matched_rule_ids') or [])
            if matched_rule_ids:
                lines.append(f"命中规则: {', '.join(str(item) for item in matched_rule_ids[:3])}")
        latest_card = result.get('latest_round_card') or result.get('current_round_card') or {}
        if latest_card:
            lines.append(f"最近轮次卡: 第 {latest_card.get('round_index')} 轮 / {latest_card.get('status')}")
            knowledge_gate = latest_card.get('knowledge_gate') or {}
            if knowledge_gate:
                gate_label = knowledge_gate.get('outcome_label') or _knowledge_gate_outcome_label(knowledge_gate.get('outcome'))
                action = knowledge_gate.get('action_label') or knowledge_gate.get('action') or 'unknown'
                lines.append(f"知识闸门: {gate_label} / {action}")
                if knowledge_gate.get('decision_reason'):
                    lines.append(f"闸门理由: {knowledge_gate.get('decision_reason')}")
                if knowledge_gate.get('user_summary'):
                    lines.append(f"轮次结论: {knowledge_gate.get('user_summary')}")
            vs_previous = (latest_card.get('vs_previous') or {}).get('highlights') or []
            if vs_previous:
                lines.append('对比上一轮:')
                lines.extend(f'- {item}' for item in vs_previous[:2])
            next_plan = latest_card.get('next_plan') or {}
            if next_plan.get('change_set'):
                lines.append('下一轮计划:')
                for item in list(next_plan.get('change_set') or [])[:3]:
                    lines.append(f"- {item.get('field')}: {item.get('old')} -> {item.get('new')}")
        final_summary = result.get('final_summary') or {}
        if final_summary:
            lines.append(f"停止原因: {final_summary.get('termination_detail') or final_summary.get('stop_reason')}")
            gate_overview = final_summary.get('knowledge_gate_overview') or {}
            if gate_overview:
                lines.append(
                    f"闸门总览: 共 {gate_overview.get('count')} 次 / 最后一次为 "
                    f"{gate_overview.get('last_outcome_label') or _knowledge_gate_outcome_label(gate_overview.get('last_outcome'))}"
                )
                if gate_overview.get('last_summary'):
                    lines.append(f"最后闸门说明: {gate_overview.get('last_summary')}")
            last_knowledge_gate = final_summary.get('last_knowledge_gate') or {}
            if last_knowledge_gate:
                gate_label = last_knowledge_gate.get('outcome_label') or _knowledge_gate_outcome_label(last_knowledge_gate.get('outcome'))
                action = last_knowledge_gate.get('action_label') or last_knowledge_gate.get('action') or 'unknown'
                lines.append(f"最后知识结论: 第 {last_knowledge_gate.get('round_index')} 轮 / {gate_label} / {action}")
            if final_summary.get('best_model_path'):
                lines.append(f"最佳模型目录: {final_summary.get('best_model_path')}")
        return _join(lines)
    if tool_name == 'compare_training_runs':
        lines = [result.get('summary', '训练记录对比已完成')]
        if result.get('left_run_id') and result.get('right_run_id'):
            lines.append(f"对比对象: {result.get('left_run_id')} vs {result.get('right_run_id')}")
        highlights = result.get('highlights') or []
        if highlights:
            lines.append('主要变化:')
            lines.extend(f'- {item}' for item in highlights[:4])
        metric_deltas = result.get('metric_deltas') or {}
        preferred = []
        for key, label in (('precision', 'precision'), ('recall', 'recall'), ('map50', 'mAP50'), ('map', 'mAP50-95')):
            item = metric_deltas.get(key)
            if isinstance(item, dict) and isinstance(item.get('delta'), (int, float)):
                preferred.append(f"{label}={item.get('delta'):+.4f}")
        if preferred:
            lines.append('关键差异: ' + '，'.join(preferred))
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'pause_training_loop':
        lines = [result.get('summary', '环训练已暂停')]
        if result.get('loop_id'):
            lines.append(f"Loop ID: {result.get('loop_id')}")
        if result.get('status'):
            lines.append(f"当前状态: {result.get('status')}")
        return _join(lines)
    if tool_name == 'resume_training_loop':
        lines = [result.get('summary', '环训练已恢复')]
        if result.get('loop_id'):
            lines.append(f"Loop ID: {result.get('loop_id')}")
        if result.get('status'):
            lines.append(f"当前状态: {result.get('status')}")
        return _join(lines)
    if tool_name == 'stop_training_loop':
        lines = [result.get('summary', '环训练已停止')]
        if result.get('loop_id'):
            lines.append(f"Loop ID: {result.get('loop_id')}")
        if result.get('status'):
            lines.append(f"当前状态: {result.get('status')}")
        return _join(lines)
    if tool_name == 'select_best_training_run':
        lines = [result.get('summary', '最佳训练记录已选出')]
        if result.get('best_run_id'):
            lines.append(f"最佳训练: {result.get('best_run_id')}")
        if result.get('ranking_basis'):
            lines.append(f"选择依据: {result.get('ranking_basis')}")
        candidates = result.get('candidates') or []
        if candidates:
            lines.append('候选记录:')
            for item in candidates[:3]:
                lines.append(f"- {item.get('run_id')}: {item.get('run_state')}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'prepare_dataset_for_training':
        lines = [result.get('summary', '数据准备完成')]
        overview = result.get('prepare_overview') or {}
        data_yaml = result.get('data_yaml') or overview.get('data_yaml')
        if data_yaml:
            lines.append(f"已准备好的 YAML: {data_yaml}")
        if result.get('warnings'):
            lines.append('风险:')
            lines.extend(f'- {item}' for item in (result.get('warnings') or [])[:2])
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'preview_extract_images':
        lines = [result.get('summary', '图片抽取预览完成')]
        overview = result.get('extract_preview_overview') or {}
        available_images = result.get('available_images')
        if available_images is None:
            available_images = overview.get('available_images', 0)
        planned_extract_count = result.get('planned_extract_count')
        if planned_extract_count is None:
            planned_extract_count = overview.get('planned_extract_count', 0)
        lines.append(f"统计: 可用 {available_images or 0} 张 / 计划抽取 {planned_extract_count or 0} 张")
        if result.get('selected_dirs'):
            lines.append(f"目录过滤: {', '.join(result.get('selected_dirs', [])[:4])}")
        if result.get('sample_images'):
            lines.append('样例图片:')
            lines.extend(f'- {item}' for item in (result.get('sample_images') or [])[:2])
        output_dir = result.get('output_dir') or overview.get('output_dir')
        if output_dir:
            lines.append(f"计划输出目录: {output_dir}")
        workflow_ready_path = result.get('workflow_ready_path') or overview.get('workflow_ready_path')
        if workflow_ready_path:
            lines.append(f"后续可复用目录: {workflow_ready_path}")
        if result.get('warnings'):
            lines.append('提示:')
            lines.extend(f'- {item}' for item in (result.get('warnings') or [])[:2])
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'extract_images':
        lines = [result.get('summary', '图片抽取完成')]
        overview = result.get('extract_overview') or {}
        extracted = result.get('extracted')
        if extracted is None:
            extracted = overview.get('extracted', 0)
        labels_copied = result.get('labels_copied')
        if labels_copied is None:
            labels_copied = overview.get('labels_copied', 0)
        conflict_count = result.get('conflict_count')
        if conflict_count is None:
            conflict_count = overview.get('conflict_count', 0)
        lines.append(
            f"统计: 已抽取 {extracted or 0} 张 / 复制标签 {labels_copied or 0} / 冲突 {conflict_count or 0}"
        )
        if result.get('sample_images'):
            lines.append('抽取样例:')
            lines.extend(f'- {item}' for item in (result.get('sample_images') or [])[:2])
        output_dir = result.get('output_dir') or overview.get('output_dir')
        if output_dir:
            lines.append(f"输出目录: {output_dir}")
        workflow_ready_path = result.get('workflow_ready_path') or overview.get('workflow_ready_path')
        if workflow_ready_path:
            lines.append(f"可继续接主链的目录: {workflow_ready_path}")
        if result.get('warnings'):
            lines.append('提示:')
            lines.extend(f'- {item}' for item in (result.get('warnings') or [])[:2])
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'scan_videos':
        lines = [result.get('summary', '视频扫描完成')]
        overview = result.get('video_scan_overview') or {}
        total_videos = result.get('total_videos')
        if total_videos is None:
            total_videos = overview.get('total_videos', 0)
        lines.append(f"统计: 发现 {total_videos or 0} 个视频")
        if result.get('sample_videos'):
            lines.append('视频样例:')
            lines.extend(f'- {item}' for item in (result.get('sample_videos') or [])[:2])
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'extract_video_frames':
        lines = [result.get('summary', '视频抽帧完成')]
        overview = result.get('frame_extract_overview') or {}
        total_frames = result.get('total_frames')
        if total_frames is None:
            total_frames = overview.get('total_frames', 0)
        extracted = result.get('extracted')
        if extracted is None:
            extracted = overview.get('extracted', 0)
        final_count = result.get('final_count')
        if final_count is None:
            final_count = overview.get('final_count', 0)
        lines.append(
            f"统计: 总帧数 {total_frames or 0} / 原始抽取 {extracted or 0} / 最终保留 {final_count or 0}"
        )
        output_dir = result.get('output_dir') or overview.get('output_dir')
        if output_dir:
            lines.append(f"输出目录: {output_dir}")
        if result.get('warnings'):
            lines.append('提示:')
            lines.extend(f'- {item}' for item in (result.get('warnings') or [])[:2])
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'summarize_training_run':
        lines = [result.get('summary', '训练结果汇总完成')]
        overview = result.get('summary_overview') or {}
        run_state = result.get('run_state') or overview.get('run_state')
        if run_state:
            lines.append(f"运行状态: {run_state}")
        save_dir = result.get('save_dir') or overview.get('save_dir')
        if save_dir:
            lines.append(f"结果目录: {save_dir}")
        observation_stage = result.get('observation_stage') or overview.get('observation_stage')
        if observation_stage:
            lines.append(f"观察阶段: {_observation_stage_label(observation_stage)}")
        progress = result.get('progress') or {}
        if progress.get('epoch') is not None and progress.get('total_epochs') is not None:
            progress_ratio = progress.get('progress_ratio')
            ratio_text = f" ({progress_ratio:.0%})" if isinstance(progress_ratio, (int, float)) else ''
            lines.append(f"训练进度: {progress.get('epoch')}/{progress.get('total_epochs')}{ratio_text}")
        elif overview.get('epoch') is not None and overview.get('total_epochs') is not None:
            lines.append(f"训练进度: {overview.get('epoch')}/{overview.get('total_epochs')}")
        _append_training_metric_snapshot(lines, result, eval_label='关键指标')
        if not result.get('analysis_ready'):
            shortages: list[str] = []
            signals = [str(item) for item in (result.get('signals') or [])]
            if any(item in signals for item in ('metrics_missing', 'missing_eval_metrics', 'insufficient_eval_metrics', 'loss_only_metrics')):
                shortages.append('缺少稳定评估指标')
            if not result.get('minimum_facts_ready'):
                shortages.append('缺少可分析日志事实')
            if shortages:
                lines.append('当前不足: ' + '；'.join(shortages))
        elif str(result.get('observation_stage') or '') == 'early':
            lines.append('提示: 已有阶段性指标，但当前仍属早期观察，不能当成最终结论')
        facts = result.get('facts') or []
        if facts:
            lines.append('事实:')
            lines.extend(f'- {item}' for item in facts[:4])
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'stop_training':
        lines = [result.get('summary', '训练已停止')]
        run_state = result.get('run_state')
        if run_state:
            lines.append(f"运行状态: {run_state}")
        return_code = result.get('return_code')
        if return_code is not None:
            lines.append(f"返回码: {return_code}")
        pid = result.get('pid')
        if pid is not None:
            lines.append(f"进程 ID: {pid}")
        suggestions = _recommendation_lines(result, limit=2)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'check_training_status':
        lines = [result.get('summary', '训练状态已更新')]
        overview = result.get('status_overview') or {}
        run_state = result.get('run_state') or overview.get('run_state')
        if run_state:
            lines.append(f"运行状态: {run_state}")
        observation_stage = result.get('observation_stage') or overview.get('observation_stage')
        if observation_stage:
            lines.append(f"观察阶段: {_observation_stage_label(observation_stage)}")
        progress = result.get('progress') or {}
        if progress.get('epoch') is not None and progress.get('total_epochs') is not None:
            progress_ratio = progress.get('progress_ratio')
            ratio_text = f" ({progress_ratio:.0%})" if isinstance(progress_ratio, (int, float)) else ''
            lines.append(f"最近进度: {progress.get('epoch')}/{progress.get('total_epochs')}{ratio_text}")
        elif overview.get('epoch') is not None and overview.get('total_epochs') is not None:
            lines.append(f"最近进度: {overview.get('epoch')}/{overview.get('total_epochs')}")
        _append_training_metric_snapshot(lines, result, eval_label='最近指标')
        if not result.get('analysis_ready'):
            shortages: list[str] = []
            signals = [str(item) for item in (result.get('signals') or [])]
            if any(item in signals for item in ('metrics_missing', 'missing_eval_metrics', 'insufficient_eval_metrics', 'loss_only_metrics')):
                shortages.append('缺少稳定评估指标')
            if not result.get('minimum_facts_ready'):
                shortages.append('缺少可分析日志事实')
            if shortages:
                lines.append('当前不足: ' + '；'.join(shortages))
        elif str(result.get('observation_stage') or '') == 'early':
            lines.append('提示: 已有阶段性指标，但当前仍属早期观察，不能当成最终结论')
        suggestions = _recommendation_lines(result, limit=1)
        if suggestions:
            lines.append(f"下一步: {suggestions[0]}")
        save_dir = result.get('save_dir') or overview.get('save_dir')
        if save_dir:
            lines.append(f"结果目录: {save_dir}")
        return _join(lines)
    if tool_name == 'predict_images':
        if result.get('started_in_background'):
            lines = [result.get('summary', '后台图片预测已启动')]
            if result.get('session_id'):
                lines.append(f"会话 ID: {result.get('session_id')}")
            if result.get('total_images') is not None:
                lines.append(f"总图片数: {result.get('total_images')}")
            if result.get('output_dir'):
                lines.append(f"输出目录: {result.get('output_dir')}")
            suggestions = _recommendation_lines(result, limit=3)
            if suggestions:
                lines.append('建议:')
                lines.extend(f'- {item}' for item in suggestions)
            return _join(lines)
        lines = [result.get('summary', '预测完成')]
        overview = result.get('prediction_overview') or {}
        processed_images = result.get('processed_images')
        if processed_images is None:
            processed_images = overview.get('processed_images', 0)
        detected_images = result.get('detected_images')
        if detected_images is None:
            detected_images = overview.get('detected_images', 0)
        empty_images = result.get('empty_images')
        if empty_images is None:
            empty_images = overview.get('empty_images', 0)
        lines.append(
            f"统计: 已处理 {processed_images or 0} 张 / 有检测 {detected_images or 0} / 无检测 {empty_images or 0}"
        )
        class_counts = result.get('class_counts') or {}
        if class_counts:
            preview = '，'.join(f"{k}={v}" for k, v in list(class_counts.items())[:4])
            lines.append(f'主要类别: {preview}')
        detected_samples = result.get('detected_samples') or []
        if detected_samples:
            lines.append('有检测样例:')
            lines.extend(f'- {item}' for item in detected_samples[:2])
        empty_samples = result.get('empty_samples') or []
        if empty_samples:
            lines.append('无检测样例:')
            lines.extend(f'- {item}' for item in empty_samples[:2])
        warnings = result.get('warnings') or []
        if warnings:
            lines.append('提示:')
            lines.extend(f'- {item}' for item in warnings[:2])
        annotated_dir = result.get('annotated_dir') or overview.get('annotated_dir')
        if annotated_dir:
            lines.append(f"标注结果目录: {annotated_dir}")
        report_path = result.get('report_path') or overview.get('report_path')
        if report_path:
            lines.append(f"预测报告: {report_path}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name in {'start_image_prediction', 'check_image_prediction_status', 'stop_image_prediction'}:
        lines = [result.get('summary', '后台图片预测状态已更新')]
        if result.get('session_id'):
            lines.append(f"会话 ID: {result.get('session_id')}")
        if result.get('status'):
            lines.append(f"状态: {result.get('status')}")
        total_images = result.get('total_images', 0)
        processed_images = result.get('processed_images', 0)
        detected_images = result.get('detected_images', 0)
        empty_images = result.get('empty_images', 0)
        lines.append(
            f"统计: 已处理 {processed_images}/{total_images} 张图片 / 有检测 {detected_images} / 无检测 {empty_images}"
        )
        class_counts = result.get('class_counts') or {}
        if class_counts:
            preview = '，'.join(f"{k}={v}" for k, v in list(class_counts.items())[:4])
            lines.append(f'主要类别: {preview}')
        if result.get('report_path'):
            lines.append(f"预测报告: {result.get('report_path')}")
        if result.get('error'):
            lines.append(f"异常: {result.get('error')}")
        suggestions = _recommendation_lines(result, limit=3)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'predict_videos':
        lines = [result.get('summary', '视频预测完成')]
        overview = result.get('prediction_overview') or {}
        processed_videos = result.get('processed_videos')
        if processed_videos is None:
            processed_videos = overview.get('processed_videos', 0)
        total_frames = result.get('total_frames')
        if total_frames is None:
            total_frames = overview.get('total_frames', 0)
        detected_frames = result.get('detected_frames')
        if detected_frames is None:
            detected_frames = overview.get('detected_frames', 0)
        total_detections = result.get('total_detections')
        if total_detections is None:
            total_detections = overview.get('total_detections', 0)
        lines.append(
            f"统计: 已处理 {processed_videos or 0} 个视频 / 总帧数 {total_frames or 0} / 有检测帧 {detected_frames or 0} / 总检测框 {total_detections or 0}"
        )
        class_counts = result.get('class_counts') or {}
        if class_counts:
            preview = '，'.join(f"{k}={v}" for k, v in list(class_counts.items())[:4])
            lines.append(f'主要类别: {preview}')
        detected_samples = result.get('detected_samples') or []
        if detected_samples:
            lines.append('有检测视频样例:')
            lines.extend(f'- {item}' for item in detected_samples[:2])
        empty_samples = result.get('empty_samples') or []
        if empty_samples:
            lines.append('无检测视频样例:')
            lines.extend(f'- {item}' for item in empty_samples[:2])
        warnings = result.get('warnings') or []
        if warnings:
            lines.append('提示:')
            lines.extend(f'- {item}' for item in warnings[:2])
        output_dir = result.get('output_dir') or overview.get('output_dir')
        if output_dir:
            lines.append(f"视频预测输出目录: {output_dir}")
        report_path = result.get('report_path') or overview.get('report_path')
        if report_path:
            lines.append(f"预测报告: {report_path}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'summarize_prediction_results':
        lines = [result.get('summary', '预测结果汇总完成')]
        overview = result.get('prediction_summary_overview') or result.get('prediction_overview') or {}
        mode = result.get('mode') or overview.get('mode')
        if mode == 'videos':
            processed_videos = result.get('processed_videos')
            if processed_videos is None:
                processed_videos = overview.get('processed_videos', 0)
            total_frames = result.get('total_frames')
            if total_frames is None:
                total_frames = overview.get('total_frames', 0)
            detected_frames = result.get('detected_frames')
            if detected_frames is None:
                detected_frames = overview.get('detected_frames', 0)
            total_detections = result.get('total_detections')
            if total_detections is None:
                total_detections = overview.get('total_detections', 0)
            lines.append(
                f"统计: 已处理 {processed_videos or 0} 个视频 / 总帧数 {total_frames or 0} / 有检测帧 {detected_frames or 0} / 总检测框 {total_detections or 0}"
            )
        else:
            processed_images = result.get('processed_images')
            if processed_images is None:
                processed_images = overview.get('processed_images', 0)
            detected_images = result.get('detected_images')
            if detected_images is None:
                detected_images = overview.get('detected_images', 0)
            empty_images = result.get('empty_images')
            if empty_images is None:
                empty_images = overview.get('empty_images', 0)
            total_detections = result.get('total_detections')
            if total_detections is None:
                total_detections = overview.get('total_detections', 0)
            lines.append(
                f"统计: 已处理 {processed_images or 0} 张 / 有检测 {detected_images or 0} / 无检测 {empty_images or 0} / 总检测框 {total_detections or 0}"
            )
        class_counts = result.get('class_counts') or {}
        if class_counts:
            preview = '，'.join(f"{k}={v}" for k, v in list(class_counts.items())[:4])
            lines.append(f'主要类别: {preview}')
        detected_samples = result.get('detected_samples') or []
        if detected_samples:
            lines.append('有检测样例:')
            lines.extend(f'- {item}' for item in detected_samples[:2])
        empty_samples = result.get('empty_samples') or []
        if empty_samples:
            lines.append('无检测样例:')
            lines.extend(f'- {item}' for item in empty_samples[:2])
        warnings = result.get('warnings') or []
        if warnings:
            lines.append('提示:')
            lines.extend(f'- {item}' for item in warnings[:2])
        annotated_dir = result.get('annotated_dir') or overview.get('annotated_dir')
        if annotated_dir:
            lines.append(f"标注结果目录: {annotated_dir}")
        report_path = result.get('report_path') or overview.get('report_path')
        if report_path:
            lines.append(f"预测报告: {report_path}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'inspect_prediction_outputs':
        lines = [result.get('summary', '预测输出检查完成')]
        overview = result.get('prediction_output_overview') or {}
        output_dir = result.get('output_dir') or overview.get('output_dir')
        if output_dir:
            lines.append(f"输出目录: {output_dir}")
        report_path = result.get('report_path') or overview.get('report_path')
        if report_path:
            lines.append(f"预测报告: {report_path}")
        artifact_roots = result.get('artifact_roots') or []
        if artifact_roots:
            lines.append('主要产物路径:')
            lines.extend(f'- {item}' for item in artifact_roots[:3])
        path_list_files = result.get('path_list_files') or {}
        if path_list_files:
            lines.append('已有路径清单:')
            for name, path in list(path_list_files.items())[:3]:
                lines.append(f'- {name}: {path}')
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'export_prediction_report':
        lines = [result.get('summary', '预测报告导出完成')]
        overview = result.get('export_overview') or {}
        export_format = result.get('export_format') or overview.get('export_format')
        if export_format:
            lines.append(f"导出格式: {export_format}")
        export_path = result.get('export_path') or overview.get('export_path')
        if export_path:
            lines.append(f"导出路径: {export_path}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'export_prediction_path_lists':
        lines = [result.get('summary', '预测路径清单导出完成')]
        overview = result.get('path_list_overview') or result.get('export_overview') or {}
        detected_count = result.get('detected_count')
        if detected_count is None:
            detected_count = overview.get('detected_count', 0)
        empty_count = result.get('empty_count')
        if empty_count is None:
            empty_count = overview.get('empty_count', 0)
        failed_count = result.get('failed_count')
        if failed_count is None:
            failed_count = overview.get('failed_count', 0)
        lines.append(
            f"统计: 命中 {detected_count or 0} / 无命中 {empty_count or 0} / 失败 {failed_count or 0}"
        )
        export_dir = result.get('export_dir') or overview.get('export_dir')
        if export_dir:
            lines.append(f"清单目录: {export_dir}")
        for key, label in (
            ('detected_items_path', '命中清单'),
            ('empty_items_path', '无命中清单'),
            ('failed_items_path', '失败清单'),
        ):
            value = result.get(key) or overview.get(key)
            if value:
                lines.append(f"{label}: {value}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'organize_prediction_results':
        lines = [result.get('summary', '预测结果整理完成')]
        overview = result.get('organization_overview') or {}
        destination_dir = result.get('destination_dir') or overview.get('destination_dir')
        if destination_dir:
            lines.append(f"整理目录: {destination_dir}")
        organize_by = result.get('organize_by') or overview.get('organize_by')
        if organize_by:
            lines.append(f"整理方式: {organize_by}")
        bucket_stats = result.get('bucket_stats') or {}
        if bucket_stats:
            lines.append('目录桶统计:')
            for bucket, count in list(bucket_stats.items())[:5]:
                lines.append(f'- {bucket}: {count}')
        sample_outputs = result.get('sample_outputs') or []
        if sample_outputs:
            lines.append('样例产物:')
            lines.extend(f'- {item}' for item in sample_outputs[:3])
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'list_remote_profiles':
        lines = [result.get('summary', '远端配置查询完成')]
        overview = result.get('profile_overview') or {}
        profiles = result.get('profiles') or []
        if profiles:
            lines.append('远端 profile:')
            for item in profiles[:5]:
                suffix = ' (默认)' if item.get('is_default') else ''
                remote_root = f" / {item.get('remote_root')}" if item.get('remote_root') else ''
                lines.append(f"- {item.get('name')}{suffix}{remote_root}")
        aliases = result.get('ssh_aliases') or []
        if aliases:
            lines.append('SSH alias:')
            for item in aliases[:5]:
                label = item.get('name') or ''
                host = item.get('hostname') or ''
                port = item.get('port') or ''
                extra = f" -> {host}" if host else ''
                if port:
                    extra += f":{port}"
                lines.append(f"- {label}{extra}")
        profiles_path = result.get('profiles_path') or overview.get('profiles_path')
        if profiles_path:
            lines.append(f"配置文件: {profiles_path}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'upload_assets_to_remote':
        lines = [result.get('summary', '远端上传完成')]
        overview = result.get('transfer_overview') or {}
        profile_name = result.get('profile_name') or overview.get('profile_name')
        if profile_name:
            lines.append(f"远端 profile: {profile_name}")
        target_label = result.get('target_label') or overview.get('target_label')
        if target_label:
            lines.append(f"目标: {target_label}")
        remote_root = result.get('remote_root') or overview.get('remote_root')
        if remote_root:
            lines.append(f"远端目录: {remote_root}")
        if result.get('transfer_strategy_summary'):
            lines.append(f"策略: {result.get('transfer_strategy_summary')}")
        file_count = result.get('file_count')
        if file_count is None:
            file_count = overview.get('file_count')
        verified_file_count = result.get('verified_file_count')
        if verified_file_count is None:
            verified_file_count = overview.get('verified_file_count', 0)
        skipped_file_count = result.get('skipped_file_count')
        if skipped_file_count is None:
            skipped_file_count = overview.get('skipped_file_count', 0)
        if file_count is not None:
            lines.append(
                f"文件统计: 总计 {file_count or 0} / 已校验 {verified_file_count or 0} / 复用 {skipped_file_count or 0}"
            )
        total_bytes = result.get('total_bytes')
        if total_bytes is None:
            total_bytes = overview.get('total_bytes')
        transferred_bytes = result.get('transferred_bytes')
        if transferred_bytes is None:
            transferred_bytes = overview.get('transferred_bytes', 0)
        skipped_bytes = result.get('skipped_bytes')
        if skipped_bytes is None:
            skipped_bytes = overview.get('skipped_bytes', 0)
        if total_bytes is not None:
            lines.append(
                f"传输体积: 实传 {transferred_bytes or 0}B / 复用 {skipped_bytes or 0}B / 总计 {total_bytes or 0}B"
            )
        uploaded_items = result.get('uploaded_items') or []
        if uploaded_items:
            lines.append('已上传:')
            for item in uploaded_items[:5]:
                lines.append(f"- {item.get('local_path')} -> {item.get('remote_path')}")
        preview = result.get('file_results_preview') or []
        if preview:
            lines.append('文件样例:')
            for item in preview[:5]:
                lines.append(
                    f"- {item.get('relative_path') or item.get('local_path')} / {item.get('mode')} / "
                    f"{item.get('size_bytes', 0)}B"
                )
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'download_assets_from_remote':
        lines = [result.get('summary', '远端下载完成')]
        overview = result.get('download_overview') or {}
        target_label = result.get('target_label') or overview.get('target_label')
        if target_label:
            lines.append(f"来源: {target_label}")
        local_root = result.get('local_root') or overview.get('local_root')
        if local_root:
            lines.append(f"本地目录: {local_root}")
        downloaded_items = result.get('downloaded_items') or []
        if downloaded_items:
            lines.append('已下载:')
            for item in downloaded_items[:5]:
                lines.append(f"- {item.get('remote_path')} -> {item.get('local_path')}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'remote_prediction_pipeline':
        lines = [result.get('summary', '远端预测闭环完成')]
        pipeline = result.get('pipeline_overview') or {}
        execution = result.get('execution_overview') or {}
        target_label = pipeline.get('target_label')
        if target_label:
            lines.append(f"目标服务器: {target_label}")
        remote_root = pipeline.get('remote_root')
        if remote_root:
            lines.append(f"远端目录: {remote_root}")
        source_kind = pipeline.get('source_kind')
        if source_kind:
            lines.append(f"输入类型: {source_kind}")
        remote_output_dir = pipeline.get('remote_output_dir')
        if remote_output_dir:
            lines.append(f"远端预测目录: {remote_output_dir}")
        local_result_root = pipeline.get('local_result_root')
        if local_result_root:
            lines.append(f"本机回传目录: {local_result_root}")
        predict_tool_name = execution.get('predict_tool_name')
        if predict_tool_name:
            lines.append(f"预测执行: {predict_tool_name}")
        if execution:
            lines.append(
                "执行状态: 上传={upload} / 预测={predict} / 下载={download}".format(
                    upload='ok' if execution.get('upload_ok') else 'fail',
                    predict='ok' if execution.get('predict_ok') else 'fail',
                    download='ok' if execution.get('download_ok') else 'skip/fail',
                )
            )
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'remote_training_pipeline':
        lines = [result.get('summary', '远端训练闭环完成')]
        pipeline = result.get('pipeline_overview') or {}
        execution = result.get('execution_overview') or {}
        target_label = pipeline.get('target_label')
        if target_label:
            lines.append(f"目标服务器: {target_label}")
        remote_root = pipeline.get('remote_root')
        if remote_root:
            lines.append(f"远端目录: {remote_root}")
        remote_dataset_path = pipeline.get('remote_dataset_path')
        if remote_dataset_path:
            lines.append(f"远端数据集: {remote_dataset_path}")
        remote_result_path = pipeline.get('remote_result_path')
        if remote_result_path:
            lines.append(f"远端训练目录: {remote_result_path}")
        local_result_root = pipeline.get('local_result_root')
        if local_result_root:
            lines.append(f"本机回传目录: {local_result_root}")
        final_run_state = execution.get('final_run_state')
        if final_run_state:
            lines.append(f"最终状态: {final_run_state}")
        if execution:
            lines.append(
                "执行状态: 上传={upload} / readiness={readiness} / preflight={preflight} / start={start} / 下载={download}".format(
                    upload='ok' if execution.get('upload_ok') else 'fail',
                    readiness='ok' if execution.get('readiness_ok') else 'fail',
                    preflight='ok' if execution.get('preflight_ok') else 'fail',
                    start='ok' if execution.get('start_ok') else 'fail',
                    download='ok' if execution.get('download_ok') else 'skip/fail',
                )
            )
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'scan_cameras':
        lines = [result.get('summary', '摄像头扫描完成')]
        cameras = result.get('cameras') or []
        if cameras:
            lines.append('可用摄像头:')
            for item in cameras[:5]:
                lines.append(f"- id={item.get('id')} / {item.get('name')}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'scan_screens':
        lines = [result.get('summary', '屏幕扫描完成')]
        screens = result.get('screens') or []
        if screens:
            lines.append('可用屏幕:')
            for item in screens[:5]:
                size = ''
                if item.get('width') and item.get('height'):
                    size = f" ({item.get('width')}x{item.get('height')})"
                lines.append(f"- id={item.get('id')} / {item.get('name')}{size}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'test_rtsp_stream':
        lines = [result.get('summary', 'RTSP 流测试完成')]
        if result.get('rtsp_url'):
            lines.append(f"RTSP 地址: {result.get('rtsp_url')}")
        if result.get('error'):
            lines.append(f"失败原因: {result.get('error')}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name in {'start_camera_prediction', 'start_rtsp_prediction', 'start_screen_prediction'}:
        lines = [result.get('summary', '实时预测已启动')]
        if result.get('session_id'):
            lines.append(f"会话 ID: {result.get('session_id')}")
        if result.get('source_type'):
            lines.append(f"来源类型: {result.get('source_type')}")
        if result.get('source_label'):
            lines.append(f"来源: {result.get('source_label')}")
        if result.get('output_dir'):
            lines.append(f"输出目录: {result.get('output_dir')}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name in {'check_realtime_prediction_status', 'stop_realtime_prediction'}:
        lines = [result.get('summary', '实时预测状态已更新')]
        if result.get('session_id'):
            lines.append(f"会话 ID: {result.get('session_id')}")
        if result.get('source_type'):
            lines.append(f"来源类型: {result.get('source_type')}")
        if result.get('status'):
            lines.append(f"状态: {result.get('status')}")
        lines.append(
            f"统计: 已处理 {result.get('processed_frames', 0)} 帧 / 有检测 {result.get('detected_frames', 0)} 帧 / 总检测 {result.get('total_detections', 0)}"
        )
        class_counts = result.get('class_counts') or {}
        if class_counts:
            preview = '，'.join(f"{k}={v}" for k, v in list(class_counts.items())[:4])
            lines.append(f'主要类别: {preview}')
        if result.get('report_path'):
            lines.append(f"实时预测报告: {result.get('report_path')}")
        if result.get('error'):
            lines.append(f"异常: {result.get('error')}")
        suggestions = _recommendation_lines(result, limit=3)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'retrieve_training_knowledge':
        lines = [result.get('summary', '知识检索完成')]
        matched_rules = result.get('matched_rule_overview') or result.get('matched_rules') or []
        if matched_rules:
            lines.append('命中规则:')
            for item in matched_rules[:2]:
                lines.append(f"- {item.get('id')}: {item.get('interpretation') or item.get('title') or item.get('reason')}")
        source_summary = result.get('retrieval_overview') or result.get('source_summary') or {}
        if source_summary:
            preview = '，'.join(f"{k}={v}" for k, v in source_summary.items() if v not in (None, '', [], {}))
            lines.append(f"来源: {preview}")
        playbooks = result.get('playbook_overview') or result.get('playbooks') or []
        if playbooks:
            lines.append('参考资料:')
            for item in playbooks[:2]:
                lines.append(f"- {item.get('title')}: {item.get('path') or item.get('id')}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'analyze_training_outcome':
        lines = [result.get('summary', '训练结果分析完成')]
        analysis_overview = result.get('analysis_overview') or {}
        facts = result.get('facts') or []
        if facts:
            lines.append('事实:')
            lines.extend(f'- {item}' for item in facts[:4])
        elif analysis_overview.get('analysis_ready') is not None:
            lines.append(f"分析就绪: {analysis_overview.get('analysis_ready')}")
        if result.get('interpretation'):
            lines.append(f"解释: {result.get('interpretation')}")
        source_summary = analysis_overview or result.get('source_summary') or {}
        if source_summary:
            preview = '，'.join(f"{k}={v}" for k, v in source_summary.items() if v not in (None, '', [], {}))
            lines.append(f"来源: {preview}")
        if result.get('recommendation'):
            lines.append(f"建议动作: {result.get('recommendation')}")
        suggestions = _recommendation_lines(result)
        if suggestions:
            lines.append('下一步:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    if tool_name == 'recommend_next_training_step':
        lines = [result.get('summary', '下一步建议生成完成')]
        recommendation_overview = result.get('recommendation_overview') or {}
        recommended_action = result.get('recommended_action') or recommendation_overview.get('recommended_action')
        if recommended_action:
            lines.append(f"优先动作: {recommended_action}")
        basis = result.get('basis') or []
        if basis:
            lines.append('依据:')
            lines.extend(f'- {item}' for item in basis[:4])
        source_summary = recommendation_overview or result.get('source_summary') or {}
        if source_summary:
            preview = '，'.join(f"{k}={v}" for k, v in source_summary.items() if v not in (None, '', [], {}))
            lines.append(f"来源: {preview}")
        if result.get('why'):
            lines.append(f"原因: {result.get('why')}")
        suggestions = _recommendation_lines(result, limit=3)
        if suggestions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in suggestions)
        return _join(lines)
    return ''

from __future__ import annotations

from typing import Any


def _join(lines: list[str]) -> str:
    return '\n'.join(lines)


def _format_tool_action(item: Any) -> str:
    if isinstance(item, dict):
        description = str(item.get('description') or '').strip()
        tool_name = str(item.get('tool') or '').strip()
        if description and tool_name:
            return f'{description} ({tool_name})'
        if description:
            return description
        if tool_name:
            return tool_name
    return str(item)


def _observation_stage_label(stage: Any) -> str:
    mapping = {
        'early': '早期观察',
        'mid': '中段观察',
        'late': '后段观察',
        'final': '最终状态',
    }
    key = str(stage or '').strip().lower()
    return mapping.get(key, key or '未知')


def build_grounded_tool_reply(applied_results: list[tuple[str, dict[str, Any]]]) -> str:
    if not applied_results:
        return ""
    tool_name, result = applied_results[-1]
    if not result.get('ok'):
        return ""
    if tool_name == 'run_dataset_health_check':
        lines = [result.get('summary', '健康检查完成')]
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
        if result.get('duplicate_groups'):
            lines.append(
                f'重复图片: {result.get("duplicate_groups", 0)} 组，额外重复文件 {result.get("duplicate_extra_files", 0)} 个'
            )
        if result.get('report_path'):
            lines.append(f'报告路径: {result.get("report_path")}')
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'detect_duplicate_images':
        lines = [result.get('summary', '重复检测完成')]
        groups = result.get('groups') or []
        if groups:
            lines.append('示例重复组:')
            for group in groups[:3]:
                sample_paths = ', '.join(group.get('paths', [])[:2])
                lines.append(f'- {sample_paths}')
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'scan_dataset':
        lines = [result.get('summary', '扫描完成')]
        if result.get('warnings'):
            lines.append('风险:')
            lines.extend(f'- {item}' for item in (result.get('warnings') or [])[:2])
        top_classes = result.get('top_classes') or []
        if top_classes:
            preview = '，'.join(f"{item.get('class')}={item.get('count')}" for item in top_classes[:3])
            lines.append(f'主要类别: {preview}')
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'validate_dataset':
        lines = [result.get('summary', '校验完成')]
        warnings = result.get('warnings') or []
        if warnings:
            lines.append('风险:')
            lines.extend(f'- {item}' for item in warnings[:3])
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'training_readiness':
        lines = [result.get('summary', '训练前检查完成')]
        blockers = result.get('blockers') or []
        warnings = result.get('warnings') or []
        if blockers:
            lines.append('阻塞项:')
            lines.extend(f'- {item}' for item in blockers[:3])
        elif warnings:
            lines.append('风险:')
            lines.extend(f'- {item}' for item in warnings[:3])
        if result.get('preparable'):
            lines.append('当前可继续自动准备: prepare_dataset_for_training')
        if result.get('primary_blocker_type'):
            lines.append(f"主要阻塞类型: {result.get('primary_blocker_type')}")
        if result.get('resolved_data_yaml'):
            lines.append(f"当前可用 YAML: {result.get('resolved_data_yaml')}")
        if result.get('auto_device'):
            lines.append(f"当前 auto 设备策略会解析到: {result.get('auto_device')}")
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {_format_tool_action(item)}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'list_training_environments':
        lines = [result.get('summary', '训练环境查询完成')]
        environments = result.get('environments') or []
        if environments:
            default_environment = result.get('default_environment') or environments[0]
            lines.append(f"默认训练环境: {default_environment.get('display_name') or default_environment.get('name')}")
            lines.append('可用环境:')
            for env in environments[:3]:
                label = env.get('display_name') or env.get('name')
                suffix = ' (默认)' if env.get('selected_by_default') else ''
                lines.append(f'- {label}{suffix}')
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'training_preflight':
        lines = [result.get('summary', '训练预检完成')]
        if result.get('training_environment'):
            env = result.get('training_environment') or {}
            lines.append(f"训练环境: {env.get('display_name') or env.get('name')}")
        resolved_args = result.get('resolved_args') or {}
        if resolved_args.get('model'):
            lines.append(f"模型: {resolved_args.get('model')}")
        if resolved_args.get('data_yaml'):
            lines.append(f"数据 YAML: {resolved_args.get('data_yaml')}")
        output_bits: list[str] = []
        if resolved_args.get('project'):
            output_bits.append(f"project={resolved_args.get('project')}")
        if resolved_args.get('name'):
            output_bits.append(f"name={resolved_args.get('name')}")
        if output_bits:
            lines.append(f"输出组织: {', '.join(output_bits)}")
        if resolved_args.get('batch') is not None:
            lines.append(f"批大小: {resolved_args.get('batch')}")
        if resolved_args.get('imgsz') is not None:
            lines.append(f"输入尺寸: {resolved_args.get('imgsz')}")
        if resolved_args.get('fraction') is not None:
            lines.append(f"采样比例: {resolved_args.get('fraction')}")
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
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'list_training_runs':
        lines = [result.get('summary', '训练历史查询完成')]
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
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
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
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
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
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
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
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'prepare_dataset_for_training':
        lines = [result.get('summary', '数据准备完成')]
        if result.get('data_yaml'):
            lines.append(f"已准备好的 YAML: {result.get('data_yaml')}")
        if result.get('warnings'):
            lines.append('风险:')
            lines.extend(f'- {item}' for item in (result.get('warnings') or [])[:2])
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {_format_tool_action(item)}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'preview_extract_images':
        lines = [result.get('summary', '图片抽取预览完成')]
        lines.append(
            f"统计: 可用 {result.get('available_images', 0)} 张 / 计划抽取 {result.get('planned_extract_count', 0)} 张"
        )
        if result.get('selected_dirs'):
            lines.append(f"目录过滤: {', '.join(result.get('selected_dirs', [])[:4])}")
        if result.get('sample_images'):
            lines.append('样例图片:')
            lines.extend(f'- {item}' for item in (result.get('sample_images') or [])[:2])
        if result.get('output_dir'):
            lines.append(f"计划输出目录: {result.get('output_dir')}")
        if result.get('warnings'):
            lines.append('提示:')
            lines.extend(f'- {item}' for item in (result.get('warnings') or [])[:2])
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'extract_images':
        lines = [result.get('summary', '图片抽取完成')]
        lines.append(
            f"统计: 已抽取 {result.get('extracted', 0)} 张 / 复制标签 {result.get('labels_copied', 0)} / 冲突 {result.get('conflict_count', 0)}"
        )
        if result.get('sample_images'):
            lines.append('抽取样例:')
            lines.extend(f'- {item}' for item in (result.get('sample_images') or [])[:2])
        if result.get('output_dir'):
            lines.append(f"输出目录: {result.get('output_dir')}")
        if result.get('workflow_ready_path'):
            lines.append(f"可继续接主链的目录: {result.get('workflow_ready_path')}")
        if result.get('warnings'):
            lines.append('提示:')
            lines.extend(f'- {item}' for item in (result.get('warnings') or [])[:2])
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'scan_videos':
        lines = [result.get('summary', '视频扫描完成')]
        lines.append(f"统计: 发现 {result.get('total_videos', 0)} 个视频")
        if result.get('sample_videos'):
            lines.append('视频样例:')
            lines.extend(f'- {item}' for item in (result.get('sample_videos') or [])[:2])
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'extract_video_frames':
        lines = [result.get('summary', '视频抽帧完成')]
        lines.append(
            f"统计: 总帧数 {result.get('total_frames', 0)} / 原始抽取 {result.get('extracted', 0)} / 最终保留 {result.get('final_count', 0)}"
        )
        if result.get('output_dir'):
            lines.append(f"输出目录: {result.get('output_dir')}")
        if result.get('warnings'):
            lines.append('提示:')
            lines.extend(f'- {item}' for item in (result.get('warnings') or [])[:2])
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'summarize_training_run':
        lines = [result.get('summary', '训练结果汇总完成')]
        if result.get('run_state'):
            lines.append(f"运行状态: {result.get('run_state')}")
        if result.get('observation_stage'):
            lines.append(f"观察阶段: {_observation_stage_label(result.get('observation_stage'))}")
        progress = result.get('progress') or {}
        if progress.get('epoch') is not None and progress.get('total_epochs') is not None:
            progress_ratio = progress.get('progress_ratio')
            ratio_text = f" ({progress_ratio:.0%})" if isinstance(progress_ratio, (int, float)) else ''
            lines.append(f"训练进度: {progress.get('epoch')}/{progress.get('total_epochs')}{ratio_text}")
        metrics = result.get('metrics') or {}
        if metrics and any(metrics.get(key) is not None for key in ('precision', 'recall', 'map50', 'map')):
            lines.append(
                '关键指标: '
                f"precision={metrics.get('precision', 0):.3f}, "
                f"recall={metrics.get('recall', 0):.3f}, "
                f"mAP50={metrics.get('map50', 0):.3f}, "
                f"mAP50-95={metrics.get('map', 0):.3f}"
            )
        elif metrics and any(metrics.get(key) is not None for key in ('box_loss', 'cls_loss', 'dfl_loss')):
            lines.append(
                '当前仅有训练损失: '
                f"box={metrics.get('box_loss')}, cls={metrics.get('cls_loss')}, dfl={metrics.get('dfl_loss')}"
            )
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
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'check_training_status':
        lines = [result.get('summary', '训练状态已更新')]
        if result.get('run_state'):
            lines.append(f"运行状态: {result.get('run_state')}")
        if result.get('observation_stage'):
            lines.append(f"观察阶段: {_observation_stage_label(result.get('observation_stage'))}")
        progress = result.get('progress') or {}
        if progress.get('epoch') is not None and progress.get('total_epochs') is not None:
            progress_ratio = progress.get('progress_ratio')
            ratio_text = f" ({progress_ratio:.0%})" if isinstance(progress_ratio, (int, float)) else ''
            lines.append(f"最近进度: {progress.get('epoch')}/{progress.get('total_epochs')}{ratio_text}")
        metrics = ((result.get('latest_metrics') or {}).get('metrics') or {})
        if metrics and any(metrics.get(key) is not None for key in ('precision', 'recall', 'map50', 'map')):
            lines.append(
                '最近指标: '
                f"precision={metrics.get('precision', 0):.3f}, "
                f"recall={metrics.get('recall', 0):.3f}, "
                f"mAP50={metrics.get('map50', 0):.3f}, "
                f"mAP50-95={metrics.get('map', 0):.3f}"
            )
        elif metrics and any(metrics.get(key) is not None for key in ('box_loss', 'cls_loss', 'dfl_loss')):
            lines.append(
                '当前仅有训练损失: '
                f"box={metrics.get('box_loss')}, cls={metrics.get('cls_loss')}, dfl={metrics.get('dfl_loss')}"
            )
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
            lines.extend(f'- {item}' for item in facts[:3])
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'predict_images':
        lines = [result.get('summary', '预测完成')]
        lines.append(
            f"统计: 已处理 {result.get('processed_images', 0)} 张 / 有检测 {result.get('detected_images', 0)} / 无检测 {result.get('empty_images', 0)}"
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
        if result.get('annotated_dir'):
            lines.append(f"标注结果目录: {result.get('annotated_dir')}")
        if result.get('report_path'):
            lines.append(f"预测报告: {result.get('report_path')}")
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'predict_videos':
        lines = [result.get('summary', '视频预测完成')]
        lines.append(
            f"统计: 已处理 {result.get('processed_videos', 0)} 个视频 / 总帧数 {result.get('total_frames', 0)} / 有检测帧 {result.get('detected_frames', 0)} / 总检测框 {result.get('total_detections', 0)}"
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
        if result.get('output_dir'):
            lines.append(f"视频预测输出目录: {result.get('output_dir')}")
        if result.get('report_path'):
            lines.append(f"预测报告: {result.get('report_path')}")
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'summarize_prediction_results':
        lines = [result.get('summary', '预测结果汇总完成')]
        if result.get('mode') == 'videos':
            lines.append(
                f"统计: 已处理 {result.get('processed_videos', 0)} 个视频 / 总帧数 {result.get('total_frames', 0)} / 有检测帧 {result.get('detected_frames', 0)} / 总检测框 {result.get('total_detections', 0)}"
            )
        else:
            lines.append(
                f"统计: 已处理 {result.get('processed_images', 0)} 张 / 有检测 {result.get('detected_images', 0)} / 无检测 {result.get('empty_images', 0)} / 总检测框 {result.get('total_detections', 0)}"
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
        if result.get('annotated_dir'):
            lines.append(f"标注结果目录: {result.get('annotated_dir')}")
        if result.get('report_path'):
            lines.append(f"预测报告: {result.get('report_path')}")
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'retrieve_training_knowledge':
        lines = [result.get('summary', '知识检索完成')]
        matched_rules = result.get('matched_rules') or []
        if matched_rules:
            lines.append('命中规则:')
            for item in matched_rules[:2]:
                lines.append(f"- {item.get('id')}: {item.get('interpretation')}")
        source_summary = result.get('source_summary') or {}
        if source_summary:
            preview = '，'.join(f"{k}={v}" for k, v in source_summary.items())
            lines.append(f"来源: {preview}")
        playbooks = result.get('playbooks') or []
        if playbooks:
            lines.append('参考资料:')
            for item in playbooks[:2]:
                lines.append(f"- {item.get('title')}: {item.get('path')}")
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'analyze_training_outcome':
        lines = [result.get('summary', '训练结果分析完成')]
        facts = result.get('facts') or []
        if facts:
            lines.append('事实:')
            lines.extend(f'- {item}' for item in facts[:4])
        if result.get('interpretation'):
            lines.append(f"解释: {result.get('interpretation')}")
        source_summary = result.get('source_summary') or {}
        if source_summary:
            preview = '，'.join(f"{k}={v}" for k, v in source_summary.items())
            lines.append(f"来源: {preview}")
        if result.get('recommendation'):
            lines.append(f"建议动作: {result.get('recommendation')}")
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('下一步:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        return _join(lines)
    if tool_name == 'recommend_next_training_step':
        lines = [result.get('summary', '下一步建议生成完成')]
        if result.get('recommended_action'):
            lines.append(f"优先动作: {result.get('recommended_action')}")
        basis = result.get('basis') or []
        if basis:
            lines.append('依据:')
            lines.extend(f'- {item}' for item in basis[:4])
        source_summary = result.get('source_summary') or {}
        if source_summary:
            preview = '，'.join(f"{k}={v}" for k, v in source_summary.items())
            lines.append(f"来源: {preview}")
        if result.get('why'):
            lines.append(f"原因: {result.get('why')}")
        next_actions = result.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:3])
        return _join(lines)
    return ''

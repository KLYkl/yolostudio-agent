from __future__ import annotations

from typing import Any


def _join(lines: list[str]) -> str:
    return '\n'.join(lines)


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
        if result.get('resolved_data_yaml'):
            lines.append(f"当前可用 YAML: {result.get('resolved_data_yaml')}")
        if result.get('auto_device'):
            lines.append(f"当前 auto 设备策略会解析到: {result.get('auto_device')}")
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
            lines.extend(f'- {item}' for item in next_actions[:2])
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
    if tool_name == 'check_training_status':
        lines = [result.get('summary', '训练状态已更新')]
        metrics = ((result.get('latest_metrics') or {}).get('metrics') or {})
        if metrics:
            epoch = metrics.get('epoch')
            total = metrics.get('total_epochs')
            if epoch is not None and total is not None:
                lines.append(f'最近观测到 epoch: {epoch}/{total}')
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
    return ''

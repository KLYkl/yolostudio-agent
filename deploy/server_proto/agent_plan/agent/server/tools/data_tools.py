from __future__ import annotations

from pathlib import Path
from typing import Any

from yolostudio_agent.agent.server.services.dataset_root import resolve_dataset_inputs, resolve_dataset_root
from yolostudio_agent.agent.server.services.gpu_utils import describe_gpu_policy, get_effective_gpu_policy, resolve_auto_device
from yolostudio_agent.agent.server.tools.data_tool_helpers import (
    MAX_ISSUE_EXAMPLES,
    _build_missing_label_risk,
    _discover_classes_txt,
    _discover_data_yaml,
    _error_payload,
    _format_issue_examples,
    _infer_dataset_root,
    _merge_risk_levels,
    _read_classes_txt_lines,
    _read_yaml_names,
    _resolve_dataset_inputs,
    _sample_integrity_entries,
    _sample_path_strings,
    _serialize_duplicate_groups,
    _summarize_health_outputs,
    _top_class_stats,
)


def _install_headless_pyside6_stub() -> None:
    import sys
    import types

    if 'PySide6.QtCore' in sys.modules:
        return

    class _DummySignal:
        def __init__(self, *args, **kwargs) -> None:
            self._slots: list[Any] = []

        def connect(self, slot) -> None:
            self._slots.append(slot)

        def emit(self, *args, **kwargs) -> None:
            for slot in list(self._slots):
                try:
                    slot(*args, **kwargs)
                except Exception:
                    pass

    class _DummyQThread:
        def __init__(self, parent=None) -> None:
            self.parent = parent

        def start(self) -> None:
            raise RuntimeError('当前为 headless PySide6 stub，不支持启动 QThread')

    qtcore = types.ModuleType('PySide6.QtCore')
    qtcore.QThread = _DummyQThread
    qtcore.Signal = lambda *args, **kwargs: _DummySignal()

    pyside6 = types.ModuleType('PySide6')
    pyside6.QtCore = qtcore

    sys.modules.setdefault('PySide6', pyside6)
    sys.modules['PySide6.QtCore'] = qtcore


def _import_core_data_handler_module(module_name: str):
    import importlib
    import sys

    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        message = str(exc)
        if 'PySide6' not in message and 'QtCore' not in message:
            raise
        _install_headless_pyside6_stub()
        for name in ['core.data_handler', 'core.data_handler._worker']:
            sys.modules.pop(name, None)
        return importlib.import_module(module_name)


def _get_data_handler_cls():
    return _import_core_data_handler_module('core.data_handler._handler').DataHandler


def _get_data_models_module():
    return _import_core_data_handler_module('core.data_handler._models')


def summarize_scan_result(result) -> str:
    return (
        f"总图片: {result.total_images}, 已标注: {result.labeled_images}, "
        f"缺失标签: {len(result.missing_labels)}, 空标签: {result.empty_labels}, "
        f"类别数: {len(result.classes)}"
    )


def scan_dataset(img_dir: str, label_dir: str = "") -> dict[str, Any]:
    """扫描数据集并返回结构化摘要、类别统计、候选 YAML / classes.txt 信息。img_dir 支持传入 dataset root。"""
    try:
        DataHandler = _get_data_handler_cls()

        img_path, label_path, resolution = _resolve_dataset_inputs(img_dir, label_dir)
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path, label_path))
        detected_data_yaml, data_yaml_candidates = _discover_data_yaml(img_path, label_path)
        detected_classes_txt, classes_txt_candidates = _discover_classes_txt(img_path, label_path)

        handler = DataHandler()
        result = handler.scan_dataset(
            img_dir=img_path,
            label_dir=label_path,
            classes_txt=Path(detected_classes_txt) if detected_classes_txt else None,
        )
        missing_label_risk = _build_missing_label_risk(result)
        next_actions = ["可继续 validate_dataset 做标签合法性校验"]
        if detected_data_yaml:
            next_actions.append(f"可直接使用 detected_data_yaml 训练: {detected_data_yaml}")
        else:
            next_actions.append("尚未发现可直接训练的 data.yaml；如要训练需显式提供 YAML 路径")
        if detected_classes_txt:
            next_actions.append(f"已发现 classes.txt，可用于保留真实类名: {detected_classes_txt}")
        if missing_label_risk['warnings']:
            next_actions.append("建议先处理缺失标签图片，或在 readiness 阶段确认是否接受当前风险")
        class_name_source = 'classes_txt' if detected_classes_txt else ('parsed_labels' if result.classes else '')
        return {
            "ok": True,
            "summary": summarize_scan_result(result),
            "dataset_root": str(dataset_root),
            "structure_type": resolution.get('structure_type'),
            "resolved_from_root": resolution.get('resolved_from_root', False),
            "resolved_img_dir": str(img_path),
            "resolved_label_dir": str(label_path) if label_path else '',
            "total_images": result.total_images,
            "labeled_images": result.labeled_images,
            "missing_labels": len(result.missing_labels),
            "missing_label_images": missing_label_risk['missing_label_images'],
            "missing_label_ratio": missing_label_risk['missing_label_ratio'],
            "risk_level": missing_label_risk['risk_level'],
            "warnings": missing_label_risk['warnings'],
            "missing_label_examples": [str(path) for path in result.missing_labels[:MAX_ISSUE_EXAMPLES]],
            "empty_labels": result.empty_labels,
            "classes": result.classes,
            "class_stats": result.class_stats,
            "top_classes": _top_class_stats(result.class_stats),
            "label_format": result.label_format.name if result.label_format else None,
            "detected_data_yaml": detected_data_yaml,
            "data_yaml_candidates": data_yaml_candidates,
            "detected_classes_txt": detected_classes_txt,
            "classes_txt_candidates": classes_txt_candidates,
            "class_name_source": class_name_source,
            "next_actions": next_actions,
        }
    except Exception as exc:
        return _error_payload(exc, "扫描数据集")


def split_dataset(
    img_dir: str,
    label_dir: str = "",
    output_dir: str = "",
    ratio: float = 0.8,
    seed: int = 42,
    mode: str = "copy",
    ignore_orphans: bool = False,
    clear_output: bool = False,
) -> dict[str, Any]:
    """按现有 DataHandler 能力将数据集切分为 train/val。"""
    try:
        DataHandler = _get_data_handler_cls()
        SplitMode = _get_data_models_module().SplitMode

        handler = DataHandler()
        mode_map = {
            "copy": SplitMode.COPY,
            "move": SplitMode.MOVE,
            "index": SplitMode.INDEX,
        }
        selected_mode = mode_map.get(mode.lower())
        if selected_mode is None:
            raise ValueError(f"不支持的 split mode: {mode}")

        resolved_output = Path(output_dir) if output_dir else None
        result = handler.split_dataset(
            img_dir=Path(img_dir),
            label_dir=Path(label_dir) if label_dir else None,
            output_dir=resolved_output,
            ratio=ratio,
            seed=seed,
            mode=selected_mode,
            ignore_orphans=ignore_orphans,
            clear_output=clear_output,
        )
        if resolved_output:
            abs_output = str(resolved_output.resolve())
        else:
            abs_output = str((Path(img_dir).parent / f"{Path(img_dir).name}_split").resolve())
        train_ratio = round(result.train_count / max(result.train_count + result.val_count, 1), 4)
        val_ratio = round(result.val_count / max(result.train_count + result.val_count, 1), 4)
        suggested_yaml = str((Path(abs_output) / 'data.yaml').resolve())
        return {
            "ok": True,
            "summary": f"数据集已划分: train={result.train_count}, val={result.val_count}, mode={mode.lower()}",
            "output_dir": abs_output,
            "train_path": result.train_path,
            "val_path": result.val_path,
            "train_count": result.train_count,
            "val_count": result.val_count,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "mode": mode.lower(),
            "suggested_yaml_path": suggested_yaml,
            "next_actions": [
                "建议检查划分结果是否符合预期",
                f"如需训练，可基于输出目录生成/准备 YAML: {suggested_yaml}",
            ],
        }
    except Exception as exc:
        return _error_payload(exc, "划分数据集")


def generate_yaml(
    train_path: str,
    val_path: str,
    classes: list[str] | None = None,
    output_path: str = "",
    classes_txt: str = "",
    img_dir: str = "",
    label_dir: str = "",
) -> dict[str, Any]:
    """根据 split 结果或现有 data.yaml/classes.txt 生成 YOLO 训练 YAML。"""
    try:
        import os
        import yaml

        train_value = str(train_path).strip()
        val_value = str(val_path).strip()
        if not train_value or not val_value:
            raise ValueError('train_path 和 val_path 不能为空')

        class_names = [str(c) for c in (classes or []) if str(c).strip()]
        class_name_source = 'explicit_classes' if class_names else ''
        classes_txt_path = Path(classes_txt) if classes_txt else None
        if not class_names and not classes_txt_path and img_dir:
            detected_classes_txt, _ = _discover_classes_txt(Path(img_dir), Path(label_dir) if label_dir else None)
            classes_txt_path = Path(detected_classes_txt) if detected_classes_txt else None

        if not class_names and classes_txt_path and classes_txt_path.exists():
            class_names = _read_classes_txt_lines(classes_txt_path)
            class_name_source = 'classes_txt' if class_names else class_name_source

        if not class_names and img_dir:
            detected_yaml, candidates = _discover_data_yaml(Path(img_dir), Path(label_dir) if label_dir else None)
            yaml_candidate = detected_yaml or (candidates[0] if candidates else '')
            if yaml_candidate:
                class_names = _read_yaml_names(Path(yaml_candidate))
                class_name_source = 'detected_yaml' if class_names else class_name_source

        if not class_names:
            raise ValueError('无法确定 classes；请显式传入 classes，或提供 classes_txt / 可解析的 data.yaml')

        output = Path(output_path).resolve() if output_path else (Path(train_value).resolve().parent.parent / 'data.yaml')
        train_p = Path(train_value)
        val_p = Path(val_value)
        if train_p.is_absolute() and val_p.is_absolute():
            dataset_root = Path(os.path.commonpath([train_p, val_p]))
            train_yaml_value = str(train_p.relative_to(dataset_root))
            val_yaml_value = str(val_p.relative_to(dataset_root))
        else:
            dataset_root = output.parent
            train_yaml_value = train_value
            val_yaml_value = val_value

        yaml_content = {
            'path': str(dataset_root),
            'train': train_yaml_value,
            'val': val_yaml_value,
            'names': {i: name for i, name in enumerate(class_names)},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(yaml_content, allow_unicode=True, sort_keys=False), encoding='utf-8')

        return {
            'ok': True,
            'summary': f'YAML 已生成: {output}',
            'output_path': str(output),
            'dataset_root': str(dataset_root),
            'train_path': train_yaml_value,
            'val_path': val_yaml_value,
            'class_count': len(class_names),
            'classes': class_names,
            'class_name_source': class_name_source,
            'next_actions': [
                f'可直接使用该 YAML 启动训练: {output}',
            ],
        }
    except Exception as exc:
        return _error_payload(exc, '生成 YAML')


def validate_dataset(
    img_dir: str,
    label_dir: str = "",
    classes_txt: str = "",
    check_coords: bool = True,
    check_class_ids: bool = True,
    check_format: bool = True,
    check_orphans: bool = True,
) -> dict[str, Any]:
    """校验标签合法性并返回问题统计与示例。img_dir 支持传入 dataset root。"""
    try:
        DataHandler = _get_data_handler_cls()

        handler = DataHandler()
        img_path, label_path, resolution = _resolve_dataset_inputs(img_dir, label_dir)
        detected_classes_txt, classes_txt_candidates = _discover_classes_txt(img_path, label_path)
        effective_classes_txt = classes_txt or detected_classes_txt
        result = handler.validate_labels(
            img_dir=img_path,
            label_dir=label_path,
            classes_txt=Path(effective_classes_txt) if effective_classes_txt else None,
            check_coords=check_coords,
            check_class_ids=check_class_ids,
            check_format=check_format,
            check_orphans=check_orphans,
        )
        scan_result = handler.scan_dataset(
            img_dir=img_path,
            label_dir=label_path,
            classes_txt=Path(effective_classes_txt) if effective_classes_txt else None,
        )
        missing_label_risk = _build_missing_label_risk(scan_result)
        breakdown = {
            "coord_errors": len(result.coord_errors),
            "class_errors": len(result.class_errors),
            "format_errors": len(result.format_errors),
            "orphan_labels": len(result.orphan_labels),
        }
        warnings = list(missing_label_risk['warnings'])
        has_risks = bool(warnings)
        summary_parts = []
        if result.has_issues:
            summary_parts.append(
                f"发现 {result.issue_count} 个标签问题: 坐标 {breakdown['coord_errors']}, 类别 {breakdown['class_errors']}, 格式 {breakdown['format_errors']}, 孤立标签 {breakdown['orphan_labels']}"
            )
        else:
            summary_parts.append('未发现标签格式/坐标问题')
        if warnings:
            summary_parts.extend(warnings)
        summary = '；'.join(summary_parts)
        next_actions = []
        if result.has_issues:
            next_actions.append('建议先修复 issue_examples 中的问题，再继续划分或训练')
        else:
            next_actions.append('标签格式层面可继续训练或做数据划分')
        if warnings:
            next_actions.append('建议确认缺失标签图片是否属于背景图；如不是，先补齐标签再训练')
        return {
            "ok": True,
            "summary": summary,
            "dataset_root": resolution.get('dataset_root', ''),
            "resolved_img_dir": str(img_path),
            "resolved_label_dir": str(label_path) if label_path else '',
            "detected_classes_txt": detected_classes_txt,
            "classes_txt_candidates": classes_txt_candidates,
            "total_labels": result.total_labels,
            "has_issues": result.has_issues,
            "has_risks": has_risks,
            "risk_level": missing_label_risk['risk_level'],
            "warnings": warnings,
            "missing_label_images": missing_label_risk['missing_label_images'],
            "missing_label_ratio": missing_label_risk['missing_label_ratio'],
            "issue_count": result.issue_count,
            "issue_breakdown": breakdown,
            "issue_examples": _format_issue_examples(result),
            "next_actions": next_actions,
        }
    except Exception as exc:
        return _error_payload(exc, "校验数据集")


def training_readiness(
    img_dir: str,
    label_dir: str = "",
    data_yaml: str = "",
    require_clean_labels: bool = True,
) -> dict[str, Any]:
    """给出当前数据集是否适合直接训练的综合判断。优先用作训练前的标准检查入口。"""
    try:
        from yolostudio_agent.agent.server.services.gpu_utils import query_gpu_status

        scan = scan_dataset(img_dir=img_dir, label_dir=label_dir)
        if not scan.get('ok'):
            return scan
        validate = validate_dataset(img_dir=img_dir, label_dir=label_dir, classes_txt=scan.get('detected_classes_txt', ''))
        if not validate.get('ok'):
            return validate

        resolved_yaml = data_yaml or scan.get('detected_data_yaml', '')
        yaml_exists = bool(resolved_yaml and Path(resolved_yaml).exists())
        labels_clean = not validate.get('has_issues', False)
        labeled_images = int(scan.get('labeled_images', 0) or 0)
        total_images = int(scan.get('total_images', 0) or 0)
        hard_label_block = total_images > 0 and labeled_images <= 0
        gpu_info = query_gpu_status()
        available_gpus = [gpu.index for gpu in gpu_info if not gpu.busy]
        device_policy = get_effective_gpu_policy()
        auto_device, auto_error = resolve_auto_device(policy=device_policy, gpus=gpu_info)
        warnings = list(validate.get('warnings', []))
        risk_level = validate.get('risk_level', scan.get('risk_level', 'none'))
        ready = yaml_exists and (labels_clean or not require_clean_labels) and bool(available_gpus) and not hard_label_block

        primary_blocker_type = ''
        if hard_label_block:
            primary_blocker_type = 'no_valid_labels'
        elif require_clean_labels and not labels_clean:
            primary_blocker_type = 'label_issues'
        elif not yaml_exists:
            primary_blocker_type = 'missing_yaml'
        elif not available_gpus or auto_error:
            primary_blocker_type = 'gpu_unavailable'

        preparable = (
            not ready
            and bool(scan.get('resolved_img_dir'))
            and not hard_label_block
        )

        blockers: list[str] = []
        if not yaml_exists:
            blockers.append('缺少可用的 data_yaml')
        if require_clean_labels and not labels_clean:
            blockers.append('标签校验未通过')
        if hard_label_block:
            blockers.append('当前没有任何有效标注图片，无法进行训练')
        if not available_gpus:
            blockers.append('当前没有空闲 GPU')
        if auto_error:
            blockers.append(f'当前 auto 设备不可解析: {auto_error}')

        data_yaml_source = 'explicit_input' if data_yaml else ('detected_existing_yaml' if resolved_yaml else '')
        next_actions: list[dict[str, Any]] = []
        if primary_blocker_type == 'missing_yaml' and preparable:
            next_actions.append({
                'description': '当前不能直接训练，但可以先调用 prepare_dataset_for_training 自动补齐 YAML 和划分产物',
                'tool': 'prepare_dataset_for_training',
                'args_hint': {
                    'dataset_path': scan.get('dataset_root') or scan.get('resolved_img_dir', img_dir),
                },
            })
        elif primary_blocker_type == 'label_issues':
            next_actions.append({
                'description': '先根据 issue_examples 修复标签问题',
                'tool': 'validate_dataset',
                'args_hint': {'img_dir': scan.get('resolved_img_dir', img_dir), 'label_dir': scan.get('resolved_label_dir', label_dir)},
            })
        elif primary_blocker_type == 'gpu_unavailable':
            next_actions.append({
                'description': '等待空闲 GPU，或释放当前占用后再训练',
                'tool': 'check_gpu_status',
                'args_hint': {},
            })

        if not yaml_exists and not any(action.get('tool') == 'prepare_dataset_for_training' for action in next_actions):
            next_actions.append({
                'description': '可先调用 generate_yaml 生成训练 YAML',
                'tool': 'generate_yaml',
                'args_hint': {
                    'img_dir': scan.get('resolved_img_dir', img_dir),
                    'label_dir': scan.get('resolved_label_dir', label_dir),
                    'classes_txt': scan.get('detected_classes_txt', ''),
                    'classes': scan.get('classes', []),
                },
            })
        if require_clean_labels and not labels_clean and not any(action.get('tool') == 'validate_dataset' for action in next_actions):
            next_actions.append({
                'description': '先根据 issue_examples 修复标签问题',
                'tool': 'validate_dataset',
                'args_hint': {'img_dir': scan.get('resolved_img_dir', img_dir), 'label_dir': scan.get('resolved_label_dir', label_dir)},
            })
        if warnings:
            next_actions.append({
                'description': '确认缺失标签图片是否属于背景图；如不是，建议先补齐标签',
                'tool': 'scan_dataset',
                'args_hint': {'img_dir': scan.get('resolved_img_dir', img_dir), 'label_dir': scan.get('resolved_label_dir', label_dir)},
            })
        if (not available_gpus or auto_error) and not any(action.get('tool') == 'check_gpu_status' for action in next_actions):
            next_actions.append({
                'description': '等待空闲 GPU，或释放当前占用后再训练',
                'tool': 'check_gpu_status',
                'args_hint': {},
            })
        if ready:
            next_actions.append({
                'description': '可以直接调用 start_training 开始训练',
                'tool': 'start_training',
                'args_hint': {'data_yaml': resolved_yaml},
            })

        summary = '可以直接训练' if ready else f"当前还不能直接训练: {', '.join(blockers)}"
        if not ready and primary_blocker_type == 'missing_yaml' and preparable:
            summary += '；但当前数据集可以先进入 prepare_dataset_for_training'
        if ready and warnings:
            summary = f"可以训练，但存在数据质量风险: {'; '.join(warnings)}"

        return {
            'ok': True,
            'ready': ready,
            'preparable': preparable,
            'primary_blocker_type': primary_blocker_type,
            'summary': summary,
            'dataset_root': scan.get('dataset_root', ''),
            'resolved_img_dir': scan.get('resolved_img_dir', img_dir),
            'resolved_label_dir': scan.get('resolved_label_dir', label_dir),
            'resolved_data_yaml': resolved_yaml,
            'data_yaml_source': data_yaml_source,
            'detected_classes_txt': scan.get('detected_classes_txt', ''),
            'class_name_source': scan.get('class_name_source', ''),
            'recommended_start_training_args': {'data_yaml': resolved_yaml} if ready and resolved_yaml else {},
            'labels_clean': labels_clean,
            'risk_level': risk_level,
            'warnings': warnings,
            'missing_label_images': validate.get('missing_label_images', scan.get('missing_label_images', 0)),
            'missing_label_ratio': validate.get('missing_label_ratio', scan.get('missing_label_ratio', 0.0)),
            'device_policy': device_policy,
            'device_policy_summary': describe_gpu_policy(device_policy),
            'auto_device': auto_device,
            'auto_error': auto_error,
            'available_gpu_indexes': available_gpus,
            'scan_summary': scan.get('summary'),
            'validation_summary': validate.get('summary'),
            'blockers': blockers,
            'next_actions': next_actions,
        }
    except Exception as exc:
        return _error_payload(exc, '检查训练就绪状态')


def run_dataset_health_check(
    dataset_path: str,
    include_duplicates: bool = True,
    duplicate_method: str = 'md5',
    hash_threshold: int = 8,
    small_threshold: int = 32,
    large_threshold: int = 8192,
    export_report: bool = False,
    report_path: str = '',
    max_examples: int = MAX_ISSUE_EXAMPLES,
    max_duplicate_groups: int = MAX_ISSUE_EXAMPLES,
) -> dict[str, Any]:
    """对数据集图片做只读健康检查：完整性、尺寸异常、重复图片，并可选导出报告。"""
    try:
        DataHandler = _get_data_handler_cls()

        duplicate_method = duplicate_method.lower().strip() or 'md5'
        if duplicate_method not in {'md5', 'phash'}:
            raise ValueError(f'不支持的 duplicate_method: {duplicate_method}')
        if duplicate_method == 'phash':
            try:
                import imagehash  # noqa: F401
            except ImportError as exc:
                raise RuntimeError('当前环境未安装 imagehash，无法使用 phash；请改用 md5') from exc

        resolution = resolve_dataset_root(dataset_path)
        if not resolution.get('ok'):
            return resolution

        img_path = Path(resolution.get('img_dir') or dataset_path)
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path))
        handler = DataHandler()

        integrity = handler.check_image_integrity(img_path)
        sizes = handler.analyze_image_sizes(
            img_path,
            small_threshold=small_threshold,
            large_threshold=large_threshold,
        )
        duplicates = handler.detect_duplicates(
            img_path,
            method=duplicate_method,
            hash_threshold=hash_threshold,
        ) if include_duplicates else []

        health_summary = _summarize_health_outputs(integrity, sizes, duplicates, include_duplicates=include_duplicates)
        exported_report = ''
        if export_report:
            resolved_report = Path(report_path).resolve() if report_path else (dataset_root / '_health_check_report.txt').resolve()
            exported_report = str(handler.export_check_report(
                resolved_report,
                integrity=integrity,
                sizes=sizes,
                duplicates=duplicates,
            ))

        next_actions: list[str] = []
        if health_summary['warnings']:
            next_actions.append('建议先处理损坏/异常图片，再继续数据准备或训练')
        if health_summary['duplicate_group_count']:
            next_actions.append('可先人工确认重复图片是否需要清理；如需进一步查看，可单独调用 detect_duplicate_images')
        if exported_report:
            next_actions.append(f'可离线查看健康检查报告: {exported_report}')
        elif health_summary['issue_count']:
            next_actions.append('如需归档检查结果，可设置 export_report=true 导出文本报告')
        if not next_actions:
            next_actions.append('图像层面未见明显阻塞，可继续做 validate_dataset 或 training_readiness')

        return {
            'ok': True,
            'summary': health_summary['summary'],
            'dataset_root': str(dataset_root),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_path),
            'risk_level': health_summary['risk_level'],
            'warnings': health_summary['warnings'],
            'issue_count': health_summary['issue_count'],
            'report_path': exported_report,
            'integrity': {
                'total_images': integrity.total_images,
                'issue_count': integrity.issue_count,
                'corrupted_count': health_summary['corrupted_count'],
                'zero_bytes_count': health_summary['zero_bytes_count'],
                'format_mismatch_count': health_summary['format_mismatch_count'],
                'exif_rotation_count': health_summary['exif_rotation_count'],
                'corrupted_examples': _sample_integrity_entries(integrity.corrupted, max_examples),
                'zero_bytes_examples': _sample_path_strings(integrity.zero_bytes, max_examples),
                'format_mismatch_examples': _sample_integrity_entries(integrity.format_mismatch, max_examples),
                'exif_rotation_examples': _sample_integrity_entries(integrity.exif_rotation, max_examples),
            },
            'size_stats': {
                'total_images': sizes.total_images,
                'min_size': list(sizes.min_size),
                'max_size': list(sizes.max_size),
                'avg_size': list(sizes.avg_size),
                'abnormal_small_count': health_summary['abnormal_small_count'],
                'abnormal_large_count': health_summary['abnormal_large_count'],
                'abnormal_small_examples': _sample_path_strings(sizes.abnormal_small, max_examples),
                'abnormal_large_examples': _sample_path_strings(sizes.abnormal_large, max_examples),
            },
            'duplicate_method': duplicate_method if include_duplicates else '',
            'hash_threshold': hash_threshold if include_duplicates else None,
            'duplicate_groups': health_summary['duplicate_group_count'],
            'duplicate_files_total': health_summary['duplicate_files_total'],
            'duplicate_extra_files': health_summary['duplicate_extra_files'],
            'duplicate_group_samples': _serialize_duplicate_groups(duplicates, max_groups=max_duplicate_groups, max_paths_per_group=max_examples),
            'next_actions': next_actions,
        }
    except Exception as exc:
        return _error_payload(exc, '执行数据集健康检查')


def detect_duplicate_images(
    dataset_path: str,
    method: str = 'md5',
    hash_threshold: int = 8,
    max_groups: int = 10,
    max_paths_per_group: int = MAX_ISSUE_EXAMPLES,
) -> dict[str, Any]:
    """检测图片重复样本，返回重复组摘要和示例路径。dataset_path 支持 dataset root。"""
    try:
        DataHandler = _get_data_handler_cls()

        method = method.lower().strip() or 'md5'
        if method not in {'md5', 'phash'}:
            raise ValueError(f'不支持的重复检测方法: {method}')
        if method == 'phash':
            try:
                import imagehash  # noqa: F401
            except ImportError as exc:
                raise RuntimeError('当前环境未安装 imagehash，无法使用 phash；请改用 md5') from exc

        resolution = resolve_dataset_root(dataset_path)
        if not resolution.get('ok'):
            return resolution

        img_path = Path(resolution.get('img_dir') or dataset_path)
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path))
        handler = DataHandler()
        duplicates = handler.detect_duplicates(
            img_path,
            method=method,
            hash_threshold=hash_threshold,
        )

        duplicate_groups = len(duplicates)
        duplicate_files_total = sum(len(group.paths) for group in duplicates)
        duplicate_extra_files = sum(max(len(group.paths) - 1, 0) for group in duplicates)
        risk_level = 'high' if duplicate_groups >= 10 else ('medium' if duplicate_groups else 'none')
        summary = (
            f'检测完成: 发现 {duplicate_groups} 组重复图片，额外重复文件 {duplicate_extra_files} 个'
            if duplicate_groups
            else '检测完成，未发现重复图片'
        )
        next_actions = [
            '建议人工确认 sample groups 中的文件是否应合并或清理'
        ] if duplicate_groups else ['当前未发现重复图片，可继续做健康检查或训练准备']
        return {
            'ok': True,
            'summary': summary,
            'dataset_root': str(dataset_root),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_path),
            'method': method,
            'hash_threshold': hash_threshold,
            'risk_level': risk_level,
            'duplicate_groups': duplicate_groups,
            'duplicate_files_total': duplicate_files_total,
            'duplicate_extra_files': duplicate_extra_files,
            'groups': _serialize_duplicate_groups(duplicates, max_groups=max_groups, max_paths_per_group=max_paths_per_group),
            'next_actions': next_actions,
        }
    except Exception as exc:
        return _error_payload(exc, '检测重复图片')


def augment_dataset(
    img_dir: str,
    label_dir: str = "",
    output_dir: str = "",
    classes_txt: str = "",
    copies_per_image: int = 1,
    include_original: bool = True,
    seed: int = 42,
    mode: str = "random",
    enable_horizontal_flip: bool = True,
    enable_rotate: bool = False,
    rotate_degrees: float = 15.0,
    enable_brightness: bool = False,
    brightness_strength: float = 0.2,
    enable_contrast: bool = False,
    contrast_strength: float = 0.25,
    enable_noise: bool = False,
    noise_strength: float = 0.08,
) -> dict[str, Any]:
    """执行离线数据增强，默认启用最常用的水平翻转。"""
    try:
        DataHandler = _get_data_handler_cls()
        AugmentConfig = _get_data_models_module().AugmentConfig

        handler = DataHandler()
        config = AugmentConfig(
            copies_per_image=copies_per_image,
            include_original=include_original,
            seed=seed,
            mode=mode,
            enable_horizontal_flip=enable_horizontal_flip,
            enable_rotate=enable_rotate,
            rotate_degrees=rotate_degrees,
            enable_brightness=enable_brightness,
            brightness_strength=brightness_strength,
            enable_contrast=enable_contrast,
            contrast_strength=contrast_strength,
            enable_noise=enable_noise,
            noise_strength=noise_strength,
        )
        result = handler.augment_dataset(
            img_dir=Path(img_dir),
            config=config,
            label_dir=Path(label_dir) if label_dir else None,
            output_dir=Path(output_dir) if output_dir else None,
            classes_txt=Path(classes_txt) if classes_txt else None,
        )
        enabled_operations = config.enabled_operations()
        total_output = result.copied_originals + result.augmented_images
        return {
            "ok": True,
            "summary": f"增强完成: 输出 {total_output} 张（原图 {result.copied_originals} / 增强 {result.augmented_images}）",
            "output_dir": result.output_dir,
            "source_images": result.source_images,
            "copied_originals": result.copied_originals,
            "augmented_images": result.augmented_images,
            "total_output_images": total_output,
            "label_files_written": result.label_files_written,
            "skipped_images": result.skipped_images,
            "mode": mode,
            "enabled_operations": enabled_operations,
            "next_actions": [
                f"可检查增强输出目录: {result.output_dir}",
                "如结果符合预期，再将增强数据纳入训练流程",
            ],
        }
    except Exception as exc:
        return _error_payload(exc, "增强数据集")

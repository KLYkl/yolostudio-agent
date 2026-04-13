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
    _inspect_training_yaml,
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


def _normalize_conversion_target_format(target_format: str) -> tuple[str, bool]:
    normalized = str(target_format or '').strip().lower()
    if normalized in {'xml', 'voc', 'pascal_voc', 'pascal-voc'}:
        return 'xml', True
    if normalized in {'txt', 'yolo', 'yolo_txt', 'yolo-txt'}:
        return 'txt', False
    raise ValueError(f'不支持的 target_format: {target_format}；仅支持 xml / txt')


def _predict_conversion_output_dir(dataset_root: Path, normalized_target_format: str) -> Path:
    data_models = _get_data_models_module()
    base_name = 'converted_labels_xml' if normalized_target_format == 'xml' else 'converted_labels_txt'
    return data_models._get_unique_dir(dataset_root / base_name)


def _resolve_conversion_class_names(
    *,
    to_xml: bool,
    classes: list[str] | None,
    classes_txt: str,
    data_yaml: str,
    img_path: Path,
    label_path: Path | None,
) -> dict[str, Any]:
    detected_classes_txt, classes_txt_candidates = _discover_classes_txt(img_path, label_path)
    detected_data_yaml, data_yaml_candidates = _discover_data_yaml(img_path, label_path)
    warnings: list[str] = []

    explicit_classes = [str(name).strip() for name in (classes or []) if str(name).strip()]
    class_names = explicit_classes
    class_name_source = 'explicit_classes' if explicit_classes else ''

    explicit_classes_txt = Path(classes_txt).resolve() if classes_txt else None
    explicit_data_yaml = Path(data_yaml).resolve() if data_yaml else None
    if explicit_classes_txt and not explicit_classes_txt.exists():
        raise ValueError(f'classes_txt 不存在: {explicit_classes_txt}')
    if explicit_data_yaml and not explicit_data_yaml.exists():
        raise ValueError(f'data_yaml 不存在: {explicit_data_yaml}')

    if not class_names and explicit_classes_txt:
        class_names = _read_classes_txt_lines(explicit_classes_txt)
        class_name_source = 'explicit_classes_txt' if class_names else class_name_source
    if not class_names and detected_classes_txt:
        class_names = _read_classes_txt_lines(Path(detected_classes_txt))
        class_name_source = 'detected_classes_txt' if class_names else class_name_source
    if not class_names and explicit_data_yaml:
        class_names = _read_yaml_names(explicit_data_yaml)
        class_name_source = 'explicit_data_yaml' if class_names else class_name_source
    if not class_names and detected_data_yaml:
        class_names = _read_yaml_names(Path(detected_data_yaml))
        class_name_source = 'detected_data_yaml' if class_names else class_name_source

    if to_xml and not class_names:
        warnings.append('当前未找到 classes.txt 或 data.yaml；TXT 转 XML 时将使用类别 id 字符串写入 XML name 字段')
    if not to_xml and not class_name_source:
        warnings.append('当前未显式提供类别表；XML 转 TXT 时将按 XML 中的类别名排序自动分配类别 id')

    return {
        'class_names': class_names,
        'class_name_source': class_name_source,
        'warnings': warnings,
        'detected_classes_txt': detected_classes_txt,
        'classes_txt_candidates': classes_txt_candidates,
        'detected_data_yaml': detected_data_yaml,
        'data_yaml_candidates': data_yaml_candidates,
    }


def _normalize_modify_action(action: str) -> tuple[str, Any]:
    ModifyAction = _get_data_models_module().ModifyAction
    normalized = str(action or '').strip().lower()
    if normalized in {'replace', 'swap'}:
        return 'replace', ModifyAction.REPLACE
    if normalized in {'remove', 'delete'}:
        return 'remove', ModifyAction.REMOVE
    raise ValueError(f'不支持的 action: {action}；仅支持 replace / remove')


def _resolve_label_class_context(
    *,
    classes_txt: str,
    data_yaml: str,
    img_path: Path,
    label_path: Path | None,
) -> dict[str, Any]:
    detected_classes_txt, classes_txt_candidates = _discover_classes_txt(img_path, label_path)
    detected_data_yaml, data_yaml_candidates = _discover_data_yaml(img_path, label_path)

    explicit_classes_txt = Path(classes_txt).resolve() if classes_txt else None
    explicit_data_yaml = Path(data_yaml).resolve() if data_yaml else None
    if explicit_classes_txt and not explicit_classes_txt.exists():
        raise ValueError(f'classes_txt 不存在: {explicit_classes_txt}')
    if explicit_data_yaml and not explicit_data_yaml.exists():
        raise ValueError(f'data_yaml 不存在: {explicit_data_yaml}')

    class_names: list[str] = []
    class_name_source = ''
    effective_classes_txt = ''
    if explicit_classes_txt:
        class_names = _read_classes_txt_lines(explicit_classes_txt)
        class_name_source = 'explicit_classes_txt' if class_names else class_name_source
        effective_classes_txt = str(explicit_classes_txt)
    elif detected_classes_txt:
        class_names = _read_classes_txt_lines(Path(detected_classes_txt))
        class_name_source = 'detected_classes_txt' if class_names else class_name_source
        effective_classes_txt = detected_classes_txt if class_names else ''
    elif explicit_data_yaml:
        class_names = _read_yaml_names(explicit_data_yaml)
        class_name_source = 'explicit_data_yaml' if class_names else class_name_source
    elif detected_data_yaml:
        class_names = _read_yaml_names(Path(detected_data_yaml))
        class_name_source = 'detected_data_yaml' if class_names else class_name_source

    return {
        'class_names': class_names,
        'class_name_source': class_name_source,
        'effective_classes_txt': effective_classes_txt,
        'detected_classes_txt': detected_classes_txt,
        'classes_txt_candidates': classes_txt_candidates,
        'detected_data_yaml': detected_data_yaml,
        'data_yaml_candidates': data_yaml_candidates,
    }


def _write_temp_classes_txt(class_names: list[str]) -> Path:
    import os
    import tempfile

    fd, temp_name = tempfile.mkstemp(prefix='yolostudio_classes_', suffix='.txt')
    temp_path = Path(temp_name)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(class_names))
        if class_names:
            handle.write('\n')
    return temp_path


def _is_numeric_class_ref(value: str) -> bool:
    return str(value or '').strip().isdigit()


def _normalize_label_format_choice(label_format: str, detected_label_format: Any = None) -> tuple[str, Any]:
    LabelFormat = _get_data_models_module().LabelFormat

    normalized = str(label_format or 'auto').strip().lower()
    detected_normalized = ''
    if hasattr(detected_label_format, 'name'):
        detected_normalized = str(getattr(detected_label_format, 'name', '')).lower()
    elif detected_label_format:
        detected_normalized = str(detected_label_format).strip().lower()

    if normalized in {'', 'auto'}:
        normalized = 'xml' if detected_normalized == 'xml' else 'txt'

    if normalized in {'txt', 'yolo', 'yolo_txt', 'yolo-txt'}:
        return 'txt', LabelFormat.TXT
    if normalized in {'xml', 'voc', 'pascal_voc', 'pascal-voc'}:
        return 'xml', LabelFormat.XML
    raise ValueError(f'不支持的 label_format: {label_format}；仅支持 auto / txt / xml')


def _build_class_mapping(class_names: list[str]) -> dict[int, str]:
    return {index: name for index, name in enumerate(class_names)}


def _find_existing_label_for_image(handler, img_path: Path, img_root: Path, label_root: Path | None) -> tuple[Path | None, Any]:
    if label_root and label_root.exists():
        return handler._find_label_in_dir(img_path, label_root, img_dir=img_root)
    return handler._find_label(img_path, img_root.parent)


def _collect_empty_label_candidates(
    handler,
    *,
    img_root: Path,
    label_root: Path | None,
    only_missing: bool,
) -> dict[str, Any]:
    images = handler._find_images(img_root)
    existing_labels = 0
    candidates: list[Path] = []
    for img_path in images:
        label_path, _ = _find_existing_label_for_image(handler, img_path, img_root, label_root)
        if label_path is not None:
            existing_labels += 1
            if only_missing:
                continue
        candidates.append(img_path)
    return {
        'images': images,
        'candidates': candidates,
        'existing_labels': existing_labels,
    }


def _build_preserved_label_output_path(
    *,
    handler,
    img_path: Path,
    img_root: Path,
    output_dir: Path | None,
    label_root: Path | None,
    ext: str,
) -> Path:
    if output_dir:
        try:
            rel_dir = img_path.parent.relative_to(img_root)
        except ValueError:
            rel_dir = Path()
        return (output_dir / rel_dir / f'{img_path.stem}{ext}').resolve()
    if label_root:
        try:
            rel_dir = img_path.parent.relative_to(img_root)
        except ValueError:
            rel_dir = Path()
        return (label_root / rel_dir / f'{img_path.stem}{ext}').resolve()
    return handler._get_label_output_path(img_path, ext).resolve()


def _create_empty_label_file(handler, *, img_path: Path, label_path: Path, label_format_enum: Any) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_name = getattr(label_format_enum, 'name', str(label_format_enum)).upper()
    if label_name == 'TXT':
        label_path.touch(exist_ok=False)
    else:
        handler._create_empty_xml(img_path, label_path)


def _predict_categorize_output_dir(img_root: Path, output_dir: str = '') -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    data_models = _get_data_models_module()
    return data_models._get_unique_dir(img_root.parent / f'{img_root.name}_categorized')


def preview_convert_format(
    dataset_path: str,
    label_dir: str = '',
    target_format: str = 'xml',
    classes: list[str] | None = None,
    classes_txt: str = '',
    data_yaml: str = '',
) -> dict[str, Any]:
    """预览标签格式转换范围；仅分析会转换哪些标签，不写文件。dataset_path 支持传入 dataset root 或 images/ 目录。"""
    try:
        DataHandler = _get_data_handler_cls()

        normalized_target_format, to_xml = _normalize_conversion_target_format(target_format)
        resolution = resolve_dataset_inputs(dataset_path, label_dir)
        if not resolution.get('ok'):
            return resolution

        img_path = Path(resolution.get('img_dir') or dataset_path)
        label_path = Path(resolution['label_dir']) if resolution.get('label_dir') else None
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path, label_path))
        class_resolution = _resolve_conversion_class_names(
            to_xml=to_xml,
            classes=classes,
            classes_txt=classes_txt,
            data_yaml=data_yaml,
            img_path=img_path,
            label_path=label_path,
        )

        handler = DataHandler()
        preview = handler.preview_convert_format(
            root=dataset_root,
            to_xml=to_xml,
            label_dir=label_path,
        )
        planned_output_dir = _predict_conversion_output_dir(dataset_root, normalized_target_format)
        total_labels = int(preview.get('total_labels', 0) or 0)
        source_type = str(preview.get('source_type') or ('TXT' if to_xml else 'XML'))
        target_type = str(preview.get('target_type') or ('XML' if to_xml else 'TXT'))
        warnings = list(class_resolution['warnings'])
        if total_labels <= 0:
            warnings.append(f'当前未找到可从 {source_type} 转为 {target_type} 的标签文件')

        summary = (
            f"预计可将 {total_labels} 个 {source_type} 标签转换为 {target_type}"
            if total_labels > 0
            else f"未找到可从 {source_type} 转为 {target_type} 的标签文件"
        )
        next_actions: list[str] = []
        if total_labels > 0:
            next_actions.append('如确认执行，可调用 convert_format 正式写出到新目录')
        else:
            next_actions.append('请先确认数据集中是否存在目标源格式标签，或显式指定 label_dir')
        if warnings:
            next_actions.append('建议先看 warnings，避免类别名来源不明确导致输出不符合预期')

        return {
            'ok': True,
            'summary': summary,
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_path.resolve()),
            'resolved_label_dir': str(label_path.resolve()) if label_path else '',
            'source_type': source_type,
            'target_type': target_type,
            'target_format': normalized_target_format,
            'total_labels': total_labels,
            'txt_files': int(preview.get('txt_files', 0) or 0),
            'xml_files': int(preview.get('xml_files', 0) or 0),
            'planned_output_dir': str(planned_output_dir.resolve()),
            'planned_output_dir_name': planned_output_dir.name,
            'class_name_source': class_resolution['class_name_source'],
            'class_count': len(class_resolution['class_names']),
            'detected_classes_txt': class_resolution['detected_classes_txt'],
            'classes_txt_candidates': class_resolution['classes_txt_candidates'],
            'detected_data_yaml': class_resolution['detected_data_yaml'],
            'data_yaml_candidates': class_resolution['data_yaml_candidates'],
            'warnings': warnings,
            'next_actions': next_actions,
        }
    except Exception as exc:
        return _error_payload(exc, '预览标签格式转换')


def convert_format(
    dataset_path: str,
    label_dir: str = '',
    target_format: str = 'xml',
    classes: list[str] | None = None,
    classes_txt: str = '',
    data_yaml: str = '',
) -> dict[str, Any]:
    """执行标签格式转换并写入独立输出目录。默认不会覆盖旧标签，而是写到 dataset root 下的新目录。"""
    try:
        DataHandler = _get_data_handler_cls()

        normalized_target_format, to_xml = _normalize_conversion_target_format(target_format)
        resolution = resolve_dataset_inputs(dataset_path, label_dir)
        if not resolution.get('ok'):
            return resolution

        img_path = Path(resolution.get('img_dir') or dataset_path)
        label_path = Path(resolution['label_dir']) if resolution.get('label_dir') else None
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path, label_path))
        class_resolution = _resolve_conversion_class_names(
            to_xml=to_xml,
            classes=classes,
            classes_txt=classes_txt,
            data_yaml=data_yaml,
            img_path=img_path,
            label_path=label_path,
        )

        handler = DataHandler()
        preview = handler.preview_convert_format(
            root=dataset_root,
            to_xml=to_xml,
            label_dir=label_path,
        )
        total_labels = int(preview.get('total_labels', 0) or 0)
        if total_labels <= 0:
            source_type = str(preview.get('source_type') or ('TXT' if to_xml else 'XML'))
            target_type = str(preview.get('target_type') or ('XML' if to_xml else 'TXT'))
            return {
                'ok': True,
                'summary': f'未找到可从 {source_type} 转为 {target_type} 的标签文件',
                'dataset_root': str(dataset_root.resolve()),
                'resolved_img_dir': str(img_path.resolve()),
                'resolved_label_dir': str(label_path.resolve()) if label_path else '',
                'source_type': source_type,
                'target_type': target_type,
                'target_format': normalized_target_format,
                'converted_count': 0,
                'output_dir': '',
                'warnings': list(class_resolution['warnings']),
                'next_actions': ['请先确认 label_dir 或数据集中的标签格式是否正确'],
            }

        output_dir = _predict_conversion_output_dir(dataset_root, normalized_target_format)
        converted_count = handler.convert_format(
            root=dataset_root,
            to_xml=to_xml,
            classes=class_resolution['class_names'] or None,
            label_dir=label_path,
            image_dir=img_path,
        )
        output_files = sorted(str(path.resolve()) for path in output_dir.rglob('*') if path.is_file())
        output_label_count = sum(1 for path in output_files if Path(path).suffix.lower() == ('.xml' if to_xml else '.txt'))
        summary = (
            f'标签格式转换完成: 成功写出 {converted_count}/{total_labels} 个 {preview.get("target_type", "").upper()} 标签'
        )
        warnings = list(class_resolution['warnings'])
        if converted_count < total_labels:
            warnings.append('部分标签未能成功转换；建议检查输出目录和原始标签内容')

        return {
            'ok': True,
            'summary': summary,
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_path.resolve()),
            'resolved_label_dir': str(label_path.resolve()) if label_path else '',
            'source_type': str(preview.get('source_type') or ('TXT' if to_xml else 'XML')),
            'target_type': str(preview.get('target_type') or ('XML' if to_xml else 'TXT')),
            'target_format': normalized_target_format,
            'converted_count': converted_count,
            'expected_total_labels': total_labels,
            'output_dir': str(output_dir.resolve()),
            'output_label_count': output_label_count,
            'output_sample_files': output_files[:MAX_ISSUE_EXAMPLES],
            'class_name_source': class_resolution['class_name_source'],
            'class_count': len(class_resolution['class_names']),
            'detected_classes_txt': class_resolution['detected_classes_txt'],
            'detected_data_yaml': class_resolution['detected_data_yaml'],
            'warnings': warnings,
            'next_actions': [
                f'建议检查输出目录中的转换结果: {output_dir.resolve()}',
                '如结果符合预期，再将新目录纳入后续训练或治理流程',
            ],
        }
    except Exception as exc:
        return _error_payload(exc, '执行标签格式转换')


def preview_modify_labels(
    dataset_path: str,
    label_dir: str = '',
    action: str = 'replace',
    old_value: str = '',
    new_value: str = '',
    classes_txt: str = '',
    data_yaml: str = '',
) -> dict[str, Any]:
    """预览标签批量修改范围；仅统计会命中的文件和标注数量，不写文件。"""
    temp_classes_txt: Path | None = None
    try:
        if not str(old_value or '').strip():
            raise ValueError('old_value 不能为空')

        DataHandler = _get_data_handler_cls()
        normalized_action, action_enum = _normalize_modify_action(action)
        if normalized_action == 'replace' and not str(new_value or '').strip():
            raise ValueError('replace 模式下 new_value 不能为空')

        resolution = resolve_dataset_inputs(dataset_path, label_dir)
        if not resolution.get('ok'):
            return resolution

        img_path = Path(resolution.get('img_dir') or dataset_path)
        label_path = Path(resolution['label_dir']) if resolution.get('label_dir') else None
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path, label_path))
        class_context = _resolve_label_class_context(
            classes_txt=classes_txt,
            data_yaml=data_yaml,
            img_path=img_path,
            label_path=label_path,
        )

        effective_classes_txt = class_context['effective_classes_txt']
        if not effective_classes_txt and class_context['class_names']:
            temp_classes_txt = _write_temp_classes_txt(class_context['class_names'])
            effective_classes_txt = str(temp_classes_txt)

        handler = DataHandler()
        search_dir = label_path if label_path else dataset_root
        preview = handler.preview_modify_labels(
            search_dir=search_dir,
            action=action_enum,
            old_value=str(old_value).strip(),
            new_value=str(new_value).strip(),
            classes_txt=Path(effective_classes_txt) if effective_classes_txt else None,
            image_dir=img_path,
            label_dir=label_path,
        )
        warnings: list[str] = []
        txt_files = int(preview.get('txt_files', 0) or 0)
        matched_files = int(preview.get('matched_files', 0) or 0)
        matched_annotations = int(preview.get('matched_annotations', 0) or 0)
        if txt_files > 0 and not effective_classes_txt and not _is_numeric_class_ref(str(old_value)):
            warnings.append('当前未找到 classes.txt 或 data.yaml；TXT 标签按类别名匹配时可能命中不到任何项')
        if normalized_action == 'replace' and txt_files > 0 and not _is_numeric_class_ref(str(new_value)):
            if not class_context['class_names']:
                warnings.append('当前未找到类名映射；若目标包含 TXT 标签，new_value 使用名称时执行阶段可能失败')
            elif str(new_value).strip() not in class_context['class_names']:
                raise ValueError(f'new_value 不在当前类别映射中: {new_value}')
        if matched_files <= 0:
            warnings.append('当前未命中任何标签文件；执行阶段不会产生修改')

        action_phrase = '替换' if normalized_action == 'replace' else '删除'
        if matched_files > 0:
            summary = (
                f"预计会在 {matched_files} 个标签文件中{action_phrase} {matched_annotations} 处标注"
            )
        else:
            summary = '未发现可修改的目标标注'

        return {
            'ok': True,
            'summary': summary,
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_path.resolve()),
            'resolved_label_dir': str(label_path.resolve()) if label_path else '',
            'action': normalized_action,
            'old_value': str(old_value).strip(),
            'new_value': str(new_value).strip(),
            'total_label_files': int(preview.get('total_label_files', 0) or 0),
            'txt_files': txt_files,
            'xml_files': int(preview.get('xml_files', 0) or 0),
            'matched_files': matched_files,
            'matched_annotations': matched_annotations,
            'class_name_source': class_context['class_name_source'],
            'class_count': len(class_context['class_names']),
            'detected_classes_txt': class_context['detected_classes_txt'],
            'classes_txt_candidates': class_context['classes_txt_candidates'],
            'detected_data_yaml': class_context['detected_data_yaml'],
            'data_yaml_candidates': class_context['data_yaml_candidates'],
            'warnings': warnings,
            'next_actions': [
                '如确认执行，可调用 modify_labels 正式改写标签文件',
                '建议先核对 matched_files / matched_annotations 是否符合预期',
            ],
        }
    except Exception as exc:
        return _error_payload(exc, '预览标签批量修改')
    finally:
        if temp_classes_txt:
            temp_classes_txt.unlink(missing_ok=True)


def modify_labels(
    dataset_path: str,
    label_dir: str = '',
    action: str = 'replace',
    old_value: str = '',
    new_value: str = '',
    classes_txt: str = '',
    data_yaml: str = '',
    backup: bool = True,
) -> dict[str, Any]:
    """执行标签批量修改。默认原地写回标签，并创建 .bak 备份文件。"""
    temp_classes_txt: Path | None = None
    try:
        if not str(old_value or '').strip():
            raise ValueError('old_value 不能为空')

        DataHandler = _get_data_handler_cls()
        normalized_action, action_enum = _normalize_modify_action(action)
        if normalized_action == 'replace' and not str(new_value or '').strip():
            raise ValueError('replace 模式下 new_value 不能为空')

        resolution = resolve_dataset_inputs(dataset_path, label_dir)
        if not resolution.get('ok'):
            return resolution

        img_path = Path(resolution.get('img_dir') or dataset_path)
        label_path = Path(resolution['label_dir']) if resolution.get('label_dir') else None
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path, label_path))
        class_context = _resolve_label_class_context(
            classes_txt=classes_txt,
            data_yaml=data_yaml,
            img_path=img_path,
            label_path=label_path,
        )

        effective_classes_txt = class_context['effective_classes_txt']
        if not effective_classes_txt and class_context['class_names']:
            temp_classes_txt = _write_temp_classes_txt(class_context['class_names'])
            effective_classes_txt = str(temp_classes_txt)

        handler = DataHandler()
        search_dir = label_path if label_path else dataset_root
        preview = handler.preview_modify_labels(
            search_dir=search_dir,
            action=action_enum,
            old_value=str(old_value).strip(),
            new_value=str(new_value).strip(),
            classes_txt=Path(effective_classes_txt) if effective_classes_txt else None,
            image_dir=img_path,
            label_dir=label_path,
        )
        txt_files = int(preview.get('txt_files', 0) or 0)
        matched_files = int(preview.get('matched_files', 0) or 0)
        matched_annotations = int(preview.get('matched_annotations', 0) or 0)
        if normalized_action == 'replace' and txt_files > 0 and not _is_numeric_class_ref(str(new_value)):
            if not class_context['class_names']:
                raise ValueError('当前未找到 classes.txt 或 data.yaml，无法对 TXT 标签按类别名执行 replace')
            if str(new_value).strip() not in class_context['class_names']:
                raise ValueError(f'new_value 不在当前类别映射中: {new_value}')

        if matched_files <= 0:
            return {
                'ok': True,
                'summary': '未发现可修改的目标标注，本次未写入任何文件',
                'dataset_root': str(dataset_root.resolve()),
                'resolved_img_dir': str(img_path.resolve()),
                'resolved_label_dir': str(label_path.resolve()) if label_path else '',
                'action': normalized_action,
                'old_value': str(old_value).strip(),
                'new_value': str(new_value).strip(),
                'modified_files': 0,
                'expected_matched_files': matched_files,
                'expected_matched_annotations': matched_annotations,
                'backup_enabled': backup,
                'warnings': ['当前未命中任何标签文件'],
                'next_actions': ['建议先调用 preview_modify_labels 确认匹配条件是否正确'],
            }

        modified_count = handler.modify_labels(
            search_dir=search_dir,
            action=action_enum,
            old_value=str(old_value).strip(),
            new_value=str(new_value).strip(),
            backup=backup,
            classes_txt=Path(effective_classes_txt) if effective_classes_txt else None,
            image_dir=img_path,
            label_dir=label_path,
        )
        warnings: list[str] = []
        if not backup:
            warnings.append('本次未创建 .bak 备份，修改不可自动回滚')
        if modified_count < matched_files:
            warnings.append('部分命中标签未成功写回；建议检查标签内容和输出现场')

        action_phrase = '替换' if normalized_action == 'replace' else '删除'
        return {
            'ok': True,
            'summary': f'标签批量修改完成: 已在 {modified_count} 个文件中{action_phrase}目标标注',
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_path.resolve()),
            'resolved_label_dir': str(label_path.resolve()) if label_path else '',
            'action': normalized_action,
            'old_value': str(old_value).strip(),
            'new_value': str(new_value).strip(),
            'modified_files': modified_count,
            'expected_matched_files': matched_files,
            'expected_matched_annotations': matched_annotations,
            'backup_enabled': backup,
            'class_name_source': class_context['class_name_source'],
            'class_count': len(class_context['class_names']),
            'detected_classes_txt': class_context['detected_classes_txt'],
            'detected_data_yaml': class_context['detected_data_yaml'],
            'warnings': warnings,
            'next_actions': [
                '建议重新执行 scan_dataset / validate_dataset，确认修改后的标签结构仍然可用',
            ],
        }
    except Exception as exc:
        return _error_payload(exc, '执行标签批量修改')
    finally:
        if temp_classes_txt:
            temp_classes_txt.unlink(missing_ok=True)


def clean_orphan_labels(
    dataset_path: str,
    label_dir: str = '',
    backup: bool = True,
    dry_run: bool = True,
) -> dict[str, Any]:
    """检测并清理孤儿标签。默认 dry_run=true，仅返回待清理范围；实际删除需显式 dry_run=false。"""
    try:
        DataHandler = _get_data_handler_cls()

        resolution = resolve_dataset_inputs(dataset_path, label_dir)
        if not resolution.get('ok'):
            return resolution

        img_path = Path(resolution.get('img_dir') or dataset_path)
        label_path = Path(resolution['label_dir']) if resolution.get('label_dir') else None
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path, label_path))

        handler = DataHandler()
        validate_result = handler.validate_labels(
            img_dir=img_path,
            label_dir=label_path,
            check_coords=False,
            check_class_ids=False,
            check_format=False,
            check_orphans=True,
        )
        orphan_labels = list(validate_result.orphan_labels)
        orphan_count = len(orphan_labels)
        sample_paths = [str(path) for path in orphan_labels[:MAX_ISSUE_EXAMPLES]]

        if dry_run or orphan_count <= 0:
            summary = (
                f'检测到 {orphan_count} 个孤儿标签文件，当前仅预览未删除'
                if orphan_count > 0
                else '未检测到孤儿标签文件'
            )
            next_actions = ['如确认删除，可调用 clean_orphan_labels 并传 dry_run=false']
            if orphan_count <= 0:
                next_actions = ['当前无需清理，可继续做 validate_dataset 或训练准备']
            return {
                'ok': True,
                'summary': summary,
                'dataset_root': str(dataset_root.resolve()),
                'structure_type': resolution.get('structure_type'),
                'resolved_from_root': resolution.get('resolved_from_root', False),
                'resolved_img_dir': str(img_path.resolve()),
                'resolved_label_dir': str(label_path.resolve()) if label_path else '',
                'dry_run': dry_run,
                'backup_enabled': backup,
                'orphan_count': orphan_count,
                'orphan_samples': sample_paths,
                'warnings': ['实际删除前建议先确认 sample 列表'] if orphan_count > 0 else [],
                'next_actions': next_actions,
            }

        cleaned_count = handler.clean_orphan_labels(orphan_labels=orphan_labels, backup=backup)
        warnings = []
        if not backup:
            warnings.append('本次未创建 .bak 备份，删除不可自动回滚')
        if cleaned_count < orphan_count:
            warnings.append('部分孤儿标签未能成功删除；建议检查文件权限或路径状态')
        return {
            'ok': True,
            'summary': f'孤儿标签清理完成: 已删除 {cleaned_count}/{orphan_count} 个文件',
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_path.resolve()),
            'resolved_label_dir': str(label_path.resolve()) if label_path else '',
            'dry_run': False,
            'backup_enabled': backup,
            'orphan_count': orphan_count,
            'cleaned_count': cleaned_count,
            'orphan_samples': sample_paths,
            'warnings': warnings,
            'next_actions': [
                '建议重新执行 validate_dataset，确认孤儿标签已清理干净',
            ],
        }
    except Exception as exc:
        return _error_payload(exc, '清理孤儿标签')


def preview_generate_empty_labels(
    dataset_path: str,
    label_dir: str = '',
    label_format: str = 'auto',
    output_dir: str = '',
    only_missing: bool = True,
) -> dict[str, Any]:
    """预览空标签生成范围。默认 only_missing=true，只为当前缺少标签的图片生成空标签。"""
    try:
        DataHandler = _get_data_handler_cls()

        resolution = resolve_dataset_inputs(dataset_path, label_dir)
        if not resolution.get('ok'):
            return resolution

        img_path = Path(resolution.get('img_dir') or dataset_path)
        label_path = Path(resolution['label_dir']) if resolution.get('label_dir') else None
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path, label_path))

        handler = DataHandler()
        scan_result = handler.scan_dataset(img_path, label_dir=label_path)
        normalized_format, label_format_enum = _normalize_label_format_choice(label_format, scan_result.label_format)
        ext = '.txt' if normalized_format == 'txt' else '.xml'
        output_root = Path(output_dir).resolve() if output_dir else None

        candidate_info = _collect_empty_label_candidates(
            handler,
            img_root=img_path,
            label_root=label_path,
            only_missing=only_missing,
        )
        candidate_images = candidate_info['candidates']
        sample_outputs = [
            str(_build_preserved_label_output_path(
                handler=handler,
                img_path=img,
                img_root=img_path,
                output_dir=output_root,
                label_root=label_path,
                ext=ext,
            ))
            for img in candidate_images[:MAX_ISSUE_EXAMPLES]
        ]
        warnings: list[str] = []
        if normalized_format == 'xml':
            warnings.append('XML 空标签会写入带图片尺寸信息的 VOC annotation 文件')
        if output_root:
            warnings.append('本次会写入指定 output_dir，并保留图片相对子目录结构')

        planned_count = len(candidate_images)
        summary = (
            f'预计生成 {planned_count} 个 {normalized_format.upper()} 空标签文件'
            if planned_count > 0
            else '当前没有需要生成的空标签文件'
        )
        return {
            'ok': True,
            'summary': summary,
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_path.resolve()),
            'resolved_label_dir': str(label_path.resolve()) if label_path else '',
            'label_format': normalized_format,
            'only_missing': only_missing,
            'total_images': len(candidate_info['images']),
            'existing_label_images': candidate_info['existing_labels'],
            'planned_generate_count': planned_count,
            'planned_output_root': str((output_root or (label_path if label_path else dataset_root)).resolve()) if (output_root or label_path or dataset_root) else '',
            'planned_output_samples': sample_outputs,
            'warnings': warnings,
            'next_actions': [
                '如确认执行，可调用 generate_empty_labels 正式写入空标签',
            ],
        }
    except Exception as exc:
        return _error_payload(exc, '预览空标签生成')


def generate_empty_labels(
    dataset_path: str,
    label_dir: str = '',
    label_format: str = 'auto',
    output_dir: str = '',
    only_missing: bool = True,
) -> dict[str, Any]:
    """为图片生成空标签文件。默认 only_missing=true，只补齐当前缺失标签。"""
    try:
        DataHandler = _get_data_handler_cls()

        resolution = resolve_dataset_inputs(dataset_path, label_dir)
        if not resolution.get('ok'):
            return resolution

        img_path = Path(resolution.get('img_dir') or dataset_path)
        label_path = Path(resolution['label_dir']) if resolution.get('label_dir') else None
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path, label_path))
        handler = DataHandler()
        scan_result = handler.scan_dataset(img_path, label_dir=label_path)
        normalized_format, label_format_enum = _normalize_label_format_choice(label_format, scan_result.label_format)
        ext = '.txt' if normalized_format == 'txt' else '.xml'
        output_root = Path(output_dir).resolve() if output_dir else None

        candidate_info = _collect_empty_label_candidates(
            handler,
            img_root=img_path,
            label_root=label_path,
            only_missing=only_missing,
        )
        candidate_images = candidate_info['candidates']
        if not candidate_images:
            return {
                'ok': True,
                'summary': '当前没有需要生成的空标签文件',
                'dataset_root': str(dataset_root.resolve()),
                'resolved_img_dir': str(img_path.resolve()),
                'resolved_label_dir': str(label_path.resolve()) if label_path else '',
                'label_format': normalized_format,
                'generated_count': 0,
                'skipped_existing': candidate_info['existing_labels'],
                'warnings': [],
                'next_actions': ['当前无需生成，可继续做 validate_dataset 或训练准备'],
            }

        generated_count = 0
        sample_outputs: list[str] = []
        for img in candidate_images:
            label_output = _build_preserved_label_output_path(
                handler=handler,
                img_path=img,
                img_root=img_path,
                output_dir=output_root,
                label_root=label_path,
                ext=ext,
            )
            if label_output.exists():
                continue
            _create_empty_label_file(handler, img_path=img, label_path=label_output, label_format_enum=label_format_enum)
            generated_count += 1
            if len(sample_outputs) < MAX_ISSUE_EXAMPLES:
                sample_outputs.append(str(label_output))

        warnings: list[str] = []
        if output_root:
            warnings.append('标签已写入指定 output_dir；请确认后续流程使用新的标签目录')
        return {
            'ok': True,
            'summary': f'空标签生成完成: 已写入 {generated_count} 个 {normalized_format.upper()} 标签文件',
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_path.resolve()),
            'resolved_label_dir': str(label_path.resolve()) if label_path else '',
            'label_format': normalized_format,
            'only_missing': only_missing,
            'generated_count': generated_count,
            'skipped_existing': candidate_info['existing_labels'],
            'output_root': str((output_root or (label_path if label_path else dataset_root)).resolve()),
            'output_samples': sample_outputs,
            'warnings': warnings,
            'next_actions': [
                '建议重新执行 scan_dataset / validate_dataset，确认新标签已被识别',
            ],
        }
    except Exception as exc:
        return _error_payload(exc, '生成空标签')


def preview_generate_missing_labels(
    dataset_path: str,
    label_dir: str = '',
    label_format: str = 'auto',
    output_dir: str = '',
) -> dict[str, Any]:
    """预览缺失标签补齐范围。"""
    result = preview_generate_empty_labels(
        dataset_path=dataset_path,
        label_dir=label_dir,
        label_format=label_format,
        output_dir=output_dir,
        only_missing=True,
    )
    if result.get('ok'):
        planned = int(result.get('planned_generate_count', 0) or 0)
        result['summary'] = (
            f'预计补齐 {planned} 张缺失标签图片'
            if planned > 0
            else '当前没有缺失标签图片需要补齐'
        )
    return result


def generate_missing_labels(
    dataset_path: str,
    label_dir: str = '',
    label_format: str = 'auto',
    output_dir: str = '',
) -> dict[str, Any]:
    """补齐当前数据集中缺失标签的图片。"""
    result = generate_empty_labels(
        dataset_path=dataset_path,
        label_dir=label_dir,
        label_format=label_format,
        output_dir=output_dir,
        only_missing=True,
    )
    if result.get('ok'):
        generated = int(result.get('generated_count', 0) or 0)
        result['summary'] = (
            f'缺失标签补齐完成: 已生成 {generated} 个空标签文件'
            if generated > 0
            else '当前没有缺失标签图片需要补齐'
        )
    return result


def preview_categorize_by_class(
    dataset_path: str,
    label_dir: str = '',
    output_dir: str = '',
    classes_txt: str = '',
    data_yaml: str = '',
    include_no_label: bool = True,
) -> dict[str, Any]:
    """预览按类别整理数据的分桶结果；仅统计分类去向，不复制文件。"""
    temp_classes_txt: Path | None = None
    try:
        DataHandler = _get_data_handler_cls()

        resolution = resolve_dataset_inputs(dataset_path, label_dir)
        if not resolution.get('ok'):
            return resolution

        img_path = Path(resolution.get('img_dir') or dataset_path)
        label_path = Path(resolution['label_dir']) if resolution.get('label_dir') else None
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path, label_path))
        class_context = _resolve_label_class_context(
            classes_txt=classes_txt,
            data_yaml=data_yaml,
            img_path=img_path,
            label_path=label_path,
        )
        effective_classes_txt = class_context['effective_classes_txt']
        class_mapping = _build_class_mapping(class_context['class_names'])
        if not effective_classes_txt and class_context['class_names']:
            temp_classes_txt = _write_temp_classes_txt(class_context['class_names'])
            effective_classes_txt = str(temp_classes_txt)

        handler = DataHandler()
        images = handler._find_images(img_path)
        category_stats: dict[str, int] = {}
        mixed_samples: dict[str, list[str]] = {}
        for img in images:
            label_file, label_format_enum = _find_existing_label_for_image(handler, img, img_path, label_path)
            if label_file is None:
                if not include_no_label:
                    continue
                category = '_no_label'
            else:
                class_ids = handler._parse_label(label_file, label_format_enum, class_mapping=class_mapping)
                unique_ids = sorted(set(class_ids))
                if len(unique_ids) == 0:
                    category = '_empty'
                elif len(unique_ids) == 1:
                    category = unique_ids[0]
                else:
                    category = '_mixed'
                    if len(mixed_samples) < MAX_ISSUE_EXAMPLES:
                        mixed_samples[img.name] = unique_ids
            category_stats[category] = category_stats.get(category, 0) + 1

        planned_output_dir = _predict_categorize_output_dir(img_path, output_dir)
        summary = (
            f'预计会把 {sum(category_stats.values())} 张图片整理到 {len(category_stats)} 个类别桶'
            if category_stats
            else '当前没有可整理的图片'
        )
        return {
            'ok': True,
            'summary': summary,
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_path.resolve()),
            'resolved_label_dir': str(label_path.resolve()) if label_path else '',
            'include_no_label': include_no_label,
            'planned_output_dir': str(planned_output_dir.resolve()),
            'category_stats': dict(sorted(category_stats.items())),
            'category_count': len(category_stats),
            'mixed_samples': mixed_samples,
            'class_name_source': class_context['class_name_source'],
            'class_count': len(class_context['class_names']),
            'detected_classes_txt': class_context['detected_classes_txt'],
            'detected_data_yaml': class_context['detected_data_yaml'],
            'warnings': ['本操作会复制图片和标签到新目录，不会改动原始数据'],
            'next_actions': ['如确认执行，可调用 categorize_by_class 正式写出分类目录'],
        }
    except Exception as exc:
        return _error_payload(exc, '预览按类别整理数据')
    finally:
        if temp_classes_txt:
            temp_classes_txt.unlink(missing_ok=True)


def categorize_by_class(
    dataset_path: str,
    label_dir: str = '',
    output_dir: str = '',
    classes_txt: str = '',
    data_yaml: str = '',
    include_no_label: bool = True,
) -> dict[str, Any]:
    """按类别复制整理图片和标签到新目录。"""
    temp_classes_txt: Path | None = None
    try:
        DataHandler = _get_data_handler_cls()

        resolution = resolve_dataset_inputs(dataset_path, label_dir)
        if not resolution.get('ok'):
            return resolution

        img_path = Path(resolution.get('img_dir') or dataset_path)
        label_path = Path(resolution['label_dir']) if resolution.get('label_dir') else None
        dataset_root = Path(resolution.get('dataset_root') or _infer_dataset_root(img_path, label_path))
        class_context = _resolve_label_class_context(
            classes_txt=classes_txt,
            data_yaml=data_yaml,
            img_path=img_path,
            label_path=label_path,
        )
        effective_classes_txt = class_context['effective_classes_txt']
        if not effective_classes_txt and class_context['class_names']:
            temp_classes_txt = _write_temp_classes_txt(class_context['class_names'])
            effective_classes_txt = str(temp_classes_txt)

        handler = DataHandler()
        resolved_output_dir = _predict_categorize_output_dir(img_path, output_dir)
        category_stats = handler.categorize_by_class(
            img_dir=img_path,
            label_dir=label_path,
            output_dir=resolved_output_dir,
            classes_txt=Path(effective_classes_txt) if effective_classes_txt else None,
            include_no_label=include_no_label,
        )
        mixed_report = resolved_output_dir / '_mixed_report.txt'
        return {
            'ok': True,
            'summary': f'按类别整理完成: 共整理 {sum(category_stats.values())} 张图片到 {len(category_stats)} 个类别桶',
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_path.resolve()),
            'resolved_label_dir': str(label_path.resolve()) if label_path else '',
            'include_no_label': include_no_label,
            'output_dir': str(resolved_output_dir.resolve()),
            'category_stats': dict(sorted(category_stats.items())),
            'category_count': len(category_stats),
            'mixed_report_path': str(mixed_report.resolve()) if mixed_report.exists() else '',
            'class_name_source': class_context['class_name_source'],
            'class_count': len(class_context['class_names']),
            'detected_classes_txt': class_context['detected_classes_txt'],
            'detected_data_yaml': class_context['detected_data_yaml'],
            'warnings': ['本操作会复制数据到新目录，原始数据不会被改写'],
            'next_actions': [
                f'建议检查整理输出目录: {resolved_output_dir.resolve()}',
            ],
        }
    except Exception as exc:
        return _error_payload(exc, '按类别整理数据')
    finally:
        if temp_classes_txt:
            temp_classes_txt.unlink(missing_ok=True)


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
        yaml_check = _inspect_training_yaml(resolved_yaml) if yaml_exists else {
            'exists': False,
            'usable': False,
            'yaml_path': '',
            'issues': [],
            'resolved_targets': {},
            'warnings': [],
        }
        yaml_usable = bool(yaml_check.get('usable'))
        labels_clean = not validate.get('has_issues', False)
        labeled_images = int(scan.get('labeled_images', 0) or 0)
        total_images = int(scan.get('total_images', 0) or 0)
        hard_label_block = total_images > 0 and labeled_images <= 0
        gpu_info = query_gpu_status()
        available_gpus = [gpu.index for gpu in gpu_info if not gpu.busy]
        device_policy = get_effective_gpu_policy()
        auto_device, auto_error = resolve_auto_device(policy=device_policy, gpus=gpu_info)
        warnings = list(validate.get('warnings', []))
        warnings.extend(str(item) for item in (yaml_check.get('warnings') or []) if str(item))
        risk_level = validate.get('risk_level', scan.get('risk_level', 'none'))
        ready = yaml_usable and (labels_clean or not require_clean_labels) and bool(available_gpus) and not hard_label_block

        primary_blocker_type = ''
        if hard_label_block:
            primary_blocker_type = 'no_valid_labels'
        elif require_clean_labels and not labels_clean:
            primary_blocker_type = 'label_issues'
        elif yaml_exists and not yaml_usable:
            primary_blocker_type = 'invalid_yaml'
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
        if yaml_exists and not yaml_usable:
            blockers.append('当前 data_yaml 内的 train/val 路径不可用')
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
        elif primary_blocker_type == 'invalid_yaml' and preparable:
            next_actions.append({
                'description': '当前 data.yaml 的 train/val 路径在当前机器不可用；建议先调用 prepare_dataset_for_training 重新生成可训练 YAML',
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

        if (not yaml_exists or (yaml_exists and not yaml_usable)) and not any(action.get('tool') == 'prepare_dataset_for_training' for action in next_actions):
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
        if not ready and primary_blocker_type == 'invalid_yaml' and preparable:
            summary += '；但当前数据集可以先进入 prepare_dataset_for_training 重新生成远端可用 YAML'
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
            'data_yaml_usable': yaml_usable,
            'data_yaml_issues': yaml_check.get('issues', []),
            'data_yaml_targets': yaml_check.get('resolved_targets', {}),
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

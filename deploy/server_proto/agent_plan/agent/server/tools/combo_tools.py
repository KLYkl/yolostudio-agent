from __future__ import annotations

from pathlib import Path
from typing import Any

from yolostudio_agent.agent.server.services.dataset_root import resolve_dataset_root
from yolostudio_agent.agent.server.tools.data_tool_helpers import _inspect_training_yaml
from yolostudio_agent.agent.server.tools.data_tools import generate_yaml, scan_dataset, split_dataset, training_readiness, validate_dataset

_EARLY_BLOCK_TYPES = {'unknown', 'images_only', 'flat'}


def _tool_candidate(*, tool: str, reason: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {'kind': 'tool_call', 'tool': tool, 'reason': reason}
    if args:
        payload['args'] = args
    return payload


def _action_candidates_from_next_actions(next_actions: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in next_actions or []:
        if not isinstance(item, dict):
            continue
        tool = str(item.get('tool') or '').strip()
        if not tool:
            continue
        candidates.append(_tool_candidate(
            tool=tool,
            reason=str(item.get('description') or '').strip() or tool,
            args=dict(item.get('args_hint') or {}),
        ))
    return candidates


def prepare_dataset_for_training(
    dataset_path: str,
    split_ratio: float = 0.8,
    force_split: bool = False,
    classes_txt: str = '',
    classes_text: str = '',
) -> dict[str, Any]:
    """将数据集准备到可训练状态：解析根目录、扫描、校验、按需划分并生成 YAML。

注意：本工具只做数据准备，不会启动训练；若用户最终目标是训练，Agent 应在 ready=true 后继续调用 start_training。"""
    resolution = resolve_dataset_root(dataset_path)
    if not resolution.get('ok'):
        return resolution

    dataset_root = resolution.get('dataset_root', dataset_path)
    img_dir = resolution.get('img_dir', dataset_path)
    label_dir = resolution.get('label_dir', '')
    steps_completed: list[dict[str, Any]] = [
        {
            'step': 'resolve_root',
            'ok': True,
            'dataset_root': dataset_root,
            'img_dir': img_dir,
            'label_dir': label_dir,
            'structure_type': resolution.get('structure_type'),
            'resolution_method': resolution.get('resolution_method'),
            'summary': resolution.get('summary', ''),
        }
    ]

    if resolution.get('structure_type') in _EARLY_BLOCK_TYPES:
        return {
            'ok': False,
            'summary': '准备失败：当前目录结构还不足以安全进入自动准备流程',
            'blocked_at': 'resolve_root',
            'dataset_root': dataset_root,
            'img_dir': img_dir,
            'label_dir': label_dir,
            'steps_completed': steps_completed,
            'next_actions': resolution.get('next_actions', []) or [
                '请显式提供 img_dir / label_dir，或将目录整理为标准 YOLO 结构后重试',
            ],
        }

    scan = scan_dataset(img_dir=img_dir, label_dir=label_dir)
    steps_completed.append({'step': 'scan', **scan})
    if not scan.get('ok'):
        return {
            'ok': False,
            'summary': '准备失败：scan_dataset 执行失败',
            'blocked_at': 'scan',
            'steps_completed': steps_completed,
            'next_actions': ['请先修复扫描错误，再继续准备数据集'],
        }

    validate = validate_dataset(
        img_dir=img_dir,
        label_dir=label_dir,
        classes_txt=classes_txt or scan.get('detected_classes_txt', ''),
    )
    steps_completed.append({'step': 'validate', **validate})
    if not validate.get('ok'):
        return {
            'ok': False,
            'summary': '准备失败：validate_dataset 执行失败',
            'blocked_at': 'validate',
            'steps_completed': steps_completed,
            'next_actions': ['请先修复标签校验错误，再继续准备数据集'],
        }

    detected_yaml = scan.get('detected_data_yaml') or ''
    yaml_check = _inspect_training_yaml(detected_yaml) if detected_yaml and Path(detected_yaml).exists() else {
        'exists': False,
        'usable': False,
        'issues': [],
    }
    has_yaml = bool(detected_yaml and Path(detected_yaml).exists() and yaml_check.get('usable'))
    should_split = force_split or (not has_yaml and not resolution.get('is_split', False))
    should_regenerate_yaml = bool(not has_yaml and resolution.get('is_split', False))

    generated_yaml = detected_yaml
    split_reason = 'user_requested' if force_split else ('missing_yaml' if should_split else ('already_split' if resolution.get('is_split', False) else 'existing_yaml'))
    data_yaml_source = 'detected_existing_yaml' if generated_yaml else ''
    if should_split:
        split_result = split_dataset(
            img_dir=img_dir,
            label_dir=label_dir,
            ratio=split_ratio,
        )
        steps_completed.append({'step': 'split', **split_result})
        if not split_result.get('ok'):
            return {
                'ok': False,
                'summary': '准备失败：split_dataset 执行失败',
                'blocked_at': 'split',
                'steps_completed': steps_completed,
                'next_actions': ['请检查数据集是否适合划分，或显式指定已准备好的 data.yaml'],
            }

        split_train_path = str(
            split_result.get('resolved_train_path')
            or split_result.get('train_path')
            or ''
        ).strip()
        split_val_path = str(
            split_result.get('resolved_val_path')
            or split_result.get('val_path')
            or ''
        ).strip()
        split_output_dir = str(split_result.get('output_dir') or '').strip()

        yaml_result = generate_yaml(
            train_path=split_train_path,
            val_path=split_val_path,
            classes=scan.get('classes', []),
            classes_text=classes_text,
            classes_txt=classes_txt or scan.get('detected_classes_txt', ''),
            img_dir=split_output_dir,
            output_path=split_result.get('suggested_yaml_path', ''),
        )
        steps_completed.append({'step': 'generate_yaml', **yaml_result})
        if not yaml_result.get('ok'):
            return {
                'ok': False,
                'summary': '准备失败：generate_yaml 执行失败',
                'blocked_at': 'generate_yaml',
                'steps_completed': steps_completed,
                'next_actions': ['请检查 split 结果路径和 classes / classes.txt 信息'],
            }
        generated_yaml = yaml_result.get('output_path', generated_yaml)
        data_yaml_source = 'generated_from_split'
    elif should_regenerate_yaml:
        split_info = dict(resolution.get('split_info') or {})
        train_path = str(split_info.get('train_img_dir') or '')
        val_path = str(split_info.get('val_img_dir') or '')
        if not train_path or not val_path:
            return {
                'ok': False,
                'summary': '准备失败：当前已识别为 split 数据集，但无法解析 train/val 路径来重建 YAML',
                'blocked_at': 'generate_yaml',
                'dataset_root': dataset_root,
                'img_dir': img_dir,
                'label_dir': label_dir,
                'steps_completed': steps_completed,
                'next_actions': ['请显式提供可用的 data.yaml，或检查 images/train / images/val 结构'],
            }

        yaml_result = generate_yaml(
            train_path=train_path,
            val_path=val_path,
            classes=scan.get('classes', []),
            classes_text=classes_text,
            classes_txt=classes_txt or scan.get('detected_classes_txt', ''),
            img_dir=img_dir,
            label_dir=label_dir,
            output_path=str((Path(dataset_root) / 'data.yaml').resolve()),
        )
        steps_completed.append({'step': 'generate_yaml', **yaml_result})
        if not yaml_result.get('ok'):
            return {
                'ok': False,
                'summary': '准备失败：generate_yaml 执行失败',
                'blocked_at': 'generate_yaml',
                'steps_completed': steps_completed,
                'next_actions': ['请检查 split 结果路径和 classes / classes.txt 信息'],
            }
        generated_yaml = yaml_result.get('output_path', generated_yaml)
        data_yaml_source = 'regenerated_from_split'

    readiness = training_readiness(
        img_dir=img_dir,
        label_dir=label_dir,
        data_yaml=generated_yaml,
    )
    steps_completed.append({'step': 'readiness', **readiness})

    ready = readiness.get('ready', False)
    warnings = readiness.get('warnings', [])
    summary = '数据集已准备到可训练状态，尚未启动训练' if ready else readiness.get('summary', '数据集尚未准备完成')
    if ready and warnings:
        summary = f"数据集已准备到可训练状态，但存在数据质量风险：{'；'.join(warnings)}"

    return {
        'ok': readiness.get('ok', False),
        'summary': summary,
        'prepare_overview': {
            'ready': ready,
            'blocked_at': None if ready else 'readiness',
            'dataset_root': dataset_root,
            'data_yaml': generated_yaml,
            'data_yaml_source': data_yaml_source,
            'force_split_applied': bool(should_split),
            'split_reason': split_reason,
            'risk_level': readiness.get('risk_level', validate.get('risk_level', 'none')),
            'warning_count': len(warnings),
            'action_count': len([step['step'] for step in steps_completed if step.get('ok')]),
        },
        'dataset_root': dataset_root,
        'img_dir': img_dir,
        'label_dir': label_dir,
        'data_yaml': generated_yaml,
        'data_yaml_source': data_yaml_source,
        'detected_classes_txt': scan.get('detected_classes_txt', ''),
        'effective_classes_txt': classes_txt or scan.get('detected_classes_txt', ''),
        'class_name_source': scan.get('class_name_source', ''),
        'risk_level': readiness.get('risk_level', validate.get('risk_level', 'none')),
        'warnings': warnings,
        'missing_label_images': readiness.get('missing_label_images', validate.get('missing_label_images', 0)),
        'missing_label_ratio': readiness.get('missing_label_ratio', validate.get('missing_label_ratio', 0.0)),
        'ready': ready,
        'force_split_applied': bool(should_split),
        'split_reason': split_reason,
        'recommended_start_training_args': readiness.get('recommended_start_training_args', {'data_yaml': generated_yaml} if ready and generated_yaml else {}),
        'blocked_at': None if ready else 'readiness',
        'actions_taken': [step['step'] for step in steps_completed if step.get('ok')],
        'steps_completed': steps_completed,
        'action_candidates': _action_candidates_from_next_actions(readiness.get('next_actions', [])),
        'next_actions': readiness.get('next_actions', []),
    }

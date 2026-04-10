from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_plan.agent.server.services.dataset_root import resolve_dataset_root
from agent_plan.agent.server.tools.data_tools import generate_yaml, scan_dataset, split_dataset, training_readiness, validate_dataset


def prepare_dataset_for_training(
    dataset_path: str,
    split_ratio: float = 0.8,
    force_split: bool = False,
) -> dict[str, Any]:
    """将数据集准备到可训练状态：解析根目录、扫描、校验、按需划分并生成 YAML。"""
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
            'summary': resolution.get('summary', ''),
        }
    ]

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

    validate = validate_dataset(img_dir=img_dir, label_dir=label_dir)
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
    has_yaml = bool(detected_yaml and Path(detected_yaml).exists())
    should_split = force_split or (not has_yaml and not resolution.get('is_split', False))

    generated_yaml = detected_yaml
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

        yaml_result = generate_yaml(
            train_path=split_result.get('train_path', ''),
            val_path=split_result.get('val_path', ''),
            classes=scan.get('classes', []),
            output_path=split_result.get('suggested_yaml_path', ''),
        )
        steps_completed.append({'step': 'generate_yaml', **yaml_result})
        if not yaml_result.get('ok'):
            return {
                'ok': False,
                'summary': '准备失败：generate_yaml 执行失败',
                'blocked_at': 'generate_yaml',
                'steps_completed': steps_completed,
                'next_actions': ['请检查 split 结果路径和 classes 信息'],
            }
        generated_yaml = yaml_result.get('output_path', generated_yaml)

    readiness = training_readiness(
        img_dir=img_dir,
        label_dir=label_dir,
        data_yaml=generated_yaml,
    )
    steps_completed.append({'step': 'readiness', **readiness})

    return {
        'ok': readiness.get('ok', False),
        'summary': '数据集已准备到可训练状态' if readiness.get('ready') else readiness.get('summary', '数据集尚未准备完成'),
        'dataset_root': dataset_root,
        'img_dir': img_dir,
        'label_dir': label_dir,
        'data_yaml': generated_yaml,
        'ready': readiness.get('ready', False),
        'blocked_at': None if readiness.get('ready') else 'readiness',
        'actions_taken': [step['step'] for step in steps_completed if step.get('ok')],
        'steps_completed': steps_completed,
        'next_actions': readiness.get('next_actions', []),
    }

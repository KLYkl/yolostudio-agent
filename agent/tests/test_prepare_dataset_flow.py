from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import agent_plan.agent.server.tools.combo_tools as combo_tools


def _run_success_path() -> None:
    steps: list[str] = []

    def fake_resolve_dataset_root(dataset_path: str):
        steps.append('resolve_root')
        return {
            'ok': True,
            'dataset_root': dataset_path,
            'img_dir': f'{dataset_path}/images',
            'label_dir': f'{dataset_path}/labels',
            'summary': 'ok',
            'is_split': False,
            'structure_type': 'yolo_standard',
            'resolution_method': 'name',
        }

    def fake_scan_dataset(img_dir: str, label_dir: str = ''):
        steps.append('scan')
        return {
            'ok': True,
            'summary': 'scan ok',
            'classes': ['cat', 'dog'],
            'detected_data_yaml': '',
            'total_images': 10,
            'labeled_images': 10,
            'missing_labels': 0,
            'empty_labels': 0,
        }

    def fake_validate_dataset(img_dir: str, label_dir: str = ''):
        steps.append('validate')
        return {
            'ok': True,
            'summary': 'validate ok',
            'has_issues': False,
            'issue_count': 0,
        }

    def fake_split_dataset(img_dir: str, label_dir: str = '', ratio: float = 0.8):
        steps.append('split')
        assert ratio == 0.8
        return {
            'ok': True,
            'summary': 'split ok',
            'train_path': '/tmp/images/train',
            'val_path': '/tmp/images/val',
            'train_count': 8,
            'val_count': 2,
            'output_dir': '/tmp',
            'suggested_yaml_path': '/tmp/data.yaml',
        }

    def fake_generate_yaml(train_path: str, val_path: str, classes: list[str] | None = None, output_path: str = ''):
        steps.append('generate_yaml')
        assert classes == ['cat', 'dog']
        assert output_path == '/tmp/data.yaml'
        return {
            'ok': True,
            'summary': 'yaml ok',
            'output_path': '/tmp/data.yaml',
        }

    def fake_training_readiness(img_dir: str, label_dir: str = '', data_yaml: str = ''):
        steps.append('readiness')
        assert data_yaml == '/tmp/data.yaml'
        return {
            'ok': True,
            'ready': True,
            'summary': '可以直接训练',
            'next_actions': [{'tool': 'start_training'}],
        }

    combo_tools.resolve_dataset_root = fake_resolve_dataset_root
    combo_tools.scan_dataset = fake_scan_dataset
    combo_tools.validate_dataset = fake_validate_dataset
    combo_tools.split_dataset = fake_split_dataset
    combo_tools.generate_yaml = fake_generate_yaml
    combo_tools.training_readiness = fake_training_readiness

    result = combo_tools.prepare_dataset_for_training('/dataset')
    assert result['ok'] is True
    assert result['ready'] is True
    assert result['data_yaml'] == '/tmp/data.yaml'
    assert steps == ['resolve_root', 'scan', 'validate', 'split', 'generate_yaml', 'readiness']


def _run_early_block_path() -> None:
    steps: list[str] = []

    def fake_resolve_dataset_root(dataset_path: str):
        steps.append('resolve_root')
        return {
            'ok': True,
            'dataset_root': dataset_path,
            'img_dir': f'{dataset_path}/pics',
            'label_dir': '',
            'summary': '只识别到图片目录，未找到标签目录',
            'is_split': False,
            'structure_type': 'images_only',
            'resolution_method': 'image=content_score,label=none',
            'next_actions': ['请显式提供 label_dir'],
        }

    def fail_if_called(*args, **kwargs):
        raise AssertionError('early-block path should stop before downstream tools')

    combo_tools.resolve_dataset_root = fake_resolve_dataset_root
    combo_tools.scan_dataset = fail_if_called
    combo_tools.validate_dataset = fail_if_called
    combo_tools.split_dataset = fail_if_called
    combo_tools.generate_yaml = fail_if_called
    combo_tools.training_readiness = fail_if_called

    result = combo_tools.prepare_dataset_for_training('/dataset')
    assert result['ok'] is False
    assert result['blocked_at'] == 'resolve_root'
    assert result['steps_completed'][0]['step'] == 'resolve_root'
    assert steps == ['resolve_root']


def main() -> None:
    _run_success_path()
    _run_early_block_path()
    print('prepare dataset flow smoke ok')


if __name__ == '__main__':
    main()

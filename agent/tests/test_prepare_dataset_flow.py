from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

import yolostudio_agent.agent.server.tools.combo_tools as combo_tools


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
            'detected_classes_txt': f'{img_dir}/../labels/classes.txt',
            'class_name_source': 'classes_txt',
            'detected_data_yaml': '',
            'total_images': 10,
            'labeled_images': 8,
            'missing_labels': 2,
            'missing_label_images': 2,
            'missing_label_ratio': 0.2,
            'risk_level': 'high',
            'warnings': ['发现 2 张图片缺少标签（占比 20.0%），训练结果可能受到明显影响'],
            'empty_labels': 0,
        }

    def fake_validate_dataset(img_dir: str, label_dir: str = '', classes_txt: str = ''):
        steps.append('validate')
        assert classes_txt.endswith('classes.txt')
        return {
            'ok': True,
            'summary': 'validate ok',
            'has_issues': False,
            'has_risks': True,
            'risk_level': 'high',
            'warnings': ['发现 2 张图片缺少标签（占比 20.0%），训练结果可能受到明显影响'],
            'missing_label_images': 2,
            'missing_label_ratio': 0.2,
            'issue_count': 0,
        }

    def fake_split_dataset(img_dir: str, label_dir: str = '', ratio: float = 0.8):
        steps.append('split')
        assert ratio == 0.8
        return {
            'ok': True,
            'summary': 'split ok',
            'train_path': 'images/train',
            'val_path': 'images/val',
            'resolved_train_path': '/tmp/images/train',
            'resolved_val_path': '/tmp/images/val',
            'train_count': 8,
            'val_count': 2,
            'output_dir': '/tmp',
            'suggested_yaml_path': '/tmp/data.yaml',
        }

    def fake_generate_yaml(train_path: str, val_path: str, classes: list[str] | None = None, output_path: str = '', classes_txt: str = '', img_dir: str = '', label_dir: str = '', classes_text: str = ''):
        steps.append('generate_yaml')
        assert classes == ['cat', 'dog']
        assert classes_txt.endswith('classes.txt')
        assert classes_text == ''
        assert output_path == '/tmp/data.yaml'
        assert train_path == '/tmp/images/train'
        assert val_path == '/tmp/images/val'
        assert img_dir == '/tmp'
        return {
            'ok': True,
            'summary': 'yaml ok',
            'output_path': '/tmp/data.yaml',
            'class_name_source': 'classes_txt',
        }

    def fake_training_readiness(img_dir: str, label_dir: str = '', data_yaml: str = ''):
        steps.append('readiness')
        assert data_yaml == '/tmp/data.yaml'
        return {
            'ok': True,
            'ready': True,
            'summary': '可以训练，但存在数据质量风险: 发现 2 张图片缺少标签（占比 20.0%），训练结果可能受到明显影响',
            'risk_level': 'high',
            'warnings': ['发现 2 张图片缺少标签（占比 20.0%），训练结果可能受到明显影响'],
            'missing_label_images': 2,
            'missing_label_ratio': 0.2,
            'recommended_start_training_args': {'data_yaml': '/tmp/data.yaml'},
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
    assert result['force_split_applied'] is True
    assert result['split_reason'] == 'missing_yaml'
    assert result['data_yaml_source'] == 'generated_from_split'
    assert result['detected_classes_txt'].endswith('classes.txt')
    assert result['class_name_source'] == 'classes_txt'
    assert result['risk_level'] == 'high'
    assert result['missing_label_images'] == 2
    assert result['recommended_start_training_args'] == {'data_yaml': '/tmp/data.yaml'}
    assert '数据质量风险' in result['summary']
    assert result['prepare_overview']['ready'] is True
    assert result['prepare_overview']['force_split_applied'] is True
    assert result['action_candidates'][0]['tool'] == 'start_training'
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


def _run_split_dataset_regenerates_invalid_yaml() -> None:
    steps: list[str] = []

    def fake_resolve_dataset_root(dataset_path: str):
        steps.append('resolve_root')
        return {
            'ok': True,
            'dataset_root': dataset_path,
            'img_dir': f'{dataset_path}/images',
            'label_dir': f'{dataset_path}/labels',
            'summary': 'detected split dataset',
            'is_split': True,
            'structure_type': 'yolo_split',
            'resolution_method': 'name',
            'split_info': {
                'train_img_dir': f'{dataset_path}/images/train',
                'val_img_dir': f'{dataset_path}/images/val',
                'train_label_dir': f'{dataset_path}/labels/train',
                'val_label_dir': f'{dataset_path}/labels/val',
            },
            'detected_data_yaml': f'{dataset_path}/data.yaml',
        }

    def fake_scan_dataset(img_dir: str, label_dir: str = ''):
        steps.append('scan')
        return {
            'ok': True,
            'summary': 'scan ok',
            'classes': ['car'],
            'detected_classes_txt': f'{img_dir}/../labels/classes.txt',
            'class_name_source': 'classes_txt',
            'detected_data_yaml': f'{Path(img_dir).parent}/data.yaml',
            'total_images': 3,
            'labeled_images': 3,
            'missing_labels': [],
            'missing_label_images': 0,
            'missing_label_ratio': 0.0,
            'risk_level': 'none',
            'warnings': [],
            'empty_labels': 0,
        }

    def fake_validate_dataset(img_dir: str, label_dir: str = '', classes_txt: str = ''):
        steps.append('validate')
        return {
            'ok': True,
            'summary': 'validate ok',
            'has_issues': False,
            'has_risks': False,
            'risk_level': 'none',
            'warnings': [],
            'missing_label_images': 0,
            'missing_label_ratio': 0.0,
            'issue_count': 0,
        }

    def fake_generate_yaml(train_path: str, val_path: str, classes: list[str] | None = None, output_path: str = '', classes_txt: str = '', img_dir: str = '', label_dir: str = '', classes_text: str = ''):
        steps.append('generate_yaml')
        assert train_path.endswith('/images/train')
        assert val_path.endswith('/images/val')
        assert classes == ['car']
        assert classes_text == ''
        assert output_path.replace('\\', '/').endswith('/data.yaml')
        return {
            'ok': True,
            'summary': 'yaml regenerated',
            'output_path': output_path,
            'class_name_source': 'classes_txt',
        }

    def fake_training_readiness(img_dir: str, label_dir: str = '', data_yaml: str = ''):
        steps.append('readiness')
        assert data_yaml.replace('\\', '/').endswith('/data.yaml')
        return {
            'ok': True,
            'ready': True,
            'summary': '可以直接训练',
            'risk_level': 'none',
            'warnings': [],
            'missing_label_images': 0,
            'missing_label_ratio': 0.0,
            'recommended_start_training_args': {'data_yaml': data_yaml},
            'next_actions': [{'tool': 'start_training'}],
        }

    def fake_inspect_training_yaml(yaml_path: str):
        return {
            'exists': True,
            'usable': False,
            'issues': [{'field': 'val', 'raw': 'images/val', 'resolved': '/bad/val', 'reason': 'missing_target'}],
        }

    combo_tools.resolve_dataset_root = fake_resolve_dataset_root
    combo_tools.scan_dataset = fake_scan_dataset
    combo_tools.validate_dataset = fake_validate_dataset
    combo_tools.split_dataset = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('split should not run for already-split dataset'))
    combo_tools.generate_yaml = fake_generate_yaml
    combo_tools.training_readiness = fake_training_readiness
    combo_tools._inspect_training_yaml = fake_inspect_training_yaml

    result = combo_tools.prepare_dataset_for_training('/dataset')
    assert result['ok'] is True
    assert result['ready'] is True
    assert result['data_yaml'].replace('\\', '/').endswith('/data.yaml')
    assert result['data_yaml_source'] == 'regenerated_from_split'
    assert result['force_split_applied'] is False
    assert result['split_reason'] == 'already_split'
    assert result['prepare_overview']['split_reason'] == 'already_split'
    assert result['action_candidates'][0]['tool'] == 'start_training'
    assert steps == ['resolve_root', 'scan', 'validate', 'generate_yaml', 'readiness']


def _run_explicit_classes_txt_overrides_detected_classes_txt() -> None:
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
            'classes': ['scan_a', 'scan_b'],
            'detected_classes_txt': f'{img_dir}/../labels/classes.txt',
            'class_name_source': 'classes_txt',
            'detected_data_yaml': '',
            'total_images': 12,
            'labeled_images': 12,
            'missing_labels': [],
            'missing_label_images': 0,
            'missing_label_ratio': 0.0,
            'risk_level': 'none',
            'warnings': [],
            'empty_labels': 0,
        }

    def fake_validate_dataset(img_dir: str, label_dir: str = '', classes_txt: str = ''):
        steps.append('validate')
        assert classes_txt == '/meta/custom-classes.txt'
        return {
            'ok': True,
            'summary': 'validate ok',
            'has_issues': False,
            'has_risks': False,
            'risk_level': 'none',
            'warnings': [],
            'missing_label_images': 0,
            'missing_label_ratio': 0.0,
            'issue_count': 0,
        }

    def fake_split_dataset(img_dir: str, label_dir: str = '', ratio: float = 0.8):
        steps.append('split')
        return {
            'ok': True,
            'summary': 'split ok',
            'train_path': 'images/train',
            'val_path': 'images/val',
            'resolved_train_path': '/tmp/custom/images/train',
            'resolved_val_path': '/tmp/custom/images/val',
            'train_count': 9,
            'val_count': 3,
            'output_dir': '/tmp/custom',
            'suggested_yaml_path': '/tmp/custom/data.yaml',
        }

    def fake_generate_yaml(
        train_path: str,
        val_path: str,
        classes: list[str] | None = None,
        output_path: str = '',
        classes_txt: str = '',
        img_dir: str = '',
        label_dir: str = '',
        classes_text: str = '',
    ):
        steps.append('generate_yaml')
        assert train_path == '/tmp/custom/images/train'
        assert val_path == '/tmp/custom/images/val'
        assert classes_txt == '/meta/custom-classes.txt'
        assert classes_text == ''
        return {
            'ok': True,
            'summary': 'yaml ok',
            'output_path': '/tmp/custom/data.yaml',
            'class_name_source': 'explicit_classes_txt',
        }

    def fake_training_readiness(img_dir: str, label_dir: str = '', data_yaml: str = ''):
        steps.append('readiness')
        assert data_yaml == '/tmp/custom/data.yaml'
        return {
            'ok': True,
            'ready': True,
            'summary': '可以训练',
            'risk_level': 'none',
            'warnings': [],
            'missing_label_images': 0,
            'missing_label_ratio': 0.0,
            'recommended_start_training_args': {'data_yaml': data_yaml},
            'next_actions': [{'tool': 'start_training'}],
        }

    combo_tools.resolve_dataset_root = fake_resolve_dataset_root
    combo_tools.scan_dataset = fake_scan_dataset
    combo_tools.validate_dataset = fake_validate_dataset
    combo_tools.split_dataset = fake_split_dataset
    combo_tools.generate_yaml = fake_generate_yaml
    combo_tools.training_readiness = fake_training_readiness

    result = combo_tools.prepare_dataset_for_training('/dataset', classes_txt='/meta/custom-classes.txt')
    assert result['ok'] is True
    assert result['ready'] is True
    assert result['data_yaml'] == '/tmp/custom/data.yaml'
    assert result['effective_classes_txt'] == '/meta/custom-classes.txt'
    assert steps == ['resolve_root', 'scan', 'validate', 'split', 'generate_yaml', 'readiness']


def main() -> None:
    _run_success_path()
    _run_early_block_path()
    _run_split_dataset_regenerates_invalid_yaml()
    _run_explicit_classes_txt_overrides_detected_classes_txt()
    print('prepare dataset flow smoke ok')


if __name__ == '__main__':
    main()

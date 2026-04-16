from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from types import SimpleNamespace

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.tools.data_tools import (
    _discover_classes_txt,
    _discover_data_yaml,
    _infer_dataset_root,
    generate_yaml,
    training_readiness,
)
import yolostudio_agent.agent.server.tools.data_tools as data_tools
import yolostudio_agent.agent.server.services.gpu_utils as gpu_utils


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix='data_tool_test_'))
    if root.exists():
        shutil.rmtree(root)
    (root / 'dataset_1ch' / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (root / 'dataset_1ch' / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (root / 'dataset_1ch' / 'labels').mkdir(parents=True, exist_ok=True)
    yaml_path = root / 'dataset_1ch.yaml'
    yaml_path.write_text("path: dataset_1ch\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n  1: dog\n", encoding='utf-8')
    classes_txt = root / 'dataset_1ch' / 'labels' / 'classes.txt'
    classes_txt.write_text('cat\ndog\n', encoding='utf-8')

    try:
        img_dir = root / 'dataset_1ch' / 'images'
        label_dir = root / 'dataset_1ch' / 'labels'
        dataset_root = _infer_dataset_root(img_dir, label_dir)
        detected, candidates = _discover_data_yaml(img_dir, label_dir)
        detected_classes, class_candidates = _discover_classes_txt(img_dir, label_dir)
        assert dataset_root == root / 'dataset_1ch'
        assert detected == str(yaml_path.resolve())
        assert str(yaml_path.resolve()) in candidates
        assert detected_classes == str(classes_txt.resolve())
        assert str(classes_txt.resolve()) in class_candidates

        output_yaml = root / 'generated.yaml'
        generated = generate_yaml(
            train_path=str((root / 'dataset_1ch' / 'images' / 'train').resolve()),
            val_path=str((root / 'dataset_1ch' / 'images' / 'val').resolve()),
            img_dir=str(img_dir),
            label_dir=str(label_dir),
            output_path=str(output_yaml),
            classes_txt=str(classes_txt),
        )
        assert generated['ok'] is True
        assert output_yaml.exists()
        assert generated['output_path'] == str(output_yaml.resolve())
        assert generated['class_name_source'] == 'classes_txt'
        yaml_data = yaml.safe_load(output_yaml.read_text(encoding='utf-8'))
        assert yaml_data['names'][0] == 'cat'
        assert yaml_data['names'][1] == 'dog'

        output_yaml_from_text = root / 'generated_from_text.yaml'
        generated_from_text = generate_yaml(
            train_path=str((root / 'dataset_1ch' / 'images' / 'train').resolve()),
            val_path=str((root / 'dataset_1ch' / 'images' / 'val').resolve()),
            img_dir=str(img_dir),
            label_dir=str(label_dir),
            output_path=str(output_yaml_from_text),
            classes_text='1. cat\n2. dog\n',
        )
        assert generated_from_text['ok'] is True
        assert output_yaml_from_text.exists()
        assert generated_from_text['class_name_source'] == 'explicit_classes_text'
        yaml_text_data = yaml.safe_load(output_yaml_from_text.read_text(encoding='utf-8'))
        assert yaml_text_data['names'][0] == 'cat'
        assert yaml_text_data['names'][1] == 'dog'

        invalid_yaml = root / 'invalid_remote.yaml'
        invalid_yaml.write_text(
            "path: Z:/definitely-missing/remote_training_smoke_dataset\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: cat\n",
            encoding='utf-8',
        )

        orig_scan_dataset = data_tools.scan_dataset
        orig_validate_dataset = data_tools.validate_dataset
        orig_module_resolve_auto_device = data_tools.resolve_auto_device
        orig_query_gpu_status = gpu_utils.query_gpu_status
        orig_resolve_auto_device = gpu_utils.resolve_auto_device
        try:
            data_tools.scan_dataset = lambda img_dir, label_dir='': {
                'ok': True,
                'summary': 'scan ok',
                'dataset_root': str((root / 'dataset_1ch').resolve()),
                'resolved_img_dir': str((root / 'dataset_1ch' / 'images').resolve()),
                'resolved_label_dir': str((root / 'dataset_1ch' / 'labels').resolve()),
                'detected_data_yaml': str(invalid_yaml.resolve()),
                'detected_classes_txt': str(classes_txt.resolve()),
                'class_name_source': 'classes_txt',
                'labeled_images': 3,
                'total_images': 3,
                'missing_labels': [],
                'missing_label_images': 0,
                'missing_label_ratio': 0.0,
                'risk_level': 'none',
            }
            data_tools.validate_dataset = lambda img_dir, label_dir='', classes_txt='': {
                'ok': True,
                'summary': 'validate ok',
                'has_issues': False,
                'warnings': [],
                'risk_level': 'none',
                'missing_label_images': 0,
                'missing_label_ratio': 0.0,
            }
            gpu_utils.query_gpu_status = lambda: [SimpleNamespace(index='0', busy=False)]
            gpu_utils.resolve_auto_device = lambda policy='single_idle_gpu', gpus=None: ('0', None)
            data_tools.resolve_auto_device = lambda policy='single_idle_gpu', gpus=None: ('0', None)

            readiness = training_readiness(str(root / 'dataset_1ch'))
            assert readiness['ok'] is True
            assert readiness['ready'] is False
            assert readiness['preparable'] is True
            assert readiness['primary_blocker_type'] == 'invalid_yaml'
            assert readiness['data_yaml_usable'] is False
            assert readiness['resolved_data_yaml'] == str(invalid_yaml.resolve())
            assert readiness['data_yaml_issues']
            assert any(issue['field'] in {'path', 'train', 'val'} for issue in readiness['data_yaml_issues'])
            assert any(action['tool'] == 'prepare_dataset_for_training' for action in readiness['next_actions'])
        finally:
            data_tools.scan_dataset = orig_scan_dataset
            data_tools.validate_dataset = orig_validate_dataset
            data_tools.resolve_auto_device = orig_module_resolve_auto_device
            gpu_utils.query_gpu_status = orig_query_gpu_status
            gpu_utils.resolve_auto_device = orig_resolve_auto_device

        print('data tool helper smoke ok')
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    main()

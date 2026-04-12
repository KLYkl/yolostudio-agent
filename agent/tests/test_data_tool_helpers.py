from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

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
)


def main() -> None:
    root = Path('C:/workspace/yolodo2.0/agent_plan/.tmp_data_tool_test')
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

        print('data tool helper smoke ok')
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    main()

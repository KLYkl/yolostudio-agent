from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.tools import data_tools


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix='data_tool_execution_guards_'))
    try:
        missing_split = data_tools.split_dataset(str(root / 'missing_dataset'))
        assert missing_split['ok'] is False, missing_split

        empty_dataset = root / 'empty_dataset'
        (empty_dataset / 'images').mkdir(parents=True, exist_ok=True)
        (empty_dataset / 'labels').mkdir(parents=True, exist_ok=True)

        empty_split = data_tools.split_dataset(str(empty_dataset))
        assert empty_split['ok'] is False, empty_split
        assert '未产出有效 train/val 结果' in empty_split['error'], empty_split

        missing_yaml = data_tools.generate_yaml(
            train_path=str((empty_dataset / 'images' / 'train').resolve()),
            val_path=str((empty_dataset / 'images' / 'val').resolve()),
            classes=['cat'],
            output_path=str((empty_dataset / 'data.yaml').resolve()),
        )
        assert missing_yaml['ok'] is False, missing_yaml
        assert 'train/val 路径不存在' in missing_yaml['error'], missing_yaml

        no_label_dataset = root / 'no_label_dataset'
        (no_label_dataset / 'images').mkdir(parents=True, exist_ok=True)
        (no_label_dataset / 'labels').mkdir(parents=True, exist_ok=True)
        convert_no_labels = data_tools.convert_format(str(no_label_dataset), target_format='xml')
        assert convert_no_labels['ok'] is False, convert_no_labels
        assert convert_no_labels['error_type'] == 'NoConvertibleLabels', convert_no_labels

        empty_categorize = data_tools.categorize_by_class(str(empty_dataset))
        assert empty_categorize['ok'] is False, empty_categorize
        assert '按类别整理未产出有效结果' in empty_categorize['error'], empty_categorize

        empty_augment = data_tools.augment_dataset(str(empty_dataset), enable_horizontal_flip=True)
        assert empty_augment['ok'] is False, empty_augment
        assert '增强未找到可处理的图片' in empty_augment['error'], empty_augment

        print('data tool execution guards ok')
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    main()

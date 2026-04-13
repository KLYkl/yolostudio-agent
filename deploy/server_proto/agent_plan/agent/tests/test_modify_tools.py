from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.tools import data_tools


TMP_ROOT = Path(__file__).resolve().parents[2] / '.tmp_modify_tools'


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (96, 96), color).save(path, format='JPEG')


def _make_label(path: Path, class_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{class_id} 0.5 0.5 0.4 0.4\n', encoding='utf-8')


def main() -> None:
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    dataset_root = TMP_ROOT / 'dataset'
    img_dir = dataset_root / 'images'
    label_dir = dataset_root / 'labels'
    try:
        _make_image(img_dir / 'a.jpg', (255, 0, 0))
        _make_image(img_dir / 'b.jpg', (0, 255, 0))
        _make_label(label_dir / 'a.txt', 0)
        _make_label(label_dir / 'b.txt', 1)
        _make_label(label_dir / 'orphan.txt', 0)
        (label_dir / 'classes.txt').write_text('car\nperson\ntruck\n', encoding='utf-8')

        preview = data_tools.preview_modify_labels(
            str(dataset_root),
            action='replace',
            old_value='car',
            new_value='truck',
        )
        assert preview['ok'] is True, preview
        assert preview['matched_files'] == 2, preview
        assert preview['matched_annotations'] == 2, preview
        assert preview['class_name_source'] == 'detected_classes_txt', preview

        modified = data_tools.modify_labels(
            str(dataset_root),
            action='replace',
            old_value='car',
            new_value='truck',
        )
        assert modified['ok'] is True, modified
        assert modified['modified_files'] == 2, modified
        assert (label_dir / 'a.txt').read_text(encoding='utf-8').startswith('2 '), modified
        assert (label_dir / 'orphan.txt').read_text(encoding='utf-8').startswith('2 '), modified
        assert (label_dir / 'a.txt.bak').exists(), modified

        orphan_preview = data_tools.clean_orphan_labels(str(dataset_root))
        assert orphan_preview['ok'] is True, orphan_preview
        assert orphan_preview['dry_run'] is True, orphan_preview
        assert orphan_preview['orphan_count'] == 1, orphan_preview
        assert any(path.endswith('orphan.txt') for path in orphan_preview['orphan_samples']), orphan_preview

        orphan_cleaned = data_tools.clean_orphan_labels(str(dataset_root), dry_run=False)
        assert orphan_cleaned['ok'] is True, orphan_cleaned
        assert orphan_cleaned['cleaned_count'] == 1, orphan_cleaned
        assert not (label_dir / 'orphan.txt').exists(), orphan_cleaned
        assert (label_dir / 'orphan.txt.bak').exists(), orphan_cleaned

        dataset_yaml_only = TMP_ROOT / 'dataset_yaml_only'
        _make_image(dataset_yaml_only / 'images' / 'c.jpg', (0, 0, 255))
        _make_label(dataset_yaml_only / 'labels' / 'c.txt', 0)
        (dataset_yaml_only / 'data.yaml').write_text(
            'path: .\ntrain: images\nval: images\nnames:\n  0: car\n  1: person\n  2: bus\n',
            encoding='utf-8',
        )

        preview_yaml = data_tools.preview_modify_labels(
            str(dataset_yaml_only),
            action='replace',
            old_value='car',
            new_value='bus',
        )
        assert preview_yaml['ok'] is True, preview_yaml
        assert preview_yaml['matched_files'] == 1, preview_yaml
        assert preview_yaml['class_name_source'] == 'detected_data_yaml', preview_yaml

        modified_yaml = data_tools.modify_labels(
            str(dataset_yaml_only),
            action='replace',
            old_value='car',
            new_value='bus',
        )
        assert modified_yaml['ok'] is True, modified_yaml
        assert modified_yaml['modified_files'] == 1, modified_yaml
        assert (dataset_yaml_only / 'labels' / 'c.txt').read_text(encoding='utf-8').startswith('2 '), modified_yaml

        print('modify tools ok')
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()

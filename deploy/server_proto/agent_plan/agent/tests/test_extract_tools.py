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
from yolostudio_agent.agent.server.tools import extract_tools


TMP_ROOT = Path(__file__).resolve().parents[2] / '.tmp_extract_tools'


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (96, 96), color).save(path, format='JPEG')


def _make_label(path: Path, class_id: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{class_id} 0.5 0.5 0.4 0.4\n', encoding='utf-8')


def main() -> None:
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    dataset_root = TMP_ROOT / 'dataset'
    img_dir = dataset_root / 'images'
    label_dir = dataset_root / 'labels'
    try:
        _make_image(img_dir / 'cam_a' / 'a1.jpg', (255, 0, 0))
        _make_image(img_dir / 'cam_a' / 'a2.jpg', (200, 0, 0))
        _make_image(img_dir / 'cam_b' / 'b1.jpg', (0, 255, 0))
        _make_image(img_dir / 'cam_b' / 'b2.jpg', (0, 200, 0))
        _make_label(label_dir / 'cam_a' / 'a1.txt', 0)
        _make_label(label_dir / 'cam_a' / 'a2.txt', 0)
        _make_label(label_dir / 'cam_b' / 'b1.txt', 1)
        _make_label(label_dir / 'cam_b' / 'b2.txt', 1)

        preview_output = TMP_ROOT / 'preview_out'
        preview = extract_tools.preview_extract_images(
            source_path=str(dataset_root),
            output_dir=str(preview_output),
            selection_mode='count',
            count=1,
            grouping_mode='per_directory',
            output_layout='flat',
        )
        assert preview['ok'] is True, preview
        assert preview['available_images'] == 4, preview
        assert preview['planned_extract_count'] == 2, preview
        assert preview['planned_dir_stats'] == {'cam_a': 1, 'cam_b': 1}, preview
        assert preview['copy_labels_effective'] is True, preview
        assert preview['workflow_ready_path'] == str(preview_output.resolve()), preview

        extract_output = TMP_ROOT / 'extract_out'
        extracted = extract_tools.extract_images(
            source_path=str(dataset_root),
            output_dir=str(extract_output),
            selection_mode='count',
            count=1,
            grouping_mode='per_directory',
            output_layout='flat',
            seed=7,
        )
        assert extracted['ok'] is True, extracted
        assert extracted['extracted'] == 2, extracted
        assert extracted['labels_copied'] == 2, extracted
        assert extracted['output_dir'] == str(extract_output.resolve()), extracted
        assert Path(extracted['output_img_dir']).is_dir(), extracted
        assert Path(extracted['output_label_dir']).is_dir(), extracted
        assert extracted['workflow_ready'] is True, extracted

        scan = data_tools.scan_dataset(extracted['workflow_ready_path'])
        assert scan['ok'] is True, scan
        assert scan['total_images'] == 2, scan
        assert scan['missing_label_images'] == 0, scan

        validate = data_tools.validate_dataset(extracted['workflow_ready_path'])
        assert validate['ok'] is True, validate
        assert validate['has_issues'] is False, validate

        conflict_preview = extract_tools.preview_extract_images(
            source_path=str(dataset_root),
            output_dir=str(extract_output),
            selection_mode='count',
            count=1,
            grouping_mode='per_directory',
            output_layout='flat',
            seed=7,
        )
        assert conflict_preview['ok'] is True, conflict_preview
        assert conflict_preview['conflict_count'] >= 2, conflict_preview

        empty_root = TMP_ROOT / 'empty_dataset'
        (empty_root / 'images').mkdir(parents=True, exist_ok=True)
        empty_preview = extract_tools.preview_extract_images(str(empty_root))
        assert empty_preview['ok'] is True, empty_preview
        assert empty_preview['available_images'] == 0, empty_preview
        assert empty_preview['planned_extract_count'] == 0, empty_preview

        print('extract tools ok')
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()

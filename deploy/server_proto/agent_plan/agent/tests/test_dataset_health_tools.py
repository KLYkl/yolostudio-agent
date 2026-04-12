from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

import yolostudio_agent.agent.server.tools.data_tools as data_tools


TMP_ROOT = Path('D:/yolodo2.0/agent_plan/.tmp_dataset_health_tools')


def _make_image(path: Path, size: tuple[int, int], color: tuple[int, int, int], *, fmt: str = 'JPEG') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', size, color)
    image.save(path, format=fmt)


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)

    dataset_root = TMP_ROOT / 'dataset'
    img_dir = dataset_root / 'images'
    label_dir = dataset_root / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    try:
        _make_image(img_dir / 'good_a.jpg', (96, 96), (255, 0, 0))
        shutil.copy2(img_dir / 'good_a.jpg', img_dir / 'good_a_copy.jpg')
        _make_image(img_dir / 'small.jpg', (16, 16), (0, 255, 0))
        _make_image(img_dir / 'large.jpg', (9001, 64), (0, 0, 255))
        _make_image(img_dir / 'mismatch.jpg', (80, 80), (128, 128, 0), fmt='PNG')
        (img_dir / 'zero.jpg').write_bytes(b'')
        (img_dir / 'corrupt.jpg').write_text('not-an-image', encoding='utf-8')

        report_path = TMP_ROOT / 'health_report.txt'
        result = data_tools.run_dataset_health_check(
            str(dataset_root),
            export_report=True,
            report_path=str(report_path),
            max_examples=5,
            max_duplicate_groups=5,
        )
        assert result['ok'] is True, result
        assert result['resolved_img_dir'] == str(img_dir.resolve())
        assert result['risk_level'] == 'critical'
        assert result['integrity']['total_images'] == 7
        assert result['integrity']['zero_bytes_count'] == 1
        assert result['integrity']['corrupted_count'] >= 1
        assert result['integrity']['format_mismatch_count'] == 1
        assert result['size_stats']['abnormal_small_count'] == 1
        assert result['size_stats']['abnormal_large_count'] == 1
        assert result['duplicate_groups'] == 1
        assert result['duplicate_extra_files'] == 1
        assert report_path.exists(), result
        assert result['report_path'] == str(report_path.resolve())
        assert result['warnings'], result

        dup = data_tools.detect_duplicate_images(str(dataset_root), method='md5')
        assert dup['ok'] is True, dup
        assert dup['duplicate_groups'] == 1, dup
        assert dup['duplicate_extra_files'] == 1, dup
        assert dup['groups'], dup
        sample_paths = dup['groups'][0]['paths']
        assert any(path.endswith('good_a.jpg') for path in sample_paths), dup
        assert any(path.endswith('good_a_copy.jpg') for path in sample_paths), dup

        print('dataset health tools ok')
    finally:
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)


if __name__ == '__main__':
    main()

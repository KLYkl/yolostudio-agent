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


TMP_ROOT = Path(__file__).resolve().parents[2] / '.tmp_label_generation_tools'


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (128, 72), color).save(path, format='JPEG')


def _make_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')


def main() -> None:
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    try:
        dataset_missing = TMP_ROOT / 'dataset_missing'
        _make_image(dataset_missing / 'images' / 'train' / 'a.jpg', (255, 0, 0))
        _make_image(dataset_missing / 'images' / 'train' / 'b.jpg', (0, 255, 0))
        _make_image(dataset_missing / 'images' / 'val' / 'c.jpg', (0, 0, 255))
        _make_label(dataset_missing / 'labels' / 'train' / 'a.txt', ['0 0.5 0.5 0.4 0.4'])

        preview_missing = data_tools.preview_generate_missing_labels(str(dataset_missing))
        assert preview_missing['ok'] is True, preview_missing
        assert preview_missing['planned_generate_count'] == 2, preview_missing
        assert any(path.endswith('labels\\train\\b.txt') or path.endswith('labels/train/b.txt') for path in preview_missing['planned_output_samples']), preview_missing
        assert any(path.endswith('labels\\val\\c.txt') or path.endswith('labels/val/c.txt') for path in preview_missing['planned_output_samples']), preview_missing

        generated_missing = data_tools.generate_missing_labels(str(dataset_missing))
        assert generated_missing['ok'] is True, generated_missing
        assert generated_missing['generated_count'] == 2, generated_missing
        assert (dataset_missing / 'labels' / 'train' / 'b.txt').exists(), generated_missing
        assert (dataset_missing / 'labels' / 'val' / 'c.txt').exists(), generated_missing

        dataset_empty = TMP_ROOT / 'dataset_empty'
        _make_image(dataset_empty / 'images' / 'cam1' / 'x.jpg', (100, 0, 0))
        _make_image(dataset_empty / 'images' / 'cam2' / 'y.jpg', (0, 100, 0))
        custom_output = TMP_ROOT / 'xml_output'

        preview_empty = data_tools.preview_generate_empty_labels(
            str(dataset_empty),
            label_format='xml',
            output_dir=str(custom_output),
            only_missing=False,
        )
        assert preview_empty['ok'] is True, preview_empty
        assert preview_empty['planned_generate_count'] == 2, preview_empty
        assert any(path.endswith('xml_output\\cam1\\x.xml') or path.endswith('xml_output/cam1/x.xml') for path in preview_empty['planned_output_samples']), preview_empty

        generated_empty = data_tools.generate_empty_labels(
            str(dataset_empty),
            label_format='xml',
            output_dir=str(custom_output),
            only_missing=False,
        )
        assert generated_empty['ok'] is True, generated_empty
        assert generated_empty['generated_count'] == 2, generated_empty
        xml_x = custom_output / 'cam1' / 'x.xml'
        xml_y = custom_output / 'cam2' / 'y.xml'
        assert xml_x.exists() and xml_y.exists(), generated_empty
        assert '<annotation>' in xml_x.read_text(encoding='utf-8')

        dataset_categorize = TMP_ROOT / 'dataset_categorize'
        _make_image(dataset_categorize / 'images' / 'one.jpg', (255, 0, 0))
        _make_image(dataset_categorize / 'images' / 'two.jpg', (0, 255, 0))
        _make_image(dataset_categorize / 'images' / 'empty.jpg', (0, 0, 255))
        _make_image(dataset_categorize / 'images' / 'nolabel.jpg', (255, 255, 0))
        _make_image(dataset_categorize / 'images' / 'mixed.jpg', (255, 0, 255))
        _make_label(dataset_categorize / 'labels' / 'one.txt', ['0 0.5 0.5 0.4 0.4'])
        _make_label(dataset_categorize / 'labels' / 'two.txt', ['1 0.5 0.5 0.3 0.3'])
        _make_label(dataset_categorize / 'labels' / 'empty.txt', [])
        _make_label(dataset_categorize / 'labels' / 'mixed.txt', ['0 0.5 0.5 0.4 0.4', '1 0.6 0.6 0.2 0.2'])
        (dataset_categorize / 'data.yaml').write_text(
            'path: .\ntrain: images\nval: images\nnames:\n  0: car\n  1: person\n',
            encoding='utf-8',
        )

        preview_categorize = data_tools.preview_categorize_by_class(str(dataset_categorize))
        assert preview_categorize['ok'] is True, preview_categorize
        assert preview_categorize['category_stats']['car'] == 1, preview_categorize
        assert preview_categorize['category_stats']['person'] == 1, preview_categorize
        assert preview_categorize['category_stats']['_empty'] == 1, preview_categorize
        assert preview_categorize['category_stats']['_no_label'] == 1, preview_categorize
        assert preview_categorize['category_stats']['_mixed'] == 1, preview_categorize

        categorized = data_tools.categorize_by_class(str(dataset_categorize))
        assert categorized['ok'] is True, categorized
        out_dir = Path(categorized['output_dir'])
        assert (out_dir / 'car' / 'images' / 'one.jpg').exists(), categorized
        assert (out_dir / 'person' / 'labels' / 'two.txt').exists(), categorized
        assert (out_dir / '_empty' / 'images' / 'empty.jpg').exists(), categorized
        assert (out_dir / '_no_label' / 'images' / 'nolabel.jpg').exists(), categorized
        assert (out_dir / '_mixed_report.txt').exists(), categorized

        print('label generation tools ok')
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()

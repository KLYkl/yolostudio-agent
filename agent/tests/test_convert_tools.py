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


TMP_ROOT = Path(__file__).resolve().parents[2] / '.tmp_convert_tools'


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (120, 80), color).save(path, format='JPEG')


def _make_label(path: Path, class_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{class_id} 0.5 0.5 0.4 0.4\n', encoding='utf-8')


def main() -> None:
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    dataset_root = TMP_ROOT / 'dataset'
    img_dir = dataset_root / 'images'
    label_dir = dataset_root / 'labels'
    try:
        _make_image(img_dir / 'cam_a' / 'a1.jpg', (255, 0, 0))
        _make_image(img_dir / 'cam_b' / 'b1.jpg', (0, 255, 0))
        _make_label(label_dir / 'cam_a' / 'a1.txt', 0)
        _make_label(label_dir / 'cam_b' / 'b1.txt', 1)
        (label_dir / 'classes.txt').write_text('car\nperson\n', encoding='utf-8')

        preview_xml = data_tools.preview_convert_format(str(dataset_root), target_format='xml')
        assert preview_xml['ok'] is True, preview_xml
        assert preview_xml['total_labels'] == 2, preview_xml
        assert preview_xml['source_type'] == 'TXT', preview_xml
        assert preview_xml['target_type'] == 'XML', preview_xml
        assert preview_xml['class_name_source'] == 'detected_classes_txt', preview_xml
        assert preview_xml['class_count'] == 2, preview_xml
        assert preview_xml['warnings'] == [], preview_xml

        converted_xml = data_tools.convert_format(str(dataset_root), target_format='xml')
        assert converted_xml['ok'] is True, converted_xml
        assert converted_xml['converted_count'] == 2, converted_xml
        assert converted_xml['output_label_count'] == 2, converted_xml
        xml_output_dir = Path(converted_xml['output_dir'])
        assert xml_output_dir.is_dir(), converted_xml

        xml_a = (xml_output_dir / 'cam_a' / 'a1.xml').read_text(encoding='utf-8')
        xml_b = (xml_output_dir / 'cam_b' / 'b1.xml').read_text(encoding='utf-8')
        assert '<name>car</name>' in xml_a, xml_a
        assert '<name>person</name>' in xml_b, xml_b

        preview_txt = data_tools.preview_convert_format(
            str(dataset_root),
            label_dir=str(xml_output_dir),
            target_format='txt',
        )
        assert preview_txt['ok'] is True, preview_txt
        assert preview_txt['total_labels'] == 2, preview_txt
        assert preview_txt['source_type'] == 'XML', preview_txt
        assert preview_txt['target_type'] == 'TXT', preview_txt

        converted_txt = data_tools.convert_format(
            str(dataset_root),
            label_dir=str(xml_output_dir),
            target_format='txt',
        )
        assert converted_txt['ok'] is True, converted_txt
        assert converted_txt['converted_count'] == 2, converted_txt
        txt_output_dir = Path(converted_txt['output_dir'])
        assert txt_output_dir.is_dir(), converted_txt
        txt_a = (txt_output_dir / 'cam_a' / 'a1.txt').read_text(encoding='utf-8').strip()
        txt_b = (txt_output_dir / 'cam_b' / 'b1.txt').read_text(encoding='utf-8').strip()
        assert txt_a.startswith('0 '), txt_a
        assert txt_b.startswith('1 '), txt_b

        dataset_no_classes = TMP_ROOT / 'dataset_no_classes'
        _make_image(dataset_no_classes / 'images' / 'only.jpg', (0, 0, 255))
        _make_label(dataset_no_classes / 'labels' / 'only.txt', 0)
        preview_without_classes = data_tools.preview_convert_format(str(dataset_no_classes), target_format='xml')
        assert preview_without_classes['ok'] is True, preview_without_classes
        assert preview_without_classes['warnings'], preview_without_classes
        converted_without_classes = data_tools.convert_format(str(dataset_no_classes), target_format='xml')
        assert converted_without_classes['ok'] is True, converted_without_classes
        xml_without_classes = Path(converted_without_classes['output_dir']) / 'only.xml'
        assert '<name>0</name>' in xml_without_classes.read_text(encoding='utf-8')

        print('convert tools ok')
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()

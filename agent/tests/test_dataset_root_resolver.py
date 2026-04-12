from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.dataset_root import resolve_dataset_inputs, resolve_dataset_root

WORK = Path(r'C:\workspace\yolodo2.0\agent_plan\agent\tests\_tmp_dataset_root')


def _mkdirs(base: Path, *parts: str) -> Path:
    path = base.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        standard = WORK / 'standard'
        _mkdirs(standard, 'images')
        _mkdirs(standard, 'labels')

        split = WORK / 'split'
        _mkdirs(split, 'images', 'train')
        _mkdirs(split, 'images', 'val')
        _mkdirs(split, 'labels', 'train')
        _mkdirs(split, 'labels', 'val')
        (split / 'images' / 'train' / 'a.jpg').write_text('x', encoding='utf-8')
        (split / 'images' / 'val' / 'b.jpg').write_text('x', encoding='utf-8')
        (split / 'labels' / 'train' / 'a.txt').write_text('0 0.5 0.5 0.2 0.2', encoding='utf-8')
        (split / 'labels' / 'val' / 'b.txt').write_text('0 0.5 0.5 0.2 0.2', encoding='utf-8')

        direct_images = WORK / 'direct_images'
        direct_images.mkdir()
        (direct_images / 'a.jpg').write_text('x', encoding='utf-8')

        alias_named = WORK / 'alias_named'
        _mkdirs(alias_named, 'pics')
        _mkdirs(alias_named, 'ann')
        (alias_named / 'pics' / 'a.jpg').write_text('x', encoding='utf-8')
        (alias_named / 'ann' / 'a.txt').write_text('0 0.5 0.5 0.2 0.2', encoding='utf-8')

        images_only = WORK / 'images_only'
        _mkdirs(images_only, 'pics')
        (images_only / 'pics' / 'a.jpg').write_text('x', encoding='utf-8')

        unknown = WORK / 'unknown'
        _mkdirs(unknown, 'misc')
        (unknown / 'README.md').write_text('placeholder', encoding='utf-8')

        standard_result = resolve_dataset_root(str(standard))
        split_result = resolve_dataset_root(str(split))
        direct_images_result = resolve_dataset_inputs(str(direct_images))
        alias_named_result = resolve_dataset_root(str(alias_named))
        images_only_result = resolve_dataset_root(str(images_only))
        unknown_result = resolve_dataset_root(str(unknown))

        assert standard_result['ok'] is True
        assert standard_result['structure_type'] == 'yolo_standard'
        assert standard_result['resolved_from_root'] is True
        assert standard_result['img_dir'].endswith('standard\\images')
        assert standard_result['label_dir'].endswith('standard\\labels')

        assert split_result['ok'] is True
        assert split_result['structure_type'] == 'yolo_split'
        assert split_result['is_split'] is True
        assert split_result['split_info']['train_img_dir'].endswith('split\\images\\train')

        assert direct_images_result['ok'] is True
        assert direct_images_result['structure_type'] == 'images_dir'
        assert direct_images_result['img_dir'].endswith('direct_images')

        assert alias_named_result['ok'] is True
        assert alias_named_result['structure_type'] == 'yolo_standard'
        assert alias_named_result['resolved_from_root'] is True
        assert alias_named_result['img_dir'].endswith('alias_named\\pics')
        assert alias_named_result['label_dir'].endswith('alias_named\\ann')
        assert alias_named_result['resolution_method'] == 'image=name,label=name'

        assert images_only_result['ok'] is True
        assert images_only_result['structure_type'] == 'images_only'
        assert images_only_result['img_dir'].endswith('images_only\\pics')
        assert images_only_result['label_dir'] == ''

        assert unknown_result['ok'] is True
        assert unknown_result['structure_type'] == 'unknown'
        assert 'directory_entries' in unknown_result

        print(json.dumps({
            'standard': standard_result,
            'split': split_result,
            'direct_images': direct_images_result,
            'alias_named': alias_named_result,
            'images_only': images_only_result,
            'unknown': unknown_result,
        }, ensure_ascii=False, indent=2))
        print('dataset root resolver smoke ok')
    finally:
        if WORK.exists():
            shutil.rmtree(WORK)


if __name__ == '__main__':
    main()

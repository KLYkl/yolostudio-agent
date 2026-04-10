from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.server.services.dataset_root import resolve_dataset_inputs, resolve_dataset_root

WORK = Path(r'D:\yolodo2.0\agent_plan\agent\tests\_tmp_dataset_root')


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

        flat = WORK / 'flat'
        flat.mkdir()
        (flat / 'a.jpg').write_text('x', encoding='utf-8')

        standard_result = resolve_dataset_root(str(standard))
        split_result = resolve_dataset_root(str(split))
        flat_result = resolve_dataset_inputs(str(flat))

        assert standard_result['ok'] is True
        assert standard_result['structure_type'] == 'yolo_standard'
        assert standard_result['resolved_from_root'] is True
        assert standard_result['img_dir'].endswith('standard\\images')
        assert standard_result['label_dir'].endswith('standard\\labels')

        assert split_result['ok'] is True
        assert split_result['structure_type'] == 'yolo_split'
        assert split_result['is_split'] is True
        assert split_result['split_info']['train_img_dir'].endswith('split\\images\\train')

        assert flat_result['ok'] is True
        assert flat_result['structure_type'] == 'flat'
        assert flat_result['img_dir'].endswith('flat')

        print(json.dumps({
            'standard': standard_result,
            'split': split_result,
            'flat': flat_result,
        }, ensure_ascii=False, indent=2))
        print('dataset root resolver smoke ok')
    finally:
        if WORK.exists():
            shutil.rmtree(WORK)


if __name__ == '__main__':
    main()

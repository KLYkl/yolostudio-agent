from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGE_DIR_NAMES = {'images', 'imgs', 'jpegimages'}
LABEL_DIR_NAMES = {'labels', 'annotations', 'label'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _find_subdir(base: Path, names: set[str]) -> Path | None:
    for name in names:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return None


def _detect_yaml_candidates(dataset_root: Path) -> list[str]:
    names = {
        'data.yaml',
        'data.yml',
        'dataset.yaml',
        'dataset.yml',
        f'{dataset_root.name}.yaml',
        f'{dataset_root.name}.yml',
    }
    candidates: list[str] = []
    seen: set[str] = set()
    for base in (dataset_root, dataset_root.parent):
        for name in names:
            path = (base / name).resolve()
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                candidates.append(key)
    return candidates


def _has_image_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
            return True
    return False


def resolve_dataset_root(path: str, label_dir: str = '') -> dict[str, Any]:
    base = Path(path)
    explicit_label = Path(label_dir) if label_dir else None

    if not base.exists():
        return {
            'ok': False,
            'error': f'路径不存在: {path}',
            'dataset_root': str(base),
            'structure_type': 'missing',
            'img_dir': str(base),
            'label_dir': str(explicit_label) if explicit_label else '',
            'resolved_from_root': False,
            'is_split': False,
            'split_info': {},
            'detected_data_yaml': '',
            'data_yaml_candidates': [],
            'summary': '路径不存在',
            'next_actions': ['请确认数据集路径是否正确'],
        }

    if base.is_dir() and base.name.lower() in IMAGE_DIR_NAMES:
        dataset_root = base.parent
        inferred_label = explicit_label or _find_subdir(dataset_root, LABEL_DIR_NAMES)
        candidates = _detect_yaml_candidates(dataset_root)
        is_split = (base / 'train').is_dir() and (base / 'val').is_dir()
        return {
            'ok': True,
            'dataset_root': str(dataset_root),
            'structure_type': 'images_dir',
            'img_dir': str(base),
            'label_dir': str(inferred_label) if inferred_label else '',
            'resolved_from_root': False,
            'is_split': is_split,
            'split_info': {
                'train_img_dir': str(base / 'train'),
                'val_img_dir': str(base / 'val'),
                'train_label_dir': str((inferred_label / 'train')) if inferred_label and (inferred_label / 'train').is_dir() else '',
                'val_label_dir': str((inferred_label / 'val')) if inferred_label and (inferred_label / 'val').is_dir() else '',
            } if is_split else {},
            'detected_data_yaml': candidates[0] if candidates else '',
            'data_yaml_candidates': candidates,
            'summary': '已直接使用 images 目录',
            'next_actions': ['可直接用 img_dir/label_dir 调用 scan_dataset'],
        }

    images_dir = _find_subdir(base, IMAGE_DIR_NAMES) if base.is_dir() else None
    labels_dir = explicit_label or (_find_subdir(base, LABEL_DIR_NAMES) if base.is_dir() else None)
    if images_dir:
        is_split = (images_dir / 'train').is_dir() and (images_dir / 'val').is_dir()
        split_info = {}
        if is_split:
            split_info = {
                'train_img_dir': str(images_dir / 'train'),
                'val_img_dir': str(images_dir / 'val'),
                'train_label_dir': str((labels_dir / 'train')) if labels_dir and (labels_dir / 'train').is_dir() else '',
                'val_label_dir': str((labels_dir / 'val')) if labels_dir and (labels_dir / 'val').is_dir() else '',
            }
        candidates = _detect_yaml_candidates(base)
        return {
            'ok': True,
            'dataset_root': str(base.resolve()),
            'structure_type': 'yolo_split' if is_split else 'yolo_standard',
            'img_dir': str(images_dir.resolve()),
            'label_dir': str(labels_dir.resolve()) if labels_dir else '',
            'resolved_from_root': True,
            'is_split': is_split,
            'split_info': split_info,
            'detected_data_yaml': candidates[0] if candidates else '',
            'data_yaml_candidates': candidates,
            'summary': '检测到已划分数据集结构' if is_split else '检测到 YOLO 标准目录结构 (images/ + labels/)',
            'next_actions': ['可直接用解析后的 img_dir/label_dir 调用 scan_dataset'],
        }

    if base.is_dir() and _has_image_files(base):
        dataset_root = base.parent if base.name.lower() in IMAGE_DIR_NAMES else base
        candidates = _detect_yaml_candidates(dataset_root)
        return {
            'ok': True,
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': 'flat',
            'img_dir': str(base.resolve()),
            'label_dir': str(explicit_label.resolve()) if explicit_label else '',
            'resolved_from_root': False,
            'is_split': False,
            'split_info': {},
            'detected_data_yaml': candidates[0] if candidates else '',
            'data_yaml_candidates': candidates,
            'summary': '检测到扁平目录结构（图片直接位于当前目录）',
            'next_actions': ['如标签不在同目录，请显式提供 label_dir'],
        }

    children = []
    if base.is_dir():
        try:
            children = sorted(child.name for child in base.iterdir())[:10]
        except OSError:
            children = []
    return {
        'ok': True,
        'dataset_root': str(base.resolve()),
        'structure_type': 'unknown',
        'img_dir': str(base.resolve()),
        'label_dir': str(explicit_label.resolve()) if explicit_label else '',
        'resolved_from_root': False,
        'is_split': False,
        'split_info': {},
        'detected_data_yaml': '',
        'data_yaml_candidates': [],
        'summary': '未识别出标准数据集目录结构',
        'directory_entries': children,
        'next_actions': ['请确认是否存在 images/ 与 labels/ 子目录，或直接提供准确的 img_dir/label_dir'],
    }



def resolve_dataset_inputs(path: str, label_dir: str = '') -> dict[str, Any]:
    resolution = resolve_dataset_root(path, label_dir)
    if not resolution.get('ok'):
        return resolution

    img_dir = resolution.get('img_dir') or path
    resolved_label = resolution.get('label_dir') or label_dir
    return {
        **resolution,
        'img_dir': str(Path(img_dir)),
        'label_dir': str(Path(resolved_label)) if resolved_label else '',
    }

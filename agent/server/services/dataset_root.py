from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGE_DIR_NAMES = {'images', 'imgs', 'jpegimages', 'pics', 'pictures', 'imageset'}
LABEL_DIR_NAMES = {'labels', 'annotations', 'label', 'ann', 'anns', 'txt_labels'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
LABEL_SUFFIXES = {'.txt', '.xml', '.json'}


def _count_files(path: Path, suffixes: set[str], max_depth: int = 2) -> int:
    if not path.is_dir():
        return 0
    count = 0
    base_depth = len(path.parts)
    try:
        for child in path.rglob('*'):
            if not child.is_file():
                continue
            if len(child.parts) - base_depth > max_depth:
                continue
            if child.suffix.lower() in suffixes:
                count += 1
    except OSError:
        return 0
    return count


def _find_named_subdir(base: Path, names: set[str]) -> Path | None:
    for name in names:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return None


def _find_best_subdir_by_content(base: Path, suffixes: set[str]) -> Path | None:
    if not base.is_dir():
        return None
    best: tuple[int, Path] | None = None
    try:
        children = [child for child in base.iterdir() if child.is_dir()]
    except OSError:
        return None
    for child in children:
        score = _count_files(child, suffixes)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, child)
    return best[1] if best else None


def _find_subdir(base: Path, names: set[str], suffixes: set[str] | None = None) -> tuple[Path | None, str]:
    named = _find_named_subdir(base, names)
    if named:
        return named, 'name'
    if suffixes is not None:
        guessed = _find_best_subdir_by_content(base, suffixes)
        if guessed:
            return guessed, 'content_score'
    return None, 'none'


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


def _has_direct_files(path: Path, suffixes: set[str]) -> bool:
    if not path.is_dir():
        return False
    try:
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in suffixes:
                return True
    except OSError:
        return False
    return False


def _has_split_subdirs(path: Path, suffixes: set[str]) -> bool:
    return (path / 'train').is_dir() and (path / 'val').is_dir() and (
        _count_files(path / 'train', suffixes, max_depth=1) > 0 or _count_files(path / 'val', suffixes, max_depth=1) > 0
    )


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
            'resolution_method': 'none',
            'is_split': False,
            'split_info': {},
            'detected_data_yaml': '',
            'data_yaml_candidates': [],
            'summary': '路径不存在',
            'next_actions': ['请确认数据集路径是否正确'],
        }

    if base.is_dir() and (
        base.name.lower() in IMAGE_DIR_NAMES or _has_direct_files(base, IMAGE_SUFFIXES) or _has_split_subdirs(base, IMAGE_SUFFIXES)
    ):
        dataset_root = base.parent
        inferred_label = explicit_label or _find_named_subdir(dataset_root, LABEL_DIR_NAMES)
        candidates = _detect_yaml_candidates(dataset_root)
        is_split = _has_split_subdirs(base, IMAGE_SUFFIXES)
        return {
            'ok': True,
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': 'images_dir',
            'img_dir': str(base.resolve()),
            'label_dir': str(inferred_label.resolve()) if inferred_label else '',
            'resolved_from_root': False,
            'resolution_method': 'direct',
            'is_split': is_split,
            'split_info': {
                'train_img_dir': str((base / 'train').resolve()),
                'val_img_dir': str((base / 'val').resolve()),
                'train_label_dir': str((inferred_label / 'train').resolve()) if inferred_label and (inferred_label / 'train').is_dir() else '',
                'val_label_dir': str((inferred_label / 'val').resolve()) if inferred_label and (inferred_label / 'val').is_dir() else '',
            } if is_split else {},
            'detected_data_yaml': candidates[0] if candidates else '',
            'data_yaml_candidates': candidates,
            'summary': '已直接使用图片目录',
            'next_actions': ['可直接用 img_dir/label_dir 调用 scan_dataset'],
        }

    images_dir, image_method = _find_subdir(base, IMAGE_DIR_NAMES, IMAGE_SUFFIXES) if base.is_dir() else (None, 'none')
    labels_dir = explicit_label
    label_method = 'explicit' if explicit_label else 'none'
    if labels_dir is None and base.is_dir():
        labels_dir, label_method = _find_subdir(base, LABEL_DIR_NAMES, LABEL_SUFFIXES)

    if images_dir and labels_dir:
        is_split = _has_split_subdirs(images_dir, IMAGE_SUFFIXES)
        split_info = {}
        if is_split:
            split_info = {
                'train_img_dir': str((images_dir / 'train').resolve()),
                'val_img_dir': str((images_dir / 'val').resolve()),
                'train_label_dir': str((labels_dir / 'train').resolve()) if (labels_dir / 'train').is_dir() else '',
                'val_label_dir': str((labels_dir / 'val').resolve()) if (labels_dir / 'val').is_dir() else '',
            }
        candidates = _detect_yaml_candidates(base)
        structure_type = 'yolo_split' if is_split else 'yolo_standard'
        if image_method == 'content_score' or label_method == 'content_score':
            structure_type = 'heuristic_split' if is_split else 'heuristic_standard'
        return {
            'ok': True,
            'dataset_root': str(base.resolve()),
            'structure_type': structure_type,
            'img_dir': str(images_dir.resolve()),
            'label_dir': str(labels_dir.resolve()),
            'resolved_from_root': True,
            'resolution_method': f'image={image_method},label={label_method}',
            'is_split': is_split,
            'split_info': split_info,
            'detected_data_yaml': candidates[0] if candidates else '',
            'data_yaml_candidates': candidates,
            'summary': '通过目录内容推断出数据集结构' if 'content_score' in {image_method, label_method} else ('检测到已划分数据集结构' if is_split else '检测到 YOLO 标准目录结构 (images/ + labels/)'),
            'next_actions': ['可直接用解析后的 img_dir/label_dir 调用 scan_dataset'],
        }

    if images_dir and not labels_dir:
        return {
            'ok': True,
            'dataset_root': str(base.resolve()),
            'structure_type': 'images_only',
            'img_dir': str(images_dir.resolve()),
            'label_dir': '',
            'resolved_from_root': True,
            'resolution_method': f'image={image_method},label=none',
            'is_split': _has_split_subdirs(images_dir, IMAGE_SUFFIXES),
            'split_info': {},
            'detected_data_yaml': '',
            'data_yaml_candidates': [],
            'summary': '只识别到图片目录，未找到标签目录',
            'next_actions': ['请显式提供 label_dir，或将标签目录整理为 labels/、ann/、annotations/ 等常见名称'],
        }

    if base.is_dir() and _has_direct_files(base, IMAGE_SUFFIXES):
        dataset_root = base.parent if base.name.lower() in IMAGE_DIR_NAMES else base
        candidates = _detect_yaml_candidates(dataset_root)
        return {
            'ok': True,
            'dataset_root': str(dataset_root.resolve()),
            'structure_type': 'flat',
            'img_dir': str(base.resolve()),
            'label_dir': str(explicit_label.resolve()) if explicit_label else '',
            'resolved_from_root': False,
            'resolution_method': 'flat',
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
        'resolution_method': 'none',
        'is_split': False,
        'split_info': {},
        'detected_data_yaml': '',
        'data_yaml_candidates': [],
        'summary': '未识别出标准数据集目录结构',
        'directory_entries': children,
        'next_actions': ['请确认是否存在 images/labels 或 pics/ann 等子目录，或直接提供准确的 img_dir/label_dir'],
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

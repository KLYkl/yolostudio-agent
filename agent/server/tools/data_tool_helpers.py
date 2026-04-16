from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from yolostudio_agent.agent.server.services.dataset_root import resolve_dataset_inputs, resolve_dataset_root

MAX_ISSUE_EXAMPLES = 3


def _error_payload(exc: Exception, action: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": f"{action}失败: {exc}",
        "error_type": exc.__class__.__name__,
    }


def _infer_dataset_root(img_dir: Path, label_dir: Path | None = None) -> Path:
    resolution = resolve_dataset_root(str(img_dir), str(label_dir) if label_dir else '')
    dataset_root = resolution.get('dataset_root') if isinstance(resolution, dict) else ''
    if dataset_root:
        return Path(dataset_root)
    return img_dir.parent if img_dir.name.lower() in {"images", "imgs", "jpegimages"} else img_dir


def _resolve_dataset_inputs(img_dir: str, label_dir: str = '') -> tuple[Path, Path | None, dict[str, Any]]:
    resolution = resolve_dataset_inputs(img_dir, label_dir)
    resolved_img = Path(resolution.get('img_dir') or img_dir)
    resolved_label = Path(resolution['label_dir']) if resolution.get('label_dir') else (Path(label_dir) if label_dir else None)
    return resolved_img, resolved_label, resolution


def _resolve_path_from_base(path_value: str, *, base_dir: Path) -> Path:
    candidate = Path(str(path_value or '').strip())
    if not candidate:
        return candidate
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _path_has_files(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    try:
        return any(child.is_file() for child in path.rglob('*'))
    except OSError:
        return False


def _discover_data_yaml(img_dir: Path, label_dir: Path | None = None) -> tuple[str, list[str]]:
    dataset_root = _infer_dataset_root(img_dir, label_dir)
    search_dirs = [dataset_root, dataset_root.parent, img_dir.parent, img_dir.parent.parent]
    names = {
        "data.yaml",
        "data.yml",
        "dataset.yaml",
        "dataset.yml",
        f"{dataset_root.name}.yaml",
        f"{dataset_root.name}.yml",
        f"{img_dir.parent.name}.yaml",
        f"{img_dir.parent.name}.yml",
    }
    candidates: list[str] = []
    seen: set[str] = set()
    for base in search_dirs:
        if not base:
            continue
        for name in names:
            path = (base / name).resolve()
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                candidates.append(key)
    detected = candidates[0] if candidates else ""
    return detected, candidates


def _discover_classes_txt(img_dir: Path, label_dir: Path | None = None) -> tuple[str, list[str]]:
    dataset_root = _infer_dataset_root(img_dir, label_dir)
    search_dirs = [
        label_dir,
        dataset_root / 'labels',
        dataset_root,
        img_dir.parent,
    ]
    names = {'classes.txt', 'class.txt', 'classes.names'}
    candidates: list[str] = []
    seen: set[str] = set()
    for base in search_dirs:
        if not base:
            continue
        for name in names:
            path = (base / name).resolve()
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                candidates.append(key)
    detected = candidates[0] if candidates else ''
    return detected, candidates


def _top_class_stats(class_stats: dict[str, int], limit: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(class_stats.items(), key=lambda item: item[1], reverse=True)
    return [{"class": name, "count": count} for name, count in ordered[:limit]]


def _format_issue_examples(result) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = {}
    if result.coord_errors:
        examples["coord_errors"] = [
            f"{path} [{where}] {reason}" for path, where, reason in result.coord_errors[:MAX_ISSUE_EXAMPLES]
        ]
    if result.class_errors:
        examples["class_errors"] = [
            f"{path} [{where}] {reason}" for path, where, reason in result.class_errors[:MAX_ISSUE_EXAMPLES]
        ]
    if result.format_errors:
        examples["format_errors"] = [
            f"{path} {reason}" for path, reason in result.format_errors[:MAX_ISSUE_EXAMPLES]
        ]
    if result.orphan_labels:
        examples["orphan_labels"] = [str(path) for path in result.orphan_labels[:MAX_ISSUE_EXAMPLES]]
    return examples


def _read_yaml_names(yaml_path: Path) -> list[str]:
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text(encoding='utf-8')) or {}
        names = data.get('names', {})
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
        if isinstance(names, list):
            return [str(x) for x in names]
    except Exception:
        pass
    return []


_WINDOWS_ABS_RE = re.compile(r'^[A-Za-z]:[\\/].*')


def _resolve_yaml_target_path(value: str, *, base_dir: Path) -> tuple[str, bool]:
    text = str(value or '').strip()
    if not text:
        return '', False
    if _WINDOWS_ABS_RE.match(text):
        return text.replace('\\', '/'), (os.name == 'nt' and Path(text).exists())
    if text.startswith('/'):
        target = Path(text)
        return str(target), target.exists()
    target = (base_dir / text).resolve()
    return str(target), target.exists()


def _inspect_training_yaml(yaml_path: str | Path) -> dict[str, Any]:
    path = Path(yaml_path)
    if not path.exists():
        return {
            'exists': False,
            'usable': False,
            'yaml_path': str(path),
            'base_dir': '',
            'issues': [{'field': 'yaml', 'raw': str(path), 'resolved': str(path), 'reason': 'missing_yaml_file'}],
            'resolved_targets': {},
            'warnings': ['data.yaml 文件不存在'],
        }

    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except Exception as exc:
        return {
            'exists': True,
            'usable': False,
            'yaml_path': str(path.resolve()),
            'base_dir': '',
            'issues': [{'field': 'yaml', 'raw': str(path), 'resolved': str(path.resolve()), 'reason': f'invalid_yaml:{exc.__class__.__name__}'}],
            'resolved_targets': {},
            'warnings': [f'data.yaml 解析失败: {exc}'],
        }

    yaml_dir = path.resolve().parent
    path_value = str(payload.get('path') or '').strip()
    if path_value:
        base_dir_text, base_exists = _resolve_yaml_target_path(path_value, base_dir=yaml_dir)
        base_issue = ''
        if _WINDOWS_ABS_RE.match(path_value) and os.name != 'nt':
            base_issue = 'windows_absolute_path_on_non_windows'
        elif base_dir_text and not base_exists:
            base_issue = 'missing_base_path'
        if base_issue:
            base_dir = yaml_dir
            issues = [{
                'field': 'path',
                'raw': path_value,
                'resolved': base_dir_text,
                'reason': base_issue,
            }]
        else:
            base_dir = Path(base_dir_text) if base_dir_text else yaml_dir
            issues = []
    else:
        base_dir_text = str(yaml_dir)
        base_dir = yaml_dir
        issues = []

    resolved_targets: dict[str, list[str]] = {}
    for field in ('train', 'val', 'test'):
        raw_value = payload.get(field)
        if not raw_value:
            continue
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        resolved_list: list[str] = []
        for item in values:
            resolved_text, exists = _resolve_yaml_target_path(str(item), base_dir=base_dir)
            if resolved_text:
                resolved_list.append(resolved_text)
            if not exists:
                issues.append({
                    'field': field,
                    'raw': str(item),
                    'resolved': resolved_text,
                    'reason': 'missing_target',
                })
        if resolved_list:
            resolved_targets[field] = resolved_list

    warnings: list[str] = []
    if issues:
        warnings.append('data.yaml 已存在，但其中 train/val 路径在当前机器不可用')

    return {
        'exists': True,
        'usable': not issues,
        'yaml_path': str(path.resolve()),
        'base_dir': base_dir_text,
        'issues': issues,
        'resolved_targets': resolved_targets,
        'warnings': warnings,
    }


def _read_classes_txt_lines(classes_txt_path: Path) -> list[str]:
    try:
        return [line.strip() for line in classes_txt_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except Exception:
        return []


def _parse_class_names_text(classes_text: str) -> list[str]:
    text = str(classes_text or '').strip()
    if not text:
        return []

    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    normalized = normalized.replace('，', ',').replace('、', ',').replace('；', ';')
    lines = [line.strip() for line in normalized.split('\n') if line.strip()]

    def _clean_item(value: str) -> str:
        item = str(value or '').strip()
        item = re.sub(r'^\d+\s*[\.\)\-]\s*', '', item)
        item = re.sub(r'^(?:class|类别)\s*[:：-]?\s*', '', item, flags=re.I)
        return item.strip()

    candidates: list[str] = []
    if len(lines) > 1:
        for line in lines:
            cleaned = _clean_item(line)
            if cleaned:
                candidates.append(cleaned)
    else:
        for chunk in re.split(r'[\n,;]+', normalized):
            cleaned = _clean_item(chunk)
            if cleaned:
                candidates.append(cleaned)

    seen: set[str] = set()
    parsed: list[str] = []
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        parsed.append(item)
    return parsed


def _build_missing_label_risk(scan_result) -> dict[str, Any]:
    total_images = getattr(scan_result, 'total_images', 0) or 0
    missing_count = len(getattr(scan_result, 'missing_labels', []) or [])
    ratio = round(missing_count / total_images, 4) if total_images else 0.0
    warnings: list[str] = []
    risk_level = 'none'
    if missing_count <= 0:
        return {
            'missing_label_images': 0,
            'missing_label_ratio': 0.0,
            'risk_level': risk_level,
            'warnings': warnings,
        }

    if ratio >= 0.5:
        risk_level = 'critical'
    elif ratio >= 0.2:
        risk_level = 'high'
    elif ratio >= 0.05:
        risk_level = 'medium'
    else:
        risk_level = 'low'

    warnings.append(
        f'发现 {missing_count} 张图片缺少标签（占比 {ratio:.1%}），训练结果可能受到明显影响'
    )
    return {
        'missing_label_images': missing_count,
        'missing_label_ratio': ratio,
        'risk_level': risk_level,
        'warnings': warnings,
    }


_RISK_LEVEL_ORDER = {
    'none': 0,
    'low': 1,
    'medium': 2,
    'high': 3,
    'critical': 4,
}


def _merge_risk_levels(*levels: str) -> str:
    best = 'none'
    best_rank = -1
    for level in levels:
        rank = _RISK_LEVEL_ORDER.get(str(level or 'none').lower(), -1)
        if rank > best_rank:
            best_rank = rank
            best = str(level or 'none').lower()
    return best


def _sample_path_strings(paths: list[Path], limit: int = MAX_ISSUE_EXAMPLES) -> list[str]:
    return [str(path) for path in paths[:limit]]


def _sample_integrity_entries(entries: list[tuple[Any, ...]], limit: int = MAX_ISSUE_EXAMPLES) -> list[str]:
    samples: list[str] = []
    for item in entries[:limit]:
        if isinstance(item, (str, Path)):
            samples.append(str(item))
        elif not isinstance(item, tuple):
            samples.append(str(item))
        elif len(item) == 2:
            path, reason = item
            samples.append(f'{path} - {reason}')
        elif len(item) == 3:
            path, left, right = item
            samples.append(f'{path} - {left} -> {right}')
        else:
            samples.append(' | '.join(str(part) for part in item))
    return samples


def _serialize_duplicate_groups(duplicates, max_groups: int = MAX_ISSUE_EXAMPLES, max_paths_per_group: int = MAX_ISSUE_EXAMPLES) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for group in duplicates[:max_groups]:
        paths = [str(path) for path in group.paths[:max_paths_per_group]]
        groups.append({
            'hash_value': group.hash_value,
            'count': len(group.paths),
            'paths': paths,
            'truncated_paths': max(0, len(group.paths) - len(paths)),
        })
    return groups


def _summarize_health_outputs(integrity, sizes, duplicates, *, include_duplicates: bool) -> dict[str, Any]:
    corrupted_count = len(getattr(integrity, 'corrupted', []) or [])
    zero_bytes_count = len(getattr(integrity, 'zero_bytes', []) or [])
    format_mismatch_count = len(getattr(integrity, 'format_mismatch', []) or [])
    exif_rotation_count = len(getattr(integrity, 'exif_rotation', []) or [])
    abnormal_small_count = len(getattr(sizes, 'abnormal_small', []) or [])
    abnormal_large_count = len(getattr(sizes, 'abnormal_large', []) or [])
    duplicate_group_count = len(duplicates or [])
    duplicate_files_total = sum(len(group.paths) for group in duplicates or [])
    duplicate_extra_files = sum(max(len(group.paths) - 1, 0) for group in duplicates or [])

    risk_level = _merge_risk_levels(
        'critical' if (corrupted_count or zero_bytes_count) else 'none',
        'high' if format_mismatch_count else 'none',
        'medium' if (abnormal_small_count or abnormal_large_count) else 'none',
        'medium' if duplicate_group_count else 'none',
        'low' if exif_rotation_count else 'none',
    )

    warnings: list[str] = []
    if corrupted_count:
        warnings.append(f'发现 {corrupted_count} 张损坏图片')
    if zero_bytes_count:
        warnings.append(f'发现 {zero_bytes_count} 个零字节文件')
    if format_mismatch_count:
        warnings.append(f'发现 {format_mismatch_count} 个文件扩展名与真实格式不匹配')
    if abnormal_small_count:
        warnings.append(f'发现 {abnormal_small_count} 张异常小图片')
    if abnormal_large_count:
        warnings.append(f'发现 {abnormal_large_count} 张异常大图片')
    if include_duplicates and duplicate_group_count:
        warnings.append(f'发现 {duplicate_group_count} 组重复图片（额外重复文件 {duplicate_extra_files} 个）')
    if exif_rotation_count:
        warnings.append(f'发现 {exif_rotation_count} 张图片带 EXIF 旋转标记')

    issue_count = (
        getattr(integrity, 'issue_count', 0)
        + abnormal_small_count
        + abnormal_large_count
        + duplicate_group_count
    )
    return {
        'health_ok': issue_count == 0,
        'issue_count': issue_count,
        'integrity': {
            'corrupted_count': corrupted_count,
            'zero_bytes_count': zero_bytes_count,
            'format_mismatch_count': format_mismatch_count,
            'exif_rotation_count': exif_rotation_count,
            'corrupted_samples': _sample_integrity_entries(getattr(integrity, 'corrupted', []) or []),
            'zero_bytes_samples': _sample_path_strings(getattr(integrity, 'zero_bytes', []) or []),
            'format_mismatch_samples': _sample_integrity_entries(getattr(integrity, 'format_mismatch', []) or []),
            'exif_rotation_samples': _sample_path_strings(getattr(integrity, 'exif_rotation', []) or []),
        },
        'sizes': {
            'abnormal_small_count': abnormal_small_count,
            'abnormal_large_count': abnormal_large_count,
            'abnormal_small_samples': _sample_integrity_entries(getattr(sizes, 'abnormal_small', []) or []),
            'abnormal_large_samples': _sample_integrity_entries(getattr(sizes, 'abnormal_large', []) or []),
        },
        'duplicates': {
            'group_count': duplicate_group_count,
            'files_total': duplicate_files_total,
            'extra_files': duplicate_extra_files,
            'samples': _serialize_duplicate_groups(duplicates or []),
        },
        'risk_level': risk_level,
        'warnings': warnings,
    }

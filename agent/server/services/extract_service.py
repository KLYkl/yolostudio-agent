from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Iterable

from agent_plan.agent.server.services.dataset_root import resolve_dataset_inputs

MAX_EXAMPLES = 3


def _error_payload(action: str, exc: Exception) -> dict[str, Any]:
    return {
        'ok': False,
        'error': f'{action}失败: {exc}',
        'error_type': exc.__class__.__name__,
        'summary': f'{action}失败',
        'next_actions': ['请查看错误信息并调整参数后重试'],
    }


def _install_headless_pyside6_stub() -> None:
    import sys
    import types

    if 'PySide6.QtCore' in sys.modules:
        return

    class _DummySignal:
        def __init__(self, *args, **kwargs) -> None:
            self._slots: list[Any] = []

        def connect(self, slot) -> None:
            self._slots.append(slot)

        def emit(self, *args, **kwargs) -> None:
            for slot in list(self._slots):
                try:
                    slot(*args, **kwargs)
                except Exception:
                    pass

    class _DummyQThread:
        def __init__(self, parent=None) -> None:
            self.parent = parent

        def start(self) -> None:
            raise RuntimeError('当前为 headless PySide6 stub，不支持启动 QThread')

    qtcore = types.ModuleType('PySide6.QtCore')
    qtcore.QThread = _DummyQThread
    qtcore.Signal = lambda *args, **kwargs: _DummySignal()

    pyside6 = types.ModuleType('PySide6')
    pyside6.QtCore = qtcore

    sys.modules.setdefault('PySide6', pyside6)
    sys.modules['PySide6.QtCore'] = qtcore


def _import_core_data_handler_module(module_name: str):
    import importlib
    import sys

    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        message = str(exc)
        if 'PySide6' not in message and 'QtCore' not in message:
            raise
        _install_headless_pyside6_stub()
        for name in ['core.data_handler', 'core.data_handler._worker']:
            sys.modules.pop(name, None)
        return importlib.import_module(module_name)


def _get_data_handler_cls():
    return _import_core_data_handler_module('core.data_handler._handler').DataHandler


def _get_data_models_module():
    return _import_core_data_handler_module('core.data_handler._models')


def _normalize_rel_dir(value: str) -> str:
    cleaned = str(value or '').strip().replace('\\', '/')
    if not cleaned or cleaned == '.':
        return '.'
    return cleaned.strip('/') or '.'


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace('；', ',').replace(';', ',').split(',')]
        return [_normalize_rel_dir(part) for part in parts if part.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        items: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(_normalize_rel_dir(text))
        return items
    return []


def _sample_strings(paths: list[Path], limit: int = MAX_EXAMPLES) -> list[str]:
    return [str(path) for path in paths[:limit]]


def _group_by_directory(images: list[Path], img_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for image_path in images:
        try:
            rel_dir = str(image_path.parent.relative_to(img_dir))
        except ValueError:
            rel_dir = '.'
        rel_dir = _normalize_rel_dir(rel_dir)
        grouped.setdefault(rel_dir, []).append(image_path)
    return grouped


def _dir_stats_from_images(images: list[Path], img_dir: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    for image_path in images:
        try:
            rel_dir = str(image_path.parent.relative_to(img_dir))
        except ValueError:
            rel_dir = '.'
        rel_dir = _normalize_rel_dir(rel_dir)
        stats[rel_dir] = stats.get(rel_dir, 0) + 1
    return dict(sorted(stats.items(), key=lambda item: item[0]))


def _resolve_selection_count(total: int, selection_mode: str, count: int, ratio: float) -> int:
    if total <= 0:
        return 0
    mode = str(selection_mode or 'count').lower()
    if mode == 'all':
        return total
    if mode == 'count':
        if count <= 0:
            raise ValueError('selection_mode=count 时 count 必须大于 0')
        return min(count, total)
    if mode == 'ratio':
        if ratio <= 0 or ratio > 1:
            raise ValueError('selection_mode=ratio 时 ratio 必须在 (0, 1] 范围内')
        return min(total, max(1, int(total * ratio)))
    raise ValueError(f'不支持的 selection_mode: {selection_mode}')


def _plan_selection(
    images: list[Path],
    img_dir: Path,
    *,
    selection_mode: str,
    count: int,
    ratio: float,
    grouping_mode: str,
    seed: int,
) -> list[Path]:
    rng = random.Random(seed)
    mode = str(grouping_mode or 'global').lower()
    if mode == 'global':
        planned = _resolve_selection_count(len(images), selection_mode, count, ratio)
        if planned <= 0:
            return []
        if planned >= len(images):
            return sorted(images)
        return sorted(rng.sample(images, planned))
    if mode == 'per_directory':
        grouped = _group_by_directory(images, img_dir)
        selected: list[Path] = []
        for _, group_images in sorted(grouped.items(), key=lambda item: item[0]):
            planned = _resolve_selection_count(len(group_images), selection_mode, count, ratio)
            if planned <= 0:
                continue
            if planned >= len(group_images):
                selected.extend(group_images)
            else:
                selected.extend(rng.sample(group_images, planned))
        return sorted(selected)
    raise ValueError(f'不支持的 grouping_mode: {grouping_mode}')


def _default_extract_output_dir(img_dir: Path) -> Path:
    return (img_dir.parent / f'{img_dir.name}_extracted').resolve()


def _build_extract_dest_paths(
    handler: Any,
    image_path: Path,
    *,
    img_dir: Path,
    label_dir: Path | None,
    output_root: Path,
    output_layout: str,
) -> tuple[Path, Path | None, Path | None]:
    dest_img = handler._build_extract_dest_path(
        image_path,
        img_dir,
        output_root,
        output_layout,
        category=None,
    )
    label_path = handler._find_extract_label(image_path, label_dir, img_dir)
    dest_label: Path | None = None
    if label_path and label_path.exists():
        dest_label = handler._build_extract_dest_path(
            label_path,
            label_dir if label_dir else img_dir.parent,
            output_root,
            output_layout,
            label_mode=True,
            img_name=image_path.stem,
            category=None,
        )
    return dest_img, label_path, dest_label


def _build_flat_base_dest(source: Path, *, output_root: Path, label_mode: bool = False) -> Path:
    sub_dir = 'labels' if label_mode else 'images'
    return output_root / sub_dir / source.name


class ExtractService:
    def _resolve_image_source(self, source_path: str, label_dir: str = '') -> tuple[Path, Path | None, Path, dict[str, Any]]:
        resolution = resolve_dataset_inputs(source_path, label_dir)
        if not resolution.get('ok', True):
            raise ValueError(resolution.get('error') or f'无法解析路径: {source_path}')
        img_dir = Path(resolution.get('img_dir') or source_path).resolve()
        if not img_dir.exists():
            raise FileNotFoundError(f'路径不存在: {img_dir}')
        if img_dir.is_file():
            raise ValueError('图片抽取只支持目录输入，不支持单个文件')
        resolved_label = Path(resolution['label_dir']).resolve() if resolution.get('label_dir') else (Path(label_dir).resolve() if label_dir else None)
        dataset_root = Path(resolution.get('dataset_root') or img_dir).resolve()
        return img_dir, resolved_label, dataset_root, resolution

    def _collect_extract_images(
        self,
        handler: Any,
        img_dir: Path,
        *,
        selected_dirs: list[str],
    ) -> tuple[list[Path], dict[str, int], list[str]]:
        directory_counts = { _normalize_rel_dir(key): value for key, value in handler.scan_subdirs(img_dir).items() }
        missing_dirs: list[str] = []
        if not selected_dirs:
            return sorted(handler._find_images(img_dir)), directory_counts, missing_dirs

        collected: list[Path] = []
        for rel_dir in selected_dirs:
            normalized = _normalize_rel_dir(rel_dir)
            if normalized == '.':
                if directory_counts.get('.'):
                    collected.extend(handler._find_images_flat(img_dir))
                else:
                    missing_dirs.append(normalized)
                continue
            target_dir = img_dir / normalized
            if target_dir.exists() and target_dir.is_dir():
                collected.extend(handler._find_images(target_dir))
            else:
                missing_dirs.append(normalized)
        return sorted(set(collected)), directory_counts, missing_dirs

    def _estimate_extract_preview(
        self,
        *,
        source_path: str,
        label_dir: str = '',
        output_dir: str = '',
        selection_mode: str = 'count',
        count: int = 100,
        ratio: float = 0.1,
        grouping_mode: str = 'global',
        selected_dirs: list[str] | str | None = None,
        copy_labels: bool = True,
        output_layout: str = 'flat',
        seed: int = 42,
        max_examples: int = MAX_EXAMPLES,
    ) -> dict[str, Any]:
        DataHandler = _get_data_handler_cls()
        handler = DataHandler()
        img_dir, resolved_label_dir, dataset_root, resolution = self._resolve_image_source(source_path, label_dir)
        normalized_dirs = _normalize_str_list(selected_dirs)
        images, directory_counts, missing_dirs = self._collect_extract_images(handler, img_dir, selected_dirs=normalized_dirs)
        selected = _plan_selection(
            images,
            img_dir,
            selection_mode=selection_mode,
            count=count,
            ratio=ratio,
            grouping_mode=grouping_mode,
            seed=seed,
        )
        planned_dir_stats = _dir_stats_from_images(selected, img_dir)
        requested_output_dir = Path(output_dir).resolve() if output_dir else _default_extract_output_dir(img_dir)
        copy_labels_effective = bool(copy_labels and resolved_label_dir and resolved_label_dir.exists())
        label_missing_count = 0
        conflict_count = 0
        conflict_samples: list[str] = []

        for image_path in selected:
            dest_img, label_path, dest_label = _build_extract_dest_paths(
                handler,
                image_path,
                img_dir=img_dir,
                label_dir=resolved_label_dir,
                output_root=requested_output_dir,
                output_layout=output_layout,
            )
            if output_layout == 'flat':
                base_img_dest = _build_flat_base_dest(image_path, output_root=requested_output_dir, label_mode=False)
                if base_img_dest.exists():
                    conflict_count += 1
                    if len(conflict_samples) < max_examples:
                        conflict_samples.append(str(base_img_dest))
            elif dest_img.exists():
                conflict_count += 1
                if len(conflict_samples) < max_examples:
                    conflict_samples.append(str(dest_img))
            if copy_labels_effective:
                if label_path is None or not label_path.exists():
                    label_missing_count += 1
                elif dest_label:
                    if output_layout == 'flat':
                        base_label_dest = _build_flat_base_dest(label_path, output_root=requested_output_dir, label_mode=True)
                        if base_label_dest.exists():
                            conflict_count += 1
                            if len(conflict_samples) < max_examples:
                                conflict_samples.append(str(base_label_dest))
                    elif dest_label.exists():
                        conflict_count += 1
                        if len(conflict_samples) < max_examples:
                            conflict_samples.append(str(dest_label))

        warnings: list[str] = []
        if missing_dirs:
            warnings.append(f'部分 selected_dirs 不存在: {", ".join(missing_dirs[:max_examples])}')
        if copy_labels and not copy_labels_effective:
            warnings.append('当前未解析到可用标签目录，本次只能抽取图片，无法同时复制标签')
        if label_missing_count:
            warnings.append(f'计划抽取的样本中有 {label_missing_count} 张图片未找到对应标签')
        if conflict_count:
            warnings.append(f'目标输出目录已存在 {conflict_count} 个潜在同名冲突')
        if output_layout != 'flat':
            warnings.append('当前输出布局不是 flat；若后续要直接接 scan/validate/训练链，建议改用 flat')

        workflow_ready_path = str(requested_output_dir) if output_layout == 'flat' else ''
        summary = (
            f'预览完成: 可用图片 {len(images)} 张，计划抽取 {len(selected)} 张'
            f'（{grouping_mode} / {selection_mode}）'
        )
        if normalized_dirs:
            summary += f'，目录过滤 {len(normalized_dirs)} 个'

        next_actions = []
        if selected:
            next_actions.append('可继续调用 extract_images 真正执行抽取')
            if workflow_ready_path:
                next_actions.append(f'抽取完成后可直接对输出目录继续 scan_dataset / validate_dataset: {workflow_ready_path}')
        else:
            next_actions.append('当前没有可抽取的样本；请调整 selected_dirs、count 或 ratio 后重试')

        return {
            'ok': True,
            'summary': summary,
            'dataset_root': str(dataset_root),
            'structure_type': resolution.get('structure_type'),
            'resolved_from_root': resolution.get('resolved_from_root', False),
            'resolved_img_dir': str(img_dir),
            'resolved_label_dir': str(resolved_label_dir) if resolved_label_dir else '',
            'selection_mode': selection_mode,
            'count': count,
            'ratio': ratio,
            'grouping_mode': grouping_mode,
            'selected_dirs': normalized_dirs,
            'available_images': len(images),
            'available_dir_stats': directory_counts,
            'planned_extract_count': len(selected),
            'planned_dir_stats': planned_dir_stats,
            'copy_labels_requested': bool(copy_labels),
            'copy_labels_effective': copy_labels_effective,
            'label_missing_count': label_missing_count,
            'output_layout': output_layout,
            'output_dir': str(requested_output_dir),
            'workflow_ready_path': workflow_ready_path,
            'workflow_ready': bool(workflow_ready_path and copy_labels_effective),
            'sample_images': _sample_strings(selected, max_examples),
            'conflict_count': conflict_count,
            'conflict_samples': conflict_samples,
            'warnings': warnings,
            'artifacts': {
                'planned_output_dir': str(requested_output_dir),
            },
            'next_actions': next_actions,
        }

    def preview_extract_images(self, **kwargs: Any) -> dict[str, Any]:
        try:
            return self._estimate_extract_preview(**kwargs)
        except Exception as exc:
            return _error_payload('预览图片抽取', exc)

    def extract_images(
        self,
        source_path: str,
        label_dir: str = '',
        output_dir: str = '',
        selection_mode: str = 'count',
        count: int = 100,
        ratio: float = 0.1,
        grouping_mode: str = 'global',
        selected_dirs: list[str] | str | None = None,
        copy_labels: bool = True,
        output_layout: str = 'flat',
        seed: int = 42,
        max_examples: int = MAX_EXAMPLES,
    ) -> dict[str, Any]:
        try:
            preview = self._estimate_extract_preview(
                source_path=source_path,
                label_dir=label_dir,
                output_dir=output_dir,
                selection_mode=selection_mode,
                count=count,
                ratio=ratio,
                grouping_mode=grouping_mode,
                selected_dirs=selected_dirs,
                copy_labels=copy_labels,
                output_layout=output_layout,
                seed=seed,
                max_examples=max_examples,
            )
            if not preview.get('ok'):
                return preview

            DataHandler = _get_data_handler_cls()
            DataModels = _get_data_models_module()
            handler = DataHandler()
            img_dir = Path(preview['resolved_img_dir'])
            resolved_label_dir = Path(preview['resolved_label_dir']) if preview.get('resolved_label_dir') else None

            images, _, _ = self._collect_extract_images(handler, img_dir, selected_dirs=preview.get('selected_dirs') or [])
            selected = _plan_selection(
                images,
                img_dir,
                selection_mode=selection_mode,
                count=count,
                ratio=ratio,
                grouping_mode=grouping_mode,
                seed=seed,
            )

            explicit_output = bool(output_dir)
            if explicit_output:
                target_output_dir = Path(output_dir).resolve()
                target_output_dir.mkdir(parents=True, exist_ok=True)
            else:
                target_output_dir = DataModels._get_unique_dir(_default_extract_output_dir(img_dir))
                target_output_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            labels_copied = 0
            label_missing_count = 0
            occupied_slots = 0
            skipped_conflicts = 0
            conflicts: list[str] = []
            copied_images: list[Path] = []

            for image_path in selected:
                dest_img, label_path, dest_label = _build_extract_dest_paths(
                    handler,
                    image_path,
                    img_dir=img_dir,
                    label_dir=resolved_label_dir,
                    output_root=target_output_dir,
                    output_layout=output_layout,
                )
                if output_layout == 'flat':
                    base_img_dest = _build_flat_base_dest(image_path, output_root=target_output_dir, label_mode=False)
                    if base_img_dest.exists():
                        occupied_slots += 1
                        if len(conflicts) < max_examples:
                            conflicts.append(str(base_img_dest))
                if dest_img.exists():
                    skipped_conflicts += 1
                    if len(conflicts) < max_examples:
                        conflicts.append(str(dest_img))
                    continue
                dest_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(image_path), str(dest_img))
                copied += 1
                copied_images.append(image_path)

                if preview.get('copy_labels_effective'):
                    if label_path is None or not label_path.exists():
                        label_missing_count += 1
                    elif dest_label is not None:
                        if output_layout == 'flat':
                            base_label_dest = _build_flat_base_dest(label_path, output_root=target_output_dir, label_mode=True)
                            if base_label_dest.exists():
                                occupied_slots += 1
                                if len(conflicts) < max_examples:
                                    conflicts.append(str(base_label_dest))
                        dest_label.parent.mkdir(parents=True, exist_ok=True)
                        if not dest_label.exists():
                            shutil.copy2(str(label_path), str(dest_label))
                            labels_copied += 1
                        else:
                            skipped_conflicts += 1
                            if len(conflicts) < max_examples:
                                conflicts.append(str(dest_label))

            dir_stats = _dir_stats_from_images(copied_images, img_dir)
            warnings = list(preview.get('warnings') or [])
            if label_missing_count:
                missing_warning = f'实际抽取结果中有 {label_missing_count} 张图片未复制到标签'
                if missing_warning not in warnings:
                    warnings.append(missing_warning)
            if occupied_slots:
                warnings.append(f'目标输出目录中已有 {occupied_slots} 个同名槽位，本次已自动改名避免覆盖')
            if skipped_conflicts:
                warnings.append(f'实际执行遇到 {skipped_conflicts} 个无法写入的同名冲突，已跳过这些文件')

            output_img_dir = str((target_output_dir / 'images').resolve()) if output_layout == 'flat' else ''
            output_label_dir = str((target_output_dir / 'labels').resolve()) if output_layout == 'flat' and preview.get('copy_labels_effective') else ''
            workflow_ready_path = str(target_output_dir.resolve()) if output_layout == 'flat' else ''
            workflow_ready = bool(workflow_ready_path and output_layout == 'flat' and preview.get('copy_labels_effective'))
            summary = (
                f'图片抽取完成: 实际抽取 {copied} 张图片'
                + (f'，复制标签 {labels_copied} 个' if labels_copied else '')
                + (f'，冲突 {len(conflicts)} 个' if conflicts else '')
            )
            next_actions = []
            if workflow_ready_path:
                next_actions.append(f'可直接对输出目录继续 scan_dataset / validate_dataset: {workflow_ready_path}')
                if workflow_ready:
                    next_actions.append(f'如需进入训练链，可继续 prepare_dataset_for_training(dataset_path={workflow_ready_path})')
            else:
                next_actions.append('如需继续接训练链，建议下次使用 output_layout=flat')

            return {
                'ok': True,
                'summary': summary,
                'dataset_root': str(target_output_dir.resolve()) if workflow_ready_path else preview.get('dataset_root', ''),
                'source_path': source_path,
                'resolved_img_dir': preview.get('resolved_img_dir', ''),
                'resolved_label_dir': preview.get('resolved_label_dir', ''),
                'selection_mode': selection_mode,
                'count': count,
                'ratio': ratio,
                'grouping_mode': grouping_mode,
                'selected_dirs': preview.get('selected_dirs', []),
                'available_images': preview.get('available_images', 0),
                'extracted': copied,
                'labels_copied': labels_copied,
                'label_missing_count': label_missing_count,
                'dir_stats': dir_stats,
                'output_layout': output_layout,
                'output_dir': str(target_output_dir.resolve()),
                'output_img_dir': output_img_dir,
                'output_label_dir': output_label_dir,
                'workflow_ready_path': workflow_ready_path,
                'workflow_ready': workflow_ready,
                'copy_labels_requested': bool(copy_labels),
                'copy_labels_effective': bool(preview.get('copy_labels_effective')),
                'conflict_count': occupied_slots + skipped_conflicts,
                'conflict_samples': conflicts[:max_examples],
                'sample_images': _sample_strings(copied_images, max_examples),
                'warnings': warnings,
                'artifacts': {
                    'output_dir': str(target_output_dir.resolve()),
                    'output_img_dir': output_img_dir,
                    'output_label_dir': output_label_dir,
                },
                'next_actions': next_actions,
            }
        except Exception as exc:
            return _error_payload('执行图片抽取', exc)

    def scan_videos(self, source_path: str, max_examples: int = MAX_EXAMPLES) -> dict[str, Any]:
        try:
            DataHandler = _get_data_handler_cls()
            handler = DataHandler()
            video_path = Path(source_path).resolve()
            if not video_path.exists():
                raise FileNotFoundError(f'路径不存在: {video_path}')

            directory_counts = { _normalize_rel_dir(key): value for key, value in handler.scan_videos(video_path).items() }
            videos = handler._collect_videos(video_path)
            summary = f'视频扫描完成: 发现 {len(videos)} 个视频文件'
            next_actions = []
            if videos:
                next_actions.append('如需抽帧，可继续调用 extract_video_frames')
                next_actions.append('如需直接做视频推理，可调用 predict_videos')
            else:
                next_actions.append('请确认目录中是否包含 .mp4/.avi/.mkv 等视频文件')
            return {
                'ok': True,
                'summary': summary,
                'source_path': str(video_path),
                'directory_counts': directory_counts,
                'total_videos': len(videos),
                'sample_videos': _sample_strings(videos, max_examples),
                'warnings': [],
                'artifacts': {},
                'next_actions': next_actions,
            }
        except Exception as exc:
            return _error_payload('扫描视频目录', exc)

    def extract_video_frames(
        self,
        source_path: str,
        output_dir: str = '',
        mode: str = 'interval',
        frame_interval: int = 30,
        time_interval: float = 1.0,
        scene_threshold: float = 0.4,
        min_scene_gap: int = 15,
        enable_dedup: bool = True,
        dedup_threshold: int = 8,
        max_frames: int = 0,
        start_time: float = 0.0,
        end_time: float = 0.0,
        output_format: str = 'jpg',
        jpg_quality: int = 95,
        name_prefix: str = '',
    ) -> dict[str, Any]:
        try:
            DataHandler = _get_data_handler_cls()
            DataModels = _get_data_models_module()
            handler = DataHandler()
            video_path = Path(source_path).resolve()
            if not video_path.exists():
                raise FileNotFoundError(f'路径不存在: {video_path}')

            mode = str(mode or 'interval').lower()
            if mode not in {'interval', 'time', 'scene'}:
                raise ValueError(f'不支持的抽帧模式: {mode}')

            config = DataModels.VideoExtractConfig(
                mode=mode,
                frame_interval=frame_interval,
                time_interval=time_interval,
                scene_threshold=scene_threshold,
                min_scene_gap=min_scene_gap,
                enable_dedup=enable_dedup,
                dedup_threshold=dedup_threshold,
                max_frames=max_frames,
                start_time=start_time,
                end_time=end_time,
                output_format=output_format,
                jpg_quality=jpg_quality,
                name_prefix=name_prefix,
                output_dir=Path(output_dir).resolve() if output_dir else None,
            )
            result = handler.extract_video_frames(video_path, config=config)
            warnings: list[str] = []
            if result.final_count <= 0:
                warnings.append('未保留任何有效帧，请检查时间区间、抽帧模式或视频内容')
            if result.dedup_removed > 0:
                warnings.append(f'去重移除了 {result.dedup_removed} 帧')
            next_actions = []
            if result.output_dir:
                next_actions.append(f'可将抽帧输出目录继续作为图片输入使用: {result.output_dir}')
            next_actions.append('如需训练，后续仍需为抽帧结果补充或整理标签')
            summary = (
                f'视频抽帧完成: 最终保留 {result.final_count} 帧'
                f'（原始抽取 {result.extracted} / 去重移除 {result.dedup_removed}）'
            )
            return {
                'ok': True,
                'summary': summary,
                'source_path': str(video_path),
                'mode': mode,
                'frame_interval': frame_interval,
                'time_interval': time_interval,
                'scene_threshold': scene_threshold,
                'max_frames': max_frames,
                'output_dir': result.output_dir,
                'total_frames': result.total_frames,
                'extracted': result.extracted,
                'dedup_removed': result.dedup_removed,
                'final_count': result.final_count,
                'video_stats': result.video_stats,
                'duration_seconds': round(result.duration, 3),
                'skipped': result.skipped,
                'warnings': warnings,
                'artifacts': {
                    'output_dir': result.output_dir,
                },
                'next_actions': next_actions,
            }
        except Exception as exc:
            return _error_payload('执行视频抽帧', exc)

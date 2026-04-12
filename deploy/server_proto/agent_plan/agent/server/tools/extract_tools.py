from __future__ import annotations

from typing import Any, Callable

from yolostudio_agent.agent.server.services.extract_service import ExtractService

service = ExtractService()


def _wrap(action: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, dict):
            return result
        return {'ok': True, 'result': result}
    except Exception as exc:
        return {
            'ok': False,
            'error': f'{action}失败: {exc}',
            'error_type': exc.__class__.__name__,
            'summary': f'{action}失败',
            'next_actions': ['请查看错误信息并调整参数后重试'],
        }


def preview_extract_images(
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
    max_examples: int = 3,
) -> dict[str, Any]:
    """预览图片抽取计划。默认以 flat 布局输出，方便后续直接接 scan/validate/训练链；不会修改原始数据。"""
    return _wrap(
        '预览图片抽取',
        service.preview_extract_images,
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


def extract_images(
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
    max_examples: int = 3,
) -> dict[str, Any]:
    """执行图片抽取。默认输出为 flat 数据集结构，便于后续直接继续 scan_dataset / validate_dataset / prepare_dataset_for_training。"""
    return _wrap(
        '执行图片抽取',
        service.extract_images,
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


def scan_videos(source_path: str, max_examples: int = 3) -> dict[str, Any]:
    """扫描视频目录或单个视频，返回可处理视频数量与目录分布。"""
    return _wrap('扫描视频目录', service.scan_videos, source_path=source_path, max_examples=max_examples)


def extract_video_frames(
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
    """执行视频抽帧。支持 interval/time/scene 三种最常见模式，输出目录可继续作为图片输入使用。"""
    return _wrap(
        '执行视频抽帧',
        service.extract_video_frames,
        source_path=source_path,
        output_dir=output_dir,
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
    )

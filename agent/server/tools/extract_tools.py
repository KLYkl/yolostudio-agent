from __future__ import annotations

from typing import Annotated, Any, Callable

from pydantic import Field
from yolostudio_agent.agent.server.services.extract_service import ExtractService

service = ExtractService()

_SOURCE_PATH_PARAM = Annotated[
    str,
    Field(
        description='待抽取图片或视频的源路径。可以是单个文件，也可以是目录。',
        examples=['/data/images', '/data/videos/demo.mp4'],
    ),
]
_LABEL_DIR_PARAM = Annotated[
    str,
    Field(
        description='标签目录路径。留空时默认按 source_path 的同级 labels 结构自动推断。',
        examples=['/data/labels'],
    ),
]
_OUTPUT_DIR_PARAM = Annotated[
    str,
    Field(
        description='抽取结果输出目录。留空时由服务端自动创建默认输出目录。',
        examples=['/tmp/extract_out', 'runs/extract'],
    ),
]
_SELECTION_MODE_PARAM = Annotated[
    str,
    Field(
        description='图片抽取选择模式。常用 count 或 ratio。',
        examples=['count', 'ratio'],
    ),
]
_COUNT_PARAM = Annotated[
    int,
    Field(
        description='按 count 模式抽取时的图片数量。',
        ge=1,
        examples=[100, 500],
    ),
]
_RATIO_PARAM = Annotated[
    float,
    Field(
        description='按 ratio 模式抽取时的比例，范围 0 到 1。',
        gt=0.0,
        le=1.0,
        examples=[0.1, 0.25],
    ),
]
_GROUPING_MODE_PARAM = Annotated[
    str,
    Field(
        description='图片抽取分组策略。常用 global 或 per_dir。',
        examples=['global', 'per_dir'],
    ),
]
_SELECTED_DIRS_PARAM = Annotated[
    list[str] | str | None,
    Field(
        description='显式限制参与抽取的子目录。可传目录字符串列表，也兼容旧的单个字符串输入。',
        examples=[['cam01', 'cam02'], 'cam01'],
    ),
]
_OUTPUT_LAYOUT_PARAM = Annotated[
    str,
    Field(
        description='抽取输出布局。常用 flat 或 preserve_dirs。',
        examples=['flat', 'preserve_dirs'],
    ),
]
_SEED_PARAM = Annotated[
    int,
    Field(
        description='随机抽样种子。',
        examples=[42, 1234],
    ),
]
_MAX_EXAMPLES_PARAM = Annotated[
    int,
    Field(
        description='预览或扫描时返回的样例数量上限。',
        ge=0,
        examples=[3, 10],
    ),
]
_FRAME_MODE_PARAM = Annotated[
    str,
    Field(
        description='视频抽帧模式。常用 interval、time 或 scene。',
        examples=['interval', 'time', 'scene'],
    ),
]
_FRAME_INTERVAL_PARAM = Annotated[
    int,
    Field(
        description='按 interval 模式抽帧时的帧间隔。',
        ge=1,
        examples=[30, 60],
    ),
]
_MAX_FRAMES_PARAM = Annotated[
    int,
    Field(
        description='最多抽取的帧数；0 表示不限制。',
        ge=0,
        examples=[0, 300],
    ),
]
_OUTPUT_FORMAT_PARAM = Annotated[
    str,
    Field(
        description='抽帧图片格式。常用 jpg 或 png。',
        examples=['jpg', 'png'],
    ),
]


def _action_candidates_from_next_actions(next_actions: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not isinstance(next_actions, list):
        return candidates
    for item in next_actions:
        if isinstance(item, dict):
            compact = {
                'action': item.get('action'),
                'tool': item.get('tool'),
                'description': item.get('description') or item.get('reason') or item.get('summary'),
            }
            compact = {key: value for key, value in compact.items() if value not in (None, '', [], {})}
            if compact:
                candidates.append(compact)
        else:
            text = str(item or '').strip()
            if text:
                candidates.append({'description': text})
    return candidates


def _extract_preview_overview(result: dict[str, Any]) -> dict[str, Any]:
    overview = {
        'available_images': result.get('available_images'),
        'planned_extract_count': result.get('planned_extract_count'),
        'selection_mode': result.get('selection_mode'),
        'grouping_mode': result.get('grouping_mode'),
        'output_layout': result.get('output_layout'),
        'workflow_ready': result.get('workflow_ready'),
        'workflow_ready_path': result.get('workflow_ready_path'),
        'conflict_count': result.get('conflict_count'),
    }
    return {key: value for key, value in overview.items() if value not in (None, '', [], {})}


def _extract_overview(result: dict[str, Any]) -> dict[str, Any]:
    overview = {
        'extracted': result.get('extracted'),
        'labels_copied': result.get('labels_copied'),
        'conflict_count': result.get('conflict_count'),
        'output_dir': result.get('output_dir'),
        'workflow_ready': result.get('workflow_ready'),
        'workflow_ready_path': result.get('workflow_ready_path'),
    }
    return {key: value for key, value in overview.items() if value not in (None, '', [], {})}


def _video_scan_overview(result: dict[str, Any]) -> dict[str, Any]:
    overview = {
        'total_videos': result.get('total_videos'),
        'source_path': result.get('source_path'),
        'sample_count': len(result.get('sample_videos') or []),
    }
    return {key: value for key, value in overview.items() if value not in (None, '', [], {})}


def _frame_extract_overview(result: dict[str, Any]) -> dict[str, Any]:
    overview = {
        'total_frames': result.get('total_frames'),
        'extracted': result.get('extracted'),
        'final_count': result.get('final_count'),
        'output_dir': result.get('output_dir'),
        'mode': result.get('mode'),
    }
    return {key: value for key, value in overview.items() if value not in (None, '', [], {})}


def _apply_structured_defaults(result: dict[str, Any], *, overview_key: str, overview_value: dict[str, Any]) -> dict[str, Any]:
    result.setdefault(overview_key, overview_value)
    result.setdefault('action_candidates', _action_candidates_from_next_actions(result.get('next_actions')))
    return result


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
    source_path: _SOURCE_PATH_PARAM,
    label_dir: _LABEL_DIR_PARAM = '',
    output_dir: _OUTPUT_DIR_PARAM = '',
    selection_mode: _SELECTION_MODE_PARAM = 'count',
    count: _COUNT_PARAM = 100,
    ratio: _RATIO_PARAM = 0.1,
    grouping_mode: _GROUPING_MODE_PARAM = 'global',
    selected_dirs: _SELECTED_DIRS_PARAM = None,
    copy_labels: bool = True,
    output_layout: _OUTPUT_LAYOUT_PARAM = 'flat',
    seed: _SEED_PARAM = 42,
    max_examples: _MAX_EXAMPLES_PARAM = 3,
) -> dict[str, Any]:
    """预览图片抽取计划。默认以 flat 布局输出，方便后续直接接 scan/validate/训练链；不会修改原始数据。"""
    result = _wrap(
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
    return _apply_structured_defaults(result, overview_key='extract_preview_overview', overview_value=_extract_preview_overview(result))


def extract_images(
    source_path: _SOURCE_PATH_PARAM,
    label_dir: _LABEL_DIR_PARAM = '',
    output_dir: _OUTPUT_DIR_PARAM = '',
    selection_mode: _SELECTION_MODE_PARAM = 'count',
    count: _COUNT_PARAM = 100,
    ratio: _RATIO_PARAM = 0.1,
    grouping_mode: _GROUPING_MODE_PARAM = 'global',
    selected_dirs: _SELECTED_DIRS_PARAM = None,
    copy_labels: bool = True,
    output_layout: _OUTPUT_LAYOUT_PARAM = 'flat',
    seed: _SEED_PARAM = 42,
    max_examples: _MAX_EXAMPLES_PARAM = 3,
) -> dict[str, Any]:
    """执行图片抽取。默认输出为 flat 数据集结构，便于后续直接继续 scan_dataset / validate_dataset / prepare_dataset_for_training。"""
    result = _wrap(
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
    return _apply_structured_defaults(result, overview_key='extract_overview', overview_value=_extract_overview(result))


def scan_videos(source_path: _SOURCE_PATH_PARAM, max_examples: _MAX_EXAMPLES_PARAM = 3) -> dict[str, Any]:
    """扫描视频目录或单个视频，返回可处理视频数量与目录分布。"""
    result = _wrap('扫描视频目录', service.scan_videos, source_path=source_path, max_examples=max_examples)
    return _apply_structured_defaults(result, overview_key='video_scan_overview', overview_value=_video_scan_overview(result))


def extract_video_frames(
    source_path: _SOURCE_PATH_PARAM,
    output_dir: _OUTPUT_DIR_PARAM = '',
    mode: _FRAME_MODE_PARAM = 'interval',
    frame_interval: _FRAME_INTERVAL_PARAM = 30,
    time_interval: float = 1.0,
    scene_threshold: float = 0.4,
    min_scene_gap: int = 15,
    enable_dedup: bool = True,
    dedup_threshold: int = 8,
    max_frames: _MAX_FRAMES_PARAM = 0,
    start_time: float = 0.0,
    end_time: float = 0.0,
    output_format: _OUTPUT_FORMAT_PARAM = 'jpg',
    jpg_quality: int = 95,
    name_prefix: str = '',
) -> dict[str, Any]:
    """执行视频抽帧。支持 interval/time/scene 三种最常见模式，输出目录可继续作为图片输入使用。"""
    result = _wrap(
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
    return _apply_structured_defaults(result, overview_key='frame_extract_overview', overview_value=_frame_extract_overview(result))

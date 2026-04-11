from __future__ import annotations

import re
from typing import Any


MODEL_SUFFIXES = ('.pt', '.onnx', '.yaml', '.yml')
VIDEO_SUFFIXES = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm')


def extract_all_paths_from_text(text: str) -> list[str]:
    patterns = [
        r"([A-Za-z]:\\[^\s，,。；;\"']+)",
        r"(/[^\s，,。；;\"']+)",
    ]
    items: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = match.group(1).rstrip('。，“”,,;；')
            if value and value not in seen:
                seen.add(value)
                items.append(value)
    return items


def looks_like_model_path(path: str) -> bool:
    return str(path).lower().endswith(MODEL_SUFFIXES)


def extract_dataset_path_from_text(text: str) -> str:
    paths = extract_all_paths_from_text(text)
    for item in paths:
        if not looks_like_model_path(item):
            return item
    return paths[0] if paths else ''


def extract_model_from_text(text: str) -> str:
    match = re.search(r'([A-Za-z0-9_./\-]+\.(?:pt|onnx|yaml))', text)
    if match:
        return match.group(1)
    match = re.search(r'(yolo[a-zA-Z0-9._-]+)', text, flags=re.I)
    if match:
        token = match.group(1)
        return token if '.' in token else f'{token}.pt'
    return ''


def looks_like_video_path(path: str) -> bool:
    return str(path).lower().endswith(VIDEO_SUFFIXES)


def should_use_video_prediction(user_text: str, path: str) -> bool:
    normalized = user_text.lower()
    if looks_like_video_path(path):
        return True
    return any(token in user_text for token in ('视频', '录像')) or 'video' in normalized


def extract_output_path_from_text(text: str, source_path: str = '') -> str:
    paths = extract_all_paths_from_text(text)
    source_key = str(source_path or '')
    candidates = [item for item in paths if not looks_like_model_path(item) and item != source_key]
    return candidates[0] if candidates else ''


def extract_count_from_text(text: str) -> int | None:
    match = re.search(r'(\d+)\s*(张|个|images?)', text, flags=re.I)
    if match:
        return int(match.group(1))
    return None


def extract_ratio_from_text(text: str) -> float | None:
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if match:
        return float(match.group(1)) / 100.0
    match = re.search(r'比例\s*[:=]?\s*(0?\.\d+|1(?:\.0+)?)', text)
    if match:
        return float(match.group(1))
    return None


def build_image_extract_args_from_text(user_text: str, source_path: str) -> dict[str, Any]:
    args: dict[str, Any] = {'source_path': source_path}
    output_path = extract_output_path_from_text(user_text, source_path)
    if output_path:
        args['output_dir'] = output_path
    ratio = extract_ratio_from_text(user_text)
    count = extract_count_from_text(user_text)
    if '全部' in user_text or 'all' in user_text.lower():
        args['selection_mode'] = 'all'
    elif ratio is not None:
        args['selection_mode'] = 'ratio'
        args['ratio'] = ratio
    else:
        args['selection_mode'] = 'count'
        args['count'] = count if count is not None else 100
    args['grouping_mode'] = 'per_directory' if any(token in user_text for token in ('每个目录', '按目录', '各目录')) else 'global'
    args['copy_labels'] = not any(token in user_text for token in ('不复制标签', '不要标签', 'labels false'))
    args['output_layout'] = 'keep' if '保持目录结构' in user_text else 'flat'
    return args


def build_video_extract_args_from_text(user_text: str, source_path: str) -> dict[str, Any]:
    args: dict[str, Any] = {'source_path': source_path}
    output_path = extract_output_path_from_text(user_text, source_path)
    if output_path:
        args['output_dir'] = output_path
    normalized = user_text.lower()
    if '场景' in user_text or 'scene' in normalized:
        args['mode'] = 'scene'
    else:
        time_match = re.search(r'每\s*(\d+(?:\.\d+)?)\s*秒', user_text)
        frame_match = re.search(r'每\s*(\d+)\s*帧', user_text)
        if time_match:
            args['mode'] = 'time'
            args['time_interval'] = float(time_match.group(1))
        else:
            args['mode'] = 'interval'
            if frame_match:
                args['frame_interval'] = int(frame_match.group(1))
    max_frames = re.search(r'最多\s*(\d+)\s*帧', user_text)
    if max_frames:
        args['max_frames'] = int(max_frames.group(1))
    return args


def extract_epochs_from_text(text: str) -> int | None:
    match = re.search(r'(\d+)\s*轮', text)
    if match:
        return int(match.group(1))
    match = re.search(r'epochs?\s*[=:]?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    return None

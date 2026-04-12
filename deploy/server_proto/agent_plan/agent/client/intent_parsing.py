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


def extract_batch_size_from_text(text: str) -> int | None:
    match = re.search(r'batch\s*[=:]?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'batch\s*(?:改成|设成|设置为|为)?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'批大小\s*(?:改成|设成|设置为|为)?\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None


def extract_image_size_from_text(text: str) -> int | None:
    match = re.search(r'imgsz\s*[=:]?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'imgsz\s*(?:改成|设成|设置为|为)?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'图像尺寸\s*(?:改成|设成|设置为|为)?\s*(\d+)', text)
    if match:
        return int(match.group(1))
    match = re.search(r'输入尺寸\s*(?:改成|设成|设置为|为)?\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None


def extract_device_from_text(text: str) -> str:
    lowered = text.lower()
    if 'device=auto' in lowered or '设备自动' in text or '自动选卡' in text or 'auto device' in lowered:
        return 'auto'
    match = re.search(r'device\s*[=:]?\s*([0-9,]+|cpu|auto)', text, flags=re.I)
    if match:
        return match.group(1).lower()
    match = re.search(r'设备\s*(?:改成|设成|设置为|为|用)?\s*([0-9,]+|cpu|auto)', text, flags=re.I)
    if match:
        return match.group(1).lower()
    match = re.search(r'gpu\s*([0-9]+(?:,[0-9]+)*)', lowered)
    if match:
        return match.group(1)
    return ''


def extract_training_environment_from_text(text: str, known_environments: list[dict[str, Any]] | None = None) -> str:
    known = known_environments or []
    lowered = text.lower()
    for item in known:
        for candidate in (
            str(item.get('display_name') or '').strip(),
            str(item.get('name') or '').strip(),
        ):
            token = candidate.lower()
            if token and token in lowered:
                return candidate

    patterns = [
        r'(?:用|使用|切到|切换到|改成|换成)\s*([A-Za-z][A-Za-z0-9._-]*)\s*环境',
        r'环境\s*(?:改成|设成|设置为|切到|切换到|为|用)?\s*([A-Za-z][A-Za-z0-9._-]*)',
        r'conda\s*环境\s*([A-Za-z][A-Za-z0-9._-]*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1)
    return ''


def extract_optimizer_from_text(text: str) -> str:
    match = re.search(r'optimizer\s*[=:]?\s*([A-Za-z][A-Za-z0-9_-]*)', text, flags=re.I)
    if match:
        return match.group(1)
    match = re.search(r'优化器\s*(?:改成|设成|设置为|为|用)?\s*([A-Za-z][A-Za-z0-9_-]*)', text, flags=re.I)
    if match:
        return match.group(1)
    return ''


def extract_freeze_from_text(text: str) -> int | None:
    match = re.search(r'freeze\s*[=:]?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'冻结\s*(\d+)\s*层', text)
    if match:
        return int(match.group(1))
    return None


def extract_lr0_from_text(text: str) -> float | None:
    match = re.search(r'lr0\s*[=:]?\s*([0-9]*\.?[0-9]+)', text, flags=re.I)
    if match:
        return float(match.group(1))
    match = re.search(r'lr0\s*(?:改成|改为|设成|设置为|为|用)?\s*([0-9]*\.?[0-9]+)', text, flags=re.I)
    if match:
        return float(match.group(1))
    match = re.search(r'初始学习率\s*(?:改成|设成|设置为|为|用)?\s*([0-9]*\.?[0-9]+)', text)
    if match:
        return float(match.group(1))
    match = re.search(r'学习率\s*(?:改成|设成|设置为|为|用)?\s*([0-9]*\.?[0-9]+)', text)
    if match:
        return float(match.group(1))
    return None


def extract_patience_from_text(text: str) -> int | None:
    match = re.search(r'patience\s*[=:]?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'早停\s*(?:改成|设成|设置为|为|用)?\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None


def extract_workers_from_text(text: str) -> int | None:
    match = re.search(r'workers?\s*[=:]?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'线程数\s*(?:改成|设成|设置为|为|用)?\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None


def extract_amp_flag_from_text(text: str) -> bool | None:
    lowered = text.lower()
    if any(token in text for token in ('关闭 amp', '禁用 amp', '不要 amp', '不要混合精度', '关闭混合精度')) or 'amp off' in lowered or 'amp=false' in lowered:
        return False
    if any(token in text for token in ('开启 amp', '启用 amp', '打开 amp', '开启混合精度', '启用混合精度')) or 'amp on' in lowered or 'amp=true' in lowered:
        return True
    return None


def extract_resume_flag_from_text(text: str) -> bool | None:
    lowered = text.lower()
    if any(token in text for token in ('继续训练', '恢复训练', '接着训')) or 'resume' in lowered:
        return True
    if any(token in text for token in ('不要恢复训练', '不要继续训练', '重新开始训练')):
        return False
    return None


def extract_custom_training_script_from_text(text: str) -> str:
    match = re.search(r'([A-Za-z0-9_./\\\\-]+\.py)', text)
    if match:
        return match.group(1)
    return ''


def extract_training_execution_backend_from_text(text: str) -> str:
    lowered = text.lower()
    script_path = extract_custom_training_script_from_text(text)
    if script_path or any(token in text for token in ('自定义训练脚本', 'python脚本训练', '脚本训练')):
        return 'custom_script'
    if 'trainer' in lowered or '自定义trainer' in lowered or '自定义训练器' in text:
        return 'custom_trainer'
    return 'standard_yolo'


def is_training_discussion_only(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in text or token in lowered
        for token in (
            '先别执行',
            '先不要执行',
            '先别启动',
            '先不要启动',
            '先看计划',
            '先看看计划',
            '先给我计划',
            '先讨论',
            '只讨论',
            '先别急着执行',
            '先做方案',
            '先 dry-run',
            '先 preflight',
        )
    )


def wants_training_advanced_details(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in text or token in lowered
        for token in (
            '高级参数',
            '高级配置',
            '展开参数',
            '详细参数',
            '更多参数',
            'advanced',
            'hyperparameter',
        )
    )


def extract_metric_signals_from_text(text: str) -> list[str]:
    normalized = text.lower()
    signals: list[str] = []
    if ((("precision" in normalized) or ('精确率' in text)) and ((("recall" in normalized) or ('召回' in text)))):
        if re.search(r'(precision|精确率).{0,8}(高|偏高).{0,12}(recall|召回).{0,8}(低|偏低)', text, flags=re.I):
            signals.append('high_precision_low_recall')
        if re.search(r'(precision|精确率).{0,8}(低|偏低).{0,12}(recall|召回).{0,8}(高|偏高)', text, flags=re.I):
            signals.append('low_precision_high_recall')
    if re.search(r'(map50|mAP50|mAP).{0,8}(低|偏低)', text, flags=re.I) or 'map低' in normalized:
        signals.append('low_map_overall')
    if '只有loss' in normalized or '只看loss' in normalized or '只有 loss' in text:
        signals.append('loss_only_metrics')
    return signals

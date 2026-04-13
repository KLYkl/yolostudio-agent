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
    return ''


def extract_model_from_text(text: str) -> str:
    match = re.search(r'([A-Za-z0-9_./\-]+\.(?:pt|onnx|yaml))', text)
    if match:
        return match.group(1)
    match = re.search(r'\b(yolo[a-zA-Z0-9._-]+)\b', text, flags=re.I)
    if match:
        token = match.group(1)
        if '.' in token or re.search(r'\d', token):
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
    match = re.search(r'batch\s*(?:改成|改为|改|设成|设置为|为)?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'批大小\s*(?:改成|改为|改|设成|设置为|为)?\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None


def extract_image_size_from_text(text: str) -> int | None:
    match = re.search(r'imgsz\s*[=:]?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'imgsz\s*(?:改成|改为|改|设成|设置为|为)?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'图像尺寸\s*(?:改成|改为|改|设成|设置为|为)?\s*(\d+)', text)
    if match:
        return int(match.group(1))
    match = re.search(r'输入尺寸\s*(?:改成|改为|改|设成|设置为|为)?\s*(\d+)', text)
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
    matches: list[tuple[int, str]] = []

    def _canonicalize(explicit: str) -> str:
        for item in known:
            for candidate in (
                str(item.get('display_name') or '').strip(),
                str(item.get('name') or '').strip(),
            ):
                if candidate and candidate.lower() == explicit.lower():
                    return candidate
        return explicit

    patterns = [
        r'(?:用|使用|切到|切换到|改成|换成)\s*([A-Za-z][A-Za-z0-9._-]*)\s*环境',
        r'环境\s*(?:先)?\s*(?:改成|设成|设置为|切到|切换到|为|用)?\s*([A-Za-z][A-Za-z0-9._-]*)',
        r'conda\s*环境\s*([A-Za-z][A-Za-z0-9._-]*)',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            matches.append((match.start(), _canonicalize(match.group(1))))

    lowered = text.lower()
    for item in known:
        for candidate in (
            str(item.get('display_name') or '').strip(),
            str(item.get('name') or '').strip(),
        ):
            token = candidate.lower()
            if token and token in lowered:
                matches.append((lowered.rfind(token), candidate))
    if matches:
        _, candidate = max(matches, key=lambda item: item[0])
        return candidate
    return ''


def extract_project_from_text(text: str) -> str:
    patterns = [
        r'project\s*[=:]?\s*([A-Za-z]:\\[^\s，,。；;\"\']+)',
        r'project\s*[=:]?\s*(/[^\s，,。；;\"\']+)',
        r'project\s*(?:改成|改为|设成|设置为|为|用)?\s*([A-Za-z]:\\[^\s，,。；;\"\']+)',
        r'project\s*(?:改成|改为|设成|设置为|为|用)?\s*(/[^\s，,。；;\"\']+)',
        r'输出目录\s*(?:改成|改为|设成|设置为|为|用)?\s*([A-Za-z]:\\[^\s，,。；;\"\']+)',
        r'输出目录\s*(?:改成|改为|设成|设置为|为|用)?\s*(/[^\s，,。；;\"\']+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).rstrip('。，“”,,;；')
    return ''


def extract_run_name_from_text(text: str) -> str:
    patterns = [
        r'name\s*[=:]?\s*([A-Za-z0-9._-]+)',
        r'name\s*(?:改成|改为|设成|设置为|为|用)?\s*([A-Za-z0-9._-]+)',
        r'(?:实验名|运行名|run名|任务名)\s*(?:改成|改为|设成|设置为|为|用)?\s*([A-Za-z0-9._-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1)
    return ''


def extract_fraction_from_text(text: str) -> float | None:
    match = re.search(r'fraction\s*[=:]?\s*(-?\d+(?:\.\d+)?)', text, flags=re.I)
    if match:
        return float(match.group(1))
    match = re.search(r'fraction\s*(?:改成|改为|改|设成|设置为|为|用)?\s*(-?\d+(?:\.\d+)?)', text, flags=re.I)
    if match:
        return float(match.group(1))
    match = re.search(r'只用\s*(\d+(?:\.\d+)?)\s*%\s*数据', text)
    if match:
        return float(match.group(1)) / 100.0
    match = re.search(r'使用\s*(\d+(?:\.\d+)?)\s*%\s*数据', text)
    if match:
        return float(match.group(1)) / 100.0
    return None


def wants_clear_batch(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in text or token in lowered
        for token in (
            '恢复默认 batch',
            'batch 恢复默认',
            'batch 不要了',
            '不要 batch',
            'batch 清掉',
            'batch 取消',
            'batch 先取消',
        )
    )


def extract_classes_from_text(text: str) -> list[int] | None:
    patterns = [
        r'classes\s*[=:]?\s*([0-9,\s]+)',
        r'classes\s*(?:改成|改为|设成|设置为|为|用)?\s*([0-9,\s]+)',
        r'只训练类别\s*([0-9,\s和]+)',
        r'只训\s*([0-9,\s和]+)\s*类',
        r'类别\s*(?:改成|改为|设成|设置为|为|用)\s*([0-9,\s和]+)',
        r'类别限制\s*([0-9,\s和]+)',
        r'类别限制\s*(?:改成|改为|设成|设置为|为|用)?\s*([0-9,\s和]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        raw = match.group(1).replace('和', ',')
        values = [part.strip() for part in raw.split(',') if part.strip()]
        if values and all(part.isdigit() for part in values):
            return [int(part) for part in values]
    return None


def extract_single_cls_flag_from_text(text: str) -> bool | None:
    lowered = text.lower()
    if any(token in text for token in ('关闭 single_cls', '禁用 single_cls', '不要单类别训练', '不要单类训练')) or 'single_cls=false' in lowered:
        return False
    if any(token in text for token in ('开启 single_cls', '启用 single_cls', '单类别训练', '单类训练')) or 'single_cls=true' in lowered or 'single-cls' in lowered:
        return True
    return None


def extract_optimizer_from_text(text: str) -> str:
    match = re.search(r'optimizer\s*[=:]?\s*([A-Za-z][A-Za-z0-9_-]*)', text, flags=re.I)
    if match:
        return match.group(1)
    match = re.search(r'optimizer\s*(?:改成|改为|设成|设置为|为|用)?\s*([A-Za-z][A-Za-z0-9_-]*)', text, flags=re.I)
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
    if any(token in text for token in ('关闭 amp', 'amp 关闭', '禁用 amp', '不要 amp', '不要混合精度', '关闭混合精度')) or 'amp off' in lowered or 'amp=false' in lowered:
        return False
    if any(token in text for token in ('开启 amp', '启用 amp', '打开 amp', '开启混合精度', '启用混合精度')) or 'amp on' in lowered or 'amp=true' in lowered:
        return True
    return None


def extract_resume_flag_from_text(text: str) -> bool | None:
    lowered = text.lower()
    if any(token in text for token in ('不要恢复训练', '不要继续训练', '重新开始训练', 'resume 不要', '不要 resume')) or 'resume=false' in lowered:
        return False
    if any(token in text for token in ('继续训练', '恢复训练', '接着训')) or 'resume' in lowered:
        return True
    return None


def extract_custom_training_script_from_text(text: str) -> str:
    match = re.search(r'([A-Za-z0-9_./\\\\-]+\.py)', text)
    if match:
        return match.group(1)
    return ''


def extract_training_execution_backend_from_text(text: str) -> str:
    lowered = text.lower()
    if any(token in text for token in ('不用自定义脚本', '不用脚本了', '切回标准 yolo', '改成标准 yolo', '用标准 yolo')) or any(token in lowered for token in ('don\'t use custom script', 'switch back to standard yolo')):
        return 'standard_yolo'
    if any(token in text for token in ('不用 trainer', '不用自定义trainer', '不用自定义训练器', '切回标准训练器')) or any(token in lowered for token in ('switch back to standard trainer',)):
        return 'standard_yolo'
    script_path = extract_custom_training_script_from_text(text)
    if script_path or any(token in text for token in ('自定义训练脚本', 'python脚本训练', '脚本训练')):
        return 'custom_script'
    trainer_explicit = any(token in text for token in ('自定义 trainer', '自定义trainer', '自定义训练器'))
    trainer_context = any(token in text for token in ('trainer 讨论', 'trainer方案', 'trainer 先讨论', 'trainer 先不管'))
    trainer_switch = re.search(r'(?:改成|切到|换成|用|讨论)\s*(?:自定义\s*)?trainer\b', lowered) is not None
    if trainer_explicit or trainer_context or trainer_switch:
        return 'custom_trainer'
    return 'standard_yolo'


def wants_default_training_environment(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in text or token in lowered
        for token in (
            '恢复默认环境',
            '用默认环境',
            '切回默认环境',
            '环境恢复默认',
            '不要指定环境',
        )
    )


def wants_clear_project(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in text or token in lowered
        for token in (
            'project 不要了',
            '不要 project',
            '清空 project',
            '恢复默认输出目录',
            '输出目录用默认',
            '不要输出目录',
        )
    )


def wants_clear_run_name(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in text or token in lowered
        for token in (
            'name 不要了',
            '不要 name',
            '清空 name',
            '运行名不要了',
            '实验名不要了',
            '不要输出名',
        )
    )


def wants_clear_fraction(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in text or token in lowered
        for token in (
            '恢复全量数据',
            '恢复全部数据',
            '全部数据都训练',
            '取消抽样',
            '不做抽样',
            '取消 fraction',
            'fraction 取消',
            'fraction 不要了',
            '不要 fraction',
            '不限制数据比例',
        )
    )


def wants_clear_classes(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in text or token in lowered
        for token in (
            '取消类别限制',
            '类别限制取消',
            '类别限制先取消',
            '把类别限制取消',
            '不要类别限制',
            '类别限制去掉',
            '不限制类别',
            '恢复全类别',
            '全部类别都训练',
            '不要 classes',
            '取消 classes',
        )
    )


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

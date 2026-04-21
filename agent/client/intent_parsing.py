from __future__ import annotations

import re
from typing import Any


MODEL_SUFFIXES = ('.pt', '.onnx')
MODEL_CONFIG_SUFFIXES = ('.yaml', '.yml')
VIDEO_SUFFIXES = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm')
PATH_ACTION_SUFFIX_MARKERS = (
    '按默认比例',
    '先不要开始训练',
    '不要开始训练',
    '先不要训练',
    '不要训练',
    '准备训练数据',
    '准备数据集',
    '准备数据',
    '划分训练集',
    '划分数据集',
    '生成data.yaml',
    '生成 data.yaml',
    '生成yaml',
    '生成 yaml',
    '开始训练',
    '启动训练',
    '开始预测',
    '启动预测',
    '导出报告',
    '总结结果',
    '分析结果',
    '然后',
    '并且',
    '并',
)


def _trim_trailing_path_noise(value: str) -> str:
    text = str(value or '').strip().rstrip('。，“”,,;；')
    if not text:
        return ''
    for marker in PATH_ACTION_SUFFIX_MARKERS:
        index = text.find(marker)
        if index <= 0:
            continue
        prefix = text[:index].rstrip('/\\ ')
        if prefix.startswith('/') or re.match(r'^[A-Za-z]:[\\/]', prefix) or prefix.startswith('~'):
            text = prefix
            break
    return text.rstrip('。，“”,,;；')


def extract_all_paths_from_text(text: str) -> list[str]:
    patterns = [
        r"([A-Za-z]:\\[^\s，,。；;\"']+)",
        r"(/[^\s，,。；;\"']+)",
    ]
    items: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = _trim_trailing_path_noise(match.group(1))
            if value and value not in seen:
                seen.add(value)
                items.append(value)
    return items


def extract_remote_root_from_text(text: str) -> str:
    patterns = (
        r'(?:remote_root|remote_dir|remote_path)\s*[=:：]?\s*(/[^\s，,。；;\"\'<>]+)',
        r'(?:远端目录|服务器目录|目标目录)\s*[=:：]?\s*(/[^\s，,。；;\"\'<>]+)',
        r'(?:上传到|传到|同步到|放到)\s*(/[^\s，,。；;\"\'<>]+)',
        r'(?:上传到|传到|同步到|放到)(?:[^\n]{0,80}?)(/[^\s，,。；;\"\'<>]+)',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return _trim_trailing_path_noise(match.group(1))
    paths = extract_all_paths_from_text(text)
    for item in reversed(paths):
        if item.startswith('/'):
            return item
    return ''


def extract_remote_server_from_text(text: str) -> str:
    patterns = (
        r'(?:服务器|远端|节点)\s*(?:[:=：]|是|为|用|走|到)?\s*([A-Za-z0-9_.@-]+)',
        r'(?:\bserver\b|\bremote\b)\s*(?:[:=：]|is|=|to)?\s*([A-Za-z0-9_.@-]+)',
        r'到\s*([A-Za-z0-9_.@-]+)\s*(?:服务器|节点)',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        value = match.group(1).strip().strip('。，“”,,;；')
        if value and not any(ch in value for ch in ('\\', '/', ':')):
            return value
    return ''


def looks_like_model_path(path: str) -> bool:
    text = str(path or '').strip().lower()
    if not text:
        return False
    if text.endswith(MODEL_SUFFIXES):
        return True
    if text.endswith(MODEL_CONFIG_SUFFIXES):
        basename = text.replace('\\', '/').rsplit('/', 1)[-1]
        return basename.startswith('yolo') or 'model' in basename or 'cfg' in basename
    return False


def extract_dataset_path_from_text(text: str) -> str:
    paths = extract_all_paths_from_text(text)
    for item in paths:
        if not looks_like_model_path(item):
            return item
    return ''


def extract_classes_txt_from_text(text: str) -> str:
    patterns = (
        r'(?:classes(?:_txt)?|类名(?:文件|列表|来源)?)\s*(?:使用|用|来自|是|为|路径是|路径为|[:=：])?\s*([A-Za-z]:\\[^\s，,。；;\"\'<>]+\.txt)',
        r'(?:classes(?:_txt)?|类名(?:文件|列表|来源)?)\s*(?:使用|用|来自|是|为|路径是|路径为|[:=：])?\s*(/[^\s，,。；;\"\'<>]+\.txt)',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return _trim_trailing_path_noise(match.group(1))
    for item in extract_all_paths_from_text(text):
        lowered = item.replace('\\', '/').rsplit('/', 1)[-1].lower()
        if lowered.endswith('.txt') and 'classes' in lowered:
            return item
    return ''


def extract_model_from_text(text: str) -> str:
    match = re.search(r'([A-Za-z0-9_./\-]+\.(?:pt|onnx))', text)
    if match:
        return match.group(1)
    yaml_match = re.search(r'([A-Za-z0-9_./\-]+\.(?:yaml|yml))', text, flags=re.I)
    if yaml_match:
        candidate = yaml_match.group(1)
        if looks_like_model_path(candidate):
            return candidate
    match = re.search(r'\b(yolo[a-zA-Z0-9._-]+)\b', text, flags=re.I)
    if match:
        token = match.group(1)
        if '.' in token or re.search(r'\d', token):
            return token if '.' in token else f'{token}.pt'
    return ''


def looks_like_video_path(path: str) -> bool:
    return str(path).lower().endswith(VIDEO_SUFFIXES)


def extract_rtsp_url_from_text(text: str) -> str:
    match = re.search(r'(rtsp://[^\s，,。；;\"\'<>]+)', text, flags=re.I)
    if match:
        return match.group(1)
    return ''


def extract_realtime_session_id_from_text(text: str) -> str:
    match = re.search(r'\b(realtime-[a-z]+-[0-9a-f]{6,})\b', text, flags=re.I)
    if match:
        return match.group(1)
    return ''


def extract_camera_id_from_text(text: str) -> int | None:
    patterns = (
        r'camera(?:_id)?\s*[=:]?\s*(\d+)',
        r'摄像头\s*(?:id)?\s*[=:：]?\s*(\d+)',
        r'(\d+)\s*号摄像头',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return int(match.group(1))
    return None


def extract_screen_id_from_text(text: str) -> int | None:
    patterns = (
        r'screen(?:_id)?\s*[=:]?\s*(\d+)',
        r'屏幕\s*(?:id)?\s*[=:：]?\s*(\d+)',
        r'(\d+)\s*号屏幕',
        r'第\s*(\d+)\s*块屏幕',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return int(match.group(1))
    return None


def extract_frame_interval_ms_from_text(text: str) -> int | None:
    match = re.search(r'frame(?:_interval)?(?:_ms)?\s*[=:]?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'(?:每|间隔)\s*(\d+(?:\.\d+)?)\s*(毫秒|ms|秒|s)\b', text, flags=re.I)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    return int(value * 1000) if unit in {'秒', 's'} else int(value)


def extract_max_frames_from_text(text: str) -> int | None:
    match = re.search(r'max_frames?\s*[=:]?\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'最多\s*(\d+)\s*帧', text)
    if match:
        return int(match.group(1))
    return None


def extract_timeout_ms_from_text(text: str) -> int | None:
    match = re.search(r'timeout(?:_ms)?\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(毫秒|ms|秒|s)?', text, flags=re.I)
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or 'ms').lower()
        return int(value * 1000) if unit in {'秒', 's'} else int(value)
    match = re.search(r'超时\s*(\d+(?:\.\d+)?)\s*(毫秒|ms|秒|s)', text, flags=re.I)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        return int(value * 1000) if unit in {'秒', 's'} else int(value)
    return None


def extract_output_path_from_text(text: str, source_path: str = '') -> str:
    paths = extract_all_paths_from_text(text)
    source_key = str(source_path or '')
    candidates = [item for item in paths if not looks_like_model_path(item) and item != source_key]
    if '://' in source_key:
        candidates = [item for item in candidates if not source_key.endswith(item)]
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
    if not match:
        match = re.search(r'device\s*(?:改成|设成|设置为|为|用)\s*([0-9,]+|cpu|auto)', text, flags=re.I)
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


from __future__ import annotations

import json
import re
from typing import Any

from yolostudio_agent.agent.client.session_state import SessionState

DATASET_FACT_SNAPSHOT_PREFIX = 'DATASET_FACT_SNAPSHOT='


def looks_like_dataset_fact_question(user_text: str) -> bool:
    text = str(user_text or '').strip()
    lowered = text.lower()
    if not text:
        return False
    if any(token in text or token in lowered for token in ('修复缺失标签', '补齐缺失标签', '生成缺失标签', 'preview_generate_missing_labels', 'generate_missing_labels')):
        return False
    patterns = (
        '哪个类别最少',
        '哪类最少',
        '最少的是哪',
        '标注最少',
        '最少类别',
        '哪个类别最多',
        '哪类最多',
        '最多的是哪',
        '标注最多',
        '最多类别',
        '有哪些类别',
        '类别有哪些',
        '都有什么类别',
        '总共几张',
        '多少张图',
        '总图片数',
        '缺失标签有多少',
        '缺少标签有多少',
        '多少张没标签',
        '缺失标签比例',
        '缺标签比例',
    )
    if any(token in text or token in lowered for token in patterns):
        return True
    if any(token in text or token in lowered for token in ('多少', '几张', '多少张', '多少个', '几个标注')):
        return True
    return False


def _normalize_path_key(value: str) -> str:
    return str(value or '').strip().replace('\\', '/').rstrip('/').lower()


def _same_dataset_target_from_context(dataset_context: dict[str, Any], requested_dataset_path: str) -> bool:
    requested = _normalize_path_key(requested_dataset_path)
    if not requested:
        return True
    candidates = {
        _normalize_path_key(dataset_context.get('dataset_root')),
        _normalize_path_key(dataset_context.get('img_dir')),
        _normalize_path_key(dataset_context.get('label_dir')),
    }
    candidates.discard('')
    return requested in candidates


def _same_dataset_target(session_state: SessionState, requested_dataset_path: str) -> bool:
    return _same_dataset_target_from_context(
        {
            'dataset_root': session_state.active_dataset.dataset_root,
            'img_dir': session_state.active_dataset.img_dir,
            'label_dir': session_state.active_dataset.label_dir,
        },
        requested_dataset_path,
    )


def _format_percent(value: Any) -> str:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return ''
    return f'{ratio:.1%}'


def _derive_extreme_from_stats(class_stats: dict[str, int], *, reverse: bool) -> dict[str, Any]:
    if not class_stats:
        return {}
    ordered = sorted(class_stats.items(), key=lambda item: (-item[1], item[0]) if reverse else (item[1], item[0]))
    name, count = ordered[0]
    return {'name': name, 'count': count}


def _extract_class_count_query(user_text: str, class_names: list[str]) -> str:
    lowered = str(user_text or '').lower()
    matches = [name for name in class_names if name and name.lower() in lowered]
    if len(matches) != 1:
        return ''
    if not any(token in user_text or token in lowered for token in ('多少', '几张', '多少张', '多少个', '几个')):
        return ''
    return matches[0]


def build_dataset_fact_snapshot_payload(session_state: SessionState) -> dict[str, Any] | None:
    ds = session_state.active_dataset
    scan = dict(ds.last_scan or {})
    if not scan:
        return None
    return {
        'dataset_root': str(ds.dataset_root or ''),
        'img_dir': str(ds.img_dir or ''),
        'label_dir': str(ds.label_dir or ''),
        'scan': {
            'summary': str(scan.get('summary') or ''),
            'total_images': scan.get('total_images'),
            'missing_labels': scan.get('missing_labels'),
            'missing_label_images': scan.get('missing_label_images'),
            'missing_label_ratio': scan.get('missing_label_ratio'),
            'classes': list(scan.get('classes') or []),
            'class_stats': dict(scan.get('class_stats') or {}),
            'top_classes': list(scan.get('top_classes') or []),
            'least_class': dict(scan.get('least_class') or {}),
            'most_class': dict(scan.get('most_class') or {}),
            'class_name_source': str(scan.get('class_name_source') or ''),
            'detected_classes_txt': str(scan.get('detected_classes_txt') or ''),
        },
    }


def build_dataset_fact_snapshot_message(session_state: SessionState) -> str | None:
    payload = build_dataset_fact_snapshot_payload(session_state)
    if not payload:
        return None
    return f'{DATASET_FACT_SNAPSHOT_PREFIX}{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}'


def extract_dataset_fact_snapshot(messages: list[Any]) -> dict[str, Any] | None:
    for message in reversed(messages):
        content = getattr(message, 'content', '')
        if not isinstance(content, str) or not content.startswith(DATASET_FACT_SNAPSHOT_PREFIX):
            continue
        raw = content[len(DATASET_FACT_SNAPSHOT_PREFIX):].strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        if isinstance(payload, dict):
            return payload
        return None
    return None


def build_dataset_fact_followup_reply_from_snapshot(
    snapshot: dict[str, Any],
    *,
    user_text: str,
    requested_dataset_path: str = '',
) -> str | None:
    if not looks_like_dataset_fact_question(user_text):
        return None
    dataset_context = {
        'dataset_root': str(snapshot.get('dataset_root') or ''),
        'img_dir': str(snapshot.get('img_dir') or ''),
        'label_dir': str(snapshot.get('label_dir') or ''),
    }
    if not _same_dataset_target_from_context(dataset_context, requested_dataset_path):
        return None

    scan = dict(snapshot.get('scan') or {})
    if not scan:
        return None

    class_stats = {
        str(name): int(count)
        for name, count in (scan.get('class_stats') or {}).items()
        if str(name).strip()
    }
    classes = [str(item) for item in (scan.get('classes') or []) if str(item).strip()]
    least_class = dict(scan.get('least_class') or {}) or _derive_extreme_from_stats(class_stats, reverse=False)
    most_class = dict(scan.get('most_class') or {}) or _derive_extreme_from_stats(class_stats, reverse=True)
    text = str(user_text or '').strip()
    lowered = text.lower()

    if any(token in text or token in lowered for token in ('哪个类别最少', '哪类最少', '最少的是哪', '标注最少', '最少类别')):
        if least_class:
            return f"标注最少的类别是 {least_class['name']}，共有 {least_class['count']} 条标注。"
        return None

    if any(token in text or token in lowered for token in ('哪个类别最多', '哪类最多', '最多的是哪', '标注最多', '最多类别')):
        if most_class:
            return f"标注最多的类别是 {most_class['name']}，共有 {most_class['count']} 条标注。"
        return None

    if any(token in text or token in lowered for token in ('有哪些类别', '类别有哪些', '都有什么类别')):
        if classes:
            return f"识别到 {len(classes)} 个类别：{', '.join(classes)}。"
        return None

    if any(token in text or token in lowered for token in ('缺失标签有多少', '缺少标签有多少', '多少张没标签', '缺失标签比例', '缺标签比例')):
        missing_count = scan.get('missing_label_images', scan.get('missing_labels'))
        ratio_text = _format_percent(scan.get('missing_label_ratio'))
        if missing_count is None:
            return None
        if ratio_text:
            return f'缺失标签图片 {missing_count} 张，占比 {ratio_text}。'
        return f'缺失标签图片 {missing_count} 张。'

    if any(token in text or token in lowered for token in ('总共几张', '多少张图', '总图片数')):
        total_images = scan.get('total_images')
        if total_images is None:
            return None
        return f'总图片数是 {total_images} 张。'

    class_query = _extract_class_count_query(text, list(class_stats))
    if class_query:
        return f'类别 {class_query} 共有 {class_stats[class_query]} 条标注。'
    return None


def build_dataset_fact_followup_reply_from_messages(
    messages: list[Any],
    *,
    user_text: str,
    requested_dataset_path: str = '',
) -> str | None:
    snapshot = extract_dataset_fact_snapshot(messages)
    if not snapshot:
        return None
    return build_dataset_fact_followup_reply_from_snapshot(
        snapshot,
        user_text=user_text,
        requested_dataset_path=requested_dataset_path,
    )


def build_dataset_fact_followup_reply(
    session_state: SessionState,
    *,
    user_text: str,
    requested_dataset_path: str = '',
) -> str | None:
    if not looks_like_dataset_fact_question(user_text):
        return None
    if not _same_dataset_target(session_state, requested_dataset_path):
        return None
    payload = build_dataset_fact_snapshot_payload(session_state)
    if not payload:
        return None
    return build_dataset_fact_followup_reply_from_snapshot(
        payload,
        user_text=user_text,
        requested_dataset_path=requested_dataset_path,
    )

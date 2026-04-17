from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.session_state import SessionState

DATASET_FACT_CONTEXT_KEY = 'dataset_fact_context'


def build_dataset_fact_context_payload(session_state: SessionState) -> dict[str, Any] | None:
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


def extract_dataset_fact_context_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    payload = state.get(DATASET_FACT_CONTEXT_KEY)
    if isinstance(payload, dict):
        return dict(payload)
    return None

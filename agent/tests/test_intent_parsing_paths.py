from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.intent_parsing import extract_dataset_path_from_text, extract_model_from_text


def main() -> None:
    text = r'用 D:\models\yolov8n.pt 做一次训练前检查'
    assert extract_model_from_text(text) == 'yolov8n.pt'
    assert extract_dataset_path_from_text(text) == ''

    mixed = r'用 D:\data\demo_dataset 和 D:\models\yolov8n.pt 做训练前检查'
    assert extract_dataset_path_from_text(mixed) == r'D:\data\demo_dataset'
    assert extract_model_from_text(mixed) == 'yolov8n.pt'

    print('intent parsing paths ok')


if __name__ == '__main__':
    main()

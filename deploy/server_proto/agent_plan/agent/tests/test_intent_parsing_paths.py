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

from yolostudio_agent.agent.client.intent_parsing import (
    extract_classes_txt_from_text,
    extract_dataset_path_from_text,
    extract_model_from_text,
)


def main() -> None:
    text = r'用 D:\models\yolov8n.pt 做一次训练前检查'
    assert extract_model_from_text(text) == 'yolov8n.pt'
    assert extract_dataset_path_from_text(text) == ''

    mixed = r'用 D:\data\demo_dataset 和 D:\models\yolov8n.pt 做训练前检查'
    assert extract_dataset_path_from_text(mixed) == r'D:\data\demo_dataset'
    assert extract_model_from_text(mixed) == 'yolov8n.pt'

    linux_prepare = r'用 /home/kly/ct_loop/data_ct 按默认比例准备训练数据'
    assert extract_dataset_path_from_text(linux_prepare) == '/home/kly/ct_loop/data_ct'

    linux_noisy = r'用 /home/kly/ct_loop/data按默认比例准备训练数据'
    assert extract_dataset_path_from_text(linux_noisy) == '/home/kly/ct_loop/data'

    data_yaml_prompt = r'请把 /home/kly/ct_loop/data_ct 按默认比例准备好，然后用 /home/kly/ct_loop/data_ct/data.yaml 开始训练'
    assert extract_model_from_text(data_yaml_prompt) == ''
    assert extract_dataset_path_from_text(data_yaml_prompt) == '/home/kly/ct_loop/data_ct'

    model_yaml_prompt = r'请用 /models/yolov8n.yaml 作为模型结构来训练'
    assert extract_model_from_text(model_yaml_prompt) == '/models/yolov8n.yaml'

    classes_txt_prompt = r'请把 /home/kly/ct_loop/data_ct 按默认比例划分，类名使用 /home/kly/ct_loop/data_ct/classes.txt，然后训练'
    assert extract_classes_txt_from_text(classes_txt_prompt) == '/home/kly/ct_loop/data_ct/classes.txt'

    print('intent parsing paths ok')


if __name__ == '__main__':
    main()

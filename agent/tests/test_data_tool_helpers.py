from __future__ import annotations

import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.server.tools.data_tools import _discover_data_yaml, _infer_dataset_root


def main() -> None:
    root = Path('D:/yolodo2.0/agent_plan/.tmp_data_tool_test')
    if root.exists():
        shutil.rmtree(root)
    (root / 'dataset_1ch' / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'dataset_1ch' / 'labels').mkdir(parents=True, exist_ok=True)
    yaml_path = root / 'dataset_1ch.yaml'
    yaml_path.write_text("path: dataset_1ch\ntrain: images/train\nval: images/val\n", encoding='utf-8')

    try:
        img_dir = root / 'dataset_1ch' / 'images'
        label_dir = root / 'dataset_1ch' / 'labels'
        dataset_root = _infer_dataset_root(img_dir, label_dir)
        detected, candidates = _discover_data_yaml(img_dir, label_dir)
        assert dataset_root == root / 'dataset_1ch'
        assert detected == str(yaml_path.resolve())
        assert str(yaml_path.resolve()) in candidates
        print('data tool helper smoke ok')
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    main()

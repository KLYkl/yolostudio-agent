from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.tool_adapter import canonical_tool_name, normalize_tool_args


def main() -> None:
    assert canonical_tool_name('detect_duplicates') == 'detect_duplicate_images'
    assert canonical_tool_name('detect_corrupted_images') == 'run_dataset_health_check'
    assert canonical_tool_name('dataset_manager.prepare_dataset') == 'prepare_dataset_for_training'

    health_args = normalize_tool_args('detect_corrupted_images', {'path': '/data/set'})
    assert health_args['dataset_path'] == '/data/set'

    dup_args = normalize_tool_args('detect_duplicates', {'img_dir': '/data/set'})
    assert dup_args['dataset_path'] == '/data/set'

    prep_args = normalize_tool_args('dataset_manager.prepare_dataset', {'path': '/data/set', 'force_split': True})
    assert prep_args['dataset_path'] == '/data/set'
    assert prep_args['force_split'] is True

    readiness_args = normalize_tool_args('training_readiness', {'path': '/data/set'})
    assert readiness_args['img_dir'] == '/data/set'

    scan_args = normalize_tool_args('scan_dataset', {'dataset_path': '/data/set'})
    assert scan_args['img_dir'] == '/data/set'

    print('tool alias adapter ok')


if __name__ == '__main__':
    main()
from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.gpu_utils import GpuInfo
import yolostudio_agent.agent.server.services.gpu_utils as gpu_utils
import yolostudio_agent.agent.server.tools.data_tools as data_tools


class _FakeScanResult:
    total_images = 5
    labeled_images = 2
    missing_labels = [Path('c.jpg'), Path('d.jpg'), Path('e.jpg')]
    empty_labels = 0
    classes = ['Excavator', 'bulldozer']
    class_stats = {'Excavator': 3, 'bulldozer': 2}
    label_format = type('Fmt', (), {'name': 'TXT'})()


class _FakeValidateResult:
    total_labels = 2
    has_issues = False
    issue_count = 0
    coord_errors = []
    class_errors = []
    format_errors = []
    orphan_labels = []


class _FakeDataHandler:
    def scan_dataset(self, **kwargs):
        return _FakeScanResult()

    def validate_labels(self, **kwargs):
        return _FakeValidateResult()



def main() -> None:
    root = Path('C:/workspace/yolodo2.0/agent_plan/.tmp_data_quality_risk')
    if root.exists():
        shutil.rmtree(root)
    img_dir = root / 'dataset' / 'images'
    label_dir = root / 'dataset' / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    try:
        for name in ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg', 'e.jpg']:
            (img_dir / name).write_bytes(b'')
        (label_dir / 'classes.txt').write_text('Excavator\nbulldozer\n', encoding='utf-8')
        yaml_path = root / 'dataset' / 'data.yaml'
        yaml_path.write_text(
            'path: .\ntrain: images\nval: images\nnames:\n  0: Excavator\n  1: bulldozer\n',
            encoding='utf-8',
        )

        fake_handler_mod = types.ModuleType('core.data_handler._handler')
        fake_handler_mod.DataHandler = _FakeDataHandler
        original_handler_mod = sys.modules.get('core.data_handler._handler')
        original_query = gpu_utils.query_gpu_status
        original_policy = data_tools.get_effective_gpu_policy
        original_resolve = data_tools.resolve_auto_device
        sys.modules['core.data_handler._handler'] = fake_handler_mod
        gpu_utils.query_gpu_status = lambda: [GpuInfo(index='0', uuid='gpu-0', free_mb=12000, busy=False)]
        data_tools.get_effective_gpu_policy = lambda: 'single_idle_gpu'
        data_tools.resolve_auto_device = lambda policy=None, gpus=None: ('0', None)
        try:
            scan = data_tools.scan_dataset(str(root / 'dataset'))
            assert scan['ok'] is True
            assert scan['detected_classes_txt'].endswith('classes.txt')
            assert scan['class_name_source'] == 'classes_txt'
            assert scan['classes'] == ['Excavator', 'bulldozer']
            assert scan['missing_label_images'] == 3
            assert abs(scan['missing_label_ratio'] - 0.6) < 1e-6
            assert scan['risk_level'] == 'critical'

            validate = data_tools.validate_dataset(str(root / 'dataset'))
            assert validate['ok'] is True
            assert validate['has_issues'] is False
            assert validate['has_risks'] is True
            assert validate['missing_label_images'] == 3
            assert validate['risk_level'] == 'critical'
            assert '缺少标签' in validate['summary']

            readiness = data_tools.training_readiness(str(root / 'dataset'))
            assert readiness['ok'] is True
            assert readiness['ready'] is True
            assert readiness['resolved_data_yaml'] == str(yaml_path.resolve())
            assert readiness['detected_classes_txt'].endswith('classes.txt')
            assert readiness['risk_level'] == 'critical'
            assert readiness['missing_label_images'] == 3
            assert readiness['auto_device'] == '0'
            assert '数据质量风险' in readiness['summary']
        finally:
            if original_handler_mod is None:
                sys.modules.pop('core.data_handler._handler', None)
            else:
                sys.modules['core.data_handler._handler'] = original_handler_mod
            gpu_utils.query_gpu_status = original_query
            data_tools.get_effective_gpu_policy = original_policy
            data_tools.resolve_auto_device = original_resolve

        print('data quality risk contract ok')
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    main()

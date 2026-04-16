from __future__ import annotations

import shutil
import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient


class _DummyGraph:
    def get_state(self, config):
        return None


WORK = Path(__file__).resolve().parent / '_tmp_tool_result_facts'


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='tool-result-facts-smoke', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})

        readiness = {
            'ok': False,
            'summary': '当前还不能直接开训',
            'readiness_overview': {
                'scope': 'execution',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'blocker_codes': ['missing_yaml'],
            },
            'device_overview': {
                'device_policy': 'single_idle_gpu',
                'auto_device': '1',
            },
            'action_candidates': [
                {
                    'action': 'prepare_dataset',
                    'tool': 'prepare_dataset_for_training',
                    'description': '先准备数据',
                }
            ],
        }
        facts = client._tool_result_user_facts('training_readiness', readiness)
        assert facts['readiness_overview']['scope'] == 'execution'
        assert facts['device_overview']['auto_device'] == '1'
        assert facts['action_candidates'][0]['tool'] == 'prepare_dataset_for_training'

        prepare = {
            'ok': True,
            'ready': True,
            'summary': '数据准备完成',
            'data_yaml': '/dataset/data.yaml',
            'prepare_overview': {
                'ready': True,
                'data_yaml': '/dataset/data.yaml',
                'force_split_applied': True,
            },
            'action_candidates': [
                {
                    'action': 'start_training',
                    'tool': 'start_training',
                    'description': '现在可以开始训练',
                }
            ],
        }
        compact = client._compact_training_loop_start_fact('prepare_dataset_for_training', prepare)
        assert compact['prepare_overview']['force_split_applied'] is True
        assert compact['action_candidates'][0]['tool'] == 'start_training'
        print('tool result facts smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()

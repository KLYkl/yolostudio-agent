from __future__ import annotations

import sys
import types
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

try:
    import langchain_core.messages  # type: ignore  # noqa: F401
except Exception:
    core_mod = types.ModuleType('langchain_core')
    messages_mod = types.ModuleType('langchain_core.messages')

    class _BaseMessage:
        def __init__(self, content=''):
            self.content = content

    class _SystemMessage(_BaseMessage):
        pass

    messages_mod.BaseMessage = _BaseMessage
    messages_mod.SystemMessage = _SystemMessage
    core_mod.messages = messages_mod
    sys.modules['langchain_core'] = core_mod
    sys.modules['langchain_core.messages'] = messages_mod

from yolostudio_agent.agent.client.context_builder import ContextBuilder
from yolostudio_agent.agent.client.event_retriever import MemoryDigest
from yolostudio_agent.agent.client.session_state import SessionState


def _scenario_sparse_summary() -> None:
    builder = ContextBuilder('system')
    state = SessionState(session_id='ctx-sparse')
    summary = builder.build_state_summary(state)
    assert '当前结构化上下文:' in summary
    assert '- session_id: ctx-sparse' in summary
    assert '数据集:' not in summary
    assert '训练:' in summary
    assert 'workflow_state: idle' in summary
    assert 'loop_workflow_state: loop_idle' in summary
    assert '预测:' not in summary
    assert '知识:' not in summary
    assert '远端传输:' not in summary
    assert '待确认操作:' not in summary
    assert '偏好:' not in summary
    assert '历史摘要:' not in summary


def _scenario_compact_populated_summary() -> None:
    builder = ContextBuilder('system')
    state = SessionState(session_id='ctx-populated')
    state.active_dataset.dataset_root = '/data/demo'
    state.active_dataset.data_yaml = '/data/demo/data.yaml'
    state.active_dataset.last_scan = {
        'summary': '扫描完成: 共 90 张图片，类别 2 个',
        'classes': ['epidural', 'subdural'],
        'top_classes': [{'class_name': 'epidural', 'count': 66}, {'class_name': 'subdural', 'count': 24}],
        'least_class': {'name': 'subdural', 'count': 24},
        'most_class': {'name': 'epidural', 'count': 66},
        'missing_label_ratio': 0.0667,
        'class_name_source': 'detected_classes_txt',
    }
    state.active_dataset.last_readiness = {'ready': True}
    state.active_training.running = True
    state.active_training.model = 'yolov8n.pt'
    state.active_training.active_loop_id = 'loop-1'
    state.active_training.last_status = {'run_state': 'running'}
    state.active_prediction.realtime_session_id = 'rt-1'
    state.active_prediction.last_realtime_status = {'status': 'running'}
    state.active_knowledge.last_analysis = {'summary': 'ok'}
    state.active_remote_transfer.target_label = 'server-a'
    state.pending_confirmation.tool_name = 'start_training'
    state.pending_confirmation.allowed_decisions = ['approve', 'reject']
    state.preferences.default_model = 'demo-model'
    digest = MemoryDigest(summary_lines=['最近调用过的工具: check_training_status'], recent_events=[])
    summary = builder.build_state_summary(state, digest)
    assert '数据集:' in summary
    assert 'dataset_root: /data/demo' in summary
    assert 'last_scan_classes: epidural, subdural' in summary
    assert 'last_scan_top_classes: epidural=66, subdural=24' in summary
    assert 'last_scan_least_class: subdural (24)' in summary
    assert 'last_scan_most_class: epidural (66)' in summary
    assert 'last_scan_missing_label_ratio: 6.7%' in summary
    assert 'readiness_cache: 已缓存' in summary
    assert '训练:' in summary
    assert 'running: True' in summary
    assert 'active_loop_id: loop-1' in summary
    assert '预测:' in summary
    assert 'realtime_session_id: rt-1' in summary
    assert '知识:' in summary
    assert 'analysis_cache: 已缓存' in summary
    assert '远端传输:' in summary
    assert 'target_label: server-a' in summary
    assert '待确认操作:' in summary
    assert 'tool: start_training' in summary
    assert '偏好:' in summary
    assert 'default_model: demo-model' in summary
    assert '历史摘要:' in summary
    assert '- 最近调用过的工具: check_training_status' in summary


def main() -> None:
    _scenario_sparse_summary()
    _scenario_compact_populated_summary()
    print('context builder ok')


if __name__ == '__main__':
    main()

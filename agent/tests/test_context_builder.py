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
    state.active_training.recent_runs = [{'run_id': 'run-a', 'run_state': 'completed'}]
    state.active_training.best_run_selection = {
        'summary': '最近最佳训练为 best-run。',
        'best_run': {
            'run_id': 'best-run',
            'best_weight_path': '/runs/train/best-run/weights/best.pt',
        },
    }
    training_plan_context = {
        'status': 'ready_for_confirmation',
        'execution_mode': 'direct_train',
        'execution_backend': 'standard_yolo',
        'dataset_path': '/data/demo',
        'training_environment': 'yolodo',
        'reasoning_summary': '当前数据已具备训练条件，确认后即可启动。',
        'preflight_summary': '训练预检通过',
        'next_step_tool': 'start_training',
        'next_step_args': {'model': 'yolov8n.pt', 'force_split': False},
        'planned_training_args': {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/demo/data.yaml',
            'epochs': 100,
            'device': '0',
            'training_environment': 'yolodo',
            'project': '/runs/train',
            'name': 'demo-run',
            'batch': 8,
            'imgsz': 960,
            'optimizer': 'AdamW',
        },
        'command_preview': ['yolo', 'train', 'model=yolov8n.pt', 'data=/data/demo/data.yaml'],
        'warnings': ['样本量偏小，建议先小步验证'],
        'risks': ['样本量偏小'],
    }
    state.active_prediction.realtime_session_id = 'rt-1'
    state.active_prediction.last_realtime_status = {'status': 'running'}
    state.active_knowledge.last_analysis = {'summary': 'ok'}
    state.active_remote_transfer.target_label = 'server-a'
    state.preferences.default_model = 'demo-model'
    digest = MemoryDigest(summary_lines=['最近调用过的工具: check_training_status'], recent_events=[])
    pending_confirmation = {
        'tool_name': 'start_training',
        'allowed_decisions': ['approve', 'reject'],
    }
    summary = builder.build_state_summary(
        state,
        digest,
        pending_confirmation=pending_confirmation,
        training_plan_context=training_plan_context,
    )
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
    assert 'best_run_id: best-run' in summary
    assert 'best_run_weight_path: /runs/train/best-run/weights/best.pt' in summary
    assert 'training_plan_draft: 待确认' in summary
    assert 'training_plan_next_step_tool: start_training' in summary
    assert 'training_plan_model: yolov8n.pt' in summary
    assert 'training_plan_data_yaml: /data/demo/data.yaml' in summary
    assert 'training_plan_batch: 8' in summary
    assert 'training_plan_imgsz: 960' in summary
    assert 'training_plan_reasoning: 当前数据已具备训练条件，确认后即可启动。' in summary
    assert 'training_plan_command_preview: yolo, train, model=yolov8n.pt, data=/data/demo/data.yaml' in summary
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

    messages = builder.build_messages(
        state,
        [],
        pending_confirmation=pending_confirmation,
        training_plan_context=training_plan_context,
    )
    assert len(messages) == 2, messages
    assert getattr(messages[0], 'content', '') == 'system'
    assert 'epidural' in getattr(messages[1], 'content', '')
    assert 'subdural' in getattr(messages[1], 'content', '')
    assert 'training_plan_next_step_tool: start_training' in getattr(messages[1], 'content', '')
    assert 'list_training_runs' not in getattr(messages[1], 'content', '')


def _scenario_graph_training_context_overrides_stale_mirror() -> None:
    builder = ContextBuilder('system')
    state = SessionState(session_id='ctx-graph-priority')
    training_plan_context = {
        'status': 'ready_for_confirmation',
        'execution_mode': 'prepare_then_loop',
        'dataset_path': '/data/fresh',
        'next_step_tool': 'start_training_loop',
        'planned_training_args': {
            'model': 'fresh.pt',
            'data_yaml': '/data/fresh/data.yaml',
            'epochs': 10,
            'project': '/runs/fresh',
        },
        'next_step_args': {
            'model': 'fresh.pt',
            'data_yaml': '/data/fresh/data.yaml',
            'epochs': 10,
            'max_rounds': 5,
            'loop_name': 'ctx-loop',
        },
        'reasoning_summary': '应以 graph training context 为准。',
    }
    summary = builder.build_state_summary(state, training_plan_context=training_plan_context)
    assert 'training_plan_dataset_path: /data/fresh' in summary
    assert 'training_plan_model: fresh.pt' in summary
    assert 'training_plan_data_yaml: /data/fresh/data.yaml' in summary
    assert 'training_plan_epochs: 10' in summary
    assert 'training_plan_next_step_tool: start_training_loop' in summary
    assert 'stale.pt' not in summary
    assert '/data/stale/data.yaml' not in summary


def main() -> None:
    _scenario_sparse_summary()
    _scenario_compact_populated_summary()
    _scenario_graph_training_context_overrides_stale_mirror()
    print('context builder ok')


if __name__ == '__main__':
    main()

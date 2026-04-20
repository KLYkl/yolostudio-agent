"""
Harness 状态预设快照
每个预设是 session_state 的 dotted-key → value 映射
"""
from __future__ import annotations
from typing import Any

STATE_PRESETS: dict[str, dict[str, Any]] = {
    # 空白状态
    'empty': {},

    # 有 prepare pending (synthetic)
    'pending_prepare': {
        'active_training.training_plan_draft': {
            'dataset_path': '/home/kly/ct_loop/data_ct',
            'next_step_tool': 'prepare_dataset_for_training',
            'next_step_args': {'dataset_path': '/home/kly/ct_loop/data_ct'},
            'execution_mode': 'prepare_then_train',
            'planned_training_args': {'model': 'yolov8n.pt', 'epochs': 30, 'batch': 16},
        },
        'pending_confirmation.tool_name': 'prepare_dataset_for_training',
        'pending_confirmation.tool_args': {'dataset_path': '/home/kly/ct_loop/data_ct'},
        'pending_confirmation.source': 'synthetic',
        'pending_confirmation.interrupt_kind': 'tool_approval',
        'pending_confirmation.summary': '准备数据集 /home/kly/ct_loop/data_ct',
        'pending_confirmation.objective': '准备数据集',
        'pending_confirmation.thread_id': 'harness-pending-prepare',
        'pending_confirmation.created_at': '2026-04-19T00:00:00Z',
    },

    # 有 split pending (graph source)
    'pending_split_graph': {
        'pending_confirmation.tool_name': 'split_dataset',
        'pending_confirmation.tool_args': {'dataset_path': '/data/ct', 'ratio': '0.8/0.1/0.1'},
        'pending_confirmation.source': 'graph',
        'pending_confirmation.interrupt_kind': 'high_risk_tool',
        'pending_confirmation.summary': '划分数据集 /data/ct',
        'pending_confirmation.objective': '划分数据集',
        'pending_confirmation.thread_id': 'harness-pending-split',
        'pending_confirmation.created_at': '2026-04-19T00:00:00Z',
    },

    # 训练正在运行
    'training_running': {
        'active_training.running': True,
        'active_training.workflow_state': 'training',
        'active_training.model': 'yolov8n.pt',
        'active_training.data_yaml': '/data/ct/data.yaml',
        'active_training.training_plan_draft': {
            'dataset_path': '/data/ct',
            'next_step_tool': '',
            'execution_mode': 'prepare_then_train',
        },
    },

    # startup 恢复态：session state 有 pending 但 graph 是空的
    'startup_with_stale_pending': {
        'pending_confirmation.tool_name': 'prepare_dataset_for_training',
        'pending_confirmation.tool_args': {'dataset_path': '/data/ct'},
        'pending_confirmation.source': 'synthetic',
        'pending_confirmation.thread_id': 'harness-startup-stale',
        'pending_confirmation.summary': '准备数据集',
        'pending_confirmation.objective': '准备数据集',
        'pending_confirmation.created_at': '2026-04-18T12:00:00Z',
    },
}

# 用于确认语义测试的同义族
CONFIRMATION_SYNONYMS: dict[str, list[str]] = {
    'approve': [
        '确认', '可以开始', '没问题', '就按这个来', '继续',
        '好的', '行', '就这样', '开始吧', 'yes', 'y',
    ],
    'approve_with_training': [
        '没问题，开始训练', '行，启动吧', '可以，直接训练',
        '那就训练吧', '确认，开始训练',
    ],
    'deny': [
        '取消', '算了', '先不做', '不用了', '不继续',
    ],
    'edit': [
        '把 batch 改成 12', 'epochs 改 50 再继续',
        '把学习率调到 0.001', '换成 yolov8s.pt',
    ],
    'clarify': [
        '为什么用这个模型', '解释一下', '先给我看计划',
    ],
}

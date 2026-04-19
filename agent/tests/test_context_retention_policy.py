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

from yolostudio_agent.agent.client.context_retention_policy import build_context_retention_decision
from yolostudio_agent.agent.client.session_state import SessionState


def test_explicit_reference_reuses_history() -> None:
    state = SessionState(session_id='retention-explicit')
    decision = build_context_retention_decision(
        state=state,
        user_text='刚才那个结果再详细一点',
        explicitly_references_previous_context=True,
    )
    assert decision.reuse_history is True
    assert decision.reason == 'explicit_reference'


def test_active_workflow_reuses_history_without_new_targets() -> None:
    state = SessionState(session_id='retention-active')
    state.active_training.running = True
    state.active_training.workflow_state = 'running'
    decision = build_context_retention_decision(
        state=state,
        user_text='现在什么情况了？详细一点',
        explicitly_references_previous_context=False,
    )
    assert decision.reuse_history is True
    assert decision.reason == 'active_workflow'


def test_cached_followup_reuses_history_without_new_task_targets() -> None:
    state = SessionState(session_id='retention-cached')
    state.active_prediction.last_export = {
        'ok': True,
        'summary': '预测报告导出完成',
        'export_path': '/tmp/report.md',
    }
    decision = build_context_retention_decision(
        state=state,
        user_text='那个报告再详细一点',
        explicitly_references_previous_context=False,
    )
    assert decision.reuse_history is True
    assert decision.reason == 'prediction_followup_context'


def test_new_task_targets_strip_ephemeral_context() -> None:
    state = SessionState(session_id='retention-new-target')
    state.active_prediction.last_export = {
        'ok': True,
        'summary': '预测报告导出完成',
        'export_path': '/tmp/report.md',
    }
    decision = build_context_retention_decision(
        state=state,
        user_text='用 /data/new_images 和 /models/best.pt 重新预测',
        explicitly_references_previous_context=False,
    )
    assert decision.reuse_history is False
    assert decision.reason == 'strip_ephemeral_context'


def test_best_run_prediction_followup_keeps_training_history_with_new_target() -> None:
    state = SessionState(session_id='retention-best-run-followup')
    state.active_training.best_run_selection = {
        'summary': '最近最佳训练为 train_log_best。',
        'best_run': {
            'run_id': 'train_log_best',
            'best_weight_path': '/weights/best.pt',
        },
    }
    decision = build_context_retention_decision(
        state=state,
        user_text='用最佳训练去预测图片 /data/images。',
        explicitly_references_previous_context=False,
    )
    assert decision.reuse_history is True
    assert decision.reason == 'best_run_prediction_followup'


def test_read_only_prediction_followup_preserves_state_without_reusing_history() -> None:
    state = SessionState(session_id='retention-read-only-prediction-followup')
    state.active_prediction.last_result = {
        'ok': True,
        'summary': '预测完成: 已处理 2 张图片',
        'report_path': '/tmp/predict/prediction_report.json',
        'output_dir': '/tmp/predict',
    }
    decision = build_context_retention_decision(
        state=state,
        user_text='现在是什么情况了？我需要详细一点的信息',
        explicitly_references_previous_context=False,
    )
    assert decision.reuse_history is False
    assert decision.reason == 'read_only_prediction_followup'
    assert decision.preserve_state_context is True


def test_read_only_training_followup_preserves_state_without_reusing_history() -> None:
    state = SessionState(session_id='retention-read-only-training-followup')
    state.active_training.best_run_selection = {
        'summary': '最近最佳训练为 train_log_best。',
        'best_run': {
            'run_id': 'train_log_best',
            'best_weight_path': '/weights/best.pt',
        },
    }
    decision = build_context_retention_decision(
        state=state,
        user_text='查看这次最佳训练详情。',
        explicitly_references_previous_context=False,
    )
    assert decision.reuse_history is False
    assert decision.reason == 'read_only_training_followup'
    assert decision.preserve_state_context is True


def test_dataset_followup_uses_dataset_domain_context() -> None:
    state = SessionState(session_id='retention-dataset-followup')
    state.active_dataset.last_scan = {
        'ok': True,
        'summary': '扫描完成: 共 120 张图片',
        'classes': ['epidural', 'subdural'],
    }
    decision = build_context_retention_decision(
        state=state,
        user_text='有哪些类别？',
        explicitly_references_previous_context=False,
    )
    assert decision.reuse_history is True
    assert decision.reason == 'dataset_followup_context'


def test_dataset_class_name_followup_reuses_dataset_context() -> None:
    state = SessionState(session_id='retention-dataset-class-followup')
    state.active_dataset.last_scan = {
        'ok': True,
        'summary': '扫描完成: 共 120 张图片',
        'classes': ['epidural', 'subdural'],
    }
    decision = build_context_retention_decision(
        state=state,
        user_text='Epidural 有多少个？',
        explicitly_references_previous_context=False,
    )
    assert decision.reuse_history is True
    assert decision.reason == 'dataset_followup_context'


def test_knowledge_followup_uses_knowledge_domain_context() -> None:
    state = SessionState(session_id='retention-knowledge-followup')
    state.active_knowledge.last_analysis = {
        'ok': True,
        'summary': '当前主要瓶颈是数据质量。',
    }
    decision = build_context_retention_decision(
        state=state,
        user_text='依据是什么？',
        explicitly_references_previous_context=False,
    )
    assert decision.reuse_history is True
    assert decision.reason == 'knowledge_followup_context'


def main() -> None:
    test_explicit_reference_reuses_history()
    test_active_workflow_reuses_history_without_new_targets()
    test_cached_followup_reuses_history_without_new_task_targets()
    test_new_task_targets_strip_ephemeral_context()
    test_best_run_prediction_followup_keeps_training_history_with_new_target()
    test_read_only_prediction_followup_preserves_state_without_reusing_history()
    test_read_only_training_followup_preserves_state_without_reusing_history()
    test_dataset_followup_uses_dataset_domain_context()
    test_dataset_class_name_followup_reuses_dataset_context()
    test_knowledge_followup_uses_knowledge_domain_context()
    print('context retention policy ok')


if __name__ == '__main__':
    main()

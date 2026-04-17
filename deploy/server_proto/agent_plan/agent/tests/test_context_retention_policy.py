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
    assert decision.reason == 'cached_followup_context'


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


def main() -> None:
    test_explicit_reference_reuses_history()
    test_active_workflow_reuses_history_without_new_targets()
    test_cached_followup_reuses_history_without_new_task_targets()
    test_new_task_targets_strip_ephemeral_context()
    print('context retention policy ok')


if __name__ == '__main__':
    main()

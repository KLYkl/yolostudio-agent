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

from yolostudio_agent.agent.client.prediction_request_service import resolve_prediction_request_followup_action


def _run() -> None:
    missing_best_weight = resolve_prediction_request_followup_action(
        wants_predict=True,
        training_command_like=False,
        wants_best_weight_prediction=True,
        best_run_selection={
            'summary': '最近最佳训练为 train_log_best。',
            'best_run': {'run_id': 'train_log_best'},
        },
    )
    assert missing_best_weight == {
        'action': 'reply',
        'reply': '我当前不能直接假定“最佳训练”的权重文件路径；请先查看最佳训练详情，或明确给我可用的权重路径。',
        'status': 'completed',
    }, missing_best_weight

    best_weight_ready = resolve_prediction_request_followup_action(
        wants_predict=True,
        training_command_like=False,
        wants_best_weight_prediction=True,
        best_run_selection={
            'summary': '最近最佳训练为 train_log_best。',
            'best_run': {'run_id': 'train_log_best', 'best_weight_path': '/weights/best.pt'},
        },
    )
    assert best_weight_ready == {'action': 'none'}, best_weight_ready

    best_weight_from_matching_inspection = resolve_prediction_request_followup_action(
        wants_predict=True,
        training_command_like=False,
        wants_best_weight_prediction=True,
        best_run_selection={
            'summary': '最近最佳训练为 train_log_best。',
            'best_run': {'run_id': 'train_log_best'},
        },
        last_run_inspection={
            'summary': '最佳训练详情已就绪',
            'selected_run_id': 'train_log_best',
            'best_weight_path': '/weights/best.pt',
        },
    )
    assert best_weight_from_matching_inspection == {'action': 'none'}, best_weight_from_matching_inspection

    mismatched_inspection_still_blocks = resolve_prediction_request_followup_action(
        wants_predict=True,
        training_command_like=False,
        wants_best_weight_prediction=True,
        best_run_selection={
            'summary': '最近最佳训练为 train_log_best。',
            'best_run': {'run_id': 'train_log_best'},
        },
        last_run_inspection={
            'summary': '另一个训练详情已就绪',
            'selected_run_id': 'train_log_other',
            'best_weight_path': '/weights/other.pt',
        },
    )
    assert mismatched_inspection_still_blocks == {
        'action': 'reply',
        'reply': '我当前不能直接假定“最佳训练”的权重文件路径；请先查看最佳训练详情，或明确给我可用的权重路径。',
        'status': 'completed',
    }, mismatched_inspection_still_blocks

    non_prediction = resolve_prediction_request_followup_action(
        wants_predict=False,
        training_command_like=False,
        wants_best_weight_prediction=True,
        best_run_selection={},
    )
    assert non_prediction == {'action': 'none'}, non_prediction

    print('prediction request service ok')


if __name__ == '__main__':
    _run()

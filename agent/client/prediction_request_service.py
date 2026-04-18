from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.execution_contracts import PredictionRequestFollowupAction


def resolve_prediction_request_followup_action(
    *,
    wants_predict: bool,
    training_command_like: bool,
    wants_best_weight_prediction: bool,
    best_run_selection: dict[str, Any] | None,
) -> PredictionRequestFollowupAction:
    if wants_best_weight_prediction and wants_predict and not training_command_like:
        best_selection = dict(best_run_selection or {})
        best_run = dict(best_selection.get('best_run') or {})
        weight_path = str(best_run.get('best_weight_path') or best_run.get('weights_path') or '').strip()
        if not weight_path:
            return {
                'action': 'reply',
                'reply': '我当前不能直接假定“最佳训练”的权重文件路径；请先查看最佳训练详情，或明确给我可用的权重路径。',
                'status': 'completed',
            }
    return {'action': 'none'}

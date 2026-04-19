from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.execution_contracts import PredictionRequestFollowupAction


def _extract_weight_path(payload: dict[str, Any] | None) -> str:
    data = dict(payload or {})
    nested_best_run = dict(data.get('best_run') or {})
    summary_overview = dict(data.get('summary_overview') or {})
    return str(
        data.get('resolved_weight_path')
        or data.get('best_weight_path')
        or data.get('weights_path')
        or data.get('weight_path')
        or nested_best_run.get('best_weight_path')
        or nested_best_run.get('weights_path')
        or summary_overview.get('best_weight_path')
        or summary_overview.get('weights_path')
        or ''
    ).strip()


def _resolve_best_run_id(best_run_selection: dict[str, Any] | None) -> str:
    selection = dict(best_run_selection or {})
    best_run = dict(selection.get('best_run') or {})
    return str(best_run.get('run_id') or selection.get('best_run_id') or '').strip()


def _resolve_inspected_run_id(last_run_inspection: dict[str, Any] | None) -> str:
    inspection = dict(last_run_inspection or {})
    return str(inspection.get('selected_run_id') or inspection.get('run_id') or '').strip()


def resolve_prediction_request_followup_action(
    *,
    wants_predict: bool,
    training_command_like: bool,
    wants_best_weight_prediction: bool,
    best_run_selection: dict[str, Any] | None,
    last_run_inspection: dict[str, Any] | None = None,
) -> PredictionRequestFollowupAction:
    if wants_best_weight_prediction and wants_predict and not training_command_like:
        best_selection = dict(best_run_selection or {})
        weight_path = _extract_weight_path(best_selection)
        if not weight_path:
            best_run_id = _resolve_best_run_id(best_selection)
            inspected_run_id = _resolve_inspected_run_id(last_run_inspection)
            if best_run_id and inspected_run_id and best_run_id == inspected_run_id:
                weight_path = _extract_weight_path(last_run_inspection)
        if not weight_path:
            return {
                'action': 'reply',
                'reply': '我当前不能直接假定“最佳训练”的权重文件路径；请先查看最佳训练详情，或明确给我可用的权重路径。',
                'status': 'completed',
            }
    return {'action': 'none'}

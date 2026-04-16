from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.state_projectors.common import (
    _prediction_management_snapshot,
    _prediction_result_snapshot,
)


def apply_prediction_tool_result(
    session_state: SessionState,
    tool_name: str,
    result: dict[str, Any],
    tool_args: dict[str, Any],
) -> None:
    pred = session_state.active_prediction
    if tool_name == 'predict_images' and result.get('ok'):
        pred.source_path = str(result.get('source_path') or tool_args.get('source_path', pred.source_path))
        pred.model = str(result.get('model') or tool_args.get('model', pred.model))
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_result = _prediction_result_snapshot(result, mode='images')
    elif tool_name == 'predict_videos' and result.get('ok'):
        pred.source_path = str(result.get('source_path') or tool_args.get('source_path', pred.source_path))
        pred.model = str(result.get('model') or tool_args.get('model', pred.model))
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_result = _prediction_result_snapshot(result, mode='videos')
    elif tool_name == 'summarize_prediction_results' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        if result.get('model'):
            pred.model = str(result.get('model'))
        if result.get('source_path'):
            pred.source_path = str(result.get('source_path'))
        pred.last_summary = _prediction_result_snapshot(result, mode=str(result.get('mode') or 'images'))
        pred.last_result = dict(pred.last_summary)
    elif tool_name == 'inspect_prediction_outputs' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_inspection = _prediction_management_snapshot(
            result,
            overview_key='prediction_output_overview',
            extra_keys=('artifact_roots', 'path_list_files', 'output_dir', 'report_path'),
        )
    elif tool_name == 'export_prediction_report' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_export = _prediction_management_snapshot(
            result,
            overview_key='export_overview',
            extra_keys=('export_path', 'export_format', 'output_dir', 'report_path'),
        )
    elif tool_name == 'export_prediction_path_lists' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.last_path_lists = _prediction_management_snapshot(
            result,
            overview_key='path_list_overview',
            fallback_overview_key='export_overview',
            extra_keys=('export_dir', 'output_dir', 'report_path', 'detected_items_path', 'empty_items_path', 'failed_items_path', 'detected_count', 'empty_count', 'failed_count'),
        )
    elif tool_name == 'organize_prediction_results' and result.get('ok'):
        pred.output_dir = str(result.get('source_output_dir') or pred.output_dir)
        pred.report_path = str(result.get('source_report_path') or pred.report_path)
        pred.last_organized_result = _prediction_management_snapshot(
            result,
            overview_key='organization_overview',
            extra_keys=('destination_dir', 'organize_by', 'artifact_preference', 'copied_items', 'bucket_stats', 'sample_outputs'),
        )

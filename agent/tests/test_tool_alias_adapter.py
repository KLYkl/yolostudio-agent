from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import StructuredTool

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.tool_adapter import adapt_tools_for_chat_model, canonical_tool_name, normalize_tool_args


def _noop(**kwargs):
    return kwargs


def main() -> None:
    assert canonical_tool_name('detect_duplicates') == 'detect_duplicate_images'
    assert canonical_tool_name('detect_corrupted_images') == 'run_dataset_health_check'
    assert canonical_tool_name('dataset_manager.prepare_dataset') == 'prepare_dataset_for_training'
    assert canonical_tool_name('predict_directory') == 'predict_images'
    assert canonical_tool_name('predict_video_directory') == 'predict_videos'
    assert canonical_tool_name('summarize_predictions') == 'summarize_prediction_results'

    health_args = normalize_tool_args('detect_corrupted_images', {'path': '/data/set'})
    assert health_args['dataset_path'] == '/data/set'

    dup_args = normalize_tool_args('detect_duplicates', {'img_dir': '/data/set'})
    assert dup_args['dataset_path'] == '/data/set'

    prep_args = normalize_tool_args('dataset_manager.prepare_dataset', {'path': '/data/set', 'force_split': True})
    assert prep_args['dataset_path'] == '/data/set'
    assert prep_args['force_split'] is True

    readiness_args = normalize_tool_args('training_readiness', {'path': '/data/set'})
    assert readiness_args['img_dir'] == '/data/set'

    scan_args = normalize_tool_args('scan_dataset', {'dataset_path': '/data/set'})
    assert scan_args['img_dir'] == '/data/set'

    predict_args = normalize_tool_args('predict_directory', {'path': '/data/images', 'model': 'yolov8n.pt'})
    assert predict_args['source_path'] == '/data/images'
    assert predict_args['model'] == 'yolov8n.pt'

    predict_video_args = normalize_tool_args('predict_video_directory', {'path': '/data/videos', 'model': 'yolov8n.pt'})
    assert predict_video_args['source_path'] == '/data/videos'
    assert predict_video_args['model'] == 'yolov8n.pt'

    predict_summary_args = normalize_tool_args('summarize_predictions', {'path': '/tmp/predict/prediction_report.json'})
    assert predict_summary_args['report_path'] == '/tmp/predict/prediction_report.json'

    tools = [
        StructuredTool.from_function(func=_noop, name='detect_duplicate_images', description='dup'),
        StructuredTool.from_function(func=_noop, name='run_dataset_health_check', description='health'),
        StructuredTool.from_function(func=_noop, name='prepare_dataset_for_training', description='prepare'),
        StructuredTool.from_function(func=_noop, name='predict_images', description='predict'),
        StructuredTool.from_function(func=_noop, name='predict_videos', description='predict-videos'),
        StructuredTool.from_function(func=_noop, name='summarize_prediction_results', description='predict-summary'),
    ]
    adapted = adapt_tools_for_chat_model(tools)
    names = {tool.name for tool in adapted}
    for alias in (
        'detect_duplicates',
        'detect_corrupted_images',
        'prepare_dataset',
        'dataset_manager.prepare_dataset',
        'predict_directory',
        'batch_predict_images',
        'predict_images_in_dir',
        'predict_video_directory',
        'batch_predict_videos',
        'predict_videos_in_dir',
        'summarize_predictions',
        'summarize_prediction_report',
        'analyze_prediction_report',
    ):
        assert alias in names, names

    print('tool alias adapter ok')


if __name__ == '__main__':
    main()

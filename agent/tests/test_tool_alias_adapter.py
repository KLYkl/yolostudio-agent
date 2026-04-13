from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import StructuredTool

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.tool_adapter import adapt_tools_for_chat_model, canonical_tool_name, normalize_tool_args


def _noop(**kwargs):
    return kwargs


def main() -> None:
    assert canonical_tool_name('detect_duplicates') == 'detect_duplicate_images'
    assert canonical_tool_name('detect_corrupted_images') == 'run_dataset_health_check'
    assert canonical_tool_name('dataset_manager.prepare_dataset') == 'prepare_dataset_for_training'
    assert canonical_tool_name('predict_directory') == 'predict_images'
    assert canonical_tool_name('predict_video_directory') == 'predict_videos'
    assert canonical_tool_name('summarize_predictions') == 'summarize_prediction_results'
    assert canonical_tool_name('preview_convert_labels') == 'preview_convert_format'
    assert canonical_tool_name('convert_labels_format') == 'convert_format'
    assert canonical_tool_name('preview_replace_labels') == 'preview_modify_labels'
    assert canonical_tool_name('replace_labels') == 'modify_labels'
    assert canonical_tool_name('fill_missing_labels') == 'generate_missing_labels'
    assert canonical_tool_name('create_empty_labels') == 'generate_empty_labels'
    assert canonical_tool_name('preview_group_by_class') == 'preview_categorize_by_class'
    assert canonical_tool_name('group_by_class') == 'categorize_by_class'
    assert canonical_tool_name('compare_training_history') == 'compare_training_runs'
    assert canonical_tool_name('best_training_run') == 'select_best_training_run'

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

    convert_args = normalize_tool_args('convert_labels_format', {'path': '/data/set', 'format': 'xml'})
    assert convert_args['dataset_path'] == '/data/set'
    assert convert_args['target_format'] == 'xml'

    modify_args = normalize_tool_args('replace_labels', {'path': '/data/set', 'from': 'car', 'to': 'truck', 'operation': 'replace'})
    assert modify_args['dataset_path'] == '/data/set'
    assert modify_args['old_value'] == 'car'
    assert modify_args['new_value'] == 'truck'
    assert modify_args['action'] == 'replace'

    missing_args = normalize_tool_args('fill_missing_labels', {'path': '/data/set', 'format': 'xml'})
    assert missing_args['dataset_path'] == '/data/set'
    assert missing_args['label_format'] == 'xml'

    categorize_args = normalize_tool_args('group_by_class', {'path': '/data/set', 'out_dir': '/tmp/cats'})
    assert categorize_args['dataset_path'] == '/data/set'
    assert categorize_args['output_dir'] == '/tmp/cats'

    tools = [
        StructuredTool.from_function(func=_noop, name='detect_duplicate_images', description='dup'),
        StructuredTool.from_function(func=_noop, name='run_dataset_health_check', description='health'),
        StructuredTool.from_function(func=_noop, name='prepare_dataset_for_training', description='prepare'),
        StructuredTool.from_function(func=_noop, name='predict_images', description='predict'),
        StructuredTool.from_function(func=_noop, name='predict_videos', description='predict-videos'),
        StructuredTool.from_function(func=_noop, name='summarize_prediction_results', description='predict-summary'),
        StructuredTool.from_function(func=_noop, name='preview_convert_format', description='preview-convert'),
        StructuredTool.from_function(func=_noop, name='convert_format', description='convert'),
        StructuredTool.from_function(func=_noop, name='preview_modify_labels', description='preview-modify'),
        StructuredTool.from_function(func=_noop, name='modify_labels', description='modify'),
        StructuredTool.from_function(func=_noop, name='generate_missing_labels', description='missing'),
        StructuredTool.from_function(func=_noop, name='generate_empty_labels', description='empty'),
        StructuredTool.from_function(func=_noop, name='preview_categorize_by_class', description='preview-categorize'),
        StructuredTool.from_function(func=_noop, name='categorize_by_class', description='categorize'),
        StructuredTool.from_function(func=_noop, name='compare_training_runs', description='compare-training'),
        StructuredTool.from_function(func=_noop, name='select_best_training_run', description='best-training'),
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
        'preview_convert_labels',
        'convert_labels_format',
        'preview_replace_labels',
        'replace_labels',
        'fill_missing_labels',
        'create_empty_labels',
        'preview_group_by_class',
        'group_by_class',
        'compare_training_history',
        'compare_training_results',
        'best_training_run',
        'pick_best_training_run',
    ):
        assert alias in names, names

    print('tool alias adapter ok')


if __name__ == '__main__':
    main()

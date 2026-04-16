from __future__ import annotations

import sys
import types
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def _install_fake_tool_dependencies() -> None:
    core_mod = types.ModuleType('langchain_core')
    tools_mod = types.ModuleType('langchain_core.tools')

    class _BaseTool:
        name = 'fake'
        description = 'fake'
        args_schema = None

    class _StructuredTool(_BaseTool):
        @classmethod
        def from_function(cls, func=None, coroutine=None, name='', description='', args_schema=None, return_direct=False):
            tool = cls()
            tool.func = func
            tool.coroutine = coroutine
            tool.name = name
            tool.description = description
            tool.args_schema = args_schema
            tool.return_direct = return_direct
            return tool

    tools_mod.BaseTool = _BaseTool
    tools_mod.StructuredTool = _StructuredTool
    core_mod.tools = tools_mod
    sys.modules['langchain_core'] = core_mod
    sys.modules['langchain_core.tools'] = tools_mod

    pyd_mod = types.ModuleType('pydantic')

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _Field(default=None, **kwargs):
        del kwargs
        return default

    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['pydantic'] = pyd_mod


_install_fake_tool_dependencies()

from langchain_core.tools import StructuredTool

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
    assert canonical_tool_name('inspect_prediction_output') == 'inspect_prediction_outputs'
    assert canonical_tool_name('export_prediction_summary') == 'export_prediction_report'
    assert canonical_tool_name('export_prediction_paths') == 'export_prediction_path_lists'
    assert canonical_tool_name('collect_prediction_hits') == 'organize_prediction_results'
    assert canonical_tool_name('scan_available_cameras') == 'scan_cameras'
    assert canonical_tool_name('scan_available_screens') == 'scan_screens'
    assert canonical_tool_name('probe_rtsp_stream') == 'test_rtsp_stream'
    assert canonical_tool_name('start_live_camera_prediction') == 'start_camera_prediction'
    assert canonical_tool_name('start_live_rtsp_prediction') == 'start_rtsp_prediction'
    assert canonical_tool_name('start_live_screen_prediction') == 'start_screen_prediction'
    assert canonical_tool_name('check_live_prediction_status') == 'check_realtime_prediction_status'
    assert canonical_tool_name('stop_live_prediction') == 'stop_realtime_prediction'
    assert canonical_tool_name('preview_convert_labels') == 'preview_convert_format'
    assert canonical_tool_name('convert_labels_format') == 'convert_format'
    assert canonical_tool_name('preview_replace_labels') == 'preview_modify_labels'
    assert canonical_tool_name('replace_labels') == 'modify_labels'
    assert canonical_tool_name('fill_missing_labels') == 'generate_missing_labels'
    assert canonical_tool_name('create_empty_labels') == 'generate_empty_labels'
    assert canonical_tool_name('preview_group_by_class') == 'preview_categorize_by_class'
    assert canonical_tool_name('group_by_class') == 'categorize_by_class'
    assert canonical_tool_name('list_remote_servers') == 'list_remote_profiles'
    assert canonical_tool_name('upload_to_server') == 'upload_assets_to_remote'
    assert canonical_tool_name('scp_to_server') == 'upload_assets_to_remote'
    assert canonical_tool_name('download_from_server') == 'download_assets_from_remote'
    assert canonical_tool_name('sync_remote_to_local') == 'download_assets_from_remote'

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

    inspect_args = normalize_tool_args('inspect_prediction_output', {'folder': '/tmp/predict'})
    assert inspect_args['output_dir'] == '/tmp/predict'

    export_report_args = normalize_tool_args(
        'export_prediction_summary',
        {'path': '/tmp/predict/prediction_report.json', 'out_dir': '/tmp/out/report.md', 'format': 'markdown'},
    )
    assert export_report_args['report_path'] == '/tmp/predict/prediction_report.json'
    assert export_report_args['export_path'] == '/tmp/out/report.md'
    assert export_report_args['export_format'] == 'markdown'

    export_paths_args = normalize_tool_args('export_prediction_paths', {'folder': '/tmp/predict', 'out_dir': '/tmp/lists'})
    assert export_paths_args['output_dir'] == '/tmp/predict'
    assert export_paths_args['export_dir'] == '/tmp/lists'

    organize_args = normalize_tool_args('collect_prediction_hits', {'folder': '/tmp/predict', 'out_dir': '/tmp/hits', 'mode': 'by_class'})
    assert organize_args['output_dir'] == '/tmp/predict'
    assert organize_args['destination_dir'] == '/tmp/hits'
    assert organize_args['organize_by'] == 'by_class'

    rtsp_probe_args = normalize_tool_args('probe_rtsp_stream', {'url': 'rtsp://demo/live'})
    assert rtsp_probe_args['rtsp_url'] == 'rtsp://demo/live'

    start_camera_args = normalize_tool_args('start_live_camera_prediction', {'source': 'demo.pt', 'camera_id': 1})
    assert start_camera_args['model'] == 'demo.pt'
    assert start_camera_args['camera_id'] == 1

    start_rtsp_args = normalize_tool_args('start_live_rtsp_prediction', {'source': 'demo.pt', 'url': 'rtsp://demo/live'})
    assert start_rtsp_args['model'] == 'demo.pt'
    assert start_rtsp_args['rtsp_url'] == 'rtsp://demo/live'

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

    remote_list_args = normalize_tool_args('list_remote_servers', {})
    assert remote_list_args == {}

    remote_upload_args = normalize_tool_args(
        'upload_to_server',
        {
            'path': r'D:\weights\best.pt',
            'server_name': 'lab',
            'remote_dir': '/srv/stage',
            'user': 'tester',
            'resume_transfer': False,
            'verify': False,
            'hash_algo': 'md5',
            'threshold_mb': 32,
            'chunk_mb': 8,
            'progress': False,
        },
    )
    assert remote_upload_args['paths_text'] == r'D:\weights\best.pt'
    assert remote_upload_args['server'] == 'lab'
    assert remote_upload_args['remote_root'] == '/srv/stage'
    assert remote_upload_args['username'] == 'tester'
    assert remote_upload_args['resume'] is False
    assert remote_upload_args['verify_hash'] is False
    assert remote_upload_args['hash_algorithm'] == 'md5'
    assert remote_upload_args['large_file_threshold_mb'] == 32
    assert remote_upload_args['chunk_size_mb'] == 8
    assert remote_upload_args['show_progress'] is False

    remote_download_args = normalize_tool_args(
        'download_from_server',
        {
            'remote_path': '/srv/output/report.json',
            'server_name': 'lab',
            'user': 'tester',
            'local_dir': r'D:\downloads',
        },
    )
    assert remote_download_args['paths_text'] == '/srv/output/report.json'
    assert remote_download_args['server'] == 'lab'
    assert remote_download_args['username'] == 'tester'
    assert remote_download_args['local_root'] == r'D:\downloads'

    tools = [
        StructuredTool.from_function(func=_noop, name='detect_duplicate_images', description='dup'),
        StructuredTool.from_function(func=_noop, name='run_dataset_health_check', description='health'),
        StructuredTool.from_function(func=_noop, name='dataset_training_readiness', description='dataset-readiness'),
        StructuredTool.from_function(func=_noop, name='training_readiness', description='training-readiness'),
        StructuredTool.from_function(func=_noop, name='prepare_dataset_for_training', description='prepare'),
        StructuredTool.from_function(func=_noop, name='predict_images', description='predict'),
        StructuredTool.from_function(func=_noop, name='predict_videos', description='predict-videos'),
        StructuredTool.from_function(func=_noop, name='summarize_prediction_results', description='predict-summary'),
        StructuredTool.from_function(func=_noop, name='inspect_prediction_outputs', description='predict-inspect'),
        StructuredTool.from_function(func=_noop, name='export_prediction_report', description='predict-export'),
        StructuredTool.from_function(func=_noop, name='export_prediction_path_lists', description='predict-paths'),
        StructuredTool.from_function(func=_noop, name='organize_prediction_results', description='predict-organize'),
        StructuredTool.from_function(func=_noop, name='scan_cameras', description='scan-cameras'),
        StructuredTool.from_function(func=_noop, name='scan_screens', description='scan-screens'),
        StructuredTool.from_function(func=_noop, name='test_rtsp_stream', description='probe-rtsp'),
        StructuredTool.from_function(func=_noop, name='start_camera_prediction', description='start-camera'),
        StructuredTool.from_function(func=_noop, name='start_rtsp_prediction', description='start-rtsp'),
        StructuredTool.from_function(func=_noop, name='start_screen_prediction', description='start-screen'),
        StructuredTool.from_function(func=_noop, name='check_realtime_prediction_status', description='check-realtime'),
        StructuredTool.from_function(func=_noop, name='stop_realtime_prediction', description='stop-realtime'),
        StructuredTool.from_function(func=_noop, name='preview_convert_format', description='preview-convert'),
        StructuredTool.from_function(func=_noop, name='convert_format', description='convert'),
        StructuredTool.from_function(func=_noop, name='preview_modify_labels', description='preview-modify'),
        StructuredTool.from_function(func=_noop, name='modify_labels', description='modify'),
        StructuredTool.from_function(func=_noop, name='generate_missing_labels', description='missing'),
        StructuredTool.from_function(func=_noop, name='generate_empty_labels', description='empty'),
        StructuredTool.from_function(func=_noop, name='preview_categorize_by_class', description='preview-categorize'),
        StructuredTool.from_function(func=_noop, name='categorize_by_class', description='categorize'),
        StructuredTool.from_function(func=_noop, name='list_remote_profiles', description='remote-profiles'),
        StructuredTool.from_function(func=_noop, name='upload_assets_to_remote', description='remote-upload'),
        StructuredTool.from_function(func=_noop, name='download_assets_from_remote', description='remote-download'),
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
        'inspect_prediction_output',
        'show_prediction_outputs',
        'prediction_output_overview',
        'export_prediction_summary',
        'write_prediction_report',
        'export_prediction_paths',
        'collect_prediction_hits',
        'group_prediction_results',
        'scan_available_cameras',
        'scan_available_screens',
        'probe_rtsp_stream',
        'start_live_camera_prediction',
        'start_live_rtsp_prediction',
        'start_live_screen_prediction',
        'check_live_prediction_status',
        'stop_live_prediction',
        'preview_convert_labels',
        'convert_labels_format',
        'preview_replace_labels',
        'replace_labels',
        'fill_missing_labels',
        'create_empty_labels',
        'preview_group_by_class',
        'group_by_class',
        'list_remote_servers',
        'list_remote_targets',
        'show_remote_profiles',
        'upload_to_server',
        'upload_to_remote',
        'sync_local_to_remote',
        'scp_to_server',
        'download_from_server',
        'download_from_remote',
        'pull_from_server',
        'sync_remote_to_local',
        'scp_from_server',
    ):
        assert alias in names, names

    canonical_only = adapt_tools_for_chat_model(tools, include_aliases=False)
    canonical_names = {tool.name for tool in canonical_only}
    assert canonical_names == {
        'detect_duplicate_images',
        'run_dataset_health_check',
        'dataset_training_readiness',
        'training_readiness',
        'prepare_dataset_for_training',
        'predict_images',
        'predict_videos',
        'summarize_prediction_results',
        'inspect_prediction_outputs',
        'export_prediction_report',
        'export_prediction_path_lists',
        'organize_prediction_results',
        'scan_cameras',
        'scan_screens',
        'test_rtsp_stream',
        'start_camera_prediction',
        'start_rtsp_prediction',
        'start_screen_prediction',
        'check_realtime_prediction_status',
        'stop_realtime_prediction',
        'preview_convert_format',
        'convert_format',
        'preview_modify_labels',
        'modify_labels',
        'generate_missing_labels',
        'generate_empty_labels',
        'preview_categorize_by_class',
        'categorize_by_class',
        'list_remote_profiles',
        'upload_assets_to_remote',
        'download_assets_from_remote',
    }, canonical_names
    canonical_map = {tool.name: tool for tool in canonical_only}
    assert '只检查数据集本身是否已经具备直接训练的结构条件' in canonical_map['dataset_training_readiness'].description
    assert '不要用于纯数据集可训练性问题' in canonical_map['training_readiness'].description

    print('tool alias adapter ok')


if __name__ == '__main__':
    main()

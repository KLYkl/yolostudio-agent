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


def main() -> None:
    try:
        from yolostudio_agent.agent.server.mcp_server import mcp
    except Exception as exc:
        print(f'mcp server metadata skipped: {exc}')
        return

    manager = getattr(mcp, '_tool_manager', None)
    assert manager is not None, 'missing tool manager'
    tools = getattr(manager, '_tools', None)
    assert isinstance(tools, dict) and tools, 'missing registered tools'

    def _schema_types(schema: dict) -> set[str]:
        any_of = schema.get('anyOf') or []
        if any_of:
            return {item.get('type') for item in any_of}
        return {str(schema.get('type') or '').strip()} if schema.get('type') else set()

    read_tool = tools['check_training_loop_status']
    assert read_tool.annotations is not None, read_tool
    assert read_tool.annotations.readOnlyHint is True, read_tool.annotations
    assert read_tool.annotations.destructiveHint is False, read_tool.annotations
    assert read_tool.annotations.idempotentHint is True, read_tool.annotations
    assert read_tool.output_schema is not None, read_tool.output_schema

    write_tool = tools['prepare_dataset_for_training']
    assert write_tool.annotations is not None, write_tool
    assert write_tool.annotations.readOnlyHint is False, write_tool.annotations
    assert write_tool.annotations.destructiveHint is True, write_tool.annotations
    assert write_tool.output_schema is not None, write_tool.output_schema
    dataset_path_schema = write_tool.parameters['properties']['dataset_path']
    assert _schema_types(dataset_path_schema) == {'string'}, dataset_path_schema
    assert dataset_path_schema.get('description'), dataset_path_schema
    assert dataset_path_schema.get('examples'), dataset_path_schema
    split_ratio_schema = write_tool.parameters['properties']['split_ratio']
    assert _schema_types(split_ratio_schema) == {'number'}, split_ratio_schema
    assert split_ratio_schema.get('description'), split_ratio_schema
    assert split_ratio_schema.get('examples'), split_ratio_schema

    extract_preview_tool = tools['preview_extract_images']
    extract_source_schema = extract_preview_tool.parameters['properties']['source_path']
    assert _schema_types(extract_source_schema) == {'string'}, extract_source_schema
    assert extract_source_schema.get('description'), extract_source_schema
    assert extract_source_schema.get('examples'), extract_source_schema
    selection_mode_schema = extract_preview_tool.parameters['properties']['selection_mode']
    assert _schema_types(selection_mode_schema) == {'string'}, selection_mode_schema
    assert selection_mode_schema.get('description'), selection_mode_schema
    assert selection_mode_schema.get('examples'), selection_mode_schema
    selected_dirs_schema = extract_preview_tool.parameters['properties']['selected_dirs']
    selected_dir_types = {item.get('type') for item in selected_dirs_schema.get('anyOf', [])}
    assert selected_dir_types == {'array', 'string', 'null'}, selected_dirs_schema
    assert selected_dirs_schema.get('description'), selected_dirs_schema
    assert selected_dirs_schema.get('examples'), selected_dirs_schema

    frame_tool = tools['extract_video_frames']
    frame_mode_schema = frame_tool.parameters['properties']['mode']
    assert _schema_types(frame_mode_schema) == {'string'}, frame_mode_schema
    assert frame_mode_schema.get('description'), frame_mode_schema
    assert frame_mode_schema.get('examples'), frame_mode_schema
    frame_interval_schema = frame_tool.parameters['properties']['frame_interval']
    assert _schema_types(frame_interval_schema) == {'integer'}, frame_interval_schema
    assert frame_interval_schema.get('description'), frame_interval_schema
    assert frame_interval_schema.get('examples'), frame_interval_schema
    frame_max_schema = frame_tool.parameters['properties']['max_frames']
    assert _schema_types(frame_max_schema) == {'integer'}, frame_max_schema
    assert frame_max_schema.get('description'), frame_max_schema
    assert frame_max_schema.get('examples'), frame_max_schema

    scan_tool = tools['scan_dataset']
    scan_img_dir_schema = scan_tool.parameters['properties']['img_dir']
    assert _schema_types(scan_img_dir_schema) == {'string'}, scan_img_dir_schema
    assert scan_img_dir_schema.get('description'), scan_img_dir_schema
    assert scan_img_dir_schema.get('examples'), scan_img_dir_schema
    scan_label_dir_schema = scan_tool.parameters['properties']['label_dir']
    assert _schema_types(scan_label_dir_schema) == {'string'}, scan_label_dir_schema
    assert scan_label_dir_schema.get('description'), scan_label_dir_schema
    assert scan_label_dir_schema.get('examples'), scan_label_dir_schema

    split_tool = tools['split_dataset']
    split_ratio_schema = split_tool.parameters['properties']['ratio']
    assert _schema_types(split_ratio_schema) == {'number'}, split_ratio_schema
    assert split_ratio_schema.get('description'), split_ratio_schema
    assert split_ratio_schema.get('examples'), split_ratio_schema
    split_mode_schema = split_tool.parameters['properties']['mode']
    assert _schema_types(split_mode_schema) == {'string'}, split_mode_schema
    assert split_mode_schema.get('description'), split_mode_schema
    assert split_mode_schema.get('examples'), split_mode_schema

    readiness_tool = tools['training_readiness']
    readiness_yaml_schema = readiness_tool.parameters['properties']['data_yaml']
    assert _schema_types(readiness_yaml_schema) == {'string'}, readiness_yaml_schema
    assert readiness_yaml_schema.get('description'), readiness_yaml_schema
    assert readiness_yaml_schema.get('examples'), readiness_yaml_schema
    readiness_clean_schema = readiness_tool.parameters['properties']['require_clean_labels']
    assert _schema_types(readiness_clean_schema) == {'boolean'}, readiness_clean_schema
    assert readiness_clean_schema.get('description'), readiness_clean_schema
    assert readiness_clean_schema.get('examples'), readiness_clean_schema

    predict_tool = tools['predict_images']
    assert predict_tool.annotations is not None, predict_tool
    assert predict_tool.annotations.readOnlyHint is False, predict_tool.annotations
    assert predict_tool.annotations.destructiveHint is False, predict_tool.annotations
    assert predict_tool.output_schema is not None, predict_tool.output_schema
    predict_source_schema = predict_tool.parameters['properties']['source_path']
    assert _schema_types(predict_source_schema) == {'string'}, predict_source_schema
    assert predict_source_schema.get('description'), predict_source_schema
    assert predict_source_schema.get('examples'), predict_source_schema
    predict_model_schema = predict_tool.parameters['properties']['model']
    assert _schema_types(predict_model_schema) == {'string'}, predict_model_schema
    assert predict_model_schema.get('description'), predict_model_schema
    assert predict_model_schema.get('examples'), predict_model_schema
    predict_conf_schema = predict_tool.parameters['properties']['conf']
    assert _schema_types(predict_conf_schema) == {'number'}, predict_conf_schema
    assert predict_conf_schema.get('description'), predict_conf_schema
    assert predict_conf_schema.get('examples'), predict_conf_schema
    predict_output_schema = predict_tool.parameters['properties']['output_dir']
    assert _schema_types(predict_output_schema) == {'string'}, predict_output_schema
    assert predict_output_schema.get('description'), predict_output_schema
    assert predict_output_schema.get('examples'), predict_output_schema

    image_status_tool = tools['check_image_prediction_status']
    image_session_schema = image_status_tool.parameters['properties']['session_id']
    assert _schema_types(image_session_schema) == {'string'}, image_session_schema
    assert image_session_schema.get('description'), image_session_schema
    assert image_session_schema.get('examples'), image_session_schema

    predict_summary_tool = tools['summarize_prediction_results']
    summary_report_schema = predict_summary_tool.parameters['properties']['report_path']
    assert _schema_types(summary_report_schema) == {'string'}, summary_report_schema
    assert summary_report_schema.get('description'), summary_report_schema
    assert summary_report_schema.get('examples'), summary_report_schema

    export_tool = tools['export_prediction_report']
    export_format_schema = export_tool.parameters['properties']['export_format']
    assert _schema_types(export_format_schema) == {'string'}, export_format_schema
    assert export_format_schema.get('description'), export_format_schema
    assert export_format_schema.get('examples'), export_format_schema

    rtsp_tool = tools['start_rtsp_prediction']
    rtsp_url_schema = rtsp_tool.parameters['properties']['rtsp_url']
    assert _schema_types(rtsp_url_schema) == {'string'}, rtsp_url_schema
    assert rtsp_url_schema.get('description'), rtsp_url_schema
    assert rtsp_url_schema.get('examples'), rtsp_url_schema
    frame_interval_schema = rtsp_tool.parameters['properties']['frame_interval_ms']
    assert _schema_types(frame_interval_schema) == {'integer'}, frame_interval_schema
    assert frame_interval_schema.get('description'), frame_interval_schema
    assert frame_interval_schema.get('examples'), frame_interval_schema

    realtime_status_tool = tools['check_realtime_prediction_status']
    realtime_session_schema = realtime_status_tool.parameters['properties']['session_id']
    assert _schema_types(realtime_session_schema) == {'string'}, realtime_session_schema
    assert realtime_session_schema.get('description'), realtime_session_schema
    assert realtime_session_schema.get('examples'), realtime_session_schema

    video_tool = tools['predict_videos']
    video_max_videos_schema = video_tool.parameters['properties']['max_videos']
    assert _schema_types(video_max_videos_schema) == {'integer'}, video_max_videos_schema
    assert video_max_videos_schema.get('description'), video_max_videos_schema
    assert video_max_videos_schema.get('examples'), video_max_videos_schema
    video_max_frames_schema = video_tool.parameters['properties']['max_frames']
    assert _schema_types(video_max_frames_schema) == {'integer'}, video_max_frames_schema
    assert video_max_frames_schema.get('description'), video_max_frames_schema
    assert video_max_frames_schema.get('examples'), video_max_frames_schema

    action_tool = tools['start_training']
    assert action_tool.annotations is not None, action_tool
    assert action_tool.annotations.readOnlyHint is False, action_tool.annotations
    assert action_tool.annotations.destructiveHint is False, action_tool.annotations
    assert action_tool.output_schema is not None, action_tool.output_schema
    model_schema = action_tool.parameters['properties']['model']
    assert _schema_types(model_schema) == {'string'}, model_schema
    assert model_schema.get('description'), model_schema
    assert model_schema.get('examples'), model_schema
    data_yaml_schema = action_tool.parameters['properties']['data_yaml']
    assert _schema_types(data_yaml_schema) == {'string'}, data_yaml_schema
    assert data_yaml_schema.get('description'), data_yaml_schema
    assert data_yaml_schema.get('examples'), data_yaml_schema
    epochs_schema = action_tool.parameters['properties']['epochs']
    assert _schema_types(epochs_schema) == {'integer'}, epochs_schema
    assert epochs_schema.get('description'), epochs_schema
    assert epochs_schema.get('examples'), epochs_schema
    training_environment_schema = action_tool.parameters['properties']['training_environment']
    assert _schema_types(training_environment_schema) == {'string'}, training_environment_schema
    assert training_environment_schema.get('description'), training_environment_schema
    assert training_environment_schema.get('examples'), training_environment_schema
    classes_schema = action_tool.parameters['properties']['classes']
    class_types = {item.get('type') for item in classes_schema.get('anyOf', [])}
    assert class_types == {'array', 'string', 'null'}, classes_schema
    assert classes_schema.get('description'), classes_schema
    assert classes_schema.get('examples'), classes_schema

    preflight_tool = tools['training_preflight']
    preflight_classes_schema = preflight_tool.parameters['properties']['classes']
    preflight_class_types = {item.get('type') for item in preflight_classes_schema.get('anyOf', [])}
    assert preflight_class_types == {'array', 'string', 'null'}, preflight_classes_schema
    assert preflight_classes_schema.get('description'), preflight_classes_schema
    assert preflight_classes_schema.get('examples'), preflight_classes_schema

    list_runs_tool = tools['list_training_runs']
    run_state_schema = list_runs_tool.parameters['properties']['run_state']
    run_state_types = _schema_types(run_state_schema)
    assert run_state_types == {'string'}, run_state_schema
    assert run_state_schema.get('description'), run_state_schema
    assert run_state_schema.get('examples'), run_state_schema
    model_keyword_schema = list_runs_tool.parameters['properties']['model_keyword']
    assert _schema_types(model_keyword_schema) == {'string'}, model_keyword_schema
    assert model_keyword_schema.get('description'), model_keyword_schema
    assert model_keyword_schema.get('examples'), model_keyword_schema

    knowledge_tool = tools['analyze_training_outcome']
    metrics_schema = knowledge_tool.parameters['properties']['metrics']
    metrics_types = {item.get('type') for item in metrics_schema.get('anyOf', [])}
    assert metrics_types == {'object', 'null'}, metrics_schema
    assert metrics_schema.get('description'), metrics_schema
    assert metrics_schema.get('examples'), metrics_schema

    retrieval_tool = tools['retrieve_training_knowledge']
    signals_schema = retrieval_tool.parameters['properties']['signals']
    signal_types = {item.get('type') for item in signals_schema.get('anyOf', [])}
    assert signal_types == {'array', 'null'}, signals_schema
    assert signals_schema.get('examples'), signals_schema

    loop_tool = tools['start_training_loop']
    allowed_schema = loop_tool.parameters['properties']['allowed_tuning_params']
    allowed_types = {item.get('type') for item in allowed_schema.get('anyOf', [])}
    assert allowed_types == {'array', 'null'}, allowed_schema
    enum_values = set()
    for item in allowed_schema.get('anyOf', []):
        if item.get('type') != 'array':
            continue
        enum_values.update(item.get('items', {}).get('enum') or [])
    assert enum_values == {'lr0', 'batch', 'imgsz', 'epochs', 'optimizer'}, allowed_schema
    assert allowed_schema.get('examples'), allowed_schema

    convert_tool = tools['convert_format']
    convert_classes_schema = convert_tool.parameters['properties']['classes']
    convert_class_types = {item.get('type') for item in convert_classes_schema.get('anyOf', [])}
    assert convert_class_types == {'array', 'null'}, convert_classes_schema
    assert convert_classes_schema.get('description'), convert_classes_schema
    assert convert_classes_schema.get('examples'), convert_classes_schema

    yaml_tool = tools['generate_yaml']
    yaml_train_schema = yaml_tool.parameters['properties']['train_path']
    assert _schema_types(yaml_train_schema) == {'string'}, yaml_train_schema
    assert yaml_train_schema.get('description'), yaml_train_schema
    assert yaml_train_schema.get('examples'), yaml_train_schema
    yaml_val_schema = yaml_tool.parameters['properties']['val_path']
    assert _schema_types(yaml_val_schema) == {'string'}, yaml_val_schema
    assert yaml_val_schema.get('description'), yaml_val_schema
    assert yaml_val_schema.get('examples'), yaml_val_schema
    yaml_classes_schema = yaml_tool.parameters['properties']['classes']
    yaml_class_types = {item.get('type') for item in yaml_classes_schema.get('anyOf', [])}
    assert yaml_class_types == {'array', 'null'}, yaml_classes_schema
    assert yaml_classes_schema.get('description'), yaml_classes_schema
    assert yaml_classes_schema.get('examples'), yaml_classes_schema

    print('mcp server metadata ok')


if __name__ == '__main__':
    main()

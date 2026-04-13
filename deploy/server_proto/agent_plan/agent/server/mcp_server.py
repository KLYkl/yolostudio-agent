from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from yolostudio_agent.agent.server.tools.combo_tools import prepare_dataset_for_training
from yolostudio_agent.agent.server.tools.data_tools import (
    augment_dataset,
    categorize_by_class,
    clean_orphan_labels,
    convert_format,
    detect_duplicate_images,
    generate_empty_labels,
    generate_yaml,
    generate_missing_labels,
    modify_labels,
    preview_categorize_by_class,
    preview_convert_format,
    preview_generate_empty_labels,
    preview_generate_missing_labels,
    preview_modify_labels,
    run_dataset_health_check,
    scan_dataset,
    split_dataset,
    training_readiness,
    validate_dataset,
)
from yolostudio_agent.agent.server.tools.extract_tools import (
    extract_images,
    extract_video_frames,
    preview_extract_images,
    scan_videos,
)
from yolostudio_agent.agent.server.tools.knowledge_tools import (
    analyze_training_outcome,
    recommend_next_training_step,
    retrieve_training_knowledge,
)
from yolostudio_agent.agent.server.tools.predict_tools import (
    check_realtime_prediction_status,
    export_prediction_path_lists,
    export_prediction_report,
    inspect_prediction_outputs,
    organize_prediction_results,
    predict_images,
    predict_videos,
    scan_cameras,
    scan_screens,
    start_camera_prediction,
    start_rtsp_prediction,
    start_screen_prediction,
    stop_realtime_prediction,
    summarize_prediction_results,
    test_rtsp_stream,
)
from yolostudio_agent.agent.server.tools.train_tools import (
    check_gpu_status,
    check_training_status,
    compare_training_runs,
    inspect_training_run,
    list_training_environments,
    list_training_runs,
    select_best_training_run,
    start_training,
    stop_training,
    summarize_training_run,
    training_preflight,
)

mcp = FastMCP("yolostudio", host="127.0.0.1", port=8080)

mcp.tool()(scan_dataset)
mcp.tool()(split_dataset)
mcp.tool()(validate_dataset)
mcp.tool()(run_dataset_health_check)
mcp.tool()(detect_duplicate_images)
mcp.tool()(augment_dataset)
mcp.tool()(preview_convert_format)
mcp.tool()(convert_format)
mcp.tool()(preview_modify_labels)
mcp.tool()(modify_labels)
mcp.tool()(clean_orphan_labels)
mcp.tool()(preview_generate_empty_labels)
mcp.tool()(generate_empty_labels)
mcp.tool()(preview_generate_missing_labels)
mcp.tool()(generate_missing_labels)
mcp.tool()(preview_categorize_by_class)
mcp.tool()(categorize_by_class)
mcp.tool()(generate_yaml)
mcp.tool()(training_readiness)
mcp.tool()(prepare_dataset_for_training)
mcp.tool()(preview_extract_images)
mcp.tool()(extract_images)
mcp.tool()(scan_videos)
mcp.tool()(extract_video_frames)
mcp.tool()(retrieve_training_knowledge)
mcp.tool()(analyze_training_outcome)
mcp.tool()(recommend_next_training_step)
mcp.tool()(predict_images)
mcp.tool()(predict_videos)
mcp.tool()(summarize_prediction_results)
mcp.tool()(inspect_prediction_outputs)
mcp.tool()(export_prediction_report)
mcp.tool()(export_prediction_path_lists)
mcp.tool()(organize_prediction_results)
mcp.tool()(scan_cameras)
mcp.tool()(scan_screens)
mcp.tool()(test_rtsp_stream)
mcp.tool()(start_camera_prediction)
mcp.tool()(start_rtsp_prediction)
mcp.tool()(start_screen_prediction)
mcp.tool()(check_realtime_prediction_status)
mcp.tool()(stop_realtime_prediction)
mcp.tool()(list_training_environments)
mcp.tool()(training_preflight)
mcp.tool()(list_training_runs)
mcp.tool()(inspect_training_run)
mcp.tool()(compare_training_runs)
mcp.tool()(select_best_training_run)
mcp.tool()(start_training)
mcp.tool()(check_training_status)
mcp.tool()(summarize_training_run)
mcp.tool()(stop_training)
mcp.tool()(check_gpu_status)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

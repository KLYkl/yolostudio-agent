from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from yolostudio_agent.agent.client.llm_factory import LlmProviderSettings, build_llm, resolve_llm_settings
from yolostudio_agent.agent.server.tools.combo_tools import prepare_dataset_for_training
from yolostudio_agent.agent.server.tools.data_tools import (
    augment_dataset,
    categorize_by_class,
    clean_orphan_labels,
    convert_format,
    detect_duplicate_images,
    dataset_training_readiness,
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
from yolostudio_agent.agent.server.tools.training_loop_tools import (
    check_training_loop_status,
    configure_loop_planner_llm,
    inspect_training_loop,
    list_training_loops,
    pause_training_loop,
    resume_training_loop,
    start_training_loop,
    stop_training_loop,
)

mcp = FastMCP("yolostudio", host="127.0.0.1", port=8080)


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, '') or '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _has_explicit_loop_llm_hint() -> bool:
    return any(
        str(os.getenv(name, '') or '').strip()
        for name in (
            'YOLOSTUDIO_LOOP_LLM_PROVIDER',
            'YOLOSTUDIO_LOOP_LLM_MODEL',
            'YOLOSTUDIO_LOOP_LLM_BASE_URL',
            'YOLOSTUDIO_LOOP_LLM_API_KEY',
        )
    )


def _configure_host_side_loop_planner() -> None:
    """
    host-side planner wiring:
    - 默认不强制启用（避免引入新依赖/新失败面）
    - 仅在显式启用或显式提供 loop llm 配置时注入
    """
    if not (_env_flag('YOLOSTUDIO_ENABLE_LOOP_PLANNER') or _has_explicit_loop_llm_hint()):
        return
    try:
        loop_settings = resolve_llm_settings(LlmProviderSettings(role='loop'), role='loop')
        loop_llm = build_llm(loop_settings, role='loop')
    except Exception:
        return
    configure_loop_planner_llm(loop_llm)

def _annotations(
    *,
    read_only: bool,
    destructive: bool,
    idempotent: bool | None,
    open_world: bool,
) -> ToolAnnotations:
    return ToolAnnotations(
        readOnlyHint=read_only,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=open_world,
    )


def _register_tool(
    fn,
    *,
    annotations: ToolAnnotations | None = None,
    structured_output: bool | None = None,
) -> None:
    mcp.tool(annotations=annotations, structured_output=structured_output)(fn)


def _register_read_tool(fn, *, structured_output: bool = True, open_world: bool = True) -> None:
    _register_tool(
        fn,
        annotations=_annotations(
            read_only=True,
            destructive=False,
            idempotent=True,
            open_world=open_world,
        ),
        structured_output=structured_output,
    )


def _register_action_tool(
    fn,
    *,
    destructive: bool = False,
    idempotent: bool | None = False,
    structured_output: bool = True,
    open_world: bool = True,
) -> None:
    _register_tool(
        fn,
        annotations=_annotations(
            read_only=False,
            destructive=destructive,
            idempotent=idempotent,
            open_world=open_world,
        ),
        structured_output=structured_output,
    )


_register_read_tool(scan_dataset)
_register_action_tool(split_dataset, destructive=True)
_register_read_tool(validate_dataset)
_register_read_tool(run_dataset_health_check)
_register_read_tool(detect_duplicate_images)
_register_action_tool(augment_dataset, destructive=True)
_register_read_tool(preview_convert_format)
_register_action_tool(convert_format, destructive=True)
_register_read_tool(preview_modify_labels)
_register_action_tool(modify_labels, destructive=True)
_register_action_tool(clean_orphan_labels, destructive=True)
_register_read_tool(preview_generate_empty_labels)
_register_action_tool(generate_empty_labels, destructive=True)
_register_read_tool(preview_generate_missing_labels)
_register_action_tool(generate_missing_labels, destructive=True)
_register_read_tool(preview_categorize_by_class)
_register_action_tool(categorize_by_class, destructive=True)
_register_action_tool(generate_yaml, destructive=True)
_register_read_tool(dataset_training_readiness)
_register_read_tool(training_readiness)
_register_action_tool(prepare_dataset_for_training, destructive=True)
_register_read_tool(preview_extract_images)
_register_action_tool(extract_images, destructive=True)
_register_read_tool(scan_videos)
_register_action_tool(extract_video_frames, destructive=True)
_register_read_tool(retrieve_training_knowledge, open_world=False)
_register_read_tool(analyze_training_outcome, open_world=False)
_register_read_tool(recommend_next_training_step, open_world=False)
_register_action_tool(predict_images, destructive=False)
_register_action_tool(predict_videos, destructive=False)
_register_read_tool(summarize_prediction_results)
_register_read_tool(inspect_prediction_outputs)
_register_read_tool(export_prediction_report)
_register_read_tool(export_prediction_path_lists)
_register_action_tool(organize_prediction_results, destructive=True)
_register_read_tool(scan_cameras)
_register_read_tool(scan_screens)
_register_read_tool(test_rtsp_stream)
_register_action_tool(start_camera_prediction, destructive=False)
_register_action_tool(start_rtsp_prediction, destructive=False)
_register_action_tool(start_screen_prediction, destructive=False)
_register_read_tool(check_realtime_prediction_status)
_register_action_tool(stop_realtime_prediction, destructive=True, idempotent=True)
_register_read_tool(list_training_environments)
_register_read_tool(training_preflight)
_register_read_tool(list_training_runs)
_register_read_tool(inspect_training_run)
_register_read_tool(compare_training_runs)
_register_read_tool(select_best_training_run)
_register_action_tool(start_training, destructive=False)
_register_read_tool(check_training_status)
_register_read_tool(summarize_training_run)
_register_action_tool(stop_training, destructive=True, idempotent=True)
_register_read_tool(check_gpu_status)
_register_action_tool(start_training_loop, destructive=False)
_register_read_tool(list_training_loops)
_register_read_tool(check_training_loop_status)
_register_read_tool(inspect_training_loop)
_register_action_tool(pause_training_loop, destructive=False, idempotent=True)
_register_action_tool(resume_training_loop, destructive=False, idempotent=True)
_register_action_tool(stop_training_loop, destructive=True, idempotent=True)

_configure_host_side_loop_planner()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

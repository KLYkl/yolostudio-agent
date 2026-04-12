from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from agent_plan.agent.server.tools.combo_tools import prepare_dataset_for_training
from agent_plan.agent.server.tools.data_tools import (
    augment_dataset,
    detect_duplicate_images,
    generate_yaml,
    run_dataset_health_check,
    scan_dataset,
    split_dataset,
    training_readiness,
    validate_dataset,
)
from agent_plan.agent.server.tools.extract_tools import (
    extract_images,
    extract_video_frames,
    preview_extract_images,
    scan_videos,
)
from agent_plan.agent.server.tools.knowledge_tools import (
    analyze_training_outcome,
    recommend_next_training_step,
    retrieve_training_knowledge,
)
from agent_plan.agent.server.tools.predict_tools import predict_images, predict_videos, summarize_prediction_results
from agent_plan.agent.server.tools.train_tools import (
    check_gpu_status,
    check_training_status,
    start_training,
    stop_training,
)

mcp = FastMCP("yolostudio", host="127.0.0.1", port=8080)

mcp.tool()(scan_dataset)
mcp.tool()(split_dataset)
mcp.tool()(validate_dataset)
mcp.tool()(run_dataset_health_check)
mcp.tool()(detect_duplicate_images)
mcp.tool()(augment_dataset)
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
mcp.tool()(start_training)
mcp.tool()(check_training_status)
mcp.tool()(stop_training)
mcp.tool()(check_gpu_status)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")



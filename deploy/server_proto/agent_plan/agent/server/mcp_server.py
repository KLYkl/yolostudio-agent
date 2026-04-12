from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from agent_plan.agent.server.tools.data_tools import (
    augment_dataset,
    scan_dataset,
    split_dataset,
    validate_dataset,
)
from agent_plan.agent.server.tools.train_tools import (
    check_training_status,
    start_training,
    stop_training,
)

mcp = FastMCP("yolostudio")

mcp.tool()(scan_dataset)
mcp.tool()(split_dataset)
mcp.tool()(validate_dataset)
mcp.tool()(augment_dataset)
mcp.tool()(start_training)
mcp.tool()(check_training_status)
mcp.tool()(stop_training)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8080)

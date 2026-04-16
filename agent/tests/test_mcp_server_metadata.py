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

    action_tool = tools['start_training']
    assert action_tool.annotations is not None, action_tool
    assert action_tool.annotations.readOnlyHint is False, action_tool.annotations
    assert action_tool.annotations.destructiveHint is False, action_tool.annotations
    assert action_tool.output_schema is not None, action_tool.output_schema

    print('mcp server metadata ok')


if __name__ == '__main__':
    main()

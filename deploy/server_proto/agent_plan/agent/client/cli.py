from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.agent_client import AgentSettings, build_agent_client

for stream_name in ("stdin", "stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

SLASH_COMMANDS = {
    "/help": "显示可用命令",
    "/context": "显示当前结构化上下文",
    "/gpu": "直接查询 GPU 状态（不经过 LLM）",
    "/status": "直接查询训练状态（不经过 LLM）",
    "/session": "显示当前会话元信息",
}


def _pretty_result(result: dict) -> str:
    if not isinstance(result, dict):
        return str(result)
    if result.get('summary'):
        extra = {k: v for k, v in result.items() if k not in {'summary'}}
        return result['summary'] if not extra else result['summary'] + "\n" + json.dumps(extra, ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False, indent=2)


async def handle_slash_command(agent, command: str) -> str | None:
    cmd = command.strip().lower()
    if cmd == "/help":
        return "\n".join(f"  {name}: {desc}" for name, desc in SLASH_COMMANDS.items())
    if cmd == "/context":
        return agent.context_summary()
    if cmd == "/gpu":
        return _pretty_result(await agent.direct_tool("check_gpu_status"))
    if cmd == "/status":
        return _pretty_result(await agent.direct_tool("check_training_status"))
    if cmd == "/session":
        return agent.session_summary()
    return None


async def main() -> None:
    session_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    agent = await build_agent_client(AgentSettings(session_id=session_id))
    print(agent.preview())
    print("输入 exit / quit 退出；输入 /help 查看快捷命令。")

    pending_thread_id: str | None = None
    while True:
        prompt = "Confirm" if pending_thread_id else "You"
        user = input(f"{prompt}: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        if pending_thread_id:
            lowered = user.lower()
            approved = lowered in {"y", "yes"}
            if lowered not in {"y", "yes", "n", "no"}:
                print("Agent: 请输入 y 或 n。")
                continue
            result = await agent.confirm(pending_thread_id, approved=approved)
            pending_thread_id = None
            print(f"Agent: {result['message']}")
            continue

        if user.startswith("/"):
            result = await handle_slash_command(agent, user)
            if result is not None:
                print(f"[System] {result}")
            else:
                print(f"[System] 未知命令: {user}，输入 /help 查看可用命令")
            continue

        result = await agent.chat(user)
        print(f"Agent: {result['message']}")
        if result["status"] == "needs_confirmation":
            pending_thread_id = result["thread_id"]


if __name__ == "__main__":
    asyncio.run(main())

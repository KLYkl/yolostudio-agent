from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.agent_client import AgentSettings, build_agent_client

for stream_name in ("stdin", "stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


async def main() -> None:
    session_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    agent = await build_agent_client(AgentSettings(session_id=session_id))
    print(agent.preview())
    print("输入 exit / quit 退出。")

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

        result = await agent.chat(user)
        print(f"Agent: {result['message']}")
        if result["status"] == "needs_confirmation":
            pending_thread_id = result["thread_id"]


if __name__ == "__main__":
    asyncio.run(main())

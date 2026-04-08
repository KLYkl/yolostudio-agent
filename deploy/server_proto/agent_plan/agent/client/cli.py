from __future__ import annotations

import asyncio

from agent_plan.agent.client.agent_client import build_agent


async def main() -> None:
    agent = await build_agent()
    print("YoloStudio Agent CLI prototype")
    while True:
        user = input("You: ").strip()
        if user in {"exit", "quit"}:
            break
        result = await agent.ainvoke({"messages": [{"role": "user", "content": user}]})
        print(result)


if __name__ == "__main__":
    asyncio.run(main())

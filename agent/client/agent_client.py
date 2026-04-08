from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


async def build_agent():
    client = MultiServerMCPClient(
        {
            "yolostudio": {
                "transport": "streamable-http",
                "url": "http://127.0.0.1:8080/mcp",
            }
        }
    )
    tools = await client.get_tools()
    llm = ChatOllama(model="gemma4:e4b", base_url="http://127.0.0.1:11434")
    return create_react_agent(llm, tools)

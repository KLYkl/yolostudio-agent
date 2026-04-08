"""快速检查 MCP Server 可用工具列表 (v2: 带 session 管理)"""
import httpx
import json

BASE = "http://127.0.0.1:8080/mcp"

def call_mcp(client: httpx.Client, method: str, params: dict | None = None, msg_id: int | None = 1) -> dict | None:
    body = {"jsonrpc": "2.0", "method": method}
    if msg_id is not None:
        body["id"] = msg_id
    if params:
        body["params"] = params
    
    r = client.post(BASE, json=body, headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    })
    
    # 保存 session ID
    if "mcp-session-id" in r.headers:
        client.headers["mcp-session-id"] = r.headers["mcp-session-id"]
    
    # 解析 SSE
    for line in r.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    return None

with httpx.Client(timeout=15) as client:
    # 1. initialize
    init = call_mcp(client, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "check_tools", "version": "0.1"},
    }, msg_id=1)
    print("=== 初始化 ===")
    print(f"Server: {init['result']['serverInfo']['name']} v{init['result']['serverInfo']['version']}")
    
    # 2. notifications/initialized (no id = notification)
    call_mcp(client, "notifications/initialized", msg_id=None)
    
    # 3. tools/list
    tools_resp = call_mcp(client, "tools/list", msg_id=2)
    tools = tools_resp.get("result", {}).get("tools", [])
    print(f"\n=== 可用工具 ({len(tools)} 个) ===\n")
    for i, t in enumerate(tools, 1):
        name = t["name"]
        desc = t.get("description", "").split("\n")[0]  # 第一行
        props = t.get("inputSchema", {}).get("properties", {})
        required = t.get("inputSchema", {}).get("required", [])
        print(f"{i}. 📌 {name}")
        print(f"   描述: {desc}")
        for pname, pinfo in props.items():
            req_mark = " *必填*" if pname in required else ""
            ptype = pinfo.get("type", "any")
            pdefault = f" (默认: {pinfo['default']})" if "default" in pinfo else ""
            pdesc = pinfo.get("description", "")
            print(f"   - {pname}: {ptype}{pdefault}{req_mark}  {pdesc}")
        print()

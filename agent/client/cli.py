from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
import locale
import os

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.agent_client import AgentSettings, build_agent_client
from yolostudio_agent.agent.client.cli_output import (
    normalize_cli_reply as _normalize_cli_reply,
    should_print_final_agent_message as _should_print_final_agent_message,
)

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def _candidate_input_encodings() -> list[str]:
    candidates: list[str] = []
    for raw in (
        getattr(sys.stdin, "encoding", None),
        os.getenv("PYTHONIOENCODING"),
        locale.getpreferredencoding(False),
        "utf-8",
        "utf-8-sig",
        "gb18030",
        "gbk",
        "cp936",
    ):
        if not raw:
            continue
        encoding = str(raw).split(":", 1)[0].strip()
        if encoding and encoding not in candidates:
            candidates.append(encoding)
    return candidates


def _decode_terminal_input(raw: bytes) -> str:
    for encoding in _candidate_input_encodings():
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_user_input(prompt: str) -> str:
    print(f"{prompt}: ", end="", flush=True)
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:
        line = sys.stdin.readline()
        if line == "":
            raise EOFError
        return line.rstrip("\r\n")

    raw = buffer.readline()
    if raw == b"":
        raise EOFError
    return _decode_terminal_input(raw).rstrip("\r\n")


def _parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("session_id", nargs="?", default="default", help="会话 ID")
    parser.add_argument("--confirm", choices=["manual", "auto"], default=None, help="确认模式")
    parser.add_argument(
        "--provider",
        choices=["ollama", "deepseek", "openai_compatible", "openai-compatible", "openai"],
        default=None,
        help="LLM provider",
    )
    parser.add_argument("--model", default=None, help="LLM 模型名")
    parser.add_argument("--base-url", dest="base_url", default=None, help="LLM base URL")
    return parser.parse_args(argv)


def _is_exit_command(user: str) -> bool:
    return str(user or "").strip().lower() in {"exit", "quit"}


async def main() -> None:
    args = _parse_cli_args(sys.argv[1:])
    settings = AgentSettings(session_id=args.session_id)
    if args.confirm:
        settings.confirmation_mode = args.confirm
    if args.provider:
        settings.provider = args.provider
    if args.model:
        settings.model = args.model
    if args.base_url:
        settings.base_url = args.base_url
    agent = await build_agent_client(settings)
    print(agent.preview())
    print("输入 exit / quit 退出。")

    while True:
        try:
            user = _read_user_input("You").strip()
        except EOFError:
            print()
            break
        if not user:
            print("Agent: 请输入内容。")
            continue
        if _is_exit_command(user):
            break

        stream_state = {"token_started": False, "tool_calls": set(), "streamed_text_seen": False, "streamed_text": ""}
        show_tool_calls = os.getenv("YOLOSTUDIO_CLI_SHOW_TOOL_CALLS") == "1"

        async def _handle_stream(event: dict) -> None:
            event_type = str(event.get("type") or "")
            if event_type == "token":
                text = str(event.get("text") or "")
                if not text:
                    return
                if not stream_state["token_started"]:
                    print("Agent: ", end="", flush=True)
                    stream_state["token_started"] = True
                stream_state["streamed_text_seen"] = True
                stream_state["streamed_text"] = f"{stream_state['streamed_text']}{text}"
                print(text, end="", flush=True)
                return
            if event_type == "tool_call":
                tool_name = str(event.get("tool_name") or "").strip()
                if not tool_name or tool_name in stream_state["tool_calls"]:
                    return
                stream_state["tool_calls"].add(tool_name)
                if not show_tool_calls:
                    return
                if stream_state["token_started"]:
                    print()
                    stream_state["token_started"] = False
                print(f"[Agent] 调用工具: {tool_name}", flush=True)

        result = await agent.chat(user, stream_handler=_handle_stream)
        if stream_state["token_started"]:
            print()
        if _should_print_final_agent_message(stream_state, str(result.get('message') or '')):
            print(f"Agent: {result['message']}")


if __name__ == "__main__":
    asyncio.run(main())

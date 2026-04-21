from __future__ import annotations


def normalize_cli_reply(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def should_print_final_agent_message(stream_state: dict[str, object], final_message: str) -> bool:
    if bool(stream_state.get("streamed_text_seen")) is not True:
        return True
    streamed_text = normalize_cli_reply(str(stream_state.get("streamed_text") or ""))
    final_text = normalize_cli_reply(final_message)
    if not streamed_text or not final_text:
        return False
    if streamed_text == final_text:
        return False
    if streamed_text.endswith(final_text) or final_text.endswith(streamed_text):
        return False
    return True

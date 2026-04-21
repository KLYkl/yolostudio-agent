from __future__ import annotations

from yolostudio_agent.agent.client.cli_output import should_print_final_agent_message


def main() -> None:
    should_print = should_print_final_agent_message(
        {
            'streamed_text_seen': False,
            'streamed_text': '',
        },
        '训练已启动',
    )
    assert should_print is True, should_print

    should_print = should_print_final_agent_message(
        {
            'streamed_text_seen': True,
            'streamed_text': '最近有 5 条环训练记录',
        },
        '最近有 5 条环训练记录',
    )
    assert should_print is False, should_print

    should_print = should_print_final_agent_message(
        {
            'streamed_text_seen': True,
            'streamed_text': '我来帮你查看最近的环训练记录。最近有 5 条环训练记录。',
        },
        '最近有 5 条环训练记录。',
    )
    assert should_print is False, should_print

    should_print = should_print_final_agent_message(
        {
            'streamed_text_seen': True,
            'streamed_text': '我先帮你检查现有环训练。',
        },
        '最近有 5 条环训练记录。',
    )
    assert should_print is True, should_print

    print('cli stream dedup ok')


if __name__ == '__main__':
    main()

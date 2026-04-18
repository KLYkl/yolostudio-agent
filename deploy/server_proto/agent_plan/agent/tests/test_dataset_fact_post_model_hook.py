from __future__ import annotations

import sys
import types
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

def _install_fake_test_dependencies() -> None:
    fake_mod = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_mod

    fake_mod = types.ModuleType('langchain_ollama')

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOllama = _FakeChatOllama
    sys.modules['langchain_ollama'] = fake_mod

    core_mod = types.ModuleType('langchain_core')
    messages_mod = types.ModuleType('langchain_core.messages')

    class _BaseMessage:
        def __init__(self, content='', **kwargs):
            self.content = content
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _AIMessage(_BaseMessage):
        pass

    class _HumanMessage(_BaseMessage):
        pass

    class _SystemMessage(_BaseMessage):
        pass

    messages_mod.AIMessage = _AIMessage
    messages_mod.HumanMessage = _HumanMessage
    messages_mod.SystemMessage = _SystemMessage
    core_mod.messages = messages_mod
    sys.modules['langchain_core'] = core_mod
    sys.modules['langchain_core.messages'] = messages_mod


_install_fake_test_dependencies()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from yolostudio_agent.agent.client.middleware.fact_validation_middleware import build_fact_validation_middleware


def main() -> None:
    snapshot = {
        'dataset_root': '/data/demo',
        'img_dir': '/data/demo/images',
        'label_dir': '/data/demo/labels',
        'scan': {
            'classes': ['Epidural', 'Subdural'],
            'class_stats': {'Epidural': 66, 'Subdural': 24},
            'least_class': {'name': 'Subdural', 'count': 24},
            'most_class': {'name': 'Epidural', 'count': 66},
            'missing_label_images': 3,
            'missing_label_ratio': 0.015,
            'total_images': 200,
        },
    }
    state = {
        'messages': [
            SystemMessage(content='system'),
            SystemMessage(content='summary'),
            HumanMessage(content='哪个类别的标注最少？'),
            AIMessage(content='', tool_calls=[{'id': 'tc-1', 'name': 'scan_dataset', 'args': {'img_dir': '/data/demo'}}]),
        ],
        'dataset_fact_context': snapshot,
    }

    hook = build_fact_validation_middleware(
        replace_last_ai_message=lambda messages, replacement: {
            'messages': [*messages[:-1], AIMessage(content=replacement)],
        }
    )
    update = hook(state)
    assert update == {}, update
    print('dataset fact post model hook ok')


if __name__ == '__main__':
    main()

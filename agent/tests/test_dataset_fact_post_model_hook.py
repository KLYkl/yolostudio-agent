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

try:
    import langchain_openai  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_mod

try:
    import langchain_ollama  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_ollama')

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOllama = _FakeChatOllama
    sys.modules['langchain_ollama'] = fake_mod

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from yolostudio_agent.agent.client.agent_client import _dataset_fact_post_model_hook


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

    update = _dataset_fact_post_model_hook(state)
    assert 'messages' in update, update
    messages = update['messages']
    assert getattr(messages[0], 'id', '') == REMOVE_ALL_MESSAGES, messages
    assert isinstance(messages[-1], AIMessage), messages[-1]
    assert 'Subdural' in messages[-1].content, messages[-1].content
    assert '24' in messages[-1].content, messages[-1].content
    print('dataset fact post model hook ok')


if __name__ == '__main__':
    main()

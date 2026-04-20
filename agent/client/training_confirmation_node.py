from typing import Any

import yolostudio_agent.agent.client.training_workflow_graph as _workflow_graph

interrupt = _workflow_graph.interrupt


def training_confirmation_node(state: Any, config: Any = None) -> Any:
    previous_interrupt = _workflow_graph.interrupt
    _workflow_graph.interrupt = interrupt
    try:
        return _workflow_graph.training_confirmation_node(state, config)
    finally:
        _workflow_graph.interrupt = previous_interrupt


def post_prepare_node(state: Any) -> Any:
    return _workflow_graph.post_prepare_node(state)


def answer_training_status_node(state: Any) -> Any:
    return _workflow_graph.answer_training_status_node(state)


__all__ = ['interrupt', 'training_confirmation_node', 'post_prepare_node', 'answer_training_status_node']

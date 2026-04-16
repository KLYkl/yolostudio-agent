from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.state_projectors.common import _knowledge_snapshot


def apply_knowledge_tool_result(
    session_state: SessionState,
    tool_name: str,
    result: dict[str, Any],
    tool_args: dict[str, Any],
) -> None:
    del tool_args
    kn = session_state.active_knowledge
    if tool_name == 'retrieve_training_knowledge' and result.get('ok'):
        kn.last_retrieval = _knowledge_snapshot(
            result,
            overview_key='retrieval_overview',
            extra_keys=('topic', 'stage', 'model_family', 'task_type', 'matched_rule_ids', 'signals'),
        )
    elif tool_name == 'analyze_training_outcome' and result.get('ok'):
        kn.last_analysis = _knowledge_snapshot(
            result,
            overview_key='analysis_overview',
            extra_keys=('assessment', 'matched_rule_ids', 'signals'),
        )
    elif tool_name == 'recommend_next_training_step' and result.get('ok'):
        kn.last_recommendation = _knowledge_snapshot(
            result,
            overview_key='recommendation_overview',
            extra_keys=('recommended_action', 'matched_rule_ids', 'signals'),
        )

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from langchain_core.messages import SystemMessage

from yolostudio_agent.agent.client.session_state import SessionState


StructuredActionClassifier = Callable[..., Awaitable[str]]


async def classify_training_loop_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    loop_id: str,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    facts = {
        'active_loop_id': loop_id or session_state.active_training.active_loop_id or '',
        'last_loop_status_summary': str((session_state.active_training.last_loop_status or {}).get('summary') or '').strip(),
        'last_loop_detail_summary': str((session_state.active_training.last_loop_detail or {}).get('summary') or '').strip(),
        'has_last_loop_status': bool(session_state.active_training.last_loop_status),
        'has_last_loop_detail': bool(session_state.active_training.last_loop_detail),
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的循环训练跟进路由器。'
                '当前同一会话里已经存在活动中的循环训练上下文。'
                '你只负责判断用户这句跟进，应该查看当前环训练状态、查看当前环训练详情、暂停当前环训练、恢复当前环训练、停止当前环训练，还是不属于当前环训练上下文。'
                '如果用户想要更详细的进展、更多训练信息、轮次信息、完整详情、详细状态，返回 inspect。'
                '如果用户只是询问现在怎么样、进度如何、当前情况、现在什么情况、训练情况，返回 status。'
                '像“环训练状态怎么样”“查看训练情况”“现在是什么情况了”这类泛状态追问，默认返回 status；'
                '只有当用户明确要求“详细一点”“训练详情”“完整详情”“更多训练信息”时，才返回 inspect。'
                '如果用户是在说“这一轮结束后停住”“别自动开下一轮”“下一轮先别跑”，返回 pause。'
                '如果用户是在说“继续环训练”“从下一轮开始继续”“恢复自动复训”，返回 resume。'
                '如果用户是在说“停止环训练”“终止环训练”“结束环训练”，返回 stop。'
                '如果用户是在说新的训练任务、准备数据、换数据集、换模型，返回 other。'
                '输出必须是 JSON，对象格式固定为 '
                '{"action":"status|inspect|pause|resume|stop|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'status', 'inspect', 'pause', 'resume', 'stop'},
    )


async def resolve_training_loop_route(
    *,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    wants_predict: bool,
    wants_stop_training: bool,
    explicit_run_ids: list[str] | None,
    classify_training_loop_followup_action_fn: Callable[..., Awaitable[str]],
) -> dict[str, Any]:
    explicit_run_ids = list(explicit_run_ids or [])
    has_training_loop_context = bool(
        session_state.active_training.active_loop_id
        or session_state.active_training.active_loop_name
        or session_state.active_training.active_loop_status
        or session_state.active_training.active_loop_request
        or session_state.active_training.last_loop_status
        or session_state.active_training.last_loop_detail
    )
    loop_id = session_state.active_training.active_loop_id or ''
    mentions_loop = any(
        token in user_text for token in ('环训练', '循环训练', '循环训', '循环跑', '自动复训', '自动续训', '自动下一轮', 'agent环训练')
    ) or any(
        token in normalized_text for token in ('training loop', 'loop training', 'auto retrain', 'auto training loop')
    )
    start_like = any(
        token in user_text for token in ('开', '启动', '开始', '跑', '来一个', '开启', '创建', '循环训', '循环跑', '训一下', '跑几轮', '训几轮', '试几轮')
    )
    loop_status_phrase = any(
        token in user_text for token in ('状态', '进度', '到哪了', '第几轮', '跑到哪了', '现在怎么样', '怎么样', '怎么样了', '咋样', '咋样了', '情况如何')
    ) or any(token in normalized_text for token in ('training loop status', 'loop status'))
    generic_training_status_in_loop = any(
        token in user_text for token in (
            '训练状态', '当前训练状态', '训练进度', '当前进度',
            '查看训练状态', '再次查看训练状态', '看一下训练状态', '再看一下训练状态',
            '训练情况', '查看训练情况', '看看训练情况', '看训练情况',
            '查看当前状态', '当前状态', '再看当前状态', '再次查看当前状态',
            '查看情况', '看情况', '看下情况', '看看情况', '现在情况', '现在什么情况',
        )
    )
    generic_training_detail_in_loop = any(
        token in user_text for token in (
            '查看训练详情', '训练详情', '查看详情', '完整详情', '详细情况', '完整情况', '轮次详情', '轮次对比',
            '训练信息', '详细训练信息',
        )
    )
    explicit_loop_detail = any(
        token in user_text for token in ('查看环训练详情', '环训练详情', '循环训练详情', '查看自动复训详情')
    )

    if wants_predict:
        return {'action': '', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if any(token in user_text for token in ('环训练列表', '最近环训练', '环训练历史', '有哪些环训练', '最近自动复训')) or any(
        token in normalized_text for token in ('list training loops', 'training loop history')
    ):
        return {'action': 'list', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if mentions_loop and start_like:
        return {'action': 'start', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if has_training_loop_context and not explicit_run_ids:
        classified_action = await classify_training_loop_followup_action_fn(
            user_text=user_text,
            normalized_text=normalized_text,
            loop_id=loop_id,
        )
        if classified_action:
            return {'action': classified_action, 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if any(token in user_text for token in ('暂停环训练', '循环训练暂停', '这一轮结束后停住', '别自动开下一轮', '下一轮先别跑')) or any(
        token in normalized_text for token in ('pause training loop', 'pause loop')
    ):
        return {'action': 'pause', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if any(token in user_text for token in ('恢复环训练', '继续环训练', '继续自动复训', '从下一轮开始继续')) or any(
        token in normalized_text for token in ('resume training loop', 'resume loop')
    ):
        return {'action': 'resume', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if any(token in user_text for token in ('停止环训练', '终止环训练', '结束环训练', '马上停掉环训练', '立即终止当前环训练')) or any(
        token in normalized_text for token in ('stop training loop', 'stop loop')
    ):
        return {'action': 'stop', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if has_training_loop_context and wants_stop_training and not any(
        token in user_text for token in ('当前轮', '这一轮', '本轮', '单轮', '只停训练', '不结束环训练')
    ):
        return {'action': 'stop', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if explicit_loop_detail or (
        has_training_loop_context
        and not explicit_run_ids
        and (
            generic_training_detail_in_loop
            or (('详细' in user_text or 'detail' in normalized_text) and ('情况' in user_text or '信息' in user_text or '状态' in user_text or 'training' in normalized_text))
            or (mentions_loop and any(token in user_text for token in ('第几轮', '轮次详情', '轮次对比', '完整详情')))
        )
    ):
        return {'action': 'inspect', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if any(token in user_text for token in ('环训练状态', '循环训练状态', '自动复训状态', '环训练进度', '循环训练进度')) or (
        has_training_loop_context
        and (
            (mentions_loop and loop_status_phrase)
            or generic_training_status_in_loop
            or (loop_status_phrase and not wants_predict)
        )
    ):
        return {'action': 'status', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    return {'action': '', 'loop_id': loop_id, 'has_context': has_training_loop_context}


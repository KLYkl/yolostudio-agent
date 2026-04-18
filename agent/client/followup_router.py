from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.mainline_guard_policy import (
    compute_initial_train_predict_flags,
    suppress_deferred_cross_domain_intents,
)
from yolostudio_agent.agent.client.session_state import SessionState


def _is_training_provenance_request(user_text: str, normalized_text: str) -> bool:
    text = str(user_text or '')
    normalized = str(normalized_text or '')
    return any(token in text for token in (
        '你基于哪次训练说的', '你是基于哪次训练说的', '基于哪次训练', '根据哪次训练', '依据哪次训练',
        '你上次不是说', '你不是说过',
    )) and any(
        token in text or token in normalized
        for token in ('训练', 'run', '最好', '最值得参考', '分析', '结论')
    )


def _is_training_evidence_request(user_text: str) -> bool:
    text = str(user_text or '')
    return any(token in text for token in (
        '依据是什么', '根据什么说的', '为什么这么说', '为什么说数据有问题',
    ))


def resolve_mainline_request_signals(
    *,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
) -> dict[str, Any]:
    training_status_phrase = (
        any(token in user_text for token in (
            '训练状态', '当前训练状态', '训练进度', '当前进度',
            '还在训练吗', '训练还在吗', '刚才训练还在吗', '上次训练还在吗', '还在跑吗',
            '训练到哪了', '训练到第几轮', '跑到第几轮',
            '第几轮了', '到第几轮了', '现在第几轮了',
            '现在状态呢', '训练情况', '查看训练情况', '看看训练情况', '看训练情况',
            '训练停了吗', '停了吗', '训练结束了吗', '结束了没', '跑完了吗', '训练完成了吗',
            '训练失败了吗', '失败了吗', '是不是训练失败了', '是不是失败了', '训练挂了吗',
            '查看训练状态', '再次查看训练状态', '看一下训练状态', '再看一下训练状态',
        ))
        or any(token in normalized_text for token in (
            'training status', 'training progress', 'check status',
            'is training done', 'training finished', 'training failed', 'did training fail'
        ))
    )
    wants_train, wants_predict = compute_initial_train_predict_flags(
        user_text,
        normalized_text,
        training_status_phrase=training_status_phrase,
    )
    no_train = any(token in user_text for token in ('不要训练', '不训练', '只检查', '仅检查', '不要启动'))
    wants_readiness = any(token in user_text for token in ('能不能直接训练', '是否可以直接训练', '可不可以直接训练', '直接训练', '训练前检查', '适合训练吗', '适不适合训练'))
    wants_split = any(token in user_text for token in ('默认划分', '划分比例', '先划分', 'split'))
    wants_train, wants_predict = suppress_deferred_cross_domain_intents(
        user_text,
        normalized_text,
        wants_train=wants_train,
        wants_predict=wants_predict,
    )
    wants_remote_profile_list = any(
        token in user_text
        for token in (
            '远端配置', '服务器配置', '远端 profile', 'remote profile', '可用服务器', '可用节点', '有哪些节点', '有哪些服务器', 'SSH alias'
        )
    ) or any(token in normalized_text for token in ('list remote profiles', 'list remote servers', 'list ssh aliases'))
    wants_remote_upload = (
        any(token in user_text for token in ('上传', '传到服务器', '传到远端', '同步到服务器', '同步到远端', '发到服务器'))
        or any(token in normalized_text for token in ('upload', 'scp', 'sync to server', 'sync to remote'))
    ) and any(
        token in user_text or token in normalized_text
        for token in ('服务器', '远端', '节点', 'server', 'remote')
    )
    wants_remote_prediction_pipeline = wants_remote_upload and wants_predict and not wants_train
    wants_remote_training_pipeline = wants_remote_upload and wants_train
    asks_metric_terms = any(token in normalized_text for token in ('precision', 'recall', 'map', 'loss', 'epoch', 'epochs', 'batch', 'imgsz', 'patience', 'lr')) or any(token in user_text for token in ('精确率', '召回', '损失', '学习率', '轮数', '批大小'))
    wants_stop_training = any(token in user_text for token in (
        '停止训练', '停掉训练', '停一下训练', '先停训练', '先把训练停掉', '停止当前训练', '先停一下', '再停一次', '再停一下',
    )) or any(token in normalized_text for token in ('stop training', 'stop current training'))
    wants_training_revision = any(
        token in normalized_text or token in user_text
        for token in (
            'batch', 'imgsz', 'device', 'epochs', '轮数', '轮', 'optimizer', '优化器',
            'freeze', '冻结', 'lr0', '学习率', 'resume', 'project', 'name',
            'fraction', 'classes', '类别', 'single_cls', '环境', '继续训练', '别停', '不要停',
        )
    )
    training_command_like = any(token in user_text for token in ('开始训练', '启动训练', '训练这个数据', '用这个数据训练', '直接开训', 'start_training'))
    return {
        'training_status_phrase': training_status_phrase,
        'wants_train': wants_train,
        'wants_predict': wants_predict,
        'no_train': no_train,
        'wants_readiness': wants_readiness,
        'wants_split': wants_split,
        'wants_remote_profile_list': wants_remote_profile_list,
        'wants_remote_upload': wants_remote_upload,
        'wants_remote_prediction_pipeline': wants_remote_prediction_pipeline,
        'wants_remote_training_pipeline': wants_remote_training_pipeline,
        'asks_metric_terms': asks_metric_terms,
        'wants_stop_training': wants_stop_training,
        'wants_training_revision': wants_training_revision,
        'training_command_like': training_command_like,
    }




def resolve_training_run_query_signals(
    *,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    has_training_context: bool,
    asks_metric_terms: bool,
    metric_signals: list[str],
    explicit_run_ids: list[str] | None,
) -> dict[str, Any]:
    explicit_run_ids = list(explicit_run_ids or [])
    wants_training_outcome_analysis = (
        any(token in user_text for token in ('训练效果怎么样', '这次训练效果怎么样', '训练结果怎么样', '训练效果如何', '结果更像', '训练效果'))
        or any(token in user_text for token in ('是不是已经收敛了', '已经收敛了吗', '收敛了吗'))
        or (
            has_training_context
            and any(token in user_text for token in ('效果怎么样', '结果怎么样', '效果如何', '结果如何'))
        )
        or (asks_metric_terms and any(token in user_text for token in ('怎么看', '说明什么', '意味着什么', '结果如何')))
    )
    references_prior_statement = any(token in user_text for token in ('你上次不是说', '你不是说过'))
    wants_training_run_compare = any(token in user_text for token in (
        '对比最近两次训练', '比较最近两次训练', '最近两次训练对比',
        '对比两次训练', '比较两次训练', '训练结果对比', '训练记录对比',
        '刚刚那次和上次比哪个好',
    )) or any(token in normalized_text for token in ('compare training runs', 'compare last two runs'))
    wants_best_training_run = any(token in user_text for token in (
        '最近哪次训练最好', '哪次训练最好', '最好的训练记录', '最好的训练结果',
        '最近哪次最值得参考', '哪次最值得参考',
        '最值得参考的训练记录', '最值得参考的训练结果',
    )) or any(token in normalized_text for token in ('best training run', 'best run'))
    if references_prior_statement:
        wants_training_run_compare = False
        wants_best_training_run = False

    wants_training_run_list = any(token in user_text for token in (
        '最近训练有哪些', '最近一次训练', '训练历史', '训练记录'
    )) or any(token in normalized_text for token in ('recent training runs', 'training history', 'list training runs'))
    wants_failed_training_run_list = any(token in user_text for token in ('失败的训练', '失败训练', '失败记录'))
    wants_completed_training_run_list = any(token in user_text for token in ('已完成的训练', '完成的训练', '跑完的训练'))
    wants_stopped_training_run_list = any(token in user_text for token in ('停止的训练', '中断的训练', '停掉的训练'))
    wants_running_training_run_list = any(token in user_text for token in ('运行中的训练', '还在跑的训练', '正在训练的记录'))
    wants_analysis_ready_run_list = any(token in user_text for token in ('可分析的训练', '有完整指标的训练', '值得分析的训练'))

    explicit_run_outcome_phrase = bool(explicit_run_ids) and any(
        token in user_text for token in (
            '效果怎么样', '结果怎么样', '效果如何', '结果如何',
            '怎么看', '说明什么', '意味着什么',
            '是不是已经收敛了', '已经收敛了吗', '收敛了吗',
        )
    )
    explicit_compare_hint = any(token in user_text for token in ('对比', '比较', '哪个好', '哪次更好'))
    repeat_training_run_compare = any(
        token in user_text
        for token in (
            '刚才那个对比再比较一次',
            '刚才那个对比再来一次',
            '把刚才那两次训练再比较一次',
            '把刚才那两次训练重新比较',
            '再比较一次',
            '再对比一次',
            '重新比较一下',
            '重新对比一下',
        )
    )
    comparison_run_ids: list[str] = list(explicit_run_ids)
    if references_prior_statement:
        comparison_run_ids = []
    if repeat_training_run_compare and not comparison_run_ids:
        last_comparison = session_state.active_training.last_run_comparison or {}
        left_run = last_comparison.get('left_run') or {}
        right_run = last_comparison.get('right_run') or {}
        left_run_id = str(left_run.get('run_id') or left_run.get('log_file') or '').strip()
        right_run_id = str(right_run.get('run_id') or right_run.get('log_file') or '').strip()
        if left_run_id:
            comparison_run_ids.append(left_run_id)
        if right_run_id:
            comparison_run_ids.append(right_run_id)
    wants_training_run_compare = wants_training_run_compare or (bool(comparison_run_ids) and (repeat_training_run_compare or explicit_compare_hint))
    wants_training_run_inspect = (not references_prior_statement) and bool(explicit_run_ids) and any(
        token in user_text for token in ('详情', '记录', '具体情况')
    )
    wants_next_step_guidance = any(token in user_text for token in ('下一步', '先补数据还是先调参数', '先补数据', '先调参数', '怎么优化', '如何优化下一步训练', '下一轮怎么做'))
    wants_training_knowledge = bool(metric_signals) or (asks_metric_terms and any(token in user_text for token in ('说明什么', '什么意思', '意味着什么', '怎么看')))
    wants_training_knowledge = wants_training_knowledge or _is_training_provenance_request(user_text, normalized_text) or _is_training_evidence_request(user_text)
    wants_training_outcome_analysis = wants_training_outcome_analysis or explicit_run_outcome_phrase
    return {
        'wants_training_outcome_analysis': wants_training_outcome_analysis,
        'wants_training_run_compare': wants_training_run_compare,
        'wants_best_training_run': wants_best_training_run,
        'wants_training_run_list': wants_training_run_list,
        'wants_failed_training_run_list': wants_failed_training_run_list,
        'wants_completed_training_run_list': wants_completed_training_run_list,
        'wants_stopped_training_run_list': wants_stopped_training_run_list,
        'wants_running_training_run_list': wants_running_training_run_list,
        'wants_analysis_ready_run_list': wants_analysis_ready_run_list,
        'comparison_run_ids': comparison_run_ids,
        'wants_training_run_inspect': wants_training_run_inspect,
        'wants_next_step_guidance': wants_next_step_guidance,
        'wants_training_knowledge': wants_training_knowledge,
    }


async def resolve_training_loop_route(
    *,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    wants_predict: bool,
    wants_stop_training: bool,
    explicit_run_ids: list[str] | None,
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
        return {'action': 'followup', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if mentions_loop and start_like:
        return {'action': 'start', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if any(token in user_text for token in ('暂停环训练', '循环训练暂停', '这一轮结束后停住', '别自动开下一轮', '下一轮先别跑')) or any(
        token in normalized_text for token in ('pause training loop', 'pause loop')
    ):
        return {'action': 'followup', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if any(token in user_text for token in ('恢复环训练', '继续环训练', '继续自动复训', '从下一轮开始继续')) or any(
        token in normalized_text for token in ('resume training loop', 'resume loop')
    ):
        return {'action': 'followup', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if any(token in user_text for token in ('停止环训练', '终止环训练', '结束环训练', '马上停掉环训练', '立即终止当前环训练')) or any(
        token in normalized_text for token in ('stop training loop', 'stop loop')
    ):
        return {'action': 'followup', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if has_training_loop_context and wants_stop_training and not any(
        token in user_text for token in ('当前轮', '这一轮', '本轮', '单轮', '只停训练', '不结束环训练')
    ):
        return {'action': 'followup', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if explicit_loop_detail or (
        has_training_loop_context
        and not explicit_run_ids
        and (
            generic_training_detail_in_loop
            or (('详细' in user_text or 'detail' in normalized_text) and ('情况' in user_text or '信息' in user_text or '状态' in user_text or 'training' in normalized_text))
            or (mentions_loop and any(token in user_text for token in ('第几轮', '轮次详情', '轮次对比', '完整详情')))
        )
    ):
        return {'action': 'followup', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    if any(token in user_text for token in ('环训练状态', '循环训练状态', '自动复训状态', '环训练进度', '循环训练进度')) or (
        has_training_loop_context
        and (
            (mentions_loop and loop_status_phrase)
            or generic_training_status_in_loop
            or (loop_status_phrase and not wants_predict)
        )
    ):
        return {'action': 'followup', 'loop_id': loop_id, 'has_context': has_training_loop_context}
    return {'action': '', 'loop_id': loop_id, 'has_context': has_training_loop_context}

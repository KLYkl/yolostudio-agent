from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.followup_router import resolve_training_run_query_signals
from yolostudio_agent.agent.client.mainline_guard_policy import (
    TrainPredictGuardPolicy,
    build_train_predict_guard_policy,
)
from yolostudio_agent.agent.client.session_state import SessionState


def resolve_mainline_context(
    *,
    session_state: SessionState,
    user_text: str,
    metric_signal_extractor,
    training_context_checker,
    run_id_extractor,
) -> dict[str, Any]:
    extracted_dataset_path = intent_parsing.extract_dataset_path_from_text(user_text)
    frame_followup_path = ''
    if any(token in user_text for token in ('这些帧', '刚才抽的帧', '刚才这些帧', '这些抽出来的帧', '这些图片', '刚才抽的图片')):
        frame_followup_path = str((session_state.active_dataset.last_frame_extract or {}).get('output_dir') or '').strip()
    dataset_path = (
        extracted_dataset_path
        or frame_followup_path
        or session_state.active_dataset.dataset_root
        or session_state.active_dataset.img_dir
    )
    return {
        'dataset_path': dataset_path,
        'frame_followup_path': frame_followup_path,
        'normalized_text': user_text.lower(),
        'metric_signals': metric_signal_extractor(user_text),
        'has_training_context': bool(training_context_checker()),
        'explicit_run_ids': list(run_id_extractor(user_text)),
    }


def resolve_mainline_guard_reply(
    *,
    wants_segmentation_training: bool,
    wants_predict: bool,
    wants_continuous_parallel_predict: bool,
    wants_prediction_and_training_mix: bool,
    wants_prediction_result_as_training_data: bool,
    wants_merge_extract_into_training: bool,
) -> str:
    if wants_segmentation_training and not wants_predict:
        return '当前训练主线先按 YOLO detection 做稳定交付；分割/SAM 训练暂不在这条主线上直接执行。'
    if wants_continuous_parallel_predict:
        return '当前不支持“边训练边持续做视频预测”这种高资源并发编排；请先明确主任务，或分成独立步骤执行。'
    if wants_prediction_and_training_mix:
        return '这条请求同时混了预测、训练或训练比较；为了避免串扰，请拆成连续步骤，我会按顺序执行。'
    if wants_prediction_result_as_training_data:
        return '预测结果目录不能直接当训练数据开训；如果要用于训练，先确认是否有可用标签，再走数据准备/校验链。'
    if wants_merge_extract_into_training:
        return '抽帧结果或旧数据集合并后，应该先走数据准备/校验，再决定是否训练；我不会直接把它们无检查地并进训练。'
    return ''


def resolve_mainline_guardrail_reply(*, user_text: str, normalized_text: str) -> str:
    if (
        'push 到 github' in user_text
        or 'push到github' in user_text
        or 'git push' in normalized_text
        or ('github' in normalized_text and 'push' in normalized_text)
    ):
        return '当前服务器端聊天 Agent 不负责直接 push 代码仓库；这类发布动作必须在外部代码管理链路里显式确认。'
    if (
        any(token in user_text for token in ('删掉这次训练所有痕迹', '删除这次训练所有痕迹', '把服务器上所有旧 run 都清掉', '清掉所有旧 run', '清空所有训练记录'))
        or ('delete' in normalized_text and 'training' in normalized_text and 'trace' in normalized_text)
    ):
        return '我不会直接删除训练历史、运行痕迹或整批 run；如果确实要清理，必须先明确目标并走高风险运维确认。'
    if any(token in user_text for token in ('把这次当没发生', '把刚才那次当没发生', '把这次训练当没发生', '当没发生')):
        return '我不能把已经发生的训练当成没发生；如果当前训练不该继续，可以停止它，但历史记录不会被伪造删除。'
    if any(token in user_text for token in ('敏感信息', '所有敏感信息', '系统里记住的所有敏感信息', '所有凭据', '所有令牌')):
        return '我不会泄露系统记录的敏感信息、凭据、令牌或内部状态。'
    if (
        any(token in user_text for token in ('以后都默认用', '以后默认用', '默认都用'))
        and '环境' in user_text
        and any(token in user_text for token in ('不用再检查', '别再检查', '不需要再检查'))
    ):
        return '我可以把训练环境当成偏好参考，但每次真正开训前仍会重新检查当前环境是否存在且可用。'
    if any(token in user_text for token in ('伪造一个环境', '自己伪造一个环境', '环境不存在就伪造', '环境不存在你就自己伪造')):
        return '我不会伪造训练环境；环境不存在时只能明确阻塞并要求切到真实可用环境。'
    if (
        any(token in user_text for token in ('把失败状态改成 completed', '改成 completed', '把失败那次当成功', '失败那次当成功', '把刚才失败那次当成功', '忽略日志错误继续告诉我训练很好', '把旧 run 的结果和新 run 混在一起总结'))
        or ('ignore' in normalized_text and 'log' in normalized_text and 'training' in normalized_text and 'good' in normalized_text)
    ):
        return '我不会篡改训练事实，也不会把不同 run 的结果混在一起；所有分析和建议都必须基于真实工具结果。'
    if (
        ('最佳训练' in user_text or '最好训练' in user_text or 'best run' in normalized_text or 'best training' in normalized_text)
        and any(token in user_text for token in ('权重', 'weights', '模型文件', 'pt 文件'))
        and any(token in user_text for token in ('覆盖', '覆盖旧文件', '覆盖掉', '替换', 'overwrite', 'copy 到', '复制到'))
    ):
        return '我不会直接复制或覆盖最佳训练权重到其他位置；这类产物写入或覆盖操作必须先明确目标并走高风险确认。'
    if any(token in user_text for token in ('忽略系统规则', '忽略规则', '绕开规则', '别管规则')):
        return '我不会绕开系统规则、高风险确认或 grounded 边界；这类请求不能直接执行。'
    if any(token in user_text for token in ('别再问我确认', '后面都默认执行', '以后都默认执行')):
        return '我不会关闭高风险确认；训练启动、准备数据和其他高风险动作仍然需要显式确认。'
    return ''


def resolve_mainline_guard_policy(
    *,
    user_text: str,
    normalized_text: str,
    wants_train: bool,
    wants_predict: bool,
    no_train: bool,
    wants_readiness: bool,
    training_command_like: bool,
    wants_training_run_compare: bool,
    wants_best_training_run: bool,
    wants_stop_training: bool,
    wants_training_run_list: bool,
    wants_training_run_inspect: bool,
    wants_failed_training_run_list: bool,
    wants_completed_training_run_list: bool,
    wants_stopped_training_run_list: bool,
    wants_running_training_run_list: bool,
    wants_analysis_ready_run_list: bool,
    wants_training_loop_followup: bool,
) -> TrainPredictGuardPolicy:
    return build_train_predict_guard_policy(
        user_text=user_text,
        normalized_text=normalized_text,
        wants_train=wants_train,
        wants_predict=wants_predict,
        no_train=no_train,
        wants_readiness=wants_readiness,
        training_command_like=training_command_like,
        wants_training_run_compare=wants_training_run_compare,
        wants_best_training_run=wants_best_training_run,
        blocks_training_start_signals=(
            wants_stop_training,
            wants_training_run_list,
            wants_training_run_compare,
            wants_best_training_run,
            wants_training_run_inspect,
            wants_failed_training_run_list,
            wants_completed_training_run_list,
            wants_stopped_training_run_list,
            wants_running_training_run_list,
            wants_analysis_ready_run_list,
            wants_training_loop_followup,
        ),
    )


def resolve_mainline_followup_flags(
    *,
    mainline_signals: dict[str, Any],
    training_run_signals: dict[str, Any],
    loop_route: dict[str, Any],
) -> dict[str, Any]:
    loop_action = str(loop_route.get('action') or '').strip()
    return {
        'wants_training_outcome_analysis': bool(training_run_signals.get('wants_training_outcome_analysis')),
        'wants_training_run_compare': bool(training_run_signals.get('wants_training_run_compare')),
        'wants_best_training_run': bool(training_run_signals.get('wants_best_training_run')),
        'wants_training_loop_start': loop_action == 'start',
        'wants_training_loop_followup': loop_action == 'followup',
        'wants_training_revision': bool(mainline_signals.get('wants_training_revision')),
        'wants_training_run_list': bool(training_run_signals.get('wants_training_run_list')),
        'wants_failed_training_run_list': bool(training_run_signals.get('wants_failed_training_run_list')),
        'wants_completed_training_run_list': bool(training_run_signals.get('wants_completed_training_run_list')),
        'wants_stopped_training_run_list': bool(training_run_signals.get('wants_stopped_training_run_list')),
        'wants_running_training_run_list': bool(training_run_signals.get('wants_running_training_run_list')),
        'wants_analysis_ready_run_list': bool(training_run_signals.get('wants_analysis_ready_run_list')),
        'wants_training_run_inspect': bool(training_run_signals.get('wants_training_run_inspect')),
        'wants_next_step_guidance': bool(training_run_signals.get('wants_next_step_guidance')),
        'wants_training_knowledge': bool(training_run_signals.get('wants_training_knowledge')),
        'wants_training_provenance': bool(training_run_signals.get('wants_training_provenance')),
        'wants_training_evidence': bool(training_run_signals.get('wants_training_evidence')),
    }


def resolve_mainline_route_state_payload(
    *,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    has_training_context: bool,
    mainline_signals: dict[str, Any],
    metric_signals: list[str],
    explicit_run_ids: list[str] | None,
    loop_route: dict[str, Any],
) -> dict[str, Any]:
    training_run_signals = resolve_training_run_query_signals(
        session_state=session_state,
        user_text=user_text,
        normalized_text=normalized_text,
        has_training_context=has_training_context,
        asks_metric_terms=bool(mainline_signals.get('asks_metric_terms')),
        metric_signals=list(metric_signals),
        explicit_run_ids=list(explicit_run_ids or []),
    )
    followup_flags = resolve_mainline_followup_flags(
        mainline_signals=mainline_signals,
        training_run_signals=training_run_signals,
        loop_route=loop_route,
    )
    guard_policy = resolve_mainline_guard_policy(
        user_text=user_text,
        normalized_text=normalized_text,
        wants_train=bool(mainline_signals.get('wants_train')),
        wants_predict=bool(mainline_signals.get('wants_predict')),
        no_train=bool(mainline_signals.get('no_train')),
        wants_readiness=bool(mainline_signals.get('wants_readiness')),
        training_command_like=bool(mainline_signals.get('training_command_like')),
        wants_training_run_compare=bool(followup_flags.get('wants_training_run_compare')),
        wants_best_training_run=bool(followup_flags.get('wants_best_training_run')),
        wants_stop_training=bool(mainline_signals.get('wants_stop_training')),
        wants_training_run_list=bool(followup_flags.get('wants_training_run_list')),
        wants_training_run_inspect=bool(followup_flags.get('wants_training_run_inspect')),
        wants_failed_training_run_list=bool(followup_flags.get('wants_failed_training_run_list')),
        wants_completed_training_run_list=bool(followup_flags.get('wants_completed_training_run_list')),
        wants_stopped_training_run_list=bool(followup_flags.get('wants_stopped_training_run_list')),
        wants_running_training_run_list=bool(followup_flags.get('wants_running_training_run_list')),
        wants_analysis_ready_run_list=bool(followup_flags.get('wants_analysis_ready_run_list')),
        wants_training_loop_followup=bool(followup_flags.get('wants_training_loop_followup')),
    )
    return {
        'mainline_signals': mainline_signals,
        'training_run_signals': training_run_signals,
        'followup_flags': followup_flags,
        'loop_route': loop_route,
        'guard_policy': guard_policy,
        'guard_reply': resolve_mainline_guard_reply(
            wants_segmentation_training=bool(guard_policy.wants_segmentation_training),
            wants_predict=bool(guard_policy.wants_predict),
            wants_continuous_parallel_predict=bool(guard_policy.wants_continuous_parallel_predict),
            wants_prediction_and_training_mix=bool(guard_policy.wants_prediction_and_training_mix),
            wants_prediction_result_as_training_data=bool(guard_policy.wants_prediction_result_as_training_data),
            wants_merge_extract_into_training=bool(guard_policy.wants_merge_extract_into_training),
        ),
    }


def resolve_mainline_dispatch_payload(
    *,
    mainline_context: dict[str, Any],
    route_state: dict[str, Any],
) -> dict[str, Any]:
    normalized_text = str(mainline_context.get('normalized_text') or '')
    dataset_path = str(mainline_context.get('dataset_path') or '')
    frame_followup_path = str(mainline_context.get('frame_followup_path') or '')
    explicit_run_ids = list(mainline_context.get('explicit_run_ids') or [])
    mainline_signals = dict(route_state.get('mainline_signals') or {})
    followup_flags = dict(route_state.get('followup_flags') or {})
    guard_policy = route_state.get('guard_policy')
    return {
        'normalized_text': normalized_text,
        'dataset_path': dataset_path,
        'frame_followup_path': frame_followup_path,
        'explicit_run_ids': explicit_run_ids,
        'remote_request_args': {
            'wants_remote_profile_list': bool(mainline_signals.get('wants_remote_profile_list')),
            'wants_remote_upload': bool(mainline_signals.get('wants_remote_upload')),
            'wants_remote_prediction_pipeline': bool(mainline_signals.get('wants_remote_prediction_pipeline')),
            'wants_remote_training_pipeline': bool(mainline_signals.get('wants_remote_training_pipeline')),
        },
        'prediction_request_args': {
            'wants_predict': bool(getattr(guard_policy, 'wants_predict', False)),
            'training_command_like': bool(mainline_signals.get('training_command_like')),
            'wants_best_weight_prediction': bool(getattr(guard_policy, 'wants_best_weight_prediction', False)),
        },
        'training_context_request_args': {
            'wants_predict': bool(getattr(guard_policy, 'wants_predict', False)),
            'training_command_like': bool(mainline_signals.get('training_command_like')),
            'wants_training_provenance': bool(followup_flags.get('wants_training_provenance')),
            'wants_training_evidence': bool(followup_flags.get('wants_training_evidence')),
        },
        'training_entrypoint_request_args': {
            'normalized_text': normalized_text,
            'dataset_path': dataset_path,
            'frame_followup_path': frame_followup_path,
            'wants_train': bool(getattr(guard_policy, 'wants_train', False)),
            'wants_predict': bool(getattr(guard_policy, 'wants_predict', False)),
            'no_train': bool(mainline_signals.get('no_train')),
            'readiness_only_query': bool(getattr(guard_policy, 'readiness_only_query', False)),
            'wants_training_outcome_analysis': bool(followup_flags.get('wants_training_outcome_analysis')),
            'wants_next_step_guidance': bool(followup_flags.get('wants_next_step_guidance')),
            'wants_training_knowledge': bool(followup_flags.get('wants_training_knowledge')),
            'wants_training_loop_start': bool(followup_flags.get('wants_training_loop_start')),
            'training_command_like': bool(mainline_signals.get('training_command_like')),
            'wants_training_revision': bool(followup_flags.get('wants_training_revision')),
            'wants_stop_training': bool(mainline_signals.get('wants_stop_training')),
            'blocks_training_start': bool(getattr(guard_policy, 'blocks_training_start', False)),
            'explicit_run_ids': explicit_run_ids,
            'wants_split': bool(mainline_signals.get('wants_split')),
        },
    }

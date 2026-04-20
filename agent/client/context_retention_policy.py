from __future__ import annotations

from dataclasses import dataclass

from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.session_state import SessionState

PREDICTION_FOLLOWUP_TOKENS = (
    '预测',
    '推理',
    '识别',
    '报告',
    '结果目录',
    '输出目录',
    '产物',
    '导出',
    '清单',
    '命中',
    '空结果',
    '实时预测',
    '整理',
    'report',
    'output',
    'export',
    'realtime',
)

DATASET_FOLLOWUP_TOKENS = (
    '数据集',
    '类别',
    '标注',
    '缺失标签',
    '重复',
    '健康检查',
    '质量',
    'data yaml',
    'yaml',
    'split',
    '划分',
    '样本',
    'images',
    'labels',
    'duplicate',
    'health',
)

TRAINING_FOLLOWUP_TOKENS = (
    '训练',
    'run',
    'epoch',
    '轮',
    '收敛',
    '下一步',
    '优化',
    '状态',
    '进度',
    '历史',
    '对比',
    '比较',
    '最佳',
    '最好',
    'summary',
    'status',
    'details',
)

KNOWLEDGE_FOLLOWUP_TOKENS = (
    '为什么',
    '原因',
    '依据',
    '建议',
    '分析',
    '结论',
    'why',
    'reason',
    'analysis',
    'recommend',
)

REMOTE_FOLLOWUP_TOKENS = (
    '远端',
    '服务器',
    '节点',
    '上传',
    '下载',
    'profile',
    'server',
    'remote',
    'scp',
    'ssh',
)


@dataclass(frozen=True)
class ContextRetentionDecision:
    reuse_history: bool
    reason: str
    preserve_state_context: bool = False


def _mentions_any(text: str, lowered: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text or token in lowered for token in tokens)


def _has_prediction_followup_context(state: SessionState) -> bool:
    prediction = state.active_prediction
    return any(
        (
            prediction.last_result,
            prediction.last_summary,
            prediction.last_inspection,
            prediction.last_export,
            prediction.last_path_lists,
            prediction.last_organized_result,
            prediction.last_realtime_status,
            prediction.last_remote_roundtrip,
        )
    )


def _has_dataset_followup_context(state: SessionState) -> bool:
    dataset = state.active_dataset
    return any(
        (
            dataset.last_scan,
            dataset.last_validate,
            dataset.last_health_check,
            dataset.last_duplicate_check,
            dataset.last_extract_preview,
            dataset.last_extract_result,
            dataset.last_video_scan,
            dataset.last_frame_extract,
        )
    )


def _mentions_cached_dataset_label(state: SessionState, text: str, lowered: str) -> bool:
    dataset = state.active_dataset
    last_scan = dataset.last_scan or {}
    label_names: set[str] = set()
    for item in list(last_scan.get('classes') or []):
        candidate = str(item or '').strip().lower()
        if candidate:
            label_names.add(candidate)
    for item in list(last_scan.get('top_classes') or []):
        if not isinstance(item, dict):
            continue
        candidate = str(item.get('class_name') or item.get('name') or '').strip().lower()
        if candidate:
            label_names.add(candidate)
    for key in ('least_class', 'most_class'):
        payload = last_scan.get(key) or {}
        if not isinstance(payload, dict):
            continue
        candidate = str(payload.get('name') or payload.get('class_name') or '').strip().lower()
        if candidate:
            label_names.add(candidate)
    return any(label and (label in lowered or label in text.lower()) for label in label_names)


def _has_training_followup_context(state: SessionState) -> bool:
    training = state.active_training
    return any(
        (
            training.last_status,
            training.last_summary,
            training.training_run_summary,
            training.last_remote_roundtrip,
            training.last_run_inspection,
            training.last_run_comparison,
            training.best_run_selection,
            training.recent_runs,
            training.last_loop_status,
            training.last_loop_detail,
            training.recent_loops,
        )
    )


def _has_knowledge_followup_context(state: SessionState) -> bool:
    knowledge = state.active_knowledge
    return any(
        (
            knowledge.last_retrieval,
            knowledge.last_analysis,
            knowledge.last_recommendation,
        )
    )


def _has_remote_followup_context(state: SessionState) -> bool:
    remote = state.active_remote_transfer
    return any(
        (
            remote.last_profile_listing,
            remote.last_upload,
            remote.last_download,
        )
    )


def _has_active_workflow_context(state: SessionState) -> bool:
    training = state.active_training
    prediction = state.active_prediction
    return any(
        (
            str(training.workflow_state or '').strip().lower() not in {'', 'idle'},
            str(training.loop_workflow_state or '').strip().lower() not in {'', 'loop_idle'},
            training.running,
            str(training.active_loop_id or '').strip(),
            str(prediction.image_prediction_status or '').strip().lower() in {'queued', 'running', 'stopping'},
            str(prediction.realtime_status or '').strip().lower() in {'starting', 'running', 'stopping'},
        )
    )


def _has_new_task_targets(user_text: str) -> bool:
    text = str(user_text or '')
    return any(
        (
            intent_parsing.extract_all_paths_from_text(text),
            intent_parsing.extract_model_from_text(text),
            intent_parsing.extract_remote_server_from_text(text),
            intent_parsing.extract_remote_root_from_text(text),
            intent_parsing.extract_rtsp_url_from_text(text),
            intent_parsing.extract_realtime_session_id_from_text(text),
        )
    )


def _needs_best_run_prediction_context(state: SessionState, user_text: str) -> bool:
    if not state.active_training.best_run_selection:
        return False
    text = str(user_text or '')
    lowered = text.lower()
    mentions_best_run = _mentions_any(text, lowered, ('最佳训练', '最好权重', 'best run', 'best weight'))
    mentions_prediction = _mentions_any(text, lowered, ('预测', '推理', '识别', '图片', '视频', 'predict', 'infer'))
    return mentions_best_run and mentions_prediction


def _needs_read_only_prediction_followup_without_history(state: SessionState, user_text: str) -> bool:
    if not _has_prediction_followup_context(state):
        return False
    text = str(user_text or '').strip()
    lowered = text.lower()
    if not text:
        return False
    mentions_prediction_target = _mentions_any(
        text,
        lowered,
        (
            '预测',
            '推理',
            '识别',
            '预测输出',
            '预测报告',
            '报告',
            '输出',
            '输出目录',
            '结果目录',
            '产物',
            'prediction',
            'predict',
            'report',
            'output',
        ),
    )
    mentions_read_only = _mentions_any(
        text,
        lowered,
        (
            '查看',
            '详情',
            '详细',
            '记录',
            '信息',
            '状态',
            '结果',
            '情况',
            '怎么看',
            '说明',
            'detail',
            'details',
            'status',
            'explain',
        ),
    )
    mentions_execution = _mentions_any(
        text,
        lowered,
        (
            '开始训练',
            '启动训练',
            '执行',
            '上传',
            '下载',
            '暂停',
            '停止',
            '覆盖',
            '复制',
            'start training',
            'upload',
            'download',
            'pause',
            'stop',
            'overwrite',
            'copy',
        ),
    )
    if not mentions_read_only or mentions_execution:
        return False
    if mentions_prediction_target:
        return False
    return not any(
        (
            _has_dataset_followup_context(state),
            _has_training_followup_context(state),
            _has_knowledge_followup_context(state),
            _has_remote_followup_context(state),
        )
    )


def _needs_read_only_training_followup_without_history(state: SessionState, user_text: str) -> bool:
    training = state.active_training
    if not (training.best_run_selection or training.last_run_inspection):
        return False
    text = str(user_text or '').strip()
    lowered = text.lower()
    if not text:
        return False
    mentions_target = _mentions_any(
        text,
        lowered,
        (
            '最佳训练',
            '最好权重',
            '训练详情',
            '训练记录',
            '训练结果',
            'best run',
            'best weight',
            'training detail',
            'training details',
            'training result',
        ),
    )
    mentions_read_only = _mentions_any(
        text,
        lowered,
        (
            '查看',
            '详情',
            '详细',
            '记录',
            '信息',
            '状态',
            '结果',
            '怎么看',
            '说明',
            'detail',
            'details',
            'status',
            'explain',
        ),
    )
    mentions_execution = _mentions_any(
        text,
        lowered,
        (
            '预测',
            '推理',
            '识别',
            '开始训练',
            '启动训练',
            '执行',
            '上传',
            '下载',
            '暂停',
            '停止',
            '覆盖',
            '复制',
            'predict',
            'infer',
            'start training',
            'upload',
            'download',
            'pause',
            'stop',
            'overwrite',
            'copy',
        ),
    )
    return mentions_target and mentions_read_only and not mentions_execution


def _cached_domain_followup_reason(state: SessionState, user_text: str) -> str:
    text = str(user_text or '').strip()
    lowered = text.lower()
    if not text:
        return ''
    if _has_prediction_followup_context(state) and _mentions_any(text, lowered, PREDICTION_FOLLOWUP_TOKENS):
        return 'prediction_followup_context'
    if _has_dataset_followup_context(state) and (
        _mentions_any(text, lowered, DATASET_FOLLOWUP_TOKENS)
        or _mentions_cached_dataset_label(state, text, lowered)
    ):
        return 'dataset_followup_context'
    if _has_training_followup_context(state) and _mentions_any(text, lowered, TRAINING_FOLLOWUP_TOKENS):
        return 'training_followup_context'
    if _has_knowledge_followup_context(state) and _mentions_any(text, lowered, KNOWLEDGE_FOLLOWUP_TOKENS):
        return 'knowledge_followup_context'
    if _has_remote_followup_context(state) and _mentions_any(text, lowered, REMOTE_FOLLOWUP_TOKENS):
        return 'remote_followup_context'
    return ''


def build_context_retention_decision(
    *,
    state: SessionState,
    user_text: str,
    explicitly_references_previous_context: bool,
    has_pending_confirmation: bool = False,
    training_plan_context: dict[str, object] | None = None,
) -> ContextRetentionDecision:
    if explicitly_references_previous_context:
        return ContextRetentionDecision(True, 'explicit_reference')
    if has_pending_confirmation:
        return ContextRetentionDecision(True, 'pending_confirmation')
    if dict(training_plan_context or {}):
        return ContextRetentionDecision(True, 'training_plan_context')
    if state.active_training.training_plan_draft:
        return ContextRetentionDecision(True, 'training_plan_draft')
    if _has_active_workflow_context(state):
        return ContextRetentionDecision(True, 'active_workflow')
    if _needs_read_only_prediction_followup_without_history(state, user_text):
        return ContextRetentionDecision(False, 'read_only_prediction_followup', preserve_state_context=True)
    if _needs_read_only_training_followup_without_history(state, user_text):
        return ContextRetentionDecision(False, 'read_only_training_followup', preserve_state_context=True)
    if _needs_best_run_prediction_context(state, user_text):
        return ContextRetentionDecision(True, 'best_run_prediction_followup')
    if not _has_new_task_targets(user_text):
        domain_reason = _cached_domain_followup_reason(state, user_text)
        if domain_reason:
            return ContextRetentionDecision(True, domain_reason)
    return ContextRetentionDecision(False, 'strip_ephemeral_context')

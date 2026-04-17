from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

PREDICTION_FOLLOWUP_ONLY_TOKENS = (
    '预测结果',
    '预测摘要',
    '预测报告',
    '预测输出',
    '总结预测',
    '总结一下预测',
    '分析预测',
    '输出目录',
    '结果目录',
    '路径清单',
    '整理预测结果',
    '整理预测产物',
)


@dataclass(frozen=True)
class TrainPredictGuardPolicy:
    wants_train: bool
    wants_predict: bool
    wants_segmentation_training: bool
    wants_prediction_and_training_mix: bool
    wants_prediction_result_as_training_data: bool
    wants_merge_extract_into_training: bool
    wants_best_weight_prediction: bool
    wants_continuous_parallel_predict: bool
    blocks_training_start: bool
    readiness_only_query: bool


def compute_initial_train_predict_flags(
    user_text: str,
    normalized_text: str,
    *,
    training_status_phrase: bool,
) -> tuple[bool, bool]:
    wants_train = (
        any(token in normalized_text for token in ('train', 'fine-tune', 'fit'))
        or any(token in user_text for token in ('训练', '开训', '重训', '重新训', '训一下', '直接训'))
    ) and not training_status_phrase
    wants_predict = (
        bool(re.search(r'\b(predict|prediction|infer|inference)\b', normalized_text))
        or any(token in user_text for token in ('预测', '推理', '识别'))
    ) and not any(token in user_text for token in PREDICTION_FOLLOWUP_ONLY_TOKENS)
    return wants_train, wants_predict


def suppress_deferred_cross_domain_intents(
    user_text: str,
    normalized_text: str,
    *,
    wants_train: bool,
    wants_predict: bool,
) -> tuple[bool, bool]:
    text = str(user_text or '')
    normalized = str(normalized_text or '')
    if wants_predict and any(
        token in text or token in normalized
        for token in (
            '训练先放着',
            '训练先别动',
            '先别训练',
            '先不训练',
            '训练稍后再说',
            '训练以后再说',
            '训练晚点再说',
            '训练先放一放',
            'train later',
            'skip training for now',
        )
    ):
        wants_train = False
    if wants_train and any(
        token in text or token in normalized
        for token in (
            '预测先放着',
            '预测先别动',
            '先别预测',
            '先不预测',
            '先不要预测',
            '不要预测',
            '预测稍后再说',
            '预测以后再说',
            '预测晚点再说',
            '预测先放一放',
            'predict later',
            'skip prediction for now',
        )
    ):
        wants_predict = False
    return wants_train, wants_predict


def has_contextual_training_mention(user_text: str, normalized_text: str) -> bool:
    text = str(user_text or '')
    normalized = str(normalized_text or '')
    return any(
        token in text or token in normalized
        for token in (
            '训练中',
            '正在训练',
            '当前在训',
            '现在在训',
            '训练还在跑',
            '训练还跑着',
            '当前训练',
            '现在训练',
            '刚才训练',
            '上次训练',
            'running training',
            'training is running',
        )
    )


def has_explicit_extract_or_scan_intent(user_text: str, normalized_text: str) -> bool:
    text = str(user_text or '')
    normalized = str(normalized_text or '')
    return any(
        token in text or token in normalized
        for token in (
            '抽帧',
            '提帧',
            '抽一版',
            '再抽一版',
            '抽图',
            '抽图片',
            '抽样图片',
            '预览抽取',
            '预览一下',
            '扫描视频',
            '视频扫描',
            'scan video',
            'scan videos',
            'extract frame',
            'extract frames',
            'extract image',
            'extract images',
            'preview extract',
        )
    )


def build_train_predict_guard_policy(
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
    blocks_training_start_signals: Iterable[bool],
) -> TrainPredictGuardPolicy:
    text = str(user_text or '')
    normalized = str(normalized_text or '')

    wants_best_weight_prediction = wants_predict and any(
        token in text or token in normalized
        for token in ('最佳训练', '最好权重', 'best run', 'best weight')
    )
    if (
        wants_best_weight_prediction
        and wants_predict
        and not any(
            token in text or token in normalized
            for token in (
                '然后',
                '再',
                '同时',
                '一边',
                '边训练边',
                '继续训练',
                '开训',
                '启动训练',
                '直接训练',
                '训练计划',
            )
        )
    ):
        wants_train = False

    suppress_contextual_train_for_extract = (
        wants_train
        and not training_command_like
        and has_contextual_training_mention(text, normalized)
        and has_explicit_extract_or_scan_intent(text, normalized)
        and not any(
            token in text or token in normalized
            for token in (
                '然后训练',
                '再训练',
                '继续训练',
                '训练计划',
                '开训',
                '直接训练',
                '启动训练',
            )
        )
    )
    if suppress_contextual_train_for_extract:
        wants_train = False

    wants_segmentation_training = wants_train and any(
        token in text or token in normalized
        for token in ('分割', 'segmentation', 'segment', 'sam')
    )
    prediction_only_with_training_exclusion = wants_predict and any(
        token in text
        for token in (
            '不要把训练',
            '别把训练',
            '不要混进训练',
            '训练准备的内容混进来',
            '排除训练',
            '只总结预测结果',
        )
    )
    wants_prediction_and_training_mix = (
        wants_predict
        and not prediction_only_with_training_exclusion
        and (wants_train or wants_training_run_compare or wants_best_training_run)
        and any(token in text for token in ('然后', '再', '同时', '边训练边', '一边'))
    )
    wants_prediction_result_as_training_data = (
        not prediction_only_with_training_exclusion
        and wants_train
        and any(token in text for token in ('预测结果', 'prediction 结果', '预测输出', '推理结果', '识别结果'))
    )
    wants_merge_extract_into_training = any(token in text for token in ('合并', '并到', '合到')) and any(
        token in text for token in ('抽帧', '帧', '旧数据集', '训练')
    )
    wants_continuous_parallel_predict = wants_train and wants_predict and any(
        token in text for token in ('边训练边', '不断做视频预测', '一直预测', '同时不断预测')
    )
    blocks_training_start = any(blocks_training_start_signals)
    readiness_only_query = wants_readiness and (no_train or any(token in text for token in ('吗', '是否', '能不能', '可不可以')))

    return TrainPredictGuardPolicy(
        wants_train=wants_train,
        wants_predict=wants_predict,
        wants_segmentation_training=wants_segmentation_training,
        wants_prediction_and_training_mix=wants_prediction_and_training_mix,
        wants_prediction_result_as_training_data=wants_prediction_result_as_training_data,
        wants_merge_extract_into_training=wants_merge_extract_into_training,
        wants_best_weight_prediction=wants_best_weight_prediction,
        wants_continuous_parallel_predict=wants_continuous_parallel_predict,
        blocks_training_start=blocks_training_start,
        readiness_only_query=readiness_only_query,
    )

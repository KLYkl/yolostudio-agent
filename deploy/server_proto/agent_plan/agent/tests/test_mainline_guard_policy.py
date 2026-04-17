from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.mainline_guard_policy import (
    build_train_predict_guard_policy,
    compute_initial_train_predict_flags,
    suppress_deferred_cross_domain_intents,
)


def test_deferred_cross_domain_predict_suppresses_train() -> None:
    wants_train, wants_predict = compute_initial_train_predict_flags(
        '训练先放着，先帮我预测 /data/images。',
        '训练先放着，先帮我预测 /data/images。'.lower(),
        training_status_phrase=False,
    )
    wants_train, wants_predict = suppress_deferred_cross_domain_intents(
        '训练先放着，先帮我预测 /data/images。',
        '训练先放着，先帮我预测 /data/images。'.lower(),
        wants_train=wants_train,
        wants_predict=wants_predict,
    )
    assert wants_train is False
    assert wants_predict is True


def test_best_weight_prediction_does_not_force_training_plan() -> None:
    policy = build_train_predict_guard_policy(
        user_text='用最佳训练去预测视频 /data/videos。',
        normalized_text='用最佳训练去预测视频 /data/videos。'.lower(),
        wants_train=True,
        wants_predict=True,
        no_train=False,
        wants_readiness=False,
        training_command_like=False,
        wants_training_run_compare=False,
        wants_best_training_run=False,
        blocks_training_start_signals=(),
    )
    assert policy.wants_best_weight_prediction is True
    assert policy.wants_train is False
    assert policy.wants_predict is True


def test_contextual_training_extract_request_stays_out_of_train() -> None:
    policy = build_train_predict_guard_policy(
        user_text='训练中，先把原视频 /data/raw.mp4 再抽一版。',
        normalized_text='训练中，先把原视频 /data/raw.mp4 再抽一版。'.lower(),
        wants_train=True,
        wants_predict=False,
        no_train=False,
        wants_readiness=False,
        training_command_like=False,
        wants_training_run_compare=False,
        wants_best_training_run=False,
        blocks_training_start_signals=(),
    )
    assert policy.wants_train is False
    assert policy.wants_merge_extract_into_training is False


def test_readiness_only_query_does_not_capture_explicit_train_command() -> None:
    knowledge_policy = build_train_predict_guard_policy(
        user_text='这个数据能不能直接训练？',
        normalized_text='这个数据能不能直接训练？'.lower(),
        wants_train=True,
        wants_predict=False,
        no_train=False,
        wants_readiness=True,
        training_command_like=False,
        wants_training_run_compare=False,
        wants_best_training_run=False,
        blocks_training_start_signals=(),
    )
    command_policy = build_train_predict_guard_policy(
        user_text='给视频目录 /data/videos 直接训练。',
        normalized_text='给视频目录 /data/videos 直接训练。'.lower(),
        wants_train=True,
        wants_predict=False,
        no_train=False,
        wants_readiness=True,
        training_command_like=True,
        wants_training_run_compare=False,
        wants_best_training_run=False,
        blocks_training_start_signals=(),
    )
    assert knowledge_policy.readiness_only_query is True
    assert command_policy.readiness_only_query is False

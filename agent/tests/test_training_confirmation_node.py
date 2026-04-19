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

from yolostudio_agent.agent.client.training_confirmation_node import (
    answer_training_status_node,
    post_prepare_node,
    training_confirmation_node,
)
from yolostudio_agent.agent.client.training_schemas import (
    LoopTrainPlan,
    PendingTurnIntent,
    TrainPlan,
    TrainingEdits,
    merge_training_plan_edits,
)
import yolostudio_agent.agent.client.training_confirmation_node as confirmation_mod


def _assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f'{message}: expected {expected!r}, got {actual!r}')


def _scenario_train_plan_edits_update_epochs_batch() -> None:
    plan = TrainPlan(dataset_path='/data/demo', model='yolov8n.pt')
    updated = merge_training_plan_edits(plan, TrainingEdits(batch=12, epochs=20))
    _assert_equal(updated.batch, 12, 'train plan batch')
    _assert_equal(updated.epochs, 20, 'train plan epochs')


def _scenario_loop_plan_edits_map_epochs_to_epochs_per_round() -> None:
    plan = LoopTrainPlan(dataset_path='/data/demo', model='yolov8n.pt', max_rounds=5, epochs_per_round=10, loop_name='ctxloop5')
    updated = merge_training_plan_edits(plan, {'epochs': 8, 'max_rounds': 3})
    _assert_equal(updated.max_rounds, 3, 'loop max_rounds')
    _assert_equal(updated.epochs_per_round, 8, 'loop epochs_per_round')


def _scenario_training_confirmation_approve_prepare() -> None:
    original_interrupt = confirmation_mod.interrupt
    try:
        confirmation_mod.interrupt = lambda payload: PendingTurnIntent(action='approve', reason='go').model_dump()
        command = training_confirmation_node({
            'training_plan': TrainPlan(dataset_path='/data/demo', model='yolov8n.pt').model_dump(),
            'training_phase': 'prepare',
        })
        _assert_equal(getattr(command, 'goto', None), 'execute_prepare', 'approve prepare goto')
    finally:
        confirmation_mod.interrupt = original_interrupt


def _scenario_training_confirmation_status_routes_to_answer_node() -> None:
    original_interrupt = confirmation_mod.interrupt
    try:
        confirmation_mod.interrupt = lambda payload: {'action': 'status', 'reason': 'data.yaml 会生成到 split 目录'}
        command = training_confirmation_node({
            'training_plan': TrainPlan(dataset_path='/data/demo', model='yolov8n.pt').model_dump(),
            'training_phase': 'start',
        })
        _assert_equal(getattr(command, 'goto', None), 'answer_training_status', 'status goto')
        update = dict(getattr(command, 'update', {}) or {})
        _assert_equal(update.get('training_status_reply'), 'data.yaml 会生成到 split 目录', 'status reply payload')
    finally:
        confirmation_mod.interrupt = original_interrupt


def _scenario_training_confirmation_new_task_suspends_plan() -> None:
    original_interrupt = confirmation_mod.interrupt
    try:
        confirmation_mod.interrupt = lambda payload: {'action': 'new_task', 'reason': '最近有哪些环训练'}
        plan = LoopTrainPlan(dataset_path='/data/demo', model='yolov8n.pt', max_rounds=5, epochs_per_round=10, loop_name='ctxloop5')
        command = training_confirmation_node({
            'training_plan': plan.model_dump(),
            'training_phase': 'start',
        })
        _assert_equal(getattr(command, 'goto', None), 'route_new_task', 'new_task goto')
        update = dict(getattr(command, 'update', {}) or {})
        suspended = dict(update.get('suspended_training_plan') or {})
        _assert_equal(dict(suspended.get('plan') or {}).get('mode'), 'loop', 'suspended loop mode')
        _assert_equal(suspended.get('next_step_tool'), '', 'suspended next step defaults empty')
        _assert_equal(update.get('pending_new_task'), '最近有哪些环训练', 'pending_new_task')
    finally:
        confirmation_mod.interrupt = original_interrupt


def _scenario_post_prepare_updates_plan_and_phase() -> None:
    command = post_prepare_node({
        'training_plan': TrainPlan(dataset_path='/data/demo', model='yolov8n.pt').model_dump(),
        'prepare_result': {'resolved_data_yaml': '/data/demo/split/data.yaml', 'summary': '已完成自动划分'},
        'training_readiness': {'summary': '训练预检通过', 'warnings': ['6.1% 标签缺失']},
    })
    _assert_equal(getattr(command, 'goto', None), 'training_confirmation', 'post_prepare goto')
    update = dict(getattr(command, 'update', {}) or {})
    plan = dict(update.get('training_plan') or {})
    _assert_equal(update.get('training_phase'), 'start', 'post_prepare phase')
    _assert_equal(plan.get('data_yaml'), '/data/demo/split/data.yaml', 'post_prepare data_yaml')
    _assert_equal(plan.get('prepare_summary'), '已完成自动划分', 'post_prepare summary')
    _assert_equal(plan.get('warnings'), ['6.1% 标签缺失'], 'post_prepare warnings')


def _scenario_answer_training_status_returns_to_confirmation() -> None:
    command = answer_training_status_node({
        'training_plan': TrainPlan(dataset_path='/data/demo', model='yolov8n.pt').model_dump(),
        'training_phase': 'start',
        'training_status_reply': 'data.yaml 会生成到 split 目录',
    })
    _assert_equal(getattr(command, 'goto', None), 'training_confirmation', 'answer status goto')


def main() -> None:
    _scenario_train_plan_edits_update_epochs_batch()
    _scenario_loop_plan_edits_map_epochs_to_epochs_per_round()
    _scenario_training_confirmation_approve_prepare()
    _scenario_training_confirmation_status_routes_to_answer_node()
    _scenario_training_confirmation_new_task_suspends_plan()
    _scenario_post_prepare_updates_plan_and_phase()
    _scenario_answer_training_status_returns_to_confirmation()
    print('training confirmation node ok')


if __name__ == '__main__':
    main()

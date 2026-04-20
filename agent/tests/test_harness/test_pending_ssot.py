"""
第 1 层 Harness 测试：Pending 单一真相源验证
验证 pending 的 set/get/persist/reload/clear 全生命周期
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from . conftest import (
    DummyGraph, make_client, make_client_at_state,
    setup_harness, cleanup_harness, HARNESS_WORK,
)


# ===== 测试 1: set → get 一致性 =====
async def _test_set_get_consistency() -> None:
    """设置 pending 后，get_pending_action 应立即返回"""
    client = make_client('ssot-set-get')
    client._set_pending_confirmation(
        'ssot-t1',
        {
            'name': 'prepare_dataset_for_training',
            'args': {'dataset_path': '/data/test'},
            'id': None,
            'synthetic': True,
        },
    )
    pending = client.get_pending_action()
    assert pending is not None, 'set 后 get_pending_action 不应为 None'
    assert pending['tool_name'] == 'prepare_dataset_for_training'
    assert pending['tool_args']['dataset_path'] == '/data/test'
    assert pending['decision_state'] == 'pending'


# ===== 测试 2: set → runtime pending shadow =====
async def _test_set_updates_runtime_pending_shadow() -> None:
    """set_pending 后，runtime pending SSOT 应立即更新"""
    client = make_client('ssot-mirror')
    client._set_pending_confirmation(
        'ssot-mirror-t1',
        {
            'name': 'split_dataset',
            'args': {'dataset_path': '/data/ct', 'ratio': '0.8/0.1/0.1'},
            'id': None,
            'synthetic': True,
        },
    )
    pending = client._pending_from_state()
    assert pending is not None, 'set 后 runtime pending 不应为空'
    assert pending['tool_name'] == 'split_dataset', f"期望 split_dataset, 实际 {pending['tool_name']}"
    assert pending['tool_args'].get('dataset_path') == '/data/ct'


# ===== 测试 3: set → save → reload（ephemeral pending 不入 session_state）=====
async def _test_persist_and_reload() -> None:
    """pending 不应再序列化进 session state"""
    scenario_id = 'ssot-persist'
    scenario_root = HARNESS_WORK / scenario_id
    scenario_root.mkdir(parents=True, exist_ok=True)

    # 第一个 client: 设 pending
    c1 = make_client(scenario_id)
    c1._set_pending_confirmation(
        'ssot-persist-t1',
        {
            'name': 'prepare_dataset_for_training',
            'args': {'dataset_path': '/data/persist_test'},
            'id': None,
            'synthetic': True,
        },
    )
    # 验证 c1 有 pending
    p1 = c1.get_pending_action()
    assert p1 is not None, 'c1 设置后应有 pending'

    # 手动持久化 session state
    state_dict = c1.session_state.to_dict()
    assert 'pending_confirmation' not in state_dict, state_dict
    state_file = scenario_root / scenario_id / 'session_state_harness.json'
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state_dict, ensure_ascii=False, indent=2))

    # 第二个 client: 从持久化中恢复
    from yolostudio_agent.agent.client.session_state import SessionState
    loaded_dict = json.loads(state_file.read_text())
    restored_state = SessionState.from_dict(loaded_dict, session_id_fallback=scenario_id)

    # 恢复后的 session_state 不应带 pending mirror
    assert not hasattr(restored_state, 'pending_confirmation')


# ===== 测试 4: 状态预设启动 =====
async def _test_preset_pending_prepare() -> None:
    """make_client_at_state('pending_prepare') 应有正确 pending"""
    client = make_client_at_state('ssot-preset', preset='pending_prepare')
    pending = client.get_pending_action()
    assert pending is not None, '预设 pending 未生效'
    assert pending['tool_name'] == 'prepare_dataset_for_training', f"预设未生效: {pending}"
    assert pending['thread_id'] == 'harness-pending-prepare'


# ===== 测试 5: clear 后 pending 消失 =====
async def _test_clear_pending() -> None:
    """clear 后 pending 应为 None"""
    client = make_client_at_state('ssot-clear', preset='pending_prepare')
    # 先用 _set 让 shadow 也有值
    client._set_pending_confirmation(
        'ssot-clear-t1',
        {
            'name': 'prepare_dataset_for_training',
            'args': {'dataset_path': '/data/clear_test'},
            'id': None,
            'synthetic': True,
        },
    )
    assert client.get_pending_action() is not None

    client._clear_pending_confirmation(thread_id='ssot-clear-t1')
    assert client.get_pending_action() is None, 'clear 后 pending 应为 None'
    assert client._pending_from_state() is None, 'clear 后 runtime pending shadow 应清空'


# ===== 测试 6: confirm reject =====
async def _test_reject_clears_pending() -> None:
    """reject 后 pending 应消失"""
    client = make_client('ssot-reject')
    client._set_pending_confirmation(
        'ssot-reject-t1',
        {
            'name': 'prepare_dataset_for_training',
            'args': {'dataset_path': '/data/reject_test'},
            'id': None,
            'synthetic': True,
        },
    )
    result = await client.confirm('ssot-reject-t1', approved=False)
    assert result['status'] == 'cancelled'
    assert client.get_pending_action() is None


# ===== 入口 =====
async def _run_all() -> None:
    tests = [
        ('set→get 一致性', _test_set_get_consistency),
        ('set→runtime pending shadow', _test_set_updates_runtime_pending_shadow),
        ('set→save→reload', _test_persist_and_reload),
        ('状态预设启动', _test_preset_pending_prepare),
        ('clear 后消失', _test_clear_pending),
        ('reject 清 pending', _test_reject_clears_pending),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            await fn()
            passed += 1
            print(f'  [OK] {name}')
        except Exception as e:
            failed += 1
            print(f'  [FAIL] {name}: {e}')

    print(f'\npending ssot 测试: {passed} 通过, {failed} 失败')
    if failed:
        raise SystemExit(1)


def main() -> None:
    setup_harness()
    try:
        asyncio.run(_run_all())
    finally:
        cleanup_harness()


if __name__ == '__main__':
    main()

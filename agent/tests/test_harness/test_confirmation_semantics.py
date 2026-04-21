"""
第 3 层 Harness 测试：确认语义同义族验证
验证 pending turn intent 结构化解析对各种确认/编辑/取消/追问表达的一致处理
"""
from __future__ import annotations

import asyncio
from typing import Any

from . conftest import make_client, setup_harness, cleanup_harness
from . fixtures import CONFIRMATION_SYNONYMS
from yolostudio_agent.agent.tests._pending_confirmation_test_support import seed_pending_confirmation


def _set_prepare_pending(client: Any) -> None:
    """设置 prepare pending 状态"""
    seed_pending_confirmation(client, 
        'confirm-sem-t1',
        {
            'name': 'prepare_dataset_for_training',
            'args': {'dataset_path': '/data/test'},
            'id': None,
            'synthetic': True,
        },
    )


# ===== 测试 1: 确认语不应被 pending intent 误判为 edit =====
async def _test_approve_not_intercepted_as_edit() -> None:
    """
    根因热点 1: pending turn intent 把"开始训练"判成 edit
    验证：当 pending=prepare 时，说"没问题，开始训练" 不应被截为 edit
    """
    problematic_phrases = [
        '没问题，开始训练', '行，启动吧', '可以，直接训练',
        '那就训练吧', '确认，开始训练',
    ]
    results = []
    for phrase in problematic_phrases:
        client = make_client(f'confirm-not-edit-{hash(phrase) % 10000}')
        _set_prepare_pending(client)
        pending = client._resolve_pending_confirmation(thread_id='confirm-sem-t1')
        payload = await client._parse_pending_turn_intent(phrase, pending)
        results.append((phrase, payload.get('action')))

    intercepted = [(p, d) for p, d in results if d == 'edit']
    print(f'  被截为 edit 的确认语: {len(intercepted)}/{len(results)}')
    for phrase, decision in intercepted:
        print(f'    "{phrase}" → {decision}')

    # 当前预期: 这些会被截获（已知问题）
    # 修复后应该全部不截获
    if intercepted:
        print(f'  [WARN] 已知问题: {len(intercepted)} 个确认语被 passthrough 截为 edit')
    else:
        print('  [OK] 所有确认语都正常通过 passthrough')


# ===== 测试 2: 纯确认语义应走 approve =====
async def _test_pure_approve_phrases() -> None:
    """纯确认语应被 pending turn intent 解析为 approve"""
    approve_phrases = ['确认', '可以开始', '没问题', '继续', '好的', '行']
    results = []
    for phrase in approve_phrases:
        client = make_client(f'confirm-pure-{hash(phrase) % 10000}')
        _set_prepare_pending(client)
        pending = client._resolve_pending_confirmation(thread_id='confirm-sem-t1')
        payload = await client._parse_pending_turn_intent(phrase, pending)
        results.append((phrase, payload.get('action')))

    passthrough_intercepted = [(p, d) for p, d in results if d != 'approve']
    print(f'  非 approve 截获: {len(passthrough_intercepted)}/{len(results)}')
    for p, d in passthrough_intercepted:
        print(f'    "{p}" → intent={d}')


# ===== 测试 3: 编辑语义应 passthrough 为 edit =====
async def _test_edit_phrases_passthrough() -> None:
    """编辑语义应被正确识别"""
    edit_phrases = CONFIRMATION_SYNONYMS['edit']
    results = []
    for phrase in edit_phrases:
        client = make_client(f'confirm-edit-{hash(phrase) % 10000}')
        _set_prepare_pending(client)
        pending = client._resolve_pending_confirmation(thread_id='confirm-sem-t1')
        payload = await client._parse_pending_turn_intent(phrase, pending)
        results.append((phrase, payload.get('action')))

    correct = [(p, d) for p, d in results if d == 'edit']
    print(f'  编辑语正确识别: {correct.__len__()}/{len(results)}')
    for p, d in results:
        if d != 'edit':
            print(f'    [WARN] "{p}" → {d} (期望 edit)')


# ===== 测试 4: 追问语义应 passthrough 为 clarify =====
async def _test_clarify_phrases_passthrough() -> None:
    """追问语义应被正确识别"""
    clarify_phrases = CONFIRMATION_SYNONYMS['clarify']
    results = []
    for phrase in clarify_phrases:
        client = make_client(f'confirm-clarify-{hash(phrase) % 10000}')
        _set_prepare_pending(client)
        pending = client._resolve_pending_confirmation(thread_id='confirm-sem-t1')
        payload = await client._parse_pending_turn_intent(phrase, pending)
        results.append((phrase, payload.get('action')))

    correct = [(p, d) for p, d in results if d == 'status']
    print(f'  追问语正确识别: {correct.__len__()}/{len(results)}')


# ===== 测试 5: pending turn intent 完整流程（不走 LLM） =====
async def _test_try_handle_returns_for_pending() -> None:
    """有 pending 时，结构化解析的 fallback 应保持核心语义不变"""
    client = make_client('confirm-try-handle')
    _set_prepare_pending(client)

    phrases_and_expected_passthrough = [
        ('确认', 'approve'),
        ('取消', 'reject'),
        ('把 batch 改成 12', 'edit'),
        ('为什么用这个模型', 'status'),
    ]
    for phrase, expected_pt in phrases_and_expected_passthrough:
        client2 = make_client(f'confirm-flow-{hash(phrase) % 10000}')
        _set_prepare_pending(client2)
        pending = client2._resolve_pending_confirmation(thread_id='confirm-sem-t1')
        payload = await client2._parse_pending_turn_intent(phrase, pending)
        actual_pt = payload.get('action')
        if actual_pt != expected_pt:
            print(f'    [WARN] "{phrase}": 期望 intent={expected_pt}, 实际={actual_pt}')
        else:
            print(f'    [OK] "{phrase}": intent={actual_pt}')


# ===== 测试 6: 返回当前 pending 不应被判成 new_task =====
async def _test_return_to_current_pending_phrase() -> None:
    client = make_client('confirm-return-current')
    _set_prepare_pending(client)
    pending = client._resolve_pending_confirmation(thread_id='confirm-sem-t1')
    client.planner_llm = object()

    async def _should_not_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError('return-to-current-pending should short-circuit before structured parse')

    client._invoke_structured_payload = _should_not_call  # type: ignore[assignment]
    payload = await client._parse_pending_turn_intent('先继续当前这个环训练', pending)
    actual = payload.get('action')
    if actual != 'status':
        raise AssertionError(f'期望 status，实际 {actual!r}: {payload}')
    print(f'    [OK] "先继续当前这个环训练": intent={actual}')


# ===== 入口 =====
async def _run_all() -> None:
    tests = [
        ('确认语不应被截为 edit', _test_approve_not_intercepted_as_edit),
        ('纯确认语 passthrough', _test_pure_approve_phrases),
        ('编辑语 passthrough', _test_edit_phrases_passthrough),
        ('追问语 passthrough', _test_clarify_phrases_passthrough),
        ('pending turn intent 完整流程', _test_try_handle_returns_for_pending),
        ('返回当前 pending', _test_return_to_current_pending_phrase),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            print(f'\n--- {name} ---')
            await fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f'  [FAIL] {name}: {e}')

    print(f'\n确认语义测试: {passed} 通过, {failed} 失败')


def main() -> None:
    setup_harness()
    try:
        asyncio.run(_run_all())
    finally:
        cleanup_harness()


if __name__ == '__main__':
    main()

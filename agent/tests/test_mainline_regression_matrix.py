from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.agent_client import AgentSettings, YoloStudioAgentClient, build_agent_client

OUT_JSON = Path(r'D:\yolodo2.0\agent_plan\agent\tests\test_mainline_regression_matrix_output.json')
OUT_MD = Path(r'D:\yolodo2.0\agent_plan\doc\mainline_regression_matrix_report_2026-04-11.md')
STANDARD_ROOT = '/home/kly/test_dataset/'
DIRTY_ROOT = '/home/kly/agent_cap_tests/zyb'
NONSTANDARD_ROOT = '/home/kly/agent_cap_tests/nonstandard_dataset'
UNKNOWN_ROOT = '/home/kly/agent_cap_tests/unknown_dataset'
MODEL_PATH = '/home/kly/yolov8n.pt'


def _now_id() -> str:
    return time.strftime('%Y%m%d_%H%M%S')


async def make_agent(provider: str, model: str, session_id: str) -> YoloStudioAgentClient:
    settings = AgentSettings(provider=provider, model=model, session_id=session_id)
    if provider != 'ollama':
        settings.base_url = os.getenv('YOLOSTUDIO_LLM_BASE_URL', settings.base_url)
        settings.api_key = os.getenv('YOLOSTUDIO_LLM_API_KEY', settings.api_key)
    return await build_agent_client(settings)


async def ensure_no_training(agent: YoloStudioAgentClient) -> dict[str, Any]:
    status = await agent.direct_tool('check_training_status')
    if status.get('running'):
        stop = await agent.direct_tool('stop_training')
        return {'status': status, 'stop': stop}
    return {'status': status}


async def run_prompt(agent: YoloStudioAgentClient, prompt: str, decisions: list[bool] | None = None) -> dict[str, Any]:
    decisions = list(decisions or [])
    transcript: list[dict[str, Any]] = []
    result = await agent.chat(prompt)
    transcript.append({'kind': 'chat', 'prompt': prompt, 'result': result})
    idx = 0
    while result.get('status') == 'needs_confirmation' and idx < len(decisions):
        approved = decisions[idx]
        idx += 1
        result = await agent.confirm(result['thread_id'], approved=approved)
        transcript.append({'kind': 'confirm', 'approved': approved, 'result': result})
    return {'result': result, 'transcript': transcript}


async def cleanup_training(agent: YoloStudioAgentClient) -> dict[str, Any]:
    status = await agent.direct_tool('check_training_status')
    payload: dict[str, Any] = {'status': status}
    if status.get('running'):
        payload['stop'] = await agent.direct_tool('stop_training')
    return payload


def recent_tools(agent: YoloStudioAgentClient, limit: int = 40) -> list[str]:
    events = agent.memory.read_events(agent.session_state.session_id, limit=limit)
    tools: list[str] = []
    for event in events:
        if event.get('type') == 'tool_result':
            tool = str(event.get('tool', ''))
            if tool:
                tools.append(tool)
    return tools


def contains_in_order(seq: list[str], expected: list[str]) -> bool:
    if not expected:
        return True
    idx = 0
    for item in seq:
        if item == expected[idx]:
            idx += 1
            if idx == len(expected):
                return True
    return False


def score_bool(*flags: bool) -> dict[str, Any]:
    total = len(flags)
    passed = sum(1 for x in flags if x)
    return {'passed_checks': passed, 'total_checks': total, 'score': round(passed / total, 3) if total else 1.0}


async def case_tool_standard_root_scan(agent: YoloStudioAgentClient) -> dict[str, Any]:
    started = time.time()
    result = await agent.direct_tool('scan_dataset', img_dir=STANDARD_ROOT)
    assessment = score_bool(
        result.get('ok') is True,
        result.get('resolved_img_dir', '').endswith('/images'),
        result.get('resolved_label_dir', '').endswith('/labels'),
        isinstance(result.get('total_images'), int),
    )
    return {
        'id': 'tool_standard_root_scan',
        'kind': 'tool',
        'duration_sec': round(time.time() - started, 2),
        'input': {'img_dir': STANDARD_ROOT},
        'result': result,
        'assessment': assessment,
        'expected': '标准 root 被正确解析到 images/labels，scan 返回结构化统计',
    }


async def case_tool_unknown_fail_fast(agent: YoloStudioAgentClient) -> dict[str, Any]:
    started = time.time()
    result = await agent.direct_tool('prepare_dataset_for_training', dataset_path=UNKNOWN_ROOT)
    assessment = score_bool(
        result.get('ok') is False,
        result.get('blocked_at') == 'resolve_root',
        '未识别出可训练的数据集结构' in str(result.get('error', '')),
    )
    return {
        'id': 'tool_unknown_fail_fast',
        'kind': 'tool',
        'duration_sec': round(time.time() - started, 2),
        'result': result,
        'assessment': assessment,
        'expected': '未知目录在 resolve_root 阶段提前失败，不继续 split/generate_yaml',
    }


async def case_tool_nonstandard_prepare(agent: YoloStudioAgentClient) -> dict[str, Any]:
    started = time.time()
    result = await agent.direct_tool('prepare_dataset_for_training', dataset_path=NONSTANDARD_ROOT)
    assessment = score_bool(
        result.get('ok') is True,
        result.get('ready') is True,
        str(result.get('img_dir', '')).endswith('/pics'),
        str(result.get('label_dir', '')).endswith('/ann'),
    )
    return {
        'id': 'tool_nonstandard_prepare',
        'kind': 'tool',
        'duration_sec': round(time.time() - started, 2),
        'result': result,
        'assessment': assessment,
        'expected': 'pics/ann 非标准目录被解析并准备到可训练状态',
    }


async def case_tool_dirty_health(agent: YoloStudioAgentClient) -> dict[str, Any]:
    started = time.time()
    result = await agent.direct_tool('run_dataset_health_check', dataset_path=DIRTY_ROOT, include_duplicates=True, max_duplicate_groups=3)
    assessment = score_bool(
        result.get('ok') is True,
        result.get('duplicate_groups') == 83,
        result.get('integrity', {}).get('format_mismatch_count') == 5,
        result.get('risk_level') in {'high', 'critical'},
    )
    return {
        'id': 'tool_dirty_health',
        'kind': 'tool',
        'duration_sec': round(time.time() - started, 2),
        'result': result,
        'assessment': assessment,
        'expected': '大数据脏数据集健康检查能返回重复组、格式问题与风险等级',
    }


async def case_tool_dirty_readiness(agent: YoloStudioAgentClient) -> dict[str, Any]:
    started = time.time()
    result = await agent.direct_tool('training_readiness', img_dir=DIRTY_ROOT)
    assessment = score_bool(
        result.get('ok') is True,
        result.get('risk_level') == 'critical',
        any('5179' in str(item) or '73.7%' in str(item) for item in (result.get('warnings') or [])),
        'data_yaml' in ''.join(result.get('blockers') or []),
    )
    return {
        'id': 'tool_dirty_readiness',
        'kind': 'tool',
        'duration_sec': round(time.time() - started, 2),
        'result': result,
        'assessment': assessment,
        'expected': 'readiness 显式暴露 dirty dataset 风险与缺少 data_yaml 的 blocker',
    }


async def case_agent_standard_no_train(provider: str, model: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'matrix-{provider}-no-train-{uuid.uuid4().hex[:8]}')
    await ensure_no_training(agent)
    started = time.time()
    prompt = f'请检查 {STANDARD_ROOT} 是否可以直接训练。不要启动训练，只告诉我结论、原因和下一步建议。'
    run = await run_prompt(agent, prompt)
    tools = recent_tools(agent)
    message = str(run['result'].get('message', ''))
    assessment = score_bool(
        run['result'].get('status') == 'completed',
        'start_training' not in tools,
        any(t in tools for t in ('training_readiness', 'prepare_dataset_for_training', 'scan_dataset')),
        ('训练' in message and ('建议' in message or '原因' in message or '数据质量' in message)),
    )
    return {
        'id': f'agent_{provider}_standard_no_train',
        'kind': 'agent',
        'provider': provider,
        'model': model,
        'duration_sec': round(time.time() - started, 2),
        'transcript': run['transcript'],
        'tools': tools,
        'message': message,
        'assessment': assessment,
        'expected': '遵守只检查不训练约束，并给出 grounded 结论',
    }


async def case_agent_complex_chain(provider: str, model: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'matrix-{provider}-complex-{uuid.uuid4().hex[:8]}')
    await ensure_no_training(agent)
    started = time.time()
    prompt = f'数据在 {STANDARD_ROOT}，按默认划分比例，然后用 yolov8n 模型进行训练。'
    run = await run_prompt(agent, prompt, [True, False])
    tools = recent_tools(agent)
    assessment = score_bool(
        contains_in_order(tools, ['prepare_dataset_for_training']),
        'start_training' in tools or run['transcript'][-1]['result'].get('status') == 'cancelled',
        any(item.get('kind') == 'confirm' for item in run['transcript']),
        run['transcript'][-1]['result'].get('status') in {'cancelled', 'completed', 'needs_confirmation'},
    )
    return {
        'id': f'agent_{provider}_complex_chain',
        'kind': 'agent',
        'provider': provider,
        'model': model,
        'duration_sec': round(time.time() - started, 2),
        'transcript': run['transcript'],
        'tools': tools,
        'assessment': assessment,
        'expected': '复杂训练意图应收敛到 prepare -> start_training 两段式流程',
    }


async def case_agent_dirty_summary(provider: str, model: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'matrix-{provider}-dirty-{uuid.uuid4().hex[:8]}')
    await ensure_no_training(agent)
    started = time.time()
    prompt = f'请分析 {DIRTY_ROOT} 这个数据集的质量问题，并列出最值得注意的 3 个风险。不要修改任何数据。'
    run = await run_prompt(agent, prompt)
    tools = recent_tools(agent)
    message = str(run['result'].get('message', ''))
    assessment = score_bool(
        run['result'].get('status') == 'completed',
        any(t in tools for t in ('scan_dataset', 'validate_dataset', 'run_dataset_health_check', 'training_readiness')),
        ('5179' in message) or ('73.7%' in message) or ('critical' in message) or ('高风险' in message) or ('缺失标签' in message),
        ('Excavator' in message) or ('classes.txt' in message) or ('bulldozer' in message),
    )
    return {
        'id': f'agent_{provider}_dirty_summary',
        'kind': 'agent',
        'provider': provider,
        'model': model,
        'duration_sec': round(time.time() - started, 2),
        'transcript': run['transcript'],
        'tools': tools,
        'message': message,
        'assessment': assessment,
        'expected': '脏数据总结尽量 grounded，能说出缺失标签风险与真实类名/来源',
    }


async def case_agent_health_grounded(provider: str, model: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'matrix-{provider}-health-{uuid.uuid4().hex[:8]}')
    await ensure_no_training(agent)
    started = time.time()
    prompt = f'请检查 {DIRTY_ROOT} 的图片是否有损坏、尺寸异常或重复图片，但不要修改任何数据。'
    run = await run_prompt(agent, prompt)
    tools = recent_tools(agent)
    message = str(run['result'].get('message', ''))
    assessment = score_bool(
        'run_dataset_health_check' in tools,
        ('完整性问题 5' in message) or ('格式不匹配 5' in message) or ('重复组 83' in message),
        ('不会修改' in prompt) and ('删除' not in message),
        '建议' in message,
    )
    return {
        'id': f'agent_{provider}_health_grounded',
        'kind': 'agent',
        'provider': provider,
        'model': model,
        'duration_sec': round(time.time() - started, 2),
        'transcript': run['transcript'],
        'tools': tools,
        'message': message,
        'assessment': assessment,
        'expected': '健康检查应命中专用工具，并以 grounded 方式总结，不编造修改动作',
    }


async def case_agent_duplicate_grounded(provider: str, model: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'matrix-{provider}-dup-{uuid.uuid4().hex[:8]}')
    await ensure_no_training(agent)
    started = time.time()
    prompt = f'只帮我找出 {DIRTY_ROOT} 里重复的图片，并给我两组样例路径，不要删除任何文件。'
    run = await run_prompt(agent, prompt)
    tools = recent_tools(agent)
    message = str(run['result'].get('message', ''))
    assessment = score_bool(
        'detect_duplicate_images' in tools,
        ('83' in message) or ('重复组' in message),
        ('删除' not in message) or ('不要删除' in prompt),
        ('/home/kly/agent_cap_tests/zyb' in message) or ('样例' in message),
    )
    return {
        'id': f'agent_{provider}_duplicate_grounded',
        'kind': 'agent',
        'provider': provider,
        'model': model,
        'duration_sec': round(time.time() - started, 2),
        'transcript': run['transcript'],
        'tools': tools,
        'message': message,
        'assessment': assessment,
        'expected': '重复检测应命中专用工具，并基于样例路径做 grounded 总结',
    }


async def case_agent_state_purity(provider: str, model: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'matrix-{provider}-purity-{uuid.uuid4().hex[:8]}')
    await ensure_no_training(agent)
    started = time.time()
    prompt = '如果现在有训练在跑就停止；如果没有，就只告诉我现在没有训练。'
    run = await run_prompt(agent, prompt)
    state = {
        'running': agent.session_state.active_training.running,
        'model': agent.session_state.active_training.model,
        'data_yaml': agent.session_state.active_training.data_yaml,
        'device': agent.session_state.active_training.device,
        'last_status': agent.session_state.active_training.last_status,
    }
    assessment = score_bool(
        run['result'].get('status') == 'completed',
        state['running'] is False,
        (not state['model']) and (not state['data_yaml']) and (not state['device']),
        isinstance(state['last_status'], dict),
    )
    return {
        'id': f'agent_{provider}_state_purity',
        'kind': 'agent',
        'provider': provider,
        'model': model,
        'duration_sec': round(time.time() - started, 2),
        'transcript': run['transcript'],
        'state': state,
        'assessment': assessment,
        'expected': 'fresh session 仅查状态时，不应被旧训练参数污染',
    }


async def run_case(case_id: str, expected: str, coro) -> dict[str, Any]:
    started = time.time()
    try:
        result = await coro
        if 'duration_sec' not in result:
            result['duration_sec'] = round(time.time() - started, 2)
        return result
    except Exception as exc:
        return {
            'id': case_id,
            'kind': 'runtime',
            'duration_sec': round(time.time() - started, 2),
            'expected': expected,
            'error': f'{type(exc).__name__}: {exc}',
            'assessment': {'passed_checks': 0, 'total_checks': 1, 'score': 0.0},
        }


async def main() -> None:
    run_id = _now_id()
    cases: list[dict[str, Any]] = []

    tool_agent = await make_agent('ollama', 'gemma4:e4b', f'matrix-tool-{run_id}')
    await ensure_no_training(tool_agent)
    cases.append(await run_case('tool_standard_root_scan', '标准 root 被正确解析到 images/labels，scan 返回结构化统计', case_tool_standard_root_scan(tool_agent)))
    cases.append(await run_case('tool_unknown_fail_fast', '未知目录在 resolve_root 阶段提前失败，不继续 split/generate_yaml', case_tool_unknown_fail_fast(tool_agent)))
    cases.append(await run_case('tool_nonstandard_prepare', 'pics/ann 非标准目录被解析并准备到可训练状态', case_tool_nonstandard_prepare(tool_agent)))
    cases.append(await run_case('tool_dirty_health', '大数据脏数据集健康检查能返回重复组、格式问题与风险等级', case_tool_dirty_health(tool_agent)))
    cases.append(await run_case('tool_dirty_readiness', 'readiness 显式暴露 dirty dataset 风险与缺少 data_yaml 的 blocker', case_tool_dirty_readiness(tool_agent)))
    await cleanup_training(tool_agent)

    cases.append(await run_case('agent_ollama_standard_no_train', '遵守只检查不训练约束，并给出 grounded 结论', case_agent_standard_no_train('ollama', 'gemma4:e4b')))
    cases.append(await run_case('agent_ollama_complex_chain', '复杂训练意图应收敛到 prepare -> start_training 两段式流程', case_agent_complex_chain('ollama', 'gemma4:e4b')))
    cases.append(await run_case('agent_ollama_dirty_summary', '脏数据总结尽量 grounded，能说出缺失标签风险与真实类名/来源', case_agent_dirty_summary('ollama', 'gemma4:e4b')))
    cases.append(await run_case('agent_ollama_health_grounded', '健康检查应命中专用工具，并以 grounded 方式总结，不编造修改动作', case_agent_health_grounded('ollama', 'gemma4:e4b')))
    cases.append(await run_case('agent_ollama_duplicate_grounded', '重复检测应命中专用工具，并基于样例路径做 grounded 总结', case_agent_duplicate_grounded('ollama', 'gemma4:e4b')))
    cases.append(await run_case('agent_ollama_state_purity', 'fresh session 仅查状态时，不应被旧训练参数污染', case_agent_state_purity('ollama', 'gemma4:e4b')))

    if os.getenv('DEEPSEEK_API_KEY'):
        cases.append(await run_case('agent_deepseek_standard_no_train', '遵守只检查不训练约束，并给出 grounded 结论', case_agent_standard_no_train('deepseek', 'deepseek-chat')))
        cases.append(await run_case('agent_deepseek_complex_chain', '复杂训练意图应收敛到 prepare -> start_training 两段式流程', case_agent_complex_chain('deepseek', 'deepseek-chat')))
        cases.append(await run_case('agent_deepseek_dirty_summary', '脏数据总结尽量 grounded，能说出缺失标签风险与真实类名/来源', case_agent_dirty_summary('deepseek', 'deepseek-chat')))
        cases.append(await run_case('agent_deepseek_health_grounded', '健康检查应命中专用工具，并以 grounded 方式总结，不编造修改动作', case_agent_health_grounded('deepseek', 'deepseek-chat')))
        cases.append(await run_case('agent_deepseek_duplicate_grounded', '重复检测应命中专用工具，并基于样例路径做 grounded 总结', case_agent_duplicate_grounded('deepseek', 'deepseek-chat')))
        cases.append(await run_case('agent_deepseek_state_purity', 'fresh session 仅查状态时，不应被旧训练参数污染', case_agent_state_purity('deepseek', 'deepseek-chat')))
    else:
        cases.append({'id': 'deepseek_matrix', 'skipped': True, 'reason': 'DEEPSEEK_API_KEY 未配置'})

    total_checks = 0
    passed_checks = 0
    failed_cases: list[str] = []
    for case in cases:
        assessment = case.get('assessment') or {}
        total_checks += int(assessment.get('total_checks', 0))
        passed_checks += int(assessment.get('passed_checks', 0))
        if assessment and assessment.get('passed_checks') != assessment.get('total_checks'):
            failed_cases.append(case['id'])

    payload = {
        'run_id': run_id,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'case_count': len(cases),
        'passed_checks': passed_checks,
        'total_checks': total_checks,
        'score': round(passed_checks / total_checks, 3) if total_checks else 1.0,
        'failed_cases': failed_cases,
        'cases': cases,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# 主线回归矩阵测试报告（2026-04-11）',
        '',
        f'- 运行批次: `{run_id}`',
        f'- Case 数量: `{len(cases)}`',
        f'- 检查项通过: `{passed_checks}/{total_checks}`',
        f'- 总体得分: `{payload["score"]}`',
        '',
        '## 1. 总结',
        '',
        f'- 未满分 case: `{", ".join(failed_cases) if failed_cases else "无"}`',
        '- 说明：本报告把“执行层是否完成”与“解释层是否 grounded”混合纳入同一轮回归，用来观察当前版本的主线能力边界。',
        '',
        '## 2. Case 结果',
        '',
    ]
    for case in cases:
        lines.append(f"### {case['id']}")
        if case.get('skipped'):
            lines.append(f"- 结果: 跳过（{case.get('reason', '')}）")
            lines.append('')
            continue
        assessment = case.get('assessment') or {}
        lines.append(f"- 预期: {case.get('expected', '')}")
        lines.append(f"- 得分: {assessment.get('passed_checks', 0)}/{assessment.get('total_checks', 0)} ({assessment.get('score', 0)})")
        if case.get('provider'):
            lines.append(f"- Provider: `{case.get('provider')}` / `{case.get('model')}`")
        if case.get('tools'):
            lines.append(f"- 实际工具链: `{', '.join(case.get('tools', []))}`")
        if case.get('message'):
            preview = str(case['message']).replace('\n', ' ')[:240]
            lines.append(f"- 回复摘要: {preview}")
        if case.get('state'):
            lines.append(f"- 状态摘要: `{json.dumps(case['state'], ensure_ascii=False)}`")
        lines.append('')

    if failed_cases:
        lines.extend([
            '## 3. 本轮暴露的问题',
            '',
        ])
        for case_id in failed_cases:
            lines.append(f'- `{case_id}`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。')
        lines.append('')
    else:
        lines.extend([
            '## 3. 本轮暴露的问题',
            '',
            '- 本轮矩阵未出现失败项，但仍应继续保留 provider 对照和脏数据回归。',
            '',
        ])

    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'run_id': run_id, 'case_count': len(cases), 'score': payload['score'], 'output_json': str(OUT_JSON), 'output_md': str(OUT_MD)}, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())

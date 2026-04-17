from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.client.tool_adapter import adapt_tools_for_chat_model
from yolostudio_agent.agent.server.tools import data_tools


class _DummyGraph:
    def get_state(self, config):
        return None


class _ReplyGraph:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.invocations: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        self.invocations.append((payload, config))
        messages = list(payload.get('messages') or [])
        messages.append(AIMessage(content=self.reply))
        return {'messages': messages}


WORK = Path(__file__).resolve().parent / '_tmp_dataset_training_readiness'


def _server_tool_contract() -> None:
    original_resolve = data_tools.resolve_dataset_root
    original_scan = data_tools.scan_dataset
    original_validate = data_tools.validate_dataset
    original_inspect_yaml = data_tools._inspect_training_yaml
    try:
        data_tools.resolve_dataset_root = lambda img_dir, label_dir='': {
            'ok': True,
            'dataset_root': '/dataset',
            'structure_type': 'yolo_standard',
            'img_dir': '/dataset/images',
            'label_dir': '/dataset/labels',
            'resolved_from_root': True,
            'resolution_method': 'direct',
            'is_split': False,
            'split_info': {},
            'detected_data_yaml': '',
            'data_yaml_candidates': [],
            'summary': '检测到 YOLO 标准目录结构 (images/ + labels/)',
            'next_actions': [],
        }
        data_tools.scan_dataset = lambda img_dir, label_dir='': {
            'ok': True,
            'summary': '总图片: 100, 已标注: 94, 缺失标签: 6, 空标签: 0, 类别数: 2',
            'dataset_root': '/dataset',
            'structure_type': 'yolo_standard',
            'resolved_img_dir': '/dataset/images',
            'resolved_label_dir': '/dataset/labels',
            'total_images': 100,
            'labeled_images': 94,
            'missing_label_images': 6,
            'missing_label_ratio': 0.06,
            'risk_level': 'medium',
            'warnings': ['发现 6 张图片缺少标签（占比 6.0%），训练结果可能受到影响'],
            'detected_data_yaml': '',
            'detected_classes_txt': '/dataset/classes.txt',
            'class_name_source': 'classes_txt',
            'classes': ['cat', 'dog'],
        }
        data_tools.validate_dataset = lambda img_dir, label_dir='', classes_txt='': {
            'ok': True,
            'summary': '未发现标签格式/坐标问题；发现 6 张图片缺少标签（占比 6.0%），训练结果可能受到影响',
            'has_issues': False,
            'has_risks': True,
            'risk_level': 'medium',
            'warnings': ['发现 6 张图片缺少标签（占比 6.0%），训练结果可能受到影响'],
            'missing_label_images': 6,
            'missing_label_ratio': 0.06,
            'issue_count': 0,
            'issue_breakdown': {'coord_errors': 0, 'class_errors': 0, 'format_errors': 0, 'orphan_labels': 0},
        }
        data_tools._inspect_training_yaml = lambda path: {
            'exists': True,
            'usable': True,
            'yaml_path': path,
            'issues': [],
            'resolved_targets': {'train': '/dataset/images/train', 'val': '/dataset/images/val'},
            'warnings': [],
        }

        readiness = data_tools.dataset_training_readiness('/dataset')
        assert readiness['ok'] is True
        assert readiness['ready'] is False
        assert readiness['needs_data_yaml'] is True
        assert readiness['needs_split'] is True
        assert readiness['dataset_structure'] == 'yolo_standard'
        assert '缺少可用的 data.yaml' in readiness['blockers']
        assert '训练/验证集还没准备好' in readiness['blockers']
        assert readiness['next_step_summary'] == '可以先准备数据，补齐 data.yaml 和划分产物。'
        assert readiness['readiness_overview']['scope'] == 'dataset'
        assert readiness['readiness_overview']['needs_data_yaml'] is True
        assert readiness['action_candidates'][0]['tool'] == 'prepare_dataset_for_training'
        assert 'auto_device' not in readiness
        assert 'available_gpu_indexes' not in readiness
        assert 'device_policy' not in readiness
    finally:
        data_tools.resolve_dataset_root = original_resolve
        data_tools.scan_dataset = original_scan
        data_tools.validate_dataset = original_validate
        data_tools._inspect_training_yaml = original_inspect_yaml


async def _llm_reply_path_smoke() -> None:
    scenario_root = WORK / 'llm_reply_path'
    shutil.rmtree(scenario_root, ignore_errors=True)
    scenario_root.mkdir(parents=True, exist_ok=True)
    try:
        graph = _ReplyGraph('我看了一下，这份数据现在还不能直接训练，主要是还没有可用的 data.yaml，而且训练/验证集也还没准备好。先把数据整理一下就可以继续。')
        settings = AgentSettings(session_id='dataset-training-readiness-llm', memory_root=str(scenario_root))
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})

        async def _unexpected_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError(f'readiness query should not shortcut direct tool anymore: {tool_name} {kwargs}')

        client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
        turn = await client.chat('帮我看一下 /home/kly/ct_loop/data_ct 能不能直接训练')
        assert turn['status'] == 'completed', turn
        assert 'data.yaml' in turn['message'], turn
        assert '训练/验证集' in turn['message'], turn
        assert len(graph.invocations) == 1, graph.invocations
    finally:
        shutil.rmtree(scenario_root, ignore_errors=True)


async def _dataset_quality_llm_reply_path_smoke() -> None:
    scenario_root = WORK / 'dataset_quality_llm_reply_path'
    shutil.rmtree(scenario_root, ignore_errors=True)
    scenario_root.mkdir(parents=True, exist_ok=True)
    try:
        graph = _ReplyGraph('我看了一下，这份数据集目前主要风险是缺少完整校验事实；如果要更稳妥，先继续做质量校验和健康检查。')
        settings = AgentSettings(session_id='dataset-quality-llm', memory_root=str(scenario_root))
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})

        async def _unexpected_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError(f'dataset quality query should not shortcut direct tool anymore: {tool_name} {kwargs}')

        client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
        turn = await client.chat('这个数据集 /home/kly/ct_loop/data_ct 质量怎么样？请先校验再回答。')
        assert turn['status'] == 'completed', turn
        assert '质量' in turn['message'] or '校验' in turn['message'], turn
        assert len(graph.invocations) == 1, graph.invocations
    finally:
        shutil.rmtree(scenario_root, ignore_errors=True)


def _compose_reply_fallback_smoke() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='dataset-training-readiness-fallback', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        readiness = {
            'ok': True,
            'readiness_scope': 'dataset',
            'summary': '从数据集角度看，这份数据还不能直接训练: 缺少可用的 data.yaml, 训练/验证集还没准备好；但可以先准备数据',
            'dataset_root': '/home/kly/ct_loop/data_ct',
            'dataset_structure': 'yolo_standard',
            'is_split': False,
            'needs_split': True,
            'needs_data_yaml': True,
            'resolved_img_dir': '/home/kly/ct_loop/data_ct/images',
            'resolved_label_dir': '/home/kly/ct_loop/data_ct/labels',
            'resolved_data_yaml': '',
            'ready': False,
            'preparable': True,
            'primary_blocker_type': 'missing_yaml',
            'blocker_codes': ['missing_yaml', 'split_required'],
            'warnings': [],
            'blockers': ['缺少可用的 data.yaml', '训练/验证集还没准备好'],
            'next_step_summary': '可以先准备数据，补齐 data.yaml 和划分产物。',
        }
        reply = client._compose_final_reply([], [('dataset_training_readiness', readiness)])
        assert '模型这次没有生成最终回复' in reply, reply
        assert 'dataset_training_readiness' in reply, reply
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


async def _stream_tools_suppression_smoke() -> None:
    scenario_root = WORK / 'stream_tools_suppression'
    shutil.rmtree(scenario_root, ignore_errors=True)
    scenario_root.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='stream-tools-suppression', memory_root=str(scenario_root))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        seen: list[dict[str, Any]] = []

        async def _capture(event: dict[str, Any]) -> None:
            seen.append(dict(event))

        await client._handle_stream_mode_event(
            'messages',
            (
                ToolMessage(
                    content='{"ok": true, "summary": "tool json should not stream as model token"}',
                    tool_call_id='tool-call-1',
                    name='dataset_training_readiness',
                ),
                {'langgraph_node': 'tools'},
            ),
            _capture,
        )
        assert not seen, seen
    finally:
        shutil.rmtree(scenario_root, ignore_errors=True)


async def _tool_none_sanitization_smoke() -> None:
    class _Args(BaseModel):
        img_dir: str = Field(default='')
        label_dir: str | None = Field(default='')
        data_yaml: str | None = Field(default='')

    async def _echo(img_dir: str, label_dir: str = '', data_yaml: str = '') -> dict[str, Any]:
        return {'img_dir': img_dir, 'label_dir': label_dir, 'data_yaml': data_yaml}

    base_tool = StructuredTool.from_function(
        func=None,
        coroutine=_echo,
        name='dataset_training_readiness',
        description='dataset readiness',
        args_schema=_Args,
        return_direct=False,
    )
    adapted = adapt_tools_for_chat_model([base_tool], include_aliases=False)[0]
    raw = await adapted.ainvoke({'img_dir': '/dataset', 'label_dir': None, 'data_yaml': None})
    assert '"label_dir": ""' in raw or '"label_dir":""' in raw, raw
    assert '"data_yaml": ""' in raw or '"data_yaml":""' in raw, raw


async def _tool_result_compaction_smoke() -> None:
    class _Args(BaseModel):
        dataset_path: str = Field(default='')

    async def _echo(dataset_path: str = '') -> dict[str, Any]:
        return {
            'ok': True,
            'summary': '这份数据还不能直接训练，但可以先准备数据',
            'blockers': ['缺少可用的 data.yaml', '训练/验证集还没准备好'],
            'next_step_summary': '先补齐 data.yaml 并划分训练/验证集。',
            'next_actions': [
                {
                    'description': '先准备数据，补齐 data.yaml 和划分产物',
                    'tool': 'prepare_dataset_for_training',
                    'args_hint': {'dataset_path': dataset_path, 'force_split': True},
                }
            ],
        }

    base_tool = StructuredTool.from_function(
        func=None,
        coroutine=_echo,
        name='dataset_training_readiness',
        description='dataset readiness',
        args_schema=_Args,
        return_direct=False,
    )
    adapted = adapt_tools_for_chat_model([base_tool], include_aliases=False)[0]
    raw = await adapted.ainvoke({'dataset_path': '/dataset'})
    assert '这份数据还不能直接训练' in raw, raw
    assert '先准备数据，补齐 data.yaml 和划分产物' in raw, raw
    assert 'prepare_dataset_for_training' not in raw, raw
    assert 'args_hint' not in raw, raw


def main() -> None:
    _server_tool_contract()
    asyncio.run(_llm_reply_path_smoke())
    asyncio.run(_dataset_quality_llm_reply_path_smoke())
    asyncio.run(_stream_tools_suppression_smoke())
    asyncio.run(_tool_none_sanitization_smoke())
    asyncio.run(_tool_result_compaction_smoke())
    _compose_reply_fallback_smoke()
    print('dataset training readiness ok')


if __name__ == '__main__':
    main()

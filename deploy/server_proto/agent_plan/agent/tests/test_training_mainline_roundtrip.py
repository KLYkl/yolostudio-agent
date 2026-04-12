from __future__ import annotations

import asyncio
import shutil
import sys
import types
from pathlib import Path
from typing import Any

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import langchain_openai  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_mod

try:
    import langchain_ollama  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_ollama')

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOllama = _FakeChatOllama
    sys.modules['langchain_ollama'] = fake_mod

try:
    import langchain_core.messages  # type: ignore  # noqa: F401
except Exception:
    core_mod = types.ModuleType('langchain_core')
    messages_mod = types.ModuleType('langchain_core.messages')
    tools_mod = types.ModuleType('langchain_core.tools')

    class _BaseMessage:
        def __init__(self, content=''):
            self.content = content

    class _AIMessage(_BaseMessage):
        def __init__(self, content='', tool_calls=None):
            super().__init__(content)
            self.tool_calls = tool_calls or []

    class _HumanMessage(_BaseMessage):
        pass

    class _SystemMessage(_BaseMessage):
        pass

    class _ToolMessage(_BaseMessage):
        def __init__(self, content='', name='', tool_call_id=''):
            super().__init__(content)
            self.name = name
            self.tool_call_id = tool_call_id

    class _BaseTool:
        name = 'fake'
        description = 'fake'
        args_schema = None

    class _StructuredTool(_BaseTool):
        @classmethod
        def from_function(cls, func=None, coroutine=None, name='', description='', args_schema=None, return_direct=False):
            tool = cls()
            tool.func = func
            tool.coroutine = coroutine
            tool.name = name
            tool.description = description
            tool.args_schema = args_schema
            tool.return_direct = return_direct
            return tool

    messages_mod.AIMessage = _AIMessage
    messages_mod.BaseMessage = _BaseMessage
    messages_mod.HumanMessage = _HumanMessage
    messages_mod.SystemMessage = _SystemMessage
    messages_mod.ToolMessage = _ToolMessage
    tools_mod.BaseTool = _BaseTool
    tools_mod.StructuredTool = _StructuredTool
    core_mod.messages = messages_mod
    core_mod.tools = tools_mod
    sys.modules['langchain_core'] = core_mod
    sys.modules['langchain_core.messages'] = messages_mod
    sys.modules['langchain_core.tools'] = tools_mod

try:
    import langchain_mcp_adapters.client  # type: ignore  # noqa: F401
except Exception:
    client_mod = types.ModuleType('langchain_mcp_adapters.client')

    class _FakeMCPClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def get_tools(self):
            return []

    client_mod.MultiServerMCPClient = _FakeMCPClient
    sys.modules['langchain_mcp_adapters.client'] = client_mod

try:
    import pydantic  # type: ignore  # noqa: F401
except Exception:
    pyd_mod = types.ModuleType('pydantic')

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _Field(default=None, description=''):
        del description
        return default

    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['pydantic'] = pyd_mod

try:
    import langgraph.prebuilt  # type: ignore  # noqa: F401
    import langgraph.types  # type: ignore  # noqa: F401
    import langgraph.checkpoint.memory  # type: ignore  # noqa: F401
except Exception:
    prebuilt_mod = types.ModuleType('langgraph.prebuilt')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')

    def _fake_create_react_agent(*args, **kwargs):
        raise AssertionError('create_react_agent should not be called in training mainline roundtrip smoke')

    class _Command:
        def __init__(self, resume=None):
            self.resume = resume

    class _InMemorySaver:
        def __init__(self, *args, **kwargs):
            self.storage = {}
            self.writes = {}
            self.blobs = {}

    prebuilt_mod.create_react_agent = _fake_create_react_agent
    types_mod.Command = _Command
    checkpoint_mod.InMemorySaver = _InMemorySaver
    sys.modules['langgraph.prebuilt'] = prebuilt_mod
    sys.modules['langgraph.types'] = types_mod
    sys.modules['langgraph.checkpoint.memory'] = checkpoint_mod

from agent_plan.agent.client.agent_client import AgentSettings, YoloStudioAgentClient


class _DummyGraph:
    def get_state(self, config):
        return None


WORK = Path(__file__).resolve().parent / '_tmp_training_mainline_roundtrip'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='training-mainline-roundtrip', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        calls: list[tuple[str, dict[str, Any]]] = []
        prepared = {'value': False}

        async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'training_readiness':
                if prepared['value']:
                    result = {
                        'ok': True,
                        'summary': '训练前检查完成：准备后的数据已具备训练条件。',
                        'dataset_root': '/data/mainline',
                        'resolved_img_dir': '/data/mainline/images',
                        'resolved_label_dir': '/data/mainline/labels',
                        'resolved_data_yaml': '/data/mainline/data.yaml',
                        'ready': True,
                        'preparable': False,
                        'warnings': ['当前是 prepare 后的首轮训练，建议持续观察验证指标'],
                        'blockers': [],
                    }
                else:
                    result = {
                        'ok': True,
                        'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                        'dataset_root': '/data/mainline',
                        'resolved_img_dir': '/data/mainline/images',
                        'resolved_label_dir': '/data/mainline/labels',
                        'resolved_data_yaml': '',
                        'ready': False,
                        'preparable': True,
                        'primary_blocker_type': 'missing_yaml',
                        'warnings': ['先准备数据，再启动训练更稳'],
                        'blockers': ['缺少可用的 data_yaml'],
                    }
            elif tool_name == 'list_training_environments':
                result = {
                    'ok': True,
                    'summary': '发现 2 个可用训练环境，默认将使用 yolodo',
                    'environments': [
                        {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True},
                        {'name': 'base', 'display_name': 'base', 'selected_by_default': False},
                    ],
                    'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                }
            elif tool_name == 'prepare_dataset_for_training':
                prepared['value'] = True
                result = {
                    'ok': True,
                    'ready': True,
                    'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                    'dataset_root': '/data/mainline',
                    'img_dir': '/data/mainline/images',
                    'label_dir': '/data/mainline/labels',
                    'data_yaml': '/data/mainline/data.yaml',
                    'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
                    'next_actions': ['如需训练，可继续 start_training'],
                }
            elif tool_name == 'training_preflight':
                selected_environment = kwargs.get('training_environment') or 'yolodo'
                result = {
                    'ok': True,
                    'ready_to_start': True,
                    'summary': f"训练预检通过：将使用 {selected_environment}，device={kwargs.get('device', 'auto') or 'auto'}",
                    'training_environment': {'name': selected_environment, 'display_name': selected_environment},
                    'resolved_args': {
                        'model': kwargs['model'],
                        'data_yaml': kwargs['data_yaml'],
                        'epochs': kwargs['epochs'],
                        'device': kwargs.get('device', 'auto') or 'auto',
                        'training_environment': selected_environment,
                        'batch': kwargs.get('batch'),
                        'imgsz': kwargs.get('imgsz'),
                    },
                    'command_preview': ['yolo', 'train'],
                    'blockers': [],
                    'warnings': ['训练已启动前最后一步确认'],
                }
            elif tool_name == 'start_training':
                result = {
                    'ok': True,
                    'summary': '训练已启动: model=yolov8n.pt, data=/data/mainline/data.yaml, device=auto',
                    'device': kwargs.get('device', 'auto') or 'auto',
                    'pid': 4567,
                    'log_file': '/runs/train_mainline.txt',
                    'started_at': 345.6,
                    'resolved_args': dict(kwargs),
                }
            elif tool_name == 'check_training_status':
                result = {
                    'ok': True,
                    'summary': '当前训练正在进行中，最近一次观察停在第 2/30 轮。',
                    'running': True,
                    'run_state': 'running',
                    'observation_stage': 'early',
                    'progress': {'epoch': 2, 'total_epochs': 30, 'progress_ratio': 2 / 30},
                    'latest_metrics': {
                        'ok': True,
                        'metrics': {
                            'epoch': 2,
                            'total_epochs': 30,
                            'box_loss': 1.245,
                            'cls_loss': 0.876,
                            'dfl_loss': 0.934,
                        },
                    },
                    'analysis_ready': False,
                    'minimum_facts_ready': True,
                    'signals': ['training_running', 'loss_only_metrics'],
                    'facts': ['epoch=2/30', 'batch=12', 'imgsz=960'],
                    'next_actions': ['可继续轮询 check_training_status', '也可以调用 summarize_training_run 查看阶段性事实'],
                }
            elif tool_name == 'summarize_training_run':
                result = {
                    'ok': True,
                    'summary': '训练结果汇总: 当前训练仍在早期阶段，只有损失指标，尚不能下最终结论。',
                    'run_state': 'running',
                    'model_family': 'yolo',
                    'task_type': 'detection',
                    'metrics': {'box_loss': 1.245, 'cls_loss': 0.876, 'dfl_loss': 0.934},
                    'signals': ['training_running', 'loss_only_metrics', 'insufficient_eval_metrics'],
                    'facts': ['epoch=2/30', '当前仅有训练损失', '尚无稳定 precision/recall/mAP'],
                    'next_actions': ['继续训练到更多 epoch', '再观察验证指标'],
                    'analysis_ready': False,
                    'minimum_facts_ready': True,
                    'observation_stage': 'early',
                }
            elif tool_name == 'analyze_training_outcome':
                assert kwargs['metrics'] == client.session_state.active_training.last_summary
                result = {
                    'ok': True,
                    'summary': '训练结果分析: 当前仍属早期观察，更适合继续收集验证指标。',
                    'assessment': 'continue_training',
                    'interpretation': '现在只有训练损失，暂时不能把阶段性变化当成最终效果。',
                    'recommendation': '继续训练并关注后续验证指标。',
                    'matched_rule_ids': ['generic_training_early_observation'],
                    'signals': ['training_running', 'insufficient_eval_metrics'],
                    'facts': ['epoch=2/30', '当前仅有训练损失'],
                    'next_actions': ['继续训练到更多 epoch', '之后再看 precision/recall/mAP'],
                    'source_summary': {'official': 2, 'workflow': 1},
                }
            elif tool_name == 'recommend_next_training_step':
                assert kwargs['status'] == client.session_state.active_training.last_summary
                result = {
                    'ok': True,
                    'summary': '下一步建议: 当前先继续训练并收集更稳定的验证指标，不急着改参数。',
                    'recommended_action': 'continue_training_and_collect_metrics',
                    'basis': ['当前训练仅到第 2/30 轮', '尚无稳定 precision/recall/mAP'],
                    'why': '现在更像事实不足，而不是已经可以判断数据或参数问题。',
                    'matched_rule_ids': ['generic_training_early_observation'],
                    'signals': ['training_running', 'insufficient_eval_metrics'],
                    'next_actions': ['继续训练到更多 epoch', '等有验证指标后再判断是补数据还是调参数'],
                    'source_summary': {'official': 2, 'workflow': 1},
                }
            else:
                raise AssertionError(f'unexpected tool call: {tool_name}')

            client._apply_to_state(tool_name, result, kwargs)
            client._record_secondary_event(tool_name, result)
            if tool_name == 'start_training' and result.get('ok'):
                client._clear_training_plan_draft()
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        turn1 = await client.chat('数据在 /data/mainline，用 yolov8n.pt 训练 30轮，batch 12，imgsz 960。')
        assert turn1['status'] == 'needs_confirmation', turn1
        assert turn1['tool_call']['name'] == 'prepare_dataset_for_training'
        assert turn1['tool_call']['args']['dataset_path'] == '/data/mainline'

        turn2 = await client.confirm(turn1['thread_id'], approved=True)
        assert turn2['status'] == 'needs_confirmation', turn2
        assert turn2['tool_call']['name'] == 'start_training'
        assert turn2['tool_call']['args']['model'] == 'yolov8n.pt'
        assert turn2['tool_call']['args']['data_yaml'] == '/data/mainline/data.yaml'
        assert turn2['tool_call']['args']['batch'] == 12
        assert turn2['tool_call']['args']['imgsz'] == 960

        turn3 = await client.confirm(turn2['thread_id'], approved=True)
        assert turn3['status'] == 'completed', turn3
        assert '训练已启动' in turn3['message']
        assert client.session_state.active_training.running is True
        assert client.session_state.active_training.model == 'yolov8n.pt'
        assert client.session_state.active_training.data_yaml == '/data/mainline/data.yaml'

        turn4 = await client.chat('现在训练到第几轮了？')
        assert turn4['status'] == 'completed', turn4
        assert calls[-1][0] == 'check_training_status'
        assert '运行状态: running' in turn4['message']
        assert '最近进度: 2/30' in turn4['message']
        assert client.session_state.active_training.last_status.get('run_state') == 'running'

        turn5 = await client.chat('这次训练效果怎么样？')
        assert turn5['status'] == 'completed', turn5
        assert calls[-2][0] == 'summarize_training_run'
        assert calls[-1][0] == 'analyze_training_outcome'
        assert '训练结果汇总:' in turn5['message']
        assert '训练结果分析:' in turn5['message']
        assert '当前仍属早期观察' in turn5['message']
        assert client.session_state.active_training.last_summary.get('run_state') == 'running'

        turn6 = await client.chat('下一步先补数据还是调参数？')
        assert turn6['status'] == 'completed', turn6
        assert calls[-3][0] == 'training_readiness'
        assert calls[-2][0] == 'summarize_training_run'
        assert calls[-1][0] == 'recommend_next_training_step'
        assert '下一步建议:' in turn6['message']
        assert '优先动作: continue_training_and_collect_metrics' in turn6['message']
        assert '继续训练到更多 epoch' in turn6['message']
        print('training mainline roundtrip ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()

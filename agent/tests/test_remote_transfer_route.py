
from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path
from typing import Any

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def _install_fake_test_dependencies() -> None:
    fake_openai = types.ModuleType('langchain_openai')
    fake_ollama = types.ModuleType('langchain_ollama')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_openai.ChatOpenAI = _FakeChatOpenAI
    fake_ollama.ChatOllama = _FakeChatOllama
    sys.modules['langchain_openai'] = fake_openai
    sys.modules['langchain_ollama'] = fake_ollama

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

        async def ainvoke(self, args):
            return args

        def invoke(self, args):
            return args

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

        async def ainvoke(self, args):
            if self.coroutine:
                return await self.coroutine(**args)
            if self.func:
                return self.func(**args)
            return args

        def invoke(self, args):
            if self.func:
                return self.func(**args)
            return args

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

    client_mod = types.ModuleType('langchain_mcp_adapters.client')

    class _FakeMCPClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def get_tools(self):
            return []

    client_mod.MultiServerMCPClient = _FakeMCPClient
    sys.modules['langchain_mcp_adapters.client'] = client_mod

    pyd_mod = types.ModuleType('pydantic')

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _Field(default=None, **kwargs):
        del kwargs
        return default

    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['pydantic'] = pyd_mod

    prebuilt_mod = types.ModuleType('langgraph.prebuilt')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')

    def _fake_create_react_agent(*args, **kwargs):
        raise AssertionError('create_react_agent should not be called in remote transfer route tests')

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


_install_fake_test_dependencies()

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.tests._coroutine_runner import run


class _NoLLMGraph:
    def get_state(self, config):
        return None

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError('remote transfer route should stay on routed flows, not fallback to graph')


WORK = Path(__file__).resolve().parent / '_tmp_remote_transfer_route'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    client = YoloStudioAgentClient(graph=_NoLLMGraph(), settings=settings, tool_registry={})
    return client


async def _scenario_list_remote_profiles_route() -> None:
    client = _make_client('profiles')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '远端 profile 1 个 / SSH alias 1 个。 默认 profile: lab。',
            'profiles_path': '/tmp/remote_profiles.json',
            'default_profile': 'lab',
            'profiles': [{'name': 'lab', 'target_label': 'lab', 'remote_root': '/srv/agent_stage', 'is_default': True}],
            'ssh_aliases': [{'name': 'lab-ssh', 'hostname': 'demo-host', 'port': '22'}],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('先看看有哪些可用服务器配置')
    assert turn['status'] == 'completed', turn
    assert calls == [('list_remote_profiles', {})], calls
    assert 'lab' in turn['message'], turn
    assert client.session_state.active_remote_transfer.profile_name == 'lab'
    assert client.session_state.active_remote_transfer.remote_root == '/srv/agent_stage'


async def _scenario_upload_route_requires_confirmation_and_then_executes() -> None:
    client = _make_client('upload')
    calls: list[tuple[str, dict[str, Any]]] = []

    local_root = WORK / 'upload_artifacts'
    weight_path = local_root / 'best.pt'
    dataset_dir = local_root / 'dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    weight_path.write_text('fake-weight', encoding='utf-8')
    (dataset_dir / 'a.txt').write_text('demo', encoding='utf-8')

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '远端上传完成：已上传 2 个本地项到 yolostudio:/tmp/agent_stage',
            'target_label': 'yolostudio',
            'profile_name': '',
            'remote_root': '/tmp/agent_stage',
            'uploaded_count': 2,
            'uploaded_items': [
                {'local_path': str(weight_path), 'remote_path': '/tmp/agent_stage/best.pt', 'item_type': 'file'},
                {'local_path': str(dataset_dir), 'remote_path': '/tmp/agent_stage/dataset', 'item_type': 'directory'},
            ],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat(f'把 "{weight_path}" 和 "{dataset_dir}" 上传到服务器 yolostudio 的 /tmp/agent_stage')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'upload_assets_to_remote', turn
    assert turn['tool_call']['args']['server'] == 'yolostudio', turn
    assert turn['tool_call']['args']['remote_root'] == '/tmp/agent_stage', turn
    assert len(turn['tool_call']['args']['local_paths']) == 2, turn

    done = await client.confirm(turn['thread_id'], True)
    assert done['status'] == 'completed', done
    assert calls and calls[0][0] == 'upload_assets_to_remote', calls
    assert '/tmp/agent_stage' in done['message'], done
    assert client.session_state.active_remote_transfer.remote_root == '/tmp/agent_stage'
    assert client.session_state.active_remote_transfer.last_upload['uploaded_count'] == 2


async def _scenario_remote_prediction_pipeline_route() -> None:
    client = _make_client('remote-predict')
    calls: list[tuple[str, dict[str, Any]]] = []

    local_root = WORK / 'remote_predict_artifacts'
    weight_path = local_root / 'best.pt'
    image_dir = local_root / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    weight_path.write_text('fake-weight', encoding='utf-8')
    (image_dir / 'a.jpg').write_bytes(b'fake-image')

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'upload_assets_to_remote':
            result = {
                'ok': True,
                'summary': '远端上传完成：已上传模型和图片目录到 yolostudio:/tmp/predict_stage',
                'target_label': 'yolostudio',
                'profile_name': '',
                'remote_root': '/tmp/predict_stage',
                'uploaded_count': 2,
                'uploaded_items': [
                    {'local_path': str(weight_path), 'remote_path': '/tmp/predict_stage/best.pt', 'item_type': 'file'},
                    {'local_path': str(image_dir), 'remote_path': '/tmp/predict_stage/images', 'item_type': 'directory'},
                ],
            }
        elif tool_name == 'predict_images':
            result = {
                'ok': True,
                'summary': '远端图片预测完成：处理 1 张，命中 1 张。',
                'source_path': '/tmp/predict_stage/images',
                'model': '/tmp/predict_stage/best.pt',
                'output_dir': '/tmp/predict_stage/_agent_prediction_output_20260413_190000',
                'report_path': '/tmp/predict_stage/_agent_prediction_output_20260413_190000/prediction_report.json',
                'processed_images': 1,
                'detected_images': 1,
                'empty_images': 0,
                'class_counts': {'car': 1},
            }
        elif tool_name == 'download_assets_from_remote':
            local_result_root = Path(str(kwargs.get('local_root') or ''))
            returned_dir = local_result_root / 'prediction_output'
            returned_dir.mkdir(parents=True, exist_ok=True)
            (returned_dir / 'prediction_report.json').write_text('{"ok": true}', encoding='utf-8')
            result = {
                'ok': True,
                'summary': '远端下载完成：已把 prediction 输出拉回本机。',
                'target_label': 'yolostudio',
                'profile_name': '',
                'local_root': str(local_result_root),
                'downloaded_count': 1,
                'downloaded_items': [
                    {
                        'remote_path': '/tmp/predict_stage/_agent_prediction_output_20260413_190000',
                        'local_path': str(returned_dir),
                        'item_type': 'directory',
                        'size_bytes': 0,
                    }
                ],
            }
        else:
            raise AssertionError(f'unexpected tool: {tool_name}')
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat(
        f'把 "{weight_path}" 和 "{image_dir}" 上传到服务器 yolostudio 的 /tmp/predict_stage，'
        '然后做图片预测并把结果拉回本机'
    )
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'remote_prediction_pipeline', turn
    assert '远端预测闭环' in turn['message'], turn
    assert '结果下载回本机' in turn['message'], turn

    local_result_root = Path(turn['tool_call']['args']['local_result_root'])
    done = await client.confirm(turn['thread_id'], True)
    assert done['status'] == 'completed', done
    assert [name for name, _ in calls] == ['upload_assets_to_remote', 'predict_images', 'download_assets_from_remote'], calls
    assert 'prediction 输出拉回本机' in done['message'], done
    assert client.session_state.active_prediction.last_remote_roundtrip['remote_output_dir'] == '/tmp/predict_stage/_agent_prediction_output_20260413_190000'
    assert client.session_state.active_prediction.last_remote_roundtrip['local_result_root'] == str(local_result_root)
    assert client.session_state.active_remote_transfer.last_download['downloaded_count'] == 1
    if local_result_root.exists():
        shutil.rmtree(local_result_root, ignore_errors=True)


async def _scenario_remote_training_pipeline_route() -> None:
    client = _make_client('remote-train')
    calls: list[tuple[str, dict[str, Any]]] = []

    local_root = WORK / 'remote_train_artifacts'
    weight_path = local_root / 'best.pt'
    dataset_dir = local_root / 'dataset'
    (dataset_dir / 'images').mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'labels').mkdir(parents=True, exist_ok=True)
    weight_path.write_text('fake-weight', encoding='utf-8')
    (dataset_dir / 'images' / 'a.jpg').write_bytes(b'fake-image')
    (dataset_dir / 'labels' / 'a.txt').write_text('0 0.5 0.5 0.2 0.2', encoding='utf-8')

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'upload_assets_to_remote':
            result = {
                'ok': True,
                'summary': '远端上传完成：已上传模型和数据集到 yolostudio:/tmp/train_stage',
                'target_label': 'yolostudio',
                'profile_name': '',
                'remote_root': '/tmp/train_stage',
                'uploaded_count': 2,
                'uploaded_items': [
                    {'local_path': str(weight_path), 'remote_path': '/tmp/train_stage/best.pt', 'item_type': 'file'},
                    {'local_path': str(dataset_dir), 'remote_path': '/tmp/train_stage/dataset', 'item_type': 'directory'},
                ],
            }
        elif tool_name == 'training_readiness':
            result = {
                'ok': True,
                'ready': False,
                'preparable': True,
                'summary': '数据集还缺 data.yaml，但可以自动补齐。',
                'dataset_root': '/tmp/train_stage/dataset',
                'resolved_img_dir': '/tmp/train_stage/dataset/images',
                'resolved_label_dir': '/tmp/train_stage/dataset/labels',
                'resolved_data_yaml': '',
                'primary_blocker_type': 'missing_data_yaml',
            }
        elif tool_name == 'prepare_dataset_for_training':
            result = {
                'ok': True,
                'ready': True,
                'summary': '数据准备完成：已生成 data.yaml。',
                'dataset_root': '/tmp/train_stage/dataset',
                'img_dir': '/tmp/train_stage/dataset/images',
                'label_dir': '/tmp/train_stage/dataset/labels',
                'data_yaml': '/tmp/train_stage/dataset/data.yaml',
            }
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过，可直接启动。',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': '/tmp/train_stage/best.pt',
                    'data_yaml': '/tmp/train_stage/dataset/data.yaml',
                    'epochs': 20,
                    'device': 'auto',
                    'project': '/tmp/train_stage/runs',
                    'name': 'remote-train-demo',
                },
                'command_preview': ['python', 'train.py', '--data', '/tmp/train_stage/dataset/data.yaml'],
            }
        elif tool_name == 'start_training':
            result = {
                'ok': True,
                'summary': '训练已启动：run=remote-train-demo',
                'resolved_args': {
                    'model': '/tmp/train_stage/best.pt',
                    'data_yaml': '/tmp/train_stage/dataset/data.yaml',
                    'epochs': 20,
                    'device': 'auto',
                    'project': '/tmp/train_stage/runs',
                    'name': 'remote-train-demo',
                    'training_environment': 'yolodo',
                },
                'device': 'auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'pid': 4321,
                'log_file': '/tmp/train_stage/runs/remote-train-demo/train.log',
            }
        else:
            raise AssertionError(f'unexpected tool: {tool_name}')
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat(
        f'把 "{weight_path}" 和 "{dataset_dir}" 上传到服务器 yolostudio 的 /tmp/train_stage，'
        '然后直接开始训练 20 轮'
    )
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'remote_training_pipeline', turn
    assert '远端训练闭环' in turn['message'], turn
    assert '启动训练' in turn['message'], turn

    done = await client.confirm(turn['thread_id'], True)
    assert done['status'] == 'completed', done
    assert [name for name, _ in calls] == [
        'upload_assets_to_remote',
        'training_readiness',
        'prepare_dataset_for_training',
        'training_preflight',
        'start_training',
    ], calls
    assert '训练已启动' in done['message'], done
    roundtrip = client.session_state.active_training.last_remote_roundtrip
    assert roundtrip['remote_dataset_path'] == '/tmp/train_stage/dataset'
    assert roundtrip['remote_model_path'] == '/tmp/train_stage/best.pt'
    assert client.session_state.active_training.last_preflight['ready_to_start'] is True
    assert client.session_state.active_training.running is True


async def _scenario_remote_training_pipeline_waits_and_downloads() -> None:
    client = _make_client('remote-train-download')
    calls: list[tuple[str, dict[str, Any]]] = []
    status_checks = 0

    local_root = WORK / 'remote_train_download_artifacts'
    weight_path = local_root / 'best.pt'
    dataset_dir = local_root / 'dataset'
    (dataset_dir / 'images').mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'labels').mkdir(parents=True, exist_ok=True)
    weight_path.write_text('fake-weight', encoding='utf-8')
    (dataset_dir / 'images' / 'a.jpg').write_bytes(b'fake-image')
    (dataset_dir / 'labels' / 'a.txt').write_text('0 0.5 0.5 0.2 0.2', encoding='utf-8')

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal status_checks
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'upload_assets_to_remote':
            result = {
                'ok': True,
                'summary': '远端上传完成：已上传模型和数据集到 yolostudio:/tmp/train_stage',
                'target_label': 'yolostudio',
                'profile_name': '',
                'remote_root': '/tmp/train_stage',
                'uploaded_count': 2,
                'uploaded_items': [
                    {'local_path': str(weight_path), 'remote_path': '/tmp/train_stage/best.pt', 'item_type': 'file'},
                    {'local_path': str(dataset_dir), 'remote_path': '/tmp/train_stage/dataset', 'item_type': 'directory'},
                ],
            }
        elif tool_name == 'training_readiness':
            result = {
                'ok': True,
                'ready': True,
                'preparable': False,
                'summary': '训练前检查通过，数据已就绪。',
                'dataset_root': '/tmp/train_stage/dataset',
                'resolved_img_dir': '/tmp/train_stage/dataset/images',
                'resolved_label_dir': '/tmp/train_stage/dataset/labels',
                'resolved_data_yaml': '/tmp/train_stage/dataset/data.yaml',
            }
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过，可直接启动。',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': '/tmp/train_stage/best.pt',
                    'data_yaml': '/tmp/train_stage/dataset/data.yaml',
                    'epochs': 5,
                    'device': 'auto',
                    'project': '/tmp/train_stage/runs',
                    'name': 'remote-train-demo',
                },
                'command_preview': ['python', 'train.py', '--data', '/tmp/train_stage/dataset/data.yaml'],
            }
        elif tool_name == 'start_training':
            result = {
                'ok': True,
                'summary': '训练已启动：run=remote-train-demo',
                'resolved_args': {
                    'model': '/tmp/train_stage/best.pt',
                    'data_yaml': '/tmp/train_stage/dataset/data.yaml',
                    'epochs': 5,
                    'device': 'auto',
                    'project': '/tmp/train_stage/runs',
                    'name': 'remote-train-demo',
                    'training_environment': 'yolodo',
                },
                'device': 'auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'pid': 5432,
                'log_file': '/tmp/train_stage/runs/remote-train-demo/train.log',
            }
        elif tool_name == 'check_training_status':
            status_checks += 1
            if status_checks == 1:
                result = {
                    'ok': True,
                    'running': True,
                    'run_state': 'running',
                    'summary': '训练进行中 (device=auto, pid=5432，epoch 2/5，当前仍属早期观察)',
                    'resolved_args': {
                        'model': '/tmp/train_stage/best.pt',
                        'data_yaml': '/tmp/train_stage/dataset/data.yaml',
                        'epochs': 5,
                        'device': 'auto',
                        'project': '/tmp/train_stage/runs',
                        'name': 'remote-train-demo',
                    },
                    'log_file': '/tmp/train_stage/runs/remote-train-demo/train.log',
                    'save_dir': '/tmp/train_stage/runs/remote-train-demo',
                }
            else:
                result = {
                    'ok': True,
                    'running': False,
                    'run_state': 'completed',
                    'summary': '当前无训练在跑，最近一次训练已完成，return_code=0，已有可分析指标',
                    'resolved_args': {
                        'model': '/tmp/train_stage/best.pt',
                        'data_yaml': '/tmp/train_stage/dataset/data.yaml',
                        'epochs': 5,
                        'device': 'auto',
                        'project': '/tmp/train_stage/runs',
                        'name': 'remote-train-demo',
                    },
                    'log_file': '/tmp/train_stage/runs/remote-train-demo/train.log',
                    'save_dir': '/tmp/train_stage/runs/remote-train-demo',
                }
        elif tool_name == 'summarize_training_run':
            result = {
                'ok': True,
                'summary': '训练结果汇总: 最近一次训练已完成，并且已有可分析指标。',
                'run_state': 'completed',
                'analysis_ready': True,
                'minimum_facts_ready': True,
                'save_dir': '/tmp/train_stage/runs/remote-train-demo',
                'facts': ['save_dir=/tmp/train_stage/runs/remote-train-demo'],
                'metrics': {'precision': 0.88},
            }
        elif tool_name == 'inspect_training_run':
            result = {
                'ok': True,
                'summary': '训练记录详情已就绪',
                'selected_run_id': 'train_log_remote_train_demo',
                'run_state': 'completed',
                'analysis_ready': True,
                'minimum_facts_ready': True,
                'resolved_args': {
                    'project': '/tmp/train_stage/runs',
                    'name': 'remote-train-demo',
                },
                'save_dir': '/tmp/train_stage/runs/remote-train-demo',
                'facts': ['save_dir=/tmp/train_stage/runs/remote-train-demo'],
            }
        elif tool_name == 'download_assets_from_remote':
            local_result_root = Path(str(kwargs.get('local_root') or ''))
            returned_dir = local_result_root / 'remote-train-demo'
            returned_dir.mkdir(parents=True, exist_ok=True)
            (returned_dir / 'results.csv').write_text('epoch,precision\n5,0.88\n', encoding='utf-8')
            result = {
                'ok': True,
                'summary': '远端下载完成：已把训练 run 目录拉回本机。',
                'target_label': 'yolostudio',
                'profile_name': '',
                'local_root': str(local_result_root),
                'downloaded_count': 1,
                'downloaded_items': [
                    {
                        'remote_path': '/tmp/train_stage/runs/remote-train-demo',
                        'local_path': str(returned_dir),
                        'item_type': 'directory',
                        'size_bytes': 0,
                    }
                ],
            }
        else:
            raise AssertionError(f'unexpected tool: {tool_name}')
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat(
        f'把 "{weight_path}" 和 "{dataset_dir}" 上传到服务器 yolostudio 的 /tmp/train_stage，'
        '开始训练 5 轮，等训练结束后把产物拉回本机'
    )
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'remote_training_pipeline', turn
    assert turn['tool_call']['args']['wait_for_completion'] is True, turn
    assert turn['tool_call']['args']['download_after_completion'] is True, turn
    assert '等待策略' in turn['message'], turn
    assert '自动把远端 run 目录下载回本机' in turn['message'], turn

    local_result_root = Path(turn['tool_call']['args']['local_result_root'])
    client.session_state.pending_confirmation.tool_args['poll_interval_seconds'] = 0
    client.session_state.pending_confirmation.tool_args['max_wait_seconds'] = 1

    done = await client.confirm(turn['thread_id'], True)
    assert done['status'] == 'completed', done
    assert [name for name, _ in calls] == [
        'upload_assets_to_remote',
        'training_readiness',
        'training_preflight',
        'start_training',
        'check_training_status',
        'check_training_status',
        'summarize_training_run',
        'inspect_training_run',
        'download_assets_from_remote',
    ], calls
    assert '训练结果汇总' in done['message'], done
    assert '训练 run 目录拉回本机' in done['message'], done
    roundtrip = client.session_state.active_training.last_remote_roundtrip
    assert roundtrip['wait_for_completion'] is True
    assert roundtrip['download_after_completion'] is True
    assert roundtrip['final_run_state'] == 'completed'
    assert roundtrip['remote_result_path'] == '/tmp/train_stage/runs/remote-train-demo'
    assert roundtrip['local_result_root'] == str(local_result_root)
    assert client.session_state.active_remote_transfer.last_download['downloaded_count'] == 1
    assert client.session_state.active_training.training_run_summary['run_state'] == 'completed'
    if local_result_root.exists():
        shutil.rmtree(local_result_root, ignore_errors=True)


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    try:
        run(_scenario_list_remote_profiles_route())
        run(_scenario_upload_route_requires_confirmation_and_then_executes())
        run(_scenario_remote_prediction_pipeline_route())
        run(_scenario_remote_training_pipeline_route())
        run(_scenario_remote_training_pipeline_waits_and_downloads())
        print('remote transfer route ok')
    finally:
        if WORK.exists():
            shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()

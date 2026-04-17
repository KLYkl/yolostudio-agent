from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.tests._chaos_test_support import WORK as P0_WORK, _make_client
from yolostudio_agent.agent.tests._coroutine_runner import run
from langchain_core.messages import AIMessage, ToolMessage


def _fresh_client(session_id: str):
    P0_WORK.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(P0_WORK / session_id, ignore_errors=True)
    client = _make_client(session_id)
    client.session_state.active_training.model = 'yolov8n.pt'
    client.session_state.active_prediction.model = 'yolov8n.pt'
    client.memory.save_state(client.session_state)
    return client


class _SingleToolGraph:
    def __init__(self, tool_name: str, args: dict[str, Any], result: dict[str, Any], final_text: str) -> None:
        self.tool_name = tool_name
        self.args = dict(args)
        self.result = dict(result)
        self.final_text = final_text
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        messages = list(payload['messages'])
        self.calls.append((self.tool_name, dict(self.args)))
        tool_call_id = f'call-{len(self.calls)}'
        return {
            'messages': messages + [
                AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': self.tool_name, 'args': self.args}]),
                ToolMessage(content=json.dumps(self.result, ensure_ascii=False), name=self.tool_name, tool_call_id=tool_call_id),
                AIMessage(content=self.final_text),
            ]
        }


@dataclass(frozen=True)
class _MatrixCase:
    case_id: str
    prompt: str
    category: str
    expected_status: str
    expected_tool_name: str = ''
    expected_message_contains: tuple[str, ...] = ()
    graph_tool_name: str = ''
    graph_args: dict[str, Any] = field(default_factory=dict)
    graph_result: dict[str, Any] = field(default_factory=dict)
    graph_final_text: str = ''
    state_preset: str = ''


def _install_training_matrix_tools(
    client,
    *,
    train_dataset_root: str,
    preparable_dataset_root: str,
):
    calls: list[tuple[str, dict[str, Any]]] = []
    train_aliases = _path_aliases(train_dataset_root)
    preparable_aliases = _path_aliases(preparable_dataset_root)

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            dataset_root = str(kwargs.get('img_dir') or '')
            if dataset_root in train_aliases:
                result = {
                    'ok': True,
                    'summary': '训练前检查完成：数据已具备训练条件。',
                    'dataset_root': dataset_root,
                    'resolved_img_dir': _path_text(Path(dataset_root) / 'images'),
                    'resolved_label_dir': _path_text(Path(dataset_root) / 'labels'),
                    'resolved_data_yaml': _path_text(Path(dataset_root) / 'data.yaml'),
                    'ready': True,
                    'preparable': False,
                    'warnings': [],
                    'blockers': [],
                }
            elif dataset_root in preparable_aliases:
                result = {
                    'ok': True,
                    'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                    'dataset_root': dataset_root,
                    'resolved_img_dir': _path_text(Path(dataset_root) / 'images'),
                    'resolved_label_dir': _path_text(Path(dataset_root) / 'labels'),
                    'resolved_data_yaml': '',
                    'ready': False,
                    'preparable': True,
                    'primary_blocker_type': 'missing_yaml',
                    'warnings': [],
                    'blockers': ['缺少可用的 data_yaml'],
                }
            else:
                raise AssertionError(f'unexpected readiness dataset: {dataset_root}')
        elif tool_name == 'dataset_training_readiness':
            dataset_root = str(kwargs.get('img_dir') or '')
            if dataset_root not in preparable_aliases:
                raise AssertionError(f'unexpected dataset readiness dataset: {dataset_root}')
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': dataset_root,
                'resolved_img_dir': _path_text(Path(dataset_root) / 'images'),
                'resolved_label_dir': _path_text(Path(dataset_root) / 'labels'),
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': [],
                'blockers': ['缺少可用的 data_yaml'],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': dict(kwargs),
                'command_preview': ['yolo', 'train'],
                'warnings': [],
                'blockers': [],
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return calls


def _install_no_tools(client):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'no direct tool expected: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return calls


def _apply_case_preset(client, case: _MatrixCase) -> None:
    if case.state_preset == 'best_weight_present':
        client.session_state.active_training.best_run_selection = {
            'summary': '最近最佳训练为 train_log_best。',
            'best_run': {
                'run_id': 'train_log_best',
                'best_weight_path': '/weights/best.pt',
            },
        }
        client.memory.save_state(client.session_state)
        return
    if case.state_preset == 'running_training_context':
        client.session_state.active_training.running = True
        client.session_state.active_training.pid = 9527
        client.session_state.active_training.data_yaml = '/data/train/data.yaml'
        client.memory.save_state(client.session_state)


def _path_text(path: Path) -> str:
    return path.resolve().as_posix()


def _path_aliases(path_text: str) -> set[str]:
    aliases = {path_text}
    normalized = path_text.replace('\\', '/')
    if len(normalized) >= 3 and normalized[1] == ':' and normalized[2] == '/':
        aliases.add(normalized[2:])
    return aliases


def _prepare_case_path_map(case: _MatrixCase, *, phase: str) -> dict[str, str]:
    if '/data/train' not in case.prompt and '/data/preparable' not in case.prompt:
        return {}
    dataset_root = P0_WORK / f'{case.case_id}-{phase}-datasets'
    shutil.rmtree(dataset_root, ignore_errors=True)
    train_root = dataset_root / 'train'
    preparable_root = dataset_root / 'preparable'
    for target in (train_root, preparable_root):
        (target / 'images').mkdir(parents=True, exist_ok=True)
        (target / 'labels').mkdir(parents=True, exist_ok=True)
    return {
        '/data/train': _path_text(train_root),
        '/data/preparable': _path_text(preparable_root),
    }


def _rewrite_case_text(value: str, replacements: dict[str, str]) -> str:
    updated = value
    for original, replacement in replacements.items():
        updated = updated.replace(original, replacement)
    return updated


def _make_graph_case(
    case_id: str,
    prompt: str,
    tool_name: str,
    args: dict[str, Any],
    final_text: str,
    *,
    state_preset: str = '',
) -> _MatrixCase:
    result = {
        'ok': True,
        'summary': final_text,
        'model': str(args.get('model') or ''),
        'source_path': str(args.get('source_path') or ''),
        'output_dir': '/tmp/intent-ab',
        'report_path': '/tmp/intent-ab/report.json',
    }
    if tool_name == 'predict_images':
        result.update({'processed_images': 2, 'detected_images': 1, 'empty_images': 1})
    else:
        result.update({'processed_videos': 1, 'total_frames': 12, 'detected_frames': 6, 'total_detections': 8})
    return _MatrixCase(
        case_id=case_id,
        prompt=prompt,
        category='graph',
        expected_status='completed',
        expected_message_contains=(final_text,),
        graph_tool_name=tool_name,
        graph_args=args,
        graph_result=result,
        graph_final_text=final_text,
        state_preset=state_preset,
    )


CASES: list[_MatrixCase] = [
    _make_graph_case('ab01', '先帮我预测 /data/images。', 'predict_images', {'source_path': '/data/images', 'model': 'yolov8n.pt'}, '图片预测完成'),
    _make_graph_case('ab02', '训练先放着，先帮我预测 /data/images。', 'predict_images', {'source_path': '/data/images', 'model': 'yolov8n.pt'}, '图片预测完成'),
    _make_graph_case('ab03', '训练先别动，先预测 /data/images。', 'predict_images', {'source_path': '/data/images', 'model': 'yolov8n.pt'}, '图片预测完成'),
    _make_graph_case('ab04', '先不训练，直接预测 /data/images。', 'predict_images', {'source_path': '/data/images', 'model': 'yolov8n.pt'}, '图片预测完成'),
    _make_graph_case('ab05', '训练稍后再说，先识别 /data/images。', 'predict_images', {'source_path': '/data/images', 'model': 'yolov8n.pt'}, '图片预测完成'),
    _make_graph_case('ab06', '训练以后再说，先推理 /data/images。', 'predict_images', {'source_path': '/data/images', 'model': 'yolov8n.pt'}, '图片预测完成'),
    _make_graph_case('ab07', 'train later, predict /data/images first.', 'predict_images', {'source_path': '/data/images', 'model': 'yolov8n.pt'}, '图片预测完成'),
    _make_graph_case('ab08', 'skip training for now and predict /data/images.', 'predict_images', {'source_path': '/data/images', 'model': 'yolov8n.pt'}, '图片预测完成'),
    _make_graph_case('ab09', '用最佳训练去预测图片 /data/images。', 'predict_images', {'source_path': '/data/images', 'model': '/weights/best.pt'}, '图片预测完成', state_preset='best_weight_present'),
    _make_graph_case('ab10', '训练晚点再说，先预测 /data/images，用 /models/yolov8n.pt。', 'predict_images', {'source_path': '/data/images', 'model': '/models/yolov8n.pt'}, '图片预测完成'),
    _make_graph_case('ab11', '先帮我预测 /data/videos。', 'predict_videos', {'source_path': '/data/videos', 'model': 'yolov8n.pt'}, '视频预测完成'),
    _make_graph_case('ab12', '训练先放着，先帮我预测这两个视频 /data/videos。', 'predict_videos', {'source_path': '/data/videos', 'model': 'yolov8n.pt'}, '视频预测完成'),
    _make_graph_case('ab13', '先不训练，先预测视频 /data/videos。', 'predict_videos', {'source_path': '/data/videos', 'model': 'yolov8n.pt'}, '视频预测完成'),
    _make_graph_case('ab14', '训练稍后再说，先预测视频 /data/videos。', 'predict_videos', {'source_path': '/data/videos', 'model': 'yolov8n.pt'}, '视频预测完成'),
    _make_graph_case('ab15', '训练以后再说，先推理 /data/videos。', 'predict_videos', {'source_path': '/data/videos', 'model': 'yolov8n.pt'}, '视频预测完成'),
    _make_graph_case('ab16', '用最佳训练去预测视频 /data/videos。', 'predict_videos', {'source_path': '/data/videos', 'model': '/weights/best.pt'}, '视频预测完成', state_preset='best_weight_present'),
    _make_graph_case('ab17', 'train later, predict /data/videos first.', 'predict_videos', {'source_path': '/data/videos', 'model': 'yolov8n.pt'}, '视频预测完成'),
    _make_graph_case('ab18', 'skip training for now and predict /data/videos.', 'predict_videos', {'source_path': '/data/videos', 'model': 'yolov8n.pt'}, '视频预测完成'),
    _make_graph_case('ab19', '训练晚点再说，先预测 /data/videos，用 /models/yolov8n.pt。', 'predict_videos', {'source_path': '/data/videos', 'model': '/models/yolov8n.pt'}, '视频预测完成'),
    _make_graph_case('ab20', '训练先放一放，先识别 /data/videos。', 'predict_videos', {'source_path': '/data/videos', 'model': 'yolov8n.pt'}, '视频预测完成'),
    _MatrixCase('ab21', '预测先别做，直接用 /data/train 和 yolov8n.pt 训练。', 'plan_start', 'needs_confirmation', expected_tool_name='start_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab22', '先不预测，用 /data/train 和 yolov8n.pt 训练。', 'plan_start', 'needs_confirmation', expected_tool_name='start_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab23', '不要预测，数据在 /data/train，用 yolov8n.pt 训练。', 'plan_start', 'needs_confirmation', expected_tool_name='start_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab24', '预测稍后再说，数据在 /data/train，用 yolov8n.pt 训练。', 'plan_start', 'needs_confirmation', expected_tool_name='start_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab25', '预测以后再说，数据在 /data/train，用 yolov8n.pt 训练。', 'plan_start', 'needs_confirmation', expected_tool_name='start_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab26', '预测晚点再说，数据在 /data/train，用 yolov8n.pt 训练。', 'plan_start', 'needs_confirmation', expected_tool_name='start_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab27', 'skip prediction for now, train /data/train with yolov8n.pt.', 'plan_start', 'needs_confirmation', expected_tool_name='start_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab28', 'predict later, train /data/train with yolov8n.pt.', 'plan_start', 'needs_confirmation', expected_tool_name='start_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab29', '预测先放着，数据在 /data/train，用 yolov8n.pt 训练，执行。', 'plan_start', 'needs_confirmation', expected_tool_name='start_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab30', '先别预测，/data/train 用 yolov8n.pt 训练 20 轮。', 'plan_start', 'needs_confirmation', expected_tool_name='start_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab31', '数据在 /data/preparable，用 yolov8n.pt 训练，只做准备。', 'prepare_only', 'needs_confirmation', expected_tool_name='prepare_dataset_for_training', expected_message_contains=('准备执行：数据准备',)),
    _MatrixCase('ab32', '预测先放着，数据在 /data/preparable，用 yolov8n.pt 训练，只做准备。', 'prepare_only', 'needs_confirmation', expected_tool_name='prepare_dataset_for_training', expected_message_contains=('准备执行：数据准备',)),
    _MatrixCase('ab33', '先不预测，数据在 /data/preparable，用 yolov8n.pt 训练，只做准备。', 'prepare_only', 'needs_confirmation', expected_tool_name='prepare_dataset_for_training', expected_message_contains=('准备执行：数据准备',)),
    _MatrixCase('ab34', '预测以后再说，数据在 /data/preparable，用 yolov8n.pt 训练，只做准备。', 'prepare_only', 'needs_confirmation', expected_tool_name='prepare_dataset_for_training', expected_message_contains=('准备执行：数据准备',)),
    _MatrixCase('ab35', '预测先别做，直接用 /data/preparable 和 yolov8n.pt 训练，执行。', 'prepare_then_train', 'needs_confirmation', expected_tool_name='prepare_dataset_for_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab36', '预测稍后再说，数据在 /data/preparable，用 yolov8n.pt 训练，执行。', 'prepare_then_train', 'needs_confirmation', expected_tool_name='prepare_dataset_for_training', expected_message_contains=('训练计划草案',)),
    _MatrixCase('ab37', '先做准备，数据在 /data/preparable，用 yolov8n.pt 训练。', 'prepare_only', 'needs_confirmation', expected_tool_name='prepare_dataset_for_training', expected_message_contains=('准备执行：数据准备',)),
    _MatrixCase('ab38', '预测晚点再说，先做准备，数据在 /data/preparable，用 yolov8n.pt 训练。', 'prepare_ambiguous', 'needs_confirmation', expected_tool_name='prepare_dataset_for_training'),
    _MatrixCase('ab39', '先预测视频 /data/videos，再继续上次训练。', 'blocked_mix', 'completed', expected_message_contains=('请拆成连续步骤',)),
    _MatrixCase('ab40', '先训练，再预测视频 /data/videos。', 'blocked_mix', 'completed', expected_message_contains=('请拆成连续步骤',)),
    _MatrixCase('ab41', '同时训练并预测 /data/videos。', 'blocked_mix', 'completed', expected_message_contains=('请拆成连续步骤',)),
    _MatrixCase('ab42', '一边训练一边预测 /data/videos。', 'blocked_mix', 'completed', expected_message_contains=('请拆成连续步骤',)),
    _MatrixCase('ab43', '边训练边不断做视频预测。', 'blocked_parallel', 'completed', expected_message_contains=('不支持“边训练边持续做视频预测”',)),
    _MatrixCase('ab44', '先预测视频 /data/videos，再比较两次训练。', 'blocked_mix', 'completed', expected_message_contains=('请拆成连续步骤',)),
    _MatrixCase('ab45', '先预测视频 /data/videos，然后对比最近两次训练。', 'blocked_mix', 'completed', expected_message_contains=('请拆成连续步骤',)),
    _MatrixCase('ab46', '先帮我预测 /data/videos，再用 /data/train 和 yolov8n.pt 开训，同时给出训练比较。', 'blocked_mix', 'completed', expected_message_contains=('请拆成连续步骤',)),
    _MatrixCase('ab47', '预测 /data/videos，然后训练 /data/train，再看哪个好。', 'blocked_mix', 'completed', expected_message_contains=('请拆成连续步骤',)),
    _MatrixCase('ab48', '一边预测一边训练一边比较结果。', 'blocked_mix', 'completed', expected_message_contains=('请拆成连续步骤',)),
    _MatrixCase('ab49', '就拿刚才的预测结果目录直接开始训练。', 'blocked_prediction_as_training_data', 'completed', expected_message_contains=('不能直接当训练数据开训',)),
    _MatrixCase('ab50', '训练失败后，用最好权重去预测视频 /data/videos。', 'blocked_missing_best_weight', 'completed', expected_message_contains=('不能直接假定“最佳训练”的权重文件路径',)),
    _MatrixCase(
        'ab51',
        '训练中，先把原视频 /data/raw.mp4 再抽一版。',
        'graph',
        'completed',
        expected_message_contains=('抽帧完成',),
        graph_tool_name='extract_video_frames',
        graph_args={},
        graph_result={
            'ok': True,
            'summary': '抽帧完成：已输出 20 帧。',
            'output_dir': '/tmp/intent-ab-frames',
            'saved_frames': 20,
            'warnings': [],
        },
        graph_final_text='抽帧完成：已输出 20 帧。',
        state_preset='running_training_context',
    ),
]


def _prepare_client_for_case(case: _MatrixCase, *, phase: str):
    client = _fresh_client(f'{case.case_id}-{phase}')
    path_map = _prepare_case_path_map(case, phase=phase)
    _apply_case_preset(client, case)
    if case.category == 'graph':
        calls = _install_no_tools(client)
        client.graph = _SingleToolGraph(
            case.graph_tool_name,
            case.graph_args,
            case.graph_result,
            case.graph_final_text,
        )  # type: ignore[assignment]
        return client, calls, _rewrite_case_text(case.prompt, path_map)
    if case.category in {'plan_start', 'prepare_only', 'prepare_then_train', 'prepare_ambiguous'}:
        calls = _install_training_matrix_tools(
            client,
            train_dataset_root=path_map.get('/data/train', '/data/train'),
            preparable_dataset_root=path_map.get('/data/preparable', '/data/preparable'),
        )
        return client, calls, _rewrite_case_text(case.prompt, path_map)
    calls = _install_no_tools(client)
    return client, calls, _rewrite_case_text(case.prompt, path_map)


async def _run_case(case: _MatrixCase) -> None:
    phase_a_client, phase_a_calls, phase_a_prompt = _prepare_client_for_case(case, phase='phase-a')
    phase_a_result = await phase_a_client._try_handle_mainline_intent(phase_a_prompt, f'{case.case_id}-phase-a')
    if case.category == 'graph':
        assert phase_a_result is None, (case.case_id, phase_a_result)
        assert phase_a_calls == [], (case.case_id, phase_a_calls)
    else:
        assert phase_a_result is not None, case.case_id
        assert phase_a_result['status'] == case.expected_status, (case.case_id, phase_a_result)
        if case.expected_tool_name:
            tool_call = phase_a_result.get('tool_call') or {}
            asserted_tool_name = str(tool_call.get('name') or '')
            if asserted_tool_name:
                assert asserted_tool_name == case.expected_tool_name, (case.case_id, phase_a_result)
        for text in case.expected_message_contains:
            assert text in str(phase_a_result.get('message') or ''), (case.case_id, phase_a_result)
        if case.category.startswith('blocked'):
            assert phase_a_calls == [], (case.case_id, phase_a_calls)

    phase_b_client, phase_b_calls, phase_b_prompt = _prepare_client_for_case(case, phase='phase-b')
    chat_result = await phase_b_client.chat(phase_b_prompt)
    assert chat_result['status'] == case.expected_status, (case.case_id, chat_result)
    if case.expected_tool_name:
        tool_call = chat_result.get('tool_call') or {}
        asserted_tool_name = str(tool_call.get('name') or '')
        if asserted_tool_name:
            assert asserted_tool_name == case.expected_tool_name, (case.case_id, chat_result)
    for text in case.expected_message_contains:
        assert text in str(chat_result.get('message') or ''), (case.case_id, chat_result)
    if case.category == 'graph':
        graph = phase_b_client.graph
        assert getattr(graph, 'calls', []) == [(case.graph_tool_name, case.graph_args)], (case.case_id, getattr(graph, 'calls', []))
        assert phase_b_calls == [], (case.case_id, phase_b_calls)
    elif case.category == 'plan_start':
        assert phase_b_calls[0][0] == 'training_readiness', (case.case_id, phase_b_calls)
        assert phase_b_calls[-1][0] == 'training_preflight', (case.case_id, phase_b_calls)
    elif case.category == 'prepare_only':
        assert phase_b_calls[0][0] == 'dataset_training_readiness', (case.case_id, phase_b_calls)
    elif case.category == 'prepare_ambiguous':
        assert phase_b_calls, (case.case_id, phase_b_calls)
        assert phase_b_calls[0][0] in {'training_readiness', 'dataset_training_readiness'}, (case.case_id, phase_b_calls)
        assert all(name != 'training_preflight' for name, _ in phase_b_calls), (case.case_id, phase_b_calls)
    elif case.category == 'prepare_then_train':
        assert phase_b_calls[0][0] == 'training_readiness', (case.case_id, phase_b_calls)
        assert all(name != 'training_preflight' for name, _ in phase_b_calls), (case.case_id, phase_b_calls)
    else:
        assert phase_b_calls == [], (case.case_id, phase_b_calls)


async def _run() -> None:
    for case in CASES:
        await _run_case(case)

    output = {
        'ok': True,
        'scenario_total': len(CASES),
        'graph_cases': sum(1 for case in CASES if case.category == 'graph'),
        'plan_cases': sum(1 for case in CASES if case.category in {'plan_start', 'prepare_only', 'prepare_then_train'}),
        'guard_cases': sum(1 for case in CASES if case.category.startswith('blocked')),
    }
    out_path = P0_WORK / 'test_agent_server_chaos_p3_intent_ab_matrix_output.json'
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'agent server chaos p3 intent ab matrix ok ({len(CASES)} prompts)')


if __name__ == '__main__':
    run(_run())

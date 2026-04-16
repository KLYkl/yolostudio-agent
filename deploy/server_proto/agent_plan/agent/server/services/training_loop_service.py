from __future__ import annotations

import asyncio
import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from yolostudio_agent.agent.server.services.knowledge_service import KnowledgeService
from yolostudio_agent.agent.server.services.train_service import TrainService

_TERMINAL_LOOP_STATES = {'completed', 'stopped', 'failed'}
_WAITING_LOOP_STATES = {'paused', 'awaiting_review'}
_ACTIVE_LOOP_STATES = {'queued', 'starting_round', 'running_round', 'analyzing_round', 'stopping'}
_SUPPORTED_TUNING_PARAMS = {'lr0', 'batch', 'imgsz', 'epochs', 'optimizer'}
_DEFAULT_TUNING_PARAMS = ('lr0', 'batch', 'imgsz', 'epochs')
_MANAGED_LEVELS = {'review', 'conservative_auto', 'full_auto'}
_PLANNER_DECISION_TYPES = {'auto_continue', 'await_review', 'stop'}
_PLANNER_PARAM_STRATEGIES = {'keep_current', 'apply_heuristic'}
_HARD_STOP_RECOMMENDED_ACTIONS = {
    'fix_data_quality',
    'collect_metrics_first',
    'inspect_labels_and_mapping',
}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = str(os.getenv(name, '') or '').strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except Exception:
        return default


_DEFAULT_LOOP_EPOCHS = _env_int('YOLOSTUDIO_LOOP_DEFAULT_EPOCHS', 10, minimum=1)

_REVIEW_REQUIRED_ACTIONS = {
    'inspect_missed_samples',
    'inspect_false_positives',
    'run_error_analysis',
}
_KNOWLEDGE_ACTION_LABELS = {
    'fix_data_quality': '先修数据质量',
    'collect_metrics_first': '先补训练事实',
    'inspect_labels_and_mapping': '先检查标签与类别映射',
    'inspect_missed_samples': '先分析漏检样本',
    'inspect_false_positives': '先分析误检样本',
    'run_error_analysis': '先做误差分析',
    'continue_observing': '继续观察',
}
_KNOWLEDGE_CATEGORY_LABELS = {
    'hard_stop': '硬停止建议',
    'analysis_review': '分析优先建议',
    'continue_observing': '继续观察',
    'other': '一般建议',
}
_KNOWLEDGE_OUTCOME_LABELS = {
    'awaiting_review': '等待审阅',
    'auto_continue': '自动继续',
    'hard_stop': '直接停止',
    'continue_observing': '继续观察',
    'paused': '暂停待定',
    'stopped': '已停止',
    'other': '一般处理',
}
_METRIC_ALIASES = {
    'map50': ('map50', 'mAP50'),
    'map': ('map', 'mAP50-95'),
    'precision': ('precision',),
    'recall': ('recall',),
}
_TRAINING_ARG_KEYS = {
    'model',
    'data_yaml',
    'epochs',
    'device',
    'training_environment',
    'project',
    'name',
    'batch',
    'imgsz',
    'fraction',
    'classes',
    'single_cls',
    'optimizer',
    'freeze',
    'resume',
    'lr0',
    'patience',
    'workers',
    'amp',
}


class TrainingLoopService:
    def __init__(
        self,
        state_dir: str | Path | None = None,
        *,
        train_service: TrainService | None = None,
        knowledge_service: KnowledgeService | None = None,
        loop_llm: Any | None = None,
        poll_interval: float = 3.0,
        time_fn: Any | None = None,
        sleep_fn: Any | None = None,
    ) -> None:
        root = Path(state_dir) if state_dir else Path(os.getenv('YOLOSTUDIO_TRAIN_LOOP_STATE_DIR', 'runs/training_loops'))
        self._state_dir = root
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._active_registry_path = self._state_dir / 'active_training_loop.json'
        self._last_registry_path = self._state_dir / 'last_training_loop.json'
        self._state_lock = threading.RLock()
        self._worker_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._worker_loop_id: str | None = None

        self.train_service = train_service or TrainService()
        self.knowledge_service = knowledge_service or KnowledgeService()
        self.loop_llm = None if loop_llm is False else loop_llm
        self._poll_interval = max(0.05, float(poll_interval))
        self._time_fn = time_fn or time.time
        self._sleep_fn = sleep_fn or time.sleep

        self._restore_active_worker_if_needed()

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        parts.append(text)
                    continue
                if isinstance(item, dict):
                    text = str(item.get('text') or item.get('content') or '').strip()
                    if text:
                        parts.append(text)
            return '\n'.join(parts).strip()
        return str(content or '').strip()

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        raw = str(text or '').strip()
        if not raw:
            return None
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(raw):
            if ch != '{':
                continue
            try:
                payload, _ = decoder.raw_decode(raw[idx:])
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _invoke_loop_llm_json(self, prompt: str) -> dict[str, Any] | None:
        if self.loop_llm is None:
            return None
        try:
            if hasattr(self.loop_llm, 'invoke'):
                response = self.loop_llm.invoke(prompt)
            elif hasattr(self.loop_llm, 'ainvoke'):
                response = asyncio.run(self.loop_llm.ainvoke(prompt))
            else:
                return None
        except Exception:
            return None
        text = self._message_text(getattr(response, 'content', response))
        return self._extract_json_object(text)

    @staticmethod
    def _valid_recommended_action(value: Any) -> str:
        text = str(value or '').strip().lower()
        return text if text in _KNOWLEDGE_ACTION_LABELS else ''

    def _request_llm_round_review(
        self,
        *,
        state: dict[str, Any],
        round_record: dict[str, Any],
        round_summary: dict[str, Any],
        analysis: dict[str, Any],
        recommendation: dict[str, Any],
        comparison_previous: dict[str, Any] | None,
        comparison_best: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        payload = {
            'task': 'round_review',
            'managed_level': state.get('managed_level'),
            'target_metric': str((state.get('boundaries') or {}).get('target_metric') or 'map50'),
            'round_index': round_record.get('round_index'),
            'round_summary': {
                'summary': round_summary.get('summary'),
                'run_state': round_summary.get('run_state'),
                'metrics': round_summary.get('metrics') or {},
                'error_lines': list(round_summary.get('error_lines') or [])[:8],
            },
            'analysis': {
                'assessment': analysis.get('assessment'),
                'interpretation': analysis.get('interpretation'),
                'recommendation': analysis.get('recommendation'),
                'signals': list(analysis.get('signals') or [])[:8],
                'facts': list(analysis.get('facts') or [])[:8],
            },
            'recommendation': {
                'recommended_action': recommendation.get('recommended_action'),
                'recommendation': recommendation.get('recommendation'),
                'why': recommendation.get('why'),
                'basis': list(recommendation.get('basis') or [])[:8],
            },
            'comparison_previous': {
                'highlights': list((comparison_previous or {}).get('highlights') or [])[:4],
                'metric_deltas': (comparison_previous or {}).get('metric_deltas') or {},
            } if comparison_previous else None,
            'comparison_best': {
                'highlights': list((comparison_best or {}).get('highlights') or [])[:4],
                'metric_deltas': (comparison_best or {}).get('metric_deltas') or {},
            } if comparison_best else None,
        }
        prompt = (
            '你是 YoloStudio 训练循环复盘器。'
            '只能基于给定事实做轮次复盘，不要补充未验证事实。'
            '输出严格 JSON，不要 markdown，不要解释。\n'
            'JSON schema:\n'
            '{\n'
            '  "review_summary": "一句中文总结",\n'
            '  "recommended_action": "continue_observing|fix_data_quality|collect_metrics_first|inspect_labels_and_mapping|inspect_missed_samples|inspect_false_positives|run_error_analysis",\n'
            '  "why": "一句中文原因",\n'
            '  "carry_forward": ["最多4条下一轮应记住的经验"],\n'
            '  "blockers": ["最多4条阻塞/风险"],\n'
            '  "next_focus": "下一轮最该关注什么",\n'
            '  "confidence": 0.0\n'
            '}\n'
            f'facts={json.dumps(payload, ensure_ascii=False)}'
        )
        result = self._invoke_loop_llm_json(prompt)
        if not isinstance(result, dict):
            return None
        review = {
            'review_summary': str(result.get('review_summary') or '').strip(),
            'recommended_action': self._valid_recommended_action(result.get('recommended_action')),
            'why': str(result.get('why') or '').strip(),
            'carry_forward': [str(item).strip() for item in list(result.get('carry_forward') or []) if str(item).strip()][:4],
            'blockers': [str(item).strip() for item in list(result.get('blockers') or []) if str(item).strip()][:4],
            'next_focus': str(result.get('next_focus') or '').strip(),
        }
        confidence = result.get('confidence')
        if isinstance(confidence, (int, float)):
            review['confidence'] = max(0.0, min(float(confidence), 1.0))
        return review

    def _request_llm_next_round_plan(
        self,
        *,
        state: dict[str, Any],
        round_record: dict[str, Any],
        round_summary: dict[str, Any],
        current_args: dict[str, Any],
        heuristic_change_set: list[dict[str, Any]],
        heuristic_transition_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        payload = {
            'task': 'next_round_plan',
            'managed_level': state.get('managed_level'),
            'target_metric': str((state.get('boundaries') or {}).get('target_metric') or 'map50'),
            'allowed_tuning_params': list((state.get('boundaries') or {}).get('allowed_tuning_params') or []),
            'max_imgsz': (state.get('boundaries') or {}).get('max_imgsz'),
            'min_batch': (state.get('boundaries') or {}).get('min_batch'),
            'round_index': round_record.get('round_index'),
            'current_args': current_args,
            'round_review': round_record.get('round_review') or {},
            'round_memory': round_record.get('round_memory') or {},
            'planner_input': round_record.get('planner_input') or {},
            'run_summary': {
                'summary': round_summary.get('summary'),
                'run_state': round_summary.get('run_state'),
                'metrics': round_summary.get('metrics') or {},
            },
            'heuristic_change_set': heuristic_change_set,
            'heuristic_transition_hint': dict(heuristic_transition_hint or {}),
        }
        prompt = (
            '你是 YoloStudio 训练循环下一轮规划器。'
            '只能在 allowed_tuning_params 内建议保守修改，只能基于给定事实回答。'
            '你还需要决定下一轮是自动继续、进入人工审阅，还是直接停止。'
            '如果没有把握，就输出空的 suggested_param_updates。'
            '输出严格 JSON，不要 markdown，不要解释。\n'
            'JSON schema:\n'
            '{\n'
            '  "plan_summary": "一句中文计划摘要",\n'
            '  "reason": "一句中文原因",\n'
            '  "suggested_decision_type": "auto_continue|await_review|stop",\n'
            '  "suggested_decision_reason": "一句中文决策原因",\n'
            '  "suggested_param_strategy": "keep_current|apply_heuristic",\n'
            '  "suggested_param_updates": [\n'
            '    {"field": "epochs|batch|imgsz|lr0|optimizer", "value": 123, "reason": "为什么这样调"}\n'
            '  ]\n'
            '}\n'
            f'facts={json.dumps(payload, ensure_ascii=False)}'
        )
        result = self._invoke_loop_llm_json(prompt)
        if not isinstance(result, dict):
            return None
        updates: list[dict[str, Any]] = []
        for item in list(result.get('suggested_param_updates') or []):
            if not isinstance(item, dict):
                continue
            field = str(item.get('field') or '').strip()
            if field not in _SUPPORTED_TUNING_PARAMS:
                continue
            updates.append(
                {
                    'field': field,
                    'value': item.get('value'),
                    'reason': str(item.get('reason') or '').strip(),
                }
            )
        suggested_decision_type = str(result.get('suggested_decision_type') or '').strip().lower()
        if suggested_decision_type not in _PLANNER_DECISION_TYPES:
            suggested_decision_type = ''
        suggested_param_strategy = str(result.get('suggested_param_strategy') or '').strip().lower()
        if suggested_param_strategy not in _PLANNER_PARAM_STRATEGIES:
            suggested_param_strategy = ''
        if not (
            updates
            or suggested_decision_type
            or suggested_param_strategy
            or str(result.get('plan_summary') or '').strip()
            or str(result.get('reason') or '').strip()
        ):
            return None
        return {
            'plan_summary': str(result.get('plan_summary') or '').strip(),
            'reason': str(result.get('reason') or '').strip(),
            'suggested_decision_type': suggested_decision_type or None,
            'suggested_decision_reason': str(result.get('suggested_decision_reason') or '').strip(),
            'suggested_param_strategy': suggested_param_strategy or None,
            'suggested_param_updates': updates,
        }

    def start_loop(
        self,
        *,
        model: str,
        data_yaml: str = '',
        epochs: int = _DEFAULT_LOOP_EPOCHS,
        device: str = 'auto',
        training_environment: str = '',
        project: str = '',
        name: str = '',
        batch: int | None = None,
        imgsz: int | None = None,
        fraction: float | None = None,
        classes: list[int] | str | None = None,
        single_cls: bool | None = None,
        optimizer: str = '',
        freeze: int | None = None,
        resume: bool | None = None,
        lr0: float | None = None,
        patience: int | None = None,
        workers: int | None = None,
        amp: bool | None = None,
        loop_name: str = '',
        managed_level: str = 'conservative_auto',
        max_rounds: int = 5,
        target_metric: str = 'map50',
        target_metric_value: float | None = None,
        min_improvement: float = 0.005,
        no_improvement_rounds: int = 2,
        max_failures: int = 2,
        allowed_tuning_params: list[str] | None = None,
        auto_handle_oom: bool = True,
        include_case_sources: bool = False,
        include_test_sources: bool = False,
        max_imgsz: int = 1536,
        min_batch: int = 1,
    ) -> dict[str, Any]:
        with self._state_lock:
            active = self._read_json(self._active_registry_path)
            if active:
                active_loop = self._load_loop(str(active.get('loop_id') or ''))
                if active_loop and str(active_loop.get('status') or '').strip().lower() not in _TERMINAL_LOOP_STATES:
                    loop_id = str(active_loop.get('loop_id') or '')
                    return {
                        'ok': False,
                        'error': f'已有环训练在运行或等待处理中（loop_id={loop_id}）',
                        'summary': '当前已有环训练任务未结束',
                        'active_loop_id': loop_id,
                        'next_actions': [
                            '可先调用 check_training_loop_status 查看当前环训练状态',
                            '如需停止当前环训练，可调用 stop_training_loop',
                        ],
                    }

        normalized_level = self._normalize_managed_level(managed_level)
        normalized_tuning_params = self._normalize_tuning_params(allowed_tuning_params)
        loop_name_text = self._normalize_loop_name(loop_name, model=model, data_yaml=data_yaml)
        base_args = {
            'model': model,
            'data_yaml': data_yaml,
            'epochs': int(epochs),
            'device': device,
            'training_environment': training_environment,
            'project': project,
            'name': name,
            'batch': batch,
            'imgsz': imgsz,
            'fraction': fraction,
            'classes': classes,
            'single_cls': single_cls,
            'optimizer': optimizer,
            'freeze': freeze,
            'resume': resume,
            'lr0': lr0,
            'patience': patience,
            'workers': workers,
            'amp': amp,
        }
        first_round_args = self._build_round_args(base_args, loop_name=loop_name_text, round_index=1)
        preflight = self.train_service.training_preflight(**first_round_args)
        if not preflight.get('ok') or not preflight.get('ready_to_start'):
            blockers = list(preflight.get('blockers') or [])
            summary = preflight.get('summary') or '环训练启动前预检未通过'
            return {
                'ok': False,
                'error': blockers[0] if blockers else summary,
                'summary': summary,
                'preflight': preflight,
                'next_actions': preflight.get('next_actions') or [
                    '请先修复预检中的阻塞项',
                    '修复后再重新启动环训练',
                ],
            }

        now = self._time_fn()
        loop_id = self._generate_loop_id(loop_name_text)
        next_round_plan = self._make_round_plan(
            round_index=1,
            training_args=first_round_args,
            reason='首轮按基线参数启动',
            decision_type='baseline_start',
        )
        state = {
            'state_version': 1,
            'loop_id': loop_id,
            'loop_name': loop_name_text,
            'status': 'queued',
            'summary': f'环训练已创建，准备启动第 1 轮（托管级别：{normalized_level}）',
            'created_at': now,
            'updated_at': now,
            'started_at': None,
            'stopped_at': None,
            'managed_level': normalized_level,
            'base_training_args': base_args,
            'boundaries': {
                'max_rounds': max(1, min(int(max_rounds), 200)),
                'target_metric': self._normalize_target_metric(target_metric),
                'target_metric_value': float(target_metric_value) if target_metric_value is not None else None,
                'min_improvement': max(0.0, float(min_improvement)),
                'no_improvement_rounds': max(1, min(int(no_improvement_rounds), 100)),
                'max_failures': max(1, min(int(max_failures), 10)),
                'allowed_tuning_params': normalized_tuning_params,
                'auto_handle_oom': bool(auto_handle_oom),
                'include_case_sources': bool(include_case_sources),
                'include_test_sources': bool(include_test_sources),
                'max_imgsz': max(320, int(max_imgsz)),
                'min_batch': max(1, int(min_batch)),
            },
            'current_round_index': 0,
            'best_round_index': None,
            'failure_count': 0,
            'no_improvement_streak': 0,
            'pause_requested': False,
            'stop_requested': False,
            'stop_reason': None,
            'termination_reason': None,
            'termination_detail': None,
            'preflight': preflight,
            'rounds': [],
            'next_round_plan': next_round_plan,
        }

        with self._state_lock:
            self._save_loop(state)
        self._spawn_worker_if_needed(loop_id)
        return {
            'ok': True,
            'summary': f'环训练已启动：{loop_name_text}（loop_id={loop_id}）',
            'loop_id': loop_id,
            'loop_name': loop_name_text,
            'status': 'queued',
            'managed_level': normalized_level,
            'boundaries': state['boundaries'],
            'next_round_plan': next_round_plan,
            'next_actions': [
                '可调用 check_training_loop_status 查看当前环训练状态',
                '如需训练完当前轮后停住，可调用 pause_training_loop',
                '如需立即终止整个环训练，可调用 stop_training_loop',
            ],
        }

    def list_loops(self, limit: int = 5) -> dict[str, Any]:
        with self._state_lock:
            loops = self._load_all_loops(limit=max(1, min(int(limit), 20)))
            active = self._read_json(self._active_registry_path) or {}
            active_loop_id = str(active.get('loop_id') or '')

        items: list[dict[str, Any]] = []
        for state in loops:
            items.append(self._build_loop_brief(state, active_loop_id=active_loop_id))
        summary = f'找到 {len(items)} 条环训练记录' if items else '当前没有可用的环训练记录'
        next_actions = ['如需查看某个环训练详情，可调用 inspect_training_loop']
        if active_loop_id:
            next_actions.insert(0, '当前存在活动环训练，可继续调用 check_training_loop_status')
        return {
            'ok': True,
            'summary': summary,
            'active_loop_id': active_loop_id or None,
            'loops': items,
            'next_actions': next_actions,
        }

    def check_loop_status(self, loop_id: str = '') -> dict[str, Any]:
        state = self._resolve_loop(loop_id)
        if not state:
            return {
                'ok': False,
                'error': f'未找到环训练: {loop_id or "active"}',
                'summary': '当前没有可查看的环训练状态',
                'next_actions': ['可先调用 start_training_loop 创建新的环训练'],
            }
        return self._build_status_payload(state, detailed=False)

    def inspect_loop(self, loop_id: str = '') -> dict[str, Any]:
        state = self._resolve_loop(loop_id)
        if not state:
            return {
                'ok': False,
                'error': f'未找到环训练: {loop_id or "active"}',
                'summary': '当前没有可查看的环训练详情',
                'next_actions': ['可先调用 start_training_loop 创建新的环训练'],
            }
        return self._build_status_payload(state, detailed=True)

    def pause_loop(self, loop_id: str = '') -> dict[str, Any]:
        with self._state_lock:
            state = self._resolve_loop(loop_id, use_lock=False)
            if not state:
                return {
                    'ok': False,
                    'error': f'未找到环训练: {loop_id or "active"}',
                    'summary': '暂停失败：未找到对应环训练',
                    'next_actions': ['可先调用 list_training_loops 查看最近记录'],
                }
            status = str(state.get('status') or '').strip().lower()
            if status in _TERMINAL_LOOP_STATES:
                return {
                    'ok': False,
                    'error': f'当前环训练已结束（status={status}）',
                    'summary': '当前环训练已结束，不能再暂停',
                    'loop_id': state.get('loop_id'),
                    'next_actions': ['如需新的环训练，可重新调用 start_training_loop'],
                }
            state['pause_requested'] = True
            if status in {'queued', 'starting_round', 'awaiting_review'} and state.get('next_round_plan'):
                state['status'] = 'paused'
                state['summary'] = f"环训练已暂停，将停在第 {int((state.get('next_round_plan') or {}).get('round_index') or 1)} 轮启动前"
            elif status == 'paused':
                state['summary'] = '环训练已保持暂停状态'
            else:
                round_index = int(state.get('current_round_index') or 0)
                state['summary'] = f'已记录暂停请求：当前第 {round_index} 轮结束后将停住'
            self._save_loop(state)

        return {
            'ok': True,
            'summary': state.get('summary') or '已记录环训练暂停请求',
            'loop_id': state.get('loop_id'),
            'status': state.get('status'),
            'pause_requested': True,
            'next_actions': [
                '可继续调用 check_training_loop_status 查看当前轮状态',
                '如需恢复自动续跑，可调用 resume_training_loop',
            ],
        }

    def resume_loop(self, loop_id: str = '') -> dict[str, Any]:
        with self._state_lock:
            state = self._resolve_loop(loop_id, use_lock=False)
            if not state:
                return {
                    'ok': False,
                    'error': f'未找到环训练: {loop_id or "active"}',
                    'summary': '恢复失败：未找到对应环训练',
                    'next_actions': ['可先调用 list_training_loops 查看最近记录'],
                }
            status = str(state.get('status') or '').strip().lower()
            if status in _TERMINAL_LOOP_STATES:
                return {
                    'ok': False,
                    'error': f'当前环训练已结束（status={status}）',
                    'summary': '当前环训练已结束，不能再恢复',
                    'loop_id': state.get('loop_id'),
                    'next_actions': ['如需新的环训练，可重新调用 start_training_loop'],
                }
            if status == 'running_round':
                return {
                    'ok': True,
                    'summary': '当前环训练本来就在运行，无需恢复',
                    'loop_id': state.get('loop_id'),
                    'status': status,
                    'next_actions': ['可继续调用 check_training_loop_status 查看当前状态'],
                }
            if status in {'queued', 'starting_round'} and not bool(state.get('pause_requested')):
                return {
                    'ok': True,
                    'summary': '当前环训练已在恢复流程中，无需重复恢复',
                    'loop_id': state.get('loop_id'),
                    'status': status,
                    'next_actions': ['可继续调用 check_training_loop_status 查看当前状态'],
                }
            if not state.get('next_round_plan') and status not in {'queued', 'starting_round', 'analyzing_round'}:
                return {
                    'ok': False,
                    'error': '当前没有待执行的下一轮计划',
                    'summary': '恢复失败：当前没有可继续的下一轮',
                    'loop_id': state.get('loop_id'),
                    'next_actions': ['可先调用 inspect_training_loop 查看当前环训练详情'],
                }
            state['pause_requested'] = False
            state['stop_requested'] = False
            if status in _WAITING_LOOP_STATES or status == 'queued':
                state['status'] = 'queued'
                next_round = state.get('next_round_plan') or {}
                next_round_index = int(next_round.get('round_index') or (int(state.get('current_round_index') or 0) + 1))
                state['summary'] = f'环训练已恢复，准备启动第 {next_round_index} 轮'
            self._save_loop(state)

        self._spawn_worker_if_needed(str(state.get('loop_id') or ''))
        return {
            'ok': True,
            'summary': state.get('summary') or '环训练已恢复',
            'loop_id': state.get('loop_id'),
            'status': state.get('status'),
            'next_actions': [
                '可继续调用 check_training_loop_status 查看当前环训练状态',
                '如需再次暂停，可调用 pause_training_loop',
            ],
        }

    def stop_loop(self, loop_id: str = '') -> dict[str, Any]:
        with self._state_lock:
            state = self._resolve_loop(loop_id, use_lock=False)
            if not state:
                return {
                    'ok': False,
                    'error': f'未找到环训练: {loop_id or "active"}',
                    'summary': '停止失败：未找到对应环训练',
                    'next_actions': ['可先调用 list_training_loops 查看最近记录'],
                }
            status = str(state.get('status') or '').strip().lower()
            if status in _TERMINAL_LOOP_STATES:
                return {
                    'ok': False,
                    'error': f'当前环训练已结束（status={status}）',
                    'summary': '当前环训练已结束，无需再停止',
                    'loop_id': state.get('loop_id'),
                    'next_actions': ['如需新的环训练，可重新调用 start_training_loop'],
                }
            state['stop_requested'] = True
            state['pause_requested'] = False
            state['status'] = 'stopping' if status == 'running_round' else 'stopped'
            state['summary'] = '已请求立即终止当前环训练'
            if status != 'running_round':
                state['stop_reason'] = 'manual_stop'
                state['termination_reason'] = 'manual_stop'
                state['termination_detail'] = '用户手动终止'
                state['stopped_at'] = self._time_fn()
                state['next_round_plan'] = None
            self._save_loop(state)

        stop_result: dict[str, Any] | None = None
        if status == 'running_round':
            stop_result = self.train_service.stop()
            self._spawn_worker_if_needed(str(state.get('loop_id') or ''))

        return {
            'ok': True,
            'summary': '当前轮已收到终止请求，环训练即将结束' if status == 'running_round' else '环训练已停止',
            'loop_id': state.get('loop_id'),
            'status': 'stopping' if status == 'running_round' else 'stopped',
            'stop_result': stop_result,
            'next_actions': [
                '可继续调用 check_training_loop_status 确认最终停止状态',
                '如需新的环训练，可重新调用 start_training_loop',
            ],
        }

    def _restore_active_worker_if_needed(self) -> None:
        active = self._read_json(self._active_registry_path)
        if not active:
            return
        loop_id = str(active.get('loop_id') or '')
        if not loop_id:
            return
        state = self._load_loop(loop_id)
        if not state:
            return
        status = str(state.get('status') or '').strip().lower()
        if status in _ACTIVE_LOOP_STATES:
            self._spawn_worker_if_needed(loop_id)

    def _spawn_worker_if_needed(self, loop_id: str) -> None:
        if not loop_id:
            return
        with self._worker_lock:
            if self._worker_thread and self._worker_thread.is_alive() and self._worker_loop_id == loop_id:
                return
            if self._worker_thread and self._worker_thread.is_alive():
                return
            thread = threading.Thread(target=self._worker_main, args=(loop_id,), daemon=True, name=f'train-loop-{loop_id}')
            self._worker_thread = thread
            self._worker_loop_id = loop_id
            thread.start()

    def _worker_main(self, loop_id: str) -> None:
        try:
            while True:
                state = self._resolve_loop(loop_id)
                if not state:
                    return
                status = str(state.get('status') or '').strip().lower()
                if status in _TERMINAL_LOOP_STATES or status in _WAITING_LOOP_STATES:
                    return

                if bool(state.get('stop_requested')) and status != 'running_round':
                    self._finish_loop(state, status='stopped', reason='manual_stop', detail='用户手动终止')
                    return

                if status in {'queued', 'starting_round'}:
                    self._start_next_round(loop_id)
                    continue

                if status == 'running_round':
                    training_status = self.train_service.status()
                    running = bool(training_status.get('running'))
                    if running:
                        self._update_round_runtime(loop_id, training_status)
                        self._sleep_fn(self._poll_interval)
                        continue
                    self._mark_loop_status(loop_id, status='analyzing_round', summary='当前轮训练已结束，正在整理结果与下一轮计划')
                    continue

                if status == 'analyzing_round':
                    self._finalize_current_round(loop_id)
                    continue

                if status == 'stopping':
                    training_status = self.train_service.status()
                    if training_status.get('running'):
                        self._sleep_fn(min(self._poll_interval, 0.5))
                        continue
                    self._mark_loop_status(loop_id, status='analyzing_round', summary='当前轮已停止，正在完成环训练收尾')
                    continue

                self._sleep_fn(self._poll_interval)
        finally:
            with self._worker_lock:
                if self._worker_loop_id == loop_id:
                    self._worker_loop_id = None
                    self._worker_thread = None

    def _start_next_round(self, loop_id: str) -> None:
        state = self._resolve_loop(loop_id)
        if not state:
            return
        plan = dict(state.get('next_round_plan') or {})
        if not plan:
            self._finish_loop(state, status='failed', reason='missing_next_round_plan', detail='缺少下一轮训练计划')
            return
        round_index = int(plan.get('round_index') or (int(state.get('current_round_index') or 0) + 1))
        training_args = self._sanitize_training_args(dict(plan.get('training_args') or {}))
        self._mark_loop_status(loop_id, status='starting_round', summary=f'准备启动第 {round_index} 轮训练')
        try:
            start_result = self.train_service.start(**training_args)
        except Exception as exc:
            start_result = {'ok': False, 'error': f'训练启动异常: {exc}'}
        if not start_result.get('ok'):
            detail = start_result.get('error') or start_result.get('summary') or '训练启动失败'
            round_record = {
                'round_index': round_index,
                'status': 'failed_to_start',
                'started_at': self._time_fn(),
                'stopped_at': self._time_fn(),
                'training_args': training_args,
                'change_set': plan.get('change_set') or [],
                'reason': plan.get('reason'),
                'decision_type': plan.get('decision_type'),
                'planner_source': plan.get('planner_source'),
                'planner_summary': plan.get('planner_summary'),
                'planner_reason': plan.get('planner_reason'),
                'start_error': detail,
            }
            state = self._resolve_loop(loop_id) or state
            state['rounds'] = list(state.get('rounds') or []) + [round_record]
            state['failure_count'] = int(state.get('failure_count') or 0) + 1
            state['next_round_plan'] = None
            self._save_loop(state)
            self._finish_loop(state, status='failed', reason='round_start_failed', detail=detail)
            return

        round_record = {
            'round_index': round_index,
            'status': 'running',
            'started_at': start_result.get('started_at') or self._time_fn(),
            'stopped_at': None,
            'training_args': training_args,
            'effective_args': start_result.get('resolved_args') or training_args,
            'change_set': plan.get('change_set') or [],
            'reason': plan.get('reason'),
            'decision_type': plan.get('decision_type'),
            'experience_context': copy.deepcopy(plan.get('experience_context') or {}),
            'planner_source': plan.get('planner_source'),
            'planner_summary': plan.get('planner_summary'),
            'planner_reason': plan.get('planner_reason'),
            'command': start_result.get('command'),
            'pid': start_result.get('pid'),
            'log_file': start_result.get('log_file'),
            'device': start_result.get('device'),
            'training_environment': start_result.get('training_environment'),
            'start_result': {
                'pid': start_result.get('pid'),
                'device': start_result.get('device'),
                'requested_device': start_result.get('requested_device'),
                'log_file': start_result.get('log_file'),
                'yolo_executable': start_result.get('yolo_executable'),
            },
        }
        state = self._resolve_loop(loop_id) or state
        state['rounds'] = list(state.get('rounds') or []) + [round_record]
        state['current_round_index'] = round_index
        state['next_round_plan'] = None
        if not state.get('started_at'):
            state['started_at'] = start_result.get('started_at') or self._time_fn()
        state['status'] = 'running_round'
        state['summary'] = f'第 {round_index} 轮训练进行中'
        self._save_loop(state)

    def _update_round_runtime(self, loop_id: str, training_status: dict[str, Any]) -> None:
        with self._state_lock:
            state = self._resolve_loop(loop_id, use_lock=False)
            if not state:
                return
            rounds = list(state.get('rounds') or [])
            if not rounds:
                return
            round_record = dict(rounds[-1])
            round_record['live_status'] = {
                'running': training_status.get('running'),
                'summary': training_status.get('summary'),
                'progress': training_status.get('progress'),
                'latest_metrics': training_status.get('latest_metrics'),
                'observed_at': self._time_fn(),
            }
            if training_status.get('pid') is not None:
                round_record['pid'] = training_status.get('pid')
            rounds[-1] = round_record
            state['rounds'] = rounds
            self._save_loop(state)

    def _finalize_current_round(self, loop_id: str) -> None:
        state = self._resolve_loop(loop_id)
        if not state:
            return
        rounds = list(state.get('rounds') or [])
        if not rounds:
            self._finish_loop(state, status='failed', reason='missing_round_record', detail='缺少当前轮记录')
            return
        round_record = dict(rounds[-1])
        log_file = str(round_record.get('log_file') or '')
        round_summary = self.train_service.inspect_training_run(log_file or 'latest')
        if not round_summary.get('ok'):
            detail = round_summary.get('error') or '训练结果读取失败'
            round_record['status'] = 'failed'
            round_record['stopped_at'] = self._time_fn()
            round_record['inspection_error'] = detail
            rounds[-1] = round_record
            state['rounds'] = rounds
            state['failure_count'] = int(state.get('failure_count') or 0) + 1
            self._save_loop(state)
            self._finish_loop(state, status='failed', reason='inspect_training_run_failed', detail=detail)
            return

        analysis = self.knowledge_service.analyze_training_outcome(
            metrics=round_summary,
            model_family='yolo',
            task_type='detection',
            include_case_sources=bool((state.get('boundaries') or {}).get('include_case_sources')),
            include_test_sources=bool((state.get('boundaries') or {}).get('include_test_sources')),
        )
        recommendation = self.knowledge_service.recommend_next_training_step(
            status=round_summary,
            model_family='yolo',
            task_type='detection',
            include_case_sources=bool((state.get('boundaries') or {}).get('include_case_sources')),
            include_test_sources=bool((state.get('boundaries') or {}).get('include_test_sources')),
        )

        previous_round = dict(rounds[-2]) if len(rounds) >= 2 else None
        comparison_previous = None
        if previous_round and previous_round.get('log_file') and round_summary.get('log_file'):
            comparison_previous = self.train_service.compare_training_runs(
                left_run_id=str(round_summary.get('log_file')),
                right_run_id=str(previous_round.get('log_file')),
            )

        previous_best_index = state.get('best_round_index')
        comparison_best = None
        if previous_best_index and int(previous_best_index) != int(round_record.get('round_index') or 0):
            best_round = self._get_round_by_index(rounds, int(previous_best_index))
            if best_round and best_round.get('log_file') and round_summary.get('log_file'):
                comparison_best = self.train_service.compare_training_runs(
                    left_run_id=str(round_summary.get('log_file')),
                    right_run_id=str(best_round.get('log_file')),
                )

        round_record['status'] = str(round_summary.get('run_state') or 'completed')
        round_record['stopped_at'] = round_summary.get('stopped_at') or self._time_fn()
        round_record['summary'] = round_summary.get('summary')
        round_record['run_summary'] = round_summary
        round_record['analysis'] = analysis
        round_record['recommendation'] = recommendation
        round_record['comparison_to_previous'] = comparison_previous
        round_record['comparison_to_best'] = comparison_best
        rounds[-1] = round_record
        state['rounds'] = rounds

        best_round_index = self._choose_best_round_index(rounds, target_metric=str((state.get('boundaries') or {}).get('target_metric') or 'map50'))
        prior_best_metric = None
        if previous_best_index:
            best_before = self._get_round_by_index(rounds[:-1], int(previous_best_index))
            prior_best_metric = self._metric_from_round(best_before, str((state.get('boundaries') or {}).get('target_metric') or 'map50'))
        current_metric = self._metric_from_summary(round_summary, str((state.get('boundaries') or {}).get('target_metric') or 'map50'))
        improved = False
        if current_metric is not None:
            if prior_best_metric is None:
                improved = True
            else:
                improved = (current_metric - prior_best_metric) >= float((state.get('boundaries') or {}).get('min_improvement') or 0.0)
        if best_round_index is not None:
            state['best_round_index'] = best_round_index
        if improved:
            state['no_improvement_streak'] = 0
        elif int(round_record.get('round_index') or 0) > 1:
            state['no_improvement_streak'] = int(state.get('no_improvement_streak') or 0) + 1

        if str(round_summary.get('run_state') or '').lower() in {'failed', 'stopped'}:
            state['failure_count'] = int(state.get('failure_count') or 0) + 1

        llm_round_review = self._request_llm_round_review(
            state=state,
            round_record=round_record,
            round_summary=round_summary,
            analysis=analysis,
            recommendation=recommendation,
            comparison_previous=comparison_previous,
            comparison_best=comparison_best,
        )
        round_review = self._build_round_review(
            state=state,
            round_record=round_record,
            round_summary=round_summary,
            analysis=analysis,
            recommendation=recommendation,
            comparison_previous=comparison_previous,
            comparison_best=comparison_best,
            llm_review=llm_round_review,
        )
        round_record['round_review'] = round_review
        round_record['planner_input'] = self._build_planner_input(
            state=state,
            round_record=round_record,
            round_summary=round_summary,
            round_review=round_review,
        )
        rounds[-1] = round_record
        state['rounds'] = rounds

        decision = self._decide_next_step(
            state=state,
            round_record=round_record,
            round_summary=round_summary,
            analysis=analysis,
            recommendation=recommendation,
        )
        round_record['decision'] = decision
        planner_output = self._build_planner_output(round_record=round_record, decision=decision)
        round_record['planner_output'] = planner_output
        round_record['round_memory'] = self._build_round_memory(
            state=state,
            round_record=round_record,
            round_summary=round_summary,
            round_review=round_review,
            planner_output=planner_output,
        )
        next_round_plan = dict(decision.get('next_round_plan') or {})
        if next_round_plan:
            experience_context = dict(next_round_plan.get('experience_context') or {})
            recent_round_memory = list(experience_context.get('recent_round_memory') or [])
            recent_round_memory.append(
                {
                    'round_index': round_record.get('round_index'),
                    'target_metric_value': round_record['round_memory'].get('target_metric_value'),
                    'recommended_action': round_record['round_memory'].get('recommended_action'),
                    'next_focus': round_record['round_memory'].get('next_focus'),
                    'decision_type': round_record['round_memory'].get('decision_type'),
                    'change_set': list(round_record['round_memory'].get('change_set') or [])[:4],
                }
            )
            experience_context['recent_round_memory'] = recent_round_memory[-3:]
            next_round_plan['experience_context'] = experience_context
            decision['next_round_plan'] = next_round_plan
        rounds[-1] = round_record
        state['rounds'] = rounds
        self._save_loop(state)

        decision_type = str(decision.get('decision_type') or '')
        if decision_type == 'stop':
            self._finish_loop(
                state,
                status=str(decision.get('final_status') or 'completed'),
                reason=str(decision.get('reason_code') or 'completed'),
                detail=str(decision.get('reason') or '环训练已结束'),
            )
            return

        next_round_plan = decision.get('next_round_plan')
        if not isinstance(next_round_plan, dict):
            self._finish_loop(state, status='failed', reason='missing_next_round_plan', detail='下一轮计划为空')
            return

        with self._state_lock:
            latest = self._resolve_loop(loop_id, use_lock=False) or state
            latest['next_round_plan'] = next_round_plan
            latest['summary'] = str(decision.get('reason') or f'第 {int(round_record.get("round_index") or 0)} 轮已完成')
            if bool(latest.get('pause_requested')):
                latest['status'] = 'paused'
            elif decision_type == 'await_review':
                latest['status'] = 'awaiting_review'
            else:
                latest['status'] = 'queued'
            self._save_loop(latest)

    def _decide_next_step(
        self,
        *,
        state: dict[str, Any],
        round_record: dict[str, Any],
        round_summary: dict[str, Any],
        analysis: dict[str, Any],
        recommendation: dict[str, Any],
    ) -> dict[str, Any]:
        boundaries = dict(state.get('boundaries') or {})
        target_metric = str(boundaries.get('target_metric') or 'map50')
        round_index = int(round_record.get('round_index') or 0)
        max_rounds = int(boundaries.get('max_rounds') or 1)
        target_metric_value = boundaries.get('target_metric_value')
        current_metric = self._metric_from_summary(round_summary, target_metric)
        managed_level = str(state.get('managed_level') or 'conservative_auto')
        recommended_action = str(recommendation.get('recommended_action') or analysis.get('assessment') or '').strip().lower()

        if bool(state.get('stop_requested')):
            return {
                'decision_type': 'stop',
                'final_status': 'stopped',
                'reason_code': 'manual_stop',
                'reason': '用户已手动终止环训练',
            }

        run_state = str(round_summary.get('run_state') or '').strip().lower()
        if run_state in {'failed', 'stopped'}:
            if bool(boundaries.get('auto_handle_oom')) and self._looks_like_oom(round_summary):
                retry_plan = self._build_next_round_plan(
                    state=state,
                    round_record=round_record,
                    round_summary=round_summary,
                    reason='检测到显存不足，已准备降低 batch/imgsz 后重试',
                    decision_type='retry_adjusted',
                    force_mode='oom',
                )
                if retry_plan:
                    if managed_level == 'full_auto':
                        return {
                            'decision_type': 'auto_continue',
                            'reason': '检测到显存不足，已自动生成降载重试计划',
                            'next_round_plan': retry_plan,
                        }
                    return {
                        'decision_type': 'await_review',
                        'reason': '检测到显存不足，已生成降载重试计划，等待确认',
                        'next_round_plan': retry_plan,
                    }
            if int(state.get('failure_count') or 0) >= int(boundaries.get('max_failures') or 1):
                return {
                    'decision_type': 'stop',
                    'final_status': 'failed',
                    'reason_code': 'too_many_failures',
                    'reason': f'连续失败次数已达到上限（{state.get("failure_count")}）',
                }
            return {
                'decision_type': 'stop',
                'final_status': 'failed',
                'reason_code': 'round_failed',
                'reason': '当前轮训练未正常完成，已停止环训练等待人工处理',
            }

        if target_metric_value is not None and current_metric is not None and current_metric >= float(target_metric_value):
            return {
                'decision_type': 'stop',
                'final_status': 'completed',
                'reason_code': 'target_metric_reached',
                'reason': f'已达到目标指标：{target_metric}={current_metric:.4f}',
            }

        if round_index >= max_rounds:
            return {
                'decision_type': 'stop',
                'final_status': 'completed',
                'reason_code': 'max_rounds_reached',
                'reason': f'已达到最大轮数 {max_rounds}',
            }

        if recommended_action in _HARD_STOP_RECOMMENDED_ACTIONS:
            return {
                'decision_type': 'stop',
                'final_status': 'stopped',
                'reason_code': recommended_action or 'manual_intervention_required',
                'reason': str(recommendation.get('recommendation') or analysis.get('recommendation') or '当前更适合先人工分析，再决定下一轮'),
            }

        default_decision_type = 'auto_continue'
        default_reason = '下一轮计划已通过轮间闸门，准备自动继续'
        default_final_status = ''
        default_reason_code = ''
        plan_reason = str(recommendation.get('recommendation') or analysis.get('recommendation') or '继续下一轮小步试探')
        plan_decision_type = 'tune_next_round'

        if int(state.get('no_improvement_streak') or 0) >= int(boundaries.get('no_improvement_rounds') or 1):
            default_decision_type = 'stop'
            default_final_status = 'completed'
            default_reason_code = 'no_improvement'
            default_reason = f'连续 {state.get("no_improvement_streak")} 轮提升不足，默认停止自动续跑'
            plan_reason = '指标已连续多轮提升不足，先生成保守下一轮计划，再由规划器决定是否继续'
            plan_decision_type = 'plateau_reassessment'
        elif recommended_action in _REVIEW_REQUIRED_ACTIONS:
            default_decision_type = 'auto_continue' if managed_level == 'full_auto' else 'await_review'
            default_reason = (
                '当前托管级别为全托管，已记录需要人工复盘的结论并继续下一轮'
                if managed_level == 'full_auto'
                else '知识策略建议先人工分析，已停在轮间闸门等待确认'
            )
            plan_reason = str(recommendation.get('recommendation') or analysis.get('recommendation') or '当前更适合先人工分析，再决定下一轮')
            plan_decision_type = 'knowledge_review_gate'

        next_round_plan = self._build_next_round_plan(
            state=state,
            round_record=round_record,
            round_summary=round_summary,
            reason=plan_reason,
            decision_type=plan_decision_type,
            heuristic_transition_hint={
                'default_decision_type': default_decision_type,
                'default_reason': default_reason,
                'default_final_status': default_final_status,
                'default_reason_code': default_reason_code,
            },
        )
        if not next_round_plan:
            if default_decision_type == 'stop':
                return {
                    'decision_type': 'stop',
                    'final_status': default_final_status or 'completed',
                    'reason_code': default_reason_code or 'no_safe_next_step',
                    'reason': default_reason,
                }
            return {
                'decision_type': 'stop',
                'final_status': 'stopped',
                'reason_code': 'no_safe_next_step',
                'reason': '当前没有安全的自动下一轮计划，已停止等待人工介入',
            }
        return self._resolve_model_led_round_transition(
            state=state,
            next_round_plan=next_round_plan,
            default_reason=default_reason,
            default_decision_type=default_decision_type,
            default_final_status=default_final_status,
            default_reason_code=default_reason_code,
        )

    @staticmethod
    def _planner_transition_hint(next_round_plan: dict[str, Any]) -> tuple[str, str]:
        decision_type = str(next_round_plan.get('planner_decision_type') or '').strip().lower()
        if decision_type not in _PLANNER_DECISION_TYPES:
            return '', ''
        reason = str(next_round_plan.get('planner_decision_reason') or '').strip()
        return decision_type, reason

    def _resolve_model_led_round_transition(
        self,
        *,
        state: dict[str, Any],
        next_round_plan: dict[str, Any],
        default_reason: str,
        default_decision_type: str,
        default_final_status: str = '',
        default_reason_code: str = '',
    ) -> dict[str, Any]:
        if bool(state.get('pause_requested')):
            return {
                'decision_type': 'paused',
                'reason': '已按请求在当前轮结束后停住',
                'next_round_plan': next_round_plan,
            }

        planner_decision_type, planner_decision_reason = self._planner_transition_hint(next_round_plan)
        if planner_decision_type == 'stop':
            return {
                'decision_type': 'stop',
                'final_status': 'stopped',
                'reason_code': 'planner_stop',
                'reason': planner_decision_reason or default_reason or '规划器建议先停下当前环训练',
            }

        managed_level = str(state.get('managed_level') or 'conservative_auto')
        if managed_level == 'review':
            return {
                'decision_type': 'await_review',
                'reason': '当前托管级别为审阅模式，下一轮等待确认',
                'next_round_plan': next_round_plan,
            }

        if managed_level == 'conservative_auto' and self._requires_review(next_round_plan):
            return {
                'decision_type': 'await_review',
                'reason': '下一轮变更幅度偏大，已停在轮间闸门等待确认',
                'next_round_plan': next_round_plan,
            }

        if planner_decision_type == 'await_review':
            return {
                'decision_type': 'await_review',
                'reason': planner_decision_reason or default_reason,
                'next_round_plan': next_round_plan,
            }
        if planner_decision_type == 'auto_continue':
            return {
                'decision_type': 'auto_continue',
                'reason': planner_decision_reason or default_reason,
                'next_round_plan': next_round_plan,
            }

        if default_decision_type == 'stop':
            return {
                'decision_type': 'stop',
                'final_status': default_final_status or 'completed',
                'reason_code': default_reason_code or 'model_led_default_stop',
                'reason': default_reason,
            }

        return {
            'decision_type': default_decision_type,
            'reason': default_reason,
            'next_round_plan': next_round_plan,
        }

    def _build_next_round_plan(
        self,
        *,
        state: dict[str, Any],
        round_record: dict[str, Any],
        round_summary: dict[str, Any],
        reason: str,
        decision_type: str,
        force_mode: str = '',
        heuristic_transition_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        current_args = self._sanitize_training_args(dict(round_record.get('effective_args') or round_record.get('training_args') or {}))
        next_args = copy.deepcopy(current_args)
        boundaries = dict(state.get('boundaries') or {})
        allowed = set(str(item) for item in boundaries.get('allowed_tuning_params') or [])
        target_metric = str(boundaries.get('target_metric') or 'map50')
        planner_source = 'heuristic'
        planner_summary = ''
        planner_reason = ''
        planner_decision_type = ''
        planner_decision_reason = ''

        if force_mode == 'oom':
            changed = self._apply_oom_adjustment(
                next_args,
                allowed=allowed,
                min_batch=int(boundaries.get('min_batch') or 1),
                max_imgsz=int(boundaries.get('max_imgsz') or 1536),
            )
            if not changed:
                return None
        else:
            recommended_action = str((round_record.get('recommendation') or {}).get('recommended_action') or '').strip().lower()
            current_metric = self._metric_from_summary(round_summary, target_metric)
            prior_best_metric = None
            best_round = self._get_round_by_index(list(state.get('rounds') or []), int(state.get('best_round_index') or 0))
            if best_round and int(best_round.get('round_index') or 0) != int(round_record.get('round_index') or 0):
                prior_best_metric = self._metric_from_round(best_round, target_metric)

            changed = False
            heuristic_args = copy.deepcopy(next_args)
            if recommended_action == 'continue_observing' and 'epochs' in allowed:
                changed = self._increase_epochs(heuristic_args)
            elif current_metric is not None and prior_best_metric is not None and current_metric <= prior_best_metric and 'lr0' in allowed:
                changed = self._reduce_lr(heuristic_args)
            elif 'epochs' in allowed:
                changed = self._increase_epochs(heuristic_args)
            elif 'imgsz' in allowed:
                changed = self._increase_imgsz(heuristic_args, max_imgsz=int(boundaries.get('max_imgsz') or 1536))
            heuristic_change_set = self._build_change_set(current_args, heuristic_args)
            llm_plan = self._request_llm_next_round_plan(
                state=state,
                round_record=round_record,
                round_summary=round_summary,
                current_args=current_args,
                heuristic_change_set=heuristic_change_set,
                heuristic_transition_hint=heuristic_transition_hint,
            )
            if llm_plan:
                llm_changed = self._apply_llm_param_updates(
                    next_args,
                    current_args=current_args,
                    updates=list(llm_plan.get('suggested_param_updates') or []),
                    allowed=allowed,
                    min_batch=int(boundaries.get('min_batch') or 1),
                    max_imgsz=int(boundaries.get('max_imgsz') or 1536),
                )
                if llm_changed:
                    changed = True
                    planner_source = 'llm'
                    planner_summary = str(llm_plan.get('plan_summary') or '').strip()
                    planner_reason = str(llm_plan.get('reason') or '').strip()
                    planner_decision_type = str(llm_plan.get('suggested_decision_type') or '').strip().lower()
                    planner_decision_reason = str(llm_plan.get('suggested_decision_reason') or '').strip()
                else:
                    planner_source = 'llm'
                    planner_summary = str(llm_plan.get('plan_summary') or '').strip()
                    planner_reason = str(llm_plan.get('reason') or '').strip()
                    planner_decision_type = str(llm_plan.get('suggested_decision_type') or '').strip().lower()
                    planner_decision_reason = str(llm_plan.get('suggested_decision_reason') or '').strip()
                    planner_param_strategy = str(llm_plan.get('suggested_param_strategy') or '').strip().lower()
                    if planner_param_strategy == 'apply_heuristic':
                        next_args = heuristic_args
                    else:
                        next_args = copy.deepcopy(current_args)
            else:
                next_args = heuristic_args

        next_round_index = int(round_record.get('round_index') or 0) + 1
        next_args = self._build_round_args(
            next_args,
            loop_name=str(state.get('loop_name') or ''),
            round_index=next_round_index,
            preserve_project=True,
        )
        change_set = self._build_change_set(current_args, next_args)
        experience_context = self._build_experience_context(state)
        current_round_review = dict(round_record.get('round_review') or {})
        if current_round_review:
            experience_context['current_round_review'] = copy.deepcopy(current_round_review)
        return self._make_round_plan(
            round_index=next_round_index,
            training_args=next_args,
            reason=reason,
            decision_type=decision_type,
            change_set=change_set,
            experience_context=experience_context,
            planner_source=planner_source,
            planner_summary=planner_summary,
            planner_reason=planner_reason,
            planner_decision_type=planner_decision_type,
            planner_decision_reason=planner_decision_reason,
        )

    @classmethod
    def _build_round_review(
        cls,
        *,
        state: dict[str, Any],
        round_record: dict[str, Any],
        round_summary: dict[str, Any],
        analysis: dict[str, Any],
        recommendation: dict[str, Any],
        comparison_previous: dict[str, Any] | None,
        comparison_best: dict[str, Any] | None,
        llm_review: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target_metric = str((state.get('boundaries') or {}).get('target_metric') or 'map50')
        normalized_action = cls._valid_recommended_action((llm_review or {}).get('recommended_action'))
        review = {
            'round_index': round_record.get('round_index'),
            'summary': str((llm_review or {}).get('review_summary') or round_summary.get('summary') or round_record.get('summary') or '').strip(),
            'run_state': round_summary.get('run_state'),
            'target_metric': target_metric,
            'target_metric_value': cls._metric_from_summary(round_summary, target_metric),
            'recommended_action': normalized_action or recommendation.get('recommended_action') or analysis.get('assessment'),
            'recommendation': recommendation.get('recommendation') or analysis.get('recommendation'),
            'why': str((llm_review or {}).get('why') or recommendation.get('why') or analysis.get('interpretation') or '').strip(),
            'signals': list(recommendation.get('signals') or analysis.get('signals') or [])[:6],
            'basis': list(recommendation.get('basis') or analysis.get('facts') or [])[:6],
            'matched_rule_ids': list(recommendation.get('matched_rule_ids') or analysis.get('matched_rule_ids') or [])[:6],
            'review_source': 'llm' if llm_review else 'heuristic',
        }
        carry_forward = [str(item).strip() for item in list((llm_review or {}).get('carry_forward') or []) if str(item).strip()]
        blockers = [str(item).strip() for item in list((llm_review or {}).get('blockers') or []) if str(item).strip()]
        next_focus = str((llm_review or {}).get('next_focus') or '').strip()
        if carry_forward:
            review['carry_forward'] = carry_forward[:4]
        if blockers:
            review['blockers'] = blockers[:4]
        if next_focus:
            review['next_focus'] = next_focus
        confidence = (llm_review or {}).get('confidence')
        if isinstance(confidence, (int, float)):
            review['confidence'] = max(0.0, min(float(confidence), 1.0))
        if comparison_previous:
            review['vs_previous'] = {
                'summary': comparison_previous.get('summary'),
                'highlights': list(comparison_previous.get('highlights') or [])[:4],
                'metric_deltas': comparison_previous.get('metric_deltas') or {},
            }
        if comparison_best:
            review['vs_best'] = {
                'summary': comparison_best.get('summary'),
                'highlights': list(comparison_best.get('highlights') or [])[:4],
                'metric_deltas': comparison_best.get('metric_deltas') or {},
            }
        return review

    def _build_planner_input(
        self,
        *,
        state: dict[str, Any],
        round_record: dict[str, Any],
        round_summary: dict[str, Any],
        round_review: dict[str, Any],
    ) -> dict[str, Any]:
        experience_context = self._build_experience_context(state)
        return {
            'round_index': round_record.get('round_index'),
            'managed_level': state.get('managed_level'),
            'target_metric': str((state.get('boundaries') or {}).get('target_metric') or 'map50'),
            'target_metric_value': round_review.get('target_metric_value'),
            'run_state': round_summary.get('run_state'),
            'change_set': list(round_record.get('change_set') or []),
            'failure_count': state.get('failure_count'),
            'no_improvement_streak': state.get('no_improvement_streak'),
            'best_round_index': state.get('best_round_index'),
            'best_target_metric': self._metric_from_round(
                self._get_round_by_index(list(state.get('rounds') or []), int(state.get('best_round_index') or 0)),
                str((state.get('boundaries') or {}).get('target_metric') or 'map50'),
            ),
            'recommended_action': round_review.get('recommended_action'),
            'experience_context': experience_context,
            'recent_round_memory': list(experience_context.get('recent_round_memory') or []),
        }

    @staticmethod
    def _build_planner_output(*, round_record: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        next_round_plan = dict(decision.get('next_round_plan') or {})
        return {
            'round_index': round_record.get('round_index'),
            'decision_type': decision.get('decision_type'),
            'decision_reason': decision.get('reason'),
            'reason_code': decision.get('reason_code'),
            'final_status': decision.get('final_status'),
            'next_round_index': next_round_plan.get('round_index'),
            'next_change_set': list(next_round_plan.get('change_set') or [])[:4],
            'planner_source': next_round_plan.get('planner_source') or round_record.get('planner_source') or 'heuristic',
            'planner_summary': next_round_plan.get('planner_summary') or round_record.get('planner_summary'),
            'planner_reason': next_round_plan.get('planner_reason') or round_record.get('planner_reason'),
            'requires_review': str(decision.get('decision_type') or '') in {'await_review', 'paused'},
        }

    @classmethod
    def _build_round_memory(
        cls,
        *,
        state: dict[str, Any],
        round_record: dict[str, Any],
        round_summary: dict[str, Any],
        round_review: dict[str, Any],
        planner_output: dict[str, Any],
    ) -> dict[str, Any]:
        recommended_action = str(round_review.get('recommended_action') or '').strip().lower()
        carry_forward: list[str] = [str(item).strip() for item in list(round_review.get('carry_forward') or []) if str(item).strip()]
        change_fields = [str((item or {}).get('field') or '').strip() for item in list(round_record.get('change_set') or []) if str((item or {}).get('field') or '').strip()]
        if change_fields:
            carry_forward.append(f'本轮已验证参数变更: {", ".join(change_fields[:4])}')
        if round_review.get('why'):
            carry_forward.append(str(round_review.get('why')))
        vs_previous = dict(round_review.get('vs_previous') or {})
        if vs_previous.get('highlights'):
            carry_forward.extend([str(item).strip() for item in list(vs_previous.get('highlights') or [])[:2] if str(item).strip()])
        blockers: list[str] = [str(item).strip() for item in list(round_review.get('blockers') or []) if str(item).strip()]
        if cls._looks_like_oom(round_summary):
            blockers.append('本轮出现显存不足，需要优先降载或调整资源。')
        if str(round_summary.get('run_state') or '').strip().lower() in {'failed', 'stopped'}:
            blockers.append(str(round_summary.get('summary') or round_record.get('summary') or '当前轮未正常完成').strip())
        memory = {
            'round_index': round_record.get('round_index'),
            'summary': round_summary.get('summary') or round_record.get('summary'),
            'target_metric': str((state.get('boundaries') or {}).get('target_metric') or 'map50'),
            'target_metric_value': round_review.get('target_metric_value'),
            'recommended_action': recommended_action or None,
            'next_focus': str(round_review.get('next_focus') or '').strip() or cls._knowledge_action_label(recommended_action),
            'change_set': list(round_record.get('change_set') or []),
            'carry_forward': carry_forward[:4],
            'blockers': blockers[:4],
            'decision_type': planner_output.get('decision_type'),
            'decision_reason': planner_output.get('decision_reason'),
        }
        return memory

    def _build_experience_context(self, state: dict[str, Any], *, limit: int = 3) -> dict[str, Any]:
        rounds = list(state.get('rounds') or [])
        recent_round_memory: list[dict[str, Any]] = []
        for item in rounds:
            memory = dict(item.get('round_memory') or {})
            if not memory:
                continue
            recent_round_memory.append(
                {
                    'round_index': item.get('round_index'),
                    'target_metric_value': memory.get('target_metric_value'),
                    'recommended_action': memory.get('recommended_action'),
                    'next_focus': memory.get('next_focus'),
                    'decision_type': memory.get('decision_type'),
                    'change_set': list(memory.get('change_set') or [])[:4],
                }
            )
        return {
            'target_metric': str((state.get('boundaries') or {}).get('target_metric') or 'map50'),
            'best_round_index': state.get('best_round_index'),
            'recent_round_memory': recent_round_memory[-limit:],
        }

    @classmethod
    def _apply_llm_param_updates(
        cls,
        args: dict[str, Any],
        *,
        current_args: dict[str, Any],
        updates: list[dict[str, Any]],
        allowed: set[str],
        min_batch: int,
        max_imgsz: int,
    ) -> bool:
        changed = False
        for item in updates:
            if not isinstance(item, dict):
                continue
            field = str(item.get('field') or '').strip()
            if field not in allowed:
                continue
            value = item.get('value')
            current_value = current_args.get(field)
            if field == 'epochs':
                if isinstance(value, (int, float)):
                    normalized = max(1, min(500, int(value)))
                    if normalized != current_value:
                        args[field] = normalized
                        changed = True
            elif field == 'batch':
                if isinstance(value, (int, float)):
                    normalized = max(min_batch, int(value))
                    if normalized != current_value:
                        args[field] = normalized
                        changed = True
            elif field == 'imgsz':
                if isinstance(value, (int, float)):
                    normalized = min(max_imgsz, cls._normalize_imgsz_value(int(value)))
                    if normalized != current_value:
                        args[field] = normalized
                        changed = True
            elif field == 'lr0':
                if isinstance(value, (int, float)) and float(value) > 0:
                    normalized = round(float(value), 6)
                    if current_value is None or abs(float(current_value) - normalized) >= 1e-9:
                        args[field] = normalized
                        changed = True
            elif field == 'optimizer':
                text = str(value or '').strip()
                if text and text != str(current_value or '').strip():
                    args[field] = text
                    changed = True
        return changed

    @staticmethod
    def _apply_oom_adjustment(args: dict[str, Any], *, allowed: set[str], min_batch: int, max_imgsz: int) -> bool:
        batch = args.get('batch')
        if 'batch' in allowed and isinstance(batch, int) and batch > min_batch:
            args['batch'] = max(min_batch, max(1, batch // 2))
            return args['batch'] != batch
        imgsz = args.get('imgsz')
        if 'imgsz' in allowed and isinstance(imgsz, int) and imgsz > 320:
            reduced = max(320, min(max_imgsz, TrainingLoopService._normalize_imgsz_value(int(imgsz * 0.75))))
            if reduced != imgsz:
                args['imgsz'] = reduced
                return True
        return False

    @staticmethod
    def _increase_epochs(args: dict[str, Any]) -> bool:
        epochs = args.get('epochs')
        if not isinstance(epochs, int) or epochs <= 0:
            return False
        max_epochs = 500
        if epochs >= max_epochs:
            return False
        delta = max(10, min(50, int(round(epochs * 0.2))))
        args['epochs'] = min(max_epochs, epochs + delta)
        return args['epochs'] != epochs

    @staticmethod
    def _reduce_lr(args: dict[str, Any]) -> bool:
        lr0 = args.get('lr0')
        if not isinstance(lr0, (int, float)) or float(lr0) <= 0:
            return False
        lowered = round(float(lr0) * 0.5, 6)
        if lowered <= 0 or abs(lowered - float(lr0)) < 1e-9:
            return False
        args['lr0'] = lowered
        return True

    @classmethod
    def _increase_imgsz(cls, args: dict[str, Any], *, max_imgsz: int) -> bool:
        imgsz = args.get('imgsz')
        if not isinstance(imgsz, int) or imgsz <= 0:
            return False
        raised = min(max_imgsz, cls._normalize_imgsz_value(int(round(imgsz * 1.25))))
        if raised == imgsz:
            return False
        args['imgsz'] = raised
        return True

    @staticmethod
    def _normalize_imgsz_value(value: int) -> int:
        return max(320, int(round(value / 32.0)) * 32)

    def _requires_review(self, next_round_plan: dict[str, Any]) -> bool:
        change_set = list(next_round_plan.get('change_set') or [])
        if not change_set:
            return False
        if len(change_set) >= 3:
            return True
        for item in change_set:
            field = str(item.get('field') or '')
            old = item.get('old')
            new = item.get('new')
            if field == 'optimizer':
                return True
            if field == 'batch' and isinstance(old, int) and isinstance(new, int) and old > 0:
                if abs(new - old) / float(old) >= 0.5:
                    return True
            if field == 'imgsz' and isinstance(old, int) and isinstance(new, int) and old > 0:
                if abs(new - old) / float(old) >= 0.25:
                    return True
            if field == 'epochs' and isinstance(old, int) and isinstance(new, int):
                if new - old >= max(20, int(round(old * 0.5))):
                    return True
            if field == 'lr0' and isinstance(old, (int, float)) and isinstance(new, (int, float)) and old > 0:
                if abs(float(new) - float(old)) / float(old) >= 0.5:
                    return True
        return False

    @staticmethod
    def _make_round_plan(
        *,
        round_index: int,
        training_args: dict[str, Any],
        reason: str,
        decision_type: str,
        change_set: list[dict[str, Any]] | None = None,
        experience_context: dict[str, Any] | None = None,
        planner_source: str = 'heuristic',
        planner_summary: str = '',
        planner_reason: str = '',
        planner_decision_type: str = '',
        planner_decision_reason: str = '',
    ) -> dict[str, Any]:
        return {
            'round_index': int(round_index),
            'training_args': copy.deepcopy(training_args),
            'reason': reason,
            'decision_type': decision_type,
            'change_set': list(change_set or []),
            'experience_context': copy.deepcopy(experience_context or {}),
            'planner_source': planner_source,
            'planner_summary': planner_summary,
            'planner_reason': planner_reason,
            'planner_decision_type': planner_decision_type,
            'planner_decision_reason': planner_decision_reason,
        }

    def _build_round_args(
        self,
        base_args: dict[str, Any],
        *,
        loop_name: str,
        round_index: int,
        preserve_project: bool = False,
    ) -> dict[str, Any]:
        args = copy.deepcopy(base_args)
        args['name'] = f'{loop_name}-r{round_index}'
        if preserve_project:
            args['project'] = args.get('project') or ''
        return args

    @staticmethod
    def _sanitize_training_args(args: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in dict(args or {}).items() if key in _TRAINING_ARG_KEYS}

    @staticmethod
    def _build_change_set(old_args: dict[str, Any], new_args: dict[str, Any]) -> list[dict[str, Any]]:
        interesting_fields = ['epochs', 'batch', 'imgsz', 'lr0', 'optimizer', 'name']
        changes: list[dict[str, Any]] = []
        for field in interesting_fields:
            old_value = old_args.get(field)
            new_value = new_args.get(field)
            if old_value != new_value:
                changes.append({'field': field, 'old': old_value, 'new': new_value})
        return changes

    @staticmethod
    def _looks_like_oom(round_summary: dict[str, Any]) -> bool:
        texts = [str(item) for item in (round_summary.get('error_lines') or [])]
        texts.append(str(round_summary.get('summary') or ''))
        combined = '\n'.join(texts).lower()
        return 'out of memory' in combined or 'cuda oom' in combined or 'cuda out of memory' in combined

    def _finish_loop(self, state: dict[str, Any], *, status: str, reason: str, detail: str) -> None:
        with self._state_lock:
            latest = self._resolve_loop(str(state.get('loop_id') or ''), use_lock=False) or state
            latest['status'] = status
            latest['summary'] = detail
            latest['stop_reason'] = reason
            latest['termination_reason'] = reason
            latest['termination_detail'] = detail
            latest['stopped_at'] = self._time_fn()
            latest['pause_requested'] = False
            latest['stop_requested'] = False
            if status in _TERMINAL_LOOP_STATES:
                latest['next_round_plan'] = None
            self._save_loop(latest)

    def _mark_loop_status(self, loop_id: str, *, status: str, summary: str) -> None:
        with self._state_lock:
            state = self._resolve_loop(loop_id, use_lock=False)
            if not state:
                return
            state['status'] = status
            state['summary'] = summary
            self._save_loop(state)

    def _build_status_payload(self, state: dict[str, Any], *, detailed: bool) -> dict[str, Any]:
        rounds = list(state.get('rounds') or [])
        current_round = self._get_round_by_index(rounds, int(state.get('current_round_index') or 0))
        best_round = self._get_round_by_index(rounds, int(state.get('best_round_index') or 0))
        latest_finished_round = self._find_latest_finished_round(rounds)
        completed_rounds = sum(1 for item in rounds if str(item.get('status') or '').strip().lower() not in {'', 'running'})
        payload: dict[str, Any] = {
            'ok': True,
            'summary': state.get('summary') or '环训练状态已就绪',
            'loop_id': state.get('loop_id'),
            'loop_name': state.get('loop_name'),
            'status': state.get('status'),
            'managed_level': state.get('managed_level'),
            'current_round_index': state.get('current_round_index'),
            'completed_rounds': completed_rounds,
            'recorded_rounds': len(rounds),
            'max_rounds': (state.get('boundaries') or {}).get('max_rounds'),
            'best_round_index': state.get('best_round_index'),
            'best_target_metric': self._metric_from_round(best_round, str((state.get('boundaries') or {}).get('target_metric') or 'map50')),
            'failure_count': state.get('failure_count'),
            'no_improvement_streak': state.get('no_improvement_streak'),
            'pause_requested': state.get('pause_requested'),
            'stop_requested': state.get('stop_requested'),
            'stop_reason': state.get('stop_reason'),
            'termination_reason': state.get('termination_reason'),
            'termination_detail': state.get('termination_detail'),
            'created_at': state.get('created_at'),
            'started_at': state.get('started_at'),
            'updated_at': state.get('updated_at'),
            'stopped_at': state.get('stopped_at'),
            'boundaries': state.get('boundaries'),
            'next_round_plan': state.get('next_round_plan'),
        }
        if current_round:
            payload['current_round'] = current_round
        if best_round:
            payload['best_round'] = best_round
        if latest_finished_round:
            payload['latest_round_card'] = self._build_round_compare_card(latest_finished_round, best_round_index=int(state.get('best_round_index') or 0))
            payload['knowledge_gate_status'] = self._build_gate_status_display(latest_finished_round)
            payload['latest_round_review'] = latest_finished_round.get('round_review')
            payload['latest_round_memory'] = latest_finished_round.get('round_memory')
            payload['latest_planner_output'] = latest_finished_round.get('planner_output')
        if current_round and current_round.get('status') == 'running':
            payload['current_round_card'] = self._build_round_compare_card(current_round, best_round_index=int(state.get('best_round_index') or 0))
        if str(state.get('status') or '').strip().lower() == 'running_round' and current_round:
            live_status = self.train_service.status()
            payload['current_training_status'] = {
                'running': live_status.get('running'),
                'summary': live_status.get('summary'),
                'progress': live_status.get('progress'),
                'latest_metrics': live_status.get('latest_metrics'),
                'pid': live_status.get('pid'),
                'log_file': live_status.get('log_file'),
            }
        if detailed:
            payload['rounds'] = rounds
            payload['round_cards'] = [
                self._build_round_compare_card(item, best_round_index=int(state.get('best_round_index') or 0))
                for item in rounds
            ]
            payload['preflight'] = state.get('preflight')
        else:
            payload['rounds'] = [
                {
                    'round_index': item.get('round_index'),
                    'status': item.get('status'),
                    'summary': item.get('summary'),
                    'change_set': item.get('change_set'),
                }
                for item in rounds[-3:]
            ]
        if str(state.get('status') or '').strip().lower() in _TERMINAL_LOOP_STATES:
            payload['final_summary'] = self._build_final_summary(state)
        next_actions = ['可继续调用 inspect_training_loop 查看完整轮次详情']
        status = str(state.get('status') or '').strip().lower()
        if status == 'running_round':
            next_actions.insert(0, '训练进行中时可继续轮询 check_training_loop_status')
        elif status in _WAITING_LOOP_STATES:
            next_actions.insert(0, '如需继续下一轮，可调用 resume_training_loop')
        elif status in _TERMINAL_LOOP_STATES:
            next_actions.insert(0, '当前环训练已结束，如需新任务可重新 start_training_loop')
        payload['next_actions'] = next_actions
        return payload

    @staticmethod
    def _find_latest_finished_round(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
        for item in reversed(rounds):
            if item.get('run_summary') or item.get('summary'):
                return item
        return None

    def _build_round_compare_card(self, round_record: dict[str, Any], *, best_round_index: int) -> dict[str, Any]:
        run_summary = dict(round_record.get('run_summary') or {})
        recommendation = dict(round_record.get('recommendation') or {})
        comparison_previous = dict(round_record.get('comparison_to_previous') or {})
        comparison_best = dict(round_record.get('comparison_to_best') or {})
        highlights_previous = list(comparison_previous.get('highlights') or [])
        highlights_best = list(comparison_best.get('highlights') or [])
        card = {
            'round_index': round_record.get('round_index'),
            'status': round_record.get('status'),
            'is_best_round': int(round_record.get('round_index') or 0) == int(best_round_index or 0),
            'summary': round_record.get('summary') or run_summary.get('summary'),
            'metrics': run_summary.get('metrics') or {},
            'changed_params': list(round_record.get('change_set') or []),
            'vs_previous': {
                'highlights': highlights_previous[:4],
                'metric_deltas': comparison_previous.get('metric_deltas') or {},
            } if comparison_previous else None,
            'vs_best': {
                'highlights': highlights_best[:4],
                'metric_deltas': comparison_best.get('metric_deltas') or {},
            } if comparison_best else None,
            'decision': {
                'type': (round_record.get('decision') or {}).get('decision_type'),
                'reason': (round_record.get('decision') or {}).get('reason'),
            } if round_record.get('decision') else None,
            'next_plan': {
                'round_index': ((round_record.get('decision') or {}).get('next_round_plan') or {}).get('round_index'),
                'change_set': ((round_record.get('decision') or {}).get('next_round_plan') or {}).get('change_set') or [],
                'experience_context': ((round_record.get('decision') or {}).get('next_round_plan') or {}).get('experience_context') or {},
                'planner_source': ((round_record.get('decision') or {}).get('next_round_plan') or {}).get('planner_source'),
                'planner_summary': ((round_record.get('decision') or {}).get('next_round_plan') or {}).get('planner_summary'),
            } if round_record.get('decision') else None,
            'knowledge_gate': self._build_knowledge_gate_summary(round_record),
            'why': recommendation.get('why') or (round_record.get('analysis') or {}).get('interpretation'),
            'recommendation': recommendation.get('recommendation') or (round_record.get('analysis') or {}).get('recommendation'),
            'round_review': round_record.get('round_review'),
            'round_memory': round_record.get('round_memory'),
            'planner_output': round_record.get('planner_output'),
            'experience_context': round_record.get('experience_context') or {},
        }
        return card

    def _build_final_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        rounds = list(state.get('rounds') or [])
        best_round = self._get_round_by_index(rounds, int(state.get('best_round_index') or 0))
        target_metric = str((state.get('boundaries') or {}).get('target_metric') or 'map50')
        best_metric = self._metric_from_round(best_round, target_metric)
        round_changes: list[dict[str, Any]] = []
        knowledge_gate_rounds: list[dict[str, Any]] = []
        experience_timeline: list[dict[str, Any]] = []
        for item in rounds:
            knowledge_gate = self._build_knowledge_gate_summary(item)
            round_review = dict(item.get('round_review') or {})
            round_memory = dict(item.get('round_memory') or {})
            planner_output = dict(item.get('planner_output') or {})
            round_changes.append({
                'round_index': item.get('round_index'),
                'status': item.get('status'),
                'target_metric': self._metric_from_round(item, target_metric),
                'change_set': item.get('change_set') or [],
            })
            if knowledge_gate:
                knowledge_gate_rounds.append({
                    'round_index': item.get('round_index'),
                    **knowledge_gate,
                })
            if round_review or round_memory or planner_output:
                experience_timeline.append({
                    'round_index': item.get('round_index'),
                    'summary': round_memory.get('summary') or round_review.get('summary') or item.get('summary'),
                    'recommended_action': round_review.get('recommended_action') or round_memory.get('recommended_action'),
                    'next_focus': round_memory.get('next_focus'),
                    'decision_type': planner_output.get('decision_type'),
                    'decision_reason': planner_output.get('decision_reason'),
                    'target_metric_value': round_review.get('target_metric_value') or round_memory.get('target_metric_value'),
                    'change_set': item.get('change_set') or [],
                })
        summary = {
            'loop_id': state.get('loop_id'),
            'loop_name': state.get('loop_name'),
            'status': state.get('status'),
            'best_round_index': state.get('best_round_index'),
            'best_target_metric_name': target_metric,
            'best_target_metric': best_metric,
            'best_model_path': ((best_round or {}).get('run_summary') or {}).get('save_dir'),
            'stop_reason': state.get('stop_reason'),
            'termination_reason': state.get('termination_reason'),
            'termination_detail': state.get('termination_detail') or state.get('summary'),
            'round_count': len(rounds),
            'round_changes': round_changes,
            'knowledge_gate_rounds': knowledge_gate_rounds,
            'last_knowledge_gate': knowledge_gate_rounds[-1] if knowledge_gate_rounds else None,
            'knowledge_gate_overview': self._build_gate_overview(knowledge_gate_rounds),
            'best_round_card': self._build_round_compare_card(best_round, best_round_index=int(state.get('best_round_index') or 0)) if best_round else None,
            'last_round_review': (rounds[-1].get('round_review') if rounds else None),
            'last_round_memory': (rounds[-1].get('round_memory') if rounds else None),
            'last_planner_output': (rounds[-1].get('planner_output') if rounds else None),
            'experience_timeline': experience_timeline,
        }
        return summary

    @staticmethod
    def _knowledge_gate_category(action: str) -> str:
        normalized = str(action or '').strip().lower()
        if not normalized:
            return ''
        if normalized in _HARD_STOP_RECOMMENDED_ACTIONS:
            return 'hard_stop'
        if normalized in _REVIEW_REQUIRED_ACTIONS:
            return 'analysis_review'
        if normalized == 'continue_observing':
            return 'continue_observing'
        return 'other'

    @classmethod
    def _build_knowledge_gate_summary(cls, round_record: dict[str, Any]) -> dict[str, Any] | None:
        recommendation = dict(round_record.get('recommendation') or {})
        analysis = dict(round_record.get('analysis') or {})
        decision = dict(round_record.get('decision') or {})
        action = str(recommendation.get('recommended_action') or analysis.get('assessment') or '').strip().lower()
        matched_rule_ids = list(recommendation.get('matched_rule_ids') or analysis.get('matched_rule_ids') or [])
        why = recommendation.get('why') or analysis.get('interpretation')
        recommendation_text = recommendation.get('recommendation') or analysis.get('recommendation')
        decision_type = str(decision.get('decision_type') or '').strip().lower()
        decision_reason = decision.get('reason')
        source_summary = recommendation.get('source_summary') or analysis.get('source_summary') or {}
        if not (action or matched_rule_ids or why or recommendation_text or decision_type):
            return None
        category = cls._knowledge_gate_category(action)
        outcome = cls._knowledge_gate_outcome(action=action, category=category, decision_type=decision_type)
        return {
            'action': action or None,
            'action_label': cls._knowledge_action_label(action),
            'category': category,
            'category_label': cls._knowledge_category_label(category),
            'matched_rule_ids': matched_rule_ids,
            'why': why,
            'recommendation': recommendation_text,
            'decision_type': decision_type or None,
            'decision_label': cls._knowledge_outcome_label(outcome),
            'decision_reason': decision_reason,
            'source_summary': source_summary,
            'outcome': outcome,
            'outcome_label': cls._knowledge_outcome_label(outcome),
            'user_summary': cls._build_gate_user_summary(
                action=action,
                category=category,
                outcome=outcome,
                recommendation_text=recommendation_text,
                decision_reason=decision_reason,
            ),
        }

    @classmethod
    def _build_gate_status_display(cls, round_record: dict[str, Any]) -> dict[str, Any] | None:
        gate = cls._build_knowledge_gate_summary(round_record)
        if not gate:
            return None
        return {
            'round_index': round_record.get('round_index'),
            'outcome': gate.get('outcome'),
            'outcome_label': gate.get('outcome_label'),
            'category': gate.get('category'),
            'category_label': gate.get('category_label'),
            'action': gate.get('action'),
            'action_label': gate.get('action_label'),
            'matched_rule_ids': gate.get('matched_rule_ids') or [],
            'summary': gate.get('user_summary'),
            'recommendation': gate.get('recommendation'),
            'why': gate.get('why'),
            'decision_type': gate.get('decision_type'),
            'decision_reason': gate.get('decision_reason'),
        }

    @classmethod
    def _build_gate_overview(cls, gate_rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not gate_rounds:
            return None
        outcome_counts: dict[str, int] = {}
        for item in gate_rounds:
            outcome = str(item.get('outcome') or 'other')
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        last_gate = gate_rounds[-1]
        return {
            'count': len(gate_rounds),
            'outcome_counts': outcome_counts,
            'last_outcome': last_gate.get('outcome'),
            'last_outcome_label': last_gate.get('outcome_label'),
            'last_action': last_gate.get('action'),
            'last_action_label': last_gate.get('action_label'),
            'last_summary': last_gate.get('user_summary'),
        }

    @staticmethod
    def _knowledge_action_label(action: str) -> str:
        return _KNOWLEDGE_ACTION_LABELS.get(str(action or '').strip().lower(), str(action or '').strip().lower() or '未知动作')

    @staticmethod
    def _knowledge_category_label(category: str) -> str:
        return _KNOWLEDGE_CATEGORY_LABELS.get(str(category or '').strip().lower(), str(category or '').strip().lower() or '未知分类')

    @staticmethod
    def _knowledge_outcome_label(outcome: str) -> str:
        return _KNOWLEDGE_OUTCOME_LABELS.get(str(outcome or '').strip().lower(), str(outcome or '').strip().lower() or '未知结果')

    @classmethod
    def _knowledge_gate_outcome(cls, *, action: str, category: str, decision_type: str) -> str:
        normalized_decision = str(decision_type or '').strip().lower()
        if category == 'hard_stop':
            return 'hard_stop'
        if normalized_decision == 'await_review':
            return 'awaiting_review'
        if normalized_decision == 'auto_continue':
            return 'auto_continue'
        if normalized_decision == 'paused':
            return 'paused'
        if normalized_decision == 'stop':
            return 'stopped'
        if category == 'continue_observing' or str(action or '').strip().lower() == 'continue_observing':
            return 'continue_observing'
        return 'other'

    @classmethod
    def _build_gate_user_summary(
        cls,
        *,
        action: str,
        category: str,
        outcome: str,
        recommendation_text: Any,
        decision_reason: Any,
    ) -> str:
        action_label = cls._knowledge_action_label(action)
        recommendation = str(recommendation_text or '').strip()
        decision = str(decision_reason or '').strip()
        if outcome == 'awaiting_review':
            return decision or f'本轮建议“{action_label}”，已停在轮间闸门等待审阅。'
        if outcome == 'auto_continue':
            return decision or f'本轮建议“{action_label}”，当前托管级别允许自动继续。'
        if outcome == 'hard_stop':
            return decision or recommendation or f'本轮建议“{action_label}”，当前更适合先停下处理。'
        if outcome == 'continue_observing':
            return decision or recommendation or '当前更适合继续观察，不建议立刻大改。'
        if outcome == 'paused':
            return decision or f'本轮建议“{action_label}”，但已按请求停在当前闸门。'
        if outcome == 'stopped':
            return decision or recommendation or f'本轮建议“{action_label}”，当前环训练已停止。'
        if category == 'analysis_review':
            return decision or recommendation or f'本轮建议“{action_label}”，建议先复盘再决定下一轮。'
        return decision or recommendation or f'本轮知识闸门已给出“{action_label}”建议。'

    def _choose_best_round_index(self, rounds: list[dict[str, Any]], *, target_metric: str) -> int | None:
        ranked: list[tuple[tuple[float, float, float], int]] = []
        for item in rounds:
            round_index = int(item.get('round_index') or 0)
            if round_index <= 0:
                continue
            score = self._round_score(item, target_metric=target_metric)
            ranked.append((score, round_index))
        if not ranked:
            return None
        ranked.sort(reverse=True)
        return ranked[0][1]

    def _round_score(self, round_record: dict[str, Any], *, target_metric: str) -> tuple[float, float, float]:
        run_summary = dict(round_record.get('run_summary') or {})
        run_state = str(run_summary.get('run_state') or round_record.get('status') or '').strip().lower()
        state_score = {
            'completed': 3.0,
            'running': 2.0,
            'stopped': 1.0,
            'failed': 0.0,
            'failed_to_start': -1.0,
        }.get(run_state, -1.0)
        metric = self._metric_from_summary(run_summary, target_metric)
        fallback_map = self._metric_from_summary(run_summary, 'map50')
        return (state_score, float(metric if metric is not None else -1.0), float(fallback_map if fallback_map is not None else -1.0))

    def _metric_from_round(self, round_record: dict[str, Any] | None, metric_name: str) -> float | None:
        if not round_record:
            return None
        summary = dict(round_record.get('run_summary') or {})
        return self._metric_from_summary(summary, metric_name)

    @staticmethod
    def _metric_from_summary(summary: dict[str, Any] | None, metric_name: str) -> float | None:
        metrics = dict((summary or {}).get('metrics') or {})
        aliases = _METRIC_ALIASES.get(metric_name, (metric_name,))
        for alias in aliases:
            value = metrics.get(alias)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _normalize_target_metric(metric_name: str) -> str:
        text = str(metric_name or 'map50').strip().lower()
        return text if text in _METRIC_ALIASES else 'map50'

    @staticmethod
    def _normalize_managed_level(value: str) -> str:
        text = str(value or 'conservative_auto').strip().lower()
        return text if text in _MANAGED_LEVELS else 'conservative_auto'

    @staticmethod
    def _normalize_tuning_params(value: list[str] | None) -> list[str]:
        explicit = value is not None
        items = list(value or []) if explicit else list(_DEFAULT_TUNING_PARAMS)
        normalized: list[str] = []
        for item in items:
            token = str(item or '').strip().lower()
            if token and token in _SUPPORTED_TUNING_PARAMS and token not in normalized:
                normalized.append(token)
        if explicit:
            return normalized
        return normalized or list(_DEFAULT_TUNING_PARAMS)

    @staticmethod
    def _normalize_loop_name(loop_name: str, *, model: str, data_yaml: str) -> str:
        text = str(loop_name or '').strip()
        if text:
            return TrainingLoopService._slugify(text)
        dataset_token = Path(str(data_yaml or 'dataset')).stem or 'dataset'
        model_token = Path(str(model or 'model')).stem or 'model'
        return TrainingLoopService._slugify(f'{dataset_token}-{model_token}')

    @staticmethod
    def _slugify(text: str) -> str:
        cleaned = ''.join(ch.lower() if ch.isalnum() else '-' for ch in str(text or 'loop'))
        while '--' in cleaned:
            cleaned = cleaned.replace('--', '-')
        cleaned = cleaned.strip('-')
        return cleaned or 'training-loop'

    def _generate_loop_id(self, loop_name: str) -> str:
        return f"{int(self._time_fn())}-{self._slugify(loop_name)}"

    def _build_loop_brief(self, state: dict[str, Any], *, active_loop_id: str) -> dict[str, Any]:
        target_metric = str((state.get('boundaries') or {}).get('target_metric') or 'map50')
        best_round = self._get_round_by_index(list(state.get('rounds') or []), int(state.get('best_round_index') or 0))
        return {
            'loop_id': state.get('loop_id'),
            'loop_name': state.get('loop_name'),
            'status': state.get('status'),
            'managed_level': state.get('managed_level'),
            'current_round_index': state.get('current_round_index'),
            'best_round_index': state.get('best_round_index'),
            'best_target_metric': self._metric_from_round(best_round, target_metric),
            'active': str(state.get('loop_id') or '') == active_loop_id,
            'updated_at': state.get('updated_at'),
            'summary': state.get('summary'),
        }

    def _resolve_loop(self, loop_id: str, *, use_lock: bool = True) -> dict[str, Any] | None:
        if use_lock:
            with self._state_lock:
                return self._resolve_loop(loop_id, use_lock=False)

        requested = str(loop_id or '').strip()
        normalized = requested.lower()
        if normalized in {'', 'active'}:
            active = self._read_json(self._active_registry_path)
            if active and active.get('loop_id'):
                return self._load_loop(str(active.get('loop_id')))
            if normalized == 'active':
                return None
        if normalized in {'', 'latest', 'last', 'recent'}:
            last = self._read_json(self._last_registry_path)
            if last and last.get('loop_id'):
                return self._load_loop(str(last.get('loop_id')))
            if normalized and normalized != '':
                return None
        if requested:
            return self._load_loop(requested)
        return None

    def _load_all_loops(self, *, limit: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self._state_dir.glob('loop_*.json'), reverse=True):
            payload = self._read_json(path)
            if payload:
                items.append(payload)
            if len(items) >= limit:
                break
        items.sort(key=lambda item: float(item.get('updated_at') or item.get('created_at') or 0.0), reverse=True)
        return items[:limit]

    def _loop_path(self, loop_id: str) -> Path:
        return self._state_dir / f'loop_{loop_id}.json'

    def _load_loop(self, loop_id: str) -> dict[str, Any] | None:
        if not loop_id:
            return None
        return self._read_json(self._loop_path(loop_id))

    def _save_loop(self, state: dict[str, Any]) -> None:
        payload = copy.deepcopy(state)
        payload['updated_at'] = self._time_fn()
        path = self._loop_path(str(payload.get('loop_id') or ''))
        self._write_json(path, payload)
        summary = {
            'loop_id': payload.get('loop_id'),
            'loop_name': payload.get('loop_name'),
            'status': payload.get('status'),
            'updated_at': payload.get('updated_at'),
            'summary': payload.get('summary'),
        }
        if str(payload.get('status') or '').strip().lower() in _TERMINAL_LOOP_STATES:
            self._write_json(self._last_registry_path, summary)
            active = self._read_json(self._active_registry_path)
            if active and str(active.get('loop_id') or '') == str(payload.get('loop_id') or ''):
                self._delete_file(self._active_registry_path)
        else:
            self._write_json(self._active_registry_path, summary)

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return None

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload, ensure_ascii=False, indent=2)
        last_error: Exception | None = None
        for attempt in range(10):
            temp_path = path.with_name(f'{path.name}.{threading.get_ident()}.{time.time_ns()}.tmp')
            try:
                temp_path.write_text(raw, encoding='utf-8')
                temp_path.replace(path)
                return
            except PermissionError as exc:
                last_error = exc
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                time.sleep(0.02 * (attempt + 1))
        if last_error is not None:
            raise last_error

    @staticmethod
    def _delete_file(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            return

    @staticmethod
    def _get_round_by_index(rounds: list[dict[str, Any]], round_index: int) -> dict[str, Any] | None:
        if round_index <= 0:
            return None
        for item in rounds:
            if int(item.get('round_index') or 0) == round_index:
                return item
        return None

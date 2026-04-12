from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from agent_plan.agent.server.services.gpu_utils import (
    GpuAllocationPolicy,
    get_effective_gpu_policy,
    query_gpu_status,
    resolve_auto_device,
)
from agent_plan.agent.server.services.train_log_parser import parse_latest_metrics, parse_training_log
from agent_plan.agent.server.services.training_result_helpers import build_training_facts, summarize_training_run as build_training_run_summary


class TrainService:
    def __init__(self, state_dir: str | Path | None = None) -> None:
        self._process: subprocess.Popen | None = None
        self._active_pid: int | None = None
        self._log_file: Path | None = None
        self._start_time: float | None = None
        self._resolved_device: str | None = None
        self._requested_device: str | None = None
        self._command: list[str] | None = None
        self._yolo_executable: str | None = None
        self._training_environment: dict[str, Any] | None = None
        self._argument_sources: dict[str, str] | None = None
        self._resolved_args: dict[str, Any] | None = None
        self._reattached: bool = False

        root = Path(state_dir) if state_dir else Path(os.getenv('YOLOSTUDIO_TRAIN_STATE_DIR', 'runs'))
        self._state_dir = root
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._active_registry_path = self._state_dir / 'active_train_job.json'
        self._last_registry_path = self._state_dir / 'last_train_job.json'

    def start(
        self,
        model: str,
        data_yaml: str = '',
        epochs: int = 100,
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
    ) -> dict:
        self._sync_runtime_state()
        if self._is_running():
            pid = self._current_pid()
            return {
                'ok': False,
                'error': f'已有训练任务在运行（pid={pid}），请先停止或等待完成',
            }

        validation_error = self._validate_inputs(
            model=model,
            data_yaml=data_yaml,
            epochs=epochs,
            project=project,
            name=name,
            batch=batch,
            imgsz=imgsz,
            fraction=fraction,
            classes=classes,
            freeze=freeze,
            lr0=lr0,
            patience=patience,
            workers=workers,
        )
        if validation_error:
            return {'ok': False, 'error': validation_error}

        resolved_device, error = self._resolve_device(device)
        if error:
            return {'ok': False, 'error': error}

        selected_environment, environment_error = self._select_training_environment(training_environment)
        if environment_error:
            return {'ok': False, 'error': environment_error}
        if not selected_environment:
            return {'ok': False, 'error': '未找到 yolo 命令。请确认某个 conda 环境中已安装 ultralytics'}
        yolo_exe = str(selected_environment.get('yolo_executable') or '')
        normalized_classes = _normalize_classes_arg(classes)

        runs_dir = self._state_dir
        runs_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = runs_dir / f'train_log_{int(time.time())}.txt'
        self._command = [
            yolo_exe, 'train',
            f'model={model}',
            f'data={data_yaml}',
            f'epochs={epochs}',
            f'device={resolved_device}',
        ]
        if project:
            self._command.append(f'project={project}')
        if name:
            self._command.append(f'name={name}')
        if batch is not None and int(batch) > 0:
            self._command.append(f'batch={int(batch)}')
        if imgsz is not None and int(imgsz) > 0:
            self._command.append(f'imgsz={int(imgsz)}')
        if fraction is not None and 0 < float(fraction) <= 1:
            self._command.append(f'fraction={float(fraction)}')
        if normalized_classes:
            self._command.append(f"classes={','.join(str(item) for item in normalized_classes)}")
        if single_cls is not None:
            self._command.append(f'single_cls={bool(single_cls)}')
        if optimizer:
            self._command.append(f'optimizer={optimizer}')
        if freeze is not None and int(freeze) >= 0:
            self._command.append(f'freeze={int(freeze)}')
        if resume:
            self._command.append('resume=True')
        if lr0 is not None and float(lr0) > 0:
            self._command.append(f'lr0={float(lr0)}')
        if patience is not None and int(patience) >= 0:
            self._command.append(f'patience={int(patience)}')
        if workers is not None and int(workers) >= 0:
            self._command.append(f'workers={int(workers)}')
        if amp is not None:
            self._command.append(f'amp={bool(amp)}')
        self._yolo_executable = yolo_exe
        self._training_environment = selected_environment
        self._requested_device = device
        self._resolved_device = resolved_device
        self._resolved_args = {
            'model': model,
            'data_yaml': data_yaml,
            'epochs': epochs,
            'device': resolved_device,
            'training_environment': str(selected_environment.get('display_name') or selected_environment.get('name') or ''),
            'project': str(project or '').strip() or None,
            'name': str(name or '').strip() or None,
            'batch': int(batch) if batch is not None and int(batch) > 0 else None,
            'imgsz': int(imgsz) if imgsz is not None and int(imgsz) > 0 else None,
            'fraction': float(fraction) if fraction is not None and 0 < float(fraction) <= 1 else None,
            'classes': normalized_classes,
            'single_cls': bool(single_cls) if single_cls is not None else None,
            'optimizer': str(optimizer or '').strip() or None,
            'freeze': int(freeze) if freeze is not None and int(freeze) >= 0 else None,
            'resume': bool(resume) if resume is not None else None,
            'lr0': float(lr0) if lr0 is not None and float(lr0) > 0 else None,
            'patience': int(patience) if patience is not None and int(patience) >= 0 else None,
            'workers': int(workers) if workers is not None and int(workers) >= 0 else None,
            'amp': bool(amp) if amp is not None else None,
            'device_policy': get_effective_gpu_policy(),
        }
        self._argument_sources = {
            'model': 'request_or_agent_input',
            'data_yaml': 'request_or_tool_output',
            'epochs': 'request_or_default',
            'device': 'auto_resolved' if device.strip().lower() == 'auto' else 'manual_request',
            'training_environment': 'manual_request' if training_environment else 'default_environment',
            'project': 'request_or_runtime_default' if project else 'runtime_default',
            'name': 'request_or_runtime_default' if name else 'runtime_default',
            'batch': 'request_or_default' if batch is not None else 'tool_or_runtime_default',
            'imgsz': 'request_or_default' if imgsz is not None else 'tool_or_runtime_default',
            'fraction': 'request_or_runtime_default' if fraction is not None else 'runtime_default',
            'classes': 'request_or_runtime_default' if normalized_classes else 'runtime_default',
            'single_cls': 'request_or_runtime_default' if single_cls is not None else 'runtime_default',
            'optimizer': 'request_or_runtime_default' if optimizer else 'runtime_default',
            'freeze': 'request_or_runtime_default' if freeze is not None else 'runtime_default',
            'resume': 'request_or_runtime_default' if resume is not None else 'runtime_default',
            'lr0': 'request_or_runtime_default' if lr0 is not None else 'runtime_default',
            'patience': 'request_or_runtime_default' if patience is not None else 'runtime_default',
            'workers': 'request_or_runtime_default' if workers is not None else 'runtime_default',
            'amp': 'request_or_runtime_default' if amp is not None else 'runtime_default',
        }
        self._reattached = False

        log_handle = self._log_file.open('w', encoding='utf-8')
        try:
            self._process = subprocess.Popen(
                self._command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            log_handle.close()
            self._clear_runtime_state(clear_process=True)
            return {'ok': False, 'error': f'启动训练子进程失败: {exc}'}
        finally:
            log_handle.close()

        self._active_pid = self._process.pid
        self._start_time = time.time()
        self._write_active_registry()

        return {
            'ok': True,
            'message': f'训练已启动：model={model}, data={data_yaml}, epochs={epochs}, device={resolved_device}, policy={get_effective_gpu_policy()}',
            'pid': self._active_pid,
            'device': resolved_device,
            'requested_device': device,
            'log_file': str(self._log_file),
            'argument_sources': self._argument_sources,
            'command': self._command,
            'resolved_args': self._resolved_args,
            'yolo_executable': yolo_exe,
            'training_environment': selected_environment,
            'started_at': self._start_time,
            'registry_path': str(self._active_registry_path),
            'reattached': False,
        }

    def status(self) -> dict:
        self._sync_runtime_state()
        running = self._is_running()
        result: dict[str, Any] = {
            'ok': True,
            'running': running,
            'model_family': 'yolo',
            'task_type': 'detection',
            'log_file': str(self._log_file) if self._log_file else None,
            'device': self._resolved_device,
            'requested_device': self._requested_device,
            'command': self._command,
            'started_at': self._start_time,
            'yolo_executable': self._yolo_executable,
            'training_environment': self._training_environment,
            'device_policy': get_effective_gpu_policy(),
            'argument_sources': self._argument_sources,
            'resolved_args': self._resolved_args,
            'reattached': self._reattached,
            'registry_path': str(self._active_registry_path),
        }

        pid = self._current_pid()
        if pid is not None:
            result['pid'] = pid

        if running:
            result['return_code'] = None
        else:
            last = self._read_json(self._last_registry_path)
            if last:
                result.update({
                    'pid': last.get('pid'),
                    'return_code': last.get('return_code'),
                    'log_file': last.get('log_file'),
                    'device': last.get('device'),
                    'requested_device': last.get('requested_device'),
                    'command': last.get('command'),
                    'started_at': last.get('started_at'),
                    'yolo_executable': last.get('yolo_executable'),
                    'training_environment': last.get('training_environment'),
                    'argument_sources': last.get('argument_sources'),
                    'resolved_args': last.get('resolved_args'),
                    'stopped_at': last.get('stopped_at'),
                    'stop_reason': last.get('stop_reason'),
                    'last_run': last,
                })

        started_at = result.get('started_at')
        if started_at:
            base = time.time() if running else (result.get('stopped_at') or time.time())
            result['elapsed_seconds'] = round(max(0.0, base - started_at), 2)
        if result.get('log_file'):
            parsed_log = parse_training_log(Path(result['log_file']))
            result['latest_metrics'] = {'ok': bool(parsed_log.get('ok')), 'metrics': parsed_log.get('metrics')}
            result['training_facts'] = build_training_facts(result, parsed_log)
        else:
            result['latest_metrics'] = {'ok': True, 'metrics': None}
            result['training_facts'] = build_training_facts(result, None)
        result['run_state'] = result['training_facts'].get('run_state')
        result['progress'] = result['training_facts'].get('progress') or {}
        result['observation_stage'] = result['training_facts'].get('observation_stage')
        result['signals'] = result['training_facts'].get('signals') or []
        result['facts'] = result['training_facts'].get('facts') or []
        result['analysis_ready'] = bool(result['training_facts'].get('analysis_ready'))
        result['minimum_facts_ready'] = bool(result['training_facts'].get('minimum_facts_ready'))
        result['latest_eval_metrics'] = result['training_facts'].get('latest_eval_metrics')
        result['latest_train_metrics'] = result['training_facts'].get('latest_train_metrics')
        result['error_lines'] = result['training_facts'].get('error_lines') or []
        if not result['latest_metrics'].get('metrics'):
            result['latest_metrics'] = result['training_facts'].get('latest_metrics') or result['latest_metrics']
        result['summary'] = self._build_status_summary(result)
        return result

    def summarize_run(self) -> dict[str, Any]:
        status = self.status()
        parsed_log = None
        if status.get('log_file'):
            parsed_log = parse_training_log(Path(status['log_file']))
        summary = build_training_run_summary(status, parsed_log)
        return summary

    def list_training_environments(self) -> dict[str, Any]:
        environments = _discover_training_environments()
        default_environment = environments[0] if environments else None
        if not environments:
            return {
                'ok': True,
                'summary': '当前未发现可用训练环境；默认训练也无法启动',
                'environments': [],
                'default_environment': None,
                'next_actions': [
                    '请先确认某个 conda 环境中已安装 ultralytics，并且存在 yolo 可执行文件',
                    '如已安装，请检查 PATH 或 conda env list 是否可见',
                ],
            }
        return {
            'ok': True,
            'summary': f"发现 {len(environments)} 个可用训练环境，默认将使用 {default_environment.get('display_name') or default_environment.get('name')}",
            'environments': environments,
            'default_environment': default_environment,
            'next_actions': [
                '如需确认这次训练会怎么启动，可继续调用 training_preflight',
                '如需直接训练，继续调用 start_training 即可',
            ],
        }

    def training_preflight(
        self,
        model: str,
        data_yaml: str = '',
        epochs: int = 100,
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
    ) -> dict[str, Any]:
        self._sync_runtime_state()
        blockers: list[str] = []
        warnings: list[str] = []

        active_pid = self._current_pid()
        if self._is_running():
            blockers.append(f'已有训练任务在运行（pid={active_pid}），当前不建议再次启动')

        validation_error = self._validate_inputs(
            model=model,
            data_yaml=data_yaml,
            epochs=epochs,
            project=project,
            name=name,
            batch=batch,
            imgsz=imgsz,
            fraction=fraction,
            classes=classes,
            freeze=freeze,
            lr0=lr0,
            patience=patience,
            workers=workers,
        )
        if validation_error:
            blockers.append(validation_error)

        resolved_device, device_error = self._resolve_device(device)
        if device_error:
            blockers.append(device_error)

        environments = _discover_training_environments()
        selected_environment = _match_training_environment(environments, training_environment) if training_environment else (environments[0] if environments else None)
        normalized_classes = _normalize_classes_arg(classes)
        if training_environment and not selected_environment:
            available_names = ', '.join(str(item.get('display_name') or item.get('name') or '') for item in environments if (item.get('display_name') or item.get('name')))
            blockers.append(f"训练环境不存在: {training_environment}" + (f"（可用: {available_names}）" if available_names else ''))
        if not selected_environment:
            blockers.append('未找到 yolo 命令。请确认某个 conda 环境中已安装 ultralytics')

        command_preview: list[str] = []
        if selected_environment and not device_error:
            command_preview = [
                str(selected_environment.get('yolo_executable') or ''),
                'train',
                f'model={model}',
                f'data={data_yaml}',
                f'epochs={epochs}',
                f'device={resolved_device}',
            ]
            if project:
                command_preview.append(f'project={project}')
            if name:
                command_preview.append(f'name={name}')
            if batch is not None and int(batch) > 0:
                command_preview.append(f'batch={int(batch)}')
            if imgsz is not None and int(imgsz) > 0:
                command_preview.append(f'imgsz={int(imgsz)}')
            if fraction is not None and 0 < float(fraction) <= 1:
                command_preview.append(f'fraction={float(fraction)}')
            if normalized_classes:
                command_preview.append(f"classes={','.join(str(item) for item in normalized_classes)}")
            if single_cls is not None:
                command_preview.append(f'single_cls={bool(single_cls)}')
            if optimizer:
                command_preview.append(f'optimizer={optimizer}')
            if freeze is not None and int(freeze) >= 0:
                command_preview.append(f'freeze={int(freeze)}')
            if resume:
                command_preview.append('resume=True')
            if lr0 is not None and float(lr0) > 0:
                command_preview.append(f'lr0={float(lr0)}')
            if patience is not None and int(patience) >= 0:
                command_preview.append(f'patience={int(patience)}')
            if workers is not None and int(workers) >= 0:
                command_preview.append(f'workers={int(workers)}')
            if amp is not None:
                command_preview.append(f'amp={bool(amp)}')

        ready_to_start = not blockers
        if ready_to_start:
            summary = (
                f"训练预检通过：将使用 {selected_environment.get('display_name') or selected_environment.get('name')}，"
                f"device={resolved_device}"
            )
        else:
            summary = f"训练预检未通过：{blockers[0]}"

        next_actions: list[str] = []
        if ready_to_start:
            next_actions = [
                '当前参数和训练环境已可启动；如需真正开始训练，请调用 start_training',
                '如需再次确认 GPU 分配，可调用 check_gpu_status',
            ]
        else:
            blocker_text = '；'.join(blockers)
            if 'data_yaml 不能为空' in blocker_text or '数据配置文件不存在' in blocker_text:
                next_actions.append('请先提供可用的 data_yaml，或先走 training_readiness / prepare_dataset_for_training')
            if '已有训练任务在运行' in blocker_text:
                next_actions.append('可先调用 check_training_status 查看当前训练，或 stop_training 后再重试')
            if '训练环境不存在' in blocker_text:
                next_actions.append('可先调用 list_training_environments 查看当前可用环境，再重新指定 training_environment')
            if '未找到 yolo 命令' in blocker_text:
                next_actions.append('可先调用 list_training_environments 确认当前是否存在可用训练环境')
            if 'GPU' in blocker_text or 'CPU 训练' in blocker_text or 'device' in blocker_text:
                next_actions.append('可先调用 check_gpu_status 调整 device 或确认空闲 GPU')
            if not next_actions:
                next_actions.append('请先修正预检中的阻塞项后再尝试训练')

        return {
            'ok': True,
            'ready_to_start': ready_to_start,
            'summary': summary,
            'model_family': 'yolo',
            'task_type': 'detection',
            'requested_device': device,
            'resolved_device': resolved_device if not device_error else None,
            'device_policy': get_effective_gpu_policy(),
            'training_environment': selected_environment,
            'candidate_environments': environments,
            'resolved_args': {
                'model': model,
                'data_yaml': data_yaml,
                'epochs': epochs,
                'device': resolved_device if not device_error else None,
                'training_environment': str(selected_environment.get('display_name') or selected_environment.get('name') or '') if selected_environment else (training_environment or None),
                'project': str(project or '').strip() or None,
                'name': str(name or '').strip() or None,
                'batch': int(batch) if batch is not None and int(batch) > 0 else None,
                'imgsz': int(imgsz) if imgsz is not None and int(imgsz) > 0 else None,
                'fraction': float(fraction) if fraction is not None and 0 < float(fraction) <= 1 else None,
                'classes': normalized_classes,
                'single_cls': bool(single_cls) if single_cls is not None else None,
                'optimizer': str(optimizer or '').strip() or None,
                'freeze': int(freeze) if freeze is not None and int(freeze) >= 0 else None,
                'resume': bool(resume) if resume is not None else None,
                'lr0': float(lr0) if lr0 is not None and float(lr0) > 0 else None,
                'patience': int(patience) if patience is not None and int(patience) >= 0 else None,
                'workers': int(workers) if workers is not None and int(workers) >= 0 else None,
                'amp': bool(amp) if amp is not None else None,
            },
            'command_preview': command_preview,
            'blockers': blockers,
            'warnings': warnings,
            'active_run': {
                'running': self._is_running(),
                'pid': active_pid,
                'log_file': str(self._log_file) if self._log_file else None,
            },
            'next_actions': next_actions,
        }

    def list_training_runs(self, limit: int = 5) -> dict[str, Any]:
        self._sync_runtime_state()
        max_items = max(1, min(int(limit), 20))
        runs = self._collect_training_run_summaries(limit=max_items)
        if runs:
            summary = f"找到 {len(runs)} 条最近训练记录"
            next_actions = [
                '如需看最近一次训练效果，可继续调用 summarize_training_run',
                '如需判断下一步怎么做，可继续调用 analyze_training_outcome 或 recommend_next_training_step',
            ]
        else:
            summary = '当前没有可读训练记录'
            next_actions = [
                '可先调用 start_training 启动一次训练',
                '如需先确认训练参数和环境，可调用 training_preflight',
            ]
        return {
            'ok': True,
            'summary': summary,
            'count': len(runs),
            'limit': max_items,
            'runs': runs,
            'next_actions': next_actions,
        }

    def inspect_training_run(self, run_id: str = '') -> dict[str, Any]:
        self._sync_runtime_state()
        requested = str(run_id or '').strip()
        normalized = requested.lower()

        statuses = self._collect_training_run_statuses(limit=None)
        selected_status: dict[str, Any] | None = None
        found_by = 'latest'

        if normalized in {'', 'latest', 'recent', 'last'}:
            selected_status = statuses[0] if statuses else None
        elif normalized == 'active':
            selected_status = next(
                (
                    status for status in statuses
                    if str(status.get('status_source_hint') or '').strip().lower() == 'active'
                ),
                None,
            )
            found_by = 'active'
        else:
            for status in statuses:
                candidate_run_id = self._derive_run_id(status)
                candidate_log = str(status.get('log_file') or '').strip()
                if requested == candidate_run_id or requested == candidate_log:
                    selected_status = status
                    found_by = 'explicit'
                    break

        if not selected_status:
            return {
                'ok': False,
                'error': f'未找到训练记录: {requested or "latest"}',
                'summary': '未找到对应训练记录',
                'requested_run_id': requested or 'latest',
                'next_actions': [
                    '可先调用 list_training_runs 查看最近训练记录',
                    '如当前没有训练记录，可先调用 training_preflight 或 start_training',
                ],
            }

        parsed_log = None
        if selected_status.get('log_file'):
            parsed_log = parse_training_log(Path(str(selected_status['log_file'])))
        summary = build_training_run_summary(selected_status, parsed_log)
        resolved_args = selected_status.get('resolved_args') or {}
        progress = summary.get('progress') or {}

        next_actions: list[str] = []
        if str(summary.get('run_state') or '').lower() == 'running':
            next_actions.append('如需继续观察训练进度，可继续调用 check_training_status')
        if summary.get('analysis_ready'):
            next_actions.append('如需解释这次训练效果，可继续调用 analyze_training_outcome')
            next_actions.append('如需判断下一步动作，可继续调用 recommend_next_training_step')
        elif summary.get('minimum_facts_ready'):
            next_actions.append('当前已有部分训练事实；如需更稳判断，可继续调用 recommend_next_training_step')
        else:
            next_actions.append('当前训练事实不足；可先检查日志、结果目录或继续训练一段时间')
        if not next_actions:
            next_actions = list(summary.get('next_actions') or [])

        return {
            'ok': True,
            'summary': summary.get('summary') or '训练记录详情已就绪',
            'requested_run_id': requested or 'latest',
            'selected_run_id': self._derive_run_id(selected_status),
            'found_by': found_by,
            'run_state': summary.get('run_state'),
            'observation_stage': summary.get('observation_stage'),
            'analysis_ready': summary.get('analysis_ready'),
            'minimum_facts_ready': summary.get('minimum_facts_ready'),
            'model_family': 'yolo',
            'task_type': 'detection',
            'status_source': summary.get('status_source'),
            'started_at': selected_status.get('started_at'),
            'stopped_at': selected_status.get('stopped_at'),
            'updated_at': selected_status.get('updated_at'),
            'pid': selected_status.get('pid'),
            'device': selected_status.get('device'),
            'training_environment': selected_status.get('training_environment'),
            'log_file': selected_status.get('log_file'),
            'model': resolved_args.get('model'),
            'data_yaml': resolved_args.get('data_yaml'),
            'epochs': resolved_args.get('epochs'),
            'resolved_args': resolved_args,
            'progress': {
                'epoch': progress.get('epoch'),
                'total_epochs': progress.get('total_epochs'),
                'progress_ratio': progress.get('progress_ratio'),
            },
            'metrics': summary.get('metrics'),
            'signals': summary.get('signals'),
            'facts': summary.get('facts'),
            'latest_metrics': summary.get('latest_metrics'),
            'next_actions': next_actions,
        }

    def stop(self) -> dict:
        self._sync_runtime_state()
        if not self._is_running():
            return {'ok': False, 'error': '当前没有正在运行的训练任务'}

        pid = self._current_pid()
        forced = False
        return_code: int | None = None

        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
                forced = True
            return_code = self._process.returncode
        elif pid is not None:
            forced, return_code = self._terminate_pid(pid)
        else:
            return {'ok': False, 'error': '当前没有正在运行的训练任务'}

        self._finalize_active_registry(return_code=return_code, forced=forced, stop_reason='manual_stop')
        self._clear_runtime_state(clear_process=True)
        return {
            'ok': True,
            'message': '训练任务已停止' if not forced else '训练任务已强制停止',
            'forced': forced,
            'return_code': return_code,
        }

    def _sync_runtime_state(self) -> None:
        if self._process and self._process.poll() is None:
            self._active_pid = self._process.pid
            self._reattached = False
            if not self._active_registry_path.exists():
                self._write_active_registry()
            return

        if self._process and self._process.poll() is not None:
            self._finalize_active_registry(return_code=self._process.poll(), forced=False, stop_reason='process_exit')
            self._clear_runtime_state(clear_process=True)

        record = self._read_json(self._active_registry_path)
        if not record:
            return

        pid = int(record.get('pid') or 0)
        if pid <= 0:
            self._finalize_active_registry(record=record, return_code=record.get('return_code'), forced=False, stop_reason='invalid_registry_pid')
            self._clear_runtime_state(clear_process=True)
            return

        if self._pid_exists(pid):
            self._load_registry_into_runtime(record, reattached=True)
            return

        self._finalize_active_registry(record=record, return_code=record.get('return_code'), forced=False, stop_reason='process_missing_after_restart')
        self._clear_runtime_state(clear_process=True)

    def _is_running(self) -> bool:
        pid = self._current_pid()
        return pid is not None and self._pid_exists(pid)

    def _current_pid(self) -> int | None:
        if self._process and self._process.poll() is None:
            return self._process.pid
        return self._active_pid

    def _load_registry_into_runtime(self, record: dict[str, Any], reattached: bool) -> None:
        self._active_pid = int(record.get('pid') or 0) or None
        self._log_file = Path(record['log_file']) if record.get('log_file') else None
        self._start_time = record.get('started_at')
        self._resolved_device = record.get('device')
        self._requested_device = record.get('requested_device')
        self._command = list(record.get('command') or []) or None
        self._yolo_executable = record.get('yolo_executable')
        self._training_environment = record.get('training_environment')
        self._argument_sources = record.get('argument_sources')
        self._resolved_args = record.get('resolved_args')
        self._reattached = reattached
        if reattached:
            self._process = None

    def _snapshot(self, *, running: bool, return_code: int | None = None, forced: bool | None = None, stop_reason: str | None = None) -> dict[str, Any]:
        return {
            'pid': self._current_pid(),
            'log_file': str(self._log_file) if self._log_file else None,
            'started_at': self._start_time,
            'device': self._resolved_device,
            'requested_device': self._requested_device,
            'command': self._command,
            'yolo_executable': self._yolo_executable,
            'training_environment': self._training_environment,
            'argument_sources': self._argument_sources,
            'resolved_args': self._resolved_args,
            'running': running,
            'return_code': return_code,
            'forced': forced,
            'stop_reason': stop_reason,
            'updated_at': time.time(),
        }

    def _collect_training_run_summaries(self, limit: int) -> list[dict[str, Any]]:
        statuses = self._collect_training_run_statuses(limit=limit)
        summaries: list[dict[str, Any]] = []
        for status in statuses:
            summary = self._build_run_summary_from_status(status)
            summary.pop('sort_key', None)
            summaries.append(summary)
        return summaries

    def _collect_training_run_statuses(self, limit: int | None) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_logs: set[str] = set()

        active = self._read_json(self._active_registry_path)
        if active and active.get('log_file'):
            seen_logs.add(str(active.get('log_file')))
            candidates.append(self._status_from_record(active, status_source='active_run'))

        last = self._read_json(self._last_registry_path)
        if last and last.get('log_file'):
            log_path = str(last.get('log_file'))
            if log_path not in seen_logs:
                seen_logs.add(log_path)
                candidates.append(self._status_from_record(last, status_source='last_run'))

        log_files = sorted(
            self._state_dir.glob('train_log_*.txt'),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for log_file in log_files:
            log_path = str(log_file)
            if log_path in seen_logs:
                continue
            seen_logs.add(log_path)
            candidates.append(self._status_from_log(log_file))
            if limit is not None and len(candidates) >= limit:
                break

        candidates.sort(
            key=lambda item: item.get('updated_at') or item.get('stopped_at') or item.get('started_at') or 0,
            reverse=True,
        )
        if limit is None:
            return candidates
        return candidates[:limit]

    def _build_run_summary_from_status(self, status: dict[str, Any]) -> dict[str, Any]:
        parsed_log = None
        if status.get('log_file'):
            parsed_log = parse_training_log(Path(status['log_file']))
        summary = build_training_run_summary(status, parsed_log)
        resolved_args = status.get('resolved_args') or {}
        progress = summary.get('progress') or {}
        return {
            'run_id': self._derive_run_id(status),
            'summary': summary.get('summary'),
            'run_state': summary.get('run_state'),
            'observation_stage': summary.get('observation_stage'),
            'analysis_ready': summary.get('analysis_ready'),
            'minimum_facts_ready': summary.get('minimum_facts_ready'),
            'status_source': summary.get('status_source'),
            'started_at': status.get('started_at'),
            'stopped_at': status.get('stopped_at'),
            'updated_at': status.get('updated_at'),
            'pid': status.get('pid'),
            'device': status.get('device'),
            'model': resolved_args.get('model'),
            'data_yaml': resolved_args.get('data_yaml'),
            'epochs': resolved_args.get('epochs'),
            'log_file': status.get('log_file'),
            'progress': {
                'epoch': progress.get('epoch'),
                'total_epochs': progress.get('total_epochs'),
                'progress_ratio': progress.get('progress_ratio'),
            },
            'metrics': summary.get('metrics'),
            'signals': summary.get('signals'),
            'next_actions': summary.get('next_actions'),
            'sort_key': status.get('updated_at') or status.get('stopped_at') or status.get('started_at') or 0,
        }

    def _status_from_record(self, record: dict[str, Any], status_source: str) -> dict[str, Any]:
        running = bool(record.get('running'))
        status = {
            'ok': True,
            'running': running,
            'log_file': record.get('log_file'),
            'device': record.get('device'),
            'requested_device': record.get('requested_device'),
            'command': record.get('command'),
            'started_at': record.get('started_at'),
            'yolo_executable': record.get('yolo_executable'),
            'training_environment': record.get('training_environment'),
            'argument_sources': record.get('argument_sources'),
            'resolved_args': record.get('resolved_args'),
            'pid': record.get('pid'),
            'return_code': record.get('return_code'),
            'stop_reason': record.get('stop_reason'),
            'stopped_at': record.get('stopped_at'),
            'updated_at': record.get('updated_at'),
            'status_source_hint': status_source,
        }
        if status_source == 'last_run':
            status['last_run'] = record
        return status

    def _status_from_log(self, log_file: Path) -> dict[str, Any]:
        stat = log_file.stat()
        return {
            'ok': True,
            'running': False,
            'log_file': str(log_file),
            'device': None,
            'requested_device': None,
            'command': None,
            'started_at': stat.st_mtime,
            'return_code': None,
            'stop_reason': None,
            'stopped_at': stat.st_mtime,
            'updated_at': stat.st_mtime,
            'resolved_args': {},
            'status_source_hint': 'log_only',
        }

    @staticmethod
    def _derive_run_id(status: dict[str, Any]) -> str:
        log_file = str(status.get('log_file') or '').strip()
        if log_file:
            return Path(log_file).stem
        started_at = status.get('started_at')
        if started_at:
            return f"train_run_{int(float(started_at))}"
        return 'unknown_run'

    def _write_active_registry(self) -> None:
        payload = self._snapshot(running=True)
        self._write_json(self._active_registry_path, payload)

    def _finalize_active_registry(
        self,
        *,
        record: dict[str, Any] | None = None,
        return_code: int | None,
        forced: bool,
        stop_reason: str,
    ) -> None:
        base = dict(record or self._snapshot(running=False, return_code=return_code, forced=forced, stop_reason=stop_reason))
        base.update({
            'running': False,
            'return_code': return_code,
            'forced': forced,
            'stop_reason': stop_reason,
            'stopped_at': time.time(),
            'updated_at': time.time(),
        })
        self._write_json(self._last_registry_path, base)
        if self._active_registry_path.exists():
            self._active_registry_path.unlink()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def _clear_runtime_state(self, *, clear_process: bool) -> None:
        if clear_process:
            self._process = None
        self._active_pid = None
        self._log_file = None
        self._start_time = None
        self._resolved_device = None
        self._requested_device = None
        self._command = None
        self._yolo_executable = None
        self._training_environment = None
        self._argument_sources = None
        self._resolved_args = None
        self._reattached = False

    @staticmethod
    def _pid_exists(pid: int | None) -> bool:
        if not pid or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _terminate_pid(pid: int) -> tuple[bool, int | None]:
        os.kill(pid, signal.SIGTERM)
        if sys.platform == 'win32':
            time.sleep(0.5)
            return False, signal.SIGTERM

        deadline = time.time() + 5
        while time.time() < deadline:
            if not TrainService._pid_exists(pid):
                return False, -signal.SIGTERM
            time.sleep(0.2)

        forced = False
        if hasattr(signal, 'SIGKILL'):
            os.kill(pid, signal.SIGKILL)
            forced = True
            deadline = time.time() + 5
            while time.time() < deadline:
                if not TrainService._pid_exists(pid):
                    return forced, -signal.SIGKILL
                time.sleep(0.2)
        return forced, None

    @staticmethod
    def _validate_inputs(
        model: str,
        data_yaml: str,
        epochs: int,
        project: str = '',
        name: str = '',
        batch: int | None = None,
        imgsz: int | None = None,
        fraction: float | None = None,
        classes: list[int] | str | None = None,
        freeze: int | None = None,
        lr0: float | None = None,
        patience: int | None = None,
        workers: int | None = None,
    ) -> str | None:
        if not str(model).strip():
            return 'model 不能为空'
        if not str(data_yaml).strip():
            return 'data_yaml 不能为空；请先提供 YAML 路径，或先完成数据集准备后再训练'
        if int(epochs) <= 0:
            return 'epochs 必须大于 0'
        if batch is not None and int(batch) <= 0:
            return 'batch 必须大于 0'
        if imgsz is not None and int(imgsz) <= 0:
            return 'imgsz 必须大于 0'
        if project is not None and not str(project).strip() and project != '':
            return 'project 不能为空字符串'
        if name is not None and not str(name).strip() and name != '':
            return 'name 不能为空字符串'
        if fraction is not None and not (0 < float(fraction) <= 1):
            return 'fraction 必须在 (0, 1] 范围内'
        normalized_classes = _normalize_classes_arg(classes)
        classes_explicitly_cleared = isinstance(classes, (list, tuple)) and len(classes) == 0
        if classes is not None and classes != '' and not classes_explicitly_cleared and normalized_classes is None:
            return 'classes 必须是非负整数列表'
        if freeze is not None and int(freeze) < 0:
            return 'freeze 不能小于 0'
        if lr0 is not None and float(lr0) <= 0:
            return 'lr0 必须大于 0'
        if patience is not None and int(patience) < 0:
            return 'patience 不能小于 0'
        if workers is not None and int(workers) < 0:
            return 'workers 不能小于 0'
        if not Path(data_yaml).exists():
            return f'数据配置文件不存在: {data_yaml}'
        model_path = Path(model)
        if model_path.suffix and model_path.suffix in {'.pt', '.onnx', '.yaml'} and not model_path.exists() and not model.startswith('yolo'):
            return f'模型文件不存在: {model}'
        return None

    @staticmethod
    def _find_yolo_executable() -> str | None:
        selected, _ = TrainService._select_training_environment()
        return str(selected.get('yolo_executable')) if selected else None

    @staticmethod
    def _select_training_environment(requested_name: str = '') -> tuple[dict[str, Any] | None, str | None]:
        environments = _discover_training_environments()
        if not environments:
            return None, None
        if requested_name:
            matched = _match_training_environment(environments, requested_name)
            if not matched:
                available_names = ', '.join(str(item.get('display_name') or item.get('name') or '') for item in environments if (item.get('display_name') or item.get('name')))
                return None, f"训练环境不存在: {requested_name}" + (f"（可用: {available_names}）" if available_names else '')
            return matched, None
        return environments[0], None

    @staticmethod
    def _resolve_device(device: str) -> tuple[str, str | None]:
        device = device.strip().lower()
        policy = get_effective_gpu_policy()

        if device == 'cpu':
            return '', '不支持 CPU 训练，请使用 GPU'
        if device == 'auto':
            return resolve_auto_device(policy=policy)

        gpus = query_gpu_status()
        gpu_map = {gpu.index: gpu for gpu in gpus}
        requested_ids = [part.strip() for part in device.split(',') if part.strip()]
        if not requested_ids:
            return '', 'device 不能为空'

        if len(requested_ids) > 1 and policy == GpuAllocationPolicy.SINGLE_IDLE_GPU:
            return '', f'当前策略 {policy} 仅允许单卡；收到 device={device}'

        missing = [gpu_id for gpu_id in requested_ids if gpu_id not in gpu_map]
        if missing:
            valid_ids = ', '.join(sorted(gpu_map.keys())) or '无'
            return '', f"GPU {', '.join(missing)} 不存在（可用设备: {valid_ids}）"

        busy = [gpu_id for gpu_id in requested_ids if gpu_map[gpu_id].busy]
        if busy:
            return '', f"GPU {', '.join(busy)} 上有进程在运行，不建议同时训练；可改用 device=auto 选择空闲 GPU"

        return ','.join(requested_ids), None

    @staticmethod
    def _build_status_summary(result: dict[str, Any]) -> str:
        training_facts = result.get('training_facts') or {}
        run_state = str(result.get('run_state') or training_facts.get('run_state') or '')
        progress = result.get('progress') or training_facts.get('progress') or {}
        if result.get('running'):
            elapsed = result.get('elapsed_seconds')
            elapsed_text = f', 已运行 {elapsed}s' if elapsed is not None else ''
            takeover_text = '，已从注册表接管' if result.get('reattached') else ''
            observation_stage = str(result.get('observation_stage') or training_facts.get('observation_stage') or 'early')
            if observation_stage == 'late':
                stage_text = '，当前已接近结束，可做阶段性分析'
            elif observation_stage == 'mid':
                stage_text = '，当前已进入中段观察，可做阶段性分析'
            else:
                stage_text = '，当前仍属早期观察'
            progress_text = ''
            if progress.get('epoch') is not None and progress.get('total_epochs') is not None:
                progress_text = f"，epoch {progress.get('epoch')}/{progress.get('total_epochs')}"
            return f"训练进行中 (device={result.get('device')}, pid={result.get('pid')}{elapsed_text}{takeover_text}{progress_text}{stage_text})"
        if run_state == 'completed':
            suffix = '，已有可分析指标' if training_facts.get('analysis_ready') else ''
            return f"当前无训练在跑，最近一次训练已完成，return_code={result.get('return_code')}{suffix}"
        if run_state == 'failed':
            return f"当前无训练在跑，最近一次训练失败，return_code={result.get('return_code')}"
        if result.get('return_code') is None and not result.get('log_file'):
            return '当前没有训练任务'
        if result.get('stop_reason') == 'manual_stop':
            return f"当前无训练在跑，最近一次训练已手动停止，return_code={result.get('return_code')}"
        return f"当前无训练在跑，最近 return_code={result.get('return_code')}"


def _resolve_yolo_in_env(env_path: Path) -> str | None:
    if sys.platform == 'win32':
        yolo_exe = env_path / 'Scripts' / 'yolo.exe'
    else:
        yolo_exe = env_path / 'bin' / 'yolo'
    return str(yolo_exe) if yolo_exe.exists() else None


def _resolve_python_in_env(env_path: Path) -> str | None:
    if sys.platform == 'win32':
        python_exe = env_path / 'Scripts' / 'python.exe'
    else:
        python_exe = env_path / 'bin' / 'python'
    return str(python_exe) if python_exe.exists() else None


def _infer_env_path_from_yolo(yolo_executable: str) -> Path | None:
    path = Path(yolo_executable)
    if path.parent.name in {'bin', 'Scripts'}:
        return path.parent.parent
    return None


def _infer_env_name(env_path: Path | None, fallback: str = '') -> str:
    if env_path and env_path.name:
        if env_path.name in {'miniconda3', 'anaconda3', '.conda'}:
            return 'base'
        return env_path.name
    return fallback or 'unknown'


def _discover_conda_env_dirs() -> list[tuple[Path, str]]:
    discovered: list[tuple[Path, str]] = []
    seen_paths: set[str] = set()
    try:
        result = subprocess.run(
            ['conda', 'env', 'list'],
            capture_output=True,
            text=True,
            timeout=10,
            shell=(sys.platform == 'win32'),
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    env_path = Path(parts[-1])
                    if env_path.is_dir():
                        key = str(env_path.resolve())
                        if key not in seen_paths:
                            discovered.append((env_path, 'conda_env_list'))
                            seen_paths.add(key)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    if sys.platform == 'win32':
        search_roots = [
            Path.home() / 'anaconda3' / 'envs',
            Path.home() / 'miniconda3' / 'envs',
            Path.home() / '.conda' / 'envs',
        ]
    else:
        search_roots = [
            Path.home() / 'anaconda3' / 'envs',
            Path.home() / 'miniconda3' / 'envs',
            Path.home() / '.conda' / 'envs',
            Path('/opt/anaconda3/envs'),
            Path('/opt/miniconda3/envs'),
        ]

    for root in search_roots:
        if not root.exists():
            continue
        try:
            for env_dir in root.iterdir():
                if not env_dir.is_dir():
                    continue
                key = str(env_dir.resolve())
                if key in seen_paths:
                    continue
                discovered.append((env_dir, 'env_directory_scan'))
                seen_paths.add(key)
        except PermissionError:
            continue
    return discovered


def _discover_training_environments() -> list[dict[str, Any]]:
    environments: list[dict[str, Any]] = []
    seen_yolo_paths: set[str] = set()

    yolo_in_path = shutil.which('yolo')
    if yolo_in_path:
        env_path = _infer_env_path_from_yolo(yolo_in_path)
        env_name = _infer_env_name(env_path, fallback='path')
        environments.append({
            'name': env_name,
            'display_name': env_name,
            'env_path': str(env_path) if env_path else '',
            'yolo_executable': yolo_in_path,
            'python_executable': _resolve_python_in_env(env_path) if env_path else '',
            'source': 'path',
            'selected_by_default': True,
        })
        seen_yolo_paths.add(str(Path(yolo_in_path).resolve()))

    for env_path, source in _discover_conda_env_dirs():
        yolo = _resolve_yolo_in_env(env_path)
        if not yolo:
            continue
        resolved_yolo = str(Path(yolo).resolve())
        if resolved_yolo in seen_yolo_paths:
            continue
        env_name = _infer_env_name(env_path)
        environments.append({
            'name': env_name,
            'display_name': env_name,
            'env_path': str(env_path),
            'yolo_executable': yolo,
            'python_executable': _resolve_python_in_env(env_path) or '',
            'source': source,
            'selected_by_default': False,
        })
        seen_yolo_paths.add(resolved_yolo)

    if environments:
        environments[0]['selected_by_default'] = True
    return environments


def _match_training_environment(environments: list[dict[str, Any]], requested_name: str) -> dict[str, Any] | None:
    requested = str(requested_name or '').strip().lower()
    if not requested:
        return environments[0] if environments else None
    for item in environments:
        candidates = {
            str(item.get('name') or '').strip().lower(),
            str(item.get('display_name') or '').strip().lower(),
            Path(str(item.get('env_path') or '')).name.strip().lower() if item.get('env_path') else '',
        }
        if requested in candidates:
            return item
    return None


def _normalize_classes_arg(classes: list[int] | str | None) -> list[int] | None:
    if classes is None or classes == '':
        return None
    if isinstance(classes, str):
        raw = classes.strip()
        if not raw:
            return None
        parts = [part.strip() for part in raw.split(',') if part.strip()]
        if not parts or not all(part.isdigit() for part in parts):
            return None
        return [int(part) for part in parts]
    if isinstance(classes, (list, tuple)):
        normalized: list[int] = []
        for item in classes:
            if isinstance(item, bool):
                return None
            try:
                value = int(item)
            except (TypeError, ValueError):
                return None
            if value < 0:
                return None
            normalized.append(value)
        return normalized or None
    return None

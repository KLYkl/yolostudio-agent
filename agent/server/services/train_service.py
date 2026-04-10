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
from agent_plan.agent.server.services.train_log_parser import parse_latest_metrics


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
    ) -> dict:
        self._sync_runtime_state()
        if self._is_running():
            pid = self._current_pid()
            return {
                'ok': False,
                'error': f'已有训练任务在运行（pid={pid}），请先停止或等待完成',
            }

        validation_error = self._validate_inputs(model=model, data_yaml=data_yaml, epochs=epochs)
        if validation_error:
            return {'ok': False, 'error': validation_error}

        resolved_device, error = self._resolve_device(device)
        if error:
            return {'ok': False, 'error': error}

        yolo_exe = self._find_yolo_executable()
        if not yolo_exe:
            return {'ok': False, 'error': '未找到 yolo 命令。请确认某个 conda 环境中已安装 ultralytics'}

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
        self._yolo_executable = yolo_exe
        self._requested_device = device
        self._resolved_device = resolved_device
        self._resolved_args = {
            'model': model,
            'data_yaml': data_yaml,
            'epochs': epochs,
            'device': resolved_device,
            'device_policy': get_effective_gpu_policy(),
        }
        self._argument_sources = {
            'model': 'request_or_agent_input',
            'data_yaml': 'request_or_tool_output',
            'epochs': 'request_or_default',
            'device': 'auto_resolved' if device.strip().lower() == 'auto' else 'manual_request',
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
            'log_file': str(self._log_file) if self._log_file else None,
            'device': self._resolved_device,
            'requested_device': self._requested_device,
            'command': self._command,
            'started_at': self._start_time,
            'yolo_executable': self._yolo_executable,
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
            result['latest_metrics'] = parse_latest_metrics(Path(result['log_file']))
        result['summary'] = self._build_status_summary(result)
        return result

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
            'argument_sources': self._argument_sources,
            'resolved_args': self._resolved_args,
            'running': running,
            'return_code': return_code,
            'forced': forced,
            'stop_reason': stop_reason,
            'updated_at': time.time(),
        }

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
    def _validate_inputs(model: str, data_yaml: str, epochs: int) -> str | None:
        if not str(model).strip():
            return 'model 不能为空'
        if not str(data_yaml).strip():
            return 'data_yaml 不能为空；请先提供 YAML 路径，或先完成数据集准备后再训练'
        if int(epochs) <= 0:
            return 'epochs 必须大于 0'
        if not Path(data_yaml).exists():
            return f'数据配置文件不存在: {data_yaml}'
        model_path = Path(model)
        if model_path.suffix and model_path.suffix in {'.pt', '.onnx', '.yaml'} and not model_path.exists() and not model.startswith('yolo'):
            return f'模型文件不存在: {model}'
        return None

    @staticmethod
    def _find_yolo_executable() -> str | None:
        yolo_in_path = shutil.which('yolo')
        if yolo_in_path:
            return yolo_in_path

        search_roots: list[Path] = []

        try:
            result = subprocess.run(
                ['conda', 'env', 'list'],
                capture_output=True, text=True, timeout=10,
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
                            yolo = _resolve_yolo_in_env(env_path)
                            if yolo:
                                return yolo
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
                    yolo = _resolve_yolo_in_env(env_dir)
                    if yolo:
                        return yolo
            except PermissionError:
                continue

        return None

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
        if result.get('running'):
            elapsed = result.get('elapsed_seconds')
            elapsed_text = f', 已运行 {elapsed}s' if elapsed is not None else ''
            takeover_text = '，已从注册表接管' if result.get('reattached') else ''
            return f"训练进行中 (device={result.get('device')}, pid={result.get('pid')}{elapsed_text}{takeover_text})"
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

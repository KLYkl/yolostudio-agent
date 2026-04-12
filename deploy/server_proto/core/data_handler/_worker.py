"""
_worker.py - DataWorker 线程封装
============================================
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal


class DataWorker(QThread):
    """
    数据处理工作线程

    Signals:
        progress(int, int): 进度更新 (current, total)
        message(str): 日志消息
        result_ready(object): 任务完成，携带结果
        error(str): 错误信息
    """

    progress = Signal(int, int)
    message = Signal(str)
    result_ready = Signal(object)
    error = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._task: Optional[Callable[[], Any]] = None
        self._is_interrupted: bool = False

    def set_task(self, task: Callable[[], Any]) -> None:
        """设置要执行的任务"""
        self._task = task

    def request_interrupt(self) -> None:
        """请求中断任务"""
        self._is_interrupted = True

    def is_interrupted(self) -> bool:
        """检查是否已请求中断"""
        return self._is_interrupted

    def run(self) -> None:
        """执行任务"""
        self._is_interrupted = False

        if self._task is None:
            self.error.emit("未设置任务")
            return

        try:
            result = self._task()
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))

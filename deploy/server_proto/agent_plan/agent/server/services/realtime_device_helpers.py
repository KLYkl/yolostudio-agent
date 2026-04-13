from __future__ import annotations

import os
from typing import Any

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    mss = None


class DeviceScanner:
    @staticmethod
    def scan_cameras(max_devices: int = 5) -> list[dict[str, Any]]:
        if cv2 is None:
            return []

        cameras: list[dict[str, Any]] = []
        original_log_level = os.environ.get('OPENCV_LOG_LEVEL', '')
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        try:
            for index in range(max(0, int(max_devices))):
                capture = None
                try:
                    capture = cv2.VideoCapture(index, cv2.CAP_ANY)
                    if capture is not None and capture.isOpened():
                        ok, _ = capture.read()
                        if ok:
                            cameras.append({
                                'id': index,
                                'name': f'摄像头 {index}',
                            })
                except Exception:
                    continue
                finally:
                    if capture is not None:
                        try:
                            capture.release()
                        except Exception:
                            pass
        finally:
            if original_log_level:
                os.environ['OPENCV_LOG_LEVEL'] = original_log_level
            elif 'OPENCV_LOG_LEVEL' in os.environ:
                del os.environ['OPENCV_LOG_LEVEL']
        return cameras

    @staticmethod
    def scan_screens() -> list[dict[str, Any]]:
        if mss is None:
            return []

        screens: list[dict[str, Any]] = []
        try:
            with mss.mss() as sct:
                for index, monitor in enumerate(sct.monitors[1:], start=1):
                    screens.append({
                        'id': index,
                        'name': f"屏幕 {index} ({monitor['width']}x{monitor['height']})",
                        'width': monitor['width'],
                        'height': monitor['height'],
                        'left': monitor['left'],
                        'top': monitor['top'],
                    })
        except Exception:
            return []
        return screens

    @staticmethod
    def test_rtsp(url: str, timeout_ms: int = 5000) -> tuple[bool, str | None]:
        if cv2 is None:
            return False, '当前环境缺少 cv2'
        if not str(url or '').strip():
            return False, '地址不能为空'

        capture = None
        try:
            capture = cv2.VideoCapture(url)
            if capture is None:
                return False, '无法创建视频流对象'
            try:
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, max(1, int(timeout_ms)))
            except Exception:
                pass
            if not capture.isOpened():
                return False, '无法打开流'
            ok, _ = capture.read()
            if not ok:
                return False, '无法读取视频帧'
            return True, None
        except Exception as exc:
            return False, str(exc)
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass

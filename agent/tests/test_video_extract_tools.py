from __future__ import annotations

import shutil
import sys
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.server.tools import extract_tools


TMP_ROOT = Path(__file__).resolve().parents[2] / '.tmp_video_extract_tools'


def _try_make_video(path: Path) -> bool:
    try:
        import cv2
        import numpy as np
    except Exception:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (64, 64))
    if not writer.isOpened():
        return False
    try:
        for idx in range(6):
            frame = np.full((64, 64, 3), idx * 30, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path.exists()


def main() -> None:
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    video_dir = TMP_ROOT / 'videos'
    try:
        real_video = video_dir / 'cam_a.mp4'
        real_video.parent.mkdir(parents=True, exist_ok=True)
        made_real = _try_make_video(real_video)
        if not made_real:
            real_video.write_bytes(b'placeholder')
        (video_dir / 'nested').mkdir(parents=True, exist_ok=True)
        (video_dir / 'nested' / 'cam_b.mp4').write_bytes(b'placeholder')

        scanned = extract_tools.scan_videos(str(video_dir))
        assert scanned['ok'] is True, scanned
        assert scanned['total_videos'] == 2, scanned
        assert scanned['directory_counts']['.'] == 1, scanned
        assert scanned['directory_counts']['nested'] == 1, scanned

        if made_real:
            output_dir = TMP_ROOT / 'frames_out'
            extracted = extract_tools.extract_video_frames(
                source_path=str(real_video),
                output_dir=str(output_dir),
                mode='interval',
                frame_interval=2,
            )
            assert extracted['ok'] is True, extracted
            assert extracted['final_count'] > 0, extracted
            assert Path(extracted['output_dir']).is_dir(), extracted
        else:
            class _FakeService:
                def extract_video_frames(self, **kwargs):
                    return {
                        'ok': True,
                        'summary': '视频抽帧完成: 最终保留 3 帧（原始抽取 3 / 去重移除 0）',
                        'source_path': kwargs['source_path'],
                        'mode': kwargs['mode'],
                        'frame_interval': kwargs['frame_interval'],
                        'output_dir': str(TMP_ROOT / 'frames_out'),
                        'total_frames': 6,
                        'extracted': 3,
                        'dedup_removed': 0,
                        'final_count': 3,
                        'video_stats': {'cam_a.mp4': 3},
                        'duration_seconds': 1.2,
                        'skipped': 3,
                        'warnings': [],
                        'artifacts': {'output_dir': str(TMP_ROOT / 'frames_out')},
                        'next_actions': ['可将抽帧输出目录继续作为图片输入使用'],
                    }

            original_service = extract_tools.service
            extract_tools.service = _FakeService()
            try:
                extracted = extract_tools.extract_video_frames(
                    source_path=str(real_video),
                    output_dir=str(TMP_ROOT / 'frames_out'),
                    mode='interval',
                    frame_interval=2,
                )
            finally:
                extract_tools.service = original_service
            assert extracted['ok'] is True, extracted
            assert extracted['final_count'] == 3, extracted

        print('video extract tools ok')
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()

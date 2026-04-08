"""
_video_extract.py - VideoExtractMixin: 视频抽帧
============================================

支持三种抽帧模式:
    - interval: 按帧间隔抽取
    - time: 按时间间隔抽取
    - scene: 按场景切换抽取
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from core.data_handler._models import (
    VIDEO_EXTENSIONS,
    VideoExtractConfig,
    VideoExtractResult,
    _get_unique_dir,
)


class VideoExtractMixin:
    """视频抽帧功能 Mixin"""

    def scan_videos(self, video_dir: Path) -> dict[str, int]:
        """
        扫描目录结构，返回各子目录的视频数量

        Args:
            video_dir: 视频根目录或单个视频文件

        Returns:
            {相对路径字符串: 视频数量} 字典
            根目录使用 "." 表示
        """
        result: dict[str, int] = {}

        if video_dir.is_file():
            if video_dir.suffix.lower() in VIDEO_EXTENSIONS:
                result["."] = 1
            return result

        if not video_dir.exists():
            return result

        root_videos = self._find_videos_flat(video_dir)
        if root_videos:
            result["."] = len(root_videos)

        for sub_dir in sorted(video_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            if sub_dir.name.startswith(".") or sub_dir.name.startswith("_"):
                continue

            sub_videos = self._find_videos(sub_dir)
            if not sub_videos:
                continue

            try:
                rel = str(sub_dir.relative_to(video_dir))
            except ValueError:
                rel = sub_dir.name
            result[rel] = len(sub_videos)

        return result

    def extract_video_frames(
        self,
        video_dir: Path,
        config: VideoExtractConfig,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> VideoExtractResult:
        """
        执行视频抽帧

        Args:
            video_dir: 视频目录或单个视频文件
            config: 抽帧配置
            interrupt_check: 中断检查
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            VideoExtractResult
        """
        result = VideoExtractResult()
        videos = self._collect_videos(video_dir)

        if not videos:
            if message_callback:
                message_callback("未找到视频文件")
            return result

        output_dir = config.output_dir
        if output_dir is None:
            base_name = video_dir.stem if video_dir.is_file() else video_dir.name
            output_dir = _get_unique_dir(video_dir.parent / f"{base_name}_frames")
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = str(output_dir)

        if message_callback:
            message_callback(f"发现 {len(videos)} 个视频，开始抽帧...")
            message_callback(f"输出目录: {output_dir}")

        total_progress = 0
        video_meta: list[tuple[Path, int, float, int, int]] = []
        for video_path in videos:
            total_frames, fps, start_frame, end_frame = self._read_video_window_info(
                video_path, config
            )
            video_meta.append((video_path, total_frames, fps, start_frame, end_frame))
            result.total_frames += total_frames
            if fps > 0 and total_frames > 0:
                result.duration += total_frames / fps
            total_progress += max(0, end_frame - start_frame)

        if total_progress <= 0:
            total_progress = len(videos)

        processed_progress = 0

        for index, (video_path, total_frames, fps, start_frame, end_frame) in enumerate(
            video_meta, start=1
        ):
            if interrupt_check():
                if message_callback:
                    message_callback("抽帧已取消")
                break

            video_key = self._get_video_stat_key(video_path, video_dir)
            if message_callback:
                message_callback(
                    f"[{index}/{len(videos)}] 正在处理: {video_key}"
                )

            if len(videos) == 1:
                video_output_dir = output_dir
            else:
                video_output_dir = output_dir / video_path.stem
                if video_output_dir.exists():
                    video_output_dir = _get_unique_dir(video_output_dir)

            local_total = max(0, end_frame - start_frame)

            def local_progress(current: int, total: int) -> None:
                if progress_callback is None:
                    return

                if total_progress <= 0:
                    progress_callback(index, len(videos))
                    return

                bounded_current = min(current, total if total > 0 else current)
                progress_callback(
                    min(total_progress, processed_progress + bounded_current),
                    total_progress,
                )

            video_result = self._extract_single_video(
                video_path,
                video_output_dir,
                config,
                total_frames=total_frames,
                fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
                interrupt_check=interrupt_check,
                progress_callback=local_progress if local_total > 0 else None,
                message_callback=message_callback,
            )

            result.extracted += video_result.extracted
            result.dedup_removed += video_result.dedup_removed
            result.final_count += video_result.final_count
            result.skipped += video_result.skipped
            result.video_stats[video_key] = video_result.final_count

            processed_progress += local_total if local_total > 0 else 1
            if progress_callback and local_total <= 0:
                progress_callback(min(processed_progress, total_progress), total_progress)

        if message_callback:
            message_callback("=" * 40)
            message_callback("视频抽帧完成:")
            message_callback(f"  原始抽取: {result.extracted} 帧")
            if result.dedup_removed > 0:
                message_callback(f"  去重移除: {result.dedup_removed} 帧")
            message_callback(f"  最终保留: {result.final_count} 帧")
            for video_name, count in sorted(result.video_stats.items()):
                message_callback(f"  🎬 {video_name}: {count} 帧")

        return result

    def _extract_single_video(
        self,
        video_path: Path,
        output_dir: Path,
        config: VideoExtractConfig,
        *,
        total_frames: int = 0,
        fps: float = 0.0,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> VideoExtractResult:
        """
        抽取单个视频的帧

        Returns:
            单视频的 VideoExtractResult
        """
        import cv2

        result = VideoExtractResult()
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            if message_callback:
                message_callback(f"无法打开视频: {video_path.name}")
            return result

        try:
            if total_frames <= 0:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            result.total_frames = total_frames
            result.duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
            result.output_dir = str(output_dir)

            if start_frame is None or end_frame is None:
                start_frame, end_frame = self._resolve_frame_range(
                    total_frames, fps, config
                )

            if start_frame >= end_frame:
                if message_callback:
                    message_callback(f"跳过视频: {video_path.name} (时间范围内无有效帧)")
                return result

            output_dir.mkdir(parents=True, exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_step = self._resolve_frame_step(config, fps)
            mode = config.mode.lower()
            total_to_process = max(0, end_frame - start_frame)
            current_frame_index = start_frame
            processed = 0
            extracted_paths: list[Path] = []
            previous_frame: Optional[Any] = None
            last_scene_frame = start_frame - max(1, config.min_scene_gap)

            while current_frame_index < end_frame:
                if interrupt_check():
                    if message_callback:
                        message_callback("抽帧已取消")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                should_save = False
                if mode == "interval":
                    should_save = (current_frame_index - start_frame) % frame_step == 0
                elif mode == "time":
                    should_save = (current_frame_index - start_frame) % frame_step == 0
                else:
                    if previous_frame is None:
                        should_save = True
                    else:
                        diff = self._compute_histogram_diff(previous_frame, frame)
                        if (
                            diff >= config.scene_threshold
                            and current_frame_index - last_scene_frame >= max(1, config.min_scene_gap)
                        ):
                            should_save = True
                    previous_frame = frame

                if mode != "scene" and previous_frame is None:
                    previous_frame = frame

                if should_save:
                    saved_path = self._save_frame(
                        frame,
                        output_dir,
                        video_path,
                        current_frame_index,
                        fps,
                        config,
                    )
                    if saved_path is not None:
                        extracted_paths.append(saved_path)
                        result.extracted += 1
                        if mode == "scene":
                            last_scene_frame = current_frame_index
                    else:
                        result.skipped += 1

                    if config.max_frames > 0 and result.extracted >= config.max_frames:
                        processed += 1
                        if progress_callback:
                            progress_callback(processed, total_to_process)
                        break
                else:
                    result.skipped += 1

                processed += 1
                if progress_callback:
                    progress_callback(processed, total_to_process)

                current_frame_index += 1

            if config.enable_dedup and extracted_paths:
                result.dedup_removed = self._dedup_extracted_frames(
                    extracted_paths,
                    config.dedup_threshold,
                    message_callback=message_callback,
                )

            result.final_count = max(0, result.extracted - result.dedup_removed)
            result.video_stats[video_path.name] = result.final_count
            return result
        finally:
            cap.release()

    def _collect_videos(self, video_dir: Path) -> list[Path]:
        """根据输入收集视频文件"""
        if video_dir.is_file():
            if video_dir.suffix.lower() in VIDEO_EXTENSIONS:
                return [video_dir]
            return []
        return self._find_videos(video_dir)

    def _find_videos(self, root: Path) -> list[Path]:
        """递归查找所有视频文件"""
        if not root.exists():
            return []

        videos: list[Path] = []
        for ext in VIDEO_EXTENSIONS:
            videos.extend(root.rglob(f"*{ext}"))
            videos.extend(root.rglob(f"*{ext.upper()}"))
        return sorted(set(videos))

    def _find_videos_flat(self, directory: Path) -> list[Path]:
        """只查找指定目录下直接的视频文件 (不递归子目录)"""
        videos: list[Path] = []
        if not directory.exists():
            return videos

        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path)
        return videos

    def _read_video_window_info(
        self,
        video_path: Path,
        config: VideoExtractConfig,
    ) -> tuple[int, float, int, int]:
        """读取视频元信息，并计算实际抽取帧范围"""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0, 0.0, 0, 0

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            start_frame, end_frame = self._resolve_frame_range(total_frames, fps, config)
            return total_frames, fps, start_frame, end_frame
        finally:
            cap.release()

    def _resolve_frame_range(
        self,
        total_frames: int,
        fps: float,
        config: VideoExtractConfig,
    ) -> tuple[int, int]:
        """根据时间区间计算抽取帧范围 [start, end)"""
        if total_frames <= 0:
            return 0, 0

        actual_fps = fps if fps > 0 else 30.0
        start_frame = max(0, int(config.start_time * actual_fps))
        if config.end_time > 0:
            end_frame = min(total_frames, int(config.end_time * actual_fps))
        else:
            end_frame = total_frames
        return start_frame, max(start_frame, end_frame)

    def _resolve_frame_step(self, config: VideoExtractConfig, fps: float) -> int:
        """根据模式计算抽帧步长"""
        mode = config.mode.lower()
        if mode == "time":
            actual_fps = fps if fps > 0 else 30.0
            return max(1, round(config.time_interval * actual_fps))
        return max(1, config.frame_interval)

    def _get_video_stat_key(self, video_path: Path, source_root: Path) -> str:
        """生成视频统计键，优先使用相对路径避免重名冲突"""
        if source_root.is_dir():
            try:
                return str(video_path.relative_to(source_root))
            except ValueError:
                pass
        return video_path.name

    def _compute_histogram_diff(self, frame_a: Any, frame_b: Any) -> float:
        """计算两帧的 HSV 直方图差异，返回 Bhattacharyya 距离"""
        import cv2

        hsv_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)

        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))

    def _save_frame(
        self,
        frame: Any,
        output_dir: Path,
        video_path: Path,
        frame_index: int,
        fps: float,
        config: VideoExtractConfig,
    ) -> Optional[Path]:
        """保存单帧图片，成功返回路径，失败返回 None"""
        import cv2

        output_format = config.output_format.lower().lstrip(".")
        if output_format not in {"jpg", "jpeg", "png", "bmp", "webp"}:
            output_format = "jpg"
        if output_format == "jpeg":
            output_format = "jpg"

        prefix = config.name_prefix.strip() or video_path.stem
        timestamp_ms = int((frame_index / fps) * 1000) if fps > 0 else frame_index
        base_name = f"{prefix}_f{frame_index:06d}_t{timestamp_ms:010d}"
        output_path = output_dir / f"{base_name}.{output_format}"

        suffix = 1
        while output_path.exists():
            output_path = output_dir / f"{base_name}_{suffix}.{output_format}"
            suffix += 1

        params: list[int] = []
        if output_format == "jpg":
            params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, config.jpg_quality))]

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(output_path), frame, params)
            return output_path if success else None
        except Exception:
            return None

    def _dedup_extracted_frames(
        self,
        frame_paths: list[Path],
        threshold: int,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """对已抽取帧做 pHash 去重，返回移除数量"""
        try:
            import imagehash
        except ImportError:
            if message_callback:
                message_callback("imagehash 未安装，跳过抽帧去重")
            return 0

        kept_hashes: list[tuple[Path, imagehash.ImageHash]] = []
        removed = 0

        for frame_path in sorted(frame_paths):
            try:
                with Image.open(frame_path) as img:
                    current_hash = imagehash.phash(img)
            except Exception:
                continue

            is_duplicate = False
            for _, existing_hash in kept_hashes:
                if current_hash - existing_hash <= threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_hashes.append((frame_path, current_hash))
                continue

            try:
                frame_path.unlink()
                removed += 1
            except OSError:
                pass

        return removed

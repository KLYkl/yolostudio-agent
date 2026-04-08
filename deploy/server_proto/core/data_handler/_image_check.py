"""
_image_check.py - ImageCheckMixin: 图像完整性校验/格式转换/尺寸分析/重复检测
============================================
"""

from __future__ import annotations

import hashlib
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ExifTags

from core.data_handler._models import (
    IMAGE_EXTENSIONS,
    LABEL_EXTENSIONS,
    DuplicateGroup,
    ImageCheckResult,
    ImageSizeStats,
    _get_unique_dir,
)

# ============================================================
# Magic Bytes 映射表 — 用于文件头/扩展名不匹配检测
# ============================================================

_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"BM", "bmp"),
    (b"RIFF", "webp"),           # 需进一步检查 offset 8..11 == b"WEBP"
    (b"II\x2a\x00", "tiff"),     # little-endian TIFF
    (b"MM\x00\x2a", "tiff"),     # big-endian TIFF
]

# 扩展名 → 标准格式名
_EXT_TO_FORMAT: dict[str, str] = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".bmp": "bmp",
    ".webp": "webp",
    ".tiff": "tiff",
    ".tif": "tiff",
}

# PIL 保存格式名映射
_TARGET_FORMAT_MAP: dict[str, str] = {
    "JPEG (.jpg)": "JPEG",
    "PNG (.png)": "PNG",
    "BMP (.bmp)": "BMP",
    "WebP (.webp)": "WEBP",
}

# PIL 格式 → 文件扩展名
_FORMAT_TO_EXT: dict[str, str] = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "BMP": ".bmp",
    "WEBP": ".webp",
}


def _detect_real_format(file_path: Path) -> Optional[str]:
    """通过文件头 magic bytes 检测文件的真实图像格式"""
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
    except OSError:
        return None

    if len(header) < 2:
        return None

    for magic, fmt in _IMAGE_MAGIC:
        if header[: len(magic)] == magic:
            # RIFF 容器需进一步确认是否为 WebP
            if fmt == "webp" and len(header) >= 12:
                if header[8:12] != b"WEBP":
                    continue
            return fmt

    return None


class ImageCheckMixin:
    """图像完整性校验 / 格式转换 / 尺寸分析 / 重复检测 Mixin"""

    # ============================================================
    # 1. 图像完整性校验
    # ============================================================

    def check_image_integrity(
        self,
        img_dir: Path,
        *,
        check_corrupted: bool = True,
        check_zero_bytes: bool = True,
        check_format_mismatch: bool = True,
        check_exif_rotation: bool = False,
        quarantine_dir: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> ImageCheckResult:
        """
        校验图片完整性

        Args:
            img_dir: 图片目录
            check_corrupted: 检测损坏图片 (PIL verify + load)
            check_zero_bytes: 检测零字节文件
            check_format_mismatch: 检测文件头/扩展名不匹配
            check_exif_rotation: 检测 EXIF 旋转标记
            quarantine_dir: 隔离目录 (非 None 时自动移动问题文件)
            interrupt_check: 中断检查函数
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            ImageCheckResult
        """
        result = ImageCheckResult()
        images = self._find_images(img_dir)
        result.total_images = len(images)

        if result.total_images == 0:
            if message_callback:
                message_callback("未找到图片文件")
            return result

        if message_callback:
            message_callback(f"发现 {result.total_images} 张图片，开始校验...")

        quarantine_paths: list[Path] = []

        for i, img_path in enumerate(images):
            if interrupt_check():
                if message_callback:
                    message_callback("校验已取消")
                break

            is_problem = False

            # ---- 零字节检测 ----
            if check_zero_bytes:
                try:
                    if img_path.stat().st_size == 0:
                        result.zero_bytes.append(img_path)
                        is_problem = True
                        if progress_callback:
                            progress_callback(i + 1, result.total_images)
                        if is_problem and quarantine_dir:
                            quarantine_paths.append(img_path)
                        continue  # 零字节无法继续检查
                except OSError:
                    result.corrupted.append((img_path, "无法访问文件"))
                    is_problem = True
                    if progress_callback:
                        progress_callback(i + 1, result.total_images)
                    if is_problem and quarantine_dir:
                        quarantine_paths.append(img_path)
                    continue

            # ---- 文件头/扩展名不匹配 ----
            if check_format_mismatch:
                ext_format = _EXT_TO_FORMAT.get(img_path.suffix.lower())
                real_format = _detect_real_format(img_path)
                if ext_format and real_format and ext_format != real_format:
                    result.format_mismatch.append(
                        (img_path, img_path.suffix, real_format)
                    )
                    is_problem = True

            # ---- 损坏图片检测 ----
            if check_corrupted:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    # verify() 后需要重新打开才能 load()
                    with Image.open(img_path) as img:
                        img.load()
                except Exception as e:
                    error_msg = str(e)[:100] if str(e) else "未知错误"
                    result.corrupted.append((img_path, error_msg))
                    is_problem = True

            # ---- EXIF 旋转标记检测 ----
            if check_exif_rotation and not is_problem:
                try:
                    with Image.open(img_path) as img:
                        exif_data = img.getexif()
                        if exif_data:
                            orientation_key = None
                            for tag_id, tag_name in ExifTags.TAGS.items():
                                if tag_name == "Orientation":
                                    orientation_key = tag_id
                                    break
                            if orientation_key and orientation_key in exif_data:
                                orientation = exif_data[orientation_key]
                                if orientation != 1:  # 1 = 正常方向
                                    result.exif_rotation.append(
                                        (img_path, orientation)
                                    )
                except Exception:
                    pass  # EXIF 读取失败不算问题

            if is_problem and quarantine_dir:
                quarantine_paths.append(img_path)

            if progress_callback:
                progress_callback(i + 1, result.total_images)

        # 执行隔离
        if quarantine_dir and quarantine_paths:
            self._quarantine_files(quarantine_paths, quarantine_dir, message_callback)

        # 输出汇总
        if message_callback:
            if result.has_issues:
                parts = []
                if result.corrupted:
                    parts.append(f"损坏 {len(result.corrupted)}")
                if result.zero_bytes:
                    parts.append(f"零字节 {len(result.zero_bytes)}")
                if result.format_mismatch:
                    parts.append(f"格式不匹配 {len(result.format_mismatch)}")
                if result.exif_rotation:
                    parts.append(f"EXIF旋转 {len(result.exif_rotation)}")
                message_callback(
                    f"校验完成，发现 {result.issue_count} 个问题: "
                    f"{', '.join(parts)}"
                )
            else:
                message_callback("校验完成，未发现问题 ✓")

        return result

    # ============================================================
    # 2. 图像尺寸分析
    # ============================================================

    def analyze_image_sizes(
        self,
        img_dir: Path,
        *,
        small_threshold: int = 32,
        large_threshold: int = 8192,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> ImageSizeStats:
        """
        分析图片尺寸分布

        Args:
            img_dir: 图片目录
            small_threshold: 小于此值视为异常小 (默认 32px)
            large_threshold: 大于此值视为异常大 (默认 8192px)

        Returns:
            ImageSizeStats
        """
        stats = ImageSizeStats()
        images = self._find_images(img_dir)
        stats.total_images = len(images)

        if stats.total_images == 0:
            if message_callback:
                message_callback("未找到图片文件")
            return stats

        if message_callback:
            message_callback(f"分析 {stats.total_images} 张图片尺寸...")

        widths: list[int] = []
        heights: list[int] = []

        for i, img_path in enumerate(images):
            if interrupt_check():
                if message_callback:
                    message_callback("分析已取消")
                break

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    stats.sizes.append((img_path, w, h))
                    widths.append(w)
                    heights.append(h)

                    if w < small_threshold or h < small_threshold:
                        stats.abnormal_small.append(img_path)
                    if w > large_threshold or h > large_threshold:
                        stats.abnormal_large.append(img_path)
            except Exception:
                pass  # 损坏图片在完整性校验中处理

            if progress_callback:
                progress_callback(i + 1, stats.total_images)

        if widths and heights:
            stats.min_size = (min(widths), min(heights))
            stats.max_size = (max(widths), max(heights))
            stats.avg_size = (
                round(sum(widths) / len(widths)),
                round(sum(heights) / len(heights)),
            )

        if message_callback:
            message_callback(
                f"尺寸分析完成: "
                f"最小 {stats.min_size[0]}×{stats.min_size[1]}, "
                f"最大 {stats.max_size[0]}×{stats.max_size[1]}, "
                f"平均 {stats.avg_size[0]}×{stats.avg_size[1]}, "
                f"异常 {len(stats.abnormal_small) + len(stats.abnormal_large)} 张"
            )

        return stats

    # ============================================================
    # 3. 图像格式转换
    # ============================================================

    def convert_image_format(
        self,
        img_dir: Path,
        *,
        target_format: str = "JPEG",
        convert_rgb: bool = True,
        sync_labels: bool = True,
        label_dir: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        批量转换图片格式

        原始图片保留，转换结果输出到新目录。

        Args:
            img_dir: 图片目录
            target_format: 目标格式 ("JPEG" / "PNG" / "BMP" / "WEBP")
            convert_rgb: 是否统一转为 RGB 模式
            sync_labels: 是否同步复制并重命名标签文件
            label_dir: 标签目录 (可选)
            interrupt_check: 中断检查函数
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            成功转换的文件数量
        """
        images = self._find_images(img_dir)
        total = len(images)

        if total == 0:
            if message_callback:
                message_callback("未找到图片文件")
            return 0

        target_ext = _FORMAT_TO_EXT.get(target_format, ".jpg")

        # 创建输出目录
        output_dir = _get_unique_dir(
            img_dir.parent / f"{img_dir.name}_converted_{target_format.lower()}"
        )
        output_img_dir = output_dir / "images"
        output_img_dir.mkdir(parents=True, exist_ok=True)

        output_lbl_dir: Optional[Path] = None
        if sync_labels:
            output_lbl_dir = output_dir / "labels"
            output_lbl_dir.mkdir(parents=True, exist_ok=True)

        if message_callback:
            message_callback(
                f"开始转换 {total} 张图片 → {target_format}"
            )
            message_callback(f"输出目录: {output_dir}")

        converted = 0

        for i, img_path in enumerate(images):
            if interrupt_check():
                if message_callback:
                    message_callback("转换已取消")
                break

            try:
                with Image.open(img_path) as img:
                    if convert_rgb and img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")

                    new_name = img_path.stem + target_ext
                    output_path = output_img_dir / new_name

                    save_kwargs = {}
                    if target_format == "JPEG":
                        save_kwargs["quality"] = 95
                    elif target_format == "WEBP":
                        save_kwargs["quality"] = 95

                    img.save(output_path, format=target_format, **save_kwargs)
                    converted += 1

                # 同步标签
                if sync_labels and output_lbl_dir:
                    self._sync_label_for_converted_image(
                        img_path, img_path.stem, img_dir, label_dir, output_lbl_dir,
                    )

            except Exception as e:
                if message_callback:
                    message_callback(f"转换失败: {img_path.name} - {e}")

            if progress_callback:
                progress_callback(i + 1, total)

        if message_callback:
            message_callback(f"转换完成: 成功 {converted}/{total}")
            message_callback(f"文件已保存到: {output_dir}")

        return converted

    def _sync_label_for_converted_image(
        self,
        img_path: Path,
        stem: str,
        img_dir: Path,
        label_dir: Optional[Path],
        output_lbl_dir: Path,
    ) -> None:
        """为转换后的图片同步复制标签文件"""
        # 查找对应标签
        if label_dir and label_dir.exists():
            label_path, _ = self._find_label_in_dir(img_path, label_dir, img_dir=img_dir)
        else:
            label_path, _ = self._find_label(img_path, img_dir.parent)

        if label_path and label_path.exists():
            dest = output_lbl_dir / label_path.name
            shutil.copy2(str(label_path), str(dest))

    # ============================================================
    # 4. 重复图片检测
    # ============================================================

    def detect_duplicates(
        self,
        img_dir: Path,
        *,
        method: str = "md5",
        hash_threshold: int = 8,
        quarantine_dir: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> list[DuplicateGroup]:
        """
        检测重复图片

        Args:
            img_dir: 图片目录
            method: "md5" (精确匹配) 或 "phash" (感知哈希)
            hash_threshold: 感知哈希相似阈值 (汉明距离，仅 phash 模式)
            quarantine_dir: 隔离目录 (非 None 时自动移动重复文件，保留每组第一个)
            interrupt_check: 中断检查函数
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            重复图片组列表
        """
        images = self._find_images(img_dir)
        total = len(images)

        if total == 0:
            if message_callback:
                message_callback("未找到图片文件")
            return []

        method_text = "MD5 精确匹配" if method == "md5" else "感知哈希"
        if message_callback:
            message_callback(f"开始检测重复 ({method_text}): {total} 张图片")

        if method == "md5":
            groups = self._detect_duplicates_md5(
                images, interrupt_check, progress_callback, total,
            )
        else:
            groups = self._detect_duplicates_phash(
                images, hash_threshold, interrupt_check, progress_callback, total,
            )

        # 执行隔离 (保留每组第一个，移动其余)
        if quarantine_dir and groups:
            quarantine_paths = []
            for group in groups:
                quarantine_paths.extend(group.paths[1:])
            self._quarantine_files(quarantine_paths, quarantine_dir, message_callback)

        if message_callback:
            total_dup_files = sum(len(g.paths) for g in groups)
            if groups:
                message_callback(
                    f"检测完成: 发现 {len(groups)} 组重复，"
                    f"共 {total_dup_files} 个文件"
                )
            else:
                message_callback("检测完成，未发现重复图片 ✓")

        return groups

    def _detect_duplicates_md5(
        self,
        images: list[Path],
        interrupt_check: Callable[[], bool],
        progress_callback: Optional[Callable[[int, int], None]],
        total: int,
    ) -> list[DuplicateGroup]:
        """MD5 精确匹配检测"""
        hash_map: dict[str, list[Path]] = defaultdict(list)

        for i, img_path in enumerate(images):
            if interrupt_check():
                break

            try:
                md5 = hashlib.md5(img_path.read_bytes()).hexdigest()
                hash_map[md5].append(img_path)
            except OSError:
                pass

            if progress_callback:
                progress_callback(i + 1, total)

        return [
            DuplicateGroup(hash_value=h, paths=paths)
            for h, paths in hash_map.items()
            if len(paths) > 1
        ]

    def _detect_duplicates_phash(
        self,
        images: list[Path],
        threshold: int,
        interrupt_check: Callable[[], bool],
        progress_callback: Optional[Callable[[int, int], None]],
        total: int,
    ) -> list[DuplicateGroup]:
        """感知哈希 (pHash) 检测"""
        try:
            import imagehash
        except ImportError:
            return []

        # 第一步: 计算所有图片的 pHash
        hashes: list[tuple[Path, imagehash.ImageHash]] = []
        for i, img_path in enumerate(images):
            if interrupt_check():
                return []

            try:
                with Image.open(img_path) as img:
                    h = imagehash.phash(img)
                    hashes.append((img_path, h))
            except Exception:
                pass

            if progress_callback:
                progress_callback(i + 1, total)

        # 第二步: 分组 — 将汉明距离小于阈值的归为一组
        used: set[int] = set()
        groups: list[DuplicateGroup] = []

        for i in range(len(hashes)):
            if i in used:
                continue

            path_i, hash_i = hashes[i]
            group_paths = [path_i]

            for j in range(i + 1, len(hashes)):
                if j in used:
                    continue

                path_j, hash_j = hashes[j]
                if hash_i - hash_j <= threshold:
                    group_paths.append(path_j)
                    used.add(j)

            if len(group_paths) > 1:
                used.add(i)
                groups.append(
                    DuplicateGroup(hash_value=str(hash_i), paths=group_paths)
                )

        return groups

    # ============================================================
    # 5. 一键健康检查
    # ============================================================

    def run_health_check(
        self,
        img_dir: Path,
        *,
        label_dir: Optional[Path] = None,
        quarantine_dir: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        一键健康检查: 依次执行完整性校验 + 尺寸分析 + MD5 重复检测

        Returns:
            {"integrity": ImageCheckResult,
             "sizes": ImageSizeStats,
             "duplicates": list[DuplicateGroup]}
        """
        if message_callback:
            message_callback("=" * 40)
            message_callback("开始一键健康检查...")
            message_callback("=" * 40)

        # 1. 完整性校验
        if message_callback:
            message_callback("\n── 阶段 1/3: 图像完整性校验 ──")
        integrity = self.check_image_integrity(
            img_dir,
            quarantine_dir=quarantine_dir,
            interrupt_check=interrupt_check,
            progress_callback=progress_callback,
            message_callback=message_callback,
        )

        if interrupt_check():
            return {"integrity": integrity, "sizes": ImageSizeStats(), "duplicates": []}

        # 2. 尺寸分析
        if message_callback:
            message_callback("\n── 阶段 2/3: 图像尺寸分析 ──")
        sizes = self.analyze_image_sizes(
            img_dir,
            interrupt_check=interrupt_check,
            progress_callback=progress_callback,
            message_callback=message_callback,
        )

        if interrupt_check():
            return {"integrity": integrity, "sizes": sizes, "duplicates": []}

        # 3. 重复检测
        if message_callback:
            message_callback("\n── 阶段 3/3: 重复图片检测 (MD5) ──")
        duplicates = self.detect_duplicates(
            img_dir,
            method="md5",
            quarantine_dir=quarantine_dir,
            interrupt_check=interrupt_check,
            progress_callback=progress_callback,
            message_callback=message_callback,
        )

        if message_callback:
            message_callback("\n" + "=" * 40)
            message_callback("健康检查完成")
            total_issues = integrity.issue_count + len(duplicates)
            abnormal_count = len(sizes.abnormal_small) + len(sizes.abnormal_large)
            if total_issues > 0 or abnormal_count > 0:
                message_callback(
                    f"问题汇总: 完整性问题 {integrity.issue_count}, "
                    f"异常尺寸 {abnormal_count}, "
                    f"重复组 {len(duplicates)}"
                )
            else:
                message_callback("所有检查通过 ✓")

        return {
            "integrity": integrity,
            "sizes": sizes,
            "duplicates": duplicates,
        }

    # ============================================================
    # 6. 导出报告
    # ============================================================

    def export_check_report(
        self,
        output_path: Path,
        *,
        integrity: Optional[ImageCheckResult] = None,
        sizes: Optional[ImageSizeStats] = None,
        duplicates: Optional[list[DuplicateGroup]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """
        导出检查报告到文本文件

        Args:
            output_path: 输出文件路径
            integrity: 完整性校验结果
            sizes: 尺寸分析结果
            duplicates: 重复检测结果

        Returns:
            报告文件路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("图像健康检查报告")
        lines.append("=" * 60)
        lines.append("")

        # ---- 完整性校验 ----
        if integrity:
            lines.append("── 图像完整性校验 ──")
            lines.append(f"扫描图片总数: {integrity.total_images}")
            lines.append(f"问题总数: {integrity.issue_count}")
            lines.append("")

            if integrity.corrupted:
                lines.append(f"  损坏图片 ({len(integrity.corrupted)}):")
                for path, reason in integrity.corrupted:
                    lines.append(f"    {path} - {reason}")
                lines.append("")

            if integrity.zero_bytes:
                lines.append(f"  零字节文件 ({len(integrity.zero_bytes)}):")
                for path in integrity.zero_bytes:
                    lines.append(f"    {path}")
                lines.append("")

            if integrity.format_mismatch:
                lines.append(f"  格式不匹配 ({len(integrity.format_mismatch)}):")
                for path, ext, real_fmt in integrity.format_mismatch:
                    lines.append(f"    {path} (扩展名 {ext}, 实际 {real_fmt})")
                lines.append("")

            if integrity.exif_rotation:
                lines.append(f"  EXIF 旋转标记 ({len(integrity.exif_rotation)}):")
                for path, orientation in integrity.exif_rotation:
                    lines.append(f"    {path} (orientation={orientation})")
                lines.append("")

        # ---- 尺寸分析 ----
        if sizes and sizes.total_images > 0:
            lines.append("── 图像尺寸分析 ──")
            lines.append(f"图片总数: {sizes.total_images}")
            lines.append(f"最小尺寸: {sizes.min_size[0]}×{sizes.min_size[1]}")
            lines.append(f"最大尺寸: {sizes.max_size[0]}×{sizes.max_size[1]}")
            lines.append(f"平均尺寸: {sizes.avg_size[0]}×{sizes.avg_size[1]}")
            lines.append("")

            if sizes.abnormal_small:
                lines.append(f"  异常小图片 ({len(sizes.abnormal_small)}):")
                for path in sizes.abnormal_small:
                    lines.append(f"    {path}")
                lines.append("")

            if sizes.abnormal_large:
                lines.append(f"  异常大图片 ({len(sizes.abnormal_large)}):")
                for path in sizes.abnormal_large:
                    lines.append(f"    {path}")
                lines.append("")

        # ---- 重复检测 ----
        if duplicates:
            total_dup = sum(len(g.paths) for g in duplicates)
            lines.append("── 重复图片检测 ──")
            lines.append(f"重复组数: {len(duplicates)}")
            lines.append(f"涉及文件: {total_dup}")
            lines.append("")
            for idx, group in enumerate(duplicates, 1):
                lines.append(f"  组 {idx} (hash={group.hash_value}):")
                for path in group.paths:
                    lines.append(f"    {path}")
                lines.append("")

        lines.append("=" * 60)
        lines.append("报告结束")
        lines.append("=" * 60)

        output_path.write_text("\n".join(lines), encoding="utf-8")

        if message_callback:
            message_callback(f"报告已保存: {output_path}")

        return output_path

    # ============================================================
    # 辅助方法
    # ============================================================

    def _quarantine_files(
        self,
        files: list[Path],
        quarantine_dir: Path,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """将问题文件移动到隔离目录"""
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for file_path in files:
            try:
                dest = quarantine_dir / file_path.name
                # 避免覆盖
                if dest.exists():
                    dest = quarantine_dir / f"{file_path.stem}_{moved}{file_path.suffix}"
                shutil.move(str(file_path), str(dest))
                moved += 1
            except OSError:
                pass
        if message_callback and moved > 0:
            message_callback(f"已隔离 {moved} 个文件到: {quarantine_dir}")
        return moved

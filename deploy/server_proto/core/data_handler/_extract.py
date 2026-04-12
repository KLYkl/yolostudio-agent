"""
_extract.py - ExtractMixin: 图片抽取功能
============================================

支持三种抽取模式:
    - random: 随机抽取
    - by_category: 按类别抽取 (单类/混合/空标签/无标签)
    - by_directory: 按目录独立抽取
"""

from __future__ import annotations

import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from core.data_handler._models import (
    ExtractConfig,
    ExtractResult,
    LabelFormat,
    _get_unique_dir,
)


class ExtractMixin:
    """图片抽取功能 Mixin"""

    def scan_subdirs(self, img_dir: Path) -> dict[str, int]:
        """
        扫描目录结构，返回各子目录的图片数量

        Args:
            img_dir: 图片根目录

        Returns:
            {相对路径字符串: 图片数量} 字典
            根目录使用 "." 表示
        """
        result: dict[str, int] = {}

        # 根目录下直接的图片
        root_images = self._find_images_flat(img_dir)
        if root_images:
            result["."] = len(root_images)

        # 遍历子目录
        for sub_dir in sorted(img_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            # 跳过隐藏目录
            if sub_dir.name.startswith(".") or sub_dir.name.startswith("_"):
                continue
            sub_images = self._find_images(sub_dir)
            if sub_images:
                try:
                    rel = str(sub_dir.relative_to(img_dir))
                except ValueError:
                    rel = sub_dir.name
                result[rel] = len(sub_images)

        return result

    def _find_images_flat(self, directory: Path) -> list[Path]:
        """只查找指定目录下直接的图片文件 (不递归子目录)"""
        from core.data_handler._models import IMAGE_EXTENSIONS

        images = []
        if not directory.exists():
            return images

        for f in sorted(directory.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(f)
        return images

    def _classify_image(
        self,
        img_path: Path,
        label_dir: Optional[Path],
        img_dir: Path,
        class_mapping: Optional[dict[int, str]] = None,
    ) -> str:
        """
        分类单张图片 (与 categorize_by_class 逻辑一致)

        Returns:
            类别名称:
            - "{class_name}": 单一类别
            - "_mixed": 多类别
            - "_empty": 空标签
            - "_no_label": 无标签
        """
        # 查找标签
        if label_dir and label_dir.exists():
            label_path, label_format = self._find_label_in_dir(
                img_path, label_dir, img_dir=img_dir
            )
        else:
            label_path, label_format = self._find_label(img_path, img_dir.parent)

        if label_path is None:
            return "_no_label"

        # 解析标签
        class_ids = self._parse_label(
            label_path, label_format, class_mapping=class_mapping or {}
        )
        unique_ids = sorted(set(class_ids))

        if len(unique_ids) == 0:
            return "_empty"
        elif len(unique_ids) == 1:
            return unique_ids[0]
        else:
            return "_mixed"

    def preview_extract(
        self,
        img_dir: Path,
        label_dir: Optional[Path],
        config: ExtractConfig,
        classes_txt: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> ExtractResult:
        """
        预估抽取结果 (不执行文件操作)

        Args:
            img_dir: 图片根目录
            label_dir: 标签目录
            config: 抽取配置
            classes_txt: classes.txt 路径
            interrupt_check: 中断检查
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            ExtractResult (不含 output_dir 和 conflicts)
        """
        result = ExtractResult()

        # 收集图片
        images = self._collect_extract_images(img_dir, config)
        result.total_available = len(images)

        if not images:
            if message_callback:
                message_callback("未找到符合条件的图片")
            return result

        # 按模式分组 + 抽样
        if config.mode == "by_category":
            grouped, _ = self._group_by_category(
                images, img_dir, label_dir, config,
                classes_txt=classes_txt,
                interrupt_check=interrupt_check,
                progress_callback=progress_callback,
                message_callback=message_callback,
            )
            if interrupt_check():
                return result
        else:
            grouped = self._group_by_directory(images, img_dir)

        # 按每组独立抽样
        selected = self._sample_per_group(grouped, config)
        result.extracted = len(selected)

        # 统计各目录
        for img in selected:
            try:
                rel_dir = str(img.parent.relative_to(img_dir))
            except ValueError:
                rel_dir = "."
            dir_name = rel_dir if rel_dir != "." else "."
            result.dir_stats[dir_name] = result.dir_stats.get(dir_name, 0) + 1

        return result

    def extract_images(
        self,
        img_dir: Path,
        label_dir: Optional[Path],
        config: ExtractConfig,
        classes_txt: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> ExtractResult:
        """
        执行图片抽取

        Args:
            img_dir: 图片根目录
            label_dir: 标签目录
            config: 抽取配置
            classes_txt: classes.txt 路径
            interrupt_check: 中断检查
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            ExtractResult
        """
        result = ExtractResult()

        # 确定输出目录
        output_dir = config.output_dir
        if output_dir is None:
            output_dir = _get_unique_dir(img_dir.parent / f"{img_dir.name}_extracted")
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = str(output_dir)

        # 收集图片
        images = self._collect_extract_images(img_dir, config)
        result.total_available = len(images)

        if not images:
            if message_callback:
                message_callback("未找到符合条件的图片")
            return result

        if message_callback:
            message_callback(f"可用图片: {len(images)} 张")

        # 按模式分组 + 抽样
        category_map: dict[Path, str] = {}  # 图片→类别映射 (按类别布局用)
        if config.mode == "by_category":
            grouped, category_map = self._group_by_category(
                images, img_dir, label_dir, config,
                classes_txt=classes_txt,
                interrupt_check=interrupt_check,
                progress_callback=progress_callback,
                message_callback=message_callback,
            )
            if interrupt_check():
                if message_callback:
                    message_callback("抽取已取消")
                return result
        else:
            grouped = self._group_by_directory(images, img_dir)

        # 按每组独立抽样
        selected = self._sample_per_group(grouped, config)

        if message_callback:
            message_callback(f"将抽取 {len(selected)} 张图片...")

        # 复制文件
        total = len(selected)
        actually_copied: list[Path] = []
        for i, img_path in enumerate(selected):
            if interrupt_check():
                if message_callback:
                    message_callback("抽取已取消")
                break

            # 确定目标路径
            img_category = category_map.get(img_path)
            dest_img = self._build_extract_dest_path(
                img_path, img_dir, output_dir, config.output_layout,
                category=img_category,
            )

            # 检查冲突
            if dest_img.exists():
                result.conflicts.append((img_path, dest_img))
                if progress_callback:
                    progress_callback(i + 1, total)
                continue

            # 复制图片
            try:
                dest_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(img_path), str(dest_img))
                result.extracted += 1
                actually_copied.append(img_path)
            except OSError as e:
                if message_callback:
                    message_callback(f"复制失败: {img_path.name} - {e}")
                if progress_callback:
                    progress_callback(i + 1, total)
                continue

            # 统计目录
            try:
                rel_dir = str(img_path.parent.relative_to(img_dir))
            except ValueError:
                rel_dir = "."
            result.dir_stats[rel_dir] = result.dir_stats.get(rel_dir, 0) + 1

            # 复制标签
            if config.copy_labels:
                try:
                    label_path = self._find_extract_label(img_path, label_dir, img_dir)
                    if label_path and label_path.exists():
                        dest_label = self._build_extract_dest_path(
                            label_path,
                            label_dir if label_dir else img_dir.parent,
                            output_dir,
                            config.output_layout,
                            label_mode=True,
                            img_name=img_path.stem,
                            category=img_category,
                        )
                        dest_label.parent.mkdir(parents=True, exist_ok=True)
                        if not dest_label.exists():
                            shutil.copy2(str(label_path), str(dest_label))
                            result.labels_copied += 1
                except OSError as e:
                    if message_callback:
                        message_callback(f"复制标签失败: {img_path.name} - {e}")

            if progress_callback:
                progress_callback(i + 1, total)

        # 生成抽取日志 (仅记录实际成功复制的文件)
        self._write_extract_log(output_dir, config, result, actually_copied)

        # 输出统计
        if message_callback:
            message_callback("=" * 40)
            message_callback("抽取完成:")
            message_callback(f"  提取: {result.extracted} 张图片")
            if result.labels_copied > 0:
                message_callback(f"  标签: {result.labels_copied} 个")
            if result.conflicts:
                message_callback(f"  冲突: {len(result.conflicts)} 个 (待处理)")
            for dir_name, count in sorted(result.dir_stats.items()):
                message_callback(f"  📂 {dir_name}: {count} 张")

        return result

    # ==================== 内部方法 ====================

    def _collect_extract_images(
        self, img_dir: Path, config: ExtractConfig
    ) -> list[Path]:
        """根据配置收集目标图片"""
        if config.selected_dirs:
            images: list[Path] = []
            for rel_dir in config.selected_dirs:
                if str(rel_dir) == ".":
                    images.extend(self._find_images_flat(img_dir))
                else:
                    sub = img_dir / rel_dir
                    if sub.exists():
                        images.extend(self._find_images(sub))
            return sorted(set(images))
        else:
            return self._find_images(img_dir)

    def _group_by_category(
        self,
        images: list[Path],
        img_dir: Path,
        label_dir: Optional[Path],
        config: ExtractConfig,
        classes_txt: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[dict[str, list[Path]], dict[Path, str]]:
        """
        按类别分组图片

        Returns:
            (grouped, category_map)
            grouped: {类别名: [图片路径]}
            category_map: {图片路径: 类别名} (供按类别布局使用)
        """
        if not config.categories:
            return {}, {}

        # 加载类别映射
        class_mapping: dict[int, str] = {}
        if classes_txt and classes_txt.exists():
            _, class_mapping = self._read_classes_txt(classes_txt)

        if message_callback:
            message_callback(
                f"正在按类别分组 ({', '.join(config.categories)})..."
            )

        grouped: dict[str, list[Path]] = {}
        category_map: dict[Path, str] = {}
        total = len(images)

        for i, img_path in enumerate(images):
            if interrupt_check():
                break

            category = self._classify_image(
                img_path, label_dir, img_dir, class_mapping
            )
            if category in config.categories:
                grouped.setdefault(category, []).append(img_path)
                category_map[img_path] = category

            if progress_callback:
                progress_callback(i + 1, total)

        return grouped, category_map

    def _group_by_directory(
        self,
        images: list[Path],
        img_dir: Path,
    ) -> dict[str, list[Path]]:
        """按目录分组图片，返回 {相对路径: [图片路径]} 字典"""
        grouped: dict[str, list[Path]] = {}
        for img in images:
            try:
                rel_dir = str(img.parent.relative_to(img_dir))
            except ValueError:
                rel_dir = "."
            grouped.setdefault(rel_dir, []).append(img)
        return grouped

    def _sample_per_group(
        self,
        grouped: dict[str, list[Path]],
        config: ExtractConfig,
    ) -> list[Path]:
        """
        按 per_item_counts 对每组独立抽样

        per_item_counts 格式: {组名: (模式, 值)}
            - ("all", 0): 取该组全部
            - ("count", N): 随机取 N 张
            - ("ratio", R): 随机取 总数*R 张
        """
        if not grouped:
            return []

        rng = random.Random(config.seed)
        selected: list[Path] = []

        for group_name, group_images in grouped.items():
            item_config = config.per_item_counts.get(group_name)
            if not item_config:
                # 未配置的组跳过
                continue

            count_mode, value = item_config

            if count_mode == "all":
                selected.extend(group_images)
            elif count_mode == "ratio":
                count = max(1, int(len(group_images) * value))
                count = min(count, len(group_images))
                selected.extend(rng.sample(group_images, count))
            else:  # "count"
                count = min(int(value), len(group_images))
                if count <= 0:
                    continue
                selected.extend(rng.sample(group_images, count))

        return sorted(selected)

    def _build_extract_dest_path(
        self,
        source: Path,
        source_root: Path,
        output_dir: Path,
        output_layout: str,
        label_mode: bool = False,
        img_name: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Path:
        """
        构建目标文件路径

        Args:
            source: 源文件路径
            source_root: 源文件根目录
            output_dir: 输出目录
            output_layout: 输出布局 ("keep"/"flat"/"by_category")
            label_mode: 标签文件模式 (输出到 labels/ 子目录)
            img_name: 对应图片文件名 (用于标签文件扁平化命名)
            category: 图片所属类别 (仅 by_category 布局使用)
        """
        if output_layout == "by_category" and category:
            # 按类别放置: output_dir / 类别名 / images(或labels) / 文件名
            sub_dir = "labels" if label_mode else "images"
            return output_dir / category / sub_dir / source.name
        elif output_layout == "keep":
            # 保持原始目录结构
            try:
                rel = source.relative_to(source_root)
                return output_dir / rel
            except ValueError:
                return output_dir / source.name
        else:
            # 扁平化: 先尝试原名，仅同名冲突时才加目录前缀
            sub_dir = "labels" if label_mode else "images"
            base_dest = output_dir / sub_dir / source.name

            if not base_dest.exists():
                return base_dest

            # 原名已被占用 → 加目录前缀去重
            try:
                rel_dir = source.parent.relative_to(source_root)
                if str(rel_dir) != ".":
                    prefix = str(rel_dir).replace("/", "_").replace("\\", "_")
                    new_name = f"{prefix}_{source.name}"
                else:
                    new_name = source.name
            except ValueError:
                new_name = source.name

            return output_dir / sub_dir / new_name

    def _find_extract_label(
        self,
        img_path: Path,
        label_dir: Optional[Path],
        img_dir: Path,
    ) -> Optional[Path]:
        """查找图片对应的标签文件"""
        if label_dir and label_dir.exists():
            label_path, _ = self._find_label_in_dir(
                img_path, label_dir, img_dir=img_dir
            )
            return label_path
        else:
            label_path, _ = self._find_label(img_path, img_dir.parent)
            return label_path

    def _write_extract_log(
        self,
        output_dir: Path,
        config: ExtractConfig,
        result: ExtractResult,
        selected: list[Path],
    ) -> None:
        """生成抽取日志文件"""
        log_path = output_dir / "_extract_log.txt"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "# 图片抽取日志",
            f"# 生成时间: {now}",
            "# ========================================",
            "",
            f"抽取模式: {config.mode}",
        ]

        # 记录每组的抽取配置
        if config.per_item_counts:
            lines.append("各组抽取配置:")
            for name, (mode, value) in config.per_item_counts.items():
                if mode == "all":
                    lines.append(f"  {name}: 全部")
                elif mode == "count":
                    lines.append(f"  {name}: {int(value)} 张")
                elif mode == "ratio":
                    lines.append(f"  {name}: {value:.1%}")

        if config.categories:
            lines.append(f"目标类别: {', '.join(config.categories)}")
        if config.seed is not None:
            lines.append(f"随机种子: {config.seed}")

        layout_labels = {"keep": "保持目录结构", "flat": "扁平化", "by_category": "按类别放置"}
        lines.append(f"输出布局: {layout_labels.get(config.output_layout, config.output_layout)}")
        lines.append(f"复制标签: {config.copy_labels}")
        lines.append("")
        lines.append("# ========================================")
        lines.append(f"可用图片: {result.total_available}")
        lines.append(f"实际提取: {result.extracted}")
        lines.append(f"标签复制: {result.labels_copied}")
        lines.append(f"文件冲突: {len(result.conflicts)}")
        lines.append("")

        # 各目录统计
        if result.dir_stats:
            lines.append("# 目录统计:")
            for dir_name, count in sorted(result.dir_stats.items()):
                lines.append(f"  {dir_name}: {count} 张")
            lines.append("")

        # 文件清单
        lines.append("# 提取文件清单:")
        for img_path in selected:
            lines.append(f"  {img_path}")

        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except OSError:
            pass  # 日志文件写入失败不影响主流程

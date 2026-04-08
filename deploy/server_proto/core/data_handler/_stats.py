"""
_stats.py - StatsMixin: 统计/分类相关
============================================
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

from core.data_handler._models import (
    LabelFormat,
    _get_unique_dir,
)


class StatsMixin:
    """统计/分类功能 Mixin"""

    def _read_classes_txt(self, classes_file: Path) -> tuple[list[str], dict[int, str]]:
        """读取 classes.txt，返回类别列表和 ID→名称映射"""
        if not classes_file.exists():
            return [], {}

        classes = []
        with open(classes_file, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    classes.append(name)

        return classes, {i: name for i, name in enumerate(classes)}

    def load_classes_txt(self, classes_file: Path) -> list[str]:
        """
        加载 classes.txt 文件

        Args:
            classes_file: classes.txt 路径

        Returns:
            类别名称列表 (有序)
        """
        classes, class_mapping = self._read_classes_txt(classes_file)
        self._class_mapping = class_mapping
        return classes

    def _resolve_class_id(self, value: str, class_mapping: Optional[dict[int, str]] = None) -> Optional[int]:
        """将类别名称解析为 ID 或直接返回 ID"""
        normalized = value.strip()
        if not normalized:
            return None

        if normalized.isdigit():
            return int(normalized)

        for id_, name in (class_mapping or {}).items():
            if name == normalized:
                return id_

        return None

    def categorize_by_class(
        self,
        img_dir: Path,
        label_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        classes_txt: Optional[Path] = None,
        include_no_label: bool = True,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, int]:
        """
        按类别分类数据集

        将图片和标签按类别复制到对应文件夹:
            - 空标签 -> _empty/
            - 单一类别 -> {class_id}/
            - 多类别 -> _mixed/
            - 无标签 -> _no_label/

        Args:
            img_dir: 图片目录
            label_dir: 标签目录 (可选，留空则自动检测)
            output_dir: 输出目录 (默认 img_dir.parent / "{img_dir.name}_categorized")
            classes_txt: classes.txt 路径 (可选，用于 TXT 标签类别映射)
            include_no_label: 是否包含无标签图片
            interrupt_check: 中断检查函数
            progress_callback: 进度回调 (current, total)
            message_callback: 消息回调

        Returns:
            分类统计 {类别名: 数量}
        """
        # 加载类别映射
        class_mapping: dict[int, str] = {}
        if classes_txt and classes_txt.exists():
            _, class_mapping = self._read_classes_txt(classes_txt)
            if message_callback:
                message_callback(f"已加载类别文件: {len(class_mapping)} 个类别")

        # 默认输出目录 (如果已存在则添加数字后缀)
        if output_dir is None:
            output_dir = _get_unique_dir(img_dir.parent / f"{img_dir.name}_categorized")

        output_dir.mkdir(parents=True, exist_ok=True)

        if message_callback:
            message_callback(f"输出目录: {output_dir}")

        # 收集所有图片
        images = self._find_images(img_dir)
        total = len(images)

        if total == 0:
            if message_callback:
                message_callback("未找到图片文件")
            return {}

        if message_callback:
            message_callback(f"发现 {total} 张图片，开始分类...")

        # 统计结果
        stats: dict[str, int] = {}
        # 混合类别报告: {文件名: [类别ID列表]}
        mixed_report: dict[str, list[str]] = {}

        for i, img_path in enumerate(images):
            if interrupt_check():
                if message_callback:
                    message_callback("分类已取消")
                break

            # 查找标签
            if label_dir and label_dir.exists():
                label_path, label_format = self._find_label_in_dir(img_path, label_dir, img_dir=img_dir)
            else:
                label_path, label_format = self._find_label(img_path, img_dir.parent)

            # 确定分类目标
            if label_path is None:
                # 无标签
                if not include_no_label:
                    if progress_callback:
                        progress_callback(i + 1, total)
                    continue
                category = "_no_label"
                class_ids: list[str] = []
            else:
                # 解析标签获取类别 ID (使用原始 ID，不映射名称)
                class_ids = self._parse_label(label_path, label_format, class_mapping=class_mapping)
                unique_ids = sorted(set(class_ids))

                if len(unique_ids) == 0:
                    category = "_empty"
                elif len(unique_ids) == 1:
                    category = unique_ids[0]
                else:
                    category = "_mixed"
                    mixed_report[img_path.name] = unique_ids

            # 创建目标目录
            cat_dir = output_dir / category
            cat_img_dir = cat_dir / "images"
            cat_lbl_dir = cat_dir / "labels"
            cat_img_dir.mkdir(parents=True, exist_ok=True)
            if category != "_no_label":
                cat_lbl_dir.mkdir(parents=True, exist_ok=True)

            # 复制图片
            dest_img = cat_img_dir / img_path.name
            shutil.copy2(str(img_path), str(dest_img))

            # 复制标签 (如果存在)
            if label_path and label_path.exists() and category != "_no_label":
                dest_lbl = cat_lbl_dir / label_path.name
                shutil.copy2(str(label_path), str(dest_lbl))

            # 统计
            stats[category] = stats.get(category, 0) + 1

            if progress_callback:
                progress_callback(i + 1, total)

        # 生成混合类别报告
        if mixed_report:
            report_path = output_dir / "_mixed_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# 混合类别报告\n")
                f.write("# 格式: 文件名 -> 类别ID列表\n")
                f.write("# ========================================\n\n")
                for filename, ids in sorted(mixed_report.items()):
                    f.write(f"{filename} -> {', '.join(ids)}\n")
            if message_callback:
                message_callback(f"混合类别报告已保存: {report_path.name}")

        # 输出统计
        if message_callback:
            message_callback("=" * 40)
            message_callback("分类完成，统计如下:")
            for cat, count in sorted(stats.items()):
                message_callback(f"  {cat}: {count} 张")
            message_callback(f"  合计: {sum(stats.values())} 张")

        return stats

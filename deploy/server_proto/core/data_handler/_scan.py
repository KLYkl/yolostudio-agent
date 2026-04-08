"""
_scan.py - ScanMixin: 数据集扫描
============================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from core.data_handler._models import (
    IMAGE_EXTENSIONS,
    LabelFormat,
    ScanResult,
)


class ScanMixin:
    """数据集扫描功能 Mixin"""

    def scan_dataset(
        self,
        img_dir: Path,
        label_dir: Optional[Path] = None,
        classes_txt: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> ScanResult:
        """
        扫描数据集，统计图片和标签信息

        Args:
            img_dir: 图片目录
            label_dir: 标签目录 (可选，留空则自动检测)
            classes_txt: classes.txt 路径 (可选，用于 TXT 标签类别映射)
            interrupt_check: 中断检查函数
            progress_callback: 进度回调 (current, total)
            message_callback: 消息回调

        Returns:
            ScanResult: 扫描结果
        """
        result = ScanResult()
        class_mapping: dict[int, str] = {}

        # 加载类别映射
        if classes_txt and classes_txt.exists():
            _, class_mapping = self._read_classes_txt(classes_txt)
            if message_callback:
                message_callback(f"已加载类别文件: {len(class_mapping)} 个类别")

        # 收集所有图片文件
        images = self._find_images(img_dir)
        result.total_images = len(images)

        if message_callback:
            msg = f"发现 {result.total_images} 张图片"
            if label_dir:
                msg += f"，标签目录: {label_dir.name}"
            else:
                msg += "，自动搜索标签..."
            message_callback(msg)

        # 扫描每张图片的标签
        for i, img_path in enumerate(images):
            if interrupt_check():
                if message_callback:
                    message_callback("扫描已取消")
                break

            # 查找标签: 优先使用指定目录
            if label_dir and label_dir.exists():
                label_path, label_format = self._find_label_in_dir(img_path, label_dir, img_dir=img_dir)
            else:
                label_path, label_format = self._find_label(img_path, img_dir.parent)

            if label_path is None:
                result.missing_labels.append(img_path)
            else:
                result.labeled_images += 1
                result.label_format = label_format

                # 解析标签内容
                classes_in_file = self._parse_label(label_path, label_format, class_mapping=class_mapping)

                if not classes_in_file:
                    result.empty_labels += 1
                else:
                    for cls in classes_in_file:
                        result.class_stats[cls] = result.class_stats.get(cls, 0) + 1

            if progress_callback:
                progress_callback(i + 1, result.total_images)

        # 提取有序类别列表
        result.classes = sorted(result.class_stats.keys())

        return result

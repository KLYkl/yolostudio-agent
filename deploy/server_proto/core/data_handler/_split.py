"""
_split.py - SplitMixin: 数据集划分
============================================
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Callable, Optional

import yaml

from core.data_handler._models import (
    SplitMode,
    SplitResult,
    _get_unique_dir,
)

import os


class SplitMixin:
    """数据集划分功能 Mixin"""

    def split_dataset(
        self,
        img_dir: Path,
        label_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        ratio: float = 0.8,
        seed: int = 42,
        mode: SplitMode = SplitMode.COPY,
        ignore_orphans: bool = False,
        clear_output: bool = False,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> SplitResult:
        """
        划分数据集为训练集和验证集

        Args:
            img_dir: 图片目录
            label_dir: 标签目录 (可选，留空则自动检测)
            output_dir: 输出目录 (可选，默认为 img_dir 同级的 _split 目录)
            ratio: 训练集比例 (0.0-1.0)
            seed: 随机种子
            mode: 划分模式 (MOVE/COPY/INDEX)
            ignore_orphans: 是否忽略无标签图片
            clear_output: 是否清空目标目录
            interrupt_check: 中断检查函数
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            SplitResult: 划分结果
        """
        result = SplitResult()

        # 默认输出目录 (如果已存在则添加数字后缀)
        if output_dir is None:
            output_dir = _get_unique_dir(img_dir.parent / f"{img_dir.name}_split")

        # 收集所有图片
        images = self._find_images(img_dir)

        # 如果忽略无标签图片，过滤掉孤立图片
        if ignore_orphans:
            labeled_images = []
            for img in images:
                if label_dir and label_dir.exists():
                    label_path, _ = self._find_label_in_dir(img, label_dir, img_dir=img_dir)
                else:
                    label_path, _ = self._find_label(img, img_dir.parent)
                if label_path is not None:
                    labeled_images.append(img)
            if message_callback:
                skipped = len(images) - len(labeled_images)
                message_callback(f"忽略 {skipped} 张无标签图片")
            images = labeled_images

        total = len(images)

        if total == 0:
            if message_callback:
                message_callback("未找到符合条件的图片文件")
            return result

        # 随机打乱
        random.seed(seed)
        random.shuffle(images)

        # 划分
        split_idx = int(total * ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        result.train_count = len(train_images)
        result.val_count = len(val_images)

        if message_callback:
            message_callback(f"划分比例: 训练集 {result.train_count}, 验证集 {result.val_count}")
            message_callback(f"输出目录: {output_dir}")

        if mode in (SplitMode.MOVE, SplitMode.COPY):
            result.train_path, result.val_path = self._split_files(
                img_dir, train_images, val_images,
                label_dir=label_dir,
                output_dir=output_dir,
                use_copy=(mode == SplitMode.COPY),
                clear_output=clear_output,
                interrupt_check=interrupt_check,
                progress_callback=progress_callback,
                message_callback=message_callback
            )
        else:
            result.train_path, result.val_path = self._split_index(
                output_dir, train_images, val_images, message_callback
            )

        return result

    def generate_yaml(
        self,
        train_path: str,
        val_path: str,
        classes: list[str],
        output_path: Path,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        生成 YOLO 训练 YAML 配置文件

        path 字段智能推断：
            - 两个绝对路径 → 取公共父目录，train/val 转为相对路径
            - 其他情况 → 使用 YAML 所在目录

        Args:
            train_path: 训练集路径 (文件夹或 txt 索引)
            val_path: 验证集路径 (文件夹或 txt 索引)
            classes: 类别名称列表
            output_path: YAML 输出路径
            message_callback: 消息回调

        Returns:
            是否成功
        """
        try:
            train_p = Path(train_path)
            val_p = Path(val_path)

            if train_p.is_absolute() and val_p.is_absolute():
                dataset_root = Path(os.path.commonpath([train_p, val_p]))
                train_path = str(train_p.relative_to(dataset_root))
                val_path = str(val_p.relative_to(dataset_root))
            else:
                dataset_root = output_path.parent

            yaml_content = {
                "path": str(dataset_root),
                "train": train_path,
                "val": val_path,
                "names": {i: name for i, name in enumerate(classes)},
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            if message_callback:
                message_callback(f"YAML 配置已保存: {output_path}")
                message_callback(f"数据集根目录 (path): {dataset_root}")

            return True

        except Exception as e:
            if message_callback:
                message_callback(f"YAML 生成失败: {e}")
            return False

    def _split_files(
        self,
        img_dir: Path,
        train_images: list[Path],
        val_images: list[Path],
        label_dir: Optional[Path],
        output_dir: Path,
        use_copy: bool,
        clear_output: bool,
        interrupt_check: Callable[[], bool],
        progress_callback: Optional[Callable[[int, int], None]],
        message_callback: Optional[Callable[[str], None]],
    ) -> tuple[str, str]:
        """
        物理文件划分 (移动或复制, YOLO 标准目录结构)

        创建结构:
            output_dir/
            ├── images/
            │   ├── train/
            │   └── val/
            └── labels/
                ├── train/
                └── val/
        """
        # YOLO 标准目录结构
        train_img_dir = output_dir / "images" / "train"
        train_lbl_dir = output_dir / "labels" / "train"
        val_img_dir = output_dir / "images" / "val"
        val_lbl_dir = output_dir / "labels" / "val"

        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 清空目标目录
        if clear_output:
            for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
            if message_callback:
                message_callback("已清空目标目录")

        mode_str = "复制" if use_copy else "移动"
        if message_callback:
            message_callback(f"使用 {mode_str} 模式，创建 YOLO 标准目录结构")

        total = len(train_images) + len(val_images)
        current = 0

        # 处理训练集
        for img_path in train_images:
            if interrupt_check():
                break

            log_msg = self._transfer_with_label(img_path, train_img_dir, train_lbl_dir, img_dir, label_dir, use_copy)
            if log_msg and message_callback:
                message_callback(log_msg)
            current += 1
            if progress_callback:
                progress_callback(current, total)

        # 处理验证集
        for img_path in val_images:
            if interrupt_check():
                break

            log_msg = self._transfer_with_label(img_path, val_img_dir, val_lbl_dir, img_dir, label_dir, use_copy)
            if log_msg and message_callback:
                message_callback(log_msg)
            current += 1
            if progress_callback:
                progress_callback(current, total)

        # 返回相对路径用于 YAML
        return "images/train", "images/val"

    def _split_index(
        self,
        root: Path,
        train_images: list[Path],
        val_images: list[Path],
        message_callback: Optional[Callable[[str], None]],
    ) -> tuple[str, str]:
        """索引文件模式划分（使用相对路径，便于数据集迁移）"""
        root.mkdir(parents=True, exist_ok=True)
        train_txt = root / "train.txt"
        val_txt = root / "val.txt"

        try:
            with open(train_txt, "w", encoding="utf-8") as f:
                for img in train_images:
                    try:
                        rel = img.relative_to(root)
                    except ValueError:
                        rel = img.absolute()
                    f.write(str(rel) + "\n")

            with open(val_txt, "w", encoding="utf-8") as f:
                for img in val_images:
                    try:
                        rel = img.relative_to(root)
                    except ValueError:
                        rel = img.absolute()
                    f.write(str(rel) + "\n")
        except OSError as e:
            if message_callback:
                message_callback(f"索引文件写入失败: {e}")
            return str(train_txt), str(val_txt)

        if message_callback:
            message_callback(f"已生成索引文件: {train_txt.name}, {val_txt.name} (相对路径)")

        return str(train_txt), str(val_txt)

    def _transfer_with_label(
        self,
        img_path: Path,
        img_dir: Path,
        lbl_dir: Path,
        source_img_dir: Path,
        label_source_dir: Optional[Path],
        use_copy: bool
    ) -> str:
        """
        移动或复制图片及其对应的标签文件

        Returns:
            日志消息字符串
        """
        transfer_func = shutil.copy2 if use_copy else shutil.move
        action = "→" if use_copy else "⇢"

        try:
            # 传输图片
            dest_img = img_dir / img_path.name
            transfer_func(str(img_path), str(dest_img))
            log_parts = [f"{img_path.name} {action} {img_dir.parent.name}/{img_dir.name}"]

            # 查找并传输标签
            if label_source_dir and label_source_dir.exists():
                label_path, _ = self._find_label_in_dir(img_path, label_source_dir, img_dir=source_img_dir)
            else:
                label_path, _ = self._find_label(img_path, source_img_dir.parent)

            if label_path and label_path.exists():
                transfer_func(str(label_path), str(lbl_dir / label_path.name))
                log_parts.append(f"+ {label_path.suffix}")

            return " ".join(log_parts)
        except OSError as e:
            return f"失败: {img_path.name} - {e}"

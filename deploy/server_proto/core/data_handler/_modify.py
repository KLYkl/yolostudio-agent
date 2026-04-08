"""
_modify.py - ModifyMixin: 标签修改/清理
============================================
"""

from __future__ import annotations

import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

from core.data_handler._models import ModifyAction


class ModifyMixin:
    """标签修改/清理功能 Mixin"""

    def modify_labels(
        self,
        search_dir: Path,
        action: ModifyAction,
        old_value: str,
        new_value: str = "",
        backup: bool = True,
        classes_txt: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        label_dir: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        批量修改标签文件

        Args:
            search_dir: 标签搜索目录
            action: 修改动作 (REPLACE/REMOVE)
            old_value: 原始类别名/ID
            new_value: 新类别名/ID (仅 REPLACE 时使用)
            backup: 是否备份原文件
            classes_txt: classes.txt 路径 (可选，提供后可使用类别名称)
            interrupt_check: 中断检查函数
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            修改的文件数量
        """
        label_files = self._collect_modify_label_files(search_dir, image_dir=image_dir, label_dir=label_dir)
        modified_count = 0
        total = len(label_files)
        backup_count = 0

        if total == 0:
            if message_callback:
                message_callback("未找到可修改的标签文件")
            return 0

        # 加载类别映射 (TXT 替换需要)
        class_mapping: dict[int, str] = {}
        if classes_txt and classes_txt.exists():
            _, class_mapping = self._read_classes_txt(classes_txt)

        for i, label_path in enumerate(label_files):
            if interrupt_check():
                if message_callback:
                    message_callback("修改已取消")
                break

            if not label_path.exists():
                continue

            try:
                # 根据文件类型处理
                if label_path.suffix.lower() == ".xml":
                    modified, tree = self._prepare_modified_xml(label_path, action, old_value, new_value)
                    if modified and tree is not None:
                        if backup:
                            backup_path = self._get_unique_backup_path(label_path)
                            shutil.copy2(label_path, backup_path)
                            backup_count += 1
                        ET.indent(tree, space="    ")
                        self._write_xml_tree_atomic(label_path, tree)
                else:
                    modified, new_lines = self._prepare_modified_txt(
                        label_path,
                        action,
                        old_value,
                        new_value,
                        class_mapping=class_mapping,
                    )
                    if modified and new_lines is not None:
                        if backup:
                            backup_path = self._get_unique_backup_path(label_path)
                            shutil.copy2(label_path, backup_path)
                            backup_count += 1
                        self._write_lines_atomic(label_path, new_lines)

                if modified:
                    modified_count += 1
            except Exception as e:
                if message_callback:
                    message_callback(f"修改失败: {label_path.name} - {e}")

            if progress_callback:
                progress_callback(i + 1, total)

        if message_callback:
            message_callback(f"已修改 {modified_count} 个标签文件")

        if backup and message_callback and backup_count > 0:
                message_callback(
                    f"提示: 已创建 {backup_count} 个 .bak 备份文件，可手动清理"
                )

        return modified_count

    def preview_modify_labels(
        self,
        search_dir: Path,
        action: ModifyAction,
        old_value: str,
        new_value: str = "",
        classes_txt: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        label_dir: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, int]:
        """预检查标签修改影响范围"""
        label_files = self._collect_modify_label_files(search_dir, image_dir=image_dir, label_dir=label_dir)
        total = len(label_files)
        txt_files = 0
        xml_files = 0
        matched_files = 0
        matched_annotations = 0

        class_mapping: dict[int, str] = {}
        if classes_txt and classes_txt.exists():
            _, class_mapping = self._read_classes_txt(classes_txt)

        for i, label_path in enumerate(label_files):
            if interrupt_check():
                break

            if label_path.suffix.lower() == ".xml":
                xml_files += 1
                affected = self._count_xml_matches(label_path, old_value)
            else:
                txt_files += 1
                affected = self._count_txt_matches(label_path, old_value, class_mapping=class_mapping)

            if affected > 0:
                matched_files += 1
                matched_annotations += affected

            if progress_callback:
                progress_callback(i + 1, total)

        return {
            "total_label_files": total,
            "txt_files": txt_files,
            "xml_files": xml_files,
            "matched_files": matched_files,
            "matched_annotations": matched_annotations,
            "replace_mode": int(action == ModifyAction.REPLACE),
            "has_classes_txt": int(bool(classes_txt and classes_txt.exists())),
        }

    def _collect_modify_label_files(
        self,
        search_dir: Path,
        image_dir: Optional[Path] = None,
        label_dir: Optional[Path] = None,
    ) -> list[Path]:
        """Collect the label files that should be modified for the current scope."""
        if image_dir and image_dir.exists() and not (label_dir and label_dir.exists()):
            return self.collect_image_label_files(image_dir)

        return self.collect_label_files(search_dir)

    def _prepare_modified_xml(
        self,
        label_path: Path,
        action: ModifyAction,
        old_value: str,
        new_value: str,
    ) -> tuple[bool, Optional[ET.ElementTree]]:
        """构建修改后的 XML 标签树"""
        try:
            tree = ET.parse(label_path)
            root = tree.getroot()
            modified = False

            if action == ModifyAction.REPLACE:
                for name_elem in root.findall(".//object/name"):
                    if name_elem.text == old_value:
                        name_elem.text = new_value
                        modified = True
            else:  # REMOVE
                for obj in root.findall(".//object"):
                    name_elem = obj.find("name")
                    if name_elem is not None and name_elem.text == old_value:
                        root.remove(obj)
                        modified = True

            return modified, tree if modified else None

        except Exception:
            return False, None

    def _prepare_modified_txt(
        self,
        label_path: Path,
        action: ModifyAction,
        old_value: str,
        new_value: str,
        class_mapping: Optional[dict[int, str]] = None,
    ) -> tuple[bool, Optional[list[str]]]:
        """构建修改后的 TXT 标签行列表"""
        try:
            # 解析类别名为 ID
            old_id = self._resolve_class_id(old_value, class_mapping=class_mapping)
            new_id = self._resolve_class_id(new_value, class_mapping=class_mapping) if action == ModifyAction.REPLACE else None

            if old_id is None:
                return False, None

            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            modified = False

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = int(parts[0])

                if class_id == old_id:
                    if action == ModifyAction.REPLACE and new_id is not None:
                        parts[0] = str(new_id)
                        new_lines.append(" ".join(parts) + "\n")
                        modified = True
                    elif action == ModifyAction.REMOVE:
                        modified = True
                        continue  # 删除该行
                else:
                    new_lines.append(line)

            return modified, new_lines if modified else None

        except Exception:
            return False, None

    def _count_txt_matches(
        self,
        path: Path,
        old_value: str,
        class_mapping: Optional[dict[int, str]] = None,
    ) -> int:
        """统计 TXT 标签中命中的标注数量"""
        old_id = self._resolve_class_id(old_value, class_mapping=class_mapping)
        if old_id is None:
            return 0

        matches = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and int(parts[0]) == old_id:
                        matches += 1
        except Exception:
            return 0

        return matches

    def _count_xml_matches(self, path: Path, old_value: str) -> int:
        """统计 XML 标签中命中的标注数量"""
        try:
            tree = ET.parse(path)
            return sum(
                1
                for name_elem in tree.getroot().findall(".//object/name")
                if name_elem.text == old_value
            )
        except Exception:
            return 0

    def _get_unique_backup_path(self, label_path: Path) -> Path:
        """Return a non-conflicting backup path for the given label file."""
        backup_path = label_path.with_suffix(label_path.suffix + ".bak")
        if not backup_path.exists():
            return backup_path

        counter = 1
        while True:
            candidate = label_path.with_suffix(label_path.suffix + f".bak.{counter}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _write_lines_atomic(self, label_path: Path, lines: list[str]) -> None:
        """Write label text atomically to avoid partial file corruption."""
        fd, temp_name = tempfile.mkstemp(
            prefix=f"{label_path.name}.",
            suffix=".tmp",
            dir=str(label_path.parent),
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.writelines(lines)
            os.replace(str(temp_path), str(label_path))
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def _write_xml_tree_atomic(self, label_path: Path, tree: ET.ElementTree) -> None:
        """Write XML labels atomically to avoid partial file corruption."""
        fd, temp_name = tempfile.mkstemp(
            prefix=f"{label_path.name}.",
            suffix=".tmp",
            dir=str(label_path.parent),
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            tree.write(temp_path, encoding="utf-8", xml_declaration=True)
            os.replace(str(temp_path), str(label_path))
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def clean_orphan_labels(
        self,
        orphan_labels: list[Path],
        backup: bool = True,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        清理孤立标签文件 (备份后删除)

        Args:
            orphan_labels: 孤立标签文件路径列表
            backup: 是否先备份再删除
            interrupt_check: 中断检查函数
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            成功清理的文件数量
        """
        total = len(orphan_labels)
        cleaned = 0

        if message_callback:
            message_callback(f"开始清理 {total} 个孤立标签文件...")

        for i, label_path in enumerate(orphan_labels):
            if interrupt_check():
                if message_callback:
                    message_callback("清理已取消")
                break

            if not label_path.exists():
                if progress_callback:
                    progress_callback(i + 1, total)
                continue

            try:
                if backup:
                    backup_path = self._get_unique_backup_path(label_path)
                    shutil.copy2(label_path, backup_path)

                label_path.unlink()
                cleaned += 1
            except Exception as e:
                if message_callback:
                    message_callback(f"清理失败: {label_path.name} - {e}")

            if progress_callback:
                progress_callback(i + 1, total)

        if message_callback:
            message_callback(f"已清理 {cleaned} 个孤立标签文件")
            if backup:
                message_callback(f"提示: 已创建 .bak 备份文件，可手动恢复")

        return cleaned

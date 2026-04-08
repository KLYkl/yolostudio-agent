"""
_convert.py - ConvertMixin: 标签格式转换
============================================
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from core.data_handler._models import (
    LabelFormat,
    ModifyAction,
    _get_unique_dir,
)


class ConvertMixin:
    """标签格式转换功能 Mixin"""

    def convert_format(
        self,
        root: Path,
        to_xml: bool = True,
        classes: Optional[list[str]] = None,
        label_dir: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        转换标签格式 (TXT ↔ XML)，输出到独立目录

        Args:
            root: 数据集根目录
            to_xml: True=TXT→XML, False=XML→TXT
            classes: 类别列表
            interrupt_check: 中断检查
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            转换成功的文件数量
        """
        converted = 0
        search_dir = label_dir if label_dir and label_dir.exists() else root

        if to_xml:
            # TXT → XML
            label_files = self.collect_label_files(search_dir, suffixes={".txt"})
            output_dir_name = "converted_labels_xml"
            target_ext = ".xml"
        else:
            # XML → TXT
            label_files = self.collect_label_files(search_dir, suffixes={".xml"})
            output_dir_name = "converted_labels_txt"
            target_ext = ".txt"

        total = len(label_files)
        if total == 0:
            if message_callback:
                message_callback("未找到可转换的标签文件")
            return 0

        # 创建输出目录 (如果已存在则添加数字后缀)
        output_dir = _get_unique_dir(root / output_dir_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        if message_callback:
            direction = "TXT → XML" if to_xml else "XML → TXT"
            message_callback(f"开始转换 ({direction}): {total} 个文件")
            message_callback(f"输出目录: {output_dir}")

        # 构建类别映射
        class_to_id = {}
        id_to_class = {}
        if classes:
            for i, name in enumerate(classes):
                class_to_id[name] = i
                id_to_class[i] = name

        # 如果没有提供类别且要转为 TXT，先扫描所有 XML 获取类别
        if not to_xml and not classes:
            unique_names = set()
            for xml_file in label_files:
                try:
                    tree = ET.parse(xml_file)
                    for obj in tree.getroot().findall(".//object/name"):
                        if obj.text:
                            unique_names.add(obj.text.strip())
                except Exception:
                    pass
            sorted_names = sorted(unique_names)
            class_to_id = {name: i for i, name in enumerate(sorted_names)}
            if message_callback:
                message_callback(f"自动检测到 {len(sorted_names)} 个类别: {', '.join(sorted_names)}")

        failed_files: list[tuple[Path, str]] = []  # 收集失败的文件

        for i, label_path in enumerate(label_files):
            if interrupt_check():
                if message_callback:
                    message_callback("转换已取消")
                break

            try:
                # 计算输出路径
                try:
                    relative_path = label_path.relative_to(search_dir)
                except ValueError:
                    relative_path = Path(label_path.name)

                output_path = output_dir / relative_path.with_suffix(target_ext)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if to_xml:
                    # TXT → XML
                    success = self._convert_txt_to_xml(
                        label_path,
                        id_to_class,
                        root,
                        output_path,
                        image_dir=image_dir,
                        label_dir=label_dir,
                    )
                else:
                    # XML → TXT
                    success = self._convert_xml_to_txt(label_path, class_to_id, output_path)

                if success:
                    converted += 1
                else:
                    failed_files.append((label_path, "转换失败"))
            except Exception as e:
                failed_files.append((label_path, str(e)))

            if progress_callback:
                progress_callback(i + 1, total)

        if message_callback:
            message_callback(f"转换完成: 成功 {converted}/{total}")
            if failed_files:
                message_callback(f"失败 {len(failed_files)} 个文件")
            message_callback(f"文件已保存到: {output_dir}")

        return converted

    def preview_convert_format(
        self,
        root: Path,
        to_xml: bool = True,
        label_dir: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, str | int]:
        """预检查格式互转范围"""
        search_dir = label_dir if label_dir and label_dir.exists() else root
        source_suffix = ".txt" if to_xml else ".xml"
        candidates = [
            path for path in search_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == source_suffix
        ]
        total_candidates = len(candidates)
        label_files: list[Path] = []

        for i, path in enumerate(candidates):
            if interrupt_check():
                break

            is_valid = (
                self._is_txt_label_file(path)
                if source_suffix == ".txt"
                else self._is_xml_label_file(path)
            )
            if is_valid:
                label_files.append(path)

            if progress_callback:
                progress_callback(i + 1, total_candidates)

        return {
            "total_labels": len(label_files),
            "txt_files": sum(1 for path in label_files if path.suffix.lower() == ".txt"),
            "xml_files": sum(1 for path in label_files if path.suffix.lower() == ".xml"),
            "source_type": "TXT" if to_xml else "XML",
            "target_type": "XML" if to_xml else "TXT",
            "output_dir_name": "converted_labels_xml" if to_xml else "converted_labels_txt",
        }

    def _convert_txt_to_xml(
        self,
        txt_path: Path,
        id_to_class: dict,
        root: Path,
        output_path: Path,
        image_dir: Optional[Path] = None,
        label_dir: Optional[Path] = None,
    ) -> bool:
        """将 TXT 标签转换为 XML"""
        # 查找对应的图片
        img_path = self._find_image_for_label(
            txt_path,
            root,
            image_dir=image_dir,
            label_dir=label_dir,
        )
        if not img_path or not img_path.exists():
            return False

        # 读取图片尺寸
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                depth = len(img.getbands())
        except Exception:
            return False

        # 解析 TXT
        objects = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])

                    # 归一化 → 绝对坐标
                    xmin = int((x_c - w / 2) * width)
                    ymin = int((y_c - h / 2) * height)
                    xmax = int((x_c + w / 2) * width)
                    ymax = int((y_c + h / 2) * height)

                    # 获取类别名称
                    name = id_to_class.get(class_id, str(class_id))

                    objects.append({
                        "name": name,
                        "xmin": max(0, xmin),
                        "ymin": max(0, ymin),
                        "xmax": min(width, xmax),
                        "ymax": min(height, ymax),
                    })

        # 生成 XML
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = img_path.parent.name
        ET.SubElement(annotation, "filename").text = img_path.name

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)

        for obj_data in objects:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = obj_data["name"]
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(obj_data["xmin"])
            ET.SubElement(bndbox, "ymin").text = str(obj_data["ymin"])
            ET.SubElement(bndbox, "xmax").text = str(obj_data["xmax"])
            ET.SubElement(bndbox, "ymax").text = str(obj_data["ymax"])

        # 格式化 XML (添加换行和缩进)
        from xml.dom import minidom
        xml_str = ET.tostring(annotation, encoding="unicode")
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")
        # 移除多余空行
        pretty_xml = "\n".join(line for line in pretty_xml.split("\n") if line.strip())

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        return True

    def _convert_xml_to_txt(self, xml_path: Path, class_to_id: dict, output_path: Path) -> bool:
        """将 XML 标签转换为 TXT"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            return False

        # 读取尺寸
        size = root.find("size")
        if size is None:
            return False

        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))

        if width == 0 or height == 0:
            return False

        # 解析对象
        lines = []
        for obj in root.findall(".//object"):
            name_elem = obj.find("name")
            bndbox = obj.find("bndbox")

            if name_elem is None or bndbox is None:
                continue

            name = name_elem.text.strip() if name_elem.text else ""

            xmin = int(float(bndbox.findtext("xmin", "0")))
            ymin = int(float(bndbox.findtext("ymin", "0")))
            xmax = int(float(bndbox.findtext("xmax", "0")))
            ymax = int(float(bndbox.findtext("ymax", "0")))

            # 绝对坐标 → 归一化
            x_c = (xmin + xmax) / 2 / width
            y_c = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            # 获取类别 ID
            class_id = class_to_id.get(name, len(class_to_id))
            if name not in class_to_id:
                class_to_id[name] = class_id

            lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # 写入 TXT
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return True

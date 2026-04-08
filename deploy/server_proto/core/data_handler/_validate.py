"""
_validate.py - ValidateMixin: 标签验证/解析
============================================
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from core.data_handler._models import (
    IMAGE_EXTENSIONS,
    LABEL_EXTENSIONS,
    LabelFormat,
    ValidateResult,
)


class ValidateMixin:
    """标签验证/解析功能 Mixin"""

    def generate_missing_labels(
        self,
        img_dir: Path,
        label_format: LabelFormat,
        label_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        扫描缺失标签图片并生成空标签文件

        Args:
            img_dir: 图片目录
            label_format: 标签格式 (TXT/XML)
            label_dir: 标签目录 (可选)
            output_dir: 输出目录 (可选)
            interrupt_check: 中断检查函数
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            生成的标签文件数量
        """
        scan_result = self.scan_dataset(
            img_dir,
            label_dir=label_dir,
            interrupt_check=interrupt_check,
            progress_callback=progress_callback,
            message_callback=message_callback,
        )

        if interrupt_check():
            return 0

        if not scan_result.missing_labels:
            if message_callback:
                message_callback("没有需要生成标签的图片")
            return 0

        if message_callback:
            message_callback(
                f"找到 {len(scan_result.missing_labels)} 张缺失标签图片，开始生成空标签"
            )

        target_output_dir = output_dir
        if target_output_dir is None and label_dir and label_dir.exists():
            target_output_dir = label_dir

        return self.generate_empty_labels(
            scan_result.missing_labels,
            label_format,
            output_dir=target_output_dir,
            interrupt_check=interrupt_check,
            progress_callback=progress_callback,
            message_callback=message_callback,
        )

    def preview_generate_missing_labels(
        self,
        img_dir: Path,
        label_dir: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, int]:
        """预检查缺失标签数量"""
        images = self._find_images(img_dir)
        total_images = len(images)
        missing_labels = 0

        for i, img_path in enumerate(images):
            if interrupt_check():
                break

            if label_dir and label_dir.exists():
                label_path, _ = self._find_label_in_dir(img_path, label_dir, img_dir=img_dir)
            else:
                label_path, _ = self._find_label(img_path, img_dir.parent)

            if label_path is None:
                missing_labels += 1

            if progress_callback:
                progress_callback(i + 1, total_images)

        return {
            "total_images": total_images,
            "missing_labels": missing_labels,
        }

    def _find_label_in_dir(
        self,
        img_path: Path,
        label_dir: Path,
        img_dir: Optional[Path] = None,
    ) -> tuple[Optional[Path], Optional[LabelFormat]]:
        """
        在指定目录中查找图片对应的标签文件

        Args:
            img_path: 图片路径
            label_dir: 标签目录
            img_dir: 图片根目录 (可选，用于子目录结构映射)
                      例: img_dir=images/, img_path=images/train/1.jpg
                      → 在 label_dir/train/ 下查找标签

        Returns:
            (标签路径, 格式) 或 (None, None)
        """
        stem = img_path.stem

        # 1. 子目录映射: 通过 img_dir 计算相对子路径
        if img_dir and img_dir.exists():
            try:
                rel_sub = img_path.parent.relative_to(img_dir)
                sub_label_dir = label_dir / rel_sub
                for ext, fmt in [(".xml", LabelFormat.XML), (".txt", LabelFormat.TXT)]:
                    label_path = sub_label_dir / (stem + ext)
                    if label_path.exists():
                        return label_path, fmt
            except ValueError:
                pass

        # 2. 回退: 直接在 label_dir 根目录查找 (原有行为)
        for ext, fmt in [(".xml", LabelFormat.XML), (".txt", LabelFormat.TXT)]:
            label_path = label_dir / (stem + ext)
            if label_path.exists():
                return label_path, fmt

        return None, None

    def generate_empty_labels(
        self,
        images: list[Path],
        label_format: LabelFormat,
        output_dir: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        为图片生成空标签文件

        标签将保存到与 images 目录同级的 labels 目录中。
        例如: .../my_data/images/1.jpg -> .../my_data/labels/1.txt

        Args:
            images: 图片路径列表
            label_format: 标签格式 (TXT/XML)
            output_dir: 输出目录 (None 则自动检测 labels 同级目录)
            interrupt_check: 中断检查函数
            progress_callback: 进度回调
            message_callback: 消息回调

        Returns:
            生成的标签文件数量
        """
        count = 0
        total = len(images)
        ext = ".txt" if label_format == LabelFormat.TXT else ".xml"

        for i, img_path in enumerate(images):
            if interrupt_check():
                if message_callback:
                    message_callback("生成已取消")
                break

            # 确定输出路径
            if output_dir:
                label_path = output_dir / (img_path.stem + ext)
            else:
                # 查找或创建同级 labels 目录
                label_path = self._get_label_output_path(img_path, ext)

            # Another process may create the file after the scan; skip overwrite.
            if label_path.exists():
                if progress_callback:
                    progress_callback(i + 1, total)
                continue

            # 确保目录存在
            label_path.parent.mkdir(parents=True, exist_ok=True)

            if label_format == LabelFormat.TXT:
                # TXT: 空文件
                label_path.touch()
            else:
                # XML: 需要图片尺寸
                self._create_empty_xml(img_path, label_path)

            count += 1

            if progress_callback:
                progress_callback(i + 1, total)

        if message_callback:
            message_callback(f"已生成 {count} 个空标签文件")

        return count

    def _get_label_output_path(self, img_path: Path, ext: str) -> Path:
        """
        计算标签文件的输出路径 (智能路径选择)

        规则:
            - XML: 使用 Annotations/ (Pascal VOC 风格)
            - TXT: 使用 labels/ (YOLO 风格)

        Args:
            img_path: 图片路径
            ext: 标签文件扩展名 (.txt 或 .xml)

        Returns:
            标签文件路径
        """
        # 根据格式确定目标目录名
        if ext.lower() == ".xml":
            target_dir_name = "Annotations"
            source_dir_names = ["jpegimages", "images", "imgs"]
        else:
            target_dir_name = "labels"
            source_dir_names = ["images", "jpegimages", "imgs"]

        parts = list(img_path.parts)

        # 查找并替换源目录名
        found_idx = None
        for i, part in enumerate(parts):
            if part.lower() in source_dir_names:
                found_idx = i
                break

        if found_idx is not None:
            # 找到源目录，替换为目标目录
            parts[found_idx] = target_dir_name
            label_path = Path(*parts).with_suffix(ext)
        else:
            # 没有找到，在图片目录的同级创建目标目录
            parent = img_path.parent.parent
            target_dir = parent / target_dir_name
            label_path = target_dir / (img_path.stem + ext)

        return label_path

    def collect_label_class_options(
        self,
        search_dir: Path,
        classes_txt: Optional[Path] = None,
    ) -> list[str]:
        """收集标签中的类别选项，用于下拉框"""
        if classes_txt and classes_txt.exists():
            classes, _ = self._read_classes_txt(classes_txt)
            return classes

        class_values: set[str] = set()
        for label_path in self.collect_label_files(search_dir):
            label_format = LabelFormat.XML if label_path.suffix.lower() == ".xml" else LabelFormat.TXT
            class_values.update(self._parse_label(label_path, label_format, class_mapping={}))

        numeric_values = sorted(
            (value for value in class_values if value.isdigit()),
            key=lambda value: int(value),
        )
        text_values = sorted(value for value in class_values if not value.isdigit())
        return numeric_values + text_values

    def collect_label_files(
        self,
        root: Path,
        suffixes: Optional[set[str]] = None,
    ) -> list[Path]:
        """递归收集有效标签文件"""
        if not root.exists():
            return []

        allowed_suffixes = {suffix.lower() for suffix in (suffixes or LABEL_EXTENSIONS)}
        label_files: list[Path] = []

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            if suffix not in allowed_suffixes:
                continue

            if suffix == ".txt" and self._is_txt_label_file(path):
                label_files.append(path)
            elif suffix == ".xml" and self._is_xml_label_file(path):
                label_files.append(path)

        return sorted(label_files)

    def collect_image_label_files(
        self,
        img_dir: Path,
        label_dir: Optional[Path] = None,
    ) -> list[Path]:
        """Collect label files matched to images to avoid touching derived directories."""
        if not img_dir.exists():
            return []

        label_files: list[Path] = []
        seen: set[Path] = set()
        for img_path in self._find_images(img_dir):
            if label_dir and label_dir.exists():
                label_path, _ = self._find_label_in_dir(img_path, label_dir, img_dir=img_dir)
            else:
                label_path, _ = self._find_label(img_path, img_dir.parent)

            if label_path and label_path.exists() and label_path not in seen:
                seen.add(label_path)
                label_files.append(label_path)

        return sorted(label_files)

    def _find_images(self, root: Path) -> list[Path]:
        """递归查找所有图片文件"""
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(root.rglob(f"*{ext}"))
            images.extend(root.rglob(f"*{ext.upper()}"))
        return sorted(set(images))

    def _find_label(self, img_path: Path, root: Path) -> tuple[Optional[Path], Optional[LabelFormat]]:
        """
        查找图片对应的标签文件

        查找顺序:
            1. 同目录下的 .xml / .txt
            2. YOLO 目录结构: images/ -> labels/
            3. Pascal VOC 目录结构: JPEGImages/ -> Annotations/
        """
        stem = img_path.stem

        # 1. 同目录 (XML 优先)
        for ext, fmt in [(".xml", LabelFormat.XML), (".txt", LabelFormat.TXT)]:
            label_path = img_path.with_suffix(ext)
            if label_path.exists():
                return label_path, fmt

        # 2. 目录映射: 支持 YOLO 和 Pascal VOC
        # 映射规则: (图片目录名 -> 标签目录名)
        dir_mappings = [
            ("images", "labels"),           # YOLO 标准
            ("jpegimages", "annotations"),  # Pascal VOC 标准
            ("imgs", "labels"),             # 常见变体
            ("img", "label"),               # 常见变体
        ]

        try:
            rel_path = img_path.relative_to(root)
            parts = list(rel_path.parts)

            # 尝试每种映射
            for img_dir, lbl_dir in dir_mappings:
                for i, part in enumerate(parts):
                    if part.lower() == img_dir:
                        # 创建替换后的路径
                        new_parts = list(parts)
                        new_parts[i] = lbl_dir
                        label_dir = root / Path(*new_parts[:-1])

                        for ext, fmt in [(".xml", LabelFormat.XML), (".txt", LabelFormat.TXT)]:
                            label_path = label_dir / (stem + ext)
                            if label_path.exists():
                                return label_path, fmt
                        break
        except ValueError:
            pass

        return None, None

    def _parse_label(
        self,
        label_path: Path,
        label_format: LabelFormat,
        class_mapping: Optional[dict[int, str]] = None,
    ) -> list[str]:
        """
        解析标签文件中的类别标识

        - TXT 格式: 如有 class_mapping，用映射后的名称；否则用原始 ID
        - XML 格式: 直接使用 <name> 文本 (已 strip)
        """
        classes: list[str] = []
        mapping = class_mapping or {}

        try:
            if label_format == LabelFormat.TXT:
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_name = mapping.get(class_id, str(class_id))
                            classes.append(class_name)
            else:
                tree = ET.parse(label_path)
                root = tree.getroot()
                for obj in root.findall(".//object/name"):
                    if obj.text:
                        classes.append(obj.text.strip())
        except Exception:
            pass

        return classes

    def _create_empty_xml(self, img_path: Path, label_path: Path) -> None:
        """创建空的 VOC XML 标签文件"""
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                depth = len(img.getbands())
        except Exception:
            width, height, depth = 0, 0, 3

        root = ET.Element("annotation")

        # 文件夹名
        ET.SubElement(root, "folder").text = img_path.parent.name

        # 文件名
        ET.SubElement(root, "filename").text = img_path.name

        # 尺寸
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)

        # 无 object 节点

        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(label_path, encoding="utf-8", xml_declaration=True)

    def _find_image_for_label(
        self,
        label_path: Path,
        root: Path,
        image_dir: Optional[Path] = None,
        label_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """根据标签文件查找对应的图片"""
        stem = label_path.stem

        # 1. 同目录
        for ext in IMAGE_EXTENSIONS:
            img_path = label_path.with_suffix(ext)
            if img_path.exists():
                return img_path

        # 2. 目录映射 (使用指定的 image_dir 和 label_dir)
        if image_dir and label_dir:
            try:
                rel_path = label_path.relative_to(label_dir)
                img_parent = image_dir / rel_path.parent
                for ext in IMAGE_EXTENSIONS:
                    img_path = img_parent / (stem + ext)
                    if img_path.exists():
                        return img_path
            except ValueError:
                pass

        # 3. Heuristic directory mapping for common dataset layouts.
        dir_mappings = [
            ("labels", "images"),
            ("annotations", "jpegimages"),
            ("label", "img"),
        ]

        try:
            rel_path = label_path.relative_to(root)
            parts = list(rel_path.parts)

            for lbl_dir, img_dir_name in dir_mappings:
                for i, part in enumerate(parts):
                    if part.lower() == lbl_dir:
                        new_parts = list(parts)
                        new_parts[i] = img_dir_name
                        img_base = root / Path(*new_parts[:-1])

                        for ext in IMAGE_EXTENSIONS:
                            img_path = img_base / (stem + ext)
                            if img_path.exists():
                                return img_path
                        break
        except ValueError:
            pass

        return None

    def _is_txt_label_file(self, path: Path) -> bool:
        """判断 TXT 文件是否为有效检测标签"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue

                    parts = stripped.split()
                    if len(parts) < 5:
                        return False

                    int(parts[0])
                    for value in parts[1:5]:
                        float(value)

            return True

        except Exception:
            return False

    def _is_xml_label_file(self, path: Path) -> bool:
        """判断 XML 文件是否为 VOC 标签"""
        try:
            root = ET.parse(path).getroot()
        except Exception:
            return False

        return root.tag.lower() == "annotation"




    def validate_labels(
        self,
        img_dir: Path,
        label_dir: Optional[Path] = None,
        classes_txt: Optional[Path] = None,
        check_coords: bool = True,
        check_class_ids: bool = True,
        check_format: bool = True,
        check_orphans: bool = True,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> ValidateResult:
        """
        校验标签文件的合法性

        Args:
            img_dir: 图片目录
            label_dir: 标签目录 (可选，留空则自动检测)
            classes_txt: classes.txt 路径 (可选，用于类别 ID 校验)
            check_coords: 是否检查坐标越界
            check_class_ids: 是否检查类别 ID 有效性
            check_format: 是否检查格式错误
            check_orphans: 是否检查孤立标签
            interrupt_check: 中断检查函数
            progress_callback: 进度回调 (current, total)
            message_callback: 消息回调

        Returns:
            ValidateResult: 校验结果
        """
        result = ValidateResult()

        # 加载类别映射
        class_names: list[str] = []
        class_mapping: dict[int, str] = {}
        if classes_txt and classes_txt.exists():
            class_names, class_mapping = self._read_classes_txt(classes_txt)
            if message_callback:
                message_callback(f"已加载类别文件: {len(class_names)} 个类别")

        # 确定标签搜索目录
        search_dir = label_dir if (label_dir and label_dir.exists()) else img_dir
        label_files = self.collect_label_files(search_dir)
        result.total_labels = len(label_files)

        if message_callback:
            message_callback(f"发现 {result.total_labels} 个标签文件，开始校验...")

        if result.total_labels == 0:
            return result

        # 预建图片文件名 set 索引 (O(1) 查找孤立标签)
        image_stems: set[str] = set()
        if check_orphans:
            images = self._find_images(img_dir)
            image_stems = {img.stem.lower() for img in images}
            if message_callback:
                message_callback(f"索引了 {len(image_stems)} 张图片用于孤立检查")

        for i, label_path in enumerate(label_files):
            if interrupt_check():
                if message_callback:
                    message_callback("校验已取消")
                break

            suffix = label_path.suffix.lower()
            is_txt = suffix == ".txt"

            # ---- 孤立标签检查 ----
            if check_orphans:
                if label_path.stem.lower() not in image_stems:
                    result.orphan_labels.append(label_path)

            # ---- TXT (YOLO) 格式校验 ----
            if is_txt:
                self._validate_txt_label(
                    label_path, result,
                    check_coords=check_coords,
                    check_class_ids=check_class_ids,
                    check_format=check_format,
                    num_classes=len(class_names) if class_names else None,
                )
            # ---- XML (VOC) 格式校验 ----
            else:
                img_path = None
                if check_coords:
                    img_path = self._find_image_for_label(
                        label_path, img_dir.parent,
                        image_dir=img_dir, label_dir=label_dir,
                    )
                self._validate_xml_label(
                    label_path, result,
                    check_coords=check_coords,
                    check_class_ids=check_class_ids,
                    check_format=check_format,
                    class_names=class_names,
                    img_path=img_path,
                )

            if progress_callback:
                progress_callback(i + 1, result.total_labels)

        if message_callback:
            if result.has_issues:
                parts = []
                if result.coord_errors:
                    parts.append(f"坐标越界 {len(result.coord_errors)}")
                if result.class_errors:
                    parts.append(f"类别无效 {len(result.class_errors)}")
                if result.format_errors:
                    parts.append(f"格式错误 {len(result.format_errors)}")
                if result.orphan_labels:
                    parts.append(f"孤立标签 {len(result.orphan_labels)}")
                message_callback(f"校验完成，发现 {result.issue_count} 个问题: {', '.join(parts)}")
            else:
                message_callback("校验完成，未发现问题 ✓")

        return result

    def _validate_txt_label(
        self,
        label_path: Path,
        result: ValidateResult,
        *,
        check_coords: bool,
        check_class_ids: bool,
        check_format: bool,
        num_classes: Optional[int],
    ) -> None:
        """校验单个 YOLO TXT 标签文件"""
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            if check_format:
                result.format_errors.append((label_path, f"无法读取: {e}"))
            return

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            loc = f"行 {line_no}"

            # 格式检查: 应该恰好 5 个字段
            if check_format and len(parts) != 5:
                result.format_errors.append(
                    (label_path, f"{loc}: 期望 5 个字段，实际 {len(parts)} 个")
                )
                continue  # 字段数不对，跳过坐标/类别检查

            # 类别 ID 检查
            if check_class_ids and num_classes is not None:
                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= num_classes:
                        result.class_errors.append(
                            (label_path, loc,
                             f"类别 ID {class_id} 超出范围 [0, {num_classes - 1})")
                        )
                except ValueError:
                    result.class_errors.append(
                        (label_path, loc, f"类别 ID '{parts[0]}' 不是整数")
                    )

            # 坐标越界检查: x, y, w, h ∈ [0, 1]
            if check_coords and len(parts) >= 5:
                try:
                    values = [float(v) for v in parts[1:5]]
                    for j, (val, name) in enumerate(zip(
                        values, ["x", "y", "w", "h"]
                    )):
                        if val < 0.0 or val > 1.0:
                            result.coord_errors.append(
                                (label_path, loc,
                                 f"{name}={val:.4f} 超出 [0, 1] 范围")
                            )
                            break  # 每行只报一次坐标越界
                except ValueError:
                    if check_format:
                        result.format_errors.append(
                            (label_path, f"{loc}: 坐标值不是有效数字")
                        )

    def _validate_xml_label(
        self,
        label_path: Path,
        result: ValidateResult,
        *,
        check_coords: bool,
        check_class_ids: bool,
        check_format: bool,
        class_names: list[str],
        img_path: Optional[Path],
    ) -> None:
        """校验单个 VOC XML 标签文件"""
        try:
            tree = ET.parse(label_path)
            root = tree.getroot()
        except ET.ParseError as e:
            if check_format:
                result.format_errors.append((label_path, f"XML 解析失败: {e}"))
            return
        except Exception as e:
            if check_format:
                result.format_errors.append((label_path, f"无法读取: {e}"))
            return

        # 格式检查: 基础结构
        if check_format:
            size_node = root.find("size")
            if size_node is None:
                result.format_errors.append((label_path, "缺少 <size> 节点"))
            else:
                for tag in ("width", "height"):
                    if size_node.find(tag) is None:
                        result.format_errors.append(
                            (label_path, f"<size> 缺少 <{tag}> 节点")
                        )

        # 获取图片尺寸 (用于坐标校验)
        img_w, img_h = 0, 0
        if check_coords and img_path and img_path.exists():
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception:
                pass

        objects = root.findall(".//object")
        for obj in objects:
            name_node = obj.find("name")
            obj_name = name_node.text.strip() if (name_node is not None and name_node.text) else "<未知>"

            # 类别检查
            if check_class_ids and class_names:
                if name_node is None or not name_node.text or name_node.text.strip() not in class_names:
                    result.class_errors.append(
                        (label_path, obj_name,
                         f"类别 '{obj_name}' 不在 classes.txt 中")
                    )

            # 格式检查: object 必须有 bndbox
            bndbox = obj.find("bndbox")
            if check_format and bndbox is None:
                result.format_errors.append(
                    (label_path, f"对象 '{obj_name}' 缺少 <bndbox> 节点")
                )
                continue

            # 坐标越界检查
            if check_coords and bndbox is not None and img_w > 0 and img_h > 0:
                try:
                    xmin = float(bndbox.findtext("xmin", "0"))
                    ymin = float(bndbox.findtext("ymin", "0"))
                    xmax = float(bndbox.findtext("xmax", "0"))
                    ymax = float(bndbox.findtext("ymax", "0"))

                    if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
                        result.coord_errors.append(
                            (label_path, obj_name,
                             f"bbox({xmin},{ymin},{xmax},{ymax}) "
                             f"超出图片尺寸 {img_w}×{img_h}")
                        )
                    if xmin >= xmax or ymin >= ymax:
                        result.coord_errors.append(
                            (label_path, obj_name,
                             f"bbox 无效: xmin({xmin})>=xmax({xmax}) "
                             f"或 ymin({ymin})>=ymax({ymax})")
                        )
                except (ValueError, TypeError):
                    if check_format:
                        result.format_errors.append(
                            (label_path, f"对象 '{obj_name}' 的 bndbox 坐标不是有效数字")
                        )

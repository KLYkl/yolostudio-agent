"""
_augment.py - AugmentMixin: 数据增强 + 几何变换
============================================
"""

from __future__ import annotations

import math
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

from core.data_handler._models import (
    AppliedGeometryOp,
    AugmentConfig,
    AugmentRecipe,
    AugmentResult,
    LabelFormat,
    _get_unique_dir,
)


class AugmentMixin:
    """数据增强 + 几何变换功能 Mixin"""

    # ==================== 公开 API ====================

    def augment_dataset(
        self,
        img_dir: Path,
        config: AugmentConfig,
        label_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        classes_txt: Optional[Path] = None,
        interrupt_check: Callable[[], bool] = lambda: False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> AugmentResult:
        """Generate an offline augmented dataset with synchronized labels."""
        if config.copies_per_image < 1:
            raise ValueError("每张图片的增强副本数至少为 1")
        if not config.has_any_operation():
            raise ValueError("请至少启用一种增强操作")
        if config.mode == "fixed" and not config.build_fixed_recipes():
            raise ValueError("固定模式下没有可用的增强组合")

        if output_dir is None:
            output_dir = _get_unique_dir(img_dir.parent / f"{img_dir.name}_augmented")

        output_dir.mkdir(parents=True, exist_ok=True)
        images = self._find_images(img_dir)
        result = AugmentResult(output_dir=str(output_dir), source_images=len(images))

        if message_callback:
            mode_label = "固定模式" if config.mode == "fixed" else "随机模式"
            message_callback(f"输出目录: {output_dir}")
            message_callback(f"增强模式: {mode_label}")

        if classes_txt and classes_txt.exists():
            shutil.copy2(str(classes_txt), str(output_dir / classes_txt.name))
            if message_callback:
                message_callback(f"已复制类别文件: {classes_txt.name}")

        total = len(images)
        if total == 0:
            if message_callback:
                message_callback("未找到图片文件")
            return result

        rng = random.Random(config.seed)

        for index, img_path in enumerate(images):
            if interrupt_check():
                if message_callback:
                    message_callback("增强已取消")
                break

            try:
                rel_path = img_path.relative_to(img_dir)
                if label_dir and label_dir.exists():
                    label_path, label_format = self._find_label_in_dir(img_path, label_dir, img_dir=img_dir)
                else:
                    label_path, label_format = self._find_label(img_path, img_dir.parent)

                with Image.open(img_path) as source_image:
                    base_image = ImageOps.exif_transpose(source_image)
                    source_image_size = base_image.size

                    if config.include_original:
                        original_output = self._build_augmented_image_output_path(output_dir, rel_path)
                        original_output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(img_path), str(original_output))
                        result.copied_originals += 1

                        if label_path and label_format:
                            original_label_output = self._build_augmented_label_output_path(
                                output_dir,
                                rel_path,
                                img_path.stem,
                                label_format,
                            )
                            self._write_augmented_label(
                                label_path,
                                label_format,
                                original_label_output,
                                [],
                                original_output,
                                source_image_size,
                            )
                            result.label_files_written += 1

                    recipe_counters: dict[str, int] = {}
                    for recipe, display_index in self._iter_augment_recipes(config, rng):
                        augmented_image, geometry_ops = self._apply_augmentation_recipe(
                            base_image,
                            recipe.operations,
                            config,
                            rng,
                        )
                        recipe_counters[recipe.name] = recipe_counters.get(recipe.name, 0) + 1
                        recipe_suffix = recipe.name if config.mode == "fixed" else "aug"
                        aug_stem = f"{img_path.stem}_{recipe_suffix}_{recipe_counters[recipe.name]:03d}"
                        aug_rel_path = rel_path.with_name(f"{aug_stem}{img_path.suffix.lower()}")
                        aug_output = self._build_augmented_image_output_path(output_dir, aug_rel_path)

                        self._save_augmented_image(augmented_image, aug_output)
                        result.augmented_images += 1

                        if label_path and label_format:
                            aug_label_output = self._build_augmented_label_output_path(
                                output_dir,
                                rel_path,
                                aug_stem,
                                label_format,
                            )
                            self._write_augmented_label(
                                label_path,
                                label_format,
                                aug_label_output,
                                geometry_ops,
                                aug_output,
                                source_image_size,
                            )
                            result.label_files_written += 1
            except Exception as exc:
                result.skipped_images += 1
                if message_callback:
                    message_callback(f"跳过 {img_path.name}: {exc}")

            if progress_callback:
                progress_callback(index + 1, total)

        if message_callback:
            total_outputs = result.copied_originals + result.augmented_images
            message_callback(
                "增强完成: "
                f"共输出 {total_outputs} 张"
                f"(原图 {result.copied_originals} / 增强 {result.augmented_images})，"
                f"标签 {result.label_files_written} 个"
            )
            if result.skipped_images > 0:
                message_callback(f"跳过图片: {result.skipped_images} 张")

        return result

    # ==================== 路径构建 ====================

    def _build_augmented_image_output_path(self, output_dir: Path, relative_path: Path) -> Path:
        """构建增强图片输出路径"""
        return output_dir / "images" / relative_path

    def _build_augmented_label_output_path(
        self,
        output_dir: Path,
        relative_path: Path,
        stem: str,
        label_format: LabelFormat,
    ) -> Path:
        """构建增强标签输出路径"""
        label_root = "labels" if label_format == LabelFormat.TXT else "Annotations"
        suffix = ".txt" if label_format == LabelFormat.TXT else ".xml"
        return output_dir / label_root / relative_path.parent / f"{stem}{suffix}"

    # ==================== 增强配方 ====================

    def _iter_augment_recipes(
        self,
        config: AugmentConfig,
        rng: random.Random,
    ) -> list[tuple[AugmentRecipe, int]]:
        """Resolve the list of recipes to generate for one source image."""
        if config.mode == "fixed":
            recipes = config.build_fixed_recipes()
            return [
                (recipe, copy_index + 1)
                for copy_index in range(config.copies_per_image)
                for recipe in recipes
            ]

        resolved: list[tuple[AugmentRecipe, int]] = []
        for copy_index in range(config.copies_per_image):
            operations = self._sample_random_operations(config, rng)
            resolved.append((AugmentRecipe("aug", operations), copy_index + 1))
        return resolved

    def _sample_random_operations(
        self,
        config: AugmentConfig,
        rng: random.Random,
    ) -> tuple[str, ...]:
        """Sample one random augmentation recipe."""
        geometry_ops: list[str] = []
        photo_ops: list[str] = []
        geometric_candidates = config.geometric_candidates()
        photometric_candidates = config.photometric_candidates()

        if geometric_candidates and rng.random() < 0.9:
            geometry_ops.append(rng.choice(geometric_candidates))

        for operation in photometric_candidates:
            if rng.random() < 0.5:
                photo_ops.append(operation)

        if not geometry_ops and not photo_ops:
            fallback_ops = geometric_candidates + photometric_candidates
            if fallback_ops:
                chosen = rng.choice(fallback_ops)
                if chosen in geometric_candidates:
                    geometry_ops.append(chosen)
                else:
                    photo_ops.append(chosen)

        return tuple(geometry_ops + photo_ops)

    # ==================== 增强应用 ====================

    def _apply_augmentation_recipe(
        self,
        image: Image.Image,
        operations: tuple[str, ...],
        config: AugmentConfig,
        rng: random.Random,
    ) -> tuple[Image.Image, list[AppliedGeometryOp]]:
        """Apply a resolved recipe to an image."""
        augmented = image.copy()
        geometry_ops: list[AppliedGeometryOp] = []

        for operation in operations:
            if operation in {"flip_lr", "flip_ud", "rotate"}:
                augmented, applied_geometry = self._apply_geometric_augmentation(
                    augmented,
                    operation,
                    config,
                    rng,
                )
                geometry_ops.append(applied_geometry)
            else:
                augmented = self._apply_photometric_augmentation(
                    augmented,
                    operation,
                    config,
                    rng,
                )

        return augmented, geometry_ops

    def _apply_geometric_augmentation(
        self,
        image: Image.Image,
        operation: str,
        config: AugmentConfig,
        rng: random.Random,
    ) -> tuple[Image.Image, AppliedGeometryOp]:
        """Apply one geometry transform and return the concrete op."""
        if operation == "flip_lr":
            return ImageOps.mirror(image), AppliedGeometryOp("flip_lr")
        if operation == "flip_ud":
            return ImageOps.flip(image), AppliedGeometryOp("flip_ud")
        if operation == "rotate":
            angle = self._resolve_rotation_angle(config, rng)
            resample = (
                Image.Resampling.NEAREST
                if image.mode in {"1", "P"}
                else Image.Resampling.BICUBIC
            )
            return (
                image.rotate(
                    angle,
                    resample=resample,
                    expand=True,
                    fillcolor=self._get_rotation_fillcolor(image),
                ),
                AppliedGeometryOp("rotate", angle),
            )
        raise ValueError(f"不支持的几何操作: {operation}")

    def _resolve_rotation_angle(
        self,
        config: AugmentConfig,
        rng: random.Random,
    ) -> float:
        """Resolve the actual rotation angle in PIL's sign convention."""
        degrees = max(0.0, float(config.rotate_degrees))
        if config.rotate_mode == "clockwise":
            return -degrees
        if config.rotate_mode == "counterclockwise":
            return degrees
        return rng.uniform(-degrees, degrees)

    def _get_rotation_fillcolor(self, image: Image.Image):
        """Match YOLO-style neutral padding for rotated images."""
        bands = len(image.getbands())
        if image.mode == "RGBA":
            return (114, 114, 114, 0)
        if image.mode in {"RGB", "YCbCr"}:
            return (114, 114, 114)
        if bands == 1:
            return 114
        return tuple(114 for _ in range(bands))

    def _apply_photometric_augmentation(
        self,
        image: Image.Image,
        operation: str,
        config: AugmentConfig,
        rng: random.Random,
    ) -> Image.Image:
        """Apply one color or sharpness transform."""
        if operation == "brightness":
            return ImageEnhance.Brightness(image).enhance(
                self._sample_enhance_factor(config.brightness_strength, rng)
            )
        if operation == "contrast":
            return ImageEnhance.Contrast(image).enhance(
                self._sample_enhance_factor(config.contrast_strength, rng)
            )
        if operation == "color":
            return ImageEnhance.Color(image).enhance(
                self._sample_enhance_factor(config.color_strength, rng)
            )
        if operation == "noise":
            return self._apply_noise(image, config.noise_strength, rng)
        if operation == "hue":
            return self._apply_hue_shift(
                image,
                rng.uniform(-config.hue_degrees, config.hue_degrees),
            )
        if operation == "sharpness":
            return ImageEnhance.Sharpness(image).enhance(
                self._sample_enhance_factor(config.sharpness_strength, rng)
            )
        if operation == "blur":
            return image.filter(
                ImageFilter.GaussianBlur(radius=rng.uniform(0.2, config.blur_radius))
            )
        raise ValueError(f"不支持的光度操作: {operation}")

    def _sample_enhance_factor(
        self,
        strength: float,
        rng: random.Random,
    ) -> float:
        """Sample a PIL enhancement factor centered at 1.0."""
        strength = max(0.0, float(strength))
        return rng.uniform(max(0.0, 1.0 - strength), 1.0 + strength)

    def _apply_noise(
        self,
        image: Image.Image,
        strength: float,
        rng: random.Random,
    ) -> Image.Image:
        """Apply additive Gaussian-style noise without changing labels."""
        strength = max(0.0, float(strength))
        if strength <= 0:
            return image

        sigma = max(1.0, strength * 128.0)
        if image.mode == "RGBA":
            alpha = image.getchannel("A")
            noisy = self._apply_noise(image.convert("RGB"), strength, rng).convert("RGBA")
            noisy.putalpha(alpha)
            return noisy
        if image.mode == "RGB":
            channels = tuple(Image.effect_noise(image.size, sigma) for _ in range(3))
            noise_image = Image.merge("RGB", channels)
            return ImageChops.add(image, noise_image, scale=1, offset=-128)
        if image.mode == "L":
            noise_image = Image.effect_noise(image.size, sigma)
            return ImageChops.add(image, noise_image, scale=1, offset=-128)
        return image

    def _apply_hue_shift(self, image: Image.Image, degrees: float) -> Image.Image:
        """Shift hue while preserving non-RGB modes when possible."""
        if image.mode not in {"RGB", "RGBA"}:
            return image

        alpha = None
        rgb_image = image
        if image.mode == "RGBA":
            alpha = image.getchannel("A")
            rgb_image = image.convert("RGB")

        hsv_image = rgb_image.convert("HSV")
        hue, sat, val = hsv_image.split()
        offset = int(round((degrees / 360.0) * 255)) % 256
        hue = hue.point(lambda value: (value + offset) % 256)
        shifted = Image.merge("HSV", (hue, sat, val)).convert("RGB")

        if alpha is not None:
            shifted = shifted.convert("RGBA")
            shifted.putalpha(alpha)
        return shifted

    def _save_augmented_image(self, image: Image.Image, output_path: Path) -> None:
        """Persist an augmented image to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image = image
        if output_path.suffix.lower() in {".jpg", ".jpeg"} and image.mode not in {"RGB", "L"}:
            save_image = image.convert("RGB")
        save_image.save(output_path)

    # ==================== 标签变换 ====================

    def _write_augmented_label(
        self,
        label_path: Path,
        label_format: LabelFormat,
        output_path: Path,
        geometry_ops: list[AppliedGeometryOp],
        output_image_path: Path,
        source_image_size: tuple[int, int],
    ) -> None:
        """Write a label file that matches the transformed image."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(output_image_path) as output_image:
            output_image_size = output_image.size
            output_depth = len(output_image.getbands())

        effective_source_size = output_image_size if not geometry_ops else source_image_size

        if label_format == LabelFormat.TXT:
            lines = self._transform_txt_label(
                label_path,
                geometry_ops,
                effective_source_size,
                output_image_size,
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return

        tree = self._transform_xml_label(
            label_path,
            geometry_ops,
            output_image_path,
            effective_source_size,
            output_image_size,
            output_depth,
        )
        ET.indent(tree, space="    ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    def _transform_txt_label(
        self,
        label_path: Path,
        geometry_ops: list[AppliedGeometryOp],
        source_image_size: tuple[int, int],
        output_image_size: tuple[int, int],
    ) -> list[str]:
        """Transform YOLO TXT labels for the generated sample."""
        transformed_lines: list[str] = []
        with open(label_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    transformed_lines.append(line)
                    continue

                try:
                    bbox = tuple(map(float, parts[1:5]))
                except ValueError:
                    transformed_lines.append(line)
                    continue

                absolute_box = self._normalized_box_to_absolute_float(
                    bbox,
                    source_image_size[0],
                    source_image_size[1],
                )
                transformed_box = self._transform_absolute_box(
                    absolute_box,
                    source_image_size,
                    geometry_ops,
                    output_image_size,
                )
                x_c, y_c, width, height = self._absolute_box_to_normalized(
                    transformed_box,
                    output_image_size[0],
                    output_image_size[1],
                )
                extras = f" {' '.join(parts[5:])}" if len(parts) > 5 else ""
                transformed_lines.append(
                    f"{parts[0]} {x_c:.6f} {y_c:.6f} {width:.6f} {height:.6f}{extras}"
                )

        return transformed_lines

    def _transform_xml_label(
        self,
        label_path: Path,
        geometry_ops: list[AppliedGeometryOp],
        output_image_path: Path,
        source_image_size: tuple[int, int],
        output_image_size: tuple[int, int],
        output_depth: int,
    ) -> ET.ElementTree:
        """Transform Pascal VOC XML boxes and metadata."""
        tree = ET.parse(label_path)
        root = tree.getroot()

        folder_elem = root.find("folder")
        if folder_elem is not None:
            folder_elem.text = output_image_path.parent.name

        filename_elem = root.find("filename")
        if filename_elem is not None:
            filename_elem.text = output_image_path.name

        size_elem = root.find("size")
        if size_elem is None:
            size_elem = ET.SubElement(root, "size")

        width_elem = size_elem.find("width")
        if width_elem is None:
            width_elem = ET.SubElement(size_elem, "width")
        height_elem = size_elem.find("height")
        if height_elem is None:
            height_elem = ET.SubElement(size_elem, "height")
        depth_elem = size_elem.find("depth")
        if depth_elem is None:
            depth_elem = ET.SubElement(size_elem, "depth")

        width_elem.text = str(output_image_size[0])
        height_elem.text = str(output_image_size[1])
        depth_elem.text = str(output_depth)

        for obj in root.findall(".//object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            try:
                xmin = float(bndbox.findtext("xmin", "0"))
                ymin = float(bndbox.findtext("ymin", "0"))
                xmax = float(bndbox.findtext("xmax", "0"))
                ymax = float(bndbox.findtext("ymax", "0"))
            except ValueError:
                continue

            transformed_box = self._transform_absolute_box(
                (xmin, ymin, xmax, ymax),
                source_image_size,
                geometry_ops,
                output_image_size,
            )
            abs_box = self._absolute_box_to_ints(transformed_box, output_image_size)

            xmin_elem = bndbox.find("xmin")
            if xmin_elem is None:
                xmin_elem = ET.SubElement(bndbox, "xmin")
            ymin_elem = bndbox.find("ymin")
            if ymin_elem is None:
                ymin_elem = ET.SubElement(bndbox, "ymin")
            xmax_elem = bndbox.find("xmax")
            if xmax_elem is None:
                xmax_elem = ET.SubElement(bndbox, "xmax")
            ymax_elem = bndbox.find("ymax")
            if ymax_elem is None:
                ymax_elem = ET.SubElement(bndbox, "ymax")

            xmin_elem.text = str(abs_box[0])
            ymin_elem.text = str(abs_box[1])
            xmax_elem.text = str(abs_box[2])
            ymax_elem.text = str(abs_box[3])

        return tree

    # ==================== 几何计算 ====================

    def _transform_absolute_box(
        self,
        bbox: tuple[float, float, float, float],
        source_image_size: tuple[int, int],
        geometry_ops: list[AppliedGeometryOp],
        output_image_size: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        """Apply geometric transforms to an axis-aligned box."""
        points = [
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3]),
        ]
        current_size = source_image_size

        for operation in geometry_ops:
            points, current_size = self._apply_geometry_to_points(
                points,
                current_size,
                operation,
            )

        transformed_box = self._points_to_box(points)
        return self._clamp_absolute_box(transformed_box, output_image_size)

    def _apply_geometry_to_points(
        self,
        points: list[tuple[float, float]],
        current_size: tuple[int, int],
        operation: AppliedGeometryOp,
    ) -> tuple[list[tuple[float, float]], tuple[int, int]]:
        """Transform box corners using the same geometry as the image."""
        current_width, current_height = current_size

        if operation.kind == "flip_lr":
            return ([(current_width - x, y) for x, y in points], current_size)
        if operation.kind == "flip_ud":
            return ([(x, current_height - y) for x, y in points], current_size)
        if operation.kind == "rotate":
            matrix, rotated_size = self._build_rotation_reverse_matrix(
                current_width,
                current_height,
                operation.value,
                expand=True,
            )
            rotated_points = [
                self._invert_reverse_affine_point(x, y, matrix) for x, y in points
            ]
            return rotated_points, rotated_size
        raise ValueError(f"不支持的增强操作: {operation.kind}")

    def _build_rotation_reverse_matrix(
        self,
        width: int,
        height: int,
        angle: float,
        *,
        expand: bool,
    ) -> tuple[list[float], tuple[int, int]]:
        """Mirror Pillow's reverse affine matrix for rotate(expand=...)."""
        angle = angle % 360.0
        center_x = width / 2.0
        center_y = height / 2.0
        angle_radians = -math.radians(angle)
        matrix = [
            round(math.cos(angle_radians), 15),
            round(math.sin(angle_radians), 15),
            0.0,
            round(-math.sin(angle_radians), 15),
            round(math.cos(angle_radians), 15),
            0.0,
        ]

        def transform(x: float, y: float) -> tuple[float, float]:
            a, b, c, d, e, f = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(-center_x, -center_y)
        matrix[2] += center_x
        matrix[5] += center_y

        output_width, output_height = width, height
        if expand:
            xs: list[float] = []
            ys: list[float] = []
            for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
                rx, ry = transform(x, y)
                xs.append(rx)
                ys.append(ry)
            output_width = math.ceil(max(xs)) - math.floor(min(xs))
            output_height = math.ceil(max(ys)) - math.floor(min(ys))
            matrix[2], matrix[5] = transform(
                -(output_width - width) / 2.0,
                -(output_height - height) / 2.0,
            )

        return matrix, (int(output_width), int(output_height))

    def _invert_reverse_affine_point(
        self,
        x: float,
        y: float,
        matrix: list[float],
    ) -> tuple[float, float]:
        """Convert a source point through Pillow's reverse affine matrix."""
        a, b, c, d, e, f = matrix
        det = a * e - b * d
        if det == 0:
            return x, y
        return (
            (e * (x - c) - b * (y - f)) / det,
            (-d * (x - c) + a * (y - f)) / det,
        )

    def _points_to_box(
        self,
        points: list[tuple[float, float]],
    ) -> tuple[float, float, float, float]:
        """Convert transformed polygon corners back into an AABB."""
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _clamp_absolute_box(
        self,
        bbox: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        """Clamp a box to image bounds while keeping a non-zero extent."""
        width, height = image_size
        if width <= 0 or height <= 0:
            return 0.0, 0.0, 0.0, 0.0

        xmin, ymin, xmax, ymax = bbox
        max_xmin = max(float(width) - 1.0, 0.0)
        max_ymin = max(float(height) - 1.0, 0.0)

        xmin = min(max(xmin, 0.0), max_xmin)
        ymin = min(max(ymin, 0.0), max_ymin)
        xmax = min(max(xmax, xmin + 1.0), float(width))
        ymax = min(max(ymax, ymin + 1.0), float(height))
        return xmin, ymin, xmax, ymax

    def _normalized_box_to_absolute_float(
        self,
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        """Convert normalized YOLO boxes into continuous image coordinates."""
        x_c, y_c, box_w, box_h = bbox
        xmin = (x_c - box_w / 2.0) * width
        ymin = (y_c - box_h / 2.0) * height
        xmax = (x_c + box_w / 2.0) * width
        ymax = (y_c + box_h / 2.0) * height
        return xmin, ymin, xmax, ymax

    def _absolute_box_to_normalized(
        self,
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        """Convert continuous image coordinates into normalized YOLO boxes."""
        if width <= 0 or height <= 0:
            return 0.0, 0.0, 0.0, 0.0

        xmin, ymin, xmax, ymax = bbox
        x_c = ((xmin + xmax) / 2.0) / width
        y_c = ((ymin + ymax) / 2.0) / height
        box_w = max(0.0, (xmax - xmin) / width)
        box_h = max(0.0, (ymax - ymin) / height)
        return (
            min(max(x_c, 0.0), 1.0),
            min(max(y_c, 0.0), 1.0),
            min(max(box_w, 0.0), 1.0),
            min(max(box_h, 0.0), 1.0),
        )

    def _absolute_box_to_ints(
        self,
        bbox: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Round and clamp a float box for VOC XML output."""
        width, height = image_size
        xmin = int(round(bbox[0]))
        ymin = int(round(bbox[1]))
        xmax = int(round(bbox[2]))
        ymax = int(round(bbox[3]))

        xmin = max(0, min(max(width - 1, 0), xmin))
        ymin = max(0, min(max(height - 1, 0), ymin))
        xmax = max(xmin + 1, min(width, xmax)) if width > 0 else 0
        ymax = max(ymin + 1, min(height, ymax)) if height > 0 else 0
        return xmin, ymin, xmax, ymax

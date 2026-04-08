from __future__ import annotations

from pathlib import Path
from typing import Any


def _error_payload(exc: Exception, action: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": f"{action}失败: {exc}",
        "error_type": exc.__class__.__name__,
    }


def summarize_scan_result(result) -> str:
    return (
        f"总图片: {result.total_images}, 已标注: {result.labeled_images}, "
        f"缺失标签: {len(result.missing_labels)}, 空标签: {result.empty_labels}, "
        f"类别: {result.classes}, 类别统计: {result.class_stats}"
    )


def scan_dataset(img_dir: str, label_dir: str = "") -> dict[str, Any]:
    """扫描数据集并返回摘要。"""
    try:
        from core.data_handler._handler import DataHandler

        handler = DataHandler()
        result = handler.scan_dataset(
            img_dir=Path(img_dir),
            label_dir=Path(label_dir) if label_dir else None,
        )
        return {
            "ok": True,
            "summary": summarize_scan_result(result),
            "total_images": result.total_images,
            "labeled_images": result.labeled_images,
            "missing_labels": len(result.missing_labels),
            "empty_labels": result.empty_labels,
            "classes": result.classes,
            "class_stats": result.class_stats,
        }
    except Exception as exc:
        return _error_payload(exc, "扫描数据集")


def split_dataset(
    img_dir: str,
    label_dir: str = "",
    output_dir: str = "",
    ratio: float = 0.8,
    seed: int = 42,
    mode: str = "copy",
    ignore_orphans: bool = False,
    clear_output: bool = False,
) -> dict[str, Any]:
    """按现有 DataHandler 能力将数据集切分为 train/val。

    mode 可选值: copy（复制文件）、move（移动文件）、index（生成索引txt）。
    不支持其他值（如 trainval）。
    """
    try:
        from core.data_handler._handler import DataHandler
        from core.data_handler._models import SplitMode

        handler = DataHandler()
        mode_map = {
            "copy": SplitMode.COPY,
            "move": SplitMode.MOVE,
            "index": SplitMode.INDEX,
        }
        selected_mode = mode_map.get(mode.lower())
        if selected_mode is None:
            raise ValueError(f"不支持的 split mode: {mode}")

        resolved_output = Path(output_dir) if output_dir else None
        result = handler.split_dataset(
            img_dir=Path(img_dir),
            label_dir=Path(label_dir) if label_dir else None,
            output_dir=resolved_output,
            ratio=ratio,
            seed=seed,
            mode=selected_mode,
            ignore_orphans=ignore_orphans,
            clear_output=clear_output,
        )
        # 推算实际输出目录的绝对路径
        if resolved_output:
            abs_output = str(resolved_output.resolve())
        else:
            # 默认: img_dir 同级的 {name}_split 目录
            abs_output = str((Path(img_dir).parent / f"{Path(img_dir).name}_split").resolve())
        return {
            "ok": True,
            "output_dir": abs_output,
            "train_path": result.train_path,
            "val_path": result.val_path,
            "train_count": result.train_count,
            "val_count": result.val_count,
            "ratio": ratio,
            "mode": mode.lower(),
        }
    except Exception as exc:
        return _error_payload(exc, "划分数据集")


def validate_dataset(
    img_dir: str,
    label_dir: str = "",
    classes_txt: str = "",
    check_coords: bool = True,
    check_class_ids: bool = True,
    check_format: bool = True,
    check_orphans: bool = True,
) -> dict[str, Any]:
    """校验标签合法性并返回问题统计。"""
    try:
        from core.data_handler._handler import DataHandler

        handler = DataHandler()
        result = handler.validate_labels(
            img_dir=Path(img_dir),
            label_dir=Path(label_dir) if label_dir else None,
            classes_txt=Path(classes_txt) if classes_txt else None,
            check_coords=check_coords,
            check_class_ids=check_class_ids,
            check_format=check_format,
            check_orphans=check_orphans,
        )
        return {
            "ok": True,
            "total_labels": result.total_labels,
            "has_issues": result.has_issues,
            "issue_count": result.issue_count,
            "coord_errors": len(result.coord_errors),
            "class_errors": len(result.class_errors),
            "format_errors": len(result.format_errors),
            "orphan_labels": len(result.orphan_labels),
        }
    except Exception as exc:
        return _error_payload(exc, "校验数据集")


def augment_dataset(
    img_dir: str,
    label_dir: str = "",
    output_dir: str = "",
    classes_txt: str = "",
    copies_per_image: int = 1,
    include_original: bool = True,
    seed: int = 42,
    mode: str = "random",
    enable_horizontal_flip: bool = True,
    enable_rotate: bool = False,
    rotate_degrees: float = 15.0,
    enable_brightness: bool = False,
    brightness_strength: float = 0.2,
    enable_contrast: bool = False,
    contrast_strength: float = 0.25,
    enable_noise: bool = False,
    noise_strength: float = 0.08,
) -> dict[str, Any]:
    """执行离线数据增强，默认启用最常用的水平翻转。"""
    try:
        from core.data_handler._handler import DataHandler
        from core.data_handler._models import AugmentConfig

        handler = DataHandler()
        config = AugmentConfig(
            copies_per_image=copies_per_image,
            include_original=include_original,
            seed=seed,
            mode=mode,
            enable_horizontal_flip=enable_horizontal_flip,
            enable_rotate=enable_rotate,
            rotate_degrees=rotate_degrees,
            enable_brightness=enable_brightness,
            brightness_strength=brightness_strength,
            enable_contrast=enable_contrast,
            contrast_strength=contrast_strength,
            enable_noise=enable_noise,
            noise_strength=noise_strength,
        )
        result = handler.augment_dataset(
            img_dir=Path(img_dir),
            config=config,
            label_dir=Path(label_dir) if label_dir else None,
            output_dir=Path(output_dir) if output_dir else None,
            classes_txt=Path(classes_txt) if classes_txt else None,
        )
        return {
            "ok": True,
            "output_dir": result.output_dir,
            "source_images": result.source_images,
            "copied_originals": result.copied_originals,
            "augmented_images": result.augmented_images,
            "label_files_written": result.label_files_written,
            "skipped_images": result.skipped_images,
        }
    except Exception as exc:
        return _error_payload(exc, "增强数据集")

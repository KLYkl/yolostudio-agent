from __future__ import annotations

from pathlib import Path
from typing import Any


MAX_ISSUE_EXAMPLES = 3


def _error_payload(exc: Exception, action: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": f"{action}失败: {exc}",
        "error_type": exc.__class__.__name__,
    }


def _infer_dataset_root(img_dir: Path, label_dir: Path | None = None) -> Path:
    candidates = [img_dir]
    if img_dir.name.lower() in {"images", "imgs", "jpegimages"}:
        candidates.append(img_dir.parent)
    if label_dir:
        candidates.append(label_dir)
        if label_dir.name.lower() in {"labels", "annotations", "label"}:
            candidates.append(label_dir.parent)
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            if candidate.name.lower() in {"images", "labels", "annotations", "label", "imgs", "jpegimages"}:
                continue
            return candidate
    return img_dir.parent if img_dir.name.lower() in {"images", "imgs", "jpegimages"} else img_dir


def _discover_data_yaml(img_dir: Path, label_dir: Path | None = None) -> tuple[str, list[str]]:
    dataset_root = _infer_dataset_root(img_dir, label_dir)
    search_dirs = [dataset_root, dataset_root.parent, img_dir.parent, img_dir.parent.parent]
    names = {
        "data.yaml",
        "data.yml",
        "dataset.yaml",
        "dataset.yml",
        f"{dataset_root.name}.yaml",
        f"{dataset_root.name}.yml",
        f"{img_dir.parent.name}.yaml",
        f"{img_dir.parent.name}.yml",
    }
    candidates: list[str] = []
    seen: set[str] = set()
    for base in search_dirs:
        if not base:
            continue
        for name in names:
            path = (base / name).resolve()
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                candidates.append(key)
    detected = candidates[0] if candidates else ""
    return detected, candidates


def _top_class_stats(class_stats: dict[str, int], limit: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(class_stats.items(), key=lambda item: item[1], reverse=True)
    return [{"class": name, "count": count} for name, count in ordered[:limit]]


def _format_issue_examples(result) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = {}
    if result.coord_errors:
        examples["coord_errors"] = [
            f"{path} [{where}] {reason}" for path, where, reason in result.coord_errors[:MAX_ISSUE_EXAMPLES]
        ]
    if result.class_errors:
        examples["class_errors"] = [
            f"{path} [{where}] {reason}" for path, where, reason in result.class_errors[:MAX_ISSUE_EXAMPLES]
        ]
    if result.format_errors:
        examples["format_errors"] = [
            f"{path} {reason}" for path, reason in result.format_errors[:MAX_ISSUE_EXAMPLES]
        ]
    if result.orphan_labels:
        examples["orphan_labels"] = [str(path) for path in result.orphan_labels[:MAX_ISSUE_EXAMPLES]]
    return examples


def summarize_scan_result(result) -> str:
    return (
        f"总图片: {result.total_images}, 已标注: {result.labeled_images}, "
        f"缺失标签: {len(result.missing_labels)}, 空标签: {result.empty_labels}, "
        f"类别数: {len(result.classes)}"
    )


def scan_dataset(img_dir: str, label_dir: str = "") -> dict[str, Any]:
    """扫描数据集并返回结构化摘要、类别统计、候选 YAML 信息。"""
    try:
        from core.data_handler._handler import DataHandler

        img_path = Path(img_dir)
        label_path = Path(label_dir) if label_dir else None
        dataset_root = _infer_dataset_root(img_path, label_path)
        detected_data_yaml, data_yaml_candidates = _discover_data_yaml(img_path, label_path)

        handler = DataHandler()
        result = handler.scan_dataset(
            img_dir=img_path,
            label_dir=label_path,
        )
        next_actions = ["可继续 validate_dataset 做标签合法性校验"]
        if detected_data_yaml:
            next_actions.append(f"可直接使用 detected_data_yaml 训练: {detected_data_yaml}")
        else:
            next_actions.append("尚未发现可直接训练的 data.yaml；如要训练需显式提供 YAML 路径")
        return {
            "ok": True,
            "summary": summarize_scan_result(result),
            "dataset_root": str(dataset_root),
            "total_images": result.total_images,
            "labeled_images": result.labeled_images,
            "missing_labels": len(result.missing_labels),
            "missing_label_examples": [str(path) for path in result.missing_labels[:MAX_ISSUE_EXAMPLES]],
            "empty_labels": result.empty_labels,
            "classes": result.classes,
            "class_stats": result.class_stats,
            "top_classes": _top_class_stats(result.class_stats),
            "label_format": result.label_format.name if result.label_format else None,
            "detected_data_yaml": detected_data_yaml,
            "data_yaml_candidates": data_yaml_candidates,
            "next_actions": next_actions,
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
    """按现有 DataHandler 能力将数据集切分为 train/val。"""
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
        if resolved_output:
            abs_output = str(resolved_output.resolve())
        else:
            abs_output = str((Path(img_dir).parent / f"{Path(img_dir).name}_split").resolve())
        train_ratio = round(result.train_count / max(result.train_count + result.val_count, 1), 4)
        val_ratio = round(result.val_count / max(result.train_count + result.val_count, 1), 4)
        suggested_yaml = str((Path(abs_output) / 'data.yaml').resolve())
        return {
            "ok": True,
            "summary": f"数据集已划分: train={result.train_count}, val={result.val_count}, mode={mode.lower()}",
            "output_dir": abs_output,
            "train_path": result.train_path,
            "val_path": result.val_path,
            "train_count": result.train_count,
            "val_count": result.val_count,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "mode": mode.lower(),
            "suggested_yaml_path": suggested_yaml,
            "next_actions": [
                "建议检查划分结果是否符合预期",
                f"如需训练，可基于输出目录生成/准备 YAML: {suggested_yaml}",
            ],
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
    """校验标签合法性并返回问题统计与示例。"""
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
        breakdown = {
            "coord_errors": len(result.coord_errors),
            "class_errors": len(result.class_errors),
            "format_errors": len(result.format_errors),
            "orphan_labels": len(result.orphan_labels),
        }
        summary = (
            "未发现标签问题" if not result.has_issues else
            f"发现 {result.issue_count} 个问题: 坐标 {breakdown['coord_errors']}, 类别 {breakdown['class_errors']}, 格式 {breakdown['format_errors']}, 孤立标签 {breakdown['orphan_labels']}"
        )
        return {
            "ok": True,
            "summary": summary,
            "total_labels": result.total_labels,
            "has_issues": result.has_issues,
            "issue_count": result.issue_count,
            "issue_breakdown": breakdown,
            "issue_examples": _format_issue_examples(result),
            "next_actions": (
                ["可继续训练或做数据划分"] if not result.has_issues else
                ["建议先修复 issue_examples 中的问题，再继续划分或训练"]
            ),
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
        enabled_operations = config.enabled_operations()
        total_output = result.copied_originals + result.augmented_images
        return {
            "ok": True,
            "summary": f"增强完成: 输出 {total_output} 张（原图 {result.copied_originals} / 增强 {result.augmented_images}）",
            "output_dir": result.output_dir,
            "source_images": result.source_images,
            "copied_originals": result.copied_originals,
            "augmented_images": result.augmented_images,
            "total_output_images": total_output,
            "label_files_written": result.label_files_written,
            "skipped_images": result.skipped_images,
            "mode": mode,
            "enabled_operations": enabled_operations,
            "next_actions": [
                f"可检查增强输出目录: {result.output_dir}",
                "如结果符合预期，再将增强数据纳入训练流程",
            ],
        }
    except Exception as exc:
        return _error_payload(exc, "增强数据集")

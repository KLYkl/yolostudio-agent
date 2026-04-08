"""
_models.py - 数据类型定义
============================================

包含所有枚举、数据类（dataclass）类型定义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from utils.constants import IMAGE_EXTENSIONS, LABEL_EXTENSIONS, VIDEO_EXTENSIONS  # noqa: F401
from utils.file_utils import get_unique_dir as _get_unique_dir  # noqa: F401




# ============================================================
# 数据类型定义
# ============================================================

class LabelFormat(Enum):
    """标签格式枚举"""
    TXT = auto()   # YOLO TXT 格式
    XML = auto()   # Pascal VOC XML 格式


class SplitMode(Enum):
    """数据集划分模式"""
    MOVE = auto()      # 物理移动文件 (剪切)
    COPY = auto()      # 物理复制文件 (推荐，更安全)
    INDEX = auto()     # 生成索引文件 (txt)


class ModifyAction(Enum):
    """标签修改动作"""
    REPLACE = auto()   # 替换类别
    REMOVE = auto()    # 删除类别


@dataclass
class ScanResult:
    """
    数据集扫描结果

    Attributes:
        total_images: 图片总数
        labeled_images: 有标签的图片数
        missing_labels: 缺失标签的图片路径列表
        empty_labels: 空标签文件数
        class_stats: 类别统计 {类别名: 数量}
        classes: 检测到的类别列表 (有序)
        label_format: 检测到的标签格式
    """
    total_images: int = 0
    labeled_images: int = 0
    missing_labels: list[Path] = field(default_factory=list)
    empty_labels: int = 0
    class_stats: dict[str, int] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    label_format: Optional[LabelFormat] = None


@dataclass
class SplitResult:
    """
    数据集划分结果

    Attributes:
        train_path: 训练集路径 (文件夹或 txt 索引)
        val_path: 验证集路径 (文件夹或 txt 索引)
        train_count: 训练集数量
        val_count: 验证集数量
    """
    train_path: str = ""
    val_path: str = ""
    train_count: int = 0
    val_count: int = 0


@dataclass
class AugmentConfig:
    """Offline dataset augmentation settings."""

    copies_per_image: int = 1
    include_original: bool = True
    seed: int = 42
    mode: str = "random"
    custom_recipes: list[tuple[str, ...]] = field(default_factory=list)
    enable_horizontal_flip: bool = False
    enable_vertical_flip: bool = False
    enable_rotate: bool = False
    rotate_mode: str = "random"
    rotate_degrees: float = 15.0
    enable_brightness: bool = False
    brightness_strength: float = 0.2
    enable_contrast: bool = False
    contrast_strength: float = 0.25
    enable_color: bool = False
    color_strength: float = 0.25
    enable_noise: bool = False
    noise_strength: float = 0.08
    enable_hue: bool = False
    hue_degrees: float = 12.0
    enable_sharpness: bool = False
    sharpness_strength: float = 0.4
    enable_blur: bool = False
    blur_radius: float = 1.2

    def geometric_candidates(self) -> list[str]:
        ops: list[str] = []
        if self.enable_horizontal_flip:
            ops.append("flip_lr")
        if self.enable_vertical_flip:
            ops.append("flip_ud")
        if self.enable_rotate and self.rotate_degrees > 0:
            ops.append("rotate")
        return ops

    def photometric_candidates(self) -> list[str]:
        ops: list[str] = []
        if self.enable_brightness and self.brightness_strength > 0:
            ops.append("brightness")
        if self.enable_contrast and self.contrast_strength > 0:
            ops.append("contrast")
        if self.enable_color and self.color_strength > 0:
            ops.append("color")
        if self.enable_noise and self.noise_strength > 0:
            ops.append("noise")
        if self.enable_hue and self.hue_degrees > 0:
            ops.append("hue")
        if self.enable_sharpness and self.sharpness_strength > 0:
            ops.append("sharpness")
        if self.enable_blur and self.blur_radius > 0:
            ops.append("blur")
        return ops

    def enabled_operations(self) -> list[str]:
        return self.geometric_candidates() + self.photometric_candidates()

    def has_any_operation(self) -> bool:
        return bool(self.enabled_operations())

    def build_fixed_recipes(self) -> list["AugmentRecipe"]:
        """Build recipes from custom_recipes list."""
        recipes: list[AugmentRecipe] = []
        seen: set[tuple[str, ...]] = set()
        enabled = set(self.enabled_operations())

        for recipe_ops in self.custom_recipes:
            # Filter to only currently enabled operations
            filtered = tuple(op for op in recipe_ops if op in enabled)
            if not filtered or filtered in seen:
                continue
            if len(filtered) == 1:
                name = self.operation_slug(filtered[0])
            else:
                name = "combo_" + "_".join(self.operation_slug(op) for op in filtered)
            recipes.append(AugmentRecipe(name, filtered))
            seen.add(filtered)

        return recipes

    @staticmethod
    def operation_slug(operation: str) -> str:
        mapping = {
            "flip_lr": "hflip",
            "flip_ud": "vflip",
            "rotate": "rotate",
            "brightness": "brightness",
            "contrast": "contrast",
            "color": "saturation",
            "noise": "noise",
            "hue": "hue",
            "sharpness": "sharpness",
            "blur": "blur",
        }
        return mapping.get(operation, operation)


@dataclass(frozen=True)
class AugmentRecipe:
    """A deterministic augmentation recipe used by fixed mode."""

    name: str
    operations: tuple[str, ...]


@dataclass(frozen=True)
class AppliedGeometryOp:
    """A concrete geometric transform applied to one augmented sample."""

    kind: str
    value: float = 0.0


@dataclass
class AugmentResult:
    """数据增强结果"""

    output_dir: str = ""
    source_images: int = 0
    copied_originals: int = 0
    augmented_images: int = 0
    label_files_written: int = 0
    skipped_images: int = 0


@dataclass
class ValidateResult:
    """
    标签校验结果

    Attributes:
        total_labels: 扫描的标签文件总数
        coord_errors: 坐标越界的文件列表 [(文件路径, 行号/对象名, 原因)]
        class_errors: 类别无效的文件列表 [(文件路径, 行号/对象名, 原因)]
        format_errors: 格式错误的文件列表 [(文件路径, 原因)]
        orphan_labels: 孤立标签文件列表 (无对应图片)
    """
    total_labels: int = 0
    coord_errors: list[tuple[Path, str, str]] = field(default_factory=list)
    class_errors: list[tuple[Path, str, str]] = field(default_factory=list)
    format_errors: list[tuple[Path, str]] = field(default_factory=list)
    orphan_labels: list[Path] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        """是否存在任何问题"""
        return bool(self.coord_errors or self.class_errors
                    or self.format_errors or self.orphan_labels)

    @property
    def issue_count(self) -> int:
        """问题总数"""
        return (len(self.coord_errors) + len(self.class_errors)
                + len(self.format_errors) + len(self.orphan_labels))


@dataclass
class ImageCheckResult:
    """
    图像完整性校验结果

    Attributes:
        total_images: 扫描的图片总数
        corrupted: 损坏图片列表 [(文件路径, 原因)]
        zero_bytes: 零字节文件列表
        format_mismatch: 文件头/扩展名不匹配列表 [(路径, 声称扩展名, 真实格式)]
        exif_rotation: 含 EXIF 旋转标记的图片列表 [(路径, orientation值)]
    """
    total_images: int = 0
    corrupted: list[tuple[Path, str]] = field(default_factory=list)
    zero_bytes: list[Path] = field(default_factory=list)
    format_mismatch: list[tuple[Path, str, str]] = field(default_factory=list)
    exif_rotation: list[tuple[Path, int]] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.corrupted or self.zero_bytes
                    or self.format_mismatch or self.exif_rotation)

    @property
    def issue_count(self) -> int:
        return (len(self.corrupted) + len(self.zero_bytes)
                + len(self.format_mismatch) + len(self.exif_rotation))


@dataclass
class ImageSizeStats:
    """
    图像尺寸分析结果

    Attributes:
        total_images: 图片总数
        min_size: 最小尺寸 (宽, 高)
        max_size: 最大尺寸 (宽, 高)
        avg_size: 平均尺寸 (宽, 高)
        sizes: 所有图片尺寸列表 [(路径, 宽, 高)]
        abnormal_small: 异常小图片列表 (宽或高 < 阈值)
        abnormal_large: 异常大图片列表 (宽或高 > 阈值)
    """
    total_images: int = 0
    min_size: tuple[int, int] = (0, 0)
    max_size: tuple[int, int] = (0, 0)
    avg_size: tuple[int, int] = (0, 0)
    sizes: list[tuple[Path, int, int]] = field(default_factory=list)
    abnormal_small: list[Path] = field(default_factory=list)
    abnormal_large: list[Path] = field(default_factory=list)


@dataclass
class DuplicateGroup:
    """
    重复图片组

    Attributes:
        hash_value: 哈希值字符串
        paths: 该组中重复图片的路径列表
    """
    hash_value: str = ""
    paths: list[Path] = field(default_factory=list)


@dataclass
class ExtractConfig:
    """
    图片抽取配置

    Attributes:
        mode: 抽取模式 ("by_category" | "by_directory")
        per_item_counts: 每个类别/目录的独立提取配置
            {名称: (模式, 值)}
            模式: "all" | "count" | "ratio"
            值: 具体数量(int) 或 比例(float, 0~1)
            示例: {"car": ("all", 0), "person": ("count", 100)}
        categories: 目标类别列表 (含 _empty/_mixed/_no_label)
        selected_dirs: 选中的目录列表
        output_layout: 输出目录布局
            "keep" — 保持原始目录结构
            "flat" — 扁平化到 images/labels 子目录
            "by_category" — 按类别创建子目录 (仅按类别抽取模式)
        copy_labels: 是否同时复制标签
        seed: 随机种子 (None=不固定)
        output_dir: 输出目录
    """
    mode: str = "by_category"
    per_item_counts: dict[str, tuple[str, float]] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)
    selected_dirs: list[Path] = field(default_factory=list)
    output_layout: str = "keep"
    copy_labels: bool = True
    seed: Optional[int] = None
    output_dir: Optional[Path] = None


@dataclass
class ExtractResult:
    """
    图片抽取结果

    Attributes:
        output_dir: 输出目录路径
        total_available: 可用图片总数
        extracted: 实际提取数量
        labels_copied: 复制的标签数
        dir_stats: 每个目录的提取统计 {目录名: 提取数量}
        conflicts: 文件冲突列表 [(新文件路径, 已存在文件路径)]
    """
    output_dir: str = ""
    total_available: int = 0
    extracted: int = 0
    labels_copied: int = 0
    dir_stats: dict[str, int] = field(default_factory=dict)
    conflicts: list[tuple[Path, Path]] = field(default_factory=list)


@dataclass
class VideoExtractConfig:
    """视频抽帧配置"""
    mode: str = "interval"  # 抽帧模式
    frame_interval: int = 30  # 按帧间隔抽取时的间隔帧数
    time_interval: float = 1.0  # 按时间间隔抽取时的间隔秒数
    scene_threshold: float = 0.4  # 场景切换检测阈值
    min_scene_gap: int = 15  # 相邻场景切换的最小帧间隔
    enable_dedup: bool = True  # 是否启用去重
    dedup_threshold: int = 8  # 去重相似度阈值
    max_frames: int = 0  # 最大抽取帧数，0 表示不限制
    start_time: float = 0.0  # 开始抽取时间（秒）
    end_time: float = 0.0  # 结束抽取时间（秒），0 表示到视频末尾
    output_format: str = "jpg"  # 输出图片格式
    jpg_quality: int = 95  # JPG 输出质量
    name_prefix: str = ""  # 输出文件名前缀
    output_dir: Optional[Path] = None  # 输出目录


@dataclass
class VideoExtractResult:
    """视频抽帧结果"""
    output_dir: str = ""  # 输出目录路径
    total_frames: int = 0  # 视频总帧数
    extracted: int = 0  # 原始抽取帧数
    dedup_removed: int = 0  # 去重移除帧数
    final_count: int = 0  # 最终保留帧数
    video_stats: dict[str, int] = field(default_factory=dict)  # 视频统计信息
    duration: float = 0.0  # 视频时长（秒）
    skipped: int = 0  # 跳过的帧数


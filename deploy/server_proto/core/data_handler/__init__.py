"""
core.data_handler - 数据处理核心模块
============================================

保持外部 API 不变:
    from core.data_handler import DataHandler
    from core.data_handler import DataWorker
    from core.data_handler import LabelFormat, ScanResult, ...
"""

# 保持 os 在模块级可访问 (测试 monkey-patch 兼容性)
import os  # noqa: F401

from core.data_handler._models import (
    IMAGE_EXTENSIONS,
    LABEL_EXTENSIONS,
    VIDEO_EXTENSIONS,
    AppliedGeometryOp,
    AugmentConfig,
    AugmentRecipe,
    AugmentResult,
    DuplicateGroup,
    ExtractConfig,
    ExtractResult,
    ImageCheckResult,
    ImageSizeStats,
    LabelFormat,
    ModifyAction,
    ScanResult,
    SplitMode,
    SplitResult,
    VideoExtractConfig,
    VideoExtractResult,
    ValidateResult,
    _get_unique_dir,
)
from core.data_handler._handler import DataHandler
from core.data_handler._worker import DataWorker

__all__ = [
    # 常量
    "IMAGE_EXTENSIONS",
    "LABEL_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    # 数据类型
    "LabelFormat",
    "SplitMode",
    "ModifyAction",
    "ScanResult",
    "SplitResult",
    "VideoExtractConfig",
    "VideoExtractResult",
    "AugmentConfig",
    "AugmentRecipe",
    "AppliedGeometryOp",
    "AugmentResult",
    "ValidateResult",
    "ExtractConfig",
    "ExtractResult",
    "ImageCheckResult",
    "ImageSizeStats",
    "DuplicateGroup",
    # 核心类
    "DataHandler",
    "DataWorker",
    # 辅助函数
    "_get_unique_dir",
]

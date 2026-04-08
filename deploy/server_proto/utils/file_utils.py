"""
file_utils.py - 文件工具函数
============================================

提供通用的文件发现、去重、目录唯一性等纯函数。
"""

from __future__ import annotations

from pathlib import Path


def discover_files(
    source: str | Path | list[str] | list[Path],
    extensions: set[str],
) -> list[Path]:
    """
    通用的文件发现 + 去重 + 排序

    支持三种输入：
    - 单个目录路径: 扫描目录中所有匹配扩展名的文件
    - 单个文件路径: 如果扩展名匹配，返回单元素列表
    - 文件路径列表: 过滤出匹配扩展名的文件

    Args:
        source: 输入源（目录/文件/列表）
        extensions: 允许的文件扩展名集合 (如 {".jpg", ".png"})

    Returns:
        去重、排序后的文件路径列表
    """
    file_list: list[Path] = []

    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.is_dir():
            for ext in extensions:
                file_list.extend(source_path.glob(f"*{ext}"))
            # 去重 (同一文件可能有大小写不同的扩展名)
            seen: set[Path] = set()
            unique_list: list[Path] = []
            for p in file_list:
                key = p.resolve()
                if key not in seen:
                    seen.add(key)
                    unique_list.append(p)
            unique_list.sort(key=lambda p: p.name.lower())
            return unique_list
        elif source_path.is_file() and source_path.suffix.lower() in extensions:
            return [source_path]
    elif isinstance(source, list):
        for p in source:
            path = Path(p)
            if path.is_file() and path.suffix.lower() in extensions:
                file_list.append(path)

    return file_list


def get_unique_dir(base_path: Path) -> Path:
    """
    获取唯一的目录路径，如果目录已存在则添加数字后缀

    例如:
        - my_dir -> my_dir (如果不存在)
        - my_dir -> my_dir(1) (如果 my_dir 已存在)
        - my_dir -> my_dir(2) (如果 my_dir 和 my_dir(1) 都已存在)

    Args:
        base_path: 基础目录路径

    Returns:
        唯一的目录路径
    """
    if not base_path.exists():
        return base_path

    parent = base_path.parent
    name = base_path.name

    counter = 1
    while True:
        new_path = parent / f"{name}({counter})"
        if not new_path.exists():
            return new_path
        counter += 1

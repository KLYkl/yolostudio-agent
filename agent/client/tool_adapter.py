from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


TOOL_NAME_ALIASES: dict[str, str] = {
    'detect_duplicates': 'detect_duplicate_images',
    'detect_corrupted_images': 'run_dataset_health_check',
    'prepare_dataset': 'prepare_dataset_for_training',
    'dataset_manager.prepare_dataset': 'prepare_dataset_for_training',
    'predict_directory': 'predict_images',
    'batch_predict_images': 'predict_images',
    'predict_images_in_dir': 'predict_images',
    'predict_video_directory': 'predict_videos',
    'batch_predict_videos': 'predict_videos',
    'predict_videos_in_dir': 'predict_videos',
    'summarize_predictions': 'summarize_prediction_results',
    'summarize_prediction_report': 'summarize_prediction_results',
    'analyze_prediction_report': 'summarize_prediction_results',
}

_ARG_ALIASES: dict[str, dict[str, str]] = {
    'run_dataset_health_check': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
    },
    'detect_duplicate_images': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
    },
    'prepare_dataset_for_training': {
        'path': 'dataset_path',
        'dataset': 'dataset_path',
        'root': 'dataset_path',
        'img_dir': 'dataset_path',
    },
    'training_readiness': {
        'path': 'img_dir',
        'dataset_path': 'img_dir',
        'dataset': 'img_dir',
        'root': 'img_dir',
    },
    'scan_dataset': {
        'path': 'img_dir',
        'dataset_path': 'img_dir',
        'dataset': 'img_dir',
        'root': 'img_dir',
    },
    'validate_dataset': {
        'path': 'img_dir',
        'dataset_path': 'img_dir',
        'dataset': 'img_dir',
        'root': 'img_dir',
    },
    'predict_images': {
        'path': 'source_path',
        'source': 'source_path',
        'input_path': 'source_path',
        'dir_path': 'source_path',
        'folder': 'source_path',
    },
    'predict_videos': {
        'path': 'source_path',
        'source': 'source_path',
        'input_path': 'source_path',
        'dir_path': 'source_path',
        'folder': 'source_path',
    },
    'summarize_prediction_results': {
        'path': 'report_path',
        'report': 'report_path',
        'json_report': 'report_path',
        'file': 'report_path',
        'dir_path': 'output_dir',
        'folder': 'output_dir',
        'output': 'output_dir',
    },
}


def _stringify_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get('type') == 'text' and item.get('text'):
                    parts.append(str(item['text']))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return '\n'.join(part for part in parts if part)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def canonical_tool_name(name: str) -> str:
    key = (name or '').strip()
    return TOOL_NAME_ALIASES.get(key, key)


def normalize_tool_args(tool_name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    canonical_name = canonical_tool_name(tool_name)
    payload = dict(args or {})
    for alias, target in _ARG_ALIASES.get(canonical_name, {}).items():
        if not payload.get(target) and payload.get(alias):
            payload[target] = payload[alias]
    return payload


def adapt_tool_for_chat_model(tool: BaseTool) -> BaseTool:
    async def _arun(**kwargs: Any) -> str:
        result = await tool.ainvoke(normalize_tool_args(tool.name, kwargs))
        return _stringify_tool_result(result)

    def _run(**kwargs: Any) -> str:
        result = tool.invoke(normalize_tool_args(tool.name, kwargs))
        return _stringify_tool_result(result)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        return_direct=False,
    )


class _DatasetPathAliasArgs(BaseModel):
    dataset_path: str = Field(default='', description='数据集根目录或图片目录')
    path: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    img_dir: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    label_dir: str = Field(default='', description='可选标签目录')
    include_duplicates: bool = Field(default=False, description='是否包含重复图片检测')
    max_duplicate_groups: int = Field(default=5, description='最多返回多少个重复组样例')
    method: str = Field(default='md5', description='重复检测方法')
    report_path: str = Field(default='', description='可选报告输出路径')


class _PrepareAliasArgs(BaseModel):
    dataset_path: str = Field(default='', description='数据集根目录或图片目录')
    path: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    img_dir: str = Field(default='', description='旧参数兼容；等价于 dataset_path')
    label_dir: str = Field(default='', description='可选标签目录')
    force_split: bool = Field(default=False, description='是否按默认比例强制划分')


class _PredictAliasArgs(BaseModel):
    source_path: str = Field(default='', description='图片文件路径或图片目录路径')
    path: str = Field(default='', description='旧参数兼容；等价于 source_path')
    source: str = Field(default='', description='旧参数兼容；等价于 source_path')
    input_path: str = Field(default='', description='旧参数兼容；等价于 source_path')
    dir_path: str = Field(default='', description='旧参数兼容；等价于 source_path')
    folder: str = Field(default='', description='旧参数兼容；等价于 source_path')
    model: str = Field(default='', description='预测模型路径或模型名')
    conf: float = Field(default=0.25, description='置信度阈值')
    iou: float = Field(default=0.45, description='NMS IoU 阈值')
    output_dir: str = Field(default='', description='输出目录')
    save_annotated: bool = Field(default=True, description='是否保存标注图')
    save_labels: bool = Field(default=False, description='是否保存 YOLO 标签')
    save_original: bool = Field(default=False, description='是否复制原图')
    generate_report: bool = Field(default=True, description='是否生成 JSON 报告')
    max_images: int = Field(default=0, description='最多处理图片数，0 表示不限制')

class _PredictSummaryAliasArgs(BaseModel):
    report_path: str = Field(default='', description='预测报告 JSON 路径')
    path: str = Field(default='', description='旧参数兼容；等价于 report_path')
    report: str = Field(default='', description='旧参数兼容；等价于 report_path')
    json_report: str = Field(default='', description='旧参数兼容；等价于 report_path')
    file: str = Field(default='', description='旧参数兼容；等价于 report_path')
    output_dir: str = Field(default='', description='预测输出目录；若未显式给 report_path，则会尝试读取 output_dir/prediction_report.json')
    dir_path: str = Field(default='', description='旧参数兼容；等价于 output_dir')
    folder: str = Field(default='', description='旧参数兼容；等价于 output_dir')

def _build_alias_tool(alias_name: str, target_tool: BaseTool, *, description: str, args_schema: type[BaseModel]) -> BaseTool:
    async def _arun(**kwargs: Any) -> str:
        result = await target_tool.ainvoke(normalize_tool_args(alias_name, kwargs))
        return _stringify_tool_result(result)

    def _run(**kwargs: Any) -> str:
        result = target_tool.invoke(normalize_tool_args(alias_name, kwargs))
        return _stringify_tool_result(result)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=alias_name,
        description=description,
        args_schema=args_schema,
        return_direct=False,
    )


def adapt_tools_for_chat_model(tools: list[BaseTool]) -> list[BaseTool]:
    adapted = [adapt_tool_for_chat_model(tool) for tool in tools]
    tool_map = {tool.name: tool for tool in tools}
    alias_tools: list[BaseTool] = []

    if 'detect_duplicate_images' in tool_map:
        alias_tools.append(
            _build_alias_tool(
                'detect_duplicates',
                tool_map['detect_duplicate_images'],
                description='兼容旧工具名 detect_duplicates。用于检测数据集中的重复图片；优先传 dataset_path，path 也可兼容。',
                args_schema=_DatasetPathAliasArgs,
            )
        )
    if 'run_dataset_health_check' in tool_map:
        alias_tools.append(
            _build_alias_tool(
                'detect_corrupted_images',
                tool_map['run_dataset_health_check'],
                description='兼容旧工具名 detect_corrupted_images。用于检查图片损坏、格式异常、尺寸异常；如需重复图片，也可设置 include_duplicates=true。',
                args_schema=_DatasetPathAliasArgs,
            )
        )
    if 'prepare_dataset_for_training' in tool_map:
        alias_tools.append(
            _build_alias_tool(
                'prepare_dataset',
                tool_map['prepare_dataset_for_training'],
                description='兼容旧工具名 prepare_dataset。用于把数据集准备到可训练状态。',
                args_schema=_PrepareAliasArgs,
            )
        )
        alias_tools.append(
            _build_alias_tool(
                'dataset_manager.prepare_dataset',
                tool_map['prepare_dataset_for_training'],
                description='兼容旧桌面风格工具名 dataset_manager.prepare_dataset。用于把数据集准备到可训练状态。',
                args_schema=_PrepareAliasArgs,
            )
        )
    if 'predict_images' in tool_map:
        for alias_name, description in (
            ('predict_directory', '兼容旧工具名 predict_directory。用于对图片目录做批量预测。'),
            ('batch_predict_images', '兼容旧工具名 batch_predict_images。用于对图片目录做批量预测。'),
            ('predict_images_in_dir', '兼容旧工具名 predict_images_in_dir。用于对图片目录做批量预测。'),
        ):
            alias_tools.append(
                _build_alias_tool(
                    alias_name,
                    tool_map['predict_images'],
                    description=description,
                    args_schema=_PredictAliasArgs,
                )
            )
    if 'predict_videos' in tool_map:
        for alias_name, description in (
            ('predict_video_directory', '兼容旧工具名 predict_video_directory。用于对视频目录做批量预测。'),
            ('batch_predict_videos', '兼容旧工具名 batch_predict_videos。用于对视频目录做批量预测。'),
            ('predict_videos_in_dir', '兼容旧工具名 predict_videos_in_dir。用于对视频目录做批量预测。'),
        ):
            alias_tools.append(
                _build_alias_tool(
                    alias_name,
                    tool_map['predict_videos'],
                    description=description,
                    args_schema=_PredictAliasArgs,
                )
            )
    if 'summarize_prediction_results' in tool_map:
        for alias_name, description in (
            ('summarize_predictions', '兼容旧工具名 summarize_predictions。用于汇总 prediction_report.json。'),
            ('summarize_prediction_report', '兼容旧工具名 summarize_prediction_report。用于汇总 prediction_report.json。'),
            ('analyze_prediction_report', '兼容旧工具名 analyze_prediction_report。用于汇总 prediction_report.json。'),
        ):
            alias_tools.append(
                _build_alias_tool(
                    alias_name,
                    tool_map['summarize_prediction_results'],
                    description=description,
                    args_schema=_PredictSummaryAliasArgs,
                )
            )

    return adapted + alias_tools

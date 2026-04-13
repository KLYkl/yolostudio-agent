from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

REPORT_NAMES = ('prediction_report.json', 'video_prediction_report.json')
MAX_SAMPLE_ITEMS = 5


def _missing() -> dict[str, Any]:
    return {
        'ok': False,
        'error': '未找到 prediction_report.json 或 video_prediction_report.json',
        'summary': '预测结果处理失败：找不到报告文件',
        'next_actions': ['请提供 report_path，或传入包含 prediction_report.json / video_prediction_report.json 的 output_dir'],
    }


def _resolve_report(report_path: str = '', output_dir: str = '') -> Path | None:
    if str(report_path).strip():
        p = Path(report_path)
        if p.is_dir():
            for name in REPORT_NAMES:
                candidate = p / name
                if candidate.exists():
                    return candidate
        return p if p.exists() else None
    if not str(output_dir).strip():
        return None
    base = Path(output_dir)
    if base.is_file() and base.exists():
        return base
    for name in REPORT_NAMES:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def _load(report_path: str = '', output_dir: str = '') -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_report(report_path, output_dir)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError
    payload = json.loads(resolved.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('预测报告 JSON 顶层必须是对象')
    return resolved, payload


def _mode(payload: dict[str, Any], resolved: Path) -> str:
    if payload.get('video_results') is not None or resolved.name == 'video_prediction_report.json':
        return 'videos'
    if any(key in payload for key in ('processed_videos', 'detected_frames', 'total_frames')):
        return 'videos'
    return 'images'


def _out_dir(payload: dict[str, Any], resolved: Path) -> Path:
    raw = str(payload.get('output_dir') or '').strip()
    return Path(raw) if raw else resolved.parent


def _summary_image(payload: dict[str, Any], resolved: Path) -> dict[str, Any]:
    items = payload.get('image_results') or []
    class_counts = dict(sorted((payload.get('class_counts') or {}).items(), key=lambda item: (-item[1], item[0])))
    processed = int(payload.get('processed_images') or len(items))
    detected = int(payload.get('detected_images') or sum(1 for item in items if int(item.get('detections') or 0) > 0))
    empty = int(payload.get('empty_images') or max(processed - detected, 0))
    total = sum(int(item.get('detections') or 0) for item in items)
    detected_samples = [str(item.get('image_path')) for item in items if int(item.get('detections') or 0) > 0][:3]
    empty_samples = [str(item.get('image_path')) for item in items if int(item.get('detections') or 0) <= 0][:3]
    warnings: list[str] = []
    failed = payload.get('failed_reads') or []
    if failed:
        warnings.append(f'有 {len(failed)} 张图片读取失败')
    if processed and empty / max(processed, 1) >= 0.8:
        warnings.append('无检测图片占比较高，建议复核模型或调低 conf 后复测')
    if not class_counts:
        warnings.append('当前报告里没有有效类别统计')
    top = [f'{k}={v}' for k, v in list(class_counts.items())[:4]]
    summary = f'预测结果摘要: 已处理 {processed} 张图片, 有检测 {detected}, 无检测 {empty}, 总检测框 {total}'
    if top:
        summary += f'，主要类别 {", ".join(top)}'
    next_actions = []
    if payload.get('annotated_dir'):
        next_actions.append(f'可查看标注结果目录: {payload.get("annotated_dir")}')
    if payload.get('labels_dir'):
        next_actions.append(f'可复用 YOLO 标签目录: {payload.get("labels_dir")}')
    next_actions.append(f'可查看预测报告: {resolved.resolve()}')
    return {
        'ok': True, 'summary': summary, 'mode': 'images', 'report_path': str(resolved.resolve()), 'output_dir': str(_out_dir(payload, resolved).resolve()),
        'annotated_dir': str(payload.get('annotated_dir') or ''), 'labels_dir': str(payload.get('labels_dir') or ''), 'originals_dir': str(payload.get('originals_dir') or ''),
        'model': str(payload.get('model') or ''), 'source_path': str(payload.get('source_path') or ''), 'processed_images': processed, 'detected_images': detected, 'empty_images': empty,
        'failed_reads': failed, 'class_counts': class_counts, 'total_detections': total, 'detected_samples': detected_samples, 'empty_samples': empty_samples,
        'warnings': warnings, 'next_actions': next_actions,
    }


def summarize_prediction_report(*, report_path: str = '', output_dir: str = '') -> dict[str, Any]:
    try:
        resolved, payload = _load(report_path, output_dir)
    except FileNotFoundError:
        return _missing()
    except Exception as exc:
        return {
            'ok': False, 'error': f'解析预测报告失败: {exc}', 'error_type': exc.__class__.__name__,
            'summary': '预测结果汇总失败：报告文件不可解析',
            'next_actions': ['请确认 prediction_report.json / video_prediction_report.json 是否为有效 UTF-8 JSON 文件'],
        }
    if _mode(payload, resolved) == 'videos':
        return _summary_video(payload, resolved)
    return _summary_image(payload, resolved)
def _summary_video(payload: dict[str, Any], resolved: Path) -> dict[str, Any]:
    items = payload.get('video_results') or []
    class_counts = dict(sorted((payload.get('class_counts') or {}).items(), key=lambda item: (-item[1], item[0])))
    processed = int(payload.get('processed_videos') or len(items))
    total_frames = int(payload.get('total_frames') or sum(int(item.get('processed_frames') or 0) for item in items))
    detected_frames = int(payload.get('detected_frames') or sum(int(item.get('detected_frames') or 0) for item in items))
    total = int(payload.get('total_detections') or sum(int(item.get('total_detections') or 0) for item in items))
    detected_samples = [str(item.get('video_path')) for item in items if int(item.get('detected_frames') or 0) > 0][:3]
    empty_samples = [str(item.get('video_path')) for item in items if int(item.get('detected_frames') or 0) <= 0][:3]
    warnings: list[str] = []
    failed = payload.get('failed_videos') or []
    if failed:
        warnings.append(f'有 {len(failed)} 个视频读取或处理失败')
    if processed and detected_frames <= 0:
        warnings.append('当前报告里没有检测到任何目标帧')
    if not class_counts:
        warnings.append('当前报告里没有有效类别统计')
    top = [f'{k}={v}' for k, v in list(class_counts.items())[:4]]
    summary = f'视频预测结果摘要: 已处理 {processed} 个视频, 总帧数 {total_frames}, 有检测帧 {detected_frames}, 总检测框 {total}'
    if top:
        summary += f'，主要类别 {", ".join(top)}'
    return {
        'ok': True, 'summary': summary, 'mode': 'videos', 'report_path': str(resolved.resolve()), 'output_dir': str(_out_dir(payload, resolved).resolve()),
        'model': str(payload.get('model') or ''), 'source_path': str(payload.get('source_path') or ''), 'processed_videos': processed,
        'total_frames': total_frames, 'detected_frames': detected_frames, 'total_detections': total, 'failed_reads': failed,
        'class_counts': class_counts, 'detected_samples': detected_samples, 'empty_samples': empty_samples,
        'warnings': warnings, 'next_actions': [f'可查看预测报告: {resolved.resolve()}'],
    }


def _artifact_roots(payload: dict[str, Any], mode: str, out_dir: Path) -> list[str]:
    roots = [str(out_dir.resolve())]
    if mode == 'videos':
        for item in payload.get('video_results') or []:
            for key in ('output_dir', 'annotated_video', 'annotated_keyframes_dir', 'raw_keyframes_dir'):
                value = str(item.get(key) or '').strip()
                if value and value not in roots:
                    roots.append(value)
    else:
        for key in ('annotated_dir', 'labels_dir', 'originals_dir'):
            value = str(payload.get(key) or '').strip()
            if value and value not in roots:
                roots.append(value)
    return roots[:MAX_SAMPLE_ITEMS + 3]


def _existing_path_lists(out_dir: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for base in (out_dir, out_dir / 'path_lists'):
        for name in ('detected_items.txt', 'empty_items.txt', 'failed_items.txt'):
            candidate = base / name
            if candidate.exists():
                found[name] = str(candidate.resolve())
    return found


def inspect_prediction_outputs(*, report_path: str = '', output_dir: str = '') -> dict[str, Any]:
    try:
        resolved, payload = _load(report_path, output_dir)
    except FileNotFoundError:
        return _missing()
    except Exception as exc:
        return {
            'ok': False, 'error': f'读取预测输出失败: {exc}', 'error_type': exc.__class__.__name__,
            'summary': '预测输出检查失败：无法读取预测报告', 'next_actions': ['请确认 report_path / output_dir 是否正确，且报告文件可正常解析'],
        }
    mode = _mode(payload, resolved)
    out_dir = _out_dir(payload, resolved)
    roots = _artifact_roots(payload, mode, out_dir)
    lists = _existing_path_lists(out_dir)
    result = summarize_prediction_report(report_path=str(resolved))
    if not result.get('ok'):
        return result
    summary = f'预测输出检查完成: 输出目录 {out_dir.resolve()}，已识别 {len(roots)} 个产物根路径'
    result.update({
        'summary': summary, 'artifact_roots': roots, 'artifact_root_count': len(roots), 'path_list_files': lists, 'path_list_count': len(lists),
        'warnings': list(result.get('warnings') or []) + ([] if lists else ['当前还没有导出的路径清单，可按需调用 export_prediction_path_lists']),
        'next_actions': ['如需导出可读报告，可调用 export_prediction_report', '如需导出命中/空结果路径清单，可调用 export_prediction_path_lists', '如需把命中结果单独整理出来，可调用 organize_prediction_results'],
    })
    return result


def _normalize_export_format(value: str) -> str:
    mapping = {'md': 'markdown', 'markdown': 'markdown', 'txt': 'text', 'text': 'text', 'json': 'json'}
    key = str(value or 'markdown').strip().lower()
    if key not in mapping:
        raise ValueError(f'不支持的 export_format: {value}')
    return mapping[key]


def _render_export(summary: dict[str, Any], fmt: str) -> str:
    mode_label = '视频' if summary.get('mode') == 'videos' else '图片'
    bullet = '- ' if fmt == 'markdown' else '  - '
    lines = [f'# {mode_label}预测结果报告' if fmt == 'markdown' else f'{mode_label}预测结果报告', '']
    lines += [f'{bullet}摘要: {summary.get("summary")}', f'{bullet}输出目录: {summary.get("output_dir")}', f'{bullet}报告路径: {summary.get("report_path")}']
    stats = [('总检测框', summary.get('total_detections', 0))]
    if summary.get('mode') == 'videos':
        stats = [('处理视频数', summary.get('processed_videos', 0)), ('总帧数', summary.get('total_frames', 0)), ('有检测帧', summary.get('detected_frames', 0))] + stats
    else:
        stats = [('处理图片数', summary.get('processed_images', 0)), ('有检测图片', summary.get('detected_images', 0)), ('无检测图片', summary.get('empty_images', 0))] + stats
    lines += ['', '## 统计' if fmt == 'markdown' else '统计:']
    lines += [f'{bullet}{k}: {v}' for k, v in stats]
    for title, values in (('类别统计', [f'{k}: {v}' for k, v in list((summary.get('class_counts') or {}).items())[:10]]), ('提示', list(summary.get('warnings') or [])[:6]), ('有检测样例', list(summary.get('detected_samples') or [])[:5]), ('无检测样例', list(summary.get('empty_samples') or [])[:5])):
        if values:
            lines += ['', f'## {title}' if fmt == 'markdown' else f'{title}:']
            lines += [f'{bullet}{item}' for item in values]
    return '\n'.join(lines).strip() + '\n'


def export_prediction_report(*, report_path: str = '', output_dir: str = '', export_path: str = '', export_format: str = 'markdown') -> dict[str, Any]:
    summary = summarize_prediction_report(report_path=report_path, output_dir=output_dir)
    if not summary.get('ok'):
        return summary
    fmt = _normalize_export_format(export_format)
    out_dir = Path(summary.get('output_dir') or Path(summary['report_path']).parent)
    target = Path(export_path).resolve() if str(export_path).strip() else (out_dir / f'prediction_summary.{"md" if fmt == "markdown" else ("txt" if fmt == "text" else "json")}').resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'json':
        target.write_text(json.dumps({k: v for k, v in summary.items() if k not in {'ok', 'next_actions'}}, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        target.write_text(_render_export(summary, fmt), encoding='utf-8')
    return {
        'ok': True, 'summary': f'预测报告导出完成: 已写出 {fmt} 报告', 'mode': summary.get('mode'), 'report_path': summary.get('report_path'),
        'output_dir': summary.get('output_dir'), 'export_path': str(target.resolve()), 'export_format': fmt, 'warnings': summary.get('warnings') or [],
        'next_actions': [f'可直接查看导出报告: {target.resolve()}', '如需导出路径清单，可继续调用 export_prediction_path_lists'],
    }
def _write_list(path: Path, items: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n'.join(item for item in items if item).strip() + ('\n' if items else '')), encoding='utf-8')
    return str(path.resolve())


def export_prediction_path_lists(*, report_path: str = '', output_dir: str = '', export_dir: str = '') -> dict[str, Any]:
    try:
        resolved, payload = _load(report_path, output_dir)
    except FileNotFoundError:
        return _missing()
    except Exception as exc:
        return {
            'ok': False, 'error': f'导出路径清单失败: {exc}', 'error_type': exc.__class__.__name__,
            'summary': '导出路径清单失败：无法读取预测报告', 'next_actions': ['请确认 report_path / output_dir 是否正确，且报告文件可正常解析'],
        }
    mode = _mode(payload, resolved)
    out_dir = _out_dir(payload, resolved)
    target_dir = Path(export_dir).resolve() if str(export_dir).strip() else (out_dir / 'path_lists')
    if mode == 'videos':
        items = payload.get('video_results') or []
        detected = [str(item.get('video_path') or '') for item in items if int(item.get('detected_frames') or 0) > 0]
        empty = [str(item.get('video_path') or '') for item in items if int(item.get('detected_frames') or 0) <= 0]
        failed = [str(item.get('path') or '') for item in (payload.get('failed_videos') or []) if str(item.get('path') or '').strip()]
    else:
        items = payload.get('image_results') or []
        detected = [str(item.get('image_path') or '') for item in items if int(item.get('detections') or 0) > 0]
        empty = [str(item.get('image_path') or '') for item in items if int(item.get('detections') or 0) <= 0]
        failed = [str(item.get('path') or '') for item in (payload.get('failed_reads') or []) if str(item.get('path') or '').strip()]
    return {
        'ok': True, 'summary': f'预测路径清单导出完成: 命中 {len(detected)} 条 / 无命中 {len(empty)} 条 / 失败 {len(failed)} 条', 'mode': mode,
        'report_path': str(resolved.resolve()), 'output_dir': str(out_dir.resolve()), 'export_dir': str(target_dir.resolve()),
        'detected_count': len(detected), 'empty_count': len(empty), 'failed_count': len(failed),
        'detected_items_path': _write_list(target_dir / 'detected_items.txt', detected),
        'empty_items_path': _write_list(target_dir / 'empty_items.txt', empty),
        'failed_items_path': _write_list(target_dir / 'failed_items.txt', failed),
        'next_actions': [f'可直接查看路径清单目录: {target_dir.resolve()}', '如需把命中结果复制整理到新目录，可继续调用 organize_prediction_results'],
    }


def _normalize_organize_mode(value: str) -> str:
    mapping = {'detected': 'detected_only', 'detected_only': 'detected_only', 'hits_only': 'detected_only', 'by_class': 'by_class', 'class': 'by_class', 'classes': 'by_class'}
    key = str(value or 'detected_only').strip().lower()
    if key not in mapping:
        raise ValueError(f'不支持的 organize_by: {value}')
    return mapping[key]


def _normalize_preference(value: str, mode: str) -> str:
    mapping = {'auto': 'auto', 'annotated': 'annotated', 'original': 'original', 'source': 'source'} if mode == 'images' else {'auto': 'auto', 'video_dir': 'video_dir', 'dir': 'video_dir', 'annotated_video': 'annotated_video', 'video': 'annotated_video'}
    key = str(value or 'auto').strip().lower()
    if key not in mapping:
        raise ValueError(f'不支持的 artifact_preference: {value}')
    return mapping[key]


def _safe_name(value: str) -> str:
    return (re.sub(r'[\\/:*?"<>|\s]+', '_', str(value or '').strip()).strip('._') or 'unknown')


def _copy_file(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / src.name
    if not candidate.exists():
        shutil.copy2(src, candidate)
        return candidate
    stem, suffix, index = src.stem, src.suffix, 2
    while True:
        candidate = dest_dir / f'{stem}_{index}{suffix}'
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        index += 1


def _copy_dir(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / src.name
    if not candidate.exists():
        shutil.copytree(src, candidate)
        return candidate
    index = 2
    while True:
        candidate = dest_dir / f'{src.name}_{index}'
        if not candidate.exists():
            shutil.copytree(src, candidate)
            return candidate
        index += 1


def organize_prediction_results(*, report_path: str = '', output_dir: str = '', destination_dir: str = '', organize_by: str = 'detected_only', include_empty: bool = False, artifact_preference: str = 'auto') -> dict[str, Any]:
    try:
        resolved, payload = _load(report_path, output_dir)
    except FileNotFoundError:
        return _missing()
    except Exception as exc:
        return {
            'ok': False, 'error': f'整理预测结果失败: {exc}', 'error_type': exc.__class__.__name__,
            'summary': '预测结果整理失败：无法读取预测报告', 'next_actions': ['请确认 report_path / output_dir 是否正确，且报告文件可正常解析'],
        }
    mode = _mode(payload, resolved)
    organize_mode = _normalize_organize_mode(organize_by)
    preference = _normalize_preference(artifact_preference, mode)
    out_dir = _out_dir(payload, resolved)
    target_dir = Path(destination_dir).resolve() if str(destination_dir).strip() else (out_dir / f'organized_{"detected_only" if organize_mode == "detected_only" else "by_class"}')
    target_dir.mkdir(parents=True, exist_ok=True)
    bucket_stats: dict[str, int] = {}
    sample_outputs: list[str] = []
    copied = 0
    if mode == 'images':
        for item in payload.get('image_results') or []:
            detections = int(item.get('detections') or 0)
            classes = [str(v).strip() for v in (item.get('classes') or []) if str(v).strip()]
            if organize_mode == 'detected_only':
                bucket = 'detected' if detections > 0 else ('empty' if include_empty else '')
            else:
                bucket = classes[0] if len(classes) == 1 else ('_mixed' if classes else ('_empty' if include_empty else ''))
            if not bucket:
                continue
            artifacts = item.get('artifact_paths') or {}
            src = None
            for key in {'auto': ('annotated', 'original_copy'), 'annotated': ('annotated',), 'original': ('original_copy',), 'source': ()}[preference]:
                value = str(artifacts.get(key) or '').strip()
                if value:
                    src = Path(value)
                    break
            if src is None:
                raw = str(item.get('image_path') or '').strip()
                src = Path(raw) if raw else None
            if src is None or not src.exists():
                continue
            bucket_root = target_dir / _safe_name(bucket)
            copied_path = _copy_file(src, bucket_root / 'images')
            label = str(artifacts.get('label_yolo') or '').strip()
            if label and Path(label).exists():
                _copy_file(Path(label), bucket_root / 'labels_yolo')
            copied += 1
            bucket_stats[bucket] = bucket_stats.get(bucket, 0) + 1
            if len(sample_outputs) < MAX_SAMPLE_ITEMS:
                sample_outputs.append(str(copied_path.resolve()))
    else:
        for item in payload.get('video_results') or []:
            detected_frames = int(item.get('detected_frames') or 0)
            classes = sorted(str(k).strip() for k in (item.get('class_counts') or {}).keys() if str(k).strip())
            if organize_mode == 'detected_only':
                bucket = 'detected' if detected_frames > 0 else ('empty' if include_empty else '')
            else:
                bucket = classes[0] if len(classes) == 1 else ('_mixed' if classes else ('_empty' if include_empty else ''))
            if not bucket:
                continue
            bucket_root = target_dir / _safe_name(bucket)
            copied_path = None
            if preference == 'annotated_video':
                video_file = str(item.get('annotated_video') or '').strip()
                if video_file and Path(video_file).exists():
                    copied_path = _copy_file(Path(video_file), bucket_root / 'videos')
            if copied_path is None:
                video_dir = str(item.get('output_dir') or '').strip()
                if video_dir and Path(video_dir).exists():
                    copied_path = _copy_dir(Path(video_dir), bucket_root / 'video_runs')
            if copied_path is None:
                continue
            copied += 1
            bucket_stats[bucket] = bucket_stats.get(bucket, 0) + 1
            if len(sample_outputs) < MAX_SAMPLE_ITEMS:
                sample_outputs.append(str(copied_path.resolve()))
    if copied <= 0:
        return {
            'ok': True, 'summary': '当前没有符合条件的预测结果需要整理', 'mode': mode, 'source_report_path': str(resolved.resolve()), 'source_output_dir': str(out_dir.resolve()),
            'destination_dir': str(target_dir.resolve()), 'organize_by': organize_mode, 'artifact_preference': preference, 'copied_items': 0, 'bucket_stats': {},
            'warnings': ['当前筛选条件下没有可复制的命中结果'], 'next_actions': ['可改成 include_empty=true，或换成 by_class 再执行一次'],
        }
    return {
        'ok': True, 'summary': f'预测结果整理完成: 已复制 {copied} 个产物到 {len(bucket_stats)} 个目录桶', 'mode': mode, 'source_report_path': str(resolved.resolve()), 'source_output_dir': str(out_dir.resolve()),
        'destination_dir': str(target_dir.resolve()), 'organize_by': organize_mode, 'artifact_preference': preference, 'include_empty': include_empty,
        'copied_items': copied, 'bucket_stats': dict(sorted(bucket_stats.items())), 'sample_outputs': sample_outputs,
        'warnings': ['本操作只复制预测产物到新目录，不会改写原始预测输出'], 'next_actions': [f'可检查整理后的目录: {target_dir.resolve()}', '如需导出路径清单，可继续调用 export_prediction_path_lists'],
    }
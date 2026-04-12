from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.grounded_reply_builder import build_grounded_tool_reply


def main() -> None:
    health_text = build_grounded_tool_reply(
        [(
            'run_dataset_health_check',
            {
                'ok': True,
                'summary': '健康检查完成: 完整性问题 5, 异常尺寸 0, 重复组 83',
                'warnings': ['发现 5 个文件扩展名与真实格式不匹配', '发现 83 组重复图片（额外重复文件 83 个）'],
                'integrity': {'corrupted_count': 0, 'zero_bytes_count': 0, 'format_mismatch_count': 5},
                'size_stats': {'abnormal_small_count': 0, 'abnormal_large_count': 0},
                'duplicate_groups': 83,
                'duplicate_extra_files': 83,
                'next_actions': ['建议先处理损坏/异常图片，再继续数据准备或训练'],
            },
        )],
    )
    assert '健康检查完成' in health_text
    assert '格式不匹配 5' in health_text
    assert '重复图片: 83 组' in health_text
    assert '建议先处理损坏/异常图片' in health_text

    readiness_text = build_grounded_tool_reply(
        [(
            'training_readiness',
            {
                'ok': True,
                'summary': '可以训练，但存在数据质量风险',
                'warnings': ['发现 5179 张图片缺少标签（占比 73.7%），训练结果可能受到明显影响'],
                'resolved_data_yaml': '/data/dirty/data.yaml',
                'auto_device': '1',
                'next_actions': ['建议先确认是否接受当前缺失标签风险，再决定是否直接训练'],
            },
        )],
    )
    assert '可以训练，但存在数据质量风险' in readiness_text
    assert '当前可用 YAML: /data/dirty/data.yaml' in readiness_text
    assert '当前 auto 设备策略会解析到: 1' in readiness_text

    prepare_text = build_grounded_tool_reply(
        [(
            'prepare_dataset_for_training',
            {
                'ok': True,
                'summary': '数据集已准备到可训练状态，但存在数据质量风险',
                'data_yaml': '/data/dirty/images_split/data.yaml',
                'warnings': ['发现 5179 张图片缺少标签（占比 73.7%），训练结果可能受到明显影响'],
                'next_actions': ['如要训练，可直接复用上面的 data_yaml 调用 start_training'],
            },
        )],
    )
    assert '数据集已准备到可训练状态' in prepare_text
    assert '已准备好的 YAML: /data/dirty/images_split/data.yaml' in prepare_text
    assert '如要训练，可直接复用上面的 data_yaml' in prepare_text

    predict_text = build_grounded_tool_reply(
        [(
            'predict_images',
            {
                'ok': True,
                'summary': '预测完成: 已处理 3 张图片, 有检测 2, 无检测 1，主要类别 Excavator=1, bulldozer=2',
                'processed_images': 3,
                'detected_images': 2,
                'empty_images': 1,
                'class_counts': {'Excavator': 1, 'bulldozer': 2},
                'detected_samples': ['/data/predict/a.jpg', '/data/predict/c.jpg'],
                'empty_samples': ['/data/predict/b.jpg'],
                'annotated_dir': '/data/predict/out/annotated',
                'report_path': '/data/predict/out/prediction_report.json',
                'next_actions': ['可查看标注结果目录: /data/predict/out/annotated'],
            },
        )],
    )
    assert '预测完成' in predict_text
    assert '已处理 3 张' in predict_text
    assert '主要类别: Excavator=1，bulldozer=2' in predict_text
    assert '/data/predict/a.jpg' in predict_text
    assert '标注结果目录: /data/predict/out/annotated' in predict_text
    assert '预测报告: /data/predict/out/prediction_report.json' in predict_text

    predict_video_text = build_grounded_tool_reply(
        [(
            'predict_videos',
            {
                'ok': True,
                'summary': '视频预测完成: 已处理 1 个视频, 总帧数 3, 有检测帧 2, 总检测框 3，主要类别 Excavator=1, bulldozer=2',
                'processed_videos': 1,
                'total_frames': 3,
                'detected_frames': 2,
                'total_detections': 3,
                'class_counts': {'Excavator': 1, 'bulldozer': 2},
                'detected_samples': ['/data/videos/a.mp4'],
                'empty_samples': [],
                'output_dir': '/data/videos/out',
                'report_path': '/data/videos/out/video_prediction_report.json',
                'next_actions': ['可查看视频预测输出目录: /data/videos/out'],
            },
        )],
    )
    assert '视频预测完成' in predict_video_text
    assert '总帧数 3' in predict_video_text
    assert '主要类别: Excavator=1，bulldozer=2' in predict_video_text
    assert '视频预测输出目录: /data/videos/out' in predict_video_text

    summary_text = build_grounded_tool_reply(
        [(
            'summarize_prediction_results',
            {
                'ok': True,
                'summary': '预测结果摘要: 已处理 3 张图片, 有检测 2, 无检测 1, 总检测框 3，主要类别 Excavator=1, bulldozer=2',
                'processed_images': 3,
                'detected_images': 2,
                'empty_images': 1,
                'total_detections': 3,
                'class_counts': {'Excavator': 1, 'bulldozer': 2},
                'detected_samples': ['/data/predict/a.jpg', '/data/predict/c.jpg'],
                'empty_samples': ['/data/predict/b.jpg'],
                'annotated_dir': '/data/predict/out/annotated',
                'report_path': '/data/predict/out/prediction_report.json',
                'next_actions': ['可查看标注结果目录: /data/predict/out/annotated'],
            },
        )],
    )
    assert '预测结果摘要' in summary_text
    assert '总检测框 3' in summary_text
    assert '主要类别: Excavator=1，bulldozer=2' in summary_text
    assert '预测报告: /data/predict/out/prediction_report.json' in summary_text

    video_summary_text = build_grounded_tool_reply(
        [(
            'summarize_prediction_results',
            {
                'ok': True,
                'mode': 'videos',
                'summary': '视频预测结果摘要: 已处理 1 个视频, 总帧数 3, 有检测帧 2, 总检测框 3，主要类别 Excavator=1, bulldozer=2',
                'processed_videos': 1,
                'total_frames': 3,
                'detected_frames': 2,
                'total_detections': 3,
                'class_counts': {'Excavator': 1, 'bulldozer': 2},
                'detected_samples': ['/data/videos/a.mp4'],
                'empty_samples': [],
                'report_path': '/data/videos/out/video_prediction_report.json',
                'next_actions': ['可查看预测报告: /data/videos/out/video_prediction_report.json'],
            },
        )],
    )
    assert '视频预测结果摘要' in video_summary_text
    assert '有检测帧 2' in video_summary_text
    assert '预测报告: /data/videos/out/video_prediction_report.json' in video_summary_text

    dup_text = build_grounded_tool_reply(
        [(
            'detect_duplicate_images',
            {
                'ok': True,
                'summary': '检测完成: 发现 2 组重复图片，额外重复文件 2 个',
                'groups': [
                    {
                        'paths': ['/data/a.jpg', '/data/a_copy.jpg'],
                    },
                    {
                        'paths': ['/data/b.jpg', '/data/b_copy.jpg'],
                    },
                ],
                'next_actions': ['建议人工确认 sample groups 中的文件是否应合并或清理'],
            },
        )],
    )
    assert '检测完成: 发现 2 组重复图片' in dup_text
    assert '/data/a.jpg, /data/a_copy.jpg' in dup_text
    assert '建议人工确认' in dup_text

    print('grounded tool reply ok')


if __name__ == '__main__':
    main()

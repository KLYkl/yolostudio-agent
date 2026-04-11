from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.grounded_reply_builder import build_grounded_tool_reply


def main() -> None:
    text = build_grounded_tool_reply([('predict_videos', {
        'ok': True,
        'summary': '视频预测完成: 已处理 1 个视频, 总帧数 3, 有检测帧 2, 总检测框 3，主要类别 Excavator=1, bulldozer=2',
        'processed_videos': 1,
        'total_frames': 3,
        'detected_frames': 2,
        'total_detections': 3,
        'class_counts': {'Excavator': 1, 'bulldozer': 2},
        'detected_samples': ['/data/videos/a.mp4'],
        'output_dir': '/data/videos/out',
        'report_path': '/data/videos/out/video_prediction_report.json',
        'next_actions': ['可查看视频预测输出目录: /data/videos/out'],
    })])
    assert '视频预测完成' in text
    assert '总帧数 3' in text
    assert '主要类别: Excavator=1，bulldozer=2' in text
    assert '视频预测输出目录: /data/videos/out' in text

    knowledge = build_grounded_tool_reply([('recommend_next_training_step', {
        'ok': True,
        'summary': '下一步建议: 先修数据，再谈调参。',
        'recommended_action': 'fix_data_quality',
        'basis': ['缺失标签比例=0.35', '重复组=2'],
        'source_summary': {'official': 1, 'workflow': 1},
        'why': '当前更像数据问题。',
        'next_actions': ['先补标签', '清理重复图片'],
    })])
    assert '优先动作: fix_data_quality' in knowledge
    assert '原因: 当前更像数据问题。' in knowledge
    assert '来源: official=1，workflow=1' in knowledge

    empty = build_grounded_tool_reply([('predict_images', {'ok': False, 'error': 'boom'})])
    assert empty == ''
    print('grounded reply builder ok')


if __name__ == '__main__':
    main()

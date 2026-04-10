from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.agent_client import YoloStudioAgentClient


def main() -> None:
    client = object.__new__(YoloStudioAgentClient)

    health_text = YoloStudioAgentClient._build_grounded_tool_reply(
        client,
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

    readiness_text = YoloStudioAgentClient._build_grounded_tool_reply(
        client,
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

    prepare_text = YoloStudioAgentClient._build_grounded_tool_reply(
        client,
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

    dup_text = YoloStudioAgentClient._build_grounded_tool_reply(
        client,
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

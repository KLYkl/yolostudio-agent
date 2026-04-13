from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.grounded_reply_builder import build_grounded_tool_reply


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

    training = build_grounded_tool_reply([('summarize_training_run', {
        'ok': True,
        'summary': '训练结果汇总: 最近一次训练已完成，但当前只有部分可读事实，暂时不能下可靠结论。',
        'run_state': 'completed',
        'observation_stage': 'final',
        'analysis_ready': False,
        'minimum_facts_ready': True,
        'progress': {'epoch': 1, 'total_epochs': 10, 'progress_ratio': 0.1},
        'metrics': {'box_loss': 1.2, 'cls_loss': 0.7, 'dfl_loss': 0.4},
        'signals': ['loss_only_metrics', 'insufficient_eval_metrics'],
        'facts': ['仅有训练损失: box_loss=1.2, cls_loss=0.7, dfl_loss=0.4'],
        'next_actions': ['补齐评估指标后再判断'],
    })])
    assert '观察阶段: 最终状态' in training
    assert '训练进度: 1/10 (10%)' in training
    assert '当前不足: 缺少稳定评估指标' in training
    assert '当前仅有训练损失' in training

    readiness = build_grounded_tool_reply([('training_readiness', {
        'ok': True,
        'summary': '当前还不能直接训练: 缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
        'ready': False,
        'preparable': True,
        'primary_blocker_type': 'missing_yaml',
        'blockers': ['缺少可用的 data_yaml'],
        'next_actions': [
            {'description': '当前不能直接训练，但可以先调用 prepare_dataset_for_training 自动补齐 YAML 和划分产物', 'tool': 'prepare_dataset_for_training'},
        ],
    })])
    assert '当前可继续自动准备: prepare_dataset_for_training' in readiness
    assert '主要阻塞类型: missing_yaml' in readiness
    assert 'prepare_dataset_for_training' in readiness

    envs = build_grounded_tool_reply([('list_training_environments', {
        'ok': True,
        'summary': '发现 2 个可用训练环境，默认将使用 yolodo',
        'environments': [
            {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True},
            {'name': 'yolo', 'display_name': 'yolo', 'selected_by_default': False},
        ],
        'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
    })])
    assert '默认训练环境: yolodo' in envs
    assert '可用环境:' in envs

    preflight = build_grounded_tool_reply([('training_preflight', {
        'ok': True,
        'summary': '训练预检通过',
        'resolved_args': {
            'model': '/models/yolov8n.pt',
            'data_yaml': '/data/dataset.yaml',
            'project': '/runs/ablation',
            'name': 'exp-blue',
            'batch': 16,
            'imgsz': 960,
            'fraction': 0.5,
            'classes': [1, 3],
            'single_cls': False,
            'optimizer': 'AdamW',
            'freeze': 6,
            'resume': True,
            'lr0': 0.003,
            'patience': 12,
            'workers': 2,
            'amp': False,
        },
        'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
        'command_preview': ['yolo', 'train', 'model=/models/yolov8n.pt', 'data=/data/dataset.yaml', 'epochs=30', 'device=1', 'batch=16', 'imgsz=960', 'optimizer=AdamW', 'freeze=6', 'resume=True', 'lr0=0.003', 'patience=12', 'workers=2', 'amp=False'],
    })])
    assert '训练环境: yolodo' in preflight
    assert '输出组织: project=/runs/ablation, name=exp-blue' in preflight
    assert '批大小: 16' in preflight
    assert '输入尺寸: 960' in preflight
    assert '采样比例: 0.5' in preflight
    assert '高级参数: classes=[1, 3], single_cls=False, optimizer=AdamW, freeze=6, resume=True, lr0=0.003, patience=12, workers=2, amp=False' in preflight

    runs = build_grounded_tool_reply([('list_training_runs', {
        'ok': True,
        'summary': '找到 2 条最近训练记录',
        'applied_filters': {'run_state': 'failed', 'analysis_ready': True},
        'runs': [
            {
                'run_id': 'train_log_111',
                'run_state': 'stopped',
                'observation_stage': 'final',
                'progress': {'epoch': 2, 'total_epochs': 30},
            },
            {
                'run_id': 'train_log_222',
                'run_state': 'completed',
                'observation_stage': 'final',
                'progress': {'epoch': 30, 'total_epochs': 30},
            },
        ],
    })])
    assert '筛选: 状态=failed, 仅可分析训练' in runs
    assert '最近训练:' in runs
    assert 'train_log_111: stopped / 最终状态，进度 2/30' in runs

    inspect_run = build_grounded_tool_reply([('inspect_training_run', {
        'ok': True,
        'summary': '训练记录详情已就绪',
        'selected_run_id': 'train_log_111',
        'run_state': 'stopped',
        'observation_stage': 'final',
        'progress': {'epoch': 2, 'total_epochs': 30, 'progress_ratio': 2 / 30},
        'model': '/models/yolov8n.pt',
        'data_yaml': '/data/dataset.yaml',
        'log_file': '/runs/train_log_111.txt',
        'signals': ['manual_stop'],
        'facts': ['precision=0.731', 'recall=0.251'],
        'next_actions': ['可继续调用 analyze_training_outcome'],
    })])
    assert '训练记录: train_log_111' in inspect_run
    assert '观察阶段: 最终状态' in inspect_run
    assert '模型: /models/yolov8n.pt' in inspect_run
    assert '事实:' in inspect_run

    compare_runs = build_grounded_tool_reply([('compare_training_runs', {
        'ok': True,
        'summary': '训练对比完成: train_log_200 相比 train_log_100，precision提升 +0.1000；mAP50提升 +0.1000',
        'left_run_id': 'train_log_200',
        'right_run_id': 'train_log_100',
        'highlights': ['precision提升 +0.1000', 'mAP50提升 +0.1000'],
        'metric_deltas': {
            'precision': {'left': 0.52, 'right': 0.42, 'delta': 0.1},
            'map50': {'left': 0.465, 'right': 0.365, 'delta': 0.1},
        },
        'next_actions': ['可继续调用 inspect_training_run'],
    })])
    assert '对比对象: train_log_200 vs train_log_100' in compare_runs
    assert '主要变化:' in compare_runs
    assert '关键差异: precision=+0.1000，mAP50=+0.1000' in compare_runs
    print('grounded reply builder ok')


if __name__ == '__main__':
    main()

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from langchain_core.messages import SystemMessage

from yolostudio_agent.agent.client.session_state import SessionState


StructuredActionClassifier = Callable[..., Awaitable[str]]
TrainingContextChecker = Callable[[], bool]


async def classify_training_loop_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    loop_id: str,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    facts = {
        'active_loop_id': loop_id or session_state.active_training.active_loop_id or '',
        'last_loop_status_summary': str((session_state.active_training.last_loop_status or {}).get('summary') or '').strip(),
        'last_loop_detail_summary': str((session_state.active_training.last_loop_detail or {}).get('summary') or '').strip(),
        'has_last_loop_status': bool(session_state.active_training.last_loop_status),
        'has_last_loop_detail': bool(session_state.active_training.last_loop_detail),
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的循环训练跟进路由器。'
                '当前同一会话里已经存在活动中的循环训练上下文。'
                '你只负责判断用户这句跟进，应该查看当前环训练状态、查看当前环训练详情，还是不属于当前环训练上下文。'
                '如果用户想要更详细的进展、更多训练信息、轮次信息、完整详情、详细状态，返回 inspect。'
                '如果用户只是询问现在怎么样、进度如何、当前情况、现在什么情况、训练情况，返回 status。'
                '像“环训练状态怎么样”“查看训练情况”“现在是什么情况了”这类泛状态追问，默认返回 status；'
                '只有当用户明确要求“详细一点”“训练详情”“完整详情”“更多训练信息”时，才返回 inspect。'
                '如果用户是在说新的训练任务、准备数据、换数据集、换模型，返回 other。'
                '输出必须是 JSON，对象格式固定为 '
                '{"action":"status|inspect|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'status', 'inspect'},
    )


async def classify_training_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    metric_signals: list[str],
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    active_training = session_state.active_training
    knowledge = session_state.active_knowledge
    facts = {
        'running': active_training.running,
        'model': active_training.model,
        'data_yaml': active_training.data_yaml,
        'device': active_training.device,
        'training_environment': active_training.training_environment,
        'has_last_status': bool(active_training.last_status),
        'has_last_summary': bool(active_training.last_summary or active_training.training_run_summary),
        'has_last_analysis': bool(knowledge.last_analysis),
        'has_last_recommendation': bool(knowledge.last_recommendation),
        'has_last_retrieval': bool(knowledge.last_retrieval),
        'metric_signals': metric_signals,
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的训练跟进路由器。'
                '当前同一会话里已经存在 training 上下文。'
                '你只负责判断用户这句跟进，应该查看当前训练状态、查看训练结果分析、查看下一步训练建议、查看训练知识解释，还是不属于当前 training 上下文。'
                '如果用户是在问现在什么情况、训练进度、详细一点的训练信息、当前训练状态，返回 status。'
                '如果用户是在问训练效果如何、结果怎么看、这些指标说明什么、训练是不是收敛了，返回 analysis。'
                '如果用户是在问下一步怎么做、先补数据还是先调参数、怎么优化下一轮，返回 next_step。'
                '如果用户是在问术语含义、指标是什么意思、训练知识或工作流解释，返回 knowledge。'
                '如果用户是在发起新训练、切换到预测、数据集处理、远端传输、训练对比或查看特定 run，返回 other。'
                '输出必须是 JSON，对象格式固定为 {"action":"status|analysis|next_step|knowledge|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'status', 'analysis', 'next_step', 'knowledge'},
    )


async def classify_training_history_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    active_training = session_state.active_training
    best_run_selection = active_training.best_run_selection or {}
    best_run = best_run_selection.get('best_run') if isinstance(best_run_selection, dict) else {}
    facts = {
        'recent_run_ids': [
            str(item.get('run_id') or '')
            for item in list(active_training.recent_runs or [])[:5]
            if str(item.get('run_id') or '').strip()
        ],
        'has_recent_runs': bool(active_training.recent_runs),
        'has_last_run_inspection': bool(active_training.last_run_inspection),
        'last_inspection_run_id': active_training.last_run_inspection.get('selected_run_id'),
        'has_last_run_comparison': bool(active_training.last_run_comparison),
        'comparison_run_ids': [
            str(active_training.last_run_comparison.get('left_run_id') or ''),
            str(active_training.last_run_comparison.get('right_run_id') or ''),
        ],
        'has_best_run_selection': bool(best_run_selection),
        'best_run_id': str((best_run or {}).get('run_id') or best_run_selection.get('best_run_id') or ''),
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的训练历史跟进路由器。'
                '当前同一会话里已经存在 training history 上下文。'
                '你只负责判断用户这句跟进，是想继续查看训练列表、查看刚才那条训练详情、查看刚才的训练对比、查看最佳训练记录，还是不属于当前 training history 上下文。'
                '如果用户是在问刚才那些训练、最近训练、那批训练记录、历史列表、再概括一下列表，返回 runs。'
                '如果用户是在问刚才那条训练详细一点、那条记录怎么看、上一条 run 细节、训练记录详情，返回 inspect。'
                '如果用户是在问刚才两条训练差异、对比结论、哪条更好、比较结果，返回 compare。'
                '如果用户是在问最佳训练、最好的那条、表现最好的是哪个、最佳 run 详细一点，返回 best。'
                '如果用户是在发起新训练、查看当前训练状态、环训练控制、预测、数据集处理、远端传输或查看特定 run id，返回 other。'
                '输出必须是 JSON，对象格式固定为 {"action":"runs|inspect|compare|best|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'runs', 'inspect', 'compare', 'best'},
    )


async def classify_training_loop_history_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    active_training = session_state.active_training
    facts = {
        'recent_loop_ids': [
            str(item.get('loop_id') or item.get('loop_name') or '')
            for item in list(active_training.recent_loops or [])[:5]
            if str(item.get('loop_id') or item.get('loop_name') or '').strip()
        ],
        'has_recent_loops': bool(active_training.recent_loops),
        'has_last_loop_status': bool(active_training.last_loop_status),
        'last_loop_status_summary': str((active_training.last_loop_status or {}).get('summary') or '').strip(),
        'has_last_loop_detail': bool(active_training.last_loop_detail),
        'last_loop_detail_summary': str((active_training.last_loop_detail or {}).get('summary') or '').strip(),
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的环训练历史跟进路由器。'
                '当前同一会话里已经存在 training loop history 上下文。'
                '你只负责判断用户这句跟进，是想继续查看环训练列表、查看刚才那个环训练状态、查看刚才那个环训练详情，还是不属于当前环训练历史上下文。'
                '如果用户是在问刚才那些环训练、最近环训练、环训练列表、再概括一下环训练历史，返回 list。'
                '如果用户是在问刚才那个环训练现在怎么样、状态如何、当前结论、停在什么阶段，返回 status。'
                '如果用户是在问刚才那个环训练详细一点、轮次细节、完整详情、知识闸门细节，返回 inspect。'
                '如果用户是在发起新的环训练、当前活动环训练控制、新训练、预测、数据处理或远端传输，返回 other。'
                '输出必须是 JSON，对象格式固定为 {"action":"list|status|inspect|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'list', 'status', 'inspect'},
    )


async def classify_knowledge_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    metric_signals: list[str],
    has_training_state_context: TrainingContextChecker,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    knowledge = session_state.active_knowledge
    facts = {
        'has_last_retrieval': bool(knowledge.last_retrieval),
        'last_retrieval': knowledge.last_retrieval,
        'has_last_analysis': bool(knowledge.last_analysis),
        'last_analysis': knowledge.last_analysis,
        'has_last_recommendation': bool(knowledge.last_recommendation),
        'last_recommendation': knowledge.last_recommendation,
        'has_training_context': bool(has_training_state_context()),
        'metric_signals': metric_signals,
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的知识跟进路由器。'
                '当前同一会话里已经存在 training knowledge / analysis / recommendation 上下文。'
                '你只负责判断用户这句跟进，是想继续查看训练知识解释、继续查看训练结果分析、继续查看下一步训练建议，还是不属于当前知识上下文。'
                '如果用户是在追问规则、术语、这些指标是什么意思、刚才那条经验/知识/解释再详细一点，返回 knowledge。'
                '如果用户是在追问训练结果怎么看、为什么这样判断、分析再展开一点，返回 analysis。'
                '如果用户是在追问下一步该怎么做、建议再具体一点、怎么优化下一轮，返回 next_step。'
                '如果用户是在发起新训练、切到预测、数据集处理、远端传输、查看特定 run 或环训练控制，返回 other。'
                '输出必须是 JSON，对象格式固定为 {"action":"knowledge|analysis|next_step|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'knowledge', 'analysis', 'next_step'},
    )


async def classify_prediction_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    fallback_path: str,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    active_prediction = session_state.active_prediction
    facts = {
        'source_path': active_prediction.source_path or fallback_path,
        'report_path': active_prediction.report_path,
        'output_dir': active_prediction.output_dir,
        'model': active_prediction.model,
        'has_last_result': bool(active_prediction.last_result),
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的预测跟进路由器。'
                '当前同一会话里已经存在 prediction 上下文。'
                '你只负责判断用户这句跟进，应该查看预测摘要、查看预测输出详情，还是不属于当前 prediction 上下文。'
                '如果用户只是在问现在怎么样、预测情况、结果如何、总结一下，返回 summary。'
                '如果用户明确要求更详细的预测信息、输出详情、报告、产物、路径清单、更多细节，返回 inspect。'
                '如果用户是在发起新预测、换模型、换路径、切到训练或数据准备，返回 other。'
                '输出必须是 JSON，对象格式固定为 {"action":"summary|inspect|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'summary', 'inspect'},
    )


async def classify_realtime_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    active_prediction = session_state.active_prediction
    facts = {
        'realtime_session_id': active_prediction.realtime_session_id,
        'realtime_source_type': active_prediction.realtime_source_type,
        'realtime_source_label': active_prediction.realtime_source_label,
        'realtime_status': active_prediction.realtime_status,
        'output_dir': active_prediction.output_dir,
        'report_path': active_prediction.report_path,
        'has_last_realtime_status': bool(active_prediction.last_realtime_status),
        'last_realtime_status': active_prediction.last_realtime_status,
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的实时预测跟进路由器。'
                '当前同一会话里已经存在 realtime prediction 上下文。'
                '你只负责判断用户这句跟进，是否应该查看当前实时预测状态。'
                '如果用户是在问现在怎么样、还在跑吗、实时预测情况、处理了多少帧、详细一点的实时信息、当前进度或当前结果，返回 status。'
                '如果用户是在发起新的摄像头/RTSP/屏幕预测、测试 RTSP、扫描摄像头/屏幕、切到训练或其他任务，返回 other。'
                '输出必须是 JSON，对象格式固定为 {"action":"status|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'status'},
    )


async def classify_prediction_management_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    active_prediction = session_state.active_prediction
    facts = {
        'has_last_inspection': bool(active_prediction.last_inspection),
        'has_last_export': bool(active_prediction.last_export),
        'has_last_path_lists': bool(active_prediction.last_path_lists),
        'has_last_organized_result': bool(active_prediction.last_organized_result),
        'last_inspection': active_prediction.last_inspection,
        'last_export': active_prediction.last_export,
        'last_path_lists': active_prediction.last_path_lists,
        'last_organized_result': active_prediction.last_organized_result,
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的预测结果管理跟进路由器。'
                '当前同一会话里已经存在 prediction management 上下文。'
                '你只负责判断用户这句跟进，应该查看预测输出检查结果、查看预测报告导出结果、查看预测路径清单结果、查看预测结果整理结果，还是不属于当前 prediction management 上下文。'
                '如果用户是在追问输出目录、产物目录、产物路径、结果里有什么、保存到了哪里，返回 inspect。'
                '如果用户是在追问导出的报告、导出的文件、报告路径、markdown/csv 报告，返回 export。'
                '如果用户是在追问刚才导出的清单、命中清单、空结果清单、失败清单、列表详情，返回 path_lists。'
                '如果用户是在追问整理后的结果、按类别后的目录、复制到了哪里、整理详情，返回 organize。'
                '如果用户只是泛泛追问“再详细一点/现在什么情况/那个结果呢”，优先使用当前上下文里最近更具体的结果：organize > path_lists > export > inspect。'
                '如果用户是在发起新的预测、换模型、换数据路径、切到训练、抽帧、远端传输，返回 other。'
                '输出必须是 JSON，对象格式固定为 {"action":"inspect|export|path_lists|organize|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'inspect', 'export', 'path_lists', 'organize'},
    )


async def classify_dataset_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    fallback_path: str,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    active_dataset = session_state.active_dataset
    facts = {
        'dataset_root': active_dataset.dataset_root or fallback_path,
        'img_dir': active_dataset.img_dir,
        'data_yaml': active_dataset.data_yaml,
        'has_scan': bool(active_dataset.last_scan),
        'has_validate': bool(active_dataset.last_validate),
        'has_health_check': bool(active_dataset.last_health_check),
        'has_duplicate_check': bool(active_dataset.last_duplicate_check),
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的数据集跟进路由器。'
                '当前同一会话里已经存在 dataset 上下文。'
                '你只负责判断用户这句跟进，应该查看数据集质量总览、查看健康检查详情、查看重复图片详情，还是不属于当前 dataset 上下文。'
                '如果用户是在问现在怎么样、数据集情况、详细一点的数据集信息、当前风险、整体状态，返回 quality。'
                '如果用户明确在问损坏、尺寸异常、健康检查、坏图、图片质量，返回 health。'
                '如果用户明确在问重复、重复图片、相似图片，返回 duplicates。'
                '如果用户是在换数据集、发起训练、做预测、抽图、抽帧或扫描视频，返回 other。'
                '输出必须是 JSON，对象格式固定为 {"action":"quality|health|duplicates|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'quality', 'health', 'duplicates'},
    )


async def classify_extract_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    active_dataset = session_state.active_dataset
    facts = {
        'has_extract_preview': bool(active_dataset.last_extract_preview),
        'has_extract_result': bool(active_dataset.last_extract_result),
        'has_video_scan': bool(active_dataset.last_video_scan),
        'has_frame_extract': bool(active_dataset.last_frame_extract),
        'extract_preview': active_dataset.last_extract_preview,
        'extract_result': active_dataset.last_extract_result,
        'video_scan': active_dataset.last_video_scan,
        'frame_extract': active_dataset.last_frame_extract,
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的抽取流程跟进路由器。'
                '当前同一会话里已经存在 extract 上下文。'
                '你只负责判断用户这句跟进，应该查看抽图预览结果、抽图执行结果、视频扫描结果、抽帧结果，还是不属于当前 extract 上下文。'
                '如果用户在问预览、计划抽多少、预览结果，返回 preview。'
                '如果用户在问抽图结果、抽样结果、输出目录、抽出来多少图片，返回 extract。'
                '如果用户在问视频有多少、扫描结果、有哪些视频，返回 video_scan。'
                '如果用户在问抽帧结果、帧输出、帧目录、抽了多少帧，返回 frame_extract。'
                '如果用户只是泛泛问“现在什么情况了/详细一点的信息”，优先使用当前上下文里最具体的已完成结果：frame_extract > extract > preview > video_scan。'
                '如果用户是在发起新的训练、预测、远端传输、数据质量检查，返回 other。'
                '输出必须是 JSON，对象格式固定为 {"action":"preview|extract|video_scan|frame_extract|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'preview', 'extract', 'video_scan', 'frame_extract'},
    )


async def classify_remote_transfer_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    active_remote = session_state.active_remote_transfer
    facts = {
        'target_label': active_remote.target_label,
        'profile_name': active_remote.profile_name,
        'remote_root': active_remote.remote_root,
        'has_last_profile_listing': bool(active_remote.last_profile_listing),
        'has_last_upload': bool(active_remote.last_upload),
        'has_last_download': bool(active_remote.last_download),
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的远端传输跟进路由器。'
                '当前同一会话里已经存在 remote transfer 上下文。'
                '你只负责判断用户这句跟进，应该查看远端 profile 列表结果、查看最近一次上传结果、查看最近一次下载结果，还是不属于当前 remote transfer 上下文。'
                '如果用户在问远端配置、可用服务器、profile、SSH alias，返回 profiles。'
                '如果用户在问上传到哪、远端目录、传输了什么、上传详情、远端传输情况，返回 upload。'
                '如果用户在问下载到哪、本机目录、拉回来了什么、下载详情，返回 download。'
                '如果用户只是泛泛问“现在什么情况了/详细一点的信息”，优先使用当前上下文里最近完成的方向：download > upload > profiles。'
                '如果用户是在发起新的上传/下载/预测/训练闭环，返回 other。'
                '输出必须是 JSON，对象格式固定为 {"action":"profiles|upload|download|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'profiles', 'upload', 'download'},
    )


async def classify_remote_roundtrip_followup_action(
    *,
    planner_llm: Any,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    has_training_state_context: TrainingContextChecker,
    classify_structured_action: StructuredActionClassifier,
) -> str:
    del normalized_text
    if planner_llm is None:
        return ''
    active_training = session_state.active_training
    active_prediction = session_state.active_prediction
    facts = {
        'has_training_remote_roundtrip': bool(active_training.last_remote_roundtrip),
        'training_remote_roundtrip': active_training.last_remote_roundtrip,
        'training_running': bool(active_training.running),
        'has_prediction_remote_roundtrip': bool(active_prediction.last_remote_roundtrip),
        'prediction_remote_roundtrip': active_prediction.last_remote_roundtrip,
        'has_local_training_context': bool(has_training_state_context()),
        'has_local_prediction_context': bool(
            active_prediction.last_result
            or active_prediction.last_summary
            or active_prediction.last_inspection
        ),
        'user_text': user_text,
    }
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Agent 的远端闭环跟进路由器。'
                '当前同一会话里已经存在 remote training / remote prediction roundtrip 上下文。'
                '你只负责判断用户这句跟进，是想继续查看远端训练闭环结果、继续查看远端预测闭环结果，还是不属于当前远端闭环上下文。'
                '如果用户在追问远端、服务器那边、闭环结果、上传后结果、回传结果、详细一点的远端执行信息，优先返回与当前上下文匹配的 training_pipeline 或 prediction_pipeline。'
                '如果用户是在问本地训练状态、本地预测结果、或是在发起新的上传/远端训练/远端预测，返回 other。'
                '如果只存在一种远端闭环上下文，而用户是在泛泛追问刚才那次远端执行情况，也返回对应 action。'
                '输出必须是 JSON，对象格式固定为 {"action":"training_pipeline|prediction_pipeline|other","reason":"..."}。'
                '不要输出 markdown，不要解释。\n'
                f'facts={json.dumps(facts, ensure_ascii=False)}'
            )
        ),
    ]
    return await classify_structured_action(
        messages=messages,
        allowed_actions={'training_pipeline', 'prediction_pipeline'},
    )

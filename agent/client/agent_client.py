from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent_plan.agent.client.file_checkpointer import FileCheckpointSaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from agent_plan.agent.client.context_builder import ContextBuilder
from agent_plan.agent.client.event_retriever import EventRetriever
from agent_plan.agent.client.grounded_reply_builder import build_grounded_tool_reply
from agent_plan.agent.client.state_applier import apply_tool_result_to_state
from agent_plan.agent.client.intent_parsing import (
    build_image_extract_args_from_text,
    build_video_extract_args_from_text,
    extract_all_paths_from_text,
    extract_count_from_text,
    extract_dataset_path_from_text,
    extract_epochs_from_text,
    extract_model_from_text,
    extract_output_path_from_text,
    extract_ratio_from_text,
    looks_like_model_path,
    looks_like_video_path,
    should_use_video_prediction,
)
from agent_plan.agent.client.llm_factory import LlmProviderSettings, build_llm, provider_summary
from agent_plan.agent.client.memory_store import MemoryStore
from agent_plan.agent.client.session_state import SessionState, utc_now
from agent_plan.agent.client.tool_adapter import adapt_tools_for_chat_model, canonical_tool_name, normalize_tool_args
from agent_plan.agent.client.tool_result_parser import parse_tool_message

SYSTEM_PROMPT = """你是 YoloStudio Agent，负责帮助用户完成数据准备、训练管理和图片预测。

工作原则：
1. 优先使用 MCP tools 获取真实结果，不要凭空猜测文件、数据集状态或训练状态。
2. 回答用简洁中文，先给结果，再给必要说明。
3. 涉及会改动数据或启动长任务的动作，不要在自然语言里先追问确认；参数足够时直接生成对应 tool 调用，由外部客户端拦截并做人审确认。
4. 工具失败时，直接说明失败原因，并给出下一步建议。
5. 如果用户的问题不需要工具，直接回答；如果需要工具，优先选择最少必要工具。

数据集目录约定：
- 用户提供的路径可能是数据集根目录（如 /home/kly/test_dataset/），而不是直接的图片目录。
- scan_dataset、validate_dataset、training_readiness 都支持直接传 dataset root，工具层会自动解析 images/ 和 labels/ 子目录。
- 当用户表达“用这个数据训练”“按默认比例划分再训练”这类需求时，优先使用 prepare_dataset_for_training 先把数据准备到可训练状态。
- 如果用户明确表达了“按默认比例划分 / 先划分再训练 / split 后训练”，调用 prepare_dataset_for_training 时应传 force_split=true。
- 当用户明确要求检查图片损坏、尺寸异常、重复图片或导出检查报告时，必须先调用对应工具，不要直接凭经验总结：综合检查优先用 run_dataset_health_check；如果用户只关心重复图片，再用 detect_duplicate_images；这些是只读检查，不会修改原始数据。
- 当用户明确要求对目录做图片抽取 / 抽样 / 采样 / 提取时，优先使用 preview_extract_images 或 extract_images；输入路径参数名是 source_path，默认使用 flat 输出布局，这样后续可直接接 scan_dataset / validate_dataset / prepare_dataset_for_training。
- 当用户明确要求对目录做视频扫描 / 抽帧时，优先使用 scan_videos 或 extract_video_frames；输入路径参数名同样是 source_path。
- 当用户明确要求对单张图片或图片目录做预测 / 推理 / 识别时，优先使用 predict_images；输入路径参数名是 source_path，模型参数名是 model。
- 当用户明确要求对单个视频或视频目录做预测 / 推理 / 识别时，优先使用 predict_videos；同样使用 source_path 和 model。
- 当前第二主线只支持图片、图片目录、单视频和视频目录，不支持 RTSP、摄像头或屏幕实时流。
- 当用户要求“总结预测结果 / 分析 prediction_report / 汇总刚才预测”时，优先使用 summarize_prediction_results；优先复用最近一次预测留下的 report_path 或 output_dir，不要凭空编造统计结果。
- 不要自己猜测子目录名称；优先依赖工具返回的 img_dir / label_dir / data_yaml。
- 当前只允许使用 MCP 已注册的工具名；不要发明工具名，也不要把桌面功能名直接当成工具名。像 detect_duplicates、detect_corrupted_images、dataset_manager.prepare_dataset 这类旧名字只属于兼容别名，不是首选正式名称。

训练约定：
- device 默认传 auto，GPU 分配由服务器端策略决定。
- training_readiness 是训练前检查的优先入口；如果 data_yaml 已明确且参数完整，也可以直接 start_training。
- check_gpu_status 仅在用户明确询问 GPU 状态时使用；不要在每次训练前机械地多调一次。
- 如果工具返回 next_actions / args_hint / recommended_start_training_args，继续执行时优先原样复用这些参数，不要自己重新猜。
- 回答训练参数时，要区分三类来源：用户明确指定、工具检测/生成、auto 解析结果；不要把 auto 或默认推断说成用户已明确指定；如果是工具解析结果，请明确说明是“工具检测到...”或“当前 auto 会解析到 ...”。
- 如果 prepare_dataset_for_training 返回目录结构不足，不要继续猜测路径或强行训练，直接说明需要用户显式提供 img_dir / label_dir。
- prepare_dataset_for_training 只负责数据准备，不代表已经开始训练；如果用户目标明确包含训练，且准备结果 ready=true，就应继续调用 start_training，而不是直接宣布训练已完成。
- 对健康检查、重复检测这类只读工具，回答时必须优先复述工具返回的 summary / warnings / next_actions / sample paths，不要编造不存在的修复能力（例如自动删除重复图、自动修复格式问题），除非系统里真的有对应工具。
- 绝对不要声称“模型已训练完成 / 已评估 / 已保存”之类结果，除非 start_training / check_training_status 等工具明确返回了这些事实。"""

HIGH_RISK_TOOLS = {"start_training", "split_dataset", "augment_dataset", "prepare_dataset_for_training"}


@dataclass(slots=True)
class AgentSettings:
    provider: str = os.getenv("YOLOSTUDIO_LLM_PROVIDER", "ollama")
    model: str = os.getenv("YOLOSTUDIO_AGENT_MODEL", "gemma4:e4b")
    base_url: str = os.getenv("YOLOSTUDIO_LLM_BASE_URL", "")
    api_key: str = os.getenv("YOLOSTUDIO_LLM_API_KEY", "")
    temperature: float = float(os.getenv("YOLOSTUDIO_LLM_TEMPERATURE", "0"))
    ollama_url: str = os.getenv("YOLOSTUDIO_OLLAMA_URL", "http://127.0.0.1:11434")
    mcp_url: str = os.getenv("YOLOSTUDIO_MCP_URL", "http://127.0.0.1:8080/mcp")
    max_history_messages: int = int(os.getenv("YOLOSTUDIO_MAX_HISTORY_MESSAGES", "12"))
    session_id: str = os.getenv("YOLOSTUDIO_SESSION_ID", "default")
    memory_root: str = os.getenv("YOLOSTUDIO_MEMORY_ROOT", str(Path(__file__).resolve().parents[2] / "memory"))

    def to_llm_settings(self) -> LlmProviderSettings:
        base_url = self.base_url or (self.ollama_url if self.provider.strip().lower() == 'ollama' else '')
        return LlmProviderSettings(
            provider=self.provider,
            model=self.model,
            base_url=base_url,
            api_key=self.api_key,
            temperature=self.temperature,
        )


class YoloStudioAgentClient:
    def __init__(self, graph: Any, settings: AgentSettings, tool_registry: dict[str, Any]) -> None:
        self.graph = graph
        self.settings = settings
        self.tool_registry = tool_registry
        self._messages: list[BaseMessage] = []
        self._turn_index = 0
        self._applied_tool_call_ids: set[str] = set()
        self.memory = MemoryStore(settings.memory_root)
        self.context_builder = ContextBuilder(SYSTEM_PROMPT)
        self.event_retriever = EventRetriever(self.memory)
        self.session_state: SessionState = self.memory.load_state(settings.session_id)
        self._sync_preferences()
        self.memory.save_state(self.session_state)

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def preview(self) -> str:
        return (
            f"YoloStudio Agent 已就绪 ({self.settings.model})\n"
            f"MCP Server: {self.settings.mcp_url} | LLM: {provider_summary(self.settings.to_llm_settings())}\n"
            f"Session: {self.session_state.session_id}"
        )

    def context_summary(self) -> str:
        digest = self.event_retriever.build_digest(self.session_state.session_id, self.session_state)
        return self.context_builder.build_state_summary(self.session_state, digest)

    def session_summary(self) -> str:
        return (
            f"Session ID: {self.session_state.session_id}\n"
            f"Created: {self.session_state.created_at}\n"
            f"Updated: {self.session_state.updated_at}\n"
            f"History Length: {len(self._messages)}\n"
            f"Turn Index: {self._turn_index}"
        )

    async def direct_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        canonical_name = canonical_tool_name(tool_name)
        normalized_args = normalize_tool_args(canonical_name, kwargs)
        tool = self.tool_registry.get(canonical_name)
        if not tool:
            return {"ok": False, "error": f"未找到工具: {canonical_name}"}
        payload = await tool.ainvoke(normalized_args)
        parsed = self._normalize_tool_output(payload)
        self.memory.append_event(self.session_state.session_id, "tool_result", {"tool": canonical_name, "args": normalized_args, "result": parsed})
        self._apply_to_state(canonical_name, parsed, normalized_args)
        self.memory.save_state(self.session_state)
        return parsed

    async def chat(self, user_text: str, auto_approve: bool = False) -> dict[str, Any]:
        self._messages.append(HumanMessage(content=user_text))
        self._turn_index += 1
        thread_id = f"{self.session_state.session_id}-turn-{self._turn_index}"
        config = {"configurable": {"thread_id": thread_id}}

        routed = await self._try_handle_mainline_intent(user_text, thread_id)
        if routed is not None:
            self._trim_history()
            self.memory.save_state(self.session_state)
            return routed

        self._trim_history()
        digest = self.event_retriever.build_digest(self.session_state.session_id, self.session_state)
        built_messages = self.context_builder.build_messages(self.session_state, self._messages, digest=digest)
        result = await self.graph.ainvoke({"messages": built_messages}, config=config)

        while True:
            pending = self._get_pending_tool_call(config)
            if not pending:
                break
            if pending["name"] in HIGH_RISK_TOOLS and not auto_approve:
                self._set_pending_confirmation(thread_id, pending)
                self.memory.save_state(self.session_state)
                return {
                    "status": "needs_confirmation",
                    "message": self._build_confirmation_prompt(pending),
                    "tool_call": pending,
                    "thread_id": thread_id,
                }
            if pending.get('adapted'):
                parsed = await self.direct_tool(pending['name'], **pending.get('args', {}))
                final_text = self._build_grounded_tool_reply([(pending['name'], parsed)]) or parsed.get('summary') or parsed.get('message') or parsed.get('error') or '操作已完成'
                self._messages.append(AIMessage(content=final_text))
                self._trim_history()
                self.memory.save_state(self.session_state)
                return {
                    "status": "completed",
                    "message": final_text,
                    "tool_call": pending,
                }
            result = await self.graph.ainvoke(Command(resume="approved"), config=config)

        applied_results = self._apply_tool_results(result["messages"], built_messages_len=len(built_messages))
        final_text = self._build_grounded_tool_reply(applied_results) or self._extract_or_fallback(result["messages"])
        self._messages.append(AIMessage(content=final_text))
        self._trim_history()
        self.memory.save_state(self.session_state)
        return {
            "status": "completed",
            "message": final_text,
            "tool_call": None,
        }

    async def confirm(self, thread_id: str, approved: bool) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        graph_pending = self._get_pending_tool_call(config)
        pending = graph_pending or self._pending_from_state()
        if not pending:
            return {"status": "error", "message": "当前没有待确认的高风险操作。"}

        if not approved:
            cancel_message = self._build_cancel_message(pending)
            self._messages.append(AIMessage(content=cancel_message))
            self._clear_pending_confirmation()
            self._trim_history()
            self.memory.append_event(self.session_state.session_id, "confirmation_cancelled", {"tool": pending["name"], "args": pending.get("args", {})})
            self.memory.save_state(self.session_state)
            return {
                "status": "cancelled",
                "message": cancel_message,
                "tool_call": pending,
            }

        self.memory.append_event(self.session_state.session_id, "confirmation_approved", {"tool": pending["name"], "args": pending.get("args", {})})
        self._clear_pending_confirmation()

        if graph_pending is None or graph_pending.get('adapted'):
            parsed = await self.direct_tool(pending["name"], **pending.get("args", {}))
            final_text = self._build_grounded_tool_reply([(pending['name'], parsed)]) or parsed.get("summary") or parsed.get("message") or ("操作执行成功" if parsed.get("ok") else parsed.get("error", "操作执行失败"))
            self._messages.append(AIMessage(content=final_text))
            self._trim_history()
            self.memory.save_state(self.session_state)
            if pending.get('name') == 'prepare_dataset_for_training':
                synthetic_followup = self._build_followup_training_request()
                if synthetic_followup:
                    self._set_pending_confirmation(thread_id, synthetic_followup)
                    self.memory.save_state(self.session_state)
                    return {
                        "status": "needs_confirmation",
                        "message": self._build_confirmation_prompt(synthetic_followup),
                        "tool_call": synthetic_followup,
                        "thread_id": thread_id,
                    }
            return {
                "status": "completed",
                "message": final_text,
                "tool_call": pending,
                "approved": True,
            }

        result = await self.graph.ainvoke(Command(resume="approved"), config=config)
        applied_results = self._apply_tool_results(result["messages"], built_messages_len=0)

        while True:
            next_pending = self._get_pending_tool_call(config)
            if not next_pending:
                break
            if next_pending["name"] in HIGH_RISK_TOOLS:
                self._set_pending_confirmation(thread_id, next_pending)
                self.memory.save_state(self.session_state)
                return {
                    "status": "needs_confirmation",
                    "message": self._build_confirmation_prompt(next_pending),
                    "tool_call": next_pending,
                    "thread_id": thread_id,
                }
            result = await self.graph.ainvoke(Command(resume="approved"), config=config)
            applied_results = self._apply_tool_results(result["messages"], built_messages_len=0)

        synthetic_followup = None
        if pending.get("name") == "prepare_dataset_for_training":
            synthetic_followup = self._build_followup_training_request()
        if synthetic_followup:
            self._set_pending_confirmation(thread_id, synthetic_followup)
            self.memory.save_state(self.session_state)
            return {
                "status": "needs_confirmation",
                "message": self._build_confirmation_prompt(synthetic_followup),
                "tool_call": synthetic_followup,
                "thread_id": thread_id,
            }

        final_text = self._build_grounded_tool_reply(applied_results) or self._extract_or_fallback(result["messages"])
        self._messages.append(AIMessage(content=final_text))
        self._trim_history()
        self.memory.save_state(self.session_state)
        return {
            "status": "completed",
            "message": final_text,
            "tool_call": pending,
            "approved": True,
        }

    def _sync_preferences(self) -> None:
        if self.session_state.preferences.default_model == self.settings.model:
            self.session_state.preferences.default_model = ""
        if self.session_state.preferences.language != "zh-CN":
            self.session_state.preferences.language = "zh-CN"

    async def _try_handle_mainline_intent(self, user_text: str, thread_id: str) -> dict[str, Any] | None:
        dataset_path = self._extract_dataset_path_from_text(user_text) or self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir
        prediction_path = self._extract_dataset_path_from_text(user_text) or self.session_state.active_prediction.source_path
        normalized_text = user_text.lower()
        wants_train = any(token in normalized_text for token in ('train', 'fine-tune', 'fit')) or ('训练' in user_text)
        no_train = any(token in user_text for token in ('不要训练', '不训练', '只检查', '仅检查', '不要启动'))
        wants_duplicates = ('重复' in user_text) or ('duplicate' in normalized_text)
        wants_health = any(token in user_text for token in ('损坏', '尺寸异常', '健康检查', '健康状况', '图片质量'))
        wants_quality = any(token in user_text for token in ('质量问题', '质量风险', '数据集质量', '分析', '总结'))
        wants_readiness = any(token in user_text for token in ('能不能直接训练', '是否可以直接训练', '可不可以直接训练', '直接训练', '训练前检查'))
        wants_split = any(token in user_text for token in ('默认划分', '划分比例', '先划分', 'split'))
        has_image_extract_verb = any(token in user_text for token in ('抽取', '提取', '抽样', '采样', '抽图', '抽一些图')) or (
            '抽' in user_text and '图片' in user_text
        )
        wants_extract_preview = ('预览' in user_text or 'preview' in normalized_text or 'dry-run' in normalized_text) and has_image_extract_verb
        wants_extract_images = any(token in user_text for token in ('抽取图片', '提取图片', '抽样图片', '采样图片', '抽图', '抽一些图')) or (
            has_image_extract_verb and '图片' in user_text
        ) or ('extract images' in normalized_text)
        wants_scan_videos = any(token in user_text for token in ('扫描视频', '视频扫描', '统计视频')) or (
            '视频' in user_text and any(token in user_text for token in ('扫描', '统计', '多少'))
        ) or ('scan videos' in normalized_text)
        wants_extract_frames = any(token in user_text for token in ('抽帧', '提帧')) or ('extract frames' in normalized_text)
        wants_predict = any(token in normalized_text for token in ('predict', 'infer')) or any(token in user_text for token in ('预测', '推理', '识别'))
        wants_prediction_summary = any(token in user_text for token in ('预测结果', '预测摘要', '总结一下预测', '刚才预测'))

        if wants_prediction_summary:
            summary_kwargs: dict[str, Any] = {}
            explicit_summary_target = self._extract_dataset_path_from_text(user_text)
            if explicit_summary_target:
                if explicit_summary_target.lower().endswith('.json'):
                    summary_kwargs['report_path'] = explicit_summary_target
                else:
                    summary_kwargs['output_dir'] = explicit_summary_target
            elif self.session_state.active_prediction.report_path:
                summary_kwargs['report_path'] = self.session_state.active_prediction.report_path
            elif self.session_state.active_prediction.output_dir:
                summary_kwargs['output_dir'] = self.session_state.active_prediction.output_dir
            elif prediction_path:
                summary_kwargs['output_dir'] = prediction_path

            if summary_kwargs:
                return await self._complete_direct_tool_reply('summarize_prediction_results', **summary_kwargs)

            if self.session_state.active_prediction.last_result:
                reply = self._build_grounded_tool_reply([('predict_images', self.session_state.active_prediction.last_result)])
                if reply:
                    self._messages.append(AIMessage(content=reply))
                    return {'status': 'completed', 'message': reply, 'tool_call': None}

        if dataset_path and wants_quality and not wants_train:
            return await self._complete_dataset_quality_reply(dataset_path)

        if dataset_path and wants_duplicates and not wants_train and not wants_health:
            return await self._complete_direct_tool_reply('detect_duplicate_images', dataset_path=dataset_path)

        if dataset_path and wants_health and not wants_train:
            return await self._complete_direct_tool_reply(
                'run_dataset_health_check',
                dataset_path=dataset_path,
                include_duplicates=wants_duplicates,
            )

        if dataset_path and wants_readiness and no_train:
            return await self._complete_direct_tool_reply('training_readiness', img_dir=dataset_path)

        if dataset_path and wants_extract_preview and not wants_train:
            return await self._complete_direct_tool_reply('preview_extract_images', **self._build_image_extract_args_from_text(user_text, dataset_path))

        if dataset_path and wants_extract_images and not wants_train and not wants_extract_preview:
            return await self._complete_direct_tool_reply('extract_images', **self._build_image_extract_args_from_text(user_text, dataset_path))

        if prediction_path and wants_scan_videos and not wants_predict and not wants_train:
            return await self._complete_direct_tool_reply('scan_videos', source_path=prediction_path)

        if prediction_path and wants_extract_frames and not wants_predict and not wants_train:
            return await self._complete_direct_tool_reply('extract_video_frames', **self._build_video_extract_args_from_text(user_text, prediction_path))

        if prediction_path and wants_predict and not wants_train:
            model = self._extract_model_from_text(user_text) or self.session_state.active_prediction.model or self.session_state.active_training.model
            predict_tool = 'predict_videos' if self._should_use_video_prediction(user_text, prediction_path) else 'predict_images'
            return await self._complete_direct_tool_reply(predict_tool, source_path=prediction_path, model=model)

        if dataset_path and wants_train and not no_train:
            args: dict[str, Any] = {'dataset_path': dataset_path}
            if wants_split:
                args['force_split'] = True
            self._set_pending_confirmation(thread_id, {'name': 'prepare_dataset_for_training', 'args': args, 'id': None, 'synthetic': True})
            return {
                'status': 'needs_confirmation',
                'message': self._build_confirmation_prompt({'name': 'prepare_dataset_for_training', 'args': args}),
                'tool_call': {'name': 'prepare_dataset_for_training', 'args': args},
                'thread_id': thread_id,
            }
        return None

    async def _complete_direct_tool_reply(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        parsed = await self.direct_tool(tool_name, **kwargs)
        reply = self._build_grounded_tool_reply([(canonical_tool_name(tool_name), parsed)])
        if not reply:
            reply = parsed.get('summary') or parsed.get('message') or parsed.get('error') or '操作已完成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if parsed.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    async def _complete_dataset_quality_reply(self, dataset_path: str) -> dict[str, Any]:
        scan = await self.direct_tool('scan_dataset', img_dir=dataset_path)
        validate = await self.direct_tool('validate_dataset', img_dir=dataset_path)
        health = await self.direct_tool('run_dataset_health_check', dataset_path=dataset_path, include_duplicates=True, max_duplicate_groups=3)

        warnings: list[str] = []
        for source in (scan, validate, health):
            for item in source.get('warnings') or []:
                if item not in warnings:
                    warnings.append(str(item))

        lines = [validate.get('summary') or scan.get('summary') or health.get('summary') or '数据集质量分析完成']
        if warnings:
            lines.append('最值得注意的风险:')
            lines.extend(f'- {item}' for item in warnings[:3])
        classes = scan.get('classes') or []
        if classes:
            lines.append(f"涉及类别: {', '.join(str(item) for item in classes[:4])}")
        if scan.get('detected_classes_txt'):
            lines.append(f"类名来源: {scan.get('detected_classes_txt')}")
        if health.get('duplicate_groups'):
            lines.append(f"重复图片: {health.get('duplicate_groups')} 组，额外重复文件 {health.get('duplicate_extra_files', 0)} 个")
        next_actions = validate.get('next_actions') or scan.get('next_actions') or health.get('next_actions') or []
        if next_actions:
            lines.append('建议:')
            lines.extend(f'- {item}' for item in next_actions[:2])
        reply = '\n'.join(lines)
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed',
            'message': reply,
            'tool_call': None,
        }

    def _extract_dataset_path_from_text(self, text: str) -> str:
        return extract_dataset_path_from_text(text)

    @staticmethod
    def _extract_all_paths_from_text(text: str) -> list[str]:
        return extract_all_paths_from_text(text)

    @staticmethod
    def _looks_like_model_path(path: str) -> bool:
        return looks_like_model_path(path)


    def _set_pending_confirmation(self, thread_id: str, pending: dict[str, Any]) -> None:
        pc = self.session_state.pending_confirmation
        pc.thread_id = thread_id
        pc.tool_name = pending["name"]
        pc.tool_args = pending.get("args", {})
        pc.created_at = utc_now()
        self.memory.append_event(self.session_state.session_id, "confirmation_requested", {"tool": pc.tool_name, "args": pc.tool_args, "thread_id": thread_id})

    def _clear_pending_confirmation(self) -> None:
        self.session_state.pending_confirmation.thread_id = ""
        self.session_state.pending_confirmation.tool_name = ""
        self.session_state.pending_confirmation.tool_args = {}
        self.session_state.pending_confirmation.created_at = ""

    def _pending_from_state(self) -> dict[str, Any] | None:
        pc = self.session_state.pending_confirmation
        if not pc.tool_name:
            return None
        return {"name": pc.tool_name, "args": pc.tool_args, "id": None}

    def _apply_tool_results(self, messages: list[BaseMessage], built_messages_len: int) -> list[tuple[str, dict[str, Any]]]:
        delta_messages = messages[built_messages_len:] if built_messages_len <= len(messages) else messages
        tool_args_by_id: dict[str, dict[str, Any]] = {}
        applied_results: list[tuple[str, dict[str, Any]]] = []
        for message in delta_messages:
            if isinstance(message, AIMessage):
                for tool_call in getattr(message, 'tool_calls', []) or []:
                    if tool_call.get('id'):
                        tool_args_by_id[tool_call['id']] = tool_call.get('args', {})
        for message in delta_messages:
            if not isinstance(message, ToolMessage):
                continue
            if message.tool_call_id and message.tool_call_id in self._applied_tool_call_ids:
                continue
            parsed = parse_tool_message(message)
            tool_name = canonical_tool_name(message.name or "unknown_tool")
            tool_args = normalize_tool_args(tool_name, tool_args_by_id.get(message.tool_call_id or '', {}))
            self.memory.append_event(self.session_state.session_id, "tool_result", {"tool": tool_name, "args": tool_args, "result": parsed})
            self._apply_to_state(tool_name, parsed, tool_args)
            applied_results.append((tool_name, parsed))
            if message.tool_call_id:
                self._applied_tool_call_ids.add(message.tool_call_id)
        return applied_results

    def _build_grounded_tool_reply(self, applied_results: list[tuple[str, dict[str, Any]]]) -> str:
        return build_grounded_tool_reply(applied_results)

    def _apply_to_state(self, tool_name: str, result: dict[str, Any], tool_args: dict[str, Any] | None = None) -> None:
        apply_tool_result_to_state(self.session_state, tool_name, result, tool_args)

    def _get_pending_tool_call(self, config: dict[str, Any]) -> dict[str, Any] | None:
        state = self.graph.get_state(config)
        if not state or not getattr(state, "next", None):
            return None
        if tuple(state.next) != ("tools",):
            return None
        messages = state.values.get("messages", [])
        if not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return None
        tool_call = last_message.tool_calls[0]
        raw_name = tool_call.get("name") or ""
        raw_args = tool_call.get("args", {})
        tool_name = canonical_tool_name(raw_name)
        tool_args = normalize_tool_args(tool_name, raw_args)
        return {
            "id": tool_call.get("id"),
            "name": tool_name,
            "args": tool_args,
            "raw_name": raw_name,
            "raw_args": raw_args,
            "adapted": raw_name != tool_name or raw_args != tool_args,
        }

    @staticmethod
    def _extract_final_ai(messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return message
        return None

    def _extract_or_fallback(self, messages: list[BaseMessage]) -> str:
        final_message = self._extract_final_ai(messages)
        final_text = final_message.content.strip() if final_message and isinstance(final_message.content, str) else ""
        return final_text or self._build_empty_reply_fallback(messages)

    def _trim_history(self) -> None:
        max_history = max(2, self.settings.max_history_messages)
        if len(self._messages) <= max_history:
            return
        self._messages = self._messages[-max_history:]


    def _recent_user_text(self) -> str:
        for message in reversed(self._messages):
            if isinstance(message, HumanMessage) and isinstance(message.content, str):
                return message.content
        return ""

    def _build_followup_training_request(self) -> dict[str, Any] | None:
        user_text = self._recent_user_text()
        if not user_text:
            return None
        if not any(token in user_text for token in ("训练", "train", "fine-tune", "fit")):
            return None
        if any(token in user_text for token in ("不要训练", "不训练", "仅检查", "只检查", "不要启动")):
            return None
        data_yaml = self.session_state.active_dataset.data_yaml.strip()
        if not data_yaml:
            return None
        model = self._extract_model_from_text(user_text)
        if not model:
            return None
        args: dict[str, Any] = {"data_yaml": data_yaml, "model": model}
        epochs = self._extract_epochs_from_text(user_text)
        if epochs is not None:
            args["epochs"] = epochs
        return {"id": None, "name": "start_training", "args": args, "synthetic": True}

    @staticmethod
    def _extract_model_from_text(text: str) -> str:
        return extract_model_from_text(text)

    @staticmethod
    def _looks_like_video_path(path: str) -> bool:
        return looks_like_video_path(path)

    def _should_use_video_prediction(self, user_text: str, path: str) -> bool:
        return should_use_video_prediction(user_text, path)

    def _extract_output_path_from_text(self, text: str, source_path: str = '') -> str:
        return extract_output_path_from_text(text, source_path)

    @staticmethod
    def _extract_count_from_text(text: str) -> int | None:
        return extract_count_from_text(text)

    @staticmethod
    def _extract_ratio_from_text(text: str) -> float | None:
        return extract_ratio_from_text(text)

    def _build_image_extract_args_from_text(self, user_text: str, source_path: str) -> dict[str, Any]:
        return build_image_extract_args_from_text(user_text, source_path)

    def _build_video_extract_args_from_text(self, user_text: str, source_path: str) -> dict[str, Any]:
        return build_video_extract_args_from_text(user_text, source_path)

    @staticmethod
    def _extract_epochs_from_text(text: str) -> int | None:
        return extract_epochs_from_text(text)

    @staticmethod
    def _build_confirmation_prompt(tool_call: dict[str, Any]) -> str:
        args = tool_call.get("args", {})
        pretty_args = "\n".join(f"  - {k}: {v}" for k, v in args.items()) or "  - 无参数"
        return (
            f"检测到高风险操作：{tool_call['name']}\n"
            f"参数摘要：\n{pretty_args}\n"
            "确认执行？(y/n)"
        )

    @staticmethod
    def _build_cancel_message(tool_call: dict[str, Any]) -> str:
        return f"已取消操作：{tool_call['name']}。如需继续，请调整参数后重新下达指令。"

    def _build_empty_reply_fallback(self, messages: list[BaseMessage]) -> str:
        tool_calls: list[str] = []
        tool_errors: list[str] = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, 'tool_calls', []) or []:
                    name = canonical_tool_name(tc.get('name'))
                    if name and name not in tool_calls:
                        tool_calls.append(name)
            if isinstance(msg, ToolMessage):
                parsed = parse_tool_message(msg)
                if not parsed.get('ok', True):
                    tool_errors.append(f"{msg.name}: {parsed.get('error', '未知错误')}")

        parts = ["Agent 未能生成有效回复。"]
        if tool_calls:
            parts.append(f"已调用工具: {', '.join(tool_calls)}")
        if tool_errors:
            parts.append(f"工具错误: {'; '.join(tool_errors)}")
        parts.append("可输入 /context 查看当前状态，或换一种方式描述需求。")
        return "\n".join(parts)

    @staticmethod
    def _normalize_tool_output(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            texts: list[str] = []
            for item in payload:
                if isinstance(item, dict):
                    texts.append(item.get('text', ''))
                else:
                    texts.append(str(item))
            raw = "\n".join(part for part in texts if part).strip()
            if not raw:
                return {"ok": True, "raw": raw}
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {"ok": True, "value": parsed}
            except Exception:
                return {"ok": True, "raw": raw}
        return {"ok": True, "raw": str(payload)}



def _checkpoint_path(settings: AgentSettings) -> Path:
    safe_session = re.sub(r'[^A-Za-z0-9_.-]+', '_', settings.session_id).strip('._') or 'default'
    return Path(settings.memory_root) / 'checkpoints' / f'{safe_session}.pkl'

async def build_agent_client(settings: AgentSettings | None = None) -> YoloStudioAgentClient:
    settings = settings or AgentSettings()
    client = MultiServerMCPClient(
        {
            "yolostudio": {
                "transport": "streamable-http",
                "url": settings.mcp_url,
            }
        }
    )
    raw_tools = await client.get_tools()
    tools = adapt_tools_for_chat_model(raw_tools)
    llm = build_llm(settings.to_llm_settings())
    graph = create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=FileCheckpointSaver(_checkpoint_path(settings)),
        interrupt_before=["tools"],
    )
    tool_registry = {tool.name: tool for tool in raw_tools}
    return YoloStudioAgentClient(graph=graph, settings=settings, tool_registry=tool_registry)


async def build_agent():
    return await build_agent_client()





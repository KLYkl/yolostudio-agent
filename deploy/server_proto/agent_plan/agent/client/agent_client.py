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
    extract_batch_size_from_text,
    extract_count_from_text,
    extract_custom_training_script_from_text,
    extract_dataset_path_from_text,
    extract_device_from_text,
    extract_epochs_from_text,
    extract_fraction_from_text,
    extract_image_size_from_text,
    extract_metric_signals_from_text,
    extract_model_from_text,
    extract_project_from_text,
    extract_optimizer_from_text,
    extract_lr0_from_text,
    extract_output_path_from_text,
    extract_ratio_from_text,
    extract_resume_flag_from_text,
    extract_run_name_from_text,
    extract_freeze_from_text,
    extract_single_cls_flag_from_text,
    extract_patience_from_text,
    extract_classes_from_text,
    extract_training_environment_from_text,
    extract_training_execution_backend_from_text,
    extract_workers_from_text,
    extract_amp_flag_from_text,
    is_training_discussion_only,
    wants_training_advanced_details,
    wants_default_training_environment,
    wants_clear_project,
    wants_clear_run_name,
    wants_clear_fraction,
    wants_clear_classes,
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
- 当用户明确询问“当前有哪些训练环境 / 默认会用哪个环境 / 这次训练会落到哪个 conda 环境”时，优先使用 list_training_environments。
- 当用户明确要求“先预检 / 先 dry-run / 先看看这次训练会怎么启动”时，优先使用 training_preflight，而不是直接 start_training。
- 当用户明确询问“最近训练有哪些 / 最近一次训练 / 训练历史 / 训练记录”时，优先使用 list_training_runs。
- 当用户明确询问“某次训练的详情 / 某个 run 的具体情况 / 某个 train_log 的记录”时，优先使用 inspect_training_run。
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
        if canonical_name == 'start_training' and parsed.get('ok'):
            self._clear_training_plan_draft()
        elif canonical_name == 'prepare_dataset_for_training' and parsed.get('ok'):
            draft = self.session_state.active_training.training_plan_draft or {}
            if str(draft.get('execution_mode') or '').strip().lower() == 'prepare_only':
                self._clear_training_plan_draft()
        self._record_secondary_event(canonical_name, parsed)
        self.memory.save_state(self.session_state)
        return parsed

    async def chat(self, user_text: str, auto_approve: bool = False) -> dict[str, Any]:
        self._messages.append(HumanMessage(content=user_text))
        self._turn_index += 1
        thread_id = f"{self.session_state.session_id}-turn-{self._turn_index}"
        config = {"configurable": {"thread_id": thread_id}}

        plan_dialogue = await self._try_handle_training_plan_dialogue(user_text, thread_id)
        if plan_dialogue is not None:
            self._trim_history()
            self.memory.save_state(self.session_state)
            return plan_dialogue

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
                    preflight = await self.direct_tool(
                        'training_preflight',
                        model=synthetic_followup['args'].get('model', ''),
                        data_yaml=synthetic_followup['args'].get('data_yaml', ''),
                        epochs=int(synthetic_followup['args'].get('epochs', 100)),
                        device=str(synthetic_followup['args'].get('device', 'auto') or 'auto'),
                        training_environment=str(synthetic_followup['args'].get('training_environment', '') or ''),
                        project=str(synthetic_followup['args'].get('project', '') or ''),
                        name=str(synthetic_followup['args'].get('name', '') or ''),
                        batch=synthetic_followup['args'].get('batch'),
                        imgsz=synthetic_followup['args'].get('imgsz'),
                        fraction=synthetic_followup['args'].get('fraction'),
                        classes=synthetic_followup['args'].get('classes'),
                        single_cls=synthetic_followup['args'].get('single_cls'),
                        optimizer=str(synthetic_followup['args'].get('optimizer', '') or ''),
                        freeze=synthetic_followup['args'].get('freeze'),
                        resume=synthetic_followup['args'].get('resume'),
                        lr0=synthetic_followup['args'].get('lr0'),
                        patience=synthetic_followup['args'].get('patience'),
                        workers=synthetic_followup['args'].get('workers'),
                        amp=synthetic_followup['args'].get('amp'),
                    )
                    draft = self._build_training_plan_draft(
                        user_text=self._recent_user_text(),
                        dataset_path=self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir,
                        readiness=self.session_state.active_dataset.last_readiness,
                        preflight=preflight,
                        next_tool_name='start_training' if preflight.get('ready_to_start') else '',
                        next_tool_args=dict(synthetic_followup.get('args') or {}) if preflight.get('ready_to_start') else {},
                        planned_training_args=dict(synthetic_followup.get('args') or {}),
                    )
                    self._save_training_plan_draft(draft)
                    if not preflight.get('ready_to_start'):
                        reply = self._merge_grounded_sections([
                            final_text,
                            self._build_grounded_tool_reply([('training_preflight', preflight)]),
                        ]) or preflight.get('summary') or final_text
                        self._messages.append(AIMessage(content=reply))
                        self._trim_history()
                        self.memory.save_state(self.session_state)
                        return {
                            "status": "error",
                            "message": reply,
                            "tool_call": synthetic_followup,
                            "approved": True,
                        }
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
            preflight = await self.direct_tool(
                'training_preflight',
                model=synthetic_followup['args'].get('model', ''),
                data_yaml=synthetic_followup['args'].get('data_yaml', ''),
                epochs=int(synthetic_followup['args'].get('epochs', 100)),
                device=str(synthetic_followup['args'].get('device', 'auto') or 'auto'),
                training_environment=str(synthetic_followup['args'].get('training_environment', '') or ''),
                project=str(synthetic_followup['args'].get('project', '') or ''),
                name=str(synthetic_followup['args'].get('name', '') or ''),
                batch=synthetic_followup['args'].get('batch'),
                imgsz=synthetic_followup['args'].get('imgsz'),
                fraction=synthetic_followup['args'].get('fraction'),
                classes=synthetic_followup['args'].get('classes'),
                single_cls=synthetic_followup['args'].get('single_cls'),
                optimizer=str(synthetic_followup['args'].get('optimizer', '') or ''),
                freeze=synthetic_followup['args'].get('freeze'),
                resume=synthetic_followup['args'].get('resume'),
                lr0=synthetic_followup['args'].get('lr0'),
                patience=synthetic_followup['args'].get('patience'),
                workers=synthetic_followup['args'].get('workers'),
                amp=synthetic_followup['args'].get('amp'),
            )
            draft = self._build_training_plan_draft(
                user_text=self._recent_user_text(),
                dataset_path=self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir,
                readiness=self.session_state.active_dataset.last_readiness,
                preflight=preflight,
                next_tool_name='start_training' if preflight.get('ready_to_start') else '',
                next_tool_args=dict(synthetic_followup.get('args') or {}) if preflight.get('ready_to_start') else {},
                planned_training_args=dict(synthetic_followup.get('args') or {}),
            )
            self._save_training_plan_draft(draft)
            if not preflight.get('ready_to_start'):
                preflight_reply = self._build_grounded_tool_reply([('training_preflight', preflight)]) or preflight.get('summary') or '训练预检未通过'
                self._messages.append(AIMessage(content=preflight_reply))
                self._trim_history()
                self.memory.save_state(self.session_state)
                return {
                    "status": "error",
                    "message": preflight_reply,
                    "tool_call": synthetic_followup,
                    "approved": True,
                }
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
        metric_signals = self._extract_metric_signals_from_text(user_text)
        wants_train = any(token in normalized_text for token in ('train', 'fine-tune', 'fit')) or ('训练' in user_text)
        no_train = any(token in user_text for token in ('不要训练', '不训练', '只检查', '仅检查', '不要启动'))
        wants_duplicates = ('重复' in user_text) or ('duplicate' in normalized_text)
        wants_health = any(token in user_text for token in ('损坏', '尺寸异常', '健康检查', '健康状况', '图片质量'))
        wants_quality = any(token in user_text for token in ('质量问题', '质量风险', '数据集质量'))
        wants_readiness = any(token in user_text for token in ('能不能直接训练', '是否可以直接训练', '可不可以直接训练', '直接训练', '训练前检查', '适合训练吗', '适不适合训练'))
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
        asks_metric_terms = any(token in normalized_text for token in ('precision', 'recall', 'map', 'loss', 'epoch', 'epochs', 'batch', 'imgsz', 'patience', 'lr')) or any(token in user_text for token in ('精确率', '召回', '损失', '学习率', '轮数', '批大小'))
        wants_training_outcome_analysis = (
            any(token in user_text for token in ('训练效果怎么样', '这次训练效果怎么样', '训练结果怎么样', '训练效果如何', '结果更像', '训练效果'))
            or (asks_metric_terms and any(token in user_text for token in ('怎么看', '说明什么', '意味着什么', '结果如何')))
        )
        wants_next_step_guidance = any(token in user_text for token in ('下一步', '先补数据还是先调参数', '先补数据', '先调参数', '怎么优化', '如何优化下一步训练', '下一轮怎么做'))
        wants_training_knowledge = bool(metric_signals) or (asks_metric_terms and any(token in user_text for token in ('说明什么', '什么意思', '意味着什么', '怎么看')))
        readiness_only_query = wants_readiness and (no_train or any(token in user_text for token in ('吗', '是否', '能不能', '可不可以')))
        training_command_like = any(token in user_text for token in ('开始训练', '启动训练', '训练这个数据', '用这个数据训练', '直接开训', 'start_training'))

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

        if dataset_path and readiness_only_query:
            return await self._complete_readiness_knowledge_reply(dataset_path)

        if not wants_predict and not training_command_like and wants_next_step_guidance:
            return await self._complete_next_training_step_reply(dataset_path if dataset_path else '')

        if not wants_predict and not training_command_like and wants_training_knowledge:
            return await self._complete_knowledge_retrieval_reply(
                topic='training_metrics' if asks_metric_terms else 'workflow',
                stage='post_training',
                signals=metric_signals,
            )

        if not wants_predict and not training_command_like and wants_training_outcome_analysis:
            return await self._complete_training_outcome_analysis_reply()

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

        if dataset_path and wants_train and not no_train and not readiness_only_query and not wants_training_outcome_analysis and not wants_next_step_guidance and not wants_training_knowledge:
            readiness = await self.direct_tool('training_readiness', img_dir=dataset_path)
            await self.direct_tool('list_training_environments')
            requested_args = self._collect_requested_training_args(
                user_text,
                data_yaml=str(readiness.get('resolved_data_yaml') or self.session_state.active_dataset.data_yaml or ''),
            )
            requested_model = str(requested_args.get('model') or '').strip()
            discussion_only = self._is_training_discussion_only(user_text)
            execution_backend = self._extract_training_execution_backend_from_text(user_text)

            if not requested_model:
                draft = self._build_training_plan_draft(
                    user_text=user_text,
                    dataset_path=dataset_path,
                    readiness=readiness,
                    next_tool_name='',
                    next_tool_args={},
                    planned_training_args=requested_args,
                )
                blockers = list(draft.get('blockers') or [])
                blockers.insert(0, '当前缺少预训练权重/模型，先补模型后再确认训练')
                draft['blockers'] = blockers
                self._save_training_plan_draft(draft)
                reply = self._render_training_plan_draft(draft, pending=False)
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}

            if execution_backend != 'standard_yolo':
                draft = self._build_training_plan_draft(
                    user_text=user_text,
                    dataset_path=dataset_path,
                    readiness=readiness,
                    next_tool_name='',
                    next_tool_args={},
                    planned_training_args=requested_args,
                )
                self._save_training_plan_draft(draft)
                reply = self._render_training_plan_draft(draft, pending=False)
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}

            can_direct_train = bool(readiness.get('ready')) and bool(readiness.get('resolved_data_yaml'))
            if can_direct_train:
                preflight = await self.direct_tool(
                    'training_preflight',
                    model=requested_model,
                    data_yaml=str(requested_args.get('data_yaml') or readiness.get('resolved_data_yaml') or ''),
                    epochs=int(requested_args.get('epochs', 100)),
                    device=str(requested_args.get('device', 'auto') or 'auto'),
                    training_environment=str(requested_args.get('training_environment') or ''),
                    project=str(requested_args.get('project') or ''),
                    name=str(requested_args.get('name') or ''),
                    batch=requested_args.get('batch'),
                    imgsz=requested_args.get('imgsz'),
                    fraction=requested_args.get('fraction'),
                    classes=requested_args.get('classes'),
                    single_cls=requested_args.get('single_cls'),
                    optimizer=str(requested_args.get('optimizer', '') or ''),
                    freeze=requested_args.get('freeze'),
                    resume=requested_args.get('resume'),
                    lr0=requested_args.get('lr0'),
                    patience=requested_args.get('patience'),
                    workers=requested_args.get('workers'),
                    amp=requested_args.get('amp'),
                )
                next_args = {
                    'model': str((preflight.get('resolved_args') or {}).get('model') or requested_model),
                    'data_yaml': str((preflight.get('resolved_args') or {}).get('data_yaml') or readiness.get('resolved_data_yaml') or ''),
                    'epochs': int((preflight.get('resolved_args') or {}).get('epochs') or requested_args.get('epochs', 100)),
                    'device': str((preflight.get('resolved_args') or {}).get('device') or requested_args.get('device') or 'auto'),
                    'training_environment': str((preflight.get('resolved_args') or {}).get('training_environment') or requested_args.get('training_environment') or ''),
                    'project': str((preflight.get('resolved_args') or {}).get('project') or requested_args.get('project') or ''),
                    'name': str((preflight.get('resolved_args') or {}).get('name') or requested_args.get('name') or ''),
                    'batch': (preflight.get('resolved_args') or {}).get('batch', requested_args.get('batch')),
                    'imgsz': (preflight.get('resolved_args') or {}).get('imgsz', requested_args.get('imgsz')),
                    'fraction': (preflight.get('resolved_args') or {}).get('fraction', requested_args.get('fraction')),
                    'classes': (preflight.get('resolved_args') or {}).get('classes', requested_args.get('classes')),
                    'single_cls': (preflight.get('resolved_args') or {}).get('single_cls', requested_args.get('single_cls')),
                    'optimizer': str((preflight.get('resolved_args') or {}).get('optimizer') or requested_args.get('optimizer') or ''),
                    'freeze': (preflight.get('resolved_args') or {}).get('freeze', requested_args.get('freeze')),
                    'resume': (preflight.get('resolved_args') or {}).get('resume', requested_args.get('resume')),
                    'lr0': (preflight.get('resolved_args') or {}).get('lr0', requested_args.get('lr0')),
                    'patience': (preflight.get('resolved_args') or {}).get('patience', requested_args.get('patience')),
                    'workers': (preflight.get('resolved_args') or {}).get('workers', requested_args.get('workers')),
                    'amp': (preflight.get('resolved_args') or {}).get('amp', requested_args.get('amp')),
                }
                draft = self._build_training_plan_draft(
                    user_text=user_text,
                    dataset_path=dataset_path,
                    readiness=readiness,
                    preflight=preflight,
                    next_tool_name='start_training' if preflight.get('ready_to_start') else '',
                    next_tool_args=next_args if preflight.get('ready_to_start') else {},
                    planned_training_args=next_args,
                )
                self._save_training_plan_draft(draft)
                reply = self._render_training_plan_draft(draft, pending=bool(preflight.get('ready_to_start') and not discussion_only))
                self._messages.append(AIMessage(content=reply))
                if preflight.get('ready_to_start') and not discussion_only:
                    self._set_pending_confirmation(thread_id, {'name': 'start_training', 'args': next_args, 'id': None, 'synthetic': True})
                    return {
                        'status': 'needs_confirmation',
                        'message': reply,
                        'tool_call': {'name': 'start_training', 'args': next_args},
                        'thread_id': thread_id,
                    }
                return {'status': 'completed', 'message': reply, 'tool_call': None}

            if readiness.get('preparable'):
                args: dict[str, Any] = {'dataset_path': dataset_path}
                if wants_split:
                    args['force_split'] = True
                draft = self._build_training_plan_draft(
                    user_text=user_text,
                    dataset_path=dataset_path,
                    readiness=readiness,
                    preflight={},
                    next_tool_name='prepare_dataset_for_training',
                    next_tool_args=args,
                    planned_training_args=requested_args,
                )
                self._save_training_plan_draft(draft)
                reply = self._render_training_plan_draft(draft, pending=not discussion_only)
                self._messages.append(AIMessage(content=reply))
                if discussion_only:
                    return {'status': 'completed', 'message': reply, 'tool_call': None}
                self._set_pending_confirmation(thread_id, {'name': 'prepare_dataset_for_training', 'args': args, 'id': None, 'synthetic': True})
                return {
                    'status': 'needs_confirmation',
                    'message': reply,
                    'tool_call': {'name': 'prepare_dataset_for_training', 'args': args},
                    'thread_id': thread_id,
                }

            draft = self._build_training_plan_draft(
                user_text=user_text,
                dataset_path=dataset_path,
                readiness=readiness,
                preflight={},
                next_tool_name='',
                next_tool_args={},
                planned_training_args=requested_args,
            )
            self._save_training_plan_draft(draft)
            reply = self._render_training_plan_draft(draft, pending=False)
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}
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

    async def _complete_readiness_knowledge_reply(self, dataset_path: str) -> dict[str, Any]:
        readiness = await self.direct_tool('training_readiness', img_dir=dataset_path)
        training_summary = self.session_state.active_training.training_run_summary or self.session_state.active_training.last_summary
        recommendation = await self.direct_tool(
            'recommend_next_training_step',
            readiness=readiness,
            health=self.session_state.active_dataset.last_health_check,
            status=training_summary or self.session_state.active_training.last_status,
            prediction_summary=self.session_state.active_prediction.last_result,
        )
        reply = self._merge_grounded_sections([
            self._build_grounded_tool_reply([('training_readiness', readiness)]),
            self._build_grounded_tool_reply([('recommend_next_training_step', recommendation)]),
        ])
        if not reply:
            reply = readiness.get('summary') or recommendation.get('summary') or readiness.get('error') or recommendation.get('error') or '训练前知识分析已完成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if readiness.get('ok', True) and recommendation.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    async def _complete_knowledge_retrieval_reply(self, *, topic: str, stage: str, signals: list[str] | None = None) -> dict[str, Any]:
        result = await self.direct_tool(
            'retrieve_training_knowledge',
            topic=topic,
            stage=stage,
            model_family='yolo',
            task_type='detection',
            signals=signals or [],
        )
        reply = self._build_grounded_tool_reply([('retrieve_training_knowledge', result)])
        if not reply:
            reply = result.get('summary') or result.get('error') or '知识检索已完成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if result.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    async def _complete_training_outcome_analysis_reply(self) -> dict[str, Any]:
        training_summary = await self.direct_tool('summarize_training_run')
        result = await self.direct_tool(
            'analyze_training_outcome',
            metrics=training_summary,
            data_quality=self.session_state.active_dataset.last_health_check or self.session_state.active_dataset.last_validate,
            prediction_summary=self.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        reply = self._merge_grounded_sections([
            self._build_grounded_tool_reply([('summarize_training_run', training_summary)]),
            self._build_grounded_tool_reply([('analyze_training_outcome', result)]),
        ])
        if not reply:
            reply = result.get('summary') or training_summary.get('summary') or result.get('error') or '训练结果分析已完成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if training_summary.get('ok', True) and result.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    async def _complete_next_training_step_reply(self, dataset_path: str = '') -> dict[str, Any]:
        readiness: dict[str, Any] | None = None
        if dataset_path:
            readiness = await self.direct_tool('training_readiness', img_dir=dataset_path)
        training_summary = await self.direct_tool('summarize_training_run')
        result = await self.direct_tool(
            'recommend_next_training_step',
            readiness=readiness or self.session_state.active_dataset.last_readiness,
            health=self.session_state.active_dataset.last_health_check,
            status=training_summary,
            prediction_summary=self.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        sections = []
        if readiness is not None:
            sections.append(self._build_grounded_tool_reply([('training_readiness', readiness)]))
        sections.append(self._build_grounded_tool_reply([('summarize_training_run', training_summary)]))
        sections.append(self._build_grounded_tool_reply([('recommend_next_training_step', result)]))
        reply = self._merge_grounded_sections(sections)
        if not reply:
            reply = result.get('summary') or training_summary.get('summary') or result.get('error') or '下一步建议已生成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if training_summary.get('ok', True) and result.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    @staticmethod
    def _merge_grounded_sections(sections: list[str]) -> str:
        cleaned: list[str] = []
        for section in sections:
            text = str(section or '').strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return '\n\n'.join(cleaned)

    def _record_secondary_event(self, tool_name: str, result: dict[str, Any]) -> None:
        if not result.get('ok'):
            return
        if tool_name == 'analyze_training_outcome':
            self.memory.append_event(
                self.session_state.session_id,
                'training_analysis',
                {
                    'summary': result.get('summary', ''),
                    'assessment': result.get('assessment', ''),
                    'matched_rule_ids': result.get('matched_rule_ids', []),
                },
            )
        elif tool_name == 'recommend_next_training_step':
            self.memory.append_event(
                self.session_state.session_id,
                'knowledge_recommendation',
                {
                    'summary': result.get('summary', ''),
                    'recommended_action': result.get('recommended_action', ''),
                    'matched_rule_ids': result.get('matched_rule_ids', []),
                },
            )

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
            self._record_secondary_event(tool_name, parsed)
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
        draft = self.session_state.active_training.training_plan_draft or {}
        if str(draft.get('execution_mode') or '').strip().lower() == 'prepare_only':
            return None
        planned_args = dict(draft.get('planned_training_args') or {})
        if draft and planned_args.get('model') and planned_args.get('data_yaml'):
            return {
                "id": None,
                "name": "start_training",
                "args": planned_args,
                "synthetic": True,
            }
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
        batch = self._extract_batch_size_from_text(user_text)
        if batch is not None:
            args["batch"] = batch
        imgsz = self._extract_image_size_from_text(user_text)
        if imgsz is not None:
            args["imgsz"] = imgsz
        device = self._extract_device_from_text(user_text)
        if device:
            args["device"] = device
        training_environment = self._extract_training_environment_from_text(
            user_text,
            list((self.session_state.active_training.last_environment_probe or {}).get('environments') or []),
        )
        if training_environment:
            args["training_environment"] = training_environment
        project = self._extract_project_from_text(user_text)
        if project:
            args["project"] = project
        run_name = self._extract_run_name_from_text(user_text)
        if run_name:
            args["name"] = run_name
        fraction = self._extract_fraction_from_text(user_text)
        if fraction is not None:
            args["fraction"] = fraction
        classes = self._extract_classes_from_text(user_text)
        if classes is not None:
            args["classes"] = classes
        single_cls = self._extract_single_cls_flag_from_text(user_text)
        if single_cls is not None:
            args["single_cls"] = single_cls
        optimizer = self._extract_optimizer_from_text(user_text)
        if optimizer:
            args["optimizer"] = optimizer
        freeze = self._extract_freeze_from_text(user_text)
        if freeze is not None:
            args["freeze"] = freeze
        resume = self._extract_resume_flag_from_text(user_text)
        if resume is not None:
            args["resume"] = resume
        lr0 = self._extract_lr0_from_text(user_text)
        if lr0 is not None:
            args["lr0"] = lr0
        patience = self._extract_patience_from_text(user_text)
        if patience is not None:
            args["patience"] = patience
        workers = self._extract_workers_from_text(user_text)
        if workers is not None:
            args["workers"] = workers
        amp = self._extract_amp_flag_from_text(user_text)
        if amp is not None:
            args["amp"] = amp
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
    def _extract_metric_signals_from_text(text: str) -> list[str]:
        return extract_metric_signals_from_text(text)

    @staticmethod
    def _extract_batch_size_from_text(text: str) -> int | None:
        return extract_batch_size_from_text(text)

    @staticmethod
    def _extract_image_size_from_text(text: str) -> int | None:
        return extract_image_size_from_text(text)

    @staticmethod
    def _extract_device_from_text(text: str) -> str:
        return extract_device_from_text(text)

    @staticmethod
    def _extract_training_environment_from_text(text: str, known_environments: list[dict[str, Any]] | None = None) -> str:
        return extract_training_environment_from_text(text, known_environments)

    @staticmethod
    def _extract_project_from_text(text: str) -> str:
        return extract_project_from_text(text)

    @staticmethod
    def _extract_run_name_from_text(text: str) -> str:
        return extract_run_name_from_text(text)

    @staticmethod
    def _extract_fraction_from_text(text: str) -> float | None:
        return extract_fraction_from_text(text)

    @staticmethod
    def _extract_classes_from_text(text: str) -> list[int] | None:
        return extract_classes_from_text(text)

    @staticmethod
    def _extract_single_cls_flag_from_text(text: str) -> bool | None:
        return extract_single_cls_flag_from_text(text)

    @staticmethod
    def _extract_optimizer_from_text(text: str) -> str:
        return extract_optimizer_from_text(text)

    @staticmethod
    def _extract_freeze_from_text(text: str) -> int | None:
        return extract_freeze_from_text(text)

    @staticmethod
    def _extract_resume_flag_from_text(text: str) -> bool | None:
        return extract_resume_flag_from_text(text)

    @staticmethod
    def _extract_custom_training_script_from_text(text: str) -> str:
        return extract_custom_training_script_from_text(text)

    @staticmethod
    def _extract_training_execution_backend_from_text(text: str) -> str:
        return extract_training_execution_backend_from_text(text)

    @staticmethod
    def _is_training_discussion_only(text: str) -> bool:
        return is_training_discussion_only(text)

    @staticmethod
    def _extract_lr0_from_text(text: str) -> float | None:
        return extract_lr0_from_text(text)

    @staticmethod
    def _extract_patience_from_text(text: str) -> int | None:
        return extract_patience_from_text(text)

    @staticmethod
    def _extract_workers_from_text(text: str) -> int | None:
        return extract_workers_from_text(text)

    @staticmethod
    def _extract_amp_flag_from_text(text: str) -> bool | None:
        return extract_amp_flag_from_text(text)

    @staticmethod
    def _wants_default_training_environment(text: str) -> bool:
        return wants_default_training_environment(text)

    @staticmethod
    def _wants_clear_project(text: str) -> bool:
        return wants_clear_project(text)

    @staticmethod
    def _wants_clear_run_name(text: str) -> bool:
        return wants_clear_run_name(text)

    @staticmethod
    def _wants_clear_fraction(text: str) -> bool:
        return wants_clear_fraction(text)

    @staticmethod
    def _wants_clear_classes(text: str) -> bool:
        return wants_clear_classes(text)

    def _collect_training_clear_fields(self, user_text: str) -> set[str]:
        clear_fields: set[str] = set()
        if self._wants_default_training_environment(user_text):
            clear_fields.add('training_environment')
        if self._wants_clear_project(user_text):
            clear_fields.add('project')
        if self._wants_clear_run_name(user_text):
            clear_fields.add('name')
        if self._wants_clear_fraction(user_text):
            clear_fields.add('fraction')
        if self._wants_clear_classes(user_text):
            clear_fields.add('classes')
        return clear_fields

    def _collect_requested_training_args(self, user_text: str, *, data_yaml: str = '') -> dict[str, Any]:
        tr = self.session_state.active_training
        args: dict[str, Any] = {}
        known_environments = list((tr.last_environment_probe or {}).get('environments') or [])
        model = self._extract_model_from_text(user_text)
        if model:
            args['model'] = model
        resolved_yaml = str(data_yaml or tr.data_yaml or self.session_state.active_dataset.data_yaml or '').strip()
        if resolved_yaml:
            args['data_yaml'] = resolved_yaml
        epochs = self._extract_epochs_from_text(user_text)
        if epochs is not None:
            args['epochs'] = epochs
        batch = self._extract_batch_size_from_text(user_text)
        if batch is not None:
            args['batch'] = batch
        imgsz = self._extract_image_size_from_text(user_text)
        if imgsz is not None:
            args['imgsz'] = imgsz
        device = self._extract_device_from_text(user_text)
        if device:
            args['device'] = device
        training_environment = self._extract_training_environment_from_text(user_text, known_environments)
        if training_environment:
            args['training_environment'] = training_environment
        project = self._extract_project_from_text(user_text)
        if project:
            args['project'] = project
        run_name = self._extract_run_name_from_text(user_text)
        if run_name:
            args['name'] = run_name
        fraction = self._extract_fraction_from_text(user_text)
        if fraction is not None:
            args['fraction'] = fraction
        classes = self._extract_classes_from_text(user_text)
        if classes is not None:
            args['classes'] = classes
        single_cls = self._extract_single_cls_flag_from_text(user_text)
        if single_cls is not None:
            args['single_cls'] = single_cls
        optimizer = self._extract_optimizer_from_text(user_text)
        if optimizer:
            args['optimizer'] = optimizer
        freeze = self._extract_freeze_from_text(user_text)
        if freeze is not None:
            args['freeze'] = freeze
        resume = self._extract_resume_flag_from_text(user_text)
        if resume is not None:
            args['resume'] = resume
        lr0 = self._extract_lr0_from_text(user_text)
        if lr0 is not None:
            args['lr0'] = lr0
        patience = self._extract_patience_from_text(user_text)
        if patience is not None:
            args['patience'] = patience
        workers = self._extract_workers_from_text(user_text)
        if workers is not None:
            args['workers'] = workers
        amp = self._extract_amp_flag_from_text(user_text)
        if amp is not None:
            args['amp'] = amp
        return args

    def _build_training_plan_draft(
        self,
        *,
        user_text: str,
        dataset_path: str,
        readiness: dict[str, Any] | None = None,
        preflight: dict[str, Any] | None = None,
        next_tool_name: str = '',
        next_tool_args: dict[str, Any] | None = None,
        planned_training_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        readiness = readiness or {}
        preflight = preflight or {}
        next_tool_args = dict(next_tool_args or {})
        planned_training_args = dict(planned_training_args or {})
        environment = (
            next_tool_args.get('training_environment')
            or planned_training_args.get('training_environment')
            or preflight.get('training_environment')
            or self.session_state.active_training.last_environment_probe.get('default_environment')
            or {}
        )
        if isinstance(environment, dict):
            env_name = str(environment.get('display_name') or environment.get('name') or '').strip()
        else:
            env_name = str(environment or '').strip()
        execution_backend = self._extract_training_execution_backend_from_text(user_text)
        custom_script = self._extract_custom_training_script_from_text(user_text)
        if next_tool_name == 'prepare_dataset_for_training':
            execution_mode = 'prepare_then_train'
        elif next_tool_name == 'start_training':
            execution_mode = 'direct_train'
        else:
            execution_mode = 'discussion_only' if self._is_training_discussion_only(user_text) else 'blocked'
        if any(token in user_text for token in ('只做准备', '只准备', '先准备不要训练')):
            execution_mode = 'prepare_only'

        blockers = [str(item) for item in (preflight.get('blockers') or readiness.get('blockers') or []) if str(item).strip()]
        warnings = [str(item) for item in (preflight.get('warnings') or readiness.get('warnings') or []) if str(item).strip()]
        risks = list(warnings[:2])
        if readiness.get('primary_blocker_type') and readiness.get('primary_blocker_type') not in risks:
            risks.insert(0, str(readiness.get('primary_blocker_type')))
        if execution_backend != 'standard_yolo':
            blockers.insert(0, '当前自动执行链只支持标准 YOLO 训练；自定义训练后端先保留为计划草案')
        reasoning_parts: list[str] = []
        if execution_backend != 'standard_yolo':
            reasoning_parts.append('当前检测到自定义训练后端，因此先保留为计划草案，不直接自动执行。')
        elif preflight.get('ready_to_start'):
            reasoning_parts.append('当前数据已具备训练条件，训练环境和参数预检都已通过；确认后即可启动。')
        elif readiness.get('preparable'):
            reasoning_parts.append('当前数据还不能直接训练，但可以先自动准备到可训练状态。')
        elif blockers:
            reasoning_parts.append(f"当前仍有阻塞，优先解决：{blockers[0]}")
        elif self._is_training_discussion_only(user_text):
            reasoning_parts.append('当前按讨论模式生成草案，不会直接执行。')

        data_yaml = str(
            planned_training_args.get('data_yaml')
            or readiness.get('resolved_data_yaml')
            or self.session_state.active_dataset.data_yaml
            or ''
        ).strip()
        if data_yaml:
            planned_training_args['data_yaml'] = data_yaml
        if env_name:
            planned_training_args['training_environment'] = env_name

        default_environment = self.session_state.active_training.last_environment_probe.get('default_environment') or {}
        default_env_name = str(default_environment.get('display_name') or default_environment.get('name') or '').strip()
        if env_name and default_env_name and env_name != default_env_name:
            reasoning_parts.append(f'当前计划已从默认环境 {default_env_name} 切换到 {env_name}。')
        fraction_value = planned_training_args.get('fraction')
        if fraction_value is not None:
            try:
                percent = float(fraction_value) * 100.0
                reasoning_parts.append(f'当前只计划使用约 {percent:.0f}% 的训练数据。')
            except (TypeError, ValueError):
                pass
        classes_value = planned_training_args.get('classes') or []
        if classes_value:
            reasoning_parts.append(f"当前只训练指定类别 {list(classes_value)}。")
        if planned_training_args.get('single_cls') is True:
            reasoning_parts.append('当前启用了 single_cls。')
        reasoning_summary = ' '.join(reasoning_parts[:5]).strip()

        return {
            'stage': 'training_plan',
            'status': 'ready_for_confirmation' if next_tool_name else 'discussion',
            'dataset_path': dataset_path,
            'data_summary': readiness.get('summary') or '',
            'preparable': readiness.get('preparable'),
            'primary_blocker_type': readiness.get('primary_blocker_type') or '',
            'execution_mode': execution_mode,
            'execution_backend': execution_backend,
            'custom_script': custom_script,
            'training_environment': env_name,
            'planned_training_args': planned_training_args,
            'advanced_details_requested': wants_training_advanced_details(user_text),
            'preflight_summary': preflight.get('summary') or '',
            'command_preview': list(preflight.get('command_preview') or []),
            'blockers': blockers,
            'warnings': warnings,
            'risks': risks,
            'reasoning_summary': reasoning_summary,
            'next_step_tool': next_tool_name,
            'next_step_args': next_tool_args,
            'editable_fields': ['model', 'epochs', 'batch', 'imgsz', 'device', 'training_environment', 'project', 'name', 'fraction', 'classes', 'single_cls', 'execution_mode', 'execution_backend'],
        }

    def _save_training_plan_draft(self, draft: dict[str, Any]) -> None:
        self.session_state.active_training.training_plan_draft = dict(draft)

    def _clear_training_plan_draft(self) -> None:
        self.session_state.active_training.training_plan_draft = {}

    def _render_training_plan_draft(self, draft: dict[str, Any], *, pending: bool) -> str:
        if not draft:
            return ''
        def _has_value(value: Any) -> bool:
            return value is not None and value != ''

        args = dict(draft.get('planned_training_args') or {})
        lines = ['训练计划草案：']
        dataset_path = str(draft.get('dataset_path') or '').strip()
        if dataset_path:
            lines.append(f'- 数据集: {dataset_path}')
        if draft.get('data_summary'):
            lines.append(f"- 当前判断: {draft.get('data_summary')}")
        if draft.get('reasoning_summary'):
            lines.append(f"- 计划依据: {draft.get('reasoning_summary')}")
        execution_mode_map = {
            'prepare_then_train': '先准备再训练',
            'direct_train': '直接训练',
            'prepare_only': '只做准备，暂不启动训练',
            'discussion_only': '先讨论方案，暂不执行',
            'blocked': '当前存在阻塞，先解决问题',
        }
        lines.append(f"- 执行方式: {execution_mode_map.get(str(draft.get('execution_mode') or ''), draft.get('execution_mode') or '未定')}")
        backend_map = {
            'standard_yolo': '标准 YOLO 训练',
            'custom_script': '自定义训练脚本',
            'custom_trainer': '自定义 Trainer',
        }
        lines.append(f"- 执行后端: {backend_map.get(str(draft.get('execution_backend') or ''), draft.get('execution_backend') or '标准 YOLO 训练')}")
        if draft.get('custom_script'):
            lines.append(f"- 自定义脚本: {draft.get('custom_script')}")
        core_bits: list[str] = []
        for key in ('model', 'data_yaml', 'epochs', 'batch', 'imgsz', 'device'):
            value = args.get(key)
            if not _has_value(value):
                continue
            display_key = 'data' if key == 'data_yaml' else key
            core_bits.append(f'{display_key}={value}')
        if core_bits:
            lines.append(f"- 核心参数: {', '.join(core_bits)}")
        output_bits: list[str] = []
        for key in ('project', 'name'):
            value = args.get(key)
            if not _has_value(value):
                continue
            output_bits.append(f'{key}={value}')
        if output_bits:
            lines.append(f"- 输出组织: {', '.join(output_bits)}")
        advanced_bits: list[str] = []
        for key in ('fraction', 'classes', 'single_cls', 'optimizer', 'freeze', 'resume', 'lr0', 'patience', 'workers', 'amp'):
            value = args.get(key)
            if not _has_value(value):
                continue
            advanced_bits.append(f'{key}={value}')
        if advanced_bits:
            lines.append(f"- 高级参数: {', '.join(advanced_bits)}")
        elif draft.get('advanced_details_requested'):
            lines.append("- 高级参数: 当前未显式指定，将使用运行时默认值")
        if draft.get('training_environment'):
            lines.append(f"- 训练环境: {draft.get('training_environment')}")
        if draft.get('preflight_summary'):
            lines.append(f"- 预检: {draft.get('preflight_summary')}")
        blockers = draft.get('blockers') or []
        warnings = draft.get('warnings') or []
        risks = draft.get('risks') or []
        if blockers:
            lines.append('- 当前阻塞:')
            lines.extend(f'  - {item}' for item in blockers[:2])
        elif risks:
            lines.append('- 主要风险:')
            lines.extend(f'  - {item}' for item in risks[:2])
        elif warnings:
            lines.append('- 主要风险:')
            lines.extend(f'  - {item}' for item in warnings[:2])
        next_tool_name = str(draft.get('next_step_tool') or '').strip()
        if next_tool_name:
            lines.append(f'- 下一步动作: {next_tool_name}')
        if pending:
            lines.append('你可以直接确认，也可以继续改参数、追问原因、改执行方式。')
        else:
            lines.append('你可以继续讨论、改参数；如果决定执行，我会再进入确认。')
        return '\n'.join(lines)

    async def _try_handle_training_plan_dialogue(self, user_text: str, thread_id: str) -> dict[str, Any] | None:
        draft = dict(self.session_state.active_training.training_plan_draft or {})
        pending = self._pending_from_state()
        if not draft and not pending:
            return None

        normalized = user_text.lower()
        clear_fields = self._collect_training_clear_fields(user_text)
        if (
            any(token in user_text for token in ('取消', '算了', '先不做', '不用了'))
            and not clear_fields
            and not any(token in user_text for token in ('取消了', '已经取消', '刚才'))
        ):
            if pending:
                return await self.confirm(thread_id, approved=False)
            self._clear_training_plan_draft()
            self.memory.save_state(self.session_state)
            return {'status': 'cancelled', 'message': '已取消当前训练计划草案。', 'tool_call': None}
        if (
            not pending
            and any(token in user_text for token in ('不要训练', '先不要训练', '不训练了'))
            and any(token in user_text for token in ('重新检查', '检查一下', '能不能直接训练', '是否能直接训练'))
        ):
            self._clear_training_plan_draft()
            self.memory.save_state(self.session_state)
            return None

        requested_execute = any(token in user_text for token in ('执行', '开始吧', '就这样', '确认', '可以开始', '开训', '启动吧')) or normalized.strip() in {'y', 'yes'}
        has_revision = any(
            token in normalized or token in user_text
            for token in (
                'batch', 'imgsz', 'device', 'epochs', '优化器', 'optimizer', '冻结', 'freeze', 'resume',
                'lr0', '学习率', 'patience', '早停', 'workers', '线程数', 'amp', '混合精度',
                '模型', '权重', 'project', '输出目录', 'name', '实验名', '运行名',
                'fraction', '全量数据', '抽样', 'classes', '类别', 'single_cls', '单类别',
                '环境', '为什么', '原因', '依据', '先只做准备', '只做准备', '标准 yolo', '自定义脚本', 'trainer',
                '高级参数', '高级配置', '展开参数', '详细参数',
            )
        ) or bool(self._extract_custom_training_script_from_text(user_text))
        if (
            any(token in user_text for token in ('先别执行', '先不要执行', '先别启动', '先不要启动', '先讨论', '先看看计划', '先给我计划'))
            and not has_revision
            and not requested_execute
        ):
            if draft:
                return {
                    'status': 'completed' if not pending else 'needs_confirmation',
                    'message': self._render_training_plan_draft(draft, pending=bool(pending)),
                    'tool_call': {'name': pending['name'], 'args': pending.get('args', {})} if pending else None,
                    'thread_id': thread_id if pending else None,
                }
            return None

        if requested_execute and not has_revision:
            if pending:
                return await self.confirm(thread_id, approved=True)
            next_tool_name = str(draft.get('next_step_tool') or '').strip()
            next_tool_args = dict(draft.get('next_step_args') or {})
            if not next_tool_name:
                return {'status': 'completed', 'message': self._render_training_plan_draft(draft, pending=False), 'tool_call': None}
            self._set_pending_confirmation(thread_id, {'name': next_tool_name, 'args': next_tool_args, 'id': None, 'synthetic': True})
            self.memory.save_state(self.session_state)
            return {
                'status': 'needs_confirmation',
                'message': self._render_training_plan_draft(draft, pending=True),
                'tool_call': {'name': next_tool_name, 'args': next_tool_args},
                'thread_id': thread_id,
            }

        if not has_revision:
            return None

        revised_draft = dict(draft or {})
        planned_args = dict(revised_draft.get('planned_training_args') or {})
        dataset_path = str(revised_draft.get('dataset_path') or self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir or '').strip()
        readiness = self.session_state.active_dataset.last_readiness or {}
        requested_args = self._collect_requested_training_args(
            user_text,
            data_yaml=str(planned_args.get('data_yaml') or self.session_state.active_dataset.data_yaml or ''),
        )
        for field in clear_fields:
            planned_args.pop(field, None)
        planned_args.update(
            {
                key: value
                for key, value in requested_args.items()
                if value is not None and value != ''
            }
        )
        execution_backend = self._extract_training_execution_backend_from_text(user_text)
        advanced_requested = wants_training_advanced_details(user_text) or bool(revised_draft.get('advanced_details_requested'))
        if any(token in user_text for token in ('只做准备', '只准备', '先准备不要训练')):
            revised_draft['execution_mode'] = 'prepare_only'
            revised_draft['next_step_tool'] = 'prepare_dataset_for_training'
        if any(token in user_text for token in ('不要自动划分', '不要划分', '不划分')):
            next_step_args = dict(revised_draft.get('next_step_args') or {})
            next_step_args.pop('force_split', None)
            revised_draft['next_step_args'] = next_step_args

        next_tool_name = str(revised_draft.get('next_step_tool') or (pending or {}).get('name') or '').strip()
        next_tool_args = dict(revised_draft.get('next_step_args') or (pending or {}).get('args') or {})
        execution_mode = str(revised_draft.get('execution_mode') or '').strip().lower()
        if execution_backend != 'standard_yolo':
            revised_draft = self._build_training_plan_draft(
                user_text=user_text,
                dataset_path=dataset_path,
                readiness=readiness,
                preflight={},
                next_tool_name='',
                next_tool_args={},
                planned_training_args=planned_args,
            )
            revised_draft['advanced_details_requested'] = advanced_requested
        elif (
            next_tool_name == 'start_training'
            or (execution_mode in {'direct_train', 'discussion_only', 'blocked'} and readiness.get('ready') and planned_args.get('model'))
        ):
            preflight = await self.direct_tool(
                'training_preflight',
                model=str(planned_args.get('model') or ''),
                data_yaml=str(planned_args.get('data_yaml') or ''),
                epochs=int(planned_args.get('epochs', 100)),
                device=str(planned_args.get('device', 'auto') or 'auto'),
                training_environment=str(planned_args.get('training_environment') or ''),
                project=str(planned_args.get('project') or ''),
                name=str(planned_args.get('name') or ''),
                batch=planned_args.get('batch'),
                imgsz=planned_args.get('imgsz'),
                fraction=planned_args.get('fraction'),
                classes=planned_args.get('classes'),
                single_cls=planned_args.get('single_cls'),
                optimizer=str(planned_args.get('optimizer', '') or ''),
                freeze=planned_args.get('freeze'),
                resume=planned_args.get('resume'),
                lr0=planned_args.get('lr0'),
                patience=planned_args.get('patience'),
                workers=planned_args.get('workers'),
                amp=planned_args.get('amp'),
            )
            revised_draft = self._build_training_plan_draft(
                user_text=user_text,
                dataset_path=dataset_path,
                readiness=readiness,
                preflight=preflight,
                next_tool_name='start_training' if preflight.get('ready_to_start') else '',
                next_tool_args={
                    'model': str((preflight.get('resolved_args') or {}).get('model') or planned_args.get('model') or ''),
                    'data_yaml': str((preflight.get('resolved_args') or {}).get('data_yaml') or planned_args.get('data_yaml') or ''),
                    'epochs': int((preflight.get('resolved_args') or {}).get('epochs') or planned_args.get('epochs', 100)),
                    'device': str((preflight.get('resolved_args') or {}).get('device') or planned_args.get('device') or 'auto'),
                    'training_environment': str((preflight.get('resolved_args') or {}).get('training_environment') or planned_args.get('training_environment') or ''),
                    'project': str((preflight.get('resolved_args') or {}).get('project') or planned_args.get('project') or ''),
                    'name': str((preflight.get('resolved_args') or {}).get('name') or planned_args.get('name') or ''),
                    'batch': (preflight.get('resolved_args') or {}).get('batch', planned_args.get('batch')),
                    'imgsz': (preflight.get('resolved_args') or {}).get('imgsz', planned_args.get('imgsz')),
                    'fraction': (preflight.get('resolved_args') or {}).get('fraction', planned_args.get('fraction')),
                    'classes': (preflight.get('resolved_args') or {}).get('classes', planned_args.get('classes')),
                    'single_cls': (preflight.get('resolved_args') or {}).get('single_cls', planned_args.get('single_cls')),
                    'optimizer': str((preflight.get('resolved_args') or {}).get('optimizer') or planned_args.get('optimizer') or ''),
                    'freeze': (preflight.get('resolved_args') or {}).get('freeze', planned_args.get('freeze')),
                    'resume': (preflight.get('resolved_args') or {}).get('resume', planned_args.get('resume')),
                    'lr0': (preflight.get('resolved_args') or {}).get('lr0', planned_args.get('lr0')),
                    'patience': (preflight.get('resolved_args') or {}).get('patience', planned_args.get('patience')),
                    'workers': (preflight.get('resolved_args') or {}).get('workers', planned_args.get('workers')),
                    'amp': (preflight.get('resolved_args') or {}).get('amp', planned_args.get('amp')),
                } if preflight.get('ready_to_start') else {},
                planned_training_args={
                    'model': str((preflight.get('resolved_args') or {}).get('model') or planned_args.get('model') or ''),
                    'data_yaml': str((preflight.get('resolved_args') or {}).get('data_yaml') or planned_args.get('data_yaml') or ''),
                    'epochs': int((preflight.get('resolved_args') or {}).get('epochs') or planned_args.get('epochs', 100)),
                    'device': str((preflight.get('resolved_args') or {}).get('device') or planned_args.get('device') or 'auto'),
                    'training_environment': str((preflight.get('resolved_args') or {}).get('training_environment') or planned_args.get('training_environment') or ''),
                    'project': str((preflight.get('resolved_args') or {}).get('project') or planned_args.get('project') or ''),
                    'name': str((preflight.get('resolved_args') or {}).get('name') or planned_args.get('name') or ''),
                    'batch': (preflight.get('resolved_args') or {}).get('batch', planned_args.get('batch')),
                    'imgsz': (preflight.get('resolved_args') or {}).get('imgsz', planned_args.get('imgsz')),
                    'fraction': (preflight.get('resolved_args') or {}).get('fraction', planned_args.get('fraction')),
                    'classes': (preflight.get('resolved_args') or {}).get('classes', planned_args.get('classes')),
                    'single_cls': (preflight.get('resolved_args') or {}).get('single_cls', planned_args.get('single_cls')),
                    'optimizer': str((preflight.get('resolved_args') or {}).get('optimizer') or planned_args.get('optimizer') or ''),
                    'freeze': (preflight.get('resolved_args') or {}).get('freeze', planned_args.get('freeze')),
                    'resume': (preflight.get('resolved_args') or {}).get('resume', planned_args.get('resume')),
                    'lr0': (preflight.get('resolved_args') or {}).get('lr0', planned_args.get('lr0')),
                    'patience': (preflight.get('resolved_args') or {}).get('patience', planned_args.get('patience')),
                    'workers': (preflight.get('resolved_args') or {}).get('workers', planned_args.get('workers')),
                    'amp': (preflight.get('resolved_args') or {}).get('amp', planned_args.get('amp')),
                },
            )
            revised_draft['advanced_details_requested'] = advanced_requested
        elif readiness.get('preparable'):
            prepare_args = dict(next_tool_args)
            prepare_args.setdefault('dataset_path', dataset_path)
            revised_draft = self._build_training_plan_draft(
                user_text=user_text,
                dataset_path=dataset_path,
                readiness=readiness,
                preflight={},
                next_tool_name='prepare_dataset_for_training',
                next_tool_args=prepare_args,
                planned_training_args=planned_args,
            )
            revised_draft['advanced_details_requested'] = advanced_requested
        else:
            revised_draft['planned_training_args'] = planned_args
            if next_tool_name:
                revised_draft['next_step_tool'] = next_tool_name
            revised_draft['advanced_details_requested'] = advanced_requested
        self._save_training_plan_draft(revised_draft)
        if pending and revised_draft.get('next_step_tool'):
            self._set_pending_confirmation(
                thread_id,
                {
                    'name': str(revised_draft.get('next_step_tool')),
                    'args': dict(revised_draft.get('next_step_args') or {}),
                    'id': None,
                    'synthetic': True,
                },
            )
            self.memory.save_state(self.session_state)
            return {
                'status': 'needs_confirmation',
                'message': self._render_training_plan_draft(revised_draft, pending=True),
                'tool_call': {
                    'name': str(revised_draft.get('next_step_tool')),
                    'args': dict(revised_draft.get('next_step_args') or {}),
                },
                'thread_id': thread_id,
            }
        self.memory.save_state(self.session_state)
        return {
            'status': 'completed',
            'message': self._render_training_plan_draft(revised_draft, pending=False),
            'tool_call': None,
        }

    def _build_confirmation_prompt(self, tool_call: dict[str, Any]) -> str:
        args = tool_call.get("args", {})
        tool_name = str(tool_call.get('name') or '')
        ds = self.session_state.active_dataset
        tr = self.session_state.active_training
        plan_draft = tr.training_plan_draft or {}

        if plan_draft and str(plan_draft.get('next_step_tool') or '').strip() == tool_name:
            return self._render_training_plan_draft(plan_draft, pending=True)

        if tool_name == 'prepare_dataset_for_training':
            lines = ['准备执行：数据准备']
            dataset_path = str(args.get('dataset_path') or ds.dataset_root or ds.img_dir or '').strip()
            if dataset_path:
                lines.append(f'数据集: {dataset_path}')
            readiness = ds.last_readiness or {}
            if readiness.get('summary'):
                lines.append(f"当前判断: {readiness.get('summary')}")
            if readiness.get('primary_blocker_type'):
                lines.append(f"主要阻塞: {readiness.get('primary_blocker_type')}")
            if readiness.get('preparable'):
                lines.append('初步安排: 自动补齐训练产物')
            if args.get('force_split'):
                lines.append('附加安排: 按默认比例划分数据')
            lines.append('确认执行？(y/n)')
            return '\n'.join(lines)

        if tool_name == 'start_training':
            lines = ['准备执行：启动训练']
            readiness = ds.last_readiness or {}
            if readiness.get('summary'):
                lines.append(f"数据理解: {readiness.get('summary')}")
            preflight = tr.last_preflight or {}
            environment = preflight.get('training_environment') or tr.last_environment_probe.get('default_environment') or {}
            env_name = environment.get('display_name') or environment.get('name')
            if env_name:
                lines.append(f'训练环境: {env_name}')
            model = args.get('model') or tr.model
            data_yaml = args.get('data_yaml') or tr.data_yaml or ds.data_yaml
            epochs = args.get('epochs') or (preflight.get('resolved_args') or {}).get('epochs') or 100
            device = args.get('device') or (preflight.get('resolved_args') or {}).get('device') or 'auto'
            batch = args.get('batch')
            if batch is None:
                batch = (preflight.get('resolved_args') or {}).get('batch')
            imgsz = args.get('imgsz')
            if imgsz is None:
                imgsz = (preflight.get('resolved_args') or {}).get('imgsz')
            project = args.get('project')
            if not project:
                project = (preflight.get('resolved_args') or {}).get('project')
            run_name = args.get('name')
            if not run_name:
                run_name = (preflight.get('resolved_args') or {}).get('name')
            fraction = args.get('fraction')
            if fraction is None:
                fraction = (preflight.get('resolved_args') or {}).get('fraction')
            classes = args.get('classes')
            if classes is None:
                classes = (preflight.get('resolved_args') or {}).get('classes')
            single_cls = args.get('single_cls')
            if single_cls is None:
                single_cls = (preflight.get('resolved_args') or {}).get('single_cls')
            plan_bits = [f'model={model}', f'data={data_yaml}', f'epochs={epochs}', f'device={device}']
            if batch is not None:
                plan_bits.append(f'batch={batch}')
            if imgsz is not None:
                plan_bits.append(f'imgsz={imgsz}')
            lines.append(f"初步安排: {', '.join(str(item) for item in plan_bits)}")
            output_bits = []
            if project:
                output_bits.append(f'project={project}')
            if run_name:
                output_bits.append(f'name={run_name}')
            if output_bits:
                lines.append(f"输出组织: {', '.join(output_bits)}")
            advanced_bits = []
            for key, value in (('fraction', fraction), ('classes', classes), ('single_cls', single_cls)):
                if value is not None and value != '':
                    advanced_bits.append(f'{key}={value}')
            if advanced_bits:
                lines.append(f"高级参数: {', '.join(advanced_bits)}")
            if preflight.get('summary'):
                lines.append(f"预检: {preflight.get('summary')}")
            command_preview = preflight.get('command_preview') or []
            if command_preview:
                preview_text = ' '.join(str(item) for item in command_preview[:6])
                if len(command_preview) > 6:
                    preview_text += ' ...'
                lines.append(f'命令预览: {preview_text}')
            lines.append('确认执行？(y/n)')
            return '\n'.join(lines)

        pretty_args = "\n".join(f"  - {k}: {v}" for k, v in args.items()) or "  - 无参数"
        return (
            f"检测到高风险操作：{tool_name}\n"
            f"参数摘要：\n{pretty_args}\n"
            "确认执行？(y/n)"
        )

    @staticmethod
    def _build_cancel_message(tool_call: dict[str, Any]) -> str:
        return f"已取消操作：{tool_call['name']}。当前计划已保留，你可以继续追问原因、调整参数后重新确认。"

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

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from dataclasses import dataclass
import inspect
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from yolostudio_agent.agent.client.file_checkpointer import FileCheckpointSaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from yolostudio_agent.agent.client.context_builder import ContextBuilder
from yolostudio_agent.agent.client.event_retriever import EventRetriever
from yolostudio_agent.agent.client.grounded_reply_builder import build_grounded_tool_reply
from yolostudio_agent.agent.client.state_applier import apply_tool_result_to_state
from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.llm_factory import (
    LlmProviderSettings,
    build_llm,
    provider_summary,
    resolve_llm_settings,
)
from yolostudio_agent.agent.client.mcp_connection import build_mcp_connection_config
from yolostudio_agent.agent.client.memory_store import MemoryStore
from yolostudio_agent.agent.client.remote_transfer_tools import build_local_transfer_tools
from yolostudio_agent.agent.client.session_state import SessionState, utc_now
from yolostudio_agent.agent.client.tool_adapter import (
    adapt_tools_for_chat_model,
    canonical_tool_name,
    normalize_tool_args,
    stringify_tool_result_facts,
)
from yolostudio_agent.agent.client.tool_result_parser import parse_tool_message

SYSTEM_PROMPT = """你是 YoloStudio Agent，负责帮助用户解决数据准备、训练、预测和远端传输问题。

工作方式：
1. 优先使用工具拿事实，再回答；不凭空猜测文件、数据集、训练或预测状态。
2. 工具选择主要依赖工具定义、参数模式和当前上下文；优先使用最少必要工具。
3. 如果用户已经给了明确路径、模型路径、报告路径或远端路径，直接复用，不要重复追问。
4. 最终回答必须由你自己组织成自然中文；除非用户明确要求调试细节，否则不要输出工具名、字段名、原始 JSON、命令 payload 或伪代码式调用示例。
5. 会修改数据、上传文件或启动长任务时，不要先在自然语言里自作主张执行；当参数足够时生成工具调用，由外部确认流程拦截。
6. 如果工具失败，直接解释失败原因，并告诉用户下一步最实际的动作。

关键边界：
- dataset_training_readiness：只判断数据集本身是否已经具备直接训练的结构条件，不检查 GPU、device 或训练环境。
- training_readiness：只在用户准备现在启动训练，或明确询问执行条件（GPU / device / 训练环境）时使用。
- prepare_dataset_for_training：用于先准备数据、补齐 data.yaml、必要时划分 train/val。

回答要求：
- 先给结论，再给原因，再给下一步建议。
- 只有用户明确要求时，才展开参数细节、工具名或 JSON。
- 如果问题本身不需要工具，直接回答。"""

HIGH_RISK_TOOLS = {
    "start_training",
    "start_training_loop",
    "split_dataset",
    "augment_dataset",
    "prepare_dataset_for_training",
    "convert_format",
    "modify_labels",
    "clean_orphan_labels",
    "generate_empty_labels",
    "generate_missing_labels",
    "categorize_by_class",
    "upload_assets_to_remote",
}

GPU_SENSITIVE_TOOLS = {
    "check_gpu_status",
    "prepare_dataset_for_training",
    "training_readiness",
    "training_preflight",
    "start_training",
    "start_training_loop",
}

SYNTHETIC_TOOL_SURFACE_METADATA = {
    "remote_prediction_pipeline": {
        "read_only": False,
        "destructive": False,
        "confirmation_required": True,
        "open_world": True,
        "risk_level": "medium",
    },
    "remote_training_pipeline": {
        "read_only": False,
        "destructive": False,
        "confirmation_required": True,
        "open_world": True,
        "risk_level": "high",
    },
}


def _raw_tool_surface_metadata(tool: Any) -> dict[str, Any]:
    return dict(
        getattr(tool, 'metadata', None)
        or getattr(tool, 'tool_metadata', None)
        or {}
    )


def _raw_tool_surface_annotations(tool: Any) -> dict[str, Any]:
    annotations = getattr(tool, 'annotations', None) or getattr(tool, 'tool_annotations', None)
    if annotations is None:
        return {}
    if isinstance(annotations, dict):
        return dict(annotations)
    values: dict[str, Any] = {}
    for attr_name in ('readOnlyHint', 'destructiveHint', 'idempotentHint', 'openWorldHint'):
        value = getattr(annotations, attr_name, None)
        if value is not None:
            values[attr_name] = value
    return values


def _raw_tool_requires_confirmation(tool: Any) -> bool:
    name = canonical_tool_name(getattr(tool, 'name', ''))
    metadata = _raw_tool_surface_metadata(tool)
    if 'confirmation_required' in metadata:
        return bool(metadata.get('confirmation_required'))
    if bool(metadata.get('destructive')):
        return True
    annotations = _raw_tool_surface_annotations(tool)
    if 'destructiveHint' in annotations and bool(annotations.get('destructiveHint')):
        return True
    return name in HIGH_RISK_TOOLS


def _build_manual_interrupt_tool_names(raw_tools: list[Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for tool in raw_tools:
        name = canonical_tool_name(getattr(tool, 'name', ''))
        if not name or name in seen:
            continue
        if _raw_tool_requires_confirmation(tool):
            seen.add(name)
            names.append(name)
    return names


@dataclass(slots=True)
class AgentSettings:
    provider: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float | None = None
    confirmation_mode: str = os.getenv("YOLOSTUDIO_CONFIRMATION_MODE", "manual")
    ollama_url: str = os.getenv("YOLOSTUDIO_OLLAMA_URL", "http://127.0.0.1:11434")
    mcp_url: str = os.getenv("YOLOSTUDIO_MCP_URL", "http://127.0.0.1:8080/mcp")
    max_history_messages: int = int(os.getenv("YOLOSTUDIO_MAX_HISTORY_MESSAGES", "12"))
    session_id: str = os.getenv("YOLOSTUDIO_SESSION_ID", "default")
    memory_root: str = os.getenv("YOLOSTUDIO_MEMORY_ROOT", str(Path(__file__).resolve().parents[2] / "memory"))

    def to_llm_settings(self, *, role: str = 'primary', inherit: LlmProviderSettings | None = None) -> LlmProviderSettings:
        raw = LlmProviderSettings(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            role=role,
        )
        resolved = resolve_llm_settings(raw, role=role, inherit=inherit)
        if resolved.provider == 'ollama' and not resolved.base_url:
            resolved.base_url = self.ollama_url
        return resolved


class YoloStudioAgentClient:
    def __init__(
        self,
        graph: Any,
        settings: AgentSettings,
        tool_registry: dict[str, Any],
        planner_llm: Any | None = None,
        *,
        primary_llm_settings: LlmProviderSettings | None = None,
        helper_llm_settings: LlmProviderSettings | None = None,
    ) -> None:
        self.graph = graph
        self.settings = settings
        self.tool_registry = tool_registry
        self.planner_llm = planner_llm
        self.primary_llm_settings = primary_llm_settings or settings.to_llm_settings(role='primary')
        self.helper_llm_settings = helper_llm_settings or settings.to_llm_settings(
            role='helper',
            inherit=self.primary_llm_settings,
        )
        self._messages: list[BaseMessage] = []
        self._turn_index = 0
        self._applied_tool_call_ids: set[str] = set()
        self.memory = MemoryStore(settings.memory_root)
        self.context_builder = ContextBuilder(SYSTEM_PROMPT)
        self.event_retriever = EventRetriever(self.memory)
        self.session_state: SessionState = self.memory.load_state(settings.session_id)
        self._clear_stale_startup_state()
        self._sync_preferences()
        self.memory.save_state(self.session_state)
        self._record_llm_runtime_config()

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def preview(self) -> str:
        return (
            f"YoloStudio Agent 已就绪 ({self.primary_llm_settings.model})\n"
            f"MCP Server: {self.settings.mcp_url} | LLM: {provider_summary(self.primary_llm_settings)}\n"
            f"Session: {self.session_state.session_id} | ConfirmMode: {self._confirmation_mode()}"
        )

    def _confirmation_mode(self) -> str:
        mode = str(self.settings.confirmation_mode or "manual").strip().lower()
        return mode if mode in {"manual", "auto"} else "manual"

    def _auto_confirmation_enabled(self) -> bool:
        return self._confirmation_mode() == "auto"

    def _uses_local_ollama(self) -> bool:
        provider = str(self.primary_llm_settings.provider or '').strip().lower()
        return provider == 'ollama'

    @staticmethod
    def _llm_settings_payload(settings: LlmProviderSettings) -> dict[str, Any]:
        return {
            'role': settings.role,
            'provider': settings.provider,
            'model': settings.model,
            'base_url': settings.base_url,
            'api_key_configured': bool(settings.api_key),
            'temperature': settings.temperature,
            'ollama_keep_alive': settings.ollama_keep_alive,
        }

    def _record_llm_runtime_config(self) -> None:
        payload = {
            'primary': self._llm_settings_payload(self.primary_llm_settings),
            'helper': self._llm_settings_payload(self.helper_llm_settings),
            'helper_reuses_primary': bool(
                self.planner_llm is not None
                and self.primary_llm_settings == self.helper_llm_settings
            ),
        }
        self.memory.append_event(self.session_state.session_id, 'llm_runtime_config', payload)

    @staticmethod
    def _parse_gpu_compute_processes(output: str) -> list[dict[str, str]]:
        if not output:
            return []
        processes: list[dict[str, str]] = []
        for line in output.splitlines():
            parts = [item.strip() for item in line.split(',', 2)]
            if len(parts) < 2:
                continue
            processes.append(
                {
                    'pid': parts[0],
                    'process_name': parts[1],
                    'used_gpu_memory': parts[2] if len(parts) > 2 else '',
                }
            )
        return processes

    async def _gpu_compute_processes(self) -> list[dict[str, str]]:
        try:
            process = await asyncio.create_subprocess_exec(
                'nvidia-smi',
                '--query-compute-apps=pid,process_name,used_gpu_memory',
                '--format=csv,noheader,nounits',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception:
            return []
        try:
            stdout, _ = await process.communicate()
        except Exception:
            with contextlib.suppress(Exception):
                process.kill()
            return []
        if process.returncode not in (0, None):
            return []
        output = stdout.decode('utf-8', errors='ignore').strip()
        return self._parse_gpu_compute_processes(output)

    async def _wait_for_local_llm_gpu_release(self, *, timeout_s: float = 8.0, poll_interval_s: float = 0.5) -> None:
        if not self._uses_local_ollama():
            return
        deadline = time.time() + max(timeout_s, 0.0)
        observed_busy = False
        while True:
            busy = [
                item
                for item in await self._gpu_compute_processes()
                if 'ollama' in str(item.get('process_name') or '').lower()
            ]
            if not busy:
                if observed_busy:
                    self.memory.append_event(
                        self.session_state.session_id,
                        'ollama_gpu_release_wait',
                        {'released': True},
                    )
                return
            observed_busy = True
            if time.time() >= deadline:
                self.memory.append_event(
                    self.session_state.session_id,
                    'ollama_gpu_release_wait',
                    {'released': False, 'busy_processes': busy},
                )
                return
            await asyncio.sleep(max(poll_interval_s, 0.1))

    async def _maybe_auto_progress(
        self,
        result: dict[str, Any] | None,
        stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any] | None:
        current = result
        steps = 0
        while (
            current
            and current.get("status") == "needs_confirmation"
            and self._auto_confirmation_enabled()
        ):
            thread_id = str(current.get("thread_id") or "").strip()
            if not thread_id:
                break
            steps += 1
            if steps > 8:
                self.memory.append_event(
                    self.session_state.session_id,
                    "auto_confirmation_abort",
                    {"reason": "too_many_confirmation_steps", "steps": steps},
                )
                break
            current = await self.confirm(thread_id, approved=True, stream_handler=stream_handler)
        return current

    async def _execute_adapted_pending_tool(
        self,
        *,
        thread_id: str,
        pending: dict[str, Any],
        approved: bool = False,
        auto_progress_followups: bool = False,
        stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        parsed = await self.direct_tool(pending["name"], **pending.get("args", {}))
        final_text = await self._render_tool_result_message(pending['name'], parsed)
        self._messages.append(AIMessage(content=final_text))
        self._trim_history()
        self.memory.save_state(self.session_state)

        if pending.get('name') == 'prepare_dataset_for_training':
            post_prepare_followup = await self._handle_post_prepare_confirmation_followup(
                thread_id=thread_id,
                prepare_parsed=parsed,
            )
            if post_prepare_followup is not None:
                if auto_progress_followups and post_prepare_followup.get('status') == 'needs_confirmation':
                    next_thread_id = str(post_prepare_followup.get('thread_id') or thread_id).strip()
                    if next_thread_id:
                        return await self.confirm(next_thread_id, approved=True, stream_handler=stream_handler)
                return post_prepare_followup

        result = {
            "status": "completed",
            "message": final_text,
            "tool_call": pending,
        }
        if approved:
            result["approved"] = True
        return result

    def _clear_stale_startup_state(self) -> None:
        pending = self.session_state.pending_confirmation
        if not str(pending.tool_name or '').strip():
            self._clear_training_plan_draft()

    @staticmethod
    def _strip_ephemeral_context(state: SessionState) -> SessionState:
        ds = state.active_dataset
        tr = state.active_training
        pred = state.active_prediction
        kn = state.active_knowledge
        rt = state.active_remote_transfer

        ds.last_scan = {}
        ds.last_validate = {}
        ds.last_readiness = {}
        ds.last_split = {}
        ds.last_health_check = {}
        ds.last_duplicate_check = {}
        ds.last_extract_preview = {}
        ds.last_extract_result = {}
        ds.last_video_scan = {}
        ds.last_frame_extract = {}

        tr.last_status = {}
        tr.last_summary = {}
        tr.training_run_summary = {}
        tr.last_start_result = {}
        tr.last_environment_probe = {}
        tr.last_preflight = {}
        tr.recent_runs = []
        tr.last_run_inspection = {}
        tr.last_run_comparison = {}
        tr.best_run_selection = {}
        tr.training_plan_draft = {}
        tr.last_remote_roundtrip = {}
        tr.last_loop_status = {}
        tr.last_loop_detail = {}
        tr.recent_loops = []

        pred.last_result = {}
        pred.last_summary = {}
        pred.last_inspection = {}
        pred.last_export = {}
        pred.last_path_lists = {}
        pred.last_organized_result = {}
        pred.last_realtime_status = {}
        pred.last_remote_roundtrip = {}

        kn.last_retrieval = {}
        kn.last_analysis = {}
        kn.last_recommendation = {}

        rt.last_profile_listing = {}
        rt.last_upload = {}
        rt.last_download = {}
        return state

    def _should_reuse_history_context(self, user_text: str) -> bool:
        if self._explicitly_references_previous_context(user_text):
            return True
        if str(self.session_state.pending_confirmation.tool_name or '').strip():
            return True
        if self.session_state.active_training.training_plan_draft:
            return True
        return False

    def _state_for_model(self, user_text: str) -> tuple[SessionState, bool]:
        reuse_history = self._should_reuse_history_context(user_text)
        if reuse_history:
            return self.session_state, True
        cloned = SessionState.from_dict(self.session_state.to_dict())
        return self._strip_ephemeral_context(cloned), False

    async def direct_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        state_mode = str(kwargs.pop('_state_mode', 'persistent') or 'persistent').strip().lower()
        canonical_name = canonical_tool_name(tool_name)
        normalized_args = normalize_tool_args(canonical_name, kwargs)
        if canonical_name in GPU_SENSITIVE_TOOLS and self._local_llm_gpu_wait_enabled():
            await self._wait_for_local_llm_gpu_release()
        tool = self.tool_registry.get(canonical_name)
        if not tool:
            return {"ok": False, "error": f"未找到工具: {canonical_name}"}
        payload = await tool.ainvoke(normalized_args)
        parsed = self._normalize_tool_output(payload)
        if state_mode == 'observe':
            return parsed
        self.memory.append_event(self.session_state.session_id, "tool_result", {"tool": canonical_name, "args": normalized_args, "result": parsed})
        self._apply_to_state(canonical_name, parsed, normalized_args)
        if canonical_name in {'start_training', 'start_training_loop'} and parsed.get('ok'):
            self._clear_training_plan_draft()
        elif canonical_name == 'prepare_dataset_for_training' and parsed.get('ok'):
            draft = self.session_state.active_training.training_plan_draft or {}
            if str(draft.get('execution_mode') or '').strip().lower() == 'prepare_only':
                self._clear_training_plan_draft()
        self._record_secondary_event(canonical_name, parsed)
        self.memory.save_state(self.session_state)
        return parsed

    def _tool_surface_metadata(self, tool_name: str) -> dict[str, Any]:
        canonical_name = canonical_tool_name(tool_name)
        tool = self.tool_registry.get(canonical_name)
        metadata = dict(
            getattr(tool, 'metadata', None)
            or getattr(tool, 'tool_metadata', None)
            or {}
        )
        if metadata:
            return metadata
        return dict(SYNTHETIC_TOOL_SURFACE_METADATA.get(canonical_name) or {})

    def _tool_surface_annotations(self, tool_name: str) -> dict[str, Any]:
        canonical_name = canonical_tool_name(tool_name)
        tool = self.tool_registry.get(canonical_name)
        annotations = getattr(tool, 'annotations', None) or getattr(tool, 'tool_annotations', None)
        if annotations is None:
            return {}
        if isinstance(annotations, dict):
            return dict(annotations)
        values: dict[str, Any] = {}
        for attr_name in ('readOnlyHint', 'destructiveHint', 'idempotentHint', 'openWorldHint'):
            value = getattr(annotations, attr_name, None)
            if value is not None:
                values[attr_name] = value
        return values

    def _tool_is_read_only(self, tool_name: str) -> bool:
        metadata = self._tool_surface_metadata(tool_name)
        if 'read_only' in metadata:
            return bool(metadata.get('read_only'))
        annotations = self._tool_surface_annotations(tool_name)
        if 'readOnlyHint' in annotations:
            return bool(annotations.get('readOnlyHint'))
        return False

    def _tool_is_destructive(self, tool_name: str) -> bool:
        metadata = self._tool_surface_metadata(tool_name)
        if 'destructive' in metadata:
            return bool(metadata.get('destructive'))
        annotations = self._tool_surface_annotations(tool_name)
        if 'destructiveHint' in annotations:
            return bool(annotations.get('destructiveHint'))
        return False

    def _tool_requires_confirmation(self, tool_name: str) -> bool:
        canonical_name = canonical_tool_name(tool_name)
        metadata = self._tool_surface_metadata(canonical_name)
        explicit = metadata.get('confirmation_required')
        if explicit is not None:
            return bool(explicit)
        if self._tool_is_destructive(canonical_name):
            return True
        if self._tool_is_read_only(canonical_name):
            return False
        return canonical_name in HIGH_RISK_TOOLS

    @staticmethod
    def _local_llm_gpu_wait_enabled() -> bool:
        value = str(os.getenv('YOLOSTUDIO_LOCAL_LLM_GPU_WAIT', '') or '').strip().lower()
        return value in {'1', 'true', 'yes', 'on'}

    def _tool_risk_level(self, tool_name: str) -> str:
        canonical_name = canonical_tool_name(tool_name)
        metadata = self._tool_surface_metadata(canonical_name)
        explicit = str(metadata.get('risk_level') or '').strip().lower()
        if explicit in {'low', 'medium', 'high'}:
            return explicit
        if self._tool_requires_confirmation(canonical_name):
            return 'high'
        if self._tool_is_read_only(canonical_name):
            return 'low'
        return 'medium'

    async def chat(self, user_text: str, auto_approve: bool = False, stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None) -> dict[str, Any]:
        if not str(user_text).strip():
            return {"status": "completed", "message": "请输入内容。", "tool_call": None}
        self._messages.append(HumanMessage(content=user_text))
        self._turn_index += 1
        thread_id = f"{self.session_state.session_id}-turn-{self._turn_index}"
        config = {"configurable": {"thread_id": thread_id}}

        pending_dialogue = await self._try_handle_pending_confirmation_dialogue(user_text, stream_handler=stream_handler)
        if pending_dialogue is not None:
            self._trim_history()
            self.memory.save_state(self.session_state)
            progressed = await self._maybe_auto_progress(pending_dialogue, stream_handler=stream_handler)
            return progressed or pending_dialogue

        routed = await self._try_handle_mainline_intent(user_text, thread_id)
        if routed is not None:
            self._trim_history()
            self.memory.save_state(self.session_state)
            progressed = await self._maybe_auto_progress(routed, stream_handler=stream_handler)
            return progressed or routed

        unresolved_pending = self._pending_from_state()
        if unresolved_pending is not None:
            pending_thread_id = self._pending_confirmation_thread_id()
            last_decision = str((unresolved_pending.get('decision_context') or {}).get('decision') or '').strip()
            if last_decision == 'edit':
                reply = (
                    '我已经保留这一步待执行动作，但这句还不足以直接改完参数。'
                    '你可以继续明确要改哪一项，例如“不要自动划分”、“batch 改成 12”或“环境改成 base”。'
                )
            elif last_decision == 'clarify':
                reply = await self._build_confirmation_message(unresolved_pending)
            else:
                reply = await self._build_confirmation_message(unresolved_pending)
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return self._needs_confirmation_result(pending_thread_id, unresolved_pending, reply)

        self._trim_history()
        state_for_model, include_history_context = self._state_for_model(user_text)
        digest = self.event_retriever.build_digest(
            self.session_state.session_id,
            state_for_model,
            include_history_context=include_history_context,
        )
        built_messages = self.context_builder.build_messages(state_for_model, self._messages, digest=digest)
        result = await self._graph_invoke({"messages": built_messages}, config=config, stream_handler=stream_handler)

        while True:
            pending = self._get_pending_tool_call(config)
            if not pending:
                break
            if self._tool_requires_confirmation(pending["name"]) and not (auto_approve or self._auto_confirmation_enabled()):
                self._set_pending_confirmation(thread_id, pending)
                self.memory.save_state(self.session_state)
                return self._needs_confirmation_result(thread_id, pending, await self._build_confirmation_message(pending))
            if pending.get('adapted'):
                return await self._execute_adapted_pending_tool(
                    thread_id=thread_id,
                    pending=pending,
                    auto_progress_followups=bool(auto_approve or self._auto_confirmation_enabled()),
                    stream_handler=stream_handler,
                )
            result = await self._graph_invoke(Command(resume="approved"), config=config, stream_handler=stream_handler)

        applied_results = self._apply_tool_results(result["messages"], built_messages_len=len(built_messages))
        final_text = self._compose_final_reply(result["messages"], applied_results)
        self._messages.append(AIMessage(content=final_text))
        self._trim_history()
        self.memory.save_state(self.session_state)
        return {
            "status": "completed",
            "message": final_text,
            "tool_call": None,
        }

    async def review_pending_action(
        self,
        decision_payload: dict[str, Any],
        *,
        stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        pending = self._pending_from_state()
        if not pending:
            return {"status": "error", "message": "当前没有待确认的高风险操作。"}

        normalized_decision = self._normalize_confirmation_reply_decision(decision_payload.get('decision'))
        if normalized_decision == 'deny':
            normalized_decision = 'reject'
        if normalized_decision not in {'approve', 'reject', 'edit', 'clarify', 'unclear', 'restate'}:
            normalized_decision = 'unclear'

        pending_thread_id = self._pending_confirmation_thread_id()
        self._record_pending_action_review(
            pending,
            decision=normalized_decision,
            reason=str(decision_payload.get('reason') or '').strip(),
            raw_user_text=str(decision_payload.get('raw_user_text') or '').strip(),
            source=str(decision_payload.get('source') or 'runtime_review').strip(),
            edits=dict(decision_payload.get('edits') or {}),
        )

        if normalized_decision == 'approve':
            return await self.confirm(pending_thread_id, approved=True, stream_handler=stream_handler)
        if normalized_decision == 'reject':
            return await self.confirm(pending_thread_id, approved=False, stream_handler=stream_handler)
        if normalized_decision == 'restate':
            self.memory.save_state(self.session_state)
            return self._needs_confirmation_result(
                pending_thread_id,
                self._pending_from_state() or pending,
                await self._build_confirmation_message(pending),
            )
        if normalized_decision == 'clarify':
            self.memory.save_state(self.session_state)
            return self._needs_confirmation_result(
                pending_thread_id,
                self._pending_from_state() or pending,
                await self._build_confirmation_message(self._pending_from_state() or pending),
            )
        if normalized_decision == 'edit':
            self.memory.save_state(self.session_state)
            return self._needs_confirmation_result(
                pending_thread_id,
                self._pending_from_state() or pending,
                '我先保留当前待执行动作。你可以直接继续改参数、换环境、换模型，'
                '等你改完后我会基于新的事实重新给出可执行方案。',
            )
        self.memory.save_state(self.session_state)
        return self._needs_confirmation_result(
            pending_thread_id,
            self._pending_from_state() or pending,
            '我先不擅自执行。你可以直接说“继续执行”、'
            '“先不要做”、'
            '“把 batch 改成 12 再继续”、'
            '或继续追问这一步为什么要做。',
        )

    async def confirm(self, thread_id: str, approved: bool, stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        graph_pending = self._get_pending_tool_call(config)
        pending = graph_pending or self._pending_from_state()
        if not pending:
            return {"status": "error", "message": "当前没有待确认的高风险操作。"}
        pending = dict(pending)
        decision_context = dict(self.session_state.pending_confirmation.decision_context or {})
        if decision_context:
            pending['decision_context'] = decision_context

        if not approved:
            cancel_message = self._build_cancel_message(pending)
            self._messages.append(AIMessage(content=cancel_message))
            self._clear_pending_confirmation()
            self._trim_history()
            self.memory.append_event(self.session_state.session_id, "confirmation_cancelled", {"tool": pending["name"], "args": pending.get("args", {})})
            self.memory.save_state(self.session_state)
            return self._cancelled_result(pending, cancel_message)

        self.memory.append_event(self.session_state.session_id, "confirmation_approved", {"tool": pending["name"], "args": pending.get("args", {})})
        self._clear_pending_confirmation()

        if pending.get('name') == 'remote_prediction_pipeline':
            return await self._execute_remote_prediction_pipeline(pending.get('args') or {})
        if pending.get('name') == 'remote_training_pipeline':
            return await self._execute_remote_training_pipeline(pending.get('args') or {})

        if graph_pending is None or graph_pending.get('adapted'):
            return await self._execute_adapted_pending_tool(
                thread_id=thread_id,
                pending=pending,
                approved=True,
                auto_progress_followups=self._auto_confirmation_enabled(),
                stream_handler=stream_handler,
            )

        result = await self._graph_invoke(Command(resume="approved"), config=config, stream_handler=stream_handler)
        applied_results = self._apply_tool_results(result["messages"], built_messages_len=0)

        while True:
            next_pending = self._get_pending_tool_call(config)
            if not next_pending:
                break
            if self._tool_requires_confirmation(next_pending["name"]):
                self._set_pending_confirmation(thread_id, next_pending)
                self.memory.save_state(self.session_state)
                return self._needs_confirmation_result(thread_id, next_pending, await self._build_confirmation_message(next_pending))
            result = await self._graph_invoke(Command(resume="approved"), config=config, stream_handler=stream_handler)
            applied_results = self._apply_tool_results(result["messages"], built_messages_len=0)

        if pending.get("name") == "prepare_dataset_for_training":
            prepare_parsed = self._find_applied_tool_result(applied_results, 'prepare_dataset_for_training') or {
                'ok': True,
                'summary': '数据准备已完成',
                'data_yaml': str(self.session_state.active_dataset.data_yaml or ''),
            }
            post_prepare_followup = await self._handle_post_prepare_confirmation_followup(
                thread_id=thread_id,
                prepare_parsed=prepare_parsed,
            )
            if post_prepare_followup is not None:
                return post_prepare_followup

        final_text = self._compose_final_reply(result["messages"], applied_results)
        self._messages.append(AIMessage(content=final_text))
        self._trim_history()
        self.memory.save_state(self.session_state)
        return {
            "status": "completed",
            "message": final_text,
            "tool_call": pending,
            "approved": True,
        }


    def _pending_confirmation_thread_id(self) -> str:
        return str(self.session_state.pending_confirmation.thread_id or '').strip() or f"{self.session_state.session_id}-pending"

    def get_pending_action(self) -> dict[str, Any] | None:
        pending = self._pending_from_state()
        if not pending:
            return None
        return self._build_pending_action_payload(pending, thread_id=self._pending_confirmation_thread_id())

    def _pending_passthrough_decision(self, user_text: str, pending: dict[str, Any]) -> str | None:
        tool_name = str((pending or {}).get('name') or '').strip()
        if not tool_name:
            return None
        normalized = str(user_text or '').lower()
        training_like_tools = {'start_training', 'prepare_dataset_for_training', 'start_training_loop'}
        revision_tokens = (
            '改成', '换成', '换个', '调整', '改一下', '不要了', '去掉', '取消类别', '类别限制', '别用',
            'batch', 'imgsz', 'device', 'epochs', '轮数', 'optimizer', 'freeze', 'resume', 'lr0', 'patience', 'workers', 'amp',
            'fraction', 'classes', 'single_cls', 'project', 'name', '环境', '模型', '权重', '最大轮数', 'managed_level',
            '训练环境', '类别', '学习率', '优化器', '冻结', '早停', '线程数', '混合精度',
            '划分', '自动划分', '不划分', '不要划分', '默认比例', 'force_split',
        )
        if tool_name in training_like_tools and any(token in user_text or token in normalized for token in revision_tokens):
            return 'edit'
        question_tokens = (
            '为什么', '原因', '依据', '怎么看', '会不会', '会生成到哪里', '会上传到哪里', '产物路径', '输出路径', '先给我计划', '先看计划',
            '先讨论', '解释一下', '再解释一下', '说详细一点', '详细说说',
        )
        if any(token in user_text for token in question_tokens):
            return 'clarify'
        return None

    def _record_pending_action_review(
        self,
        pending: dict[str, Any],
        *,
        decision: str,
        raw_user_text: str = '',
        reason: str = '',
        source: str = 'runtime_review',
        edits: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_decision = self._normalize_confirmation_reply_decision(decision)
        if normalized_decision == 'deny':
            normalized_decision = 'reject'
        if normalized_decision not in {'approve', 'reject', 'edit', 'clarify', 'unclear', 'restate'}:
            normalized_decision = 'unclear'
        pending_thread_id = self._pending_confirmation_thread_id()
        decision_context = {
            'decision': normalized_decision,
            'reason': str(reason or '').strip(),
            'raw_user_text': str(raw_user_text or '').strip(),
            'source': str(source or 'runtime_review').strip(),
            'edits': dict(edits or {}),
        }
        self.session_state.pending_confirmation.decision_context = decision_context
        self.memory.append_event(
            self.session_state.session_id,
            'pending_action_reviewed',
            {
                'tool': str(pending.get('name') or ''),
                'thread_id': pending_thread_id,
                'decision': normalized_decision,
                'reason': decision_context['reason'],
                'source': decision_context['source'],
                'has_edits': bool(decision_context['edits']),
            },
        )
        return decision_context

    async def _try_handle_pending_confirmation_dialogue(
        self,
        user_text: str,
        *,
        stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any] | None:
        pending = self._pending_from_state()
        if not pending:
            return None
        passthrough_decision = self._pending_passthrough_decision(user_text, pending)
        if passthrough_decision is not None:
            self._record_pending_action_review(
                pending,
                decision=passthrough_decision,
                raw_user_text=user_text,
                source='natural_language_chat',
            )
            return None
        decision = await self._classify_confirmation_reply(user_text, pending)
        if decision in {'approve', 'deny'}:
            return await self.review_pending_action(
                {
                    'decision': 'approve' if decision == 'approve' else 'reject',
                    'raw_user_text': user_text,
                    'source': 'natural_language_chat',
                },
                stream_handler=stream_handler,
            )
        pending_thread_id = self._pending_confirmation_thread_id()
        if decision == 'restate':
            return self._needs_confirmation_result(pending_thread_id, pending, await self._build_confirmation_message(pending))
        pending_followup_action = await self._classify_pending_followup_action(
            user_text=user_text,
            pending=pending,
        )
        if pending_followup_action == 'status_or_detail':
            return self._needs_confirmation_result(
                pending_thread_id,
                pending,
                await self._build_confirmation_message(pending),
            )
        if self._looks_like_pending_status_or_detail_query(user_text):
            return self._needs_confirmation_result(
                pending_thread_id,
                pending,
                await self._build_confirmation_message(pending),
            )
        return None


    @staticmethod
    def _should_leave_pending_confirmation_to_dialogue(user_text: str, pending: dict[str, Any]) -> bool:
        del user_text, pending
        return False

    @staticmethod
    def _looks_like_pending_status_or_detail_query(user_text: str) -> bool:
        text = str(user_text or '').strip()
        normalized = text.lower()
        return any(
            token in text
            for token in (
                '查看状态', '当前状态', '查看情况', '看情况', '查看训练状态', '训练状态',
                '查看训练详情', '训练详情', '查看详情', '完整详情', '详细情况',
            )
        ) or any(token in normalized for token in ('status', 'details', 'detail'))

    async def _classify_structured_action(
        self,
        *,
        messages: list[Any],
        allowed_actions: set[str],
    ) -> str:
        parsed = await self._invoke_structured_payload(
            messages=messages,
            schema=self._action_router_schema(allowed_actions),
        )
        action = str(parsed.get('action') or '').strip().lower()
        return action if action in allowed_actions else ''

    async def _invoke_structured_payload(
        self,
        *,
        messages: list[Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        if self.planner_llm is None:
            return {}
        if self._planner_supports_native_structured_output():
            try:
                structured_llm = self.planner_llm.with_structured_output(schema)
                parsed = await structured_llm.ainvoke(messages)
                normalized = self._structured_output_payload(parsed)
                if normalized:
                    return normalized
            except Exception:
                pass
        try:
            response = await self.planner_llm.ainvoke(messages)
            return self._parse_json_object_payload(self._message_text(getattr(response, 'content', response)))
        except Exception:
            return {}

    def _planner_supports_native_structured_output(self) -> bool:
        provider = str(
            self.helper_llm_settings.provider
            or self.primary_llm_settings.provider
            or ''
        ).strip().lower()
        if provider != 'ollama':
            return False
        return callable(getattr(self.planner_llm, 'with_structured_output', None))

    @staticmethod
    def _action_router_schema(allowed_actions: set[str]) -> dict[str, Any]:
        return {
            'title': 'yolostudio_action_router',
            'type': 'object',
            'properties': {
                'action': {
                    'type': 'string',
                    'enum': sorted(allowed_actions),
                },
                'reason': {
                    'type': 'string',
                },
            },
            'required': ['action', 'reason'],
            'additionalProperties': False,
        }

    @staticmethod
    def _structured_output_value(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        if hasattr(value, 'model_dump'):
            try:
                dumped = value.model_dump()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dumped.get(key)
        if hasattr(value, key):
            return getattr(value, key)
        return None

    @classmethod
    def _structured_output_payload(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, 'model_dump'):
            try:
                dumped = value.model_dump()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dict(dumped)
        if hasattr(value, '__dict__'):
            data = {
                key: cls._structured_output_value(value, key)
                for key in getattr(value, '__dict__', {}).keys()
            }
            return {key: item for key, item in data.items() if item is not None}
        return {}

    async def _invoke_renderer_text(
        self,
        *,
        messages: list[Any],
        failure_event: str = '',
        failure_payload: dict[str, Any] | None = None,
    ) -> str:
        if self.planner_llm is None:
            return ''
        try:
            response = await self.planner_llm.ainvoke(messages)
            return self._message_text(getattr(response, 'content', response)).strip()
        except Exception as exc:
            if failure_event:
                payload = dict(failure_payload or {})
                payload.setdefault('error', str(exc))
                self.memory.append_event(
                    self.session_state.session_id,
                    failure_event,
                    payload,
                )
            return ''

    async def _classify_pending_followup_action(
        self,
        *,
        user_text: str,
        pending: dict[str, Any],
    ) -> str:
        if self.planner_llm is None:
            return ''
        facts = {
            'tool_name': str(pending.get('name') or ''),
            'tool_args': dict(pending.get('args') or {}),
            'objective': str(pending.get('objective') or self.session_state.pending_confirmation.objective or ''),
            'summary': str(pending.get('summary') or self.session_state.pending_confirmation.summary or ''),
            'allowed_decisions': list(pending.get('allowed_decisions') or self.session_state.pending_confirmation.allowed_decisions or []),
            'review_config': dict(pending.get('review_config') or self.session_state.pending_confirmation.review_config or {}),
            'decision_context': dict(pending.get('decision_context') or self.session_state.pending_confirmation.decision_context or {}),
            'user_text': user_text,
        }
        messages = [
            SystemMessage(
                content=(
                    '你是 YoloStudio Agent 的待确认动作跟进路由器。'
                    '当前同一会话里已经存在一个待确认动作。'
                    '你只负责判断用户这句跟进，是在追问当前待确认动作的状态、详情、原因、计划安排，还是不属于当前待确认动作上下文。'
                    '如果用户是在问现在什么情况、详细一点、为什么这样安排、这一步会做什么、当前计划是什么，返回 status_or_detail。'
                    '如果用户是在发起新的训练/预测/数据处理/远端传输或切换到别的话题，返回 other。'
                    '输出必须是 JSON，对象格式固定为 {"action":"status_or_detail|other","reason":"..."}。'
                    '不要输出 markdown，不要解释。\n'
                    f'facts={json.dumps(facts, ensure_ascii=False)}'
                )
            ),
        ]
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'status_or_detail'},
        )

    async def _classify_confirmation_reply(self, user_text: str, pending: dict[str, Any]) -> str:
        heuristic = self._classify_confirmation_reply_fallback(user_text)
        if heuristic in {'approve', 'deny', 'restate'}:
            return heuristic
        if self.planner_llm is None:
            return heuristic
        facts = self._confirmation_user_facts(pending)
        messages = [
            SystemMessage(
                content=(
                    '你是 YoloStudio Agent 的审批回复解释器。'
                    '你只负责判断用户这句回复，对当前待确认操作意味着什么。'
                    '输出必须是 JSON，对象格式固定为 '
                    '{"decision":"approve|deny|edit|clarify|unclear|restate","reason":"..."}。'
                    '只有当用户明确表达“继续执行/批准”时，decision 才能是 approve；'
                    '只有当用户明确表达“不要执行/取消”时，decision 才能是 deny；'
                    '如果用户主要是在改参数、换环境、换模型、调整执行方式，返回 edit；'
                    '如果用户主要是在追问原因、要求解释或想先听说明，返回 clarify；'
                    '如果用户只是表达犹豫但没有明确批准/拒绝/修改/追问，返回 unclear；'
                    '如果用户只是让你重复当前确认内容，返回 restate。'
                    '不要输出 Markdown，不要输出额外文字。'
                )
            ),
            HumanMessage(
                content=(
                    '当前待确认事实：\n'
                    f'{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n'
                    f'用户回复：{user_text}'
                )
            ),
        ]
        try:
            parsed = await self._invoke_structured_payload(
                messages=messages,
                schema={
                    'title': 'yolostudio_confirmation_reply',
                    'type': 'object',
                    'properties': {
                        'decision': {
                            'type': 'string',
                            'enum': ['approve', 'deny', 'edit', 'clarify', 'unclear', 'restate'],
                        },
                        'reason': {'type': 'string'},
                    },
                    'required': ['decision', 'reason'],
                    'additionalProperties': False,
                },
            )
            decision = self._normalize_confirmation_reply_decision(parsed.get('decision'))
            if decision != 'unclear':
                return decision
        except Exception as exc:
            self.memory.append_event(
                self.session_state.session_id,
                'confirmation_reply_classify_failed',
                {'tool': str(pending.get('name') or ''), 'error': str(exc)},
            )
        return heuristic

    @staticmethod
    def _normalize_confirmation_reply_decision(value: Any) -> str:
        text = str(value or '').strip().lower()
        mapping = {
            'approve': 'approve',
            'approved': 'approve',
            'continue': 'approve',
            'yes': 'approve',
            'allow': 'approve',
            'deny': 'deny',
            'denied': 'deny',
            'cancel': 'deny',
            'reject': 'deny',
            'no': 'deny',
            'edit': 'edit',
            'edited': 'edit',
            'revise': 'edit',
            'modify': 'edit',
            'clarify': 'clarify',
            'question': 'clarify',
            'unclear': 'unclear',
            'unknown': 'unclear',
            'ask_info': 'unclear',
            'restate': 'restate',
            'repeat': 'restate',
        }
        return mapping.get(text, 'unclear')

    @staticmethod
    def _parse_confirmation_reply_payload(raw: str) -> dict[str, Any]:
        return YoloStudioAgentClient._parse_json_object_payload(raw)

    @staticmethod
    def _parse_json_object_payload(raw: str) -> dict[str, Any]:
        text = str(raw or '').strip()
        if not text:
            return {}
        fenced = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
        candidates = [fenced]
        match = re.search(r'\{.*\}', fenced, flags=re.DOTALL)
        if match:
            candidates.insert(0, match.group(0))
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _classify_confirmation_reply_fallback(user_text: str) -> str:
        text = str(user_text or '').strip().lower()
        if not text:
            return 'unclear'
        normalized = re.sub(r'\s+', '', text)
        restate_tokens = {
            '再说一遍', '再说一次', '重说一遍', '重说一次', '重复一下', '重复一遍', '重新说一下', '再讲一遍', '再讲一次'
        }
        if any(token in normalized for token in restate_tokens):
            return 'restate'

        deny_exact = {
            'n', 'no', 'cancel', 'stop', 'deny', 'abort',
            '不同意', '取消', '否', '不用', '不要', '停止', '算了', '不继续', '不继续了', '不执行', '不做了', '先不要', '先别', '先不做',
        }
        approve_exact = {
            'y', 'yes', 'ok', 'okay', 'sure', 'confirm', 'continue', 'go', 'goahead',
            '同意', '确认', '继续', '继续吧', '是', '好的', '可以', '行', '执行', '开始', '开始吧',
            '开始训练', '启动训练', '开始循环训练', '启动循环训练',
            '就这样', '按这个来', '没问题', '好',
        }
        if normalized in deny_exact:
            return 'deny'
        if normalized in approve_exact:
            return 'approve'

        deny_markers = (
            '不继续', '不继续了', '先别', '先不要', '不执行', '别执行', '先不做', '不要做', '不要执行', '不同意', '取消', '算了', '不做了', '停止'
        )
        if any(marker in normalized for marker in deny_markers):
            return 'deny'

        approve_markers = (
            '继续吧', '继续执行', '可以开始', '可以执行', '确认执行', '同意执行', '按这个来', '就这样',
            '开始训练', '启动训练', '开始循环训练', '启动循环训练',
            '开始吧', '执行吧', '没问题', '可以', '行', '继续', '确认'
        )
        if any(marker in normalized for marker in approve_markers) and not any(token in normalized for token in ('不', '别', '先不', '不要', '取消', '算了')):
            return 'approve'
        return 'unclear'
    def _sync_preferences(self) -> None:
        if self.session_state.preferences.default_model == self.primary_llm_settings.model:
            self.session_state.preferences.default_model = ""
        if self.session_state.preferences.language != "zh-CN":
            self.session_state.preferences.language = "zh-CN"

    async def _complete_cached_tool_result_reply(
        self,
        tool_name: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not payload:
            return None
        reply = await self._render_tool_result_message(tool_name, payload)
        if not reply:
            return None
        self._messages.append(AIMessage(content=reply))
        return {'status': 'completed', 'message': reply, 'tool_call': None}

    async def _try_handle_prediction_management_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        action = await self._classify_prediction_management_followup_action(
            user_text=user_text,
            normalized_text=normalized_text,
        )
        result = self._prediction_management_followup_result(action)
        if not result:
            return None
        tool_name, payload = result
        return await self._complete_cached_tool_result_reply(tool_name, payload)

    async def _try_handle_prediction_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
        fallback_path: str,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        action = await self._classify_prediction_followup_action(
            user_text=user_text,
            normalized_text=normalized_text,
            fallback_path=fallback_path,
        )
        cached_result = self._prediction_followup_result(action)
        if cached_result:
            tool_name, payload = cached_result
            return await self._complete_cached_tool_result_reply(tool_name, payload)
        if action == 'inspect':
            inspect_kwargs = self._prediction_followup_kwargs(
                user_text,
                fallback_path=fallback_path,
                allow_context_fallback=True,
            )
            if inspect_kwargs:
                return await self._complete_direct_tool_reply('inspect_prediction_outputs', **inspect_kwargs)
        if action == 'summary':
            summary_kwargs = self._prediction_followup_kwargs(
                user_text,
                fallback_path=fallback_path,
                allow_context_fallback=True,
            )
            if summary_kwargs:
                return await self._complete_direct_tool_reply('summarize_prediction_results', **summary_kwargs)
        return None

    async def _try_handle_extract_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        action = await self._classify_extract_followup_action(
            user_text=user_text,
            normalized_text=normalized_text,
        )
        result = self._extract_followup_result(action)
        if not result:
            return None
        tool_name, payload = result
        return await self._complete_cached_tool_result_reply(tool_name, payload)

    async def _try_handle_dataset_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
        dataset_path: str,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        action = await self._classify_dataset_followup_action(
            user_text=user_text,
            normalized_text=normalized_text,
            fallback_path=dataset_path,
        )
        if action == 'quality':
            return await self._complete_dataset_quality_reply(dataset_path)
        cached_result = self._dataset_followup_result(action)
        if cached_result:
            tool_name, payload = cached_result
            return await self._complete_cached_tool_result_reply(tool_name, payload)
        if action == 'health':
            return await self._complete_direct_tool_reply(
                'run_dataset_health_check',
                dataset_path=dataset_path,
                include_duplicates=True,
                max_duplicate_groups=3,
            )
        if action == 'duplicates':
            return await self._complete_direct_tool_reply('detect_duplicate_images', dataset_path=dataset_path)
        return None

    async def _try_handle_realtime_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        action = await self._classify_realtime_followup_action(
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if action == 'status':
            status_kwargs = self._build_realtime_session_kwargs(user_text)
            return await self._complete_direct_tool_reply('check_realtime_prediction_status', **status_kwargs)
        return None

    async def _try_handle_remote_roundtrip_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        action = await self._classify_remote_roundtrip_followup_action(
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if action == 'training_pipeline':
            return await self._complete_cached_tool_result_reply(
                'remote_training_pipeline',
                self.session_state.active_training.last_remote_roundtrip,
            )
        if action == 'prediction_pipeline':
            return await self._complete_cached_tool_result_reply(
                'remote_prediction_pipeline',
                self.session_state.active_prediction.last_remote_roundtrip,
            )
        return None

    async def _try_handle_remote_transfer_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        action = await self._classify_remote_transfer_followup_action(
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if action == 'profiles':
            return await self._complete_cached_tool_result_reply(
                'list_remote_profiles',
                self.session_state.active_remote_transfer.last_profile_listing,
            )
        if action == 'upload':
            return await self._complete_cached_tool_result_reply(
                'upload_assets_to_remote',
                self.session_state.active_remote_transfer.last_upload,
            )
        if action == 'download':
            return await self._complete_cached_tool_result_reply(
                'download_assets_from_remote',
                self.session_state.active_remote_transfer.last_download,
            )
        return None

    async def _try_handle_prediction_requests(
        self,
        *,
        user_text: str,
        normalized_text: str,
        prediction_path: str,
        wants_train: bool,
        training_command_like: bool,
        wants_scan_videos: bool,
        wants_extract_frames: bool,
        wants_extract_preview: bool,
        wants_extract_images: bool,
        wants_remote_upload: bool,
        wants_prediction_summary: bool,
        wants_prediction_output_inspection: bool,
        wants_prediction_report_export: bool,
        wants_prediction_path_lists: bool,
        wants_prediction_result_organize: bool,
        has_prediction_followup_context: bool,
        has_prediction_management_followup_context: bool,
        has_explicit_prediction_target: bool,
    ) -> dict[str, Any] | None:
        if wants_prediction_output_inspection:
            inspect_kwargs = self._prediction_followup_kwargs(
                user_text,
                fallback_path=prediction_path,
                allow_context_fallback=True,
            )
            cached_inspect = self._prediction_request_cached_result('inspect', inspect_kwargs)
            if cached_inspect:
                tool_name, payload = cached_inspect
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            if inspect_kwargs:
                return await self._complete_direct_tool_reply('inspect_prediction_outputs', **inspect_kwargs)

        if wants_prediction_report_export:
            export_path = intent_parsing.extract_output_path_from_text(
                user_text,
                self.session_state.active_prediction.output_dir or prediction_path,
            )
            export_kwargs = self._prediction_followup_kwargs(
                user_text,
                fallback_path=prediction_path,
                allow_context_fallback=True,
            )
            if export_path:
                export_kwargs['export_path'] = export_path
                if str(export_kwargs.get('output_dir') or '').strip() == export_path:
                    export_kwargs.pop('output_dir', None)
            cached_export = self._prediction_management_request_cached_result('export', export_kwargs)
            if cached_export:
                tool_name, payload = cached_export
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            if export_kwargs:
                return await self._complete_direct_tool_reply('export_prediction_report', **export_kwargs)

        if wants_prediction_path_lists:
            export_dir = intent_parsing.extract_output_path_from_text(
                user_text,
                self.session_state.active_prediction.output_dir or prediction_path,
            )
            path_list_kwargs = self._prediction_followup_kwargs(
                user_text,
                fallback_path=prediction_path,
                allow_context_fallback=True,
            )
            if export_dir:
                path_list_kwargs['export_dir'] = export_dir
                if str(path_list_kwargs.get('output_dir') or '').strip() == export_dir:
                    path_list_kwargs.pop('output_dir', None)
            cached_path_lists = self._prediction_management_request_cached_result('path_lists', path_list_kwargs)
            if cached_path_lists:
                tool_name, payload = cached_path_lists
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            if path_list_kwargs:
                return await self._complete_direct_tool_reply('export_prediction_path_lists', **path_list_kwargs)

        if wants_prediction_result_organize:
            organize_kwargs = self._prediction_followup_kwargs(
                user_text,
                fallback_path=prediction_path,
                allow_context_fallback=True,
            )
            destination_dir = intent_parsing.extract_output_path_from_text(user_text, self.session_state.active_prediction.output_dir or prediction_path)
            organize_by = 'by_class' if '类别' in user_text else 'detected_only'
            include_empty = '无命中' in user_text or '空结果' in user_text
            if destination_dir:
                organize_kwargs['destination_dir'] = destination_dir
            organize_kwargs['organize_by'] = organize_by
            organize_kwargs['include_empty'] = include_empty
            if organize_kwargs:
                return await self._complete_direct_tool_reply('organize_prediction_results', **organize_kwargs)

        if wants_prediction_summary:
            summary_kwargs = self._prediction_followup_kwargs(user_text, fallback_path=prediction_path)
            cached_summary = self._prediction_request_cached_result('summary', summary_kwargs)
            if cached_summary:
                tool_name, payload = cached_summary
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            if summary_kwargs:
                return await self._complete_direct_tool_reply('summarize_prediction_results', **summary_kwargs)

            if self._explicitly_references_previous_context(user_text) and self.session_state.active_prediction.last_result:
                reply = await self._render_tool_result_message('predict_images', self.session_state.active_prediction.last_result)
                if reply:
                    self._messages.append(AIMessage(content=reply))
                    return {'status': 'completed', 'message': reply, 'tool_call': None}

        prediction_management_followup = await self._try_handle_prediction_management_followup(
            enabled=(
                has_prediction_management_followup_context
                and not wants_train
                and not training_command_like
                and not wants_scan_videos
                and not wants_extract_frames
                and not wants_extract_preview
                and not wants_extract_images
                and not wants_remote_upload
                and not has_explicit_prediction_target
                and not wants_prediction_output_inspection
                and not wants_prediction_report_export
                and not wants_prediction_path_lists
                and not wants_prediction_result_organize
            ),
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if prediction_management_followup:
            return prediction_management_followup

        prediction_followup = await self._try_handle_prediction_followup(
            enabled=(
                has_prediction_followup_context
                and not wants_train
                and not training_command_like
                and not wants_scan_videos
                and not wants_extract_frames
                and not wants_extract_preview
                and not wants_extract_images
                and not wants_remote_upload
                and not has_explicit_prediction_target
            ),
            user_text=user_text,
            normalized_text=normalized_text,
            fallback_path=prediction_path,
        )
        if prediction_followup:
            return prediction_followup
        return None

    async def _try_handle_dataset_and_extract_requests(
        self,
        *,
        user_text: str,
        normalized_text: str,
        dataset_path: str,
        prediction_path: str,
        extracted_dataset_path: str,
        wants_train: bool,
        wants_predict: bool,
        training_command_like: bool,
        wants_remote_upload: bool,
        wants_scan_videos: bool,
        wants_extract_frames: bool,
        wants_extract_preview: bool,
        wants_extract_images: bool,
        wants_quality: bool,
        wants_health: bool,
        wants_duplicates: bool,
        wants_readiness: bool,
        has_extract_followup_context: bool,
        has_dataset_followup_context: bool,
    ) -> dict[str, Any] | None:
        extract_followup = await self._try_handle_extract_followup(
            enabled=(
                has_extract_followup_context
                and not wants_train
                and not wants_predict
                and not training_command_like
                and not wants_remote_upload
                and not wants_quality
                and not wants_health
                and not wants_duplicates
                and not extracted_dataset_path
            ),
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if extract_followup:
            return extract_followup

        dataset_followup = await self._try_handle_dataset_followup(
            enabled=(
                has_dataset_followup_context
                and bool(dataset_path)
                and not wants_train
                and not wants_predict
                and not training_command_like
                and not wants_scan_videos
                and not wants_extract_frames
                and not wants_extract_preview
                and not wants_extract_images
                and not wants_remote_upload
                and not wants_quality
                and not wants_health
                and not wants_duplicates
                and not extracted_dataset_path
            ),
            user_text=user_text,
            normalized_text=normalized_text,
            dataset_path=dataset_path,
        )
        if dataset_followup:
            return dataset_followup

        if dataset_path and wants_quality and not wants_train:
            return await self._complete_dataset_quality_reply(dataset_path)

        if dataset_path and wants_duplicates and not wants_train and not wants_health:
            if has_dataset_followup_context and self._dataset_request_cache_allowed(dataset_path):
                cached_duplicates = self._dataset_followup_result('duplicates')
                if cached_duplicates:
                    tool_name, payload = cached_duplicates
                    return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('detect_duplicate_images', dataset_path=dataset_path)

        if dataset_path and wants_health and not wants_train:
            if has_dataset_followup_context and self._dataset_request_cache_allowed(dataset_path):
                cached_health = self._dataset_followup_result('health')
                if cached_health:
                    tool_name, payload = cached_health
                    return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply(
                'run_dataset_health_check',
                dataset_path=dataset_path,
                include_duplicates=wants_duplicates,
            )

        if dataset_path and wants_readiness and not wants_predict and not training_command_like:
            return await self._complete_readiness_knowledge_reply(dataset_path)

        if dataset_path and wants_extract_preview and not wants_train:
            preview_kwargs = intent_parsing.build_image_extract_args_from_text(user_text, dataset_path)
            if has_dataset_followup_context and self._dataset_request_cache_allowed(dataset_path):
                cached_preview = self._dataset_extract_request_cached_result('preview', preview_kwargs, dataset_path=dataset_path)
                if cached_preview:
                    tool_name, payload = cached_preview
                    return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('preview_extract_images', **preview_kwargs)

        if dataset_path and wants_extract_images and not wants_train and not wants_extract_preview:
            return await self._complete_direct_tool_reply('extract_images', **intent_parsing.build_image_extract_args_from_text(user_text, dataset_path))

        if prediction_path and wants_scan_videos and not wants_predict and not training_command_like:
            cached_video_scan = self._dataset_extract_request_cached_result('video_scan', {'source_path': prediction_path})
            if cached_video_scan:
                tool_name, payload = cached_video_scan
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('scan_videos', source_path=prediction_path)

        if prediction_path and wants_extract_frames and not wants_predict and not training_command_like:
            return await self._complete_direct_tool_reply('extract_video_frames', **intent_parsing.build_video_extract_args_from_text(user_text, prediction_path))
        return None

    async def _try_handle_training_history_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        action = await self._classify_training_history_followup_action(
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if action == 'runs' and self.session_state.active_training.recent_runs:
            return await self._complete_cached_tool_result_reply(
                'list_training_runs',
                {
                    'ok': True,
                    'summary': '训练历史查询完成',
                    'runs': list(self.session_state.active_training.recent_runs),
                },
            )
        if action == 'inspect':
            return await self._complete_cached_tool_result_reply(
                'inspect_training_run',
                self.session_state.active_training.last_run_inspection,
            )
        if action == 'compare':
            return await self._complete_cached_tool_result_reply(
                'compare_training_runs',
                self.session_state.active_training.last_run_comparison,
            )
        if action == 'best':
            return await self._complete_cached_tool_result_reply(
                'select_best_training_run',
                self.session_state.active_training.best_run_selection,
            )
        return None

    async def _try_handle_training_loop_history_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        action = await self._classify_training_loop_history_followup_action(
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if action == 'list' and self.session_state.active_training.recent_loops:
            return await self._complete_cached_tool_result_reply(
                'list_training_loops',
                {
                    'ok': True,
                    'summary': '环训练列表已就绪',
                    'loops': list(self.session_state.active_training.recent_loops),
                },
            )
        if action == 'status':
            return await self._complete_cached_tool_result_reply(
                'check_training_loop_status',
                self.session_state.active_training.last_loop_status,
            )
        if action == 'inspect':
            payload = self.session_state.active_training.last_loop_detail or self.session_state.active_training.last_loop_status
            return await self._complete_cached_tool_result_reply('inspect_training_loop', payload)
        return None

    async def _try_handle_knowledge_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
        metric_signals: list[str],
        asks_metric_terms: bool,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        action = await self._classify_knowledge_followup_action(
            user_text=user_text,
            normalized_text=normalized_text,
            metric_signals=metric_signals,
        )
        cached_result = self._knowledge_followup_result(action)
        if cached_result:
            tool_name, payload = cached_result
            return await self._complete_cached_tool_result_reply(tool_name, payload)
        if action == 'knowledge':
            retrieval = self.session_state.active_knowledge.last_retrieval
            retrieval_topic = str(retrieval.get('topic') or '').strip() or ('training_metrics' if asks_metric_terms else 'workflow')
            retrieval_stage = str(retrieval.get('stage') or '').strip() or 'post_training'
            retrieval_signals = list(retrieval.get('signals') or metric_signals)
            return await self._complete_knowledge_retrieval_reply(
                topic=retrieval_topic,
                stage=retrieval_stage,
                signals=retrieval_signals,
            )
        if action == 'analysis':
            return await self._complete_training_outcome_analysis_reply()
        if action == 'next_step':
            return await self._complete_next_training_step_reply('')
        return None

    async def _try_handle_training_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
        metric_signals: list[str],
        asks_metric_terms: bool,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        action = await self._classify_training_followup_action(
            user_text=user_text,
            normalized_text=normalized_text,
            metric_signals=metric_signals,
        )
        cached_result = self._training_followup_cached_result(action)
        if cached_result:
            tool_name, payload = cached_result
            return await self._complete_cached_tool_result_reply(tool_name, payload)
        if action == 'status':
            return await self._complete_direct_tool_reply('check_training_status')
        if action == 'analysis':
            return await self._complete_training_outcome_analysis_reply()
        if action == 'next_step':
            return await self._complete_next_training_step_reply('')
        if action == 'knowledge':
            return await self._complete_knowledge_retrieval_reply(
                topic='training_metrics' if asks_metric_terms else 'workflow',
                stage='post_training',
                signals=metric_signals,
            )
        return None

    async def _try_handle_realtime_requests(
        self,
        *,
        user_text: str,
        normalized_text: str,
        wants_train: bool,
        wants_camera_scan: bool,
        wants_screen_scan: bool,
        wants_rtsp_test: bool,
        wants_realtime_status: bool,
        wants_realtime_stop: bool,
        wants_camera_prediction: bool,
        wants_rtsp_prediction: bool,
        wants_screen_prediction: bool,
        has_realtime_context: bool,
        has_explicit_realtime_target: bool,
        rtsp_url: str,
    ) -> dict[str, Any] | None:
        if wants_camera_scan and not wants_train:
            return await self._complete_direct_tool_reply('scan_cameras')

        if wants_screen_scan and not wants_train:
            return await self._complete_direct_tool_reply('scan_screens')

        if wants_rtsp_test and rtsp_url and not wants_train:
            timeout_ms = intent_parsing.extract_timeout_ms_from_text(user_text)
            test_kwargs: dict[str, Any] = {'rtsp_url': rtsp_url}
            if timeout_ms is not None:
                test_kwargs['timeout_ms'] = timeout_ms
            return await self._complete_direct_tool_reply('test_rtsp_stream', **test_kwargs)

        if wants_realtime_status and not wants_train:
            status_kwargs = self._build_realtime_session_kwargs(user_text)
            return await self._complete_direct_tool_reply('check_realtime_prediction_status', **status_kwargs)

        if wants_realtime_stop and not wants_train:
            stop_kwargs = self._build_realtime_session_kwargs(user_text)
            return await self._complete_direct_tool_reply('stop_realtime_prediction', **stop_kwargs)

        if wants_camera_prediction and not wants_train:
            return await self._complete_direct_tool_reply(
                'start_camera_prediction',
                **self._build_realtime_prediction_args(user_text, source_type='camera'),
            )

        if wants_rtsp_prediction and not wants_train:
            return await self._complete_direct_tool_reply(
                'start_rtsp_prediction',
                **self._build_realtime_prediction_args(user_text, source_type='rtsp'),
            )

        if wants_screen_prediction and not wants_train:
            return await self._complete_direct_tool_reply(
                'start_screen_prediction',
                **self._build_realtime_prediction_args(user_text, source_type='screen'),
            )

        realtime_followup = await self._try_handle_realtime_followup(
            enabled=(
                has_realtime_context
                and not wants_train
                and not wants_camera_scan
                and not wants_screen_scan
                and not wants_rtsp_test
                and not wants_realtime_stop
                and not wants_camera_prediction
                and not wants_rtsp_prediction
                and not wants_screen_prediction
                and not has_explicit_realtime_target
            ),
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if realtime_followup:
            return realtime_followup
        return None

    async def _try_handle_remote_requests(
        self,
        *,
        thread_id: str,
        user_text: str,
        normalized_text: str,
        wants_train: bool,
        wants_predict: bool,
        training_command_like: bool,
        wants_remote_profile_list: bool,
        wants_remote_upload: bool,
        wants_remote_prediction_pipeline: bool,
        wants_remote_training_pipeline: bool,
        wants_remote_result_return: bool,
        has_remote_transfer_followup_context: bool,
        has_remote_roundtrip_followup_context: bool,
        has_explicit_remote_target: bool,
        has_explicit_transfer_paths: bool,
    ) -> dict[str, Any] | None:
        if wants_remote_profile_list:
            cached_profiles = self.session_state.active_remote_transfer.last_profile_listing
            if cached_profiles:
                return await self._complete_cached_tool_result_reply(
                    'list_remote_profiles',
                    cached_profiles,
                )
            return await self._complete_direct_tool_reply('list_remote_profiles')

        if wants_remote_prediction_pipeline:
            pipeline_args = self._build_remote_prediction_pipeline_args(user_text)
            upload_args = dict(pipeline_args.get('upload_args') or {})
            local_paths = list(upload_args.get('local_paths') or [])
            if not local_paths:
                reply = '当前还不能发起远端预测闭环：请明确给我要上传的本地模型和图片/视频路径。'
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}
            if not str(upload_args.get('server') or '').strip() and not str(upload_args.get('host') or '').strip():
                reply = '当前还不能发起远端预测闭环：请明确目标服务器，或先列出并选择一个远端 profile。'
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}
            if not str(upload_args.get('remote_root') or '').strip():
                reply = '当前还不能发起远端预测闭环：请明确远端目录，或先配置一个带默认 remote_root 的远端 profile。'
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}
            pending = {'name': 'remote_prediction_pipeline', 'args': pipeline_args, 'id': None, 'synthetic': True}
            self._set_pending_confirmation(thread_id, pending)
            reply = await self._build_confirmation_message(pending)
            self._messages.append(AIMessage(content=reply))
            return self._needs_confirmation_result(thread_id, pending, reply)

        if wants_remote_training_pipeline:
            pipeline_args = self._build_remote_training_pipeline_args(user_text)
            upload_args = dict(pipeline_args.get('upload_args') or {})
            local_paths = list(upload_args.get('local_paths') or [])
            if not local_paths:
                reply = '当前还不能发起远端训练闭环：请明确给我要上传的本地模型和数据集路径。'
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}
            if not str(upload_args.get('server') or '').strip() and not str(upload_args.get('host') or '').strip():
                reply = '当前还不能发起远端训练闭环：请明确目标服务器，或先列出并选择一个远端 profile。'
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}
            if not str(upload_args.get('remote_root') or '').strip():
                reply = '当前还不能发起远端训练闭环：请明确远端目录，或先配置一个带默认 remote_root 的远端 profile。'
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}
            pending = {'name': 'remote_training_pipeline', 'args': pipeline_args, 'id': None, 'synthetic': True}
            self._set_pending_confirmation(thread_id, pending)
            reply = await self._build_confirmation_message(pending)
            self._messages.append(AIMessage(content=reply))
            return self._needs_confirmation_result(thread_id, pending, reply)

        remote_roundtrip_followup = await self._try_handle_remote_roundtrip_followup(
            enabled=(
                has_remote_roundtrip_followup_context
                and not wants_remote_profile_list
                and not wants_remote_upload
                and not wants_remote_prediction_pipeline
                and not wants_remote_training_pipeline
                and not wants_remote_result_return
                and not has_explicit_remote_target
                and not has_explicit_transfer_paths
            ),
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if remote_roundtrip_followup:
            return remote_roundtrip_followup

        if wants_remote_upload:
            upload_args = self._build_remote_upload_args(user_text)
            upload_args = self._apply_remote_defaults(upload_args)
            local_paths = list(upload_args.get('local_paths') or [])
            if not local_paths:
                reply = '当前还不能发起远端上传：请明确给我要上传的本地路径；至少提供一个本地权重、数据集目录或视频路径。'
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}
            if not str(upload_args.get('remote_root') or '').strip():
                reply = '当前还不能发起远端上传：请明确远端目录，或先配置一个带默认 remote_root 的远端 profile。'
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}
            pending = {'name': 'upload_assets_to_remote', 'args': upload_args, 'id': None, 'synthetic': True}
            self._set_pending_confirmation(thread_id, pending)
            reply = await self._build_confirmation_message(pending)
            self._messages.append(AIMessage(content=reply))
            return self._needs_confirmation_result(thread_id, pending, reply)

        remote_transfer_followup = await self._try_handle_remote_transfer_followup(
            enabled=(
                has_remote_transfer_followup_context
                and not wants_train
                and not wants_predict
                and not training_command_like
                and not wants_remote_profile_list
                and not wants_remote_upload
                and not wants_remote_prediction_pipeline
                and not wants_remote_training_pipeline
                and not wants_remote_result_return
                and not has_explicit_remote_target
                and not has_explicit_transfer_paths
            ),
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if remote_transfer_followup:
            return remote_transfer_followup
        return None

    async def _try_handle_training_history_and_loop_requests(
        self,
        *,
        user_text: str,
        normalized_text: str,
        has_training_history_followup_context: bool,
        has_training_loop_history_followup_context: bool,
        has_training_context: bool,
        wants_predict: bool,
        training_command_like: bool,
        explicit_run_ids: list[str],
        wants_training_run_inspect: bool,
        wants_failed_training_run_list: bool,
        wants_completed_training_run_list: bool,
        wants_stopped_training_run_list: bool,
        wants_running_training_run_list: bool,
        wants_analysis_ready_run_list: bool,
        wants_training_loop_list: bool,
        wants_training_loop_status: bool,
        wants_inspect_training_loop: bool,
        wants_pause_training_loop: bool,
        wants_resume_training_loop: bool,
        wants_stop_training_loop: bool,
        comparison_run_ids: list[str],
        wants_training_run_compare: bool,
        wants_next_step_guidance: bool,
        wants_best_training_run: bool,
        wants_training_outcome_analysis: bool,
        loop_route: dict[str, Any],
    ) -> dict[str, Any] | None:
        training_history_followup = await self._try_handle_training_history_followup(
            enabled=(
                has_training_history_followup_context
                and not has_training_context
                and not wants_predict
                and not training_command_like
                and not explicit_run_ids
                and not wants_training_run_inspect
                and not wants_failed_training_run_list
                and not wants_completed_training_run_list
                and not wants_stopped_training_run_list
                and not wants_running_training_run_list
                and not wants_analysis_ready_run_list
                and not wants_training_loop_list
                and not wants_training_loop_status
                and not wants_inspect_training_loop
                and not wants_pause_training_loop
                and not wants_resume_training_loop
                and not wants_stop_training_loop
            ),
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if training_history_followup:
            return training_history_followup

        if wants_training_run_compare and wants_next_step_guidance and not wants_predict and not training_command_like:
            compare_kwargs: dict[str, Any] = {}
            if comparison_run_ids:
                compare_kwargs['left_run_id'] = comparison_run_ids[0]
                if len(comparison_run_ids) > 1:
                    compare_kwargs['right_run_id'] = comparison_run_ids[1]
            return await self._complete_training_compare_next_step_reply(**compare_kwargs)

        if wants_best_training_run and wants_next_step_guidance and not wants_predict and not training_command_like:
            return await self._complete_best_training_next_step_reply()

        if wants_training_run_compare and wants_training_outcome_analysis and not wants_predict and not training_command_like:
            compare_kwargs: dict[str, Any] = {}
            if comparison_run_ids:
                compare_kwargs['left_run_id'] = comparison_run_ids[0]
                if len(comparison_run_ids) > 1:
                    compare_kwargs['right_run_id'] = comparison_run_ids[1]
            return await self._complete_training_compare_analysis_reply(**compare_kwargs)

        if wants_best_training_run and wants_training_outcome_analysis and not wants_predict and not training_command_like:
            return await self._complete_best_training_outcome_analysis_reply()

        if explicit_run_ids and wants_next_step_guidance and not wants_predict and not training_command_like and not wants_training_run_compare:
            return await self._complete_specific_training_run_next_step_reply(explicit_run_ids[0])

        if explicit_run_ids and wants_training_outcome_analysis and not wants_predict and not training_command_like and not wants_training_run_compare:
            return await self._complete_specific_training_run_outcome_analysis_reply(explicit_run_ids[0])

        if wants_training_run_compare and not wants_predict and not training_command_like:
            cached_compare = self._training_history_request_cached_result(
                request='compare',
                left_run_id=comparison_run_ids[0] if comparison_run_ids else '',
                right_run_id=comparison_run_ids[1] if len(comparison_run_ids) > 1 else '',
            )
            if cached_compare:
                tool_name, payload = cached_compare
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            compare_kwargs: dict[str, Any] = {}
            if comparison_run_ids:
                compare_kwargs['left_run_id'] = comparison_run_ids[0]
                if len(comparison_run_ids) > 1:
                    compare_kwargs['right_run_id'] = comparison_run_ids[1]
            return await self._complete_direct_tool_reply('compare_training_runs', **compare_kwargs)

        if wants_best_training_run and not wants_predict and not training_command_like:
            cached_best = self._training_history_request_cached_result(request='best')
            if cached_best:
                tool_name, payload = cached_best
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('select_best_training_run')

        if wants_training_run_inspect and not wants_predict and not training_command_like:
            cached_inspection = self._training_history_request_cached_result(
                request='inspect',
                run_id=explicit_run_ids[0],
            )
            if cached_inspection:
                tool_name, payload = cached_inspection
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('inspect_training_run', run_id=explicit_run_ids[0])

        if wants_failed_training_run_list and not wants_predict and not training_command_like:
            cached_failed = self._training_history_request_cached_result(request='runs', run_state='failed')
            if cached_failed:
                tool_name, payload = cached_failed
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('list_training_runs', run_state='failed')

        if wants_completed_training_run_list and not wants_predict and not training_command_like:
            cached_completed = self._training_history_request_cached_result(request='runs', run_state='completed')
            if cached_completed:
                tool_name, payload = cached_completed
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('list_training_runs', run_state='completed')

        if wants_stopped_training_run_list and not wants_predict and not training_command_like:
            cached_stopped = self._training_history_request_cached_result(request='runs', run_state='stopped')
            if cached_stopped:
                tool_name, payload = cached_stopped
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('list_training_runs', run_state='stopped')

        if wants_running_training_run_list and not wants_predict and not training_command_like:
            cached_running = self._training_history_request_cached_result(request='runs', run_state='running')
            if cached_running:
                tool_name, payload = cached_running
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('list_training_runs', run_state='running')

        if wants_analysis_ready_run_list and not wants_predict and not training_command_like:
            cached_analysis_ready = self._training_history_request_cached_result(request='runs', analysis_ready=True)
            if cached_analysis_ready:
                tool_name, payload = cached_analysis_ready
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('list_training_runs', analysis_ready=True)

        if wants_training_loop_list and not wants_predict and not training_command_like:
            cached_loops = self._training_loop_request_cached_result('list')
            if cached_loops:
                tool_name, payload = cached_loops
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('list_training_loops')

        active_loop_id = str(loop_route.get('loop_id') or '').strip() or self.session_state.active_training.active_loop_id

        if wants_training_loop_status and not wants_predict and not training_command_like:
            return await self._complete_direct_tool_reply('check_training_loop_status', loop_id=active_loop_id)

        if wants_inspect_training_loop and not wants_predict and not training_command_like:
            cached_loop_detail = self._training_loop_request_cached_result('inspect', loop_id=active_loop_id)
            if cached_loop_detail:
                tool_name, payload = cached_loop_detail
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('inspect_training_loop', loop_id=active_loop_id)

        if wants_pause_training_loop and not wants_predict and not training_command_like:
            return await self._complete_direct_tool_reply('pause_training_loop', loop_id=active_loop_id)

        if wants_resume_training_loop and not wants_predict and not training_command_like:
            return await self._complete_direct_tool_reply('resume_training_loop', loop_id=active_loop_id)

        if wants_stop_training_loop and not wants_predict and not training_command_like:
            return await self._complete_direct_tool_reply('stop_training_loop', loop_id=active_loop_id)

        loop_history_followup = await self._try_handle_training_loop_history_followup(
            enabled=(
                has_training_loop_history_followup_context
                and not self.session_state.active_training.active_loop_id
                and not wants_predict
                and not training_command_like
                and not explicit_run_ids
                and not wants_training_loop_list
                and not wants_training_loop_status
                and not wants_inspect_training_loop
                and not wants_pause_training_loop
                and not wants_resume_training_loop
                and not wants_stop_training_loop
            ),
            user_text=user_text,
            normalized_text=normalized_text,
        )
        if loop_history_followup:
            return loop_history_followup

        return None

    async def _try_handle_training_context_requests(
        self,
        *,
        user_text: str,
        normalized_text: str,
        dataset_path: str,
        wants_predict: bool,
        training_command_like: bool,
        wants_stop_training: bool,
        wants_training_provenance: bool,
        wants_training_evidence: bool,
        wants_next_step_guidance: bool,
        wants_training_knowledge: bool,
        wants_training_outcome_analysis: bool,
        wants_training_status: bool,
        wants_training_run_list: bool,
        wants_training_run_compare: bool,
        wants_best_training_run: bool,
        wants_training_run_inspect: bool,
        wants_failed_training_run_list: bool,
        wants_completed_training_run_list: bool,
        wants_stopped_training_run_list: bool,
        wants_running_training_run_list: bool,
        wants_analysis_ready_run_list: bool,
        wants_training_loop_list: bool,
        wants_training_loop_status: bool,
        wants_inspect_training_loop: bool,
        wants_pause_training_loop: bool,
        wants_resume_training_loop: bool,
        wants_stop_training_loop: bool,
        wants_training_loop_start: bool,
        has_knowledge_followup_context: bool,
        has_training_followup_context: bool,
        has_training_context: bool,
        explicit_run_ids: list[str],
        has_explicit_training_target: bool,
        metric_signals: list[str],
        asks_metric_terms: bool,
    ) -> dict[str, Any] | None:
        if wants_training_run_list and not wants_predict and not training_command_like:
            cached_runs = self._training_history_request_cached_result(request='runs')
            if cached_runs:
                tool_name, payload = cached_runs
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_direct_tool_reply('list_training_runs')

        knowledge_followup = await self._try_handle_knowledge_followup(
            enabled=(
                has_knowledge_followup_context
                and not has_training_context
                and not wants_predict
                and not training_command_like
                and not wants_stop_training
                and not explicit_run_ids
                and not has_explicit_training_target
                and not wants_training_run_compare
                and not wants_best_training_run
                and not wants_training_run_inspect
                and not wants_training_run_list
                and not wants_failed_training_run_list
                and not wants_completed_training_run_list
                and not wants_stopped_training_run_list
                and not wants_running_training_run_list
                and not wants_analysis_ready_run_list
                and not wants_training_loop_list
                and not wants_training_loop_status
                and not wants_inspect_training_loop
                and not wants_pause_training_loop
                and not wants_resume_training_loop
                and not wants_stop_training_loop
                and not wants_training_loop_start
            ),
            user_text=user_text,
            normalized_text=normalized_text,
            metric_signals=metric_signals,
            asks_metric_terms=asks_metric_terms,
        )
        if knowledge_followup:
            return knowledge_followup

        training_followup = await self._try_handle_training_followup(
            enabled=(
                has_training_followup_context
                and not wants_predict
                and not training_command_like
                and not wants_stop_training
                and not explicit_run_ids
                and not has_explicit_training_target
                and not wants_training_run_compare
                and not wants_best_training_run
                and not wants_training_run_inspect
                and not wants_training_run_list
                and not wants_failed_training_run_list
                and not wants_completed_training_run_list
                and not wants_stopped_training_run_list
                and not wants_running_training_run_list
                and not wants_analysis_ready_run_list
                and not wants_training_loop_list
                and not wants_training_loop_status
                and not wants_inspect_training_loop
                and not wants_pause_training_loop
                and not wants_resume_training_loop
                and not wants_stop_training_loop
                and not wants_training_loop_start
            ),
            user_text=user_text,
            normalized_text=normalized_text,
            metric_signals=metric_signals,
            asks_metric_terms=asks_metric_terms,
        )
        if training_followup:
            return training_followup

        if wants_training_status and not wants_predict and not training_command_like:
            return await self._complete_direct_tool_reply('check_training_status')

        if wants_stop_training and not wants_predict and not training_command_like:
            return await self._complete_direct_tool_reply('stop_training')

        if not wants_predict and not training_command_like and wants_training_provenance:
            return self._complete_training_provenance_reply()

        if not wants_predict and not training_command_like and wants_training_evidence:
            return self._complete_training_evidence_reply()

        if not wants_predict and not training_command_like and wants_next_step_guidance:
            cached_next_step = self._knowledge_followup_result('next_step')
            if cached_next_step and not dataset_path:
                tool_name, payload = cached_next_step
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_next_training_step_reply(dataset_path if dataset_path else '')

        if not wants_predict and not training_command_like and wants_training_knowledge:
            cached_knowledge = self._knowledge_followup_result('knowledge')
            if cached_knowledge and not metric_signals and not asks_metric_terms:
                tool_name, payload = cached_knowledge
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_knowledge_retrieval_reply(
                topic='training_metrics' if asks_metric_terms else 'workflow',
                stage='post_training',
                signals=metric_signals,
            )

        if not wants_predict and not training_command_like and wants_training_outcome_analysis:
            cached_analysis = self._knowledge_followup_result('analysis')
            if cached_analysis and not metric_signals and not asks_metric_terms:
                tool_name, payload = cached_analysis
                return await self._complete_cached_tool_result_reply(tool_name, payload)
            return await self._complete_training_outcome_analysis_reply()
        return None

    async def _try_handle_training_and_prediction_requests(
        self,
        *,
        thread_id: str,
        user_text: str,
        normalized_text: str,
        dataset_path: str,
        prediction_path: str,
        frame_followup_path: str,
        wants_train: bool,
        wants_predict: bool,
        no_train: bool,
        readiness_only_query: bool,
        wants_training_outcome_analysis: bool,
        wants_next_step_guidance: bool,
        wants_training_knowledge: bool,
        wants_training_loop_start: bool,
        training_command_like: bool,
        wants_training_revision: bool,
        wants_stop_training: bool,
        explicit_run_ids: list[str],
        wants_best_weight_prediction: bool,
        wants_split: bool,
    ) -> dict[str, Any] | None:
        if wants_training_loop_start and not wants_predict and not training_command_like:
            active_dataset_root = str(self.session_state.active_dataset.dataset_root or '').strip()
            active_img_dir = str(self.session_state.active_dataset.img_dir or '').strip()
            can_reuse_session_yaml = not dataset_path or dataset_path in {active_dataset_root, active_img_dir}
            resolved_yaml: str | None = None
            if can_reuse_session_yaml:
                resolved_yaml = str(
                    self.session_state.active_dataset.data_yaml
                    or self.session_state.active_training.data_yaml
                    or ''
                ).strip()
            loop_args = self._collect_requested_training_loop_args(user_text, data_yaml=resolved_yaml)
            return await self._run_training_loop_start_orchestration(
                user_text=user_text,
                thread_id=thread_id,
                dataset_path=dataset_path,
                loop_args=loop_args,
            )

        if (
            self.session_state.active_training.running
            and wants_train
            and wants_training_revision
            and not wants_stop_training
            and not any(token in user_text for token in ('新数据', '新数据集', '另一个数据集', '换数据集', '改数据集'))
            and not wants_predict
            and not explicit_run_ids
            and not any(token in user_text for token in ('数据', '数据集', 'dataset', 'img_dir', 'label_dir', '换成', '改成', '改用', '现在用'))
            and 'resume' not in normalized_text
        ):
            reply = (
                '当前训练还在运行，不能直接热更新 batch、轮数、优化器或设备等核心参数。'
                '如果要改参数，请先停止当前训练，再生成新的训练计划。'
            )
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        if wants_best_weight_prediction and wants_predict and not training_command_like:
            best_selection = self.session_state.active_training.best_run_selection or {}
            best_run = best_selection.get('best_run') or {}
            weight_path = str(best_run.get('best_weight_path') or best_run.get('weights_path') or '').strip()
            if not weight_path:
                reply = '我当前不能直接假定“最佳训练”的权重文件路径；请先查看最佳训练详情，或明确给我可用的权重路径。'
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}

        if prediction_path and wants_predict and not training_command_like:
            model = intent_parsing.extract_model_from_text(user_text) or self.session_state.active_prediction.model or self.session_state.active_training.model
            predict_tool = 'predict_videos' if self._should_use_video_prediction(user_text, prediction_path) else 'predict_images'
            return await self._complete_direct_tool_reply(predict_tool, source_path=prediction_path, model=model)

        if wants_train and not dataset_path and not no_train and not readiness_only_query and not wants_training_outcome_analysis and not wants_next_step_guidance and not wants_training_knowledge:
            requested_model = intent_parsing.extract_model_from_text(user_text)
            missing_fields = ['数据集路径']
            if not requested_model:
                missing_fields.append('预训练权重/模型')
            lines = ['当前还不能开始训练：']
            for field in missing_fields:
                lines.append(f'- 缺少{field}')
            lines.append('请先补充最少必要信息；我至少需要数据集目录，训练时还需要可用的预训练权重/模型。')
            reply = '\n'.join(lines)
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        if dataset_path and wants_train and not no_train and not readiness_only_query and not wants_training_outcome_analysis and not wants_next_step_guidance and not wants_training_knowledge:
            readiness = await self.direct_tool('training_readiness', img_dir=dataset_path)
            await self.direct_tool('list_training_environments')
            requested_args = self._collect_requested_training_args(
                user_text,
                data_yaml=str(readiness.get('resolved_data_yaml') or self.session_state.active_dataset.data_yaml or ''),
            )
            if not str(requested_args.get('model') or '').strip():
                draft_model = str((((self.session_state.active_training.training_plan_draft or {}).get('planned_training_args') or {}).get('model')) or '').strip()
                preserved_model = ''
                if frame_followup_path:
                    preserved_model = draft_model or str(self.session_state.active_training.model or '').strip()
                elif any(token in user_text for token in ('继续', '刚才', '上次', '恢复')):
                    preserved_model = draft_model or str(self.session_state.active_training.model or '').strip()
                if preserved_model:
                    requested_args['model'] = preserved_model
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
                reply = await self._render_training_plan_message(draft, pending=False)
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
                reply = await self._render_training_plan_message(draft, pending=False)
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
                reply = await self._render_training_plan_message(draft, pending=bool(preflight.get('ready_to_start') and not discussion_only))
                self._messages.append(AIMessage(content=reply))
                if preflight.get('ready_to_start') and not discussion_only:
                    pending = {'name': 'start_training', 'args': next_args, 'id': None, 'synthetic': True}
                    self._set_pending_confirmation(thread_id, pending)
                    return self._needs_confirmation_result(thread_id, pending, reply)
                return {'status': 'completed', 'message': reply, 'tool_call': None}

            if readiness.get('preparable'):
                args: dict[str, Any] = {'dataset_path': dataset_path}
                if wants_split:
                    args['force_split'] = True
                explicit_classes_txt = str(requested_args.get('classes_txt') or '').strip()
                if explicit_classes_txt:
                    args['classes_txt'] = explicit_classes_txt
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
                reply = await self._render_training_plan_message(draft, pending=not discussion_only)
                self._messages.append(AIMessage(content=reply))
                if discussion_only:
                    return {'status': 'completed', 'message': reply, 'tool_call': None}
                pending = {'name': 'prepare_dataset_for_training', 'args': args, 'id': None, 'synthetic': True}
                self._set_pending_confirmation(thread_id, pending)
                return self._needs_confirmation_result(thread_id, pending, reply)

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
            reply = await self._render_training_plan_message(draft, pending=False)
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}
        return None

    async def _try_handle_mainline_intent(self, user_text: str, thread_id: str) -> dict[str, Any] | None:
        guardrail = self._try_handle_guardrail_intent(user_text)
        if guardrail is not None:
            return guardrail
        plan_dialogue = await self._try_handle_training_plan_dialogue(user_text, thread_id)
        if plan_dialogue is not None:
            return plan_dialogue
        extracted_dataset_path = self._extract_dataset_path_from_text(user_text)
        frame_followup_path = ''
        if any(token in user_text for token in ('这些帧', '刚才抽的帧', '刚才这些帧', '这些抽出来的帧', '这些图片', '刚才抽的图片')):
            frame_followup_path = str((self.session_state.active_dataset.last_frame_extract or {}).get('output_dir') or '').strip()
        dataset_path = extracted_dataset_path or frame_followup_path or self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir
        prediction_path = self._extract_dataset_path_from_text(user_text) or self.session_state.active_prediction.source_path
        normalized_text = user_text.lower()
        metric_signals = self._extract_metric_signals_from_text(user_text)
        has_training_context = bool(
            self.session_state.active_training.training_run_summary
            or self.session_state.active_training.last_summary
            or self.session_state.active_training.last_status
        )
        has_knowledge_followup_context = bool(
            self.session_state.active_knowledge.last_retrieval
            or self.session_state.active_knowledge.last_analysis
            or self.session_state.active_knowledge.last_recommendation
        )
        has_training_followup_context = bool(
            has_training_context
            or self.session_state.active_knowledge.last_analysis
            or self.session_state.active_knowledge.last_recommendation
            or self.session_state.active_knowledge.last_retrieval
        )
        has_training_history_followup_context = bool(
            self.session_state.active_training.recent_runs
            or self.session_state.active_training.last_run_inspection
            or self.session_state.active_training.last_run_comparison
            or self.session_state.active_training.best_run_selection
        )
        has_training_loop_history_followup_context = bool(
            self.session_state.active_training.recent_loops
            or self.session_state.active_training.last_loop_status
            or self.session_state.active_training.last_loop_detail
        )
        training_status_phrase = (
            any(token in user_text for token in (
                '训练状态', '当前训练状态', '训练进度', '当前进度',
                '还在训练吗', '训练还在吗', '刚才训练还在吗', '上次训练还在吗', '还在跑吗',
                '训练到哪了', '训练到第几轮', '跑到第几轮',
                '第几轮了', '到第几轮了', '现在第几轮了',
                '现在状态呢', '训练情况', '查看训练情况', '看看训练情况', '看训练情况',
                '训练停了吗', '停了吗', '训练结束了吗', '结束了没', '跑完了吗', '训练完成了吗',
                '训练失败了吗', '失败了吗', '是不是训练失败了', '是不是失败了', '训练挂了吗',
                '查看训练状态', '再次查看训练状态', '看一下训练状态', '再看一下训练状态',
            ))
            or any(token in normalized_text for token in (
                'training status', 'training progress', 'check status',
                'is training done', 'training finished', 'training failed', 'did training fail'
            ))
        )
        wants_train = (
            any(token in normalized_text for token in ('train', 'fine-tune', 'fit'))
            or any(token in user_text for token in ('训练', '开训', '重训', '重新训', '训一下', '直接训'))
        ) and not training_status_phrase
        no_train = any(token in user_text for token in ('不要训练', '不训练', '只检查', '仅检查', '不要启动'))
        wants_duplicates = ('重复' in user_text) or ('duplicate' in normalized_text)
        wants_health = any(token in user_text for token in ('损坏', '尺寸异常', '健康检查', '健康状况', '图片质量'))
        wants_quality = any(token in user_text for token in ('质量问题', '质量风险', '数据集质量'))
        has_dataset_followup_context = bool(
            self.session_state.active_dataset.dataset_root
            or self.session_state.active_dataset.img_dir
            or self.session_state.active_dataset.data_yaml
            or self.session_state.active_dataset.last_scan
            or self.session_state.active_dataset.last_validate
            or self.session_state.active_dataset.last_health_check
            or self.session_state.active_dataset.last_duplicate_check
        )
        has_extract_followup_context = bool(
            self.session_state.active_dataset.last_extract_preview
            or self.session_state.active_dataset.last_extract_result
            or self.session_state.active_dataset.last_video_scan
            or self.session_state.active_dataset.last_frame_extract
        )
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
        wants_extract_frames = (
            any(token in user_text for token in ('抽帧', '提帧'))
            or (
                '视频' in user_text
                and any(token in user_text for token in ('抽一版', '抽一遍', '再抽一版', '再抽一遍', '重新抽一版', '重新抽一遍'))
            )
            or ('extract frames' in normalized_text)
        )
        prediction_followup_only_tokens = (
            '预测结果',
            '预测摘要',
            '预测报告',
            '总结预测',
            '总结一下预测',
            '分析预测',
            '输出目录',
            '结果目录',
            '路径清单',
            '整理预测结果',
            '整理预测产物',
        )
        wants_predict = (
            bool(re.search(r'\b(predict|prediction|infer|inference)\b', normalized_text))
            or any(token in user_text for token in ('预测', '推理', '识别'))
        ) and not any(token in user_text for token in prediction_followup_only_tokens)
        wants_prediction_summary = any(token in user_text for token in ('预测结果', '预测摘要', '总结一下预测', '刚才预测'))
        has_prediction_followup_context = bool(
            self.session_state.active_prediction.report_path
            or self.session_state.active_prediction.output_dir
            or self.session_state.active_prediction.last_result
        )
        has_prediction_management_followup_context = bool(
            self.session_state.active_prediction.last_inspection
            or self.session_state.active_prediction.last_export
            or self.session_state.active_prediction.last_path_lists
            or self.session_state.active_prediction.last_organized_result
        )
        wants_remote_profile_list = any(
            token in user_text
            for token in (
                '远端配置', '服务器配置', '远端 profile', 'remote profile', '可用服务器', '可用节点', '有哪些节点', '有哪些服务器', 'SSH alias'
            )
        ) or any(token in normalized_text for token in ('list remote profiles', 'list remote servers', 'list ssh aliases'))
        wants_remote_upload = (
            any(token in user_text for token in ('上传', '传到服务器', '传到远端', '同步到服务器', '同步到远端', '发到服务器'))
            or any(token in normalized_text for token in ('upload', 'scp', 'sync to server', 'sync to remote'))
        ) and any(
            token in user_text or token in normalized_text
            for token in ('服务器', '远端', '节点', 'server', 'remote')
        )
        wants_remote_result_return = any(
            token in user_text or token in normalized_text
            for token in ('拉回来', '拉回本机', '回传', '取回', '下载回来', 'download back', 'bring back', 'pull back')
        )
        has_remote_transfer_followup_context = bool(
            self.session_state.active_remote_transfer.last_upload
            or self.session_state.active_remote_transfer.last_download
            or self.session_state.active_remote_transfer.last_profile_listing
            or self.session_state.active_remote_transfer.target_label
            or self.session_state.active_remote_transfer.remote_root
        )
        has_remote_roundtrip_followup_context = bool(
            self.session_state.active_training.last_remote_roundtrip
            or self.session_state.active_prediction.last_remote_roundtrip
        )
        wants_remote_prediction_pipeline = wants_remote_upload and wants_predict and not wants_train
        wants_remote_training_pipeline = wants_remote_upload and wants_train
        wants_prediction_output_inspection = (
            any(token in user_text for token in ('输出目录', '结果目录', '保存在哪', '产物路径', '输出里有什么', '有哪些结果文件'))
            and (wants_predict or has_prediction_followup_context)
        )
        wants_prediction_report_export = (
            (any(token in user_text for token in ('导出', '导成', '生成')) and '报告' in user_text)
            and (wants_predict or has_prediction_followup_context)
        )
        wants_prediction_path_lists = (
            any(token in user_text for token in ('路径清单', '结果清单', '命中清单', '空结果清单'))
            and (wants_predict or has_prediction_followup_context)
        )
        wants_prediction_result_organize = any(
            token in user_text for token in ('只看有命中的结果', '只保留有命中', '整理预测结果', '整理预测产物', '按类别整理预测', '按类别整理结果', '把命中的结果单独列出来', '把有目标的结果单独列出来')
        ) and (wants_predict or has_prediction_followup_context)
        rtsp_url = intent_parsing.extract_rtsp_url_from_text(user_text)
        has_realtime_context = bool(
            self.session_state.active_prediction.realtime_session_id
            or self.session_state.active_prediction.realtime_status
            or self.session_state.active_prediction.last_realtime_status
        )
        mentions_camera = any(token in user_text for token in ('摄像头', 'camera', 'webcam'))
        mentions_screen = any(token in user_text for token in ('屏幕', 'screen', '显示器'))
        mentions_rtsp = bool(rtsp_url) or 'rtsp' in normalized_text
        wants_camera_scan = any(token in user_text for token in ('扫描摄像头', '扫描可用摄像头', '摄像头列表', '可用摄像头', '有哪些摄像头')) or (
            mentions_camera and any(token in user_text for token in ('扫描', '列出', '看看有哪些', '有哪些'))
        ) or any(token in normalized_text for token in ('scan camera', 'scan cameras', 'camera list'))
        wants_screen_scan = any(token in user_text for token in ('扫描屏幕', '扫描可用屏幕', '屏幕列表', '可用屏幕', '有哪些屏幕')) or (
            mentions_screen and any(token in user_text for token in ('扫描', '列出', '看看有哪些', '有哪些'))
        ) or any(token in normalized_text for token in ('scan screen', 'scan screens', 'screen list'))
        wants_realtime_start = any(token in user_text for token in ('开始', '启动', '跑', '接入', '接这个', '开启')) or any(
            token in normalized_text for token in ('start', 'begin', 'run')
        )
        wants_realtime_infer = wants_realtime_start or any(token in user_text for token in ('预测', '检测', '识别'))
        wants_camera_prediction = mentions_camera and wants_realtime_infer
        wants_rtsp_prediction = mentions_rtsp and wants_realtime_infer
        wants_screen_prediction = mentions_screen and wants_realtime_infer
        wants_rtsp_test = mentions_rtsp and not wants_rtsp_prediction and any(
            token in user_text for token in ('测试', '测一下', '试一下', '检查', '探测', '能不能用', '可不可用', '通不通')
        )
        wants_realtime_status = any(
            token in user_text for token in ('实时预测状态', '实时预测进度', '摄像头预测状态', 'RTSP 预测状态', 'rtsp 预测状态', '屏幕预测状态')
        ) or (
            has_realtime_context and any(token in user_text for token in ('还在跑吗', '现在状态呢', '当前状态', '处理了多少帧', '跑到哪了', '实时状态'))
        )
        wants_realtime_stop = any(
            token in user_text for token in ('停止实时预测', '停掉实时预测', '停止摄像头预测', '停止 RTSP 预测', '停止rtsp预测', '停止屏幕预测', '停掉摄像头', '停掉rtsp', '停掉屏幕')
        ) or any(token in normalized_text for token in ('stop realtime prediction', 'stop camera prediction', 'stop rtsp prediction', 'stop screen prediction'))
        has_explicit_realtime_target = bool(
            intent_parsing.extract_realtime_session_id_from_text(user_text)
            or intent_parsing.extract_camera_id_from_text(user_text)
            or intent_parsing.extract_screen_id_from_text(user_text)
            or rtsp_url
        )
        asks_metric_terms = any(token in normalized_text for token in ('precision', 'recall', 'map', 'loss', 'epoch', 'epochs', 'batch', 'imgsz', 'patience', 'lr')) or any(token in user_text for token in ('精确率', '召回', '损失', '学习率', '轮数', '批大小'))
        wants_training_outcome_analysis = (
            any(token in user_text for token in ('训练效果怎么样', '这次训练效果怎么样', '训练结果怎么样', '训练效果如何', '结果更像', '训练效果'))
            or any(token in user_text for token in ('是不是已经收敛了', '已经收敛了吗', '收敛了吗'))
            or (
                has_training_context
                and any(token in user_text for token in ('效果怎么样', '结果怎么样', '效果如何', '结果如何'))
            )
            or (asks_metric_terms and any(token in user_text for token in ('怎么看', '说明什么', '意味着什么', '结果如何')))
        )
        wants_training_status = training_status_phrase
        references_prior_statement = any(token in user_text for token in ('你上次不是说', '你不是说过'))
        wants_training_run_compare = any(token in user_text for token in (
            '对比最近两次训练', '比较最近两次训练', '最近两次训练对比',
            '对比两次训练', '比较两次训练', '训练结果对比', '训练记录对比',
            '刚刚那次和上次比哪个好',
        )) or any(token in normalized_text for token in ('compare training runs', 'compare last two runs'))
        wants_best_training_run = any(token in user_text for token in (
            '最近哪次训练最好', '哪次训练最好', '最好的训练记录', '最好的训练结果',
            '最近哪次最值得参考', '哪次最值得参考',
            '最值得参考的训练记录', '最值得参考的训练结果',
        )) or any(token in normalized_text for token in ('best training run', 'best run'))
        if references_prior_statement:
            wants_training_run_compare = False
            wants_best_training_run = False
        wants_stop_training = any(token in user_text for token in (
            '停止训练', '停掉训练', '停一下训练', '先停训练', '先把训练停掉', '停止当前训练', '先停一下', '再停一次', '再停一下',
        )) or any(token in normalized_text for token in ('stop training', 'stop current training'))
        explicit_run_ids = self._extract_training_run_ids_from_text(user_text)
        loop_route = await self._resolve_training_loop_route(
            user_text=user_text,
            normalized_text=normalized_text,
            wants_predict=wants_predict,
            wants_train=wants_train,
            wants_stop_training=wants_stop_training,
            explicit_run_ids=explicit_run_ids,
        )
        has_training_loop_context = bool(loop_route.get('has_context'))
        wants_training_loop_start = loop_route.get('action') == 'start'
        wants_training_loop_status = loop_route.get('action') == 'status'
        wants_training_loop_list = loop_route.get('action') == 'list'
        wants_pause_training_loop = loop_route.get('action') == 'pause'
        wants_resume_training_loop = loop_route.get('action') == 'resume'
        wants_stop_training_loop = loop_route.get('action') == 'stop'
        wants_inspect_training_loop = loop_route.get('action') == 'inspect'
        wants_training_revision = any(
            token in normalized_text or token in user_text
            for token in (
                'batch', 'imgsz', 'device', 'epochs', '轮数', '轮', 'optimizer', '优化器',
                'freeze', '冻结', 'lr0', '学习率', 'resume', 'project', 'name',
                'fraction', 'classes', '类别', 'single_cls', '环境', '继续训练', '别停', '不要停',
            )
        )
        wants_training_run_list = any(token in user_text for token in (
            '最近训练有哪些', '最近一次训练', '训练历史', '训练记录'
        )) or any(token in normalized_text for token in ('recent training runs', 'training history', 'list training runs'))
        wants_failed_training_run_list = any(token in user_text for token in ('失败的训练', '失败训练', '失败记录'))
        wants_completed_training_run_list = any(token in user_text for token in ('已完成的训练', '完成的训练', '跑完的训练'))
        wants_stopped_training_run_list = any(token in user_text for token in ('停止的训练', '中断的训练', '停掉的训练'))
        wants_running_training_run_list = any(token in user_text for token in ('运行中的训练', '还在跑的训练', '正在训练的记录'))
        wants_analysis_ready_run_list = any(token in user_text for token in ('可分析的训练', '有完整指标的训练', '值得分析的训练'))
        explicit_run_outcome_phrase = bool(explicit_run_ids) and any(
            token in user_text for token in (
                '效果怎么样', '结果怎么样', '效果如何', '结果如何',
                '怎么看', '说明什么', '意味着什么',
                '是不是已经收敛了', '已经收敛了吗', '收敛了吗',
            )
        )
        explicit_compare_hint = any(token in user_text for token in ('对比', '比较', '哪个好', '哪次更好'))
        repeat_training_run_compare = any(
            token in user_text
            for token in (
                '刚才那个对比再比较一次',
                '刚才那个对比再来一次',
                '把刚才那两次训练再比较一次',
                '把刚才那两次训练重新比较',
                '再比较一次',
                '再对比一次',
                '重新比较一下',
                '重新对比一下',
            )
        )
        comparison_run_ids: list[str] = list(explicit_run_ids)
        if references_prior_statement:
            comparison_run_ids = []
        if repeat_training_run_compare and not comparison_run_ids:
            last_comparison = self.session_state.active_training.last_run_comparison or {}
            left_run = last_comparison.get('left_run') or {}
            right_run = last_comparison.get('right_run') or {}
            left_run_id = str(left_run.get('run_id') or left_run.get('log_file') or '').strip()
            right_run_id = str(right_run.get('run_id') or right_run.get('log_file') or '').strip()
            if left_run_id:
                comparison_run_ids.append(left_run_id)
            if right_run_id:
                comparison_run_ids.append(right_run_id)
        wants_training_run_compare = wants_training_run_compare or (bool(comparison_run_ids) and (repeat_training_run_compare or explicit_compare_hint))
        wants_training_run_inspect = (not references_prior_statement) and bool(explicit_run_ids) and any(
            token in user_text for token in ('详情', '记录', '具体情况')
        )
        wants_next_step_guidance = any(token in user_text for token in ('下一步', '先补数据还是先调参数', '先补数据', '先调参数', '怎么优化', '如何优化下一步训练', '下一轮怎么做'))
        wants_training_knowledge = bool(metric_signals) or (asks_metric_terms and any(token in user_text for token in ('说明什么', '什么意思', '意味着什么', '怎么看')))
        wants_training_provenance = any(token in user_text for token in (
            '你基于哪次训练说的', '你是基于哪次训练说的', '基于哪次训练', '根据哪次训练', '依据哪次训练',
            '你上次不是说', '你不是说过',
        )) and any(
            token in user_text or token in normalized_text
            for token in ('训练', 'run', '最好', '最值得参考', '分析', '结论')
        )
        wants_training_evidence = any(token in user_text for token in (
            '依据是什么', '根据什么说的', '为什么这么说', '为什么说数据有问题',
        ))
        has_explicit_training_target = bool(
            extracted_dataset_path
            or intent_parsing.extract_model_from_text(user_text)
            or intent_parsing.extract_epochs_from_text(user_text) is not None
        )
        wants_segmentation_training = wants_train and any(
            token in user_text or token in normalized_text
            for token in ('分割', 'segmentation', 'segment', 'sam')
        )
        prediction_only_with_training_exclusion = wants_predict and any(
            token in user_text
            for token in (
                '不要把训练',
                '别把训练',
                '不要混进训练',
                '训练准备的内容混进来',
                '排除训练',
                '只总结预测结果',
            )
        )
        wants_prediction_and_training_mix = wants_predict and not prediction_only_with_training_exclusion and (
            wants_train or wants_training_run_compare or wants_best_training_run
        ) and any(token in user_text for token in ('然后', '再', '同时', '边训练边', '一边'))
        wants_prediction_result_as_training_data = not prediction_only_with_training_exclusion and wants_train and any(
            token in user_text for token in ('预测结果', 'prediction 结果', '预测输出', '推理结果', '识别结果')
        )
        wants_merge_extract_into_training = any(token in user_text for token in ('合并', '并到', '合到')) and any(
            token in user_text for token in ('抽帧', '帧', '旧数据集', '训练')
        )
        wants_best_weight_prediction = wants_predict and any(
            token in user_text or token in normalized_text
            for token in ('最佳训练', '最好权重', 'best run', 'best weight')
        )
        wants_continuous_parallel_predict = wants_train and wants_predict and any(
            token in user_text for token in ('边训练边', '不断做视频预测', '一直预测', '同时不断预测')
        )
        wants_best_training_run_analysis = wants_best_training_run and (
            wants_training_outcome_analysis
            or any(token in user_text for token in ('怎么看', '结果怎么样', '结果如何', '效果怎么样', '效果如何', '意味着什么'))
        )
        wants_training_outcome_analysis = wants_training_outcome_analysis or explicit_run_outcome_phrase
        readiness_only_query = wants_readiness and (no_train or any(token in user_text for token in ('吗', '是否', '能不能', '可不可以')))
        training_command_like = any(token in user_text for token in ('开始训练', '启动训练', '训练这个数据', '用这个数据训练', '直接开训', 'start_training'))

        if wants_segmentation_training and not wants_predict:
            reply = '当前训练主线先按 YOLO detection 做稳定交付；分割/SAM 训练暂不在这条主线上直接执行。'
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        if wants_continuous_parallel_predict:
            reply = '当前不支持“边训练边持续做视频预测”这种高资源并发编排；请先明确主任务，或分成独立步骤执行。'
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        if wants_prediction_and_training_mix:
            reply = '这条请求同时混了预测、训练或训练比较；为了避免串扰，请拆成连续步骤，我会按顺序执行。'
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        if wants_prediction_result_as_training_data:
            reply = '预测结果目录不能直接当训练数据开训；如果要用于训练，先确认是否有可用标签，再走数据准备/校验链。'
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        if wants_merge_extract_into_training:
            reply = '抽帧结果或旧数据集合并后，应该先走数据准备/校验，再决定是否训练；我不会直接把它们无检查地并进训练。'
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        realtime_request = await self._try_handle_realtime_requests(
            user_text=user_text,
            normalized_text=normalized_text,
            wants_train=wants_train,
            wants_camera_scan=wants_camera_scan,
            wants_screen_scan=wants_screen_scan,
            wants_rtsp_test=wants_rtsp_test,
            wants_realtime_status=wants_realtime_status,
            wants_realtime_stop=wants_realtime_stop,
            wants_camera_prediction=wants_camera_prediction,
            wants_rtsp_prediction=wants_rtsp_prediction,
            wants_screen_prediction=wants_screen_prediction,
            has_realtime_context=has_realtime_context,
            has_explicit_realtime_target=has_explicit_realtime_target,
            rtsp_url=rtsp_url,
        )
        if realtime_request:
            return realtime_request

        has_explicit_remote_target = bool(self._extract_remote_server_from_text(user_text) or self._extract_remote_root_from_text(user_text))
        has_explicit_transfer_paths = bool(self._extract_all_paths_from_text(user_text))
        remote_request = await self._try_handle_remote_requests(
            thread_id=thread_id,
            user_text=user_text,
            normalized_text=normalized_text,
            wants_train=wants_train,
            wants_predict=wants_predict,
            training_command_like=training_command_like,
            wants_remote_profile_list=wants_remote_profile_list,
            wants_remote_upload=wants_remote_upload,
            wants_remote_prediction_pipeline=wants_remote_prediction_pipeline,
            wants_remote_training_pipeline=wants_remote_training_pipeline,
            wants_remote_result_return=wants_remote_result_return,
            has_remote_transfer_followup_context=has_remote_transfer_followup_context,
            has_remote_roundtrip_followup_context=has_remote_roundtrip_followup_context,
            has_explicit_remote_target=has_explicit_remote_target,
            has_explicit_transfer_paths=has_explicit_transfer_paths,
        )
        if remote_request:
            return remote_request

        has_explicit_prediction_target = bool(self._extract_dataset_path_from_text(user_text) or intent_parsing.extract_model_from_text(user_text))
        prediction_request = await self._try_handle_prediction_requests(
            user_text=user_text,
            normalized_text=normalized_text,
            prediction_path=prediction_path,
            wants_train=wants_train,
            training_command_like=training_command_like,
            wants_scan_videos=wants_scan_videos,
            wants_extract_frames=wants_extract_frames,
            wants_extract_preview=wants_extract_preview,
            wants_extract_images=wants_extract_images,
            wants_remote_upload=wants_remote_upload,
            wants_prediction_summary=wants_prediction_summary,
            wants_prediction_output_inspection=wants_prediction_output_inspection,
            wants_prediction_report_export=wants_prediction_report_export,
            wants_prediction_path_lists=wants_prediction_path_lists,
            wants_prediction_result_organize=wants_prediction_result_organize,
            has_prediction_followup_context=has_prediction_followup_context,
            has_prediction_management_followup_context=has_prediction_management_followup_context,
            has_explicit_prediction_target=has_explicit_prediction_target,
        )
        if prediction_request:
            return prediction_request

        dataset_or_extract_request = await self._try_handle_dataset_and_extract_requests(
            user_text=user_text,
            normalized_text=normalized_text,
            dataset_path=dataset_path,
            prediction_path=prediction_path,
            extracted_dataset_path=extracted_dataset_path,
            wants_train=wants_train,
            wants_predict=wants_predict,
            training_command_like=training_command_like,
            wants_remote_upload=wants_remote_upload,
            wants_scan_videos=wants_scan_videos,
            wants_extract_frames=wants_extract_frames,
            wants_extract_preview=wants_extract_preview,
            wants_extract_images=wants_extract_images,
            wants_quality=wants_quality,
            wants_health=wants_health,
            wants_duplicates=wants_duplicates,
            wants_readiness=wants_readiness,
            has_extract_followup_context=has_extract_followup_context,
            has_dataset_followup_context=has_dataset_followup_context,
        )
        if dataset_or_extract_request:
            return dataset_or_extract_request

        training_history_or_loop = await self._try_handle_training_history_and_loop_requests(
            user_text=user_text,
            normalized_text=normalized_text,
            has_training_history_followup_context=has_training_history_followup_context,
            has_training_loop_history_followup_context=has_training_loop_history_followup_context,
            has_training_context=has_training_context,
            wants_predict=wants_predict,
            training_command_like=training_command_like,
            explicit_run_ids=explicit_run_ids,
            wants_training_run_inspect=wants_training_run_inspect,
            wants_failed_training_run_list=wants_failed_training_run_list,
            wants_completed_training_run_list=wants_completed_training_run_list,
            wants_stopped_training_run_list=wants_stopped_training_run_list,
            wants_running_training_run_list=wants_running_training_run_list,
            wants_analysis_ready_run_list=wants_analysis_ready_run_list,
            wants_training_loop_list=wants_training_loop_list,
            wants_training_loop_status=wants_training_loop_status,
            wants_inspect_training_loop=wants_inspect_training_loop,
            wants_pause_training_loop=wants_pause_training_loop,
            wants_resume_training_loop=wants_resume_training_loop,
            wants_stop_training_loop=wants_stop_training_loop,
            comparison_run_ids=comparison_run_ids,
            wants_training_run_compare=wants_training_run_compare,
            wants_next_step_guidance=wants_next_step_guidance,
            wants_best_training_run=wants_best_training_run_analysis or wants_best_training_run,
            wants_training_outcome_analysis=wants_training_outcome_analysis,
            loop_route=loop_route,
        )
        if training_history_or_loop:
            return training_history_or_loop

        training_context_request = await self._try_handle_training_context_requests(
            user_text=user_text,
            normalized_text=normalized_text,
            dataset_path=dataset_path,
            wants_predict=wants_predict,
            training_command_like=training_command_like,
            wants_stop_training=wants_stop_training,
            wants_training_provenance=wants_training_provenance,
            wants_training_evidence=wants_training_evidence,
            wants_next_step_guidance=wants_next_step_guidance,
            wants_training_knowledge=wants_training_knowledge,
            wants_training_outcome_analysis=wants_training_outcome_analysis,
            wants_training_status=wants_training_status,
            wants_training_run_list=wants_training_run_list,
            wants_training_run_compare=wants_training_run_compare,
            wants_best_training_run=wants_best_training_run,
            wants_training_run_inspect=wants_training_run_inspect,
            wants_failed_training_run_list=wants_failed_training_run_list,
            wants_completed_training_run_list=wants_completed_training_run_list,
            wants_stopped_training_run_list=wants_stopped_training_run_list,
            wants_running_training_run_list=wants_running_training_run_list,
            wants_analysis_ready_run_list=wants_analysis_ready_run_list,
            wants_training_loop_list=wants_training_loop_list,
            wants_training_loop_status=wants_training_loop_status,
            wants_inspect_training_loop=wants_inspect_training_loop,
            wants_pause_training_loop=wants_pause_training_loop,
            wants_resume_training_loop=wants_resume_training_loop,
            wants_stop_training_loop=wants_stop_training_loop,
            wants_training_loop_start=wants_training_loop_start,
            has_knowledge_followup_context=has_knowledge_followup_context,
            has_training_followup_context=has_training_followup_context,
            has_training_context=has_training_context,
            explicit_run_ids=explicit_run_ids,
            has_explicit_training_target=has_explicit_training_target,
            metric_signals=metric_signals,
            asks_metric_terms=asks_metric_terms,
        )
        if training_context_request:
            return training_context_request

        training_or_prediction_request = await self._try_handle_training_and_prediction_requests(
            thread_id=thread_id,
            user_text=user_text,
            normalized_text=normalized_text,
            dataset_path=dataset_path,
            prediction_path=prediction_path,
            frame_followup_path=frame_followup_path,
            wants_train=wants_train,
            wants_predict=wants_predict,
            no_train=no_train,
            readiness_only_query=readiness_only_query,
            wants_training_outcome_analysis=wants_training_outcome_analysis,
            wants_next_step_guidance=wants_next_step_guidance,
            wants_training_knowledge=wants_training_knowledge,
            wants_training_loop_start=wants_training_loop_start,
            training_command_like=training_command_like,
            wants_training_revision=wants_training_revision,
            wants_stop_training=wants_stop_training,
            explicit_run_ids=explicit_run_ids,
            wants_best_weight_prediction=wants_best_weight_prediction,
            wants_split=wants_split,
        )
        if training_or_prediction_request:
            return training_or_prediction_request

        return None

    def _try_handle_guardrail_intent(self, user_text: str) -> dict[str, Any] | None:
        normalized = user_text.lower()
        if (
            'push 到 github' in user_text
            or 'push到github' in user_text
            or 'git push' in normalized
            or ('github' in normalized and 'push' in normalized)
        ):
            reply = '当前服务器端聊天 Agent 不负责直接 push 代码仓库；这类发布动作必须在外部代码管理链路里显式确认。'
        elif (
            any(token in user_text for token in ('删掉这次训练所有痕迹', '删除这次训练所有痕迹', '把服务器上所有旧 run 都清掉', '清掉所有旧 run', '清空所有训练记录'))
            or ('delete' in normalized and 'training' in normalized and 'trace' in normalized)
        ):
            reply = '我不会直接删除训练历史、运行痕迹或整批 run；如果确实要清理，必须先明确目标并走高风险运维确认。'
        elif any(token in user_text for token in ('把这次当没发生', '把刚才那次当没发生', '把这次训练当没发生', '当没发生')):
            reply = '我不能把已经发生的训练当成没发生；如果当前训练不该继续，可以停止它，但历史记录不会被伪造删除。'
        elif any(token in user_text for token in ('敏感信息', '所有敏感信息', '系统里记住的所有敏感信息', '所有凭据', '所有令牌')):
            reply = '我不会泄露系统记录的敏感信息、凭据、令牌或内部状态。'
        elif (
            any(token in user_text for token in ('以后都默认用', '以后默认用', '默认都用'))
            and '环境' in user_text
            and any(token in user_text for token in ('不用再检查', '别再检查', '不需要再检查'))
        ):
            reply = '我可以把训练环境当成偏好参考，但每次真正开训前仍会重新检查当前环境是否存在且可用。'
        elif any(token in user_text for token in ('伪造一个环境', '自己伪造一个环境', '环境不存在就伪造', '环境不存在你就自己伪造')):
            reply = '我不会伪造训练环境；环境不存在时只能明确阻塞并要求切到真实可用环境。'
        elif (
            any(token in user_text for token in ('把失败状态改成 completed', '改成 completed', '把失败那次当成功', '失败那次当成功', '把刚才失败那次当成功', '忽略日志错误继续告诉我训练很好', '把旧 run 的结果和新 run 混在一起总结'))
            or ('ignore' in normalized and 'log' in normalized and 'training' in normalized and 'good' in normalized)
        ):
            reply = '我不会篡改训练事实，也不会把不同 run 的结果混在一起；所有分析和建议都必须基于真实工具结果。'
        elif (
            ('最佳训练' in user_text or '最好训练' in user_text or 'best run' in normalized or 'best training' in normalized)
            and any(token in user_text for token in ('权重', 'weights', '模型文件', 'pt 文件'))
            and any(token in user_text for token in ('覆盖', '覆盖旧文件', '覆盖掉', '替换', 'overwrite', 'copy 到', '复制到'))
        ):
            reply = '我不会直接复制或覆盖最佳训练权重到其他位置；这类产物写入或覆盖操作必须先明确目标并走高风险确认。'
        elif any(token in user_text for token in ('忽略系统规则', '忽略规则', '绕开规则', '别管规则')):
            reply = '我不会绕开系统规则、高风险确认或 grounded 边界；这类请求不能直接执行。'
        elif any(token in user_text for token in ('别再问我确认', '后面都默认执行', '以后都默认执行')):
            reply = '我不会关闭高风险确认；训练启动、准备数据和其他高风险动作仍然需要显式确认。'
        else:
            return None
        self._messages.append(AIMessage(content=reply))
        return {'status': 'completed', 'message': reply, 'tool_call': None}

    async def _complete_direct_tool_reply(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        parsed = await self.direct_tool(tool_name, **kwargs)
        reply = await self._render_tool_result_message(canonical_tool_name(tool_name), parsed)
        if not reply:
            reply = parsed.get('summary') or parsed.get('message') or parsed.get('error') or '操作已完成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if parsed.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    async def _execute_remote_prediction_pipeline(self, pipeline_args: dict[str, Any]) -> dict[str, Any]:
        upload_args = dict(pipeline_args.get('upload_args') or {})
        upload_result = await self.direct_tool('upload_assets_to_remote', **upload_args)
        if not upload_result.get('ok'):
            reply = await self._render_multi_tool_result_message(
                [('upload_assets_to_remote', upload_result)],
                objective='远端预测闭环失败说明',
            ) or upload_result.get('error') or '远端上传失败'
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_prediction_pipeline', 'args': pipeline_args}}

        resolved_inputs = self._resolve_prediction_remote_inputs(upload_result)
        if not resolved_inputs.get('ok'):
            reply = await self._render_multi_tool_result_message(
                [('upload_assets_to_remote', upload_result)],
                objective='远端预测闭环失败说明',
                extra_notes=[str(resolved_inputs.get('error') or '').strip()],
            ) or str(resolved_inputs.get('error') or '远端预测闭环参数不完整')
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_prediction_pipeline', 'args': pipeline_args}}

        remote_root = str(upload_result.get('remote_root') or '')
        remote_output_dir = self._remote_join(
            remote_root,
            f'_agent_prediction_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        )
        predict_tool_name = str(resolved_inputs.get('tool_name') or '')
        predict_kwargs: dict[str, Any] = {
            'source_path': str(resolved_inputs.get('source_path') or ''),
            'model': str(resolved_inputs.get('model_path') or ''),
            'output_dir': remote_output_dir,
            'generate_report': True,
        }
        if predict_tool_name == 'predict_videos':
            predict_kwargs.update({
                'save_video': False,
                'save_keyframes_annotated': True,
                'save_keyframes_raw': False,
            })
        else:
            predict_kwargs.update({
                'save_annotated': True,
                'save_labels': False,
                'save_original': False,
            })
        predict_result = await self.direct_tool(predict_tool_name, **predict_kwargs)
        if not predict_result.get('ok'):
            reply = await self._render_multi_tool_result_message(
                [
                    ('upload_assets_to_remote', upload_result),
                    (predict_tool_name, predict_result),
                ],
                objective='远端预测闭环失败说明',
            ) or predict_result.get('error') or '远端预测执行失败'
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_prediction_pipeline', 'args': pipeline_args}}

        download_result: dict[str, Any] = {}
        if pipeline_args.get('download_after_predict', True):
            download_args = {
                'remote_paths': [str(predict_result.get('output_dir') or remote_output_dir)],
                'server': upload_args.get('server', ''),
                'profile': upload_args.get('profile', ''),
                'host': upload_args.get('host', ''),
                'username': upload_args.get('username', ''),
                'port': upload_args.get('port', 0),
                'local_root': pipeline_args.get('local_result_root', ''),
                'recursive': True,
            }
            download_result = await self.direct_tool('download_assets_from_remote', **download_args)

        pipeline_result = {
            'ok': predict_result.get('ok') is True and (not download_result or download_result.get('ok') is True),
            'upload': upload_result,
            'predict': predict_result,
            'download': download_result,
            'remote_source_path': str(resolved_inputs.get('source_path') or ''),
            'remote_model_path': str(resolved_inputs.get('model_path') or ''),
            'remote_output_dir': str(predict_result.get('output_dir') or remote_output_dir),
            'local_result_root': str((download_result or {}).get('local_root') or pipeline_args.get('local_result_root') or ''),
            'source_kind': str(resolved_inputs.get('source_kind') or ''),
            'predict_tool_name': predict_tool_name,
        }
        pipeline_result['pipeline_overview'] = {
            'target_label': str(upload_result.get('target_label') or upload_args.get('server') or '').strip(),
            'remote_root': str(upload_result.get('remote_root') or upload_args.get('remote_root') or '').strip(),
            'remote_source_path': pipeline_result['remote_source_path'],
            'remote_model_path': pipeline_result['remote_model_path'],
            'remote_output_dir': pipeline_result['remote_output_dir'],
            'local_result_root': pipeline_result['local_result_root'],
            'source_kind': pipeline_result['source_kind'],
        }
        pipeline_result['execution_overview'] = {
            'upload_ok': bool(upload_result.get('ok')),
            'predict_ok': bool(predict_result.get('ok')),
            'download_ok': bool((not download_result) or download_result.get('ok')),
            'predict_tool_name': predict_tool_name,
            'download_after_predict': bool(pipeline_args.get('download_after_predict', True)),
        }
        action_candidates: list[dict[str, Any]] = []
        if pipeline_result['local_result_root']:
            action_candidates.append({
                'tool': 'inspect_prediction_outputs',
                'description': f"可继续查看本机结果目录: {pipeline_result['local_result_root']}",
            })
        elif pipeline_result['remote_output_dir']:
            action_candidates.append({
                'tool': 'download_assets_from_remote',
                'description': f"如需回传，可继续下载远端预测目录: {pipeline_result['remote_output_dir']}",
            })
        report_path = str((predict_result.get('report_path') or '')).strip()
        if report_path:
            action_candidates.append({
                'tool': 'summarize_prediction_results',
                'description': f"可继续汇总远端预测报告: {report_path}",
            })
        if action_candidates:
            pipeline_result['action_candidates'] = action_candidates[:4]
        self.session_state.active_prediction.last_remote_roundtrip = pipeline_result
        self.memory.append_event(self.session_state.session_id, 'remote_prediction_pipeline', pipeline_result)

        reply = await self._render_multi_tool_result_message(
            [
                ('upload_assets_to_remote', upload_result),
                (predict_tool_name, predict_result),
                ('download_assets_from_remote', download_result) if download_result else ('', {}),
            ],
            objective='远端预测闭环执行结果',
        ) or predict_result.get('summary') or '远端预测闭环已完成'
        self._messages.append(AIMessage(content=reply))
        self._trim_history()
        self.memory.save_state(self.session_state)
        return {
            'status': 'completed' if pipeline_result['ok'] else 'error',
            'message': reply,
            'tool_call': {'name': 'remote_prediction_pipeline', 'args': pipeline_args},
        }

    async def _execute_remote_training_pipeline(self, pipeline_args: dict[str, Any]) -> dict[str, Any]:
        upload_args = dict(pipeline_args.get('upload_args') or {})
        upload_result = await self.direct_tool('upload_assets_to_remote', **upload_args)
        if not upload_result.get('ok'):
            reply = await self._render_multi_tool_result_message(
                [('upload_assets_to_remote', upload_result)],
                objective='远端训练闭环失败说明',
            ) or upload_result.get('error') or '远端上传失败'
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_training_pipeline', 'args': pipeline_args}}

        resolved_inputs = self._resolve_training_remote_inputs(upload_result)
        if not resolved_inputs.get('ok'):
            reply = await self._render_multi_tool_result_message(
                [('upload_assets_to_remote', upload_result)],
                objective='远端训练闭环失败说明',
                extra_notes=[str(resolved_inputs.get('error') or '').strip()],
            ) or str(resolved_inputs.get('error') or '远端训练闭环参数不完整')
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_training_pipeline', 'args': pipeline_args}}

        dataset_path = str(resolved_inputs.get('dataset_path') or '')
        model_path = str(resolved_inputs.get('model_path') or '')
        readiness = await self.direct_tool('training_readiness', img_dir=dataset_path)
        if not readiness.get('ok'):
            reply = await self._render_multi_tool_result_message(
                [
                    ('upload_assets_to_remote', upload_result),
                    ('training_readiness', readiness),
                ],
                objective='远端训练闭环失败说明',
            ) or readiness.get('error') or '远端训练前检查失败'
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_training_pipeline', 'args': pipeline_args}}

        prepare_result: dict[str, Any] = {}
        data_yaml = str(readiness.get('resolved_data_yaml') or '').strip()
        if not readiness.get('ready'):
            prepare_args: dict[str, Any] = {'dataset_path': dataset_path}
            if pipeline_args.get('force_split'):
                prepare_args['force_split'] = True
            prepare_result = await self.direct_tool('prepare_dataset_for_training', **prepare_args)
            if not prepare_result.get('ok') or not prepare_result.get('ready'):
                reply = await self._render_multi_tool_result_message(
                    [
                        ('upload_assets_to_remote', upload_result),
                        ('training_readiness', readiness),
                        ('prepare_dataset_for_training', prepare_result),
                    ],
                    objective='远端训练闭环失败说明',
                ) or prepare_result.get('summary') or prepare_result.get('error') or '远端数据准备未通过'
                self._messages.append(AIMessage(content=reply))
                self._trim_history()
                self.memory.save_state(self.session_state)
                return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_training_pipeline', 'args': pipeline_args}}
            data_yaml = str(prepare_result.get('data_yaml') or '').strip()

        requested_args = self._collect_requested_training_args(str(pipeline_args.get('user_text') or ''), data_yaml=data_yaml)
        requested_args['model'] = model_path
        requested_args['data_yaml'] = data_yaml
        requested_args['device'] = str(requested_args.get('device') or 'auto')
        requested_args['epochs'] = int(requested_args.get('epochs', 100))

        preflight = await self.direct_tool(
            'training_preflight',
            model=str(requested_args.get('model') or ''),
            data_yaml=str(requested_args.get('data_yaml') or ''),
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
        if not preflight.get('ok') or not preflight.get('ready_to_start'):
            reply = await self._render_multi_tool_result_message(
                [
                    ('upload_assets_to_remote', upload_result),
                    ('training_readiness', readiness),
                    ('prepare_dataset_for_training', prepare_result) if prepare_result else ('', {}),
                    ('training_preflight', preflight),
                ],
                objective='远端训练闭环失败说明',
            ) or preflight.get('summary') or preflight.get('error') or '远端训练预检未通过'
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_training_pipeline', 'args': pipeline_args}}

        resolved_args = dict(preflight.get('resolved_args') or {})
        start_result = await self.direct_tool(
            'start_training',
            model=str(resolved_args.get('model') or requested_args.get('model') or ''),
            data_yaml=str(resolved_args.get('data_yaml') or requested_args.get('data_yaml') or ''),
            epochs=int(resolved_args.get('epochs') or requested_args.get('epochs', 100)),
            device=str(resolved_args.get('device') or requested_args.get('device') or 'auto'),
            training_environment=str(resolved_args.get('training_environment') or requested_args.get('training_environment') or ''),
            project=str(resolved_args.get('project') or requested_args.get('project') or ''),
            name=str(resolved_args.get('name') or requested_args.get('name') or ''),
            batch=resolved_args.get('batch', requested_args.get('batch')),
            imgsz=resolved_args.get('imgsz', requested_args.get('imgsz')),
            fraction=resolved_args.get('fraction', requested_args.get('fraction')),
            classes=resolved_args.get('classes', requested_args.get('classes')),
            single_cls=resolved_args.get('single_cls', requested_args.get('single_cls')),
            optimizer=str(resolved_args.get('optimizer') or requested_args.get('optimizer') or ''),
            freeze=resolved_args.get('freeze', requested_args.get('freeze')),
            resume=resolved_args.get('resume', requested_args.get('resume')),
            lr0=resolved_args.get('lr0', requested_args.get('lr0')),
            patience=resolved_args.get('patience', requested_args.get('patience')),
            workers=resolved_args.get('workers', requested_args.get('workers')),
            amp=resolved_args.get('amp', requested_args.get('amp')),
        )

        wait_result: dict[str, Any] = {}
        final_status: dict[str, Any] = {}
        final_summary: dict[str, Any] = {}
        final_inspection: dict[str, Any] = {}
        download_result: dict[str, Any] = {}
        remote_result_path = ''

        if start_result.get('ok') and pipeline_args.get('wait_for_completion'):
            poll_interval = pipeline_args.get('poll_interval_seconds', 15)
            max_wait = pipeline_args.get('max_wait_seconds', 7200)
            wait_result = await self._wait_for_remote_training_terminal_state(
                poll_interval_seconds=int(15 if poll_interval is None else poll_interval),
                max_wait_seconds=int(7200 if max_wait is None else max_wait),
            )
            final_status = dict(wait_result.get('status_result') or {})
            final_summary = dict(wait_result.get('summary_result') or {})
            final_inspection = dict(wait_result.get('inspect_result') or {})
            remote_result_path = self._resolve_remote_training_result_path(
                start_result=start_result,
                status_result=final_status,
                summary_result=final_summary,
                inspection_result=final_inspection,
            )
            if wait_result.get('ok') and pipeline_args.get('download_after_completion'):
                if remote_result_path:
                    download_args = {
                        'remote_paths': [remote_result_path],
                        'server': upload_args.get('server', ''),
                        'profile': upload_args.get('profile', ''),
                        'host': upload_args.get('host', ''),
                        'username': upload_args.get('username', ''),
                        'port': upload_args.get('port', 0),
                        'local_root': pipeline_args.get('local_result_root', ''),
                        'recursive': True,
                    }
                    download_result = await self.direct_tool('download_assets_from_remote', **download_args)
                else:
                    download_result = {
                        'ok': False,
                        'summary': '训练已结束，但当前无法解析远端结果目录，未执行自动回传。',
                        'error': 'missing_remote_result_path',
                    }

        final_run_state = str(
            (final_summary.get('run_state') if isinstance(final_summary, dict) else '')
            or (final_status.get('run_state') if isinstance(final_status, dict) else '')
            or ''
        ).strip().lower()
        wait_required = bool(pipeline_args.get('wait_for_completion'))
        wait_ok = True
        if wait_required:
            wait_ok = bool(wait_result.get('ok')) and final_run_state == 'completed'
        download_required = bool(pipeline_args.get('download_after_completion'))
        download_ok = (not download_required) or bool(download_result.get('ok'))
        pipeline_result = {
            'ok': start_result.get('ok') is True and wait_ok and download_ok,
            'upload': upload_result,
            'readiness': readiness,
            'prepare': prepare_result,
            'preflight': preflight,
            'start': start_result,
            'wait': wait_result,
            'final_status': final_status,
            'final_summary': final_summary,
            'final_inspection': final_inspection,
            'download': download_result,
            'remote_dataset_path': dataset_path,
            'remote_model_path': model_path,
            'remote_result_path': remote_result_path,
            'local_result_root': str((download_result or {}).get('local_root') or pipeline_args.get('local_result_root') or ''),
            'wait_for_completion': wait_required,
            'download_after_completion': download_required,
            'final_run_state': final_run_state,
        }
        pipeline_result['pipeline_overview'] = {
            'target_label': str(upload_result.get('target_label') or upload_args.get('server') or '').strip(),
            'remote_root': str(upload_result.get('remote_root') or upload_args.get('remote_root') or '').strip(),
            'remote_dataset_path': pipeline_result['remote_dataset_path'],
            'remote_model_path': pipeline_result['remote_model_path'],
            'remote_result_path': pipeline_result['remote_result_path'],
            'local_result_root': pipeline_result['local_result_root'],
        }
        pipeline_result['execution_overview'] = {
            'upload_ok': bool(upload_result.get('ok')),
            'readiness_ok': bool(readiness.get('ok')),
            'prepare_ok': bool((not prepare_result) or prepare_result.get('ok')),
            'preflight_ok': bool(preflight.get('ok')),
            'start_ok': bool(start_result.get('ok')),
            'wait_ok': bool(wait_ok),
            'download_ok': bool(download_ok),
            'final_run_state': final_run_state,
        }
        action_candidates: list[dict[str, Any]] = []
        if pipeline_result['local_result_root']:
            action_candidates.append({
                'tool': 'summarize_training_run',
                'description': f"可继续查看本机训练产物目录: {pipeline_result['local_result_root']}",
            })
        elif pipeline_result['remote_result_path']:
            action_candidates.append({
                'tool': 'download_assets_from_remote',
                'description': f"如需回传，可继续下载远端训练目录: {pipeline_result['remote_result_path']}",
            })
        if final_run_state == 'completed':
            action_candidates.append({
                'tool': 'summarize_training_run',
                'description': '可继续查看训练总结或下一步建议',
            })
        if action_candidates:
            pipeline_result['action_candidates'] = action_candidates[:4]
        self.session_state.active_training.last_remote_roundtrip = pipeline_result
        self.memory.append_event(self.session_state.session_id, 'remote_training_pipeline', pipeline_result)

        reply = await self._render_multi_tool_result_message(
            [
                ('upload_assets_to_remote', upload_result),
                ('training_readiness', readiness),
                ('prepare_dataset_for_training', prepare_result) if prepare_result else ('', {}),
                ('training_preflight', preflight),
                ('start_training', start_result),
                ('check_training_status', final_status) if final_status else ('', {}),
                ('summarize_training_run', final_summary) if final_summary else ('', {}),
                ('download_assets_from_remote', download_result) if download_result else ('', {}),
            ],
            objective='远端训练闭环执行结果',
            extra_notes=[str(wait_result.get('message') or '').strip()] if wait_result and not wait_result.get('ok') else None,
        ) or start_result.get('summary') or start_result.get('error') or '远端训练闭环已完成'
        self._messages.append(AIMessage(content=reply))
        self._trim_history()
        self.memory.save_state(self.session_state)
        return {
            'status': 'completed' if pipeline_result['ok'] else 'error',
            'message': reply,
            'tool_call': {'name': 'remote_training_pipeline', 'args': pipeline_args},
        }

    async def _wait_for_remote_training_terminal_state(
        self,
        *,
        poll_interval_seconds: int = 15,
        max_wait_seconds: int = 7200,
    ) -> dict[str, Any]:
        started = time.monotonic()
        status_checks: list[dict[str, Any]] = []
        interval = max(0, int(poll_interval_seconds))
        wait_limit = max(1, int(max_wait_seconds))

        while True:
            status_result = await self.direct_tool('check_training_status')
            status_checks.append({
                'summary': status_result.get('summary'),
                'running': status_result.get('running'),
                'run_state': status_result.get('run_state'),
                'save_dir': status_result.get('save_dir'),
                'log_file': status_result.get('log_file'),
            })
            if not status_result.get('ok'):
                return {
                    'ok': False,
                    'message': '训练已启动，但轮询训练状态失败；未执行自动回传。',
                    'status_result': status_result,
                    'status_checks': status_checks,
                }

            run_state = str(status_result.get('run_state') or '').strip().lower()
            if not status_result.get('running') and run_state not in {'', 'running'}:
                summary_result = await self.direct_tool('summarize_training_run')
                inspect_result = await self.direct_tool('inspect_training_run')
                return {
                    'ok': True,
                    'status_result': status_result,
                    'summary_result': summary_result,
                    'inspect_result': inspect_result,
                    'status_checks': status_checks,
                }

            if (time.monotonic() - started) >= wait_limit:
                return {
                    'ok': False,
                    'timed_out': True,
                    'message': f'训练已启动，但在等待窗口 {wait_limit}s 内仍未结束；未执行自动回传。',
                    'status_result': status_result,
                    'status_checks': status_checks,
                }

            if interval > 0:
                await asyncio.sleep(interval)

    def _extract_training_save_dir(self, payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ''
        save_dir = str(payload.get('save_dir') or '').strip()
        if save_dir:
            return save_dir
        for fact in payload.get('facts') or []:
            text = str(fact or '').strip()
            if text.startswith('save_dir='):
                return text.split('=', 1)[1].strip()
        resolved_args = payload.get('resolved_args') or {}
        project = str(resolved_args.get('project') or '').strip()
        name = str(resolved_args.get('name') or '').strip()
        if project and name:
            return self._remote_join(project, name)
        return ''

    def _resolve_remote_training_result_path(
        self,
        *,
        start_result: dict[str, Any],
        status_result: dict[str, Any],
        summary_result: dict[str, Any],
        inspection_result: dict[str, Any],
    ) -> str:
        for payload in (inspection_result, summary_result, status_result, start_result):
            save_dir = self._extract_training_save_dir(payload)
            if save_dir:
                return save_dir
        return ''

    async def _complete_dataset_quality_reply(self, dataset_path: str) -> dict[str, Any]:
        scan = await self.direct_tool('scan_dataset', img_dir=dataset_path)
        validate = await self.direct_tool('validate_dataset', img_dir=dataset_path)
        health = await self.direct_tool('run_dataset_health_check', dataset_path=dataset_path, include_duplicates=True, max_duplicate_groups=3)

        reply = await self._render_multi_tool_result_message(
            [
                ('scan_dataset', scan),
                ('validate_dataset', validate),
                ('run_dataset_health_check', health),
            ],
            objective='数据集质量分析说明',
        )
        if not reply:
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
            action_candidates = (
                validate.get('action_candidates')
                or scan.get('action_candidates')
                or health.get('action_candidates')
                or []
            )
            next_actions = validate.get('next_actions') or scan.get('next_actions') or health.get('next_actions') or []
            if action_candidates:
                lines.append('建议:')
                for item in action_candidates[:2]:
                    if not isinstance(item, dict):
                        continue
                    fragment = str(item.get('description') or item.get('reason') or item.get('tool') or '').strip()
                    if fragment:
                        lines.append(f'- {fragment}')
            elif next_actions:
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
        readiness = await self.direct_tool('dataset_training_readiness', img_dir=dataset_path)
        recommendation = await self.direct_tool(
            'recommend_next_training_step',
            readiness=readiness,
            health=self.session_state.active_dataset.last_health_check,
            status=self.session_state.active_training.training_run_summary
            or self.session_state.active_training.last_summary
            or self.session_state.active_training.last_status,
            comparison=self.session_state.active_training.last_run_comparison,
            prediction_summary=self.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        reply = await self._render_multi_tool_result_message(
            [
                ('dataset_training_readiness', readiness),
                ('recommend_next_training_step', recommendation),
            ],
            objective='数据集训练就绪与下一步建议说明',
        )
        if not reply:
            reply = (
                recommendation.get('summary')
                or readiness.get('summary')
                or recommendation.get('error')
                or readiness.get('error')
                or '数据集可训练性检查已完成'
            )
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
        reply = await self._render_tool_result_message('retrieve_training_knowledge', result)
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
            comparison=self.session_state.active_training.last_run_comparison,
            prediction_summary=self.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        reply = await self._render_multi_tool_result_message(
            [
                ('summarize_training_run', training_summary),
                ('analyze_training_outcome', result),
            ],
            objective='训练结果分析说明',
        )
        if not reply:
            reply = result.get('summary') or training_summary.get('summary') or result.get('error') or '训练结果分析已完成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if training_summary.get('ok', True) and result.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    async def _complete_specific_training_run_outcome_analysis_reply(self, run_id: str) -> dict[str, Any]:
        inspection = await self.direct_tool('inspect_training_run', run_id=run_id)
        result = await self.direct_tool(
            'analyze_training_outcome',
            metrics=inspection,
            data_quality=self.session_state.active_dataset.last_health_check or self.session_state.active_dataset.last_validate,
            comparison=self.session_state.active_training.last_run_comparison,
            prediction_summary=self.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        reply = await self._render_multi_tool_result_message(
            [
                ('inspect_training_run', inspection),
                ('analyze_training_outcome', result),
            ],
            objective='指定训练结果分析说明',
        )
        if not reply:
            reply = result.get('summary') or inspection.get('summary') or result.get('error') or '指定训练结果分析已完成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if inspection.get('ok', True) and result.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    async def _complete_best_training_outcome_analysis_reply(self) -> dict[str, Any]:
        selection = await self.direct_tool('select_best_training_run')
        best_run = selection.get('best_run') if selection.get('ok') else None
        result = await self.direct_tool(
            'analyze_training_outcome',
            metrics=best_run or self.session_state.active_training.training_run_summary or self.session_state.active_training.last_summary or self.session_state.active_training.last_status,
            data_quality=self.session_state.active_dataset.last_health_check or self.session_state.active_dataset.last_validate,
            prediction_summary=self.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        reply = await self._render_multi_tool_result_message(
            [
                ('select_best_training_run', selection),
                ('analyze_training_outcome', result),
            ],
            objective='最佳训练结果分析说明',
        )
        if not reply:
            reply = result.get('summary') or selection.get('summary') or result.get('error') or '最佳训练结果分析已完成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if selection.get('ok', True) and result.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    async def _complete_training_compare_analysis_reply(self, left_run_id: str = '', right_run_id: str = '') -> dict[str, Any]:
        comparison = await self.direct_tool('compare_training_runs', left_run_id=left_run_id, right_run_id=right_run_id)
        latest_run = comparison.get('left_run') if comparison.get('ok') else None
        result = await self.direct_tool(
            'analyze_training_outcome',
            metrics=latest_run or self.session_state.active_training.training_run_summary or self.session_state.active_training.last_summary or self.session_state.active_training.last_status,
            data_quality=self.session_state.active_dataset.last_health_check or self.session_state.active_dataset.last_validate,
            comparison=comparison if comparison.get('ok') else None,
            prediction_summary=self.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        reply = await self._render_multi_tool_result_message(
            [
                ('compare_training_runs', comparison),
                ('analyze_training_outcome', result),
            ],
            objective='训练对比分析说明',
        )
        if not reply:
            reply = result.get('summary') or comparison.get('summary') or result.get('error') or '训练对比分析已完成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if comparison.get('ok', True) and result.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    async def _complete_specific_training_run_next_step_reply(self, run_id: str) -> dict[str, Any]:
        inspection = await self.direct_tool('inspect_training_run', run_id=run_id)
        result = await self.direct_tool(
            'recommend_next_training_step',
            readiness=self.session_state.active_dataset.last_readiness,
            health=self.session_state.active_dataset.last_health_check,
            status=inspection,
            comparison=self.session_state.active_training.last_run_comparison,
            prediction_summary=self.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        reply = await self._render_multi_tool_result_message(
            [
                ('inspect_training_run', inspection),
                ('recommend_next_training_step', result),
            ],
            objective='指定训练下一步建议说明',
        )
        if not reply:
            reply = result.get('summary') or inspection.get('summary') or result.get('error') or '指定训练的下一步建议已生成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if inspection.get('ok', True) and result.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    async def _complete_best_training_next_step_reply(self) -> dict[str, Any]:
        selection = await self.direct_tool('select_best_training_run')
        best_run = selection.get('best_run') if selection.get('ok') else None
        result = await self.direct_tool(
            'recommend_next_training_step',
            readiness=self.session_state.active_dataset.last_readiness,
            health=self.session_state.active_dataset.last_health_check,
            status=best_run or self.session_state.active_training.training_run_summary or self.session_state.active_training.last_summary or self.session_state.active_training.last_status,
            prediction_summary=self.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        reply = await self._render_multi_tool_result_message(
            [
                ('select_best_training_run', selection),
                ('recommend_next_training_step', result),
            ],
            objective='最佳训练下一步建议说明',
        )
        if not reply:
            reply = result.get('summary') or selection.get('summary') or result.get('error') or '最佳训练的下一步建议已生成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if selection.get('ok', True) and result.get('ok', True) else 'error',
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
            comparison=self.session_state.active_training.last_run_comparison,
            prediction_summary=self.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        results: list[tuple[str, dict[str, Any]]] = []
        if readiness is not None:
            results.append(('training_readiness', readiness))
        results.append(('summarize_training_run', training_summary))
        results.append(('recommend_next_training_step', result))
        reply = await self._render_multi_tool_result_message(results, objective='下一步训练建议说明')
        if not reply:
            reply = result.get('summary') or training_summary.get('summary') or result.get('error') or '下一步建议已生成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if training_summary.get('ok', True) and result.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    async def _complete_training_compare_next_step_reply(self, left_run_id: str = '', right_run_id: str = '') -> dict[str, Any]:
        comparison = await self.direct_tool('compare_training_runs', left_run_id=left_run_id, right_run_id=right_run_id)
        latest_run = comparison.get('left_run') if comparison.get('ok') else None
        result = await self.direct_tool(
            'recommend_next_training_step',
            readiness=self.session_state.active_dataset.last_readiness,
            health=self.session_state.active_dataset.last_health_check,
            status=latest_run or self.session_state.active_training.training_run_summary or self.session_state.active_training.last_summary or self.session_state.active_training.last_status,
            comparison=comparison if comparison.get('ok') else None,
            prediction_summary=self.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        reply = await self._render_multi_tool_result_message(
            [
                ('compare_training_runs', comparison),
                ('recommend_next_training_step', result),
            ],
            objective='训练对比后的下一步建议说明',
        )
        if not reply:
            reply = result.get('summary') or comparison.get('summary') or result.get('error') or '训练对比后的下一步建议已生成'
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if comparison.get('ok', True) and result.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    def _complete_training_provenance_reply(self) -> dict[str, Any]:
        tr = self.session_state.active_training
        lines: list[str] = []
        comparison = tr.last_run_comparison or {}
        best_selection = tr.best_run_selection or {}
        inspected = tr.last_run_inspection or {}
        summary = tr.training_run_summary or tr.last_summary or tr.last_status or {}

        if comparison:
            left_run = comparison.get('left_run') or {}
            right_run = comparison.get('right_run') or {}
            left_id = str(left_run.get('run_id') or left_run.get('log_file') or '最近一次训练').strip()
            right_id = str(right_run.get('run_id') or right_run.get('log_file') or '上一次训练').strip()
            lines.append(f'我当前主要基于训练对比结果：{left_id} 对比 {right_id}。')
            if comparison.get('summary'):
                lines.append(f"- 对比摘要: {comparison.get('summary')}")
        elif best_selection:
            best_run = best_selection.get('best_run') or {}
            best_id = str(best_run.get('run_id') or best_run.get('log_file') or '最近最佳训练').strip()
            lines.append(f'我当前主要基于最值得参考的训练记录：{best_id}。')
            if best_selection.get('summary'):
                lines.append(f"- 选择依据: {best_selection.get('summary')}")
        elif inspected:
            selected_id = str(inspected.get('selected_run_id') or inspected.get('log_file') or '指定训练记录').strip()
            lines.append(f'我当前主要基于你刚查看的训练记录：{selected_id}。')
            if inspected.get('summary'):
                lines.append(f"- 记录摘要: {inspected.get('summary')}")
        elif summary:
            run_label = str(summary.get('run_id') or summary.get('log_file') or summary.get('summary') or '最近一次训练').strip()
            lines.append(f'我当前主要基于最近一次训练结果：{run_label}。')
        else:
            lines.append('我当前没有可追溯的训练依据；请先查看训练状态、训练详情、训练对比或最佳训练。')

        reply = '\n'.join(lines)
        self._messages.append(AIMessage(content=reply))
        return {'status': 'completed', 'message': reply, 'tool_call': None}

    def _complete_training_evidence_reply(self) -> dict[str, Any]:
        tr = self.session_state.active_training
        kn = self.session_state.active_knowledge
        lines = ['当前判断主要基于这些事实：']

        summary = tr.training_run_summary or tr.last_summary or tr.last_status or {}
        if summary.get('summary'):
            lines.append(f"- 训练事实: {summary.get('summary')}")
        if summary.get('signals'):
            lines.append(f"- 训练信号: {', '.join(str(item) for item in list(summary.get('signals') or [])[:4])}")

        comparison = tr.last_run_comparison or {}
        if comparison.get('summary'):
            lines.append(f"- 对比依据: {comparison.get('summary')}")
        if comparison.get('signals'):
            lines.append(f"- 对比信号: {', '.join(str(item) for item in list(comparison.get('signals') or [])[:4])}")

        analysis = kn.last_analysis or {}
        if analysis.get('summary'):
            lines.append(f"- 分析结论: {analysis.get('summary')}")
        if analysis.get('signals'):
            lines.append(f"- 分析信号: {', '.join(str(item) for item in list(analysis.get('signals') or [])[:4])}")

        recommendation = kn.last_recommendation or {}
        if recommendation.get('summary'):
            lines.append(f"- 建议依据: {recommendation.get('summary')}")
        if recommendation.get('recommended_action'):
            lines.append(f"- 当前建议动作: {recommendation.get('recommended_action')}")

        if len(lines) == 1:
            lines.append('- 当前没有足够的训练分析上下文；请先查看训练结果或重新分析。')

        reply = '\n'.join(lines)
        self._messages.append(AIMessage(content=reply))
        return {'status': 'completed', 'message': reply, 'tool_call': None}

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
        return intent_parsing.extract_dataset_path_from_text(text)

    @staticmethod
    def _extract_all_paths_from_text(text: str) -> list[str]:
        return intent_parsing.extract_all_paths_from_text(text)

    def _looks_like_prepare_only_request(self, text: str) -> bool:
        normalized = str(text or '').strip().lower()
        if not normalized:
            return False
        positive_tokens = (
            '准备训练数据',
            '准备数据',
            '准备数据集',
            '划分训练集',
            '划分数据集',
            '默认比例',
            '生成data.yaml',
            '生成 data.yaml',
            '生成yaml',
            '生成 yaml',
            '先做准备',
            '只做准备',
            '只准备',
        )
        if not any(token in text or token in normalized for token in positive_tokens):
            return False
        prepare_only_overrides = ('不要开始训练', '先不要开始训练', '不要训练', '先不要训练', '只做准备', '只准备')
        has_followup_training_intent = (
            any(token in text for token in ('开始训练', '启动训练', '开训', '直接训练', '训练计划', '训练草案', '模型来训练', '模型训练', '继续训练'))
            or bool(re.search(r'(?:然后|再|之后|接着|完成后|准备好后|整理好后)[^\n]{0,120}(?:训练|开训|启动)', text))
        )
        if has_followup_training_intent and not any(token in text for token in prepare_only_overrides):
            return False
        return bool(self._extract_dataset_path_from_text(text))

    async def _try_handle_prepare_only_intent(self, user_text: str, thread_id: str) -> dict[str, Any] | None:
        if not self._looks_like_prepare_only_request(user_text):
            return None
        dataset_path = str(self._extract_dataset_path_from_text(user_text) or '').strip()
        if not dataset_path:
            return None
        dataset_candidate = Path(dataset_path).expanduser()
        if dataset_candidate.is_absolute() and not dataset_candidate.exists():
            reply = f'我还没核实到这个路径存在：{dataset_path}。请先检查路径是否写对。'
            self._messages.append(AIMessage(content=reply))
            self._clear_training_plan_draft()
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        readiness = await self.direct_tool('dataset_training_readiness', img_dir=dataset_path)
        if not readiness.get('ok'):
            reply = await self._render_tool_result_message('dataset_training_readiness', readiness)
            if not reply:
                reply = readiness.get('error') or '数据准备前检查失败'
            self._messages.append(AIMessage(content=reply))
            return {'status': 'error', 'message': reply, 'tool_call': None}

        if readiness.get('ready') and str(readiness.get('resolved_data_yaml') or '').strip():
            data_yaml = str(readiness.get('resolved_data_yaml') or '').strip()
            reply = f'当前数据已经可训练，现成 data.yaml: {data_yaml}。如果你只是想准备数据，这一步已经完成。'
            self._messages.append(AIMessage(content=reply))
            self._clear_training_plan_draft()
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        resolved_img_dir = str(readiness.get('resolved_img_dir') or '').strip()
        resolved_label_dir = str(readiness.get('resolved_label_dir') or '').strip()
        if not resolved_img_dir or not resolved_label_dir:
            reply = (
                f'我还没核实到可用的数据集结构：{dataset_path}。'
                '当前没有确认到 images/ 和 labels/ 目录，请检查路径是否写对。'
            )
            self._messages.append(AIMessage(content=reply))
            self._clear_training_plan_draft()
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        prepare_args: dict[str, Any] = {'dataset_path': dataset_path}
        classes_txt = str(intent_parsing.extract_classes_txt_from_text(user_text) or '').strip()
        if classes_txt:
            prepare_args['classes_txt'] = classes_txt
        if any(token in user_text for token in ('按默认比例', '默认比例', '先划分', '划分训练集', '划分数据集', 'split')):
            prepare_args['force_split'] = True

        pending = {
            'name': 'prepare_dataset_for_training',
            'args': prepare_args,
            'id': None,
            'synthetic': True,
            'prepare_only': True,
        }
        self._set_pending_confirmation(thread_id, pending)
        reply = await self._build_confirmation_message(pending)
        self._messages.append(AIMessage(content=reply))
        draft = self._build_training_plan_draft(
            user_text=user_text,
            dataset_path=dataset_path,
            readiness=readiness,
            preflight={},
            next_tool_name='prepare_dataset_for_training',
            next_tool_args=dict(prepare_args),
            planned_training_args={},
        )
        draft['execution_mode'] = 'prepare_only'
        self._save_training_plan_draft(draft)
        return self._needs_confirmation_result(thread_id, pending, reply)

    @staticmethod
    def _extract_training_run_ids_from_text(text: str) -> list[str]:
        return re.findall(r'train_log_[A-Za-z0-9_-]+', text)

    @staticmethod
    def _looks_like_model_path(path: str) -> bool:
        return intent_parsing.looks_like_model_path(path)

    @staticmethod
    def _extract_remote_server_from_text(text: str) -> str:
        return intent_parsing.extract_remote_server_from_text(text)

    @staticmethod
    def _extract_remote_root_from_text(text: str) -> str:
        return intent_parsing.extract_remote_root_from_text(text)

    @staticmethod
    def _explicitly_references_previous_context(text: str) -> bool:
        normalized = str(text or '').strip().lower()
        if not normalized:
            return False
        tokens = (
            '刚才',
            '刚刚',
            '上次',
            '上一个',
            '上一次',
            '之前',
            '前一个',
            '最近',
            '继续',
            '恢复',
            '沿用',
            '复用',
            'reuse',
            'previous',
            'last',
        )
        return any(token in normalized or token in text for token in tokens)

    def _build_remote_upload_args(self, user_text: str) -> dict[str, Any]:
        quoted_paths = [
            item.rstrip('。，“”,,;；')
            for item in re.findall(r'[\"\']((?:[A-Za-z]:\\|/)[^\"\']+)[\"\']', user_text)
        ]
        raw_paths = []
        seen_paths: set[str] = set()
        for item in [*quoted_paths, *self._extract_all_paths_from_text(user_text)]:
            if item and item not in seen_paths:
                seen_paths.add(item)
                raw_paths.append(item)
        remote_root = self._extract_remote_root_from_text(user_text)
        server = self._extract_remote_server_from_text(user_text)
        allow_reuse = self._explicitly_references_previous_context(user_text)

        local_paths: list[str] = []
        remote_candidates: list[str] = []
        seen_local: set[str] = set()
        for item in raw_paths:
            value = str(item or '').strip()
            if not value:
                continue
            if remote_root and value == remote_root:
                continue
            path_obj = Path(value).expanduser()
            if path_obj.exists() or re.match(r'^[A-Za-z]:\\', value):
                normalized = str(path_obj)
                if normalized not in seen_local:
                    seen_local.add(normalized)
                    local_paths.append(normalized)
            elif value.startswith('/'):
                remote_candidates.append(value)

        if not remote_root and remote_candidates:
            remote_root = remote_candidates[-1]

        if not local_paths and allow_reuse:
            lower_text = user_text.lower()
            if any(token in user_text for token in ('权重', '模型', 'pt 文件')) or 'weight' in lower_text or 'model' in lower_text:
                model_path = str(self.session_state.active_training.model or self.session_state.active_prediction.model or '').strip()
                if model_path:
                    path_obj = Path(model_path).expanduser()
                    if path_obj.exists():
                        local_paths.append(str(path_obj))
            if any(token in user_text for token in ('数据', '数据集', 'dataset')) or 'dataset' in lower_text:
                dataset_path = str(self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir or '').strip()
                if dataset_path:
                    path_obj = Path(dataset_path).expanduser()
                    if path_obj.exists() and str(path_obj) not in local_paths:
                        local_paths.append(str(path_obj))
            if any(token in user_text for token in ('视频', '录像', 'video')) and not any(token in user_text for token in ('预测结果', 'prediction 输出')):
                source_path = str(self.session_state.active_prediction.source_path or '').strip()
                if source_path:
                    path_obj = Path(source_path).expanduser()
                    if path_obj.exists() and str(path_obj) not in local_paths:
                        local_paths.append(str(path_obj))

        args: dict[str, Any] = {'local_paths': local_paths}
        remembered_server = str(self.session_state.active_remote_transfer.profile_name or self.session_state.active_remote_transfer.target_label or '').strip()
        remembered_root = str(self.session_state.active_remote_transfer.remote_root or '').strip()
        if server:
            args['server'] = server
        elif allow_reuse and remembered_server:
            args['server'] = remembered_server
        if remote_root:
            args['remote_root'] = remote_root
        elif allow_reuse and remembered_root:
            args['remote_root'] = remembered_root
        return args

    @staticmethod
    def _remote_join(root: str, *parts: str) -> str:
        path = PurePosixPath(str(root or '/'))
        for part in parts:
            text = str(part or '')
            if not text:
                continue
            for chunk in text.replace('\\', '/').split('/'):
                if chunk and chunk not in {'.'}:
                    path /= chunk
        return path.as_posix()

    @staticmethod
    def _path_looks_like_video(path: str) -> bool:
        suffix = Path(path).suffix.lower()
        return suffix in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.ts', '.webm'}

    @staticmethod
    def _path_looks_like_image(path: str) -> bool:
        suffix = Path(path).suffix.lower()
        return suffix in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

    def _directory_kind(self, path: str) -> str:
        path_obj = Path(path)
        if not path_obj.is_dir():
            return ''
        limit = 32
        seen = 0
        has_video = False
        has_image = False
        for child in path_obj.rglob('*'):
            if not child.is_file():
                continue
            seen += 1
            child_path = str(child)
            if self._path_looks_like_video(child_path):
                has_video = True
            if self._path_looks_like_image(child_path):
                has_image = True
            if seen >= limit or (has_video and has_image):
                break
        if has_video and not has_image:
            return 'video'
        if has_image and not has_video:
            return 'image'
        if has_video:
            return 'video'
        if has_image:
            return 'image'
        return ''

    def _next_local_roundtrip_root(self, kind: str) -> str:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        root = Path(__file__).resolve().parents[2] / 'output' / kind / stamp
        root.mkdir(parents=True, exist_ok=True)
        return str(root)

    def _apply_remote_defaults(self, args: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(args or {})
        allow_reuse = bool(resolved.pop('_allow_previous_context', False))
        if allow_reuse and not str(resolved.get('server') or '').strip() and not str(resolved.get('host') or '').strip():
            remembered_server = str(
                self.session_state.active_remote_transfer.profile_name
                or self.session_state.active_remote_transfer.target_label
                or ''
            ).strip()
            if remembered_server:
                resolved['server'] = remembered_server
        if allow_reuse and not str(resolved.get('remote_root') or '').strip():
            remembered_root = str(self.session_state.active_remote_transfer.remote_root or '').strip()
            if remembered_root:
                resolved['remote_root'] = remembered_root
        return resolved

    def _build_remote_prediction_pipeline_args(self, user_text: str) -> dict[str, Any]:
        upload_args = self._build_remote_upload_args(user_text)
        upload_args['_allow_previous_context'] = self._explicitly_references_previous_context(user_text)
        upload_args = self._apply_remote_defaults(upload_args)
        return {
            'user_text': user_text,
            'upload_args': upload_args,
            'download_after_predict': True,
            'local_result_root': self._next_local_roundtrip_root('remote_prediction_roundtrip'),
        }

    def _build_remote_training_pipeline_args(self, user_text: str) -> dict[str, Any]:
        upload_args = self._build_remote_upload_args(user_text)
        upload_args['_allow_previous_context'] = self._explicitly_references_previous_context(user_text)
        upload_args = self._apply_remote_defaults(upload_args)
        force_split = any(token in user_text for token in ('按默认比例划分', '先划分再训练', 'split 后训练', 'split后训练'))
        normalized = user_text.lower()
        wait_for_completion = any(
            token in user_text or token in normalized
            for token in (
                '等训练结束',
                '等训练完成',
                '训练完后',
                '训练完成后',
                '训练结束后',
                '跑完后',
                'wait until training',
                'wait for training',
            )
        )
        download_after_completion = any(
            token in user_text or token in normalized
            for token in (
                '回传',
                '拉回本机',
                '下载回来',
                '下载回本机',
                '训练产物拉回',
                '结果拉回',
                'download back',
                'bring back',
                'pull back',
            )
        )
        if download_after_completion:
            wait_for_completion = True
        return {
            'user_text': user_text,
            'upload_args': upload_args,
            'force_split': force_split,
            'wait_for_completion': wait_for_completion,
            'download_after_completion': download_after_completion,
            'local_result_root': self._next_local_roundtrip_root('remote_training_roundtrip') if download_after_completion else '',
            'poll_interval_seconds': 15,
            'max_wait_seconds': 7200,
        }

    def _resolve_prediction_remote_inputs(self, upload_result: dict[str, Any]) -> dict[str, Any]:
        uploaded_items = list(upload_result.get('uploaded_items') or [])
        if not uploaded_items:
            return {'ok': False, 'error': '远端上传完成了，但没有可用于预测的远端输入项。'}

        model_items = [item for item in uploaded_items if self._looks_like_model_path(str(item.get('local_path') or item.get('remote_path') or ''))]
        source_items = [item for item in uploaded_items if item not in model_items]
        if not model_items:
            return {'ok': False, 'error': '当前远端预测闭环缺少模型文件；请至少上传一个 .pt / .onnx 模型。'}
        if not source_items:
            return {'ok': False, 'error': '当前远端预测闭环缺少图片或视频输入。'}

        source_kinds: set[str] = set()
        for item in source_items:
            local_path = str(item.get('local_path') or '')
            item_type = str(item.get('item_type') or '')
            if item_type == 'directory':
                kind = self._directory_kind(local_path)
            elif self._path_looks_like_video(local_path):
                kind = 'video'
            elif self._path_looks_like_image(local_path):
                kind = 'image'
            else:
                kind = ''
            if kind:
                source_kinds.add(kind)
        if not source_kinds:
            return {'ok': False, 'error': '当前还无法判断上传内容是图片还是视频；请显式上传图片/图片目录或视频/视频目录。'}
        if len(source_kinds) > 1:
            return {'ok': False, 'error': '当前远端预测闭环不支持把图片和视频混在一次请求里；请拆成独立步骤执行。'}
        if len(source_items) > 1:
            return {
                'ok': False,
                'error': '当前远端预测闭环要求待预测输入是单个文件或单个目录；如果有多个图片/视频，请先整理进一个目录再上传。',
            }

        source_kind = next(iter(source_kinds))
        source_path = str(source_items[0].get('remote_path') or '')
        tool_name = 'predict_videos' if source_kind == 'video' else 'predict_images'
        return {
            'ok': True,
            'tool_name': tool_name,
            'model_path': str(model_items[-1].get('remote_path') or ''),
            'source_path': source_path,
            'source_kind': source_kind,
        }

    def _resolve_training_remote_inputs(self, upload_result: dict[str, Any]) -> dict[str, Any]:
        uploaded_items = list(upload_result.get('uploaded_items') or [])
        if not uploaded_items:
            return {'ok': False, 'error': '远端上传完成了，但没有可用于训练的远端输入项。'}
        model_items = [item for item in uploaded_items if self._looks_like_model_path(str(item.get('local_path') or item.get('remote_path') or ''))]
        dataset_items = [
            item for item in uploaded_items
            if item not in model_items and str(item.get('item_type') or '') == 'directory'
        ]
        if not model_items:
            return {'ok': False, 'error': '当前远端训练闭环缺少模型文件；请至少上传一个 .pt / .onnx 模型。'}
        if not dataset_items:
            return {'ok': False, 'error': '当前远端训练闭环缺少数据集目录；请上传一个数据集根目录。'}
        return {
            'ok': True,
            'model_path': str(model_items[-1].get('remote_path') or ''),
            'dataset_path': str(dataset_items[-1].get('remote_path') or ''),
        }


    def _pending_allowed_decisions(self, tool_name: str) -> list[str]:
        normalized = canonical_tool_name(str(tool_name or '').strip())
        if self._tool_is_read_only(normalized):
            return ['approve', 'reject', 'clarify']
        if self._tool_requires_confirmation(normalized):
            return ['approve', 'reject', 'edit', 'clarify']
        return ['approve', 'reject', 'clarify']

    def _pending_action_objective(self, tool_name: str, args: dict[str, Any]) -> str:
        tool_name = str(tool_name or '').strip()
        dataset_path = str(args.get('dataset_path') or self.session_state.active_dataset.dataset_root or '').strip()
        data_yaml = str(args.get('data_yaml') or self.session_state.active_training.data_yaml or self.session_state.active_dataset.data_yaml or '').strip()
        model = str(args.get('model') or self.session_state.active_training.model or '').strip()
        if tool_name == 'prepare_dataset_for_training':
            return f'把数据集准备到可训练状态{f"（{dataset_path}）" if dataset_path else ""}'
        if tool_name == 'start_training':
            parts = [part for part in [model, data_yaml] if part]
            return '启动训练' + (f"（{' / '.join(parts)}）" if parts else '')
        if tool_name == 'start_training_loop':
            parts = [part for part in [model, data_yaml] if part]
            return '启动循环训练' + (f"（{' / '.join(parts)}）" if parts else '')
        if tool_name == 'upload_assets_to_remote':
            return '把本地资源上传到远端服务器'
        if tool_name == 'remote_training_pipeline':
            return '执行远端训练闭环'
        if tool_name == 'remote_prediction_pipeline':
            return '执行远端预测闭环'
        return f'执行 {tool_name}'

    def _pending_action_summary(self, tool_name: str, args: dict[str, Any]) -> str:
        tool_name = str(tool_name or '').strip()
        dataset_path = str(args.get('dataset_path') or self.session_state.active_dataset.dataset_root or '').strip()
        data_yaml = str(args.get('data_yaml') or self.session_state.active_training.data_yaml or self.session_state.active_dataset.data_yaml or '').strip()
        model = str(args.get('model') or self.session_state.active_training.model or '').strip()
        if tool_name == 'prepare_dataset_for_training':
            details = [item for item in [dataset_path or None, 'force_split=true' if args.get('force_split') else None] if item]
            return '准备数据集' + (f"：{'，'.join(details)}" if details else '')
        if tool_name == 'start_training':
            details = [item for item in [f'model={model}' if model else None, f'data={data_yaml}' if data_yaml else None, f"epochs={args.get('epochs')}" if args.get('epochs') is not None else None] if item]
            return '启动训练' + (f"：{', '.join(details)}" if details else '')
        if tool_name == 'start_training_loop':
            details = [item for item in [f'model={model}' if model else None, f'data={data_yaml}' if data_yaml else None, f"max_rounds={args.get('max_rounds')}" if args.get('max_rounds') is not None else None] if item]
            return '启动循环训练' + (f"：{', '.join(details)}" if details else '')
        if tool_name == 'upload_assets_to_remote':
            return '上传资源到远端服务器'
        if tool_name == 'remote_training_pipeline':
            return '执行远端训练闭环'
        if tool_name == 'remote_prediction_pipeline':
            return '执行远端预测闭环'
        return f'待确认动作：{tool_name}'

    def _pending_review_config(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        normalized = canonical_tool_name(tool_name)
        metadata = self._tool_surface_metadata(normalized)
        annotations = self._tool_surface_annotations(normalized)
        read_only = self._tool_is_read_only(normalized)
        destructive = self._tool_is_destructive(normalized)
        return {
            'risk_level': self._tool_risk_level(normalized),
            'allow_edit': not read_only,
            'allow_clarify': True,
            'tool_name': normalized,
            'confirmation_required': self._tool_requires_confirmation(normalized),
            'read_only': read_only,
            'destructive': destructive,
            'open_world': bool(
                metadata.get('open_world')
                if 'open_world' in metadata
                else annotations.get('openWorldHint')
            ),
        }

    def _build_pending_action_payload(
        self,
        pending: dict[str, Any],
        *,
        thread_id: str | None = None,
        decision_state: str = 'pending',
    ) -> dict[str, Any]:
        tool_name = str(pending.get('name') or '').strip()
        args = dict(pending.get('args') or {})
        return {
            'interrupt_kind': 'tool_approval',
            'decision_state': decision_state,
            'thread_id': str(thread_id or pending.get('thread_id') or self.session_state.pending_confirmation.thread_id or '').strip(),
            'tool_name': tool_name,
            'tool_args': args,
            'summary': str(pending.get('summary') or self._pending_action_summary(tool_name, args)).strip(),
            'objective': str(pending.get('objective') or self._pending_action_objective(tool_name, args)).strip(),
            'allowed_decisions': list(pending.get('allowed_decisions') or self._pending_allowed_decisions(tool_name)),
            'review_config': dict(pending.get('review_config') or self._pending_review_config(tool_name, args)),
            'decision_context': dict(pending.get('decision_context') or self.session_state.pending_confirmation.decision_context or {}),
        }

    def _needs_confirmation_result(self, thread_id: str, pending: dict[str, Any], message: str) -> dict[str, Any]:
        return {
            'status': 'needs_confirmation',
            'message': message,
            'tool_call': {'name': pending['name'], 'args': pending.get('args', {})},
            'thread_id': thread_id,
            'pending_action': self._build_pending_action_payload(pending, thread_id=thread_id),
        }

    def _cancelled_result(self, pending: dict[str, Any], message: str) -> dict[str, Any]:
        payload = self._build_pending_action_payload(pending, decision_state='rejected')
        return {
            'status': 'cancelled',
            'message': message,
            'tool_call': {'name': pending['name'], 'args': pending.get('args', {})},
            'pending_action': payload,
        }

    def _set_pending_confirmation(self, thread_id: str, pending: dict[str, Any]) -> None:
        payload = self._build_pending_action_payload(pending, thread_id=thread_id)
        pc = self.session_state.pending_confirmation
        pc.thread_id = thread_id
        pc.tool_name = payload['tool_name']
        pc.tool_args = payload['tool_args']
        pc.interrupt_kind = payload['interrupt_kind']
        pc.objective = payload['objective']
        pc.summary = payload['summary']
        pc.allowed_decisions = list(payload['allowed_decisions'])
        pc.review_config = dict(payload['review_config'])
        pc.decision_context = dict(payload.get('decision_context') or {})
        pc.created_at = utc_now()
        self.memory.append_event(
            self.session_state.session_id,
            'confirmation_requested',
            {
                'tool': pc.tool_name,
                'args': pc.tool_args,
                'thread_id': thread_id,
                'summary': pc.summary,
                'objective': pc.objective,
                'allowed_decisions': pc.allowed_decisions,
            },
        )

    def _clear_pending_confirmation(self) -> None:
        pc = self.session_state.pending_confirmation
        pc.thread_id = ""
        pc.tool_name = ""
        pc.tool_args = {}
        pc.interrupt_kind = 'tool_approval'
        pc.objective = ''
        pc.summary = ''
        pc.allowed_decisions = ['approve', 'reject', 'edit', 'clarify']
        pc.review_config = {}
        pc.decision_context = {}
        pc.created_at = ""

    def _pending_from_state(self) -> dict[str, Any] | None:
        pc = self.session_state.pending_confirmation
        if not pc.tool_name:
            return None
        return {
            'name': pc.tool_name,
            'args': dict(pc.tool_args),
            'id': None,
            'summary': pc.summary,
            'objective': pc.objective,
            'allowed_decisions': list(pc.allowed_decisions),
            'review_config': dict(pc.review_config),
            'decision_context': dict(pc.decision_context),
            'thread_id': pc.thread_id,
        }

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

    async def _emit_stream_event(self, stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None, event: dict[str, Any]) -> None:
        if stream_handler is None:
            return
        maybe = stream_handler(event)
        if inspect.isawaitable(maybe):
            await maybe

    @staticmethod
    def _extract_stream_text(chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        text_value = getattr(chunk, 'text', None)
        if isinstance(text_value, str) and text_value:
            return text_value
        content = getattr(chunk, 'content', None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_part = item.get('text')
                    if isinstance(text_part, str):
                        parts.append(text_part)
            return ''.join(parts)
        return ''

    async def _handle_stream_mode_event(self, mode: str, data: Any, stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None) -> None:
        if stream_handler is None:
            return
        if mode == 'messages' and isinstance(data, tuple) and len(data) == 2:
            chunk, metadata = data
            node_name = str((metadata or {}).get('langgraph_node') or (metadata or {}).get('node') or '').strip().lower()
            if node_name == 'tools' or isinstance(chunk, ToolMessage) or getattr(chunk, 'tool_call_id', None):
                return
            text = self._extract_stream_text(chunk)
            if text:
                await self._emit_stream_event(stream_handler, {'type': 'token', 'text': text, 'metadata': metadata})
            return
        if mode != 'updates' or not isinstance(data, dict):
            return
        for node_name, payload in data.items():
            if not isinstance(payload, dict):
                continue
            messages = payload.get('messages')
            if not isinstance(messages, list):
                continue
            for message in messages:
                tool_calls = getattr(message, 'tool_calls', None) or []
                for tool_call in tool_calls:
                    tool_name = canonical_tool_name(str(tool_call.get('name') or '').strip())
                    if tool_name:
                        await self._emit_stream_event(stream_handler, {'type': 'tool_call', 'tool_name': tool_name, 'node': node_name})

    async def _graph_invoke(self, payload: Any, config: dict[str, Any], stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None) -> dict[str, Any]:
        if stream_handler is None:
            return await self.graph.ainvoke(payload, config=config)
        async for item in self.graph.astream(
            payload,
            config=config,
            stream_mode=['messages', 'updates'],
            version='v1',
        ):
            if isinstance(item, tuple) and len(item) == 2:
                mode, data = item
                await self._handle_stream_mode_event(str(mode), data, stream_handler)
        state = self.graph.get_state(config)
        values = getattr(state, 'values', {}) if state else {}
        messages = list(values.get('messages') or [])
        return {'messages': messages}

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

    def _compose_final_reply(self, messages: list[BaseMessage], applied_results: list[tuple[str, dict[str, Any]]]) -> str:
        final_message = self._extract_final_ai(messages)
        final_text = final_message.content.strip() if final_message and isinstance(final_message.content, str) else ""
        if final_text:
            return final_text
        if applied_results:
            tool_names = [name for name, _ in applied_results if name]
            tool_hint = f"（已执行工具: {'、'.join(tool_names[:3])}）" if tool_names else ""
            return f"模型这次没有生成最终回复{tool_hint}。我没有再替模型拼凑答案；如果你愿意，我可以继续重试，或者你也可以直接查看这次工具执行结果。"
        return self._build_empty_reply_fallback(messages)

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

    async def _continue_followup_training_after_prepare(
        self,
        *,
        thread_id: str,
        synthetic_followup: dict[str, Any],
        prepare_parsed: dict[str, Any],
    ) -> dict[str, Any]:
        followup_args = dict(synthetic_followup.get('args') or {})
        preflight = await self.direct_tool(
            'training_preflight',
            model=followup_args.get('model', ''),
            data_yaml=followup_args.get('data_yaml', ''),
            epochs=int(followup_args.get('epochs', 100)),
            device=str(followup_args.get('device', 'auto') or 'auto'),
            training_environment=str(followup_args.get('training_environment', '') or ''),
            project=str(followup_args.get('project', '') or ''),
            name=str(followup_args.get('name', '') or ''),
            batch=followup_args.get('batch'),
            imgsz=followup_args.get('imgsz'),
            fraction=followup_args.get('fraction'),
            classes=followup_args.get('classes'),
            single_cls=followup_args.get('single_cls'),
            optimizer=str(followup_args.get('optimizer', '') or ''),
            freeze=followup_args.get('freeze'),
            resume=followup_args.get('resume'),
            lr0=followup_args.get('lr0'),
            patience=followup_args.get('patience'),
            workers=followup_args.get('workers'),
            amp=followup_args.get('amp'),
        )
        draft = self._build_training_plan_draft(
            user_text=self._recent_user_text(),
            dataset_path=self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir,
            readiness=self.session_state.active_dataset.last_readiness,
            preflight=preflight,
            next_tool_name='start_training' if preflight.get('ready_to_start') else '',
            next_tool_args=followup_args if preflight.get('ready_to_start') else {},
            planned_training_args=followup_args,
        )
        self._save_training_plan_draft(draft)
        if not preflight.get('ready_to_start'):
            reply = await self._render_prepare_followup_message(prepare_parsed, preflight)
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
        return self._needs_confirmation_result(
            thread_id,
            synthetic_followup,
            await self._build_confirmation_message(synthetic_followup),
        )

    @staticmethod
    def _find_applied_tool_result(
        applied_results: list[tuple[str, dict[str, Any]]],
        tool_name: str,
    ) -> dict[str, Any] | None:
        canonical_name = canonical_tool_name(tool_name)
        for applied_tool_name, parsed in reversed(applied_results):
            if canonical_tool_name(applied_tool_name) == canonical_name:
                return dict(parsed or {})
        return None

    async def _handle_post_prepare_confirmation_followup(
        self,
        *,
        thread_id: str,
        prepare_parsed: dict[str, Any],
    ) -> dict[str, Any] | None:
        loop_followup_result = await self._continue_training_loop_start_after_prepare(
            thread_id=thread_id,
            prepare_result=prepare_parsed,
        )
        if loop_followup_result is not None:
            return loop_followup_result
        loop_followup = self._build_followup_training_loop_request()
        if loop_followup:
            self._set_pending_confirmation(thread_id, loop_followup)
            self.memory.save_state(self.session_state)
            return self._needs_confirmation_result(
                thread_id,
                loop_followup,
                await self._build_confirmation_message(loop_followup),
            )
        if not prepare_parsed or prepare_parsed.get('ok') is False:
            return None
        synthetic_followup = None
        if not prepare_parsed.get('prepare_only'):
            synthetic_followup = self._build_followup_training_request()
        if synthetic_followup:
            return await self._continue_followup_training_after_prepare(
                thread_id=thread_id,
                synthetic_followup=synthetic_followup,
                prepare_parsed=prepare_parsed,
            )
        return None

    def _build_followup_training_request(self) -> dict[str, Any] | None:
        draft = self.session_state.active_training.training_plan_draft or {}
        if str(draft.get('execution_mode') or '').strip().lower() == 'prepare_only':
            return None
        planned_args = dict(draft.get('planned_training_args') or {})
        latest_readiness = self.session_state.active_dataset.last_readiness or {}
        if draft and planned_args.get('model'):
            prepared_yaml = str(planned_args.get('data_yaml') or self.session_state.active_dataset.data_yaml or '').strip()
            if prepared_yaml:
                planned_args['data_yaml'] = prepared_yaml
                draft['planned_training_args'] = dict(planned_args)
                latest_summary = str(latest_readiness.get('summary') or '').strip()
                if latest_summary:
                    draft['data_summary'] = latest_summary
                draft['reasoning_summary'] = '数据已经准备到可训练状态；下一步直接启动训练。'
                draft['blockers'] = [str(item).strip() for item in (latest_readiness.get('blockers') or []) if str(item).strip()]
                draft['warnings'] = [str(item).strip() for item in (latest_readiness.get('warnings') or []) if str(item).strip()]
                draft['next_step_tool'] = 'start_training'
                draft['next_step_args'] = dict(planned_args)
                self._save_training_plan_draft(draft)
                self.memory.save_state(self.session_state)
                return {
                    "id": None,
                    "name": "start_training",
                    "args": planned_args,
                    "synthetic": True,
                }
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
        args = self._collect_requested_training_args(user_text, data_yaml=data_yaml)
        args.pop('classes_txt', None)
        if not str(args.get('model') or '').strip():
            return None
        return {"id": None, "name": "start_training", "args": args, "synthetic": True}

    def _build_followup_training_loop_request(self) -> dict[str, Any] | None:
        draft = self.session_state.active_training.training_plan_draft or {}
        if str(draft.get('execution_mode') or '').strip().lower() != 'prepare_then_loop':
            return None
        planned_args = dict(draft.get('planned_loop_args') or draft.get('planned_training_args') or {})
        latest_readiness = self.session_state.active_dataset.last_readiness or {}
        prepared_yaml = str(
            planned_args.get('data_yaml')
            or self.session_state.active_dataset.data_yaml
            or self.session_state.active_training.data_yaml
            or ''
        ).strip()
        if not prepared_yaml:
            return None
        model = str(planned_args.get('model') or self.session_state.active_training.model or '').strip()
        if not model:
            return None
        planned_args['model'] = model
        planned_args['data_yaml'] = prepared_yaml
        planned_args.pop('classes_txt', None)
        if not str(planned_args.get('managed_level') or '').strip():
            planned_args['managed_level'] = 'conservative_auto'
        max_rounds = planned_args.get('max_rounds')
        if max_rounds is None or str(max_rounds).strip() == '':
            planned_args['max_rounds'] = 5
        draft['planned_loop_args'] = dict(planned_args)
        latest_summary = str(latest_readiness.get('summary') or '').strip()
        if latest_summary:
            draft['data_summary'] = latest_summary
        draft['reasoning_summary'] = '数据已经准备到可训练状态；下一步进入循环训练。'
        draft['blockers'] = [str(item).strip() for item in (latest_readiness.get('blockers') or []) if str(item).strip()]
        draft['warnings'] = [str(item).strip() for item in (latest_readiness.get('warnings') or []) if str(item).strip()]
        draft['next_step_tool'] = 'start_training_loop'
        draft['next_step_args'] = dict(planned_args)
        self._save_training_plan_draft(draft)
        self.memory.save_state(self.session_state)
        return {
            "id": None,
            "name": "start_training_loop",
            "args": planned_args,
            "synthetic": True,
        }

    @staticmethod
    def _looks_like_video_path(path: str) -> bool:
        return intent_parsing.looks_like_video_path(path)

    def _should_use_video_prediction(self, user_text: str, path: str) -> bool:
        normalized = str(user_text or '').lower()
        if intent_parsing.looks_like_video_path(path):
            return True
        return any(token in user_text for token in ('视频', '录像')) or 'video' in normalized

    def _build_realtime_session_kwargs(self, user_text: str) -> dict[str, Any]:
        session_id = intent_parsing.extract_realtime_session_id_from_text(user_text) or self.session_state.active_prediction.realtime_session_id
        return {'session_id': session_id} if session_id else {}

    def _build_realtime_prediction_args(self, user_text: str, *, source_type: str) -> dict[str, Any]:
        args: dict[str, Any] = {}
        model = intent_parsing.extract_model_from_text(user_text) or self.session_state.active_prediction.model or self.session_state.active_training.model
        if model:
            args['model'] = model
        frame_interval_ms = intent_parsing.extract_frame_interval_ms_from_text(user_text)
        if frame_interval_ms is not None:
            args['frame_interval_ms'] = frame_interval_ms
        max_frames = intent_parsing.extract_max_frames_from_text(user_text)
        if max_frames is not None:
            args['max_frames'] = max_frames
        if source_type == 'camera':
            camera_id = intent_parsing.extract_camera_id_from_text(user_text)
            if camera_id is not None:
                args['camera_id'] = camera_id
        elif source_type == 'rtsp':
            rtsp_url = intent_parsing.extract_rtsp_url_from_text(user_text)
            if rtsp_url:
                args['rtsp_url'] = rtsp_url
        elif source_type == 'screen':
            screen_id = intent_parsing.extract_screen_id_from_text(user_text)
            if screen_id is not None:
                args['screen_id'] = screen_id
        source_hint = str(args.get('rtsp_url') or '')
        output_dir = intent_parsing.extract_output_path_from_text(user_text, source_hint)
        if output_dir:
            args['output_dir'] = output_dir
        return args

    def _prediction_followup_kwargs(
        self,
        user_text: str,
        fallback_path: str = '',
        *,
        allow_context_fallback: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        explicit_target = self._extract_dataset_path_from_text(user_text)
        allow_reuse = self._explicitly_references_previous_context(user_text)
        should_reuse_context = allow_reuse or allow_context_fallback
        if explicit_target:
            if explicit_target.lower().endswith('.json'):
                kwargs['report_path'] = explicit_target
            else:
                kwargs['output_dir'] = explicit_target
        elif should_reuse_context and self.session_state.active_prediction.report_path:
            kwargs['report_path'] = self.session_state.active_prediction.report_path
        elif should_reuse_context and self.session_state.active_prediction.output_dir:
            kwargs['output_dir'] = self.session_state.active_prediction.output_dir
        elif should_reuse_context and fallback_path:
            kwargs['output_dir'] = fallback_path
        return kwargs

    @staticmethod
    def _extract_metric_signals_from_text(text: str) -> list[str]:
        normalized = text.lower()
        signals: list[str] = []
        if ((('precision' in normalized) or ('精确率' in text)) and ((('recall' in normalized) or ('召回' in text)))):
            if re.search(r'(precision|精确率).{0,8}(高|偏高).{0,12}(recall|召回).{0,8}(低|偏低)', text, flags=re.I):
                signals.append('high_precision_low_recall')
            if re.search(r'(precision|精确率).{0,8}(低|偏低).{0,12}(recall|召回).{0,8}(高|偏高)', text, flags=re.I):
                signals.append('low_precision_high_recall')
        if re.search(r'(map50|mAP50|mAP).{0,8}(低|偏低)', text, flags=re.I) or 'map低' in normalized:
            signals.append('low_map_overall')
        if '只有loss' in normalized or '只看loss' in normalized or '只有 loss' in text:
            signals.append('loss_only_metrics')
        return signals

    @staticmethod
    def _extract_training_execution_backend_from_text(text: str) -> str:
        lowered = text.lower()
        if any(token in text for token in ('不用自定义脚本', '不用脚本了', '切回标准 yolo', '改成标准 yolo', '用标准 yolo')) or any(token in lowered for token in ("don't use custom script", 'switch back to standard yolo')):
            return 'standard_yolo'
        if any(token in text for token in ('不用 trainer', '不用自定义trainer', '不用自定义训练器', '切回标准训练器')) or any(token in lowered for token in ('switch back to standard trainer',)):
            return 'standard_yolo'
        script_path = intent_parsing.extract_custom_training_script_from_text(text)
        if script_path or any(token in text for token in ('自定义训练脚本', 'python脚本训练', '脚本训练')):
            return 'custom_script'
        trainer_explicit = any(token in text for token in ('自定义 trainer', '自定义trainer', '自定义训练器'))
        trainer_context = any(token in text for token in ('trainer 讨论', 'trainer方案', 'trainer 先讨论', 'trainer 先不管'))
        trainer_switch = re.search(r'(?:改成|切到|换成|用|讨论)\s*(?:自定义\s*)?trainer', lowered) is not None
        if trainer_explicit or trainer_context or trainer_switch:
            return 'custom_trainer'
        return 'standard_yolo'

    @staticmethod
    def _is_training_discussion_only(text: str) -> bool:
        lowered = str(text or '').lower()
        return any(
            token in text or token in lowered
            for token in (
                '先别执行',
                '先不要执行',
                '先别启动',
                '先不要启动',
                '先看计划',
                '先看看计划',
                '先给我计划',
                '先讨论',
                '只讨论',
                '先别急着执行',
                '先做方案',
                '先 dry-run',
                '先 preflight',
            )
        )

    @staticmethod
    def _wants_default_training_environment(text: str) -> bool:
        lowered = str(text or '').lower()
        return any(
            token in text or token in lowered
            for token in (
                '恢复默认环境',
                '用默认环境',
                '切回默认环境',
                '环境恢复默认',
                '不要指定环境',
            )
        )

    @staticmethod
    def _wants_clear_project(text: str) -> bool:
        lowered = str(text or '').lower()
        return any(
            token in text or token in lowered
            for token in (
                'project 不要了',
                '不要 project',
                '清空 project',
                '恢复默认输出目录',
                '输出目录用默认',
                '不要输出目录',
            )
        )

    @staticmethod
    def _wants_clear_run_name(text: str) -> bool:
        lowered = str(text or '').lower()
        return any(
            token in text or token in lowered
            for token in (
                'name 不要了',
                '不要 name',
                '清空 name',
                '运行名不要了',
                '实验名不要了',
                '不要输出名',
            )
        )

    @staticmethod
    def _wants_clear_batch(text: str) -> bool:
        lowered = str(text or '').lower()
        return any(
            token in text or token in lowered
            for token in (
                '恢复默认 batch',
                'batch 恢复默认',
                'batch 不要了',
                '不要 batch',
                'batch 清掉',
                'batch 取消',
                'batch 先取消',
            )
        )

    @staticmethod
    def _wants_clear_fraction(text: str) -> bool:
        lowered = str(text or '').lower()
        return any(
            token in text or token in lowered
            for token in (
                '恢复全量数据',
                '恢复全部数据',
                '全部数据都训练',
                '取消抽样',
                '不做抽样',
                '取消 fraction',
                'fraction 取消',
                'fraction 不要了',
                '不要 fraction',
                '不限制数据比例',
            )
        )

    @staticmethod
    def _wants_clear_classes(text: str) -> bool:
        lowered = str(text or '').lower()
        return any(
            token in text or token in lowered
            for token in (
                '取消类别限制',
                '类别限制取消',
                '类别限制先取消',
                '把类别限制取消',
                '不要类别限制',
                '类别限制去掉',
                '不限制类别',
                '恢复全类别',
                '全部类别都训练',
                '不要 classes',
                '取消 classes',
            )
        )

    @staticmethod
    def _wants_training_advanced_details(text: str) -> bool:
        lowered = str(text or '').lower()
        return any(
            token in text or token in lowered
            for token in (
                '高级参数',
                '高级配置',
                '展开参数',
                '详细参数',
                '更多参数',
                'advanced',
                'hyperparameter',
            )
        )

    def _collect_training_clear_fields(self, user_text: str) -> set[str]:
        clear_fields: set[str] = set()
        if self._wants_default_training_environment(user_text):
            clear_fields.add('training_environment')
        if self._wants_clear_batch(user_text):
            clear_fields.add('batch')
        if self._wants_clear_project(user_text):
            clear_fields.add('project')
        if self._wants_clear_run_name(user_text):
            clear_fields.add('name')
        if self._wants_clear_fraction(user_text):
            clear_fields.add('fraction')
        if self._wants_clear_classes(user_text):
            clear_fields.add('classes')
        return clear_fields

    def _collect_requested_training_args(self, user_text: str, *, data_yaml: str | None = '') -> dict[str, Any]:
        tr = self.session_state.active_training
        args: dict[str, Any] = {}
        known_environments = list((tr.last_environment_probe or {}).get('environments') or [])
        model = intent_parsing.extract_model_from_text(user_text)
        if any(token in user_text for token in ('现在用', '改用', '换成', '改成')):
            explicit_model_paths = re.findall(r'([A-Za-z0-9_./\\-]+\.(?:pt|onnx|yaml|yml))', user_text, flags=re.I)
            explicit_model_tokens = [token if '.' in token else f'{token}.pt' for token in re.findall(r'\b(yolo[a-zA-Z0-9._-]+)\b', user_text, flags=re.I)]
            candidates: list[str] = []
            for item in explicit_model_paths + explicit_model_tokens:
                if item not in candidates:
                    candidates.append(item)
            if candidates:
                model = candidates[-1]
        if model:
            args['model'] = model
        resolved_yaml = '' if data_yaml is None else str(data_yaml or tr.data_yaml or self.session_state.active_dataset.data_yaml or '').strip()
        if resolved_yaml:
            args['data_yaml'] = resolved_yaml
        return self._apply_training_text_overrides(
            user_text,
            args,
            known_environments=known_environments,
            include_classes_txt=True,
        )

    def _apply_training_text_overrides(
        self,
        user_text: str,
        args: dict[str, Any],
        *,
        known_environments: list[dict[str, Any]] | None = None,
        include_classes_txt: bool,
    ) -> dict[str, Any]:
        classes_txt = intent_parsing.extract_classes_txt_from_text(user_text)
        if include_classes_txt and classes_txt:
            args['classes_txt'] = classes_txt
        epochs = intent_parsing.extract_epochs_from_text(user_text)
        if epochs is not None:
            args['epochs'] = epochs
        batch = intent_parsing.extract_batch_size_from_text(user_text)
        if batch is not None:
            args['batch'] = batch
        imgsz = intent_parsing.extract_image_size_from_text(user_text)
        if imgsz is not None:
            args['imgsz'] = imgsz
        device = intent_parsing.extract_device_from_text(user_text)
        if device:
            args['device'] = device
        training_environment = intent_parsing.extract_training_environment_from_text(user_text, known_environments)
        if training_environment:
            args['training_environment'] = training_environment
        project = intent_parsing.extract_project_from_text(user_text)
        if project:
            args['project'] = project
        run_name = intent_parsing.extract_run_name_from_text(user_text)
        if run_name:
            args['name'] = run_name
        fraction = intent_parsing.extract_fraction_from_text(user_text)
        if fraction is not None:
            args['fraction'] = fraction
        classes = intent_parsing.extract_classes_from_text(user_text)
        if classes is not None:
            args['classes'] = classes
        single_cls = intent_parsing.extract_single_cls_flag_from_text(user_text)
        if single_cls is not None:
            args['single_cls'] = single_cls
        optimizer = intent_parsing.extract_optimizer_from_text(user_text)
        if optimizer:
            args['optimizer'] = optimizer
        freeze = intent_parsing.extract_freeze_from_text(user_text)
        if freeze is not None:
            args['freeze'] = freeze
        resume = intent_parsing.extract_resume_flag_from_text(user_text)
        if resume is not None:
            args['resume'] = resume
        lr0 = intent_parsing.extract_lr0_from_text(user_text)
        if lr0 is not None:
            args['lr0'] = lr0
        patience = intent_parsing.extract_patience_from_text(user_text)
        if patience is not None:
            args['patience'] = patience
        workers = intent_parsing.extract_workers_from_text(user_text)
        if workers is not None:
            args['workers'] = workers
        amp = intent_parsing.extract_amp_flag_from_text(user_text)
        if amp is not None:
            args['amp'] = amp
        return args

    def _collect_requested_training_loop_args(self, user_text: str, *, data_yaml: str | None = '') -> dict[str, Any]:
        tr = self.session_state.active_training
        args = self._collect_requested_training_args(user_text, data_yaml=data_yaml)
        if not str(args.get('model') or '').strip():
            preserved_model = str(
                (((tr.training_plan_draft or {}).get('planned_training_args') or {}).get('model'))
                or tr.model
                or ''
            ).strip()
            if preserved_model:
                args['model'] = preserved_model
        managed_level = self._extract_training_loop_managed_level(user_text)
        if managed_level:
            args['managed_level'] = managed_level
        max_rounds = self._extract_training_loop_max_rounds(user_text)
        if max_rounds is not None:
            args['max_rounds'] = max_rounds
            explicit_epoch_signal = bool(re.search(r'\bepochs?\b', user_text, flags=re.I)) or any(
                token in user_text for token in ('每轮训练', '每轮训', '每轮跑', '每次训练')
            )
            if args.get('epochs') == max_rounds and not explicit_epoch_signal:
                args.pop('epochs', None)
        target_metric, target_metric_value = self._extract_training_loop_target_metric(user_text)
        if target_metric:
            args['target_metric'] = target_metric
        if target_metric_value is not None:
            args['target_metric_value'] = target_metric_value
        loop_name = self._extract_training_loop_name(user_text) or intent_parsing.extract_run_name_from_text(user_text)
        if loop_name:
            args['loop_name'] = loop_name
        allowed_tuning_params = self._extract_training_loop_allowed_tuning_params(user_text)
        if allowed_tuning_params:
            args['allowed_tuning_params'] = allowed_tuning_params
        if any(token in user_text for token in ('自动处理OOM', '自动处理 OOM', 'OOM 自动恢复', 'oom自动恢复')):
            args['auto_handle_oom'] = True
        return args

    def _build_loop_prepare_args(self, user_text: str, dataset_path: str) -> dict[str, Any]:
        prepare_args: dict[str, Any] = {'dataset_path': dataset_path}
        explicit_classes_txt = str(intent_parsing.extract_classes_txt_from_text(user_text) or '').strip()
        if explicit_classes_txt:
            prepare_args['classes_txt'] = explicit_classes_txt
        if any(token in user_text for token in ('按默认比例', '默认比例', '先划分', '划分训练集', '划分数据集', 'split')):
            prepare_args['force_split'] = True
        return prepare_args

    def _compact_training_loop_start_fact(self, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        if tool_name == 'training_readiness':
            compact = {
                'ok': result.get('ok'),
                'ready': result.get('ready'),
                'preparable': result.get('preparable'),
                'summary': result.get('summary'),
                'resolved_data_yaml': result.get('resolved_data_yaml'),
                'blockers': result.get('blockers') or [],
                'warnings': result.get('warnings') or [],
            }
            readiness_overview = dict(result.get('readiness_overview') or {})
            if readiness_overview:
                compact['readiness_overview'] = readiness_overview
            device_overview = dict(result.get('device_overview') or {})
            if device_overview:
                compact['device_overview'] = device_overview
            action_candidates = self._compact_action_candidates(result.get('action_candidates'))
            if action_candidates:
                compact['action_candidates'] = action_candidates
            return compact
        if tool_name == 'list_training_environments':
            environments = []
            for item in list(result.get('environments') or []):
                if not isinstance(item, dict):
                    continue
                environments.append(
                    {
                        'name': item.get('name'),
                        'display_name': item.get('display_name'),
                        'selected_by_default': item.get('selected_by_default'),
                    }
                )
            compact = {
                'ok': result.get('ok'),
                'summary': result.get('summary'),
                'default_environment': result.get('default_environment'),
                'environments': environments,
            }
            environment_overview = dict(result.get('environment_overview') or {})
            if environment_overview:
                compact['environment_overview'] = environment_overview
            action_candidates = self._compact_action_candidates(result.get('action_candidates'))
            if action_candidates:
                compact['action_candidates'] = action_candidates
            return compact
        if tool_name == 'prepare_dataset_for_training':
            compact = {
                'ok': result.get('ok'),
                'ready': result.get('ready'),
                'summary': result.get('summary'),
                'data_yaml': result.get('data_yaml'),
                'warnings': result.get('warnings') or [],
                'risk_level': result.get('risk_level'),
            }
            prepare_overview = dict(result.get('prepare_overview') or {})
            if prepare_overview:
                compact['prepare_overview'] = prepare_overview
            action_candidates = self._compact_action_candidates(result.get('action_candidates'))
            if action_candidates:
                compact['action_candidates'] = action_candidates
            else:
                compact['next_actions'] = result.get('next_actions') or []
            return compact
        return {'ok': result.get('ok'), 'summary': result.get('summary')}

    def _known_training_loop_data_yaml(
        self,
        loop_args: dict[str, Any],
        observed_tools: dict[str, dict[str, Any]] | None = None,
        dataset_path: str = '',
    ) -> str:
        observed_tools = dict(observed_tools or {})
        prepare_result = dict(observed_tools.get('prepare_dataset_for_training') or {})
        readiness = dict(observed_tools.get('training_readiness') or {})
        active_dataset_root = str(self.session_state.active_dataset.dataset_root or '').strip()
        active_img_dir = str(self.session_state.active_dataset.img_dir or '').strip()
        can_reuse_session_yaml = not dataset_path or dataset_path in {active_dataset_root, active_img_dir}
        session_yaml = ''
        if can_reuse_session_yaml:
            session_yaml = str(
                self.session_state.active_dataset.data_yaml
                or self.session_state.active_training.data_yaml
                or ''
            ).strip()
        return str(
            loop_args.get('data_yaml')
            or prepare_result.get('data_yaml')
            or readiness.get('resolved_data_yaml')
            or session_yaml
            or ''
        ).strip()

    def _build_training_loop_start_fallback_plan(
        self,
        *,
        user_text: str,
        dataset_path: str,
        loop_args: dict[str, Any],
        observed_tools: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        observed_tools = dict(observed_tools or {})
        readiness = dict(observed_tools.get('training_readiness') or {})
        prepare_result = dict(observed_tools.get('prepare_dataset_for_training') or {})
        model = str(loop_args.get('model') or '').strip()
        data_yaml = self._known_training_loop_data_yaml(loop_args, observed_tools, dataset_path=dataset_path)
        if not model:
            return {
                'decision': 'block',
                'reason': '当前还不能开启环训练：缺少预训练权重/模型。请先明确模型，例如 yolov8n.pt。',
                'planner_source': 'fallback',
            }
        if prepare_result.get('ok') and data_yaml:
            next_args = dict(loop_args)
            next_args['model'] = model
            next_args['data_yaml'] = data_yaml
            if not str(next_args.get('managed_level') or '').strip():
                next_args['managed_level'] = 'conservative_auto'
            if next_args.get('max_rounds') in {None, ''}:
                next_args['max_rounds'] = 5
            return {
                'decision': 'start',
                'next_tool': 'start_training_loop',
                'next_args': next_args,
                'reason': '数据已经准备完成，可以直接启动循环训练。',
                'planner_source': 'fallback',
            }
        if data_yaml:
            next_args = dict(loop_args)
            next_args['model'] = model
            next_args['data_yaml'] = data_yaml
            if not str(next_args.get('managed_level') or '').strip():
                next_args['managed_level'] = 'conservative_auto'
            if next_args.get('max_rounds') in {None, ''}:
                next_args['max_rounds'] = 5
            return {
                'decision': 'start',
                'next_tool': 'start_training_loop',
                'next_args': next_args,
                'reason': '当前数据已具备训练条件，可以直接进入循环训练。',
                'planner_source': 'fallback',
            }
        if not readiness:
            if dataset_path:
                return {
                    'decision': 'observe',
                    'next_tool': 'training_readiness',
                    'next_args': {'img_dir': dataset_path},
                    'reason': '先读取训练前检查结果，再决定是 prepare 还是 start。',
                    'planner_source': 'fallback',
                }
            return {
                'decision': 'block',
                'reason': '当前还不能开启环训练：缺少可用数据路径，无法判断是否需要先 prepare。',
                'planner_source': 'fallback',
            }
        if readiness and not readiness.get('ok', True) and not readiness.get('preparable'):
            blockers = [str(item).strip() for item in (readiness.get('blockers') or []) if str(item).strip()]
            blocker_detail = str(readiness.get('error') or (blockers[0] if blockers else '') or readiness.get('summary') or '').strip()
            return {
                'decision': 'block',
                'reason': f'当前还不能开启环训练：{blocker_detail or "训练前检查失败"}',
                'planner_source': 'fallback',
            }
        if dataset_path and readiness.get('preparable'):
            return {
                'decision': 'prepare',
                'next_tool': 'prepare_dataset_for_training',
                'next_args': self._build_loop_prepare_args(user_text, dataset_path),
                'reason': '当前数据还不能直接进入循环训练，先准备数据集，再继续启动 loop。',
                'planner_source': 'fallback',
            }
        blockers = [str(item).strip() for item in (readiness.get('blockers') or []) if str(item).strip()]
        blocker_detail = str(readiness.get('error') or (blockers[0] if blockers else '') or readiness.get('summary') or '').strip()
        return {
            'decision': 'block',
            'reason': f'当前还不能开启环训练：{blocker_detail or "缺少可训练的 data_yaml。"}',
            'planner_source': 'fallback',
        }

    async def _plan_training_loop_start(
        self,
        *,
        user_text: str,
        dataset_path: str,
        loop_args: dict[str, Any],
        observed_tools: dict[str, dict[str, Any]] | None = None,
        step_index: int = 1,
    ) -> dict[str, Any]:
        observed_tools = dict(observed_tools or {})
        fallback = self._build_training_loop_start_fallback_plan(
            user_text=user_text,
            dataset_path=dataset_path,
            loop_args=loop_args,
            observed_tools=observed_tools,
        )
        if self.planner_llm is None:
            return fallback

        current_data_yaml = self._known_training_loop_data_yaml(loop_args, observed_tools, dataset_path=dataset_path)
        facts = {
            'user_request': user_text,
            'dataset_path': dataset_path,
            'step_index': step_index,
            'requested_loop_args': dict(loop_args),
            'current_data_yaml': current_data_yaml,
            'observed_tool_names': list(observed_tools.keys()),
            'observed_tools': {
                name: self._compact_training_loop_start_fact(name, result)
                for name, result in observed_tools.items()
            },
            'available_tools': [
                {
                    'tool': 'training_readiness',
                    'when': '需要先确认数据是否已具备直接训练条件',
                    'kind': 'read_only',
                    'args_rule': {'img_dir': dataset_path or '<dataset_path>'},
                },
                {
                    'tool': 'list_training_environments',
                    'when': '只有在确实需要补充训练环境事实时才调用',
                    'kind': 'read_only',
                    'args_rule': {},
                },
                {
                    'tool': 'prepare_dataset_for_training',
                    'when': '需要生成 data_yaml / 划分数据 / 准备训练产物时调用',
                    'kind': 'high_risk',
                    'args_rule': {'dataset_path': dataset_path or '<dataset_path>'},
                },
                {
                    'tool': 'start_training_loop',
                    'when': '模型已明确且 data_yaml 已就绪时调用',
                    'kind': 'high_risk',
                    'args_rule': {'model': loop_args.get('model') or '<model>', 'data_yaml': current_data_yaml or '<data_yaml>'},
                },
            ],
        }
        messages = [
            SystemMessage(
                content=(
                    '你是 YoloStudio Agent 的循环训练启动编排器。'
                    '你要根据当前事实决定“下一步只调用一个什么工具”，而不是写解释性散文。'
                    '请优先选择最小必要的下一步：如果事实不足，优先选只读工具；如果事实已经足够，直接选真正要执行的高风险工具。'
                    '不要默认先收集所有事实；只选择你当前真正需要的下一步。'
                    '输出必须是 JSON，对象格式固定为 '
                    '{"next_tool":"training_readiness|list_training_environments|prepare_dataset_for_training|start_training_loop|block",'
                    '"reason":"..."}。'
                    '如果模型缺失，返回 block。'
                    '如果 data_yaml 已就绪且模型明确，优先 start_training_loop。'
                    '如果 data_yaml 缺失但数据可能可准备，优先 training_readiness 或 prepare_dataset_for_training。'
                    '不要输出 Markdown，不要输出额外文字。'
                )
            ),
            HumanMessage(
                content=(
                    '当前事实：\n'
                    f'{json.dumps(facts, ensure_ascii=False, indent=2)}'
                )
            ),
        ]
        try:
            parsed = await self._invoke_structured_payload(
                messages=messages,
                schema={
                    'title': 'yolostudio_training_loop_start_plan',
                    'type': 'object',
                    'properties': {
                        'next_tool': {
                            'type': 'string',
                            'enum': [
                                'training_readiness',
                                'list_training_environments',
                                'prepare_dataset_for_training',
                                'start_training_loop',
                                'block',
                            ],
                        },
                        'reason': {'type': 'string'},
                    },
                    'required': ['next_tool', 'reason'],
                    'additionalProperties': False,
                },
            )
            plan = self._normalize_training_loop_start_plan(
                parsed=parsed,
                user_text=user_text,
                dataset_path=dataset_path,
                loop_args=loop_args,
                observed_tools=observed_tools,
            )
            if plan is not None:
                plan['planner_source'] = 'llm'
                plan['planner_payload'] = parsed
                self.memory.append_event(
                    self.session_state.session_id,
                    'loop_start_planned',
                    {
                        'source': 'llm',
                        'decision': plan.get('decision'),
                        'next_tool': plan.get('next_tool'),
                        'step_index': step_index,
                    },
                )
                return plan
        except Exception as exc:
            self.memory.append_event(
                self.session_state.session_id,
                'loop_start_plan_failed',
                {'error': str(exc)},
            )
        self.memory.append_event(
            self.session_state.session_id,
            'loop_start_planned',
            {
                'source': 'fallback',
                'decision': fallback.get('decision'),
                'next_tool': fallback.get('next_tool'),
                'step_index': step_index,
            },
        )
        return fallback

    def _normalize_training_loop_start_plan(
        self,
        *,
        parsed: dict[str, Any],
        user_text: str,
        dataset_path: str,
        loop_args: dict[str, Any],
        observed_tools: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        observed_tools = dict(observed_tools or {})
        readiness = dict(observed_tools.get('training_readiness') or {})
        decision = str(parsed.get('decision') or '').strip().lower()
        next_tool = str(parsed.get('next_tool') or '').strip()
        reason = str(parsed.get('reason') or '').strip()
        next_tool = next_tool or {'prepare': 'prepare_dataset_for_training', 'start': 'start_training_loop', 'block': 'block'}.get(decision, '')

        model = str(loop_args.get('model') or '').strip()
        data_yaml = self._known_training_loop_data_yaml(loop_args, observed_tools, dataset_path=dataset_path)
        if next_tool == 'training_readiness':
            if not dataset_path:
                return None
            return {
                'decision': 'observe',
                'next_tool': 'training_readiness',
                'next_args': {'img_dir': dataset_path},
                'reason': reason or '先读取训练前检查结果，再决定是否 prepare 或直接 start。',
            }
        if next_tool == 'list_training_environments':
            return {
                'decision': 'observe',
                'next_tool': 'list_training_environments',
                'next_args': {},
                'reason': reason or '先确认当前可用训练环境，再决定是否直接启动循环训练。',
            }
        if next_tool == 'start_training_loop':
            if not model or not data_yaml:
                return None
            next_args = dict(loop_args)
            next_args['model'] = model
            next_args['data_yaml'] = data_yaml
            if not str(next_args.get('managed_level') or '').strip():
                next_args['managed_level'] = 'conservative_auto'
            if next_args.get('max_rounds') in {None, ''}:
                next_args['max_rounds'] = 5
            return {
                'decision': 'start',
                'next_tool': 'start_training_loop',
                'next_args': next_args,
                'reason': reason or '当前数据已具备训练条件，可以直接进入循环训练。',
            }
        if next_tool == 'prepare_dataset_for_training':
            if not dataset_path:
                return None
            return {
                'decision': 'prepare',
                'next_tool': 'prepare_dataset_for_training',
                'next_args': self._build_loop_prepare_args(user_text, dataset_path),
                'reason': reason or '当前数据还不能直接进入循环训练，先准备数据集，再继续启动 loop。',
            }
        if next_tool in {'block', 'none', ''} or decision == 'block':
            blockers = [str(item).strip() for item in (readiness.get('blockers') or []) if str(item).strip()]
            blocker_detail = str(readiness.get('error') or (blockers[0] if blockers else '') or readiness.get('summary') or '').strip()
            return {
                'decision': 'block',
                'reason': reason or f'当前还不能开启环训练：{blocker_detail or "缺少可训练的 data_yaml。"}',
            }
        return None

    def _build_training_loop_start_draft(
        self,
        *,
        user_text: str,
        dataset_path: str,
        loop_args: dict[str, Any],
        observed_tools: dict[str, dict[str, Any]] | None,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        observed_tools = dict(observed_tools or {})
        readiness = dict(observed_tools.get('training_readiness') or {})
        prepare_result = dict(observed_tools.get('prepare_dataset_for_training') or {})
        latest_summary = str(
            prepare_result.get('summary')
            or readiness.get('summary')
            or self.session_state.active_dataset.last_readiness.get('summary')
            or ''
        ).strip()
        planned_args = dict(loop_args)
        data_yaml = self._known_training_loop_data_yaml(planned_args, observed_tools, dataset_path=dataset_path)
        if data_yaml:
            planned_args['data_yaml'] = data_yaml
        next_tool_name = str(plan.get('next_tool') or '').strip()
        previous_draft = dict(self.session_state.active_training.training_plan_draft or {})
        execution_mode = 'prepare_then_loop' if next_tool_name == 'prepare_dataset_for_training' else 'direct_loop'
        if next_tool_name == 'start_training_loop' and (
            'prepare_dataset_for_training' in observed_tools
            or str(previous_draft.get('execution_mode') or '').strip().lower() == 'prepare_then_loop'
        ):
            execution_mode = 'prepare_then_loop'
        return {
            'source_intent': 'training_loop',
            'execution_mode': execution_mode,
            'execution_backend': 'standard_yolo',
            'dataset_path': dataset_path,
            'data_summary': latest_summary,
            'reasoning_summary': str(plan.get('reason') or '').strip(),
            'planned_training_args': dict(planned_args),
            'planned_loop_args': dict(planned_args),
            'next_step_tool': next_tool_name,
            'next_step_args': dict(plan.get('next_args') or {}),
            'planner_decision_source': str(plan.get('planner_source') or 'fallback'),
            'planner_decision': 'prepare' if next_tool_name == 'prepare_dataset_for_training' else 'start',
            'planner_output': dict(plan.get('planner_payload') or {}),
            'planner_user_request': user_text,
            'planner_observed_tools': list(observed_tools.keys()),
            'editable_fields': ['model', 'epochs', 'batch', 'imgsz', 'device', 'training_environment', 'project', 'name'],
        }

    async def _run_training_loop_start_orchestration(
        self,
        *,
        user_text: str,
        thread_id: str,
        dataset_path: str,
        loop_args: dict[str, Any],
        observed_tools: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        observed: dict[str, dict[str, Any]] = dict(observed_tools or {})
        if not str(loop_args.get('model') or '').strip():
            reply = '当前还不能开启环训练：缺少预训练权重/模型。请先明确模型，例如 yolov8n.pt。'
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        seen_observe_calls: set[tuple[str, str]] = set()
        for step_index in range(1, 5):
            known_data_yaml = self._known_training_loop_data_yaml(loop_args, observed, dataset_path=dataset_path)
            if known_data_yaml:
                loop_args['data_yaml'] = known_data_yaml
            plan = await self._plan_training_loop_start(
                user_text=user_text,
                dataset_path=dataset_path,
                loop_args=loop_args,
                observed_tools=observed,
                step_index=step_index,
            )
            if plan.get('decision') == 'block':
                reply = str(plan.get('reason') or '当前还不能开启环训练。').strip()
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}

            next_tool_name = str(plan.get('next_tool') or '').strip()
            next_tool_args = dict(plan.get('next_args') or {})
            if next_tool_name in {'training_readiness', 'list_training_environments'}:
                observe_key = (next_tool_name, json.dumps(next_tool_args, ensure_ascii=False, sort_keys=True))
                if observe_key in seen_observe_calls:
                    reply = '当前还不能稳定规划下一步；读到的事实没有继续收敛。请换一种方式说明需求，或直接明确 data.yaml / 模型。'
                    self._messages.append(AIMessage(content=reply))
                    return {'status': 'completed', 'message': reply, 'tool_call': None}
                seen_observe_calls.add(observe_key)
                observed_result = await self.direct_tool(next_tool_name, _state_mode='observe', **next_tool_args)
                observed[next_tool_name] = observed_result
                self.memory.append_event(
                    self.session_state.session_id,
                    'loop_start_observed_tool',
                    {
                        'tool': next_tool_name,
                        'args': next_tool_args,
                        'result': self._compact_training_loop_start_fact(next_tool_name, observed_result),
                        'step_index': step_index,
                    },
                )
                continue

            draft = self._build_training_loop_start_draft(
                user_text=user_text,
                dataset_path=dataset_path,
                loop_args=loop_args,
                observed_tools=observed,
                plan=plan,
            )
            self._save_training_plan_draft(draft)
            pending = {
                'name': next_tool_name,
                'args': next_tool_args,
                'id': None,
                'synthetic': True,
            }
            self._set_pending_confirmation(thread_id, pending)
            reply = await self._build_confirmation_message(pending)
            self._messages.append(AIMessage(content=reply))
            return self._needs_confirmation_result(thread_id, pending, reply)

        reply = '当前还不能稳定规划循环训练启动步骤：内部编排步数已达到上限。请换一种方式说明需求，或直接给出可训练的 data.yaml。'
        self._messages.append(AIMessage(content=reply))
        return {'status': 'completed', 'message': reply, 'tool_call': None}

    async def _continue_training_loop_start_after_prepare(
        self,
        *,
        thread_id: str,
        prepare_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        draft = dict(self.session_state.active_training.training_plan_draft or {})
        if str(draft.get('source_intent') or '').strip().lower() != 'training_loop':
            return None
        user_text = str(draft.get('planner_user_request') or self._recent_user_text() or '').strip()
        if not user_text:
            return None
        dataset_path = str(
            draft.get('dataset_path')
            or self.session_state.active_dataset.dataset_root
            or self.session_state.active_dataset.img_dir
            or ''
        ).strip()
        loop_args = dict(draft.get('planned_loop_args') or draft.get('planned_training_args') or {})
        prepared_yaml = str(prepare_result.get('data_yaml') or '').strip()
        if prepared_yaml:
            loop_args['data_yaml'] = prepared_yaml
        return await self._run_training_loop_start_orchestration(
            user_text=user_text,
            thread_id=thread_id,
            dataset_path=dataset_path,
            loop_args=loop_args,
            observed_tools={'prepare_dataset_for_training': dict(prepare_result or {})},
        )

    @staticmethod
    def _extract_training_loop_id_from_text(text: str) -> str:
        match = re.search(r'\b\d{9,}-[A-Za-z0-9-]+\b', text)
        return match.group(0) if match else ''

    async def _resolve_training_loop_route(
        self,
        *,
        user_text: str,
        normalized_text: str,
        wants_predict: bool,
        wants_train: bool,
        wants_stop_training: bool,
        explicit_run_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        explicit_run_ids = list(explicit_run_ids or [])
        has_training_loop_context = bool(
            self.session_state.active_training.active_loop_id
            or self.session_state.active_training.last_loop_status
            or self.session_state.active_training.last_loop_detail
        )
        mentions_loop = any(
            token in user_text for token in ('环训练', '循环训练', '循环训', '循环跑', '自动复训', '自动续训', '自动下一轮', 'agent环训练')
        ) or any(
            token in normalized_text for token in ('training loop', 'loop training', 'auto retrain', 'auto training loop')
        )
        start_like = any(
            token in user_text for token in ('开', '启动', '开始', '跑', '来一个', '开启', '创建', '循环训', '循环跑', '训一下', '跑几轮', '训几轮', '试几轮')
        )
        loop_status_phrase = any(
            token in user_text for token in ('状态', '进度', '到哪了', '第几轮', '跑到哪了', '现在怎么样', '怎么样', '怎么样了', '咋样', '咋样了', '情况如何')
        ) or any(token in normalized_text for token in ('training loop status', 'loop status'))
        generic_training_status_in_loop = any(
            token in user_text for token in (
                '训练状态', '当前训练状态', '训练进度', '当前进度',
                '查看训练状态', '再次查看训练状态', '看一下训练状态', '再看一下训练状态',
                '训练情况', '查看训练情况', '看看训练情况', '看训练情况',
                '查看当前状态', '当前状态', '再看当前状态', '再次查看当前状态',
                '查看情况', '看情况', '看下情况', '看看情况', '现在情况', '现在什么情况',
            )
        )
        generic_training_detail_in_loop = any(
            token in user_text for token in (
                '查看训练详情', '训练详情', '查看详情', '完整详情', '详细情况', '完整情况', '轮次详情', '轮次对比',
                '训练信息', '详细训练信息',
            )
        )
        explicit_loop_detail = any(
            token in user_text for token in ('查看环训练详情', '环训练详情', '循环训练详情', '查看自动复训详情')
        )
        loop_id = self._extract_training_loop_id_from_text(user_text) or self.session_state.active_training.active_loop_id

        if wants_predict:
            return {'action': '', 'loop_id': loop_id, 'has_context': has_training_loop_context}
        if any(token in user_text for token in ('环训练列表', '最近环训练', '环训练历史', '有哪些环训练', '最近自动复训')) or any(
            token in normalized_text for token in ('list training loops', 'training loop history')
        ):
            return {'action': 'list', 'loop_id': loop_id, 'has_context': has_training_loop_context}
        if any(token in user_text for token in ('暂停环训练', '循环训练暂停', '这一轮结束后停住', '别自动开下一轮', '下一轮先别跑')) or any(
            token in normalized_text for token in ('pause training loop', 'pause loop')
        ):
            return {'action': 'pause', 'loop_id': loop_id, 'has_context': has_training_loop_context}
        if any(token in user_text for token in ('恢复环训练', '继续环训练', '继续自动复训', '从下一轮开始继续')) or any(
            token in normalized_text for token in ('resume training loop', 'resume loop')
        ):
            return {'action': 'resume', 'loop_id': loop_id, 'has_context': has_training_loop_context}
        if any(token in user_text for token in ('停止环训练', '终止环训练', '结束环训练', '马上停掉环训练', '立即终止当前环训练')) or any(
            token in normalized_text for token in ('stop training loop', 'stop loop')
        ):
            return {'action': 'stop', 'loop_id': loop_id, 'has_context': has_training_loop_context}
        if has_training_loop_context and wants_stop_training and not any(
            token in user_text for token in ('当前轮', '这一轮', '本轮', '单轮', '只停训练', '不结束环训练')
        ):
            return {'action': 'stop', 'loop_id': loop_id, 'has_context': has_training_loop_context}
        if mentions_loop and start_like:
            return {'action': 'start', 'loop_id': loop_id, 'has_context': has_training_loop_context}
        if has_training_loop_context and not explicit_run_ids:
            classified_action = await self._classify_training_loop_followup_action(
                user_text=user_text,
                normalized_text=normalized_text,
                loop_id=loop_id,
            )
            if classified_action:
                return {'action': classified_action, 'loop_id': loop_id, 'has_context': has_training_loop_context}
        if explicit_loop_detail or (
            has_training_loop_context
            and not explicit_run_ids
            and (
                generic_training_detail_in_loop
                or (('详细' in user_text or 'detail' in normalized_text) and ('情况' in user_text or '信息' in user_text or '状态' in user_text or 'training' in normalized_text))
                or (mentions_loop and any(token in user_text for token in ('第几轮', '轮次详情', '轮次对比', '完整详情')))
            )
        ):
            return {'action': 'inspect', 'loop_id': loop_id, 'has_context': has_training_loop_context}
        if any(token in user_text for token in ('环训练状态', '循环训练状态', '自动复训状态', '环训练进度', '循环训练进度')) or (
            has_training_loop_context
            and (
                (mentions_loop and loop_status_phrase)
                or generic_training_status_in_loop
                or (loop_status_phrase and not wants_train)
            )
        ):
            return {'action': 'status', 'loop_id': loop_id, 'has_context': has_training_loop_context}
        return {'action': '', 'loop_id': loop_id, 'has_context': has_training_loop_context}

    async def _classify_training_loop_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
        loop_id: str,
    ) -> str:
        if self.planner_llm is None:
            return ''
        facts = {
            'active_loop_id': loop_id or self.session_state.active_training.active_loop_id or '',
            'last_loop_status_summary': str((self.session_state.active_training.last_loop_status or {}).get('summary') or '').strip(),
            'last_loop_detail_summary': str((self.session_state.active_training.last_loop_detail or {}).get('summary') or '').strip(),
            'has_last_loop_status': bool(self.session_state.active_training.last_loop_status),
            'has_last_loop_detail': bool(self.session_state.active_training.last_loop_detail),
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'status', 'inspect'},
        )

    async def _classify_training_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
        metric_signals: list[str],
    ) -> str:
        if self.planner_llm is None:
            return ''
        active_training = self.session_state.active_training
        knowledge = self.session_state.active_knowledge
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'status', 'analysis', 'next_step', 'knowledge'},
        )

    async def _classify_training_history_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
    ) -> str:
        if self.planner_llm is None:
            return ''
        active_training = self.session_state.active_training
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'runs', 'inspect', 'compare', 'best'},
        )

    async def _classify_training_loop_history_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
    ) -> str:
        if self.planner_llm is None:
            return ''
        active_training = self.session_state.active_training
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'list', 'status', 'inspect'},
        )

    async def _classify_knowledge_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
        metric_signals: list[str],
    ) -> str:
        if self.planner_llm is None:
            return ''
        knowledge = self.session_state.active_knowledge
        active_training = self.session_state.active_training
        facts = {
            'has_last_retrieval': bool(knowledge.last_retrieval),
            'last_retrieval': knowledge.last_retrieval,
            'has_last_analysis': bool(knowledge.last_analysis),
            'last_analysis': knowledge.last_analysis,
            'has_last_recommendation': bool(knowledge.last_recommendation),
            'last_recommendation': knowledge.last_recommendation,
            'has_training_context': bool(
                active_training.training_run_summary
                or active_training.last_summary
                or active_training.last_status
                or active_training.running
            ),
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'knowledge', 'analysis', 'next_step'},
        )

    async def _classify_prediction_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
        fallback_path: str,
    ) -> str:
        if self.planner_llm is None:
            return ''
        active_prediction = self.session_state.active_prediction
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'summary', 'inspect'},
        )

    async def _classify_realtime_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
    ) -> str:
        if self.planner_llm is None:
            return ''
        active_prediction = self.session_state.active_prediction
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'status'},
        )

    async def _classify_prediction_management_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
    ) -> str:
        if self.planner_llm is None:
            return ''
        active_prediction = self.session_state.active_prediction
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'inspect', 'export', 'path_lists', 'organize'},
        )

    async def _classify_dataset_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
        fallback_path: str,
    ) -> str:
        if self.planner_llm is None:
            return ''
        active_dataset = self.session_state.active_dataset
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'quality', 'health', 'duplicates'},
        )

    async def _classify_extract_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
    ) -> str:
        if self.planner_llm is None:
            return ''
        active_dataset = self.session_state.active_dataset
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'preview', 'extract', 'video_scan', 'frame_extract'},
        )

    def _extract_followup_result(self, action: str) -> tuple[str, dict[str, Any]] | None:
        ds = self.session_state.active_dataset
        mapping: dict[str, tuple[str, dict[str, Any]]] = {
            'preview': ('preview_extract_images', dict(ds.last_extract_preview or {})),
            'extract': ('extract_images', dict(ds.last_extract_result or {})),
            'video_scan': ('scan_videos', dict(ds.last_video_scan or {})),
            'frame_extract': ('extract_video_frames', dict(ds.last_frame_extract or {})),
        }
        result = mapping.get(action)
        if result and result[1]:
            return result
        return None

    def _prediction_followup_result(self, action: str) -> tuple[str, dict[str, Any]] | None:
        pred = self.session_state.active_prediction
        summary_payload = dict(pred.last_summary or {})
        if not summary_payload:
            last_result = dict(pred.last_result or {})
            if any(
                key in last_result
                for key in ('summary_overview', 'action_candidates', 'total_detections')
            ):
                summary_payload = last_result
        mapping: dict[str, tuple[str, dict[str, Any]]] = {
            'inspect': ('inspect_prediction_outputs', dict(pred.last_inspection or {})),
            'summary': ('summarize_prediction_results', summary_payload),
        }
        result = mapping.get(action)
        if result and result[1]:
            return result
        return None

    def _prediction_management_followup_result(self, action: str) -> tuple[str, dict[str, Any]] | None:
        pred = self.session_state.active_prediction
        mapping: dict[str, tuple[str, dict[str, Any]]] = {
            'inspect': ('inspect_prediction_outputs', dict(pred.last_inspection or {})),
            'export': ('export_prediction_report', dict(pred.last_export or {})),
            'path_lists': ('export_prediction_path_lists', dict(pred.last_path_lists or {})),
            'organize': ('organize_prediction_results', dict(pred.last_organized_result or {})),
        }
        result = mapping.get(action)
        if result and result[1]:
            return result
        return None

    @staticmethod
    def _cached_request_matches_targets(
        payload: dict[str, Any],
        request_kwargs: dict[str, Any],
        *,
        target_keys: tuple[str, ...],
    ) -> bool:
        for key in target_keys:
            expected = str(request_kwargs.get(key) or '').strip()
            if not expected:
                continue
            actual = str(payload.get(key) or '').strip()
            if not actual or actual != expected:
                return False
        return True

    def _prediction_request_cached_result(
        self,
        request: str,
        request_kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any]] | None:
        if request == 'inspect':
            cached = self._prediction_management_followup_result('inspect')
        elif request == 'summary':
            cached = self._prediction_followup_result('summary')
        else:
            cached = None
        if not cached:
            return None
        tool_name, payload = cached
        if self._cached_request_matches_targets(payload, request_kwargs, target_keys=('report_path', 'output_dir')):
            return tool_name, payload
        return None

    def _prediction_management_request_cached_result(
        self,
        request: str,
        request_kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any]] | None:
        if request == 'export':
            cached = self._prediction_management_followup_result('export')
            target_keys = ('report_path', 'output_dir', 'export_path')
        elif request == 'path_lists':
            cached = self._prediction_management_followup_result('path_lists')
            target_keys = ('report_path', 'output_dir', 'export_dir')
        else:
            return None
        if not cached:
            return None
        tool_name, payload = cached
        if self._cached_request_matches_targets(payload, request_kwargs, target_keys=target_keys):
            return tool_name, payload
        return None

    def _knowledge_followup_result(self, action: str) -> tuple[str, dict[str, Any]] | None:
        knowledge = self.session_state.active_knowledge
        mapping: dict[str, tuple[str, dict[str, Any]]] = {
            'knowledge': ('retrieve_training_knowledge', dict(knowledge.last_retrieval or {})),
            'analysis': ('analyze_training_outcome', dict(knowledge.last_analysis or {})),
            'next_step': ('recommend_next_training_step', dict(knowledge.last_recommendation or {})),
        }
        result = mapping.get(action)
        if result and result[1]:
            payload = result[1]
            if any(
                key in payload
                for key in ('retrieval_overview', 'analysis_overview', 'recommendation_overview', 'action_candidates')
            ):
                return result
            return None
        return None

    def _training_followup_cached_result(self, action: str) -> tuple[str, dict[str, Any]] | None:
        if action == 'status':
            return None
        return self._knowledge_followup_result(action)

    def _dataset_followup_result(self, action: str) -> tuple[str, dict[str, Any]] | None:
        ds = self.session_state.active_dataset
        mapping: dict[str, tuple[str, dict[str, Any]]] = {
            'health': ('run_dataset_health_check', dict(ds.last_health_check or {})),
            'duplicates': ('detect_duplicate_images', dict(ds.last_duplicate_check or {})),
        }
        result = mapping.get(action)
        if result and result[1]:
            return result
        return None

    def _dataset_extract_request_cached_result(
        self,
        request: str,
        request_kwargs: dict[str, Any],
        *,
        dataset_path: str = '',
    ) -> tuple[str, dict[str, Any]] | None:
        ds = self.session_state.active_dataset
        if request == 'preview':
            if dataset_path and not self._dataset_request_cache_allowed(dataset_path):
                return None
            cached = ('preview_extract_images', dict(ds.last_extract_preview or {}))
            target_keys = ('source_path', 'output_dir', 'workflow_ready_path')
        elif request == 'video_scan':
            cached = ('scan_videos', dict(ds.last_video_scan or {}))
            target_keys = ('source_path',)
        else:
            return None
        tool_name, payload = cached
        if not payload:
            return None
        if self._cached_request_matches_targets(payload, request_kwargs, target_keys=target_keys):
            return tool_name, payload
        return None

    def _dataset_request_cache_allowed(self, dataset_path: str) -> bool:
        target = str(dataset_path or '').strip()
        if not target:
            return True
        ds = self.session_state.active_dataset
        candidates = {
            str(ds.dataset_root or '').strip(),
            str(ds.img_dir or '').strip(),
            str(ds.data_yaml or '').strip(),
        }
        candidates.discard('')
        return target in candidates

    def _training_history_request_cached_result(
        self,
        *,
        request: str,
        run_state: str | None = None,
        analysis_ready: bool = False,
        run_id: str | None = None,
        left_run_id: str | None = None,
        right_run_id: str | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        training = self.session_state.active_training
        if request == 'runs':
            runs = list(training.recent_runs or [])
            if analysis_ready:
                runs = [item for item in runs if bool(item.get('analysis_ready'))]
            elif run_state:
                runs = [item for item in runs if str(item.get('run_state') or '').strip().lower() == run_state]
            if runs:
                return (
                    'list_training_runs',
                    {
                        'ok': True,
                        'summary': '训练历史查询完成',
                        'runs': runs,
                    },
                )
            return None
        if request == 'compare':
            payload = dict(training.last_run_comparison or {})
            cached_left = str(payload.get('left_run_id') or '').strip()
            cached_right = str(payload.get('right_run_id') or '').strip()
            expected_left = str(left_run_id or '').strip()
            expected_right = str(right_run_id or '').strip()
            if payload and (
                (not expected_left and not expected_right)
                or (
                    (not expected_left or cached_left == expected_left)
                    and (not expected_right or cached_right == expected_right)
                )
            ):
                return ('compare_training_runs', payload)
            return None
        if request == 'best':
            payload = dict(training.best_run_selection or {})
            return ('select_best_training_run', payload) if payload else None
        if request == 'inspect':
            payload = dict(training.last_run_inspection or {})
            cached_run_id = str(payload.get('selected_run_id') or payload.get('run_id') or '').strip()
            if payload and (not run_id or not cached_run_id or cached_run_id == str(run_id).strip()):
                return ('inspect_training_run', payload)
        return None

    def _training_loop_request_cached_result(
        self,
        request: str,
        *,
        loop_id: str | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        training = self.session_state.active_training
        if request == 'list':
            loops = list(training.recent_loops or [])
            if loops:
                return (
                    'list_training_loops',
                    {
                        'ok': True,
                        'summary': '环训练列表已就绪',
                        'loops': loops,
                    },
                )
            return None
        if request == 'inspect':
            payload = dict(training.last_loop_detail or {})
            cached_loop_id = str(payload.get('loop_id') or '').strip()
            if payload and (not loop_id or not cached_loop_id or cached_loop_id == str(loop_id).strip()):
                return ('inspect_training_loop', payload)
        return None

    async def _classify_remote_transfer_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
    ) -> str:
        if self.planner_llm is None:
            return ''
        active_remote = self.session_state.active_remote_transfer
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'profiles', 'upload', 'download'},
        )

    async def _classify_remote_roundtrip_followup_action(
        self,
        *,
        user_text: str,
        normalized_text: str,
    ) -> str:
        if self.planner_llm is None:
            return ''
        active_training = self.session_state.active_training
        active_prediction = self.session_state.active_prediction
        facts = {
            'has_training_remote_roundtrip': bool(active_training.last_remote_roundtrip),
            'training_remote_roundtrip': active_training.last_remote_roundtrip,
            'training_running': bool(active_training.running),
            'has_prediction_remote_roundtrip': bool(active_prediction.last_remote_roundtrip),
            'prediction_remote_roundtrip': active_prediction.last_remote_roundtrip,
            'has_local_training_context': bool(
                active_training.last_status
                or active_training.last_summary
                or active_training.training_run_summary
                or active_training.running
            ),
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
        return await self._classify_structured_action(
            messages=messages,
            allowed_actions={'training_pipeline', 'prediction_pipeline'},
        )

    @staticmethod
    def _extract_training_loop_max_rounds(text: str) -> int | None:
        patterns = [
            r'最多\s*(\d+)\s*轮',
            r'max\s*rounds?\s*[=:]?\s*(\d+)',
            r'循环训练\s*(\d+)\s*轮',
            r'循环\s*(\d+)\s*轮',
            r'循环训\s*(\d+)\s*轮',
            r'循环跑\s*(\d+)\s*轮',
            r'自动复训\s*(\d+)\s*轮',
            r'(?:跑|训)\s*(\d+)\s*轮',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                value = int(match.group(1))
                if value > 0:
                    return value
        return None

    @staticmethod
    def _extract_training_loop_managed_level(text: str) -> str:
        normalized = text.lower()
        if any(token in text for token in ('全托管', '完全放权', '完全自动')) or 'full auto' in normalized:
            return 'full_auto'
        if any(token in text for token in ('审阅模式', '每轮审阅', '每轮都停', '每轮都给我看')) or 'review mode' in normalized:
            return 'review'
        if any(token in text for token in ('保守自动', '小改自动', '大改再问')) or 'conservative' in normalized:
            return 'conservative_auto'
        return ''

    @staticmethod
    def _extract_training_loop_target_metric(text: str) -> tuple[str, float | None]:
        metric_patterns = [
            ('map50', r'(?:mAP50|map50)[^0-9]{0,8}(?:到|>=|>|达到|目标(?:是|为)?)\s*([0-9]*\.?[0-9]+)'),
            ('map', r'(?:mAP50-95|map50-95)[^0-9]{0,8}(?:到|>=|>|达到|目标(?:是|为)?)\s*([0-9]*\.?[0-9]+)'),
            ('precision', r'(?:precision|精确率)[^0-9]{0,8}(?:到|>=|>|达到|目标(?:是|为)?)\s*([0-9]*\.?[0-9]+)'),
            ('recall', r'(?:recall|召回)[^0-9]{0,8}(?:到|>=|>|达到|目标(?:是|为)?)\s*([0-9]*\.?[0-9]+)'),
        ]
        for metric_name, pattern in metric_patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                return metric_name, float(match.group(1))
        return '', None

    @staticmethod
    def _extract_training_loop_name(text: str) -> str:
        for pattern in (
            r'(?:环训练|循环训练|自动复训)(?:名称|名字)?(?:叫|命名为)\s*([A-Za-z0-9_.-]+)',
            r'loop(?:\s*name)?\s*[=:]\s*([A-Za-z0-9_.-]+)',
        ):
            match = re.search(pattern, text, flags=re.I)
            if match:
                return match.group(1)
        return ''

    @staticmethod
    def _extract_training_loop_allowed_tuning_params(text: str) -> list[str]:
        if not any(token in text for token in ('允许', '只允许', '自动调', '自动调整', '调参')) and 'allow' not in text.lower():
            return []
        mapping = [
            ('lr0', ('lr0', '学习率')),
            ('batch', ('batch', '批大小')),
            ('imgsz', ('imgsz', '图像尺寸', '输入尺寸')),
            ('epochs', ('epochs', '轮数')),
            ('optimizer', ('optimizer', '优化器')),
        ]
        result: list[str] = []
        for key, tokens in mapping:
            if any(token in text or token in text.lower() for token in tokens):
                result.append(key)
        return result

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
        custom_script = intent_parsing.extract_custom_training_script_from_text(user_text)
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
        explicit_classes_txt = str(planned_training_args.get('classes_txt') or next_tool_args.get('classes_txt') or '').strip()
        if explicit_classes_txt:
            reasoning_parts.append(f'当前已显式指定类名文件 {explicit_classes_txt}。')
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
            'advanced_details_requested': self._wants_training_advanced_details(user_text),
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
            'prepare_then_loop': '先准备再进入循环训练',
            'direct_train': '直接训练',
            'direct_loop': '直接启动循环训练',
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
        classes_txt = str(args.get('classes_txt') or '').strip()
        if classes_txt:
            lines.append(f'- 类名来源文件: {classes_txt}')
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

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item.strip())
                    continue
                if isinstance(item, dict):
                    text = str(item.get('text') or item.get('content') or '').strip()
                    if text:
                        parts.append(text)
            return '\n'.join(parts).strip()
        return str(content or '').strip()

    @staticmethod
    def _human_training_step_name(tool_name: str) -> str:
        normalized = str(tool_name or '').strip()
        mapping = {
            'prepare_dataset_for_training': '先准备数据集',
            'start_training': '启动训练',
            'start_training_loop': '启动循环训练',
            'training_preflight': '先做训练预检',
        }
        return mapping.get(normalized, normalized)

    def _training_plan_user_facts(self, draft: dict[str, Any], *, pending: bool) -> dict[str, Any]:
        execution_mode_raw = str(draft.get('execution_mode') or '').strip().lower()
        next_step_tool = str(draft.get('next_step_tool') or '').strip()
        loop_like = 'loop' in execution_mode_raw or next_step_tool == 'start_training_loop'
        args_source = draft.get('planned_loop_args') if loop_like else draft.get('planned_training_args')
        args = dict(args_source or draft.get('planned_training_args') or {})
        next_args = dict(draft.get('next_step_args') or {})
        execution_mode_map = {
            'prepare_then_train': '先准备再训练',
            'prepare_then_loop': '先准备再进入循环训练',
            'direct_train': '直接训练',
            'direct_loop': '直接启动循环训练',
            'prepare_only': '只做准备，暂不启动训练',
            'discussion_only': '先讨论方案，暂不执行',
            'blocked': '当前存在阻塞，先解决问题',
        }
        execution_backend_map = {
            'standard_yolo': '标准 YOLO 训练',
            'custom_script': '自定义训练脚本',
            'custom_trainer': '自定义 Trainer',
        }
        return {
            'pending_confirmation': bool(pending),
            'dataset_path': str(draft.get('dataset_path') or '').strip(),
            'current_judgment': str(draft.get('data_summary') or '').strip(),
            'plan_reason': str(draft.get('reasoning_summary') or '').strip(),
            'execution_mode': execution_mode_map.get(execution_mode_raw, execution_mode_raw),
            'execution_backend': execution_backend_map.get(str(draft.get('execution_backend') or ''), str(draft.get('execution_backend') or '').strip()),
            'training_environment': str(draft.get('training_environment') or '').strip(),
            'model': str(args.get('model') or '').strip(),
            'data_yaml': str(args.get('data_yaml') or '').strip(),
            'classes_txt': str(args.get('classes_txt') or next_args.get('classes_txt') or '').strip(),
            'project': str(args.get('project') or '').strip(),
            'name': str(args.get('name') or '').strip(),
            'epochs': args.get('epochs'),
            'device': str(args.get('device') or '').strip(),
            'loop_requested': loop_like,
            'managed_level': str(args.get('managed_level') or '').strip(),
            'max_rounds': args.get('max_rounds'),
            'next_step': self._human_training_step_name(next_step_tool),
            'next_step_tool': next_step_tool,
            'blockers': [str(item).strip() for item in (draft.get('blockers') or []) if str(item).strip()],
            'warnings': [str(item).strip() for item in (draft.get('warnings') or []) if str(item).strip()],
        }

    def _training_plan_render_error(self, draft: dict[str, Any], *, pending: bool, error: Exception | None = None) -> str:
        facts = self._training_plan_user_facts(draft, pending=pending)
        summary_bits: list[str] = []
        if facts.get('dataset_path'):
            summary_bits.append(f"数据集：{facts['dataset_path']}")
        if facts.get('model'):
            summary_bits.append(f"模型：{facts['model']}")
        if facts.get('classes_txt'):
            summary_bits.append(f"类名文件：{facts['classes_txt']}")
        if facts.get('next_step'):
            summary_bits.append(f"下一步：{facts['next_step']}")
        prefix = '模型这次没有成功生成计划说明。'
        if error:
            prefix = f'{prefix} 我不会再用固定模板冒充模型输出。'
        if summary_bits:
            return f"{prefix} 当前已确认的计划事实：{'；'.join(summary_bits)}。请稍后重试。"
        return f'{prefix} 请稍后重试。'

    async def _render_training_plan_message(self, draft: dict[str, Any], *, pending: bool) -> str:
        if not draft:
            return ''
        if self.planner_llm is None:
            return self._render_training_plan_draft(draft, pending=pending)

        facts = self._training_plan_user_facts(draft, pending=pending)
        messages = [
            SystemMessage(
                content=(
                    '你是 YoloStudio Training Agent 的计划说明器。'
                    '请基于已验证事实，用自然中文向用户说明当前训练计划。'
                    '不要输出工具名、字段名、JSON、命令、payload、函数名，'
                    '也不要使用“训练计划草案：”“原因和说明”“关键风险提示”这类固定模板标题。'
                    '像同一个 Agent 在继续对话一样说明，不要每次都套相同句式。'
                    '如果是循环训练，请明确说“循环训练”，不要混成普通训练。'
                    '优先用 2 到 4 句自然中文：先说当前结论，再解释原因，最后说明下一步。'
                    '如果 pending_confirmation=true，请用一句自然中文说明“如果你同意，我就按这个计划执行”。'
                    '不要补充未验证事实。'
                )
            ),
            HumanMessage(
                content=(
                    '请根据以下已验证事实，直接给用户一段自然中文说明：\n'
                    f'{json.dumps(facts, ensure_ascii=False, indent=2)}'
                )
            ),
        ]
        text = await self._invoke_renderer_text(
            messages=messages,
            failure_event='planner_render_failed',
            failure_payload={
                'dataset_path': facts.get('dataset_path', ''),
                'next_step': facts.get('next_step', ''),
            },
        )
        if text:
            return text
        return self._training_plan_render_error(draft, pending=pending)

    async def _build_confirmation_message(self, tool_call: dict[str, Any]) -> str:
        args = tool_call.get('args', {})
        tool_name = str(tool_call.get('name') or '')
        plan_draft = self.session_state.active_training.training_plan_draft or {}
        if plan_draft and str(plan_draft.get('next_step_tool') or '').strip() == tool_name:
            return await self._render_training_plan_message(plan_draft, pending=True)
        return await self._render_confirmation_message({'name': tool_name, 'args': args})

    def _confirmation_user_facts(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        args = dict(tool_call.get('args') or {})
        tool_name = str(tool_call.get('name') or '').strip()
        ds = self.session_state.active_dataset
        tr = self.session_state.active_training
        facts: dict[str, Any] = {
            'tool_name': tool_name,
            'tool_action': self._human_training_step_name(tool_name),
            'confirmation_mode': self._confirmation_mode(),
            'dataset_path': str(args.get('dataset_path') or ds.dataset_root or ds.img_dir or '').strip(),
            'data_yaml': str(args.get('data_yaml') or tr.data_yaml or ds.data_yaml or '').strip(),
            'model': str(args.get('model') or tr.model or '').strip(),
            'classes_txt': str(args.get('classes_txt') or '').strip(),
            'force_split': bool(args.get('force_split')),
            'device': str(args.get('device') or tr.device or '').strip(),
            'training_environment': str(args.get('training_environment') or tr.training_environment or '').strip(),
            'project': str(args.get('project') or tr.project or '').strip(),
            'run_name': str(args.get('name') or tr.run_name or '').strip(),
            'managed_level': str(args.get('managed_level') or '').strip(),
            'max_rounds': args.get('max_rounds'),
        }
        readiness = ds.last_readiness or {}
        summary = str(readiness.get('summary') or '').strip()
        if summary:
            facts['dataset_summary'] = summary
        readiness_overview = dict(readiness.get('readiness_overview') or {})
        if readiness_overview:
            facts['dataset_readiness'] = {
                'ready': readiness_overview.get('ready'),
                'preparable': readiness_overview.get('preparable'),
                'primary_blocker_type': readiness_overview.get('primary_blocker_type'),
                'blocker_codes': list(readiness_overview.get('blocker_codes') or [])[:4],
                'risk_level': readiness_overview.get('risk_level'),
                'warning_count': readiness_overview.get('warning_count'),
                'blocker_count': readiness_overview.get('blocker_count'),
                'needs_split': readiness_overview.get('needs_split'),
                'needs_data_yaml': readiness_overview.get('needs_data_yaml'),
            }
        blockers = [str(item).strip() for item in (readiness.get('blockers') or []) if str(item).strip()]
        if blockers:
            facts['dataset_blockers'] = blockers[:4]
        warnings = [str(item).strip() for item in (readiness.get('warnings') or []) if str(item).strip()]
        if warnings:
            facts['dataset_warnings'] = warnings[:4]
        action_candidates = self._compact_action_candidates(readiness.get('action_candidates'))
        if action_candidates:
            facts['action_candidates'] = action_candidates
        return {
            key: value
            for key, value in facts.items()
            if value is not None and value != '' and value != [] and value != {}
        }

    def _confirmation_render_error(self, tool_call: dict[str, Any], error: Exception | None = None) -> str:
        if error:
            self.memory.append_event(
                self.session_state.session_id,
                'confirmation_render_failed',
                {'tool': str(tool_call.get('name') or ''), 'error': str(error)},
            )
        return self._build_confirmation_prompt(tool_call)

    async def _render_confirmation_message(self, tool_call: dict[str, Any]) -> str:
        if self.planner_llm is None:
            return self._build_confirmation_prompt(tool_call)
        facts = self._confirmation_user_facts(tool_call)
        messages = [
            SystemMessage(
                content=(
                    '你是 YoloStudio Agent 的确认说明器。'
                    '请基于已验证事实，用自然中文说明即将执行的动作、原因和关键风险。'
                    '不要输出工具名、字段名、JSON、命令或 payload。'
                    '不要使用“原因和说明”“关键风险提示”这类固定小标题，也不要每次都复用同一套句式。'
                    '如果这是循环训练相关动作，要明确说“循环训练”，不要混成普通训练。'
                    '最后用一句自然中文询问用户是否继续，不要把确认限制写成 y/n。'
                    '不要补充未验证事实。'
                )
            ),
            HumanMessage(
                content=(
                    '请根据以下已验证事实，直接给用户一段自然中文确认说明：\n'
                    f'{json.dumps(facts, ensure_ascii=False, indent=2)}'
                )
            ),
        ]
        text = await self._invoke_renderer_text(
            messages=messages,
            failure_event='confirmation_render_failed',
            failure_payload={'tool': str(tool_call.get('name') or '')},
        )
        if text:
            return text
        return self._confirmation_render_error(tool_call)

    def _tool_result_user_facts(self, tool_name: str, parsed: dict[str, Any]) -> dict[str, Any]:
        facts: dict[str, Any] = {
            'tool_name': tool_name,
            'ok': bool(parsed.get('ok')),
            'summary': str(parsed.get('summary') or parsed.get('message') or parsed.get('error') or '').strip(),
            'error': str(parsed.get('error') or '').strip(),
        }
        for overview_key, overview_value in self._structured_overview_payloads(parsed).items():
            facts[overview_key] = overview_value
        action_candidates = self._compact_action_candidates(parsed.get('action_candidates'))
        if action_candidates:
            facts['action_candidates'] = action_candidates
        if tool_name == 'start_training' and parsed.get('ok'):
            facts['can_check_progress'] = True
            facts['can_stop_run'] = True
        for key in (
            'data_yaml',
            'output_dir',
            'save_dir',
            'project',
            'name',
            'device',
            'pid',
            'log_file',
            'train_count',
            'val_count',
            'resolved_train_path',
            'resolved_val_path',
            'forced',
            'return_code',
        ):
            value = parsed.get(key)
            if value is None or value == '':
                continue
            facts[key] = value
        if tool_name in {
            'start_training_loop',
            'list_training_loops',
            'check_training_loop_status',
            'inspect_training_loop',
            'pause_training_loop',
            'resume_training_loop',
            'stop_training_loop',
        }:
            for key in (
                'loop_id',
                'loop_name',
                'status',
                'managed_level',
                'current_round_index',
                'completed_rounds',
                'max_rounds',
                'best_round_index',
                'best_target_metric',
                'failure_count',
                'no_improvement_streak',
                'termination_reason',
                'termination_detail',
                'active_loop_id',
            ):
                value = parsed.get(key)
                if value is None or value == '':
                    continue
                facts[key] = value
            boundaries = dict(parsed.get('boundaries') or {})
            if boundaries:
                facts['target_metric'] = boundaries.get('target_metric')
                if boundaries.get('target_metric_value') is not None:
                    facts['target_metric_value'] = boundaries.get('target_metric_value')
            next_round_plan = dict(parsed.get('next_round_plan') or {})
            if next_round_plan:
                facts['next_round_plan'] = {
                    'round_index': next_round_plan.get('round_index'),
                    'reason': next_round_plan.get('reason'),
                    'decision_type': next_round_plan.get('decision_type'),
                    'change_set': [
                        {
                            'field': item.get('field'),
                            'old': item.get('old'),
                            'new': item.get('new'),
                        }
                        for item in list(next_round_plan.get('change_set') or [])[:4]
                        if isinstance(item, dict)
                    ],
                    'experience_context': next_round_plan.get('experience_context'),
                }
            latest_round_card = dict(parsed.get('latest_round_card') or {})
            if latest_round_card:
                facts['latest_round_card'] = {
                    'round_index': latest_round_card.get('round_index'),
                    'status': latest_round_card.get('status'),
                    'summary': latest_round_card.get('summary'),
                    'metrics': latest_round_card.get('metrics') or {},
                    'changed_params': list(latest_round_card.get('changed_params') or [])[:4],
                    'knowledge_gate': latest_round_card.get('knowledge_gate'),
                    'decision': latest_round_card.get('decision'),
                    'next_plan': latest_round_card.get('next_plan'),
                    'why': latest_round_card.get('why'),
                    'recommendation': latest_round_card.get('recommendation'),
                    'round_review': latest_round_card.get('round_review'),
                    'round_memory': latest_round_card.get('round_memory'),
                    'planner_output': latest_round_card.get('planner_output'),
                    'experience_context': latest_round_card.get('experience_context'),
                }
            round_cards = list(parsed.get('round_cards') or [])
            if round_cards:
                facts['recent_round_cards'] = [
                    {
                        'round_index': item.get('round_index'),
                        'status': item.get('status'),
                        'summary': item.get('summary'),
                        'knowledge_gate': item.get('knowledge_gate'),
                        'decision': item.get('decision'),
                        'round_review': item.get('round_review'),
                        'round_memory': item.get('round_memory'),
                        'planner_output': item.get('planner_output'),
                    }
                    for item in round_cards[-3:]
                    if isinstance(item, dict)
                ]
            loops = list(parsed.get('loops') or [])
            if loops:
                facts['loop_count'] = len(loops)
                facts['recent_loops'] = [
                    {
                        'loop_id': item.get('loop_id'),
                        'loop_name': item.get('loop_name'),
                        'status': item.get('status'),
                        'managed_level': item.get('managed_level'),
                        'current_round_index': item.get('current_round_index'),
                        'max_rounds': item.get('max_rounds'),
                        'best_round_index': item.get('best_round_index'),
                        'best_target_metric': item.get('best_target_metric'),
                    }
                    for item in loops[:5]
                    if isinstance(item, dict)
                ]
        for key in ('knowledge_gate_status', 'latest_round_review', 'latest_round_memory', 'latest_planner_output'):
            value = parsed.get(key)
            if isinstance(value, dict) and value:
                facts[key] = value
            final_summary = dict(parsed.get('final_summary') or {})
            if final_summary:
                facts['final_summary'] = {
                    'status': final_summary.get('status'),
                    'best_round_index': final_summary.get('best_round_index'),
                    'best_target_metric_name': final_summary.get('best_target_metric_name'),
                    'best_target_metric': final_summary.get('best_target_metric'),
                    'stop_reason': final_summary.get('stop_reason'),
                    'termination_detail': final_summary.get('termination_detail'),
                    'round_count': final_summary.get('round_count'),
                    'last_round_review': final_summary.get('last_round_review'),
                    'last_round_memory': final_summary.get('last_round_memory'),
                    'last_planner_output': final_summary.get('last_planner_output'),
                    'experience_timeline': list(final_summary.get('experience_timeline') or [])[-3:],
                }
            if 'action_candidates' not in facts:
                next_actions = list(parsed.get('next_actions') or [])
                if next_actions:
                    facts['next_actions'] = next_actions[:4]
        if tool_name in {'remote_prediction_pipeline', 'remote_training_pipeline'}:
            for key in (
                'remote_source_path',
                'remote_model_path',
                'remote_dataset_path',
                'remote_output_dir',
                'remote_result_path',
                'local_result_root',
                'source_kind',
                'predict_tool_name',
                'final_run_state',
                'wait_for_completion',
                'download_after_completion',
            ):
                value = parsed.get(key)
                if value is None or value == '':
                    continue
                facts[key] = value
        return facts

    @staticmethod
    def _compact_action_candidates(action_candidates: Any) -> list[dict[str, Any]]:
        if not isinstance(action_candidates, list):
            return []
        compacted: list[dict[str, Any]] = []
        for item in action_candidates[:4]:
            if not isinstance(item, dict):
                continue
            compact = {
                'action': item.get('action'),
                'tool': item.get('tool'),
                'description': item.get('description'),
            }
            compacted.append({key: value for key, value in compact.items() if value not in (None, '', [], {})})
        return [item for item in compacted if item]

    @classmethod
    def _structured_overview_payloads(cls, parsed: dict[str, Any]) -> dict[str, Any]:
        payloads: dict[str, Any] = {}
        for key, value in parsed.items():
            if not key.endswith('_overview'):
                continue
            if isinstance(value, dict) and value:
                payloads[key] = dict(value)
            elif isinstance(value, list) and value:
                payloads[key] = list(value)
        for key in ('matched_rule_overview', 'playbook_overview'):
            value = parsed.get(key)
            if isinstance(value, list) and value:
                payloads[key] = list(value)[:4]
        return payloads

    @staticmethod
    def _remote_pipeline_applied_results(tool_name: str, parsed: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        normalized = canonical_tool_name(tool_name)
        if normalized == 'remote_prediction_pipeline':
            predict_tool_name = str(parsed.get('predict_tool_name') or 'predict_images').strip() or 'predict_images'
            ordered = [
                ('upload_assets_to_remote', dict(parsed.get('upload') or {})),
                (predict_tool_name, dict(parsed.get('predict') or {})),
                ('download_assets_from_remote', dict(parsed.get('download') or {})),
            ]
            return [(name, payload) for name, payload in ordered if payload]
        if normalized == 'remote_training_pipeline':
            ordered = [
                ('upload_assets_to_remote', dict(parsed.get('upload') or {})),
                ('training_readiness', dict(parsed.get('readiness') or {})),
                ('prepare_dataset_for_training', dict(parsed.get('prepare') or {})),
                ('training_preflight', dict(parsed.get('preflight') or {})),
                ('start_training', dict(parsed.get('start') or {})),
                ('check_training_status', dict(parsed.get('final_status') or {})),
                ('summarize_training_run', dict(parsed.get('final_summary') or {})),
                ('download_assets_from_remote', dict(parsed.get('download') or {})),
            ]
            return [(name, payload) for name, payload in ordered if payload]
        return []

    def _fallback_tool_result_text(self, tool_name: str, parsed: dict[str, Any]) -> str:
        if self._structured_overview_payloads(parsed) or parsed.get('action_candidates'):
            structured_text = stringify_tool_result_facts(parsed).strip()
            if structured_text:
                return structured_text
        return (
            self._build_grounded_tool_reply([(tool_name, parsed)])
            or str(parsed.get('summary') or parsed.get('message') or parsed.get('error') or '').strip()
            or ('操作执行成功' if parsed.get('ok') else '操作执行失败')
        )

    def _fallback_multi_tool_result_message(
        self,
        applied_results: list[tuple[str, dict[str, Any]]],
        *,
        extra_notes: list[str] | None = None,
    ) -> str:
        sections: list[str] = []
        for tool_name, parsed in applied_results:
            normalized_name = str(canonical_tool_name(tool_name) or '').strip()
            if not normalized_name:
                continue
            normalized_parsed = parsed if isinstance(parsed, dict) else {'ok': False, 'summary': str(parsed or '').strip()}
            sections.append(self._fallback_tool_result_text(normalized_name, normalized_parsed))
        for note in extra_notes or []:
            text = str(note or '').strip()
            if text:
                sections.append(text)
        return self._merge_grounded_sections(sections)

    async def _render_multi_tool_result_message(
        self,
        applied_results: list[tuple[str, dict[str, Any]]],
        *,
        objective: str = '',
        extra_notes: list[str] | None = None,
    ) -> str:
        normalized_results: list[tuple[str, dict[str, Any]]] = []
        for tool_name, parsed in applied_results:
            normalized_name = str(canonical_tool_name(tool_name) or '').strip()
            if not normalized_name:
                continue
            normalized_parsed = parsed if isinstance(parsed, dict) else {'ok': False, 'summary': str(parsed or '').strip()}
            normalized_results.append((normalized_name, normalized_parsed))

        cleaned_notes = [str(note).strip() for note in (extra_notes or []) if str(note).strip()]
        if not normalized_results:
            return self._merge_grounded_sections(cleaned_notes)
        if len(normalized_results) == 1 and not cleaned_notes:
            tool_name, parsed = normalized_results[0]
            return await self._render_tool_result_message(tool_name, parsed)
        if self.planner_llm is None:
            return self._fallback_multi_tool_result_message(normalized_results, extra_notes=cleaned_notes)

        facts = {
            'objective': str(objective or '').strip(),
            'results': [self._tool_result_user_facts(tool_name, parsed) for tool_name, parsed in normalized_results],
            'extra_notes': cleaned_notes[:4],
        }
        messages = [
            SystemMessage(
                content=(
                    '你是 YoloStudio Agent 的组合结果说明器。'
                    '请基于多条已验证的工具结果，用自然中文向用户给出一个连贯结论。'
                    '优先回答最终结果，再补关键事实和下一步。'
                    '不要输出工具名、字段名、JSON、命令或 payload。'
                    '不要补充未验证事实。'
                )
            ),
            HumanMessage(
                content=(
                    '请根据以下已验证事实，直接给用户一段自然中文说明：\n'
                    f'{json.dumps(facts, ensure_ascii=False, indent=2)}'
                )
            ),
        ]
        text = await self._invoke_renderer_text(
            messages=messages,
            failure_event='multi_tool_result_render_failed',
            failure_payload={
                'tools': [tool_name for tool_name, _ in normalized_results],
                'objective': str(objective or '').strip(),
            },
        )
        if text:
            return text
        return self._fallback_multi_tool_result_message(normalized_results, extra_notes=cleaned_notes)

    def _tool_result_render_error(self, tool_name: str, parsed: dict[str, Any], error: Exception | None = None) -> str:
        summary = str(parsed.get('summary') or parsed.get('message') or parsed.get('error') or '').strip()
        if error and summary:
            return f'模型这次没有成功整理执行结果。我先给你真实摘要：{summary}'
        if error:
            return '模型这次没有成功整理执行结果，但工具已经执行完成。请稍后重试。'
        return summary or ('操作执行成功' if parsed.get('ok') else '操作执行失败')

    async def _render_prepare_followup_message(self, prepare_parsed: dict[str, Any], preflight: dict[str, Any]) -> str:
        if self.planner_llm is None:
            return await self._render_multi_tool_result_message(
                [
                    ('prepare_dataset_for_training', prepare_parsed),
                    ('training_preflight', preflight),
                ],
                objective='数据准备后的训练衔接说明',
            ) or preflight.get('summary') or prepare_parsed.get('summary') or '后续训练预检未通过'

        facts = {
            'prepare_summary': str(prepare_parsed.get('summary') or prepare_parsed.get('message') or '').strip(),
            'data_yaml': str(prepare_parsed.get('data_yaml') or self.session_state.active_dataset.data_yaml or '').strip(),
            'preflight_summary': str(preflight.get('summary') or preflight.get('message') or preflight.get('error') or '').strip(),
            'ready_to_start': bool(preflight.get('ready_to_start')),
            'blockers': [str(item).strip() for item in (preflight.get('blockers') or []) if str(item).strip()][:4],
            'training_environment': str((preflight.get('training_environment') or {}).get('display_name') or (preflight.get('training_environment') or {}).get('name') or '').strip(),
            'device': str((preflight.get('resolved_args') or {}).get('device') or '').strip(),
            'model': str((preflight.get('resolved_args') or {}).get('model') or '').strip(),
        }
        messages = [
            SystemMessage(
                content=(
                    '你是 YoloStudio Agent 的训练衔接说明器。'
                    '请基于已验证事实，用自然中文说明：数据准备后的训练衔接情况。'
                    '如果训练不能继续，直接解释阻塞原因和下一步。'
                    '不要输出工具名、字段名、JSON、命令或 payload。'
                    '不要补充未验证事实。'
                )
            ),
            HumanMessage(
                content=(
                    '请根据以下已验证事实，直接给用户一段自然中文说明：\n'
                    f'{json.dumps({k: v for k, v in facts.items() if v is not None and v != "" and v != [] and v != {}}, ensure_ascii=False, indent=2)}'
                )
            ),
        ]
        text = await self._invoke_renderer_text(
            messages=messages,
            failure_event='prepare_followup_render_failed',
            failure_payload={'ready_to_start': bool(preflight.get('ready_to_start'))},
        )
        if text:
            return text
        return await self._render_multi_tool_result_message(
            [
                ('prepare_dataset_for_training', prepare_parsed),
                ('training_preflight', preflight),
            ],
            objective='数据准备后的训练衔接说明',
        ) or preflight.get('summary') or prepare_parsed.get('summary') or '后续训练预检未通过'

    async def _render_tool_result_message(self, tool_name: str, parsed: dict[str, Any]) -> str:
        remote_pipeline_results = self._remote_pipeline_applied_results(tool_name, parsed)
        if remote_pipeline_results:
            objective = '远端训练闭环执行结果' if canonical_tool_name(tool_name) == 'remote_training_pipeline' else '远端预测闭环执行结果'
            extra_notes: list[str] = []
            remote_result_path = str(parsed.get('remote_result_path') or '').strip()
            local_result_root = str(parsed.get('local_result_root') or '').strip()
            final_run_state = str(parsed.get('final_run_state') or '').strip()
            if remote_result_path:
                extra_notes.append(f'远端结果目录: {remote_result_path}')
            if local_result_root:
                extra_notes.append(f'本机回传目录: {local_result_root}')
            if final_run_state:
                extra_notes.append(f'最终运行状态: {final_run_state}')
            return await self._render_multi_tool_result_message(
                remote_pipeline_results,
                objective=objective,
                extra_notes=extra_notes or None,
            )
        if self.planner_llm is None:
            return self._fallback_tool_result_text(tool_name, parsed)

        facts = self._tool_result_user_facts(tool_name, parsed)
        messages = [
            SystemMessage(
                content=(
                    '你是 YoloStudio Agent 的结果说明器。'
                    '请基于已验证的工具执行结果，用自然中文向用户说明本次执行结果。'
                    '不要输出工具名、字段名、JSON、命令或 payload。'
                    '如果成功，先说结果，再补关键事实；如果失败，直接解释失败原因和下一步。'
                    '不要补充未验证事实。'
                )
            ),
            HumanMessage(
                content=(
                    '请根据以下已验证事实，直接给用户一段自然中文说明：\n'
                    f'{json.dumps(facts, ensure_ascii=False, indent=2)}'
                )
            ),
        ]
        text = await self._invoke_renderer_text(
            messages=messages,
            failure_event='tool_result_render_failed',
            failure_payload={'tool': tool_name, 'ok': bool(parsed.get('ok'))},
        )
        if text:
            return text
        return self._tool_result_render_error(tool_name, parsed)

    async def _try_handle_training_plan_bootstrap(
        self,
        user_text: str,
        thread_id: str,
        *,
        explicit_run_ids: list[str],
        normalized: str,
        latest_dataset_path: str,
        requested_execute: bool,
        wants_repeat_prepare: bool,
        wants_retry_last_plan: bool,
        wants_resume_recent_training: bool,
        wants_analysis_only: bool,
    ) -> dict[str, Any] | None:
        prepare_only = await self._try_handle_prepare_only_intent(user_text, thread_id)
        if prepare_only is not None:
            return prepare_only
        if self.session_state.active_training.running and explicit_run_ids and wants_resume_recent_training:
            reply = '当前已有活动训练；如果你想恢复或切换到另一个历史 run，请先停止当前训练，再明确要恢复的 run。'
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}
        running_hot_update_intent = (
            self.session_state.active_training.running
            and not latest_dataset_path
            and not explicit_run_ids
            and not any(token in user_text for token in ('预测', '推理', '识别', '视频', '图片'))
            and not any(token in user_text for token in ('第几轮', '训练到哪了', '训练到第几轮', '跑到第几轮', '训练状态', '训练进度', '当前进度', '还在训练吗', '还在跑吗', '现在状态'))
            and (
                any(token in normalized for token in ('batch', 'device', 'epochs', 'optimizer', 'freeze', 'lr0', 'resume', 'imgsz', 'fraction', 'classes', 'single_cls'))
                or any(token in user_text for token in ('轮数', '轮', '优化器', '冻结', '学习率', '环境', 'GPU', '显卡'))
            )
        )
        if running_hot_update_intent:
            reply = (
                '当前训练还在运行，不能直接热更新 batch、轮数、优化器或设备等核心参数。'
                '如果要改参数，请先停止当前训练，再生成新的训练计划。'
            )
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}
        if requested_execute and self.session_state.active_training.running:
            reply = '当前训练已经在运行；如果要新开训练，请先停止当前训练，或明确给出新的数据集和模型。'
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}
        readiness = self.session_state.active_dataset.last_readiness or {}
        data_yaml = str(self.session_state.active_dataset.data_yaml or '').strip()
        if wants_repeat_prepare and readiness.get('ready') and data_yaml:
            reply = f'当前数据集已经准备完成：{data_yaml}；不需要重复 prepare。你可以直接继续训练或重新规划。'
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}
        if not (wants_retry_last_plan or wants_resume_recent_training):
            return None
        if wants_analysis_only:
            if (
                self.session_state.active_training.training_run_summary
                or self.session_state.active_training.last_summary
                or self.session_state.active_training.last_status
            ):
                return await self._complete_training_outcome_analysis_reply()
            return None
        tr = self.session_state.active_training
        base_args = dict((tr.last_start_result or {}).get('resolved_args') or {})
        if not str(base_args.get('model') or '').strip():
            base_args['model'] = str(tr.model or '').strip()
        if not str(base_args.get('data_yaml') or '').strip():
            base_args['data_yaml'] = str(tr.data_yaml or self.session_state.active_dataset.data_yaml or '').strip()
        if not str(base_args.get('training_environment') or '').strip() and str(tr.training_environment or '').strip():
            base_args['training_environment'] = str(tr.training_environment).strip()
        dataset_path = str(self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir or '').strip()
        if not dataset_path and str(base_args.get('data_yaml') or '').strip():
            dataset_path = str(self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir or '').strip()
        if not str(base_args.get('model') or '').strip() or not str(base_args.get('data_yaml') or '').strip():
            reply = '当前缺少足够的历史训练参数，暂时不能直接恢复这次训练计划；请先明确数据集和模型。'
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}
        run_state = str((tr.training_run_summary or {}).get('run_state') or (tr.last_summary or {}).get('run_state') or (tr.last_status or {}).get('run_state') or '').strip().lower()
        if wants_resume_recent_training:
            if run_state != 'stopped':
                reply = '当前只有已停止的训练才适合按最近状态继续；失败或已完成的训练更适合按原计划重试或重新规划。'
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}
            base_args['resume'] = True
        else:
            base_args['resume'] = False
        if dataset_path and not readiness:
            readiness = await self.direct_tool('training_readiness', img_dir=dataset_path)
        if not (tr.last_environment_probe or {}).get('environments'):
            await self.direct_tool('list_training_environments')
        preflight = await self.direct_tool(
            'training_preflight',
            model=str(base_args.get('model') or ''),
            data_yaml=str(base_args.get('data_yaml') or ''),
            epochs=int(base_args.get('epochs', 100)),
            device=str(base_args.get('device', 'auto') or 'auto'),
            training_environment=str(base_args.get('training_environment') or ''),
            project=str(base_args.get('project') or ''),
            name=str(base_args.get('name') or ''),
            batch=base_args.get('batch'),
            imgsz=base_args.get('imgsz'),
            fraction=base_args.get('fraction'),
            classes=base_args.get('classes'),
            single_cls=base_args.get('single_cls'),
            optimizer=str(base_args.get('optimizer', '') or ''),
            freeze=base_args.get('freeze'),
            resume=base_args.get('resume'),
            lr0=base_args.get('lr0'),
            patience=base_args.get('patience'),
            workers=base_args.get('workers'),
            amp=base_args.get('amp'),
        )
        next_args = {
            'model': str((preflight.get('resolved_args') or {}).get('model') or base_args.get('model') or ''),
            'data_yaml': str((preflight.get('resolved_args') or {}).get('data_yaml') or base_args.get('data_yaml') or ''),
            'epochs': int((preflight.get('resolved_args') or {}).get('epochs') or base_args.get('epochs', 100)),
            'device': str((preflight.get('resolved_args') or {}).get('device') or base_args.get('device') or 'auto'),
            'training_environment': str((preflight.get('resolved_args') or {}).get('training_environment') or base_args.get('training_environment') or ''),
            'project': str((preflight.get('resolved_args') or {}).get('project') or base_args.get('project') or ''),
            'name': str((preflight.get('resolved_args') or {}).get('name') or base_args.get('name') or ''),
            'batch': (preflight.get('resolved_args') or {}).get('batch', base_args.get('batch')),
            'imgsz': (preflight.get('resolved_args') or {}).get('imgsz', base_args.get('imgsz')),
            'fraction': (preflight.get('resolved_args') or {}).get('fraction', base_args.get('fraction')),
            'classes': (preflight.get('resolved_args') or {}).get('classes', base_args.get('classes')),
            'single_cls': (preflight.get('resolved_args') or {}).get('single_cls', base_args.get('single_cls')),
            'optimizer': str((preflight.get('resolved_args') or {}).get('optimizer') or base_args.get('optimizer') or ''),
            'freeze': (preflight.get('resolved_args') or {}).get('freeze', base_args.get('freeze')),
            'resume': (preflight.get('resolved_args') or {}).get('resume', base_args.get('resume')),
            'lr0': (preflight.get('resolved_args') or {}).get('lr0', base_args.get('lr0')),
            'patience': (preflight.get('resolved_args') or {}).get('patience', base_args.get('patience')),
            'workers': (preflight.get('resolved_args') or {}).get('workers', base_args.get('workers')),
            'amp': (preflight.get('resolved_args') or {}).get('amp', base_args.get('amp')),
        }
        rebuilt_draft = self._build_training_plan_draft(
            user_text=user_text,
            dataset_path=dataset_path,
            readiness=readiness,
            preflight=preflight,
            next_tool_name='start_training' if preflight.get('ready_to_start') else '',
            next_tool_args=next_args if preflight.get('ready_to_start') else {},
            planned_training_args=next_args,
        )
        self._save_training_plan_draft(rebuilt_draft)
        if preflight.get('ready_to_start'):
            pending = {'name': 'start_training', 'args': next_args, 'id': None, 'synthetic': True}
            self._set_pending_confirmation(thread_id, pending)
            return self._needs_confirmation_result(thread_id, pending, await self._render_training_plan_message(rebuilt_draft, pending=True))
        reply = await self._render_training_plan_message(rebuilt_draft, pending=False)
        self._messages.append(AIMessage(content=reply))
        return {'status': 'completed', 'message': reply, 'tool_call': None}

    async def _try_handle_training_plan_dialogue(self, user_text: str, thread_id: str) -> dict[str, Any] | None:
        draft = dict(self.session_state.active_training.training_plan_draft or {})
        pending = self._pending_from_state()
        explicit_run_ids = self._extract_training_run_ids_from_text(user_text)

        normalized = user_text.lower()
        if any(token in user_text for token in ('环训练', '循环训练', '循环训', '循环跑', '自动复训', '自动续训')) or any(
            token in normalized for token in ('training loop', 'loop training', 'auto retrain', 'auto training loop')
        ):
            return None
        clear_fields = self._collect_training_clear_fields(user_text)
        discussion_only_hint = self._is_training_discussion_only(user_text) or any(token in user_text for token in ('不执行', '不要执行', '暂不执行', '先不执行'))
        training_readiness_question = any(
            token in user_text
            for token in (
                '能不能直接训练',
                '能否直接训练',
                '可不可以直接训练',
                '可以直接训练吗',
                '是否可以直接训练',
                '是否能直接训练',
                '还能不能直接训练',
                '可否直接训练',
                '能直接训练吗',
            )
        )
        contradictory_train_intent = not training_readiness_question and any(token in user_text for token in ('不要训练', '先不要训练', '不训练了')) and any(
            token in user_text for token in ('开始训练', '启动训练', '直接训练', '直接开始训练', '开训', '执行')
        )
        requested_execute = (
            any(token in user_text for token in ('执行', '开始吧', '就这样', '确认', '可以开始', '开训', '启动吧', '直接训练', '直接开始训练'))
            or normalized.strip() in {'y', 'yes'}
        ) and not discussion_only_hint and not contradictory_train_intent
        if any(token in user_text for token in ('为什么', '原因', '依据', '怎么看')):
            requested_execute = False
        wants_repeat_prepare = any(token in user_text for token in ('再 prepare 一次', '再准备一次', '重新 prepare 一次', '重新准备一次', '再做一次准备', '重新准备一遍'))
        wants_retry_last_plan = any(token in user_text for token in ('按原计划重试一次', '按原计划重试', '重试刚才那次训练', '重试上次训练', '按刚才的计划再来一次', '按原计划再来一次'))
        wants_resume_recent_training = (
            any(
                token in user_text
                for token in (
                    '从最近状态继续训练',
                    '从最近状态继续',
                    '从最近状态恢复训练',
                    '恢复刚才训练',
                    '接着上次训练',
                    '恢复上次训练',
                    'resume 上次训练',
                    'resume 刚才训练',
                    'resume 另一个 run',
                    'resume run',
                    '继续另一个 run',
                )
            )
            or (
                'resume' in normalized
                and (
                    bool(explicit_run_ids)
                    or any(token in user_text for token in ('上次', '刚才', '最近', '继续', '恢复', '另一个', '历史', 'run'))
                )
            )
        )
        wants_analysis_only = any(token in user_text for token in ('只分析', '只看结果', '不要接着训', '不要继续训', '不要继续训练'))
        latest_dataset_path = self._extract_dataset_path_from_text(user_text)
        all_paths = self._extract_all_paths_from_text(user_text)
        project_path_hint = intent_parsing.extract_project_from_text(user_text)
        custom_script_hint = intent_parsing.extract_custom_training_script_from_text(user_text)
        dataset_candidates = [
            item for item in all_paths
            if not self._looks_like_model_path(item)
            and item != project_path_hint
            and item != custom_script_hint
        ]
        if dataset_candidates and any(token in user_text for token in ('换成', '现在用', '改成', '改用')):
            latest_dataset_path = dataset_candidates[-1]
        if (
            latest_dataset_path
            and project_path_hint
            and latest_dataset_path == project_path_hint
            and not any(token in user_text for token in ('数据', 'dataset', 'img_dir', 'label_dir'))
        ):
            latest_dataset_path = ''
        if (
            latest_dataset_path
            and custom_script_hint
            and latest_dataset_path == custom_script_hint
            and not any(token in user_text for token in ('数据', 'dataset', 'img_dir', 'label_dir'))
        ):
            latest_dataset_path = ''
        dataset_path_revision_requested = bool(latest_dataset_path) and (
            any(token in user_text for token in ('数据', '数据集', 'dataset', 'img_dir', 'label_dir'))
            or any(token in user_text for token in ('现在用', '改用', '换成', '改成'))
            or (
                any(token in user_text for token in ('训练', '开训', '开始训练', '继续训练'))
                and not any(token in user_text for token in ('预测', '推理', '识别', '抽帧', '提帧', '视频', '图片'))
            )
        )
        if contradictory_train_intent:
            reply = '你这句话里同时出现了“不要训练”和“开始训练”；我先按保守方式处理，只保留讨论态，不会直接执行。'
            if draft or pending:
                reply = f'{reply}\n\n{await self._render_training_plan_message(draft or {}, pending=bool(pending))}'
            self._messages.append(AIMessage(content=reply))
            return {
                'status': 'completed' if not pending else 'needs_confirmation',
                'message': reply,
                'tool_call': {'name': pending['name'], 'args': pending.get('args', {})} if pending else None,
                'thread_id': thread_id if pending else None,
            }
        if not draft and not pending:
            return await self._try_handle_training_plan_bootstrap(
                user_text,
                thread_id,
                explicit_run_ids=explicit_run_ids,
                normalized=normalized,
                latest_dataset_path=latest_dataset_path,
                requested_execute=requested_execute,
                wants_repeat_prepare=wants_repeat_prepare,
                wants_retry_last_plan=wants_retry_last_plan,
                wants_resume_recent_training=wants_resume_recent_training,
                wants_analysis_only=wants_analysis_only,
            )

        has_revision = any(
            token in normalized or token in user_text
            for token in (
                'batch', 'imgsz', 'device', 'epochs', '轮', '轮数', '优化器', 'optimizer', '冻结', 'freeze', 'resume',
                'lr0', '学习率', 'patience', '早停', 'workers', '线程数', 'amp', '混合精度',
                '模型', '权重', 'project', '输出目录', 'name', '实验名', '运行名',
                'fraction', '全量数据', '抽样', 'classes', '类别', 'single_cls', '单类别',
                '环境', '为什么', '原因', '依据', '先只做准备', '只做准备', '标准 yolo', '自定义脚本', 'trainer',
                '高级参数', '高级配置', '展开参数', '详细参数',
                '划分', '自动划分', '不划分', '不要划分', '默认比例',
            )
        ) or wants_retry_last_plan or wants_resume_recent_training or bool(intent_parsing.extract_custom_training_script_from_text(user_text)) or dataset_path_revision_requested
        if (
            any(token in user_text for token in ('取消', '算了', '先不做', '不用了', '先别开始训练', '先不要开始训练', '先别开训', '先不要开训', '先别开始', '先不要开始'))
            and not clear_fields
            and not requested_execute
            and not has_revision
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
        if pending and any(token in user_text for token in ('等等', '等一下', '先等等', '先等下', '稍等', '先稍等')):
            return self._needs_confirmation_result(thread_id, pending, await self._build_confirmation_message(pending))
        if (
            pending
            and pending.get('name') == 'prepare_dataset_for_training'
            and any(token in user_text for token in ('data_yaml', 'yaml', '产物路径', '输出路径', '会生成到哪里'))
        ):
            dataset_path = str((pending.get('args') or {}).get('dataset_path') or draft.get('dataset_path') or '').strip()
            expected_yaml = f'{dataset_path.rstrip("/")}/data.yaml' if dataset_path else '准备输出目录中的 data.yaml'
            dataset_label = dataset_path or '<当前数据集>'
            reply = (
                f'如果继续 prepare，我会基于数据集 {dataset_label} 生成可训练产物；'
                f'预期会产出可用的 data_yaml（通常是 {expected_yaml}），完成后我会把真实路径写回状态。'
            )
            self._messages.append(AIMessage(content=reply))
            return self._needs_confirmation_result(thread_id, pending, reply)
        if wants_repeat_prepare:
            readiness = self.session_state.active_dataset.last_readiness or {}
            data_yaml = str(self.session_state.active_dataset.data_yaml or '').strip()
            if readiness.get('ready') and data_yaml:
                if pending:
                    return self._needs_confirmation_result(thread_id, pending, await self._build_confirmation_message(pending))
                reply = f'当前数据集已经准备完成：{data_yaml}；不需要重复 prepare。你可以直接继续训练或重新规划。'
                self._messages.append(AIMessage(content=reply))
                return {'status': 'completed', 'message': reply, 'tool_call': None}
        if (
            any(token in user_text for token in ('先别执行', '先不要执行', '先别启动', '先不要启动', '先讨论', '先看看计划', '先给我计划', '想先看计划', '记一下我想先看计划'))
            and not has_revision
            and not requested_execute
        ):
            if draft:
                return {
                    'status': 'completed' if not pending else 'needs_confirmation',
                    'message': await self._render_training_plan_message(draft, pending=bool(pending)),
                    'tool_call': {'name': pending['name'], 'args': pending.get('args', {})} if pending else None,
                    'thread_id': thread_id if pending else None,
                }
            return None

        if (
            draft
            and not has_revision
            and not requested_execute
            and any(token in user_text for token in ('最开始那套呢', '最开始那个计划呢', '第一版计划呢', '最早那套呢', '最开始那版呢'))
        ):
            reply = (
                '当前只保留最新训练计划草案；最开始那套已经被后续修订覆盖。'
                '如果要回退，请直接说明要恢复的数据集、模型或关键参数。\n\n'
                f"{await self._render_training_plan_message(draft, pending=bool(pending))}"
            )
            self._messages.append(AIMessage(content=reply))
            return {
                'status': 'completed' if not pending else 'needs_confirmation',
                'message': reply,
                'tool_call': {'name': pending['name'], 'args': pending.get('args', {})} if pending else None,
                'thread_id': thread_id if pending else None,
            }

        if (
            draft
            and not has_revision
            and not requested_execute
            and any(token in user_text for token in ('训练计划继续', '继续刚才训练计划', '继续刚才的训练计划', '继续刚才那个训练计划', '刚才训练计划继续'))
        ):
            return {
                'status': 'completed' if not pending else 'needs_confirmation',
                'message': await self._render_training_plan_message(draft, pending=bool(pending)),
                'tool_call': {'name': pending['name'], 'args': pending.get('args', {})} if pending else None,
                'thread_id': thread_id if pending else None,
            }

        if requested_execute and not has_revision:
            if pending:
                return await self.confirm(thread_id, approved=True)
            next_tool_name = str(draft.get('next_step_tool') or '').strip()
            next_tool_args = dict(draft.get('next_step_args') or {})
            if not next_tool_name:
                return {'status': 'completed', 'message': await self._render_training_plan_message(draft, pending=False), 'tool_call': None}
            self._set_pending_confirmation(thread_id, {'name': next_tool_name, 'args': next_tool_args, 'id': None, 'synthetic': True})
            self.memory.save_state(self.session_state)
            return self._needs_confirmation_result(
                thread_id,
                {'name': next_tool_name, 'args': next_tool_args, 'id': None, 'synthetic': True},
                await self._render_training_plan_message(draft, pending=True),
            )

        if not has_revision:
            return None

        revised_draft = dict(draft or {})
        planned_args = dict(revised_draft.get('planned_training_args') or {})
        dataset_path = str(latest_dataset_path or revised_draft.get('dataset_path') or self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir or '').strip()
        readiness = self.session_state.active_dataset.last_readiness or {}
        previous_dataset_path = str(revised_draft.get('dataset_path') or '').strip()
        if latest_dataset_path and dataset_path and dataset_path != previous_dataset_path:
            readiness = await self.direct_tool('training_readiness', img_dir=dataset_path)
            await self.direct_tool('list_training_environments')
            resolved_yaml = str(readiness.get('resolved_data_yaml') or '').strip()
            if resolved_yaml:
                planned_args['data_yaml'] = resolved_yaml
            else:
                planned_args.pop('data_yaml', None)
        requested_data_yaml_hint: str | None = str(planned_args.get('data_yaml') or '').strip()
        if not requested_data_yaml_hint:
            requested_data_yaml_hint = None if latest_dataset_path else str(self.session_state.active_dataset.data_yaml or '').strip()
        requested_args = self._collect_requested_training_args(
            user_text,
            data_yaml=requested_data_yaml_hint,
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
        advanced_requested = self._wants_training_advanced_details(user_text) or bool(revised_draft.get('advanced_details_requested'))
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
            (next_tool_name == 'start_training' or execution_mode in {'direct_train', 'discussion_only', 'blocked'})
            and readiness.get('ready')
            and planned_args.get('model')
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
            prepare_args: dict[str, Any] = {'dataset_path': dataset_path}
            if next_tool_args.get('force_split'):
                prepare_args['force_split'] = next_tool_args.get('force_split')
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
        self._save_training_plan_draft(revised_draft)
        force_confirmation = wants_retry_last_plan or wants_resume_recent_training
        if revised_draft.get('next_step_tool') and (pending or force_confirmation or requested_execute):
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
            return self._needs_confirmation_result(
                thread_id,
                {
                    'name': str(revised_draft.get('next_step_tool')),
                    'args': dict(revised_draft.get('next_step_args') or {}),
                    'id': None,
                    'synthetic': True,
                },
                await self._render_training_plan_message(revised_draft, pending=True),
            )
        self.memory.save_state(self.session_state)
        return {
            'status': 'completed',
            'message': await self._render_training_plan_message(revised_draft, pending=False),
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
            resolved_img_dir = str(readiness.get('resolved_img_dir') or ds.img_dir or '').strip()
            resolved_label_dir = str(readiness.get('resolved_label_dir') or ds.label_dir or '').strip()
            resolved_yaml = str(readiness.get('resolved_data_yaml') or ds.data_yaml or '').strip()
            if resolved_img_dir and resolved_label_dir:
                lines.append('当前状态: 已识别图片目录和标注目录')
            if not resolved_yaml:
                lines.append('当前状态: 还没有可用的 data.yaml，本次会自动补齐训练产物')
            elif readiness.get('ready'):
                lines.append(f'当前状态: 已有可用 data.yaml（{resolved_yaml}）')
            if args.get('force_split'):
                lines.append('附加安排: 按默认比例划分数据')
            planned_yaml = resolved_yaml or self._remote_join(dataset_path, 'data.yaml') if dataset_path else ''
            if planned_yaml:
                lines.append(f'预期产物: data.yaml -> {planned_yaml}')
            lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
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
            lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
            return '\n'.join(lines)

        if tool_name == 'start_training_loop':
            lines = ['准备执行：启动环训练']
            readiness = ds.last_readiness or {}
            if readiness.get('summary'):
                lines.append(f"数据理解: {readiness.get('summary')}")
            if args.get('loop_name'):
                lines.append(f"环训练名称: {args.get('loop_name')}")
            model = args.get('model') or tr.model
            data_yaml = args.get('data_yaml') or tr.data_yaml or ds.data_yaml
            if model:
                lines.append(f'模型: {model}')
            if data_yaml:
                lines.append(f'数据 YAML: {data_yaml}')
            lines.append(f"托管级别: {args.get('managed_level') or 'conservative_auto'}")
            lines.append(f"最大轮数: {args.get('max_rounds') or 5}")
            if args.get('target_metric'):
                target_line = f"目标指标: {args.get('target_metric')}"
                if args.get('target_metric_value') is not None:
                    target_line += f" >= {args.get('target_metric_value')}"
                lines.append(target_line)
            plan_bits = []
            for key in ('epochs', 'batch', 'imgsz', 'device'):
                value = args.get(key)
                if value is not None and value != '':
                    plan_bits.append(f'{key}={value}')
            if plan_bits:
                lines.append(f"首轮参数: {', '.join(str(item) for item in plan_bits)}")
            allowed_tuning_params = list(args.get('allowed_tuning_params') or [])
            if allowed_tuning_params:
                lines.append(f"允许自动调整: {', '.join(str(item) for item in allowed_tuning_params)}")
            lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
            return '\n'.join(lines)

        if tool_name == 'upload_assets_to_remote':
            lines = ['准备执行：远端上传']
            target_label = str(args.get('server') or self.session_state.active_remote_transfer.target_label or '').strip()
            remote_root = str(args.get('remote_root') or self.session_state.active_remote_transfer.remote_root or '').strip()
            if target_label:
                lines.append(f'目标服务器: {target_label}')
            if remote_root:
                lines.append(f'远端目录: {remote_root}')
            local_paths = list(args.get('local_paths') or [])
            if local_paths:
                lines.append('本地上传项:')
                lines.extend(f'- {item}' for item in local_paths[:5])
                if len(local_paths) > 5:
                    lines.append(f'- 其余 {len(local_paths) - 5} 项已省略')
            lines.append(
                '默认策略: 大文件自动分块 + 断点续传 + 哈希校验'
                f" (threshold={args.get('large_file_threshold_mb', 256)}MB, chunk={args.get('chunk_size_mb', 64)}MB)"
            )
            lines.append('说明: 这会把本机文件/目录复制到远端服务器。')
            lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
            return '\n'.join(lines)

        if tool_name == 'remote_prediction_pipeline':
            lines = ['准备执行：远端预测闭环']
            pipeline_args = dict(args or {})
            upload_args = dict(pipeline_args.get('upload_args') or {})
            target_label = str(upload_args.get('server') or self.session_state.active_remote_transfer.target_label or '').strip()
            remote_root = str(upload_args.get('remote_root') or self.session_state.active_remote_transfer.remote_root or '').strip()
            local_paths = list(upload_args.get('local_paths') or [])
            if target_label:
                lines.append(f'目标服务器: {target_label}')
            if remote_root:
                lines.append(f'远端目录: {remote_root}')
            if local_paths:
                lines.append('本地上传项:')
                lines.extend(f'- {item}' for item in local_paths[:5])
                if len(local_paths) > 5:
                    lines.append(f'- 其余 {len(local_paths) - 5} 项已省略')
            local_result_root = str(pipeline_args.get('local_result_root') or '').strip()
            if local_result_root:
                lines.append(f'本机回传目录: {local_result_root}')
            lines.append('执行链路: 上传本地模型/图片或视频 -> 远端执行 prediction -> 结果下载回本机')
            lines.append('限制: 待预测输入当前要求是单个文件或单个目录；多个散文件请先整理进目录。')
            lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
            return '\n'.join(lines)

        if tool_name == 'remote_training_pipeline':
            lines = ['准备执行：远端训练闭环']
            pipeline_args = dict(args or {})
            upload_args = dict(pipeline_args.get('upload_args') or {})
            target_label = str(upload_args.get('server') or self.session_state.active_remote_transfer.target_label or '').strip()
            remote_root = str(upload_args.get('remote_root') or self.session_state.active_remote_transfer.remote_root or '').strip()
            local_paths = list(upload_args.get('local_paths') or [])
            if target_label:
                lines.append(f'目标服务器: {target_label}')
            if remote_root:
                lines.append(f'远端目录: {remote_root}')
            if local_paths:
                lines.append('本地上传项:')
                lines.extend(f'- {item}' for item in local_paths[:5])
                if len(local_paths) > 5:
                    lines.append(f'- 其余 {len(local_paths) - 5} 项已省略')
            lines.append('执行链路: 上传本地模型/数据集 -> 远端做 readiness/prepare/preflight -> 启动训练')
            if pipeline_args.get('force_split'):
                lines.append('附加安排: 数据未就绪时自动按默认比例划分并补齐训练产物')
            if pipeline_args.get('wait_for_completion'):
                lines.append(
                    f"等待策略: 启动后轮询训练状态直到结束 (poll={pipeline_args.get('poll_interval_seconds', 15)}s, "
                    f"max_wait={pipeline_args.get('max_wait_seconds', 7200)}s)"
                )
            if pipeline_args.get('download_after_completion'):
                local_result_root = str(pipeline_args.get('local_result_root') or '').strip()
                if local_result_root:
                    lines.append(f'训练产物回传目录: {local_result_root}')
                lines.append('附加安排: 训练结束后自动把远端 run 目录下载回本机')
            lines.append('说明: 这会在远端真正启动训练进程，属于高风险动作。')
            lines.append('如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。')
            return '\n'.join(lines)

        pretty_args = "\n".join(f"  - {k}: {v}" for k, v in args.items()) or "  - 无参数"
        return (
            f"检测到高风险操作：{tool_name}\n"
            f"参数摘要：\n{pretty_args}\n"
            "如果你想继续，我就执行；如果想先停一下，也可以直接告诉我。"
        )

    @staticmethod
    def _build_cancel_message(tool_call: dict[str, Any]) -> str:
        return '好，我先不执行这一步。当前计划已保留；如果你想改参数、换模型、追问原因，或者稍后重新确认，都可以直接告诉我。'

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
        parts.append("你可以直接继续描述需求，或换一种方式说明你想做什么。")
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
            "yolostudio": build_mcp_connection_config(settings.mcp_url)
        }
    )
    raw_tools = list(await client.get_tools())
    raw_tools.extend(build_local_transfer_tools())
    include_tool_aliases = os.getenv("YOLOSTUDIO_MODEL_VISIBLE_TOOL_ALIASES", "").strip().lower() in {"1", "true", "yes", "on"}
    tools = adapt_tools_for_chat_model(raw_tools, include_aliases=include_tool_aliases)
    primary_llm_settings = settings.to_llm_settings(role='primary')
    llm = build_llm(primary_llm_settings, role='primary')
    helper_llm_enabled = str(os.getenv('YOLOSTUDIO_HELPER_LLM_ENABLED', '1')).strip().lower() not in {'0', 'false', 'no', 'off'}
    helper_llm_settings = settings.to_llm_settings(role='helper', inherit=primary_llm_settings)
    helper_llm = llm
    if helper_llm_enabled:
        if helper_llm_settings != primary_llm_settings:
            helper_llm = build_llm(helper_llm_settings, role='helper', inherit=primary_llm_settings)
    else:
        helper_llm = None
    react_kwargs: dict[str, Any] = {
        'prompt': SYSTEM_PROMPT,
        'checkpointer': FileCheckpointSaver(_checkpoint_path(settings)),
    }
    if str(settings.confirmation_mode or 'manual').strip().lower() == 'manual':
        interrupt_tool_names = _build_manual_interrupt_tool_names(raw_tools)
        if interrupt_tool_names:
            react_kwargs['interrupt_before'] = interrupt_tool_names
    graph = create_react_agent(
        llm,
        tools,
        **react_kwargs,
    )
    tool_registry = {tool.name: tool for tool in raw_tools}
    return YoloStudioAgentClient(
        graph=graph,
        settings=settings,
        tool_registry=tool_registry,
        planner_llm=helper_llm,
        primary_llm_settings=primary_llm_settings,
        helper_llm_settings=helper_llm_settings,
    )


async def build_agent():
    return await build_agent_client()

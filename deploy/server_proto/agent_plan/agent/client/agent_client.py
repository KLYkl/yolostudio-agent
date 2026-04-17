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
from typing import Any, Awaitable, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
try:
    from langchain_core.runnables.config import set_config_context
except Exception:
    @contextlib.contextmanager
    def set_config_context(config: Any):
        del config
        yield
from langchain_mcp_adapters.client import MultiServerMCPClient
from yolostudio_agent.agent.client.file_checkpointer import FileCheckpointSaver
from langgraph.prebuilt import create_react_agent
try:
    from langgraph.types import Command, interrupt
except Exception:
    from langgraph.types import Command

    def interrupt(value: Any) -> Any:
        return value
try:
    from langgraph.prebuilt.tool_node import ToolRuntime as InjectedToolRuntime
except Exception:
    InjectedToolRuntime = Any  # type: ignore[misc,assignment]

from yolostudio_agent.agent.client.context_builder import ContextBuilder
from yolostudio_agent.agent.client.context_retention_policy import build_context_retention_decision
from yolostudio_agent.agent.client.event_retriever import EventRetriever
from yolostudio_agent.agent.client.followup_router import (
    classify_training_loop_followup_action,
)
from yolostudio_agent.agent.client.grounded_reply_builder import build_grounded_tool_reply
from yolostudio_agent.agent.client.hitl_manager import (
    build_cancel_message as build_pending_cancel_message,
    build_pending_action_payload,
    confirmation_user_facts,
    pending_action_objective,
    pending_action_summary,
    pending_review_config,
)
from yolostudio_agent.agent.client.state_applier import apply_tool_result_to_state
from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.cached_tool_reply_service import build_cached_tool_context_payload
from yolostudio_agent.agent.client.dataset_fact_service import build_dataset_fact_context_payload
from yolostudio_agent.agent.client.llm_factory import (
    LlmProviderSettings,
    build_llm,
    provider_summary,
    resolve_llm_settings,
)
from yolostudio_agent.agent.client.mcp_connection import (
    load_mcp_tools_with_recovery,
)
from yolostudio_agent.agent.client.memory_store import MemoryStore
from yolostudio_agent.agent.client.mainline_guard_policy import (
    build_train_predict_guard_policy,
    compute_initial_train_predict_flags,
    suppress_deferred_cross_domain_intents,
)
from yolostudio_agent.agent.client.reply_renderer import (
    build_confirmation_message as build_confirmation_message_reply,
    build_confirmation_prompt as build_confirmation_prompt_reply,
    compact_action_candidates,
    confirmation_render_error as confirmation_render_error_reply,
    remote_pipeline_applied_results,
    render_confirmation_message as render_confirmation_message_reply,
    render_multi_tool_result_message as render_multi_tool_result_message_reply,
    render_tool_result_message as render_tool_result_message_reply,
    structured_overview_payloads,
    tool_result_render_error,
    tool_result_user_facts,
    fallback_multi_tool_result_message as fallback_multi_tool_result_message_reply,
    fallback_tool_result_text as fallback_tool_result_text_reply,
)
from yolostudio_agent.agent.client.remote_transfer_tools import build_local_transfer_tools
from yolostudio_agent.agent.client.session_state import SessionState, utc_now
from yolostudio_agent.agent.client.tool_adapter import (
    adapt_tools_for_chat_model,
    canonical_tool_name,
    normalize_tool_args,
    stringify_tool_result_facts,
)
from yolostudio_agent.agent.client.tool_policy import (
    pending_allowed_decisions,
    resolve_tool_execution_policy,
)
from yolostudio_agent.agent.client.tool_result_parser import parse_tool_message
from yolostudio_agent.agent.client.training_workflow import sync_training_workflow_state
from yolostudio_agent.agent.client.training_plan_service import (
    build_training_loop_start_draft as build_training_loop_start_draft_service,
    build_training_loop_start_fallback_plan as build_training_loop_start_fallback_plan_service,
    plan_training_loop_start as plan_training_loop_start_service,
    render_training_plan_message as render_training_plan_message_service,
    run_training_loop_start_orchestration as run_training_loop_start_orchestration_service,
    training_plan_render_error,
    training_plan_user_facts,
)
from yolostudio_agent.agent.client.training_followup_service import (
    complete_training_evidence_reply as complete_training_evidence_reply_service,
    complete_training_provenance_reply as complete_training_provenance_reply_service,
)

SYSTEM_PROMPT = """你是 YoloStudio Agent，负责帮助用户解决数据准备、训练、预测和远端传输问题。

工作方式：
1. 优先使用工具拿事实，再回答；不凭空猜测文件、数据集、训练或预测状态。
2. 工具选择主要依赖工具定义、参数模式和当前上下文；优先使用最少必要工具。
3. 如果用户已经给了明确路径、模型路径、报告路径或远端路径，直接复用，不要重复追问。
4. 最终回答必须由你自己组织成自然中文；除非用户明确要求调试细节，否则不要输出工具名、字段名、原始 JSON、命令 payload 或伪代码式调用示例。
5. 会修改数据、上传文件或启动长任务时，不要先在自然语言里自作主张执行；当参数足够时生成工具调用，由外部确认流程拦截。
6. 如果工具失败，直接解释失败原因，并告诉用户下一步最实际的动作。
7. 对 dataset / prediction / knowledge 的低风险追问，先看结构化上下文里是否已经有刚才的缓存结果；如果用户没有给新路径、新报告、新导出目标或新的知识主题，就优先复用这些事实直接回答，不要机械重复调用同一个工具。
8. 如果用户明确给了新的数据集路径、报告路径、输出目录、导出路径或整理目录，再调用对应工具；否则优先沿用上下文中的 dataset_root/img_dir、report_path/output_dir、topic/stage/signals。
9. 用户只要求“更详细一点”“刚才那个再展开”时，优先基于已缓存结果解释；只有缓存缺关键事实时才补调用轻量工具。

关键边界：
- dataset_training_readiness：只判断数据集本身是否已经具备直接训练的结构条件，不检查 GPU、device 或训练环境。
- training_readiness：只在用户准备现在启动训练，或明确询问执行条件（GPU / device / 训练环境）时使用。
- prepare_dataset_for_training：用于先准备数据、补齐 data.yaml、必要时划分 train/val。

回答要求：
- 先给结论，再给原因，再给下一步建议。
- 只有用户明确要求时，才展开参数细节、工具名或 JSON。
- 如果问题本身不需要工具，直接回答。"""

GPU_SENSITIVE_TOOLS = {
    "check_gpu_status",
    "prepare_dataset_for_training",
    "training_readiness",
    "training_preflight",
    "start_training",
    "start_training_loop",
}


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
        self._recorded_route_tool_call_ids: set[str] = set()
        self._recorded_tool_error_signatures: set[str] = set()
        self._pending_confirmation_shadow: dict[str, Any] | None = None
        self._pending_review_shadow: dict[str, Any] = {}
        self.memory = MemoryStore(settings.memory_root)
        self.context_builder = ContextBuilder(SYSTEM_PROMPT)
        self.event_retriever = EventRetriever(self.memory)
        self.session_state: SessionState = self.memory.load_state(settings.session_id)
        self._bootstrap_pending_confirmation_state()
        self._clear_stale_startup_state()
        self._sync_training_workflow_state(reason='startup_sync')
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

    def _record_route_ownership(
        self,
        route: str,
        *,
        thread_id: str = '',
        tool_name: str = '',
        tool_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            'route': str(route or '').strip(),
            'turn_index': self._turn_index,
        }
        if thread_id:
            payload['thread_id'] = thread_id
        normalized_tool_name = canonical_tool_name(tool_name) if tool_name else ''
        if normalized_tool_name:
            payload['tool'] = normalized_tool_name
        normalized_tool_names = [canonical_tool_name(name) for name in list(tool_names or []) if str(name).strip()]
        normalized_tool_names = [name for name in normalized_tool_names if name]
        if normalized_tool_names:
            payload['tools'] = normalized_tool_names
        if metadata:
            payload.update({key: value for key, value in metadata.items() if value not in (None, '', [], {}, ())})
        self.memory.append_event(self.session_state.session_id, 'route_ownership', payload)

    def route_ownership_report(self, limit: int | None = None) -> list[dict[str, Any]]:
        return self.memory.read_events_by_type(self.session_state.session_id, 'route_ownership', limit=limit)

    def _record_bypass_route(self, thread_id: str, result: dict[str, Any]) -> None:
        tool_call = dict(result.get('tool_call') or {})
        self._record_route_ownership(
            'graph-external-bypass',
            thread_id=thread_id,
            tool_name=str(tool_call.get('name') or ''),
            metadata={
                'status': str(result.get('status') or '').strip(),
            },
        )

    def _record_graph_selected_tools(
        self,
        messages: list[BaseMessage],
        *,
        thread_id: str,
        built_messages_len: int,
    ) -> None:
        delta_messages = messages[built_messages_len:] if built_messages_len <= len(messages) else messages
        for message in delta_messages:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in getattr(message, 'tool_calls', []) or []:
                tool_name = canonical_tool_name(tool_call.get('name') or '')
                if not tool_name:
                    continue
                tool_call_id = str(tool_call.get('id') or '').strip()
                signature = tool_call_id or json.dumps(
                    {
                        'tool': tool_name,
                        'args': normalize_tool_args(tool_name, tool_call.get('args', {})),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if signature in self._recorded_route_tool_call_ids:
                    continue
                self._recorded_route_tool_call_ids.add(signature)
                self._record_route_ownership(
                    'graph-selected-tool',
                    thread_id=thread_id,
                    tool_name=tool_name,
                    metadata={'tool_call_id': tool_call_id},
                )

    def _record_tool_error_recovery(
        self,
        messages: list[BaseMessage],
        *,
        thread_id: str,
        built_messages_len: int,
    ) -> None:
        delta_messages = messages[built_messages_len:] if built_messages_len <= len(messages) else messages
        errors: list[dict[str, Any]] = []
        for message in delta_messages:
            if not isinstance(message, ToolMessage):
                continue
            parsed = parse_tool_message(message)
            if parsed.get('ok', True):
                continue
            tool_name = canonical_tool_name(message.name or 'unknown_tool')
            error_text = str(parsed.get('error') or parsed.get('summary') or '未知错误').strip() or '未知错误'
            signature = json.dumps(
                {
                    'tool_call_id': str(message.tool_call_id or ''),
                    'tool': tool_name,
                    'error': error_text,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if signature in self._recorded_tool_error_signatures:
                continue
            self._recorded_tool_error_signatures.add(signature)
            errors.append(
                {
                    'tool': tool_name,
                    'tool_call_id': str(message.tool_call_id or ''),
                    'error': error_text,
                }
            )
        if not errors:
            return
        self._record_route_ownership(
            'tool-error-recovery',
            thread_id=thread_id,
            tool_names=[str(item.get('tool') or '') for item in errors],
            metadata={'errors': errors},
        )

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

    def _sync_training_workflow_state(self, *, reason: str = '') -> None:
        sync_training_workflow_state(
            self.session_state,
            append_event=lambda event_type, payload: self.memory.append_event(
                self.session_state.session_id,
                event_type,
                payload,
            ),
            reason=reason,
        )

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
        sync_training_workflow_state(state)
        return state

    def _should_reuse_history_context(self, user_text: str) -> bool:
        decision = build_context_retention_decision(
            state=self.session_state,
            user_text=user_text,
            explicitly_references_previous_context=self._explicitly_references_previous_context(user_text),
        )
        return decision.reuse_history

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

    def _tool_policy(self, tool_name: str):
        return resolve_tool_execution_policy(tool_name, tool_registry=self.tool_registry)

    def _tool_is_read_only(self, tool_name: str) -> bool:
        return self._tool_policy(tool_name).read_only

    def _tool_is_destructive(self, tool_name: str) -> bool:
        return self._tool_policy(tool_name).destructive

    def _tool_requires_confirmation(self, tool_name: str) -> bool:
        return self._tool_policy(tool_name).confirmation_required

    @staticmethod
    def _local_llm_gpu_wait_enabled() -> bool:
        value = str(os.getenv('YOLOSTUDIO_LOCAL_LLM_GPU_WAIT', '') or '').strip().lower()
        return value in {'1', 'true', 'yes', 'on'}

    def _tool_risk_level(self, tool_name: str) -> str:
        return self._tool_policy(tool_name).risk_level

    def _has_training_state_context(self) -> bool:
        return self.session_state.active_training.workflow_state in {
            'preflight_ready',
            'pending_confirmation',
            'running',
            'completed',
            'failed',
            'stopped',
        }

    def _has_training_followup_context(self) -> bool:
        knowledge = self.session_state.active_knowledge
        return bool(
            self._has_training_state_context()
            or knowledge.last_analysis
            or knowledge.last_recommendation
            or knowledge.last_retrieval
        )

    def _has_training_loop_context(self) -> bool:
        return self.session_state.active_training.loop_workflow_state != 'loop_idle'

    def _has_training_loop_history_followup_context(self) -> bool:
        training = self.session_state.active_training
        return bool(
            training.recent_loops
            or training.last_loop_status
            or training.last_loop_detail
        )

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
            self._record_bypass_route(thread_id, routed)
            self._trim_history()
            self.memory.save_state(self.session_state)
            progressed = await self._maybe_auto_progress(routed, stream_handler=stream_handler)
            return progressed or routed

        unresolved_pending = self._resolve_pending_confirmation(thread_id=self._pending_confirmation_thread_id())
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
        # Keep the prompt summary lean when history should be trimmed, but always
        # pass the full typed runtime context into the graph so post-model cached
        # follow-up handling does not depend on synthetic message snapshots.
        graph_input = {
            "messages": built_messages,
            "cached_tool_context": build_cached_tool_context_payload(self.session_state),
            "dataset_fact_context": build_dataset_fact_context_payload(self.session_state),
        }
        result = await self._graph_invoke(graph_input, config=config, stream_handler=stream_handler)
        self._record_graph_selected_tools(result["messages"], thread_id=thread_id, built_messages_len=len(built_messages))
        self._record_tool_error_recovery(result["messages"], thread_id=thread_id, built_messages_len=len(built_messages))
        pending = self._resolve_pending_confirmation(thread_id=thread_id, config=config)
        if pending is not None:
            if auto_approve or self._auto_confirmation_enabled():
                return await self.confirm(thread_id, approved=True, stream_handler=stream_handler)
            self.memory.save_state(self.session_state)
            return self._needs_confirmation_result(thread_id, pending, await self._build_confirmation_message(pending))

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
        pending_thread_id = self._pending_confirmation_thread_id()
        pending = self._resolve_pending_confirmation(thread_id=pending_thread_id)
        if not pending:
            return {"status": "error", "message": "当前没有待确认的高风险操作。"}

        normalized_decision = self._normalize_confirmation_reply_decision(decision_payload.get('decision'))
        if normalized_decision == 'deny':
            normalized_decision = 'reject'
        if normalized_decision not in {'approve', 'reject', 'edit', 'clarify', 'unclear', 'restate'}:
            normalized_decision = 'unclear'

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
        config = self._pending_config(thread_id)
        pending = self._resolve_pending_confirmation(thread_id=thread_id, config=config)
        if not pending:
            return {"status": "error", "message": "当前没有待确认的高风险操作。"}
        pending = dict(pending)
        decision_context = dict((self._pending_from_state() or pending).get('decision_context') or {})
        if decision_context:
            pending['decision_context'] = decision_context

        if not approved:
            cancel_message = self._build_cancel_message(pending)
            self._messages.append(AIMessage(content=cancel_message))
            if str(pending.get('source') or '').strip().lower() == 'graph':
                self._clear_pending_confirmation(thread_id=thread_id, persist_graph=False)
                with contextlib.suppress(Exception):
                    await self._graph_invoke(Command(resume={'decision': 'reject'}), config=config, stream_handler=stream_handler)
            else:
                self._clear_pending_confirmation(thread_id=thread_id)
            self._trim_history()
            self.memory.append_event(self.session_state.session_id, "confirmation_cancelled", {"tool": pending["name"], "args": pending.get("args", {})})
            self.memory.save_state(self.session_state)
            return self._cancelled_result(pending, cancel_message)

        self.memory.append_event(self.session_state.session_id, "confirmation_approved", {"tool": pending["name"], "args": pending.get("args", {})})
        self._clear_pending_confirmation(
            thread_id=thread_id,
            persist_graph=str(pending.get('source') or '').strip().lower() != 'graph',
        )

        if pending.get('name') == 'remote_prediction_pipeline':
            return await self._execute_remote_prediction_pipeline(pending.get('args') or {})
        if pending.get('name') == 'remote_training_pipeline':
            return await self._execute_remote_training_pipeline(pending.get('args') or {})

        if pending.get('source') != 'graph' or pending.get('adapted'):
            return await self._execute_adapted_pending_tool(
                thread_id=thread_id,
                pending=pending,
                approved=True,
                auto_progress_followups=self._auto_confirmation_enabled(),
                stream_handler=stream_handler,
            )

        result = await self._graph_invoke(Command(resume={'decision': 'approve'}), config=config, stream_handler=stream_handler)
        self._record_graph_selected_tools(result["messages"], thread_id=thread_id, built_messages_len=0)
        self._record_tool_error_recovery(result["messages"], thread_id=thread_id, built_messages_len=0)
        next_pending = self._resolve_pending_confirmation(thread_id=thread_id, config=config)
        if next_pending is not None:
            self.memory.save_state(self.session_state)
            return self._needs_confirmation_result(thread_id, next_pending, await self._build_confirmation_message(next_pending))
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
        pending = self._pending_confirmation_shadow or self._compat_pending_from_session() or {}
        return str(pending.get('thread_id') or '').strip() or f"{self.session_state.session_id}-pending"

    def get_pending_action(self) -> dict[str, Any] | None:
        pending = self._resolve_pending_confirmation(thread_id=self._pending_confirmation_thread_id())
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
        if (
            tool_name == 'prepare_dataset_for_training'
            and any(token in user_text or token in normalized for token in ('直接训练', '开始训练', '启动训练', '执行训练', '那就训练', '那你训练'))
        ):
            return 'edit'
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
        self._pending_review_shadow = dict(decision_context)
        current_pending = self._pending_from_state()
        if current_pending is not None:
            current_pending['decision_context'] = dict(decision_context)
            self._pending_confirmation_shadow = dict(current_pending)
            self._mirror_pending_confirmation(current_pending)
            if str(current_pending.get('source') or '').strip().lower() != 'graph':
                self._update_graph_pending_state(
                    thread_id=pending_thread_id,
                    pending=current_pending,
                    review=decision_context,
                )
            else:
                self._update_graph_pending_state(
                    thread_id=pending_thread_id,
                    pending=None,
                    review=decision_context,
                )
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
        pending = self._resolve_pending_confirmation(thread_id=self._pending_confirmation_thread_id())
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
            'objective': str(pending.get('objective') or ''),
            'summary': str(pending.get('summary') or ''),
            'allowed_decisions': list(pending.get('allowed_decisions') or []),
            'review_config': dict(pending.get('review_config') or {}),
            'decision_context': dict(pending.get('decision_context') or {}),
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

    async def _complete_cached_or_direct_tool_reply(
        self,
        tool_name: str,
        *,
        cached_result: tuple[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if cached_result:
            cached_tool_name, payload = cached_result
            cached_reply = await self._complete_cached_tool_result_reply(cached_tool_name, payload)
            if cached_reply is not None:
                return cached_reply
        return await self._complete_direct_tool_reply(tool_name, **kwargs)

    async def _try_handle_prediction_management_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        del enabled, user_text, normalized_text
        return None

    async def _try_handle_prediction_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
        fallback_path: str,
    ) -> dict[str, Any] | None:
        del enabled, user_text, normalized_text, fallback_path
        return None

    async def _try_handle_extract_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        del enabled, user_text, normalized_text
        return None

    async def _try_handle_dataset_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
        dataset_path: str,
    ) -> dict[str, Any] | None:
        del enabled, user_text, normalized_text, dataset_path
        return None

    async def _try_handle_realtime_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        del enabled, user_text, normalized_text
        return None

    async def _try_handle_remote_roundtrip_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        del enabled, user_text, normalized_text
        return None

    async def _try_handle_remote_transfer_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        del enabled, user_text, normalized_text
        return None

    async def _try_handle_prediction_requests(
        self,
        *,
        user_text: str,
        normalized_text: str,
        prediction_path: str,
        wants_predict: bool,
        wants_train: bool,
        training_command_like: bool,
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
        del normalized_text, prediction_path, has_prediction_followup_context, has_prediction_management_followup_context, has_explicit_prediction_target
        explicit_source_path = intent_parsing.extract_dataset_path_from_text(user_text)
        explicit_model = intent_parsing.extract_model_from_text(user_text)
        can_fastpath_explicit_prediction = (
            wants_predict
            and bool(explicit_source_path)
            and bool(explicit_model)
            and not wants_train
            and not training_command_like
            and not wants_remote_upload
            and not wants_prediction_summary
            and not wants_prediction_output_inspection
            and not wants_prediction_report_export
            and not wants_prediction_path_lists
            and not wants_prediction_result_organize
        )
        if can_fastpath_explicit_prediction:
            tool_name = 'predict_videos' if self._should_use_video_prediction(user_text, explicit_source_path) else 'predict_images'
            tool_args: dict[str, Any] = {
                'source_path': explicit_source_path,
                'model': explicit_model,
            }
            output_dir = intent_parsing.extract_output_path_from_text(user_text, explicit_source_path)
            if output_dir:
                tool_args['output_dir'] = output_dir
            return await self._complete_direct_tool_reply(tool_name, **tool_args)
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
        wants_quality: bool,
        wants_health: bool,
        wants_duplicates: bool,
        wants_readiness: bool,
        readiness_only_query: bool,
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
                and not wants_train
                and not wants_predict
                and not training_command_like
                and not wants_remote_upload
                and not wants_quality
                and not wants_health
                and not wants_duplicates
                and not wants_readiness
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

        if dataset_path and wants_readiness and readiness_only_query and not wants_predict and not training_command_like:
            return await self._complete_readiness_knowledge_reply(dataset_path)
        return None

    async def _try_handle_training_history_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        del enabled, user_text, normalized_text
        return None

    async def _try_handle_training_loop_history_followup(
        self,
        *,
        enabled: bool,
        user_text: str,
        normalized_text: str,
    ) -> dict[str, Any] | None:
        del enabled, user_text, normalized_text
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
        del enabled, user_text, normalized_text, metric_signals, asks_metric_terms
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
        del enabled, user_text, normalized_text, metric_signals, asks_metric_terms
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
        wants_realtime_stop: bool,
        wants_camera_prediction: bool,
        wants_rtsp_prediction: bool,
        wants_screen_prediction: bool,
        has_realtime_context: bool,
        has_explicit_realtime_target: bool,
        rtsp_url: str,
    ) -> dict[str, Any] | None:
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
            return await self._complete_cached_or_direct_tool_reply(
                'list_remote_profiles',
                cached_result=(
                    ('list_remote_profiles', self.session_state.active_remote_transfer.last_profile_listing)
                    if self.session_state.active_remote_transfer.last_profile_listing else None
                ),
            )

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
        wants_training_summary: bool,
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
                and has_training_context
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

        if not wants_predict and not training_command_like and wants_training_summary:
            return await self._complete_cached_or_direct_tool_reply(
                'summarize_training_run',
                cached_result=self._training_summary_request_cached_result(),
            )

        if not wants_predict and not training_command_like:
            suggested_tool = self._suggested_training_followup_tool(user_text)
            if suggested_tool:
                cached_result = (
                    self._training_summary_request_cached_result()
                    if suggested_tool == 'summarize_training_run'
                    else self._knowledge_tool_cached_result(suggested_tool)
                )
                return await self._complete_cached_or_direct_tool_reply(
                    suggested_tool,
                    cached_result=cached_result,
                )

        if (
            wants_training_status
            and not wants_predict
            and not training_command_like
            and not wants_training_loop_list
            and not wants_training_loop_status
            and not wants_inspect_training_loop
            and not wants_pause_training_loop
            and not wants_resume_training_loop
            and not wants_stop_training_loop
            and not wants_training_loop_start
        ):
            return await self._complete_cached_or_direct_tool_reply(
                'check_training_status',
                cached_result=self._training_status_request_cached_result(),
            )

        if not wants_predict and not training_command_like and wants_training_provenance:
            return self._complete_training_provenance_reply()

        if not wants_predict and not training_command_like and wants_training_evidence:
            return self._complete_training_evidence_reply()

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
        blocks_training_start: bool,
        explicit_run_ids: list[str],
        wants_best_weight_prediction: bool,
        wants_split: bool,
        has_explicit_realtime_target: bool,
    ) -> dict[str, Any] | None:
        if wants_training_loop_start and not wants_predict and not training_command_like:
            resolved_yaml = self._session_training_data_yaml(dataset_path=dataset_path)
            loop_args = self._collect_requested_training_loop_args(
                user_text,
                data_yaml=resolved_yaml if resolved_yaml else None,
            )
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

        if wants_train and not dataset_path and not no_train and not readiness_only_query and not wants_training_outcome_analysis and not wants_next_step_guidance and not wants_training_knowledge and not blocks_training_start:
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

        if dataset_path and wants_train and not no_train and not readiness_only_query and not wants_training_outcome_analysis and not wants_next_step_guidance and not wants_training_knowledge and not blocks_training_start:
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
        self._sync_training_workflow_state(reason='route_eval')
        guardrail = self._try_handle_guardrail_intent(user_text)
        if guardrail is not None:
            return guardrail
        plan_dialogue = await self._try_handle_training_plan_dialogue(user_text, thread_id)
        if plan_dialogue is not None:
            return plan_dialogue
        extracted_dataset_path = intent_parsing.extract_dataset_path_from_text(user_text)
        frame_followup_path = ''
        if any(token in user_text for token in ('这些帧', '刚才抽的帧', '刚才这些帧', '这些抽出来的帧', '这些图片', '刚才抽的图片')):
            frame_followup_path = str((self.session_state.active_dataset.last_frame_extract or {}).get('output_dir') or '').strip()
        dataset_path = extracted_dataset_path or frame_followup_path or self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir
        prediction_path = intent_parsing.extract_dataset_path_from_text(user_text) or self.session_state.active_prediction.source_path
        normalized_text = user_text.lower()
        metric_signals = self._extract_metric_signals_from_text(user_text)
        has_training_context = self._has_training_state_context()
        has_knowledge_followup_context = bool(
            self.session_state.active_knowledge.last_retrieval
            or self.session_state.active_knowledge.last_analysis
            or self.session_state.active_knowledge.last_recommendation
        )
        has_training_followup_context = self._has_training_followup_context()
        has_training_history_followup_context = bool(
            self.session_state.active_training.recent_runs
            or self.session_state.active_training.last_run_inspection
            or self.session_state.active_training.last_run_comparison
            or self.session_state.active_training.best_run_selection
        )
        has_training_loop_history_followup_context = self._has_training_loop_history_followup_context()
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
        wants_train, wants_predict = compute_initial_train_predict_flags(
            user_text,
            normalized_text,
            training_status_phrase=training_status_phrase,
        )
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
        wants_train, wants_predict = suppress_deferred_cross_domain_intents(
            user_text,
            normalized_text,
            wants_train=wants_train,
            wants_predict=wants_predict,
        )
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
        has_training_summary_context = bool(
            self.session_state.active_training.training_run_summary
            or self.session_state.active_training.last_summary
            or self.session_state.active_training.last_status
            or self.session_state.active_training.last_start_result
            or self.session_state.active_training.running
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
        wants_training_summary = (
            any(
                token in user_text
                for token in (
                    '训练总结',
                    '训练摘要',
                    '训练汇总',
                    '训练结果汇总',
                    '训练结果总结',
                    '结果总结',
                )
            )
            or (
                has_training_summary_context
                and '数据总结' in user_text
                and not any(
                    token in user_text
                    for token in (
                        '数据集质量',
                        '数据质量',
                        '类别分布',
                        '有哪些类别',
                        '缺失标签',
                        '最少类别',
                        '最多类别',
                    )
                )
            )
        )
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
        wants_best_training_run_analysis = wants_best_training_run and (
            wants_training_outcome_analysis
            or any(token in user_text for token in ('怎么看', '结果怎么样', '结果如何', '效果怎么样', '效果如何', '意味着什么'))
        )
        wants_training_outcome_analysis = wants_training_outcome_analysis or explicit_run_outcome_phrase
        training_command_like = any(token in user_text for token in ('开始训练', '启动训练', '训练这个数据', '用这个数据训练', '直接开训', 'start_training'))
        guard_policy = build_train_predict_guard_policy(
            user_text=user_text,
            normalized_text=normalized_text,
            wants_train=wants_train,
            wants_predict=wants_predict,
            no_train=no_train,
            wants_readiness=wants_readiness,
            training_command_like=training_command_like,
            wants_training_run_compare=wants_training_run_compare,
            wants_best_training_run=wants_best_training_run,
            blocks_training_start_signals=(
                wants_stop_training,
                wants_training_run_list,
                wants_training_run_compare,
                wants_best_training_run,
                wants_training_run_inspect,
                wants_failed_training_run_list,
                wants_completed_training_run_list,
                wants_stopped_training_run_list,
                wants_running_training_run_list,
                wants_analysis_ready_run_list,
                wants_training_loop_list,
                wants_training_loop_status,
                wants_inspect_training_loop,
                wants_pause_training_loop,
                wants_resume_training_loop,
                wants_stop_training_loop,
            ),
        )
        wants_train = guard_policy.wants_train
        wants_predict = guard_policy.wants_predict
        wants_segmentation_training = guard_policy.wants_segmentation_training
        wants_prediction_and_training_mix = guard_policy.wants_prediction_and_training_mix
        wants_prediction_result_as_training_data = guard_policy.wants_prediction_result_as_training_data
        wants_merge_extract_into_training = guard_policy.wants_merge_extract_into_training
        wants_best_weight_prediction = guard_policy.wants_best_weight_prediction
        wants_continuous_parallel_predict = guard_policy.wants_continuous_parallel_predict
        blocks_training_start = guard_policy.blocks_training_start
        readiness_only_query = guard_policy.readiness_only_query

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

        has_explicit_remote_target = bool(intent_parsing.extract_remote_server_from_text(user_text) or intent_parsing.extract_remote_root_from_text(user_text))
        has_explicit_transfer_paths = bool(intent_parsing.extract_all_paths_from_text(user_text))
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

        has_explicit_prediction_target = bool(intent_parsing.extract_dataset_path_from_text(user_text) or intent_parsing.extract_model_from_text(user_text))
        prediction_request = await self._try_handle_prediction_requests(
            user_text=user_text,
            normalized_text=normalized_text,
            prediction_path=prediction_path,
            wants_predict=wants_predict,
            wants_train=wants_train,
            training_command_like=training_command_like,
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
            wants_quality=wants_quality,
            wants_health=wants_health,
            wants_duplicates=wants_duplicates,
            wants_readiness=wants_readiness,
            readiness_only_query=readiness_only_query,
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
            wants_training_summary=wants_training_summary,
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
            blocks_training_start=blocks_training_start,
            explicit_run_ids=explicit_run_ids,
            wants_best_weight_prediction=wants_best_weight_prediction,
            wants_split=wants_split,
            has_explicit_realtime_target=has_explicit_realtime_target,
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
        training_summary = dict(
            self.session_state.active_training.training_run_summary
            or self.session_state.active_training.last_summary
            or self.session_state.active_training.last_status
            or {}
        )
        if not training_summary:
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
            reply = str(result.get('summary') or training_summary.get('summary') or result.get('error') or '训练结果分析已完成')
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if training_summary.get('ok', True) and result.get('ok', True) else 'error',
            'message': reply,
            'tool_call': None,
        }

    def _complete_training_provenance_reply(self) -> dict[str, Any]:
        return complete_training_provenance_reply_service(
            self.session_state,
            append_ai_message=lambda reply: self._messages.append(AIMessage(content=reply)),
        )

    def _complete_training_evidence_reply(self) -> dict[str, Any]:
        return complete_training_evidence_reply_service(
            self.session_state,
            append_ai_message=lambda reply: self._messages.append(AIMessage(content=reply)),
        )

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
        return bool(intent_parsing.extract_dataset_path_from_text(text))

    async def _try_handle_prepare_only_intent(self, user_text: str, thread_id: str) -> dict[str, Any] | None:
        if not self._looks_like_prepare_only_request(user_text):
            return None
        dataset_path = str(intent_parsing.extract_dataset_path_from_text(user_text) or '').strip()
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
        planned_training_args = self._collect_requested_training_args(
            user_text,
            data_yaml=None,
        )
        draft = self._build_training_plan_draft(
            user_text=user_text,
            dataset_path=dataset_path,
            readiness=readiness,
            preflight={},
            next_tool_name='prepare_dataset_for_training',
            next_tool_args=dict(prepare_args),
            planned_training_args=planned_training_args,
        )
        draft['execution_mode'] = 'prepare_only'
        self._save_training_plan_draft(draft)
        return self._needs_confirmation_result(thread_id, pending, reply)

    @staticmethod
    def _extract_training_run_ids_from_text(text: str) -> list[str]:
        return re.findall(r'train_log_[A-Za-z0-9_-]+', text)

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
        for item in [*quoted_paths, *intent_parsing.extract_all_paths_from_text(user_text)]:
            if item and item not in seen_paths:
                seen_paths.add(item)
                raw_paths.append(item)
        remote_root = intent_parsing.extract_remote_root_from_text(user_text)
        server = intent_parsing.extract_remote_server_from_text(user_text)
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
    def _normalize_context_path(path: str) -> str:
        text = str(path or '').strip().replace('\\', '/')
        text = re.sub(r'/+', '/', text)
        if len(text) > 1 and text.endswith('/'):
            text = text.rstrip('/')
        return text

    @classmethod
    def _dataset_scope_candidates(cls, dataset_path: str) -> list[str]:
        normalized = cls._normalize_context_path(dataset_path)
        if not normalized:
            return []
        scopes = [normalized]
        leaf_name = normalized.rsplit('/', 1)[-1].lower()
        if leaf_name in {'images', 'labels', 'train', 'val', 'test'} and '/' in normalized:
            parent = normalized.rsplit('/', 1)[0] or '/'
            if parent not in scopes:
                scopes.append(parent)
        return scopes

    @classmethod
    def _path_within_scope(cls, candidate: str, scope: str) -> bool:
        candidate_normalized = cls._normalize_context_path(candidate)
        scope_normalized = cls._normalize_context_path(scope)
        if not candidate_normalized or not scope_normalized:
            return False
        return candidate_normalized == scope_normalized or candidate_normalized.startswith(scope_normalized + '/')

    def _session_training_data_yaml(self, dataset_path: str = '') -> str:
        ds = self.session_state.active_dataset
        tr = self.session_state.active_training
        candidates = [
            str(ds.data_yaml or '').strip(),
            str((tr.active_loop_request or {}).get('data_yaml') or '').strip(),
            str(tr.data_yaml or '').strip(),
        ]
        if not str(dataset_path or '').strip():
            return next((candidate for candidate in candidates if candidate), '')

        scopes = self._dataset_scope_candidates(dataset_path)
        for candidate in candidates:
            if candidate and any(self._path_within_scope(candidate, scope) for scope in scopes):
                return candidate
        return ''

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

        model_items = [item for item in uploaded_items if intent_parsing.looks_like_model_path(str(item.get('local_path') or item.get('remote_path') or ''))]
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
        model_items = [item for item in uploaded_items if intent_parsing.looks_like_model_path(str(item.get('local_path') or item.get('remote_path') or ''))]
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

    @staticmethod
    def _pending_signature(pending: dict[str, Any] | None) -> str:
        if not pending:
            return ''
        return json.dumps(
            {
                'thread_id': str(pending.get('thread_id') or '').strip(),
                'tool': str(pending.get('name') or pending.get('tool_name') or '').strip(),
                'args': dict(pending.get('args') or pending.get('tool_args') or {}),
                'source': str(pending.get('source') or '').strip(),
                'tool_call_id': str(pending.get('tool_call_id') or pending.get('id') or '').strip(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _compat_pending_from_session(self) -> dict[str, Any] | None:
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
            'source': str(pc.source or 'synthetic').strip().lower() or 'synthetic',
        }

    def _mirror_pending_confirmation(self, pending: dict[str, Any] | None) -> None:
        pc = self.session_state.pending_confirmation
        if not pending:
            pc.thread_id = ""
            pc.tool_name = ""
            pc.tool_args = {}
            pc.source = 'synthetic'
            pc.interrupt_kind = 'tool_approval'
            pc.objective = ''
            pc.summary = ''
            pc.allowed_decisions = ['approve', 'reject', 'edit', 'clarify']
            pc.review_config = {}
            pc.decision_context = {}
            pc.created_at = ""
            return
        pc.thread_id = str(pending.get('thread_id') or '').strip()
        pc.tool_name = str(pending.get('name') or pending.get('tool_name') or '').strip()
        pc.tool_args = dict(pending.get('args') or pending.get('tool_args') or {})
        pc.source = str(pending.get('source') or 'synthetic').strip().lower() or 'synthetic'
        pc.interrupt_kind = str(pending.get('interrupt_kind') or 'tool_approval').strip() or 'tool_approval'
        pc.objective = str(pending.get('objective') or '').strip()
        pc.summary = str(pending.get('summary') or '').strip()
        pc.allowed_decisions = list(pending.get('allowed_decisions') or ['approve', 'reject', 'edit', 'clarify'])
        pc.review_config = dict(pending.get('review_config') or {})
        pc.decision_context = dict(pending.get('decision_context') or {})
        pc.created_at = str(pending.get('created_at') or utc_now()).strip()

    def _bootstrap_pending_confirmation_state(self) -> None:
        legacy_pending = self._compat_pending_from_session()
        if not legacy_pending:
            self._mirror_pending_confirmation(None)
            return
        self._pending_confirmation_shadow = dict(legacy_pending)
        self._pending_review_shadow = dict(legacy_pending.get('decision_context') or {})
        self._mirror_pending_confirmation(legacy_pending)

    @staticmethod
    def _pending_config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _graph_state_snapshot(self, config: dict[str, Any]) -> Any | None:
        if not hasattr(self.graph, 'get_state'):
            return None
        try:
            return self.graph.get_state(config)
        except Exception:
            return None

    def _graph_pending_review(self, config: dict[str, Any]) -> dict[str, Any]:
        state = self._graph_state_snapshot(config)
        values = getattr(state, 'values', {}) if state else {}
        if isinstance(values, dict):
            review = values.get('pending_review')
            if isinstance(review, dict):
                return dict(review)
        return {}

    @staticmethod
    def _merge_pending_review_context(pending: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(pending)
        review_context = dict(review or {})
        if review_context:
            merged['decision_context'] = review_context
        else:
            merged['decision_context'] = dict(merged.get('decision_context') or {})
        return merged

    def _graph_pending_from_values(self, config: dict[str, Any]) -> dict[str, Any] | None:
        state = self._graph_state_snapshot(config)
        values = getattr(state, 'values', {}) if state else {}
        if not isinstance(values, dict):
            return None
        pending = values.get('pending_confirmation')
        if not isinstance(pending, dict):
            return None
        merged = dict(pending)
        tool_name = canonical_tool_name(str(merged.get('name') or merged.get('tool_name') or '').strip())
        if not tool_name:
            return None
        merged['name'] = tool_name
        merged['tool_name'] = tool_name
        merged['args'] = normalize_tool_args(tool_name, dict(merged.get('args') or merged.get('tool_args') or {}))
        merged['tool_args'] = dict(merged['args'])
        merged['source'] = str(merged.get('source') or 'synthetic').strip().lower() or 'synthetic'
        return self._merge_pending_review_context(merged, self._graph_pending_review(config))

    def _graph_pending_from_interrupt(self, config: dict[str, Any]) -> dict[str, Any] | None:
        state = self._graph_state_snapshot(config)
        if state is None:
            return None
        interrupts = list(getattr(state, 'interrupts', ()) or ())
        for task in getattr(state, 'tasks', ()) or ():
            interrupts.extend(list(getattr(task, 'interrupts', ()) or ()))
        for item in interrupts:
            raw_value = getattr(item, 'value', None)
            if not isinstance(raw_value, dict):
                continue
            tool_name = canonical_tool_name(str(raw_value.get('name') or raw_value.get('tool_name') or '').strip())
            if not tool_name:
                continue
            raw_args = dict(raw_value.get('raw_args') or raw_value.get('args') or raw_value.get('tool_args') or {})
            merged = {
                'id': raw_value.get('id') or raw_value.get('tool_call_id'),
                'tool_call_id': str(raw_value.get('tool_call_id') or raw_value.get('id') or '').strip(),
                'name': tool_name,
                'tool_name': tool_name,
                'args': normalize_tool_args(tool_name, dict(raw_value.get('args') or raw_value.get('tool_args') or {})),
                'tool_args': normalize_tool_args(tool_name, dict(raw_value.get('args') or raw_value.get('tool_args') or {})),
                'raw_name': str(raw_value.get('raw_name') or raw_value.get('name') or raw_value.get('tool_name') or '').strip(),
                'raw_args': raw_args,
                'adapted': bool(raw_value.get('adapted')),
                'summary': str(raw_value.get('summary') or '').strip(),
                'objective': str(raw_value.get('objective') or '').strip(),
                'allowed_decisions': list(raw_value.get('allowed_decisions') or []),
                'review_config': dict(raw_value.get('review_config') or {}),
                'decision_context': dict(raw_value.get('decision_context') or {}),
                'thread_id': str(raw_value.get('thread_id') or '').strip(),
                'source': 'graph',
                'interrupt_kind': str(raw_value.get('interrupt_kind') or 'tool_approval').strip() or 'tool_approval',
            }
            return self._merge_pending_review_context(merged, self._graph_pending_review(config))
        return None

    def _update_graph_pending_state(
        self,
        *,
        thread_id: str,
        pending: dict[str, Any] | None,
        review: dict[str, Any] | None = None,
    ) -> None:
        if not thread_id or not hasattr(self.graph, 'update_state'):
            return
        update = {
            'pending_confirmation': dict(pending) if isinstance(pending, dict) else None,
            'pending_review': dict(review or {}),
        }
        try:
            self.graph.update_state(self._pending_config(thread_id), update)
        except Exception:
            return

    def _pending_allowed_decisions(self, tool_name: str) -> list[str]:
        return pending_allowed_decisions(self._tool_policy(tool_name))

    def _pending_action_objective(self, tool_name: str, args: dict[str, Any]) -> str:
        return pending_action_objective(self.session_state, tool_name, args)

    def _pending_action_summary(self, tool_name: str, args: dict[str, Any]) -> str:
        return pending_action_summary(self.session_state, tool_name, args)

    def _pending_review_config(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        del args
        return pending_review_config(tool_name, self._tool_policy(tool_name))

    def _build_pending_action_payload(
        self,
        pending: dict[str, Any],
        *,
        thread_id: str | None = None,
        decision_state: str = 'pending',
    ) -> dict[str, Any]:
        return build_pending_action_payload(
            self.session_state,
            pending,
            tool_policy_resolver=self._tool_policy,
            thread_id=thread_id,
            decision_state=decision_state,
        )

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

    def _remember_pending_confirmation(
        self,
        pending: dict[str, Any],
        *,
        emit_event: bool,
        persist_graph: bool,
    ) -> None:
        normalized = dict(pending)
        normalized['thread_id'] = str(normalized.get('thread_id') or '').strip()
        normalized['source'] = str(normalized.get('source') or 'synthetic').strip().lower() or 'synthetic'
        previous_signature = self._pending_signature(self._pending_confirmation_shadow)
        self._pending_confirmation_shadow = normalized
        self._pending_review_shadow = dict(normalized.get('decision_context') or {})
        self._mirror_pending_confirmation(normalized)
        if persist_graph and normalized['source'] != 'graph':
            self._update_graph_pending_state(
                thread_id=normalized['thread_id'],
                pending=normalized,
                review=self._pending_review_shadow,
            )
        if emit_event and self._pending_signature(normalized) != previous_signature:
            self.memory.append_event(
                self.session_state.session_id,
                'confirmation_requested',
                {
                    'tool': str(normalized.get('name') or ''),
                    'args': dict(normalized.get('args') or {}),
                    'thread_id': normalized['thread_id'],
                    'summary': str(normalized.get('summary') or ''),
                    'objective': str(normalized.get('objective') or ''),
                    'allowed_decisions': list(normalized.get('allowed_decisions') or []),
                    'pending_source': normalized['source'],
                },
            )
        self._sync_training_workflow_state(reason='pending_set')

    def _set_pending_confirmation(self, thread_id: str, pending: dict[str, Any]) -> None:
        payload = self._build_pending_action_payload(pending, thread_id=thread_id)
        source = str(
            pending.get('source')
            or ('synthetic' if pending.get('synthetic') else 'graph')
        ).strip().lower()
        if source not in {'graph', 'synthetic'}:
            source = 'synthetic'
        normalized = {
            'id': pending.get('id'),
            'tool_call_id': str(pending.get('tool_call_id') or pending.get('id') or '').strip(),
            'name': payload['tool_name'],
            'tool_name': payload['tool_name'],
            'args': dict(payload['tool_args']),
            'tool_args': dict(payload['tool_args']),
            'summary': payload['summary'],
            'objective': payload['objective'],
            'allowed_decisions': list(payload['allowed_decisions']),
            'review_config': dict(payload['review_config']),
            'decision_context': dict(payload.get('decision_context') or {}),
            'thread_id': thread_id,
            'source': source,
            'interrupt_kind': payload['interrupt_kind'],
            'created_at': utc_now(),
        }
        self._remember_pending_confirmation(
            normalized,
            emit_event=True,
            persist_graph=True,
        )

    def _clear_pending_confirmation(self, *, thread_id: str = '', persist_graph: bool = True) -> None:
        resolved_thread_id = str(
            thread_id
            or (self._pending_confirmation_shadow or {}).get('thread_id')
            or (self._compat_pending_from_session() or {}).get('thread_id')
            or ''
        ).strip()
        self._pending_confirmation_shadow = None
        self._pending_review_shadow = {}
        self._mirror_pending_confirmation(None)
        if persist_graph and resolved_thread_id:
            self._update_graph_pending_state(thread_id=resolved_thread_id, pending=None, review={})
        self._sync_training_workflow_state(reason='pending_cleared')

    def _pending_from_state(self) -> dict[str, Any] | None:
        if self._pending_confirmation_shadow:
            return self._merge_pending_review_context(self._pending_confirmation_shadow, self._pending_review_shadow)
        legacy = self._compat_pending_from_session()
        if not legacy:
            return None
        self._pending_confirmation_shadow = dict(legacy)
        self._pending_review_shadow = dict(legacy.get('decision_context') or {})
        return self._merge_pending_review_context(legacy, self._pending_review_shadow)

    def _resolve_pending_confirmation(
        self,
        *,
        thread_id: str = '',
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        shadow_pending = self._pending_from_state()
        pending_thread_id = str(thread_id or (shadow_pending or {}).get('thread_id') or self._pending_confirmation_thread_id()).strip()
        pending_config = config or self._pending_config(pending_thread_id)
        graph_pending = self._graph_pending_from_values(pending_config)
        if graph_pending is not None:
            self._remember_pending_confirmation(graph_pending, emit_event=False, persist_graph=False)
            return graph_pending
        interrupt_pending = self._graph_pending_from_interrupt(pending_config)
        if interrupt_pending is not None:
            if pending_thread_id and not str(interrupt_pending.get('thread_id') or '').strip():
                interrupt_pending['thread_id'] = pending_thread_id
            self._remember_pending_confirmation(interrupt_pending, emit_event=False, persist_graph=False)
            return interrupt_pending
        pending = shadow_pending or self._pending_from_state()
        if not pending:
            return None
        if str(pending.get('source') or 'synthetic').strip().lower() == 'graph':
            self._clear_pending_confirmation(thread_id=pending_thread_id, persist_graph=False)
            self.memory.save_state(self.session_state)
            return None
        return pending

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
        self._sync_training_workflow_state(reason=f'tool:{tool_name}')

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
            "source": "graph",
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

    def _recent_ai_text(self) -> str:
        for message in reversed(self._messages):
            if isinstance(message, AIMessage):
                return self._message_text(getattr(message, 'content', ''))
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
            or self._session_training_data_yaml(dataset_path=str(draft.get('dataset_path') or ''))
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
        explicit_target = intent_parsing.extract_dataset_path_from_text(user_text)
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
        session_yaml = self._session_training_data_yaml(dataset_path=dataset_path)
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
        return build_training_loop_start_fallback_plan_service(
            user_text=user_text,
            dataset_path=dataset_path,
            loop_args=loop_args,
            observed_tools=observed_tools,
            known_training_loop_data_yaml=self._known_training_loop_data_yaml,
            build_loop_prepare_args=self._build_loop_prepare_args,
        )

    async def _plan_training_loop_start(
        self,
        *,
        user_text: str,
        dataset_path: str,
        loop_args: dict[str, Any],
        observed_tools: dict[str, dict[str, Any]] | None = None,
        step_index: int = 1,
    ) -> dict[str, Any]:
        return await plan_training_loop_start_service(
            planner_llm=self.planner_llm,
            session_id=self.session_state.session_id,
            user_text=user_text,
            dataset_path=dataset_path,
            loop_args=loop_args,
            observed_tools=observed_tools,
            step_index=step_index,
            build_training_loop_start_fallback_plan_fn=self._build_training_loop_start_fallback_plan,
            known_training_loop_data_yaml=self._known_training_loop_data_yaml,
            compact_training_loop_start_fact=self._compact_training_loop_start_fact,
            invoke_structured_payload=self._invoke_structured_payload,
            normalize_training_loop_start_plan=self._normalize_training_loop_start_plan,
            append_event=lambda event, payload: self.memory.append_event(self.session_state.session_id, event, payload),
        )

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
        return build_training_loop_start_draft_service(
            self.session_state,
            user_text=user_text,
            dataset_path=dataset_path,
            loop_args=loop_args,
            observed_tools=observed_tools,
            plan=plan,
            known_training_loop_data_yaml=self._known_training_loop_data_yaml,
        )

    async def _run_training_loop_start_orchestration(
        self,
        *,
        user_text: str,
        thread_id: str,
        dataset_path: str,
        loop_args: dict[str, Any],
        observed_tools: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await run_training_loop_start_orchestration_service(
            self.session_state,
            user_text=user_text,
            thread_id=thread_id,
            dataset_path=dataset_path,
            loop_args=loop_args,
            observed_tools=observed_tools,
            direct_tool=self.direct_tool,
            plan_training_loop_start_fn=self._plan_training_loop_start,
            known_training_loop_data_yaml=self._known_training_loop_data_yaml,
            append_event=lambda event, payload: self.memory.append_event(self.session_state.session_id, event, payload),
            compact_training_loop_start_fact=self._compact_training_loop_start_fact,
            build_training_loop_start_draft_fn=self._build_training_loop_start_draft,
            save_training_plan_draft=self._save_training_plan_draft,
            set_pending_confirmation=self._set_pending_confirmation,
            build_confirmation_message=self._build_confirmation_message,
            needs_confirmation_result=self._needs_confirmation_result,
            append_ai_message=lambda reply: self._messages.append(AIMessage(content=reply)),
        )

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
        has_training_loop_context = self._has_training_loop_context()
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
        return await classify_training_loop_followup_action(
            planner_llm=self.planner_llm,
            session_state=self.session_state,
            user_text=user_text,
            normalized_text=normalized_text,
            loop_id=loop_id,
            classify_structured_action=self._classify_structured_action,
        )
    def _training_status_request_cached_result(self) -> tuple[str, dict[str, Any]] | None:
        training = self.session_state.active_training
        payload = dict(training.last_status or {})
        if not payload:
            return None
        run_state = str(
            payload.get('run_state')
            or (payload.get('status_overview') or {}).get('run_state')
            or ''
        ).strip().lower()
        if training.running or run_state == 'running':
            return None
        return ('check_training_status', payload)

    def _training_summary_request_cached_result(self) -> tuple[str, dict[str, Any]] | None:
        payload = dict(
            self.session_state.active_training.training_run_summary
            or self.session_state.active_training.last_summary
            or {}
        )
        if not payload:
            return None
        return ('summarize_training_run', payload)

    def _knowledge_tool_cached_result(self, tool_name: str) -> tuple[str, dict[str, Any]] | None:
        canonical_name = canonical_tool_name(tool_name)
        knowledge = self.session_state.active_knowledge
        if canonical_name == 'analyze_training_outcome' and knowledge.last_analysis:
            return ('analyze_training_outcome', dict(knowledge.last_analysis))
        if canonical_name == 'recommend_next_training_step' and knowledge.last_recommendation:
            return ('recommend_next_training_step', dict(knowledge.last_recommendation))
        return None

    def _suggested_training_followup_tool(self, user_text: str) -> str:
        if self._classify_confirmation_reply_fallback(user_text) != 'approve':
            return ''
        recent_ai_text = self._recent_ai_text()
        if not recent_ai_text or not any(
            token in recent_ai_text
            for token in (
                '训练结果汇总',
                '训练总结',
                '训练摘要',
                '训练结果分析',
                '下一步建议',
                '可继续调用',
                '是否同意',
                '可以继续分析训练结果',
            )
        ):
            return ''
        training = self.session_state.active_training
        for payload in (
            training.training_run_summary,
            training.last_summary,
            training.last_status,
        ):
            if not isinstance(payload, dict) or not payload:
                continue
            for item in list(payload.get('action_candidates') or []):
                if not isinstance(item, dict):
                    continue
                tool_name = canonical_tool_name(item.get('tool') or '')
                if tool_name in {'summarize_training_run', 'analyze_training_outcome', 'recommend_next_training_step'}:
                    return tool_name
        return ''

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
        return training_plan_user_facts(draft, pending=pending)

    def _training_plan_render_error(self, draft: dict[str, Any], *, pending: bool, error: Exception | None = None) -> str:
        return training_plan_render_error(draft, pending=pending, error=error)

    async def _render_training_plan_message(self, draft: dict[str, Any], *, pending: bool) -> str:
        return await render_training_plan_message_service(
            planner_llm=self.planner_llm,
            draft=draft,
            pending=pending,
            render_training_plan_draft=self._render_training_plan_draft,
            invoke_renderer_text=self._invoke_renderer_text,
        )

    async def _build_confirmation_message(self, tool_call: dict[str, Any]) -> str:
        return await build_confirmation_message_reply(
            self.session_state,
            tool_call,
            render_training_plan_message=self._render_training_plan_message,
            render_confirmation_message=self._render_confirmation_message,
        )

    def _confirmation_user_facts(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        return confirmation_user_facts(
            self.session_state,
            tool_call,
            confirmation_mode=self._confirmation_mode(),
            human_training_step_name=self._human_training_step_name,
            compact_action_candidates=self._compact_action_candidates,
        )

    def _confirmation_render_error(self, tool_call: dict[str, Any], error: Exception | None = None) -> str:
        return confirmation_render_error_reply(
            tool_call,
            error=error,
            append_event=lambda event, payload: self.memory.append_event(self.session_state.session_id, event, payload),
            build_confirmation_prompt=self._build_confirmation_prompt,
        )

    async def _render_confirmation_message(self, tool_call: dict[str, Any]) -> str:
        return await render_confirmation_message_reply(
            planner_llm=self.planner_llm,
            tool_call=tool_call,
            build_confirmation_prompt=self._build_confirmation_prompt,
            confirmation_user_facts=self._confirmation_user_facts,
            invoke_renderer_text=self._invoke_renderer_text,
            confirmation_render_error=self._confirmation_render_error,
        )

    def _tool_result_user_facts(self, tool_name: str, parsed: dict[str, Any]) -> dict[str, Any]:
        return tool_result_user_facts(tool_name, parsed)

    @staticmethod
    def _compact_action_candidates(action_candidates: Any) -> list[dict[str, Any]]:
        return compact_action_candidates(action_candidates)

    @classmethod
    def _structured_overview_payloads(cls, parsed: dict[str, Any]) -> dict[str, Any]:
        return structured_overview_payloads(parsed)

    @staticmethod
    def _remote_pipeline_applied_results(tool_name: str, parsed: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        return remote_pipeline_applied_results(tool_name, parsed)

    def _fallback_tool_result_text(self, tool_name: str, parsed: dict[str, Any]) -> str:
        return fallback_tool_result_text_reply(
            tool_name,
            parsed,
            build_grounded_tool_reply=self._build_grounded_tool_reply,
        )

    def _fallback_multi_tool_result_message(
        self,
        applied_results: list[tuple[str, dict[str, Any]]],
        *,
        extra_notes: list[str] | None = None,
    ) -> str:
        return fallback_multi_tool_result_message_reply(
            applied_results,
            extra_notes=extra_notes,
            build_grounded_tool_reply=self._build_grounded_tool_reply,
            merge_grounded_sections=self._merge_grounded_sections,
        )

    async def _render_multi_tool_result_message(
        self,
        applied_results: list[tuple[str, dict[str, Any]]],
        *,
        objective: str = '',
        extra_notes: list[str] | None = None,
    ) -> str:
        return await render_multi_tool_result_message_reply(
            planner_llm=self.planner_llm,
            applied_results=applied_results,
            objective=objective,
            extra_notes=extra_notes,
            invoke_renderer_text=self._invoke_renderer_text,
            render_tool_result_message=self._render_tool_result_message,
            build_grounded_tool_reply=self._build_grounded_tool_reply,
            merge_grounded_sections=self._merge_grounded_sections,
        )

    def _tool_result_render_error(self, tool_name: str, parsed: dict[str, Any], error: Exception | None = None) -> str:
        return tool_result_render_error(tool_name, parsed, error=error)

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
        return await render_tool_result_message_reply(
            planner_llm=self.planner_llm,
            tool_name=tool_name,
            parsed=parsed,
            render_multi_tool_result_message=self._render_multi_tool_result_message,
            invoke_renderer_text=self._invoke_renderer_text,
            build_grounded_tool_reply=self._build_grounded_tool_reply,
            merge_grounded_sections=self._merge_grounded_sections,
        )

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
        latest_dataset_path = intent_parsing.extract_dataset_path_from_text(user_text)
        all_paths = intent_parsing.extract_all_paths_from_text(user_text)
        project_path_hint = intent_parsing.extract_project_from_text(user_text)
        custom_script_hint = intent_parsing.extract_custom_training_script_from_text(user_text)
        dataset_candidates = [
            item for item in all_paths
            if not intent_parsing.looks_like_model_path(item)
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
        switching_prepare_only_to_train = bool(
            pending
            and pending.get('name') == 'prepare_dataset_for_training'
            and str(draft.get('execution_mode') or '').strip().lower() == 'prepare_only'
            and any(token in user_text for token in ('直接训练', '开始训练', '启动训练', '执行训练', '那就训练', '那你训练'))
        )
        has_revision = has_revision or switching_prepare_only_to_train
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
        if switching_prepare_only_to_train:
            revised_draft['execution_mode'] = 'prepare_then_train'
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
        return build_confirmation_prompt_reply(
            self.session_state,
            tool_call,
            render_training_plan_draft=self._render_training_plan_draft,
            remote_join=self._remote_join,
        )

    @staticmethod
    def _build_cancel_message(tool_call: dict[str, Any]) -> str:
        return build_pending_cancel_message(tool_call)

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


try:
    from langgraph.prebuilt.chat_agent_executor import AgentState as _LangGraphAgentState

    class _AgentRuntimeGraphState(_LangGraphAgentState, total=False):
        cached_tool_context: dict[str, Any] | None
        dataset_fact_context: dict[str, Any] | None
        pending_confirmation: dict[str, Any] | None
        pending_review: dict[str, Any] | None

except Exception:
    class _AgentRuntimeGraphState(TypedDict, total=False):
        messages: list[Any]
        remaining_steps: int
        cached_tool_context: dict[str, Any] | None
        dataset_fact_context: dict[str, Any] | None
        pending_confirmation: dict[str, Any] | None
        pending_review: dict[str, Any] | None


def _normalize_confirmation_resume_value(value: Any) -> str:
    candidate = value
    if isinstance(value, dict):
        candidate = (
            value.get('decision')
            or value.get('type')
            or value.get('action')
            or value.get('resume')
        )
    text = str(candidate or '').strip().lower()
    if text in {'approve', 'approved', 'accept', 'accepted', 'true', '1'}:
        return 'approve'
    if text in {'reject', 'rejected', 'deny', 'denied', 'ignore', 'cancel', 'false', '0'}:
        return 'reject'
    return text


def _tool_response_with_pending_clear(response: Any) -> Any:
    clear_update = {'pending_confirmation': None, 'pending_review': {}}
    if isinstance(response, Command):
        update = response.update
        if isinstance(update, dict):
            merged_update = dict(update)
            merged_update.update(clear_update)
            return Command(
                graph=response.graph,
                update=merged_update,
                resume=response.resume,
                goto=response.goto,
            )
        if isinstance(update, list):
            return Command(
                graph=response.graph,
                update={'messages': list(update), **clear_update},
                resume=response.resume,
                goto=response.goto,
            )
        if update is None:
            return Command(
                graph=response.graph,
                update=dict(clear_update),
                resume=response.resume,
                goto=response.goto,
            )
        return response
    if isinstance(response, ToolMessage):
        return Command(update={'messages': [response], **clear_update})
    return response


def _tool_rejection_command(*, tool_name: str, tool_call_id: str, message: str) -> Command:
    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=message,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            ],
            'pending_confirmation': None,
            'pending_review': {},
        }
    )


async def build_agent_client(settings: AgentSettings | None = None) -> YoloStudioAgentClient:
    settings = settings or AgentSettings()
    route_event_store = MemoryStore(settings.memory_root)
    client_holder: dict[str, YoloStudioAgentClient] = {}

    def _record_post_hook_route(route: str, payload: dict[str, Any]) -> None:
        route_event_store.append_event(
            settings.session_id,
            'route_ownership',
            {
                'route': str(route or '').strip(),
                **dict(payload or {}),
            },
        )

    raw_tools = await load_mcp_tools_with_recovery(
        settings.mcp_url,
        client_factory=MultiServerMCPClient,
    )
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
        'post_model_hook': _build_agent_post_model_hook(helper_llm, route_reporter=_record_post_hook_route),
        'state_schema': _AgentRuntimeGraphState,
    }
    graph_tools = _build_graph_tool_surface(
        tools,
        confirmation_mode=settings.confirmation_mode,
        tool_policy_resolver=lambda tool_name: resolve_tool_execution_policy(tool_name, tool_registry={tool.name: tool for tool in raw_tools}),
        client_getter=lambda: client_holder.get('client'),
    )
    graph = create_react_agent(
        llm,
        graph_tools,
        **react_kwargs,
    )
    tool_registry = {tool.name: tool for tool in raw_tools}
    client = YoloStudioAgentClient(
        graph=graph,
        settings=settings,
        tool_registry=tool_registry,
        planner_llm=helper_llm,
        primary_llm_settings=primary_llm_settings,
        helper_llm_settings=helper_llm_settings,
    )
    client_holder['client'] = client
    return client


def _build_graph_pending_interrupt_payload(
    request: Any,
    client: YoloStudioAgentClient | None,
) -> dict[str, Any]:
    tool_call = dict(getattr(request, 'tool_call', None) or {})
    raw_name = str(tool_call.get('name') or '').strip()
    canonical_name = canonical_tool_name(raw_name)
    raw_args = dict(tool_call.get('args') or {})
    tool_args = normalize_tool_args(canonical_name, raw_args)
    runtime = getattr(request, 'runtime', None)
    config = getattr(runtime, 'config', {}) if runtime is not None else {}
    thread_id = str(((config or {}).get('configurable') or {}).get('thread_id') or '').strip()
    tool_call_id = str(tool_call.get('id') or '').strip()
    pending = {
        'id': tool_call_id,
        'tool_call_id': tool_call_id,
        'name': canonical_name,
        'tool_name': canonical_name,
        'args': tool_args,
        'tool_args': dict(tool_args),
        'raw_name': raw_name,
        'raw_args': raw_args,
        'adapted': raw_name != canonical_name or raw_args != tool_args,
        'thread_id': thread_id,
        'source': 'graph',
        'interrupt_kind': 'tool_approval',
    }
    if client is not None:
        payload = client._build_pending_action_payload(pending, thread_id=thread_id)
    else:
        payload = {
            'interrupt_kind': 'tool_approval',
            'summary': '',
            'objective': '',
            'allowed_decisions': ['approve', 'reject', 'edit', 'clarify'],
            'review_config': {},
            'decision_context': {},
        }
    payload = {
        **dict(payload or {}),
        **pending,
        'thread_id': thread_id,
        'source': 'graph',
    }
    return payload


def _build_graph_tool_surface(
    tools: list[Any],
    *,
    confirmation_mode: str,
    tool_policy_resolver: Callable[[str], Any],
    client_getter: Callable[[], YoloStudioAgentClient | None],
) -> Any:
    try:
        from langgraph.prebuilt.tool_node import ToolNode
    except Exception:
        return tools
    try:
        from langgraph.errors import GraphBubbleUp
        from langgraph.prebuilt.tool_node import ToolCallRequest, _handle_tool_error
    except Exception:
        return ToolNode(tools, handle_tool_errors=True)
    manual_confirmation = str(confirmation_mode or 'manual').strip().lower() == 'manual'

    async def _awrap_tool_call(request: Any, execute: Callable[[Any], Awaitable[Any]]) -> Any:
        tool_call = dict(getattr(request, 'tool_call', None) or {})
        tool_name = canonical_tool_name(str(tool_call.get('name') or '').strip())
        policy = tool_policy_resolver(tool_name)
        if manual_confirmation and getattr(policy, 'confirmation_required', False):
            pending_payload = _build_graph_pending_interrupt_payload(request, client_getter())
            runtime = getattr(request, 'runtime', None)
            runtime_config = getattr(runtime, 'config', {}) if runtime is not None else {}
            with set_config_context(runtime_config) as runtime_context:
                if hasattr(runtime_context, 'run'):
                    resume_value = runtime_context.run(interrupt, pending_payload)
                else:
                    resume_value = interrupt(pending_payload)
            if _normalize_confirmation_resume_value(resume_value) != 'approve':
                return _tool_rejection_command(
                    tool_name=tool_name,
                    tool_call_id=str(tool_call.get('id') or ''),
                    message=build_pending_cancel_message(pending_payload),
                )
            response = await execute(request)
            return _tool_response_with_pending_clear(response)
        return await execute(request)

    class InterruptibleToolNode(ToolNode):
        def _run_one(self, call: Any, input_type: str, tool_runtime: Any) -> Any:
            tool = self.tools_by_name.get(call["name"])
            tool_request = ToolCallRequest(
                tool_call=call,
                tool=tool,
                state=tool_runtime.state,
                runtime=tool_runtime,
            )
            config = tool_runtime.config
            if self._wrap_tool_call is None:
                return self._execute_tool_sync(tool_request, input_type, config)

            def execute(req: Any) -> Any:
                return self._execute_tool_sync(req, input_type, config)

            try:
                return self._wrap_tool_call(tool_request, execute)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                if not self._handle_tool_errors:
                    raise
                content = _handle_tool_error(exc, flag=self._handle_tool_errors)
                return ToolMessage(
                    content=content,
                    name=tool_request.tool_call["name"],
                    tool_call_id=tool_request.tool_call["id"],
                    status="error",
                )

        async def _arun_one(self, call: Any, input_type: str, tool_runtime: Any) -> Any:
            tool = self.tools_by_name.get(call["name"])
            tool_request = ToolCallRequest(
                tool_call=call,
                tool=tool,
                state=tool_runtime.state,
                runtime=tool_runtime,
            )
            config = tool_runtime.config
            if self._awrap_tool_call is None and self._wrap_tool_call is None:
                return await self._execute_tool_async(tool_request, input_type, config)

            async def execute(req: Any) -> Any:
                return await self._execute_tool_async(req, input_type, config)

            def _sync_execute(req: Any) -> Any:
                return self._execute_tool_sync(req, input_type, config)

            try:
                if self._awrap_tool_call is not None:
                    return await self._awrap_tool_call(tool_request, execute)
                return self._wrap_tool_call(tool_request, _sync_execute)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                if not self._handle_tool_errors:
                    raise
                content = _handle_tool_error(exc, flag=self._handle_tool_errors)
                return ToolMessage(
                    content=content,
                    name=tool_request.tool_call["name"],
                    tool_call_id=tool_request.tool_call["id"],
                    status="error",
                )

    return InterruptibleToolNode(tools, handle_tool_errors=True, awrap_tool_call=_awrap_tool_call)


def _message_text_static(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
                continue
            if isinstance(item, dict):
                text = str(item.get('text') or item.get('content') or '').strip()
                if text:
                    parts.append(text)
        return '\n'.join(parts).strip()
    return str(content or '').strip()


def _merge_grounded_sections_static(sections: list[str]) -> str:
    cleaned: list[str] = []
    for section in sections:
        text = str(section or '').strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return '\n\n'.join(cleaned)


def _replace_last_ai_message(messages: list[Any], reply: str) -> dict[str, Any]:
    replacement = AIMessage(content=reply)
    try:
        from langchain_core.messages import RemoveMessage
        from langgraph.graph.message import REMOVE_ALL_MESSAGES
    except Exception:
        return {'messages': [*messages[:-1], replacement]}
    return {'messages': [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages[:-1], replacement]}


def _dataset_fact_post_model_hook(state: dict[str, Any]) -> dict[str, Any]:
    del state
    return {}


def _build_agent_post_model_hook(
    planner_llm: Any,
    *,
    route_reporter: Callable[[str, dict[str, Any]], None] | None = None,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    del planner_llm, route_reporter

    async def _post_model_hook(state: dict[str, Any]) -> dict[str, Any]:
        del state
        return {}

    return _post_model_hook


async def build_agent():
    return await build_agent_client()

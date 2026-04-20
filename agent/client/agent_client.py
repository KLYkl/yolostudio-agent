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
    from langgraph.graph import END, START, StateGraph
except Exception:
    END = '__end__'  # type: ignore[assignment]
    START = '__start__'  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
try:
    from langgraph.prebuilt.tool_node import ToolRuntime as InjectedToolRuntime
except Exception:
    InjectedToolRuntime = Any  # type: ignore[misc,assignment]

from yolostudio_agent.agent.client.context_builder import ContextBuilder
from yolostudio_agent.agent.client.context_retention_policy import build_context_retention_decision
from yolostudio_agent.agent.client.event_retriever import EventRetriever
from yolostudio_agent.agent.client.followup_router import (
    resolve_mainline_request_signals,
    resolve_training_loop_route,
)
from yolostudio_agent.agent.client.mainline_route_support import (
    resolve_mainline_context,
    resolve_mainline_dispatch_payload,
    resolve_mainline_guardrail_reply,
    resolve_mainline_route_state_payload,
)
from yolostudio_agent.agent.client.grounded_reply_builder import build_grounded_tool_reply
from yolostudio_agent.agent.client.hitl_manager import (
    build_cancel_message as build_pending_cancel_message,
    build_pending_action_payload,
    confirmation_user_facts,
    pending_action_objective,
    pending_action_summary,
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
from yolostudio_agent.agent.client.execution_contracts import PredictionRequestFollowupAction
from yolostudio_agent.agent.client.prediction_request_service import resolve_prediction_request_followup_action
from yolostudio_agent.agent.client.prediction_execution_service import (
    run_remote_prediction_pipeline_flow,
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
)
from yolostudio_agent.agent.client.tool_policy import (
    pending_allowed_decisions,
    resolve_tool_execution_policy,
)
from yolostudio_agent.agent.client.tool_result_parser import parse_tool_message
from yolostudio_agent.agent.client.training_workflow import sync_training_workflow_state
from yolostudio_agent.agent.client.training_request_service import (
    run_training_request_entrypoint,
)
from yolostudio_agent.agent.client.training_execution_service import (
    run_post_prepare_training_start_flow,
    run_remote_training_pipeline_flow,
    wait_for_remote_training_terminal_state as wait_for_remote_training_terminal_state_service,
)
from yolostudio_agent.agent.client.training_dialogue_service import (
    run_training_plan_dialogue_flow,
)
from yolostudio_agent.agent.client.training_contracts import TrainingPlanFollowupAction
from yolostudio_agent.agent.client.training_plan_service import (
    build_training_loop_start_draft as build_training_loop_start_draft_service,
    build_training_loop_start_fallback_plan as build_training_loop_start_fallback_plan_service,
    render_training_plan_message as render_training_plan_message_service,
    run_training_loop_start_orchestration as run_training_loop_start_orchestration_service,
    training_plan_render_error,
    training_plan_user_facts,
)
from yolostudio_agent.agent.client.training_plan_context_service import (
    build_training_plan_context_from_draft,
    build_training_plan_context_payload,
    extract_training_plan_context_from_state,
)
from yolostudio_agent.agent.client.training_confirmation_node import (
    answer_training_status_node,
    post_prepare_node,
    training_confirmation_node,
)
from yolostudio_agent.agent.client.training_schemas import (
    PendingTurnIntent,
    TrainingEdits,
    coerce_training_plan,
    merge_training_plan_edits,
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

_DEFER_TO_GRAPH = object()
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
        memory: MemoryStore | None = None,
        checkpointer: FileCheckpointSaver | None = None,
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
        self._recorded_graph_text_signatures: set[str] = set()
        self._pending_confirmation_shadow: dict[str, Any] | None = None
        self._pending_review_shadow: dict[str, Any] = {}
        self._graph_has_training_entry = bool(getattr(graph, '__codex_training_entry__', False))
        self.memory = memory or MemoryStore(settings.memory_root)
        self.checkpointer = checkpointer
        self.context_builder = ContextBuilder(SYSTEM_PROMPT)
        self.event_retriever = EventRetriever(self.memory)
        self.session_state: SessionState = self.memory.load_state(settings.session_id)
        self._bootstrap_pending_confirmation_state()
        self._reconcile_startup_pending_confirmation()
        self._bootstrap_turn_index()
        self._clear_stale_startup_state()
        self._sync_training_workflow_state(reason='startup_sync')
        self._record_startup_checkpoint_health()
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

    def _record_graph_text_response(
        self,
        messages: list[BaseMessage],
        *,
        thread_id: str,
        built_messages_len: int,
    ) -> None:
        delta_messages = messages[built_messages_len:] if built_messages_len <= len(messages) else messages
        if any(isinstance(message, AIMessage) and list(getattr(message, 'tool_calls', []) or []) for message in delta_messages):
            return
        final_text = ''
        for message in reversed(delta_messages):
            if not isinstance(message, AIMessage):
                continue
            text = self._message_text(getattr(message, 'content', ''))
            if text:
                final_text = text
                break
        if not final_text:
            return
        signature = json.dumps(
            {
                'thread_id': thread_id,
                'reply': final_text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if signature in self._recorded_graph_text_signatures:
            return
        self._recorded_graph_text_signatures.add(signature)
        self._record_route_ownership(
            'graph-text-response',
            thread_id=thread_id,
            metadata={
                'response_kind': 'text',
                'reply_preview': final_text[:120],
            },
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
                stream_handler=stream_handler,
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
        if str(pending.tool_name or '').strip():
            return
        draft = dict(self.session_state.active_training.training_plan_draft or {})
        if not draft:
            self._clear_training_plan_draft()
            return
        materialized = self._build_materialized_training_pending(
            draft=draft,
            user_text=str(draft.get('planner_user_request') or ''),
        )
        if materialized is None:
            self._clear_training_plan_draft()
            return
        rebuilt_draft, _ = materialized
        self._ensure_materialized_training_state_context(draft=rebuilt_draft)
        self._save_training_plan_draft(rebuilt_draft)

    def _record_startup_checkpoint_health(self) -> None:
        if self.checkpointer is None or not hasattr(self.checkpointer, 'health_payload'):
            return
        try:
            payload = dict(self.checkpointer.health_payload() or {})
        except Exception:
            return
        status = str(payload.get('status') or '').strip().lower()
        if status not in {'corrupt_detected', 'corrupt_recovered'}:
            return
        self.memory.append_event(
            self.session_state.session_id,
            'startup_checkpoint_degraded',
            payload,
        )

    def _startup_checkpoint_thread_ids(self) -> list[str]:
        thread_ids: set[str] = set()
        if self.checkpointer is None or not hasattr(self.checkpointer, 'thread_ids'):
            pass
        else:
            prefix = f"{self.session_state.session_id}-"
            try:
                thread_ids.update(
                    str(thread_id).strip()
                    for thread_id in self.checkpointer.thread_ids(prefix=prefix)
                    if str(thread_id).strip()
                )
            except Exception:
                pass
        startup_pending = self._startup_pending_from_session()
        if startup_pending and str(startup_pending.get('source') or '').strip().lower() == 'graph':
            hinted_thread_id = str(startup_pending.get('thread_id') or '').strip()
            if hinted_thread_id:
                thread_ids.add(hinted_thread_id)
        return sorted(thread_ids)

    def _startup_graph_pending_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for thread_id in self._startup_checkpoint_thread_ids():
            config = self._pending_config(thread_id)
            pending = self._graph_pending_from_values(config) or self._graph_pending_from_interrupt(config)
            if not pending:
                continue
            pending_source = str(pending.get('source') or 'synthetic').strip().lower() or 'synthetic'
            if pending_source != 'graph':
                continue
            restored = dict(pending)
            restored['thread_id'] = str(restored.get('thread_id') or thread_id).strip()
            candidates.append(restored)
        return candidates

    def _hydrate_startup_pending_confirmation(self) -> None:
        if self._pending_from_state():
            return
        candidates = self._startup_graph_pending_candidates()
        if len(candidates) > 1:
            self.memory.append_event(
                self.session_state.session_id,
                'startup_pending_restore_skipped',
                {
                    'reason': 'multiple_graph_pending_candidates',
                    'candidate_count': len(candidates),
                    'thread_ids': [str(candidate.get('thread_id') or '').strip() for candidate in candidates],
                },
            )
            return
        if len(candidates) != 1:
            return
        restored = candidates[0]
        config = self._pending_config(str(restored.get('thread_id') or '').strip())
        self._remember_pending_confirmation(restored, emit_event=False, persist_graph=False)
        self._restore_training_plan_draft_from_context(config=config, pending=restored)
        self.memory.append_event(
            self.session_state.session_id,
            'startup_graph_pending_restored',
            {
                'tool': str(restored.get('name') or restored.get('tool_name') or '').strip(),
                'thread_id': str(restored.get('thread_id') or '').strip(),
                'pending_source': str(restored.get('source') or 'graph').strip().lower() or 'graph',
            },
        )

    def _reconcile_startup_pending_confirmation(self) -> None:
        pending = self._pending_from_state()
        if not pending:
            self._hydrate_startup_pending_confirmation()
            restored = self._pending_from_state()
            if restored is not None:
                return
            startup_pending = self._startup_pending_from_session()
            if startup_pending and str(startup_pending.get('source') or '').strip().lower() == 'graph':
                thread_id = str(startup_pending.get('thread_id') or '').strip()
                self._clear_pending_confirmation(thread_id=thread_id, persist_graph=False)
                self._clear_training_plan_draft()
                self.memory.append_event(
                    self.session_state.session_id,
                    'startup_stale_pending_cleared',
                    {
                        'tool': str(startup_pending.get('name') or startup_pending.get('tool_name') or '').strip(),
                        'thread_id': thread_id,
                        'pending_source': 'graph',
                    },
                )
            return
        pending_source = str(pending.get('source') or 'synthetic').strip().lower() or 'synthetic'
        if pending_source != 'graph':
            candidates = self._startup_graph_pending_candidates()
            if len(candidates) == 1:
                restored = candidates[0]
                config = self._pending_config(str(restored.get('thread_id') or '').strip())
                self._clear_training_plan_draft()
                self._remember_pending_confirmation(restored, emit_event=False, persist_graph=False)
                self._restore_training_plan_draft_from_context(config=config, pending=restored)
                self.memory.append_event(
                    self.session_state.session_id,
                    'startup_legacy_pending_replaced',
                    {
                        'previous_tool': str(pending.get('name') or pending.get('tool_name') or '').strip(),
                        'previous_source': pending_source,
                        'restored_tool': str(restored.get('name') or restored.get('tool_name') or '').strip(),
                        'thread_id': str(restored.get('thread_id') or '').strip(),
                    },
                )
                self.memory.append_event(
                    self.session_state.session_id,
                    'startup_graph_pending_restored',
                    {
                        'tool': str(restored.get('name') or restored.get('tool_name') or '').strip(),
                        'thread_id': str(restored.get('thread_id') or '').strip(),
                        'pending_source': str(restored.get('source') or 'graph').strip().lower() or 'graph',
                    },
                )
            return
        thread_id = str(pending.get('thread_id') or '').strip() or self._pending_confirmation_thread_id()
        resolved = self._resolve_pending_confirmation(
            thread_id=thread_id,
            config=self._pending_config(thread_id),
        )
        if resolved is not None:
            self._restore_training_plan_draft_from_context(
                config=self._pending_config(thread_id),
                pending=resolved,
            )
        if resolved is None:
            self.memory.append_event(
                self.session_state.session_id,
                'startup_stale_pending_cleared',
                {
                    'tool': str(pending.get('name') or pending.get('tool_name') or '').strip(),
                    'thread_id': thread_id,
                    'pending_source': pending_source,
                },
            )

    def _sync_training_workflow_state(self, *, reason: str = '') -> None:
        sync_training_workflow_state(
            self.session_state,
            pending_confirmation=dict(self._pending_confirmation_shadow or {}) if self._pending_confirmation_shadow else None,
            append_event=lambda event_type, payload: self.memory.append_event(
                self.session_state.session_id,
                event_type,
                payload,
            ),
            reason=reason,
        )

    def _bootstrap_turn_index(self) -> None:
        prefix = f'{self.session_state.session_id}-turn-'
        candidate_thread_ids = set(self._startup_checkpoint_thread_ids())
        for candidate in (
            str(self.session_state.pending_confirmation.thread_id or '').strip(),
            str((self._pending_confirmation_shadow or {}).get('thread_id') or '').strip(),
        ):
            if candidate:
                candidate_thread_ids.add(candidate)
        max_turn_index = 0
        for thread_id in candidate_thread_ids:
            if not thread_id.startswith(prefix):
                continue
            suffix = thread_id[len(prefix):]
            match = re.match(r'(\d+)', suffix)
            if not match:
                continue
            with contextlib.suppress(Exception):
                max_turn_index = max(max_turn_index, int(match.group(1)))
        self._turn_index = max_turn_index

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
        pending = state.pending_confirmation
        pending.thread_id = ""
        pending.tool_name = ""
        pending.tool_args = {}
        pending.source = 'synthetic'
        pending.interrupt_kind = 'tool_approval'
        pending.objective = ''
        pending.summary = ''
        pending.allowed_decisions = ['approve', 'reject', 'edit', 'clarify']
        pending.review_config = {}
        pending.decision_context = {}
        pending.created_at = ""
        sync_training_workflow_state(state, pending_confirmation={})
        return state

    def _context_retention_decision(self, user_text: str):
        return build_context_retention_decision(
            state=self.session_state,
            user_text=user_text,
            explicitly_references_previous_context=self._explicitly_references_previous_context(user_text),
        )

    def _state_for_model(self, user_text: str) -> tuple[SessionState, bool, list[BaseMessage]]:
        decision = self._context_retention_decision(user_text)
        if decision.reuse_history:
            return self.session_state, True, list(self._messages)
        recent_messages = list(self._messages[-1:]) if self._messages else []
        if decision.preserve_state_context:
            return self.session_state, False, recent_messages
        cloned = SessionState.from_dict(self.session_state.to_dict())
        return self._strip_ephemeral_context(cloned), False, recent_messages

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

    async def chat(self, user_text: str, auto_approve: bool = False, stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None) -> dict[str, Any]:
        if not str(user_text).strip():
            return {"status": "completed", "message": "请输入内容。", "tool_call": None}
        self._messages.append(HumanMessage(content=user_text))
        self._turn_index += 1
        thread_id = f"{self.session_state.session_id}-turn-{self._turn_index}"

        active_interrupt = self._active_graph_interrupt(preferred_thread_id=thread_id)
        if active_interrupt is not None:
            interrupt_thread_id, interrupt_payload = active_interrupt
            if self._is_training_confirmation_interrupt_payload(interrupt_payload):
                decision = await self._parse_user_decision(
                    user_text=user_text,
                    interrupt_payload=interrupt_payload,
                )
                graph_result = await self._graph_invoke(
                    Command(resume=decision),
                    config=self._pending_config(interrupt_thread_id),
                    stream_handler=stream_handler,
                )
                next_interrupt = self._active_graph_interrupt(preferred_thread_id=interrupt_thread_id)
                response = self._format_interrupt_or_result(
                    graph_result,
                    thread_id=interrupt_thread_id,
                    interrupt_payload=next_interrupt[1] if next_interrupt is not None else None,
                )
                self._trim_history()
                self.memory.save_state(self.session_state)
                return response

        pending_dialogue = await self._try_handle_pending_confirmation_dialogue(user_text, stream_handler=stream_handler)
        pending_passthrough_requested = pending_dialogue is _DEFER_TO_GRAPH
        if pending_dialogue is not None and not pending_passthrough_requested:
            self._trim_history()
            self.memory.save_state(self.session_state)
            progressed = await self._maybe_auto_progress(pending_dialogue, stream_handler=stream_handler)
            return progressed or pending_dialogue

        routed = await self._try_handle_mainline_intent(
            user_text,
            thread_id,
            skip_training_plan_dialogue=pending_passthrough_requested,
        )
        if routed is _DEFER_TO_GRAPH:
            routed = None
        if routed is not None:
            self._record_bypass_route(thread_id, routed)
            self._trim_history()
            self.memory.save_state(self.session_state)
            progressed = await self._maybe_auto_progress(routed, stream_handler=stream_handler)
            return progressed or routed

        if pending_passthrough_requested:
            return await self._handoff_current_runtime_to_graph(
                thread_id=thread_id,
                user_text_hint=user_text,
                auto_approve=auto_approve,
                ignore_existing_pending=True,
                suppress_training_plan_context=True,
                suppress_ephemeral_state_context=True,
                stream_handler=stream_handler,
            )

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

        return await self._handoff_current_runtime_to_graph(
            thread_id=thread_id,
            user_text_hint=user_text,
            auto_approve=auto_approve,
            stream_handler=stream_handler,
        )

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
        active_interrupt = self._active_graph_interrupt(preferred_thread_id=thread_id)
        if active_interrupt is not None:
            interrupt_thread_id, interrupt_payload = active_interrupt
            if self._is_training_confirmation_interrupt_payload(interrupt_payload):
                decision = {
                    'action': 'approve' if approved else 'reject',
                    'reason': 'confirm() resume',
                }
                graph_result = await self._graph_invoke(
                    Command(resume=decision),
                    config=self._pending_config(interrupt_thread_id),
                    stream_handler=stream_handler,
                )
                next_interrupt = self._active_graph_interrupt(preferred_thread_id=interrupt_thread_id)
                response = self._format_interrupt_or_result(
                    graph_result,
                    thread_id=interrupt_thread_id,
                    interrupt_payload=next_interrupt[1] if next_interrupt is not None else None,
                )
                self._trim_history()
                self.memory.save_state(self.session_state)
                return response

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

        graph_stream_handler = stream_handler
        if pending.get('name') == 'prepare_dataset_for_training':
            graph_stream_handler = None
        result = await self._graph_invoke(
            Command(resume={'decision': 'approve'}),
            config=config,
            stream_handler=graph_stream_handler,
        )
        self._record_graph_selected_tools(result["messages"], thread_id=thread_id, built_messages_len=0)
        self._record_tool_error_recovery(result["messages"], thread_id=thread_id, built_messages_len=0)
        self._record_graph_text_response(result["messages"], thread_id=thread_id, built_messages_len=0)
        applied_results: list[tuple[str, dict[str, Any]]] | None = None

        if pending.get("name") == "prepare_dataset_for_training":
            applied_results = self._apply_tool_results(result["messages"], built_messages_len=0)
            prepare_parsed = self._find_applied_tool_result(applied_results, 'prepare_dataset_for_training') or {
                'ok': True,
                'summary': '数据准备已完成',
                'data_yaml': str(self.session_state.active_dataset.data_yaml or ''),
            }
            post_prepare_followup = await self._handle_post_prepare_confirmation_followup(
                thread_id=thread_id,
                prepare_parsed=prepare_parsed,
                stream_handler=stream_handler,
            )
            if post_prepare_followup is not None:
                return post_prepare_followup

        next_pending = self._resolve_pending_confirmation(thread_id=thread_id, config=config)
        if next_pending is not None:
            self.memory.save_state(self.session_state)
            return self._needs_confirmation_result(thread_id, next_pending, await self._build_confirmation_message(next_pending))
        if applied_results is None:
            applied_results = self._apply_tool_results(result["messages"], built_messages_len=0)

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
        pending = self._pending_confirmation_shadow or {}
        return str(pending.get('thread_id') or '').strip() or f"{self.session_state.session_id}-pending"

    def get_pending_action(self) -> dict[str, Any] | None:
        active_interrupt = self._active_graph_interrupt()
        if active_interrupt is not None:
            _, interrupt_payload = active_interrupt
            payload = self._pending_action_from_training_confirmation_interrupt(interrupt_payload)
            if payload is not None:
                return payload
        pending = self._resolve_pending_confirmation(thread_id=self._pending_confirmation_thread_id())
        if not pending:
            return None
        return self._build_pending_action_payload(pending, thread_id=self._pending_confirmation_thread_id())

    def _looks_like_pending_approve_request(self, user_text: str) -> bool:
        normalized = self._normalize_confirmation_text(user_text)
        if not normalized:
            return False
        if self._classify_confirmation_reply_fallback(user_text) != 'approve':
            return False
        if any(token in normalized for token in ('不', '别', '先不', '不要', '取消', '算了')):
            return False
        strong_approve_tokens = (
            '继续',
            '继续执行',
            '开始训练',
            '启动训练',
            '开始循环训练',
            '启动循环训练',
            '确认开始训练',
            '确认启动训练',
            '就这样',
            '按这个来',
            '没问题',
            '可以开始',
            '执行吧',
        )
        return any(token in normalized for token in strong_approve_tokens)

    def _looks_like_pending_edit_request(self, user_text: str, pending: dict[str, Any]) -> bool:
        text = str(user_text or '').strip()
        normalized = text.lower()
        if not text:
            return False
        tool_name = str((pending or {}).get('name') or '').strip()
        if not tool_name:
            return False
        if any(token in text or token in normalized for token in ('取消执行', '不要执行', '不执行', '先不要做', '先别做', '停止执行', '取消这一步')):
            return False
        training_like_tools = {'start_training', 'prepare_dataset_for_training', 'start_training_loop'}
        if tool_name in training_like_tools:
            clear_fields = self._collect_training_clear_fields(text)
            requested_training_args = self._collect_requested_training_args(text, data_yaml=None)
            split_tokens = (
                '自动划分', '不划分', '不要划分', '默认比例', '按默认比例', 'force_split', 'split',
                '只做准备', '先做准备', 'prepare only',
            )
            split_requested = any(token in text or token in normalized for token in split_tokens)
            if self._looks_like_pending_approve_request(text) and not clear_fields and not requested_training_args and not split_requested:
                return False
            if clear_fields:
                return True
            if requested_training_args:
                return True
            if split_requested:
                return True
        generic_revision_markers = (
            '改成', '换成', '换个', '调整', '改一下', '去掉', '取消类别', '类别限制',
            'batch', 'imgsz', 'device', 'epochs', '轮数', 'optimizer', 'freeze', 'resume', 'lr0', 'patience',
            'workers', 'amp', 'fraction', 'classes', 'single_cls', 'project', 'name', '环境', '模型', '权重',
            '训练环境', '学习率', '优化器', '冻结', '早停', '线程数', '混合精度',
        )
        return any(token in text or token in normalized for token in generic_revision_markers)

    @staticmethod
    def _looks_like_pending_clarify_request(user_text: str) -> bool:
        text = str(user_text or '').strip()
        if not text:
            return False
        question_tokens = (
            '为什么', '原因', '依据', '怎么看', '会不会', '会生成到哪里', '会上传到哪里', '产物路径', '输出路径', '先给我计划', '先看计划',
            '先讨论', '解释一下', '再解释一下', '说详细一点', '详细说说',
        )
        return any(token in text for token in question_tokens)

    def _pending_passthrough_decision(self, user_text: str, pending: dict[str, Any]) -> str | None:
        if self._looks_like_pending_edit_request(user_text, pending):
            return 'edit'
        if self._looks_like_pending_clarify_request(user_text):
            return 'clarify'
        return None

    async def _classify_pending_turn_intent(self, user_text: str, pending: dict[str, Any]) -> str:
        heuristic = self._pending_passthrough_decision(user_text, pending)
        if heuristic == 'edit':
            return 'edit'
        if heuristic == 'clarify':
            return 'status'
        if self._looks_like_pending_passthrough_request(user_text):
            return 'passthrough'
        decision = await self._classify_confirmation_reply(user_text, pending)
        if decision == 'approve':
            return 'approve'
        if decision == 'deny':
            return 'reject'
        if decision == 'restate':
            return 'restate'
        if decision == 'edit':
            return 'edit'
        if decision == 'clarify':
            return 'status'
        pending_followup_action = await self._classify_pending_followup_action(
            user_text=user_text,
            pending=pending,
        )
        if pending_followup_action == 'status_or_detail' or self._looks_like_pending_status_or_detail_query(user_text):
            return 'status'
        return 'restate'

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
    ) -> dict[str, Any] | object | None:
        pending = self._resolve_pending_confirmation(thread_id=self._pending_confirmation_thread_id())
        if not pending:
            return None
        pending_turn_intent = await self._classify_pending_turn_intent(user_text, pending)
        if pending_turn_intent == 'edit':
            self._record_pending_action_review(
                pending,
                decision='edit',
                raw_user_text=user_text,
                source='natural_language_chat',
            )
            return None
        if pending_turn_intent in {'approve', 'reject'}:
            return await self.review_pending_action(
                {
                    'decision': pending_turn_intent,
                    'raw_user_text': user_text,
                    'source': 'natural_language_chat',
                },
                stream_handler=stream_handler,
            )
        pending_thread_id = self._pending_confirmation_thread_id()
        if pending_turn_intent == 'restate':
            return self._needs_confirmation_result(pending_thread_id, pending, await self._build_confirmation_message(pending))
        if pending_turn_intent == 'status':
            return self._needs_confirmation_result(
                pending_thread_id,
                pending,
                await self._build_confirmation_message(pending),
            )
        if pending_turn_intent == 'passthrough':
            return _DEFER_TO_GRAPH
        return None

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

    @staticmethod
    def _looks_like_pending_passthrough_request(user_text: str) -> bool:
        text = str(user_text or '').strip()
        normalized = text.lower()
        if not text:
            return False
        explicit_override_tokens = (
            '我不是要继续当前计划',
            '不是要继续当前计划',
            '不是继续当前计划',
            '先不管当前计划',
        )
        if any(token in text for token in explicit_override_tokens):
            return True
        list_or_history_tokens = (
            '列表',
            '历史',
            '最近有哪些',
            '最近有啥',
            '最近有什么',
            '已有',
            '列出来',
            '看一下有哪些',
        )
        subject_tokens = (
            '环训练',
            '循环训练',
            '训练记录',
            '训练 run',
            'run',
            '服务器配置',
            '远端配置',
            'profile',
            'profiles',
        )
        return any(token in text or token in normalized for token in list_or_history_tokens) and any(
            token in text or token in normalized for token in subject_tokens
        )

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
    def _pending_turn_intent_schema() -> dict[str, Any]:
        return {
            'title': 'yolostudio_pending_turn_intent',
            'type': 'object',
            'properties': {
                'action': {
                    'type': 'string',
                    'enum': ['approve', 'reject', 'edit', 'status', 'new_task', 'unclear'],
                },
                'edits': {
                    'type': 'object',
                    'properties': {
                        'model': {'type': 'string'},
                        'epochs': {'type': 'integer'},
                        'batch': {'type': 'integer'},
                        'imgsz': {'type': 'integer'},
                        'device': {'type': 'string'},
                        'training_environment': {'type': 'string'},
                        'data_yaml': {'type': 'string'},
                        'max_rounds': {'type': 'integer'},
                        'epochs_per_round': {'type': 'integer'},
                        'loop_name': {'type': 'string'},
                    },
                    'additionalProperties': False,
                },
                'reason': {'type': 'string'},
            },
            'required': ['action', 'reason'],
            'additionalProperties': False,
        }

    def _normalize_pending_turn_intent_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized_source: dict[str, Any]
        try:
            normalized_source = self._model_dump_compat(PendingTurnIntent.model_validate(payload or {}))
        except Exception:
            normalized_source = dict(payload or {})

        action = str((normalized_source or {}).get('action') or '').strip().lower()
        if action not in {'approve', 'reject', 'edit', 'status', 'new_task', 'unclear'}:
            action = 'unclear'
        edits = normalized_source.get('edits') if isinstance(normalized_source, dict) else None
        normalized_edits: dict[str, Any] | None = None
        if isinstance(edits, dict):
            try:
                normalized_edits = self._model_dump_compat(TrainingEdits.model_validate(edits))
            except Exception:
                normalized_edits = {
                    key: value
                    for key, value in edits.items()
                    if value is not None
                }
        normalized = {
            'action': action,
            'reason': str((normalized_source or {}).get('reason') or '').strip(),
        }
        if normalized_edits:
            normalized['edits'] = normalized_edits
        return normalized

    async def _parse_user_decision(
        self,
        *,
        user_text: str,
        interrupt_payload: dict[str, Any],
    ) -> dict[str, Any]:
        plan_payload = {}
        try:
            plan_payload = self._model_dump_compat(coerce_training_plan(dict(interrupt_payload.get('plan') or {})))
        except Exception:
            raw_plan = interrupt_payload.get('plan')
            if isinstance(raw_plan, dict):
                plan_payload = dict(raw_plan)
        messages = [
            SystemMessage(
                content=(
                    '你是训练确认路由器。'
                    '根据当前训练计划和用户回复，输出结构化 JSON。'
                    'action 只能是 approve / reject / edit / status / new_task / unclear。'
                    '如果用户在修改参数，action=edit，且 edits 里只填写明确提到的新值。'
                    '如果用户在追问当前计划细节、产物路径、原因、当前参数，action=status。'
                    '如果用户在问与当前计划无关的新任务，例如查看历史、列出环训练、查其他对象，action=new_task。'
                    '不要编造参数，不要输出额外字段。'
                )
            ),
            HumanMessage(
                content=(
                    f'当前阶段: {str(interrupt_payload.get("phase") or "prepare")}\n'
                    f'当前计划: {json.dumps(plan_payload, ensure_ascii=False, indent=2)}\n'
                    f'用户回复: {user_text}'
                )
            ),
        ]
        parsed = await self._invoke_structured_payload(
            messages=messages,
            schema=self._pending_turn_intent_schema(),
        )
        normalized = self._normalize_pending_turn_intent_payload(parsed)
        if normalized.get('action') == 'unclear':
            fallback = self._fallback_pending_turn_intent(
                user_text=user_text,
                interrupt_payload=interrupt_payload,
            )
            if fallback:
                normalized = fallback
        if normalized.get('action') == 'unclear' and not normalized.get('reason'):
            normalized['reason'] = str(user_text or '').strip()
        return normalized

    def _fallback_pending_turn_intent(
        self,
        *,
        user_text: str,
        interrupt_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        text = str(user_text or '').strip()
        if not text:
            return None
        if self._looks_like_pending_passthrough_request(text):
            return {'action': 'new_task', 'reason': text}
        if self._looks_like_pending_status_or_detail_query(text):
            return {'action': 'status', 'reason': text}
        pending_like: dict[str, Any] = {}
        tool_call = self._training_confirmation_tool_call(interrupt_payload)
        if tool_call:
            pending_like = {'name': str(tool_call.get('name') or '').strip(), 'args': dict(tool_call.get('args') or {})}
        if self._looks_like_pending_edit_request(text, pending_like):
            edits = dict(self._collect_requested_training_args(text, data_yaml=None) or {})
            if edits:
                edits.pop('data_yaml', None)
            clear_fields = self._collect_training_clear_fields(text)
            for field in clear_fields:
                edits[field] = None
            if edits:
                return {'action': 'edit', 'reason': text, 'edits': edits}
        confirmation = self._classify_confirmation_reply_fallback(text)
        if confirmation == 'approve':
            return {'action': 'approve', 'reason': text}
        if confirmation in {'deny', 'reject'}:
            return {'action': 'reject', 'reason': text}
        return {'action': 'unclear', 'reason': text}

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

    @staticmethod
    def _model_dump_compat(value: Any, *, exclude_none: bool = True) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            payload = dict(value)
        elif hasattr(value, 'model_dump'):
            try:
                payload = value.model_dump(exclude_none=exclude_none)
            except TypeError:
                payload = value.model_dump()
            except Exception:
                payload = {}
        else:
            payload = dict(getattr(value, '__dict__', {}) or {})
        if exclude_none and isinstance(payload, dict):
            return {key: item for key, item in payload.items() if item is not None}
        return dict(payload or {})

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
        normalized = YoloStudioAgentClient._normalize_confirmation_text(text)
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
            '可以继续', '好继续', '没问题开始训练', '没问题开始训练吧',
            '开始训练吧', '确认开始训练', '确认启动训练', '可以开始训练', '可以继续执行',
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
            '开始吧', '执行吧', '没问题', '可以', '行', '继续', '确认',
            '可以继续', '好继续', '确认开始训练', '确认启动训练',
        )
        if any(marker in normalized for marker in approve_markers) and not any(token in normalized for token in ('不', '别', '先不', '不要', '取消', '算了')):
            return 'approve'
        return 'unclear'

    @staticmethod
    def _normalize_confirmation_text(value: Any) -> str:
        text = str(value or '').strip().lower()
        if not text:
            return ''
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[，。,\.!！?？、；;：:“”\"\'‘’（）()\[\]{}<>《》]+', '', text)
        return text
    def _sync_preferences(self) -> None:
        if self.session_state.preferences.default_model == self.primary_llm_settings.model:
            self.session_state.preferences.default_model = ""
        if self.session_state.preferences.language != "zh-CN":
            self.session_state.preferences.language = "zh-CN"

    def _apply_prediction_request_action(self, followup_action: PredictionRequestFollowupAction | None) -> dict[str, Any] | None:
        followup_action = dict(followup_action or {})
        action = str(followup_action.get('action') or '').strip()
        if action == 'none':
            return None
        if action == 'reply':
            return self._complete_dialogue_text(str(followup_action.get('reply') or ''))
        return None

    async def _execute_mainline_tool_bypass(
        self,
        tool_name: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tool_args = dict(args or {})
        parsed = await self.direct_tool(tool_name, **tool_args)
        reply = await self._render_tool_result_message(tool_name, parsed)
        if not reply:
            reply = str(
                parsed.get('summary')
                or parsed.get('message')
                or parsed.get('error')
                or ('操作执行成功' if parsed.get('ok') else '操作执行失败')
            ).strip()
        self._messages.append(AIMessage(content=reply))
        return {
            'status': 'completed' if parsed.get('ok') else 'error',
            'message': reply,
            'tool_call': {'name': tool_name, 'args': tool_args},
        }

    async def _try_handle_live_training_control_request(
        self,
        *,
        route_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        mainline_signals = dict(route_state.get('mainline_signals') or {})
        followup_flags = dict(route_state.get('followup_flags') or {})
        loop_route = dict(route_state.get('loop_route') or {})
        loop_action = str(loop_route.get('action') or '').strip()
        if loop_action in {'start', 'followup'}:
            return None
        followup_flags.pop('wants_training_revision', None)
        if any(followup_flags.values()):
            return None
        if mainline_signals.get('wants_stop_training'):
            if any(
                mainline_signals.get(key)
                for key in (
                    'wants_remote_profile_list',
                    'wants_remote_upload',
                    'wants_remote_prediction_pipeline',
                    'wants_remote_training_pipeline',
                    'wants_predict',
                )
            ):
                return None
            return await self._execute_mainline_tool_bypass('stop_training')
        if mainline_signals.get('training_status_phrase'):
            if any(
                mainline_signals.get(key)
                for key in (
                    'wants_remote_profile_list',
                    'wants_remote_upload',
                    'wants_remote_prediction_pipeline',
                    'wants_remote_training_pipeline',
                    'wants_predict',
                    'wants_train',
                    'no_train',
                    'wants_readiness',
                    'wants_split',
                    'training_command_like',
                )
            ):
                return None
            return await self._execute_mainline_tool_bypass('check_training_status')
        return None

    async def _try_handle_remote_requests(
        self,
        *,
        user_text: str,
        wants_remote_profile_list: bool,
        wants_remote_upload: bool,
        wants_remote_prediction_pipeline: bool,
        wants_remote_training_pipeline: bool,
    ) -> dict[str, Any] | object | None:
        if wants_remote_profile_list:
            cached_listing = dict(self.session_state.active_remote_transfer.last_profile_listing or {})
            if cached_listing:
                reply = await self._render_tool_result_message('list_remote_profiles', cached_listing)
                if reply:
                    self._messages.append(AIMessage(content=reply))
                    return {'status': 'completed', 'message': reply, 'tool_call': None}
            return _DEFER_TO_GRAPH

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
            return _DEFER_TO_GRAPH

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
            return _DEFER_TO_GRAPH

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
            return _DEFER_TO_GRAPH
        return None

    def _complete_training_entrypoint_result(self, entrypoint_result: dict[str, Any]) -> dict[str, Any] | object:
        draft = dict(entrypoint_result.get('draft') or {})
        reply = str(entrypoint_result.get('reply') or '').strip()
        if draft:
            self._save_training_plan_draft(draft)
        if reply:
            self._messages.append(AIMessage(content=reply))
        if entrypoint_result.get('defer_to_graph'):
            return _DEFER_TO_GRAPH
        return {'status': 'completed', 'message': reply, 'tool_call': None}

    async def _try_handle_training_entrypoints(
        self,
        *,
        thread_id: str,
        user_text: str,
        normalized_text: str,
        dataset_path: str,
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
        wants_split: bool,
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

        entrypoint_result = await run_training_request_entrypoint(
            session_state=self.session_state,
            user_text=user_text,
            normalized_text=normalized_text,
            dataset_path=dataset_path,
            frame_followup_path=frame_followup_path,
            wants_train=wants_train,
            wants_predict=wants_predict,
            no_train=no_train,
            readiness_only_query=readiness_only_query,
            wants_training_outcome_analysis=wants_training_outcome_analysis,
            wants_next_step_guidance=wants_next_step_guidance,
            wants_training_knowledge=wants_training_knowledge,
            wants_training_revision=wants_training_revision,
            wants_stop_training=wants_stop_training,
            blocks_training_start=blocks_training_start,
            explicit_run_ids=explicit_run_ids,
            wants_split=wants_split,
            direct_tool=self.direct_tool,
            collect_requested_training_args=self._collect_requested_training_args,
            is_training_discussion_only=self._is_training_discussion_only,
            extract_training_execution_backend=self._extract_training_execution_backend_from_text,
            build_training_plan_draft_fn=self._build_training_plan_draft,
            render_training_plan_message=self._render_training_plan_message,
        )
        if entrypoint_result is not None:
            return self._complete_training_entrypoint_result(entrypoint_result)
        return None

    async def _dispatch_mainline_requests(
        self,
        *,
        thread_id: str,
        user_text: str,
        mainline_context: dict[str, Any],
        route_state: dict[str, Any],
    ) -> dict[str, Any] | object | None:
        dispatch_payload = resolve_mainline_dispatch_payload(
            mainline_context=mainline_context,
            route_state=route_state,
        )
        remote_request = await self._try_handle_remote_requests(
            user_text=user_text,
            **dict(dispatch_payload.get('remote_request_args') or {}),
        )
        if remote_request is _DEFER_TO_GRAPH:
            return _DEFER_TO_GRAPH
        if remote_request:
            return remote_request

        prediction_request = self._apply_prediction_request_action(
            resolve_prediction_request_followup_action(
                **dict(dispatch_payload.get('prediction_request_args') or {}),
                best_run_selection=self.session_state.active_training.best_run_selection,
                last_run_inspection=self.session_state.active_training.last_run_inspection,
            )
        )
        if prediction_request:
            return prediction_request

        live_training_control = await self._try_handle_live_training_control_request(
            route_state=route_state,
        )
        if live_training_control:
            return live_training_control

        training_entrypoint_args = dict(dispatch_payload.get('training_entrypoint_request_args') or {})
        if bool(training_entrypoint_args.get('wants_training_loop_start')) or not self._graph_has_training_entry:
            training_entrypoint_request = await self._try_handle_training_entrypoints(
                thread_id=thread_id,
                user_text=user_text,
                **training_entrypoint_args,
            )
            if training_entrypoint_request is _DEFER_TO_GRAPH:
                return None
            return training_entrypoint_request
        return None

    def _collect_mainline_context(self, user_text: str) -> dict[str, Any]:
        return resolve_mainline_context(
            session_state=self.session_state,
            user_text=user_text,
            metric_signal_extractor=self._extract_metric_signals_from_text,
            training_context_checker=self._has_training_state_context,
            run_id_extractor=self._extract_training_run_ids_from_text,
        )

    async def _resolve_mainline_route_state(self, user_text: str, mainline_context: dict[str, Any]) -> dict[str, Any]:
        normalized_text = str(mainline_context.get('normalized_text') or '')
        metric_signals = list(mainline_context.get('metric_signals') or [])
        has_training_context = bool(mainline_context.get('has_training_context'))
        explicit_run_ids = list(mainline_context.get('explicit_run_ids') or [])
        mainline_signals = resolve_mainline_request_signals(
            session_state=self.session_state,
            user_text=user_text,
            normalized_text=normalized_text,
        )
        loop_route = await self._resolve_training_loop_route(
            user_text=user_text,
            normalized_text=normalized_text,
            wants_predict=bool(mainline_signals.get('wants_predict')),
            wants_stop_training=bool(mainline_signals.get('wants_stop_training')),
            explicit_run_ids=explicit_run_ids,
        )
        return resolve_mainline_route_state_payload(
            session_state=self.session_state,
            user_text=user_text,
            normalized_text=normalized_text,
            has_training_context=has_training_context,
            mainline_signals=mainline_signals,
            metric_signals=metric_signals,
            explicit_run_ids=explicit_run_ids,
            loop_route=loop_route,
        )

    async def _try_handle_mainline_intent(
        self,
        user_text: str,
        thread_id: str,
        *,
        skip_training_plan_dialogue: bool = False,
    ) -> dict[str, Any] | None:
        self._sync_training_workflow_state(reason='route_eval')
        guardrail = self._try_handle_guardrail_intent(user_text)
        if guardrail is not None:
            return guardrail
        if not skip_training_plan_dialogue:
            plan_dialogue = await self._try_handle_training_plan_dialogue(user_text, thread_id)
            if plan_dialogue is _DEFER_TO_GRAPH:
                return None
            if plan_dialogue is not None:
                return plan_dialogue
        mainline_context = self._collect_mainline_context(user_text)
        route_state = await self._resolve_mainline_route_state(user_text, mainline_context)
        guard_reply = str(route_state.get('guard_reply') or '')
        if guard_reply:
            reply = guard_reply
            self._messages.append(AIMessage(content=reply))
            return {'status': 'completed', 'message': reply, 'tool_call': None}

        return await self._dispatch_mainline_requests(
            thread_id=thread_id,
            user_text=user_text,
            mainline_context=mainline_context,
            route_state=route_state,
        )

    def _try_handle_guardrail_intent(self, user_text: str) -> dict[str, Any] | None:
        reply = resolve_mainline_guardrail_reply(user_text=user_text, normalized_text=user_text.lower())
        if not reply:
            return None
        self._messages.append(AIMessage(content=reply))
        return {'status': 'completed', 'message': reply, 'tool_call': None}

    async def _execute_remote_prediction_pipeline(self, pipeline_args: dict[str, Any]) -> dict[str, Any]:
        flow_result = await run_remote_prediction_pipeline_flow(
            pipeline_args=pipeline_args,
            direct_tool=self.direct_tool,
            resolve_prediction_remote_inputs=self._resolve_prediction_remote_inputs,
            build_remote_output_dir=lambda upload_result: self._remote_join(
                str(upload_result.get('remote_root') or ''),
                f'_agent_prediction_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            ),
        )
        upload_result = dict(flow_result.get('upload') or {})
        resolved_inputs = dict(flow_result.get('resolved_inputs') or {})
        predict_result = dict(flow_result.get('predict') or {})
        download_result = dict(flow_result.get('download') or {})
        stage = str(flow_result.get('stage') or '').strip()
        if stage == 'upload':
            reply = await self._render_multi_tool_result_message(
                [('upload_assets_to_remote', upload_result)],
                objective='远端预测闭环失败说明',
            ) or upload_result.get('error') or '远端上传失败'
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_prediction_pipeline', 'args': pipeline_args}}

        if stage == 'resolve':
            reply = await self._render_multi_tool_result_message(
                [('upload_assets_to_remote', upload_result)],
                objective='远端预测闭环失败说明',
                extra_notes=[str(resolved_inputs.get('error') or '').strip()],
            ) or str(resolved_inputs.get('error') or '远端预测闭环参数不完整')
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_prediction_pipeline', 'args': pipeline_args}}

        predict_tool_name = str(resolved_inputs.get('tool_name') or '')
        if stage == 'predict':
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

        pipeline_result = dict(flow_result.get('pipeline_result') or {})
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
        flow_result = await run_remote_training_pipeline_flow(
            pipeline_args=pipeline_args,
            direct_tool=self.direct_tool,
            resolve_training_remote_inputs=self._resolve_training_remote_inputs,
            collect_requested_training_args=self._collect_requested_training_args,
            wait_for_remote_training_terminal_state=lambda **kwargs: self._wait_for_remote_training_terminal_state(**kwargs),
            resolve_remote_training_result_path=lambda **kwargs: self._resolve_remote_training_result_path(**kwargs),
        )
        upload_result = dict(flow_result.get('upload') or {})
        resolved_inputs = dict(flow_result.get('resolved_inputs') or {})
        start_flow = dict(flow_result.get('start_flow') or {})
        wait_result = dict(flow_result.get('wait') or {})
        download_result = dict(flow_result.get('download') or {})
        stage = str(flow_result.get('stage') or '').strip()
        if stage == 'upload':
            reply = await self._render_multi_tool_result_message(
                [('upload_assets_to_remote', upload_result)],
                objective='远端训练闭环失败说明',
            ) or upload_result.get('error') or '远端上传失败'
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_training_pipeline', 'args': pipeline_args}}

        if stage == 'resolve':
            reply = await self._render_multi_tool_result_message(
                [('upload_assets_to_remote', upload_result)],
                objective='远端训练闭环失败说明',
                extra_notes=[str(resolved_inputs.get('error') or '').strip()],
            ) or str(resolved_inputs.get('error') or '远端训练闭环参数不完整')
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_training_pipeline', 'args': pipeline_args}}

        readiness = dict(start_flow.get('readiness') or {})
        prepare_result = dict(start_flow.get('prepare') or {})
        preflight = dict(start_flow.get('preflight') or {})
        start_result = dict(start_flow.get('start') or {})
        start_stage = str(start_flow.get('stage') or '').strip()
        if stage == 'start':
            reply = await self._render_multi_tool_result_message(
                [
                    ('upload_assets_to_remote', upload_result),
                    ('training_readiness', readiness),
                    ('prepare_dataset_for_training', prepare_result) if prepare_result else ('', {}),
                    ('training_preflight', preflight) if preflight else ('', {}),
                    ('start_training', start_result) if start_result else ('', {}),
                ],
                objective='远端训练闭环失败说明',
            ) or (
                readiness.get('error')
                if start_stage == 'readiness'
                else prepare_result.get('summary') or prepare_result.get('error')
                if start_stage == 'prepare'
                else preflight.get('summary') or preflight.get('error')
                if start_stage == 'preflight'
                else start_result.get('summary') or start_result.get('error')
                if start_stage == 'start'
                else '远端训练启动失败'
            )
            self._messages.append(AIMessage(content=reply))
            self._trim_history()
            self.memory.save_state(self.session_state)
            return {'status': 'error', 'message': reply, 'tool_call': {'name': 'remote_training_pipeline', 'args': pipeline_args}}

        final_status: dict[str, Any] = {}
        final_summary: dict[str, Any] = {}
        if wait_result:
            final_status = dict(wait_result.get('status_result') or {})
            final_summary = dict(wait_result.get('summary_result') or {})
        pipeline_result = dict(flow_result.get('pipeline_result') or {})

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
        return await wait_for_remote_training_terminal_state_service(
            direct_tool=self.direct_tool,
            poll_interval_seconds=poll_interval_seconds,
            max_wait_seconds=max_wait_seconds,
            sleep=asyncio.sleep,
        )

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

    def _complete_training_plan_followup_text(
        self,
        reply: str,
        *,
        status: str = 'completed',
    ) -> dict[str, Any]:
        result = self._complete_dialogue_text(reply)
        result['status'] = str(status or 'completed')
        return result

    async def _apply_training_plan_followup_action(
        self,
        *,
        followup_action: TrainingPlanFollowupAction | None,
        thread_id: str | None = None,
        user_text: str = '',
        handoff_mode: str = 'defer',
        pending: dict[str, Any] | None = None,
        draft: dict[str, Any] | None = None,
    ) -> dict[str, Any] | object | None:
        followup_action = dict(followup_action or {})
        pending = dict(pending or {})
        draft = dict(draft or {})
        thread_id = str(thread_id or '')
        action = str(followup_action.get('action') or '').strip()
        if action == 'none':
            return None
        if action == 'cancel_pending':
            return await self.confirm(thread_id, approved=False)
        if action == 'cancel_draft':
            self._clear_training_plan_draft()
            self.memory.save_state(self.session_state)
            return {'status': 'cancelled', 'message': '已取消当前训练计划草案。', 'tool_call': None}
        if action == 'clear_and_recheck':
            self._clear_training_plan_draft()
            self.memory.save_state(self.session_state)
            return None
        if action == 'confirmation_message':
            if draft and thread_id:
                return await self._enter_graph_training_confirmation(
                    draft=draft,
                    thread_id=thread_id,
                    user_text_hint=user_text or '请进入当前训练确认。',
                )
            return self._needs_confirmation_result(thread_id, pending, await self._build_confirmation_message(pending))
        if action == 'reply_with_pending':
            reply = str(followup_action.get('reply') or '').strip()
            self._messages.append(AIMessage(content=reply))
            return self._needs_confirmation_result(thread_id, pending, reply)
        if action == 'reply':
            return self._complete_dialogue_text(str(followup_action.get('reply') or ''))
        if action == 'defer_to_graph':
            return _DEFER_TO_GRAPH
        if action == 'render_plan':
            if not draft:
                return None
            if thread_id:
                return await self._enter_graph_training_confirmation(
                    draft=draft,
                    thread_id=thread_id,
                    user_text_hint=user_text or '请按当前训练方案进入确认。',
                )
            return await self._render_training_plan_dialogue_response(
                draft=draft,
                pending=pending,
                thread_id=thread_id,
                preamble=str(followup_action.get('preamble') or ''),
                append_message=bool(followup_action.get('append_message')),
            )
        if action == 'render_original_plan':
            if draft and thread_id:
                return await self._enter_graph_training_confirmation(
                    draft=draft,
                    thread_id=thread_id,
                    user_text_hint=user_text or '请回到原始训练方案确认。',
                )
            return await self._render_training_plan_dialogue_response(
                draft=draft,
                pending=pending,
                thread_id=thread_id,
                preamble=str(followup_action.get('preamble') or ''),
                append_message=True,
            )
        if action == 'approve_pending':
            return await self.confirm(thread_id, approved=True)
        if action == 'render_draft':
            if draft and thread_id:
                return await self._enter_graph_training_confirmation(
                    draft=draft,
                    thread_id=thread_id,
                    user_text_hint=user_text or '请按当前训练草案进入确认。',
                )
            return await self._render_training_plan_dialogue_response(
                draft=draft,
                pending=None,
                thread_id=thread_id,
            )
        if action == 'noop':
            return None
        if action == 'build_plan':
            return None
        if action == 'refresh_confirmation':
            if draft and thread_id:
                return await self._enter_graph_training_confirmation(
                    draft=draft,
                    thread_id=thread_id,
                    user_text_hint='请按更新后的训练草案进入确认。',
                )
            if self._should_refresh_training_plan_confirmation_locally(pending, draft):
                return await self._refresh_training_plan_confirmation_locally(
                    thread_id=thread_id,
                    pending=pending,
                    revised_draft=draft,
                    user_text_hint='请按更新后的训练草案进入确认。',
                )
            return await self._refresh_training_plan_confirmation_via_graph(
                thread_id=thread_id,
                pending=pending,
                revised_draft=draft,
                user_text_hint='请按更新后的训练草案进入确认。',
            )
        if action == 'render_completed':
            self.memory.save_state(self.session_state)
            return await self._render_training_plan_dialogue_response(
                draft=draft,
                pending=None,
                thread_id=thread_id,
            )
        if action == 'save_draft_and_reply' or action == 'save_draft_and_handoff':
            draft = dict(followup_action.get('draft') or {})
            if draft:
                self._save_training_plan_draft(draft)
        elif action == 'clear_draft_and_reply':
            self._clear_training_plan_draft()

        if action == 'save_draft_and_handoff':
            if draft and thread_id:
                return await self._enter_graph_training_confirmation(
                    draft=draft,
                    thread_id=thread_id,
                    user_text_hint=user_text,
                )
            if handoff_mode == 'handoff' and thread_id:
                return await self._handoff_current_runtime_to_graph(
                    thread_id=thread_id,
                    user_text_hint=user_text,
                    auto_approve=False,
                )
            return _DEFER_TO_GRAPH

        if action == 'clear_draft_and_reply' or action == 'save_draft_and_reply':
            return self._complete_training_plan_followup_text(
                str(followup_action.get('reply') or ''),
                status=str(followup_action.get('status') or 'completed'),
            )
        return None

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

    def _startup_pending_from_session(self) -> dict[str, Any] | None:
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
        legacy_pending = self._startup_pending_from_session()
        if not legacy_pending:
            self._pending_confirmation_shadow = None
            self._pending_review_shadow = {}
            self._mirror_pending_confirmation(None)
            return
        if str(legacy_pending.get('source') or 'synthetic').strip().lower() == 'graph':
            self._pending_confirmation_shadow = None
            self._pending_review_shadow = {}
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

    def _graph_training_plan_context(self, config: dict[str, Any]) -> dict[str, Any] | None:
        state = self._graph_state_snapshot(config)
        values = getattr(state, 'values', {}) if state else {}
        if not isinstance(values, dict):
            return None
        return extract_training_plan_context_from_state(values)

    def _restore_training_plan_draft_from_context(
        self,
        *,
        config: dict[str, Any],
        pending: dict[str, Any],
    ) -> None:
        existing_draft = dict(self.session_state.active_training.training_plan_draft or {})
        context = self._graph_training_plan_context(config) or {}
        next_step_tool = str(context.get('next_step_tool') or '').strip()
        pending_tool = str(pending.get('name') or pending.get('tool_name') or '').strip()
        if not next_step_tool or next_step_tool != pending_tool:
            return
        draft_meaningful_keys = (
            'dataset_path',
            'execution_mode',
            'reasoning_summary',
            'data_summary',
            'preflight_summary',
            'planned_training_args',
            'command_preview',
            'warnings',
            'risks',
            'blockers',
        )
        restored_draft = {
            'stage': str(context.get('stage') or ''),
            'status': str(context.get('status') or ''),
            'dataset_path': str(context.get('dataset_path') or ''),
            'execution_mode': str(context.get('execution_mode') or ''),
            'execution_backend': str(context.get('execution_backend') or ''),
            'training_environment': str(context.get('training_environment') or ''),
            'advanced_details_requested': bool(context.get('advanced_details_requested')),
            'reasoning_summary': str(context.get('reasoning_summary') or ''),
            'data_summary': str(context.get('data_summary') or ''),
            'preflight_summary': str(context.get('preflight_summary') or ''),
            'next_step_tool': next_step_tool,
            'next_step_args': dict(context.get('next_step_args') or {}),
            'planned_training_args': dict(context.get('planned_training_args') or {}),
            'command_preview': list(context.get('command_preview') or []),
            'blockers': [str(item).strip() for item in (context.get('blockers') or []) if str(item).strip()],
            'warnings': [str(item).strip() for item in (context.get('warnings') or []) if str(item).strip()],
            'risks': [str(item).strip() for item in (context.get('risks') or []) if str(item).strip()],
        }
        if not any(restored_draft.get(key) for key in draft_meaningful_keys):
            return
        if existing_draft and all(existing_draft.get(key) == restored_draft.get(key) for key in restored_draft):
            return
        self._save_training_plan_draft(restored_draft)
        self.memory.append_event(
            self.session_state.session_id,
            'startup_training_plan_draft_restored',
            {
                'next_step_tool': next_step_tool,
                'dataset_path': str(restored_draft.get('dataset_path') or '').strip(),
                'status': str(restored_draft.get('status') or '').strip(),
                'replaced_existing': bool(existing_draft),
            },
        )

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

    @staticmethod
    def _graph_interrupt_payloads(state: Any) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        if state is None:
            return payloads
        interrupts = list(getattr(state, 'interrupts', ()) or ())
        for task in getattr(state, 'tasks', ()) or ():
            interrupts.extend(list(getattr(task, 'interrupts', ()) or ()))
        for item in interrupts:
            raw_value = getattr(item, 'value', item)
            if isinstance(raw_value, dict):
                payloads.append(dict(raw_value))
        return payloads

    def _active_graph_interrupt(self, *, preferred_thread_id: str = '') -> tuple[str, dict[str, Any]] | None:
        candidate_thread_ids: list[str] = []
        for candidate in (
            preferred_thread_id,
            str((self._pending_confirmation_shadow or {}).get('thread_id') or '').strip(),
            str(self.session_state.pending_confirmation.thread_id or '').strip(),
        ):
            if candidate:
                candidate_thread_ids.append(candidate)
        candidate_thread_ids.extend(self._startup_checkpoint_thread_ids())
        if not candidate_thread_ids and preferred_thread_id:
            candidate_thread_ids.append(preferred_thread_id)
        seen: set[str] = set()
        for thread_id in candidate_thread_ids:
            if not thread_id or thread_id in seen:
                continue
            seen.add(thread_id)
            state = self._graph_state_snapshot(self._pending_config(thread_id))
            if state is None or not getattr(state, 'next', None):
                continue
            payloads = self._graph_interrupt_payloads(state)
            if payloads:
                return thread_id, payloads[0]
        return None

    @staticmethod
    def _is_training_confirmation_interrupt_payload(payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        kind = str(payload.get('type') or payload.get('interrupt_kind') or '').strip().lower()
        return kind == 'training_confirmation'

    def _training_confirmation_tool_call(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        plan_payload = dict(payload.get('plan') or {})
        if not plan_payload:
            return None
        phase = str(payload.get('phase') or 'prepare').strip().lower() or 'prepare'
        try:
            plan = coerce_training_plan(plan_payload)
            plan_args = self._model_dump_compat(plan)
            explicit_next_step_args = dict(payload.get('next_step_args') or {})
            if phase == 'prepare':
                return {
                    'name': 'prepare_dataset_for_training',
                    'args': explicit_next_step_args or {'dataset_path': plan.dataset_path},
                }
            if getattr(plan, 'mode', 'train') == 'loop':
                return {'name': 'start_training_loop', 'args': explicit_next_step_args or plan_args}
            return {'name': 'start_training', 'args': explicit_next_step_args or plan_args}
        except Exception:
            return None

    def _pending_action_from_training_confirmation_interrupt(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        tool_call = self._training_confirmation_tool_call(payload)
        if not tool_call:
            return None
        pending = {
            'name': str(tool_call.get('name') or '').strip(),
            'args': dict(tool_call.get('args') or {}),
            'thread_id': str(payload.get('thread_id') or '').strip(),
            'source': 'synthetic',
            'decision_context': dict(self._pending_review_shadow or {}),
        }
        return self._build_pending_action_payload(
            pending,
            thread_id=str(payload.get('thread_id') or '').strip() or None,
        )

    def _render_plan_for_cli(self, plan_payload: dict[str, Any], phase: str = '', execution_mode: str = '', status_reply: str = '') -> str:
        plan = coerce_training_plan(dict(plan_payload or {}))
        phase_text = '数据准备确认' if str(phase or '').strip().lower() == 'prepare' else '训练启动确认'
        lines: list[str] = []
        status_reply = str(status_reply or '').strip()
        if status_reply:
            lines.append(status_reply)
            lines.append('')
        lines.append(f'{phase_text}：')
        mode = getattr(plan, 'mode', 'train')
        lines.append(f"- 类型: {'循环训练' if mode == 'loop' else '普通训练'}")
        lines.append(f'- 数据集: {plan.dataset_path}')
        lines.append(f'- 模型: {plan.model}')
        if mode == 'loop':
            lines.append(f'- 循环参数: max_rounds={getattr(plan, "max_rounds", "")}, epochs_per_round={getattr(plan, "epochs_per_round", "")}')
            loop_name = str(getattr(plan, 'loop_name', '') or '').strip()
            if loop_name:
                lines.append(f'- 循环名称: {loop_name}')
        else:
            lines.append(f'- 训练参数: epochs={getattr(plan, "epochs", "")}, batch={plan.batch}, imgsz={plan.imgsz}')
        if str(plan.device or '').strip():
            lines.append(f'- 设备: {plan.device}')
        if str(plan.training_environment or '').strip():
            lines.append(f'- 环境: {plan.training_environment}')
        if str(plan.data_yaml or '').strip():
            lines.append(f'- data.yaml: {plan.data_yaml}')
        execution_mode_map = {
            'prepare_then_train': '先准备再训练',
            'prepare_then_loop': '先准备再进入循环训练',
            'direct_train': '直接训练',
            'direct_loop': '直接启动循环训练',
            'prepare_only': '只做准备，暂不启动训练',
            'discussion_only': '先讨论方案，暂不执行',
            'blocked': '当前存在阻塞，先解决问题',
        }
        if execution_mode:
            lines.append(f"- 执行方式: {execution_mode_map.get(str(execution_mode).strip(), str(execution_mode).strip())}")
        blockers = [str(item).strip() for item in (getattr(plan, 'blockers', None) or []) if str(item).strip()]
        warnings = [str(item).strip() for item in (getattr(plan, 'warnings', None) or []) if str(item).strip()]
        if blockers:
            lines.append('- 当前阻塞:')
            lines.extend(f'  - {item}' for item in blockers[:3])
        elif warnings:
            lines.append('- 主要风险:')
            lines.extend(f'  - {item}' for item in warnings[:3])
        summary = str(getattr(plan, 'prepare_summary', '') or getattr(plan, 'readiness_summary', '') or '').strip()
        if summary:
            lines.append(f'- 说明: {summary}')
        lines.append('你可以直接确认、改参数、追问原因，或者切去处理别的新任务。')
        return '\n'.join(lines)

    def _format_interrupt_or_result(
        self,
        result: dict[str, Any],
        *,
        thread_id: str,
        interrupt_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = interrupt_payload
        if payload is None:
            interrupts = result.get('__interrupt__')
            if isinstance(interrupts, list):
                for item in interrupts:
                    raw_value = getattr(item, 'value', item)
                    if isinstance(raw_value, dict):
                        payload = dict(raw_value)
                        break
        if self._is_training_confirmation_interrupt_payload(payload):
            self._sync_training_draft_from_interrupt_payload(dict(payload or {}))
            message = self._render_plan_for_cli(
                dict((payload or {}).get('plan') or {}),
                str((payload or {}).get('phase') or ''),
                str((payload or {}).get('execution_mode') or ''),
                status_reply=str((payload or {}).get('status_reply') or '').strip(),
            )
            pending_action = self._pending_action_from_training_confirmation_interrupt(payload or {})
            if message:
                self._messages.append(AIMessage(content=message))
            return {
                'status': 'needs_confirmation',
                'message': message,
                'tool_call': self._training_confirmation_tool_call(payload or {}),
                'thread_id': thread_id,
                'interrupt_payload': dict(payload or {}),
                'pending_action': pending_action,
            }
        messages = list(result.get('messages') or [])
        final_text = self._extract_or_fallback(messages)
        if final_text:
            self._messages.append(AIMessage(content=final_text))
        return {
            'status': 'completed',
            'message': final_text,
            'tool_call': None,
        }

    def _draft_to_training_confirmation_interrupt(
        self,
        draft: dict[str, Any],
        *,
        thread_id: str,
    ) -> dict[str, Any] | None:
        draft = dict(draft or {})
        next_step_tool = str(draft.get('next_step_tool') or '').strip()
        if next_step_tool not in {'prepare_dataset_for_training', 'start_training', 'start_training_loop'}:
            return None
        planned_training_args = dict(draft.get('planned_training_args') or {})
        planned_loop_args = dict(draft.get('planned_loop_args') or planned_training_args)
        dataset_path = str(draft.get('dataset_path') or self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir or '').strip()
        model = str(planned_training_args.get('model') or planned_loop_args.get('model') or self.session_state.active_training.model or '').strip()
        if not dataset_path or not model:
            return None
        phase = 'prepare' if next_step_tool == 'prepare_dataset_for_training' else 'start'
        mode = 'loop' if next_step_tool == 'start_training_loop' or 'loop' in str(draft.get('execution_mode') or '').strip().lower() else 'train'
        batch_value = planned_training_args.get('batch')
        if batch_value in (None, ''):
            batch_value = planned_loop_args.get('batch')
        if batch_value in (None, ''):
            batch_value = 16
        imgsz_value = planned_training_args.get('imgsz')
        if imgsz_value in (None, ''):
            imgsz_value = planned_loop_args.get('imgsz')
        if imgsz_value in (None, ''):
            imgsz_value = 640
        plan_payload: dict[str, Any] = {
            'mode': mode,
            'dataset_path': dataset_path,
            'model': model,
            'batch': batch_value,
            'imgsz': imgsz_value,
            'device': str(planned_training_args.get('device') or planned_loop_args.get('device') or draft.get('device') or '').strip(),
            'training_environment': str(draft.get('training_environment') or '').strip(),
            'data_yaml': str(planned_training_args.get('data_yaml') or planned_loop_args.get('data_yaml') or self.session_state.active_training.data_yaml or '').strip(),
            'project': str(planned_training_args.get('project') or planned_loop_args.get('project') or '').strip(),
            'name': str(planned_training_args.get('name') or planned_loop_args.get('name') or '').strip(),
            'fraction': planned_training_args.get('fraction', planned_loop_args.get('fraction')),
            'classes': list(planned_training_args.get('classes') or planned_loop_args.get('classes') or []),
            'single_cls': planned_training_args.get('single_cls', planned_loop_args.get('single_cls')),
            'optimizer': str(planned_training_args.get('optimizer') or planned_loop_args.get('optimizer') or '').strip(),
            'freeze': planned_training_args.get('freeze', planned_loop_args.get('freeze')),
            'resume': planned_training_args.get('resume', planned_loop_args.get('resume')),
            'lr0': planned_training_args.get('lr0', planned_loop_args.get('lr0')),
            'patience': planned_training_args.get('patience', planned_loop_args.get('patience')),
            'workers': planned_training_args.get('workers', planned_loop_args.get('workers')),
            'amp': planned_training_args.get('amp', planned_loop_args.get('amp')),
            'blockers': [str(item).strip() for item in (draft.get('blockers') or []) if str(item).strip()],
            'warnings': [str(item).strip() for item in (draft.get('warnings') or []) if str(item).strip()],
            'readiness_summary': str(draft.get('preflight_summary') or draft.get('data_summary') or '').strip(),
            'prepare_summary': str(draft.get('reasoning_summary') or '').strip(),
        }
        if mode == 'loop':
            plan_payload.update(
                {
                    'max_rounds': planned_loop_args.get('max_rounds') or 5,
                    'epochs_per_round': planned_loop_args.get('epochs_per_round') or planned_loop_args.get('epochs') or 10,
                    'loop_name': str(planned_loop_args.get('loop_name') or planned_loop_args.get('name') or '').strip(),
                }
            )
        else:
            plan_payload['epochs'] = planned_training_args.get('epochs') or 100
        try:
            normalized_plan = self._model_dump_compat(coerce_training_plan(plan_payload))
        except Exception:
            normalized_plan = plan_payload
        return {
            'type': 'training_confirmation',
            'phase': phase,
            'plan': normalized_plan,
            'execution_mode': str(draft.get('execution_mode') or '').strip(),
            'next_step_tool': next_step_tool,
            'next_step_args': dict(draft.get('next_step_args') or {}),
            'thread_id': thread_id,
        }

    def _sync_training_draft_from_interrupt_payload(self, payload: dict[str, Any]) -> None:
        draft = dict(self.session_state.active_training.training_plan_draft or {})
        plan = coerce_training_plan(dict(payload.get('plan') or {}))
        phase = str(payload.get('phase') or 'prepare').strip().lower() or 'prepare'
        next_step_tool = 'prepare_dataset_for_training' if phase == 'prepare' else ('start_training_loop' if getattr(plan, 'mode', 'train') == 'loop' else 'start_training')
        draft['dataset_path'] = plan.dataset_path
        draft['training_environment'] = str(getattr(plan, 'training_environment', '') or '').strip()
        draft['blockers'] = [str(item).strip() for item in (getattr(plan, 'blockers', None) or []) if str(item).strip()]
        draft['warnings'] = [str(item).strip() for item in (getattr(plan, 'warnings', None) or []) if str(item).strip()]
        draft['preflight_summary'] = str(getattr(plan, 'readiness_summary', '') or '').strip()
        draft['reasoning_summary'] = str(getattr(plan, 'prepare_summary', '') or '').strip()
        draft['next_step_tool'] = str(payload.get('next_step_tool') or next_step_tool).strip() or next_step_tool
        draft['next_step_args'] = dict(payload.get('next_step_args') or draft.get('next_step_args') or {})
        draft['execution_mode'] = str(payload.get('execution_mode') or '').strip() or (
            'prepare_then_loop' if next_step_tool == 'prepare_dataset_for_training' and getattr(plan, 'mode', 'train') == 'loop' else (
                'prepare_then_train' if next_step_tool == 'prepare_dataset_for_training' else (
                    'direct_loop' if getattr(plan, 'mode', 'train') == 'loop' else 'direct_train'
                )
            )
        )
        if getattr(plan, 'mode', 'train') == 'loop':
            draft['planned_loop_args'] = {
                'model': plan.model,
                'data_yaml': str(getattr(plan, 'data_yaml', '') or '').strip(),
                'batch': getattr(plan, 'batch', None),
                'imgsz': getattr(plan, 'imgsz', None),
                'device': str(getattr(plan, 'device', '') or '').strip(),
                'project': str(getattr(plan, 'project', '') or '').strip(),
                'name': str(getattr(plan, 'name', '') or '').strip(),
                'fraction': getattr(plan, 'fraction', None),
                'classes': list(getattr(plan, 'classes', None) or []),
                'single_cls': getattr(plan, 'single_cls', None),
                'optimizer': str(getattr(plan, 'optimizer', '') or '').strip(),
                'freeze': getattr(plan, 'freeze', None),
                'resume': getattr(plan, 'resume', None),
                'lr0': getattr(plan, 'lr0', None),
                'patience': getattr(plan, 'patience', None),
                'workers': getattr(plan, 'workers', None),
                'amp': getattr(plan, 'amp', None),
                'max_rounds': getattr(plan, 'max_rounds', None),
                'epochs_per_round': getattr(plan, 'epochs_per_round', None),
                'loop_name': str(getattr(plan, 'loop_name', '') or '').strip(),
            }
            draft['planned_training_args'] = dict(draft['planned_loop_args'])
        else:
            draft['planned_training_args'] = {
                'model': plan.model,
                'data_yaml': str(getattr(plan, 'data_yaml', '') or '').strip(),
                'batch': getattr(plan, 'batch', None),
                'imgsz': getattr(plan, 'imgsz', None),
                'device': str(getattr(plan, 'device', '') or '').strip(),
                'project': str(getattr(plan, 'project', '') or '').strip(),
                'name': str(getattr(plan, 'name', '') or '').strip(),
                'fraction': getattr(plan, 'fraction', None),
                'classes': list(getattr(plan, 'classes', None) or []),
                'single_cls': getattr(plan, 'single_cls', None),
                'optimizer': str(getattr(plan, 'optimizer', '') or '').strip(),
                'freeze': getattr(plan, 'freeze', None),
                'resume': getattr(plan, 'resume', None),
                'lr0': getattr(plan, 'lr0', None),
                'patience': getattr(plan, 'patience', None),
                'workers': getattr(plan, 'workers', None),
                'amp': getattr(plan, 'amp', None),
                'epochs': getattr(plan, 'epochs', None),
            }
        self._save_training_plan_draft(draft)

    async def _enter_graph_training_confirmation(
        self,
        *,
        draft: dict[str, Any],
        thread_id: str,
        user_text_hint: str,
        stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        draft = dict(draft or {})
        handoff = await self._handoff_current_runtime_to_graph(
            thread_id=thread_id,
            user_text_hint=user_text_hint,
            auto_approve=False,
            suppress_training_plan_context=False,
            training_plan_context_override=build_training_plan_context_from_draft(draft),
            skip_text_only_training_confirmation=True,
            stream_handler=stream_handler,
        )
        if self._is_training_confirmation_interrupt_payload(dict(handoff.get('interrupt_payload') or {})):
            return handoff
        interrupt_payload = self._draft_to_training_confirmation_interrupt(draft, thread_id=thread_id)
        if interrupt_payload is not None:
            return self._format_interrupt_or_result({}, thread_id=thread_id, interrupt_payload=interrupt_payload)
        return handoff

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
        merged_pending = self._merge_pending_review_context(pending, self._pending_review_shadow)
        return {
            'status': 'needs_confirmation',
            'message': message,
            'tool_call': {'name': merged_pending['name'], 'args': merged_pending.get('args', {})},
            'thread_id': thread_id,
            'pending_action': self._build_pending_action_payload(merged_pending, thread_id=thread_id),
        }

    def _cancelled_result(self, pending: dict[str, Any], message: str) -> dict[str, Any]:
        merged_pending = self._merge_pending_review_context(pending, self._pending_review_shadow)
        payload = self._build_pending_action_payload(merged_pending, decision_state='rejected')
        return {
            'status': 'cancelled',
            'message': message,
            'tool_call': {'name': merged_pending['name'], 'args': merged_pending.get('args', {})},
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
        merged_pending = self._merge_pending_review_context(pending, self._pending_review_shadow)
        payload = self._build_pending_action_payload(merged_pending, thread_id=thread_id)
        source = str(
            merged_pending.get('source')
            or ('synthetic' if merged_pending.get('synthetic') else 'graph')
        ).strip().lower()
        if source not in {'graph', 'synthetic'}:
            source = 'synthetic'
        normalized = {
            'id': merged_pending.get('id'),
            'tool_call_id': str(merged_pending.get('tool_call_id') or merged_pending.get('id') or '').strip(),
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

    def _update_pending_confirmation_args(self, thread_id: str, updates: dict[str, Any]) -> bool:
        pending = self._resolve_pending_confirmation(thread_id=thread_id, config=self._pending_config(thread_id))
        if not pending:
            return False
        normalized_updates = dict(updates or {})
        if not normalized_updates:
            return True
        updated_pending = dict(pending)
        updated_args = dict(updated_pending.get('args') or updated_pending.get('tool_args') or {})
        updated_args.update(normalized_updates)
        updated_pending['args'] = dict(updated_args)
        updated_pending['tool_args'] = dict(updated_args)
        self._remember_pending_confirmation(
            updated_pending,
            emit_event=False,
            persist_graph=False,
        )
        resolved_thread_id = str(updated_pending.get('thread_id') or thread_id or '').strip()
        if resolved_thread_id:
            self._update_graph_pending_state(
                thread_id=resolved_thread_id,
                pending=updated_pending,
                review=self._pending_review_shadow,
            )
        self.memory.save_state(self.session_state)
        return True

    def _clear_pending_confirmation(self, *, thread_id: str = '', persist_graph: bool = True) -> None:
        resolved_thread_id = str(
            thread_id
            or (self._pending_confirmation_shadow or {}).get('thread_id')
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
        return None

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
            if callable(getattr(self.graph, 'ainvoke', None)):
                return pending
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
            parsed = parse_tool_message(message)
            tool_name = canonical_tool_name(message.name or "unknown_tool")
            tool_args = normalize_tool_args(tool_name, tool_args_by_id.get(message.tool_call_id or '', {}))
            apply_signature = ''
            if message.tool_call_id:
                apply_signature = json.dumps(
                    {
                        'tool_call_id': str(message.tool_call_id or ''),
                        'tool': tool_name,
                        'args': tool_args,
                        'result': parsed,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if apply_signature in self._applied_tool_call_ids:
                    continue
            self.memory.append_event(self.session_state.session_id, "tool_result", {"tool": tool_name, "args": tool_args, "result": parsed})
            self._apply_to_state(tool_name, parsed, tool_args)
            self._record_secondary_event(tool_name, parsed)
            applied_results.append((tool_name, parsed))
            if apply_signature:
                self._applied_tool_call_ids.add(apply_signature)
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

    async def _invoke_graph_from_current_runtime(
        self,
        *,
        thread_id: str,
        user_text_hint: str,
        suppress_training_plan_context: bool = False,
        training_plan_context_override: dict[str, Any] | None = None,
        suppress_ephemeral_state_context: bool = False,
        stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], int]:
        config = {"configurable": {"thread_id": thread_id}}
        self._trim_history()
        if suppress_ephemeral_state_context:
            stripped_state = self._strip_ephemeral_context(SessionState.from_dict(self.session_state.to_dict()))
            state_for_model, include_history_context = stripped_state, False
            recent_messages_for_model = list(self._messages[-1:]) if self._messages else []
        else:
            state_for_model, include_history_context, recent_messages_for_model = self._state_for_model(user_text_hint)
        digest = self.event_retriever.build_digest(
            self.session_state.session_id,
            state_for_model,
            include_history_context=include_history_context,
        )
        built_messages = self.context_builder.build_messages(state_for_model, recent_messages_for_model, digest=digest)
        graph_input = {
            "messages": built_messages,
            "cached_tool_context": build_cached_tool_context_payload(self.session_state),
            "dataset_fact_context": build_dataset_fact_context_payload(self.session_state),
            "training_plan_context": (
                None
                if suppress_training_plan_context
                else dict(training_plan_context_override or build_training_plan_context_payload(self.session_state) or {})
            ) or None,
        }
        result = await self._graph_invoke(graph_input, config=config, stream_handler=stream_handler)
        built_messages_len = len(built_messages)
        self._record_graph_selected_tools(result["messages"], thread_id=thread_id, built_messages_len=built_messages_len)
        self._record_tool_error_recovery(result["messages"], thread_id=thread_id, built_messages_len=built_messages_len)
        self._record_graph_text_response(result["messages"], thread_id=thread_id, built_messages_len=built_messages_len)
        return result, config, built_messages_len

    async def _handoff_current_runtime_to_graph(
        self,
        *,
        thread_id: str,
        user_text_hint: str,
        auto_approve: bool = False,
        ignore_existing_pending: bool = False,
        suppress_training_plan_context: bool = False,
        training_plan_context_override: dict[str, Any] | None = None,
        suppress_ephemeral_state_context: bool = False,
        skip_text_only_training_confirmation: bool = False,
        stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        graph_result, config, built_messages_len = await self._invoke_graph_from_current_runtime(
            thread_id=thread_id,
            user_text_hint=user_text_hint,
            suppress_training_plan_context=suppress_training_plan_context,
            training_plan_context_override=training_plan_context_override,
            suppress_ephemeral_state_context=suppress_ephemeral_state_context,
            stream_handler=stream_handler,
        )
        active_interrupt = self._active_graph_interrupt(preferred_thread_id=thread_id)
        if active_interrupt is not None:
            interrupt_thread_id, interrupt_payload = active_interrupt
            if self._is_training_confirmation_interrupt_payload(interrupt_payload):
                formatted = self._format_interrupt_or_result(
                    graph_result,
                    thread_id=interrupt_thread_id,
                    interrupt_payload=interrupt_payload,
                )
                self._trim_history()
                self.memory.save_state(self.session_state)
                return formatted
        if ignore_existing_pending:
            pending = self._graph_pending_from_values(config)
            if pending is None:
                pending = self._graph_pending_from_interrupt(config)
                if pending is not None and thread_id and not str(pending.get('thread_id') or '').strip():
                    pending['thread_id'] = thread_id
            if pending is not None:
                self._remember_pending_confirmation(pending, emit_event=False, persist_graph=False)
        else:
            pending = self._resolve_pending_confirmation(thread_id=thread_id, config=config)
        if pending is not None:
            if auto_approve or self._auto_confirmation_enabled():
                return await self.confirm(thread_id, approved=True, stream_handler=stream_handler)
            self.memory.save_state(self.session_state)
            return self._needs_confirmation_result(thread_id, pending, await self._build_confirmation_message(pending))

        applied_results = self._apply_tool_results(graph_result["messages"], built_messages_len=built_messages_len)
        final_text = self._compose_final_reply(graph_result["messages"], applied_results)
        if not skip_text_only_training_confirmation:
            materialized_confirmation = await self._maybe_materialize_text_only_training_confirmation(
                thread_id=thread_id,
                user_text_hint=user_text_hint,
                final_text=final_text,
                applied_results=applied_results,
                stream_handler=stream_handler,
            )
            if materialized_confirmation is not None:
                self._trim_history()
                self.memory.save_state(self.session_state)
                return materialized_confirmation
        self._messages.append(AIMessage(content=final_text))
        self._trim_history()
        self.memory.save_state(self.session_state)
        return {
            "status": "completed",
            "message": final_text,
            "tool_call": None,
        }

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

    async def _continue_followup_training_after_prepare(
        self,
        *,
        thread_id: str,
        synthetic_followup: dict[str, Any],
        prepare_parsed: dict[str, Any],
        stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        flow_result = await run_post_prepare_training_start_flow(
            user_text=self._recent_user_text(),
            dataset_path=self.session_state.active_dataset.dataset_root or self.session_state.active_dataset.img_dir,
            readiness=self.session_state.active_dataset.last_readiness,
            synthetic_followup=synthetic_followup,
            prepare_parsed=prepare_parsed,
            direct_tool=self.direct_tool,
            build_training_plan_draft_fn=self._build_training_plan_draft,
            render_prepare_followup_message=self._render_prepare_followup_message,
        )
        followup_action = dict(flow_result.get('followup_action') or {})
        followup_draft = dict(flow_result.get('draft') or followup_action.get('draft') or {})
        preflight = dict(flow_result.get('preflight') or {})
        self.memory.append_event(
            self.session_state.session_id,
            'post_prepare_followup_resolved',
            {
                'thread_id': thread_id,
                'action': str(followup_action.get('action') or '').strip(),
                'status': str(followup_action.get('status') or '').strip(),
                'next_step_tool': str(followup_draft.get('next_step_tool') or '').strip(),
                'ready_to_start': bool(preflight.get('ready_to_start')),
                'blocker_count': len([str(item).strip() for item in (preflight.get('blockers') or []) if str(item).strip()]),
                'warning_count': len([str(item).strip() for item in (preflight.get('warnings') or []) if str(item).strip()]),
            },
        )
        if self._should_localize_post_prepare_start_confirmation(
            followup_action=followup_action,
            draft=followup_draft,
        ):
            return await self._refresh_training_plan_confirmation_locally(
                thread_id=f"{thread_id}-post-prepare-start",
                pending=None,
                revised_draft=followup_draft,
                user_text_hint='当前数据已具备训练条件，确认后即可启动训练。',
            )
        result = await self._apply_training_plan_followup_action(
            followup_action=followup_action,
            thread_id=f"{thread_id}-post-prepare-start",
            user_text=self._recent_user_text(),
            handoff_mode='handoff',
        )
        if result is not None:
            return result
        return {
            "status": "error",
            "message": "",
            "tool_call": synthetic_followup,
            "approved": True,
        }

    @staticmethod
    def _should_localize_post_prepare_start_confirmation(
        *,
        followup_action: dict[str, Any] | None,
        draft: dict[str, Any] | None,
    ) -> bool:
        followup_action = dict(followup_action or {})
        draft = dict(draft or {})
        if str(followup_action.get('action') or '').strip() != 'save_draft_and_handoff':
            return False
        if canonical_tool_name(str(draft.get('next_step_tool') or '').strip()) != 'start_training':
            return False
        execution_backend = str(draft.get('execution_backend') or '').strip().lower()
        if execution_backend and execution_backend != 'standard_yolo':
            return False
        return True

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
        stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any] | None:
        loop_followup_result = await self._continue_training_loop_start_after_prepare(
            thread_id=thread_id,
            prepare_result=prepare_parsed,
        )
        if loop_followup_result is not None:
            return loop_followup_result
        loop_followup = self._build_followup_training_loop_request()
        if loop_followup:
            followup_thread_id = f"{thread_id}-post-prepare-loop"
            return await self._handoff_current_runtime_to_graph(
                thread_id=followup_thread_id,
                user_text_hint=self._recent_user_text(),
                auto_approve=False,
                stream_handler=stream_handler,
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
                stream_handler=stream_handler,
            )
        return None

    async def _refresh_training_plan_confirmation_via_graph(
        self,
        *,
        thread_id: str,
        pending: dict[str, Any] | None,
        revised_draft: dict[str, Any],
        user_text_hint: str,
    ) -> dict[str, Any]:
        prior_review_context = dict((pending or {}).get('decision_context') or self._pending_review_shadow)
        pending_thread_id = str(
            (pending or {}).get('thread_id')
            or self._pending_confirmation_thread_id()
            or thread_id
        ).strip()
        if pending_thread_id:
            self._clear_pending_confirmation(thread_id=pending_thread_id)
        handoff = await self._handoff_current_runtime_to_graph(
            thread_id=thread_id,
            user_text_hint=user_text_hint,
            auto_approve=False,
        )
        if handoff.get('status') != 'needs_confirmation':
            return handoff
        if self._is_training_confirmation_interrupt_payload(dict(handoff.get('interrupt_payload') or {})):
            return handoff
        refreshed_pending = self._pending_from_state() or {}
        if prior_review_context:
            refreshed_pending = self._merge_pending_review_context(refreshed_pending, prior_review_context)
            self._remember_pending_confirmation(
                refreshed_pending,
                emit_event=False,
                persist_graph=True,
            )
        confirmation_thread_id = str(
            refreshed_pending.get('thread_id')
            or handoff.get('thread_id')
            or thread_id
        ).strip()
        return self._needs_confirmation_result(
            confirmation_thread_id,
            refreshed_pending,
            await self._render_training_plan_message(revised_draft, pending=True),
        )

    @staticmethod
    def _should_refresh_training_plan_confirmation_locally(
        pending: dict[str, Any] | None,
        revised_draft: dict[str, Any] | None,
    ) -> bool:
        pending_tool = canonical_tool_name(str((pending or {}).get('name') or (pending or {}).get('tool_name') or '').strip())
        next_step_tool = canonical_tool_name(str((revised_draft or {}).get('next_step_tool') or '').strip())
        execution_backend = str((revised_draft or {}).get('execution_backend') or '').strip().lower()
        if pending_tool != 'prepare_dataset_for_training':
            return False
        if not next_step_tool:
            return False
        if execution_backend and execution_backend != 'standard_yolo':
            return False
        return True

    async def _refresh_training_plan_confirmation_locally(
        self,
        *,
        thread_id: str,
        pending: dict[str, Any] | None,
        revised_draft: dict[str, Any],
        user_text_hint: str,
    ) -> dict[str, Any]:
        rebuilt_draft = dict(revised_draft or {})
        next_tool_name = canonical_tool_name(str(rebuilt_draft.get('next_step_tool') or '').strip())
        next_tool_args = dict(rebuilt_draft.get('next_step_args') or {})
        if not next_tool_name:
            materialized = self._build_materialized_training_pending(
                draft=revised_draft,
                user_text=user_text_hint or self._recent_user_text(),
            )
            if materialized is None:
                return await self._refresh_training_plan_confirmation_via_graph(
                    thread_id=thread_id,
                    pending=pending,
                    revised_draft=revised_draft,
                    user_text_hint=user_text_hint,
                )
            rebuilt_draft, refreshed_pending = materialized
        else:
            if not next_tool_args and next_tool_name == 'prepare_dataset_for_training':
                dataset_path = str(
                    rebuilt_draft.get('dataset_path')
                    or self.session_state.active_dataset.dataset_root
                    or self.session_state.active_dataset.img_dir
                    or ''
                ).strip()
                if dataset_path:
                    next_tool_args = {'dataset_path': dataset_path}
                    rebuilt_draft['next_step_args'] = dict(next_tool_args)
            refreshed_pending = {
                'id': None,
                'name': next_tool_name,
                'args': dict(next_tool_args),
                'synthetic': True,
            }
        confirmation_thread_id = str(
            (pending or {}).get('thread_id')
            or self._pending_confirmation_thread_id()
            or thread_id
        ).strip() or thread_id
        self._save_training_plan_draft(rebuilt_draft)
        interrupt_payload = self._draft_to_training_confirmation_interrupt(rebuilt_draft, thread_id=confirmation_thread_id)
        self.memory.append_event(
            self.session_state.session_id,
            'training_plan_confirmation_refreshed_locally',
            {
                'thread_id': confirmation_thread_id,
                'tool': str(refreshed_pending.get('name') or ''),
                'execution_mode': str(rebuilt_draft.get('execution_mode') or ''),
            },
        )
        self.memory.save_state(self.session_state)
        if interrupt_payload is not None:
            return await self._enter_graph_training_confirmation(
                draft=rebuilt_draft,
                thread_id=confirmation_thread_id,
                user_text_hint='请按更新后的训练草案进入确认。',
            )
        return await self._render_training_plan_dialogue_response(
            draft=rebuilt_draft,
            pending=None,
            thread_id=confirmation_thread_id,
        )

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

    @staticmethod
    def _looks_like_training_confirmation_invitation(text: str) -> bool:
        content = str(text or '').strip()
        if not content:
            return False
        lowered = content.lower()
        confirmation_tokens = (
            '确认执行',
            '是否确认',
            '是否继续',
            '如果你同意',
            '如果您同意',
            '同意的话',
            '确认后',
            '继续吗',
            '可以继续',
        )
        execution_tokens = (
            '开始执行',
            '开始训练',
            '启动训练',
            '数据准备',
            '准备数据集',
            '准备数据',
            '准备训练数据',
            'prepare',
            'data.yaml',
        )
        explicit_prepare_question_patterns = (
            r'(?:现在|先)?开始准备(?:数据集|数据|训练数据)?吗',
            r'是否(?:现在|先)?开始准备(?:数据集|数据|训练数据)?',
            r'要不要(?:先)?准备(?:数据集|数据|训练数据)?',
            r'先准备(?:数据集|数据|训练数据)?吗',
        )
        if any(re.search(pattern, content) or re.search(pattern, lowered) for pattern in explicit_prepare_question_patterns) and any(
            token in content or token in lowered for token in execution_tokens
        ):
            return True
        return any(token in content or token in lowered for token in confirmation_tokens) and any(
            token in content or token in lowered for token in execution_tokens
        )

    def _build_materialized_training_pending(
        self,
        *,
        draft: dict[str, Any],
        user_text: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        draft = dict(draft or {})
        if not draft:
            return None
        if str(draft.get('source_intent') or '').strip().lower() == 'training_loop':
            return None
        execution_backend = str(draft.get('execution_backend') or '').strip().lower()
        if execution_backend and execution_backend != 'standard_yolo':
            return None

        dataset_path = str(
            draft.get('dataset_path')
            or self.session_state.active_dataset.dataset_root
            or self.session_state.active_dataset.img_dir
            or ''
        ).strip()
        planned_training_args = dict(draft.get('planned_training_args') or {})
        model = str(
            planned_training_args.get('model')
            or self.session_state.active_training.model
            or ''
        ).strip()
        if not dataset_path or not model:
            return None

        readiness = dict(self.session_state.active_dataset.last_readiness or {})
        next_tool_name = str(draft.get('next_step_tool') or '').strip()
        next_tool_args = dict(draft.get('next_step_args') or {})

        if not next_tool_name:
            data_yaml = str(
                planned_training_args.get('data_yaml')
                or readiness.get('resolved_data_yaml')
                or self._session_training_data_yaml(dataset_path=dataset_path)
                or ''
            ).strip()
            if data_yaml:
                planned_training_args['data_yaml'] = data_yaml
                next_tool_name = 'start_training'
                next_tool_args = dict(planned_training_args)
            else:
                next_tool_name = 'prepare_dataset_for_training'
                next_tool_args = {'dataset_path': dataset_path}
                classes_txt = str(planned_training_args.get('classes_txt') or '').strip()
                if classes_txt:
                    next_tool_args['classes_txt'] = classes_txt

        rebuilt_draft = self._build_training_plan_draft(
            user_text=user_text,
            dataset_path=dataset_path,
            readiness=readiness,
            preflight={},
            next_tool_name=next_tool_name,
            next_tool_args=next_tool_args,
            planned_training_args=planned_training_args,
        )
        for key in (
            'source_intent',
            'advanced_details_requested',
            'planner_user_request',
            'planner_decision_source',
            'planner_decision',
            'planner_output',
            'planner_observed_tools',
        ):
            if draft.get(key) not in (None, '', [], {}):
                rebuilt_draft[key] = draft.get(key)
        if not rebuilt_draft.get('data_summary') and draft.get('data_summary'):
            rebuilt_draft['data_summary'] = draft.get('data_summary')
        if not rebuilt_draft.get('reasoning_summary') and draft.get('reasoning_summary'):
            rebuilt_draft['reasoning_summary'] = draft.get('reasoning_summary')

        pending = {
            'id': None,
            'name': next_tool_name,
            'args': dict(next_tool_args),
            'synthetic': True,
        }
        return rebuilt_draft, pending

    def _ensure_materialized_training_state_context(self, *, draft: dict[str, Any]) -> None:
        draft = dict(draft or {})
        if not draft:
            return
        dataset_path = str(
            draft.get('dataset_path')
            or self.session_state.active_dataset.dataset_root
            or self.session_state.active_dataset.img_dir
            or ''
        ).strip()
        planned_training_args = dict(draft.get('planned_training_args') or {})
        next_tool_name = str(draft.get('next_step_tool') or '').strip()
        next_tool_args = dict(draft.get('next_step_args') or {})
        data_yaml = str(
            next_tool_args.get('data_yaml')
            or planned_training_args.get('data_yaml')
            or self.session_state.active_dataset.data_yaml
            or ''
        ).strip()
        model = str(
            planned_training_args.get('model')
            or self.session_state.active_training.model
            or ''
        ).strip()

        if dataset_path and not str(self.session_state.active_dataset.dataset_root or '').strip():
            self.session_state.active_dataset.dataset_root = dataset_path
        if data_yaml:
            self.session_state.active_dataset.data_yaml = data_yaml
        if model:
            self.session_state.active_training.model = model

        readiness = dict(self.session_state.active_dataset.last_readiness or {})
        summary = str(draft.get('data_summary') or draft.get('reasoning_summary') or '').strip()
        warnings = [str(item).strip() for item in (draft.get('warnings') or []) if str(item).strip()]
        blockers = [str(item).strip() for item in (draft.get('blockers') or []) if str(item).strip()]
        if not blockers and next_tool_name == 'prepare_dataset_for_training' and not data_yaml:
            blockers = ['缺少可用的 data_yaml']

        if dataset_path and not str(readiness.get('dataset_root') or '').strip():
            readiness['dataset_root'] = dataset_path
        if data_yaml and not str(readiness.get('resolved_data_yaml') or '').strip():
            readiness['resolved_data_yaml'] = data_yaml
        if warnings and not readiness.get('warnings'):
            readiness['warnings'] = warnings
        if blockers and not readiness.get('blockers'):
            readiness['blockers'] = blockers
        if summary and not str(readiness.get('summary') or '').strip():
            readiness['summary'] = summary

        if next_tool_name == 'prepare_dataset_for_training':
            readiness.setdefault('ok', True)
            readiness.setdefault('ready', False)
            readiness.setdefault('preparable', True)
        elif next_tool_name == 'start_training':
            readiness.setdefault('ok', True)
            readiness.setdefault('ready', True)
            readiness.setdefault('preparable', False)

        if readiness:
            self.session_state.active_dataset.last_readiness = readiness

    async def _maybe_materialize_text_only_training_confirmation(
        self,
        *,
        thread_id: str,
        user_text_hint: str,
        final_text: str,
        applied_results: list[tuple[str, dict[str, Any]]],
        stream_handler: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any] | None:
        if applied_results:
            return None
        if self._pending_from_state():
            return None
        if not self._looks_like_training_confirmation_invitation(final_text):
            return None
        materialized = self._build_materialized_training_pending(
            draft=dict(self.session_state.active_training.training_plan_draft or {}),
            user_text=user_text_hint or self._recent_user_text(),
        )
        if materialized is None:
            return None
        rebuilt_draft, pending = materialized
        self._ensure_materialized_training_state_context(draft=rebuilt_draft)
        self._save_training_plan_draft(rebuilt_draft)
        interrupt_payload = self._draft_to_training_confirmation_interrupt(rebuilt_draft, thread_id=thread_id)
        self.memory.append_event(
            self.session_state.session_id,
            'graph_text_training_plan_materialized',
            {
                'thread_id': thread_id,
                'tool': str(pending.get('name') or ''),
                'execution_mode': str(rebuilt_draft.get('execution_mode') or ''),
            },
        )
        if interrupt_payload is not None:
            return await self._enter_graph_training_confirmation(
                draft=rebuilt_draft,
                thread_id=thread_id,
                user_text_hint=user_text_hint or self._recent_user_text(),
                stream_handler=stream_handler,
            )
        return None

    def _should_use_video_prediction(self, user_text: str, path: str) -> bool:
        normalized = str(user_text or '').lower()
        if intent_parsing.looks_like_video_path(path):
            return True
        return any(token in user_text for token in ('视频', '录像')) or 'video' in normalized

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
            build_training_loop_start_fallback_plan_fn=self._build_training_loop_start_fallback_plan,
            known_training_loop_data_yaml=self._known_training_loop_data_yaml,
            append_event=lambda event, payload: self.memory.append_event(self.session_state.session_id, event, payload),
            compact_training_loop_start_fact=self._compact_training_loop_start_fact,
            build_training_loop_start_draft_fn=self._build_training_loop_start_draft,
            save_training_plan_draft=self._save_training_plan_draft,
            append_ai_message=lambda reply: self._messages.append(AIMessage(content=reply)),
            handoff_to_graph=lambda handoff_thread_id, handoff_user_text: self._handoff_current_runtime_to_graph(
                thread_id=handoff_thread_id,
                user_text_hint=handoff_user_text,
                auto_approve=False,
            ),
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
            thread_id=f"{thread_id}-post-prepare-loop",
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
        wants_stop_training: bool,
        explicit_run_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return await resolve_training_loop_route(
            session_state=self.session_state,
            user_text=user_text,
            normalized_text=normalized_text,
            wants_predict=wants_predict,
            wants_stop_training=wants_stop_training,
            explicit_run_ids=explicit_run_ids,
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
        return training_plan_user_facts(draft, pending=pending)

    def _training_plan_render_error(self, draft: dict[str, Any], *, pending: bool, error: Exception | None = None) -> str:
        return training_plan_render_error(draft, pending=pending, error=error)

    @staticmethod
    def _should_force_structured_training_surface(draft: dict[str, Any], *, pending: bool) -> bool:
        if not draft or not pending:
            return False
        execution_mode = str(draft.get('execution_mode') or '').strip().lower()
        if execution_mode in {'prepare_only', 'discussion_only', 'blocked'}:
            return False
        next_step_tool = str(draft.get('next_step_tool') or '').strip().lower()
        critical_modes = {'prepare_then_train', 'prepare_then_loop', 'direct_train', 'direct_loop'}
        critical_tools = {'prepare_dataset_for_training', 'start_training', 'start_training_loop', 'training_preflight'}
        return execution_mode in critical_modes or next_step_tool in critical_tools

    async def _render_training_plan_message(self, draft: dict[str, Any], *, pending: bool) -> str:
        if self._should_force_structured_training_surface(draft, pending=pending):
            return self._render_training_plan_draft(draft, pending=pending)
        return await render_training_plan_message_service(
            planner_llm=self.planner_llm,
            draft=draft,
            pending=pending,
            render_training_plan_draft=self._render_training_plan_draft,
            invoke_renderer_text=self._invoke_renderer_text,
        )

    async def _render_training_plan_dialogue_response(
        self,
        *,
        draft: dict[str, Any] | None,
        pending: dict[str, Any] | None,
        thread_id: str,
        preamble: str = '',
        append_message: bool = False,
    ) -> dict[str, Any]:
        draft = dict(draft or {})
        pending = dict(pending or {})
        rendered_plan = await self._render_training_plan_message(draft, pending=bool(pending))
        message = rendered_plan
        preamble = str(preamble or '').strip()
        if preamble:
            message = f'{preamble}\n\n{rendered_plan}' if rendered_plan else preamble
        if append_message and message:
            self._messages.append(AIMessage(content=message))
        return {
            'status': 'completed' if not pending else 'needs_confirmation',
            'message': message,
            'tool_call': {'name': pending['name'], 'args': pending.get('args', {})} if pending else None,
            'thread_id': thread_id if pending else None,
        }

    def _complete_dialogue_text(self, reply: str, *, append_message: bool = True) -> dict[str, Any]:
        reply = str(reply or '').strip()
        if append_message and reply:
            self._messages.append(AIMessage(content=reply))
        return {'status': 'completed', 'message': reply, 'tool_call': None}

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

    def _build_grounded_prepare_followup_message(self, prepare_parsed: dict[str, Any], preflight: dict[str, Any]) -> str:
        sections: list[str] = []
        for tool_name, parsed in (
            ('prepare_dataset_for_training', prepare_parsed),
            ('training_preflight', preflight),
        ):
            text = self._build_grounded_tool_reply([(tool_name, parsed)])
            if not text:
                text = str(parsed.get('summary') or parsed.get('message') or parsed.get('error') or '').strip()
            if text:
                sections.append(text)
        return self._merge_grounded_sections(sections)

    async def _render_prepare_followup_message(self, prepare_parsed: dict[str, Any], preflight: dict[str, Any]) -> str:
        grounded_text = self._build_grounded_prepare_followup_message(prepare_parsed, preflight)
        if not preflight.get('ready_to_start'):
            self.memory.append_event(
                self.session_state.session_id,
                'post_prepare_followup_rendered',
                {
                    'mode': 'grounded',
                    'ready_to_start': False,
                    'blockers': [str(item).strip() for item in (preflight.get('blockers') or []) if str(item).strip()][:4],
                    'warnings': [str(item).strip() for item in (preflight.get('warnings') or []) if str(item).strip()][:4],
                },
            )
            if grounded_text:
                return grounded_text
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
            self.memory.append_event(
                self.session_state.session_id,
                'post_prepare_followup_rendered',
                {
                    'mode': 'llm',
                    'ready_to_start': bool(preflight.get('ready_to_start')),
                    'blockers': [str(item).strip() for item in (preflight.get('blockers') or []) if str(item).strip()][:4],
                },
            )
            return text
        return grounded_text or await self._render_multi_tool_result_message(
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

    async def _try_handle_training_plan_dialogue(self, user_text: str, thread_id: str) -> dict[str, Any] | None:
        draft = dict(self.session_state.active_training.training_plan_draft or {})
        pending = self._pending_from_state()
        explicit_run_ids = self._extract_training_run_ids_from_text(user_text)
        clear_fields = self._collect_training_clear_fields(user_text)
        dialogue_result = await run_training_plan_dialogue_flow(
            session_state=self.session_state,
            user_text=user_text,
            draft=draft,
            pending=pending,
            explicit_run_ids=explicit_run_ids,
            clear_fields=clear_fields,
            readiness=self.session_state.active_dataset.last_readiness or {},
            data_yaml=str(self.session_state.active_dataset.data_yaml or '').strip(),
            is_training_discussion_only=self._is_training_discussion_only,
            custom_training_script_requested=bool(intent_parsing.extract_custom_training_script_from_text(user_text)),
            looks_like_prepare_only_request=self._looks_like_prepare_only_request,
            extract_dataset_path=intent_parsing.extract_dataset_path_from_text,
            local_path_exists=lambda path: Path(path).expanduser().exists(),
            collect_requested_training_args=self._collect_requested_training_args,
            extract_training_execution_backend=self._extract_training_execution_backend_from_text,
            wants_training_advanced_details=self._wants_training_advanced_details,
            direct_tool=self.direct_tool,
            build_training_plan_draft_fn=self._build_training_plan_draft,
            render_tool_result_message=self._render_tool_result_message,
            render_training_plan_message=self._render_training_plan_message,
        )
        followup_action = dict(dialogue_result.get('followup_action') or {})
        draft_to_save = dict(dialogue_result.get('draft_to_save') or {})
        if draft_to_save:
            self._save_training_plan_draft(draft_to_save)
            draft = draft_to_save
        return await self._apply_training_plan_followup_action(
            followup_action=followup_action,
            thread_id=thread_id,
            user_text=user_text,
            handoff_mode=str(followup_action.get('handoff_mode') or 'defer'),
            pending=pending,
            draft=draft,
        )

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
        training_plan_context: dict[str, Any] | None
        pending_confirmation: dict[str, Any] | None
        pending_review: dict[str, Any] | None
        training_plan: dict[str, Any] | None
        training_phase: str | None
        training_execution_mode: str | None
        training_next_step_tool: str | None
        training_next_step_args: dict[str, Any] | None
        suspended_training_plan: dict[str, Any] | None
        pending_new_task: str | None
        training_status_reply: str | None
        prepare_result: dict[str, Any] | None
        training_preflight: dict[str, Any] | None

except Exception:
    class _AgentRuntimeGraphState(TypedDict, total=False):
        messages: list[Any]
        remaining_steps: int
        cached_tool_context: dict[str, Any] | None
        dataset_fact_context: dict[str, Any] | None
        training_plan_context: dict[str, Any] | None
        pending_confirmation: dict[str, Any] | None
        pending_review: dict[str, Any] | None
        training_plan: dict[str, Any] | None
        training_phase: str | None
        training_execution_mode: str | None
        training_next_step_tool: str | None
        training_next_step_args: dict[str, Any] | None
        suspended_training_plan: dict[str, Any] | None
        pending_new_task: str | None
        training_status_reply: str | None
        prepare_result: dict[str, Any] | None
        training_preflight: dict[str, Any] | None


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


def _training_state_update_from_interrupt_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(payload or {})
    return {
        'training_plan': dict(payload.get('plan') or {}) or None,
        'training_phase': str(payload.get('phase') or '').strip() or None,
        'training_execution_mode': str(payload.get('execution_mode') or '').strip() or None,
        'training_next_step_tool': str(payload.get('next_step_tool') or '').strip() or None,
        'training_next_step_args': dict(payload.get('next_step_args') or {}) or None,
        'suspended_training_plan': dict(payload.get('suspended_training_plan') or {}) or None,
        'pending_new_task': str(payload.get('pending_new_task') or '').strip() or None,
        'training_status_reply': str(payload.get('status_reply') or '').strip() or None,
        'training_plan_context': None,
        'pending_confirmation': None,
        'pending_review': {},
    }


def _latest_human_text_from_messages(messages: list[Any] | None) -> str:
    for message in reversed(list(messages or [])):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content.strip()
    return ''


def _tool_result_message(tool_name: str, *, parsed: dict[str, Any], tool_call_id: str = '', status: str = '') -> ToolMessage:
    kwargs: dict[str, Any] = {
        'content': json.dumps(parsed, ensure_ascii=False),
        'name': tool_name,
        'tool_call_id': tool_call_id,
    }
    if status:
        kwargs['status'] = status
    return ToolMessage(**kwargs)


async def build_agent_client(settings: AgentSettings | None = None) -> YoloStudioAgentClient:
    settings = settings or AgentSettings()
    memory_store = MemoryStore(settings.memory_root)
    client_holder: dict[str, YoloStudioAgentClient] = {}
    checkpointer = FileCheckpointSaver(_checkpoint_path(settings))

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
        'state_schema': _AgentRuntimeGraphState,
    }
    if StateGraph is None:
        react_kwargs['checkpointer'] = checkpointer
    graph_tools = _build_graph_tool_surface(
        tools,
        confirmation_mode=settings.confirmation_mode,
        tool_policy_resolver=lambda tool_name: resolve_tool_execution_policy(tool_name, tool_registry={tool.name: tool for tool in raw_tools}),
        client_getter=lambda: client_holder.get('client'),
    )
    react_graph = create_react_agent(
        llm,
        graph_tools,
        **react_kwargs,
    )

    def _thread_id_from_config(config: Any) -> str:
        if isinstance(config, dict):
            configurable = dict(config.get('configurable') or {})
            return str(configurable.get('thread_id') or '').strip()
        configurable = dict(getattr(config, 'configurable', {}) or {})
        return str(configurable.get('thread_id') or '').strip()

    def _latest_human_text_from_state(state: _AgentRuntimeGraphState) -> str:
        for message in reversed(list((state or {}).get('messages') or [])):
            if isinstance(message, HumanMessage):
                return str(getattr(message, 'content', '') or '').strip()
            if getattr(message.__class__, '__name__', '') == 'HumanMessage':
                return str(getattr(message, 'content', '') or '').strip()
        return ''

    def _update_plan_from_preflight(
        client: YoloStudioAgentClient,
        *,
        plan_payload: dict[str, Any],
        preflight: dict[str, Any],
        next_tool_name: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        updated_plan = dict(plan_payload or {})
        resolved_args = dict(preflight.get('resolved_args') or {})
        if next_tool_name == 'start_training_loop' and 'epochs' in resolved_args and 'epochs_per_round' not in resolved_args:
            resolved_args['epochs_per_round'] = resolved_args.get('epochs')
        if next_tool_name == 'start_training_loop':
            resolved_args.pop('epochs', None)
        for key in (
            'model',
            'data_yaml',
            'batch',
            'imgsz',
            'device',
            'training_environment',
            'project',
            'name',
            'fraction',
            'classes',
            'single_cls',
            'optimizer',
            'freeze',
            'resume',
            'lr0',
            'patience',
            'workers',
            'amp',
        ):
            value = resolved_args.get(key)
            if value not in (None, ''):
                updated_plan[key] = value
            elif key in {'fraction', 'classes', 'single_cls', 'freeze', 'resume', 'lr0', 'patience', 'workers', 'amp'}:
                updated_plan[key] = [] if key == 'classes' and value is None else value
        if str(updated_plan.get('mode') or '').strip().lower() == 'loop':
            for key in ('max_rounds', 'epochs_per_round', 'loop_name'):
                value = resolved_args.get(key)
                if value not in (None, ''):
                    updated_plan[key] = value
        else:
            epochs_value = resolved_args.get('epochs')
            if epochs_value not in (None, ''):
                updated_plan['epochs'] = epochs_value
        blockers = [str(item).strip() for item in (preflight.get('blockers') or []) if str(item).strip()]
        warnings = [str(item).strip() for item in (preflight.get('warnings') or []) if str(item).strip()]
        summary = str(preflight.get('summary') or '').strip()
        updated_plan['blockers'] = blockers
        updated_plan['warnings'] = warnings
        updated_plan['readiness_summary'] = summary
        return updated_plan, (resolved_args or dict(plan_payload or {}))

    async def _route_training_entry(state: _AgentRuntimeGraphState, config: Any = None) -> Command:
        client = client_holder.get('client')
        if client is None:
            return Command(goto='agent_runtime')
        thread_id = _thread_id_from_config(config)
        context = extract_training_plan_context_from_state(dict(state or {})) or {}
        if context:
            interrupt_payload = client._draft_to_training_confirmation_interrupt(
                context,
                thread_id=thread_id,
            )
            if interrupt_payload:
                return Command(
                    update=_training_state_update_from_interrupt_payload(interrupt_payload),
                    goto='training_confirmation',
                )

        latest_user_text = _latest_human_text_from_state(state)
        if not latest_user_text:
            return Command(goto='agent_runtime')
        mainline_context = client._collect_mainline_context(latest_user_text)
        route_state = await client._resolve_mainline_route_state(latest_user_text, mainline_context)
        dispatch_payload = resolve_mainline_dispatch_payload(
            mainline_context=mainline_context,
            route_state=route_state,
        )
        training_entrypoint_args = dict(dispatch_payload.get('training_entrypoint_request_args') or {})
        if not training_entrypoint_args or bool(training_entrypoint_args.get('wants_training_loop_start')):
            return Command(goto='agent_runtime')
        entrypoint_result = await run_training_request_entrypoint(
            session_state=client.session_state,
            user_text=latest_user_text,
            direct_tool=client.direct_tool,
            collect_requested_training_args=client._collect_requested_training_args,
            is_training_discussion_only=client._is_training_discussion_only,
            extract_training_execution_backend=client._extract_training_execution_backend_from_text,
            build_training_plan_draft_fn=client._build_training_plan_draft,
            render_training_plan_message=client._render_training_plan_message,
            **training_entrypoint_args,
        )
        if not entrypoint_result:
            return Command(goto='agent_runtime')
        draft = dict(entrypoint_result.get('draft') or {})
        if bool(entrypoint_result.get('defer_to_graph')) and draft:
            interrupt_payload = client._draft_to_training_confirmation_interrupt(
                draft,
                thread_id=thread_id,
            )
            if interrupt_payload:
                return Command(
                    update=_training_state_update_from_interrupt_payload(interrupt_payload),
                    goto='training_confirmation',
                )
        reply = str(entrypoint_result.get('reply') or '').strip()
        if reply:
            return Command(
                update={
                    'messages': [AIMessage(content=reply)],
                    'training_plan_context': None,
                },
                goto=END,
            )
        return Command(goto='agent_runtime')

    async def _refresh_training_start_after_edit(state: _AgentRuntimeGraphState, config: Any = None) -> Command:
        del config
        client = client_holder.get('client')
        if client is None:
            return Command(goto='training_confirmation')
        plan_payload = dict((state or {}).get('training_plan') or {})
        if not plan_payload:
            return Command(goto='training_confirmation')
        next_tool_name = canonical_tool_name(str((state or {}).get('training_next_step_tool') or '').strip())
        if next_tool_name not in {'start_training', 'start_training_loop'}:
            return Command(goto='training_confirmation')
        preflight_args = dict((state or {}).get('training_next_step_args') or plan_payload)
        preflight = await client.direct_tool('training_preflight', **preflight_args)
        updated_plan, resolved_args = _update_plan_from_preflight(
            client,
            plan_payload=plan_payload,
            preflight=preflight,
            next_tool_name=next_tool_name,
        )
        return Command(
            update={
                'training_plan': updated_plan,
                'training_next_step_tool': next_tool_name,
                'training_next_step_args': resolved_args,
                'training_preflight': preflight,
                'training_status_reply': '',
            },
            goto='training_confirmation',
        )

    async def _execute_prepare(state: _AgentRuntimeGraphState, config: Any = None) -> Command:
        del config
        client = client_holder.get('client')
        if client is None:
            return Command(goto='training_confirmation')
        plan_payload = dict((state or {}).get('training_plan') or {})
        if not plan_payload:
            return Command(goto='training_confirmation')
        plan = coerce_training_plan(plan_payload)
        next_args = dict((state or {}).get('training_next_step_args') or {'dataset_path': plan.dataset_path})
        prepare_result = await client.direct_tool('prepare_dataset_for_training', **next_args)
        messages: list[Any] = [_tool_result_message('prepare_dataset_for_training', parsed=prepare_result)]
        if not prepare_result.get('ok'):
            failed_plan = dict(plan_payload)
            failed_plan['blockers'] = [str(prepare_result.get('error') or prepare_result.get('summary') or '数据准备失败').strip()]
            failed_plan['prepare_summary'] = str(prepare_result.get('summary') or prepare_result.get('error') or '').strip()
            return Command(
                update={
                    'messages': messages,
                    'training_plan': failed_plan,
                    'training_status_reply': failed_plan['prepare_summary'],
                },
                goto='training_confirmation',
            )
        preflight_args = client._model_dump_compat(plan)
        prepared_yaml = str(prepare_result.get('data_yaml') or prepare_result.get('resolved_data_yaml') or '').strip()
        if prepared_yaml:
            preflight_args['data_yaml'] = prepared_yaml
        if getattr(plan, 'mode', 'train') == 'loop':
            preflight_args.setdefault('epochs', getattr(plan, 'epochs_per_round', None))
        preflight = await client.direct_tool('training_preflight', **preflight_args)
        messages.append(_tool_result_message('training_preflight', parsed=preflight))
        return Command(
            update={
                'messages': messages,
                'prepare_result': prepare_result,
                'training_preflight': preflight,
                'training_status_reply': '',
            },
            goto='post_prepare',
        )

    async def _execute_training(state: _AgentRuntimeGraphState, config: Any = None) -> Command:
        del config
        client = client_holder.get('client')
        if client is None:
            return Command(goto='training_confirmation')
        plan_payload = dict((state or {}).get('training_plan') or {})
        if not plan_payload:
            return Command(goto='training_confirmation')
        plan = coerce_training_plan(plan_payload)
        next_tool_name = canonical_tool_name(str((state or {}).get('training_next_step_tool') or '').strip())
        if next_tool_name not in {'start_training', 'start_training_loop'}:
            next_tool_name = 'start_training_loop' if getattr(plan, 'mode', 'train') == 'loop' else 'start_training'
        next_args = dict((state or {}).get('training_next_step_args') or client._model_dump_compat(plan))
        parsed = await client.direct_tool(next_tool_name, **next_args)
        reply = await client._render_tool_result_message(next_tool_name, parsed)
        messages: list[Any] = [_tool_result_message(next_tool_name, parsed=parsed)]
        if reply:
            messages.append(AIMessage(content=reply))
        if parsed.get('ok'):
            return Command(
                update={
                    'messages': messages,
                    'training_plan': None,
                    'training_phase': None,
                    'training_execution_mode': None,
                    'training_next_step_tool': None,
                    'training_next_step_args': None,
                    'suspended_training_plan': None,
                    'pending_new_task': None,
                    'training_status_reply': '',
                    'training_plan_context': None,
                },
                goto=END,
            )
        failed_plan = dict(plan_payload)
        failed_plan['blockers'] = [str(parsed.get('error') or parsed.get('summary') or '训练启动失败').strip()]
        failed_plan['readiness_summary'] = str(parsed.get('summary') or parsed.get('error') or '').strip()
        return Command(
            update={
                'messages': messages,
                'training_plan': failed_plan,
                'training_status_reply': str(reply or failed_plan['readiness_summary']).strip(),
            },
            goto='training_confirmation',
        )

    async def _route_new_task(state: _AgentRuntimeGraphState, config: Any = None) -> Command:
        del config
        pending_new_task = str((state or {}).get('pending_new_task') or '').strip()
        if not pending_new_task:
            suspended = dict((state or {}).get('suspended_training_plan') or {})
            if suspended:
                return Command(
                    update={
                        'training_plan': dict(suspended.get('plan') or {}) or None,
                        'training_phase': str(suspended.get('phase') or '').strip() or None,
                        'training_execution_mode': str(suspended.get('execution_mode') or '').strip() or None,
                        'training_next_step_tool': str(suspended.get('next_step_tool') or '').strip() or None,
                        'training_next_step_args': dict(suspended.get('next_step_args') or {}) or None,
                        'suspended_training_plan': None,
                        'training_status_reply': '',
                    },
                    goto='training_confirmation',
                )
            return Command(goto='agent_runtime')
        return Command(
            update={
                'messages': [HumanMessage(content=pending_new_task)],
                'pending_new_task': None,
                'training_status_reply': '',
            },
            goto='agent_runtime',
        )

    if StateGraph is not None:
        workflow = StateGraph(_AgentRuntimeGraphState)
        workflow.add_node('route_training_entry', _route_training_entry)
        workflow.add_node('agent_runtime', react_graph)
        workflow.add_node('training_confirmation', training_confirmation_node)
        workflow.add_node('refresh_training_start_after_edit', _refresh_training_start_after_edit)
        workflow.add_node('execute_prepare', _execute_prepare)
        workflow.add_node('post_prepare', post_prepare_node)
        workflow.add_node('execute_training', _execute_training)
        workflow.add_node('answer_training_status', answer_training_status_node)
        workflow.add_node('route_new_task', _route_new_task)
        workflow.add_edge(START, 'route_training_entry')
        workflow.add_edge('agent_runtime', END)
        graph = workflow.compile(checkpointer=checkpointer)
        try:
            setattr(graph, '__codex_training_entry__', True)
        except Exception:
            pass
    else:
        graph = react_graph
    tool_registry = {tool.name: tool for tool in raw_tools}
    client = YoloStudioAgentClient(
        graph=graph,
        settings=settings,
        tool_registry=tool_registry,
        planner_llm=helper_llm,
        primary_llm_settings=primary_llm_settings,
        helper_llm_settings=helper_llm_settings,
        memory=memory_store,
        checkpointer=checkpointer,
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
    contract_error = _interruptible_tool_node_contract_error(ToolNode)
    if contract_error:
        raise RuntimeError(
            'LangGraph ToolNode private API is incompatible with interruptible approvals: '
            f'{contract_error}'
        )
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

    # LangGraph does not currently expose a public hook that lets us pause a
    # tool call at the ToolNode execution boundary and still resume through the
    # graph checkpoint contract. We therefore override the private _run_one /
    # _arun_one methods so the interrupt-aware wrap hooks are applied before the
    # tool body executes. Keep _interruptible_tool_node_contract_error() in sync
    # with these assumptions so upstream LangGraph drift fails loudly.
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


def _interruptible_tool_node_contract_error(tool_node_cls: type[Any]) -> str | None:
    try:
        init_signature = inspect.signature(tool_node_cls.__init__)
    except (TypeError, ValueError):
        return 'ToolNode.__init__ signature is unavailable'
    init_parameters = init_signature.parameters
    missing_init = [
        name for name in ('handle_tool_errors', 'awrap_tool_call')
        if name not in init_parameters
    ]
    if missing_init:
        return f"ToolNode.__init__ is missing parameters: {', '.join(missing_init)}"
    missing_class_attrs = [
        name for name in ('_execute_tool_sync', '_execute_tool_async')
        if not hasattr(tool_node_cls, name)
    ]
    if missing_class_attrs:
        return f"ToolNode is missing methods: {', '.join(missing_class_attrs)}"
    try:
        probe = tool_node_cls([], handle_tool_errors=True, awrap_tool_call=None)
    except Exception as exc:
        return f'ToolNode probe construction failed: {exc}'
    missing_instance_attrs = [
        name for name in ('tools_by_name', '_handle_tool_errors', '_wrap_tool_call', '_awrap_tool_call')
        if not hasattr(probe, name)
    ]
    if missing_instance_attrs:
        return f"ToolNode instance is missing attrs: {', '.join(missing_instance_attrs)}"
    return None


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


async def build_agent():
    return await build_agent_client()

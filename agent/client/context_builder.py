from __future__ import annotations

from typing import Iterable

from langchain_core.messages import BaseMessage, SystemMessage

from agent_plan.agent.client.event_retriever import MemoryDigest
from agent_plan.agent.client.session_state import SessionState


def _fmt_mapping(data: dict) -> str:
    if not data:
        return '无'
    return '; '.join(f'{k}={v}' for k, v in data.items())


class ContextBuilder:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def build_messages(
        self,
        state: SessionState,
        recent_messages: Iterable[BaseMessage],
        digest: MemoryDigest | None = None,
    ) -> list[BaseMessage]:
        summary = self.build_state_summary(state, digest)
        return [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=summary),
            *list(recent_messages),
        ]

    def build_state_summary(self, state: SessionState, digest: MemoryDigest | None = None) -> str:
        ds = state.active_dataset
        tr = state.active_training
        pc = state.pending_confirmation
        pred = state.active_prediction
        kn = state.active_knowledge
        pref = state.preferences
        digest_text = digest.to_text() if digest else '无历史摘要'
        return (
            '当前结构化上下文:\n'
            f'- session_id: {state.session_id}\n'
            '数据集:\n'
            f'  dataset_root: {ds.dataset_root or "未设置"}\n'
            f'  img_dir: {ds.img_dir or "未设置"}\n'
            f'  label_dir: {ds.label_dir or "未设置"}\n'
            f'  data_yaml: {ds.data_yaml or "未设置"}\n'
            f'  last_scan: {_fmt_mapping(ds.last_scan)}\n'
            f'  last_validate: {_fmt_mapping(ds.last_validate)}\n'
            f'  last_readiness: {_fmt_mapping(ds.last_readiness)}\n'
            f'  last_split: {_fmt_mapping(ds.last_split)}\n'
            f'  last_health_check: {_fmt_mapping(ds.last_health_check)}\n'
            f'  last_duplicate_check: {_fmt_mapping(ds.last_duplicate_check)}\n'
            f'  last_extract_preview: {_fmt_mapping(ds.last_extract_preview)}\n'
            f'  last_extract_result: {_fmt_mapping(ds.last_extract_result)}\n'
            f'  last_video_scan: {_fmt_mapping(ds.last_video_scan)}\n'
            f'  last_frame_extract: {_fmt_mapping(ds.last_frame_extract)}\n'
            '训练:\n'
            f'  running: {tr.running}\n'
            f'  model: {tr.model or "未设置"}\n'
            f'  data_yaml: {tr.data_yaml or "未设置"}\n'
            f'  device: {tr.device or "未设置"}\n'
            f'  pid: {tr.pid if tr.pid is not None else "无"}\n'
            f'  log_file: {tr.log_file or "无"}\n'
            f'  last_status: {_fmt_mapping(tr.last_status)}\n'
            '预测:\n'
            f'  source_path: {pred.source_path or "未设置"}\n'
            f'  model: {pred.model or "未设置"}\n'
            f'  output_dir: {pred.output_dir or "无"}\n'
            f'  report_path: {pred.report_path or "无"}\n'
            f'  last_result: {_fmt_mapping(pred.last_result)}\n'
            '知识:\n'
            f'  last_retrieval: {_fmt_mapping(kn.last_retrieval)}\n'
            f'  last_analysis: {_fmt_mapping(kn.last_analysis)}\n'
            f'  last_recommendation: {_fmt_mapping(kn.last_recommendation)}\n'
            '待确认操作:\n'
            f'  tool: {pc.tool_name or "无"}\n'
            f'  args: {_fmt_mapping(pc.tool_args)}\n'
            '偏好:\n'
            f'  default_model: {pref.default_model or "未设置"}\n'
            f'  default_epochs: {pref.default_epochs if pref.default_epochs is not None else "未设置"}\n'
            '历史摘要:\n'
            f'{digest_text}\n'
        )

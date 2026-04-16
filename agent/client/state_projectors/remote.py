from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.state_projectors.common import _remote_transfer_snapshot


def apply_remote_transfer_tool_result(
    session_state: SessionState,
    tool_name: str,
    result: dict[str, Any],
    tool_args: dict[str, Any],
) -> None:
    del tool_args
    rt = session_state.active_remote_transfer
    if tool_name == 'list_remote_profiles' and result.get('ok'):
        rt.last_profile_listing = _remote_transfer_snapshot(
            result,
            overview_key='profile_overview',
            extra_keys=('profiles_path', 'default_profile', 'profile_count', 'ssh_alias_count'),
        )
        rt.last_profile_listing['profile_count'] = len(result.get('profiles') or [])
        rt.last_profile_listing['ssh_alias_count'] = len(result.get('ssh_aliases') or [])
        profiles = result.get('profiles') or []
        default_profile = str(result.get('default_profile') or '').strip()
        if default_profile:
            rt.profile_name = default_profile
            for item in profiles:
                if str(item.get('name') or '').strip() == default_profile:
                    rt.target_label = str(item.get('target_label') or rt.target_label or default_profile)
                    rt.remote_root = str(item.get('remote_root') or rt.remote_root)
                    break
    elif tool_name == 'upload_assets_to_remote' and result.get('ok'):
        rt.target_label = str(result.get('target_label') or rt.target_label)
        rt.profile_name = str(result.get('profile_name') or rt.profile_name)
        rt.remote_root = str(result.get('remote_root') or rt.remote_root)
        rt.last_upload = _remote_transfer_snapshot(
            result,
            overview_key='transfer_overview',
            extra_keys=(
                'target_label',
                'profile_name',
                'remote_root',
                'uploaded_count',
                'uploaded_items',
                'file_count',
                'verified_file_count',
                'skipped_file_count',
                'chunked_file_count',
                'scp_file_count',
                'transferred_bytes',
                'skipped_bytes',
                'total_bytes',
                'resume_enabled',
                'verify_hash',
                'hash_algorithm',
                'large_file_threshold_mb',
                'chunk_size_mb',
                'transfer_strategy_summary',
                'file_results_preview',
            ),
        )
        rt.last_upload['target_label'] = rt.target_label
        rt.last_upload['profile_name'] = rt.profile_name
        rt.last_upload['remote_root'] = rt.remote_root
        rt.last_upload['uploaded_items'] = result.get('uploaded_items') or []
        rt.last_upload['file_results_preview'] = result.get('file_results_preview') or []
    elif tool_name == 'download_assets_from_remote' and result.get('ok'):
        rt.target_label = str(result.get('target_label') or rt.target_label)
        rt.profile_name = str(result.get('profile_name') or rt.profile_name)
        rt.last_download = _remote_transfer_snapshot(
            result,
            overview_key='download_overview',
            extra_keys=('target_label', 'profile_name', 'local_root', 'downloaded_count', 'downloaded_items'),
        )
        rt.last_download['target_label'] = rt.target_label
        rt.last_download['profile_name'] = rt.profile_name
        rt.last_download['downloaded_items'] = result.get('downloaded_items') or []

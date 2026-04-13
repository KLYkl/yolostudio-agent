from __future__ import annotations

import json
import runpy
import shutil
from datetime import datetime, UTC
from pathlib import Path

SUITES = [
    'test_agent_server_chaos_p0.py',
    'test_agent_server_chaos_p1_followup.py',
    'test_agent_server_chaos_p1_replanning.py',
    'test_agent_server_chaos_p1_crossmainline.py',
    'test_agent_server_chaos_p1_repeat_tolerance.py',
    'test_agent_server_chaos_p1_recovery.py',
    'test_agent_server_chaos_p1_compare_resilience.py',
    'test_agent_server_chaos_p1_input_matrix.py',
    'test_agent_server_chaos_p1_revision_matrix.py',
    'test_agent_server_chaos_p1_running_matrix.py',
    'test_agent_server_chaos_p1_confirmation_matrix.py',
    'test_agent_server_chaos_p2_context.py',
    'test_agent_server_chaos_p2_integrity.py',
    'test_agent_server_chaos_p2_mismatch_matrix.py',
    'test_agent_server_chaos_p2_guardrail_misc.py',
    'test_agent_server_chaos_p2_crossmainline_extra.py',
]

SCENARIO_TOTAL = 100


def main() -> None:
    tests_dir = Path(__file__).resolve().parent
    tmp_root = tests_dir / '_tmp_agent_server_chaos_p0'
    shutil.rmtree(tmp_root, ignore_errors=True)
    for suite in SUITES:
        runpy.run_path(str(tests_dir / suite), run_name='__main__')
    output = {
        'ok': True,
        'scenario_total': SCENARIO_TOTAL,
        'suite_count': len(SUITES),
        'suites': SUITES,
        'generated_at_utc': datetime.now(UTC).isoformat(),
    }
    (tests_dir / 'test_agent_server_chaos_full_matrix_output.json').write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'agent server chaos full matrix ok ({SCENARIO_TOTAL} scenarios planned)')


if __name__ == '__main__':
    main()

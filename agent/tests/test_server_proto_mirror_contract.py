from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    here = Path(__file__).resolve()
    bootstrap_candidates: list[Path] = []
    for candidate in here.parents:
        if (candidate / 'deploy' / 'scripts' / 'sync_server_proto.py').exists():
            bootstrap_candidates.append(candidate)
            bootstrap_candidates.append(candidate.parent)
            break
    for candidate in bootstrap_candidates:
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from deploy.scripts.sync_server_proto import (
    DEPLOY_ONLY_PATHS,
    MANAGED_MAPPINGS,
    SERVER_PROTO_ROOT,
    collect_drift,
)


def main() -> None:
    assert MANAGED_MAPPINGS, 'managed mirror mappings must not be empty'
    for mapping in MANAGED_MAPPINGS:
        assert mapping.source.exists(), mapping.source
        assert str(mapping.dest).startswith(str(SERVER_PROTO_ROOT)), mapping.dest

    for relative_path in DEPLOY_ONLY_PATHS:
        deploy_only_path = SERVER_PROTO_ROOT / relative_path
        assert deploy_only_path.exists(), deploy_only_path

    drift = collect_drift()
    assert not drift, '\n'.join(f'{record.kind}: {record.dest} <= {record.source}' for record in drift[:50])
    print('server proto mirror contract ok')


if __name__ == '__main__':
    main()

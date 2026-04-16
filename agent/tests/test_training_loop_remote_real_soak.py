from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.tests.training_loop_soak_support import (
    parse_allowed_tuning_params,
    run_training_loop_soak,
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run real training-loop soak validation.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data-yaml', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', default='0')
    parser.add_argument('--training-environment', default='')
    parser.add_argument('--project', default='')
    parser.add_argument('--name', default='')
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--imgsz', type=int, default=None)
    parser.add_argument('--fraction', type=float, default=None)
    parser.add_argument('--classes', default='')
    parser.add_argument('--single-cls', action='store_true')
    parser.add_argument('--optimizer', default='')
    parser.add_argument('--freeze', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lr0', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--loop-name', default='')
    parser.add_argument('--managed-level', default='full_auto')
    parser.add_argument('--max-rounds', type=int, default=20)
    parser.add_argument('--target-metric', default='map50')
    parser.add_argument('--target-metric-value', type=float, default=None)
    parser.add_argument('--min-improvement', type=float, default=0.0)
    parser.add_argument('--no-improvement-rounds', type=int, default=999)
    parser.add_argument('--max-failures', type=int, default=2)
    parser.add_argument('--allowed-tuning-params', default='none')
    parser.add_argument('--auto-handle-oom', action='store_true')
    parser.add_argument('--include-case-sources', action='store_true')
    parser.add_argument('--include-test-sources', action='store_true')
    parser.add_argument('--max-imgsz', type=int, default=1536)
    parser.add_argument('--min-batch', type=int, default=1)
    parser.add_argument('--knowledge-mode', choices=['forced', 'real'], default='forced')
    parser.add_argument('--forced-action', default='continue_observing')
    parser.add_argument('--state-dir', default='')
    parser.add_argument('--loop-poll-interval', type=float, default=5.0)
    parser.add_argument('--watch-poll-interval', type=float, default=5.0)
    parser.add_argument('--wait-mode', choices=['terminal', 'review_or_terminal'], default='terminal')
    parser.add_argument('--auto-resume-reviews', type=int, default=0)
    parser.add_argument('--recreate-service-on-review-resume', action='store_true')
    parser.add_argument('--timeout', type=float, default=0.0)
    args = parser.parse_args()

    classes = args.classes.strip() if isinstance(args.classes, str) else args.classes
    if classes == '':
        classes = None

    payload = run_training_loop_soak(
        output_path=args.output,
        model=args.model,
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        device=args.device,
        training_environment=args.training_environment,
        project=args.project,
        name=args.name,
        batch=args.batch,
        imgsz=args.imgsz,
        fraction=args.fraction,
        classes=classes,
        single_cls=args.single_cls if args.single_cls else None,
        optimizer=args.optimizer,
        freeze=args.freeze,
        resume=args.resume if args.resume else None,
        lr0=args.lr0,
        patience=args.patience,
        workers=args.workers,
        amp=args.amp if args.amp else None,
        loop_name=args.loop_name,
        managed_level=args.managed_level,
        max_rounds=args.max_rounds,
        target_metric=args.target_metric,
        target_metric_value=args.target_metric_value,
        min_improvement=args.min_improvement,
        no_improvement_rounds=args.no_improvement_rounds,
        max_failures=args.max_failures,
        allowed_tuning_params=parse_allowed_tuning_params(args.allowed_tuning_params),
        auto_handle_oom=args.auto_handle_oom,
        include_case_sources=args.include_case_sources,
        include_test_sources=args.include_test_sources,
        max_imgsz=args.max_imgsz,
        min_batch=args.min_batch,
        knowledge_mode=args.knowledge_mode,
        forced_action=args.forced_action,
        state_dir=args.state_dir or None,
        loop_poll_interval=args.loop_poll_interval,
        watch_poll_interval=args.watch_poll_interval,
        wait_mode=args.wait_mode,
        auto_resume_reviews=args.auto_resume_reviews,
        recreate_service_on_review_resume=args.recreate_service_on_review_resume,
        timeout=args.timeout if args.timeout > 0 else None,
    )
    print(payload.get('output_path') or json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()

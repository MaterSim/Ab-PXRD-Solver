import argparse
import os
import sys
from pxrd_app.cli import build_common_parser, build_run_state, collect_input_csv_files, run_csv_batch
from pxrd_app.runtime import emit_timing_summary, write_results_csv
from pxrd_app.constants import DEFAULT_STATE as default_state
from pxrd_app.core import attach_run_log, detach_run_log, run_pipeline, logger

def run_deterministic(csv_path: str, args: argparse.Namespace) -> dict | None:
    run_state = build_run_state(default_state, logger, args, csv_path)
    system_log_handler = attach_run_log(run_state)

    try:
        print("Starting deterministic PXRD pipeline.")
        run_state = run_pipeline(run_state)
    except KeyboardInterrupt:
        print("Process interrupted by user")
    finally:
        detach_run_log(system_log_handler)
        emit_timing_summary(logger, run_state)
        write_results_csv(csv_path, run_state)
        print("Exiting deterministic main thread")

def _parse_args() -> argparse.Namespace:
    parser = build_common_parser("Run PXRD pipeline in deterministic mode")
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    try:
        csv_files = collect_input_csv_files(args.input)
        if args.reverse: csv_files = list(reversed(csv_files))
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)
    try:
        os.makedirs(args.output + "/cifs", exist_ok=True)
        os.makedirs(args.output + "/logs", exist_ok=True)
        run_csv_batch(csv_files, args, run_deterministic)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)

if __name__ == "__main__":
    main()
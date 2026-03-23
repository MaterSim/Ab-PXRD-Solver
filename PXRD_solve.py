import argparse
import sys

from pxrd_app.cli import build_common_parser, build_run_state_from_args, collect_input_csv_files, run_csv_batch
from pxrd_app.runtime import print_result_summary, write_results_csv
from pxrd_app.constants import DEFAULT_STATE as default_state
from pxrd_app.core import (
    _attach_system_run_log,
    _detach_system_run_log,
    _run_pipeline_fallback,
    logger,
)

# Test logger output at startup
logger.info("[LOGGER TEST] PXRD_solve.py logger.info is working. This should appear in both PXRD_solver.log and the console.")


def run_deterministic(csv_path: str, args: argparse.Namespace) -> dict | None:
    run_state = build_run_state_from_args(default_state, logger, args, csv_path)
    system_log_handler = _attach_system_run_log(run_state)
    result = None

    try:
        print("Starting deterministic PXRD pipeline.")
        result = _run_pipeline_fallback(
            run_state,
            announce_bug_switch=False,
            status_label="deterministic_success",
        )
        return result
    except KeyboardInterrupt:
        print("Process interrupted by user")
        return None
    finally:
        system_log_path = run_state.get("system_run_log")
        if system_log_path:
            print(f"Saved consolidated run log to {system_log_path}")
        _detach_system_run_log(system_log_handler)
        print_result_summary(
            logger,
            run_state,
            result,
            success_message="Deterministic pipeline completed successfully!",
            failure_prefix="Deterministic pipeline finished without a solution",
        )
        write_results_csv(csv_path, run_state, result.get("status") if isinstance(result, dict) else "unknown")
        print("Exiting deterministic main thread")


def _parse_args() -> argparse.Namespace:
    parser = build_common_parser("Run PXRD pipeline in deterministic mode")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        csv_files = collect_input_csv_files(args.input)
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)
    try:
        run_csv_batch(csv_files, args, run_deterministic)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
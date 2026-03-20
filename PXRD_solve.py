import argparse
import sys

from pxrd_cli import build_common_parser, build_run_state_from_args, collect_input_csv_files, run_csv_batch
from PXRD_agent import (
    _attach_system_run_log,
    _detach_system_run_log,
    _run_pipeline_fallback,
    default_state,
    logger,
)


FAILURE_STATUSES = {"no_cells", "no_solution"}


def _fmt_seconds(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    total_minutes = int(total_seconds // 60)
    seconds_remain = total_seconds - (60 * total_minutes)
    if total_minutes >= 60:
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours}h {minutes}m {seconds_remain:04.1f}s"
    return f"{total_minutes}m {seconds_remain:04.1f}s"


def _print_result_summary(run_state: dict, result: dict | None) -> None:
    result_status = result.get("status", "") if isinstance(result, dict) else ""

    timing_breakdown = run_state.get("timing_breakdown_seconds") if isinstance(run_state, dict) else None
    if isinstance(timing_breakdown, dict):
        spg_cell_s = float(timing_breakdown.get("spg_and_cell", 0.0))
        structure_s = float(timing_breakdown.get("structure_inference", 0.0))
        total_s = float(timing_breakdown.get("total", spg_cell_s + structure_s))
        timing_line = (
            f"Timing summary: SPG+Cell={_fmt_seconds(spg_cell_s)} | "
            f"Structure={_fmt_seconds(structure_s)} | Total={_fmt_seconds(total_s)}"
        )
        logger.info(timing_line)
        print(timing_line)

    if result_status in FAILURE_STATUSES:
        reason = {
            "no_cells": "no valid unit cells found",
            "no_solution": "no accepted structure found",
        }.get(result_status, result_status)
        print(f"Deterministic pipeline finished without a solution ({reason}).")
    else:
        print("Deterministic pipeline completed successfully!")


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
        _print_result_summary(run_state, result)
        print("Exiting deterministic main thread")


def _parse_args() -> argparse.Namespace:
    parser = build_common_parser("Run PXRD pipeline in deterministic mode")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        csv_files = collect_input_csv_files(args.input_csv)
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
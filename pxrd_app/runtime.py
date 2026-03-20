FAILURE_STATUSES = {"no_cells", "no_solution"}


def format_seconds(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    total_minutes = int(total_seconds // 60)
    seconds_remain = total_seconds - (60 * total_minutes)
    if total_minutes >= 60:
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours}h {minutes}m {seconds_remain:04.1f}s"
    return f"{total_minutes}m {seconds_remain:04.1f}s"


def build_timing_summary_line(run_state: dict | None) -> str | None:
    timing_breakdown = run_state.get("timing_breakdown_seconds") if isinstance(run_state, dict) else None
    if not isinstance(timing_breakdown, dict):
        return None

    spg_cell_s = float(timing_breakdown.get("spg_and_cell", 0.0))
    structure_s = float(timing_breakdown.get("structure_inference", 0.0))
    total_s = float(timing_breakdown.get("total", spg_cell_s + structure_s))
    return (
        f"Timing summary: SPG+Cell={format_seconds(spg_cell_s)} | "
        f"Structure={format_seconds(structure_s)} | Total={format_seconds(total_s)}"
    )


def emit_timing_summary(logger, run_state: dict | None) -> str | None:
    timing_line = build_timing_summary_line(run_state)
    if timing_line is None:
        return None
    logger.info(timing_line)
    print(timing_line)
    return timing_line


def print_result_summary(
    logger,
    run_state: dict | None,
    result: dict | None,
    *,
    success_message: str,
    failure_prefix: str,
) -> None:
    result_status = result.get("status", "") if isinstance(result, dict) else ""
    emit_timing_summary(logger, run_state)

    if result_status in FAILURE_STATUSES:
        reason = {
            "no_cells": "no valid unit cells found",
            "no_solution": "no accepted structure found",
        }.get(result_status, result_status)
        print(f"{failure_prefix} ({reason}).")
    else:
        print(success_message)
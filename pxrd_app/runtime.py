import csv
import os
from pathlib import Path

FAILURE_STATUSES = {"no_cells", "no_solution"}

RESULTS_CSV = Path("Results") / "results_summary.csv"
CSV_COLUMNS = [
    "csv_file_name", "Runtime", "Number of relaxed structures",
    "Status", "E", "dE", "R2", "Chi2", "SPG", "Wyckoff", "Cell",
]


def _extract_xtal_cell(xtal) -> str | None:
    """Return a compact cell-parameter string from a pyxtal object."""
    if xtal is None:
        return None
    try:
        return xtal.lattice.encode()
    except Exception:
        pass
    try:
        para = xtal.lattice.get_para()
        a, b, c, alpha, beta, gamma = para
        return f"{a:.4f} {b:.4f} {c:.4f} {alpha:.2f} {beta:.2f} {gamma:.2f}"
    except Exception:
        return None


def _extract_xtal_wyckoff(xtal) -> str | None:
    """Return Wyckoff labels as a compact string, e.g. '[6f], [1a]'."""
    if xtal is None:
        return None
    try:
        sites = xtal.atom_sites
        labels = []
        for site in sites:
            wp = getattr(site, "wp", None)
            label = None
            multiplicity = None
            if wp is not None:
                try:
                    label = wp.get_label()
                except Exception:
                    multiplicity = getattr(wp, "multiplicity", None)
                    letter = getattr(wp, "letter", None)
                    if multiplicity is not None and letter is not None:
                        label = f"{int(multiplicity)}{letter}"
                if multiplicity is None:
                    try:
                        multiplicity = int(getattr(wp, "multiplicity", None))
                    except Exception:
                        multiplicity = None
            if label:
                labels.append((int(multiplicity) if multiplicity is not None else 0, str(label)))

        if not labels:
            return None

        labels.sort(key=lambda item: (-item[0], item[1]))
        return ", ".join(f"[{label}]" for _mult, label in labels)
    except Exception:
        return None


def write_results_csv(input_csv: str, run_state: dict | None, result: dict | None) -> None:
    """Append one summary row to Results/results_summary.csv."""
    run_state = run_state or {}
    result = result or {}

    # --- Status ---
    status_raw = result.get("status", "unknown")
    is_failure = status_raw in FAILURE_STATUSES or status_raw == "unknown"
    status_label = "Failure" if is_failure else "Success"

    # --- Runtime ---
    timing = run_state.get("timing_breakdown_seconds") or {}
    spg_cell_s = float(timing.get("spg_and_cell", 0.0))
    structure_s = float(timing.get("structure_inference", 0.0))
    runtime_s = float(timing.get("total", spg_cell_s + structure_s))
    runtime_str = format_seconds(runtime_s) if runtime_s > 0 else ""

    # --- Structure count ---
    structure_log = run_state.get("structure_log") or []
    n_relaxed = len(structure_log)

    # --- Best-structure metrics (only on success) ---
    E = dE = R2 = Chi2 = SPG = Wyckoff = Cell = ""
    if not is_failure:
        wr = run_state.get("wyckoff_result") or {}
        xtal = wr.get("xtal")
        _f = lambda v: f"{v:.6g}" if v is not None else ""
        E = _f(wr.get("selected_energy"))
        dE = _f(wr.get("eng_rel"))
        R2 = _f(wr.get("r2"))
        Chi2 = _f(wr.get("chi2"))
        SPG = str(run_state.get("spg") or wr.get("spg") or "")
        Wyckoff = _extract_xtal_wyckoff(xtal) or ""
        Cell = _extract_xtal_cell(xtal) or ""

    csv_file_name = os.path.basename(input_csv)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with open(RESULTS_CSV, "a+", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([
            csv_file_name, runtime_str, n_relaxed,
            status_label, E, dE, R2, Chi2, SPG, Wyckoff, Cell,
        ])


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
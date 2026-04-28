import csv
import os
from pathlib import Path
from typing import Optional, Dict, List

FAILURE_STATUSES = {"no_cells", "no_solution"}

_DEFAULT_RESULTS_DIR = "Results"
CSV_COLUMNS = [
    "csv_file_name", "Runtime", "N_struc", "N_attempts", "N_est",
    "Status", "E", "dE", "R2", "Chi2", "SPG", "Wyckoff", "Cell",
]


def _format_scalar(value, decimals: int = 4) -> str:
    if value is None:
        return ""
    try:
        text = f"{float(value):.{int(decimals)}f}".rstrip("0").rstrip(".")
        return text if text not in {"-0", ""} else "0"
    except Exception:
        return ""

def _extract_xtal_wyckoff(xtal) -> Optional[str]:
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


def write_results_csv(input_csv: str, run_state: Optional[dict]) -> None:
    """Upsert one summary row in <results_dir>/summary.csv by csv_file_name."""
    # --- Status ---
    status_label = run_state.get("status")
    print(f"Run status (status_label): {status_label}")

    # --- Runtime ---
    timing = run_state.get("timing_breakdown_seconds") or {}
    spg_cell_s = float(timing.get("spg_and_cell", 0.0))
    structure_s = float(timing.get("structure_inference", 0.0))
    runtime_s = float(timing.get("total", spg_cell_s + structure_s))
    runtime_str = format_seconds(runtime_s) if runtime_s > 0 else ""

    # --- Structure count ---
    structure_log = run_state.get("structure_log") or []
    n_struc = len(structure_log)
    n_attempts = run_state.get("attempt_count")
    n_est = run_state.get("Total_est", 0)

    # --- Best-structure metrics (only on success) ---
    E = dE = R2 = Chi2 = SPG = Wyckoff = Cell = ""
    if status_label == "Success":
        wr = run_state.get("wyckoff_result") or {}
        xtal = wr.get("xtal")
        E = _format_scalar(wr.get("selected_energy"), 4)
        dE = _format_scalar(wr.get("eng_rel"), 4)
        R2 = _format_scalar(wr.get("r2"), 4)
        Chi2 = _format_scalar(wr.get("chi2"), 4)
        SPG = str(run_state.get("spg") or wr.get("spg") or "")
        Wyckoff = _extract_xtal_wyckoff(xtal) or ""
        Cell = str([_format_scalar(x, 4) for x in xtal.lattice.encode()])

    csv_file_name = os.path.basename(input_csv)
    is_placeholder_failure = (
        status_label == "Failure"
        and runtime_s <= 0
        and n_struc == 0
        and int(n_est or 0) == 0
    )

    if is_placeholder_failure:
        # Skip writing empty placeholder failures that can occur when a worker dies
        # before real progress is recorded. This avoids stale false negatives.
        print(f"Skipping placeholder failure row for {csv_file_name}")
        return

    row_data = {
        "csv_file_name": csv_file_name,
        "Runtime": runtime_str,
        "N_struc": str(n_struc),
        "N_attempts": str(n_attempts),
        "N_est": str(n_est),
        "Status": status_label if status_label is not None else "Unknown",
        "E": E,
        "dE": dE,
        "R2": R2,
        "Chi2": Chi2,
        "SPG": SPG,
        "Wyckoff": Wyckoff,
        "Cell": Cell,
    }

    results_csv = Path(run_state.get("results_dir") or _DEFAULT_RESULTS_DIR) / "summary.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: List[Dict[str, str]] = []
    if results_csv.exists() and results_csv.stat().st_size > 0:
        with open(results_csv, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not row:
                    continue
                existing_rows.append({col: row.get(col, "") for col in CSV_COLUMNS})

    updated_rows: List[Dict[str, str]] = []
    replaced = False
    for row in existing_rows:
        if row.get("csv_file_name") == csv_file_name:
            if replaced:
                continue
            existing_status = (row.get("Status") or "").strip()
            if existing_status == "Success" and status_label != "Success":
                # Keep the successful row; do not downgrade it with a later failure.
                updated_rows.append(row)
            else:
                updated_rows.append(row_data)
            replaced = True
            continue
        updated_rows.append(row)

    if not replaced:
        updated_rows.append(row_data)

    with open(results_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(updated_rows)


def format_seconds(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    total_minutes = int(total_seconds // 60)
    seconds_remain = total_seconds - (60 * total_minutes)
    if total_minutes >= 60:
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours}h {minutes}m {seconds_remain:04.1f}s"
    return f"{total_minutes}m {seconds_remain:04.1f}s"


def emit_timing_summary(logger, run_state: Optional[dict]) -> Optional[str]:
    """Log and return a timing summary line for the run_state."""
    timing_breakdown = run_state.get("timing_breakdown_seconds") if isinstance(run_state, dict) else None
    if not isinstance(timing_breakdown, dict):
        return None

    spg_cell_s = float(timing_breakdown.get("spg_and_cell", 0.0))
    structure_s = float(timing_breakdown.get("structure_inference", 0.0))
    total_s = float(timing_breakdown.get("total", spg_cell_s + structure_s))
    timing_line = (
        f"Timing summary: SPG+Cell={format_seconds(spg_cell_s)} | "
        f"Structure={format_seconds(structure_s)} | Total={format_seconds(total_s)}"
    )
    logger.info(timing_line)
    return timing_line

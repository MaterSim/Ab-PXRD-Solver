import csv
import os
from pathlib import Path

FAILURE_STATUSES = {"no_cells", "no_solution"}

_DEFAULT_RESULTS_DIR = "Results"
CSV_COLUMNS = [
    "csv_file_name", "Runtime", "N_struc",
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


def _approx_equal(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(float(a) - float(b)) <= tol


def _format_cell_components(values: list[float], decimals: int = 4) -> str:
    return "[" + ", ".join(_format_scalar(v, decimals) for v in values) + "]"


def _compact_cell_from_para(para) -> str | None:
    try:
        a, b, c, alpha, beta, gamma = [float(x) for x in para]
    except Exception:
        return None

    if (
        _approx_equal(alpha, 90.0, 1e-2)
        and _approx_equal(beta, 90.0, 1e-2)
        and _approx_equal(gamma, 90.0, 1e-2)
    ):
        if _approx_equal(a, b) and _approx_equal(b, c):
            return _format_cell_components([a])
        if _approx_equal(a, b):
            return _format_cell_components([a, c])
        return _format_cell_components([a, b, c])

    if (
        _approx_equal(alpha, 90.0, 1e-2)
        and _approx_equal(beta, 90.0, 1e-2)
        and _approx_equal(gamma, 120.0, 1e-2)
        and _approx_equal(a, b)
    ):
        return _format_cell_components([a, c])

    if _approx_equal(alpha, 90.0, 1e-2) and _approx_equal(gamma, 90.0, 1e-2):
        return _format_cell_components([a, b, c, beta], decimals=4)

    return _format_cell_components([a, b, c, alpha, beta, gamma], decimals=4)


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


def write_results_csv(input_csv: str, run_state: dict | None) -> None:
    """Append one summary row to <results_dir>/summary.csv."""
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

    # --- Best-structure metrics (only on success) ---
    E = dE = R2 = Chi2 = SPG = Wyckoff = Cell = ""
    if status_label == "Success":
        wr = run_state.get("wyckoff_result") or {}
        xtal = wr.get("xtal")
        E = _format_scalar(wr.get("selected_energy"), 6)
        dE = _format_scalar(wr.get("eng_rel"), 6)
        R2 = _format_scalar(wr.get("r2"), 6)
        Chi2 = _format_scalar(wr.get("chi2"), 6)
        SPG = str(run_state.get("spg") or wr.get("spg") or "")
        Wyckoff = _extract_xtal_wyckoff(xtal) or ""
        Cell = xtal.lattice.encode()

    csv_file_name = os.path.basename(input_csv)

    results_csv = Path(run_state.get("results_dir") or _DEFAULT_RESULTS_DIR) / "summary.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists() or results_csv.stat().st_size == 0
    with open(results_csv, "a+", newline="") as fh:
        writer = csv.writer(fh)
        if write_header: writer.writerow(CSV_COLUMNS)
        writer.writerow([
            csv_file_name, runtime_str, n_struc,
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


def emit_timing_summary(logger, run_state: dict | None) -> str | None:
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
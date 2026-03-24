import logging
import os
import random
import re
import sys
import time
import copy
from pathlib import Path
from uuid import uuid4
from functools import lru_cache
import shutil
import pandas as pd
import numpy as np

import argparse
from pxrd_app.constants import VALID_LATTICE_SYMMETRIES
from pxrd_app.inference import (
    CRYSTAL_SYSTEM_PRIORITY,
    SPG_INFER_BACKENDS,
    infer_formula_spg,
    infer_spacegroups_from_backend,
    spg_to_crystal_system,
)
from pxrd_app.plot import plot_energy_vs_r2

from tools.manager import RawDataManager, CellManager
from tools.solver import (
    CellSolver,
    search_solution,
    enumerate_wyckoff_multi_spg,
    get_adaptive_wp_limits,
)
from tools.utils import parse_formula, get_volume_from_density
from tools.density import predict_density_ensemble
from pyxtal.symmetry import Group


# Configure logging with both file and console handlers, avoiding duplicate handlers
logger = logging.getLogger("pxrd_agent")
logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler('PXRD_solver.log')
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_name_token(value: str | None, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    token = "".join(cleaned).strip("_")
    return token or fallback

def _is_important_runlog_message(message: str) -> bool:
    text = str(message or "").strip()
    if not text:
        return False

    noisy_prefixes = (
        "Peak at index ",
        "Removed peak ",
        "Trying ",
        "Generated ",
        "Reached maximum number of solutions",
        "Filtered singular orthorhombic hkl guess sets",
        "Solution for ",
        "Status | SPG | Dims (Sorted)",
        "KEEP   |",
        "DROP   |",
        "SKIP   |",
        "Z=",
        "SPG: ",
        "Using Materials Project MACE",
        "Using float32 for MACECalculator",
        "Using CPU",
        "Default dtype float32 does not match model dtype float64",
    )
    if text.startswith(noisy_prefixes):
        return False

    important_prefixes = (
        "=",
        "Run started:",
        "Input:",
        "Per-system run log:",
        "Starting pipeline",
        "Applying lattice symmetry filter:",
        "Lattice symmetry filter ",
        "Unknown lattice symmetry filter",
        "Selected inferred space group:",
        "Cell solving completed",
        "Phase 1",
        "No inferred SG candidate produced valid cells",
        "No valid unit cells found.",
        "Phase 2:",
        "Phase 2 strategy:",
        "Rank  ",
        "[Pair ",
        "Pair ",
        "WP #",
        "Space group:",
        "Adaptive Wyckoff solve:",
        "Attempt ",
        "Final refinement results:",
        "Best refinement plot saved to",
        "Best structure saved to",
        "Selected attempt ",
        "No satisfactory solution found across all attempts.",
        "No inferred space group met acceptance thresholds;",
        "Completed inferred SG sweep;",
        "Best inferred-SG score observed:",
        "Best accepted inferred-SG solution details:",
        "Accepted solution details:",
        "Good solution found early:",
        "Timing breakdown:",
        "1) SPG + cell inference:",
        "2) Structure inference:",
        "Total:",
        "Timing summary:",
        "Pipeline finished without a solution",
        "Pipeline completed successfully!",
        "Process interrupted by user",
        "Exiting main thread",
        "Saved consolidated run log to ",
    )
    if text.startswith(important_prefixes):
        return True

    important_substrings = (
        "rejected for spg=",
        "precheck error for spg=",
        "accepted solution found; moving to next ranked pair",
        "Refinement triggered:",
        "Refinement skipped:",
        "Perturbation refinement triggered:",
        "Perturbed refinement skipped:",
        "Low-sim early exit:",
        "Promising local minimum for current WP setting; adding",
        "Running ",
        "Perturbation ",
        "Early-stop deferred:",
        "Good refined fit found but energy is too high for early stop:",
        "Returning best locally intensified accepted candidate",
        "returning best refined fallback candidate",
        "returning no solution",
        "metrics: Wr=",
    )
    if any(token in text for token in important_substrings):
        return True

    if re.match(r"^\d+\s+\d+\s+\d", text):
        return True
    if re.match(r"^-{20,}$", text):
        return True
    if text.startswith("*"):
        return True

    return False


class _SystemRunLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return _is_important_runlog_message(record.getMessage())
        except Exception:
            return True

def _attach_system_run_log(state: dict) -> logging.Handler | None:
    try:
        results_dir = state.get("results_dir", "Results")
        os.makedirs(results_dir, exist_ok=True)
        log_path = _get_system_run_log_path(state)
        handler = logging.FileHandler(log_path, mode="a")
        handler.setFormatter(logging.Formatter("%(message)s"))
        # Removed filter so all messages are logged
        # Attach to pxrd_agent logger instead of root
        logger.addHandler(handler)
        state["system_run_log"] = log_path
        run_banner = (
            f"\n{'=' * 80}\n"
            f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Input: {state.get('pxrd_csv')}\n"
            f"{'=' * 80}\n"
            f"Per-system run log: {log_path}\n"
        )
        logger.info(run_banner)
        return handler
    except Exception as exc:
        logger.warning(f"Warning: failed to create per-system run log ({exc}).")
        return None


def _detach_system_run_log(handler: logging.Handler | None) -> None:
    if handler is None:
        return
    try:
        logger.removeHandler(handler)
    finally:
        try:
            handler.close()
        except Exception:
            pass


def _env_int(name: str, default: int, min_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if min_value is not None:
        value = max(min_value, value)
    return value


class StreamToLogger:
    """Redirect writes to a logger instance."""
    def __init__(self, logger_instance, level):
        self.logger = logger_instance
        self.level = level
        self._buffer = ""

    def write(self, message):
        if not message:
            return

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer:
            line = self._buffer.rstrip()
            if line:
                self.logger.log(self.level, line)
            self._buffer = ""


# Redirect stdout/stderr to logging to capture all output including strands library
#sys.stdout = StreamToLogger(logging.getLogger("stdout"), logging.INFO)
#sys.stderr = StreamToLogger(logging.getLogger("stderr"), logging.WARNING)

def _get_cell_solver_kwargs(state: dict) -> dict:
    hkl_max_raw = state.get("cell_solver_hkl_max", (2, 5, 6))
    theta_tols_raw = state.get("cell_solver_theta_tols", [0.1, 0.15, 0.5])

    try:
        hkl_max = tuple(int(x) for x in hkl_max_raw)
    except Exception:
        hkl_max = (2, 5, 6)

    try:
        theta_tols = [float(x) for x in theta_tols_raw]
    except Exception:
        theta_tols = [0.1, 0.15, 0.5]

    if len(hkl_max) != 3:
        hkl_max = (2, 5, 6)
    if not theta_tols:
        theta_tols = [0.1, 0.15, 0.5]

    return {
        "max_mismatch": max(0, int(state.get("cell_solver_max_mismatch", 12))),
        "hkl_max": hkl_max,
        "max_square": max(1, int(state.get("cell_solver_max_square", 28))),
        "total_square": max(1, int(state.get("cell_solver_total_square", 40))),
        "theta_tols": theta_tols,
        "min_abc": max(0.1, float(state.get("min_abc", 2.0))),
        "max_chi2": max(1e-6, float(state.get("cell_solver_max_chi2", 0.5))),
        "max_guess": max(100, int(state.get("cell_solver_max_guess", 50000))),
    }

def _run_data_preprocessor(pxrd_csv: str, state: dict) -> dict:
    formula_from_filename, spg_from_filename = infer_formula_spg(pxrd_csv)
    state["spg_from_filename"] = int(spg_from_filename) if spg_from_filename is not None else None
    formula_override = state.get("formula")
    formula = formula_override if formula_override else formula_from_filename
    if not formula:
        raise ValueError(
            "Cannot infer formula from file name. Provide --input-formula or use PXRD_<formula>_<spg>.csv naming."
        )

    infer_spg = bool(state.get("infer_spg_from_pxrd", False))
    spg_top_k = int(state.get("spg_top_k", 25))
    max_cell_volume = state.get("max_cell_volume")
    show_spg_predictions = bool(state.get("show_spg_predictions", False))
    spg_infer_backend = str(state.get("spg_infer_backend", "model") or "model").strip().lower()
    if spg_infer_backend not in SPG_INFER_BACKENDS:
        logger.warning(f"Unknown spg_infer_backend='{spg_infer_backend}', falling back to 'model'.")
        spg_infer_backend = "model"
    spg = int(spg_from_filename) if spg_from_filename is not None else 0
    composition = parse_formula(formula)

    density = predict_density_ensemble(formula, sigma=2.5)
    density_min = float(density['min'])
    density_max = float(density['max'])
    density_pred = float(density.get('prediction', density_max))

    # V1: estimated formula volume from the predicted density.
    # Effective max-cell-volume is max(V1*5, V2), where V2 is user input.
    density_for_v1 = density_pred if density_pred > 0 else max(density_max, 1e-6)
    formula_volume_v1 = float(get_volume_from_density(composition, density_for_v1))
    formula_volume_cap = 5.0 * formula_volume_v1
    user_max_volume_v2 = None if max_cell_volume is None else float(max_cell_volume)
    effective_max_cell_volume = (
        max(formula_volume_cap, user_max_volume_v2)
        if user_max_volume_v2 is not None
        else formula_volume_cap
    )
    max_cell_volume = effective_max_cell_volume
    state["max_cell_volume"] = effective_max_cell_volume
    state["formula_volume_v1"] = formula_volume_v1
    state["formula_volume_cap"] = formula_volume_cap
    state["max_cell_volume_input_v2"] = user_max_volume_v2

    min_abc = 2.0
    wavelength = 1.54184

    df = pd.read_csv(pxrd_csv, comment='#')
    x1 = df.iloc[:, 0].values
    y1 = df.iloc[:, 1].values
    data = RawDataManager(x1, y1, bg_subtract=False)
    data.get_peaks_from_scipy()
    data.filter_peaks_by_ml(threshold=0.8, min_height=3.0)
    peaks = data.peaks
    peak_positions = x1[peaks]

    if infer_spg:
        infer_result = infer_spacegroups_from_backend(
            x1=np.array(x1, dtype=float),
            y1=np.array(y1, dtype=float),
            peak_positions=np.array(peak_positions, dtype=float),
            formula=formula,
            spg_infer_backend=spg_infer_backend,
            spg_top_k=spg_top_k,
            min_abc=min_abc,
            max_cell_volume=max_cell_volume,
        )
        predictions = infer_result.get("predictions") or []
        if infer_result.get("source"):
            state["spg_prediction_source"] = infer_result["source"]
        if infer_result.get("smart_cell_raw_solutions_by_spg"):
            state["smart_cell_raw_solutions_by_spg"] = infer_result["smart_cell_raw_solutions_by_spg"]
        if infer_result.get("smart_cell_ranked_spg_cells"):
            state["smart_cell_ranked_spg_cells"] = infer_result["smart_cell_ranked_spg_cells"]
        if predictions: state["spg_predictions"] = predictions

    min_volume = float(get_volume_from_density(composition, max(density_max, 1e-6)))

    result = {
        "spg": int(spg),
        "formula": formula,
        "x1": x1.tolist(),
        "y1": y1.tolist(),
        "peaks": peaks.tolist(),
        "peak_positions": peak_positions.tolist(),
        "composition": composition,
        "density_min": density_min,
        "density_max": density_max,
        "min_volume": min_volume,
        "formula_volume_v1": formula_volume_v1,
        "formula_volume_cap": formula_volume_cap,
        "max_cell_volume": max_cell_volume,
        "min_abc": min_abc,
        "wavelength": wavelength,
    }
    state.update(result)
    return result


@lru_cache(maxsize=256)
def _get_group(spg: int) -> Group:
    return Group(int(spg))

def _format_wyckoff_labels_from_ids(spg: int, wp_ids) -> str:
    try:
        group = _get_group(int(spg))
        labels_nested = []
        for sub in (wp_ids or []):
            one_species = []
            for wp in (sub or []):
                label = group[int(wp)].get_label()
                one_species.append(str(label))
            labels_nested.append(one_species)
        return str(labels_nested)
    except Exception:
        return "[]"


def _run_cell_solver_stage(state: dict) -> dict:
    spg = state.get("spg")
    formula = state.get("formula")
    peak_positions = state.get("peak_positions")
    max_cells = state.get("max_cells")
    max_cell_volume = state.get("max_cell_volume")
    cell_solver_kwargs = _get_cell_solver_kwargs(state)

    def _filter_cells_by_max_volume(cells: list) -> tuple[list, int]:
        if max_cell_volume is None:
            return cells, 0
        cap = float(max_cell_volume)
        kept = []
        removed = 0
        for cell in cells:
            vol = float(getattr(cell, "size", float("inf")))
            if vol <= cap:
                kept.append(cell)
            else:
                removed += 1
        return kept, removed

    smart_raw_by_spg = state.get("smart_cell_raw_solutions_by_spg") or {}
    smart_backend_active = bool(
        state.get("infer_spg_from_pxrd", False)
        and str(state.get("spg_infer_backend", "model")).strip().lower() == "smart-cell"
    )
    if smart_backend_active and int(spg) in smart_raw_by_spg:
        raw_solutions = smart_raw_by_spg.get(int(spg), [])
        if raw_solutions:
            cells = CellManager.consolidate(raw_solutions, max_solutions=max_cells, merge_tol=0.05)
            cells, removed_by_volume = _filter_cells_by_max_volume(cells)
            state["cells"] = cells
            if not cells:
                text = (
                    f"Cell solving found no valid unit cells for formula {formula} in space group {spg} "
                    f"after applying max volume filter"
                    f" ({float(max_cell_volume):.2f} Å^3).\n"
                )
                return {
                    "status": "no_cells",
                    "message": text,
                    "cells": [],
                }
            text = (
                f"Cell solving completed for formula {formula} in space group {spg} "
                f"using SmartCellSolver cache.\n"
            )
            if removed_by_volume > 0:
                text += (
                    f"Filtered out {removed_by_volume} cell solution(s) with volume > "
                    f"{float(max_cell_volume):.2f} Å^3.\n"
                )
            return {
                "status": "success",
                "message": text,
                "cells": [{"dimensions": cell.dims, "missing_peaks": cell.missing} for cell in cells],
            }

    peak_positions_np = np.array(peak_positions)
    solver = CellSolver(
        spg,
        peak_positions_np,
        max_mismatch=cell_solver_kwargs["max_mismatch"],
        hkl_max=cell_solver_kwargs["hkl_max"],
        max_square=cell_solver_kwargs["max_square"],
        total_square=cell_solver_kwargs["total_square"],
        theta_tols=cell_solver_kwargs["theta_tols"],
        min_abc=cell_solver_kwargs["min_abc"],
        max_chi2=cell_solver_kwargs["max_chi2"],
        max_guess=cell_solver_kwargs["max_guess"],
        verbose=False,
    )
    solutions = solver.solve()
    sols = [
        (spg, sol['cell'], sol['mismatch'], sol['chi2'][1], sol['errors'], sol['id'], sol['match'])
        for sol in solutions
    ]

    if not sols:
        state["cells"] = []
        text = f"Cell solving found no valid unit cells for formula {formula} in space group {spg}.\n"
        return {
            "status": "no_cells",
            "message": text,
            "cells": [],
        }

    cells = CellManager.consolidate(sols, max_solutions=max_cells, merge_tol=0.05)
    cells, removed_by_volume = _filter_cells_by_max_volume(cells)

    if not cells:
        state["cells"] = []
        text = (
            f"Cell solving found no valid unit cells for formula {formula} in space group {spg} "
            f"after applying max volume filter ({float(max_cell_volume):.2f} Å^3).\n"
        )
        return {
            "status": "no_cells",
            "message": text,
            "cells": [],
        }

    state["cells"] = cells
    text = f"Cell solving completed for formula {formula} in space group {spg}.\n"
    if removed_by_volume > 0:
        text += (
            f"Filtered out {removed_by_volume} cell solution(s) with volume > "
            f"{float(max_cell_volume):.2f} Å^3.\n"
        )
    return {
        "status": "success",
        "message": text,
        "cells": [{"dimensions": cell.dims, "missing_peaks": cell.missing} for cell in cells],
    }


def _run_wyckoff_solver(state: dict, all_structure_log: list, structure_id_counter=None) -> str:
    spg = state.get("spg")
    formula = state.get("formula")
    cells = state.get("cells")
    composition = state.get("composition")
    density_min = state.get("density_min")
    density_max = state.get("density_max")
    wavelength = state.get("wavelength")
    pxrd_csv = state.get("pxrd_csv")
    INST_FILE = state.get("INST_FILE")
    thetas = state.get("thetas")
    resolution = state.get("resolution")
    SCALED_INTENSITY_TOL = state.get("SCALED_INTENSITY_TOL")
    ref_den = (density_min, density_max)
    x1 = np.array(state.get("x1"))
    y1 = np.array(state.get("y1"))
    peaks = np.array(state.get("peaks"))
    forced_wp_solution = state.get("forced_wp_solution")
    if "forced_wp_solution" in state: state.pop("forced_wp_solution", None)
    min_r2 = state.get("min_r2")
    max_chi2 = state.get("max_chi2")
    max_force = state.get("max_force")
    max_stress = state.get("max_stress")
    max_local_boosts = max(0, int(state.get("max_local_boosts", 1)))
    max_local_perturbations = max(0, int(state.get("max_local_perturbations", 2)))
    perturb_displacement = max(0.0, float(state.get("perturb_displacement", 0.06)))
    max_eng_rel_early_stop = state.get("max_eng_rel_early_stop", state.get("max_eng_rel", None))
    min_structures_before_early_stop = max(0, int(state.get("min_structures_before_early_stop", 10)))
    eng_min, sim_max = 1e10, 0.90

    results_dir = state.get("results_dir", "Results")
    os.makedirs(results_dir, exist_ok=True)
    tmp_root = Path("tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    run_token = f"{_safe_name_token(Path(str(pxrd_csv or '')).stem)}_{os.getpid()}_{uuid4().hex[:8]}"
    run_tmp_dir = tmp_root / f"run_{run_token}"
    run_tmp_dir.mkdir(parents=True, exist_ok=True)

    title = f'{formula} PXRD Prediction: Space Group {spg}'
    match_cif = f'{results_dir}/Match_{formula}_{spg}.cif'
    stale_result_cifs = [
        *Path(results_dir).glob(f"Match_{formula}_{spg}_attempt*.cif"),
        *Path(results_dir).glob(f"Match_{formula}_{spg}_attempt*_refined.cif"),
    ]
    for stale_path in stale_result_cifs:
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass
    attempts = max(1, int(state.get("multi_attempts", 3)))
    seed_base = int(state.get("seed_base", 20260315))

    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def _attempt_schedule(i: int) -> tuple[int, int, int]:
        schedules = [
            (5, 20, 9),
            (7, 25, 10),
            (10, 30, 12),
        ]
        if i < len(schedules):
            return schedules[i]
        return schedules[-1]

    def _score_result(res: dict) -> float:
        wr = res["wr"]
        r2 = res["r2"]
        chi2 = res["chi2"]
        return float((1.5 * r2) - (0.4 * wr) - (0.2 * chi2))

    def _meets_acceptance(res: dict) -> bool:
        r2 = res.get("r2")
        chi2 = res.get("chi2")
        if r2 is None or chi2 is None:
            return False
        return bool(r2 >= min_r2 and chi2 <= max_chi2)

    best_result = None
    best_score = -1e9
    struc_count = state.get("Struc_count") or 0

    #logger.info(f"Adaptive Wyckoff solve: {attempts} attempt(s), seed_base={seed_base}")
    for attempt_idx in range(attempts):
        seed = seed_base + 9973 * attempt_idx
        N1, N2, N3 = _attempt_schedule(attempt_idx)
        _set_seed(seed)

        attempt_prefix = run_tmp_dir / f"Match_{formula}_{spg}_attempt{attempt_idx + 1}"
        attempt_png = str(attempt_prefix.with_suffix(".png"))
        attempt_cif = str(attempt_prefix.with_suffix(".cif"))
        attempt_refinement_png = str(attempt_prefix.with_name(f"{attempt_prefix.name}_refinement.png"))
        logger.info(
            f"Attempt {attempt_idx + 1}/{attempts}: seed={seed}, schedule=(N1={N1}, N2={N2}, N3={N3}), "
            f"Boosts={max_local_boosts}, Perturb: {max_local_perturbations}/{perturb_displacement:.3f}, "
            f"{struc_count} structures")

        #if struc_count != len(all_structure_log):
        #    logger.error(
        #        f"Error: Struc_count ({struc_count}) does not match length of all_structure_log "
        #        f"({len(all_structure_log)}). This may indicate a mismatch in structure logging."
        #    )
        #    import sys; sys.exit(1)

        wr, r2, chi2, xtal, eng_best, selected_eng, selected_eng_rel, struc_count = search_solution(
            cells[:N1],
            spg,
            composition,
            ref_den,
            title,
            attempt_png,
            attempt_cif,
            pxrd_csv,
            peaks,
            x1,
            y1,
            eng_min,
            sim_max,
            N1,
            N2,
            N3,
            struc_count if structure_id_counter is None else structure_id_counter,
            max_force,
            max_stress,
            wavelength,
            thetas,
            resolution,
            SCALED_INTENSITY_TOL,
            INST_FILE,
            logger,
            min_r2,
            max_chi2,
            max_local_boosts=max_local_boosts,
            max_local_perturbations=max_local_perturbations,
            perturb_displacement=perturb_displacement,
            structure_log=all_structure_log,
            max_eng_rel_early_stop=max_eng_rel_early_stop,
            min_structures_before_early_stop=min_structures_before_early_stop,
            forced_wp_solution=forced_wp_solution,
        )

        print(f"{struc_count} new structure(s). Total structures: {len(all_structure_log)}.")
        # Accumulate struc_count with the number of new structures generated in this attempt
        if wr is None: continue

        candidate = {
            "wr": float(wr),
            "r2": float(r2),
            "chi2": float(chi2),
            "xtal": xtal,
            "eng_best": float(eng_best),
            "selected_energy": float(selected_eng) if selected_eng is not None else None,
            "eng_rel": float(selected_eng_rel) if selected_eng_rel is not None else None,
            "attempt": attempt_idx + 1,
            "seed": seed,
            "png": attempt_refinement_png,
            "cif": attempt_cif,
            "accepted": False,
        }
        candidate["accepted"] = _meets_acceptance(candidate)
        score = _score_result(candidate)
        selected_energy_text = (
            f", E={candidate['selected_energy']:.4f}, dE={candidate['eng_rel']:.4f}"
            if candidate["selected_energy"] is not None and candidate["eng_rel"] is not None
            else ""
        )
        logger.info(
            f"Metrics: Wr={wr:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}, "
            f"score={score:.4f}, accepted={candidate['accepted']}{selected_energy_text}"
        )

        if score > best_score:
            best_score = score
            best_result = candidate


        # Early stop: excellent solution found
        if candidate["accepted"] and len(all_structure_log) >= min_structures_before_early_stop and (
            candidate["r2"] >= max(min_r2 + 0.02, 0.97)
            or candidate["chi2"] <= min(max_chi2 * 0.7, 0.08)
        ):
            logger.info(f"Early stop: excellent solution found at attempt {attempt_idx + 1}.")
            break

        # Hard cap: exit immediately if min_structures_before_early_stop is reached
        if len(all_structure_log) >= min_structures_before_early_stop:
            logger.info(f"Structure cap reached ({len(all_structure_log)}/{min_structures_before_early_stop}); exiting search loop.")
            break

    state["structure_log"] = all_structure_log
    state["Struc_count"] = struc_count

    if best_result is None:
        logger.info("No satisfactory solution found across all attempts.")
        state["wyckoff_result"] = {
            "spg": spg,
            "accepted": False,
                "Struc_count": struc_count,
            "wr": None,
            "r2": None,
            "chi2": None,
            "eng_best": eng_min,
            "attempt": None,
            "seed": None,
            "png": None,
            "cif": None,
            "score": None,
        }
        text = f"Wyckoff solving completed for formula {formula} in space group {spg}.\n"
        text += f"Adaptive attempts: {attempts}, seed_base: {seed_base}\n"
        text += f"Best similarity: {sim_max:.3f}, Minimum energy per atom: {eng_min:.3f} eV\n"
        text += "No satisfactory solution found.\n"
        return text

    best_result["xtal"].to_file(match_cif)

    wr = best_result["wr"]
    r2 = best_result["r2"]
    chi2 = best_result["chi2"]
    logger.info(f"\nFinal refinement results: Wr={wr:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}")
    if os.path.exists(best_result["png"]):
        logger.info(f"Best refinement plot saved to {best_result['png']}")
    logger.info(f"Best structure saved to {match_cif}")
    logger.info(
        f"Selected attempt {best_result['attempt']} (seed={best_result['seed']}, score={best_score:.4f})"
    )
    #logger.info(best_result["xtal"])
    best_result["spg"] = spg
    best_result["score"] = best_score
    state["wyckoff_result"] = best_result
    best_result["Struc_count"] = struc_count

    text = f"Wyckoff solving completed for formula {formula} in space group {spg}.\n"
    text += f"Adaptive attempts: {attempts}, seed_base: {seed_base}\n"
    text += f"Best similarity: {sim_max:.3f}, Minimum energy per atom: {best_result['eng_best']:.3f} eV\n"
    text += f"Final Rietveld refinement results: Wr={wr:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}\n"
    text += f"Selected attempt: {best_result['attempt']} (seed={best_result['seed']})\n"
    if not best_result["accepted"]:
        text += "Best refined candidate did not meet the acceptance thresholds, but was kept as a fallback result.\n"
    if os.path.exists(best_result["png"]):
        text += f"Best refinement plot saved to {best_result['png']}\n"
    text += f"Best structure saved to {match_cif}\n"
    return text


def _run_pipeline_fallback(
    state: dict,
    status_label: str = "fallback_success",
) -> dict:
    pipeline_start_time = time.perf_counter()
    spg_cell_phase_end_time: float | None = None
    structure_phase_start_time: float | None = None

    logger.info("Using deterministic pipeline execution.")
    _run_data_preprocessor(state["pxrd_csv"], state)

    def _emit_progress(message: str) -> None:
        # Use the logger to handle both console and file output
        logger.info(str(message))

    def _format_elapsed(seconds: float) -> str:
        total_seconds = max(0.0, float(seconds))
        total_minutes = int(total_seconds // 60)
        seconds_remain = total_seconds - (60 * total_minutes)
        if total_minutes >= 60:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m {seconds_remain:04.1f}s"
        return f"{total_minutes}m {seconds_remain:04.1f}s"

    def _current_timing_breakdown_seconds() -> dict:
        nonlocal spg_cell_phase_end_time, structure_phase_start_time
        now = time.perf_counter()
        if spg_cell_phase_end_time is None:
            spg_cell_seconds = now - pipeline_start_time
            structure_seconds = 0.0
        else:
            spg_cell_seconds = max(0.0, spg_cell_phase_end_time - pipeline_start_time)
            if structure_phase_start_time is None:
                structure_seconds = max(0.0, now - spg_cell_phase_end_time)
            else:
                structure_seconds = max(0.0, now - structure_phase_start_time)

        total_seconds = spg_cell_seconds + structure_seconds
        return {
            "spg_and_cell": float(spg_cell_seconds),
            "structure_inference": float(structure_seconds),
            "total": float(total_seconds),
        }

    def _emit_timing_breakdown() -> None:
        breakdown = _current_timing_breakdown_seconds()
        state["timing_breakdown_seconds"] = breakdown
        _emit_progress("Timing breakdown:")
        _emit_progress(f"  1) SPG + cell inference: {_format_elapsed(breakdown['spg_and_cell'])}")
        _emit_progress(f"  2) Structure inference: {_format_elapsed(breakdown['structure_inference'])}")
        _emit_progress(f"  Total: {_format_elapsed(breakdown['total'])}")

    def _emit_accepted_solution_details(spg_value: int, result: dict, prefix: str = "Accepted solution") -> None:
        wr = result.get("wr")
        r2 = result.get("r2")
        chi2 = result.get("chi2")
        score = result.get("score")
        selected_energy = result.get("selected_energy")
        eng_rel = result.get("eng_rel")
        attempt = result.get("attempt")
        seed = result.get("seed")
        cif = result.get("cif")
        png = result.get("png")

        global global_structure_log

        wr_text = f"{float(wr):.4f}" if wr is not None else "n/a"
        r2_text = f"{float(r2):.4f}" if r2 is not None else "n/a"
        chi2_text = f"{float(chi2):.4f}" if chi2 is not None else "n/a"
        score_text = f"{float(score):.4f}" if score is not None else "n/a"
        energy_text = f"{float(selected_energy):.4f}" if selected_energy is not None else "n/a"
        eng_rel_text = f"{float(eng_rel):.4f}" if eng_rel is not None else "n/a"
        attempt_text = str(attempt) if attempt is not None else "n/a"
        seed_text = str(seed) if seed is not None else "n/a"
        cif_text = str(cif) if cif else "n/a"
        png_text = str(png) if png else "n/a"

        _emit_progress(
            f"{prefix} details: spg={spg_value}, Wr={wr_text}, R2={r2_text}, "
            f"Chi2={chi2_text}, score={score_text}, E={energy_text}, dE={eng_rel_text}, "
            f"attempt={attempt_text}, seed={seed_text}"
        )
        _emit_progress(f"CIF={cif_text}, PNG={png_text}")

    def _validate_reused_cell_for_spg(cell_obj, spg_value: int, peak_positions: np.ndarray):
        try:
            cell_solver_kwargs = _get_cell_solver_kwargs(state)
            solver = CellSolver(
                int(spg_value),
                peak_positions,
                max_mismatch=cell_solver_kwargs["max_mismatch"],
                hkl_max=cell_solver_kwargs["hkl_max"],
                max_square=cell_solver_kwargs["max_square"],
                total_square=cell_solver_kwargs["total_square"],
                theta_tols=cell_solver_kwargs["theta_tols"],
                min_abc=cell_solver_kwargs["min_abc"],
                max_chi2=cell_solver_kwargs["max_chi2"],
                max_guess=cell_solver_kwargs["max_guess"],
                verbose=False,
            )
            solution, reason = solver.validate_cell(np.array(cell_obj.dims, dtype=float))
            if solution is None:
                return False, None, reason
            metrics = {
                "chi2": float(solution["chi2"][1]),
                "missing": int(len(solution["mismatch"])),
                "errors": [float(x) for x in solution["errors"]],
            }
            return True, metrics, None
        except Exception as exc:
            return False, None, f"precheck exception ({exc})"

    infer_spg = bool(state.get("infer_spg_from_pxrd", False))
    stop_on_first_accepted_inferred_spg = bool(state.get("stop_on_first_accepted_inferred_spg", True))
    peak_positions_np = np.array(state.get("peak_positions") or [], dtype=float)
    composition = state.get("composition", {})
    density_min = state.get("density_min", 0.0)
    density_max = state.get("density_max", 0.0)
    spg_prediction_rank = {
        int(pred_spg): idx
        for idx, (pred_spg, _prob) in enumerate(state.get("spg_predictions", []), start=1)
    }

    def _prediction_rank(spg_value: int) -> int:
        return int(spg_prediction_rank.get(int(spg_value), 10**9))

    def _chi2_bucket(value: float, tol: float = 5e-4) -> int:
        if tol <= 0:
            return int(np.round(float(value) * 1e6))
        return int(np.round(float(value) / tol))

    def _balanced_pair_priority(
        est_trials: int,
        volume: float,
        min_trials: int,
        min_volume: float,
        trial_weight: float = 0.65,
        volume_weight: float = 0.35,
    ) -> float:
        safe_trials = max(1.0, float(est_trials))
        safe_volume = max(1e-6, float(volume))
        ref_trials = max(1.0, float(min_trials))
        ref_volume = max(1e-6, float(min_volume))
        trial_ratio = safe_trials / ref_trials
        volume_ratio = safe_volume / ref_volume
        return float((trial_ratio ** trial_weight) * (volume_ratio ** volume_weight))

    def _canonical_cell_signature(cell_obj) -> tuple:
        dims = np.array(getattr(cell_obj, "dims", []), dtype=float)
        if len(dims) == 0:
            return (0, ())
        if len(dims) >= 3:
            abc = tuple(round(float(x), 3) for x in sorted(dims[:3].tolist()))
            tail = tuple(round(float(x), 2) for x in dims[3:].tolist())
            return (len(dims), abc + tail)
        return (len(dims), tuple(round(float(x), 3) for x in sorted(dims.tolist())))

    wp_candidate_cache: dict[tuple, list] = {}
    wp_cost_cache: dict[tuple, tuple[int, int]] = {}

    def _pair_key(cell_obj, spg_value: int) -> tuple:
        return (_canonical_cell_signature(cell_obj), int(spg_value))

    def _get_wp_candidates_for_pair(cell_obj, spg_value: int) -> list:
        key = _pair_key(cell_obj, spg_value)
        if key in wp_candidate_cache:
            return wp_candidate_cache[key]
        try:
            candidates = enumerate_wyckoff_multi_spg(
                cell_obj.dims,
                [int(spg_value)],
                composition,
                ref_den=(density_min, density_max),
            )
        except Exception:
            candidates = []
        wp_candidate_cache[key] = candidates
        return candidates

    def _estimate_pair_trial_cost(cell_obj, spg_value: int) -> tuple[int, int]:
        key = _pair_key(cell_obj, spg_value)
        if key in wp_cost_cache:
            return wp_cost_cache[key]

        candidates = _get_wp_candidates_for_pair(cell_obj, spg_value)
        candidate_count = len(candidates)
        top_candidates = candidates[:20]

        est_trials = 0
        for candidate in top_candidates:
            dof = int(candidate[5])
            n4 = dof * 3 if dof != 1 else 4
            est_trials += (n4 + 1)

        if candidate_count == 0:
            est_trials = 10**9

        out = (candidate_count, est_trials)
        wp_cost_cache[key] = out
        return out
    predicted_spgs = []
    for pred_spg, _prob in state.get("spg_predictions", [])[: int(state.get("spg_top_k", 5))]:
        spg_int = int(pred_spg)
        if spg_int not in predicted_spgs:
            predicted_spgs.append(spg_int)

    lattice_filter = str(state.get("lattice_symmetry", "auto") or "auto").strip().lower()
    if lattice_filter == "auto":
        filename_spg = state.get("spg_from_filename")
        target_system = spg_to_crystal_system(int(filename_spg)) if filename_spg is not None else None
    elif lattice_filter == "any":
        target_system = None
    elif lattice_filter in VALID_LATTICE_SYMMETRIES:
        target_system = lattice_filter
    else:
        _emit_progress(f"Unknown lattice symmetry filter '{lattice_filter}', using unfiltered SG candidates.")
        target_system = None

    if predicted_spgs and target_system is not None:
        filtered_spgs = [sg for sg in predicted_spgs if spg_to_crystal_system(sg) == target_system]
        if filtered_spgs:
            _emit_progress(
                f"Applying lattice symmetry filter: {target_system}. "
                f"Kept {len(filtered_spgs)}/{len(predicted_spgs)} inferred SG candidates."
            )
            predicted_spgs = filtered_spgs
        else:
            _emit_progress(
                f"Lattice symmetry filter '{target_system}' removed all inferred SG candidates; "
                f"falling back to unfiltered candidate list."
            )

    if infer_spg and predicted_spgs:
        inferred_sweep_start_time = time.perf_counter()
        best_trial_state = None
        best_trial_message = None
        best_trial_score = -1e9

        def _cell_signature(cell_obj) -> tuple:
            dims = tuple(round(float(x), 3) for x in np.array(cell_obj.dims).tolist())
            return (len(dims), dims)

        # key = (seed_spg, dims_sig) — same dims under different SPGs kept separately
        attempted_cell_keys: set = set()
        any_seed_had_cells = False
        global_structure_log: list = []

        # ── Phase 1: collect all (cell, spg) pairs from every seed SPG ──────────
        all_seed_cells: list = []  # (volume, cell, seed_spg)
        for seed_rank, seed_spg in enumerate(predicted_spgs, start=1):
            seed_state = copy.deepcopy(state)
            seed_state["spg"] = seed_spg
            _run_cell_solver_stage(seed_state)
            seed_cells = seed_state.get("cells") or []

            if not seed_cells:
                _emit_progress(f"Phase 1 — rank {seed_rank:2d}/{len(predicted_spgs)}: spg={seed_spg} | No candidate cells found.")
                continue

            # Show volume range of cells found for this SPG
            volumes = [float(getattr(cell, "size", 0.0)) for cell in seed_cells]
            vol_min, vol_max = min(volumes), max(volumes)
            vol_info = (
                f"vol={vol_min:.1f}–{vol_max:.1f} Å³"
                if vol_min != vol_max
                else f"vol={vol_min:.1f} Å³"
            )
            _emit_progress(
                f"Phase 1 — rank {seed_rank:2d}/{len(predicted_spgs)}: spg={seed_spg} | Found {len(seed_cells)} cell(s): {vol_info}"
            )

            any_seed_had_cells = True
            for cell in seed_cells:
                sig = _cell_signature(cell)
                key = (seed_spg, sig)
                if key in attempted_cell_keys:
                    continue
                attempted_cell_keys.add(key)
                all_seed_cells.append((float(getattr(cell, "size", 0.0)), cell, seed_spg))

        if all_seed_cells:
            # ── Phase 2: plan ALL (cell, spg) pairs with explicit cost estimates ─
            # 1. Group permutation-equivalent / near-identical cells into families.
            # 2. For each (cell, spg), estimate cost by Wyckoff candidate count and
            #    estimated number of generated trials.
            # 3. Globally rank every pair by a balanced score that combines
            #    relative estimated trials and relative cell volume.
            grouped_seed_cells: dict[tuple, list[tuple[float, object, int]]] = {}
            for item in all_seed_cells:
                _vol, _cell, _spg = item
                sig = _canonical_cell_signature(_cell)
                grouped_seed_cells.setdefault(sig, []).append(item)

            planned_groups = []
            skipped_pairs = []
            for sig, members in grouped_seed_cells.items():
                enriched_members = []
                for _vol, _cell, _spg in members:
                    cand_count, est_trials = _estimate_pair_trial_cost(_cell, _spg)
                    if cand_count == 0:
                        skipped_pairs.append((_vol, _spg))  # no valid Wyckoff assignments — skip
                        continue
                    enriched_members.append(
                        {
                            "vol": float(_vol),
                            "cell": _cell,
                            "spg": int(_spg),
                            "cand_count": int(cand_count),
                            "est_trials": int(est_trials),
                        }
                    )

                if not enriched_members:
                    continue

                enriched_members.sort(
                    key=lambda m: (
                        m["est_trials"],
                        round(m["vol"], 1),
                        m["cand_count"],
                        _prediction_rank(m["spg"]),
                        getattr(m["cell"], "missing", 999),
                        _chi2_bucket(getattr(m["cell"], "chi2", 1e9)),
                        getattr(m["cell"], "chi2", 1e9),
                        -CRYSTAL_SYSTEM_PRIORITY.get(spg_to_crystal_system(int(m["spg"])), 0),
                        -int(m["spg"]),
                    )
                )
                best_symmetry = max(
                    CRYSTAL_SYSTEM_PRIORITY.get(spg_to_crystal_system(int(m["spg"])), 0)
                    for m in enriched_members
                )
                best_missing = min(getattr(m["cell"], "missing", 999) for m in enriched_members)
                best_chi2 = min(float(getattr(m["cell"], "chi2", 1e9)) for m in enriched_members)
                best_pred_rank = min(_prediction_rank(m["spg"]) for m in enriched_members)
                min_group_volume = min(float(m["vol"]) for m in enriched_members)
                min_group_trials = min(int(m["est_trials"]) for m in enriched_members)
                min_group_candidates = min(int(m["cand_count"]) for m in enriched_members)
                planned_groups.append(
                    {
                        "signature": sig,
                        "members": enriched_members,
                        "best_symmetry": best_symmetry,
                        "best_missing": best_missing,
                        "best_chi2": best_chi2,
                        "best_pred_rank": best_pred_rank,
                        "min_volume": min_group_volume,
                        "min_trials": min_group_trials,
                        "min_candidates": min_group_candidates,
                    }
                )

            planned_groups.sort(
                key=lambda g: (
                    g["min_trials"],
                    round(g["min_volume"], 1),
                    g["min_candidates"],
                    -g["best_symmetry"],
                    g["best_pred_rank"],
                    g["best_missing"],
                    _chi2_bucket(g["best_chi2"]),
                )
            )

            planned_pairs = [
                member
                for group in planned_groups
                for member in group["members"]
            ]
            min_pair_trials = min(int(member["est_trials"]) for member in planned_pairs)
            min_pair_volume = min(float(member["vol"]) for member in planned_pairs)
            for member in planned_pairs:
                member["balance_score"] = _balanced_pair_priority(
                    member["est_trials"],
                    member["vol"],
                    min_pair_trials,
                    min_pair_volume,
                )
            planned_pairs.sort(
                key=lambda m: (
                    m["balance_score"],
                    m["est_trials"],
                    round(m["vol"], 1),
                    m["cand_count"],
                    -CRYSTAL_SYSTEM_PRIORITY.get(spg_to_crystal_system(int(m["spg"])), 0),
                    _prediction_rank(m["spg"]),
                    getattr(m["cell"], "missing", 999),
                    _chi2_bucket(getattr(m["cell"], "chi2", 1e9)),
                    getattr(m["cell"], "chi2", 1e9),
                    -int(m["spg"]),
                )
            )

            all_seed_cells = [
                (member["vol"], member["cell"], member["spg"])
                for member in planned_pairs
            ]

            volumes = [float(item[0]) for item in all_seed_cells]
            vol_lo = min(volumes)
            vol_hi = max(volumes)
            _emit_progress(
                f"Phase 2: planned {len(all_seed_cells)} (cell, SPG) pair(s) across "
                f"{len(planned_groups)} cell family/families. Volume range: {vol_lo:.1f}–{vol_hi:.1f} Å³"
            )
            _emit_progress(
                "Phase 2 strategy: globally rank every (cell, SPG) pair by a balanced "
                "score combining relative estimated trials and relative volume "
                "(trial_weight=0.65, volume_weight=0.35), then break ties by "
                "(fewer estimated trials, smaller volume, fewer candidates), "
                "then (symmetry, SG prediction rank, missing, chi2)."
            )

            # ── Phase 2 summary table ────────────────────────────────────────────
            _emit_progress(
                f"\n{'Rank':<5} {'SPG':<5} {'Volume(Å³)':<11} {'Chi2':<8} {'Missing':<8} {'EstTrials':<10} {'BalScore':<9} Dims"
            )
            _emit_progress("-" * 104)
            for _ri, _pair in enumerate(planned_pairs, start=1):
                _vol = _pair["vol"]
                _cell = _pair["cell"]
                _spg = _pair["spg"]
                _est_trials = _pair["est_trials"]
                _balance_score = float(_pair.get("balance_score", float("nan")))
                _dims_str = "  ".join(f"{float(x):8.3f}" for x in _cell.dims)
                _emit_progress(
                    f"{_ri:<5} {_spg:<5} {_vol:<11.1f} "
                    f"{getattr(_cell, 'chi2', float('nan')):<8.4f} "
                    f"{getattr(_cell, 'missing', -1):<8} {_est_trials:<10} {_balance_score:<9.3f} {_dims_str}"
                )
            _emit_progress("")

            # ─ Summary of skipped pairs ─
            if skipped_pairs:
                _emit_progress(
                    f"Skipped {len(skipped_pairs)} individual (cell, SPG) pair(s) due to zero valid Wyckoff "
                    f"position(s) in the given Z range."
                )

            spg_cell_phase_end_time = time.perf_counter()
            structure_phase_start_time = spg_cell_phase_end_time
            #all_structure_log: list = []

            # ── Phase 3: systematic structure generation across all ranked (cell, spg) pairs ──
            # Each entry is already a specific (cell, spg) pairing — enumerate Wyckoff
            # only for that SPG to avoid redundant work across identical cell dims.
            for rank_idx, (vol, cell, seed_spg) in enumerate(all_seed_cells, start=1):
                consolidated_wp = _get_wp_candidates_for_pair(cell, seed_spg)
                if not consolidated_wp: continue

                top_preview = [f"spg={s[0]} count={s[6]} dof={s[5]}" for s in consolidated_wp[:3]]
                logger.info(
                    f"\n[Pair {rank_idx}/{len(all_seed_cells)}] vol={vol:.1f} Å³, spg={seed_spg}, dims={[round(float(x), 3) for x in cell.dims]}: {len(consolidated_wp)} WP candidates. Top: {' | '.join(top_preview)}"
                )

                wp_limits = get_adaptive_wp_limits(len(consolidated_wp), 20)
                prev_limit = 0
                wp_attempted = 0
                cell_accepted = False

                for limit in wp_limits:
                    if wp_attempted >= len(consolidated_wp):
                        break

                    for sol in consolidated_wp[prev_limit:limit]:
                        spg_val, _comp, _lat, wp_ids, num_wps, dof, count, Z, orig_spg = sol
                        wp_attempted += 1

                        passed, _metrics, reject_reason = _validate_reused_cell_for_spg(
                            cell, spg_val, peak_positions_np
                        )
                        if not passed:
                            _emit_progress(
                                f"\nPair {rank_idx} rejected for spg={spg_val}: {reject_reason}"
                            )
                            continue

                        wp_labels_text = _format_wyckoff_labels_from_ids(spg_val, wp_ids)
                        logger.info(
                            f"WP #{wp_attempted}: spg={spg_val}, count={count}, dof={dof}, "
                            f"n_wps={num_wps}, wyckoff={wp_labels_text}"
                        )

                        trial_state = copy.deepcopy(state)
                        trial_state["spg"] = spg_val
                        trial_state["cells"] = copy.deepcopy([cell])
                        trial_state["suppress_local_energy_plot"] = True
                        forced_wp_solution = sol[:8] if len(sol) >= 9 else sol
                        trial_state["forced_wp_solution"] = forced_wp_solution

                        trial_message = _run_wyckoff_solver(trial_state, global_structure_log)
                        # After running, update the main state's Struc_count by accumulating
                        state["Struc_count"] = trial_state.get("Struc_count")
                        trial_result = trial_state.get("wyckoff_result") or {}

                        trial_score = trial_result.get("score")
                        if trial_score is not None and trial_score > best_trial_score:
                            best_trial_score = trial_score
                            best_trial_state = trial_state
                            best_trial_message = trial_message

                        if trial_result.get("accepted"):
                            _emit_accepted_solution_details(spg_val, trial_result)
                            cell_accepted = True
                            # For inferred-SPG early exit, require stricter criteria: R² > 0.93 AND χ² < 0.18
                            # instead of soft acceptance (R² ≥ 0.85 and χ² ≤ 0.24)
                            r2_val = trial_result.get("r2")
                            chi2_val = trial_result.get("chi2")
                            strict_early_exit = (
                                r2_val is not None and chi2_val is not None and
                                r2_val >= 0.93 and chi2_val < 0.18
                            )
                            candidate_energy = trial_result.get("selected_energy")
                            global_energy_values = [
                                float(entry.get("eng"))
                                for entry in global_structure_log
                                if entry.get("eng") is not None
                            ]
                            global_best_energy = min(global_energy_values) if global_energy_values else None
                            global_eng_rel = None
                            if candidate_energy is not None and global_best_energy is not None:
                                global_eng_rel = max(0.0, float(candidate_energy) - float(global_best_energy))
                            max_eng_rel_early_stop = max(
                                0.0,
                                float(state.get(
                                    "max_eng_rel_early_stop",
                                    state.get("max_eng_rel") if state.get("max_eng_rel") is not None else 0.20,
                                )),
                            )
                            energy_ok_for_global_early_exit = (
                                global_eng_rel is not None and global_eng_rel <= max_eng_rel_early_stop
                            )
                            enough_global_structures = len(global_structure_log) >= max(0, int(state.get("min_structures_before_early_stop", 10)))
                            if stop_on_first_accepted_inferred_spg and strict_early_exit and enough_global_structures and energy_ok_for_global_early_exit:
                                logger.info(
                                    f"Good solution found early: spg={spg_val}, "
                                    f"R2={trial_result.get('r2', 0):.4f}, "
                                    f"Chi2={trial_result.get('chi2', 0):.4f}, "
                                    f"dE_global={global_eng_rel:.4f}. "
                                    f"Stopping search after pair {rank_idx}/{len(all_seed_cells)} "
                                    f"and {wp_attempted} WP candidate(s)."
                                )

                                timing_breakdown = _current_timing_breakdown_seconds()
                                state["timing_breakdown_seconds"] = timing_breakdown
                                formula_str = state.get("formula", "unknown")
                                plot_energy_vs_r2(
                                    global_structure_log,
                                    formula_str,
                                    "all",
                                    f"{state.get('results_dir', 'Results')}/EnergyR2_{formula_str}.png",
                                    status="Success",
                                    elapsed_seconds=time.perf_counter() - inferred_sweep_start_time,
                                    timing_breakdown_seconds=timing_breakdown,
                                )
                                state.update(trial_state)
                                _emit_timing_breakdown()

                                return {
                                    "status": status_label,
                                    "message": trial_message,
                                    "spg": state.get("spg"),
                                    "formula": state.get("formula"),
                                }

                            if stop_on_first_accepted_inferred_spg and strict_early_exit and enough_global_structures and not energy_ok_for_global_early_exit:
                                _emit_progress(
                                    f"Good refined fit found for spg={spg_val}, but skipping early stop "
                                    f"because dE_global={global_eng_rel:.4f} exceeds "
                                    f"{max_eng_rel_early_stop:.4f} eV/atom."
                                )

                            n_structures = len(global_structure_log)
                            _emit_progress(
                                f"Accepted solution found for spg={spg_val}; continuing "
                                f"(early-stop criteria not met or disabled, {n_structures} structures explored so far)."
                            )

                    prev_limit = limit

                if cell_accepted:
                    _emit_progress(f"Pair {rank_idx}: accepted solution found; moving to next pair.")

            # End of all-pairs loop: emit global plot covering every structure tried
            if global_structure_log:
                timing_breakdown = _current_timing_breakdown_seconds()
                state["timing_breakdown_seconds"] = timing_breakdown
                formula_str = state.get("formula", "unknown")
                global_plot_status = "Success" if (best_trial_state and (best_trial_state.get("wyckoff_result") or {}).get("accepted", False)) else "Failure"
                plot_energy_vs_r2(
                    global_structure_log,
                    formula_str,
                    "all",
                    f"{state.get('results_dir', 'Results')}/EnergyR2_{formula_str}.png",
                    status=global_plot_status,
                    elapsed_seconds=time.perf_counter() - inferred_sweep_start_time,
                    timing_breakdown_seconds=timing_breakdown,
                )

        if not any_seed_had_cells:
            _emit_progress("No inferred SG candidate produced valid cells; falling back to default single-SPG flow.")
        elif best_trial_state is not None:
            best_trial_result = best_trial_state.get("wyckoff_result") or {}
            if best_trial_result.get("accepted"):
                _emit_accepted_solution_details(
                    int(best_trial_state.get("spg", 0)),
                    best_trial_result,
                    prefix="Best accepted inferred-SG solution",
                )
                _emit_progress(f"Return best result in spg={best_trial_state.get('spg')}.")
            else:
                _emit_progress(f"No acceptance; return best result in spg={best_trial_state.get('spg')}.")
            _emit_progress(f"Best inferred-SG score observed: {best_trial_score:.4f}")
            state.update(best_trial_state)
            accepted_inferred = (best_trial_state.get("wyckoff_result") or {}).get("accepted", False)

            if spg_cell_phase_end_time is None:
                spg_cell_phase_end_time = time.perf_counter()
            if accepted_inferred and structure_phase_start_time is None:
                structure_phase_start_time = spg_cell_phase_end_time
            _emit_timing_breakdown()
            return {
                "status": status_label if accepted_inferred else "no_solution",
                "message": best_trial_message,
                "spg": state.get("spg"),
                "formula": state.get("formula"),
            }

    cell_result = _run_cell_solver_stage(state)
    if not state.get("cells"):
        _emit_progress("No valid unit cells found. Pipeline did not find a solution.")
        spg_cell_phase_end_time = time.perf_counter()
        _emit_timing_breakdown()
        return {
            "status": "no_cells",
            "message": cell_result.get("message", ""),
            "spg": state.get("spg"),
            "formula": state.get("formula"),
        }
    spg_cell_phase_end_time = time.perf_counter()
    structure_phase_start_time = spg_cell_phase_end_time
    wyckoff_message = _run_wyckoff_solver_stage(state, global_structure_log)
    wyckoff_result = state.get("wyckoff_result") or {}
    accepted = wyckoff_result.get("accepted", False)
    final_status = status_label if accepted else "no_solution"
    _emit_timing_breakdown()
    return {
        "status": final_status,
        "message": wyckoff_message,
        "spg": state.get("spg"),
        "formula": state.get("formula"),
    }



def _extract_outcome(label: str, state: dict, result: dict | None) -> dict:
    wyckoff_result = state.get("wyckoff_result") or {}
    structure_log = state.get("structure_log") or []
    refined_entries = [entry for entry in structure_log if entry.get("refined")]
    best_refined_r2 = max((_safe_float(entry.get("r2"), -1.0) for entry in refined_entries), default=None)
    best_refined_chi2 = min((_safe_float(entry.get("chi2"), 1e9) for entry in refined_entries), default=None)
    min_energy = min((_safe_float(entry.get("eng"), 1e9) for entry in structure_log), default=None)
    selected_energy = _safe_float(wyckoff_result.get("selected_energy"))
    eng_rel = _safe_float(wyckoff_result.get("eng_rel"))
    if eng_rel is None and selected_energy is not None and min_energy is not None:
        eng_rel = max(0.0, selected_energy - min_energy)

    return {
        "label": label,
        "status": (result or {}).get("status", "unknown"),
        "message": (result or {}).get("message", ""),
        "spg": state.get("spg"),
        "formula": state.get("formula"),
        "accepted": bool(wyckoff_result.get("accepted", False)),
        "wr": _safe_float(wyckoff_result.get("wr")),
        "r2": _safe_float(wyckoff_result.get("r2")),
        "chi2": _safe_float(wyckoff_result.get("chi2")),
        "score": _safe_float(wyckoff_result.get("score")),
        "selected_energy": selected_energy,
        "eng_rel": eng_rel,
        "attempt": _safe_int(wyckoff_result.get("attempt")),
        "seed": _safe_int(wyckoff_result.get("seed")),
        "cell_count": len(state.get("cells") or []),
        "structure_count": len(structure_log),
        "refined_count": len(refined_entries),
        "best_refined_r2": best_refined_r2,
        "best_refined_chi2": best_refined_chi2,
        "min_energy": min_energy,
        "log_path": state.get("system_run_log"),
    }

def _outcome_rank_key(outcome: dict) -> tuple:
    accepted = 1 if outcome.get("accepted") else 0
    score = _safe_float(outcome.get("score"), -1e9)
    r2 = _safe_float(outcome.get("r2"), -1.0)
    chi2 = _safe_float(outcome.get("chi2"), 1e9)
    eng_rel = _safe_float(outcome.get("eng_rel"), 1e9)
    structure_count = _safe_int(outcome.get("structure_count"), 0)
    cell_count = _safe_int(outcome.get("cell_count"), 0)
    return (accepted, score, r2, -chi2, -eng_rel, structure_count, cell_count)


def _is_better_outcome(candidate: dict, incumbent: dict) -> bool:
    return _outcome_rank_key(candidate) > _outcome_rank_key(incumbent)

def _is_good_sampling_outcome(outcome: dict, args: argparse.Namespace) -> bool:
    if not outcome.get("accepted"):
        return False
    eng_rel = _safe_float(outcome.get("eng_rel"), None)
    if eng_rel is None:
        return True
    return eng_rel <= float(args.success_max_eng_rel)


def _artifact_paths(formula: str | None, spg: int | None) -> list[Path]:
    if not formula or not spg:
        return []
    return [
        Path("Results") / f"Match_{formula}_{spg}.cif",
        Path("Results") / f"EnergyR2_{formula}_{spg}.png",
    ]


def _restore_artifacts(snapshot: dict[str, str]) -> None:
    for target, source in snapshot.items():
        source_path = Path(source)
        if not source_path.exists():
            continue
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _outcome_rank_key(outcome: dict) -> tuple:
    accepted = 1 if outcome.get("accepted") else 0
    score = _safe_float(outcome.get("score"), -1e9)
    r2 = _safe_float(outcome.get("r2"), -1.0)
    chi2 = _safe_float(outcome.get("chi2"), 1e9)
    eng_rel = _safe_float(outcome.get("eng_rel"), 1e9)
    structure_count = _safe_int(outcome.get("structure_count"), 0)
    cell_count = _safe_int(outcome.get("cell_count"), 0)
    return (accepted, score, r2, -chi2, -eng_rel, structure_count, cell_count)

def _normalize_wp_candidate(candidate):
    return candidate[:8] if len(candidate) >= 9 else candidate

def _normalize_signature(value):
    if isinstance(value, list):
        return tuple(_normalize_signature(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_signature(item) for item in value)
    return value

def _prepare_trial_state(base_state: dict, *, spg: int | None = None, cells=None, overrides: dict | None = None) -> dict:
    trial_state = copy.deepcopy(base_state)
    if spg is not None:
        trial_state["spg"] = int(spg)
    if cells is not None:
        trial_state["cells"] = copy.deepcopy(cells)
    if overrides:
        trial_state.update(overrides)
    return trial_state


def _cell_signature(cell) -> tuple:
    dims = getattr(cell, "dims", cell)
    return tuple(round(float(item), 6) for item in dims)


def _candidate_wp_signature(candidate) -> tuple:
    normalized = _normalize_wp_candidate(candidate)
    return (
        int(normalized[0]),
        tuple(tuple(str(wp) for wp in group) for group in normalized[3]),
    )


def _observed_setting_stats(structure_log: list[dict]) -> tuple[dict, dict]:
    setting_stats: dict = {}
    wp_stats: dict = {}

    def _update(stats: dict, key, entry: dict) -> None:
        current = stats.setdefault(
            key,
            {
                "seen": 0,
                "refined": 0,
                "best_r2": -1.0,
                "best_sim": -1.0,
                "best_eng": float("inf"),
                "best_chi2": float("inf"),
            },
        )
        current["seen"] += 1
        sim = _safe_float(entry.get("sim"), -1.0)
        eng = _safe_float(entry.get("eng"), float("inf"))
        r2 = _safe_float(entry.get("r2"), -1.0)
        chi2 = _safe_float(entry.get("chi2"), float("inf"))
        current["best_sim"] = max(current["best_sim"], sim)
        current["best_eng"] = min(current["best_eng"], eng)
        current["best_r2"] = max(current["best_r2"], r2)
        current["best_chi2"] = min(current["best_chi2"], chi2)
        if entry.get("refined"):
            current["refined"] += 1

    for entry in structure_log or []:
        setting_key = _normalize_signature(entry.get("setting_signature"))
        wp_key = _normalize_signature(entry.get("wp_signature"))
        if setting_key is not None:
            _update(setting_stats, setting_key, entry)
        if wp_key is not None:
            _update(wp_stats, wp_key, entry)
    return setting_stats, wp_stats


def _artifact_paths(formula: str | None, spg: int | None) -> list[Path]:
    if not formula or not spg:
        return []
    return [
        Path("Results") / f"Match_{formula}_{spg}.cif",
        Path("Results") / f"EnergyR2_{formula}_{spg}.png",
    ]


def _snapshot_artifacts(formula: str | None, spg: int | None, label: str) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not formula or not spg:
        return snapshot
    snapshot_dir = Path("tmp") / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for artifact in _artifact_paths(formula, spg):
        if not artifact.exists():
            continue
        snapshot_path = snapshot_dir / f"{label}_{artifact.name}"
        shutil.copy2(artifact, snapshot_path)
        snapshot[str(artifact)] = str(snapshot_path)
    return snapshot



def _observed_rank_key(setting_stats: dict | None, wp_stats: dict | None) -> tuple:
    setting_stats = setting_stats or {}
    wp_stats = wp_stats or {}
    setting_best_eng = setting_stats.get("best_eng", float("inf"))
    wp_best_eng = wp_stats.get("best_eng", float("inf"))
    setting_best_chi2 = setting_stats.get("best_chi2", float("inf"))
    wp_best_chi2 = wp_stats.get("best_chi2", float("inf"))
    return (
        1 if setting_stats else 0,
        1 if setting_stats.get("refined", 0) > 0 else 0,
        setting_stats.get("best_r2", -1.0),
        -setting_best_chi2 if setting_best_chi2 != float("inf") else -1e9,
        -setting_best_eng if setting_best_eng != float("inf") else -1e9,
        setting_stats.get("best_sim", -1.0),
        1 if wp_stats else 0,
        1 if wp_stats.get("refined", 0) > 0 else 0,
        wp_stats.get("best_r2", -1.0),
        -wp_best_chi2 if wp_best_chi2 != float("inf") else -1e9,
        -wp_best_eng if wp_best_eng != float("inf") else -1e9,
        wp_stats.get("best_sim", -1.0),
    )


def _get_system_run_log_path(state: dict) -> str:
    pxrd_csv = str(state.get("pxrd_csv") or "")
    stem = Path(pxrd_csv).stem if pxrd_csv else "unknown_system"
    log_name = f"RunLog_{_safe_name_token(stem)}.log"
    results_dir = state.get("results_dir", "Results")
    return str(Path(results_dir) / log_name)


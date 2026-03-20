from strands import Agent, tool, ToolContext
from strands.models.gemini import GeminiModel
from strands.multiagent.graph import GraphBuilder
import argparse
import logging
import os
import inspect
import random
import shutil
import sys
import time
import traceback
import copy
from pathlib import Path
from importlib.metadata import version as pkg_version, PackageNotFoundError
import pandas as pd
import numpy as np
from tools.manager import RawDataManager, CellManager, WPManager
from tools.peak_prediction import predict_peaks, predict_spacegroup
from tools.XRD import Profile
from tools.solver import (
    CellSolver,
    SmartCellSolver,
    search_solution,
    enumerate_wyckoff_multi_spg,
    score_wp_candidate,
    get_adaptive_wp_limits,
)
from tools.utils import parse_formula, get_volume_from_density
from tools.density import predict_density_ensemble

# Configure logging with both file and console handlers
file_handler = logging.FileHandler('single_agent.log')
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logging.root.addHandler(file_handler)
logging.root.addHandler(console_handler)
logging.root.setLevel(logging.INFO)

logger = logging.getLogger("strands.multiagent")
logger.setLevel(logging.ERROR)


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
sys.stdout = StreamToLogger(logging.getLogger("stdout"), logging.INFO)
#sys.stderr = StreamToLogger(logging.getLogger("stderr"), logging.WARNING)


default_state = {
    # Raw inputs
    "pxrd_csv": "Examples/PXRD_PrYMg2_123.csv",
    "formula": "",
    "composition": {},
    # To be filled by Data Agents
    "x1": [],
    "y1": [],
    "peaks": [],
    "peak_positions": [],
    # To be filled by Solver Agents
    "spg": 0,
    "density_min": 0.0,
    "density_max": 0.0,
    "min_volume": 0.0,
    "min_abc": 2.0,
    "cells": [],
    # Constraints and parameters
    "wavelength": 1.54184,
    "min_r2": 0.95,
    "max_chi2": 0.12,
    "INST_FILE": "tools/INST_XRY.PRM",
    "SCALED_INTENSITY_TOL": 0.01,
    "thetas": [10, 80],
    "resolution": 0.02,
    "max_force": 0.5,
    "max_stress": 0.3,
    "max_cells": 10,
    "max_cell_volume": None,
    "multi_attempts": _env_int("PXRD_MULTI_ATTEMPTS", 1, min_value=1),
    "seed_base": _env_int("PXRD_SEED_BASE", 20260315),
    "spg_top_k": 25,
    "spg_infer_backend": "model",
    "stop_on_first_accepted_inferred_spg": True,
    "show_spg_predictions": True,
    "max_local_boosts": _env_int("PXRD_LOCAL_BOOSTS", 1, min_value=0),
    "max_local_perturbations": _env_int("PXRD_LOCAL_PERTURBS", 2, min_value=0),
    "perturb_displacement": float(os.getenv("PXRD_PERTURB_DISPLACEMENT", "0.06")),
    "max_eng_rel_early_stop": 0.20,
    "max_eng_rel": None,
    "min_structures_before_early_stop": 10,
}


gemini_model = GeminiModel(
    client_args={'api_key': 'AIzaSyA2TT4RqCvrY-RwRNmhT8AnCLwH-IwvdE8'},
    model_id='gemini-2.5-pro',
    params={"temperature": 0.7}
)

INPUT_PROMPT = "Process the PXRD data from Examples/PXRD_PrYMg2_123.csv"

VALID_LATTICE_SYMMETRIES = {
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
}

SPG_INFER_BACKENDS = {"model", "smart-cell"}

CRYSTAL_SYSTEM_PRIORITY = {
    "cubic": 7,
    "hexagonal": 6,
    "trigonal": 5,
    "tetragonal": 4,
    "orthorhombic": 3,
    "monoclinic": 2,
    "triclinic": 1,
}


def _infer_formula_spg(path: str) -> tuple[str | None, int | None]:
    tokens = Path(path).stem.split('_')
    formula_guess, spg_guess = None, None
    if len(tokens) >= 2:
        try:
            spg_guess = int(tokens[-1])
            formula_guess = '_'.join(tokens[1:-1]) if len(tokens) > 2 else None
        except ValueError:
            pass
    return formula_guess, spg_guess


def _spg_to_crystal_system(spg: int) -> str | None:
    if 1 <= spg <= 2:
        return "triclinic"
    if 3 <= spg <= 15:
        return "monoclinic"
    if 16 <= spg <= 74:
        return "orthorhombic"
    if 75 <= spg <= 142:
        return "tetragonal"
    if 143 <= spg <= 167:
        return "trigonal"
    if 168 <= spg <= 194:
        return "hexagonal"
    if 195 <= spg <= 230:
        return "cubic"
    return None


def _rank_spg_candidates_from_smart_solutions(solutions: list[dict], top_k: int = 5) -> list[tuple[int, float]]:
    """Rank SG candidates from SmartCellSolver output.

    Priority is crystal-system symmetry (high -> low), then per-SPG evidence:
    lower missing peaks, lower chi2, higher support count.
    """
    if not solutions:
        return []

    stats: dict[int, dict] = {}
    for sol in solutions:
        spg = int(sol.get("spg", 0) or 0)
        if spg <= 0:
            continue
        mismatch = len(sol.get("mismatch", []) or [])
        chi2_raw = sol.get("chi2", (1e9, 1e9))
        if isinstance(chi2_raw, (list, tuple)) and len(chi2_raw) >= 2:
            chi2_val = float(chi2_raw[1])
        else:
            chi2_val = float(chi2_raw if chi2_raw is not None else 1e9)

        rec = stats.setdefault(
            spg,
            {
                "support": 0,
                "best_missing": 10**9,
                "best_chi2": 1e9,
            },
        )
        rec["support"] += 1
        rec["best_missing"] = min(rec["best_missing"], mismatch)
        rec["best_chi2"] = min(rec["best_chi2"], chi2_val)

    ordered_spgs = sorted(
        stats.keys(),
        key=lambda sg: (
            -CRYSTAL_SYSTEM_PRIORITY.get(_spg_to_crystal_system(sg), 0),
            stats[sg]["best_missing"],
            stats[sg]["best_chi2"],
            -stats[sg]["support"],
            -sg,
        ),
    )

    ordered_spgs = ordered_spgs[: max(1, int(top_k))]
    denom = float(sum(1.0 / (idx + 1) for idx in range(len(ordered_spgs))))
    ranked = []
    for idx, spg in enumerate(ordered_spgs):
        weight = (1.0 / (idx + 1)) / denom if denom > 0 else 0.0
        ranked.append((int(spg), float(weight)))
    return ranked


def _run_data_preprocessor_stage(pxrd_csv: str, state: dict) -> dict:
    formula_from_filename, spg_from_filename = _infer_formula_spg(pxrd_csv)
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
        try:
            predictions = []
            if spg_infer_backend == "smart-cell":
                smart_solutions = SmartCellSolver(
                    np.array(peak_positions, dtype=float),
                    hkl_max=(2, 5, 6),
                    max_mismatch=12,
                    max_chi2=0.2,
                    max_square=28,
                    total_square=40,
                    theta_tols=[0.1, 0.15, 0.5],
                    min_abc=min_abc,
                    max_abc=35.0,
                    min_volume=20.0,
                    max_volume=max_cell_volume,
                    verbose=False,
                )
                predictions = _rank_spg_candidates_from_smart_solutions(
                    smart_solutions,
                    top_k=spg_top_k,
                )
                raw_by_spg: dict[int, list[tuple]] = {}
                for sol in smart_solutions:
                    spg_i = int(sol.get("spg", 0) or 0)
                    if spg_i <= 0:
                        continue
                    mismatch = sol.get("mismatch", []) or []
                    chi2_raw = sol.get("chi2", (1e9, 1e9))
                    chi2_val = float(chi2_raw[1]) if isinstance(chi2_raw, (list, tuple)) and len(chi2_raw) >= 2 else float(chi2_raw)
                    raw_tuple = (
                        spg_i,
                        sol.get("cell"),
                        mismatch,
                        chi2_val,
                        sol.get("errors", []),
                        sol.get("id", ""),
                        sol.get("match", []),
                    )
                    raw_by_spg.setdefault(spg_i, []).append(raw_tuple)
                if raw_by_spg:
                    state["smart_cell_raw_solutions_by_spg"] = raw_by_spg
                state["spg_prediction_source"] = "smart_cell_solver"

            if not predictions:
                y1_norm = (y1 - np.min(y1)) / (np.max(y1) - np.min(y1) + 1e-8)
                peak_results = predict_peaks(y1_norm, threshold=0.8)
                peak_idx = [pos for pos, _ in peak_results]
                peak_intensities = [y1_norm[pos] * 100 for pos in peak_idx]

                if peak_idx:
                    _, py = Profile("gaussian").get_profile(x1[peak_idx], peak_intensities, 10, 80)
                    predictions = predict_spacegroup(py, formula, top_k=spg_top_k, use_normalization=False)
                    state["spg_prediction_source"] = "reconstructed_profile"
                else:
                    predictions = predict_spacegroup(y1_norm, formula, top_k=spg_top_k, use_normalization=True)
                    state["spg_prediction_source"] = "raw_intensity_fallback"

            if predictions:
                spg = int(predictions[0][0])
                state["spg_predictions"] = predictions
                top_lines = [
                    f"{idx + 1}. spg={int(pred_spg)} prob={float(prob):.2%}"
                    for idx, (pred_spg, prob) in enumerate(predictions[:spg_top_k])
                ]
                top_text = "\n".join(top_lines)
                source = state.get("spg_prediction_source", "unknown")
                if show_spg_predictions:
                    logger.info(f"Top-{spg_top_k} inferred space groups from PXRD ({source}):\n{top_text}")
                    print(f"Top-{spg_top_k} inferred space groups from PXRD ({source}):\n{top_text}")
                logger.info(f"Selected inferred space group: spg={spg}")
        except Exception as exc:
            logger.warning(f"Space-group inference failed; using filename/default space group. Reason: {exc}")

    if spg <= 0:
        raise ValueError(
            "Cannot infer space group from file name. Use --infer-spg or rename file as PXRD_<formula>_<spg>.csv."
        )

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


def _run_cell_solver_stage(state: dict) -> dict:
    spg = state.get("spg")
    formula = state.get("formula")
    peak_positions = state.get("peak_positions")
    max_cells = state.get("max_cells")
    max_cell_volume = state.get("max_cell_volume")

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
        max_mismatch=12,
        hkl_max=(2, 5, 6),
        max_square=28,
        total_square=40,
        theta_tols=[0.1, 0.15, 0.5],
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


def _plot_energy_vs_r2(
    structure_log: list,
    formula: str,
    spg: int,
    output_png: str,
    status: str = "Failure",
    elapsed_seconds: float | None = None,
    timing_breakdown_seconds: dict | None = None,
) -> None:
    """Scatter plot of energy-per-atom vs R² for every relaxed structure explored.
    Structures that were never refined receive R²=0.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        engs = [e["eng"] for e in structure_log]
        r2s  = [e["r2"]  for e in structure_log]
        mask = [e.get("refined", False) for e in structure_log]

        unref = [(e, r) for e, r, m in zip(engs, r2s, mask) if not m]
        ref   = [(e, r) for e, r, m in zip(engs, r2s, mask) if m]

        fig, ax = plt.subplots(figsize=(8, 5))
        if unref:
            ue, ur = zip(*unref)
            ax.scatter(ue, ur, c="steelblue", s=25, alpha=0.5,
                       label=f"Relaxed only (N={len(ue)})")
        if ref:
            re, rr = zip(*ref)
            ax.scatter(re, rr, c="crimson", marker="*", s=140, alpha=0.7,
                       label=f"Refined (N={len(re)})")

        ax.set_xlabel("Energy per atom (eV)")
        ax.set_ylabel("R² score  (0 = not refined)")
        ax.set_ylim(-0.2, 1.1)
        if timing_breakdown_seconds and "total" in timing_breakdown_seconds:
            total_seconds = max(0.0, float(timing_breakdown_seconds.get("total", 0.0)))
        elif elapsed_seconds is not None:
            total_seconds = max(0.0, float(elapsed_seconds))
            total_minutes = int(total_seconds // 60)
            seconds_remain = total_seconds - (60 * total_minutes)
            if total_minutes >= 60:
                hours = total_minutes // 60
                minutes = total_minutes % 60
                time_text = f"{hours}h {minutes}m {seconds_remain:04.1f}s"
            else:
                time_text = f"{total_minutes}m {seconds_remain:04.1f}s"
        else:
            time_text = "n/a"
        if timing_breakdown_seconds and "total" in timing_breakdown_seconds:
            total_minutes = int(total_seconds // 60)
            seconds_remain = total_seconds - (60 * total_minutes)
            if total_minutes >= 60:
                hours = total_minutes // 60
                minutes = total_minutes % 60
                time_text = f"{hours}h {minutes}m {seconds_remain:04.1f}s"
            else:
                time_text = f"{total_minutes}m {seconds_remain:04.1f}s"
        breakdown_text = None
        if timing_breakdown_seconds:
            spg_cell_seconds = max(0.0, float(timing_breakdown_seconds.get("spg_and_cell", 0.0)))
            structure_seconds = max(0.0, float(timing_breakdown_seconds.get("structure_inference", 0.0)))

            def _fmt_breakdown(seconds: float) -> str:
                total_minutes = int(seconds // 60)
                seconds_remain = seconds - (60 * total_minutes)
                if total_minutes >= 60:
                    hours = total_minutes // 60
                    minutes = total_minutes % 60
                    return f"{hours}h {minutes}m {seconds_remain:04.1f}s"
                return f"{total_minutes}m {seconds_remain:04.1f}s"

            breakdown_text = (
                f"SPG+Cell: {_fmt_breakdown(spg_cell_seconds)} | "
                f"Structure: {_fmt_breakdown(structure_seconds)}"
            )
        ax.set_title(
            f"{formula}  SPG {spg} — Energy vs R²  ({len(structure_log)} structures)  "
            f"[{status}]  [Time: {time_text}]"
            + (f"\n[{breakdown_text}]" if breakdown_text else "")
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_png, dpi=150)
        plt.close(fig)
        logger.info(f"Energy–R² plot saved to {output_png}")
    except Exception as exc:
        logger.warning(f"Failed to generate Energy–R² plot: {exc}")


def _run_wyckoff_solver_stage(state: dict) -> str:
    stage_start_time = time.perf_counter()
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
    if "forced_wp_solution" in state:
        state.pop("forced_wp_solution", None)
    min_r2 = state.get("min_r2")
    max_chi2 = state.get("max_chi2")
    max_force = state.get("max_force")
    max_stress = state.get("max_stress")
    max_local_boosts = max(0, int(state.get("max_local_boosts", 1)))
    max_local_perturbations = max(0, int(state.get("max_local_perturbations", 2)))
    perturb_displacement = max(0.0, float(state.get("perturb_displacement", 0.06)))
    max_eng_rel_early_stop = state.get("max_eng_rel_early_stop", state.get("max_eng_rel", None))
    min_structures_before_early_stop = max(0, int(state.get("min_structures_before_early_stop", 10)))
    suppress_local_energy_plot = bool(state.get("suppress_local_energy_plot", False))

    eng_min, sim_max = 1e10, 0.90

    os.makedirs("Results", exist_ok=True)
    os.makedirs("tmp", exist_ok=True)

    title = f'{formula} PXRD Prediction: Space Group {spg}'
    match_cif = f'Results/Match_{formula}_{spg}.cif'
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
        hard_accept = bool(res["r2"] > min_r2 or res["chi2"] < max_chi2)
        soft_accept = bool(
            res["r2"] >= max(min_r2 - 0.10, 0.85)
            and res["chi2"] <= max_chi2 * 2.0
        )
        return hard_accept or soft_accept

    best_result = None
    best_score = -1e9
    all_structure_log: list = []

    logger.info(f"Adaptive Wyckoff solve: {attempts} attempt(s), seed_base={seed_base}")
    for attempt_idx in range(attempts):
        seed = seed_base + 9973 * attempt_idx
        N1, N2, N3 = _attempt_schedule(attempt_idx)
        _set_seed(seed)

        attempt_png = f"tmp/Match_{formula}_{spg}_attempt{attempt_idx + 1}.png"
        attempt_cif = f"Results/Match_{formula}_{spg}_attempt{attempt_idx + 1}.cif"
        attempt_refinement_png = attempt_cif.replace(".cif", "_refinement.png")
        logger.info(
            f"Attempt {attempt_idx + 1}/{attempts}: seed={seed}, schedule=(N1={N1}, N2={N2}, N3={N3}), "
            f"local_boosts={max_local_boosts}, local_perturbations={max_local_perturbations}, "
            f"perturb_displacement={perturb_displacement:.3f}"
        )

        wr, r2, chi2, xtal, eng_best = search_solution(
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

        if wr is None:
            logger.info(f"Attempt {attempt_idx + 1}: no refined solution found.")
            continue

        candidate = {
            "wr": float(wr),
            "r2": float(r2),
            "chi2": float(chi2),
            "xtal": xtal,
            "eng_best": float(eng_best),
            "attempt": attempt_idx + 1,
            "seed": seed,
            "png": attempt_refinement_png,
            "cif": attempt_cif,
            "accepted": False,
        }
        candidate["accepted"] = _meets_acceptance(candidate)
        score = _score_result(candidate)
        logger.info(
            f"Attempt {attempt_idx + 1} metrics: Wr={wr:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}, "
            f"score={score:.4f}, accepted={candidate['accepted']}"
        )

        if score > best_score:
            best_score = score
            best_result = candidate

        if candidate["accepted"] and len(all_structure_log) >= min_structures_before_early_stop and (
            candidate["r2"] >= max(min_r2 + 0.02, 0.97)
            or candidate["chi2"] <= min(max_chi2 * 0.7, 0.08)
        ):
            logger.info(f"Early stop: excellent solution found at attempt {attempt_idx + 1}.")
            break

    local_plot_status = "Success" if (best_result is not None and best_result.get("accepted", False)) else "Failure"
    if all_structure_log and not suppress_local_energy_plot:
        elapsed_stage = time.perf_counter() - stage_start_time
        _plot_energy_vs_r2(
            all_structure_log, formula, spg,
            f"Results/EnergyR2_{formula}_{spg}.png",
            status=local_plot_status,
            elapsed_seconds=elapsed_stage,
            timing_breakdown_seconds=state.get("timing_breakdown_seconds"),
        )

    state["structure_log"] = all_structure_log

    if best_result is None:
        logger.info("No satisfactory solution found across all attempts.")
        state["wyckoff_result"] = {
            "spg": spg,
            "accepted": False,
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

    if os.path.exists(best_result["cif"]):
        shutil.copy2(best_result["cif"], match_cif)

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
    logger.info(best_result["xtal"])
    best_result["spg"] = spg
    best_result["score"] = best_score
    state["wyckoff_result"] = best_result

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


def _is_strands_gemini_stream_bug(error: BaseException) -> bool:
    error_text = f"{error}\n{traceback.format_exc()}"
    return (
        "strands/models/gemini.py" in error_text
        and "candidate" in error_text
        and "UnboundLocalError" in error_text
    )


def _get_strands_version() -> str:
    try:
        return pkg_version("strands")
    except PackageNotFoundError:
        return "unknown"


def _has_known_gemini_stream_candidate_bug() -> bool:
    try:
        src = inspect.getsource(GeminiModel.stream)
    except Exception:
        return False
    vulnerable_finish_reason_line = "candidate.finish_reason if candidate else \"STOP\""
    has_guard_initialization = "candidate = None" in src
    return vulnerable_finish_reason_line in src and not has_guard_initialization


def _startup_runtime_mode() -> tuple[bool, str]:
    strands_version = _get_strands_version()
    force_fallback_raw = os.getenv("STRANDS_FORCE_FALLBACK", "0")
    allow_graph_raw = os.getenv("STRANDS_ALLOW_GRAPH_WITH_KNOWN_BUG", "0")
    env_force_fallback = force_fallback_raw == "1"
    env_allow_graph = allow_graph_raw == "1"
    has_known_bug = _has_known_gemini_stream_candidate_bug()

    if has_known_bug:
        logger.warning(
            "Detected known Strands Gemini streaming candidate bug in installed package "
            f"(strands=={strands_version}). "
            "Graph execution may crash; fallback mode is recommended."
        )

    use_fallback = env_force_fallback or (has_known_bug and not env_allow_graph)
    mode = "fallback" if use_fallback else "graph"
    print(
        "Startup flags: "
        f"STRANDS_FORCE_FALLBACK={force_fallback_raw}, "
        f"STRANDS_ALLOW_GRAPH_WITH_KNOWN_BUG={allow_graph_raw}"
    )
    logger.info(
        f"Runtime mode: {mode} (strands=={strands_version}, "
        f"known_gemini_stream_bug={has_known_bug}, "
        f"STRANDS_FORCE_FALLBACK={env_force_fallback}, "
        f"STRANDS_ALLOW_GRAPH_WITH_KNOWN_BUG={env_allow_graph})"
    )
    return use_fallback, strands_version


def _run_pipeline_fallback(
    state: dict,
    announce_bug_switch: bool = True,
    status_label: str = "fallback_success",
) -> dict:
    pipeline_start_time = time.perf_counter()
    spg_cell_phase_end_time: float | None = None
    structure_phase_start_time: float | None = None

    if announce_bug_switch:
        logger.info("Detected Strands Gemini streaming bug; switching to deterministic fallback pipeline.")
    else:
        logger.info("Using deterministic pipeline execution.")
    _run_data_preprocessor_stage(state["pxrd_csv"], state)

    def _emit_progress(message: str) -> None:
        print(message)

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
        attempt = result.get("attempt")
        seed = result.get("seed")
        cif = result.get("cif")
        png = result.get("png")

        wr_text = f"{float(wr):.4f}" if wr is not None else "n/a"
        r2_text = f"{float(r2):.4f}" if r2 is not None else "n/a"
        chi2_text = f"{float(chi2):.4f}" if chi2 is not None else "n/a"
        score_text = f"{float(score):.4f}" if score is not None else "n/a"
        attempt_text = str(attempt) if attempt is not None else "n/a"
        seed_text = str(seed) if seed is not None else "n/a"
        cif_text = str(cif) if cif else "n/a"
        png_text = str(png) if png else "n/a"

        _emit_progress(
            f"{prefix} details: spg={spg_value}, Wr={wr_text}, R2={r2_text}, "
            f"Chi2={chi2_text}, score={score_text}, attempt={attempt_text}, seed={seed_text}"
        )
        _emit_progress(f"Accepted artifacts: CIF={cif_text}, PNG={png_text}")

    def _validate_reused_cell_for_spg(cell_obj, spg_value: int, peak_positions: np.ndarray):
        try:
            solver = CellSolver(
                int(spg_value),
                peak_positions,
                max_mismatch=12,
                hkl_max=(2, 5, 6),
                max_square=28,
                total_square=40,
                theta_tols=[0.1, 0.15, 0.5],
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
        target_system = _spg_to_crystal_system(int(filename_spg)) if filename_spg is not None else None
    elif lattice_filter == "any":
        target_system = None
    elif lattice_filter in VALID_LATTICE_SYMMETRIES:
        target_system = lattice_filter
    else:
        _emit_progress(f"Unknown lattice symmetry filter '{lattice_filter}', using unfiltered SG candidates.")
        target_system = None

    if predicted_spgs and target_system is not None:
        filtered_spgs = [sg for sg in predicted_spgs if _spg_to_crystal_system(sg) == target_system]
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
            _emit_progress(
                f"Cell solve phase 1 — SG rank {seed_rank}/{len(predicted_spgs)}: collecting cells for spg={seed_spg}"
            )
            seed_state = copy.deepcopy(state)
            seed_state["spg"] = seed_spg
            _run_cell_solver_stage(seed_state)
            seed_cells = seed_state.get("cells") or []

            if not seed_cells:
                _emit_progress(f"spg={seed_spg} produced no candidate cells.")
                continue

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
            # 3. Rank by (small volume first, lower estimated trials first), then
            #    prediction rank and fit quality.
            grouped_seed_cells: dict[tuple, list[tuple[float, object, int]]] = {}
            for item in all_seed_cells:
                _vol, _cell, _spg = item
                sig = _canonical_cell_signature(_cell)
                grouped_seed_cells.setdefault(sig, []).append(item)

            planned_groups = []
            for sig, members in grouped_seed_cells.items():
                enriched_members = []
                for _vol, _cell, _spg in members:
                    cand_count, est_trials = _estimate_pair_trial_cost(_cell, _spg)
                    if cand_count == 0:
                        continue  # no valid Wyckoff assignments — skip
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
                    continue  # all members in this family had no valid Wyckoff assignments

                enriched_members.sort(
                    key=lambda m: (
                        m["est_trials"],
                        round(m["vol"], 1),
                        m["cand_count"],
                        _prediction_rank(m["spg"]),
                        getattr(m["cell"], "missing", 999),
                        _chi2_bucket(getattr(m["cell"], "chi2", 1e9)),
                        getattr(m["cell"], "chi2", 1e9),
                        -CRYSTAL_SYSTEM_PRIORITY.get(_spg_to_crystal_system(int(m["spg"])), 0),
                        -int(m["spg"]),
                    )
                )
                best_symmetry = max(
                    CRYSTAL_SYSTEM_PRIORITY.get(_spg_to_crystal_system(int(m["spg"])), 0)
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

            all_seed_cells = [
                (member["vol"], member["cell"], member["spg"])
                for group in planned_groups
                for member in group["members"]
            ]

            vol_lo = all_seed_cells[0][0]
            vol_hi = all_seed_cells[-1][0]
            _emit_progress(
                f"Phase 2: planned {len(all_seed_cells)} (cell, SPG) pair(s) across "
                f"{len(planned_groups)} cell family/families. Volume range: {vol_lo:.1f}–{vol_hi:.1f} Å³"
            )
            _emit_progress(
                "Phase 2 strategy: prioritize (small volume, fewer estimated trials/candidates), "
                "then (symmetry, SG prediction rank, missing, chi2)."
            )

            # ── Phase 2 summary table ────────────────────────────────────────────
            _emit_progress(
                f"\n{'Rank':<5} {'SPG':<5} {'Volume(Å³)':<11} {'Chi2':<8} {'Missing':<8} {'EstTrials':<10} Dims"
            )
            _emit_progress("-" * 92)
            for _ri, (_vol, _cell, _spg) in enumerate(all_seed_cells, start=1):
                _cand_count, _est_trials = _estimate_pair_trial_cost(_cell, _spg)
                _dims_str = "  ".join(f"{float(x):8.3f}" for x in _cell.dims)
                _emit_progress(
                    f"{_ri:<5} {_spg:<5} {_vol:<11.1f} "
                    f"{getattr(_cell, 'chi2', float('nan')):<8.4f} "
                    f"{getattr(_cell, 'missing', -1):<8} {_est_trials:<10} {_dims_str}"
                )
            _emit_progress("")

            spg_cell_phase_end_time = time.perf_counter()
            structure_phase_start_time = spg_cell_phase_end_time

            # ── Phase 3: systematic structure generation across all ranked (cell, spg) pairs ──
            # Each entry is already a specific (cell, spg) pairing — enumerate Wyckoff
            # only for that SPG to avoid redundant work across identical cell dims.
            for rank_idx, (vol, cell, seed_spg) in enumerate(all_seed_cells, start=1):
                pair_desc = (
                    f"[Pair {rank_idx}/{len(all_seed_cells)}] vol={vol:.1f} Å³, "
                    f"spg={seed_spg}, dims={[round(float(x), 3) for x in cell.dims]}"
                )

                try:
                    consolidated_wp = _get_wp_candidates_for_pair(cell, seed_spg)
                except Exception as exc:
                    _emit_progress(f"{pair_desc}: Wyckoff enumeration failed ({exc}). Skipping.")
                    continue

                if not consolidated_wp:
                    _emit_progress(f"{pair_desc}: no Wyckoff candidates found. Skipping.")
                    continue

                _emit_progress(pair_desc)

                top_preview = [
                    f"spg={s[0]} count={s[6]} dof={s[5]}"
                    for s in consolidated_wp[:3]
                ]
                _emit_progress(
                    f"  Pair {rank_idx}: {len(consolidated_wp)} WP candidates. "
                    f"Top: {' | '.join(top_preview)}"
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

                        try:
                            passed, _metrics, reject_reason = _validate_reused_cell_for_spg(
                                cell, spg_val, peak_positions_np
                            )
                            if not passed:
                                _emit_progress(
                                    f"    Pair {rank_idx} rejected for spg={spg_val}: {reject_reason}"
                                )
                                continue
                        except Exception as exc:
                            _emit_progress(
                                f"    Pair {rank_idx} precheck error for spg={spg_val}: {exc}"
                            )
                            continue

                        _emit_progress(
                            f"  WP #{wp_attempted}: spg={spg_val}, count={count}, dof={dof}, n_wps={num_wps}"
                        )

                        trial_state = copy.deepcopy(state)
                        trial_state["spg"] = spg_val
                        trial_state["cells"] = copy.deepcopy([cell])
                        trial_state["suppress_local_energy_plot"] = True
                        forced_wp_solution = sol[:8] if len(sol) >= 9 else sol
                        trial_state["forced_wp_solution"] = forced_wp_solution

                        trial_message = _run_wyckoff_solver_stage(trial_state)
                        trial_result = trial_state.get("wyckoff_result") or {}

                        # Accumulate structure log across all trials for global plot
                        global_structure_log.extend(trial_state.get("structure_log") or [])

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
                            enough_global_structures = len(global_structure_log) >= max(0, int(state.get("min_structures_before_early_stop", 10)))
                            if stop_on_first_accepted_inferred_spg and strict_early_exit and enough_global_structures:
                                _emit_progress(
                                    f"Good solution found early: spg={spg_val}, "
                                    f"R2={trial_result.get('r2', 0):.4f}, "
                                    f"Chi2={trial_result.get('chi2', 0):.4f}. "
                                    f"Stopping search after pair {rank_idx}/{len(all_seed_cells)} "
                                    f"and {wp_attempted} WP candidate(s)."
                                )
                                if global_structure_log:
                                    timing_breakdown = _current_timing_breakdown_seconds()
                                    state["timing_breakdown_seconds"] = timing_breakdown
                                    formula_str = state.get("formula", "unknown")
                                    _plot_energy_vs_r2(
                                        global_structure_log,
                                        formula_str,
                                        "all",
                                        f"Results/EnergyR2_{formula_str}_global.png",
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
                            _emit_progress(
                                f"Accepted solution found for spg={spg_val}; continuing "
                                f"(stop-on-first-accepted is disabled)."
                            )

                    prev_limit = limit

                if cell_accepted:
                    _emit_progress(
                        f"Pair {rank_idx}: accepted solution found; moving to next ranked pair."
                    )

            # End of all-pairs loop: emit global plot covering every structure tried
            if global_structure_log:
                timing_breakdown = _current_timing_breakdown_seconds()
                state["timing_breakdown_seconds"] = timing_breakdown
                formula_str = state.get("formula", "unknown")
                global_plot_status = "Success" if (best_trial_state and (best_trial_state.get("wyckoff_result") or {}).get("accepted", False)) else "Failure"
                _plot_energy_vs_r2(
                    global_structure_log,
                    formula_str,
                    "all",
                    f"Results/EnergyR2_{formula_str}_global.png",
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
                _emit_progress(
                    f"Completed inferred SG sweep; returning best accepted result from spg={best_trial_state.get('spg')}."
                )
            else:
                _emit_progress(
                    f"No inferred space group met acceptance thresholds; returning best fallback result from spg={best_trial_state.get('spg')}."
                )
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
    wyckoff_message = _run_wyckoff_solver_stage(state)
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


def _run_pipeline_graph_consistent(state: dict) -> dict:
    return _run_pipeline_fallback(
        state,
        announce_bug_switch=False,
        status_label="graph_success",
    )

@tool(context=True)
def WyckoffSolverTool(tool_context: ToolContext) -> str:
    return _run_wyckoff_solver_stage(tool_context.invocation_state)

@tool(context=True)
def CellSolverTool(tool_context: ToolContext) -> dict:
    return _run_cell_solver_stage(tool_context.invocation_state)

@tool(context=True)
def DataPreprocessor(pxrd_csv: str, tool_context: ToolContext) -> dict:
    return _run_data_preprocessor_stage(pxrd_csv, tool_context.invocation_state)

DataPreprocessAgent = Agent(
    model=gemini_model,
    tools=[DataPreprocessor],
    system_prompt=(
        "You are a PXRD (Powder X-Ray Diffraction) data analysis specialist.\n\n"
        "Your primary task is to preprocess experimental PXRD data for crystal structure determination.\n\n"
        "When given a PXRD CSV file path, you should:\n"
        "1. Extract the chemical formula and space group from the filename\n"
        "2. Load and process the diffraction pattern data\n"
        "3. Identify characteristic peaks using scipy algorithms\n"
        "4. Predict material density using ensemble ML models\n"
        "5. Calculate minimum volume constraints for unit cell indexing\n\n"
        "Always use the DataPreprocessor tool to perform these tasks.\n"
        "Report any issues with data quality or processing errors immediately.\n"
        "The return format should include status, messages, and all relevant computed parameters."
    )
)

CellManagerAgent = Agent(
    model=gemini_model,
    tools=[CellSolverTool],
    system_prompt=(
        "You are a PXRD unit cell solver specialist.\n\n"
        "Your primary task is to determine unit cell parameters from the given PXRD peak data.\n\n"
        "For the given peak positions, chemical composition, space group, and constraints, you should:\n"
        "1. Use indexing algorithms to find candidate unit cells that fit the peak data\n"
        "2. Apply constraints based on composition and predicted density to filter solutions\n"
        "3. Rank candidate cells based on fit quality and physical plausibility\n\n"
        "Always use the CellSolver tool to perform these tasks.\n"
        "Report any issues with indexing or solution quality immediately.\n"
        "The return format should include status, messages, and all relevant computed unit cell parameters."
    )
)

WyckoffSolverAgent = Agent(
    model=gemini_model,
    tools=[WyckoffSolverTool],
    system_prompt=(
        "You are a specialist in crystal structure generation and optimization.\n\n"
        "Your primary task is to generate candidate crystal structures from indexed unit cells "
        "and optimize them to match experimental PXRD data.\n\n"

        "**Your Workflow:**\n"
        "1. **Wyckoff Position Assignment**\n"
        "   - For each candidate unit cell, enumerate possible Wyckoff position combinations\n"
        "   - Consider space group symmetry constraints\n"
        "   - Filter based on composition and density constraints\n\n"

        "2. **Structure Generation**\n"
        "   - Generate initial atomic positions using symmetry operations\n"
        "   - Validate structural geometry and atomic overlaps\n"
        "   - Generate multiple random configurations per Wyckoff assignment\n\n"

        "3. **Geometry Optimization**\n"
        "   - Relax atomic positions using ASE with MACE force field\n"
        "   - Track energy minimization to identify stable configurations\n\n"
        "   - Apply stress constraints (max stress < 0.5 GPa initially)\n"

        "4. **XRD Pattern Matching**\n"
        "   - Calculate theoretical XRD patterns for optimized structures\n"
        "   - Compare with experimental data using similarity metrics\n"
        "   - Track best matches (similarity > 0.90)\n\n"

        "5. **Rietveld Refinement**\n"
        "   - Perform full-pattern refinement using GSAS-II for promising candidates\n"
        "   - Calculate fit metrics: Rwp, R², χ²\n"
        "   - Accept solutions with R² > 0.95 or χ² < 0.12\n\n"

        "**Search Strategy:**\n"
        "- Test top 5 unit cells (ranked by missing peaks)\n"
        "- Evaluate up to 20 Wyckoff position combinations per cell\n"
        "- Generate 3×DOF + 1 random structures per combination (max DOF=9)\n"
        "- Stop immediately when R² > 0.95 or χ² < 0.12 is achieved\n\n"

        "**Quality Criteria:**\n"
        "- Structural validity (no atomic overlaps)\n"
        "- Converged geometry (stress < 0.5 GPa)\n"
        "- Low potential energy per atom\n"
        "- High XRD pattern similarity (> 0.90)\n"
        "- Excellent Rietveld fit (R² > 0.95 or χ² < 0.12)\n\n"

        "**Expected Input (from previous agents):**\n"
        "- Space group number\n"
        "- Chemical formula and composition\n"
        "- List of indexed unit cells with dimensions\n"
        "- Experimental PXRD data (x1, y1, peaks)\n"
        "- Density constraints (min, max)\n"
        "- X-ray wavelength\n\n"

        "**Output Format:**\n"
        "Report the following:\n"
        "1. Number of cells tested\n"
        "2. Total structures generated and optimized\n"
        "3. Best similarity score and energy achieved\n"
        "4. Final Rietveld refinement metrics (Rwp, R², χ²)\n"
        "5. Whether a satisfactory solution was found (R² > 0.95)\n"
        "6. Paths to saved structure file (.cif) and XRD comparison plot (.png)\n\n"

        "Always use the WyckoffSolverTool to perform these computationally intensive tasks.\n"
        "This tool may take several minutes to hours depending on complexity.\n"
        "Report progress updates and immediately notify when a satisfactory solution is found.\n"
        "If no solution meets the R² threshold after exhausting the search space, "
        "recommend adjustments to constraints or suggest alternative space groups."
    )
)

builder = GraphBuilder()
builder.add_node(DataPreprocessAgent, "DataPreprocessorAgent")
builder.add_node(CellManagerAgent, "CellSolverAgent")
builder.add_node(WyckoffSolverAgent, "WyckoffSolverAgent")
builder.add_edge("DataPreprocessorAgent", "CellSolverAgent")
builder.add_edge("CellSolverAgent", "WyckoffSolverAgent")
builder.set_entry_point("DataPreprocessorAgent")
graph = builder.build()

def main(
    state: dict | None = None,
    input_prompt: str | None = None,
    pxrd_csv: str | None = None,
    formula: str | None = None,
    multi_attempts: int | None = None,
    seed_base: int | None = None,
    infer_spg_from_pxrd: bool | None = None,
    try_all_inferred_spg: bool | None = None,
    spg_top_k: int | None = None,
    spg_infer_backend: str | None = None,
    show_spg_predictions: bool | None = None,
    lattice_symmetry: str | None = None,
    max_local_boosts: int | None = None,
    max_local_perturbations: int | None = None,
    perturb_displacement: float | None = None,
    max_eng_rel: float | None = None,
    max_cell_volume: float | None = None,
) -> None:
    run_state = copy.deepcopy(default_state if state is None else state)
    if pxrd_csv is not None:
        run_state["pxrd_csv"] = pxrd_csv
    if formula is not None:
        run_state["formula"] = formula
    if multi_attempts is not None:
        run_state["multi_attempts"] = max(1, int(multi_attempts))
    if seed_base is not None:
        run_state["seed_base"] = int(seed_base)
    if infer_spg_from_pxrd is not None:
        run_state["infer_spg_from_pxrd"] = bool(infer_spg_from_pxrd)
    if try_all_inferred_spg is not None:
        run_state["stop_on_first_accepted_inferred_spg"] = not bool(try_all_inferred_spg)
    if spg_top_k is not None:
        run_state["spg_top_k"] = int(spg_top_k)
    if spg_infer_backend is not None:
        backend = str(spg_infer_backend).strip().lower()
        if backend in SPG_INFER_BACKENDS:
            run_state["spg_infer_backend"] = backend
        else:
            logger.warning(f"Unsupported spg infer backend '{spg_infer_backend}', using model backend.")
            run_state["spg_infer_backend"] = "model"
    if show_spg_predictions is not None:
        run_state["show_spg_predictions"] = bool(show_spg_predictions)
    if lattice_symmetry is not None:
        run_state["lattice_symmetry"] = str(lattice_symmetry).strip().lower()
    elif bool(run_state.get("infer_spg_from_pxrd", False)) and str(run_state.get("spg_infer_backend", "model")).strip().lower() == "smart-cell":
        run_state["lattice_symmetry"] = "any"
    if max_local_boosts is not None:
        run_state["max_local_boosts"] = max(0, int(max_local_boosts))
    if max_local_perturbations is not None:
        run_state["max_local_perturbations"] = max(0, int(max_local_perturbations))
    if perturb_displacement is not None:
        run_state["perturb_displacement"] = max(0.0, float(perturb_displacement))
    if max_eng_rel is not None:
        run_state["max_eng_rel"] = max(0.0, float(max_eng_rel))
        run_state["max_eng_rel_early_stop"] = max(0.0, float(max_eng_rel))
    if max_cell_volume is not None:
        max_cell_volume = float(max_cell_volume)
        if max_cell_volume > 0:
            run_state["max_cell_volume"] = max_cell_volume
        else:
            logger.warning(f"Ignoring non-positive max_cell_volume={max_cell_volume}; expected > 0.")
    run_prompt = INPUT_PROMPT if input_prompt is None else input_prompt
    force_fallback, _ = _startup_runtime_mode()
    _FAILURE_STATUSES = {"no_cells", "no_solution"}

    try:
        if force_fallback:
            print("Starting pipeline in fallback mode.")
            result = _run_pipeline_fallback(run_state)
        else:
            if bool(run_state.get("infer_spg_from_pxrd", False)):
                print("Starting pipeline in graph-consistent deterministic mode.")
                result = _run_pipeline_graph_consistent(run_state)
            else:
                result = graph(run_prompt,
                               invocation_state=run_state)
    except KeyboardInterrupt:
        print("Process interrupted by user")
        print("Exiting main thread")
        return
    except Exception as exc:
        if _is_strands_gemini_stream_bug(exc):
            print("Strands Gemini streaming error detected; retrying with fallback execution.")
            result = _run_pipeline_fallback(run_state)
        else:
            raise

    result_status = result.get("status", "") if isinstance(result, dict) else ""

    timing_breakdown = run_state.get("timing_breakdown_seconds") if isinstance(run_state, dict) else None
    if isinstance(timing_breakdown, dict):
        def _fmt_seconds(seconds: float) -> str:
            total_seconds = max(0.0, float(seconds))
            total_minutes = int(total_seconds // 60)
            seconds_remain = total_seconds - (60 * total_minutes)
            if total_minutes >= 60:
                hours = total_minutes // 60
                minutes = total_minutes % 60
                return f"{hours}h {minutes}m {seconds_remain:04.1f}s"
            return f"{total_minutes}m {seconds_remain:04.1f}s"

        spg_cell_s = float(timing_breakdown.get("spg_and_cell", 0.0))
        structure_s = float(timing_breakdown.get("structure_inference", 0.0))
        total_s = float(timing_breakdown.get("total", spg_cell_s + structure_s))
        timing_line = (
            f"Timing summary: SPG+Cell={_fmt_seconds(spg_cell_s)} | "
            f"Structure={_fmt_seconds(structure_s)} | Total={_fmt_seconds(total_s)}"
        )
        logger.info(timing_line)
        print(timing_line)

    if result_status in _FAILURE_STATUSES:
        reason = {
            "no_cells": "no valid unit cells found",
            "no_solution": "no accepted structure found",
        }.get(result_status, result_status)
        print(f"Pipeline finished without a solution ({reason}).")
    else:
        print("Pipeline completed successfully!")
    print("Exiting main thread")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PXRD agent pipeline")
    parser.add_argument(
        "--input-csv",
        default="Examples/PXRD_PrYMg2_123.csv",
        help=(
            "Path to a PXRD CSV file, or a directory containing CSV files. "
            "If a directory is provided, all '*.csv' files in that directory are processed."
        ),
    )
    parser.add_argument(
        "--input-formula",
        default="",
        help="Optional formula override. Leave empty to parse from filename.",
    )
    parser.add_argument(
        "--multi-attempts",
        type=int,
        default=None,
        help="Override number of adaptive attempts (same as PXRD_MULTI_ATTEMPTS).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=123456,
        help="Override base random seed (same as PXRD_SEED_BASE).",
    )
    parser.add_argument(
        "--infer-spg",
        action="store_true",
        help="Infer space group from PXRD/profile model instead of filename convention.",
    )
    parser.add_argument(
        "--try-all-inferred-spg",
        action="store_true",
        help="When --infer-spg is on, do not stop at first accepted SG; evaluate all inferred candidates and keep best.",
    )
    parser.add_argument(
        "--spg-top-k",
        type=int,
        choices=[3, 5, 10, 20, 25, 30, 50, 100],
        default=100,
        help="Number of inferred space-group options to evaluate/show (3 or 5).",
    )
    parser.add_argument(
        "--spg-infer-backend",
        type=str,
        choices=["model", "smart-cell"],
        default="model",
        help=(
            "Backend for --infer-spg: 'model' uses pretrained SG classifier, "
            "'smart-cell' uses SmartCellSolver to rank likely SGs by high->low symmetry and indexing evidence."
        ),
    )
    parser.add_argument(
        "--symmetry",
        type=str,
        choices=["auto", "any", "triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"],
        default=None,
        help=(
            "Optional crystal-system filter for inferred SG candidates. "
            "Defaults to 'auto' when --infer-spg is set; otherwise unset. "
            "'auto' uses filename SPG (if present), 'any' disables filtering."
        ),
    )
    parser.add_argument(
        "--local-boosts",
        type=int,
        default=None,
        help="Maximum number of extra regeneration boosts per promising Wyckoff setting.",
    )
    parser.add_argument(
        "--local-perturbations",
        type=int,
        default=None,
        help="Maximum number of perturb-and-relax trials per promising Wyckoff setting.",
    )
    parser.add_argument(
        "--perturb-displacement",
        type=float,
        default=None,
        help="Standard deviation of Cartesian perturbation in Å for local perturb-and-relax trials.",
    )
    parser.add_argument(
        "--max-eng-rel",
        type=float,
        default=None,
        help=(
            "Maximum allowed energy-above-best (eV/atom) for immediate early termination on excellent "
            "refined fits. If unset, uses max(refine_eng_window, 0.60)."
        ),
    )
    parser.add_argument(
        "--max-cell-volume",
        type=float,
        default=2500.0,
        help="Maximum allowed unit-cell volume (Å^3) for cell solutions. Larger cells are discarded.",
    )
    args = parser.parse_args()

    if args.symmetry is not None:
        symmetry = args.symmetry
    elif args.infer_spg and args.spg_infer_backend == "smart-cell":
        symmetry = "any"
    else:
        symmetry = "auto" if args.infer_spg else None

    input_path = Path(args.input_csv)
    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in directory: {input_path}")
            sys.exit(1)
        print(f"Found {len(csv_files)} CSV file(s) in '{input_path}'.")
    elif input_path.is_file():
        csv_files = [input_path]
    else:
        print(f"Input path does not exist: {input_path}")
        sys.exit(1)

    shared_kwargs = dict(
        formula=args.input_formula,
        multi_attempts=args.multi_attempts,
        seed_base=args.seed_base,
        infer_spg_from_pxrd=args.infer_spg,
        try_all_inferred_spg=args.try_all_inferred_spg,
        spg_top_k=args.spg_top_k,
        spg_infer_backend=args.spg_infer_backend,
        lattice_symmetry=symmetry,
        max_local_boosts=args.local_boosts,
        max_local_perturbations=args.local_perturbations,
        perturb_displacement=args.perturb_displacement,
        max_eng_rel=args.max_eng_rel,
        max_cell_volume=args.max_cell_volume,
        show_spg_predictions=True,
    )

    for idx, csv_path in enumerate(csv_files, start=1):
        csv_str = str(csv_path)
        if len(csv_files) > 1:
            print(f"\n{'=' * 60}")
            print(f"Processing file {idx}/{len(csv_files)}: {csv_str}")
            print(f"{'=' * 60}\n")
        main(
            pxrd_csv=csv_str,
            input_prompt=f"Process the PXRD data from {csv_str}",
            **shared_kwargs,
        )

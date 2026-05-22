import logging
import os
import random
import time
import copy
from pathlib import Path
#from typing import Sequence, Optional, List, Dict, Union, Any
#from uuid import uuid4
import pandas as pd
import numpy as np

from pxrd_app.constants import CRYSTAL_SYSTEM_PRIORITY
from pxrd_app.inference import infer_formula_spg, infer_spg_from_backend, spg_to_crystal_system
from pxrd_app.plot import plot_energy_vs_r2
from pxrd_app.tools.utils import parse_formula, get_volume_from_density, format_wyckoff_labels
from pxrd_app.tools.manager import RawDataManager, CellManager
from pxrd_app.tools.density import predict_density_ensemble
from pxrd_app.tools.solver import CellSolver, search_solution, enumerate_wyckoff, get_adaptive_wp_limits
from pyxtal.database.element import Element

# Configure logging with both file and console handlers, avoiding duplicate handlers
logger = logging.getLogger("pxrd_agent")
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
file_handler = logging.FileHandler('PXRD_solver.log')
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prefer higher-symmetry, lower-trial solutions and smaller volumes
def get_pair_priority(
    est_trials: int,
    volume: float,
    missing_peaks: int,
    chi2: float,
    min_trials: int,
    min_volume: float,
    max_missing: int,
    trial_weight: float = 0.5,
    vol_weight: float = 0.5,
) -> float:
    safe_trials = max(1.0, float(est_trials))
    safe_volume = max(1e-6, float(volume))
    ref_trials = max(1.0, float(min_trials))
    ref_volume = max(1e-6, float(min_volume))
    trial_ratio = safe_trials / ref_trials
    vol_ratio = safe_volume / ref_volume
    missing = (missing_peaks + 1) / (max_missing + 1)
    return (trial_ratio ** trial_weight) * (vol_ratio ** vol_weight) * (missing ** 0.5) * (chi2 ** 0.4)

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


def _get_global_best_energy(structure_log: list[dict] | None) -> float | None:
    if not structure_log:
        return None
    energies = [
        float(entry.get("eng"))
        for entry in structure_log
        if entry.get("eng") is not None
    ]
    return min(energies) if energies else None

def attach_run_log(state: dict) -> logging.Handler | None:
    results_dir = state.get("results_dir", "Results")
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, os.path.basename(_get_system_run_log_path(state)))
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

def detach_run_log(handler: logging.Handler | None) -> None:
    if handler is None:
        return
    try:
        logger.removeHandler(handler)
    finally:
        try:
            handler.close()
        except Exception:
            pass

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

def _get_cell_solver_kwargs(state: dict) -> dict:
    hkl_max_raw = state.get("cell_solver_hkl_max", (2, 5, 6))
    theta_tols_raw = state.get("cell_solver_theta_tols", [0.1, 0.15, 0.5])
    hkl_max = tuple(int(x) for x in hkl_max_raw)
    theta_tols = [float(x) for x in theta_tols_raw]
    return {
        "max_mismatch": state["cell_solver_max_mismatch"],
        "hkl_max": hkl_max,
        "max_square": state.get("cell_solver_max_square", 28),
        "total_square": state.get("cell_solver_total_square", 40),
        "theta_tols": theta_tols,
        "min_abc": state.get("min_abc"),
        "max_abc": state.get("max_abc"),
        "max_chi2": state.get("cell_solver_max_chi2", 0.5),
        "max_guess": state.get("cell_solver_max_guess", 50000),
    }


def run_data_preprocessor(pxrd_csv: str, state: dict) -> dict:
    formula_from_filename, spg_from_filename = infer_formula_spg(pxrd_csv)
    formula_override = state.get("formula")
    formula = formula_override if formula_override else formula_from_filename
    if not formula:
        raise ValueError("Cannot infer formula. Provide --formula or use PXRD_<formula>_<spg>.csv")

    infer_spg = state["infer_spg_from_pxrd"]
    spg_top_k = state["spg_top_k"]
    max_volume = state["max_volume"]
    spg = int(spg_from_filename) if spg_from_filename is not None else 0

    # Resolve the crystal-system filter that will be applied during SPG inference.
    infer_crystal_system = None

    composition = parse_formula(formula)
    min_abc = state["min_abc"]
    wavelength = state["wavelength"]
    density = predict_density_ensemble(formula, sigma=2.5)
    density_min, density_max = density['min'], density['max']

    df = pd.read_csv(pxrd_csv, comment='#')
    x1, y1 = df.iloc[:, 0].values, df.iloc[:, 1].values.copy()

    # Background subtraction and peak detection
    if y1.min() > 2.5:
        bg_subtract = True
        min_height = 7.5
        height = 7.5
    else:
        if y1.min() > 0.25:
            print(f"Background subtraction applied (min intensity {y1.min():.2f} > 0.25).")
            y1 -= y1.min()
            min_height = 3.0
            height = 3.0
        else:
            min_height = 3.0
            height = 1.5
        bg_subtract = False
    #min_height = 7.5 if bg_subtract else 5.0
    #height = min_height if bg_subtract else 1.5
    #bg_subtract = True
    data = RawDataManager(x1, y1, bg_subtract=bg_subtract)
    data.get_peaks_from_scipy(height=height)
    data.filter_peaks_by_ml(threshold=0.8, min_height=min_height)
    if bg_subtract:
        state['pxrd_csv'] = pxrd_csv.replace(".csv", "_bg_subtracted.csv")
        print(pxrd_csv, state['pxrd_csv'])
        data.to_csv(state['pxrd_csv'])
        logger.info(f"Background subtraction applied (min intensity {y1.min():.2f} > 2.5).")

    #data = RawDataManager(x1, y1, bg_subtract=False)
    #data.get_peaks_from_scipy()
    #data.filter_peaks_by_ml(threshold=0.8, min_height=3.0)
    peaks = data.peaks
    peak_positions = x1[peaks]
    #data.plot('my.pdf')#; import sys; sys.exit()

    # QZ: Just to handle cases with very few peaks with light elements.
    if len(peaks) <= 10 and bg_subtract:
        state['max_sim'] = 0.4
        state['max_chi2'] = 0.22

    if infer_spg:
        result = infer_spg_from_backend(
            peak_positions=np.array(peak_positions, dtype=float),
            formula=formula,
            spg_top_k=spg_top_k,
            max_volume=max_volume,
            crystal_system=infer_crystal_system,
        )
        predictions = result.get("predictions") or []
        state["spg_predictions"] = [spg for spg, _prob in predictions]
        if result.get("source"):
            state["spg_prediction_source"] = result["source"]
        if result.get("smart_cell_candidates_by_spg"):
            state["smart_cell_candidates_by_spg"] = result["smart_cell_candidates_by_spg"]
        if result.get("smart_cell_metrics_by_pair"):
            state["smart_cell_metrics_by_pair"] = result["smart_cell_metrics_by_pair"]
        if result.get("smart_cell_ranked_spg_cells"):
            state["smart_cell_ranked_spg_cells"] = result["smart_cell_ranked_spg_cells"]

    min_volume = get_volume_from_density(composition, density_max) * 0.7

    result = {
        "infer_spg_from_pxrd": infer_spg,
        "spg": spg,
        "formula": formula,
        "x1": x1.tolist(),
        "y1": data.y.tolist(),
        "peaks": peaks.tolist(),
        "peak_positions": peak_positions.tolist(),
        "composition": composition,
        "density_min": density_min,
        "density_max": density_max,
        "min_volume": min_volume,
        "max_volume": max_volume,
        "min_abc": min_abc,
        "wavelength": wavelength,
    }
    state.update(result)

    return result


def run_cell_solver(state: dict) -> dict:
    spg = state.get("spg")
    formula = state.get("formula")
    peak_positions = state.get("peak_positions")
    max_cells = state.get("max_cells")
    max_volume = state.get("max_volume")
    min_volume = state.get("min_volume")
    cell_solver_kwargs = _get_cell_solver_kwargs(state)

    if state['infer_spg_from_pxrd']:
        smart_raw_by_spg = state.get("smart_cell_candidates_by_spg") or {}
        if spg in smart_raw_by_spg:
            raw_solutions = smart_raw_by_spg[spg]
            if raw_solutions:
                cells = CellManager.consolidate(raw_solutions, max_solutions=max_cells, merge_tol=0.05)
                state["cells"] = cells
                if not cells:
                    text = f"Cell solving found no valid unit cells for formula {formula} in space group {spg}.\n"
                    return {"status": "no_cells", "message": text, "cells": []}
                text = (
                    f"Cell solving completed for formula {formula} in space group {spg} "
                    f"using SmartCellSolver cache.\n"
                )
                return {"status": "success", "message": text,
                    "cells": [{"dimensions": cell.dims, "missing_peaks": cell.missing} for cell in cells],
                }
        else:
            raise ValueError(f"SmartCellSolver cannot find SPG {spg}. Exiting.")

    else:
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
            max_abc=cell_solver_kwargs["max_abc"],
            max_chi2=cell_solver_kwargs["max_chi2"],
            max_guess=cell_solver_kwargs["max_guess"],
            max_volume=max_volume,
            min_volume=min_volume,
            verbose=False,
        )
        solutions = solver.solve()

        sols = []
        for sol in solutions:
            if 15 < spg < 75:
                axis_orders = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                for axis_order in axis_orders:
                    _cell = np.array([sol['cell'][i] for i in axis_order])
                    if solver.validate_cell(_cell):
                        sols.append((spg, _cell, sol['mismatch'], sol['chi2'][1], sol['errors'], sol['id'], sol['match']))
            else:
                sols.append((spg, sol['cell'], sol['mismatch'], sol['chi2'][1], sol['errors'], sol['id'], sol['match']))

        if not sols:
            state["cells"] = []
            text = f"Cell solving found no valid unit cells for formula {formula} in space group {spg}.\n"
            return {
                "status": "no_cells",
                "message": text,
                "cells": [],
            }

        cells = CellManager.consolidate(sols, max_solutions=max_cells, merge_tol=0.05)

        if not cells:
            state["cells"] = []
            text = f"Cell solving found no valid unit cells for formula {formula} in space group {spg}.\n"
            return {
                "status": "no_cells",
                "message": text,
                "cells": [],
            }

        state["cells"] = cells
        text = f"Cell solving completed for formula {formula} in space group {spg}.\n"
        return {
            "status": "success",
            "message": text,
            "cells": [{"dimensions": cell.dims, "missing_peaks": cell.missing} for cell in cells],
        }


def run_wyckoff_solver(state: dict, all_structure_log: list, structure_id_counter=None,
                       global_accepted: bool = False, factor=1) -> str:
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
    forced_wp_solution = state.get("forced_wp_solution")
    if "forced_wp_solution" in state: state.pop("forced_wp_solution", None)
    min_r2 = state.get("min_r2")
    max_chi2 = state.get("max_chi2")
    max_force = state.get("max_force")
    max_stress = state.get("max_stress")
    max_eng = state.get("max_eng")
    disable_early_termination = bool(state.get("disable_early_termination", False))
    min_structures_before_early_stop = max(0, int(state.get("min_structures_before_early_stop", 10)))
    sim_max = state.get("max_sim")
    if max([Element(ele).z for ele in composition.keys()]) <= 6: sim_max = 0.55
    if len(state.get("peaks")) <=4: sim_max = 0.2
    eng_min = _get_global_best_energy(all_structure_log) or 1e10
    max_wp = state.get("max_wp")
    max_Z = state.get("max_Z")
    max_dof = state.get("max_dof")
    per_dof = state.get("per_dof")
    max_wp_choices = state.get("max_wp_choices")
    max_atoms = state.get("max_atoms")

    results_dir = state.get("results_dir", "Results")
    cifs_dir = os.path.join(results_dir, "cifs")
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(cifs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    tmp_root = Path(results_dir) / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ["PXRD_TMP_ROOT"] = str(tmp_root)
    gsas_timeout = state.get("gsas_refine_timeout")
    if gsas_timeout is not None:
        os.environ["GSAS_REFINE_TIMEOUT"] = str(int(gsas_timeout))
    gsas_max_calls = state.get("gsas_max_calls_per_worker")
    if gsas_max_calls is not None:
        os.environ["GSAS_MAX_CALLS_PER_WORKER"] = str(int(gsas_max_calls))
    gsas_max_cyc = state.get("gsas_max_cyc")
    if gsas_max_cyc is not None:
        os.environ["GSAS_MAX_CYC"] = str(int(gsas_max_cyc))
    gsas_early_exit_wr = state.get("gsas_early_exit_wr")
    if gsas_early_exit_wr is not None:
        os.environ["GSAS_EARLY_EXIT_WR"] = str(float(gsas_early_exit_wr))
    run_token = f"{_safe_name_token(Path(str(pxrd_csv or '')).stem)}"#_{os.getpid()}_{uuid4().hex[:8]}"
    run_tmp_dir = tmp_root / f"run_{run_token}"
    run_tmp_dir.mkdir(parents=True, exist_ok=True)

    match_cif = os.path.join(cifs_dir, f'Match_{formula}_{spg}.cif')
    stale_result_cifs = [
        *Path(cifs_dir).glob(f"Match_{run_token}_attempt*.cif"),
        *Path(cifs_dir).glob(f"Match_{run_token}_attempt*_refined.cif"),
    ]
    for stale_path in stale_result_cifs:
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass
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

    struc_count = state.get("Struc_count") or 0
    update_best = False

    attempt_prefix = run_tmp_dir / f"Match_{run_token}_attempt1"
    attempt_cif = str(attempt_prefix.with_suffix(".cif"))

    wr, r2, chi2, xtal, eng_best, selected_eng, _, struc_count, attempt_count, qrs_id = search_solution(
        cells,
        spg,
        composition,
        ref_den,
        attempt_cif,
        pxrd_csv,
        x1,
        y1,
        eng_min,
        sim_max,
        max_wp_choices,
        struc_count if structure_id_counter is None else structure_id_counter,
        max_force,
        max_stress,
        wavelength,
        thetas,
        resolution,
        SCALED_INTENSITY_TOL,
        INST_FILE,
        logger,
        max_wp,
        max_Z,
        max_dof,
        per_dof,
        max_atoms,
        min_r2,
        max_chi2,
        structure_log=all_structure_log,
        max_eng=max_eng,
        min_structures_before_early_stop=min_structures_before_early_stop,
        disable_early_termination=disable_early_termination,
        forced_wp_solution=forced_wp_solution,
        ase_log=state["ase_log"],
        global_accepted=global_accepted,
        qrs_method=state["qrs"],
        factor=factor,
    )

    state["attempt_count"] += attempt_count
    #print(f"{struc_count} new structure(s). Total: {len(all_structure_log)}/{state['attempt_count']}.")
    # Accumulate struc_count with the number of new structures generated in this attempt
    if wr is not None:
        global_best_energy = _get_global_best_energy(all_structure_log)
        if global_best_energy is None and eng_best is not None:
            global_best_energy = float(eng_best)
        if global_best_energy is not None:
            eng_min = min(float(eng_min), float(global_best_energy))

        candidate_selected_energy = float(selected_eng) if selected_eng is not None else None
        candidate_eng = None
        if candidate_selected_energy is not None and global_best_energy is not None:
            candidate_eng = max(0.0, candidate_selected_energy - float(global_best_energy))

        candidate = {
            "wr": float(wr),
            "r2": float(r2),
            "chi2": float(chi2),
            "xtal": xtal,
            "eng_best": float(global_best_energy) if global_best_energy is not None else float(eng_best),
            "selected_energy": candidate_selected_energy,
            "eng_rel": candidate_eng,
            "wp_labels": state.get("wp_labels"),
            "cif": attempt_cif,
            "qrs_id": qrs_id,
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

        if score > state.get('best_score', -1e9):
            state['best_score'] = score
            state['best_result'] = candidate
            update_best = True

        # Early stop: excellent solution found
        if (not disable_early_termination and state['best_result']["accepted"]
                and len(all_structure_log) >= min_structures_before_early_stop and (
            state['best_result']["r2"] >= max(min_r2 + 0.02, 0.97)
            or state['best_result']["chi2"] <= min(max_chi2 * 0.7, 0.08)
        )):
            logger.info(f"Early stop: excellent solution found.")

    state["structure_log"] = all_structure_log
    state["Struc_count"] = struc_count

    if state['best_result'] is None: #and len(all_structure_log) >= min_structures_before_early_stop:
        state["wyckoff_result"] = {
            "spg": spg,
            "accepted": False,
            "Struc_count": struc_count,
            "wr": None,
            "r2": None,
            "chi2": None,
            "eng_best": eng_min,
            "attempt": None,
            "png": None,
            "cif": None,
            "score": None,
        }
        text = f"Wyckoff solving completed for formula {formula} in space group {spg}.\n"
        text += f"Best similarity: {sim_max:.3f}, Minimum energy per atom: {eng_min:.3f} eV\n"
        text += "No satisfactory solution found.\n"
        return text, state

    state['best_result']["xtal"].to_file(match_cif)
    wr = state['best_result']["wr"]
    r2 = state['best_result']["r2"]
    chi2 = state['best_result']["chi2"]
    if update_best:
        logger.info(f"\nFinal refinement results: Wr={wr:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}")
        logger.info(f"Best structure saved to {match_cif}")
    state['best_result']["spg"] = spg
    state['best_result']["score"] = state['best_score']
    state['best_result']["Struc_count"] = struc_count
    state["wyckoff_result"] = state['best_result']

    text = f"Wyckoff solving completed for formula {formula} in space group {spg}.\n"
    text += f"Best similarity: {sim_max:.3f}, Min_energy: {state['best_result']['eng_best']:.3f} eV\n"
    text += f"Final Rietveld refinement results: Wr={wr:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}\n"
    if not state['best_result']["accepted"]:
        text += "Best candidate did not meet the acceptance thresholds, but was kept as a fallback result.\n"
    text += f"Best structure saved to {match_cif}\n"
    return text, state

def run_pipeline(state: dict) -> dict:

    def _format_elapsed(seconds: float) -> str:
        total_seconds = max(0.0, float(seconds))
        total_minutes = int(total_seconds // 60)
        seconds_remain = total_seconds - (60 * total_minutes)
        if total_minutes >= 60:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m {seconds_remain:04.1f}s"
        return f"{total_minutes}m {seconds_remain:04.1f}s"

    def _timing_breakdown_seconds() -> dict:
        nonlocal spg_cell_end_time, structure_start_time
        now = time.perf_counter()
        if spg_cell_end_time is None:
            spg_cell_seconds = now - pipeline_start_time
            structure_seconds = 0.0
        else:
            spg_cell_seconds = spg_cell_end_time - pipeline_start_time
            if structure_start_time is None:
                structure_seconds = now - spg_cell_end_time
            else:
                structure_seconds = now - structure_start_time
        total_seconds = spg_cell_seconds + structure_seconds
        logger.info("Timing breakdown:")
        logger.info(f"  1) SPG + cell inference: {_format_elapsed(spg_cell_seconds)}")
        logger.info(f"  2) Structure inference: {_format_elapsed(structure_seconds)}")
        logger.info(f"  Total: {_format_elapsed(total_seconds)}")

        return {
            "spg_and_cell": float(spg_cell_seconds),
            "structure_inference": float(structure_seconds),
            "total": float(total_seconds),
        }

    def _emit_accepted_solution(spg_value: int, result: dict, prefix: str = "Accepted solution") -> None:
        wr = result.get("wr")
        r2 = result.get("r2")
        chi2 = result.get("chi2")
        score = result.get("score")
        selected_energy = result.get("selected_energy")
        eng_rel = result.get("eng_rel")

        global global_structure_log

        wr_text = f"{float(wr):.4f}" if wr is not None else "n/a"
        r2_text = f"{float(r2):.4f}" if r2 is not None else "n/a"
        chi2_text = f"{float(chi2):.4f}" if chi2 is not None else "n/a"
        score_text = f"{float(score):.4f}" if score is not None else "n/a"
        energy_text = f"{float(selected_energy):.4f}" if selected_energy is not None else "n/a"
        eng_rel_text = f"{float(eng_rel):.4f}" if eng_rel is not None else "n/a"

        logger.info(
            f"{prefix}: spg={spg_value}, Wr={wr_text}, R2={r2_text}, "
            f"Chi2={chi2_text}, score={score_text}, E={energy_text}, dE={eng_rel_text}, "
        )

    pipeline_start_time = time.perf_counter()
    spg_cell_end_time: float | None = None
    structure_start_time: float | None = None

    logger.info("Using deterministic pipeline execution.")
    run_data_preprocessor(state["pxrd_csv"], state)

    infer_spg = state["infer_spg_from_pxrd"]
    composition = state["composition"]
    density_min, density_max = state["density_min"], state["density_max"]
    csv_path = state.get("wp_path")

    wp_candidate_cache: dict[tuple, list] = {}
    wp_cost_cache: dict[tuple, tuple[int, int, int]] = {}

    def _cell_cache_key(cell_obj, spg_value) -> tuple:
        dims_sig = tuple(round(float(x), 3) for x in np.array(cell_obj.dims).tolist())
        return (int(spg_value), dims_sig)

    def _get_wp_candidates_for_pair(cell_obj, spg_value, max_wp, max_Z, max_dof, max_samples=None, csv_path=None) -> list:
        key = _cell_cache_key(cell_obj, spg_value)
        if key in wp_candidate_cache: return wp_candidate_cache[key]
        candidates = enumerate_wyckoff(
            cell_obj.dims,
            [spg_value],
            composition,
            max_wp, max_Z, max_dof,
            ref_den=(density_min, density_max),
            verbose=True,
            max_samples=max_samples,
            csv_path=csv_path,
        )
        #print(f"Enumerated {len(candidates)} Wyckoff candidates for cell {cell_obj.dims} under SPG {spg_value}.")
        wp_candidate_cache[key] = candidates
        return candidates

    def _estimate_pair_cost(cell_obj, spg_value, max_wp, max_Z, max_dof, per_dof, csv_path, max_samples=None) -> tuple[int, int, int]:
        key = _cell_cache_key(cell_obj, spg_value)
        if key in wp_cost_cache: return wp_cost_cache[key]

        candidates = _get_wp_candidates_for_pair(cell_obj, spg_value, max_wp, max_Z, max_dof,
                                                 max_samples=max_samples, csv_path=csv_path)
        #print(f"Estimated cell {cell_obj.dims} under SPG {spg_value}: {len(candidates)} Wyckoffs.")
        candidate_count = len(candidates)
        est_trials = sum(candidate[6] for candidate in candidates)  # sum of dof*per_dof across candidates
        out = (candidate_count, candidate_count, est_trials)
        wp_cost_cache[key] = out
        return out

    max_wp = state.get("max_wp")
    max_Z = state.get("max_Z")
    max_dof = state.get("max_dof")
    per_dof = state.get("per_dof")
    max_samples = state.get("max_enumeration_samples")
    max_trials = state.get("max_trials")
    state["attempt_count"] = 0

    if infer_spg:
        predicted_spgs = state["spg_predictions"][:state["spg_top_k"]]

        force_spg = state.get("force_spg")
        if force_spg is not None:
            before = len(predicted_spgs)
            predicted_spgs = [sg for sg in predicted_spgs if sg == int(force_spg)]
            logger.info(
                f"Applying --spg filter: restricted to spg={force_spg}. "
                f"Kept {len(predicted_spgs)}/{before} inferred SG candidate(s)."
            )
            if len(predicted_spgs) == 0:
                logger.info(f"No SPG candidates match --spg={force_spg}; aborting search.")
                return state
    else:
        force_spg = state.get("force_spg")
        if force_spg is not None:
            predicted_spgs = [int(force_spg)]
            logger.info(f"--spg override: using spg={force_spg} (ignoring filename SPG={state['spg']}).")
            state["spg"] = int(force_spg)
        else:
            predicted_spgs = [state["spg"]]
        state["spg_predictions"] = predicted_spgs
        logger.info(f"No SPG inference. Using provided SPG: {predicted_spgs[0]}")

    spg_rank = {pred_spg: idx for idx, pred_spg in enumerate(state["spg_predictions"], start=1)}

    best_trial_state = None
    best_trial_message = None
    best_trial_score = -1e9
    best_accepted_trial_state = None
    best_accepted_trial_message = None
    best_accepted_trial_score = -1e9
    global_accepted_exists = False  # tracks whether any pair has produced an accepted solution

    # key = (seed_spg, dims_sig) — same dims under different SPGs kept separately
    any_seed_had_cells = False
    attempted_cell_keys: set = set()
    global_structure_log: list = []
    max_trials = state.get("max_trials")
    max_pairs = state.get("max_pairs")

    # ── Phase 1: collect all (cell, spg) pairs from every seed SPG ──────────
    all_seed_cells: list = []  # (volume, cell, seed_spg)
    for seed_rank, seed_spg in enumerate(predicted_spgs, start=1):
        seed_state = copy.deepcopy(state)
        seed_state["spg"] = seed_spg
        run_cell_solver(seed_state)
        seed_cells = seed_state.get("cells") or []

        if not seed_cells:
            logger.info(f"Phase 1 — rank {seed_rank:2d}/{len(predicted_spgs)}: spg={seed_spg} | No candidate cells found.")
            continue

        # Show volume range of cells found for this SPG
        volumes = [float(getattr(cell, "size", 0.0)) for cell in seed_cells]
        vol_min, vol_max = min(volumes), max(volumes)
        vol_info = (
            f"vol={vol_min:.1f}–{vol_max:.1f} Å³"
            if vol_min != vol_max
            else f"vol={vol_min:.1f} Å³"
        )
        logger.info(
            f"Phase 1 — rank {seed_rank:2d}/{len(predicted_spgs)}: spg={seed_spg} | Found {len(seed_cells)} cell(s): {vol_info}"
        )

        any_seed_had_cells = True
        for cell in seed_cells:
            dims = tuple(round(float(x), 3) for x in np.array(cell.dims).tolist())
            key = (seed_spg, (len(dims), dims))
            if key in attempted_cell_keys: continue
            #print(f"Adding seed cell for SPG {seed_spg} with dims {cell.dims} to planning queue.")
            attempted_cell_keys.add(key)
            all_seed_cells.append((float(getattr(cell, "size", 0.0)), cell, seed_spg))

    if all_seed_cells:
        # ── Phase 2: plan ALL (cell, spg) pairs with explicit cost estimates ─
        # For each (cell, spg) pair, estimate cost by Wyckoff candidate count and
        # estimated number of generated trials. Globally rank every pair by a balanced
        # score combining relative estimated trials and relative volume.
        total_count = 0
        planned_pairs = []
        total_est_trials = 0
        num_cells = len(all_seed_cells)

        for _vol, _cell, _spg in all_seed_cells:
            cand_count, est_wps, est_trials = _estimate_pair_cost(_cell, _spg,
                    max_wp, max_Z, max_dof, per_dof, csv_path, max_samples=max_samples)
            if cand_count == 0: continue
            total_count += cand_count
            total_est_trials += est_trials
            planned_pairs.append(
                {"vol": _vol,
                 "cell": _cell,
                 "spg": _spg,
                 "cand_count": cand_count,
                 "est_wps": est_wps,
                 "est_trials": est_trials}
            )
            if total_est_trials > max_trials and \
                len(planned_pairs) >= min([int(num_cells * 0.2), max_pairs]):
                logger.info(
                    f"Reached maximum total estimated trials ({total_est_trials} > {max_trials}) "
                    f"after planning {len(planned_pairs)} (cell, SPG) pairs. Stopping further planning."
                )
                break


        if len(planned_pairs) == 0:
            logger.info("No viable (cell, SPG) pairs found; cannot proceed to structure generation.")
            return state

        min_pair_trials = min(int(m["est_trials"]) for m in planned_pairs)
        min_pair_volume = min(float(m["vol"]) for m in planned_pairs)
        max_missing = max(m["cell"].missing for m in planned_pairs)
        for member in planned_pairs:
            member["balance_score"] = get_pair_priority(
                member["est_trials"],
                member["vol"],
                member["cell"].missing,
                member["cell"].chi2,
                min_pair_trials,
                min_pair_volume,
                max_missing,
            )
        planned_pairs.sort(
            key=lambda m: (
                m["balance_score"],
                m["est_trials"],
                round(m["vol"], 1),
                m["cand_count"],
                -CRYSTAL_SYSTEM_PRIORITY.get(
                    (spg_to_crystal_system(m["spg"]) or '').lower(), 0
                ),
                spg_rank.get(m["spg"], 10**9),
                getattr(m["cell"], "missing", 999),
                int(np.round(float(getattr(m["cell"], "chi2", 1e9)) / 5e-4)),
                getattr(m["cell"], "chi2", 1e9),
                -int(m["spg"]),
            )
        )

        all_seed_cells = [(m["vol"], m["cell"], m["spg"]) for m in planned_pairs]
        if len(all_seed_cells) == 0:
            logger.info("No inferred SG candidate produced valid cells.")
            state["msg"] = "No valid cells are found."
            return state
        volumes = [float(item[0]) for item in all_seed_cells]
        vol_lo, vol_hi = min(volumes), max(volumes)
        logger.info(
            f"Phase 2: ranked {len(planned_pairs)} (cell, SPG) pair(s) from {len(predicted_spgs)} seed SPG candidates. "
            f"Volume range: {vol_lo:.1f}–{vol_hi:.1f} Å³"
            f" | Total candidates: {total_count}"
        )
        logger.info(
            "Phase 2 strategy: globally rank every (cell, SPG) pair by a balanced "
            "score combining relative estimated trials and relative volume "
            "(trial_weight=0.65, volume_weight=0.35), then break ties by "
            "(fewer estimated trials, smaller volume, fewer candidates), "
            "then (symmetry, SG prediction rank, missing, chi2)."
        )

        # ── Phase 2 summary table ────────────────────────────────────────────
        logger.info(
            f"\n{'Pair':<5} {'SPG':<5} {'Cell':<33}{'#WPs':<5}{'Volume(Å³)':<12}  {'Chi2':<8} {'N_m':<8}{'N_t':<6} {'Priority Score'}"
        )
        logger.info("-" * 104)
        for _ri, _pair in enumerate(planned_pairs, start=1):
            _vol = _pair["vol"]
            _cell = _pair["cell"]
            _spg = _pair["spg"]
            _est_wps = _pair["est_wps"]
            _est_trials = _pair["est_trials"]
            _balance_score = float(_pair.get("balance_score", float("nan")))
            _dims_str = "  ".join(f"{float(x):6.3f}" for x in _cell.dims)
            logger.info(
                f"{_ri:<5} {_spg:<5} [{_dims_str:<30}] {_est_wps:<5}  {_vol:<11.1f}"
                f"{_cell.chi2:<6.4f} {_cell.missing:5}"
                f"{_est_trials:8} {_balance_score:9.3f}"
            )
        logger.info("")

    list_wp_only = bool(state.get("list_wp_only", False))
    if list_wp_only: logger.info("List Wyckoff candidates and skip structure generation.")

    # ── Phase 3: systematic structure generation across all ranked (cell, spg) pairs ──
    # Each entry is already a specific (cell, spg) pairing — enumerate Wyckoff
    # only for that SPG to avoid redundant work across identical cell dims.
    spg_cell_end_time = time.perf_counter()
    structure_start_time = spg_cell_end_time
    terminate_pair = False
    stop_after_pair = False  # finish current pair's WPs, then stop
    structure_limit = state['min_structures_before_early_stop']
    max_wp_choices = state.get("max_wp_choices")
    N_cells = len(all_seed_cells)
    max_eng = max(0.0, state.get("max_eng"))
    disable_early_termination = bool(state.get("disable_early_termination", False))

    for it in range(3):
        # Just in case some strctures failed very frequently
        if len(global_structure_log) >= structure_limit and terminate_pair:
            logger.info(f"Reached maximum structure limit and Stop further generation.")
            break
        if it > 0 and len(global_structure_log) >= structure_limit:
            logger.info(f"More than one attempt reached and Stop further generation.")
            break

        for rank_idx, (vol, cell, seed_spg) in enumerate(all_seed_cells, start=1):
            #if rank_idx <= 3: continue  # skip the top 3 pairs in the first iteration to allow quick initial results and early exit if good solutions are found
            consolidated_wp = _get_wp_candidates_for_pair(cell, seed_spg, max_wp, max_Z, max_dof)
            N_wps = len(consolidated_wp)
            if not consolidated_wp: continue

            pair_str = f"Pair {rank_idx}/{N_cells}"
            dim_str = "  ".join(f"{float(x):8.3f}" for x in cell.dims)
            vol_str = f"vol={vol:.1f} Å³"
            logger.info(f"\n[{pair_str}] {vol_str}, spg={seed_spg}, dims={dim_str}: {N_wps} WP choices.")

            wp_limits = get_adaptive_wp_limits(len(consolidated_wp), max_wp_choices)
            prev_limit = 0
            wp_attempted = 0
            trial_state = copy.deepcopy(state)
            trial_state["best_score"] = -1e9
            trial_state["best_result"] = None

            for limit in wp_limits:
                if wp_attempted >= len(consolidated_wp): break

                for sol in consolidated_wp[prev_limit:limit]:
                    # Enforce early-stop structure budget immediately, not only
                    # at the outer-loop boundary, to avoid overshooting by 1+ WPs.
                    if len(global_structure_log) >= structure_limit and terminate_pair:
                        logger.info(
                            f"Reached max structure limit ({structure_limit}) while exploring pair "
                            f"{rank_idx}/{len(all_seed_cells)}; stopping further generation."
                        )
                        break

                    spg_val, _comp, _lat, wp_ids, num_wps, dof, _, count, Z, orig_spg = sol
                    wp_attempted += 1

                    wp_labels_text = format_wyckoff_labels(spg_val, wp_ids)
                    logger.info(
                        f"WP #{wp_attempted}: spg={spg_val}, count={count}, dof={dof}, "
                        f"n_wps={num_wps}, wyckoff={wp_labels_text}"
                    )

                    if list_wp_only: continue

                    trial_state["spg"] = spg_val
                    trial_state["cells"] = copy.deepcopy([cell])
                    trial_state['wp_labels'] = wp_labels_text
                    trial_state["forced_wp_solution"] = sol[:8] if len(sol) >= 9 else sol

                    trial_message, trial_state = run_wyckoff_solver(
                        trial_state, global_structure_log,
                        global_accepted=global_accepted_exists,
                        factor = 1 if len(all_seed_cells) > 5 else 3, #if len(all_seed_cells) > 2 else 10
                    )

                    # After running, update the main state's Struc_count by accumulating
                    state["Struc_count"] = trial_state.get("Struc_count")
                    state["attempt_count"] = trial_state["attempt_count"]
                    trial_result = trial_state.get("wyckoff_result") or {}
                    if state["attempt_count"] >= state['max_attempt_count']:
                        logger.info(f"Reached max attempt ({state['max_attempt_count']}); stopping further generation.")
                        terminate_pair = True
                        break
                    if state["Struc_count"] >= state['max_relax_count']:
                        logger.info(f"Reached max relaxation ({state['max_relax_count']}); stopping further generation.")
                        terminate_pair = True
                        break
                    #print("++++++++++++++++++ Debug attempt_count:", state["attempt_count"], "Struc_count:", state["Struc_count"])
                    trial_score = trial_result.get("score")
                    if trial_score is not None and trial_score > best_trial_score:
                        best_trial_score = trial_score
                        best_trial_state = trial_state
                        best_trial_message = trial_message

                    r2_val, chi2_val = trial_result.get("r2"), trial_result.get("chi2")
                    _min_r2 = float(state.get("min_r2") or 0.9)
                    # Matches the solver-level is_excellent_refinement threshold
                    excellent_r2 = (r2_val is not None and r2_val >= max(_min_r2 + 0.03, 0.98))

                    if trial_result.get("accepted"):
                        global_accepted_exists = True
                        trial_score_for_accepted = trial_score if trial_score is not None else -1e9
                        if trial_score_for_accepted > best_accepted_trial_score:
                            best_accepted_trial_score = trial_score_for_accepted
                            best_accepted_trial_state = trial_state
                            best_accepted_trial_message = trial_message
                        _emit_accepted_solution(spg_val, trial_result)
                        # For inferred-SPG early exit, require stricter criteria: R² > 0.93 AND χ² < 0.18
                        strict_early_exit = (
                            r2_val is not None and chi2_val is not None and
                            r2_val >= 0.93 and chi2_val < 0.18
                        )
                        candidate_energy = trial_result.get("selected_energy")
                        global_best_energy = _get_global_best_energy(global_structure_log)
                        global_eng_rel = trial_result.get("eng_rel")
                        if global_eng_rel is None and candidate_energy is not None and global_best_energy is not None:
                            global_eng_rel = max(0.0, float(candidate_energy) - float(global_best_energy))
                        energy_ok = (global_eng_rel is not None and global_eng_rel <= max_eng)
                        enough_structures = len(global_structure_log) >= state["min_structures_before_early_stop"]
                        if strict_early_exit and not disable_early_termination:
                            if not energy_ok:
                                logger.info(
                                    f"Good fit found for spg={spg_val}, but skipping quality-based early stop "
                                    f"because dE_global={global_eng_rel:.4f} exceeds "
                                    f"{max_eng:.4f} eV/atom; search may still stop via "
                                    f"the structure-budget criterion."
                                )
                            else:
                                if not enough_structures:
                                    logger.info(
                                        f"Good fit found for spg={spg_val}, but skipping early stop "
                                        f"because only {len(global_structure_log)} structures have been explored "
                                    )
                                else:
                                    logger.info(
                                        f"Good fit found early: spg={spg_val}, "
                                        f"R2={trial_result.get('r2', 0):.4f}, "
                                        f"Chi2={trial_result.get('chi2', 0):.4f}, "
                                        f"dE_global={global_eng_rel:.4f}. "
                                    f"Stopping search after pair {rank_idx}/{len(all_seed_cells)} "
                                    f"and {wp_attempted} WP candidate(s).")
                                    terminate_pair = True
                                    break
                    elif excellent_r2 and not disable_early_termination:
                        # The WP-level solver fired its early stop (r2 >= 0.98) but the
                        # result is not strictly accepted (chi2 slightly exceeds threshold).
                        # Finish the remaining WPs for this pair, but don't start new pairs.
                        candidate_energy = trial_result.get("selected_energy")
                        global_best_energy = _get_global_best_energy(global_structure_log)
                        global_eng_rel = trial_result.get("eng_rel")
                        if global_eng_rel is None and candidate_energy is not None and global_best_energy is not None:
                            global_eng_rel = max(0.0, float(candidate_energy) - float(global_best_energy))
                        energy_ok = (global_eng_rel is not None and global_eng_rel <= max_eng)
                        enough_structures = len(global_structure_log) >= state["min_structures_before_early_stop"]
                        if energy_ok and enough_structures and not stop_after_pair:
                            stop_after_pair = True
                            logger.info(
                                f"Excellent r2 fit (not strictly accepted: R2={r2_val:.4f}, "
                                f"Chi2={chi2_val:.4f}, dE_global={global_eng_rel:.4f}) for spg={spg_val}. "
                                f"Will finish remaining WPs for pair {rank_idx}, then stop."
                            )

                    # If a globally-accepted solution already exists (from any
                    # previous WP/pair) and the minimum-structures budget is
                    # reached, stop all further exploration.
                    if (
                        not disable_early_termination and
                        len(global_structure_log) >= structure_limit
                        and global_accepted_exists
                    ):
                        logger.info(
                            f"Accepted solution found and structure budget reached "
                            f"({len(global_structure_log)}/{structure_limit}); stopping search."
                        )
                        terminate_pair = True
                        break
                if terminate_pair: break
                prev_limit = limit
            if terminate_pair: break
            if stop_after_pair:
                logger.info(
                    f"Stopping after pair {rank_idx}/{len(all_seed_cells)} "
                    f"(excellent r2 fit found; all WPs for this pair have been explored)."
                )
                terminate_pair = True
                break

    state["Total_est"] = total_count
    state["structure_log"] = global_structure_log
    if list_wp_only:
        timing_breakdown = _timing_breakdown_seconds()
        state["timing_breakdown_seconds"] = timing_breakdown
        state["status"] = "WP-Only"
        state["msg"] = "Wyckoff combinations listed only; structure generation skipped."
        logger.info(state["msg"])
        return state

    timing_breakdown = _timing_breakdown_seconds()
    state["timing_breakdown_seconds"] = timing_breakdown
    state["structure_log"] = global_structure_log
    if not any_seed_had_cells:
        logger.info("No inferred SG candidate produced valid cells.")
        state["msg"] = "No valid cells are found."
        return state
    elif best_trial_state is not None:
        final_trial_state = best_accepted_trial_state or best_trial_state
        final_trial_message = best_accepted_trial_message or best_trial_message
        final_trial_score = best_accepted_trial_score if best_accepted_trial_state is not None else best_trial_score
        final_trial_state["WP_qrs_id"] = (
            (final_trial_state.get("wyckoff_result") or {}).get("qrs_id")
            if state.get("use_qrs") else None
        )

        best_trial_result = final_trial_state["best_result"] or {}
        best_trial_spg = final_trial_state["spg"]
        if best_trial_result.get("accepted"):
            status = "Success"
        elif (
            best_trial_result.get("r2") is not None
            and best_trial_result["r2"] >= state.get("min_r2", 0.95)
        ):
            status = "C-Success"
        else:
            status = "Failure"

        # End of all-pairs loop: emit global plot covering every structure tried
        tag = state['pxrd_csv'].split("/")[-1].split(".")[0]
        spg_str = f"SG{best_trial_spg}" if best_trial_spg is not None else "SG_unknown"
        results_dir = state.get("results_dir", "Results")
        output_png = f"{results_dir}/EnergyR2_{tag}_{spg_str}.png"
        final_trial_state['status'] = status
        plot_energy_vs_r2(global_structure_log, final_trial_state, output_png, timing_breakdown)

        if status == "Success":
            # Recompute dE against the global energy minimum from ALL explored
            # structures (pairs explored after the accepted solution may have
            # lower energies, making the earlier dE=0.000 stale).
            final_global_best_energy = _get_global_best_energy(global_structure_log)
            if final_global_best_energy is not None and best_trial_result.get("selected_energy") is not None:
                best_trial_result["eng_rel"] = max(
                    0.0, float(best_trial_result["selected_energy"]) - float(final_global_best_energy)
                )
            _emit_accepted_solution(best_trial_spg, best_trial_result,
                                    prefix="Best solution")
            logger.info(f"Return best result in spg={best_trial_spg}.")
        else:
            logger.info(f"No acceptance; return best result in spg={best_trial_spg}")
        logger.info(f"Best inferred-SG score observed: {final_trial_score:.4f}")

        # Keep the actual attempt count
        count0 = state.get("attempt_count")
        state.update(final_trial_state)
        state["attempt_count"] = count0
        # Keep the global WP-trial count from this pipeline run.
        state["status"] = status
        state["msg"] = final_trial_message
        return state
    return state

def _get_system_run_log_path(state: dict) -> str:
    pxrd_csv = str(state.get("pxrd_csv") or "")
    stem = Path(pxrd_csv).stem if pxrd_csv else "unknown_system"
    log_name = f"RunLog_{_safe_name_token(stem)}.log"
    results_dir = state.get("results_dir", "Results")
    return str(Path(results_dir) / log_name)

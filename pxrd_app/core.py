import logging
import os
import random
import time
import copy
from pathlib import Path
from uuid import uuid4
import pandas as pd
import numpy as np

from pxrd_app.constants import VALID_LATTICE_SYMMETRIES, CRYSTAL_SYSTEM_PRIORITY
from pxrd_app.inference import infer_formula_spg, infer_spg_from_backend, spg_to_crystal_system
from pxrd_app.plot import plot_energy_vs_r2
from tools.utils import parse_formula, get_volume_from_density, format_wyckoff_labels
from tools.manager import RawDataManager, CellManager
from tools.density import predict_density_ensemble
from tools.solver import CellSolver, search_solution, enumerate_wyckoff, get_adaptive_wp_limits

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
        "min_abc": state.get("min_abc", 2.0),
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
    max_cell_volume = state["max_cell_volume"]
    spg_infer_backend = state["spg_infer_backend"].strip().lower()

    spg = int(spg_from_filename) if spg_from_filename is not None else 0
    composition = parse_formula(formula)
    min_abc = state["min_abc"]
    wavelength = state["wavelength"]
    density = predict_density_ensemble(formula, sigma=2.5)
    density_min, density_max = density['min'], density['max']

    df = pd.read_csv(pxrd_csv, comment='#')
    x1, y1 = df.iloc[:, 0].values, df.iloc[:, 1].values
    data = RawDataManager(x1, y1, bg_subtract=False)
    data.get_peaks_from_scipy()
    data.filter_peaks_by_ml(threshold=0.8, min_height=3.0)
    peaks = data.peaks
    peak_positions = x1[peaks]

    if infer_spg:
        result = infer_spg_from_backend(
            x1=np.array(x1, dtype=float),
            y1=np.array(y1, dtype=float),
            peak_positions=np.array(peak_positions, dtype=float),
            formula=formula,
            spg_infer_backend=spg_infer_backend,
            spg_top_k=spg_top_k,
            max_cell_volume=max_cell_volume,
        )
        predictions = result.get("predictions") or []
        state["spg_predictions"] = [spg for spg, _prob in predictions]
        if result.get("source"):
            state["spg_prediction_source"] = result["source"]
        if result.get("smart_cell_raw_solutions_by_spg"):
            state["smart_cell_raw_solutions_by_spg"] = result["smart_cell_raw_solutions_by_spg"]
        if result.get("smart_cell_ranked_spg_cells"):
            state["smart_cell_ranked_spg_cells"] = result["smart_cell_ranked_spg_cells"]

    min_volume = float(get_volume_from_density(composition, max(density_max, 1e-6)))

    result = {
        "infer_spg_from_pxrd": infer_spg,
        "spg": spg,
        "formula": formula,
        "x1": x1.tolist(),
        "y1": y1.tolist(),
        "peaks": peaks.tolist(),
        "peak_positions": peak_positions.tolist(),
        "composition": composition,
        "density_min": density_min,
        "density_max": density_max,
        "min_volume": min_volume,
        "max_cell_volume": max_cell_volume,
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

    if state['infer_spg_from_pxrd']:
        smart_raw_by_spg = state["smart_cell_raw_solutions_by_spg"]
        if spg in smart_raw_by_spg:
            raw_solutions = smart_raw_by_spg[spg]
            if raw_solutions:
                cells = CellManager.consolidate(raw_solutions, max_solutions=max_cells, merge_tol=0.05)
                cells, removed_by_volume = _filter_cells_by_max_volume(cells)
                state["cells"] = cells
                if not cells:
                    text = (
                        f"Cell solving found no valid unit cells for formula {formula} in space group {spg} "
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
            max_chi2=cell_solver_kwargs["max_chi2"],
            max_guess=cell_solver_kwargs["max_guess"],
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


def run_wyckoff_solver(state: dict, all_structure_log: list, structure_id_counter=None) -> str:
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
    max_local_perturbations = max(0, int(state.get("max_local_perturbations", 2)))
    perturb_displacement = max(0.0, float(state.get("perturb_displacement", 0.06)))
    max_eng_rel_early_stop = state.get("max_eng_rel_early_stop", state.get("max_eng_rel", None))
    min_structures_before_early_stop = max(0, int(state.get("min_structures_before_early_stop", 10)))
    sim_max = state.get("sim_max", 0.90)
    eng_min = 1e10
    max_wp = state.get("max_wp")
    max_Z = state.get("max_Z")
    max_dof = state.get("max_dof")

    results_dir = state.get("results_dir", "Results")
    cifs_dir = os.path.join(results_dir, "cifs")
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(cifs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    tmp_root = Path("tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    run_token = f"{_safe_name_token(Path(str(pxrd_csv or '')).stem)}_{os.getpid()}_{uuid4().hex[:8]}"
    run_tmp_dir = tmp_root / f"run_{run_token}"
    run_tmp_dir.mkdir(parents=True, exist_ok=True)

    title = f'{formula} PXRD Prediction: Space Group {spg}'
    match_cif = os.path.join(cifs_dir, f'Match_{formula}_{spg}.cif')
    stale_result_cifs = [
        *Path(cifs_dir).glob(f"Match_{formula}_{spg}_attempt*.cif"),
        *Path(cifs_dir).glob(f"Match_{formula}_{spg}_attempt*_refined.cif"),
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
    N1 = state.get("N1")
    N2 = state.get("N2")

    for attempt_idx in range(attempts):
        seed = seed_base + 9973 * attempt_idx
        _set_seed(seed)

        attempt_prefix = run_tmp_dir / f"Match_{formula}_{spg}_attempt{attempt_idx + 1}"
        attempt_png = str(attempt_prefix.with_suffix(".png"))
        attempt_cif = str(attempt_prefix.with_suffix(".cif"))
        attempt_refinement_png = str(attempt_prefix.with_name(f"{attempt_prefix.name}_refinement.png"))
        logger.info(
            f"Attempt {attempt_idx + 1}/{attempts}: seed={seed}, (N1={N1}, N2={N2}), "
            f"Perturb: {max_local_perturbations}/{perturb_displacement:.3f}, "
            f"{struc_count} structures")

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
            N2,
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
            min_r2,
            max_chi2,
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

        if score > state.get('best_score', -1e9):
            state['best_score'] = score
            state['best_result'] = candidate

        # Early stop: excellent solution found
        if state['best_result']["accepted"] and len(all_structure_log) >= min_structures_before_early_stop and (
            state['best_result']["r2"] >= max(min_r2 + 0.02, 0.97)
            or state['best_result']["chi2"] <= min(max_chi2 * 0.7, 0.08)
        ):
            logger.info(f"Early stop: excellent solution found at attempt {attempt_idx + 1}.")
            break

    state["structure_log"] = all_structure_log
    state["Struc_count"] = struc_count

    if state['best_result'] is None: #and len(all_structure_log) >= min_structures_before_early_stop:
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
        return text, state

    state['best_result']["xtal"].to_file(match_cif)
    wr = state['best_result']["wr"]
    r2 = state['best_result']["r2"]
    chi2 = state['best_result']["chi2"]
    logger.info(f"\nFinal refinement results: Wr={wr:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}")
    if os.path.exists(state['best_result']["png"]):
        logger.info(f"Best refinement plot saved to {state['best_result']['png']}")
    logger.info(f"Best structure saved to {match_cif}")
    state['best_result']["spg"] = spg
    state['best_result']["score"] = state['best_score']
    state['best_result']["Struc_count"] = struc_count
    state["wyckoff_result"] = state['best_result']

    text = f"Wyckoff solving completed for formula {formula} in space group {spg}.\n"
    text += f"Adaptive attempts: {attempts}, seed_base: {seed_base}\n"
    text += f"Best similarity: {sim_max:.3f}, Minimum energy per atom: {state['best_result']['eng_best']:.3f} eV\n"
    text += f"Final Rietveld refinement results: Wr={wr:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}\n"
    text += f"Selected attempt: {state['best_result']['attempt']} \n"
    if not state['best_result']["accepted"]:
        text += "Best candidate did not meet the acceptance thresholds, but was kept as a fallback result.\n"
    if os.path.exists(state['best_result']["png"]):
        text += f"Best refinement plot saved to {state['best_result']['png']}\n"
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

        logger.info(
            f"{prefix} details: spg={spg_value}, Wr={wr_text}, R2={r2_text}, "
            f"Chi2={chi2_text}, score={score_text}, E={energy_text}, dE={eng_rel_text}, "
            f"attempt={attempt_text}, seed={seed_text}"
        )
        logger.info(f"CIF={cif_text}, PNG={png_text}")

    def _validate_reused_cell_for_spg(cell_obj, spg_value: int, peak_positions: np.ndarray):
        """
        QZ: This should be moved to solver....
        """
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

    pipeline_start_time = time.perf_counter()
    spg_cell_end_time: float | None = None
    structure_start_time: float | None = None

    logger.info("Using deterministic pipeline execution.")
    run_data_preprocessor(state["pxrd_csv"], state)

    infer_spg = state["infer_spg_from_pxrd"]
    peak_positions_np = np.array(state.get("peak_positions") or [], dtype=float)
    composition = state["composition"]
    density_min, density_max = state["density_min"], state["density_max"]

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

    def canonical_cell(cell_obj) -> tuple:
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

    def _get_wp_candidates_for_pair(cell_obj, spg_value, max_wp, max_Z, max_dof) -> list:
        key = (canonical_cell(cell_obj), spg_value)
        if key in wp_candidate_cache: return wp_candidate_cache[key]
        candidates = enumerate_wyckoff(
            cell_obj.dims,
            [spg_value],
            composition,
            max_wp, max_Z, max_dof,
            ref_den=(density_min, density_max),
            verbose=True,
        )
        #print(f"Enumerated {len(candidates)} Wyckoff candidates for cell {cell_obj.dims} under SPG {spg_value}.")
        wp_candidate_cache[key] = candidates
        return candidates

    def _estimate_pair_cost(cell_obj, spg_value, max_wp, max_Z, max_dof) -> tuple[int, int]:
        key = (canonical_cell(cell_obj), spg_value)
        #print(wp_candidate_cache)
        #print(f"Estimating cost for cell {cell_obj.dims} under SPG {spg_value} with key {key}.")
        if key in wp_cost_cache: return wp_cost_cache[key]

        candidates = _get_wp_candidates_for_pair(cell_obj, spg_value, max_wp, max_Z, max_dof)
        #print(f"Found {len(candidates)} Wyckoff candidates for cell {cell_obj.dims} under SPG {spg_value}/{max_wp}.")
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

    max_wp = state.get("max_wp")
    max_Z = state.get("max_Z")
    max_dof = state.get("max_dof")

    if infer_spg:
        predicted_spgs = state["spg_predictions"][:state["spg_top_k"]]
        lattice_filter = str(state.get("lattice_symmetry", "auto") or "auto").strip().lower()
        target_system = None
        if lattice_filter == "auto":
            filename_spg = state.get("spg_from_filename")
            target_system = spg_to_crystal_system(int(filename_spg)) if filename_spg is not None else None
        elif lattice_filter in VALID_LATTICE_SYMMETRIES:
            target_system = lattice_filter

        if predicted_spgs and target_system is not None:
            filtered_spgs = [sg for sg in predicted_spgs if spg_to_crystal_system(sg) == target_system]
            logger.info(
                f"Applying lattice symmetry filter: {target_system}. "
                f"Kept {len(filtered_spgs)}/{len(predicted_spgs)} inferred SG candidates."
            )
            predicted_spgs = filtered_spgs
            if len(predicted_spgs) == 0:
                logger.info(f"No SPG candidates after applying lattice symmetry filter '{target_system}'.")
                return state
    else:
        predicted_spgs = [state["spg"]]
        state["spg_predictions"] = predicted_spgs
        logger.info(f"No SPG inference. Using provided SPG: {predicted_spgs[0]}")

    spg_rank = {pred_spg: idx for idx, pred_spg in enumerate(state["spg_predictions"], start=1)}

    best_trial_state = None
    best_trial_message = None
    best_trial_score = -1e9

    # key = (seed_spg, dims_sig) — same dims under different SPGs kept separately
    any_seed_had_cells = False
    attempted_cell_keys: set = set()
    global_structure_log: list = []

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
        # 1. Group permutation-equivalent / near-identical cells into families.
        # 2. For each (cell, spg), estimate cost by Wyckoff candidate count and
        #    estimated number of generated trials.
        # 3. Globally rank every pair by a balanced score that combines
        #    relative estimated trials and relative cell volume.
        grouped_seed_cells: dict[tuple, list[tuple[float, object, int]]] = {}
        for item in all_seed_cells:
            _vol, _cell, _spg = item
            sig = canonical_cell(_cell)
            grouped_seed_cells.setdefault(sig, []).append(item)

        planned_groups = []
        skipped_pairs = []
        for sig, members in grouped_seed_cells.items():
            enriched_members = []
            for _vol, _cell, _spg in members:
                cand_count, est_trials = _estimate_pair_cost(_cell, _spg, max_wp, max_Z, max_dof)
                if cand_count == 0:
                    #print(f"Warning: (cell, SPG) pair with volume {_vol:.1f} Å³ and SPG {_spg} has no valid Wyckoff assignments; skipping.")
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
            if not enriched_members: continue

            enriched_members.sort(
                key=lambda m: (
                    m["est_trials"],
                    round(m["vol"], 1),
                    m["cand_count"],
                    spg_rank.get(m["spg"], 10**9),
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
            best_pred_rank = min(spg_rank.get(m["spg"], 10**9) for m in enriched_members)
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
        if len(planned_groups) == 0:
            logger.info("No viable (cell, SPG) pairs found; cannot proceed to structure generation.")
            return state
        logger.info(f"Planned {len(planned_groups)} cells across {len(predicted_spgs)} seed SPG candidates.")
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
                spg_rank.get(m["spg"], 10**9),
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
        logger.info(
            f"Phase 2: planned {len(all_seed_cells)} (cell, SPG) pair(s) across "
            f"{len(planned_groups)} cell family/families. Volume range: {vol_lo:.1f}–{vol_hi:.1f} Å³"
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
            f"\n{'Rank':<5} {'SPG':<5} {'Volume(Å³)':<11} {'Chi2':<8} {'Missing':<8} {'EstTrials':<10} {'BalScore':<9} Dims"
        )
        logger.info("-" * 104)
        for _ri, _pair in enumerate(planned_pairs, start=1):
            _vol = _pair["vol"]
            _cell = _pair["cell"]
            _spg = _pair["spg"]
            _est_trials = _pair["est_trials"]
            _balance_score = float(_pair.get("balance_score", float("nan")))
            _dims_str = "  ".join(f"{float(x):8.3f}" for x in _cell.dims)
            logger.info(
                f"{_ri:<5} {_spg:<5} {_vol:<11.1f} "
                f"{getattr(_cell, 'chi2', float('nan')):<8.4f} "
                f"{getattr(_cell, 'missing', -1):<8} {_est_trials:<10} {_balance_score:<9.3f} {_dims_str}"
            )
        logger.info("")

        # ─ Summary of skipped pairs ─
        if skipped_pairs:
            logger.info(
                f"Skipped {len(skipped_pairs)} individual (cell, SPG) pair(s) due to zero valid Wyckoff "
                f"position(s) in the given Z range.")


    if len(all_seed_cells) == 0:
        logger.info("No inferred SG candidate produced valid cells.")
        state["msg"] = "No valid cells are found."
        return state

    # ── Phase 3: systematic structure generation across all ranked (cell, spg) pairs ──
    # Each entry is already a specific (cell, spg) pairing — enumerate Wyckoff
    # only for that SPG to avoid redundant work across identical cell dims.
    spg_cell_end_time = time.perf_counter()
    structure_start_time = spg_cell_end_time
    terminate_pair = False

    for it in range(5):
        # Just in case some strctures failed very frequently
        if len(global_structure_log) >= state['min_structures_before_early_stop']:
            logger.info(f"Reached maximum structure limit and Stop further generation.")
            break

        for rank_idx, (vol, cell, seed_spg) in enumerate(all_seed_cells, start=1):
            consolidated_wp = _get_wp_candidates_for_pair(cell, seed_spg, max_wp, max_Z, max_dof)
            if not consolidated_wp: continue

            top_preview = [f"spg={s[0]} count={s[6]} dof={s[5]}" for s in consolidated_wp[:3]]
            logger.info(
                f"\n[Pair {rank_idx}/{len(all_seed_cells)}] vol={vol:.1f} Å³, spg={seed_spg}, dims={[round(float(x), 3) for x in cell.dims]}: {len(consolidated_wp)} WP candidates. Top: {' | '.join(top_preview)}"
            )

            wp_limits = get_adaptive_wp_limits(len(consolidated_wp), 20)
            prev_limit = 0
            wp_attempted = 0
            trial_state = copy.deepcopy(state)
            trial_state["best_score"] = -1e9
            trial_state["best_result"] = None

            for limit in wp_limits:
                if wp_attempted >= len(consolidated_wp): break

                for sol in consolidated_wp[prev_limit:limit]:
                    spg_val, _comp, _lat, wp_ids, num_wps, dof, count, Z, orig_spg = sol
                    wp_attempted += 1

                    passed, _metrics, reject_reason = _validate_reused_cell_for_spg(
                        cell, spg_val, peak_positions_np
                    )
                    if not passed:
                        logger.info(f"\nPair {rank_idx} rejected for spg={spg_val}: {reject_reason}")
                        continue

                    wp_labels_text = format_wyckoff_labels(spg_val, wp_ids)
                    logger.info(
                        f"WP #{wp_attempted}: spg={spg_val}, count={count}, dof={dof}, "
                        f"n_wps={num_wps}, wyckoff={wp_labels_text}"
                    )

                    trial_state["spg"] = spg_val
                    trial_state["cells"] = copy.deepcopy([cell])
                    trial_state['wp_labels'] = wp_labels_text
                    trial_state["forced_wp_solution"] = sol[:8] if len(sol) >= 9 else sol

                    trial_message, trial_state = run_wyckoff_solver(trial_state, global_structure_log)

                    # After running, update the main state's Struc_count by accumulating
                    state["Struc_count"] = trial_state.get("Struc_count")
                    trial_result = trial_state.get("wyckoff_result") or {}

                    trial_score = trial_result.get("score")
                    if trial_score is not None and trial_score > best_trial_score:
                        best_trial_score = trial_score
                        best_trial_state = trial_state
                        best_trial_message = trial_message

                    if trial_result.get("accepted"):
                        _emit_accepted_solution(spg_val, trial_result)
                        # For inferred-SPG early exit, require stricter criteria: R² > 0.93 AND χ² < 0.18
                        r2_val, chi2_val = trial_result.get("r2"), trial_result.get("chi2")
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
                        max_eng_rel_early_stop = max(0.0, float(state.get("max_eng_rel_early_stop") or state.get("max_eng_rel") or 0.20))
                        energy_ok = (global_eng_rel is not None and global_eng_rel <= max_eng_rel_early_stop)
                        enough_structures = len(global_structure_log) >= state["min_structures_before_early_stop"]
                        if strict_early_exit:
                            if not energy_ok:
                                logger.info(
                                    f"Good fit found for spg={spg_val}, but skipping early stop "
                                    f"because dE_global={global_eng_rel:.4f} exceeds "
                                    f"{max_eng_rel_early_stop:.4f} eV/atom."
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
                if terminate_pair: break
                prev_limit = limit
            if terminate_pair: break

    timing_breakdown = _timing_breakdown_seconds()
    if not any_seed_had_cells:
        logger.info("No inferred SG candidate produced valid cells.")
        state["msg"] = "No valid cells are found."
        return state
    elif best_trial_state is not None:
        best_trial_result = best_trial_state["best_result"] or {}
        best_trial_spg = best_trial_state["spg"]
        status = "Success" if best_trial_result.get("accepted") else "Failure"

        # End of all-pairs loop: emit global plot covering every structure tried
        state["timing_breakdown_seconds"] = timing_breakdown
        formula_str = state.get("formula", "unknown")
        results_dir = state.get("results_dir", "Results")
        output_png = f"{results_dir}/EnergyR2_{formula_str}.png"
        best_trial_state['status'] = status
        plot_energy_vs_r2(global_structure_log, best_trial_state, output_png, timing_breakdown)

        if status == "Success":
            _emit_accepted_solution(best_trial_spg, best_trial_result, 
                                    prefix="Best accepted solution")
            logger.info(f"Return best result in spg={best_trial_spg}.")
        else:
            logger.info(f"No acceptance; return best result in spg={best_trial_spg}")
        logger.info(f"Best inferred-SG score observed: {best_trial_score:.4f}")
        state.update(best_trial_state)
        state["status"] = status
        state["msg"] = best_trial_message 
        return state
    # QZ: why do we need this loop?
    #wyckoff_message, _ = run_wyckoff_solver(state, global_structure_log)
    #state["msg"] = wyckoff_message
    return state

def _get_system_run_log_path(state: dict) -> str:
    pxrd_csv = str(state.get("pxrd_csv") or "")
    stem = Path(pxrd_csv).stem if pxrd_csv else "unknown_system"
    log_name = f"RunLog_{_safe_name_token(stem)}.log"
    results_dir = state.get("results_dir", "Results")
    return str(Path(results_dir) / log_name)
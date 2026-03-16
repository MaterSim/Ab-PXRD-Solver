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
import traceback
import copy
from pathlib import Path
from importlib.metadata import version as pkg_version, PackageNotFoundError
import pandas as pd
import numpy as np
from tools.manager import RawDataManager, CellManager
from tools.peak_prediction import predict_peaks, predict_spacegroup
from tools.XRD import Profile
from tools.solver import CellSolver, search_solution
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
    "multi_attempts": _env_int("PXRD_MULTI_ATTEMPTS", 3, min_value=1),
    "seed_base": _env_int("PXRD_SEED_BASE", 20260315),
    "spg_top_k": 5,
    "show_spg_predictions": True,
    "max_local_boosts": _env_int("PXRD_LOCAL_BOOSTS", 1, min_value=0),
    "max_local_perturbations": _env_int("PXRD_LOCAL_PERTURBS", 2, min_value=0),
    "perturb_displacement": float(os.getenv("PXRD_PERTURB_DISPLACEMENT", "0.06")),
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
    spg_top_k = int(state.get("spg_top_k", 5))
    show_spg_predictions = bool(state.get("show_spg_predictions", False))
    spg = int(spg_from_filename) if spg_from_filename is not None else 0
    composition = parse_formula(formula)

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

    min_abc = 2.0
    wavelength = 1.54184
    density = predict_density_ensemble(formula, sigma=2.5)
    density_min = float(density['min'])
    density_max = float(density['max'])
    min_volume = float(get_volume_from_density(composition, density['max']))

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

    state["cells"] = cells
    text = f"Cell solving completed for formula {formula} in space group {spg}.\n"
    return {
        "status": "success",
        "message": text,
        "cells": [{"dimensions": cell.dims, "missing_peaks": cell.missing} for cell in cells],
    }


def _run_wyckoff_solver_stage(state: dict) -> str:
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
    min_r2 = state.get("min_r2")
    max_chi2 = state.get("max_chi2")
    max_force = state.get("max_force")
    max_stress = state.get("max_stress")
    max_local_boosts = max(0, int(state.get("max_local_boosts", 1)))
    max_local_perturbations = max(0, int(state.get("max_local_perturbations", 2)))
    perturb_displacement = max(0.0, float(state.get("perturb_displacement", 0.06)))

    eng_min, sim_max = 1e10, 0.90

    os.makedirs("Results", exist_ok=True)

    title = f'{formula} PXRD Prediction: Space Group {spg}'
    match_png = f"Results/Match_{formula}_{spg}.png"
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

    logger.info(f"Adaptive Wyckoff solve: {attempts} attempt(s), seed_base={seed_base}")
    for attempt_idx in range(attempts):
        seed = seed_base + 9973 * attempt_idx
        N1, N2, N3 = _attempt_schedule(attempt_idx)
        _set_seed(seed)

        attempt_png = f"Results/Match_{formula}_{spg}_attempt{attempt_idx + 1}.png"
        attempt_cif = f"Results/Match_{formula}_{spg}_attempt{attempt_idx + 1}.cif"
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
            "png": attempt_png,
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

        if candidate["accepted"] and (
            candidate["r2"] >= max(min_r2 + 0.02, 0.97)
            or candidate["chi2"] <= min(max_chi2 * 0.7, 0.08)
        ):
            logger.info(f"Early stop: excellent solution found at attempt {attempt_idx + 1}.")
            break

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

    if os.path.exists(best_result["png"]):
        shutil.copy2(best_result["png"], match_png)
    if os.path.exists(best_result["cif"]):
        shutil.copy2(best_result["cif"], match_cif)

    wr = best_result["wr"]
    r2 = best_result["r2"]
    chi2 = best_result["chi2"]
    logger.info(f"\nFinal refinement results: Wr={wr:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}")
    logger.info(f"Best structure saved to {match_cif} and {match_png}")
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
    text += f"Best structure saved to {match_cif} and {match_png}\n"
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
    if announce_bug_switch:
        logger.info("Detected Strands Gemini streaming bug; switching to deterministic fallback pipeline.")
    else:
        logger.info("Using deterministic pipeline execution.")
    _run_data_preprocessor_stage(state["pxrd_csv"], state)

    def _emit_progress(message: str) -> None:
        print(message)

    infer_spg = bool(state.get("infer_spg_from_pxrd", False))
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
        best_trial_state = None
        best_trial_message = None
        best_trial_score = -1e9

        base_cells = None
        base_cell_spg = None

        for rank, candidate_spg in enumerate(predicted_spgs, start=1):
            _emit_progress(
                f"Cell solve seed SG rank {rank}/{len(predicted_spgs)}: trying spg={candidate_spg}"
            )
            seed_state = copy.deepcopy(state)
            seed_state["spg"] = candidate_spg
            _run_cell_solver_stage(seed_state)
            if seed_state.get("cells"):
                base_cells = copy.deepcopy(seed_state["cells"])
                base_cell_spg = candidate_spg
                _emit_progress(
                    f"Cell solving succeeded with spg={candidate_spg}; reusing these cells across SG candidates."
                )
                break

            _emit_progress(
                f"spg={candidate_spg} produced no candidate cells; trying next SG for cell solve."
            )

        if base_cells is None:
            _emit_progress("No inferred SG candidate produced valid cells; falling back to default single-SPG flow.")
        else:
            for rank, candidate_spg in enumerate(predicted_spgs, start=1):
                _emit_progress(
                    f"Trying inferred space group rank {rank}/{len(predicted_spgs)} with reused cells: "
                    f"spg={candidate_spg} (cells from spg={base_cell_spg})"
                )
                trial_state = copy.deepcopy(state)
                trial_state["spg"] = candidate_spg
                trial_state["cells"] = copy.deepcopy(base_cells)

                trial_message = _run_wyckoff_solver_stage(trial_state)
                trial_result = trial_state.get("wyckoff_result") or {}

                if trial_result.get("accepted"):
                    _emit_progress(
                        f"Accepted solution found for inferred spg={candidate_spg}; stopping SG iteration early."
                    )
                    state.update(trial_state)
                    return {
                        "status": status_label,
                        "message": trial_message,
                        "spg": state.get("spg"),
                        "formula": state.get("formula"),
                    }

                trial_score = trial_result.get("score")
                if trial_score is not None and trial_score > best_trial_score:
                    best_trial_score = trial_score
                    best_trial_state = trial_state
                    best_trial_message = trial_message

                _emit_progress(
                    f"Inferred spg={candidate_spg} was fully processed but not accepted; trying next candidate."
                )

            if best_trial_state is not None:
                _emit_progress(
                    f"No inferred space group met acceptance thresholds; returning best fallback result from spg={best_trial_state.get('spg')}."
                )
                state.update(best_trial_state)
                return {
                    "status": status_label,
                    "message": best_trial_message,
                    "spg": state.get("spg"),
                    "formula": state.get("formula"),
                }

    _run_cell_solver_stage(state)
    wyckoff_message = _run_wyckoff_solver_stage(state)
    return {
        "status": status_label,
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
    spg_top_k: int | None = None,
    show_spg_predictions: bool | None = None,
    lattice_symmetry: str | None = None,
    max_local_boosts: int | None = None,
    max_local_perturbations: int | None = None,
    perturb_displacement: float | None = None,
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
    if spg_top_k is not None:
        run_state["spg_top_k"] = int(spg_top_k)
    if show_spg_predictions is not None:
        run_state["show_spg_predictions"] = bool(show_spg_predictions)
    if lattice_symmetry is not None:
        run_state["lattice_symmetry"] = str(lattice_symmetry).strip().lower()
    if max_local_boosts is not None:
        run_state["max_local_boosts"] = max(0, int(max_local_boosts))
    if max_local_perturbations is not None:
        run_state["max_local_perturbations"] = max(0, int(max_local_perturbations))
    if perturb_displacement is not None:
        run_state["perturb_displacement"] = max(0.0, float(perturb_displacement))
    run_prompt = INPUT_PROMPT if input_prompt is None else input_prompt
    force_fallback, _ = _startup_runtime_mode()

    try:
        if force_fallback:
            print("Starting pipeline in fallback mode.")
            result = _run_pipeline_fallback(run_state)
            print("Pipeline completed successfully via fallback execution!")
        else:
            if bool(run_state.get("infer_spg_from_pxrd", False)):
                print("Starting pipeline in graph-consistent deterministic mode.")
                result = _run_pipeline_graph_consistent(run_state)
                print("Pipeline completed successfully via graph-consistent execution!")
            else:
                result = graph(run_prompt,
                               invocation_state=run_state)
                print("Pipeline completed successfully!")
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as exc:
        if _is_strands_gemini_stream_bug(exc):
            print("Strands Gemini streaming error detected; retrying with fallback execution.")
            result = _run_pipeline_fallback(run_state)
            print("Pipeline completed successfully via fallback execution!")
        else:
            raise
    print("Exiting main thread")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PXRD agent pipeline")
    parser.add_argument(
        "--input-csv",
        default="Examples/PXRD_PrYMg2_123.csv",
        help="Path to PXRD CSV file",
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
        "--spg-top-k",
        type=int,
        choices=[3, 5, 10, 20],
        default=5,
        help="Number of inferred space-group options to evaluate/show (3 or 5).",
    )
    parser.add_argument(
        "--symmetry",
        type=str,
        choices=["auto", "any", "triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"],
        default="auto",
        help=(
            "Optional crystal-system filter for inferred SG candidates. "
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
    args = parser.parse_args()

    main(
        pxrd_csv=args.input_csv,
        formula=args.input_formula,
        multi_attempts=args.multi_attempts,
        seed_base=args.seed_base,
        infer_spg_from_pxrd=args.infer_spg,
        spg_top_k=args.spg_top_k,
        lattice_symmetry=args.symmetry,
        max_local_boosts=args.local_boosts,
        max_local_perturbations=args.local_perturbations,
        perturb_displacement=args.perturb_displacement,
        show_spg_predictions=True,
        input_prompt=f"Process the PXRD data from {args.input_csv}",
    )

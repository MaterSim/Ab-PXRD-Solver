from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from pxrd_app.tools.solver import SmartCellSolver
from pxrd_app.constants import DEFAULT_STATE, CRYSTAL_SYSTEM_PRIORITY


def infer_formula_spg(path: str) -> Tuple[Optional[str], Optional[int]]:
    stem = Path(path).stem
    # Try underscore first
    tokens = stem.split("_")
    formula_guess, spg_guess = None, None
    if len(tokens) >= 2 and tokens[-1].isdigit():
        spg_guess = int(tokens[-1])
        formula_guess = "_".join(tokens[1:-1]) if len(tokens) > 2 else tokens[0] if len(tokens) == 2 else None
        return formula_guess, spg_guess
    # Try hyphen as separator
    tokens = stem.split("-")
    if len(tokens) >= 2 and tokens[-1].isdigit():
        spg_guess = int(tokens[-1])
        formula_guess = "-".join(tokens[:-1]) if len(tokens) > 2 else tokens[0] if len(tokens) == 2 else None
        return formula_guess, spg_guess
    return None, None

def spg_to_crystal_system(spg: Optional[int]) -> Optional[str]:
    if spg is None:
        return None
    if 1 <= spg <= 2:
        return "Triclinic"
    if 3 <= spg <= 15:
        return "Monoclinic"
    if 16 <= spg <= 74:
        return "Orthorhombic"
    if 75 <= spg <= 142:
        return "Tetragonal"
    if 143 <= spg <= 167:
        return "Trigonal"
    if 168 <= spg <= 194:
        return "Hexagonal"
    if 195 <= spg <= 230:
        return "Cubic"
    return None


def _smart_solution_metrics(sol: dict) -> Optional[Dict[str, Any]]:
    spg = int(sol.get("spg", 0) or 0)
    if spg <= 0:
        return None

    mismatch = sol.get("mismatch", []) or []
    chi2_raw = sol.get("chi2", (1e9, 1e9))
    if isinstance(chi2_raw, (list, tuple)) and len(chi2_raw) >= 2:
        chi2_val = float(chi2_raw[1])
    else:
        chi2_val = float(chi2_raw if chi2_raw is not None else 1e9)

    cell = sol.get("cell")
    volume = float(getattr(cell, "size", float("inf")))
    support = len(sol.get("match", []) or [])
    return {
        "spg": spg,
        "mismatch": int(len(mismatch)),
        "chi2": chi2_val,
        "volume": volume,
        "support": int(support),
    }


def rank_spg_candidates_from_smart_solutions(solutions: list[dict], top_k: int = 5) -> list[tuple[int, float]]:
    if not solutions:
        return []

    stats: dict[int, dict] = {}
    for sol in solutions:
        metrics = _smart_solution_metrics(sol)
        if metrics is None:
            continue
        spg = int(metrics["spg"])
        rec = stats.setdefault(
            spg,
            {
                "support": 0,
                "best_missing": 10**9,
                "best_chi2": 1e9,
            },
        )
        rec["support"] += 1
        rec["best_missing"] = min(rec["best_missing"], int(metrics["mismatch"]))
        rec["best_chi2"] = min(rec["best_chi2"], float(metrics["chi2"]))

    ordered_spgs = sorted(
        stats.keys(),
        key=lambda sg: (
            -CRYSTAL_SYSTEM_PRIORITY.get(spg_to_crystal_system(sg), 0),
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


def rank_smart_cell_spg_cell_solutions(solutions: list[dict]) -> list[dict]:
    ranked = []
    for sol in solutions:
        metrics = _smart_solution_metrics(sol)
        if metrics is None:
            continue
        ranked.append((metrics, sol))

    ranked.sort(
        key=lambda item: (
            -CRYSTAL_SYSTEM_PRIORITY.get(spg_to_crystal_system(int(item[0]["spg"])), 0),
            int(item[0]["mismatch"]),
            float(item[0]["chi2"]),
            float(item[0]["volume"]),
            -int(item[0]["support"]),
            -int(item[0]["spg"]),
        )
    )
    return [sol for _metrics, sol in ranked]


def _smart_cell_pair_key(spg: int, cell) -> tuple[int, tuple[float, ...]]:
    dims = np.asarray(cell, dtype=float).tolist()
    return int(spg), tuple(round(float(x), 4) for x in dims)


def build_smart_cell_metrics_cache(solutions: list[dict]) -> dict[tuple[int, tuple[float, ...]], dict]:
    metrics_by_pair: dict[tuple[int, tuple[float, ...]], dict] = {}
    for sol in rank_smart_cell_spg_cell_solutions(solutions):
        metrics = _smart_solution_metrics(sol)
        if metrics is None:
            continue
        key = _smart_cell_pair_key(int(metrics["spg"]), sol.get("cell"))
        record = {
            "chi2": float(metrics["chi2"]),
            "missing": int(metrics["mismatch"]),
            "errors": [float(x) for x in (sol.get("errors", []) or [])],
        }
        prev = metrics_by_pair.get(key)
        if prev is None or (record["missing"], record["chi2"]) < (prev["missing"], prev["chi2"]):
            metrics_by_pair[key] = record
    return metrics_by_pair


def build_ranked_smart_cell_solution_cache(solutions: list[dict]) -> dict[int, list[tuple]]:
    raw_by_spg: dict[int, list[tuple]] = {}
    for sol in rank_smart_cell_spg_cell_solutions(solutions):
        metrics = _smart_solution_metrics(sol)
        if metrics is None: continue
        spg_i = int(metrics["spg"])
        raw_tuple = (
            spg_i,
            sol.get("cell"),
            sol.get("mismatch", []) or [],
            float(metrics["chi2"]),
            sol.get("errors", []),
            sol.get("id", ""),
            sol.get("match", []),
        )
        raw_by_spg.setdefault(spg_i, []).append(raw_tuple)
    return raw_by_spg


def infer_spg_from_backend(
    *,
    peak_positions: np.ndarray,
    formula: str,
    spg_top_k: int,
    max_volume: Optional[float],
    crystal_system: Optional[str] = None,
) -> dict:
    result = {
        "predictions": [],
        "source": None,
        "smart_cell_candidates_by_spg": {},
        "smart_cell_metrics_by_pair": {},
        "smart_cell_ranked_spg_cells": [],
    }

    smart_solutions = SmartCellSolver(
        np.array(peak_positions, dtype=float),
        hkl_max=(2, 5, 6),
        max_mismatch=DEFAULT_STATE["cell_solver_max_mismatch"],
        max_chi2=DEFAULT_STATE["cell_solver_max_chi2"],
        max_square=DEFAULT_STATE["cell_solver_max_square"],
        total_square=DEFAULT_STATE["cell_solver_total_square"],
        theta_tols=[0.1, 0.15, 0.5],
        min_abc=DEFAULT_STATE["min_abc"],
        max_abc=DEFAULT_STATE["max_abc"],
        min_volume=20.0,
        max_volume=max_volume,
        verbose=False,
        crystal_system=crystal_system,
    )
    result["smart_cell_ranked_spg_cells"] = rank_smart_cell_spg_cell_solutions(smart_solutions)
    candidate_map = build_ranked_smart_cell_solution_cache(smart_solutions)
    result["smart_cell_candidates_by_spg"] = candidate_map
    result["smart_cell_metrics_by_pair"] = build_smart_cell_metrics_cache(smart_solutions)
    result["predictions"] = rank_spg_candidates_from_smart_solutions(smart_solutions, top_k=spg_top_k)
    result["source"] = "smart_cell_solver"
    return result

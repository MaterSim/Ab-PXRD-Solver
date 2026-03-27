# pxrd_app/constants.py
"""
Constants and default configuration for PXRD agent.
"""
import os

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

DEFAULT_STATE = {
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
    "max_sim": 0.9,
    # Structure generation cap
    # Track total number of generated structures
    "Struc_count": 0,
    "INST_FILE": "tools/INST_XRY.PRM",
    "SCALED_INTENSITY_TOL": 0.01,
    "thetas": [10, 80],
    "resolution": 0.02,
    "max_force": 0.5,
    "max_stress": 0.3,
    "max_cells": 10,
    "max_wp": 10,
    "max_dof": 3,
    "max_Z": 24,
    "max_dof": 10,
    "max_abc": 35.0,
    "min_abc": 2.0,
    "max_cell_volume": None,
    "cell_solver_max_mismatch": 14,
    "cell_solver_hkl_max": (2, 5, 6),
    "cell_solver_max_square": 28,
    "cell_solver_total_square": 40,
    "cell_solver_theta_tols": [0.1, 0.15, 0.5],
    "cell_solver_max_chi2": 0.5,
    "cell_solver_max_guess": 50000,
    "multi_attempts": _env_int("PXRD_MULTI_ATTEMPTS", 1, min_value=1),
    "seed_base": _env_int("PXRD_SEED_BASE", 20260315),
    "spg_top_k": 25,
    "spg_infer_backend": "model",
    "stop_on_first_accepted_inferred_spg": True,
    "show_spg_predictions": True,
    "max_local_boosts": _env_int("PXRD_LOCAL_BOOSTS", 1, min_value=0),
    "max_local_perturbations": _env_int("PXRD_LOCAL_PERTURBS", 2, min_value=0),
    "perturb_displacement": float(os.getenv("PXRD_PERTURB_DISPLACEMENT", "0.06")),
    "max_eng_rel_early_stop": 0.05,
    "max_eng_rel": 0.025,
    "min_structures_before_early_stop": 10,
}

VALID_LATTICE_SYMMETRIES = {
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
}

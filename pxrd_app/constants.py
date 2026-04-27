# pxrd_app/constants.py
"""
Constants and default configuration for PXRD agent.
"""
import os
import re
from typing import Optional

CRYSTAL_SYSTEM_PRIORITY = {
    "cubic": 7,
    "hexagonal": 6,
    "trigonal": 5,
    "tetragonal": 4,
    "orthorhombic": 3,
    "monoclinic": 2,
    "triclinic": 1,
}

# Regular expressions for PXRD_resume parsing
PAIR_TABLE_RE = re.compile(
    r"^\s*(?P<rank>\d+)\s+"
    r"(?P<spg>\d+)\s+"
    r"(?P<volume>-?\d+(?:\.\d+)?)\s+"
    r"(?P<chi2>-?\d+(?:\.\d+)?)\s+"
    r"(?P<missing>\d+)\s+"
    r"(?P<est_trials>\d+)\s+"
    r"(?P<bal_score>-?\d+(?:\.\d+)?)\s+"
    r"(?P<dims>.+?)\s*$"
)
PAIR_HEADER_RE = re.compile(
    r"^\[Pair\s+(?P<pair_index>\d+)/(?P<pair_total>\d+)\]\s+"
    r"vol=(?P<volume>-?\d+(?:\.\d+)?)\s+Å³,\s+"
    r"spg=(?P<spg>\d+),\s+dims=(?P<dims>\[.*?\])"
    r"(?:[:;.,\s]+.*)?$"
)
WP_HEADER_RE = re.compile(
    r"^\s*WP\s+#(?P<wp_index>\d+):\s+"
    r"spg=(?P<spg>\d+),\s+count=(?P<count>\d+),\s+dof=(?P<dof>\d+),\s+"
    r"n_wps=(?P<n_wps>\d+),\s+wyckoff=(?P<wyckoff>.+?)\s*$"
)
TRIAL_LINE_RE = re.compile(r"^\*(?P<body>.*)$")
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _env_int(name: str, default: int, min_value: Optional[int] = None) -> int:
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
    "min_volume": 10.0,
    "min_abc": 2.0,
    "max_abc": 36.0,
    "max_wp": 15,
    "max_Z": 24,
    "max_dof": 25,
    "max_atoms": 200,
    "max_cell_volume": 1000.0,
    "cells": [],
    # Constraints and parameters
    "wavelength": 1.54184,
    "min_r2": 0.95,
    "max_chi2": 0.12,
    "max_sim": 0.9,
    # Structure generation cap
    # Track total number of generated structures
    "Struc_count": 0,
    "INST_FILE": "pxrd_app/tools/INST_XRY.PRM",
    "gsas_refine_timeout": 60,  # per-refinement wall-time limit in seconds
    "gsas_max_calls_per_worker": 30,  # recycle GSAS-II subprocess after N refinements to prevent memory leaks
    "gsas_max_cyc": 10,  # max Levenberg-Marquardt cycles per do_refinements() call
    "gsas_early_exit_wr": 50.0,  # skip atom/Mustrain refinement if wR% exceeds this after profile+cell steps
    "SCALED_INTENSITY_TOL": 0.01,
    "thetas": [10, 80],
    "resolution": 0.02,
    "max_force": 0.5,
    "max_stress": 0.3,
    "max_cells": 10,
    "cell_solver_max_mismatch": 30, # should be scaled with the number of peaks later
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
    "max_local_perturbations": _env_int("PXRD_LOCAL_PERTURBS", 2, min_value=0),
    "perturb_displacement": float(os.getenv("PXRD_PERTURB_DISPLACEMENT", "0.06")),
    "max_eng_rel_early_stop": 0.05,
    "max_eng_rel": 0.025,
    "min_structures_before_early_stop": 10,
    "ase_logfile": None,
    "max_attempt_count": 5000,
    "max_relax_count": 250,  # Limit total relaxations to avoid hang
    "max_enumeration_samples": 1000,  # Limit samples during cost estimation to avoid hang
    "max_trials": 50000000,  # Limit estimated trials to avoid hang
    "max_pairs": 250,  # Limit number of planned (cell, SPG) pairs to avoid hang
    "max_wp_choices": 20,  # Limit number of WP choices per cell to avoid combinatorial explosion
    "wp_csv_path": "database/spg_num_wps_mp.csv",
    "use_qrs": False,  # Whether to use Quasi-Random Sampling for structure generation
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

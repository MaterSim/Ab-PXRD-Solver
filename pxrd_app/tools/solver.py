"""
Module for PXRD indexing and lattice parameter estimation.
"""
import os
import sys
import shutil
import numpy as np
from itertools import combinations
from pyxtal.symmetry import Group, get_bravais_lattice, get_lattice_type, generate_possible_hkls

def _missing_gsas_refine_pxrd(*args, **kwargs):
    raise ModuleNotFoundError(
        "GSAS-II Python module 'GSASIIscriptable' is not available. "
        "Install GSAS-II (e.g., conda-forge gsas2pkg) to enable refinement."
    )

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)
from pxrd_app.tools.manager import RawDataManager, CellManager, WPManager, XtalManager
from pxrd_app.tools.utils import relax_structure
from pxrd_app.tools.XRD import Similarity, XRD
try:
    from pxrd_app.tools.gsas import refine_pxrd
except Exception:
    refine_pxrd = _missing_gsas_refine_pxrd

def get_cell_params(bravais, hkls, two_thetas, wave_length, min_abc, max_abc, min_volume):
    """
    Calculate cell parameters for a given set of hkls.

    Args:
        bravais (int): Bravais lattice type (0-13)
        hkls: list of (h, k, l) tuples
        two_thetas: list of 2theta values
        wave_length: X-ray wavelength
        min_abc: minimum allowed cell parameter value
        max_abc: maximum allowed cell parameter value
        min_volume: minimum allowed cell volume

    Returns:
        cells: list of cell parameters
        hkls_out: filtered list of hkls corresponding to cells
    """
    hkls = np.array(hkls)
    two_thetas = np.array(two_thetas)

    # Convert to d-spacings
    thetas = np.radians(two_thetas / 2)
    d_spacings = wave_length / (2 * np.sin(thetas))

    cells = []
    ltype = get_lattice_type(bravais)
    if ltype == 6:  # cubic, only need a
        h_sq_sum = np.sum(hkls**2, axis=1)
        cells = d_spacings * np.sqrt(h_sq_sum)
        vols = cells ** 3
        mask = (cells < max_abc) & (cells > min_abc) & (vols > min_volume)
        cells = cells[mask]
        cells = np.reshape(cells, [len(cells), 1])
        hkls_out = hkls[mask]

    elif ltype == 5:  #  hexagonal, need a and c
        # need two hkls to determine a and c
        len_solutions = len(hkls) // 2
        ds = (2 * np.sin(thetas) / wave_length)**2
        A = np.zeros([len(hkls), 2])
        A[:, 0] = 4/3 * (hkls[:, 0] ** 2 + hkls[:, 0] * hkls[:, 1] + hkls[:, 1] ** 2)
        A[:, 1] = hkls[:, 2] ** 2
        B = np.reshape(ds, [len_solutions, 2])
        A = np.reshape(A, [len_solutions, 2, 2])#; print(A.shape, B.shape)
        xs = np.linalg.solve(A, B)#; print(xs); import sys; sys.exit()
        mask1 = np.all(xs[:, :] > 0, axis=1)
        hkls_out = np.reshape(hkls, (len_solutions, 6))
        hkls_out = hkls_out[mask1]
        xs = xs[mask1]
        cells = np.sqrt(1/xs)
        vols = (np.sqrt(3)/2) * cells[:, 0]**2 * cells[:, 1]
        mask2 = (cells[:, 0] < max_abc) & (cells[:, 0] > min_abc) & (vols > min_volume)
        cells = cells[mask2]
        hkls_out = hkls_out[mask2]

    elif ltype == 4:  # tetragonal, need a and c
        # need two hkls to determine a and c
        len_solutions = len(hkls) // 2
        ds = (2 * np.sin(thetas) / wave_length)**2
        A = np.zeros([len(hkls), 2])
        A[:, 0] = hkls[:, 0] ** 2 + hkls[:, 1] ** 2
        A[:, 1] = hkls[:, 2] ** 2
        B = np.reshape(ds, [len_solutions, 2])
        A = np.reshape(A, [len_solutions, 2, 2])#; print(A.shape, B.shape)
        xs = np.linalg.solve(A, B)#; print(xs); import sys; sys.exit()
        mask1 = np.all(xs[:, :] > 0, axis=1)
        hkls_out = np.reshape(hkls, (len_solutions, 6))
        hkls_out = hkls_out[mask1]
        xs = xs[mask1]
        cells = np.sqrt(1/xs)
        vols = cells[:, 0]**2 * cells[:, 1]
        mask2 = np.all((cells[:, :2] < max_abc) & (cells[:, :2] > min_abc), axis=1) & (vols > min_volume)
        cells = cells[mask2]
        hkls_out = hkls_out[mask2]

    elif ltype == 3:  # orthorhombic, need a, b, c
        # need three hkls to determine a, b, c
        len_solutions = len(hkls) // 3
        ds = (2 * np.sin(thetas) / wave_length)**2
        A = np.zeros([len(hkls), 3])
        A[:, 0] = hkls[:, 0] ** 2
        A[:, 1] = hkls[:, 1] ** 2
        A[:, 2] = hkls[:, 2] ** 2
        B = np.reshape(ds, [len_solutions, 3])
        A = np.reshape(A, [len_solutions, 3, 3])
        xs = np.linalg.solve(A, B)#; print(xs); import sys; sys.exit()
        mask1 = np.all(xs[:, :] > 0, axis=1)
        hkls_out = np.reshape(hkls, (len_solutions, 9))
        hkls_out = hkls_out[mask1]
        xs = xs[mask1]
        cells = np.sqrt(1/xs)
        vols = cells[:, 0] * cells[:, 1] * cells[:, 2]
        mask2 = np.all((cells[:, :3] < max_abc) & (cells[:, :3] > min_abc), axis=1) & (vols > min_volume)
        cells = cells[mask2]
        hkls_out = hkls_out[mask2]

    elif ltype == 2:  # monoclinic, need a, b, c, beta
        # need four hkls to determine a, b, c, beta
        len_solutions = len(hkls) // 4
        thetas = np.radians(two_thetas/2)
        ds = (2 * np.sin(thetas) / wave_length)**2
        # hkls (4*N, 3) A=> N, 4, 4
        A = np.zeros([len(hkls), 4])
        A[:, 0] = hkls[:, 0] ** 2
        A[:, 1] = hkls[:, 1] ** 2
        A[:, 2] = hkls[:, 2] ** 2
        A[:, 3] = hkls[:, 0] * hkls[:, 2]
        B = np.reshape(ds, [len_solutions, 4])
        A = np.reshape(A, [len_solutions, 4, 4])#; print(A.shape, B.shape)
        xs = np.linalg.solve(A, B)#; print(xs); import sys; sys.exit()
        mask1 = np.all(xs[:, :3] > 0, axis=1)
        hkls_out = np.reshape(hkls, (len_solutions, 12))#;print(hkls.shape, mask1.shape, A.shape)
        hkls_out = hkls_out[mask1]
        xs = xs[mask1]

        cos_betas = -xs[:, 3] / (2 * np.sqrt(xs[:, 0] * xs[:, 2]))
        masks = np.abs(cos_betas) <= 1/np.sqrt(2)
        xs = xs[masks]
        hkls_out = hkls_out[masks]

        cos_betas = cos_betas[masks]
        sin_betas = np.sqrt(1 - cos_betas ** 2)
        cells = np.zeros([len(xs), 4])
        cells[:, 1] = np.sqrt(1/xs[:, 1])
        cells[:, 3] = np.degrees(np.arccos(cos_betas))
        cells[:, 0] = np.sqrt(1/xs[:, 0]) / sin_betas
        cells[:, 2] = np.sqrt(1/xs[:, 2]) / sin_betas

        # force angle to be less than 90
        #mask = cells[:, 3] > 90.0
        #cells[mask, 3] = 180.0 - cells[mask, 3]
        vols = cells[:, 0] * cells[:, 1] * cells[:, 2] * np.sin(np.radians(cells[:, 3]))
        mask2 = np.all((cells[:, :3] < max_abc) & (cells[:, :3] > min_abc), axis=1) & (vols > min_volume)
        cells = cells[mask2]
        hkls_out = hkls_out[mask2]
    else:
        msg = "Only cubic, tetragonal, hexagonal, and orthorhombic systems are supported."
        raise NotImplementedError(msg)

    return cells, hkls_out

def get_d_hkl_from_cell(bravais, cells, h, k, l):
    """
    Estimate the maximum hkl indices to consider based on the cell parameters and maximum 2theta.

    Args:
        bravais (int): Bravais lattice type (1-15)
        cells: cell parameters
        h: h index
        k: k index
        l: l index

    Returns:
        d: d-spacing values
    """
    ltype = get_lattice_type(bravais)
    if ltype == 6:  # cubic
        d = cells[:, 0] / np.sqrt(h**2 + k**2 + l**2)
    elif ltype == 5:  # hexagonal
        a, c = cells[:, 0], cells[:, 1]
        d = 1 / np.sqrt((4/3) * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)
    elif ltype == 4:  # tetragonal
        a, c = cells[:, 0], cells[:, 1]
        d = 1 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)
    elif ltype == 3:  # orthorhombic
        a, b, c = cells[:, 0], cells[:, 1], cells[:, 2]
        d = 1 / np.sqrt(h**2 / a**2 + k**2 / b**2 + l**2 / c**2)
    elif ltype == 2:  # monoclinic
        a, b, c, beta = cells[:, 0], cells[:, 1], cells[:, 2], np.radians(cells[:, 3])
        sin_beta = np.sin(beta)
        d = 1 / np.sqrt((h**2 / (a**2 * sin_beta**2)) + (k**2 / b**2) + (l**2 / (c**2 * sin_beta**2)) -
                (2 * h * l * np.cos(beta) / (a * c * sin_beta**2)))
    else:
        raise NotImplementedError("triclinic systems are not supported.")
    return d

def get_two_theta_from_cell(bravais, hkls, cell, wave_length=1.54184, check_unique=True):
    """
    Calculate expected 2theta values from hkls and cell parameters.

    Args:
        bravais (int): Bravais lattice type (1-15)
        hkls: hkl indices (np.array)
        cell: cell parameters
        wave_length: X-ray wavelength, default is Cu K-alpha
        check_unique: whether to filter unique 2theta values

    Returns:
        two_thetas: calculated 2theta values
        hkls_out: filtered hkls corresponding to two_thetas
    """
    h, k, l = hkls[:, 0], hkls[:, 1], hkls[:, 2]
    ltype = get_lattice_type(bravais)
    if ltype == 6:  # cubic
        a = cell[0]
        d = a / np.sqrt(h**2 + k**2 + l**2)#; print('ddddd', d)
    elif ltype == 5:  # hexagonal
        a, c = cell[0], cell[1]
        d = 1 / np.sqrt((4/3) * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)
    elif ltype == 4:  # tetragonal
        a, c = cell[0], cell[1]
        d = 1 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)
    elif ltype == 3:  # orthorhombic
        a, b, c = cell[0], cell[1], cell[2]
        d = 1 / np.sqrt(h**2 / a**2 + k**2 / b**2 + l**2 / c**2)
    elif ltype == 2:  # monoclinic
        a, b, c, beta = cell[0], cell[1], cell[2], np.radians(cell[3])
        sin = np.sin(beta)
        cos = np.cos(beta)
        d = 1 / np.sqrt((h**2 / (a**2 * sin**2)) + (k**2 / b**2) + (l**2 / (c**2 * sin**2)) -
                (2 * h * l * cos / (a * c * sin**2)))
    else:
        raise NotImplementedError("triclinic systems are not supported.")

    # Handle cases where sin_theta > 1
    sin_theta = wave_length / (2 * d)
    valid = sin_theta <= 1#; print(d[~valid]); import sys; sys.exit()
    two_thetas = 2 * np.degrees(np.arcsin(sin_theta[valid]))
    two_thetas = np.round(two_thetas, decimals=3)
    if check_unique:
        two_thetas, ids = np.unique(two_thetas, return_index=True)
        return two_thetas, hkls[valid][ids]
    else:
        return two_thetas, hkls[valid]


def _match_obs_exp_peaks(obs_thetas, exp_thetas, tol):
    """
    Fast peak matching without constructing a full N_obs x N_exp error matrix.

    Semantics are aligned with the prior implementation:
    1) Determine which observed peaks have at least one calculated peak within tolerance.
    2) For those observed peaks, assign nearest unused calculated peaks globally
       (not restricted by tolerance), in observed-peak order.
    3) Build calculated-peak "observability" mask based on tolerance windows.

    Returns:
        matched_obs_ids: indices into obs_thetas
        matched_exp_ids: indices into exp_thetas
        errs: obs - exp error for each matched pair in obs order
        used_exp_mask: boolean mask over exp_thetas that are within tolerance
                       of at least one observed peak (for mismatch accounting)
    """
    obs_thetas = np.asarray(obs_thetas)
    exp_thetas = np.asarray(exp_thetas)
    if len(obs_thetas) == 0 or len(exp_thetas) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), np.zeros(len(exp_thetas), dtype=bool)

    sort_ids = np.argsort(exp_thetas, kind='stable')
    exp_sorted = exp_thetas[sort_ids]

    left = np.searchsorted(exp_sorted, obs_thetas - tol, side='left')
    right = np.searchsorted(exp_sorted, obs_thetas + tol, side='right')

    has_obs_match = left < right
    matched_obs_ids = np.where(has_obs_match)[0]

    # Calculated peaks that are close enough to at least one observed peak.
    observable_sorted = np.zeros(len(exp_sorted), dtype=bool)
    for lo, hi in zip(left, right):
        if lo < hi:
            observable_sorted[lo:hi] = True

    used_exp_mask = np.zeros(len(exp_thetas), dtype=bool)
    if len(exp_thetas) > 0:
        used_exp_mask[sort_ids] = observable_sorted

    matched_exp_ids = []
    errs = []
    assigned_exp = np.zeros(len(exp_thetas), dtype=bool)

    for obs_id in matched_obs_ids:
        if not np.any(~assigned_exp):
            continue

        obs_theta = obs_thetas[obs_id]
        errors = np.abs(exp_thetas - obs_theta)
        masked_errors = np.where(assigned_exp, np.inf, errors)
        exp_id = int(np.argmin(masked_errors))
        if not np.isfinite(masked_errors[exp_id]):
            continue

        assigned_exp[exp_id] = True
        matched_exp_ids.append(exp_id)
        errs.append(float(obs_theta - exp_thetas[exp_id]))

    matched_obs_ids = np.asarray(matched_obs_ids[:len(matched_exp_ids)], dtype=int)
    matched_exp_ids = np.asarray(matched_exp_ids, dtype=int)
    errs = np.asarray(errs, dtype=float)
    return matched_obs_ids, matched_exp_ids, errs, used_exp_mask


def _format_large_error_details(obs_thetas, exp_thetas, exp_hkls,
                                matched_obs_ids, matched_exp_ids, errs,
                                theta_tols, max_items=8):
    """
    Build a compact debug string listing matched peak pairs that exceed
    the angle-dependent error tolerances.
    """
    if len(errs) == 0:
        return ""

    obs_thetas = np.asarray(obs_thetas)
    exp_thetas = np.asarray(exp_thetas)
    matched_obs_ids = np.asarray(matched_obs_ids, dtype=int)
    matched_exp_ids = np.asarray(matched_exp_ids, dtype=int)
    errs = np.asarray(errs, dtype=float)

    obs_matched = obs_thetas[matched_obs_ids]
    limits = np.full(len(obs_matched), float(theta_tols[2]), dtype=float)
    limits[obs_matched < 50.0] = float(theta_tols[1])
    limits[obs_matched < 30.0] = float(theta_tols[0])

    bad_ids = np.where(np.abs(errs) > limits)[0]
    if len(bad_ids) == 0:
        return ""

    # Show the largest offenders first.
    order = bad_ids[np.argsort(np.abs(errs[bad_ids]))[::-1]]
    lines = ["Large-error pairs (obs, calc, err, tol, hkl):"]
    for idx in order[:max_items]:
        obs_theta = float(obs_matched[idx])
        exp_id = int(matched_exp_ids[idx])
        exp_theta = float(exp_thetas[exp_id])
        err = float(errs[idx])
        tol = float(limits[idx])
        hkl = tuple(int(v) for v in exp_hkls[exp_id])
        lines.append(
            f"obs={obs_theta:.3f}, calc={exp_theta:.3f}, "
            f"err={err:.3f}, tol={tol:.3f}, hkl={hkl}"
        )
    return "\n".join(lines)


def _format_unmatched_obs_details(obs_thetas, exp_thetas, exp_hkls,
                                  matched_obs_ids, max_items=8):
    """
    Build a compact debug string listing observed peaks that had no
    calculated peak within tolerance, along with their nearest calculated peak.
    """
    obs_thetas = np.asarray(obs_thetas)
    exp_thetas = np.asarray(exp_thetas)
    matched_obs_ids = np.asarray(matched_obs_ids, dtype=int)

    if len(obs_thetas) == 0:
        return ""

    unmatched_obs_ids = np.setdiff1d(np.arange(len(obs_thetas)), matched_obs_ids)
    if len(unmatched_obs_ids) == 0:
        return ""

    if len(exp_thetas) == 0:
        lines = ["Unmatched observed peaks (no calculated peaks available):"]
        for obs_id in unmatched_obs_ids[:max_items]:
            lines.append(f"obs={float(obs_thetas[obs_id]):.3f}")
        return "\n".join(lines)

    nearest_ids = np.array([
        int(np.argmin(np.abs(exp_thetas - obs_thetas[obs_id])))
        for obs_id in unmatched_obs_ids
    ], dtype=int)
    nearest_errs = np.abs(obs_thetas[unmatched_obs_ids] - exp_thetas[nearest_ids])
    order = np.argsort(nearest_errs)[::-1]

    lines = ["Unmatched observed peaks (nearest calc, abs err, hkl):"]
    for idx in order[:max_items]:
        obs_id = int(unmatched_obs_ids[idx])
        exp_id = int(nearest_ids[idx])
        obs_theta = float(obs_thetas[obs_id])
        exp_theta = float(exp_thetas[exp_id])
        err = float(obs_theta - exp_theta)
        hkl = tuple(int(v) for v in exp_hkls[exp_id])
        lines.append(
            f"obs={obs_theta:.3f}, calc={exp_theta:.3f}, "
            f"abs_err={abs(err):.3f}, err={err:.3f}, hkl={hkl}"
        )
    return "\n".join(lines)


def rescue_spg_cell(spg_value, candidate_cell, state):
    """Attempt direct SPG rescue for a candidate cell.

    Args:
        spg_value: Space group number
        candidate_cell: Cell parameters [a, b, c] or permuted variant
        state: Dict containing solver state with keys:
            - direct_solver_cache: Cache of CellSolver instances by SPG
            - direct_validate_cache: Cache of validation results
            - branch_rescue_stats: Local stats dict for branch
            - rescue_stats: Global stats dict
            - thetas: Observed theta values
            - solver_hkl_max: Max hkl indices
            - max_mismatch: Max allowed mismatch
            - max_chi2: Max chi2 threshold
            - solver_max_square: Max hkl^2 sum per axis
            - solver_total_square: Max total hkl^2
            - min_abc: Min cell parameter
            - max_abc: Max cell parameter
            - min_volume: Min cell volume
            - theta_tols: Theta tolerances
            - solver_max_guess: Max guess iterations
            - max_volume: Max volume

    Returns:
        (success, solution) tuple where success is bool and solution is dict or None
    """
    branch_rescue_stats = state['branch_rescue_stats']
    rescue_stats = state['rescue_stats']
    direct_solver_cache = state['direct_solver_cache']
    direct_validate_cache = state['direct_validate_cache']

    branch_rescue_stats["triggered"] += 1
    rescue_stats["triggered"] += 1
    cell_sig = tuple(round(float(x), 4) for x in np.asarray(candidate_cell).tolist())
    cache_key = (int(spg_value), cell_sig)
    if cache_key in direct_validate_cache:
        branch_rescue_stats["cache_hits"] += 1
        rescue_stats["cache_hits"] += 1
        cached_out = direct_validate_cache[cache_key]
        if cached_out[0]:
            branch_rescue_stats["accepted"] += 1
            rescue_stats["accepted"] += 1
        return cached_out

    direct_solver = direct_solver_cache.get(int(spg_value))
    if direct_solver is None:
        direct_solver = CellSolver(
            spg=int(spg_value),
            thetas=state['thetas'],
            hkl_max=state['solver_hkl_max'],
            max_mismatch=state['max_mismatch'],
            max_chi2=state['max_chi2'],
            max_square=state['solver_max_square'],
            total_square=state['solver_total_square'],
            min_abc=state['min_abc'],
            max_abc=state['max_abc'],
            min_volume=state['min_volume'],
            theta_tols=state['theta_tols'],
            max_guess=min(4000, state['solver_max_guess']),
            max_volume=state['max_volume'],
            verbose=False,
        )
        direct_solver_cache[int(spg_value)] = direct_solver

    branch_rescue_stats["solver_checks"] += 1
    rescue_stats["solver_checks"] += 1
    sol_direct, _ = direct_solver.validate_cell(np.array(candidate_cell, dtype=float))
    if sol_direct is None:
        out = (False, None)
    else:
        out = (True, sol_direct)

    if out[0]:
        branch_rescue_stats["accepted"] += 1
        rescue_stats["accepted"] += 1
    direct_validate_cache[cache_key] = out
    return out

class CellSolver:
    def __init__(self, spg, thetas, bra=None, N_add=5, max_mismatch=20,
                 theta_tols=[0.1, 0.15, 0.5], cell_tol=0.25, hkl_max=(2, 3, 10), max_square=20,
                 total_square=50, min_abc=2.0, max_abc=30.0,
                 min_angle=30.0, max_angle=150.0, min_volume=20.0, max_chi2=0.5,
                 N_batch=20, wave_length=1.54184, max_guess=50000, max_volume=None, verbose=False):
        """
        PXRD Cell Solver from the given spg and 2theta values.
        The algorithm is based on generating possible hkl combinations
        and solving the cell parameters using Bragg's law under the space group constraints.
        Only hkl combinations that are allowed in the space group are considered.

        Args:
            spg (int): space group number (1-230)
            thetas: list of 2theta values
            bra: bravais lattice type (optional, can be inferred from spg)
            N_add: number of extra peaks to consider beyond the number of hkls
            max_mismatch: maximum number of mismatched peaks allowed
            theta_tols: list of tolerances for matching 2theta values, default is [0.1, 0.15, 0.5] degrees
            cell_tol: tolerance for considering two cells as the same, default is 0.25
            max_square: maximum square of individual hkl indices
            total_square: maximum total square of hkl indices
            min_abc: minimum a, b, c values
            max_abc: maximum a, b, c values
            min_angle: minimum angle (for monoclinic)
            max_angle: maximum angle (for monoclinic)
            min_volume: minimum cell volume
            max_chi2: maximum chi2 value for peak matching
            hkl_max: maximum h, k, l indices to consider
            N_batch: batch size for processing guesses
            wave_length: X-ray wavelength (default Cu Kα = 1.54184 Å)
            max_guess: maximum number of cell guesses to evaluate
            max_volume: optional maximum unit-cell volume to accept
            verbose: whether to print verbose output
        """
        if bra is None:
            self.spg = spg
            self.bravais = get_bravais_lattice(spg)
            self.group = Group(self.spg)
            self.trial_hkls = np.array(self.group.generate_possible_hkls(50, 50, 50, 2500))
        else:
            self.spg = None
            self.bravais = bra
            self.trial_hkls = generate_possible_hkls(self.bravais, 50, 50, 50, 2500)
        self.thetas = np.asarray(thetas)
        self.theta_count = len(self.thetas)
        self.theta_max = float(self.thetas[-1]) if self.theta_count > 0 else 0.0
        self.trial_hkls_abs = np.abs(self.trial_hkls)
        self.N_add = N_add
        self.max_mismatch = max_mismatch
        self.theta_tols = theta_tols
        self.cell_tol = cell_tol
        self.hkl_max = hkl_max
        self.max_square = max_square
        self.total_square = total_square
        self.min_abc = min_abc
        self.max_abc = max_abc
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_volume = min_volume
        self.max_volume = None if max_volume is None else float(max_volume)
        self.max_chi2 = max_chi2
        self.N_batch = N_batch
        self.wave_length = wave_length
        self.verbose = verbose
        self.lattice_type = get_lattice_type(self.bravais)
        if self.bravais > 3:
            self.max_mismatch_hkl = 3
        else:
            self.max_mismatch_hkl = 2
        self.max_guess = max_guess


    def get_cell_from_multi_hkls(self, hkls, thetas):
        """
        Estimate the cell parameters from multiple (hkl, two_theta) inputs.
        The idea is to use the Bragg's law to estimate the lattice parameters.
        We need to run multiple trials and select the best one.

        Args:
            hkls: list of (h, k, l) tuples
            thetas: list of 2theta values

        Returns:
            solutions: list of solutions
        """
        cells, hkls = get_cell_params(self.bravais, hkls, thetas,
                                      self.wave_length,
                                      self.min_abc,
                                      self.max_abc,
                                      self.min_volume)

        if len(cells) == 0: return []
        # keep cells up to 4 decimal places
        if self.bravais <= 3:
            cells[:, -1] = np.round(cells[:, -1], decimals=2)
            cells[:, :3] = np.round(cells[:, :3], decimals=4)
        elif self.bravais > 12:
            cells = np.round(cells, decimals=5)

        _, unique_ids = np.unique(cells, axis=0, return_index=True)
        hkls = hkls[unique_ids]#; print(cells)  # remove duplicates
        cells = cells[unique_ids]

        # get the maximum h from assuming the cell[-1] is (h00)
        d_100s = get_d_hkl_from_cell(self.bravais, cells, 1, 0, 0)
        d_010s = get_d_hkl_from_cell(self.bravais, cells, 0, 1, 0)
        d_001s = get_d_hkl_from_cell(self.bravais, cells, 0, 0, 1)
        theta_100s = 2*np.degrees(np.arcsin(self.wave_length / (2 * d_100s)))
        theta_010s = 2*np.degrees(np.arcsin(self.wave_length / (2 * d_010s)))
        theta_001s = 2*np.degrees(np.arcsin(self.wave_length / (2 * d_001s)))
        h_maxs = np.array(self.theta_max / theta_100s, dtype=int); h_maxs[h_maxs > 100] = 100
        k_maxs = np.array(self.theta_max / theta_010s, dtype=int); k_maxs[k_maxs > 100] = 100
        l_maxs = np.array(self.theta_max / theta_001s, dtype=int); l_maxs[l_maxs > 100] = 100

        solutions = []
        for i, cell in enumerate(cells):
            if len(cell) == 4 and cell[3] > 90: cell[3] = 180 - cell[3]
            sol, _ = self.validate_cell(cell, self.trial_hkls, hkls[i], h_maxs[i], k_maxs[i], l_maxs[i])
            if sol is not None:
                solutions.append(sol)
        return solutions

    def get_paras_from_cell(self, cell):
        """
        Get the cell parameters in a consistent format.

        Args:
            cell: cell parameters in the format of [a, b, c] or [a, b, c, alpha, beta, gamma]

        Returns:
            paras: cell parameters in the format of [a, b, c, alpha, beta, gamma]
        """
        # get the maximum h from assuming the cell[-1] is (h00)
        cell = np.reshape(cell, [1, -1])
        d_100 = get_d_hkl_from_cell(self.bravais, cell, 1, 0, 0)[0]
        d_010 = get_d_hkl_from_cell(self.bravais, cell, 0, 1, 0)[0]
        d_001 = get_d_hkl_from_cell(self.bravais, cell, 0, 0, 1)[0]
        theta_100 = 2*np.degrees(np.arcsin(self.wave_length / (2 * d_100)))
        theta_010 = 2*np.degrees(np.arcsin(self.wave_length / (2 * d_010)))
        theta_001 = 2*np.degrees(np.arcsin(self.wave_length / (2 * d_001)))
        h_max = self.theta_max // theta_100
        k_max = self.theta_max // theta_010
        l_max = self.theta_max // theta_001
        if self.bravais <= 3:
            h_max += 1; l_max += 1
        return h_max, k_max, l_max

    def validate_cell(self, cell, trial_hkls=None, hkl=None, h_max=None, k_max=None, l_max=None, verbose=False):
        """
        Validate the cell parameters by comparing the expected 2theta values with the observed ones.

        Args:
            cell: cell parameters to validate
            trial_hkls: list of hkls to consider for this cell
            hkl: the original hkl used to generate the cell (for reference)
            h_max, k_max, l_max: maximum h, k, l indices to consider based on the cell parameters
            verbose: whether to print verbose output

        Returns:
            solution: dict containing the cell parameters, (mis)matched peaks and chi2
            remark: any remark or reason for rejection (if applicable)
        """
        #print("Testing cell:", cell)
        cell_str = ", ".join([f"{c:.4f}" for c in cell])
        if trial_hkls is None:
            trial_hkls = self.trial_hkls
            trial_hkls_abs = self.trial_hkls_abs
        elif trial_hkls is self.trial_hkls:
            trial_hkls_abs = self.trial_hkls_abs
        else:
            trial_hkls_abs = np.abs(trial_hkls)
        if h_max is None: h_max, k_max, l_max = self.get_paras_from_cell(cell)
        if cell[:3].min() <= self.min_abc or cell[:3].max() >= self.max_abc:
            return None, f"Rejected cell due to abc parameters out of range: {cell_str}"
        if self.bravais == 3 and (cell[3] < self.min_angle or cell[3] > self.max_angle):
            return None, f"Rejected cell due to angle parameters out of range: {cell_str}"
        if self.max_volume is not None:
            volume = self.get_volume_from_cell(cell)
            if volume > self.max_volume:
                return None, f"Rejected cell due to volume out of range: {cell_str} ({volume:.2f} > {self.max_volume:.2f})"

        mask = (
            (trial_hkls_abs[:, 0] <= h_max)
            & (trial_hkls_abs[:, 1] <= k_max)
            & (trial_hkls_abs[:, 2] <= l_max)
        )
        test_hkls = trial_hkls[mask]
        exp_thetas, exp_hkls = get_two_theta_from_cell(self.bravais, test_hkls, cell, self.wave_length)
        #print("Generated", len(exp_thetas), "peaks for cell", cell)
        if len(exp_thetas) == 0:
            msg = f"Rejected cell due to no expected peaks within the theta range: {cell_str}"
            return None, msg

        tol = self.theta_tols[-1]
        # Fast tolerance check without full error matrix.
        sort_ids = np.argsort(exp_thetas, kind='stable')
        exp_sorted = exp_thetas[sort_ids]
        left = np.searchsorted(exp_sorted, self.thetas - tol, side='left')
        right = np.searchsorted(exp_sorted, self.thetas + tol, side='right')
        has_obs_match = left < right
        ids_matched = np.where(has_obs_match)[0]

        # Compute observability for mismatch accounting.
        observable_sorted = np.zeros(len(exp_sorted), dtype=bool)
        for lo, hi in zip(left, right):
            if lo < hi:
                observable_sorted[lo:hi] = True
        used_exp_mask = np.zeros(len(exp_thetas), dtype=bool)
        used_exp_mask[sort_ids] = observable_sorted

        if len(ids_matched) == self.theta_count:
            #print(f"Perfect match for cell: {cell_str}")
            #print("Observed thetas:", self.thetas)
            #print("Expected thetas:", exp_thetas)
            # Greedy global assignment of matched peaks.
            matched_exp_ids = []
            errs = []
            assigned_exp = np.zeros(len(exp_thetas), dtype=bool)
            for obs_id in ids_matched:
                obs_theta = self.thetas[obs_id]
                errors = np.abs(exp_thetas - obs_theta)
                masked_errors = np.where(assigned_exp, np.inf, errors)
                exp_id = int(np.argmin(masked_errors))
                if np.isfinite(masked_errors[exp_id]):
                    assigned_exp[exp_id] = True
                    matched_exp_ids.append(exp_id)
                    errs.append(float(obs_theta - exp_thetas[exp_id]))

            if len(matched_exp_ids) < self.theta_count:
                return None, f"Rejected cell: {cell_str}, matched {len(matched_exp_ids)}/{self.theta_count} peaks"

            # Get the obs. peaks
            matched_peaks = []
            for obs_id, hkl_id in zip(ids_matched, matched_exp_ids):
                obs_theta = self.thetas[obs_id]
                exp_theta = exp_thetas[hkl_id]
                matched_peaks.append((exp_hkls[hkl_id], exp_theta, obs_theta))
            obs_arr = self.thetas[ids_matched]
            errs = np.asarray(errs, dtype=float)
            # Weighted chi² using theta_tol as uncertainty
            # χ² = Σ[(obs - exp)² / σ²] where σ = theta_tol
            chi2 = np.sum((errs)**2) / (self.theta_tols[-1]**2 * len(obs_arr))
            half = self.theta_count // 2
            chi2_half = np.sum(errs[:half]**2) / (self.theta_tols[-1]**2 * half)
            N_50 = len(self.thetas[self.thetas < 50])
            N_30 = len(self.thetas[self.thetas < 30])
            max_error = np.abs(errs).max()
            max_error_50 = np.abs(errs[:N_50]).max() if N_50 > 0 else 0
            max_error_30 = np.abs(errs[:N_30]).max() if N_30 > 0 else 0
            if max_error_30 > self.theta_tols[0] or \
                max_error_50 > self.theta_tols[1] or \
                max_error > self.theta_tols[2]:
                msg = f"Rejected cell: {cell_str}, error: {max_error_30:.4f} {max_error_50:.4f} {max_error:.4f}"
                return None, msg

            if chi2 > self.max_chi2 or chi2_half > max(chi2, 0.01):
                msg = f"Rejected cell: {cell_str}, chi2: {chi2_half:.4f} {chi2:.4f}"
                return None, msg

            ids_mis_matched = np.where(~used_exp_mask)[0]

            mis_matched_peaks = []
            for id in ids_mis_matched:
                _hkl = exp_hkls[id]
                theta = exp_thetas[id]
                #if theta < min(self.thetas[-1], 50) and abs(_hkl).max() <= self.max_mismatch_hkl:
                if theta < self.theta_max and abs(_hkl).max() <= self.max_mismatch_hkl:
                    mis_matched_peaks.append((_hkl, theta))

            if len(mis_matched_peaks) <= self.max_mismatch: #and chi2_half < chi2:
                solution = {
                    'cell': cell,
                    'match': matched_peaks,
                    'mismatch': mis_matched_peaks,
                    'errors': [max_error_30, max_error_50, max_error],
                    'chi2': [chi2_half, chi2],
                    'id': hkl,
                }
                return solution, None
            else:
                msg = f"Rejected cell: {cell_str}, mismatch {len(mis_matched_peaks)}/{self.max_mismatch}"
        else:
            if verbose:
                details = _format_unmatched_obs_details(
                    self.thetas,
                    exp_thetas,
                    exp_hkls,
                    ids_matched,
                )
                if details:
                    print(details)
            msg = f"Rejected cell: {cell_str}, matched {len(ids_matched)}/{self.theta_count} peaks"
        return None, msg


    def validate_cell_loose(self, cell, trial_hkls=None, hkl=None, h_max=None, k_max=None, l_max=None, verbose=False):
        """
        Validate the cell parameters by comparing the expected 2theta values with the observed ones.
        We ignore the restriction of matching all peaks and allow some mismatches

        Args:
            cell: cell parameters to validate
            trial_hkls: list of hkls to consider for this cell
            hkl: the original hkl used to generate the cell (for reference)
            h_max, k_max, l_max: maximum h, k, l indices to consider based on the cell parameters
            verbose: whether to print verbose output

        Returns:
            solution: dict containing the cell parameters, (mis)matched peaks, and chi2
            remark: any remark or reason for rejection (if applicable)
        """
        #print("Testing cell:", cell)
        cell_str = ", ".join([f"{c:.4f}" for c in cell])
        if trial_hkls is None:
            trial_hkls = self.trial_hkls
            trial_hkls_abs = self.trial_hkls_abs
        elif trial_hkls is self.trial_hkls:
            trial_hkls_abs = self.trial_hkls_abs
        else:
            trial_hkls_abs = np.abs(trial_hkls)
        if h_max is None: h_max, k_max, l_max = self.get_paras_from_cell(cell)

        mask = (
            (trial_hkls_abs[:, 0] <= h_max)
            & (trial_hkls_abs[:, 1] <= k_max)
            & (trial_hkls_abs[:, 2] <= l_max)
        )
        test_hkls = trial_hkls[mask]
        exp_thetas, exp_hkls = get_two_theta_from_cell(self.bravais, test_hkls, cell,
                                                       self.wave_length, False)
        #print("Generated", len(exp_thetas), "peaks for cell", cell)
        msg = ''
        if len(exp_thetas) == 0:
            msg += f"Rejected cell due to no expected peaks: {cell_str}"

        ids_matched, matched_exp_ids, errs, used_exp_mask = _match_obs_exp_peaks(
            self.thetas, exp_thetas, self.theta_tols[-1]
        )

        # Get the obs. peaks
        if len(ids_matched) < len(self.thetas):
             msg += f"\nMatched {len(ids_matched)}/{len(self.thetas)} peaks"
             for id in range(len(self.thetas)):
                 if id not in ids_matched:
                     msg += f"{self.thetas[id]:.2f} "
                     #print(self.thetas[id])
                     #for e, sim, hkl in zip(errors_matrix_raw[id], exp_thetas, exp_hkls):
                     #    if hkl[0] **2 + hkl[1] **2 + hkl[2] **2 == 8:
                     #       print(f"({hkl}, {self.thetas[id]:.2f}, {sim:.2f}, {e:.2f}) ")
                     #import sys; sys.exit()
        matched_peaks = []
        for id, hkl_id in zip(ids_matched, matched_exp_ids):
            obs_theta = self.thetas[id]
            exp_theta = exp_thetas[hkl_id]
            matched_peaks.append((exp_hkls[hkl_id], exp_theta, obs_theta))
        obs_arr = self.thetas[ids_matched]

        # Weighted chi² using theta_tol as uncertainty
        # χ² = Σ[(obs - exp)² / σ²] where σ = theta_tol
        chi2 = np.sum((errs)**2) / (self.theta_tols[-1]**2 * len(obs_arr))
        half = len(self.thetas) // 2
        chi2_half = np.sum(errs[:half]**2) / (self.theta_tols[-1]**2 * half)
        N_50 = len(self.thetas[self.thetas < 50])
        N_30 = len(self.thetas[self.thetas < 30])
        max_error = np.abs(errs).max()
        max_error_50 = np.abs(errs[:N_50]).max() if N_50 > 0 else 0
        max_error_30 = np.abs(errs[:N_30]).max() if N_30 > 0 else 0
        if max_error_30 > self.theta_tols[0] or \
            max_error_50 > self.theta_tols[1] or \
                max_error > self.theta_tols[2]:
            msg = f"\nmax error: {max_error_30:.4f} {max_error_50:.4f} {max_error:.4f}"
            if verbose:
                details = _format_large_error_details(
                    self.thetas,
                    exp_thetas,
                    exp_hkls,
                    ids_matched,
                    matched_exp_ids,
                    errs,
                    self.theta_tols,
                )
                if details:
                    msg += "\n" + details

        if chi2_half > max(chi2, 0.01) or chi2 > self.max_chi2:
            msg += f"\nLarge chi2: {chi2_half:.4f} {chi2:.4f}"

        ids_mis_matched = np.where(~used_exp_mask)[0]

        mis_matched_peaks = []
        for id in ids_mis_matched:
            _hkl = exp_hkls[id]
            theta = exp_thetas[id]
            if theta < self.thetas[-1] and abs(_hkl).max() < 3:
                mis_matched_peaks.append((_hkl, theta))

        if len(mis_matched_peaks) <= self.max_mismatch:
            msg += f"\nLarge mismatches: {len(mis_matched_peaks)}/{self.max_mismatch}"

        solution = {
            'cell': cell,
            'match': matched_peaks,
            'mismatch': mis_matched_peaks,
            'chi2': [chi2_half, chi2],
            'errors': [max_error_30, max_error_50, max_error],
            'id': hkl,
            'bravais': self.bravais,
            'wave_length': self.wave_length,
        }
        return solution, msg

    def refine_cell_parameters(self, cell_init, obs_hkls,
                               max_iterations=100, verbose=False):
        """
        Refine unit cell parameters by minimizing chi² between calculated and observed peak positions.

        Args:
            cell_init: Initial cell parameters [a, b, c] or [a, b, c, α, β, γ]
            obs_hkls: Array of observed Miller indices corresponding to obs_thetas
            max_iterations: Max iterations for optimizer
            verbose: Print refinement progress

        Returns:
            cell_refined: Optimized cell parameters
            chi2_final: Final chi² value
        """
        from scipy.optimize import minimize

        def objective(cell_params, obs_hkls):
            N = len(obs_hkls)
            calc_thetas, _ = get_two_theta_from_cell(self.bravais,
                                                     obs_hkls,
                                                     cell_params,
                                                     self.wave_length,
                                                     False)
            chi2 = np.sum((self.thetas[:N] - calc_thetas)**2) / (self.theta_tols[-1]**2 * len(self.thetas))

            return chi2

        # Optimize
        result = minimize(objective, cell_init, args=(obs_hkls),
                          method='Nelder-Mead',
                          options={'maxiter': max_iterations})

        cell_refined = result.x
        chi2_final = result.fun
        chi2_half = objective(cell_refined, obs_hkls[:len(obs_hkls)//2])

        if verbose:
            chi2_init = objective(cell_init, obs_hkls)
            print(f"Refinement: {cell_init}  {chi2_init:.4f} -> {cell_refined} {chi2_final:.4f}")
        return cell_refined, chi2_final, chi2_half

    def solve(self, max_solutions=20, max_count=100):
        """
        Solve for possible cell parameters based on the provided 2theta values.

        Args:
            max_solutions: Maximum number of unique cell solutions to return.
            max_count: Maximum number of solutions with perfect peak matches (i.e., all observed peaks matched) to consider

        Returns:
            results: list of solution dictionaries containing cell parameters and matched peaks
        """

        h, k, l = self.hkl_max#; print(f"Generating hkl guesses with max indices: h={h}, k={k}, l={l}")
        guesses = self.group.generate_hkl_guesses(h, k, l, max_square=self.max_square,
                                              total_square=self.total_square,
                                              verbose=self.verbose)
        #print(f"Generated {len(guesses)} raw hkl guess sets before filtering.")
        raw_guess_count = len(guesses)
        if self.spg is not None:
            print(f"Generated {raw_guess_count} hkl guess sets for space group {self.spg}.")
        guesses = np.array(guesses)
        if len(guesses) == 0:
            return []

        # For orthorhombic systems, the linear solve in get_cell_params uses
        # A = [[h^2, k^2, l^2], ...] for 3 hkls. If det(A)=0, that guess set
        # is guaranteed singular for any theta assignment and can be removed.
        if self.lattice_type == 3 and guesses.ndim == 3 and guesses.shape[1] == 3:
            coeff = guesses.astype(float) ** 2
            dets = np.linalg.det(coeff)
            valid_mask = np.abs(dets) > 1e-10
            if np.any(~valid_mask):
                #if self.spg is not None:
                #    print(
                #        f"Filtered singular orthorhombic hkl guess sets for space group {self.spg}: "
                #        f"{len(guesses)} -> {int(np.count_nonzero(valid_mask))}."
                #    )
                guesses = guesses[valid_mask]
                if len(guesses) == 0:
                    return []

        if self.verbose: print("Total guesses:", len(guesses))
        sum_squares = np.sum(guesses**2, axis=(1,2))
        if len(guesses) > self.max_guess:
            if self.spg is not None:
                print(
                    f"Truncating hkl guess sets for space group {self.spg}: "
                    f"{len(guesses)} -> {self.max_guess}."
                )
            # Keep only the smallest-sum guesses without fully sorting the tail.
            keep_ids = np.argpartition(sum_squares, self.max_guess - 1)[:self.max_guess]
            guesses = guesses[keep_ids]
            sum_squares = sum_squares[keep_ids]

        sorted_indices = np.argsort(sum_squares)
        guesses = guesses[sorted_indices]

        n_peaks = len(guesses[0])
        N = min([n_peaks + self.N_add, self.theta_count])
        available_peaks = self.thetas[:N]

        peak_combos = np.array(list(combinations(range(N), n_peaks)), dtype=int)
        if len(peak_combos) == 0:
            return []
        N_thetas = len(peak_combos)
        thetas_by_combo = available_peaks[peak_combos]

        results = []
        cell_all = []
        perfect_match_count = 0

        for start in range(0, len(guesses), self.N_batch):
            end = min(start + self.N_batch, len(guesses))
            batch_guesses = guesses[start:end]
            batch_size = len(batch_guesses)
            if batch_size == 0:
                continue

            # Chunk combo expansion to avoid very large temporary arrays.
            target_pair_count = 40000
            combo_chunk = max(1, min(N_thetas, target_pair_count // max(1, batch_size)))

            for combo_start in range(0, N_thetas, combo_chunk):
                combo_end = min(combo_start + combo_chunk, N_thetas)
                combo_thetas = thetas_by_combo[combo_start:combo_end]
                combo_count = len(combo_thetas)

                # Repeat each guess for each peak-combination candidate in this chunk.
                hkls_t = np.repeat(batch_guesses, combo_count, axis=0).reshape(-1, 3)
                thetas = np.tile(combo_thetas.reshape(-1), batch_size)

                sols = self.get_cell_from_multi_hkls(hkls_t, thetas)
                for sol in sols:
                    guess, match, unmatch, chi2 = sol['id'], len(sol['match']), len(sol['mismatch']), sol['chi2'][1]
                    if match == self.theta_count:
                        perfect_match_count += 1
                        cell1 = sol['cell'] #np.sort(np.array(sol['cell']))
                        vol = self.get_volume_from_cell(cell1)
                        d2 = np.sum(guess**2)
                        add = False

                        if len(cell_all) == 0:
                            add = True
                        else:
                            diff2 = np.sum((cell_all - cell1)**2, axis=1)
                            ids = np.where(diff2 < self.cell_tol**2)[0]
                            if len(ids) == 0:
                                add = True
                            else:
                                # keep the one with lower chi2
                                best_existing_chi2 = min(results[i]['chi2'][1] for i in ids)
                                if chi2 < best_existing_chi2:
                                    # remove the old one
                                    for j in sorted(ids, reverse=True):
                                        del results[j]
                                        cell_all = np.delete(cell_all, j, axis=0)
                                    add = True
                                    #if self.verbose:
                                    #    print(f"Found better cell for similar solution, χ²: {chi2:.4f}")

                        if add:
                            results.append(sol)
                            if len(cell_all) == 0:
                                cell_all = np.array([cell1])
                            else:
                                cell_all = np.vstack([cell_all, cell1])

                        if self.verbose:
                            cell1_str = ' '.join([f'{p:6.3f}' for p in cell1]) + f" [{vol:7.1f}]"
                            guess_str = ' '.join([f'{g:2d}' for g in guess])
                            strs = f"{guess_str} [{d2:2d}] {cell1_str} {unmatch:2d}/{self.theta_count:3d} "
                            strs += f"χ²: {chi2:.4f} "
                            strs += ' '.join([f"{e:.3f}" for e in sol['errors']])
                            strs += f" {len(results):3d}/{add}"
                            print(strs)

                        if len(cell_all) >= max_solutions:
                            print(f"Reached maximum number of solutions ({max_solutions}). Stop!")
                            return results
                        if perfect_match_count >= max_count:
                            print(f"Reached maximum number of counts ({max_count}). Stop!")
                            return results

            if self.verbose and (start // self.N_batch) % 100 == 0:
                d2 = (guesses[start]**2).sum()
                print(f"Processed {start}/{len(guesses)} guesses, d2={d2}.")

        return results

    def get_volume_from_cell(self, cell):
        """
        Calculate the volume of the unit cell based on the cell parameters.

        Args:
            cell (list): Cell parameters

        Returns:
            volume (float): Volume of the unit cell
        """
        if self.bravais >= 13:
            volume = cell[0] ** 3
        elif self.bravais >= 11:
            volume = cell[0] ** 2 * cell[1] * np.sqrt(3) / 2
        elif self.bravais >= 9:
            volume = cell[0] ** 2 * cell[1]
        elif self.bravais >= 4:
            volume = cell[0] * cell[1] * cell[2]
        elif self.bravais >= 2:
            beta = np.radians(cell[3])
            volume = cell[0] * cell[1] * cell[2] * np.sin(beta)
        else:
            raise NotImplementedError("Triclinic system not supported.")
        return volume

    def plot_solution(self, solution):
        """
        Plot the calculated vs observed PXRD pattern for a given solution.

        Args:
            solution (dict): Solution dictionary containing cell parameters and matched peaks
        """
        import matplotlib.pyplot as plt

        cell = solution['cell']
        matched_peaks = solution['matched_peaks']
        hkl_list = [m[0] for m in matched_peaks]

        calc_thetas, _ = get_two_theta_from_cell(self.bravais,
                                                 np.array(hkl_list),
                                                 cell,
                                                 self.wave_length,
                                                 False)

        plt.figure(figsize=(8, 5))
        plt.plot(self.thetas, calc_thetas-self.thetas, 'o', alpha=0.3)
        cell_str = ', '.join([f'{p:.4f}' for p in cell])
        chi2_str = f"χ²: {solution['chi2'][0]:.4f} {solution['chi2'][1]:.4f}"
        for i in range(3):
            plt.bar(self.thetas[i], calc_thetas[i]-self.thetas[i],
                    label=f'Peak {i+1}: {hkl_list[i]}', width=0.1, alpha=0.5)
        plt.title(f'Solution Cell: {cell_str}, {chi2_str}')
        plt.xlabel('Observed 2θ (degrees)')
        plt.ylabel('Calculated 2θ (degrees)')
        plt.legend()
        plt.show()

def SmartCellSolver(thetas, hkl_max, max_mismatch, max_chi2=0.1, max_square=28, total_square=25,
                    theta_tols=[0.1, 0.15, 0.5], min_abc=2.0, max_abc=30.0,
                    min_volume=20.0, max_volume=None, verbose=False):
        """
        A smarter version of CellSolver that automatically guess the space group and uses
        more intelligent heuristics to generate hkl guesses.
        1. Check the possible crystal systems using (195, 143, 75, 16, 3) as thresholds for space group numbers.
        2. Analyze possible A/C/F/I centering based on the presence of certain peaks. For example,
            - if (001) is present, it's likely not a C-centered lattice.
            - If (110) is absent but (200) is present, it might indicate a body-centered lattice.
        3. For each Bravis lattice, loop over all space groups to filter incompatible extinction rules.
        4. Output feaisble solutions based volume, chi_2 and number of missing calculated peaks.

        Args:
            thetas: list of observed 2theta values
            max_mismatch: maximum number of mismatched peaks between observed and calculated peaks
            hkl_max: maximum h, k, l indices to consider for generating guesses
            max_chi2: maximum chi² value for peak matching to consider a solution valid
            max_square: maximum square of individual hkl indices to consider when generating guesses
            total_square: maximum total square of hkl indices to consider when generating guesses (i.e
            theta_tol: tolerance for matching 2theta values, default is 0.5 degrees.
            min_abc: minimum lattice parameter to consider for valid solutions
            max_abc: maximum lattice parameter to consider for valid solutions
            min_volume: minimum unit cell volume to consider for valid solutions
            max_volume: maximum unit cell volume to consider for valid solutions
            verbose: whether to print detailed information during the solving process

        Returns:
            results: list of solution dictionaries containing cell parameters, matched peaks, mis-matched peaks,
        """
        bra_list = [
            ('cubic-F', 15, 0, [196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228]),
            ('cubic-I', 14, 0, [197, 199, 204, 206, 211, 214, 217, 220, 229, 230]),
            ('cubic-P', 13, 0, [195, 198, 200, 201, 205, 207, 208, 212, 213, 215, 218, 221, 222, 223, 224]),
            ('hexagonal-P', 11, 0, [168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
                                    178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
                                    188, 189, 190, 191, 192, 193, 194]),
            ('hexagonal-R', 12, 0, [146, 148, 155, 160, 161, 166, 167]),
            ('trigonal-P', 11, 0, [143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159,
                                    162, 163, 164, 165]),
            ('tetragonal-I', 10, 0, [79, 80, 82, 87, 88, 97, 98, 107, 108, 109, 110,
                                     119, 120, 121, 122, 139, 140, 141, 142]),
            ('tetragonal-P', 9, 0, [75, 76, 77, 78, 81, 83, 84, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 99,
                            100, 101, 102, 103, 104, 105, 106, 111, 112, 113, 114, 115, 116, 117, 118,
                            123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138]),
            ('orthorhombic-F', 8, 0, [22, 42, 43, 69, 70]),
            ('orthorhombic-I', 7, 0, [23, 24, 44, 45, 46, 71, 72, 73, 74]),
            ('orthorhombic-C', 6, 0, [21, 20, 35, 36, 37, 63, 64, 65, 66, 67, 68]),
            ('orthorhombic-A', 5, 0, [38, 39, 40, 41]),
            ('orthorhombic-P', 4, 2, [16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 33, 34, 47, 48, 49,
                                      50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]),
            ('monoclinic-C', 3, 4, [5, 8, 9, 12, 15]),
            ('monoclinic-P', 2, 4, [3, 4, 6, 7, 10, 11, 13, 14]),
        ]
        min_mismatch = max_mismatch + 1
        rescue_stats = {
            "triggered": 0,
            "accepted": 0,
            "solver_checks": 0,
            "cache_hits": 0,
        }
        solutions = []
        for _id, (bra_type, bra_index, ideal_mismatch, spgs) in enumerate(bra_list):
            print(f"Trying {bra_type} ...")
            branch_rescue_stats = {
                "triggered": 0,
                "accepted": 0,
                "solver_checks": 0,
                "cache_hits": 0,
                "exceptions": 0,
            }
            solver_hkl_max = hkl_max
            solver_max_square = max_square
            solver_total_square = total_square
            solver_max_guess = 50000
            N_add = 5
            if bra_index in [5, 6, 7, 8]: # orthorhombic
                for id in range(_id+1, len(bra_list)-2):
                    spgs += bra_list[id][-1]

            # Runtime guardrails for low-symmetry branches where hkl-guess
            # combinatorics can explode (e.g., monoclinic-C > 1e6 raw guesses).
            if bra_type.startswith('monoclinic'):
                solver_hkl_max = (
                    min(int(hkl_max[0]), 3),
                    min(int(hkl_max[1]), 4),
                    min(int(hkl_max[2]), 4),
                )
                solver_max_square = min(int(max_square), 20)
                solver_total_square = min(int(total_square), 28)
                if bra_index == 2:
                    N_add, solver_max_guess = 3, 30000
                else:
                    N_add, solver_max_guess = 4, 25000
            elif bra_type.startswith('orthorhombic'):
                solver_max_guess = 25000
            elif bra_index in [9, 10, 11, 12]: # teragonal and hexagonal
                solver_hkl_max = (
                    min(int(hkl_max[0]), 3),
                    min(int(hkl_max[1]), 3),
                    min(int(hkl_max[2]), 10),
                )

            if bra_index in [15, 14, 8, 7]: # orthorhombic
                solver_max_square = max(int(max_square), 35)
                solver_total_square = max(int(total_square), 35)


            # use very strict criteria to get initial cell solutions for each space group,
            # then use those solutions to determine the centering and possible space groups.
            # This way we can significantly reduce the number of space groups we need to check in the later steps,
            # and also increase the chances of finding the correct solution by starting with a more accurate initial guess.
            solver = CellSolver(spg=spgs[0], thetas=thetas, hkl_max=solver_hkl_max,
                                N_add=N_add, max_mismatch=max_mismatch, max_chi2=max_chi2,
                                max_square=solver_max_square, total_square=solver_total_square,
                                min_abc=min_abc, max_abc=max_abc, min_volume=min_volume,
                                theta_tols=theta_tols, max_guess=solver_max_guess, max_volume=max_volume,
                                verbose=verbose)
            base_solutions = solver.solve(max_solutions=25, max_count=100)
            if len(base_solutions) == 0: continue

            count = 0
            direct_solver_cache = {}
            direct_validate_cache = {}

            rescue_state = {
                'direct_solver_cache': direct_solver_cache,
                'direct_validate_cache': direct_validate_cache,
                'branch_rescue_stats': branch_rescue_stats,
                'rescue_stats': rescue_stats,
                'thetas': thetas,
                'solver_hkl_max': solver_hkl_max,
                'max_mismatch': max_mismatch,
                'max_chi2': max_chi2,
                'solver_max_square': solver_max_square,
                'solver_total_square': solver_total_square,
                'min_abc': min_abc,
                'max_abc': max_abc,
                'min_volume': min_volume,
                'theta_tols': theta_tols,
                'solver_max_guess': solver_max_guess,
                'max_volume': max_volume,
            }
            early_stop = False

            for base_solution in base_solutions:
                if bra_index in [4, 7, 8]:
                    axis_orders = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                elif bra_index in [5]: #A-center
                    axis_orders = [(0, 1, 2), (0, 2, 1)]
                elif bra_index in [6]: #C-center
                    axis_orders = [(0, 1, 2), (1, 0, 2)]
                elif bra_index in [2, 3]: #Monoclinic
                    axis_orders = [(0, 1, 2), (2, 1, 0)]
                else:
                    axis_orders = [(0, 1, 2)]

                matched_hkls = [m[0] for m in base_solution['match']]
                base_cell = base_solution['cell']
                base_match = base_solution['match']
                base_mismatch = base_solution['mismatch']

                for axis_order in axis_orders:
                    # Axis permutations are only meaningful for orthorhombic SPGs (16-74).
                    has_orthorhombic_spg = any(15 < spg < 75 for spg in spgs)
                    if has_orthorhombic_spg:
                        permuted_cell = [base_cell[i] for i in axis_order]
                        permuted_match = [
                            (tuple(m[0][i] for i in axis_order), m[1], m[2])
                            for m in base_match
                        ]
                        permuted_mismatch = [
                            (tuple(m[0][i] for i in axis_order), m[1])
                            for m in base_mismatch
                        ]
                    else:
                        permuted_cell = base_cell
                        permuted_match = base_match
                        permuted_mismatch = base_mismatch
                    for spg in spgs:
                        use_axis_permutation = 15 < spg < 75
                        candidate_cell = permuted_cell if use_axis_permutation else base_cell

                        match, unmatch = check_space_group(spg, matched_hkls,
                                                           base_mismatch,
                                                           axis_order)
                        #print(f"Checking SPG {spg} for {bra_type} with axis order {axis_order}: match={match}, unmatch={len(unmatch)}")
                        #import sys; sys.exit()
                        use_direct_rescue = False
                        direct_sol = None
                        if not match:
                            rescue_ok, direct_sol = rescue_spg_cell(spg, candidate_cell, rescue_state)
                            if rescue_ok:
                                match = True
                                use_direct_rescue = True
                                unmatch = direct_sol.get('mismatch', [])

                        if match:
                            #if verbose:
                            #    print(f"Adding Space group {spg}: Match: {match} {len(unmatch)}")
                            if use_direct_rescue:
                                cell = candidate_cell
                                peaks = direct_sol.get('match', [])
                                mis_peaks = direct_sol.get('mismatch', [])
                                chi2_vals = direct_sol.get('chi2', base_solution['chi2'])
                                errors_vals = direct_sol.get('errors', base_solution['errors'])
                                id_vals = direct_sol.get('id', base_solution['id'])
                            elif 15 < spg < 75:
                                cell = candidate_cell
                                peaks = permuted_match
                                mis_peaks = permuted_mismatch
                                chi2_vals = base_solution['chi2']
                                errors_vals = base_solution['errors']
                                id_vals = base_solution['id']
                            else:
                                cell = base_cell
                                peaks = base_match
                                mis_peaks = base_mismatch
                                chi2_vals = base_solution['chi2']
                                errors_vals = base_solution['errors']
                                id_vals = base_solution['id']
                            solution = {
                                'spg': spg,
                                'cell': cell,
                                'match': peaks,
                                'mismatch': mis_peaks,
                                'chi2': chi2_vals,
                                'errors': errors_vals,
                                'id': id_vals,
                            }
                            volume = solver.get_volume_from_cell(cell)
                            if max_volume is not None and volume > max_volume: continue
                            #cell_str = '[' + ', '.join(f'{float(x):.3f}' for x in np.asarray(cell).tolist()) + ']'
                            #print(f"Solution for {bra_type}, {spg}, cell: {cell_str}, volume: {volume:.2f}, mismatch: {len(unmatch)}, chi2: {chi2_vals[1]:.4f}")
                            solutions.append(solution)
                            min_mismatch = min(min_mismatch, len(unmatch))
                            count += 1

            if branch_rescue_stats["triggered"] > 0:
                print(
                    f"Direct SG rescue stats ({bra_type}): "
                    f"triggered={branch_rescue_stats['triggered']}, "
                    f"accepted={branch_rescue_stats['accepted']}, "
                    f"cache_hits={branch_rescue_stats['cache_hits']}, "
                    f"solver_checks={branch_rescue_stats['solver_checks']}, "
                )
            # Early stop with high confidence
            if count > 0 and min_mismatch <= ideal_mismatch:
                early_stop = True

            if early_stop and bra_index not in [11, 12, 15]: # For higher symmetry branches, we can be more confident about early stopping.
                print(
                    f"SmartCellSolver early stop: {bra_type} reached ideal mismatch "
                    f"(best={min_mismatch}, ideal={ideal_mismatch})."
                )
                return solutions
        return solutions

def check_centering(matched_hkls, centering):
    """
    Check if the presence of certain hkls is consistent with the given centering type.

    Args:
        matched_hkls: List of matched (h, k, l) tuples
        centering: Centering type ('P', 'C', 'A', 'F', 'I', 'R')

    Returns:
        bool: True if the matched hkls are consistent with the centering type, False otherwise
    """
    for hkl in matched_hkls:
        if centering == 'C':
            if (hkl[0] + hkl[1]) % 2 != 0:
                return False
        elif centering == 'A':
            if (hkl[1] + hkl[2]) % 2 != 0:
                return False
        elif centering == 'F':
            if not (hkl[0]%2 == hkl[1]%2 == hkl[2]%2):
                return False
        elif centering == 'I':
            if (hkl[0] + hkl[1] + hkl[2]) % 2 != 0:
                return False
        elif centering == 'R':
            if (hkl[0] - hkl[1] - hkl[2]) % 3 != 0 and (hkl[1] - hkl[0] - hkl[2]) % 3 != 0:
                #print(hkl); import sys; sys.exit("R-centering requires (h-k-l) to be a multiple of 3. The presence of the peak with hkl = {} is inconsistent with R-centering.".format(hkl))
                return False
    return True


def check_space_group(spg, matched_hkls, unmatched_hkls, axis_order):
    """
    Check if the given space group is compatible with the observed matched and unmatched hkls based on extinction rules.

    Args:
        spg: Space group number
        matched_hkls: List of matched (h, k, l) tuples
        unmatched_hkls: List of unmatched (h, k, l) tuples
        axis_order: order of axes to consider for the hkls (e.g., (0, 1, 2) for (h, k, l))

    Returns:
        bool: True if the space group is compatible with the observed hkls, False otherwise
    """
    group = Group(spg)

    # Check if all matched hkls are allowed by the space group
    # This won't be permuted
    if spg in [148, 155, 160, 161, 166, 167]:
        for hkl in matched_hkls:
            if (hkl[0] - hkl[1] - hkl[2]) % 3 != 0 and (hkl[1] - hkl[0] - hkl[2]) % 3 != 0:
                #print(hkl); import sys; sys.exit("R-centering requires (h-k-l) to be a multiple of 3.")
                return False, []
    else:
        for hkl in matched_hkls:
            h, k, l = hkl[axis_order[0]], hkl[axis_order[1]], hkl[axis_order[2]]
            if not group.is_valid_hkl(h, k, l):
                #print(f"+++++++Space group {group.number} does not allow {h}/{k}/{l} {group.is_valid_hkl(h, k, l)}.")
                return False, []

    # Check if any unmatched hkls are allowed by the space group
    unmatched = []
    for (hkl, peak) in unmatched_hkls:
        h, k, l = hkl[axis_order[0]], hkl[axis_order[1]], hkl[axis_order[2]]
        if group.is_valid_hkl(h, k, l):
            unmatched.append((hkl, peak))
    return True, unmatched


def enumerate_wyckoff(cell_dims, spg_list, composition, max_wp, max_dof, max_Z, ref_den=None,
                       verbose=False, max_samples=None, csv_path=None):
    """
    Enumerate Wyckoff position combinations for a SINGLE CELL across MULTIPLE space groups.

    Consolidates all candidates from all SPGs and sorts them globally by count (highest first),
    then by DOF, then by other metrics. This avoids redundant structure generation and
    prioritizes real structural precedents across the entire SPG candidate set.

    Args:
        cell_dims: Cell dimensions (e.g., [a, b, c, alpha, beta, gamma])
        spg_list: List of space group integers to enumerate
        composition: Dictionary of element -> count
        max_wp: Maximum number of Wyckoff positions to consider
        max_dof: Maximum degrees of freedom to consider
        max_Z: Maximum atomic number to consider
        ref_den: (density_min, density_max) tuple for density filtering
        verbose: Whether to print detailed information during enumeration
        max_samples: If set, limit enumeration to this many samples per SPG (for cost estimation)
        csv_path: the reference csv file path from pyxtal for counting occurrences of Wyckoff combinations in known structures (optional)

    Returns:
        List of consolidated Wyckoff candidates sorted by global priority:
            [(spg, comp, lattice, wp_ids, num_wps, dof, count, Z, original_spg), ...]
        Each tuple includes the original SPG for reference.
    """
    all_candidates = []
    enumeration_count = 0  # Track total samples enumerated

    for spg in spg_list:
        wp_manager = WPManager(spg, cell_dims, composition, max_wp=max_wp, max_Z=max_Z,
                               max_dof=max_dof, ref_den=ref_den, csv=csv_path)
        local_sols = wp_manager.get_wyckoff_positions(verbose, max_samples=max_samples)
        #print(f"Enumerated {len(local_sols)} Wyckoff position combinations for SPG {spg} {ref_den}.")
        enumeration_count += len(local_sols)

        # If we're in cost estimation mode and exceeded limit, stop early
        if max_samples is not None and enumeration_count > max_samples:
            if verbose:
                print(f"WPManager for SPG {spg}: Enumeration limited to {max_samples} samples (cost estimation mode).")
            break

        # Tag each solution with which SPG it came from
        for sol in local_sols:
            # sol = (spg, comp, lattice, wp_ids, num_wps, dof, count, Z)
            # Add original SPG as 9th element for reference
            tagged_sol = sol + (spg,)
            all_candidates.append(tagged_sol)

    if not all_candidates:
        return []

    # Sort by count (descending), then by DOF (ascending), then by num_wps, then by Z
    # This prioritizes high-count (real structure) assignments globally
    all_candidates.sort(key=lambda x: (-x[6], x[5], -x[4], x[7]))

    return all_candidates


def score_wp_candidate(sol, max_dof=None):
    """
    Rank a Wyckoff-position candidate for search prioritization.

    Higher-ranked candidates should be cheaper to explore and more likely to
    produce valid structures early in the search.

    Ranking priorities:
    1. Respect the current DOF budget.
    2. Lower combined cost (dof + 1.5 * num_wps) first — balances structural
       compactness (fewer distinct sites) against coordinate freedom.  This
       promotes configurations like '8d 16e 16e' (3 sites, DOF=7) over
       sprawling '16e 8d 4b 4a 8d' (5 sites, DOF=5) because real structures
       tend to occupy fewer distinct Wyckoff orbits.
    3. Higher combinatorial count first.
    4. Less fragmented site assignment first.
    5. Lower Z first.
    """
    (_, _comp, _lattice, wp_ids, num_wps, dof, count, Z) = sol
    within_budget = 1 if max_dof is None or dof <= max_dof else 0
    fragmentation = sum(max(len(wp) - 1, 0) for wp in wp_ids)
    combined_cost = dof + 1.5 * num_wps  # lower → tried first
    return (within_budget, count, -combined_cost, -fragmentation, -Z)


def get_adaptive_wp_limits(total_candidates, max_to_try):
    """Return monotonically increasing candidate cutoffs for adaptive expansion."""
    max_to_try = min(total_candidates, max_to_try)
    if max_to_try <= 0:
        return []

    limits = []
    for limit in (3, 5, 10, max_to_try):
        limit = min(limit, max_to_try)
        if limit > 0 and (not limits or limit > limits[-1]):
            limits.append(limit)
    return limits

def should_perturb(sim, eng_rel, wr, r2, chi2, min_r2, max_chi2, refine_sim_min, refine_eng_window):
    """
    Decide whether ANY candidate deserves a perturb-and-relax trial,
    independently of the regen-boost budget.

    More permissive than should_intensify_regen: fires on any reasonable
    refined result (r2 not too far below target) or high-sim unrefined structure.
    This ensures structures like r2=0.945 get perturbed even after the regen
    boost budget is exhausted.
    """
    if wr is not None and r2 is not None and chi2 is not None:
        if r2 >= max(min_r2 - 0.15, 0.75) or chi2 <= min(max_chi2 * 3.0, 0.50):
            return True

    return sim >= max(refine_sim_min + 0.08, 0.80) and eng_rel <= (refine_eng_window + 0.40)


def is_excellent_refinement(r2, chi2, min_r2, max_chi2):
    """Return True for clearly strong refined fits that should stop immediately."""
    if r2 is None or chi2 is None:
        return False
    return r2 >= max(min_r2 + 0.03, 0.98) or chi2 <= min(max_chi2 * 0.75, 0.08)


def should_terminate(r2, chi2, eng_rel, min_r2, max_chi2, max_eng_rel_for_termination):
    """Allow immediate termination only for excellent fit quality AND near-best energy."""
    if not is_excellent_refinement(r2, chi2, min_r2, max_chi2):
        return False
    return eng_rel <= max_eng_rel_for_termination


def perturb_atoms(atoms, displacement=0.06):
    """Apply a small random Cartesian displacement to atomic positions."""
    trial_atoms = atoms.copy()
    positions = trial_atoms.get_positions()
    positions += np.random.normal(loc=0.0, scale=displacement, size=positions.shape)
    trial_atoms.set_positions(positions)
    return trial_atoms


def _normalize_signature_value(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_signature_value(item) for item in value)
    try:
        return round(float(value), 6)
    except Exception:
        return str(value)

def _normalize_signature_value(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_signature_value(item) for item in value)
    try:
        return round(float(value), 6)
    except Exception:
        return str(value)


def _cell_signature(cell_obj):
    return _normalize_signature_value(getattr(cell_obj, "dims", cell_obj))


def _wp_signature(spg_sol, wp_ids):
    return (
        int(spg_sol),
        tuple(tuple(str(wp) for wp in group) for group in wp_ids),
    )

def _make_structure_log_metadata(cell_obj, spg_sol, wp_ids, num_wps, dof, count, Z, sites):
    cell_sig = _cell_signature(cell_obj)
    wp_sig = _wp_signature(spg_sol, wp_ids)
    return {
        "spg": int(spg_sol),
        "cell_dims": list(getattr(cell_obj, "dims", [])),
        "cell_signature": cell_sig,
        "wp_signature": wp_sig,
        "setting_signature": (cell_sig, wp_sig),
        "wp_labels": [list(group) for group in sites],
        "wp_ids": [list(group) for group in wp_ids],
        "num_wps": int(num_wps),
        "dof": int(dof),
        "count": None if count is None else int(count),
        "Z": int(Z),
    }

def search_solution(cells, spg, composition, ref_den, match_cif,
                    match_csv, x1, y1, eng_min, sim_max, N2, struc_count,
                    max_force, max_stress, wavelength, thetas, resolution, SCALED_INTENSITY_TOL,
                    INST_FILE, logger, max_wp, max_Z, max_dof, max_atoms, min_r2=0.95, max_chi2=0.12, refine_margin=0.02,
                    refine_sim_min=0.7, refine_eng_window=0.5, max_local_perturbations=2,
                    perturb_displacement=0.06, structure_log=[],
                    max_eng_rel_early_stop=None, min_structures_before_early_stop=10,
                    forced_wp_solution=None, ase_logfile=None, global_accepted=False,
                    use_qrs=False):
    """
    Explore candidates and return first satisfactory refinement result.

    Args:
        cells: List of candidate cells.
        spg: Space group number.
        composition: Dictionary of element counts.
        ref_den: Tuple of (min_density, max_density).
        match_cif: Path to save match CIF.
        match_csv: Path to save match CSV.
        x1, y1: Simulated PXRD data arrays.
        eng_min: Current minimum energy.
        sim_max: Current maximum similarity.
        N2: Limits for loops.
        struc_count: Number of structures successfully generated so far.
        max_force: Maximum allowed force for relaxed structures.
        max_stress: Maximum allowed stress for relaxed structures.
        wavelength: X-ray wavelength for XRD simulation.
        thetas: 2theta values for XRD simulation.
        resolution: Resolution for XRD simulation.
        SCALED_INTENSITY_TOL: Tolerance for scaled intensity in XRD simulation.
        INST_FILE: Instrument file for refinement.
        min_r2: Minimum R² value for a good fit.
        max_chi2: Maximum chi² value for a good fit.
        logger: Logger for recording results.
        max_wp: Maximum number of Wyckoff positions to consider.
        max_Z: Maximum atomic number to consider.
        max_dof: Maximum degrees of freedom to consider.
        max_atoms: Maximum number of atoms in the unit cell to consider.
        use_qrs: Whether to use quasirandom sampling for generating trial structures.
    Returns:
        Tuple of (wr, r2, chi2, xtal, eng_best, selected_eng, selected_eng_rel)
    """
    # print(f"\n{'='*60}, struc_count={struc_count}, structure_log={len(structure_log)}")
    # Special case for single-element systems: be more permissive to allow more candidates to be refined and potentially find a good match.
    if len(composition.keys()) == 1: sim_max = min(sim_max, 0.4)  # be more permissive for single-element systems where sim is less reliable
    eng_best = eng_min
    best_refined_result = None
    best_refined_score = -1e9
    best_refined_result_energy_ok = None
    best_refined_energy_ok_score = -1e9
    min_structures_before_early_stop = max(0, int(min_structures_before_early_stop))

    if max_eng_rel_early_stop is None:
        max_eng_rel_for_termination = max(float(refine_eng_window), 0.30)
    else:
        max_eng_rel_for_termination = max(0.0, float(max_eng_rel_early_stop))


    def _finalize_result(result, attempt_count=None):
        if result is None: return None
        wr, r2, chi2, xtal, _eng_best, selected_eng, _sel_eng, count = result
        final_eng_rel = None if selected_eng is None else max(0.0, float(selected_eng) - float(eng_best))
        return (wr, r2, chi2, xtal, eng_best, selected_eng, final_eng_rel, count, attempt_count)

    def _return_best_available(local_candidate, best_refined_result_energy_ok,
                               struc_count=None, attempt_count=None):
        if local_candidate is not None:
            if struc_count is not None:
                # Replace the last item of local_candidate with struc_count
                local_candidate = tuple(list(local_candidate[:-1]) + [struc_count])
            return _finalize_result(local_candidate, attempt_count=attempt_count)

        if best_refined_result_energy_ok is not None:
            if struc_count is not None:
                best_refined_result_energy_ok = tuple(list(best_refined_result_energy_ok[:-1]) + [struc_count])
            return _finalize_result(best_refined_result_energy_ok, attempt_count=attempt_count)

        if best_refined_result is not None:
            logger.info("No accepted candidate was found")
            return (None, None, None, None, eng_best, None, None, struc_count, attempt_count)

        return (None, None, None, None, eng_best, None, None, struc_count, attempt_count)

    #trial_cells = list(cells[:N1])
    early_stop = False
    local_accepted_result = None
    attempt_count = 0

    # Track emitted structure IDs to avoid duplicates
    emitted_id_messages = set()
    for cell in cells:
        # logger.info(f"\nTrying cell: {cell.dims}, missing peaks: {cell.missing}")
        if forced_wp_solution is not None:
            normalized_forced_wp = forced_wp_solution[:8] if len(forced_wp_solution) >= 9 else forced_wp_solution
            ranked_sols = [normalized_forced_wp] if normalized_forced_wp[5] <= max_dof else []
        else:
            wp_manager = WPManager(spg, cell.dims, composition, max_wp, max_Z, max_dof, max_atoms, ref_den)
            ranked_sols = wp_manager.get_wyckoff_positions()
            ranked_sols = sorted(ranked_sols, key=lambda sol: score_wp_candidate(sol), reverse=True)
        if len(ranked_sols) == 0:
            logger.info(f"No Wyckoff candidates satisfy DOF <= {max_dof} for cell {cell.dims}.")
            continue

        if forced_wp_solution is not None:
            # Forced mode already narrows to a single WP candidate.
            wp_limits = [len(ranked_sols)]
        else:
            wp_limits = get_adaptive_wp_limits(len(ranked_sols), N2)

        prev_limit = 0
        for limit in wp_limits:
            for sol in ranked_sols[prev_limit:limit]:
                (spg_sol, comp, lattice, wp_ids, num_wps, dof, count, Z) = sol
                xm = XtalManager(spg_sol, composition.keys(), comp, lattice, wp_ids, use_seeds=use_qrs)
                log_metadata = _make_structure_log_metadata(
                    cell, spg_sol, wp_ids, num_wps, dof, count, Z, xm.sites
                )
                # If DOF=0, allow 1 trial; if DOF=1, use 4; else use DOF*3
                N4 = 1 if xm.dof == 0 else (4 if xm.dof == 1 else xm.dof * 3)
                N_false = 0
                extra_trials = 0
                local_perturbations = 0
                best_sim_in_wpset = 0.0
                valid_trials_in_wpset = 0
                local_accepted_score = -1e9
                # Exit a WP set early if the first warm-up trials all yield very low sim.
                # Use a conservative threshold — well below refine_sim_min — so only
                # truly hopeless WP combinations are skipped.
                wpset_warmup = max(4, N4 // 3)
                wpset_low_sim_exit = max(0.35, refine_sim_min - 0.35)
                trial_idx = 0
                while trial_idx < (N4 + 1 + extra_trials):
                    trial_idx += 1
                    attempt_count += 1
                    if N_false > max([4, N4 // 2]):
                        #logger.info("Too many invalid structures, skip....")
                        break
                    xtal = xm.generate_structure(trial_idx)
                    actual_idx = trial_idx + xm.skips
                    if not xtal.valid:
                        N_false += 1
                        continue
                    atoms = relax_structure(xtal.to_ase(), xm.dof, ase_logfile=ase_logfile)
                    if atoms is None:
                        N_false += 1
                        continue

                    eng = atoms.get_potential_energy() / len(atoms)
                    stress = abs(atoms.get_stress()[:3].mean())
                    fmax = abs(atoms.get_forces()).max()
                    if stress > max_stress or fmax > max_force:
                        N_false += 1
                        continue
                    prev_eng_best = eng_best
                    next_eng_best = min(prev_eng_best, eng)
                    eng_rel = max(0.0, eng - next_eng_best)
                    is_new_best_energy = eng < prev_eng_best
                    if is_new_best_energy:
                        eng_best = eng

                    xrd = XRD(atoms, wavelength=wavelength, thetas=thetas,
                              res=resolution, SCALED_INTENSITY_TOL=SCALED_INTENSITY_TOL)
                    x2, y2 = xrd.get_plot_gsas2(U=0.1, V=-0.1, W=0.5, X=0.1, Y=0.1,
                                                bg_ratio=0.0, mix_ratio=0.0)

                    y2 = RawDataManager(x2, y2, bg_subtract=False).y
                    sim = Similarity((x1, y1), (x2, y2)).value
                    valid_trials_in_wpset += 1
                    if sim > best_sim_in_wpset: best_sim_in_wpset = sim
                    # Early exit: if after the warm-up window the WP set has never reached
                    # even a very low sim, it is very unlikely to produce a useful structure.
                    if (valid_trials_in_wpset >= wpset_warmup and
                            best_sim_in_wpset < wpset_low_sim_exit and
                            extra_trials == 0):
                        logger.info(
                            f"  Low-sim early exit: best_sim={best_sim_in_wpset:.3f} < "
                            f"{wpset_low_sim_exit:.2f} after {valid_trials_in_wpset} valid trials; "
                            f"skipping remaining trials for this WP set."
                        )
                        break

                    struc_count += 1
                    cell_volume = getattr(cell, 'size', None)
                    volume_str = f" vol={cell_volume:.1f} Å³" if cell_volume is not None else ""
                    msg = f"ID{struc_count: 3d}-{actual_idx:2d}: {xtal.get_xtal_string()}, {volume_str}"
                    # Do not emit here; emission is handled after refinement/perturbation with duplicate suppression
                    refined_score = None

                    # Composite refinement trigger using two independent, system-agnostic criteria:
                    #   1. sim >= refine_sim_min: structure has meaningful pattern agreement.
                    #   2. eng_rel <= refine_eng_window: energy is within `refine_eng_window`
                    #      eV/atom of the best structure seen so far in this run.
                    if sim >= max(sim_max - refine_margin, 0.0) or (sim >= refine_sim_min and eng_rel <= refine_eng_window):
                        #title0 = title + f' {eng:.3f}/{eng_best:.3f}'
                        #plot_XRD(x1, y1, x2, y2, x1[peaks], y1[peaks], title0, match_png)
                        xtal.from_seed(atoms)
                        xtal.to_file(match_cif)
                        wr, r2, chi2, _, elapsed = refine_pxrd(match_csv, match_cif, INST_FILE)

                        if wr is None:
                            # Save a copy of the failed CIF for debugging
                            _fail_dir = os.path.join(os.getenv("PXRD_TMP_ROOT", "tmp"), "gsas_runs")
                            os.makedirs(_fail_dir, exist_ok=True)
                            _fail_dst = os.path.join(_fail_dir, f"failed_ID{struc_count}.cif")
                            try:
                                shutil.copy2(match_cif, _fail_dst)
                            except Exception:
                                pass

                        if wr is not None:
                            refined_score = float((1.5 * r2) - (0.4 * wr) - (0.2 * chi2))
                            if refined_score > best_refined_score:
                                best_refined_score = refined_score
                                best_refined_result = (wr, r2, chi2, xtal, eng_best, eng, eng_rel, struc_count)
                            if eng_rel <= max_eng_rel_for_termination and refined_score > best_refined_energy_ok_score:
                                    best_refined_energy_ok_score = refined_score
                                    best_refined_result_energy_ok = (wr, r2, chi2, xtal, eng_best, eng, eng_rel, struc_count)

                            _do_perturb = (
                                (local_perturbations < max_local_perturbations or is_new_best_energy) and
                                should_perturb(sim, eng_rel, wr, r2, chi2,
                                    min_r2, max_chi2, refine_sim_min, refine_eng_window,
                                )
                            )
                            if _do_perturb:
                                remaining_perturbations = max_local_perturbations - local_perturbations
                                # If triggered by a new-best-energy bypass, always allow at least 1 trial
                                if remaining_perturbations <= 0 and is_new_best_energy:
                                    remaining_perturbations = 1
                                perturb_trials = min(remaining_perturbations, 1 if wr is not None else 2)
                                for perturb_idx in range(perturb_trials):
                                    local_perturbations += 1
                                    displacement = max(0.02, perturb_displacement * (0.67 if wr is not None else 1.0))
                                    perturbed_atoms = perturb_atoms(atoms, displacement=displacement)
                                    perturbed_atoms = relax_structure(perturbed_atoms, xm.dof, ase_logfile=ase_logfile)
                                    if perturbed_atoms is None:
                                        continue

                                    #struc_count += 1  # Increment for each perturbed structure

                                    p_eng = perturbed_atoms.get_potential_energy() / len(perturbed_atoms)
                                    p_stress = abs(perturbed_atoms.get_stress()[:3].mean())
                                    p_fmax = abs(perturbed_atoms.get_forces()).max()
                                    if p_stress > max_stress or p_fmax > max_force:
                                        logger.info(
                                            f"  Perturbation {perturb_idx + 1}/{perturb_trials} rejected by stress/force "
                                            f"filters ({p_stress:.3f}, {p_fmax:.3f})."
                                        )
                                        continue

                                    prev_eng_best = eng_best
                                    next_eng_best = min(prev_eng_best, p_eng)
                                    p_eng_rel = max(0.0, p_eng - next_eng_best)
                                    p_is_new_best_energy = p_eng < prev_eng_best
                                    if p_is_new_best_energy: eng_best = p_eng

                                    p_xrd = XRD(perturbed_atoms, wavelength=wavelength, thetas=thetas,
                                                res=resolution, SCALED_INTENSITY_TOL=SCALED_INTENSITY_TOL)
                                    p_x2, p_y2 = p_xrd.get_plot_gsas2(U=0.1, V=-0.1, W=0.5, X=0.1, Y=0.1,
                                                                      bg_ratio=0.0, mix_ratio=0.0)
                                    p_y2 = RawDataManager(p_x2, p_y2, bg_subtract=False).y
                                    p_sim = Similarity((x1, y1), (p_x2, p_y2)).value
                                    if p_sim >= max(sim_max - refine_margin, 0.0) or (p_sim >= refine_sim_min and p_eng_rel <= refine_eng_window):
                                        xtal.from_seed(perturbed_atoms)
                                        xtal.to_file(match_cif)
                                        p_wr, p_r2, p_chi2, _, p_elapsed = refine_pxrd(match_csv, match_cif, INST_FILE)
                                        if p_wr is None:
                                            _fail_dir = os.path.join(os.getenv("PXRD_TMP_ROOT", "tmp"), "gsas_runs")
                                            os.makedirs(_fail_dir, exist_ok=True)
                                            _fail_dst = os.path.join(_fail_dir, f"failed_ID{struc_count}_perturb{perturb_idx}.cif")
                                            try:
                                                shutil.copy2(match_cif, _fail_dst)
                                            except Exception:
                                                pass
                                        if p_wr is not None and p_r2 is not None and p_chi2 is not None:
                                            if p_r2 > r2 and p_chi2 < chi2:
                                                wr, r2, chi2, elapsed = p_wr, p_r2, p_chi2, p_elapsed
                                                eng, eng_rel, sim = p_eng, p_eng_rel, p_sim
                                                stress, fmax = p_stress, p_fmax
                                                is_new_best_energy = p_is_new_best_energy
                                    msg += f" {sim:.3f}, {eng:.3f}, {stress:.3f}, {fmax:.3f}"
                                    msg += f" {wr:6.3f}, {r2:6.3f}, {chi2:6.3f}, {elapsed:.1f}s"
                                    if is_new_best_energy:
                                        msg += ' +++++'
                            else:
                                msg += f" {sim:.3f}, {eng:.3f}, {stress:.3f}, {fmax:.3f}"
                                msg += f" {wr:6.3f}, {r2:6.3f}, {chi2:6.3f}, {elapsed:.1f}s"
                        else:
                            logger.info("  Refinement failed; continuing search without refined metrics.")
                            msg += f" {sim:.3f}, {eng:.3f}, {stress:.3f}, {fmax:.3f}"
                            msg += " [refine-failed]"
                    else:
                        wr, r2, chi2 = None, None, None
                        msg += f" {sim:.3f}, {eng:.3f}, {stress:.3f}, {fmax:.3f}"

                    logger.info(msg)
                    emitted_id_messages.add(msg)
                    log_entry = {
                        "eng": eng,
                        "eng_rel": eng_rel,
                        "sim": sim,
                        "r2": r2,
                        "wr": wr,
                        "chi2": chi2,
                        "refined": r2 is not None,
                        **log_metadata,
                    }
                    structure_log.append(log_entry)

                    # If a globally-accepted solution already exists from a previous
                    # WP/pair, treat reaching the structure budget as an early stop.
                    if global_accepted and len(structure_log) >= min_structures_before_early_stop:
                        early_stop = True

                    if wr is not None and (r2 > min_r2 or chi2 < max_chi2):
                        energy_ok = eng_rel <= max_eng_rel_for_termination
                        if refined_score is not None and refined_score > local_accepted_score and energy_ok:
                            local_accepted_score = refined_score
                            local_accepted_result = (wr, r2, chi2, xtal, eng_best, eng, eng_rel, struc_count)

                        if should_terminate(r2, chi2, eng_rel, min_r2, max_chi2, max_eng_rel_for_termination):
                            early_stop = True
                            logger.info(
                                f"***Excellent fit within current WP trial (r2={r2:.3f}, chi2={chi2:.3f}) "
                                f"with energy (eng_rel={eng_rel:.3f} eV/atom); "
                            )
                    if (struc_count >= min_structures_before_early_stop
                            or len(structure_log) >= min_structures_before_early_stop) and early_stop:
                        logger.info(
                            f"Early stop triggered after {struc_count} local / "
                            f"{len(structure_log)} global structures; terminating current WP search."
                        )
                        return _return_best_available(local_accepted_result, best_refined_result_energy_ok, struc_count, attempt_count=attempt_count)

            prev_limit = limit
        # Correct the structure count in the final accepted result if it exists, to reflect the total number of structures generated so far.
    return _return_best_available(local_accepted_result, best_refined_result_energy_ok, struc_count, attempt_count=attempt_count)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from pathlib import Path

    def _infer_formula_spg(path: Path):
        """Infer formula and space group from a file name like PXRD_<formula>_<spg>.csv."""
        tokens = path.stem.split('_')
        formula_guess, spg_guess = None, None
        if len(tokens) >= 2:
            try:
                spg_guess = int(tokens[-1])
                # Join middle tokens to support names with extra underscores.
                formula_guess = '_'.join(tokens[1:-1]) if len(tokens) > 2 else None
            except ValueError:
                pass
        return formula_guess, spg_guess

    for data in [#f'Examples/PXRD_Ba14Na14LiN6_225.csv',
                      #(f'GSAS_PXRD/O2Rb_139.csv', 115.46),
                      #(f'GSAS_PXRD/Ba3P4_43.csv', 1721.07),
                      #(f'GSAS_PXRD/Fe2O4Ti_36.csv', 304.28),
                      #f'Examples/PXRD_Be2SiBi_119.csv',
                      #f'Examples/PXRD_K2SnO6_148.csv',
                      #f'Examples/PXRD_HgC2N2_122.csv',
                      #f'Examples/PXRD_Mg9Si5_176.csv',
                      #f'Examples/hardPXRD_HfTlCuS3_63.csv',
                      #f'Examples/PXRD_CoO2_12.csv',
                      #f'GSAS_PXRD/BaMn4O8_12.csv',
                      #(f'GSAS_PXRD/PdPm2Pt_225.csv', 364.50),
                      #(f'GSAS_PXRD/Pa3Te_221.csv', 95.46),
                      #(f'GSAS_PXRD/Hg3Mg_194.csv', 189.09),
                      #(f'GSAS_PXRD/AsAuEu_194.csv', 142.64),
                      #(f'GSAS_PXRD/AsMnPd_189.csv', 144.73),
                      #(f'GSAS_PXRD/Al3H2Ni3Zr3_189.csv', 146.83),
                      #(f'GSAS_PXRD/AlMg_187.csv', 37.66),
                      #(f'GSAS_PXRD/PbS_186.csv', 221.80),
                      #(f'GSAS_PXRD/AsNdPd_186.csv', 128.87),
                      #(f'GSAS_PXRD/In2Ni3S2_166.csv', 337.66),
                      #(f'GSAS_PXRD/PbS_160.csv', 160.12),
                      #(f'GSAS_PXRD/S8TlV6_147.csv', 238.46),
                      #(f'GSAS_PXRD/O2Zr_141.csv', 172.37),
                      #(f'GSAS_PXRD/O2SbSm2_139.csv', 209.28),
                      #(f'GSAS_PXRD/AsFe2_129.csv', 77.97),
                      #(f'GSAS_PXRD/ErTe2_129.csv', 173.24),
                      #(f'GSAS_PXRD/AsPd5Tl_123.csv', 115.95),
                      #(f'GSAS_PXRD/Al3CNi9_123.csv', 143.96),
                      #(f'GSAS_PXRD/CeCu4LaSi4_123.csv', 166.75),
                      #(f'GSAS_PXRD/Eu2IrPd3Si4_115.csv', 180.60),
                      #(f'GSAS_PXRD/Al3EuSi_107.csv', 205.06),
                      #(f'GSAS_PXRD/F3OV_76.csv', 277.97),
                      #(f'GSAS_PXRD/AsErTe_62.csv', 297.10),
                      #(f'GSAS_PXRD/AsHoS_62.csv', 247.71),
                      #(f'GSAS_PXRD/Pd3Rh3U2_47.csv', 133.31),
                      #(f'GSAS_PXRD/O6RbTa2_46.csv', 488.90),
                      #(f'GSAS_PXRD/Co3Ni6S8_44.csv', 488.90),
                      #(f'GSAS_PXRD/Co3Ni6S8_44.csv', 488.90),
                      #(f'GSAS_PXRD/F11Ni6O_38.csv', 411.66),
                      #(f'GSAS_PXRD/AlMg_187.csv', 37.66),
                      #(f'GSAS_PXRD/CeCu4LaSi4_123.csv', 166.75),
                      #(f'GSAS_PXRD/Al3EuSi_107.csv', 205.06),
                      #(f'GSAS_PXRD/Cu3O4_225.csv', 733.66),
                      #(f'GSAS_PXRD/Li_166.csv', 183.62),       #pass
                      #(f'GSAS_PXRD/Li_229.csv', 40.68),        #pass
                      #(f'GSAS_PXRD/Li_225.csv', 81.39),        #pass
                      #(f'GSAS_PXRD/Li_194.csv', 81.46),        #pass
                      #(f'GSAS_PXRD/Li3N_191.csv', 43.50),      #pass
                      #(f'GSAS_PXRD/HLi4N_88.csv', 227.87),     #pass
                      #(f'GSAS_PXRD/N_136.csv', 89.17),         #pass
                      #(f'GSAS_PXRD/BeH2_72.csv', 280.64),      #pass
                      #(f'GSAS_PXRD/B12BeC2_166.csv', 336.32),  #pass
                      #(f'GSAS_PXRD/B3Li_127.csv', 146.90), #pass
                      #(f'GSAS_PXRD/H8Si_31.csv', 233.55),
                      #(f'GSAS_PXRD/AsCd2Cl2_14.csv', 549.84),
                      #(f'GSAS_PXRD/RbTe6_15.csv', 1058.53), #pass
                      #(f'GSAS_PXRD/Co3LiO6_12.csv', 419.64), #++++++
                      #(f'GSAS_PXRD/AsPd5_12.csv', 363.93), #++++++
                      #(f'GSAS_PXRD/In3Rh2Sr2_12.csv', 342.05), #++++++
                      #(f'GSAS_PXRD/In4SbTe3_8.csv', 243.44), #pass
                      #(f'GSAS_PXRD/O2Si_9.csv', 363.63), #pass
                      #(f'GSAS_PXRD/BaSb2_11.csv', 656.92),
                      (f'GSAS_PXRD/S4V_15.csv', 892.36),
                      #(f'GSAS_PXRD/B2Li2Se5_15.csv', 666.82),
                      #(f'GSAS_PXRD/LiMo2O4_15.csv', 361.04),
                    ]:
        match_csv, ref_volume = data if isinstance(data, tuple) else (data, None)
        formula, ref_spg = _infer_formula_spg(Path(match_csv))
        df = pd.read_csv(match_csv)
        x1, y1 = df.iloc[:, 0].values, df.iloc[:, 1].values
        if y1.min() > 2.5:
            bg_subtract = True
        else:
            if y1.min() > 1.0:
                y1 -= y1.min()
            bg_subtract = False
        min_height = 7.5 if bg_subtract else 2.0 #4#3.5#0
        height = min_height if bg_subtract else 1.0
        data = RawDataManager(x1, y1, bg_subtract=bg_subtract) #bg_subtract=False)
        data.get_peaks_from_scipy(height=min_height)#height=25.0)
        data.filter_peaks_by_ml(threshold=0.8, min_height=min_height)#, max_theta=50.0)
        #data.get_peaks_from_scipy_adaptive()
        peaks = data.peaks
        peak_positions = x1[peaks]
        data.plot('test.png')
        min_abc = 2.0
        max_abc = 35.0
        if False: #ref_spg is not None:
            solver = CellSolver(ref_spg,
                            x1[data.peaks],
                            max_mismatch=30,
                            hkl_max=(4, 4, 12),
                            max_square=25,
                            total_square=25,
                            theta_tols=[0.1, 0.15, 0.5],
                            min_abc=min_abc,
                            max_abc=max_abc,
                            verbose=True,
                            )
            solutions = solver.solve(max_solutions=100, max_count=100)
            sols = [
                (ref_spg, sol['cell'], sol['mismatch'], sol['chi2'][1], sol['errors'], sol['id'], sol['match'])
                for sol in solutions
            ]
            #_, msg = solver.validate_cell(np.array([13.75, 3.16, 9.67, 133.90], dtype=np.float64))
            #_, msg = solver.validate_cell(np.array([6.24, 15.00, 18.40], dtype=np.float64), verbose=True)
            #print(f"Validation message for test cell: {msg}"); import sys; sys.exit(0)
        else:
            solutions = SmartCellSolver(x1[data.peaks],
                            max_mismatch=30,
                            hkl_max=(4, 4, 4),
                            max_square=25,
                            total_square=25,
                            theta_tols=[0.1, 0.15, 0.5],
                            min_abc=min_abc,
                            max_abc=max_abc,
                            verbose=True,
                            max_volume=2000,
                            )
            sols = [
                (sol['spg'], sol['cell'], sol['mismatch'], sol['chi2'][1], sol['errors'], sol['id'], sol['match'])
                for sol in solutions
            ]
        if sols is None or len(sols) == 0:
            print(f"No candidate cells found for {match_csv}")
            continue
        cells = CellManager.consolidate(sols,
                                        max_solutions=200,
                                        merge_tol=0.02,
                                        ref_spg=ref_spg,
                                        ref_volume=ref_volume,
                                        max_mismatch=40,
                                        sort_by='volume')
        print(f"Final consolidated solutions for {match_csv}\n")

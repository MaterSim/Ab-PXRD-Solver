"""
Module for PXRD indexing and lattice parameter estimation.
"""
import os
import sys
import numpy as np
from itertools import combinations
from pyxtal.symmetry import Group
from pyxtal.symmetry import get_bravais_lattice, get_lattice_type, generate_possible_hkls

def _missing_gsas_refine_pxrd(*args, **kwargs):
    raise ModuleNotFoundError(
        "GSAS-II Python module 'GSASIIscriptable' is not available. "
        "Install GSAS-II (e.g., conda-forge gsas2pkg) to enable refinement."
    )

try:
    from .manager import RawDataManager, CellManager, WPManager, XtalManager
    from .utils import plot_XRD, relax_structure
    from .XRD import Similarity, XRD
    try:
        from .gsas import refine_pxrd
    except Exception:
        refine_pxrd = _missing_gsas_refine_pxrd
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from tools.manager import RawDataManager, CellManager, WPManager, XtalManager
    from tools.utils import plot_XRD, relax_structure
    from tools.XRD import Similarity, XRD
    try:
        from tools.gsas import refine_pxrd
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

class CellSolver:
    def __init__(self, spg, thetas, bra=None, N_add=5, max_mismatch=20, theta_tols=[0.1, 0.15, 0.5],
                 cell_tol=0.25, hkl_max=(2, 3, 10), max_square=20,
                 total_square=50, min_abc=2.0, max_abc=30.0,
                 min_angle=30.0, max_angle=150.0, min_volume=20.0, max_chi2=0.5,
                 N_batch=20, wave_length=1.54184, max_guess=50000, verbose=False):
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
        return h_max, k_max, l_max

    def validate_cell(self, cell, trial_hkls=None, hkl=None, h_max=None, k_max=None, l_max=None):
        """
        Validate the cell parameters by comparing the expected 2theta values with the observed ones.

        Args:
            cell: cell parameters to validate
            trial_hkls: list of hkls to consider for this cell
            hkl: the original hkl used to generate the cell (for reference)
            h_max, k_max, l_max: maximum h, k, l indices to consider based on the cell parameters

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
            msg = f"Rejected cell: {cell_str}, matched {len(ids_matched)}/{self.theta_count} peaks"
        return None, msg


    def validate_cell_loose(self, cell, trial_hkls=None, hkl=None, h_max=None, k_max=None, l_max=None):
        """
        Validate the cell parameters by comparing the expected 2theta values with the observed ones.
        We ignore the restriction of matching all peaks and allow some mismatches

        Args:
            cell: cell parameters to validate
            trial_hkls: list of hkls to consider for this cell
            hkl: the original hkl used to generate the cell (for reference)
            h_max, k_max, l_max: maximum h, k, l indices to consider based on the cell parameters

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

    def solve(self, max_solutions=10, max_count=50):
        """
        Solve for possible cell parameters based on the provided 2theta values.

        Args:
            max_solutions: Maximum number of unique cell solutions to return.
            max_count: Maximum number of solutions with perfect peak matches (i.e., all observed peaks matched) to consider

        Returns:
            results: list of solution dictionaries containing cell parameters and matched peaks
        """

        h, k, l = self.hkl_max
        guesses = self.group.generate_hkl_guesses(h, k, l, max_square=self.max_square,
                                              total_square=self.total_square,
                                              verbose=self.verbose)
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
                if self.spg is not None:
                    print(
                        f"Filtered singular orthorhombic hkl guess sets for space group {self.spg}: "
                        f"{len(guesses)} -> {int(np.count_nonzero(valid_mask))}."
                    )
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

                        if len(cell_all) >= max_solutions or perfect_match_count >= max_count:
                            print(f"Reached maximum number of solutions ({max_solutions}). Stop!")
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
            ('orthorhombic-F', 8, 2, [22, 42, 43, 69, 70]),
            ('orthorhombic-I', 7, 2, [23, 24, 44, 45, 46, 71, 72, 73, 74]),
            ('orthorhombic-C', 6, 2, [20, 21, 35, 36, 37, 63, 64, 65, 66, 67, 68]),
            ('orthorhombic-A', 5, 2, [38, 39, 40, 41]),
            ('tetragonal-P', 9, 0, [75, 76, 77, 78, 81, 83, 84, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 99,
                            100, 101, 102, 103, 104, 105, 106, 111, 112, 113, 114, 115, 116, 117, 118,
                            123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138]),
            ('orthorhombic-P', 4, 2, [16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 33, 34, 47, 48, 49,
                                      50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]),
            ('monoclinic-C', 3, 4, [5, 8, 9, 12, 15]),
            ('monoclinic-P', 2, 4, [3, 4, 6, 7, 10, 11, 13, 14]),
        ]
        max_volume_cap = None if max_volume is None else float(max_volume)
        min_mismatch = max_mismatch + 1
        solutions = []
        for (bra_type, bra_index, ideal_mismatch, spgs) in bra_list:
            print(f"Trying {bra_type} ...")
            solver_hkl_max = hkl_max
            solver_max_square = max_square
            solver_total_square = total_square
            solver_max_guess = 50000

            # Runtime guardrails for low-symmetry branches where hkl-guess
            # combinatorics can explode (e.g., monoclinic-C > 1e6 raw guesses).
            if bra_type.startswith('monoclinic'):
                solver_hkl_max = (
                    min(int(hkl_max[0]), 2),
                    min(int(hkl_max[1]), 4),
                    min(int(hkl_max[2]), 4),
                )
                solver_max_square = min(int(max_square), 20)
                solver_total_square = min(int(total_square), 28)
                solver_max_guess = 12000
            elif bra_type.startswith('orthorhombic'):
                solver_max_guess = 25000

            # use very strict criteria to get initial cell solutions for each space group,
            # then use those solutions to determine the centering and possible space groups.
            # This way we can significantly reduce the number of space groups we need to check in the later steps,
            # and also increase the chances of finding the correct solution by starting with a more accurate initial guess.
            solver = CellSolver(spg=spgs[0], thetas=thetas, hkl_max=solver_hkl_max, max_mismatch=max_mismatch,
                                max_chi2=max_chi2, max_square=solver_max_square, total_square=solver_total_square,
                                min_abc=min_abc, max_abc=max_abc, min_volume=min_volume,
                                theta_tols=theta_tols, max_guess=solver_max_guess,
                                verbose=verbose)
            base_solutions = solver.solve(max_solutions=15, max_count=20)
            if len(base_solutions) == 0: continue

            count = 0
            # Build supercell variants: for each base solution also try n×cell (n=2,3)
            # so we don't miss a correct super-cell whose primitive hits max_mismatch
            # in the base SPG (which has no extinctions).  The super-cell is then
            # checked against the real SPG's extinction rules and often has 0 mismatch.
            extended_base_solutions = list(base_solutions)
            for _bs in base_solutions:
                for _n in (2, 3):
                    _sup_cell = [c * _n for c in _bs['cell']]
                    if any(c > max_abc for c in _sup_cell):
                        continue
                    _sup_vol = solver.get_volume_from_cell(_sup_cell)
                    if _sup_vol > max_abc ** 3:
                        continue
                    if max_volume_cap is not None and _sup_vol > max_volume_cap:
                        continue
                    _sup = {k: v for k, v in _bs.items()}
                    _sup['cell'] = _sup_cell
                    _sup['match'] = [
                        (tuple(int(v) * _n for v in hkl), obs_theta, cal_theta)
                        for (hkl, obs_theta, cal_theta) in _bs['match']
                    ]
                    _sup['mismatch'] = [
                        (tuple(int(v) * _n for v in hkl), theta)
                        for (hkl, theta) in _bs['mismatch']
                    ]
                    extended_base_solutions.append(_sup)

            direct_solver_cache = {}
            direct_validate_cache = {}

            def _try_direct_spg_rescue(spg_value, candidate_cell):
                cell_sig = tuple(round(float(x), 4) for x in np.asarray(candidate_cell).tolist())
                cache_key = (int(spg_value), cell_sig)
                if cache_key in direct_validate_cache:
                    return direct_validate_cache[cache_key]

                try:
                    direct_solver = direct_solver_cache.get(int(spg_value))
                    if direct_solver is None:
                        direct_solver = CellSolver(
                            spg=int(spg_value),
                            thetas=thetas,
                            hkl_max=solver_hkl_max,
                            max_mismatch=max_mismatch,
                            max_chi2=max_chi2,
                            max_square=solver_max_square,
                            total_square=solver_total_square,
                            min_abc=min_abc,
                            max_abc=max_abc,
                            min_volume=min_volume,
                            theta_tols=theta_tols,
                            max_guess=min(4000, solver_max_guess),
                            verbose=False,
                        )
                        direct_solver_cache[int(spg_value)] = direct_solver

                    sol_direct, _ = direct_solver.validate_cell(np.array(candidate_cell, dtype=float))
                    if sol_direct is None:
                        out = (False, None)
                    else:
                        out = (True, sol_direct)
                except Exception:
                    out = (False, None)

                direct_validate_cache[cache_key] = out
                return out

            for base_solution in extended_base_solutions:
                if bra_index in [4, 7, 8]:
                    axis_orders = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                elif bra_index in [5]: #A-center
                    axis_orders = [(0, 1, 2), (0, 2, 1)]
                elif bra_index in [6]: #C-center
                    axis_orders = [(0, 1, 2), (1, 0, 2)]
                else:
                    axis_orders = [(0, 1, 2)]

                matched_hkls = [m[0] for m in base_solution['match']]

                for axis_order in axis_orders:
                    #permuted_hkls = [tuple(m[0][i] for i in axis_order) for m in matched_hkls]
                    #if len(axis_orders)==6: print(f"Permuted hkls: {permuted_hkls}"); import sys; sys.exit("Debugging stop.")
                    for spg in spgs:
                        if 15 < spg < 75:
                            candidate_cell = [base_solution['cell'][i] for i in axis_order]
                        else:
                            candidate_cell = base_solution['cell']

                        match, unmatch = check_space_group(spg, matched_hkls,
                                                           base_solution['mismatch'],
                                                           axis_order)
                        use_direct_rescue = False
                        direct_sol = None
                        if not match:
                            rescue_ok, direct_sol = _try_direct_spg_rescue(spg, candidate_cell)
                            if rescue_ok:
                                match = True
                                use_direct_rescue = True
                                unmatch = direct_sol.get('mismatch', [])
                                #print(
                                #    f"Direct SG rescue accepted: spg={spg}, cell="
                                #    f"[{', '.join(f'{float(x):.3f}' for x in np.asarray(candidate_cell).tolist())}]"
                                #)

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
                                peaks = [(tuple(m[0][i] for i in axis_order), m[1], m[2]) for m in base_solution['match']]
                                mis_peaks = [(tuple(m[0][i] for i in axis_order), m[1]) for m in base_solution['mismatch']]
                                chi2_vals = base_solution['chi2']
                                errors_vals = base_solution['errors']
                                id_vals = base_solution['id']
                            else:
                                cell = base_solution['cell']
                                peaks = base_solution['match']
                                mis_peaks = base_solution['mismatch']
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
                            cell_str = '[' + ', '.join(f'{float(x):.3f}' for x in np.asarray(cell).tolist()) + ']'
                            volume = solver.get_volume_from_cell(cell)
                            if max_volume_cap is not None and volume > max_volume_cap:
                                continue
                            print(f"Solution for {bra_type}, {spg}, cell: {cell_str}, volume: {volume:.2f}, mismatch: {len(unmatch)}, chi2: {chi2_vals[1]:.4f}")
                            solutions.append(solution)
                            count += 1
                            if min_mismatch > len(unmatch):
                                min_mismatch = len(unmatch)
            # Early stop with high confidence
            if count > 0 and min_mismatch <= ideal_mismatch:
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
                #print(f"Space group {spg} does not allow {hkl}.")
                return False, []

    # Check if any unmatched hkls are allowed by the space group
    unmatched = []
    for (hkl, peak) in unmatched_hkls:
        h, k, l = hkl[axis_order[0]], hkl[axis_order[1]], hkl[axis_order[2]]
        if group.is_valid_hkl(h, k, l):
            unmatched.append((hkl, peak))
    return True, unmatched


def enumerate_wyckoff_multi_spg(cell_dims, spg_list, composition, max_wp, max_dof, max_Z, ref_den=None):
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

    Returns:
        List of consolidated Wyckoff candidates sorted by global priority:
            [(spg, comp, lattice, wp_ids, num_wps, dof, count, Z, original_spg), ...]
        Each tuple includes the original SPG for reference.
    """
    all_candidates = []

    for spg in spg_list:
        wp_manager = WPManager(spg, cell_dims, composition, max_wp=max_wp, max_Z=max_Z, max_dof=max_dof, ref_den=ref_den)
        local_sols = wp_manager.get_wyckoff_positions()#; print(f'+++++++ {wp_manager.spg}: Z={wp_manager.Zs}')

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
    all_candidates.sort(
        key=lambda x: (-x[6], x[5], -x[4], x[7]),
        reverse=False  # Lower values earlier except for count
    )

    # Actually, let's fix the sort: count is highest priority (descending), DOF second (ascending)
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


def should_boost(sim, eng_rel, wr, r2, chi2, min_r2, max_chi2, refine_sim_min, refine_eng_window):
    """
    Decide whether a promising candidate justifies extra regeneration trials.
    Gated by max_local_boosts counter.
    """
    if wr is not None and r2 is not None and chi2 is not None:
        if r2 >= max(min_r2 - 0.12, 0.78) or chi2 <= min(max_chi2 * 2.0, 0.35):
            return True

    strong_similarity = sim >= max(refine_sim_min + 0.10, 0.82)
    near_miss_similarity = sim >= max(refine_sim_min + 0.15, 0.85)
    low_relative_energy = eng_rel <= (refine_eng_window + 0.25)
    near_miss_energy = eng_rel <= (refine_eng_window + 0.35)
    return (strong_similarity and low_relative_energy) or (near_miss_similarity and near_miss_energy)


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

def search_solution(cells, spg, composition, ref_den, title, match_png, match_cif,
                    match_csv, peaks, x1, y1, eng_min, sim_max, N1, N2, N3, struc_count,
                    max_force, max_stress, wavelength, thetas, resolution, SCALED_INTENSITY_TOL,
                    INST_FILE, logger, max_wp, max_Z, max_dof, min_r2=0.95, max_chi2=0.12, refine_margin=0.02,
                    refine_sim_min=0.7, refine_eng_window=0.5,
                    max_local_boosts=1, max_local_perturbations=2,
                    perturb_displacement=0.06, structure_log=[],
                    max_eng_rel_early_stop=None, min_structures_before_early_stop=10,
                    forced_wp_solution=None):
    """
    Explore candidates and return first satisfactory refinement result.

    Args:
        cells: List of candidate cells.
        spg: Space group number.
        composition: Dictionary of element counts.
        ref_den: Tuple of (min_density, max_density).
        title: Title for plots.
        match_png: Path to save match plot.
        match_cif: Path to save match CIF.
        match_csv: Path to save match CSV.
        peaks: Indices of peaks used for indexing.
        x1, y1: Simulated PXRD data arrays.
        eng_min: Current minimum energy.
        sim_max: Current maximum similarity.
        N1, N2, N3: Limits for loops.
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

    Returns:
        Tuple of (wr, r2, chi2, xtal, eng_best, selected_eng, selected_eng_rel)
    """
    # print(f"\n{'='*60}, struc_count={struc_count}, structure_log={len(structure_log)}")
    # Special case for single-element systems: be more permissive to allow more candidates to be refined and potentially find a good match.
    if len(composition.keys()) == 1: sim_max = min(sim_max, 0.5)  # be more permissive for single-element systems where sim is less reliable
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


    def _finalize_result(result):
        if result is None: return None
        wr, r2, chi2, xtal, _eng_best_at_sel, selected_eng, _selected_eng_rel, count = result
        final_eng_rel = None if selected_eng is None else max(0.0, float(selected_eng) - float(eng_best))
        return (wr, r2, chi2, xtal, eng_best, selected_eng, final_eng_rel, count)

    def _return_best_available(local_candidate, best_refined_result_energy_ok, struc_count=None):
        if local_candidate is not None:
            if struc_count is not None:
                # Replace the last item of local_candidate with struc_count
                local_candidate = tuple(list(local_candidate[:-1]) + [struc_count])
            return _finalize_result(local_candidate)

        if best_refined_result_energy_ok is not None:
            if struc_count is not None:
                best_refined_result_energy_ok = tuple(list(best_refined_result_energy_ok[:-1]) + [struc_count])
            return _finalize_result(best_refined_result_energy_ok)

        if best_refined_result is not None:
            logger.info("No accepted candidate was found")
            return (None, None, None, None, eng_best, None, None, struc_count)

        return (None, None, None, None, eng_best, None, None, struc_count)

    trial_cells = list(cells[:N1])
    early_stop = False
    local_accepted_result = None

    # Track emitted structure IDs to avoid duplicates
    emitted_id_messages = set()
    for cell in trial_cells:
        # logger.info(f"\nTrying cell: {cell.dims}, missing peaks: {cell.missing}")
        if forced_wp_solution is not None:
            normalized_forced_wp = forced_wp_solution[:8] if len(forced_wp_solution) >= 9 else forced_wp_solution
            ranked_sols = [normalized_forced_wp] if normalized_forced_wp[5] <= N3 else []
        else:
            wp_manager = WPManager(spg, cell.dims, composition, max_wp, max_Z, max_dof, ref_den=ref_den)
            sols = wp_manager.get_wyckoff_positions()
            ranked_sols = [sol for sol in sols if sol[5] <= N3]
        if len(ranked_sols) == 0:
            logger.info(f"No Wyckoff candidates satisfy DOF <= {N3} for cell {cell.dims}.")
            continue

        if forced_wp_solution is None:
            ranked_sols = sorted(ranked_sols, key=lambda sol: score_wp_candidate(sol, max_dof=N3), reverse=True)
        wp_limits = get_adaptive_wp_limits(len(ranked_sols), N2)

        prev_limit = 0
        for limit in wp_limits:
            for sol in ranked_sols[prev_limit:limit]:
                (spg_sol, comp, lattice, wp_ids, num_wps, dof, count, Z) = sol
                xm = XtalManager(spg_sol, composition.keys(), comp, lattice, wp_ids, count=count, emit_summary=False)
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
                    if N_false > max([4, N4 // 2]):
                        logger.info("Too many invalid structures, skip....")
                        break
                    xtal = xm.generate_structure()
                    if not xtal.valid:
                        N_false += 1
                        continue
                    atoms = relax_structure(xtal.to_ase(), xm.dof)
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
                    msg = f"ID{struc_count: 3d}: {xtal.get_xtal_string()}, {volume_str}"
                    # Do not emit here; emission is handled after refinement/perturbation with duplicate suppression
                    refined_score = None

                    # Composite refinement trigger using two independent, system-agnostic criteria:
                    #   1. sim >= refine_sim_min: structure has meaningful pattern agreement.
                    #   2. eng_rel <= refine_eng_window: energy is within `refine_eng_window`
                    #      eV/atom of the best structure seen so far in this run.
                    if sim >= max(sim_max - refine_margin, 0.0) or (sim >= refine_sim_min and eng_rel <= refine_eng_window):
                        title0 = title + f' {eng:.3f}/{eng_best:.3f}'
                        plot_XRD(x1, y1, x2, y2, x1[peaks], y1[peaks], title0, match_png)
                        xtal.from_seed(atoms)
                        xtal.to_file(match_cif)
                        wr, r2, chi2, _ = refine_pxrd(match_csv, match_cif, INST_FILE)

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
                                    perturbed_atoms = relax_structure(perturbed_atoms, xm.dof)
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
                                    if p_is_new_best_energy:
                                        eng_best = p_eng

                                    p_xrd = XRD(perturbed_atoms, wavelength=wavelength, thetas=thetas,
                                                res=resolution, SCALED_INTENSITY_TOL=SCALED_INTENSITY_TOL)
                                    p_x2, p_y2 = p_xrd.get_plot_gsas2(U=0.1, V=-0.1, W=0.5, X=0.1, Y=0.1,
                                                                      bg_ratio=0.0, mix_ratio=0.0)
                                    p_y2 = RawDataManager(p_x2, p_y2, bg_subtract=False).y
                                    p_sim = Similarity((x1, y1), (p_x2, p_y2)).value
                                    if p_sim >= max(sim_max - refine_margin, 0.0) or (p_sim >= refine_sim_min and p_eng_rel <= refine_eng_window):
                                        xtal.from_seed(perturbed_atoms)
                                        xtal.to_file(match_cif)
                                        p_wr, p_r2, p_chi2, _ = refine_pxrd(match_csv, match_cif, INST_FILE)
                                        if p_wr is not None and p_r2 is not None and p_chi2 is not None:
                                            if p_r2 > r2 and p_chi2 < chi2:
                                                wr, r2, chi2 = p_wr, p_r2, p_chi2
                                                eng, eng_rel, sim = p_eng, p_eng_rel, p_sim
                                                stress, fmax = p_stress, p_fmax
                                                is_new_best_energy = p_is_new_best_energy
                                    msg += f" {sim:.3f}, {eng:.3f}, {stress:.3f}, {fmax:.3f}"
                                    msg += f" {wr:6.3f}, {r2:6.3f}, {chi2:6.3f}"
                                    if is_new_best_energy:
                                        msg += ' +++++'
                            else:
                                msg += f" {sim:.3f}, {eng:.3f}, {stress:.3f}, {fmax:.3f}"
                                msg += f" {wr:6.3f}, {r2:6.3f}, {chi2:6.3f}"
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

                    if wr is not None and (r2 > min_r2 or chi2 < max_chi2):
                        energy_ok = eng_rel <= max_eng_rel_for_termination
                        if refined_score is not None and refined_score > local_accepted_score and energy_ok:
                            local_accepted_score = refined_score
                            local_accepted_result = (wr, r2, chi2, xtal, eng_best, eng, eng_rel, struc_count)

                        if should_terminate(r2, chi2, eng_rel, min_r2, max_chi2, max_eng_rel_for_termination):
                            early_stop = True
                            logger.info(
                                f"***Excellent fit (r2={r2:.3f}, chi2={chi2:.3f}) with energy (eng_rel={eng_rel:.3f} eV/atom); "
                            )
                    if struc_count >= min_structures_before_early_stop and early_stop:
                        logger.info(
                            f"Early stop triggered after {struc_count} structures; terminating search."
                        )
                        return _finalize_result(local_accepted_result)

            prev_limit = limit
        # Correct the structure count in the final accepted result if it exists, to reflect the total number of structures generated so far.
    return _return_best_available(local_accepted_result, best_refined_result_energy_ok, struc_count)


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

    for match_csv in [#f'Examples/PXRD_Ba14Na14LiN6_225.csv',
                      #f'Examples/PXRD_Be2SiBi_119.csv',
                      #f'Examples/PXRD_K2SnO6_148.csv',
                      #f'Examples/PXRD_HgC2N2_122.csv',
                      #f'Examples/PXRD_Mg9Si5_176.csv',
                      #f'Examples/hardPXRD_HfTlCuS3_63.csv',
                      f'Examples/PXRD_CoO2_12.csv',
                    ]:
        formula, ref_spg = _infer_formula_spg(Path(match_csv))
        df = pd.read_csv(match_csv)
        x1 = df.iloc[:, 0].values
        y1 = df.iloc[:, 1].values
        data = RawDataManager(x1, y1, bg_subtract=False)
        data.get_peaks_from_scipy_adaptive()
        data.plot()
        min_abc = 2.0
        max_abc = 35.0
        solutions = SmartCellSolver(x1[data.peaks],
                        max_mismatch=30,
                        hkl_max=(4, 4, 4),
                        max_square=20,
                        total_square=25,
                        theta_tols=[0.1, 0.15, 0.5],
                        min_abc=min_abc,
                        max_abc=max_abc,
                        verbose=False,
                        )
        sols = [
            (sol['spg'], sol['cell'], sol['mismatch'], sol['chi2'][1], sol['errors'], sol['id'], sol['match'])
            for sol in solutions
        ]
        cells = CellManager.consolidate(sols,
                                        max_solutions=200,
                                        merge_tol=0.02,
                                        ref_spg=ref_spg,
                                        max_mismatch=20,
                                        sort_by='volume')
        print(f"Final consolidated solutions for {match_csv}\n")

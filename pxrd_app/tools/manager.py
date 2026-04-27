"""
Module for PXRD managers
- RawDataManager: process the raw PXRD data, background subtraction, peak finding and profiling
- CellManager: rank and consolidate cell solutions
- WPManager: sample Wyckoff positions according to constraints (spg, composition, density range)
- XtalManager: sample crystal structures from Spg, cell parameters and Wyckoff positions.
"""
from typing import Sequence
import numpy as np
from pandas import read_csv
from scipy.signal import find_peaks, savgol_filter
from scipy.stats.qmc import Halton, Sobol
import matplotlib.pyplot as plt
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pyxtal.symmetry import rf, Group
from pyxtal.database.element import Element


DEFAULT_BG_ORDER = 6
DEFAULT_BG_ITERS = 50
DEFAULT_BG_ASYM = 0.01
DEFAULT_SAVGOL_WINDOW = 4
DEFAULT_SAVGOL_POLYORDER = 3

class RawDataManager:
    def __init__(self, two_thetas: Sequence[float], intensities: Sequence[float],
                 bg_subtract: bool = True, smooth: bool = True):
        """
        RawDataManager is used to process the raw PXRD data, including
        - background subtraction
        - peak finding
        - profiling.

        Args:
            two_thetas (list): List of 2theta values
            intensities (list): Corresponding intensities
            bg_subtract (bool): Whether to perform background subtraction
            smooth (bool): Whether to apply smoothing
        """
        self.x = two_thetas
        self.y_raw = intensities
        if bg_subtract:
            self.y = self.background_subtraction(
                order=DEFAULT_BG_ORDER,
                n_iter=DEFAULT_BG_ITERS,
                asym=DEFAULT_BG_ASYM,
            )
        else:
            self.y = self.y_raw.copy()
        if smooth:
            self.y = savgol_filter(
                self.y,
                window_length=DEFAULT_SAVGOL_WINDOW,
                polyorder=DEFAULT_SAVGOL_POLYORDER,
            )


    def background_subtraction(self, order: int = 6, n_iter: int = 50,
                               asym: float = 0.01) -> np.ndarray:
        """
        Remove background from XRD data using asymmetric least squares polynomial fitting.

        Args:
            order (int): The order of the polynomial to fit.
            n_iter (int): Number of iterations for fitting.
            asym (float): Asymmetry parameter for weighting.
        """
        w = np.ones_like(self.y_raw)

        for _ in range(n_iter):
            p = np.polyfit(self.x, self.y_raw, order, w=w)
            baseline = np.polyval(p, self.x)
            # downweight points above baseline (likely peaks)
            w = asym * (self.y_raw > baseline) + (1 - asym) * (self.y_raw <= baseline)

        y_corrected = self.y_raw - baseline
        y_corrected[y_corrected < 0] = 0
        return y_corrected

    def get_peaks_from_scipy(self, height: float = 1.0, distance: int = 5,
                             prominence: float = 1.5) -> None:
        """
        Detect peaks in the processed XRD data using scipy's find_peaks function.
        Intentionally use small height and prominence thresholds to capture more peaks,
        including weak ones, which can be filtered later by ML or domain knowledge.

        Args:
            height (float): Minimum height of peaks.
            distance (int): Minimum distance between peaks.
            prominence (float): Minimum prominence of peaks.
        """
        self.peaks, _ = find_peaks(self.y, height=height, distance=distance, prominence=prominence)

    def filter_peaks_by_ml(self, threshold: float = 0.8, min_height: float = 5.0) -> None:
        """
        Filter detected peaks according to the probabilities predicted by a machine learning model.

        Args:
            xrd (list): List of XRD intensities corresponding to self.x.
            threshold (float): Probability threshold for keeping peaks.
        """
        from .peak_prediction import _predict_peaks
        y0 = (self.y - self.y.min()) / (self.y.max() - self.y.min() + 1e-6)
        self.probs = _predict_peaks(y0)

        # Collect indices to remove (iterate backwards to avoid index shifting issues)
        indices_to_remove = []
        for i in range(len(self.peaks) - 1, -1, -1):
            peak_idx = self.peaks[i]
            intensity = self.y[peak_idx]
            prob = self.probs[peak_idx]
            theta = self.x[peak_idx]
            print(f"Peak at index {peak_idx} ({theta:.2f}) with intensity {intensity:.2f} prob {prob:.3f}")
            if prob < threshold and intensity < min_height:
                indices_to_remove.append(i)
                print(f"Removed peak {peak_idx} with prob {prob:.3f} and intensity {intensity:.2f}")

        # Delete all marked peaks at once
        self.peaks = np.delete(self.peaks, indices_to_remove)


    def get_peaks_from_scipy_adaptive(self, distance: int = 5, height: float = 1.5,
                                      prominence: float = 2.0, max_peaks: int = 35,
                                      min_peaks: int = 15,
                                      heights_fallback: Sequence[float] | None = None,
                                      prominence_fallback: Sequence[float] | None = None) -> None:
        """
        Detect peaks in the processed XRD data using an adaptive thresholding approach.

        Handles three scenarios:
        1. Default settings (height=1.0, prominence=2): good for majority of cases
        2. Too many peaks detected: keep only the strongest max_peaks peaks
        3. Too few peaks detected: decrease height/prominence to add more peaks

        Args:
            distance (int): Minimum distance between peaks.
            height (float): Initial minimum height of peaks.
            prominence (float): Initial minimum prominence of peaks.
            max_peaks (int): Maximum number of peaks to keep (default 35).
            min_peaks (int): Minimum number of peaks desired (default 10).
            heights_fallback (list): Fallback heights to try if too few peaks.
            prominence_fallback (list): Fallback prominences to try if too few peaks.
        """
        if heights_fallback is None:
            heights_fallback = (0.5, 0.2)
        if prominence_fallback is None:
            prominence_fallback = (1.5, 1.0)

        # Scenario 1: Try default settings first
        self.peaks, _ = find_peaks(self.y, height=height, distance=distance, prominence=prominence)

        # Scenario 2: Too many peaks - keep strong peaks and fill up to max_peaks
        if len(self.peaks) > max_peaks:
            intensities = self.y[self.peaks]
            # Keep peaks with intensity > 2*height (strong peaks)
            strong_mask = intensities > 4.0 * height
            strong_peaks = self.peaks[strong_mask]

            if len(strong_peaks) >= max_peaks:
                # If strong peaks already exceed max_peaks, keep the strong peaks
                self.peaks = strong_peaks
            else:
                # Add weak peaks to reach max_peaks
                weak_peaks = self.peaks[~strong_mask]
                weak_intensities = intensities[~strong_mask]
                sorted_indices = np.argsort(weak_intensities)[::-1]
                n_weak_to_add = max_peaks - len(strong_peaks)
                weak_peaks_to_add = weak_peaks[sorted_indices[:n_weak_to_add]]
                self.peaks = np.sort(np.concatenate([strong_peaks, weak_peaks_to_add]))

            return

        # Scenario 3 (too few peaks) currently disabled to avoid introducing extra weak/noisy peaks.
        _ = (min_peaks, heights_fallback, prominence_fallback)

    def add_low_angle_peaks(self, angle_threshold: float = 35,
                            height_ratio: float = 0.25,
                            prominence_ratio: float = 0.15,
                            min_distance: int = 5,
                            max_additional_peaks: int = 10,
                            height_min: float = 1.25) -> None:
        """
        Check for and add missing peaks in the low-angle region with relaxed but adaptive criteria.
        Uses ratios relative to strongest peaks rather than absolute thresholds to avoid adding noise.

        Args:
            angle_threshold (float): Maximum 2theta angle to consider as "low angle" region.
            height_ratio (float): Minimum height as a fraction of max existing peak height.
            prominence_ratio (float): Minimum prominence as a fraction of max existing peak height.
            min_distance (int): Minimum distance between peaks in indices.
            max_additional_peaks (int): Maximum number of additional peaks to add.
            height_min (float): Minimum absolute height to consider for adding peaks.
        """
        if not hasattr(self, 'peaks'):
            raise ValueError("Peaks must be detected before adding low-angle peaks.")

        if len(self.peaks) == 0:
            return

        # Calculate adaptive thresholds based on the STRONGEST existing peaks (not median)
        existing_peak_heights = self.y[self.peaks]
        # Use 50th percentile to avoid outliers but still be conservative
        strong_peak_height = np.percentile(existing_peak_heights, 50)

        height_threshold = max(strong_peak_height * height_ratio, height_min)
        prominence_threshold = max(strong_peak_height * prominence_ratio, height_min)

        # Find indices in the low-angle region
        low_angle_mask = self.x <= angle_threshold
        low_angle_indices = np.where(low_angle_mask)[0]

        if len(low_angle_indices) == 0:
            return

        # Extract low-angle data
        low_angle_data = self.y[low_angle_mask]

        # Find all peaks in low-angle region with adaptive criteria
        low_angle_peaks, _ = find_peaks(
            low_angle_data,
            height=height_threshold,
            distance=min_distance,
            prominence=prominence_threshold
        )

        # Convert to global indices
        new_peaks_candidates = low_angle_indices[low_angle_peaks]

        # Filter out peaks that are already detected and rank by intensity
        peaks_to_add = []
        peak_intensities = []
        for new_peak_idx in new_peaks_candidates:
            # Check if this peak is already in the detected peaks
            if not np.any(np.abs(self.peaks - new_peak_idx) <= min_distance):
                peaks_to_add.append(new_peak_idx)
                peak_intensities.append(self.y[new_peak_idx])

        # Sort by intensity (strongest first) and limit to max_additional_peaks
        if len(peaks_to_add) > 0:
            sorted_indices = np.argsort(peak_intensities)[::-1]
            peaks_to_add = np.array(peaks_to_add)[sorted_indices[:max_additional_peaks]]

            # Add new peaks and re-sort
            self.peaks = np.sort(np.concatenate([self.peaks, peaks_to_add]))
            print(f"Added {len(peaks_to_add)} low-angle peaks. Total peaks: {len(self.peaks)}")

    def get_non_peak_integral(self, window: float = 0.05) -> None:
        """
        Calculate the integrals of non-peak regions for potential use in refinement or scoring.

        Args:
            window (float): Width of the window around each peak to exclude (in degrees).
        """
        if not hasattr(self, 'peaks'):
            raise ValueError("Peaks must be detected before calculating non-peak integrals.")
        non_peak_mask = np.ones_like(self.y, dtype=bool)

        # Exclude peak positions and their surrounding windows
        for peak_idx in self.peaks:
            peak_theta = self.x[peak_idx]
            # Mask out the region within ±window around each peak
            window_mask = (self.x >= peak_theta - window) & (self.x <= peak_theta + window)
            non_peak_mask[window_mask] = False

        self.non_peak_integral = np.trapz(self.y[non_peak_mask], self.x[non_peak_mask])

    def get_integrals_by_peaks(self, peaks: Sequence[float], window: float = 0.05) -> float:
        """
        Calculate the integrals around specified peaks for potential use in refinement or scoring.

        Args:
            peaks (list): List of peak values around which to calculate integrals.
            window (float): Width of the window around each index to consider for integration (in degrees).
        """
        integral = 0.0
        for peak in peaks:
            mask = (self.x >= peak - window) & (self.x <= peak + window)
            if np.sum(mask) > 1:  # Need at least 2 points for integration
                integral += np.trapz(self.y[mask], self.x[mask])
        return integral

    def plot(self, figname='xrd_debug.png', remark=None):
        """
        Plot the raw and processed XRD data along with detected peaks for debugging.

        Args:
            figname (str): Filename to save the plot.
            remark (str): Optional remark to display on the plot.
        """
        plt.figure(figsize=(8,5))
        plt.plot(self.x, self.y_raw, label='Raw Data')
        plt.plot(self.x, self.y, label='Processed Data')
        if hasattr(self, 'peaks'):
            plt.plot(self.x[self.peaks], self.y[self.peaks], "x", label=f'Detected Peaks ({len(self.peaks)})')
        if remark is not None:
            plt.title(remark)
        plt.xlabel('2θ (degrees)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
        plt.savefig(figname)
        plt.close()

    def to_csv(self, filename):
        """
        Save the processed XRD data to a CSV file.

        Args:
            filename (str): The name of the file to save the data to.
        """
        data = np.column_stack((self.x, self.y))
        np.savetxt(filename, data, delimiter=",", header="2theta,intensity", comments="")

    def plot_peaks_vs_cell_solutions(self, cell_solutions, match_ids=[],
                                     chi2_limit=None,
                                     title=None,
                                     figname='xrd_peaks_vs_cells.png',
                                     error_limit=0.25):
        """
        Plot the detected peaks against the calculated peak positions from cell solutions.

        Args:
            cell_solutions (list): List of CellManager objects representing cell solutions.
            match_ids (list): List of indices of matched cell solutions.
            chi2_limit (float): Upper limit for chi-squared values on the plot.
            error_limit (float): Maximum error to display on the plot.
            ref_cell (list or None): Reference cell parameters for comparison.
        """
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                              width_ratios=[1.5, 1],
                              height_ratios=[1.5, 1])
        if title is not None: plt.suptitle(title)

        # [0, 0]: raw/processed data and detected peaks
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(self.x, self.y_raw, label='Raw Data', alpha=0.7)
        ax00.plot(self.x, self.y, label='Processed Data', alpha=0.7)
        if hasattr(self, 'peaks'):
            ax00.plot(self.x[self.peaks], self.y[self.peaks], "x", alpha=0.4,
                      label=f'Detected Peaks ({len(self.peaks)})')
        ax00.set_xlabel('2θ (degrees)')
        ax00.set_ylabel('Intensity (a.u.)')
        ax00.legend()
        ax00.set_xlim(self.x.min()-0.5, self.x.max()+0.5)

        # [1, 0]: Best solution peaks vs errors
        if not hasattr(self, 'non_peak_integral'): self.get_non_peak_integral()
        ax10 = fig.add_subplot(gs[1, 0])
        if match_ids:
            for match_id in match_ids:
                solution = cell_solutions[match_id]
                chi2 = f"χ²: {solution.chi2:.4f}"
                errs = ' '.join([f"{e:.3f}" for e in solution.errors])
                theta_calc = np.array([m[1] for m in solution.match])
                theta_obs = np.array([m[2] for m in solution.match])
                theta_miss = np.array([m[1] for m in solution.mismatch])
                ratio = 100*self.get_integrals_by_peaks(theta_miss)/self.non_peak_integral
                N_miss = len(solution.mismatch)
                label = f'Match {solution.size:.1f} {chi2} {errs} ({N_miss}/{ratio:.2f})'
                ax10.plot(theta_obs, theta_calc-theta_obs, 'o', alpha=0.3, label=label)
                ax10.bar(theta_miss, width=0.1, height=0.5*error_limit, alpha=0.5)

        ax10.axhline(0, color='gray', linestyle='--')
        if len(cell_solutions) > 0:
            chi2s = [c.chi2 for c in cell_solutions]
            ids = np.argsort(chi2s)
            for id in ids:
                if id not in match_ids:
                    solution = cell_solutions[id]
                    chi2 = f"χ²: {solution.chi2:.4f}"
                    errs = ' '.join([f"{e:.3f}" for e in solution.errors])
                    theta_calc = np.array([m[1] for m in solution.match])
                    theta_obs = np.array([m[2] for m in solution.match])
                    theta_miss = np.array([m[1] for m in solution.mismatch])
                    ratio = self.get_integrals_by_peaks(theta_miss)/self.non_peak_integral
                    N_miss = len(solution.mismatch)
                    label = f'Other {solution.size:.1f} {chi2} {errs} ({N_miss}/{ratio:.2f})'
                    ax10.plot(theta_obs, theta_calc-theta_obs, 'o', alpha=0.3, label=label)
                    ax10.bar(theta_miss, width=0.1, height=-0.5*error_limit, alpha=0.5)
                    break
        else:
            ax10.text(0.5, 0.5, 'No matched solutions', ha='center', va='center')
        ax10.set_xlabel('Observed 2θ (degrees)')
        ax10.set_ylabel('2θ differences (degrees)')
        ax10.set_xlim(self.x.min()-0.5, self.x.max()+0.5)
        ax10.set_ylim(-error_limit, error_limit)
        ax10.legend()

        # [:, 1]: χ² vs volume scatter (spanning both rows)
        ax_chi2 = fig.add_subplot(gs[:, 1])
        if len(cell_solutions) == 0:
            ax_chi2.text(0.5, 0.5, 'No cell solutions', ha='center', va='center')
        else:
            vols = [c.size for c in cell_solutions]
            chi2s = [c.chi2 for c in cell_solutions]
            ax_chi2.scatter(vols, chi2s, alpha=0.6, label='All solutions')
            if match_ids:
                match_set = set(match_ids)
                match_vols = [v for i, v in enumerate(vols) if i in match_set]
                match_chi2s = [c for i, c in enumerate(chi2s) if i in match_set]
                if match_vols:
                    ax_chi2.scatter(match_vols, match_chi2s, color='red', marker='x', s=60, label='Matched')
        ax_chi2.set_xlabel('Cell volume (A^3)')
        ax_chi2.set_ylabel('Chi^2 (peaks)')
        if chi2_limit is not None: ax_chi2.set_ylim(-0.001, chi2_limit)
        ax_chi2.legend()

        fig.savefig(figname)
        plt.close(fig)

    def plot_peaks_vs_cell_solution(self, solution, title=None, remark=None,
                                     figname='xrd_peaks_vs_cells.png'):
        """
        Plot the detected peaks against the calculated peak positions from cell solutions.

        Args:
            cell_solutions (list): List of CellManager objects representing cell solutions.
            match_ids (list): List of indices of matched cell solutions.
            chi2_limit (float): Upper limit for chi-squared values on the plot.
            ref_cell (list or None): Reference cell parameters for comparison.
        """
        from .solver import get_two_theta_from_cell
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.3})
        thetas = self.x[self.peaks]
        if title is not None: plt.suptitle(title)

        # Left: raw/processed data and detected peaks
        axes[0].plot(self.x, self.y_raw, label='Raw Data')
        axes[0].plot(self.x, self.y, label='Processed Data')
        if hasattr(self, 'peaks'):
            axes[0].plot(thetas, self.y[self.peaks], "x",
                         label=f'Detected Peaks ({len(self.peaks)})')
        axes[0].set_xlabel('2θ (degrees)')
        axes[0].set_ylabel('Intensity (a.u.)')
        if remark is not None:
            axes[0].text(0.5, 0.9, remark, ha='center', va='center', transform=axes[0].transAxes)
        axes[0].legend()


        # Right: χ² vs volume scatter
        matched_peaks = solution['match']
        hkl_list = [m[0] for m in matched_peaks]
        thetas = np.array([m[2] for m in matched_peaks])
        calc_thetas, _ = get_two_theta_from_cell(solution['bravais'],
                                                 np.array(hkl_list),
                                                 solution['cell'],
                                                 solution['wave_length'],
                                                 False)
        #print(np.abs(calc_thetas - thetas)) #.max())
        axes[1].plot(thetas, calc_thetas-thetas, 'o', alpha=0.3)
        cell_str = ', '.join([f'{p:.4f}' for p in solution['cell']])
        for i in range(3):
            axes[1].bar(thetas[i], calc_thetas[i]-thetas[i], label=f'Peak {i+1}: {hkl_list[i]}', width=0.1, alpha=0.5)
        axes[1].set_title(f'Solution Cell: {cell_str}, χ²: {solution["chi2"][0]:.4f} {solution["chi2"][1]:.4f}')
        axes[1].set_xlabel('Observed 2θ (degrees)')
        axes[1].set_ylabel('Calculated 2θ (degrees)')

        fig.savefig(figname)
        plt.close(fig)

class CellManager:
    def __init__(self, spg, params, mismatch, chi2, errors, hkl, match):
        """
        Cell Manager is used to handle cell parameter related operations.

        Args:
            spg (int): Space group number
            params (list): Cell parameters
            mismatch (list): List of mismatches for matched peaks
            chi2 (float): Chi-squared value for peak matching
            errors (list): List of errors for matched peaks
            hkl (any): Unique identifier for the solution
            match (list): List of matched peaks
        """
        # Store raw parameters
        self.raw_params = params
        self.dims = np.array(params)
        self.missing = len(mismatch)
        self.mismatch = mismatch
        self.chi2 = chi2
        self.errors = 0 if errors is None else errors
        self.d2 = 0 if hkl is None else np.sum(hkl**2)
        self.group = Group(spg)
        self.spg = spg
        lattice = Lattice.from_1d_representation(self.dims, self.group.lattice_type)
        self.size = lattice.volume
        self.match = match

    def is_similar_to(self, other, tol=0.05, a_tol=5.0):
        if self.spg != other.spg:
            return False
        else:
            return self.is_similar_to_cell(other.dims, tol, a_tol)

    def is_similar_to_cell(self, other, tol=0.05, a_tol=5.0):
        """
        Instance method:
        Check if 'self' is nearly identical to 'other' (duplicate).
        Reduced angle tolerance from 5.0° to 3.0° for stricter merging.
        """
        if len(self.dims) <= 2:
            diff = np.abs(self.dims - other) / other
            return np.all(diff < tol)
        elif len(self.dims) == 3:
            if self.spg in [16, 19, 21, 22, 23, 24, 47, 48, 69, 70, 71]:
                axis_orders = [(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 0, 1), (1, 2, 0), (2, 1, 0)]
            elif self.spg in [17, 18, 20, 25, 26, 27, 34, 35, 36, 37, 42,
                              43, 44, 49, 51, 52, 54, 56, 58, 59, 65,
                              66, 67, 68, 74]: # c is the special, a/b same
                axis_orders = [(0, 1, 2), (1, 0, 2)]
            elif self.spg in [38, 40]: # a is the special, b/c permutable
                axis_orders = [(0, 1, 2), (0, 2, 1)]
            elif self.spg in [41]: # b is the special, a/c permutable
                axis_orders = [(0, 1, 2), (2, 1, 0)]
            else: # a/b/c not permutable like Pnma
                axis_orders = [(0, 1, 2)]
            #print(self.spg, axis_orders)
            for order in axis_orders:
                cell = [self.dims[i] for i in order]
                diff = np.abs(cell - other) / other
                if np.all(diff < tol):
                    return True
            return False
        elif len(self.dims) == 4: # monoclinic
            angle_diff = np.abs(self.dims[3:] - other[3:])
            if self.spg in [3, 4, 6, 10, 11]:
                axis_orders = [(0, 1, 2), (2, 1, 0)]
            else:
                axis_orders = [(0, 1, 2)]
            for order in axis_orders:
                cell = [self.dims[i] for i in order]
                abc_diff = np.abs(cell - other[:3]) / other[:3]
                if np.all(abc_diff < tol) and np.all(angle_diff < a_tol):
                    return True
                #else:
                #    # b-axis same and a/c
                #    if abs(self.dims[1] - other.dims[1]) < tol and \
                #        abs(self.size_proxy - other.size_proxy) / other.size_proxy < tol:
                #        return True

            return False

    def __repr__(self):
        spg_str = f"{self.spg:3d}"
        dims_str = f"{' '.join([f'{x:6.3f}' for x in self.dims]):<28}|"
        missing_str = f"  {self.d2:2d}/{self.missing:2d}  |"
        size_str = f" {self.size:<8.1f}|"
        chi2_str = f" {self.chi2:<6.4f}|"
        spg_str = f" {self.spg:<3} |"
        error_str = ' '.join([f'{x:5.3f}' for x in self.errors]) + ' |'
        return spg_str + dims_str + chi2_str + missing_str + size_str + error_str

    @classmethod
    def consolidate(cls, raw_data, merge_tol=0.15, max_solutions=10, ref_cell=None,
                    verbose=False, debug=False, ref_spg=None, ref_volume=None,
                    max_mismatch=30, chi2_tie_tol=5e-4, sort_by='chi2'):
        """
        Class method: Takes raw list of [spg, dims, missing, chi2, d2, error], instantiates objects,
        sorts, merges duplicates, removes supercells, and returns the clean list.

        Args:
            raw_data (list): List of [spg, dims, missing, chi2, d2, error] entries
            merge_tol (float): Tolerance for merging duplicates
            max_solutions (int): Maximum number of solutions to retain
            ref_cell (np.array): np.array of reference cell parameters to match against (optional)
            verbose (bool): Whether to print detailed consolidation steps
            debug (bool): Whether to print debug information
            ref_spg (int): Reference space group number (optional)
            ref_volume (float): Reference cell volume (optional)
            max_mismatch (int): Maximum allowed mismatch (optional)
            sort_by (str): Sort order used during consolidation and for returned solutions. Options:
                - 'chi2' (default): sort by chi2 / quality
                - 'volume': sort by cell volume ascending

        Returns:
            kept_solutions (list): Consolidated list of CellManager objects
        """
        # 1. Instantiate objects
        spg = raw_data[0][0]
        solutions = [cls(d[0], d[1], d[2], d[3], d[4], d[5], d[6]) for d in raw_data]
        if ref_cell is not None:
            ref_lattice = Lattice.from_1d_representation(ref_cell, Group(spg).lattice_type)
            ref_volume = ref_lattice.volume
        # 2. Sort with chi2 tie-bucketing:
        #    when chi2 values are very close (within `chi2_tie_tol`), prioritize
        #    fewer missing peaks before fine-grained chi2 differences.
        def _chi2_bucket(value):
            if chi2_tie_tol is None or chi2_tie_tol <= 0:
                return value
            return int(np.round(value / chi2_tie_tol))

        sort_mode = str(sort_by).lower()
        if sort_mode not in ('chi2', 'volume'):
            raise ValueError("sort_by must be either 'chi2' or 'volume'")

        if sort_mode == 'volume':
            # Volume-first ranking (then quality tie-breakers).
            solutions.sort(key=lambda x: (x.size, x.chi2, x.missing, -x.spg))
        elif ref_cell is not None:
            # Reference-cell mode remains volume-prioritized unless overridden.
            solutions.sort(key=lambda x: (x.size, x.chi2, x.missing, -x.spg))
        else:
            solutions.sort(key=lambda x: (_chi2_bucket(x.chi2), x.missing, x.chi2, x.size, -x.spg))

        kept_solutions = []
        indices_to_skip = set()
        match_cell = False
        if verbose:
            print(f"{'Status':<6} | SPG | {'Dims (Sorted)':<27}| {'Chi2':<6}| {'Missing':<8}| {'Volume':<8}| {'Errors':<20}")
        for i in range(len(solutions)):
            if i in indices_to_skip: continue
            base = solutions[i]
            cubic_max_mismatch = min(max_mismatch, 12)   # cubic: same max_mismatch, capped at 12
            if base.spg >= 195 and base.missing > cubic_max_mismatch: continue
            if base.spg < 195 and base.missing > max_mismatch: continue

            strs = f"{'KEEP':<6} |{str(base)} "

            if ref_spg is not None and base.spg == ref_spg:
                strs += f'+++++Matched SPG'
            if ref_volume is not None and abs(base.size - ref_volume) / ref_volume < merge_tol * 2.5:
                strs += f'++volume'
            if ref_cell is not None:
                if base.is_similar_to_cell(ref_cell, tol=merge_tol):
                    strs += f'+++++Matched cell'
                    match_cell = True
                #elif base.size / ref_volume > 1.1:
                #    print(f"Cell volume {base.size:.1f} is larger than reference {ref_volume:.1f}. Stop.")
                #    break
            if verbose:
                print(strs)

            kept_solutions.append(base)
            if verbose and len(kept_solutions) >= max_solutions:# or match_vol:
                print(f"Reached maximum of {max_solutions} solutions, stop.")
                break

            # Sweep through the rest of the list to find duplicates or supercells
            for j in range(i + 1, len(solutions)):
                if j in indices_to_skip:
                    continue

                candidate = solutions[j]
                # CHECK 1: Merge (Duplicate)
                if candidate.is_similar_to(base, tol=merge_tol):
                    indices_to_skip.add(j)

                    if candidate.chi2 < base.chi2:
                        if verbose:
                            print(f"{'SWAP':<6} | {str(candidate)} | Better (replacing)")
                        # Replace the last added solution (which was the old base)
                        kept_solutions[-1] = candidate
                        base = candidate
                    else:
                        # Otherwise, just drop the candidate
                        if verbose:
                            print(f"{'MERGE':<6} | {str(candidate)} | Similar (dropped)")
                    continue
        if verbose and len(kept_solutions) > 0:
            print(f"Consolidation: {len(kept_solutions)} unique solutions from {len(raw_data)} entries.")
        if debug and not match_cell and ref_cell is not None:
            print("Warning: No solutions matched the reference volume criteria.")
            import sys; sys.exit()

        # Optional resort of final solutions
        if sort_mode == 'volume':
            kept_solutions.sort(key=lambda x: x.size)
            print(f"Solutions resorted by volume.")

        if ref_cell is not None:
            return kept_solutions, match_cell
        else:
            return kept_solutions


class WPManager:

    @staticmethod
    def find_wp_assignments(comp, ids, nums):
        """
        Assigns Wyckoff position IDs to a composition. This function can handle
        cases where a composition number is a sum of multiple Wyckoff multiplicities.

        Args:
            comp (list): The target composition counts, e.g., [18, 6].
            ids (list): The list of Wyckoff position IDs, e.g., [1, 1, 6, 8, 9].
            nums (list): The corresponding multiplicities for each ID, e.g., [8, 8, 4, 2, 2].

        Returns:
            list: A list of all possible valid assignments. Each assignment is a list
                  of lists, where each inner list contains the WP IDs for an element.
        """

        # Group identical WPs by (id, multiplicity) to avoid combinatorial
        # explosion from choosing k-of-n identical items.  Instead of enumerating
        # C(n,k) index-permutations, we just decide "pick k from this group".
        from collections import Counter
        group_counts = Counter(zip(ids, nums))
        # Sorted descending by multiplicity for early pruning
        groups = sorted(
            [(wp_id, mult, count) for (wp_id, mult), count in group_counts.items()],
            key=lambda x: x[1], reverse=True,
        )

        def find_subsets_grouped(target, grps):
            """All distinct multi-sets of WP types whose multiplicities sum to target.
            grps: list of (wp_id, mult, available_count), sorted desc by mult.
            Returns list of lists of (group_index, n_picked).
            """
            n = len(grps)
            # Suffix max-sum for pruning
            max_sum = [0] * (n + 1)
            for i in range(n - 1, -1, -1):
                max_sum[i] = max_sum[i + 1] + grps[i][1] * grps[i][2]

            results = []
            current = []

            def _inner(remaining, start):
                if remaining == 0:
                    results.append(current[:])
                    return
                if start >= n or max_sum[start] < remaining:
                    return
                _wp_id, mult, count = grps[start]
                max_pick = min(count, remaining // mult) if mult <= remaining else 0
                # k = 0 (skip this group)
                _inner(remaining, start + 1)
                # k = 1..max_pick
                for k in range(1, max_pick + 1):
                    current.append((start, k))
                    _inner(remaining - k * mult, start + 1)
                    current.pop()

            _inner(target, 0)
            return results

        sorted_indexed_comp = sorted(enumerate(comp), key=lambda x: x[1], reverse=True)

        seen = set()
        solutions = []
        stack = []

        def solve(comp_targets, grps):
            if not comp_targets:
                sorted_stack = sorted(stack, key=lambda x: x[0])
                canonical = tuple(tuple(sorted(part)) for _, part in sorted_stack)
                if canonical not in seen:
                    seen.add(canonical)
                    solutions.append([list(part) for _, part in sorted_stack])
                return

            orig_idx, target = comp_targets[0]
            rest = comp_targets[1:]

            for picks in find_subsets_grouped(target, grps):
                # Build id list for this element
                part = []
                for gi, n_picked in picks:
                    part.extend([grps[gi][0]] * n_picked)

                # Reduce consumed counts, drop empty groups
                picks_dict = dict(picks)
                new_grps = []
                for i, (wp_id, mult, count) in enumerate(grps):
                    rem = count - picks_dict.get(i, 0)
                    if rem > 0:
                        new_grps.append((wp_id, mult, rem))

                stack.append((orig_idx, part))
                solve(rest, new_grps)
                stack.pop()

        solve(sorted_indexed_comp, groups)
        return solutions

    def __init__(self, spg, cell, composition, max_wp=9, max_Z=24, max_dof=10, max_atoms=200,
                 ref_den=None, csv='database/spg_num_wps_mp.csv'):
        """
        WP Manager is used to infer likely Wyckoff positions from the given space group,
        cell, composition, and density constraint.

        Args:
            spg (int): Space group number
            cell (list): Cell parameters
            composition (dict): Elemental composition (e.g., {'Si': 1, 'O': 2})
            max_wp (int): Maximum number of Wyckoff positions to consider
            max_Z (int): Maximum Z value to consider for volume estimation
            max_dof (int): Maximum degrees of freedom to consider
            max_atoms (int): Maximum number of atoms in the unit cell to consider
            ref_den (float): Reference density to use for Z estimation (optional)
            csv (str): Path to the CSV file containing Wyckoff position data
        """
        df = read_csv(rf("pyxtal", csv))
        self.spg = spg
        self.df = df[df['spg'] == self.spg]
        self.cell = cell
        self.composition = composition
        self.comp = [composition[key] for key in composition.keys()]
        self.group = Group(spg)
        self.orders = self.group.get_orders()
        self.lattice = Lattice.from_1d_representation(cell, self.group.lattice_type)
        self.max_wp = max_wp
        self.max_dof = max_dof
        self.max_atoms = max_atoms
        volume = self.lattice.volume

        # Calculate total number of atoms in formula unit
        vol = [0, 0]

        if ref_den is not None:
            # Use provided reference density
            if len(ref_den) == 1:
                ref_den_min = 0.75 * ref_den
                ref_den_max = 1.25 * ref_den
            else:
                ref_den_min, ref_den_max = ref_den[0], ref_den[1]
            total_mass = sum([composition[el] * Element(el).mass for el in composition.keys()])
            vol_ref_max = total_mass / ref_den_min * 1.66054  # in Å³
            vol_ref_min = total_mass / ref_den_max * 1.66054  # in Å³
            vol = [vol_ref_min, vol_ref_max]
        else:
            for el in composition.keys():
                sp = Element(el)
                vol1 = 0.8 * sp.covalent_radius**3 * np.pi * 4 / 3
                vol2 = 4 * sp.covalent_radius**3 * np.pi * 4 / 3
                vol[0] += composition[el] * vol1
                vol[1] += composition[el] * vol2
        self.Zs = (int(np.ceil(volume/vol[1])), int(np.floor(volume/vol[0])))
        # Adjust Z range based on max_atoms constraint
        if self.Zs[1] * sum(self.comp) > self.max_atoms:
            self.Zs = (self.Zs[0], self.max_atoms // sum(self.comp))
        if self.Zs[1] > max_Z:
            self.Zs = (self.Zs[0], max_Z)
        if self.Zs[0] - self.Zs[1] == 1:
            Z = int(np.round(volume/((vol[0]+vol[1])/2)))
            self.Zs = (Z, Z)
        # print(f"Estimated Z range: {self.Zs}, Vol: {volume:.2f}, Vol bounds: [{vol[0]:.2f}, {vol[1]:.2f}]")

    def get_wyckoff_positions_general(self):
        """
        Infer possible Wyckoff position combinations based on the composition and Z range.
        """
        sols = []
        for Z in range(self.Zs[0], self.Zs[1]+1):
            sols_before_z = len(sols)
            comp = [n * Z for n in self.comp]
            wps, _, ids = self.group.list_wyckoff_combinations(comp, numWp=(0, self.max_wp))
            indices = sorted(range(len(ids)), key=lambda i: sum(len(x) for x in ids[i]))
            ids = [ids[i] for i in indices]
            wps = [wps[i] for i in indices]
            if len(ids) > 0:
                #print(f"Z={Z}: Found {len(ids)} Wyckoff position combinations.")
                wp_lists = []
                for id in ids:
                    # Check if the combination is alternative to existing ones
                    duplicate = False
                    tmp = [len(self.group)-1-item for sublist in id for item in sublist]
                    for order in self.orders:
                        tmp_list = order[tmp].tolist()
                        #print("Checking order:", wps[i], tmp, tmp_list)
                        if tmp_list in wp_lists:
                            duplicate = True
                            #print(wps[i], "is duplicate")
                            break
                    if not duplicate:
                        dof = [self.group[wp].get_dof() for sublist in id for wp in sublist]
                        wp_lists.append(tmp)
                        sols.append((self.spg, comp, self.lattice, id, len(tmp), sum(dof)))
                        #print("Added:", self.spg, comp, id)
            #kept_this_z = len(sols) - sols_before_z
            #if kept_this_z > 0:
            #    print(f"Z={Z}: Kept {kept_this_z} Wyckoff combinations in {self.spg}/{self.composition}.")
            # sort sols by DOF and number of WPs
            sols = sorted(sols, key=lambda x: (x[5], x[4]))
        #for sol in sols:
        #    wp_labels = [[self.group[w].get_label() for w in wp] for wp in sol[3]]
        #    # print(f"SPG: {sol[0]}, WPs: {wp_labels}, Num WPs: {sol[4]}, DOF: {sol[5]}")
        return sols

    def get_wyckoff_positions(self, verbose=False, max_samples=None, timing=False):
        """
        Infer possible Wyckoff position combinations based on the composition and Z range.

        Args:
            verbose: Print enumeration progress
            max_samples: If set, limit total enumeration to this many samples (for cost estimation)
            timing: If True, print per-Z timing breakdown for each step
        """
        from time import time as _time
        sols = []
        enumeration_count = 0  # Track enumeration progress
        #print(f"Enumerating Wyckoff position combinations for {self.spg}/{self.composition} with Z in {self.Zs}...")
        for Z in range(self.Zs[0], self.Zs[1]+1):
            sols_before_z = len(sols)
            df_z = self.df[self.df['n_atoms'] == Z * sum(self.comp)]
            if len(df_z) == 0: continue

            comp = [n * Z for n in self.comp]
            wp_lists = []

            t_assign = 0.0
            t_dup = 0.0
            n_raw_sols = 0

            for _, row in df_z.iterrows():
                # Early exit if we've exceeded max_samples during cost estimation
                if max_samples is not None and enumeration_count >= max_samples:
                    if verbose:
                        print(f"Z={Z}: Enumeration limited to {max_samples} samples (cost estimation mode).")
                    break

                ids = [int(x) for x in row['wps'].split('-')]
                count = row['count']
                nums = [self.group[id].multiplicity for id in ids]
                if len(ids) > self.max_wp:
                    continue

                # Pre-filter: every solution from a row has the same total DOF
                # (all WPs are always fully assigned), so skip the whole row early.
                total_dof = sum(self.group[id].get_dof() for id in ids)
                if total_dof > self.max_dof:
                    continue
                n_wps = len(ids)  # also constant per row

                if timing: _t0 = _time()
                solutions = self.find_wp_assignments(comp, ids, nums)
                if timing: t_assign += _time() - _t0
                n_raw_sols += len(solutions)

                for sol in solutions:
                    enumeration_count += 1
                    if max_samples is not None and enumeration_count >= max_samples:
                        break

                    duplicate = False
                    tmp_lists = [[] for _ in range(len(self.orders))]

                    if timing: _t1 = _time()
                    for i, order in enumerate(self.orders):
                        for sublist in sol:
                            items = [len(self.group)-1-item for item in sublist]
                            tmp = order[items]
                            tmp.sort()
                            tmp_lists[i].extend(tmp.tolist())

                        if tmp_lists[i] in wp_lists:
                            duplicate = True
                            break
                    if timing: t_dup += _time() - _t1

                    if not duplicate:
                        wp_lists.append(tmp_lists[0])
                        sols.append((self.spg, comp, self.lattice, sol, n_wps, total_dof, count, Z))

            #kept_z = len(sols) - sols_before_z
            #if kept_z > 0 and verbose:
            #    print(f"Z={Z}: Kept {kept_z} Wyckoff position combinations.")
            if timing:
                print(f"  Z={Z}: rows={len(df_z)}, raw_sols={n_raw_sols}, kept={kept_z} "
                      f"| t_assign={t_assign:.4f}s  t_dup={t_dup:.4f}s")

            # Exit outer loop if we hit max_samples
            if max_samples is not None and enumeration_count >= max_samples:
                break

        # Sort the solutions by count and DOF
        sols = sorted(sols, key=lambda x: (x[7], -x[6], x[5]))
        return sols

class XtalManager:
    def __init__(self, spg, species, numIons, cell, WPs, use_seeds=False,
                 qrs_method='sobol'):
        """
        Crystal Manager is used to handle crystal structure related operations.

        Args:
            spg (int): Space group number
            species (list): List of atomic species
            numIons (list): Number of ions for each species
            cell (list): Cell parameters
            WPs (list): Wyckoff positions
            use_seeds (bool): Whether to use seed structures for generation
            qrs_method (str): Quasi-random sampler to use when seed structures are enabled.
        """
        self.spg = Group(spg)
        self.WPs = WPs
        self.cell = cell
        self.species = species#; print(f"  Species: {self.species}")
        self.numIons = numIons
        dof = 0
        sites = [[] for _ in range(len(species))]
        sites_flat = []
        elements_flat = []
        for i, wp in enumerate(WPs):
            for _wp in wp:
                dof += self.spg[_wp].get_dof()
                sites[i].append(self.spg[_wp].get_label())
                sites_flat.append(self.spg[_wp].get_label())
                elements_flat.append(list(species)[i])
        self.dof = dof
        self.sites = sites
        self.sites_flat = sites_flat
        self.elements_flat = elements_flat
        if use_seeds:
            n_seeds = 7 * dof + 2
            method = str(qrs_method).strip().lower()
            if method == 'halton':
                _sampler = Halton(d=max(dof, 1), scramble=False)
                self.seeds = _sampler.random(n=n_seeds)
            elif method == 'sobol':
                _sampler = Sobol(d=max(dof, 1), scramble=False)
                # Sobol balance properties require a power-of-two sample count.
                m = int(np.ceil(np.log2(max(n_seeds, 1))))
                self.seeds = _sampler.random_base2(m=m)[:n_seeds]
            else:
                raise ValueError(f"Unsupported qrs_method={qrs_method!r}; expected 'sobol' or 'halton'.")
            self.skips = 0
            #print(f"Using {len(self.seeds)} seed structures for generation.")
        else:
            self.seeds = None

    def generate_structure(self, idx=0):
        """
        Generate the crystal structure from the Wyckoff positions and cell parameters.

        Args:
            cell (list): Cell parameters
            use_asu (bool): Whether to use asymmetric unit for generation
        """
        xtal = pyxtal()
        if self.seeds is not None:
            skips = self.skips
            for id in range(idx + skips, len(self.seeds)):
                if self.dof > 0:
                    x = self.cell.encode() + self.seeds[id].tolist()
                else:
                    x = self.cell.encode()
                #print(f"Generating: {idx}, {x}, {self.spg.number}")
                xtal.from_spg_wps_rep(self.spg.number, self.sites_flat, x, self.elements_flat)
                if len(xtal.check_short_distances(r=0.75)) > 0:
                    self.skips += 1
                else:
                    break
        else:
            xtal.from_random(3, self.spg, self.species, self.numIons,
                         lattice=self.cell, sites=self.sites,
                         force_pass=True,
                         #use_asu=use_asu,
                         t_factor=0.8,
                         )
        return xtal


if __name__ == "__main__":
    #data = [([9, 3], [2, 2, 2, 2], [3, 3, 3, 3], 1),
    #        ([4, 4, 4, 2, 4], [10, 10, 11, 12, 13], [4, 4, 4, 4, 2], 4),
    #       ]
    #for d in data:
    #    comp, ids, nums, n = d
    #    sols = WPManager.find_wp_assignments(comp, ids, nums)
    #    print('input: ', d, '\n', sols)
    #wp = WPManager(164, [6.065, 17.283], {'Al': 13, 'Ba': 7}, max_wp=10, ref_den=(3.24, 4.44))
    #sols = wp.get_wyckoff_positions()
    #for sol in sols:
    #    wp_labels = [[wp.group[w].get_label() for w in _wp] for _wp in sol[3]]
    #    print(f"SPG: {sol[0]}, Comp: {sol[1]}, WPs: {wp_labels}, DOF: {sol[5]}, Count: {sol[6]}, Z: {sol[7]}")
    from time import time
    spgs, cell, comp, ref_den = [63, 64], [32.42842875, 2.26406458, 9.41050049], {'B': 2, 'Be': 1, 'C': 2}, (1.04, 3.80)
    spgs, cell, comp, ref_den = [142], [7.53540996, 14.84882202], {'Er': 1, 'B': 4, 'Rh': 4}, (9.13, 10.33)
    for spg in spgs:
        print(f"\n--- SPG {spg} ---")
        t0 = time()
        wp = WPManager(spg, cell, comp, max_wp=15, max_Z=36, max_dof=21, ref_den=ref_den,
                       csv='database/spg_num_wps_raw.csv')
        t_init = time() - t0
        print(f"  Init: {t_init:.4f}s")
        t1 = time()
        sols = wp.get_wyckoff_positions(timing=True)
        t_wp = time() - t1
        print(f"  get_wyckoff_positions total: {t_wp:.4f}s")
        for sol in sols:
            wp_labels = [[wp.group[w].get_label() for w in _wp] for _wp in sol[3]]
        if len(sols) > 0:
            print(f"SPG: {sol[0]}, Comp: {sol[1]}, WPs: {wp_labels}, DOF: {sol[5]}, Count: {sol[6]}, Z: {sol[7]}, T: {time()-t0}")
        else:
            print(f"SPG: {spg}, No Wyckoff solutions found for the given composition and cell.")

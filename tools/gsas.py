"""
Simulate and refine PXRD patterns using GSAS-II.
"""
import os
import sys
import json
import atexit
import warnings
import uuid
import glob
import queue as _queue_mod
import numpy as np
from importlib import import_module
from multiprocessing import Process, Queue, get_context

# Use 'spawn' so each subprocess gets a fresh Python interpreter
# (fork would inherit the corrupted GSAS-II module state).
_mp_ctx = get_context("spawn")

# Suppress GSAS-II and pydantic warnings
warnings.filterwarnings("ignore", message=".*Importing GSASIIscriptable as a top level module is deprecated.*")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import matplotlib.pyplot as plt


def _get_tmp_root() -> str:
    return os.getenv("PXRD_TMP_ROOT", "tmp")


def _load_gsas_scriptable():
    """Load GSAS-II scriptable API from common import locations."""
    gsas_candidates = []

    gsas_path_env = os.getenv("GSASII_PATH", "").strip()
    if gsas_path_env:
        gsas_candidates.extend([gsas_path_env, os.path.join(gsas_path_env, "GSASII")])

    # Common local install locations from the official gitstrap workflow.
    home = os.path.expanduser("~")
    gsas_candidates.extend([
        os.path.join(home, "GSAS-II"),
        os.path.join(home, "GSAS-II", "GSASII"),
    ])

    for candidate in gsas_candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)

    errors = []
    for modname in ("GSASIIscriptable", "GSASII.GSASIIscriptable"):
        try:
            return import_module(modname)
        except Exception as exc:
            errors.append(f"{modname}: {exc}")
    msg = (
        "GSAS-II is not available in this environment. "
        "Install GSAS-II and ensure GSASIIscriptable is importable. "
        "If GSAS-II is installed in a custom location, set GSASII_PATH to that directory. "
        "Tried imports: " + " | ".join(errors)
    )
    raise ModuleNotFoundError(msg)


def check_gsas_available():
    """Return (ok, message) describing GSAS-II availability."""
    try:
        _load_gsas_scriptable()
        # Import a compiled GSAS extension to ensure binary compatibility.
        try:
            import pyspg  # type: ignore  # noqa: F401
        except Exception as exc:
            return False, (
                "GSAS-II scriptable API imported, but binary module load failed "
                f"(pyspg): {exc}"
            )
        return True, "GSAS-II is available."
    except Exception as exc:
        return False, str(exc)


class RedirectG2Output:
    """Context manager to redirect G2sc output to a file."""
    def __init__(self, log_file='gsas_refinement.log'):
        self.log_file = log_file
        self.original_stdout = None
        self.original_stderr = None
        self.file_handle = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.file_handle = open(self.log_file, 'a')
        sys.stdout = self.file_handle
        sys.stderr = self.file_handle
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.file_handle:
            self.file_handle.close()

def simulate_pxrd(cif_file, U=0.1, V=-0.1, W=0.5, X=0.2, Y=0.2, grainsize=20,
                  bg_ratio=0.05, add_noise=False, noise_level=0.02,
                  Tmin=10.0, Tmax=80.0, Tstep=0.02, wavelength=1.54184,
                  max_counts=5000, iparams='INST_XRY.PRM'):
    """
    Simulate PXRD pattern using GSAS-II from a CIF file.

    Args:
        cif_file: Path to CIF file
        U, V, W: Caglioti parameters for Gaussian broadening (in deg²)
        X, Y: Lorentzian broadening parameters (in deg)
        grainsize: Crystallite size in micrometers (affects Lorentzian broadening)
        bg_ratio: Background intensity ratio relative to max peak (0-1)
        add_noise: If True, add Poisson + Gaussian noise
        noise_level: Relative Gaussian noise level (e.g., 0.02 = 2%)
        max_counts: Counts scaling for Poisson noise
    """
    G2sc = _load_gsas_scriptable()

    # Create project and add phase
    tmp_root = _get_tmp_root()
    os.makedirs(tmp_root, exist_ok=True)
    gpx = G2sc.G2Project(newgpx=os.path.join(tmp_root, 'simulation.gpx'))
    phase = gpx.add_phase(cif_file, phasename='MyPhase')

    # Create simulated histogram
    hist = gpx.add_simulated_powder_histogram(
        histname='Simulated',
        iparams=iparams,
        Tmin=Tmin,
        Tmax=Tmax,                # Max 2θ
        Tstep=Tstep,              # Step size
        wavelength=wavelength,    # Cu Kα average
        scale=1.0,                # Scale factor
        phases=[phase]            # Link to phase
    )

    # Modify peak broadening parameters (index [0] is the value, [1] is refine flag)
    inst_params = hist.data['Instrument Parameters'][0]
    inst_params['U'][0] = U
    inst_params['V'][0] = V
    inst_params['W'][0] = W
    inst_params['X'][0] = X
    inst_params['Y'][0] = Y

    # Set crystallite size (in micrometers) - affects peak broadening
    phase_hist = phase.data['Histograms'][hist.name]
    phase_hist['Size'][0] = [True, False]  # [isotropic flag, refine flag]
    phase_hist['Size'][1] = [grainsize, 0.0, 1.0]  # [size, mustrain, refine_flag]

    # Calculate the pattern (need to do at least one refinement cycle)
    gpx.do_refinements()

    # Access the simulated data
    data = hist.data['data'][1]
    x = data[0]      # 2θ values
    ycalc = data[3]  # Calculated intensities

    # Add background
    if bg_ratio > 0:
        bg_coeffs = np.abs(np.random.randn(6))
        bg_coeffs[0] = -bg_coeffs[0]  # Ensure decreasing trend
        bg_fun = np.poly1d(bg_coeffs)
        bg = bg_fun(x)
        bg -= bg.min()
        bg_y = bg / bg.max() * ycalc.max() * bg_ratio
        ycalc = ycalc + bg_y

    # Add noise
    if add_noise and ycalc.max() > 0:
        # Scale to counts
        counts = ycalc / ycalc.max() * max_counts
        counts = np.maximum(counts, 0)
        # Add Poisson noise
        noisy_counts = np.random.poisson(counts).astype(float)
        # Add Gaussian noise
        noisy_counts += np.random.normal(0, noise_level * max_counts, size=len(noisy_counts))
        # Scale back to intensity domain
        ycalc = noisy_counts * (ycalc.max() / max_counts)

    ycalc = ycalc / ycalc.max() * 100
    return x, ycalc

def _refine_pxrd_impl(pxrd_file, cif_file, instprm="INST_XRY.PRM",
                gpx_name=None, gsas_log=None):
    """
    Core GSAS-II refinement logic.  Runs inside a subprocess to guarantee
    a clean GSAS-II module state on every call.

    Returns:
        (wR, R2, weighted_chi2, refined_cif)  on success
        (None, None, None, None)               on failure
    """

    G2sc = _load_gsas_scriptable()

    tmp_root = os.path.join(_get_tmp_root(), "gsas_runs")
    os.makedirs(tmp_root, exist_ok=True)
    run_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    default_base = os.path.join(tmp_root, f"{os.path.splitext(os.path.basename(cif_file))[0]}_{run_id}")
    if gpx_name is None: gpx_name = f"{default_base}.gpx"
    if gsas_log is None: gsas_log = f"{default_base}.log"

    gpx_dir = os.path.dirname(gpx_name)
    if gpx_dir: os.makedirs(gpx_dir, exist_ok=True)

    try:
        with RedirectG2Output(gsas_log):
            # --------------------------------------------------
            # Create GSAS-II project
            # --------------------------------------------------
            if os.path.exists(gpx_name): os.remove(gpx_name)
            gpx = G2sc.G2Project(newgpx=gpx_name)
            gpx.data['Controls']['data']['max cyc'] = 100

            # --------------------------------------------------
            # Import PXRD data and phase
            # --------------------------------------------------
            hist = gpx.add_powder_histogram(pxrd_file, instprm)
            phase = gpx.add_phase(cif_file, phasename='Phase1', histograms=[hist])

            print("Phase loaded:", phase.name)
            print("Histogram:", hist.name)

            hist.set_refinements({'Background': {'type': 'chebyschev',
                                                 'no. coeffs': 8,
                                                 'refine': True}})
            gpx.do_refinements()

            # 2) Scale only
            gpx.set_refinement({"set": {'Sample Parameters': ['Scale']}}, phase=phase)
            gpx.do_refinements()

            # 3) Zero + basic profile (U,V,W)
            hist.set_refinements({'Instrument Parameters': ['Zero', 'U', 'V', 'W']})
            gpx.do_refinements()

            # 3b) Add Lorentzian terms to fix peak shape (no asymmetry)
            hist.set_refinements({'Instrument Parameters': ['Zero', 'U', 'V', 'W', 'X', 'Y']})
            gpx.do_refinements()

            # 3c) Refine Kα2 fraction, polarization, and additional Lorentzian term
            hist.set_refinements({'Instrument Parameters': ['Zero', 'Polariz.', 'U', 'V', 'W', 'X', 'Y', 'Z']})
            gpx.do_refinements()

            # 3d) Refine axial asymmetry terms
            hist.set_refinements({'Instrument Parameters': ['Zero', 'Polariz.', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SH/L', 'Azimuth']})
            gpx.do_refinements()

            # 3e) Refine unit cell parameters
            phase.set_refinements({'Cell': True})
            gpx.do_refinements()

            # 4) atomic positions (all atoms)
            try:
                phase.set_refinements({'Atoms': {'all': ['X']}})
                gpx.do_refinements()

                phase_hist_key = hist.name
                hdict = phase.data['Histograms'][phase_hist_key]
                # HStrain is typically [values, refineFlags]; enable all flags
                hvals, hflags = hdict['HStrain']
                hdict['HStrain'][1] = [True] * len(hflags)
                gpx.do_refinements()
                print("Enabled HStrain refinement (thermal strain).")

            except Exception as e:
                print("Failed to refine atomic positions or HStrain:", e)
                return None, None, None, None, None, None

            wR = hist.get_wR()
            if wR is None:
                print("GSAS refinement produced no wR (residuals missing).")
                return None, None, None, None, None, None
            print(f"wR: {wR:.3f}")

            # Plot the final fit using GSAS-II powder histogram arrays
            data = hist.data.get('data')
            arrays = data[1]
            x = arrays[0]
            yobs = arrays[1]
            wt = arrays[2] if len(arrays) > 2 else None
            ycalc = arrays[3] if len(arrays) > 3 else None
            ybkg = arrays[4] if len(arrays) > 4 else None

            # Compute R^2 between observed and calculated
            R2 = None
            if ycalc is not None:
                ss_res = float(((yobs - ycalc)**2).sum())
                ss_tot = float(((yobs - yobs.mean())**2).sum())
                R2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
                print(f"R2: {R2:.4f}")

            # Compute chi² from weighted residuals
            weighted_chi2 = None
            if ycalc is not None and wt is not None:
                weighted_res = (yobs - ycalc) * np.sqrt(wt)
                weighted_chi2 = float((weighted_res**2).sum() / (len(yobs) - 1))
                print(f"Weighted chi² (manual): {weighted_chi2:.3f}")

            refined_cif = cif_file
    except Exception as e:
        print(f"GSAS refinement failed for {os.path.basename(cif_file)}: {e}")
        return None, None, None, None, None, None
    finally:
        # Clean up GSAS temp files to avoid stale state
        for suffix in ('.gpx', '.log', '.lst', '.bak0.gpx'):
            p = f"{default_base}{suffix}"
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # Convert arrays to lists for pickling across process boundary
    x_calc = x.tolist() if x is not None else None
    y_calc = ycalc.tolist() if ycalc is not None else None

    return wR, R2, weighted_chi2, refined_cif, x_calc, y_calc


# ---------------------------------------------------------------------------
# Persistent GSAS-II worker subprocess
# ---------------------------------------------------------------------------
# Instead of spawning a fresh process for every refinement call (expensive
# due to GSAS-II reimport), we keep a long-lived worker subprocess that
# processes requests in a loop.  The worker is killed and restarted only
# when a refinement *fails* (wR is None), because that is when GSAS-II
# module-level state may be corrupted.
# ---------------------------------------------------------------------------

def _worker_loop(request_q, result_q):
    """Persistent GSAS-II worker: process refinement requests until shutdown."""
    while True:
        request = request_q.get()
        if request is None:          # shutdown sentinel
            break
        try:
            result = _refine_pxrd_impl(*request)
        except Exception:
            result = (None, None, None, None, None, None)
        result_q.put(result)


class _PersistentWorker:
    """Manages a persistent GSAS-II subprocess, restarting only after failures."""

    def __init__(self):
        self._proc = None
        self._req_q = None
        self._res_q = None

    # -- lifecycle ----------------------------------------------------------

    def _start(self):
        self._req_q = _mp_ctx.Queue()
        self._res_q = _mp_ctx.Queue()
        self._proc = _mp_ctx.Process(
            target=_worker_loop,
            args=(self._req_q, self._res_q),
            daemon=True,
        )
        self._proc.start()

    def _kill(self):
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=5)
            self._proc = None
            self._req_q = None
            self._res_q = None

    def shutdown(self):
        """Gracefully shut down the worker (called via atexit)."""
        if self._proc is not None and self._proc.is_alive():
            try:
                self._req_q.put(None)       # tell loop to exit
                self._proc.join(timeout=10)
            except Exception:
                pass
            if self._proc is not None and self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=5)
        self._proc = None

    # -- public API --------------------------------------------------------

    def call(self, pxrd_file, cif_file, instprm):
        """Run one refinement, reusing the subprocess when healthy."""
        if self._proc is None or not self._proc.is_alive():
            self._start()

        self._req_q.put((pxrd_file, cif_file, instprm))

        try:
            result = self._res_q.get(timeout=300)
        except _queue_mod.Empty:
            self._kill()
            return None, None, None, None, None, None

        # If the worker died while computing, discard it for next call.
        if not self._proc.is_alive():
            self._proc = None

        # After any failure, kill the worker so the next call gets a fresh
        # GSAS-II interpreter (avoids the cascading-corruption bug).
        if result[0] is None:
            self._kill()

        # Pad old-format 4-tuples for backwards compatibility
        if len(result) == 4:
            result = result + (None, None)

        return result


_gsas_worker = _PersistentWorker()
atexit.register(_gsas_worker.shutdown)


def refine_pxrd(pxrd_file, cif_file, instprm="INST_XRY.PRM",
                ax=None, plot=False, remove=False):
    """
    Refine PXRD data using GSAS-II in an isolated subprocess.

    A persistent worker subprocess is reused across consecutive successful
    calls. After any failure the worker is killed and a fresh one is
    spawned on the next call (prevents the GSAS-II state-corruption bug).

    Returns:
        (wR, R2, weighted_chi2, refined_cif) or (None, None, None, None)
    """
    # Convert to absolute paths so the subprocess can find them
    pxrd_file = os.path.abspath(pxrd_file)
    cif_file = os.path.abspath(cif_file)
    instprm = os.path.abspath(instprm)

    wR, R2, weighted_chi2, refined_cif, x_calc, y_calc = _gsas_worker.call(
        pxrd_file, cif_file, instprm,
    )

    if wR is None:
        return None, None, None, None

    # ---- Optional plotting (runs in the parent process) ----
    if plot and ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    if ax is not None:
        # Re-read the PXRD CSV to get observed data for plotting
        import pandas as pd
        df = pd.read_csv(pxrd_file)
        x_obs = df.iloc[:, 0].values
        y_obs = df.iloc[:, 1].values
        ax.plot(x_obs, y_obs, 'k.', markersize=2, label='Observed')
        # Overlay the simulated (calculated) pattern from refinement
        if x_calc is not None and y_calc is not None:
            ax.plot(x_calc, y_calc, 'r-', linewidth=0.8, label='Calculated')
        ax.set_xlabel('2θ (degrees)')
        ax.set_ylabel('Intensity (a.u.)')
        title = 'PXRD Refinement Fit'
        if wR is not None: title += f" (Rwp={wR:.3f})"
        if R2 is not None: title += f"; R2={R2:.3f}"
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)

    return wR, R2, weighted_chi2, refined_cif

if __name__ == "__main__":
    ok, msg = check_gsas_available()
    if not ok:
        print(f"[ERROR] {msg}")
        print("[HINT] Install GSAS-II in your local environment before running tools/gsas.py")
        sys.exit(1)

    # --------------------------------------------------
    # User inputs
    # --------------------------------------------------
    INST_FILE = "tools/INST_XRY.PRM"
    pxrd_csv = "Examples/PXRD_TiCuSiAs_129.csv"
    pxrd_csv = "GSAS_PXRD/BeH2_72.csv"
    for match_cif in ['Fails/failed_ID100.cif', 'Fails/failed_ID77.cif']:
        wr, r2, chi2, cif = refine_pxrd(pxrd_csv, match_cif, INST_FILE, plot=True)
        print(match_cif, wr, r2, chi2)

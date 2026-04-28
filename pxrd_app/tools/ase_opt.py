# --- Optional fix for PyTorch weights_only change (harmless if unused) ---
from torch.serialization import add_safe_globals
add_safe_globals([slice])
# ------------------------------------------------------------------------

import atexit
import queue as _queue_mod
import signal
import warnings
import os
import sys
import multiprocessing as mp
import time
import io
import contextlib
import numpy as np

# Suppress the torch.load FutureWarning emitted by e3nn at module-import time.
# This must be at module level so it is in place in spawned child processes
# before any function is called (function-level filters are too late because
# warnings.catch_warnings() in get_calculator saves/restores the filter list).
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.load.*weights_only=False.*",
    category=FutureWarning,
)

# --- Limit threads per worker BEFORE importing PyTorch/MACE ---
# Without this, each worker defaults to using ALL available CPU cores.
# With 48 parallel workers on a 48-core node that means 48×48 = 2304 threads.
# Setting to 1 keeps total threads at 48 (one per worker).
# Override via PXRD_THREADS_PER_WORKER env var if needed.
_threads_per_worker = int(os.getenv('PXRD_THREADS_PER_WORKER', '1'))
os.environ.setdefault('OMP_NUM_THREADS', str(_threads_per_worker))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(_threads_per_worker))
os.environ.setdefault('MKL_NUM_THREADS', str(_threads_per_worker))
os.environ.setdefault('NUMEXPR_NUM_THREADS', str(_threads_per_worker))
# -----------------------------------------------------------------------

from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter
from ase.optimize.fire import FIRE
import logging
from ase.atoms import Atoms

_cached_mace = None
_cached_uma = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _default_isolate_fix_symmetry(calculator=None) -> bool:
    """
    Isolated relaxation is helpful for hard symmetry-related hangs, but on macOS the
    required `spawn` start method can make every relaxation look stuck because the
    child process must cold-start PyTorch/MACE for each attempt.
    """
    if isinstance(calculator, str) and calculator.upper() == "MACE":
        return False
    return sys.platform != "darwin"


def _silence_mace_output() -> bool:
    return _env_flag("PXRD_SILENCE_MACE", default=True)


def _suppress_torch_load_futurewarning() -> bool:
    return _env_flag("PXRD_SUPPRESS_TORCH_LOAD_FUTUREWARNING", default=True)


@contextlib.contextmanager
def _silence_external_output(enabled: bool):
    if not enabled:
        yield
        return

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def _strip_atoms_for_ipc(atoms):
    """Return a light-weight `Atoms` copy that can safely cross process boundaries."""
    if atoms is None:
        return None

    clean_atoms = atoms.copy()
    clean_atoms.calc = None
    clean_atoms.set_constraint()
    return clean_atoms


def _restore_atoms_from_ipc(atoms, calculator):
    """Reattach the calculator after an `Atoms` object returns from a worker process."""
    if atoms is None:
        return None

    with _silence_external_output(isinstance(calculator, str) and calculator == "MACE" and _silence_mace_output()):
        atoms.calc = get_calculator(calculator)
    return atoms


def _ase_relax_impl(
    atoms,
    calculator="MACE",
    opt_lat=True,
    step=300,
    fmax=0.5,
    logfile=None,
    max_time=15.0,
    label="ase",
    use_fix_symmetry=True,
):
    logger = logging.getLogger()
    timeout = int(max_time * 60)  # seconds

    import threading
    is_main_thread = threading.current_thread() == threading.main_thread()
    timer = None

    if is_main_thread:
        def handler(signum, frame):
            raise TimeoutError("Optimization timed out")
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
    else:
        logger.warning("Warning: ASE_relax called from non-main thread; timeout disabled.")
        timeout_event = threading.Event()
        timer = threading.Timer(timeout, timeout_event.set)
        timer.daemon = True
        timer.start()

    step_init = min([30, int(step / 2)])

    try:
        atoms.calc = get_calculator(calculator)
        if use_fix_symmetry:
            atoms.set_constraint(FixSymmetry(atoms))

        if opt_lat:
            ecf = UnitCellFilter(atoms)
            dyn = FIRE(ecf, a=0.1, logfile=logfile) if logfile is not None else FIRE(ecf, a=0.1)
        else:
            dyn = FIRE(atoms, a=0.1, logfile=logfile) if logfile is not None else FIRE(atoms, a=0.1)

        with np.errstate(under='ignore'):
            with _silence_external_output(True):
                dyn.run(fmax=fmax, steps=step_init)
        forces = atoms.get_forces()
        _fmax = np.sqrt((forces**2).sum(axis=1).max())

        if _fmax < 1e3 and step > step_init:
            with np.errstate(under='ignore'):
                with _silence_external_output(True):
                    dyn.run(fmax=fmax, steps=step - step_init)
            forces = atoms.get_forces()
            _fmax = np.sqrt((forces**2).sum(axis=1).max())
            if _fmax > 100:
                atoms = None
        else:
            atoms = None

    except TimeoutError:
        logger.warning(f"Warning {label} timed out after {timeout} seconds.")
        atoms = None

    except TypeError:
        logger.warning(f"Warning {label} spglib error in getting the lattice")
        atoms = None

    except FloatingPointError:
        logger.warning(f"Warning {label} FloatingPointError (underflow) during symmetry-constrained relaxation")
        atoms = None

    finally:
        signal.alarm(0)
        if timer is not None:
            timer.cancel()

    return atoms


def _ase_worker_loop(req_q, res_q):
    """Event loop running inside the persistent ASE worker subprocess.

    Loads the MACE calculator once (on the first request via ``_ase_relax_impl``
    → ``get_calculator``) and then handles an unlimited stream of relaxation
    requests.  This avoids the per-call spawn + MACE-reload overhead that the
    old Pipe-based approach incurred on macOS.
    """
    # Silence CPython crash reports (spglib segfaults) at the OS fd level.
    try:
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, 2)
        os.close(_devnull_fd)
    except OSError:
        pass

    while True:
        item = req_q.get()
        if item is None:          # graceful shutdown sentinel
            break
        atoms, kwargs = item
        try:
            result = _ase_relax_impl(atoms, **kwargs)
            res_q.put(("ok", _strip_atoms_for_ipc(result)))
        except Exception as exc:
            try:
                res_q.put(("error", f"{type(exc).__name__}: {exc}"))
            except Exception:
                pass


class _PersistentAseWorker:
    """Persistent subprocess for ASE relaxations.

    MACE (or any calculator) is loaded once when the worker starts, then
    reused for every subsequent call.  On macOS (spawn start method) this
    eliminates the ~2–5 s per-call MACE reload that the old per-Pipe approach
    suffered.  On Linux (fork) the difference is small but the worker also
    avoids repeated fork overhead.

    Crash recovery: if spglib segfaults and kills the worker, the next
    ``call()`` detects the dead process, restarts it, and returns ``"crash"``
    so the caller can retry without FixSymmetry.
    """

    def __init__(self):
        self._proc = None
        self._req_q = None
        self._res_q = None
        _start_method = "fork" if sys.platform.startswith("linux") else "spawn"
        self._ctx = mp.get_context(_start_method)

    def _start(self):
        self._req_q = self._ctx.Queue()
        self._res_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_ase_worker_loop,
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
        """Graceful shutdown (registered with atexit)."""
        if self._proc is not None and self._proc.is_alive():
            try:
                self._req_q.put(None)
                self._proc.join(timeout=10)
            except Exception:
                pass
        self._kill()

    def call(self, atoms, kwargs):
        """Submit one relaxation request and block until result or crash/timeout.

        Returns ``(status, payload)`` where *status* is one of:
        ``"ok"`` – payload is the relaxed Atoms (or None on convergence failure),
        ``"crash"`` – worker process died (spglib segfault),
        ``"timeout"`` – wall-time limit exceeded,
        ``"error"`` – Python exception in worker.
        """
        if self._proc is None or not self._proc.is_alive():
            self._start()

        self._req_q.put((atoms, kwargs))
        wait_s = max(10, int(kwargs.get("max_time", 15.0) * 60) + 20)

        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                result = self._res_q.get(timeout=min(1.0, remaining))
                # Worker may have crashed after sending; clear dead ref.
                if not self._proc.is_alive():
                    self._proc = None
                return result
            except _queue_mod.Empty:
                if not self._proc.is_alive():
                    self._kill()
                    return ("crash", None)

        # Deadline exceeded — kill the hung worker.
        self._kill()
        return ("timeout", None)


_ase_worker = _PersistentAseWorker()
atexit.register(_ase_worker.shutdown)

def get_calculator(calculator):
    """
    Return an ASE calculator instance.

    Supported strings:
      - 'FAIRChem', 'ANI', 'MACE', 'MACEOFF' if you re-enable those blocks.
    Or you can pass an ASE calculator instance directly.
    """
    global _cached_mace, _cached_uma

    if isinstance(calculator, str):
        if calculator == "UMA":
            if _cached_uma is None:
                import torch
                torch.set_num_threads(_threads_per_worker)
                from fairchem.core import pretrained_mlip, FAIRChemCalculator
                predictor = pretrained_mlip.get_predict_unit("uma-s-1p1")
                _cached_uma = FAIRChemCalculator(predictor,
                                                 task_name="omc")
            calc = _cached_uma

        elif calculator == "MACE":
            if _cached_mace is None:
                import torch
                torch.set_num_threads(_threads_per_worker)
                # Keep suppression tightly scoped to MACE import/init only.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*float32.*",
                        category=UserWarning,
                        module=r"(mace|e3nn)(\.|$)",
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*float32.*",
                        category=FutureWarning,
                        module=r"(mace|e3nn)(\.|$)",
                    )
                    if _suppress_torch_load_futurewarning():
                        warnings.filterwarnings(
                            "ignore",
                            message=r".*torch\.load.*weights_only=False.*",
                            category=FutureWarning,
                            module=r"e3nn\.o3\._wigner",
                        )
                    if _silence_mace_output():
                        with _silence_external_output(True):
                            from mace.calculators import mace_mp
                            _cached_mace = mace_mp(model="small")
                    else:
                        from mace.calculators import mace_mp
                        _cached_mace = mace_mp(model="small")
            calc = _cached_mace

        else:
            raise ValueError(f"Unknown calculator: {calculator}")
    else:
        # already an ASE calculator instance
        calc = calculator

    return calc


def ASE_relax(
    struc,
    calculator="MACE",
    opt_lat=True,
    step=300,
    fmax=0.5,
    logfile=None,
    max_time=15.0,
    label="ase",
    use_fix_symmetry=None,
):
    """
    ASE optimizer used by pyxtal/DFS.

    Args:
        struc: ASE Atoms object or a pyxtal object (with .to_ase()).
        calculator: string ('FAIRChem', 'MACE', 'ANI', 'MACEOFF') or ASE calculator.
        opt_lat (bool): optimize lattice (cell) or not.
        step (int): maximum FIRE steps.
        fmax (float): force convergence criterion (eV/Å).
        logfile (str or None): FIRE log file.
        max_time (float): wall time limit in minutes.
        label (str): label for logging.

    Returns:
        ASE Atoms object if successful, otherwise None.
    """


    logger = logging.getLogger()

    # Convert to ASE Atoms if needed
    if isinstance(struc, Atoms):
        atoms = struc
    elif hasattr(struc, "to_ase"):
        atoms = struc.to_ase(resort=False)
    else:
        raise TypeError("ASE_relax expects an ASE Atoms or a pyxtal object with .to_ase().")

    if use_fix_symmetry is None:
        use_fix_symmetry = _env_flag("PXRD_USE_FIX_SYMMETRY", default=True)

    isolate_fix_symmetry = _env_flag(
        "PXRD_ISOLATE_FIX_SYMMETRY",
        default=True,
    )
    if use_fix_symmetry and isolate_fix_symmetry and isinstance(calculator, str):
        # Delegate to the persistent worker subprocess.  MACE is loaded once
        # in that process and reused across calls — no per-call spawn + reload.
        kwargs = {
            "calculator": calculator,
            "opt_lat": opt_lat,
            "step": step,
            "fmax": fmax,
            "logfile": logfile,
            "max_time": max_time,
            "label": label,
            "use_fix_symmetry": use_fix_symmetry,
        }
        status, payload = _ase_worker.call(atoms, kwargs)

        if status == "ok":
            return _restore_atoms_from_ipc(payload, calculator)
        if status == "crash":
            logger.warning(
                f"Warning {label} isolated relaxation process crashed; "
                "retrying without FixSymmetry."
            )
            return _ase_relax_impl(
                atoms,
                calculator=calculator,
                opt_lat=opt_lat,
                step=step,
                fmax=fmax,
                logfile=logfile,
                max_time=max_time,
                label=f"{label}-nosym",
                use_fix_symmetry=False,
            )
        if status == "timeout":
            logger.warning(f"Warning {label} timed out in isolated relaxation process.")
            return None
        # status == "error" or unexpected
        logger.warning(f"Warning {label} isolated relaxation failed: {payload}")
        return None

    return _ase_relax_impl(
        atoms,
        calculator=calculator,
        opt_lat=opt_lat,
        step=step,
        fmax=fmax,
        logfile=logfile,
        max_time=max_time,
        label=label,
        use_fix_symmetry=use_fix_symmetry,
    )

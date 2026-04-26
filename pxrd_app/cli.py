import argparse
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, List

SPG_INFER_BACKENDS = {"smart-cell", "model"}
COMMON_SPG_TOP_K_CHOICES = [3, 5, 10, 20, 25, 30, 50, 100]
COMMON_SYMMETRY_CHOICES = [
    "auto",
    "any",
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--begin", '-b',
        type=int,
        default=0,
        help="Index of the first CSV file to process (default: 0).",
    )
    parser.add_argument(
        "--end", '-e',
        type=int,
        default=-1,
        help="Index of the last CSV file to process (default: all).",
    )
    parser.add_argument(
        "--max-wp",
        type=int,
        help="Maximum number of Wyckoff positions to consider (default: 9).",
    )
    parser.add_argument(
        "--max-dof",
        type=int,
        help="Maximum degrees of freedom for Wyckoff position combinations (default: 10).",
    )
    parser.add_argument(
        "--max-z",
        type=int,
        help="Maximum Z value to consider for volume estimation (default: 24).",
    )
    parser.add_argument(
        "--max-sim",
        type=float,
        help="Maximum allowed similarity to known structures for accepting a solution (default: 0.9).",
    )
    parser.add_argument(
        "--input",
        default="Examples/PXRD_PrYMg2_123.csv",
        help=(
            "Path to a PXRD CSV file, or a directory containing CSV files. "
            "If a directory is provided, all '*.csv' files in that directory are processed."
        ),
    )
    parser.add_argument(
        "--use-list",
        action="store_true",
        help=(
            "Treat --input as a text file containing CSV paths (one per line). "
            "Blank lines and lines starting with '#' are ignored."
        ),
    )
    parser.add_argument(
        "--formula",
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
        "--reverse",
        action="store_true",
        help="Reverse the order of processing CSV files.",
    )
    parser.add_argument(
        "--spg-top-k",
        type=int,
        choices=COMMON_SPG_TOP_K_CHOICES,
        default=150,
        help="Number of inferred space-group options to evaluate/show.",
    )
    parser.add_argument(
        "--spg-backend",
        type=str,
        choices=sorted(SPG_INFER_BACKENDS),
        default="smart-cell",
        help=(
            "Backend for --infer-spg: 'model' uses pretrained SG classifier, "
            "'smart-cell' uses SmartCellSolver to rank likely SGs by high->low symmetry and indexing evidence."
        ),
    )
    parser.add_argument(
        "--symmetry",
        type=str,
        choices=COMMON_SYMMETRY_CHOICES,
        default=None,
        help=(
            "Optional crystal-system filter for inferred SG candidates. "
            "Defaults to 'auto' when --infer-spg is set; otherwise unset. "
            "'auto' uses filename SPG (if present), 'any' disables filtering."
        ),
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
        help="Standard deviation of Cartesian perturbation in A for local perturb-and-relax trials.",
    )
    parser.add_argument(
        "--max-eng-rel",
        type=float,
        default=None,
        help=(
            "Maximum allowed energy-above-best (eV/atom) for immediate early termination on excellent "
            "refined fits. If unset, uses max(refine_eng_window, 0.60)."
        ),
    )
    parser.add_argument(
        "--max-cell-volume",
        type=float,
        default=1500.0,
        help="Maximum allowed unit-cell volume (A^3) for cell solutions. Larger cells are discarded.",
    )
    parser.add_argument(
        "--list-wp-only",
        action="store_true",
        help=(
            "List Wyckoff candidates for each planned (cell, SPG) pair and skip "
            "all structure generation/refinement trials."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of CSV files to solve in parallel when --input points to a directory. "
            "Use 1 to run sequentially."
        ),
    )
    parser.add_argument(
        "--parallel-retry-rounds",
        type=int,
        default=3,
        help=(
            "Number of extra parallel retry rounds after the initial pool attempt when a "
            "BrokenProcessPool occurs. Set to 0 to disable retries."
        ),
    )
    parser.add_argument(
        "--parallel-retry-min-workers",
        type=int,
        default=1,
        help=(
            "Minimum worker count to use during parallel retry rounds while backing off "
            "from the initial --workers value."
        ),
    )
    parser.add_argument(
        "--ase-logfile",
        type=str,
        default=None,
        help=(
            "Filename/path for ASE FIRE optimizer logs. "
            "Default is None (disable ASE log file output)."
        ),
    )
    parser.add_argument(
        "--output",
        default="Results",
        help="Directory to write results (CIF files, CSV summary, run logs, plots) into. Defaults to 'Results'.",
    )
    return parser


def resolve_cli_symmetry(args: argparse.Namespace) -> Optional[str]:
    if args.symmetry is not None:
        return args.symmetry
    if args.infer_spg and args.spg_backend == "smart-cell":
        return "any"
    return "auto" if args.infer_spg else None


def collect_input_csv_files(input_csv: str, use_list: bool = False) -> List[Path]:
    input_path = Path(input_csv)
    if use_list:
        if not input_path.is_file():
            raise FileNotFoundError(f"List file does not exist: {input_path}")

        base_dir = ''#input_path.parent
        csv_files: list[Path] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                p = Path(entry)
                if not p.is_absolute():
                    p = (base_dir / p).resolve()
                if not p.is_file():
                    raise FileNotFoundError(f"CSV path from list file does not exist: {p}")
                csv_files.append(p)

        if not csv_files:
            raise FileNotFoundError(f"No CSV paths found in list file: {input_path}")
        return csv_files

    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path}")
        return csv_files
    if input_path.is_file():
        return [input_path]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def resolve_parallel_workers(requested_workers: Optional[int], file_count: int) -> int:
    if file_count <= 1:
        return 1
    workers = 1 if requested_workers is None else max(1, int(requested_workers))
    return min(workers, file_count)


def _isolated_run_one(run_one, csv_str, args):
    """Run a single CSV job in an isolated subprocess.

    If the child segfaults (e.g. spglib), only this subprocess dies —
    the parent pool worker survives and reports the failure.
    """
    ctx = get_context("spawn")
    proc = ctx.Process(target=run_one, args=(csv_str, args))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        sig = -proc.exitcode if proc.exitcode < 0 else proc.exitcode
        raise RuntimeError(
            f"Subprocess exited with code {proc.exitcode} (signal {sig})"
        )


def run_csv_batch(
    csv_files: List[Path],
    args: argparse.Namespace,
    run_one,
) -> None:
    workers = resolve_parallel_workers(getattr(args, "workers", 1), len(csv_files))

    if len(csv_files) > 1:
        if workers > 1:
            print(f"Found {len(csv_files)} CSV file(s) in '{args.input}'. Running with {workers} parallel worker(s).")
        else:
            print(f"Found {len(csv_files)} CSV file(s) in '{args.input}'.")

    if workers <= 1:
        for idx, csv_path in enumerate(csv_files, start=1):
            csv_str = str(csv_path)
            if len(csv_files) > 1:
                print(f"\n{'=' * 60}")
                print(f"Processing file {idx}/{len(csv_files)}: {csv_str}")
                print(f"{'=' * 60}\n")
            run_one(csv_str, args)
        return

    failures: List[tuple[str, str]] = []
    pending: List[tuple[int, str]] = [
        (idx, str(csv_path))
        for idx, csv_path in enumerate(csv_files, start=1)
    ]
    total = len(pending)
    completed_ok = 0
    max_parallel_retry_rounds = max(0, int(getattr(args, "parallel_retry_rounds", 3)))
    retry_min_workers = max(1, int(getattr(args, "parallel_retry_min_workers", 1)))
    mp_context = get_context("spawn")

    for retry_round in range(max_parallel_retry_rounds + 1):
        if not pending:
            break

        # Retry rounds use fresh process pools with reduced worker fanout for stability.
        max_round_workers = max(retry_min_workers, workers // (2 ** retry_round))
        round_workers = min(len(pending), max_round_workers)
        if retry_round == 0:
            print(
                f"Running batch in parallel pool with {round_workers} worker(s)."
            )
        else:
            print(
                f"\nRetry round {retry_round}/{max_parallel_retry_rounds}: "
                f"re-queueing {len(pending)} file(s) in parallel with {round_workers} worker(s)."
            )

        broken_pool_files: List[tuple[int, str]] = []
        with ProcessPoolExecutor(max_workers=round_workers, mp_context=mp_context) as executor:
            future_map = {
                executor.submit(_isolated_run_one, run_one, csv_str, args): (idx, csv_str)
                for idx, csv_str in pending
            }
            for future in as_completed(future_map):
                idx, csv_str = future_map[future]
                try:
                    future.result()
                    completed_ok += 1
                    print(f"[{completed_ok}/{total}] Completed file {idx}: {csv_str}")
                except BrokenProcessPool:
                    broken_pool_files.append((idx, csv_str))
                    print(
                        f"Pool crashed for file {idx}: {csv_str} "
                        f"(will retry in parallel round {retry_round + 1})"
                    )
                except Exception as exc:
                    failures.append((csv_str, str(exc)))
                    print(f"Failed file {idx}: {csv_str}")
                    print(f"  Reason: {exc}")

        pending = sorted(broken_pool_files)

    if pending:
        for idx, csv_str in pending:
            failures.append((csv_str, "process pool repeatedly crashed during parallel retries"))
            print(f"Failed file {idx}: {csv_str}")
            print("  Reason: process pool repeatedly crashed during parallel retries")

    if failures:
        failure_text = "; ".join(f"{path}: {reason}" for path, reason in failures)
        raise RuntimeError(f"{len(failures)} file(s) failed during parallel execution: {failure_text}")


def _build_state(
    default_state: dict,
    logger,
    *,
    state: Optional[dict] = None,
    pxrd_csv: Optional[str] = None,
    formula: Optional[str] = None,
    multi_attempts: Optional[int] = None,
    seed_base: Optional[int] = None,
    infer_spg_from_pxrd: Optional[bool] = None,
    spg_top_k: Optional[int] = None,
    spg_infer_backend: Optional[str] = None,
    lattice_symmetry: Optional[str] = None,
    max_local_perturbations: Optional[int] = None,
    perturb_displacement: Optional[float] = None,
    max_eng_rel: Optional[float] = None,
    max_cell_volume: Optional[float] = None,
    list_wp_only: Optional[bool] = None,
    results_dir: Optional[str] = None,
    max_wp: Optional[int] = None,
    max_dof: Optional[int] = None,
    max_Z: Optional[int] = None,
    max_atoms: Optional[int] = None,
    max_sim: Optional[float] = None,
    ase_logfile: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> dict:
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
    if spg_infer_backend is not None:
        backend = str(spg_infer_backend).strip().lower()
        if backend in SPG_INFER_BACKENDS:
            run_state["spg_infer_backend"] = backend
        else:
            run_state["spg_infer_backend"] = "model"
    if lattice_symmetry is not None:
        run_state["lattice_symmetry"] = str(lattice_symmetry).strip().lower()
    elif bool(run_state.get("infer_spg_from_pxrd", False)) and str(run_state.get("spg_infer_backend", "model")).strip().lower() == "smart-cell":
        run_state["lattice_symmetry"] = "any"
    if max_local_perturbations is not None:
        run_state["max_local_perturbations"] = max_local_perturbations
    if perturb_displacement is not None:
        run_state["perturb_displacement"] = perturb_displacement
    if max_eng_rel is not None:
        run_state["max_eng_rel"] = max(0.0, float(max_eng_rel))
        run_state["max_eng_rel_early_stop"] = max(0.0, float(max_eng_rel))
    if max_cell_volume is not None:
        max_cell_volume = float(max_cell_volume)
        if max_cell_volume > 0:
            run_state["max_cell_volume"] = max_cell_volume
        else:
            logger.warning(f"Ignoring non-positive max_cell_volume={max_cell_volume}; expected > 0.")
    if list_wp_only is not None:
        run_state["list_wp_only"] = bool(list_wp_only)
    if results_dir is not None:
        run_state["results_dir"] = str(results_dir)
    if csv_path is not None: run_state["csv_path"] = str(csv_path)
    if max_wp is not None: run_state["max_wp"] = int(max_wp)
    if max_dof is not None: run_state["max_dof"] = int(max_dof)
    if max_Z is not None: run_state["max_Z"] = int(max_Z)
    if max_atoms is not None: run_state["max_atoms"] = int(max_atoms)
    if max_sim is not None: run_state["max_sim"] = float(max_sim)
    if ase_logfile is not None:
        text = str(ase_logfile).strip()
        run_state["ase_logfile"] = text or None
    run_state["status"] = "Failure"  # default to failure unless pipeline updates to success
    return run_state


def build_run_state(default_state: dict, logger, args: argparse.Namespace, csv_path: str) -> dict:
    return _build_state(
        default_state,
        logger,
        pxrd_csv=csv_path,
        formula=args.formula,
        multi_attempts=args.multi_attempts,
        seed_base=args.seed_base,
        infer_spg_from_pxrd=args.infer_spg,
        spg_top_k=args.spg_top_k,
        spg_infer_backend=args.spg_backend,
        lattice_symmetry=resolve_cli_symmetry(args),
        max_local_perturbations=args.local_perturbations,
        perturb_displacement=args.perturb_displacement,
        max_eng_rel=args.max_eng_rel,
        max_cell_volume=args.max_cell_volume,
        list_wp_only=args.list_wp_only,
        results_dir=args.output,
        max_wp=args.max_wp,
        max_dof=args.max_dof,
        max_Z=args.max_z,
        max_sim=args.max_sim,
        ase_logfile=args.ase_logfile,
        )

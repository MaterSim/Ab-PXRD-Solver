import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, List

def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--begin",
        type=int,
        default=0,
        help="Index of the first CSV file to process (default: 0).",
    )
    parser.add_argument(
        "--end",
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
        "--infer-spg",
        action="store_true",
        help="Infer space group from PXRD/profile model instead of filename convention.",
    )
    parser.add_argument(
        "--spg",
        type=int,
        default=None,
        help="Restrict the search to a single space group number (1-230).",
    )
    parser.add_argument(
        "--max-eng",
        type=float,
        default=None,
        help=(
            "Maximum allowed energy-above-best (eV/atom) for immediate early termination on excellent "
            "refined fits. If unset, uses max(refine_eng_window, 0.60)."
        ),
    )
    parser.add_argument(
        "--max-vol",
        type=float,
        default=1500.0,
        help="Maximum allowed unit-cell volume (A^3) for cell solutions. Larger cells are discarded.",
    )
    parser.add_argument(
        "--no-early-termination",
        action="store_true",
        help="Disable early-stop shortcuts and continue searching until other run limits are reached.",
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
        "--ase-log",
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
    parser.add_argument(
        "--wp-path",
        type=str,
        default="pxrd_app/tools/spg_comp_wp.csv",
        help=(
            "Path to CSV file containing number of Wyckoff positions per space group, used for "
            "estimating enumeration cost and guiding search. Defaults to 'pxrd_app/tools/spg_num_wps_mp.csv'."
        ),
    )
    parser.add_argument(
        "--qrs",
        choices=("sobol", "halton"),
        default="halton",
        help="Quasi-random sampler to use with --use-qrs. Defaults to halton.",
    )
    return parser


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
    infer_spg_from_pxrd: Optional[bool] = None,
    spg_top_k: Optional[int] = None,
    force_spg: Optional[int] = None,
    max_eng: Optional[float] = None,
    disable_early_termination: Optional[bool] = None,
    max_volume: Optional[float] = None,
    list_wp_only: Optional[bool] = None,
    results_dir: Optional[str] = None,
    max_wp: Optional[int] = None,
    max_dof: Optional[int] = None,
    max_Z: Optional[int] = None,
    max_atoms: Optional[int] = None,
    max_sim: Optional[float] = None,
    ase_log: Optional[str] = None,
    wp_path: Optional[str] = None,
    qrs: Optional[str] = None,
) -> dict:
    run_state = copy.deepcopy(default_state if state is None else state)
    if pxrd_csv is not None: run_state["pxrd_csv"] = pxrd_csv
    if formula is not None: run_state["formula"] = formula
    if infer_spg_from_pxrd is not None: run_state["infer_spg_from_pxrd"] = bool(infer_spg_from_pxrd)
    if spg_top_k is not None: run_state["spg_top_k"] = int(spg_top_k)
    run_state["spg_infer_backend"] = "smart-cell"
    if force_spg is not None:
        spg_val = int(force_spg)
        if 1 <= spg_val <= 230:
            run_state["force_spg"] = spg_val
        else:
            logger.warning(f"Ignoring invalid --spg={spg_val}; must be between 1 and 230.")
    if max_eng is not None:
        run_state["max_eng"] = max(0.0, float(max_eng))
    if disable_early_termination is not None:
        run_state["disable_early_termination"] = bool(disable_early_termination)
    if max_volume is not None:
        max_volume = float(max_volume)
        if max_volume > 0:
            run_state["max_volume"] = max_volume
        else:
            logger.warning(f"Ignoring non-positive max_volume={max_volume}; expected > 0.")
    if list_wp_only is not None: run_state["list_wp_only"] = bool(list_wp_only)
    if results_dir is not None: run_state["results_dir"] = str(results_dir)
    if wp_path is not None: run_state["wp_path"] = str(wp_path)
    if max_wp is not None: run_state["max_wp"] = int(max_wp)
    if max_dof is not None: run_state["max_dof"] = int(max_dof)
    if max_Z is not None: run_state["max_Z"] = int(max_Z)
    if max_atoms is not None: run_state["max_atoms"] = int(max_atoms)
    if max_sim is not None: run_state["max_sim"] = float(max_sim)
    if ase_log is not None:
        text = str(ase_log).strip()
        run_state["ase_log"] = text or None
    if qrs is not None:
        method = str(qrs).strip().lower()
        run_state["qrs"] = method if method in ("sobol", "halton") else "halton"
    run_state["status"] = "Failure"  # default to failure unless pipeline updates to success
    #print(run_state["disable_early_termination"]); import sys; sys.exit()
    return run_state


def build_run_state(default_state: dict, logger, args: argparse.Namespace, csv_path: str) -> dict:
    return _build_state(
        default_state,
        logger,
        pxrd_csv=csv_path,
        formula=args.formula,
        infer_spg_from_pxrd=args.infer_spg,
        spg_top_k=160,#args.spg_top_k,
        force_spg=args.spg,
        max_eng=args.max_eng,
        disable_early_termination=args.no_early_termination,
        max_volume=args.max_vol,
        list_wp_only=args.list_wp_only,
        results_dir=args.output,
        max_wp=args.max_wp,
        max_dof=args.max_dof,
        max_Z=args.max_z,
        max_sim=args.max_sim,
        ase_log=args.ase_log,
        wp_path=args.wp_path,
        qrs=args.qrs,
        )

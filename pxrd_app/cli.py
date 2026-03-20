import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

SPG_INFER_BACKENDS = {"model", "smart-cell"}


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
        "--input-csv",
        default="Examples/PXRD_PrYMg2_123.csv",
        help=(
            "Path to a PXRD CSV file, or a directory containing CSV files. "
            "If a directory is provided, all '*.csv' files in that directory are processed."
        ),
    )
    parser.add_argument(
        "--input-formula",
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
        "--try-all-inferred-spg",
        action="store_true",
        help="When --infer-spg is on, do not stop at first accepted SG; evaluate all inferred candidates and keep best.",
    )
    parser.add_argument(
        "--spg-top-k",
        type=int,
        choices=COMMON_SPG_TOP_K_CHOICES,
        default=100,
        help="Number of inferred space-group options to evaluate/show.",
    )
    parser.add_argument(
        "--spg-infer-backend",
        type=str,
        choices=sorted(SPG_INFER_BACKENDS),
        default="model",
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
        "--local-boosts",
        type=int,
        default=None,
        help="Maximum number of extra regeneration boosts per promising Wyckoff setting.",
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
        default=2500.0,
        help="Maximum allowed unit-cell volume (A^3) for cell solutions. Larger cells are discarded.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of CSV files to solve in parallel when --input-csv points to a directory. "
            "Use 1 to run sequentially."
        ),
    )
    return parser


def resolve_cli_symmetry(args: argparse.Namespace) -> str | None:
    if args.symmetry is not None:
        return args.symmetry
    if args.infer_spg and args.spg_infer_backend == "smart-cell":
        return "any"
    return "auto" if args.infer_spg else None


def collect_input_csv_files(input_csv: str) -> list[Path]:
    input_path = Path(input_csv)
    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path}")
        return csv_files
    if input_path.is_file():
        return [input_path]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def resolve_parallel_workers(requested_workers: int | None, file_count: int) -> int:
    if file_count <= 1:
        return 1
    workers = 1 if requested_workers is None else max(1, int(requested_workers))
    return min(workers, file_count)


def run_csv_batch(
    csv_files: list[Path],
    args: argparse.Namespace,
    run_one,
) -> None:
    workers = resolve_parallel_workers(getattr(args, "workers", 1), len(csv_files))

    if len(csv_files) > 1:
        if workers > 1:
            print(f"Found {len(csv_files)} CSV file(s) in '{args.input_csv}'. Running with {workers} parallel worker(s).")
        else:
            print(f"Found {len(csv_files)} CSV file(s) in '{args.input_csv}'.")

    if workers <= 1:
        for idx, csv_path in enumerate(csv_files, start=1):
            csv_str = str(csv_path)
            if len(csv_files) > 1:
                print(f"\n{'=' * 60}")
                print(f"Processing file {idx}/{len(csv_files)}: {csv_str}")
                print(f"{'=' * 60}\n")
            run_one(csv_str, args)
        return

    failures: list[tuple[str, str]] = []
    mp_context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
        future_map = {
            executor.submit(run_one, str(csv_path), args): (idx, str(csv_path))
            for idx, csv_path in enumerate(csv_files, start=1)
        }
        completed = 0
        total = len(future_map)
        for future in as_completed(future_map):
            idx, csv_str = future_map[future]
            completed += 1
            try:
                future.result()
                print(f"[{completed}/{total}] Completed file {idx}: {csv_str}")
            except Exception as exc:
                failures.append((csv_str, str(exc)))
                print(f"[{completed}/{total}] Failed file {idx}: {csv_str}")
                print(f"  Reason: {exc}")

    if failures:
        failure_text = "; ".join(f"{path}: {reason}" for path, reason in failures)
        raise RuntimeError(f"{len(failures)} file(s) failed during parallel execution: {failure_text}")


def build_run_state(
    default_state: dict,
    logger,
    *,
    state: dict | None = None,
    pxrd_csv: str | None = None,
    formula: str | None = None,
    multi_attempts: int | None = None,
    seed_base: int | None = None,
    infer_spg_from_pxrd: bool | None = None,
    try_all_inferred_spg: bool | None = None,
    spg_top_k: int | None = None,
    spg_infer_backend: str | None = None,
    show_spg_predictions: bool | None = None,
    lattice_symmetry: str | None = None,
    max_local_boosts: int | None = None,
    max_local_perturbations: int | None = None,
    perturb_displacement: float | None = None,
    max_eng_rel: float | None = None,
    max_cell_volume: float | None = None,
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
    if try_all_inferred_spg is not None:
        run_state["stop_on_first_accepted_inferred_spg"] = not bool(try_all_inferred_spg)
    if spg_top_k is not None:
        run_state["spg_top_k"] = int(spg_top_k)
    if spg_infer_backend is not None:
        backend = str(spg_infer_backend).strip().lower()
        if backend in SPG_INFER_BACKENDS:
            run_state["spg_infer_backend"] = backend
        else:
            logger.warning(f"Unsupported spg infer backend '{spg_infer_backend}', using model backend.")
            run_state["spg_infer_backend"] = "model"
    if show_spg_predictions is not None:
        run_state["show_spg_predictions"] = bool(show_spg_predictions)
    if lattice_symmetry is not None:
        run_state["lattice_symmetry"] = str(lattice_symmetry).strip().lower()
    elif bool(run_state.get("infer_spg_from_pxrd", False)) and str(run_state.get("spg_infer_backend", "model")).strip().lower() == "smart-cell":
        run_state["lattice_symmetry"] = "any"
    if max_local_boosts is not None:
        run_state["max_local_boosts"] = max(0, int(max_local_boosts))
    if max_local_perturbations is not None:
        run_state["max_local_perturbations"] = max(0, int(max_local_perturbations))
    if perturb_displacement is not None:
        run_state["perturb_displacement"] = max(0.0, float(perturb_displacement))
    if max_eng_rel is not None:
        run_state["max_eng_rel"] = max(0.0, float(max_eng_rel))
        run_state["max_eng_rel_early_stop"] = max(0.0, float(max_eng_rel))
    if max_cell_volume is not None:
        max_cell_volume = float(max_cell_volume)
        if max_cell_volume > 0:
            run_state["max_cell_volume"] = max_cell_volume
        else:
            logger.warning(f"Ignoring non-positive max_cell_volume={max_cell_volume}; expected > 0.")
    return run_state


def build_run_state_from_args(default_state: dict, logger, args: argparse.Namespace, csv_path: str) -> dict:
    return build_run_state(
        default_state,
        logger,
        pxrd_csv=csv_path,
        formula=args.input_formula,
        multi_attempts=args.multi_attempts,
        seed_base=args.seed_base,
        infer_spg_from_pxrd=args.infer_spg,
        try_all_inferred_spg=args.try_all_inferred_spg,
        spg_top_k=args.spg_top_k,
        spg_infer_backend=args.spg_infer_backend,
        show_spg_predictions=True,
        lattice_symmetry=resolve_cli_symmetry(args),
        max_local_boosts=args.local_boosts,
        max_local_perturbations=args.local_perturbations,
        perturb_displacement=args.perturb_displacement,
        max_eng_rel=args.max_eng_rel,
        max_cell_volume=args.max_cell_volume,
    )
import argparse
import ast
import copy
import csv
import logging
import re
import sys
import os
from pathlib import Path
import time
import shutil
from pxrd_app.constants import DEFAULT_STATE, PAIR_TABLE_RE, PAIR_HEADER_RE, WP_HEADER_RE, FLOAT_RE
from pxrd_app.cli import build_common_parser, build_run_state, collect_input_csv_files
from pxrd_app.runtime import write_results_csv
from pxrd_app.plot import plot_energy_vs_r2
from pxrd_app.core import run_data_preprocessor, run_wyckoff_solver, logger
from pxrd_app.inference import infer_formula_spg, spg_to_crystal_system
from pxrd_app.tools.manager import CellManager
from pxrd_app.tools.utils import format_wyckoff_labels

def _extract_outcome(label: str, state: dict, result: dict | None) -> dict:
    wyckoff_result = state.get("wyckoff_result") or {}
    structure_log = state.get("structure_log") or []
    refined_entries = [entry for entry in structure_log if entry.get("refined")]
    best_refined_r2 = max((entry.get("r2", -1.0) for entry in refined_entries), default=None)
    best_refined_chi2 = min((entry.get("chi2", 1e9) for entry in refined_entries), default=None)
    min_energy = min((entry.get("eng", 1e9) for entry in structure_log), default=None)
    selected_energy = wyckoff_result.get("selected_energy", 0.0)
    eng_rel = wyckoff_result.get("eng_rel", 0.0)
    if eng_rel is None and selected_energy is not None and min_energy is not None:
        eng_rel = max(0.0, selected_energy - min_energy)

    return {
        "label": label,
        "status": (result or {}).get("status", "unknown"),
        "message": (result or {}).get("message", ""),
        "spg": state.get("spg"),
        "formula": state.get("formula"),
        "accepted": bool(wyckoff_result.get("accepted", False)),
        "wr": float(wyckoff_result.get("wr", 1e9)),
        "r2": float(wyckoff_result.get("r2", -1.0)),
        "chi2": float(wyckoff_result.get("chi2", 1e9)),
        "score": float(wyckoff_result.get("score")),
        "selected_energy": selected_energy,
        "eng_rel": eng_rel,
        "attempt": int(wyckoff_result.get("attempt")),
        "seed": int(wyckoff_result.get("seed")),
        "cell_count": len(state.get("cells") or []),
        "structure_count": len(structure_log),
        "refined_count": len(refined_entries),
        "best_refined_r2": best_refined_r2,
        "best_refined_chi2": best_refined_chi2,
        "min_energy": min_energy,
        "log_path": state.get("system_run_log"),
    }

def _outcome_rank_key(outcome: dict) -> tuple:
    accepted = 1 if outcome.get("accepted") else 0
    score = float(outcome.get("score", -1e9))
    r2 = float(outcome.get("r2", -1.0))
    chi2 = float(outcome.get("chi2", 1e9))
    eng_rel = float(outcome.get("eng_rel", 1e9))
    structure_count = int(outcome.get("structure_count", 0))
    cell_count = int(outcome.get("cell_count", 0))
    return (accepted, score, r2, -chi2, -eng_rel, structure_count, cell_count)

def _is_better_outcome(candidate: dict, incumbent: dict) -> bool:
    return _outcome_rank_key(candidate) > _outcome_rank_key(incumbent)

def _is_good_sampling_outcome(outcome: dict, args: argparse.Namespace) -> bool:
    if not outcome.get("accepted"):
        return False
    eng_rel = float(outcome.get("eng_rel", None))
    if eng_rel is None:
        return True
    return eng_rel <= float(args.success_max_eng_rel)

def _system_stem(csv_path: str) -> str:
    return Path(csv_path).stem or "unknown_system"

def _resume_report_paths(csv_path: str, results_dir: str) -> tuple[Path, Path]:
    stem = _system_stem(csv_path)
    return Path(results_dir) / f"ResumeReport_{stem}.md"

def _build_resume_log_handler(state: dict) -> logging.Handler:
    source_log = str(state.get("source_run_log") or "").strip()
    if not source_log:
        raise ValueError("source_run_log must be set before attaching the resume log handler")
    log_path = Path(source_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    state["system_run_log"] = str(log_path)
    banner = (
        f"\n{'=' * 80}\n"
        f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Input: {state.get('pxrd_csv')}\n"
        f"{'=' * 80}\n"
        f"Per-system run log: {log_path}\n"
    )
    print(banner)
    # Write banner to the log file directly
    try:
        with open(log_path, "a") as f:
            f.write(banner)
    except Exception as exc:
        print(f"[WARN] Could not write resume banner to log: {exc}")
    print(f"Appending resume output to log: {log_path}")
    return handler


def _parse_trial_line(line: str) -> dict | None:

    # Accept lines with 'ID ...: *' prefix by extracting the part after ':'
    stripped_line = line.strip()
    # Use regex to extract the substring starting with '*', regardless of spaces
    star_match = re.search(r"\*.*", stripped_line)
    import re as _re
    stripped_line = line.strip()
    # Use regex to extract the substring starting with '*', regardless of spaces
    star_match = _re.search(r"\*.*", stripped_line)
    if not star_match:
        return None
    stripped_line = star_match.group(0)

    # Try to extract vol=... Å³
    volume = None
    volume_match = re.search(r"vol=(?P<volume>-?\d+(?:\.\d+)?)Å³", stripped_line)
    if volume_match is not None:
        volume = float(volume_match.group("volume"))

    # Extract perturb, skip_eng_rel if present
    perturb = None
    perturb_match = re.search(r"\[perturb:(?P<idx>\d+)\]", stripped_line)
    if perturb_match is not None: perturb = int(perturb_match.group("idx"))

    skip_eng_rel = None
    skip_match = re.search(r"\[skip:\s*eng_rel=(?P<eng_rel>-?\d+(?:\.\d+)?)\]", stripped_line)
    if skip_match is not None:
        skip_eng_rel = float(skip_match.group("eng_rel"))

    # Find numbers after 'vol=... Å³' for sim, eng, ...
    sim = eng = stress = fmax = None
    wr = r2 = chi2 = None
    post_vol_match = re.search(r"vol=[^\s]+ Å³([^\n]*)", stripped_line)
    if post_vol_match:
        post_vol = post_vol_match.group(1)
        post_vol_numbers = [float(x) for x in FLOAT_RE.findall(post_vol)]
        # Assign as many as possible
        if len(post_vol_numbers) >= 1:
            sim = post_vol_numbers[0]
        if len(post_vol_numbers) >= 2:
            eng = post_vol_numbers[1]
        if len(post_vol_numbers) >= 3:
            stress = post_vol_numbers[2]
        if len(post_vol_numbers) >= 4:
            fmax = post_vol_numbers[3]
        if len(post_vol_numbers) >= 7:
            wr, r2, chi2 = post_vol_numbers[-3:]

    return {
        "sim": sim,
        "eng": eng,
        "stress": stress,
        "fmax": fmax,
        "volume": volume,
        "wr": wr,
        "r2": r2 if r2 is not None else 0.0,
        "chi2": chi2,
        "refined": wr is not None and r2 is not None and chi2 is not None,
        "skip_eng_rel": skip_eng_rel,
        "perturb": perturb,
        "new_best_energy": "+++++" in stripped_line,
    }

def _build_previous_structure_entry(pair: dict | None, wp: dict | None, trial: dict) -> dict:
    spg = int((wp or {}).get("spg") or (pair or {}).get("spg") or 0)
    wp_labels = (wp or {}).get("wyckoff_labels") or []
    return {
        "eng": float(trial["eng"]),
        "eng_rel": None,
        "sim": float(trial["sim"]),
        "r2": float(trial.get("r2") or 0.0),
        "wr": float(trial.get("wr") or 0.0),
        "chi2": float(trial.get("chi2") or 1e9),
        "refined": bool(trial.get("refined", False)),
        "spg": spg,
        "wp_labels": wp_labels,
        "pair_rank": (pair or {}).get("rank"),
        "source": "previous_log",
    }

def _parse_run_log(log_path: Path) -> dict:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    pair_rows: list[dict] = []
    pairs_by_index: dict[int, dict] = {}
    previous_structure_log: list[dict] = []
    current_pair = None
    current_wp = None
    in_phase2_table = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("Rank  SPG"):
            in_phase2_table = True
            continue

        if in_phase2_table:
            if not stripped or stripped.startswith("-"):
                continue
            table_match = PAIR_TABLE_RE.match(line)
            if table_match is not None:
                dims = [float(value) for value in FLOAT_RE.findall(table_match.group("dims"))]
                pair_row = {
                    "rank": int(table_match.group("rank")),
                    "spg": int(table_match.group("spg")),
                    "volume": float(table_match.group("volume")),
                    "chi2": float(table_match.group("chi2")),
                    "missing": int(table_match.group("missing")),
                    "est_trials": int(table_match.group("est_trials")),
                    "bal_score": float(table_match.group("bal_score")),
                    "dims": dims,
                    "wp_candidates": [],
                }
                pair_rows.append(pair_row)
                continue
            if stripped.startswith("[Pair "):
                in_phase2_table = False

        pair_match = PAIR_HEADER_RE.match(stripped)
        if pair_match is not None:
            pair_index = int(pair_match.group("pair_index"))
            pair_row = pair_rows[pair_index - 1] if 0 < pair_index <= len(pair_rows) else {
                "rank": pair_index,
                "spg": int(pair_match.group("spg")),
                "volume": float(pair_match.group("volume")),
                "chi2": None,
                "missing": None,
                "est_trials": None,
                "bal_score": None,
                "dims": [float(value) for value in FLOAT_RE.findall(pair_match.group("dims"))],
                "wp_candidates": [],
            }
            pair_row["pair_index"] = pair_index
            pair_row["pair_total"] = int(pair_match.group("pair_total"))
            pairs_by_index[pair_index] = pair_row
            current_pair = pair_row
            current_wp = None
            continue

        wp_match = WP_HEADER_RE.match(stripped)
        if wp_match is not None and current_pair is not None:
            wyckoff_labels = ast.literal_eval(wp_match.group("wyckoff"))
            current_wp = {
                "wp_index": int(wp_match.group("wp_index")),
                "spg": int(wp_match.group("spg")),
                "count": int(wp_match.group("count")),
                "dof": int(wp_match.group("dof")),
                "n_wps": int(wp_match.group("n_wps")),
                "wyckoff_labels": wyckoff_labels,
                "trial_entries": [],
            }
            current_pair["wp_candidates"].append(current_wp)
            continue
        if stripped.startswith("ID ") and current_pair is not None:
            trial_entry = _parse_trial_line(stripped)
            # print(f"Parsed trial line: {trial_entry} from line: {stripped}")
            if trial_entry is not None:
                if current_wp is not None:
                    current_wp["trial_entries"].append(trial_entry)
                previous_structure_log.append(_build_previous_structure_entry(current_pair, current_wp, trial_entry))

    min_energy = min((float(entry["eng"]) for entry in previous_structure_log), default=None)
    for entry in previous_structure_log:
        if min_energy is None:
            entry["eng_rel"] = None
        else:
            entry["eng_rel"] = max(0.0, float(entry["eng"]) - float(min_energy))

    # Group by (spg, cell dims, wyckoff_labels)
    grouped = {}
    for entry in previous_structure_log:
        spg = entry["spg"]
        cell = tuple(round(float(x), 5) for x in entry.get("dims", entry.get("cell_dims", entry.get("dimensions", []))))
        wyckoff = tuple(tuple(str(item) for item in group) for group in entry.get("wp_labels", []))
        key = (spg, cell, wyckoff)
        score = float((1.5 * entry.get("r2", 0)) - (0.4 * entry.get("wr", 0)) - (0.2 * entry.get("chi2", 1e9)))
        if key not in grouped:
            grouped[key] = {"entry": entry.copy(), "attempts": 1, "score": score}
        else:
            grouped[key]["attempts"] += 1
            # Update best solution if new one is better
            if score > grouped[key]["score"]:
                grouped[key]["entry"] = entry.copy()
                grouped[key]["score"] = score
    print(f"Parsed {len(previous_structure_log)} previous structures, grouped into {len(grouped)} unique (spg, cell, wyckoff) signatures.")
    #import sys; sys.exit(0)
    # Add attempts to each entry
    deduped_structure_log = []
    for group in grouped.values():
        e = group["entry"]
        e["attempts"] = group["attempts"]
        deduped_structure_log.append(e)

    for pair in pair_rows:
        for wp in pair.get("wp_candidates", []):
            trials = wp.get("trial_entries", [])
            refined_trials = [trial for trial in trials if trial.get("refined")]
            wp["trial_count"] = len(trials)
            wp["refined_count"] = len(refined_trials)
            wp["best_sim"] = max((float(trial.get("sim", -1.0)) for trial in trials), default=-1.0)
            wp["best_eng"] = min((float(trial.get("eng", float("inf"))) for trial in trials), default=float("inf"))
            wp["best_wr"] = min((float(trial.get("wr", float("inf"))) for trial in refined_trials), default=float("inf"))
            wp["best_r2"] = max((float(trial.get("r2", -1.0)) for trial in refined_trials), default=-1.0)
            wp["best_chi2"] = min((float(trial.get("chi2", float("inf"))) for trial in refined_trials), default=float("inf"))
    best_previous = None
    best_previous_score = None
    for group in grouped.values():
        entry = group["entry"]
        wr = float(entry.get("wr", float("inf")))
        r2 = float(entry.get("r2", -1.0))
        chi2 = float(entry.get("chi2", float("inf")))
        if wr is None or r2 is None or chi2 is None: continue
        score = float((1.5 * r2) - (0.4 * wr) - (0.2 * chi2))
        if best_previous_score is None or score > best_previous_score:
            best_previous_score = score
            best_previous = {
                "score": score,
                "wr": wr,
                "r2": r2,
                "chi2": chi2,
                "eng": float(entry.get("eng", 0.0)),
                "eng_rel": float(entry.get("eng_rel", 0.0)),
                "spg": entry["spg"],
            }

    return {
        "pairs": pair_rows,
        "previous_structure_log": deduped_structure_log,
        "previous_best": best_previous,
    }


def _make_cell_from_pair(pair: dict) -> CellManager:
    missing = max(0, int(pair.get("missing") or 0))
    mismatch = [0] * missing
    dims = pair.get("dims") or []
    return CellManager(
        int(pair["spg"]),
        dims,
        mismatch,
        float(pair.get("chi2") or 0.0),
        [],
        None,
        [],
    )

def _restore_artifacts(snapshot: dict[str, str]) -> None:
    for target, source in snapshot.items():
        source_path = Path(source)
        if not source_path.exists():
            continue
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

def _artifact_paths(formula: str | None, spg: int | None, results_dir: str) -> list[Path]:
    if not formula or not spg:
        return []
    return [
        Path(results_dir) / f"Match_{formula}_{spg}.cif",
        Path(results_dir) / f"EnergyR2_{formula}_{spg}.png",
    ]

def _snapshot_artifacts(formula: str | None, spg: int | None, label: str, results_dir: str) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not formula or not spg:
        return snapshot
    snapshot_dir = Path(results_dir) / "tmp" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for artifact in _artifact_paths(formula, spg, results_dir):
        if not artifact.exists():
            continue
        snapshot_path = snapshot_dir / f"{label}_{artifact.name}"
        shutil.copy2(artifact, snapshot_path)
        snapshot[str(artifact)] = str(snapshot_path)
    return snapshot

def _resume_rank_key(trial: dict) -> tuple:
    observed = trial.get("observed") or {}
    pair = trial.get("pair") or {}
    refined_count = int(observed.get("refined_count") or 0)
    best_r2 = float(observed.get("best_r2", -1.0))
    best_chi2 = float(observed.get("best_chi2", float("inf")))
    best_eng = float(observed.get("best_eng", float("inf")))
    best_sim = float(observed.get("best_sim", -1.0))
    pair_rank = int(pair.get("rank", 10**6))
    bal_score = float(pair.get("bal_score", float("inf")))
    return (
        1 if refined_count > 0 else 0,
        best_r2,
        -best_chi2 if best_chi2 != float("inf") else -1e9,
        -best_eng if best_eng != float("inf") else -1e9,
        best_sim,
        -pair_rank,
        -bal_score if bal_score != float("inf") else -1e9,
    )

def _emit_resume_strategy(ranked_trials: list[dict], args: argparse.Namespace) -> None:
    print("Resume strategy: rank the queue by prior evidence from the log in this order:")
    print("refined count -> best R2 -> best chi2 -> best energy -> best similarity -> original Phase 2 priority")
    print(
        f"Resume strategy: {len(ranked_trials)} queued SPG/CELL/Wyckoff trial(s), "
        f"resume_attempts={args.resume_attempts}, "
    )
    print(
        f"{'Q':<3} {'SPG':<5} {'Pair':<5} {'WP':<6} {'DOF':<4} {'PrevRef':<7} {'BestR2':<8} "
        f"{'BestChi2':<9} {'BestE':<8} {'BestSim':<8} {'BaseN':<7} {'MaxN':<7} {'Dims / Wyckoff'}"
    )
    print("-" * 160)
    for queue_idx, trial in enumerate(ranked_trials, start=1):
        observed = trial.get("observed")

        pair = trial.get("pair") or {}
        dims = " ".join(f"{float(value):.3f}" for value in trial["cell"].dims)
        # Use Wyckoff labels from the log if available, else from candidate
        wp_labels = trial.get("wyckoff_labels")
        best_r2 = float(observed.get("best_r2", None))
        best_chi2 = float(observed.get("best_chi2", None))
        best_eng = float(observed.get("best_eng", None))
        best_sim = float(observed.get("best_sim", None))
        wp_index = trial.get('wp_index') or 0

        logger.info(
            f"{queue_idx:<3} {pair.get('spg'):<5} "
            f"{pair['rank']:<5} {wp_index:<6} "
            f"{trial['dof']:<4} {int(observed.get('refined_count') or 0):<7} "
            f"{(f'{best_r2:.3f}' if best_r2 is not None and best_r2 > -1 else 'n/a'):<8} "
            f"{(f'{best_chi2:.3f}' if best_chi2 is not None and best_chi2 < float('inf') else 'n/a'):<9} "
            f"{(f'{best_eng:.3f}' if best_eng is not None and best_eng < float('inf') else 'n/a'):<8} "
            f"{(f'{best_sim:.3f}' if best_sim is not None and best_sim > -1 else 'n/a'):<8} "
            f"{int(observed.get('base_total') or 0):<7} {int(observed.get('max_total') or 0):<7} "
            f"{dims} | {wp_labels}"
        )

def _build_resume_trials(parsed_log) -> list[dict]:
    visited = []
    ranked_trials: list[dict] = []
    for pair in parsed_log.get("pairs", []):
        cell = _make_cell_from_pair(pair)
        cell_str = f"{' '.join(f'{float(x):.3f}' for x in cell.dims)}"
        # Logged candidates
        for wp in pair["wp_candidates"]:
            spg, wyckoff_labels = (pair["spg"], wp.get("wyckoff_labels"))
            entry = (spg, cell_str, wyckoff_labels)
            if entry in visited:
                # update existing entry with any new trial data if available
                for existing in ranked_trials:
                    if existing['spg'] == spg and existing['cell_str'] == cell_str and existing['wyckoff_labels'] == wyckoff_labels:
                        existing["observed"]["trial_count"] += wp.get("trial_count")
                        existing["observed"]["refined_count"] += wp.get("refined_count")
                        existing["observed"]["best_r2"] = max(existing["observed"]["best_r2"], wp["best_r2"])
                        existing["observed"]["best_chi2"] = min(existing["observed"]["best_chi2"], wp["best_chi2"])
                        existing["observed"]["best_eng"] = min(existing["observed"]["best_eng"], wp["best_eng"])
                        existing["observed"]["best_sim"] = max(existing["observed"]["best_sim"], wp["best_sim"])
                        break
                continue
            else:
                visited.append(entry)
                ranked_trials.append(
                {
                    "spg": spg,
                    "cell_str": cell_str,
                    "wyckoff_labels": wyckoff_labels,
                    "label": f"pair_{pair.get('rank')}_wp_{wp.get('wp_index')}",
                    "pair": pair,
                    "wp_index": wp.get("wp_index"),
                    "cell": cell,
                    "dof": wp.get("dof"),
                    "summary": f"rank={pair.get('rank')} spg={pair.get('spg')} wp={wp.get('wp_index')}",
                    "observed": {
                        "trial_count": wp.get("trial_count", 0),
                        "refined_count": wp.get("refined_count", 0),
                        "best_r2": wp.get("best_r2", -1.0),
                        "best_chi2": wp.get("best_chi2", float("inf")),
                        "best_eng": wp.get("best_eng", float("inf")),
                        "best_sim": wp.get("best_sim", -1.0),
                    },
                })
                #print(spg, cell_str, wyckoff_labels, f"rank={pair.get('rank')} spg={pair.get('spg')}")
    ranked_trials.sort(key=_resume_rank_key, reverse=True)
    return ranked_trials[: max(1, 50)]


def _previous_baseline_outcome(parsed_log: dict, state: dict) -> dict:
    previous_best = parsed_log.get("previous_best") or {}
    previous_structure_log = parsed_log.get("previous_structure_log") or []
    return {
        "label": "previous_log_baseline",
        "status": "previous_failure",
        "message": "Best observed candidate reconstructed from prior run log.",
        "spg": previous_best.get("spg") or state.get("spg"),
        "formula": state.get("formula"),
        "accepted": False,
        "wr": previous_best.get("wr"),
        "r2": previous_best.get("r2"),
        "chi2": previous_best.get("chi2"),
        "score": previous_best.get("score"),
        "selected_energy": previous_best.get("eng"),
        "eng_rel": previous_best.get("eng_rel"),
        "attempt": None,
        "seed": None,
        "cell_count": len(parsed_log.get("pairs") or []),
        "structure_count": len(previous_structure_log),
        "refined_count": len([entry for entry in previous_structure_log if entry.get("refined")]),
        "best_refined_r2": previous_best.get("r2"),
        "best_refined_chi2": previous_best.get("chi2"),
        "min_energy": min((float(entry.get("eng")) for entry in previous_structure_log), default=None),
        "log_path": state.get("source_run_log"),
    }

def _resolve_failure_csvs(summary_csv: str, examples_dir: str, symmetry: str = "auto") -> list:
    failures = []
    symmetry = (symmetry or "auto").strip().lower()
    use_symmetry_filter = symmetry not in {"", "auto", "any"}

    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Status", "").strip().lower() == "failure":
                csv_file = row.get("csv_file_name", "").strip()
                if csv_file:
                    if use_symmetry_filter:
                        _, spg = infer_formula_spg(csv_file)
                        csv_symmetry = spg_to_crystal_system(spg)
                        if csv_symmetry != symmetry:
                            continue
                    path = os.path.join(examples_dir, csv_file)
                    if os.path.isfile(path) and path not in failures:
                        failures.append(path)

    if use_symmetry_filter:
        print(
            f"Identified {len(failures)} failure CSV(s) from summary "
            f"with symmetry='{symmetry}': {failures}"
        )
    else:
        print(f"Identified {len(failures)} failure CSV(s) from summary: {failures}")

    for csv_file in failures:
        if not os.path.isfile(csv_file):
            print(f"[WARN] Failure CSV not found: {csv_file}")
    return failures

def run_resume(csv_path, args):
    tmp_root = os.path.join(args.output, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    os.environ["PXRD_TMP_ROOT"] = tmp_root

    structure_log = []
    # Build run state
    run_state = build_run_state(DEFAULT_STATE, logger, args, csv_path)
    run_state["pxrd_csv"] = csv_path
    min_structures = run_state["min_structures_before_early_stop"]

    # Find the original run log
    system_stem = _system_stem(csv_path)
    # Try to find the original log in the output dir, supporting both .txt and .log extensions
    log_candidates = []
    for ext in (".txt", ".log"):
        log_candidates.append(os.path.join(args.output, f"logs/RunLog_{system_stem}{ext}"))
        log_candidates.append(os.path.join(os.path.dirname(csv_path), f"logs/RunLog_{system_stem}{ext}"))
    source_log = None
    for log_path in log_candidates:
        if os.path.isfile(log_path):
            source_log = log_path
            break
    if not source_log:
        print(f"[ERROR] No RunLog found for {csv_path} (tried: {log_candidates})")
        return
    run_state["source_run_log"] = source_log
    # Ensure log handler and banner are initialized for resume runs
    _build_resume_log_handler(run_state)
    parsed_log = _parse_run_log(Path(source_log))
    if not parsed_log.get("pairs"):
        print(f"[ERROR] No Phase 2 candidates found in run log: {source_log}")
        return
    run_data_preprocessor(csv_path, run_state)
    baseline_outcome = _previous_baseline_outcome(parsed_log, run_state)
    winner_state = run_state
    winner_state["best_result"] = None
    winner_outcome = baseline_outcome
    followup_trials: list[dict] = []
    combined_plot_log = copy.deepcopy(parsed_log.get("previous_structure_log") or [])

    # Start timing for structure inference
    _start_time = time.time()
    winner_snapshot = _snapshot_artifacts(
        winner_outcome["formula"], winner_outcome["spg"], "resume_winner", args.output
    )
    ranked_trials = _build_resume_trials(parsed_log)
    logger.info(
        f"Parsed {len(parsed_log.get('pairs') or [])} Phase 2 pair(s) and selected "
        f"{len(ranked_trials)} resume trial(s)."
    )

    _emit_resume_strategy(ranked_trials, args)
    good_outcome = False
    for id in range(max(1, int(args.resume_attempts))):
        for trial in ranked_trials:
            logger.info(f"\n{'-' * 80}\nStarting trial: {trial['label']} - {trial['summary']}\n")
            trial_state, trial_outcome = _run_resume_trial(winner_state, trial, args, structure_log)
            structure_log.extend(copy.deepcopy(trial_state.get("structure_log") or []))
            followup_trials.append(trial_outcome)
            combined_plot_log.extend(copy.deepcopy(trial_state.get("structure_log") or []))
            #print(f"=============Debug struc_log {len(structure_log)} {len(combined_plot_log)}")
            if _is_better_outcome(trial_outcome, winner_outcome):
                winner_state = trial_state
                winner_outcome = trial_outcome
                winner_snapshot = _snapshot_artifacts(
                    winner_outcome["formula"], winner_outcome["spg"], "resume_winner", args.output
                )
            else:
                _restore_artifacts(winner_snapshot)

            # Enforce early termination as soon as a good outcome is found
            if _is_good_sampling_outcome(trial_outcome, args):
                logger.info(f"Accepted low-dE result found in {trial_outcome.get('label')}")
                good_outcome = True

            if good_outcome and len(structure_log) >= min_structures:
                logger.info(f"stopping resume search early.")
                winner_state["status"] = "Success"
                break
        if good_outcome:
            break

    # A resume run that produced an accepted winner should be recorded as success
    # even if it stopped before reaching the exploratory min-structure target.
    if winner_outcome.get("accepted") or good_outcome:
        winner_state["status"] = "Success"

    # End timing and update timing_breakdown_seconds
    structure_runtime = time.time() - _start_time
    timing = {}
    timing["structure_inference"] = structure_runtime
    timing["total_runtime"] = structure_runtime
    timing["spg_and_cell"] = 0.0
    formula = winner_state.get("formula") or "unknown_formula"
    spg = winner_state.get("spg") or "unknown_spg"
    merged_plot = Path(args.output) / f"EnergyR2_{formula}_{spg}_resume.png"
    if combined_plot_log:
        plot_energy_vs_r2(combined_plot_log, winner_state, merged_plot, timing)
        logger.info(f"Energy–R² plot saved to {merged_plot}")

    md_path = _resume_report_paths(csv_path, args.output)
    summary_path = Path(args.output) / "summary.csv"
    # Append new result to summary.csv
    input_csv = Path(csv_path)
    write_results_csv(input_csv, winner_state)
    # Prepare row data from final_outcome
    print(f"Saved resume report to {md_path} and {summary_path}")

def _run_resume_trial(base_state: dict, trial: dict, args: argparse.Namespace, structure_log: list) -> tuple[dict, dict]:

    trial_state = copy.deepcopy(base_state)
    trial_state["spg"] = int(trial["pair"]["spg"])
    trial_state["cells"] = [copy.deepcopy(trial["cell"])]
    trial_state["multi_attempts"] = 1
    trial_state["max_local_perturbations"] = base_state["max_local_perturbations"]
    trial_state["perturb_displacement"] = base_state["perturb_displacement"]
    trial_state["max_eng_rel_early_stop"] = float(args.success_max_eng_rel)
    trial_state["wp_labels"] = trial["wyckoff_labels"]
    message, trial_state = run_wyckoff_solver(trial_state, [], len(structure_log))
    wyckoff_result = trial_state.get("wyckoff_result") or {}
    status = f"{trial['label']}_success" if wyckoff_result.get("accepted") else "no_solution"
    if wyckoff_result["wr"] is not None:
        outcome = _extract_outcome(trial["label"], trial_state, {"status": status, "message": message})
        outcome["summary"] = trial["summary"]
    else:
        outcome = {
            "label": trial["label"],
            "status": status,
            "message": message,
            "spg": trial_state.get("spg"),
            "formula": trial_state.get("formula"),
            "accepted": False,
        }
    return trial_state, outcome

if __name__ == "__main__":
    from pxrd_app.cli import run_csv_batch, collect_input_csv_files
    parser = build_common_parser("Resume PXRD search from a failed run log")
    parser.add_argument(
        "--summary",
        default="",
        help="Optional path to Results/summary.csv. When set, all failed systems are resumed.",
    )
    parser.add_argument(
        "--input-dir",
        default="Examples",
        help="Directory used to resolve csv_file_name entries from --summary.",
    )
    parser.add_argument(
        "--log-path",
        default="",
        help="Optional explicit RunLog path for single-file resume runs.",
    )
    parser.add_argument(
        "--resume-attempts",
        type=int,
        default=3,
        help="Adaptive multi-attempt count for each resume trial.",
    )
    parser.add_argument(
        "--success-max-eng-rel",
        type=float,
        default=0.20,
        help="Stop once an accepted candidate has dE below this threshold.",
    )

    args = parser.parse_args()
    if args.summary:
        csv_paths = _resolve_failure_csvs(args.summary, args.input_dir, args.symmetry)
        if not csv_paths:
            print(f"No failed systems found in summary CSV: {args.summary}")
            sys.exit(1)
    else:
        summary_file = os.path.join(args.output, 'summary.csv')
        if os.path.isfile(summary_file):
            print(f"Found summary.csv in output directory; resuming failed systems from that summary.")
            args.summary = summary_file
            csv_paths = _resolve_failure_csvs(summary_file, args.input_dir, args.symmetry)
            if not csv_paths:
                print(f"No failed systems found in existing summary CSV: {summary_file}")
                sys.exit(1)
        else:
            try:
                csv_paths = [str(path) for path in collect_input_csv_files(args.input, use_list=args.use_list)]
            except FileNotFoundError as exc:
                print(str(exc))
                sys.exit(1)

    os.makedirs(args.output + "/cifs", exist_ok=True)
    os.makedirs(args.output + "/logs", exist_ok=True)
    os.makedirs(args.output + "/tmp", exist_ok=True)
    if len(csv_paths) > 1 and args.workers > 1:
        run_csv_batch([Path(p) for p in csv_paths], args, run_resume)
    else:
        for csv_path in csv_paths:
            run_resume(csv_path, args)

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

from pxrd_app.constants import DEFAULT_STATE as default_state
from pxrd_app.cli import build_common_parser, build_run_state_from_args, collect_input_csv_files
from pxrd_app.runtime import write_results_csv
from pxrd_app.plot import plot_energy_vs_r2
from pxrd_app.core import (
    _format_wyckoff_labels_from_ids,
    _run_data_preprocessor,
    _run_wyckoff_solver,
    logger,
    _extract_outcome,
    _is_better_outcome,
    _is_good_sampling_outcome,
    _restore_artifacts,
    _safe_float,
    _safe_int,
    _snapshot_artifacts,
)
from tools.manager import CellManager
from tools.solver import enumerate_wyckoff_multi_spg, score_wp_candidate

structure_log = []

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


def _system_stem(csv_path: str) -> str:
    return Path(csv_path).stem or "unknown_system"


def _resume_report_paths(csv_path: str, results_dir: str) -> tuple[Path, Path]:
    stem = _system_stem(csv_path)
    return Path(results_dir) / f"ResumeReport_{stem}.md"


def _merged_plot_path(formula: str, results_dir: str) -> Path:
    return Path(results_dir) / f"EnergyR2_{formula}_resume.png"


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

    # Extract all numbers (float/int) in order, including negatives
    all_numbers = [float(x) for x in FLOAT_RE.findall(stripped_line)]
    # Try to extract vol=... Å³
    volume = None
    volume_match = re.search(r"vol=(?P<volume>-?\d+(?:\.\d+)?)Å³", stripped_line)
    if volume_match is not None:
        volume = float(volume_match.group("volume"))

    # Extract perturb, boost, skip_eng_rel if present
    perturb = None
    perturb_match = re.search(r"\[perturb:(?P<idx>\d+)\]", stripped_line)
    if perturb_match is not None:
        perturb = int(perturb_match.group("idx"))

    boost = None
    boost_match = re.search(r"\[boost:\+(?P<count>\d+)\]", stripped_line)
    if boost_match is not None:
        boost = int(boost_match.group("count"))

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
        "boost": boost,
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
        "wr": _safe_float(trial.get("wr")),
        "chi2": _safe_float(trial.get("chi2")),
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

    for pair in pair_rows:
        for wp in pair.get("wp_candidates", []):
            trials = wp.get("trial_entries", [])
            refined_trials = [trial for trial in trials if trial.get("refined")]
            wp["trial_count"] = len(trials)
            wp["refined_count"] = len(refined_trials)
            wp["best_sim"] = max((_safe_float(trial.get("sim"), -1.0) for trial in trials), default=-1.0)
            wp["best_eng"] = min((_safe_float(trial.get("eng"), float("inf")) for trial in trials), default=float("inf"))
            wp["best_wr"] = min((_safe_float(trial.get("wr"), float("inf")) for trial in refined_trials), default=float("inf"))
            wp["best_r2"] = max((_safe_float(trial.get("r2"), -1.0) for trial in refined_trials), default=-1.0)
            wp["best_chi2"] = min((_safe_float(trial.get("chi2"), float("inf")) for trial in refined_trials), default=float("inf"))
            #print(f"Parsed WP candidate: SPG {wp['spg']}, count {wp['count']}, dof {wp['dof']}, wyckoff {wp['wyckoff_labels']}")
    best_previous = None
    best_previous_score = None
    for entry in previous_structure_log:
        wr = _safe_float(entry.get("wr"))
        r2 = _safe_float(entry.get("r2"))
        chi2 = _safe_float(entry.get("chi2"))
        if wr is None or r2 is None or chi2 is None:
            continue
        score = float((1.5 * r2) - (0.4 * wr) - (0.2 * chi2))
        if best_previous_score is None or score > best_previous_score:
            best_previous_score = score
            best_previous = {
                "score": score,
                "wr": wr,
                "r2": r2,
                "chi2": chi2,
                "eng": float(entry.get("eng")),
                "eng_rel": _safe_float(entry.get("eng_rel")),
                "spg": _safe_int(entry.get("spg")),
            }
    #import sys; sys.exit(0)

    return {
        "pairs": pair_rows,
        "previous_structure_log": previous_structure_log,
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


def _candidate_label_signature(spg: int, candidate) -> tuple[int, tuple[tuple[str, ...], ...]]:
    label_text = _format_wyckoff_labels_from_ids(spg, candidate[3])
    labels = ast.literal_eval(label_text)
    return int(spg), tuple(tuple(str(item) for item in group) for group in labels)


def _logged_label_signature(wp: dict) -> tuple[int, tuple[tuple[str, ...], ...]]:
    return int(wp["spg"]), tuple(tuple(str(item) for item in group) for group in wp["wyckoff_labels"])


def _resume_rank_key(trial: dict) -> tuple:
    observed = trial.get("observed") or {}
    pair = trial.get("pair") or {}
    refined_count = int(observed.get("refined_count") or 0)
    best_r2 = _safe_float(observed.get("best_r2"), -1.0)
    best_chi2 = _safe_float(observed.get("best_chi2"), float("inf"))
    best_eng = _safe_float(observed.get("best_eng"), float("inf"))
    best_sim = _safe_float(observed.get("best_sim"), -1.0)
    pair_rank = int(pair.get("rank") or 10**6)
    bal_score = _safe_float(pair.get("bal_score"), float("inf"))
    candidate_score = score_wp_candidate(trial["candidate"], max_dof=int(trial["candidate"][5]))
    return (
        1 if refined_count > 0 else 0,
        best_r2,
        -best_chi2 if best_chi2 != float("inf") else -1e9,
        -best_eng if best_eng != float("inf") else -1e9,
        best_sim,
        -pair_rank,
        -bal_score if bal_score != float("inf") else -1e9,
        candidate_score,
    )


def _resume_attempt_schedule(attempt_idx: int) -> tuple[int, int, int]:
    schedules = [
        (5, 20, 9),
        (7, 25, 10),
        (10, 30, 12),
    ]
    if attempt_idx < len(schedules):
        return schedules[attempt_idx]
    return schedules[-1]


def _estimate_structures_for_candidate(candidate, attempts: int, max_local_boosts: int) -> dict:
    dof = int(candidate[5])
    base_trials_per_attempt: list[int] = []
    max_trials_per_attempt: list[int] = []

    for attempt_idx in range(max(1, int(attempts))):
        _n1, _n2, n3 = _resume_attempt_schedule(attempt_idx)
        if dof > int(n3):
            base_trials_per_attempt.append(0)
            max_trials_per_attempt.append(0)
            continue

        n4 = (dof * 3) if dof != 1 else 4
        base_trials = n4 + 1
        added_trials = max(2, min(6, (n4 // 2) if n4 > 1 else 2))
        max_trials = base_trials + (max(0, int(max_local_boosts)) * added_trials)
        base_trials_per_attempt.append(base_trials)
        max_trials_per_attempt.append(max_trials)

    return {
        "dof": dof,
        "base_per_attempt": base_trials_per_attempt,
        "max_per_attempt": max_trials_per_attempt,
        "base_total": sum(base_trials_per_attempt),
        "max_total": sum(max_trials_per_attempt),
    }


def _emit_resume_strategy(ranked_trials: list[dict], args: argparse.Namespace) -> None:
    print("Resume strategy: rank the queue by prior evidence from the log in this order:")
    print("  refined count -> best R2 -> best chi2 -> best energy -> best similarity -> original Phase 2 priority")
    print(
        f"Resume strategy: {len(ranked_trials)} queued SPG/CELL/Wyckoff trial(s), "
        f"resume_attempts={int(args.resume_attempts)}, local_boosts={int(args.resume_local_boosts)}, "
        f"local_perturbations={int(args.resume_local_perturbations)}"
    )
    print(
        f"{'Q':<3} {'SPG':<5} {'Pair':<5} {'WP':<6} {'DOF':<4} {'PrevRef':<7} {'BestR2':<8} "
        f"{'BestChi2':<9} {'BestE':<8} {'BestSim':<8} {'BaseN':<7} {'MaxN':<7} {'Dims / Wyckoff'}"
    )
    print("-" * 160)
    for queue_idx, trial in enumerate(ranked_trials, start=1):
        observed = trial.get("observed") or {}
        pair = trial.get("pair") or {}
        est = trial.get("estimated_structures") or {}
        dims = " ".join(f"{float(value):.3f}" for value in trial["cell"].dims)
        # Use Wyckoff labels from the log if available, else from candidate
        wp_labels = None
        if pair and "wp_candidates" in pair and trial.get("wp_index") is not None:
            for wp in pair["wp_candidates"]:
                if wp.get("wp_index") == trial["wp_index"]:
                    wp_labels = wp.get("wyckoff_labels")
                    break
        if wp_labels is None:
            wp_labels = ast.literal_eval(_format_wyckoff_labels_from_ids(int(trial["candidate"][0]), trial["candidate"][3]))
        best_r2 = _safe_float(observed.get("best_r2"), None)
        best_chi2 = _safe_float(observed.get("best_chi2"), None)
        best_eng = _safe_float(observed.get("best_eng"), None)
        best_sim = _safe_float(observed.get("best_sim"), None)
        wp_index = trial.get('wp_index') or 0
        try:
            wp_index_int = int(wp_index)
        except (ValueError, TypeError):
            wp_index_int = 0  # or another default/fallback value
        logger.info(
            f"{queue_idx:<3} {int(pair.get('spg') or trial['candidate'][0]):<5} "
            f"{int(pair.get('rank') or 0):<5} {wp_index_int:<6} "
            f"{int(est.get('dof') or trial['candidate'][5]):<4} {int(observed.get('refined_count') or 0):<7} "
            f"{(f'{best_r2:.3f}' if best_r2 is not None and best_r2 > -1 else 'n/a'):<8} "
            f"{(f'{best_chi2:.3f}' if best_chi2 is not None and best_chi2 < float('inf') else 'n/a'):<9} "
            f"{(f'{best_eng:.3f}' if best_eng is not None and best_eng < float('inf') else 'n/a'):<8} "
            f"{(f'{best_sim:.3f}' if best_sim is not None and best_sim > -1 else 'n/a'):<8} "
            f"{int(est.get('base_total') or 0):<7} {int(est.get('max_total') or 0):<7} "
            f"{dims} | {wp_labels}"
        )

def _build_resume_trials(parsed_log: dict, state: dict, args: argparse.Namespace) -> list[dict]:
    composition = state.get("composition") or {}
    ref_den = (state.get("density_min", 0.0), state.get("density_max", 0.0))
    ranked_trials: list[dict] = []

    for pair in parsed_log.get("pairs", [])[: max(1, int(args.pair_limit))]:
        cell = _make_cell_from_pair(pair)
        candidates = enumerate_wyckoff_multi_spg(cell.dims, [int(pair["spg"])], composition, ref_den=ref_den)
        by_signature = {
            _candidate_label_signature(int(pair["spg"]), candidate): candidate
            for candidate in candidates
        }
        used_signatures: set = set()

        # Logged candidates
        for wp in pair.get("wp_candidates", []):
            signature = _logged_label_signature(wp)
            candidate = by_signature.get(signature)
            if candidate is None: continue
            used_signatures.add(signature)
            candidate8 = candidate[:8] if len(candidate) > 8 else candidate
            est = _estimate_structures_for_candidate(
                candidate8,
                attempts=int(args.resume_attempts),
                max_local_boosts=int(args.resume_local_boosts),
            )
            ranked_trials.append(
                {
                    "label": f"pair_{pair.get('rank')}_wp_{wp.get('wp_index')}",
                    "pair": pair,
                    "wp_index": wp.get("wp_index"),
                    "cell": cell,
                    "candidate": candidate8,
                    "estimated_structures": est,
                    "summary": f"rank={pair.get('rank')} spg={pair.get('spg')} wp={wp.get('wp_index')} labels={wp.get('wyckoff_labels')}",
                    "observed": {
                        "trial_count": wp.get("trial_count", 0),
                        "refined_count": wp.get("refined_count", 0),
                        "best_r2": wp.get("best_r2", -1.0),
                        "best_chi2": wp.get("best_chi2", float("inf")),
                        "best_eng": wp.get("best_eng", float("inf")),
                        "best_sim": wp.get("best_sim", -1.0),
                    },
                }
            )
            #logger.info(f"Resume trial {wp.get('wyckoff_labels')}: {ranked_trials[-1]['observed']}")

        # Unseen candidates
        unseen_candidates = []
        for candidate in candidates:
            signature = _candidate_label_signature(int(pair["spg"]), candidate)
            if signature in used_signatures:
                continue
            unseen_candidates.append(candidate)
        unseen_candidates8 = [c[:8] if len(c) > 8 else c for c in unseen_candidates]
        unseen_candidates8 = sorted(
            unseen_candidates8,
            key=lambda candidate: score_wp_candidate(candidate, max_dof=int(candidate[5])),
            reverse=True,
        )
        for unseen_idx, candidate8 in enumerate(unseen_candidates8[: max(0, int(args.include_unseen_per_pair))], start=1):
            label_text = _format_wyckoff_labels_from_ids(int(pair["spg"]), candidate8[3])
            est = _estimate_structures_for_candidate(
                candidate8,
                attempts=int(args.resume_attempts),
                max_local_boosts=int(args.resume_local_boosts),
            )
            ranked_trials.append(
                {
                    "label": f"pair_{pair.get('rank')}_wp_unseen{unseen_idx}",
                    "pair": pair,
                    "wp_index": f"unseen{unseen_idx}",
                    "cell": cell,
                    "candidate": candidate8,
                    "estimated_structures": est,
                    "summary": f"rank={pair.get('rank')} spg={pair.get('spg')} wp=unseen{unseen_idx} labels={label_text}",
                    "observed": {
                        "trial_count": 0,
                        "refined_count": 0,
                        "best_r2": -1.0,
                        "best_chi2": float("inf"),
                        "best_eng": float("inf"),
                        "best_sim": -1.0,
                    },
                }
            )

    #import sys; sys.exit(0)
    ranked_trials.sort(key=_resume_rank_key, reverse=True)
    return ranked_trials[: max(1, int(args.candidate_limit))]


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

def _resolve_failure_csvs_from_summary(summary_csv: str, examples_dir: str) -> list:
    failures = []
    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Status", "").strip().lower() == "failure":
                csv_file = row.get("csv_file_name", "").strip()
                if csv_file:
                    path = os.path.join(examples_dir, csv_file)
                    if os.path.isfile(path):
                        failures.append(path)
    return failures


def run_resume(csv_path, args):
    structure_log = []
    # Build run state
    run_state = build_run_state_from_args(default_state, logger, args, csv_path)
    run_state["pxrd_csv"] = csv_path
    min_structures = max(0, int(run_state.get("min_structures_before_early_stop", 10)))
    # Find the original run log
    system_stem = _system_stem(csv_path)
    # Try to find the original log in the output dir, supporting both .txt and .log extensions
    log_candidates = []
    for ext in (".txt", ".log"):
        log_candidates.append(os.path.join(args.output, f"RunLog_{system_stem}{ext}"))
        log_candidates.append(os.path.join(os.path.dirname(csv_path), f"RunLog_{system_stem}{ext}"))
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
    _run_data_preprocessor(csv_path, run_state)
    baseline_outcome = _previous_baseline_outcome(parsed_log, run_state)
    winner_state = run_state
    winner_outcome = baseline_outcome
    followup_trials: list[dict] = []
    combined_plot_log = copy.deepcopy(parsed_log.get("previous_structure_log") or [])

    # Start timing for structure inference
    _start_time = time.time()
    winner_snapshot = _snapshot_artifacts(winner_outcome.get("formula"), winner_outcome.get("spg"), "resume_winner")

    ranked_trials = _build_resume_trials(parsed_log, run_state, args)
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
            print(f"=============Debug struc_log {len(structure_log)} {len(combined_plot_log)}")
            if _is_better_outcome(trial_outcome, winner_outcome):
                winner_state = trial_state
                winner_outcome = trial_outcome
                winner_snapshot = _snapshot_artifacts(
                    winner_outcome.get("formula"),
                    winner_outcome.get("spg"),
                    "resume_winner",
                )
            else:
                _restore_artifacts(winner_snapshot)

            # Enforce early termination as soon as a good outcome is found
            if _is_good_sampling_outcome(trial_outcome, args) and len(structure_log) >= min_structures:
                logger.info(
                    f"Accepted low-dE result found in {trial_outcome.get('label')}; "
                    "stopping resume search early."
                )
                good_outcome = True
                break
        if good_outcome:
            break

    # End timing and update timing_breakdown_seconds
    _end_time = time.time()
    structure_runtime = _end_time - _start_time
    timing = winner_state.get("timing_breakdown_seconds")
    if timing is None:
        timing = {}
        winner_state["timing_breakdown_seconds"] = timing
    timing["structure_inference"] = structure_runtime

    merged_plot = _merged_plot_path(winner_state.get("formula") or _system_stem(csv_path), args.output)
    if combined_plot_log:
        plot_energy_vs_r2(
            combined_plot_log,
            winner_state.get("formula") or _system_stem(csv_path),
            "resume",
            str(merged_plot),
            status="Success" if bool(winner_outcome.get("accepted")) else "Failure",
            timing_breakdown_seconds=timing,
        )
        logger.info(f"Energy–R² plot saved to {merged_plot}")

    md_path = _resume_report_paths(csv_path, args.output)
    summary_path = Path(args.output) / "summary.csv"
    # Append new result to summary.csv
    input_csv = Path(csv_path)
    write_results_csv(input_csv, winner_state, winner_outcome.get("status") if isinstance(winner_outcome, dict) else "unknown")
    # Prepare row data from final_outcome
    print(f"Saved resume report to {md_path} and {summary_path}")

def _run_resume_trial(base_state: dict, trial: dict, args: argparse.Namespace, structure_log: list) -> tuple[dict, dict]:

    trial_state = copy.deepcopy(base_state)
    trial_state["spg"] = int(trial["pair"]["spg"])
    trial_state["cells"] = [copy.deepcopy(trial["cell"])]
    trial_state["multi_attempts"] = 1 #max(int(base_state.get("multi_attempts", 1)), int(args.resume_attempts))
    trial_state["max_local_boosts"] = max(int(base_state.get("max_local_boosts", 0)), int(args.resume_local_boosts))
    trial_state["max_local_perturbations"] = max(
        int(base_state.get("max_local_perturbations", 0)),
        int(args.resume_local_perturbations),
    )
    trial_state["perturb_displacement"] = max(
        float(base_state.get("perturb_displacement", 0.06)),
        float(args.resume_perturb_displacement),
    )
    trial_state["max_eng_rel_early_stop"] = float(args.success_max_eng_rel)
    trial_state["forced_wp_solution"] = trial["candidate"]
    trial_state["suppress_local_energy_plot"] = True
    message = _run_wyckoff_solver(trial_state, [], len(structure_log))
    wyckoff_result = trial_state.get("wyckoff_result") or {}
    status = f"{trial['label']}_success" if wyckoff_result.get("accepted") else "no_solution"
    outcome = _extract_outcome(trial["label"], trial_state, {"status": status, "message": message})
    outcome["summary"] = trial["summary"]
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
        "--examples-dir",
        default="Examples",
        help="Directory used to resolve csv_file_name entries from --summary.",
    )
    parser.add_argument(
        "--log-path",
        default="",
        help="Optional explicit RunLog path for single-file resume runs.",
    )
    parser.add_argument(
        "--pair-limit",
        type=int,
        default=8,
        help="Maximum number of Phase 2 ranked pairs to consider from the previous run log.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=12,
        help="Maximum number of targeted forced-WP resume trials to execute.",
    )
    parser.add_argument(
        "--include-unseen-per-pair",
        type=int,
        default=1,
        help="Additional untested WP candidates to include per Phase 2 pair.",
    )
    parser.add_argument(
        "--resume-attempts",
        type=int,
        default=3,
        help="Adaptive multi-attempt count for each resume trial.",
    )
    parser.add_argument(
        "--resume-local-boosts",
        type=int,
        default=2,
        help="Maximum local regeneration boosts for each resume trial.",
    )
    parser.add_argument(
        "--resume-local-perturbations",
        type=int,
        default=2,
        help="Maximum local perturb-and-relax trials for each resume trial.",
    )
    parser.add_argument(
        "--resume-perturb-displacement",
        type=float,
        default=0.08,
        help="Perturbation displacement in A for resume trials.",
    )
    parser.add_argument(
        "--success-max-eng-rel",
        type=float,
        default=0.20,
        help="Stop once an accepted candidate has dE below this threshold.",
    )

    args = parser.parse_args()
    if args.summary:
        csv_paths = _resolve_failure_csvs_from_summary(args.summary, args.examples_dir)
        if not csv_paths:
            print(f"No failed systems found in summary CSV: {args.summary}")
            sys.exit(1)
    else:
        try:
            csv_paths = [str(path) for path in collect_input_csv_files(args.input)]
        except FileNotFoundError as exc:
            print(str(exc))
            sys.exit(1)

    # Use run_csv_batch for parallel processing if multiple files
    if len(csv_paths) > 1 and args.workers > 1:
        run_csv_batch([Path(p) for p in csv_paths], args, run_resume)
    else:
        for csv_path in csv_paths:
            run_resume(csv_path, args)
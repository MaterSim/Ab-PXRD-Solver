import argparse
import copy
import json
import random
import shutil
import sys
from pathlib import Path

from pxrd_app.cli import build_common_parser, build_run_state_from_args, collect_input_csv_files, run_csv_batch
from PXRD_agent import (
    _attach_system_run_log,
    _detach_system_run_log,
    _run_cell_solver_stage,
    _run_pipeline_fallback,
    _run_wyckoff_solver_stage,
    default_state,
    logger,
)
from tools.solver import enumerate_wyckoff_multi_spg


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _system_stem(csv_path: str) -> str:
    return Path(csv_path).stem or "unknown_system"


def _report_paths(csv_path: str) -> tuple[Path, Path]:
    stem = _system_stem(csv_path)
    return (
        Path("Results") / f"Agent2Report_{stem}.md",
        Path("Results") / f"Agent2Report_{stem}.json",
    )


def _artifact_paths(formula: str | None, spg: int | None) -> list[Path]:
    if not formula or not spg:
        return []
    return [
        Path("Results") / f"Match_{formula}_{spg}.cif",
        Path("Results") / f"EnergyR2_{formula}_{spg}.png",
    ]


def _snapshot_artifacts(formula: str | None, spg: int | None, label: str) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not formula or not spg:
        return snapshot
    snapshot_dir = Path("tmp") / "agent2_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for artifact in _artifact_paths(formula, spg):
        if not artifact.exists():
            continue
        snapshot_path = snapshot_dir / f"{label}_{artifact.name}"
        shutil.copy2(artifact, snapshot_path)
        snapshot[str(artifact)] = str(snapshot_path)
    return snapshot


def _restore_artifacts(snapshot: dict[str, str]) -> None:
    for target, source in snapshot.items():
        source_path = Path(source)
        if not source_path.exists():
            continue
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _extract_outcome(label: str, state: dict, result: dict | None) -> dict:
    wyckoff_result = state.get("wyckoff_result") or {}
    structure_log = state.get("structure_log") or []
    refined_entries = [entry for entry in structure_log if entry.get("refined")]
    best_refined_r2 = max((_safe_float(entry.get("r2"), -1.0) for entry in refined_entries), default=None)
    best_refined_chi2 = min((_safe_float(entry.get("chi2"), 1e9) for entry in refined_entries), default=None)
    min_energy = min((_safe_float(entry.get("eng"), 1e9) for entry in structure_log), default=None)

    return {
        "label": label,
        "status": (result or {}).get("status", "unknown"),
        "message": (result or {}).get("message", ""),
        "spg": state.get("spg"),
        "formula": state.get("formula"),
        "accepted": bool(wyckoff_result.get("accepted", False)),
        "wr": _safe_float(wyckoff_result.get("wr")),
        "r2": _safe_float(wyckoff_result.get("r2")),
        "chi2": _safe_float(wyckoff_result.get("chi2")),
        "score": _safe_float(wyckoff_result.get("score")),
        "selected_energy": _safe_float(wyckoff_result.get("selected_energy")),
        "eng_rel": _safe_float(wyckoff_result.get("eng_rel")),
        "attempt": _safe_int(wyckoff_result.get("attempt")),
        "seed": _safe_int(wyckoff_result.get("seed")),
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
    score = _safe_float(outcome.get("score"), -1e9)
    r2 = _safe_float(outcome.get("r2"), -1.0)
    chi2 = _safe_float(outcome.get("chi2"), 1e9)
    eng_rel = _safe_float(outcome.get("eng_rel"), 1e9)
    structure_count = _safe_int(outcome.get("structure_count"), 0)
    cell_count = _safe_int(outcome.get("cell_count"), 0)
    return (accepted, score, r2, -chi2, -eng_rel, structure_count, cell_count)


def _is_better_outcome(candidate: dict, incumbent: dict) -> bool:
    return _outcome_rank_key(candidate) > _outcome_rank_key(incumbent)


def _classify_scenario(state: dict, outcome: dict, args: argparse.Namespace) -> tuple[str, list[str]]:
    if args.scenario != "auto":
        return args.scenario, [f"Scenario forced from CLI: {args.scenario}"]

    reasons = []
    cells = state.get("cells") or []
    structure_log = state.get("structure_log") or []
    refined_count = outcome.get("refined_count", 0)
    min_r2 = float(state.get("min_r2", 0.95))
    max_chi2 = float(state.get("max_chi2", 0.12))
    r2 = outcome.get("r2")
    chi2 = outcome.get("chi2")
    eng_rel = outcome.get("eng_rel")

    if outcome.get("accepted"):
        reasons.append("Deterministic run already produced an accepted structure.")
        if r2 is not None and r2 >= min_r2:
            reasons.append(f"R2={r2:.4f} meets target {min_r2:.4f}.")
        if chi2 is not None and chi2 <= max_chi2:
            reasons.append(f"Chi2={chi2:.4f} meets target {max_chi2:.4f}.")
        if eng_rel is None or eng_rel <= args.success_max_eng_rel:
            reasons.append("Selected structure is at or near the current best energy.")
            return "near-success", reasons

    if not cells:
        reasons.append("No cell solutions were retained after the deterministic run.")
        return "lack-spg-cell", reasons

    if refined_count > 0 or structure_log:
        reasons.append(
            f"Deterministic run explored {len(structure_log)} structure(s) with {refined_count} refined candidate(s)."
        )
        reasons.append("This indicates the search reached structure generation, so the main issue is likely sampling depth.")
        return "lack-sampling", reasons

    reasons.append(f"Only {len(cells)} cell solution(s) were available and no structures were refined.")
    reasons.append("This points to insufficient SPG/cell coverage rather than structure sampling.")
    return "lack-spg-cell", reasons


def _print_outcome(prefix: str, outcome: dict) -> None:
    score = _safe_float(outcome.get("score"), float("nan"))
    r2 = _safe_float(outcome.get("r2"), float("nan"))
    chi2 = _safe_float(outcome.get("chi2"), float("nan"))
    eng_rel = _safe_float(outcome.get("eng_rel"), float("nan"))
    print(
        f"{prefix}: status={outcome.get('status')}, accepted={outcome.get('accepted')}, "
        f"spg={outcome.get('spg')}, cells={outcome.get('cell_count')}, "
        f"structures={outcome.get('structure_count')}, refined={outcome.get('refined_count')}, "
        f"score={score:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}, dE={eng_rel:.4f}"
    )


def _prepare_trial_state(base_state: dict, *, spg: int | None = None, cells=None, overrides: dict | None = None) -> dict:
    trial_state = copy.deepcopy(base_state)
    if spg is not None:
        trial_state["spg"] = int(spg)
    if cells is not None:
        trial_state["cells"] = copy.deepcopy(cells)
    if overrides:
        trial_state.update(overrides)
    return trial_state


def _run_structure_trial(
    base_state: dict,
    trial_label: str,
    *,
    spg: int,
    cells,
    overrides: dict,
) -> tuple[dict, dict]:
    trial_state = _prepare_trial_state(base_state, spg=spg, cells=cells, overrides=overrides)
    print(f"Running follow-up trial '{trial_label}' on spg={spg} with {len(trial_state.get('cells') or [])} cell(s).")
    message = _run_wyckoff_solver_stage(trial_state)
    wyckoff_result = trial_state.get("wyckoff_result") or {}
    status = f"{trial_label}_success" if wyckoff_result.get("accepted") else "no_solution"
    outcome = _extract_outcome(trial_label, trial_state, {"status": status, "message": message})
    return trial_state, outcome


def _enumerate_quick_check_wp_solutions(base_state: dict, cell, spg: int) -> list:
    composition = base_state.get("composition") or {}
    density_min = base_state.get("density_min")
    density_max = base_state.get("density_max")
    ref_den = (density_min, density_max)
    return enumerate_wyckoff_multi_spg(cell.dims, [int(spg)], composition, ref_den=ref_den)


def _run_quick_check_trials(
    base_state: dict,
    *,
    cell,
    spg: int,
    quick_check_index: int,
    args: argparse.Namespace,
) -> list[tuple[dict, dict]]:
    wp_candidates = _enumerate_quick_check_wp_solutions(base_state, cell, spg)
    total_candidates = len(wp_candidates)
    max_trials = max(1, int(args.quick_check_max_trials))
    print(
        f"Quick-check cell {quick_check_index}: enumerated {total_candidates} WP candidate(s) for spg={spg}."
    )

    if total_candidates == 0:
        return []

    if total_candidates > max_trials:
        rng_seed = int(base_state.get("seed_base", 20260315)) + (1009 * quick_check_index)
        rng = random.Random(rng_seed)
        selected_candidates = rng.sample(wp_candidates, k=max_trials)
        print(
            f"Quick-check cell {quick_check_index}: estimated trial count exceeds {max_trials}; "
            f"sampling {max_trials} random WP trial(s) with seed={rng_seed}."
        )
    else:
        selected_candidates = list(wp_candidates)
        print(f"Quick-check cell {quick_check_index}: trying all {len(selected_candidates)} WP trial(s).")

    outcomes: list[tuple[dict, dict]] = []
    for trial_idx, candidate in enumerate(selected_candidates, start=1):
        forced_wp_solution = candidate[:8] if len(candidate) >= 9 else candidate
        trial_state, trial_outcome = _run_structure_trial(
            base_state,
            f"quick_check_{quick_check_index}_{trial_idx}",
            spg=int(spg),
            cells=[cell],
            overrides={
                "multi_attempts": 1,
                "max_local_boosts": 0,
                "max_local_perturbations": 0,
                "forced_wp_solution": forced_wp_solution,
                "suppress_local_energy_plot": True,
            },
        )
        outcomes.append((trial_state, trial_outcome))
    return outcomes


def _run_expanded_cell_trial(
    base_state: dict,
    trial_label: str,
    *,
    spg: int,
    overrides: dict,
) -> tuple[dict, dict]:
    trial_state = _prepare_trial_state(base_state, spg=spg, overrides=overrides)
    print(f"Running expanded cell search '{trial_label}' on spg={spg}.")
    cell_result = _run_cell_solver_stage(trial_state)
    if not trial_state.get("cells"):
        outcome = _extract_outcome(trial_label, trial_state, cell_result)
        return trial_state, outcome
    message = _run_wyckoff_solver_stage(trial_state)
    wyckoff_result = trial_state.get("wyckoff_result") or {}
    status = f"{trial_label}_success" if wyckoff_result.get("accepted") else "no_solution"
    outcome = _extract_outcome(trial_label, trial_state, {"status": status, "message": message})
    return trial_state, outcome


def _handle_near_success(base_state: dict, winner_state: dict, winner_outcome: dict, args: argparse.Namespace) -> tuple[dict, dict, list[dict]]:
    trials = []
    cells = winner_state.get("cells") or []
    if args.quick_check_cells <= 0 or len(cells) <= 1:
        return winner_state, winner_outcome, trials

    same_spg_snapshot = _snapshot_artifacts(winner_outcome.get("formula"), winner_outcome.get("spg"), "winner_near_success")
    for idx, cell in enumerate(cells[1:1 + args.quick_check_cells], start=1):
        quick_trials = _run_quick_check_trials(
            base_state,
            cell=cell,
            spg=int(winner_outcome["spg"]),
            quick_check_index=idx,
            args=args,
        )
        for trial_state, trial_outcome in quick_trials:
            trials.append(trial_outcome)
            if _is_better_outcome(trial_outcome, winner_outcome):
                winner_state = trial_state
                winner_outcome = trial_outcome
                same_spg_snapshot = _snapshot_artifacts(winner_outcome.get("formula"), winner_outcome.get("spg"), "winner_near_success")
            else:
                _restore_artifacts(same_spg_snapshot)
    return winner_state, winner_outcome, trials


def _handle_lack_sampling(base_state: dict, winner_state: dict, winner_outcome: dict, args: argparse.Namespace) -> tuple[dict, dict, list[dict]]:
    trials = []
    cells = base_state.get("cells") or []
    if not cells:
        return winner_state, winner_outcome, trials

    same_spg_snapshot = _snapshot_artifacts(winner_outcome.get("formula"), winner_outcome.get("spg"), "winner_lack_sampling")

    pooled_state, pooled_outcome = _run_structure_trial(
        base_state,
        "resample_pool",
        spg=int(base_state["spg"]),
        cells=cells[: max(1, int(args.resample_top_cells))],
        overrides={
            "multi_attempts": max(int(base_state.get("multi_attempts", 1)), int(args.resample_attempts)),
            "max_local_boosts": max(int(base_state.get("max_local_boosts", 0)), int(args.resample_local_boosts)),
            "max_local_perturbations": max(int(base_state.get("max_local_perturbations", 0)), int(args.resample_local_perturbations)),
            "perturb_displacement": max(float(base_state.get("perturb_displacement", 0.06)), float(args.resample_perturb_displacement)),
        },
    )
    trials.append(pooled_outcome)
    if _is_better_outcome(pooled_outcome, winner_outcome):
        winner_state = pooled_state
        winner_outcome = pooled_outcome
        same_spg_snapshot = _snapshot_artifacts(winner_outcome.get("formula"), winner_outcome.get("spg"), "winner_lack_sampling")
    else:
        _restore_artifacts(same_spg_snapshot)

    for idx, cell in enumerate(cells[: max(1, int(args.resample_top_cells))], start=1):
        trial_state, trial_outcome = _run_structure_trial(
            base_state,
            f"resample_cell_{idx}",
            spg=int(base_state["spg"]),
            cells=[cell],
            overrides={
                "multi_attempts": max(int(base_state.get("multi_attempts", 1)), int(args.resample_attempts)),
                "max_local_boosts": max(int(base_state.get("max_local_boosts", 0)), int(args.resample_local_boosts)),
                "max_local_perturbations": max(int(base_state.get("max_local_perturbations", 0)), int(args.resample_local_perturbations)),
                "perturb_displacement": max(float(base_state.get("perturb_displacement", 0.06)), float(args.resample_perturb_displacement)),
            },
        )
        trials.append(trial_outcome)
        if _is_better_outcome(trial_outcome, winner_outcome):
            winner_state = trial_state
            winner_outcome = trial_outcome
            same_spg_snapshot = _snapshot_artifacts(winner_outcome.get("formula"), winner_outcome.get("spg"), "winner_lack_sampling")
        else:
            _restore_artifacts(same_spg_snapshot)
    return winner_state, winner_outcome, trials


def _expanded_spg_candidates(base_state: dict, args: argparse.Namespace) -> list[int]:
    candidates = []
    base_spg = _safe_int(base_state.get("spg"))
    if base_spg is not None:
        candidates.append(base_spg)
    if bool(base_state.get("infer_spg_from_pxrd", False)):
        for pred_spg, _prob in (base_state.get("spg_predictions") or [])[: max(1, int(args.expanded_spg_top_k))]:
            pred_spg_int = int(pred_spg)
            if pred_spg_int not in candidates:
                candidates.append(pred_spg_int)
    return candidates


def _handle_lack_spg_cell(base_state: dict, winner_state: dict, winner_outcome: dict, args: argparse.Namespace) -> tuple[dict, dict, list[dict]]:
    trials = []
    spg_candidates = _expanded_spg_candidates(base_state, args)
    for idx, spg in enumerate(spg_candidates, start=1):
        trial_state, trial_outcome = _run_expanded_cell_trial(
            base_state,
            f"expanded_cell_{idx}",
            spg=spg,
            overrides={
                "max_cells": max(int(base_state.get("max_cells", 10)), int(args.expanded_max_cells)),
                "multi_attempts": max(int(base_state.get("multi_attempts", 1)), int(args.expanded_structure_attempts)),
                "max_local_boosts": max(int(base_state.get("max_local_boosts", 0)), int(args.expanded_local_boosts)),
                "max_local_perturbations": max(int(base_state.get("max_local_perturbations", 0)), int(args.expanded_local_perturbations)),
                "perturb_displacement": max(float(base_state.get("perturb_displacement", 0.06)), float(args.expanded_perturb_displacement)),
                "cell_solver_max_mismatch": int(args.expanded_cell_max_mismatch),
                "cell_solver_hkl_max": tuple(int(x) for x in args.expanded_cell_hkl_max),
                "cell_solver_max_square": int(args.expanded_cell_max_square),
                "cell_solver_total_square": int(args.expanded_cell_total_square),
                "cell_solver_theta_tols": [float(x) for x in args.expanded_cell_theta_tols],
                "cell_solver_max_chi2": float(args.expanded_cell_max_chi2),
                "cell_solver_max_guess": int(args.expanded_cell_max_guess),
            },
        )
        trials.append(trial_outcome)
        if _is_better_outcome(trial_outcome, winner_outcome):
            winner_state = trial_state
            winner_outcome = trial_outcome
    return winner_state, winner_outcome, trials


def _write_report(
    csv_path: str,
    scenario: str,
    reasons: list[str],
    baseline_outcome: dict,
    followup_trials: list[dict],
    final_outcome: dict,
) -> tuple[Path, Path]:
    md_path, json_path = _report_paths(csv_path)
    md_lines = [
        f"# PXRD Agent 2 Report: {_system_stem(csv_path)}",
        "",
        f"Input CSV: {csv_path}",
        f"Scenario: {scenario}",
        "",
        "## Classification Rationale",
    ]
    md_lines.extend(f"- {reason}" for reason in reasons)
    md_lines.extend([
        "",
        "## Baseline Outcome",
        f"- status: {baseline_outcome.get('status')}",
        f"- accepted: {baseline_outcome.get('accepted')}",
        f"- spg: {baseline_outcome.get('spg')}",
        f"- score: {baseline_outcome.get('score')}",
        f"- R2: {baseline_outcome.get('r2')}",
        f"- Chi2: {baseline_outcome.get('chi2')}",
        f"- dE: {baseline_outcome.get('eng_rel')}",
        f"- cells: {baseline_outcome.get('cell_count')}",
        f"- structures explored: {baseline_outcome.get('structure_count')}",
        "",
        "## Follow-up Trials",
    ])
    if followup_trials:
        for trial in followup_trials:
            md_lines.append(
                f"- {trial.get('label')}: status={trial.get('status')}, accepted={trial.get('accepted')}, "
                f"spg={trial.get('spg')}, score={trial.get('score')}, R2={trial.get('r2')}, Chi2={trial.get('chi2')}, dE={trial.get('eng_rel')}"
            )
    else:
        md_lines.append("- No follow-up trial was required.")
    md_lines.extend([
        "",
        "## Final Outcome",
        f"- label: {final_outcome.get('label')}",
        f"- status: {final_outcome.get('status')}",
        f"- accepted: {final_outcome.get('accepted')}",
        f"- spg: {final_outcome.get('spg')}",
        f"- score: {final_outcome.get('score')}",
        f"- R2: {final_outcome.get('r2')}",
        f"- Chi2: {final_outcome.get('chi2')}",
        f"- dE: {final_outcome.get('eng_rel')}",
        f"- log path: {final_outcome.get('log_path')}",
        "",
    ])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "csv_path": csv_path,
                "scenario": scenario,
                "reasons": reasons,
                "baseline": baseline_outcome,
                "followup_trials": followup_trials,
                "final": final_outcome,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return md_path, json_path


def run_agent2(csv_path: str, args: argparse.Namespace) -> None:
    run_state = build_run_state_from_args(default_state, logger, args, csv_path)
    system_log_handler = _attach_system_run_log(run_state)
    winner_state = run_state
    winner_outcome = None
    followup_trials: list[dict] = []

    try:
        print("Starting PXRD agent 2 workflow.")
        baseline_result = _run_pipeline_fallback(
            run_state,
            announce_bug_switch=False,
            status_label="deterministic_success",
        )
        baseline_outcome = _extract_outcome("baseline", run_state, baseline_result)
        winner_state = run_state
        winner_outcome = baseline_outcome
        _print_outcome("Baseline", baseline_outcome)

        scenario, reasons = _classify_scenario(run_state, baseline_outcome, args)
        print(f"Scenario selected: {scenario}")
        for reason in reasons:
            print(f"  - {reason}")

        followup_base_state = copy.deepcopy(run_state)
        if scenario == "near-success":
            winner_state, winner_outcome, followup_trials = _handle_near_success(
                followup_base_state,
                winner_state,
                winner_outcome,
                args,
            )
        elif scenario == "lack-sampling":
            winner_state, winner_outcome, followup_trials = _handle_lack_sampling(
                followup_base_state,
                winner_state,
                winner_outcome,
                args,
            )
        else:
            winner_state, winner_outcome, followup_trials = _handle_lack_spg_cell(
                followup_base_state,
                winner_state,
                winner_outcome,
                args,
            )

        for trial in followup_trials:
            _print_outcome(f"Follow-up {trial.get('label')}", trial)
        _print_outcome("Final", winner_outcome)

        md_path, json_path = _write_report(
            csv_path,
            scenario,
            reasons,
            baseline_outcome,
            followup_trials,
            winner_outcome,
        )
        print(f"Saved agent-2 markdown report to {md_path}")
        print(f"Saved agent-2 JSON report to {json_path}")
    except KeyboardInterrupt:
        print("Process interrupted by user")
    finally:
        system_log_path = winner_state.get("system_run_log") if isinstance(winner_state, dict) else run_state.get("system_run_log")
        if system_log_path:
            print(f"Saved consolidated run log to {system_log_path}")
        _detach_system_run_log(system_log_handler)
        print("Exiting PXRD agent 2")


def _build_parser() -> argparse.ArgumentParser:
    parser = build_common_parser("Run PXRD agent 2 workflow")
    parser.add_argument(
        "--scenario",
        choices=["auto", "near-success", "lack-sampling", "lack-spg-cell"],
        default="auto",
        help="Override automatic scenario classification.",
    )
    parser.add_argument(
        "--success-max-eng-rel",
        type=float,
        default=0.20,
        help="Maximum dE (eV/atom) still considered clearly near-successful.",
    )
    parser.add_argument(
        "--quick-check-cells",
        type=int,
        default=2,
        help="Number of alternative existing cells to quick-check in the near-success scenario.",
    )
    parser.add_argument(
        "--quick-check-attempts",
        type=int,
        default=1,
        help="Deprecated compatibility knob. Quick-check now runs single forced WP trials.",
    )
    parser.add_argument(
        "--quick-check-max-trials",
        type=int,
        default=10,
        help="Maximum number of random forced WP trials per quick-check cell.",
    )
    parser.add_argument(
        "--quick-check-local-boosts",
        type=int,
        default=0,
        help="Deprecated compatibility knob. Quick-check skips local boost passes.",
    )
    parser.add_argument(
        "--quick-check-local-perturbations",
        type=int,
        default=0,
        help="Deprecated compatibility knob. Quick-check skips local perturbation passes.",
    )
    parser.add_argument(
        "--resample-top-cells",
        type=int,
        default=3,
        help="Number of current cells to revisit when the failure looks like insufficient sampling.",
    )
    parser.add_argument(
        "--resample-attempts",
        type=int,
        default=4,
        help="Wyckoff attempts to use when resampling existing cells.",
    )
    parser.add_argument(
        "--resample-local-boosts",
        type=int,
        default=2,
        help="Local regeneration boosts for resampling existing cells.",
    )
    parser.add_argument(
        "--resample-local-perturbations",
        type=int,
        default=4,
        help="Local perturb-and-relax trials for resampling existing cells.",
    )
    parser.add_argument(
        "--resample-perturb-displacement",
        type=float,
        default=0.08,
        help="Perturbation displacement in A for resampling existing cells.",
    )
    parser.add_argument(
        "--expanded-spg-top-k",
        type=int,
        default=5,
        help="How many inferred SPGs to revisit in the expanded cell-search scenario.",
    )
    parser.add_argument(
        "--expanded-max-cells",
        type=int,
        default=20,
        help="Maximum number of cell solutions to retain in expanded cell search.",
    )
    parser.add_argument(
        "--expanded-structure-attempts",
        type=int,
        default=4,
        help="Wyckoff attempts after expanded cell search.",
    )
    parser.add_argument(
        "--expanded-local-boosts",
        type=int,
        default=2,
        help="Local regeneration boosts after expanded cell search.",
    )
    parser.add_argument(
        "--expanded-local-perturbations",
        type=int,
        default=4,
        help="Local perturb-and-relax trials after expanded cell search.",
    )
    parser.add_argument(
        "--expanded-perturb-displacement",
        type=float,
        default=0.08,
        help="Perturbation displacement in A after expanded cell search.",
    )
    parser.add_argument(
        "--expanded-cell-max-mismatch",
        type=int,
        default=18,
        help="Maximum mismatch allowed during expanded cell search.",
    )
    parser.add_argument(
        "--expanded-cell-hkl-max",
        type=int,
        nargs=3,
        default=[3, 6, 8],
        metavar=("H", "K", "L"),
        help="Expanded hkl limit for cell search.",
    )
    parser.add_argument(
        "--expanded-cell-max-square",
        type=int,
        default=40,
        help="Expanded max square for cell search.",
    )
    parser.add_argument(
        "--expanded-cell-total-square",
        type=int,
        default=60,
        help="Expanded total square for cell search.",
    )
    parser.add_argument(
        "--expanded-cell-theta-tols",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.6],
        help="Expanded theta tolerances for cell search.",
    )
    parser.add_argument(
        "--expanded-cell-max-chi2",
        type=float,
        default=0.8,
        help="Expanded chi2 cutoff for peak matching during cell search.",
    )
    parser.add_argument(
        "--expanded-cell-max-guess",
        type=int,
        default=120000,
        help="Expanded maximum number of cell guesses to evaluate.",
    )
    return parser


def run_agent2_csv(csv_path: str, args: argparse.Namespace) -> None:
    run_agent2(csv_path, args)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        csv_files = collect_input_csv_files(args.input_csv)
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)
    try:
        run_csv_batch(csv_files, args, run_agent2_csv)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
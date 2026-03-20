#!/usr/bin/env python3
"""Smoke-test tool imports and callable entry points for PXRD-agent."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    optional: bool = False


def _run_check(name: str, fn: Callable[[], str], optional: bool = False) -> CheckResult:
    try:
        detail = fn()
        return CheckResult(name=name, ok=True, detail=detail, optional=optional)
    except Exception as exc:
        return CheckResult(name=name, ok=False, detail=str(exc), optional=optional)


def check_utils() -> str:
    from tools.utils import parse_formula

    comp = parse_formula("Ba4NaBi")
    if comp.get("Ba") != 4 or comp.get("Na") != 1 or comp.get("Bi") != 1:
        raise ValueError(f"Unexpected parse_formula output: {comp}")
    return "parse_formula works"


def check_manager_imports() -> str:
    from tools.manager import RawDataManager, CellManager, WPManager  # noqa: F401

    return "RawDataManager/CellManager/WPManager imported"


def check_solver_imports() -> str:
    from tools.solver import CellSolver, SmartCellSolver, search_solution  # noqa: F401

    return "CellSolver/SmartCellSolver/search_solution imported"


def check_xrd_imports() -> str:
    from tools.XRD import Profile, Similarity, XRD  # noqa: F401

    return "Profile/Similarity/XRD imported"


def check_peak_prediction_imports() -> str:
    from tools.peak_prediction import predict_peaks, predict_spacegroup  # noqa: F401

    return "predict_peaks/predict_spacegroup imported"


def check_density_imports() -> str:
    from tools.density import predict_density_ensemble  # noqa: F401

    return "predict_density_ensemble imported"


def check_gsas() -> str:
    from tools.gsas import check_gsas_available, refine_pxrd  # noqa: F401

    ok, msg = check_gsas_available()
    if not ok:
        raise RuntimeError(msg)
    return "GSAS-II available and refine_pxrd importable"


def check_gsas_light_refinement() -> str:
    """Run a lightweight real refinement using bundled example inputs."""
    from tools.gsas import refine_pxrd

    pxrd = REPO_ROOT / "Examples" / "PXRD_PrYMg2_123.csv"
    cif = REPO_ROOT / "Examples" / "Reference_PrYMg2.cif"
    inst = REPO_ROOT / "tools" / "INST_XRY.PRM"

    if not pxrd.exists() or not cif.exists() or not inst.exists():
        raise FileNotFoundError("Missing example files required for light refinement test")

    wr, r2, chi2, refined_cif = refine_pxrd(
        pxrd_file=str(pxrd),
        cif_file=str(cif),
        instprm=str(inst),
        gpx_name=str(REPO_ROOT / "tmp" / "gsas_light_test.gpx"),
        gsas_log=str(REPO_ROOT / "tmp" / "gsas_light_test.log"),
    )

    if wr is None or r2 is None or chi2 is None or refined_cif is None:
        raise RuntimeError("Light refinement did not return valid metrics")

    return f"light refinement ok (Rwp={wr:.3f}, R2={r2:.4f}, chi2={chi2:.4f})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test PXRD-agent tool callability")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat optional checks (GSAS) as required and fail when unavailable",
    )
    parser.add_argument(
        "--skip-light-refinement",
        action="store_true",
        help="Skip the default lightweight real GSAS refinement example",
    )
    args = parser.parse_args()

    checks = [
        ("utils", check_utils, False),
        ("manager", check_manager_imports, False),
        ("solver", check_solver_imports, False),
        ("xrd", check_xrd_imports, False),
        ("peak_prediction", check_peak_prediction_imports, False),
        ("density", check_density_imports, False),
        ("gsas", check_gsas, True),
    ]

    if not args.skip_light_refinement:
        checks.append(("gsas_light_refinement", check_gsas_light_refinement, False))

    results: list[CheckResult] = []
    for name, fn, optional in checks:
        results.append(_run_check(name, fn, optional=optional))

    failed_required = 0
    failed_optional = 0

    print("=== Tool Smoke Test ===")
    for r in results:
        if r.ok:
            print(f"[OK]   {r.name}: {r.detail}")
            continue

        if r.optional and not args.strict:
            failed_optional += 1
            print(f"[WARN] {r.name}: {r.detail}")
        else:
            failed_required += 1
            print(f"[FAIL] {r.name}: {r.detail}")

    print("-----------------------")
    print(f"Required failures: {failed_required}")
    print(f"Optional warnings: {failed_optional}")

    return 1 if failed_required > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

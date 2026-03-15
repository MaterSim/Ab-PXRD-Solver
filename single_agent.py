from strands import Agent, tool, ToolContext
from strands.models.gemini import GeminiModel
from strands.multiagent.graph import GraphBuilder
import logging
import os
import inspect
import sys
import traceback
from importlib.metadata import version as pkg_version, PackageNotFoundError
import pandas as pd
import numpy as np
from tools.manager import RawDataManager, CellManager
from tools.solver import CellSolver, search_solution
from tools.utils import parse_formula, get_volume_from_density
from tools.density import predict_density_ensemble

# Configure logging with both file and console handlers
file_handler = logging.FileHandler('single_agent.log')
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logging.root.addHandler(file_handler)
logging.root.addHandler(console_handler)
logging.root.setLevel(logging.INFO)

logger = logging.getLogger("strands.multiagent")
logger.setLevel(logging.ERROR)


class StreamToLogger:
    """Redirect writes to a logger instance."""
    def __init__(self, logger_instance, level):
        self.logger = logger_instance
        self.level = level
        self._buffer = ""

    def write(self, message):
        if not message:
            return

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer:
            line = self._buffer.rstrip()
            if line:
                self.logger.log(self.level, line)
            self._buffer = ""


# Redirect stdout/stderr to logging to capture all output including strands library
sys.stdout = StreamToLogger(logging.getLogger("stdout"), logging.INFO)
#sys.stderr = StreamToLogger(logging.getLogger("stderr"), logging.WARNING)


share_state = {
    # Raw inputs
    "pxrd_csv": "Examples/PXRD_PrYMg2_123.csv",
    "formula": "",
    "composition": {},
    # To be filled by Data Agents
    "x1": [],
    "y1": [],
    "peaks": [],
    "peak_positions": [],
    # To be filled by Solver Agents
    "spg": 0,
    "density_min": 0.0,
    "density_max": 0.0,
    "min_volume": 0.0,
    "min_abc": 2.0,
    "cells": [],
    # Constraints and parameters
    "wavelength": 1.54184,
    "min_r2": 0.95,
    "max_chi2": 0.12,
    "INST_FILE": "tools/INST_XRY.PRM",
    "SCALED_INTENSITY_TOL": 0.01,
    "thetas": [10, 80],
    "resolution": 0.02,
    "max_force": 0.5,
    "max_stress": 0.3,
    "max_cells": 10,
}


gemini_model = GeminiModel(
    client_args={'api_key': 'AIzaSyA2TT4RqCvrY-RwRNmhT8AnCLwH-IwvdE8'},
    model_id='gemini-2.5-pro',
    params={"temperature": 0.7}
)

INPUT_PROMPT = "Process the PXRD data from Examples/PXRD_PrYMg2_123.csv"


def _run_data_preprocessor_stage(pxrd_csv: str, state: dict) -> dict:
    formula = pxrd_csv.split('_')[1]
    spg = int(pxrd_csv.split('_')[2].split('.')[0])
    composition = parse_formula(formula)

    df = pd.read_csv(pxrd_csv, comment='#')
    x1 = df.iloc[:, 0].values
    y1 = df.iloc[:, 1].values
    data = RawDataManager(x1, y1, bg_subtract=False)
    data.get_peaks_from_scipy()
    data.filter_peaks_by_ml(threshold=0.8, min_height=3.0)
    peaks = data.peaks
    peak_positions = x1[peaks]

    min_abc = 2.0
    wavelength = 1.54184
    density = predict_density_ensemble(formula, sigma=2.5)
    density_min = float(density['min'])
    density_max = float(density['max'])
    min_volume = float(get_volume_from_density(composition, density['max']))

    result = {
        "spg": int(spg),
        "formula": formula,
        "x1": x1.tolist(),
        "y1": y1.tolist(),
        "peaks": peaks.tolist(),
        "peak_positions": peak_positions.tolist(),
        "composition": composition,
        "density_min": density_min,
        "density_max": density_max,
        "min_volume": min_volume,
        "min_abc": min_abc,
        "wavelength": wavelength,
    }
    state.update(result)
    return result


def _run_cell_solver_stage(state: dict) -> dict:
    spg = state.get("spg")
    formula = state.get("formula")
    peak_positions = state.get("peak_positions")
    max_cells = state.get("max_cells")

    peak_positions_np = np.array(peak_positions)
    solver = CellSolver(
        spg,
        peak_positions_np,
        max_mismatch=12,
        hkl_max=(2, 5, 6),
        max_square=28,
        total_square=40,
        theta_tols=[0.1, 0.15, 0.5],
        verbose=False,
    )
    solutions = solver.solve()
    sols = [
        (spg, sol['cell'], sol['mismatch'], sol['chi2'][1], sol['errors'], sol['id'], sol['match'])
        for sol in solutions
    ]
    cells = CellManager.consolidate(sols, max_solutions=max_cells, merge_tol=0.05)

    state["cells"] = cells
    text = f"Cell solving completed for formula {formula} in space group {spg}.\n"
    return {
        "status": "success",
        "message": text,
        "cells": [{"dimensions": cell.dims, "missing_peaks": cell.missing} for cell in cells],
    }


def _run_wyckoff_solver_stage(state: dict) -> str:
    spg = state.get("spg")
    formula = state.get("formula")
    cells = state.get("cells")
    composition = state.get("composition")
    density_min = state.get("density_min")
    density_max = state.get("density_max")
    wavelength = state.get("wavelength")
    pxrd_csv = state.get("pxrd_csv")
    INST_FILE = state.get("INST_FILE")
    thetas = state.get("thetas")
    resolution = state.get("resolution")
    SCALED_INTENSITY_TOL = state.get("SCALED_INTENSITY_TOL")
    ref_den = (density_min, density_max)
    x1 = np.array(state.get("x1"))
    y1 = np.array(state.get("y1"))
    peaks = np.array(state.get("peaks"))
    min_r2 = state.get("min_r2")
    max_chi2 = state.get("max_chi2")
    max_force = state.get("max_force")
    max_stress = state.get("max_stress")

    N1, N2, N3 = 5, 20, 9
    eng_min, sim_max = 1e10, 0.90

    os.makedirs("Results", exist_ok=True)

    title = f'{formula} PXRD Prediction: Space Group {spg}'
    match_png = f"Results/Match_{formula}_{spg}.png"
    match_cif = f'Results/Match_{formula}_{spg}.cif'
    (wr, r2, chi2, xtal, _) = search_solution(
        cells[:N1],
        spg,
        composition,
        ref_den,
        title,
        match_png,
        match_cif,
        pxrd_csv,
        peaks,
        x1,
        y1,
        eng_min,
        sim_max,
        N1,
        N2,
        N3,
        max_force,
        max_stress,
        wavelength,
        thetas,
        resolution,
        SCALED_INTENSITY_TOL,
        INST_FILE,
        logger,
        min_r2,
        max_chi2,
    )

    if wr is None:
        logger.info("No satisfactory solution found.")
    else:
        logger.info(f"\nFinal refinement results: Wr={wr:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}")
        logger.info(f"Best structure saved to {match_cif} and {match_png}")
        logger.info(xtal)

    text = f"Wyckoff solving completed for formula {formula} in space group {spg}.\n"
    text += f"Best similarity: {sim_max:.3f}, Minimum energy per atom: {eng_min:.3f} eV\n"
    if wr is not None:
        text += f"Final Rietveld refinement results: Wr={wr:.4f}, R2={r2:.4f}, Chi2={chi2:.4f}\n"
        text += f"Best structure saved to {match_cif} and {match_png}\n"
    else:
        text += "No satisfactory solution found.\n"
    return text


def _is_strands_gemini_stream_bug(error: BaseException) -> bool:
    error_text = f"{error}\n{traceback.format_exc()}"
    return (
        "strands/models/gemini.py" in error_text
        and "candidate" in error_text
        and "UnboundLocalError" in error_text
    )


def _get_strands_version() -> str:
    try:
        return pkg_version("strands")
    except PackageNotFoundError:
        return "unknown"


def _has_known_gemini_stream_candidate_bug() -> bool:
    try:
        src = inspect.getsource(GeminiModel.stream)
    except Exception:
        return False
    vulnerable_finish_reason_line = "candidate.finish_reason if candidate else \"STOP\""
    has_guard_initialization = "candidate = None" in src
    return vulnerable_finish_reason_line in src and not has_guard_initialization


def _startup_runtime_mode() -> tuple[bool, str]:
    strands_version = _get_strands_version()
    force_fallback_raw = os.getenv("STRANDS_FORCE_FALLBACK", "0")
    allow_graph_raw = os.getenv("STRANDS_ALLOW_GRAPH_WITH_KNOWN_BUG", "0")
    env_force_fallback = force_fallback_raw == "1"
    env_allow_graph = allow_graph_raw == "1"
    has_known_bug = _has_known_gemini_stream_candidate_bug()

    if has_known_bug:
        logger.warning(
            "Detected known Strands Gemini streaming candidate bug in installed package "
            f"(strands=={strands_version}). "
            "Graph execution may crash; fallback mode is recommended."
        )

    use_fallback = env_force_fallback or (has_known_bug and not env_allow_graph)
    mode = "fallback" if use_fallback else "graph"
    print(
        "Startup flags: "
        f"STRANDS_FORCE_FALLBACK={force_fallback_raw}, "
        f"STRANDS_ALLOW_GRAPH_WITH_KNOWN_BUG={allow_graph_raw}"
    )
    logger.info(
        f"Runtime mode: {mode} (strands=={strands_version}, "
        f"known_gemini_stream_bug={has_known_bug}, "
        f"STRANDS_FORCE_FALLBACK={env_force_fallback}, "
        f"STRANDS_ALLOW_GRAPH_WITH_KNOWN_BUG={env_allow_graph})"
    )
    return use_fallback, strands_version


def _run_pipeline_fallback(state: dict) -> dict:
    logger.info("Detected Strands Gemini streaming bug; switching to deterministic fallback pipeline.")
    _run_data_preprocessor_stage(state["pxrd_csv"], state)
    _run_cell_solver_stage(state)
    wyckoff_message = _run_wyckoff_solver_stage(state)
    return {
        "status": "fallback_success",
        "message": wyckoff_message,
        "spg": state.get("spg"),
        "formula": state.get("formula"),
    }

@tool(context=True)
def WyckoffSolverTool(tool_context: ToolContext) -> str:
    return _run_wyckoff_solver_stage(tool_context.invocation_state)

@tool(context=True)
def CellSolverTool(tool_context: ToolContext) -> dict:
    return _run_cell_solver_stage(tool_context.invocation_state)

@tool(context=True)
def DataPreprocessor(pxrd_csv: str, tool_context: ToolContext) -> dict:
    return _run_data_preprocessor_stage(pxrd_csv, tool_context.invocation_state)

DataPreprocessAgent = Agent(
    model=gemini_model,
    tools=[DataPreprocessor],
    system_prompt=(
        "You are a PXRD (Powder X-Ray Diffraction) data analysis specialist.\n\n"
        "Your primary task is to preprocess experimental PXRD data for crystal structure determination.\n\n"
        "When given a PXRD CSV file path, you should:\n"
        "1. Extract the chemical formula and space group from the filename\n"
        "2. Load and process the diffraction pattern data\n"
        "3. Identify characteristic peaks using scipy algorithms\n"
        "4. Predict material density using ensemble ML models\n"
        "5. Calculate minimum volume constraints for unit cell indexing\n\n"
        "Always use the DataPreprocessor tool to perform these tasks.\n"
        "Report any issues with data quality or processing errors immediately.\n"
        "The return format should include status, messages, and all relevant computed parameters."
    )
)

CellManagerAgent = Agent(
    model=gemini_model,
    tools=[CellSolverTool],
    system_prompt=(
        "You are a PXRD unit cell solver specialist.\n\n"
        "Your primary task is to determine unit cell parameters from the given PXRD peak data.\n\n"
        "For the given peak positions, chemical composition, space group, and constraints, you should:\n"
        "1. Use indexing algorithms to find candidate unit cells that fit the peak data\n"
        "2. Apply constraints based on composition and predicted density to filter solutions\n"
        "3. Rank candidate cells based on fit quality and physical plausibility\n\n"
        "Always use the CellSolver tool to perform these tasks.\n"
        "Report any issues with indexing or solution quality immediately.\n"
        "The return format should include status, messages, and all relevant computed unit cell parameters."
    )
)

WyckoffSolverAgent = Agent(
    model=gemini_model,
    tools=[WyckoffSolverTool],
    system_prompt=(
        "You are a specialist in crystal structure generation and optimization.\n\n"
        "Your primary task is to generate candidate crystal structures from indexed unit cells "
        "and optimize them to match experimental PXRD data.\n\n"

        "**Your Workflow:**\n"
        "1. **Wyckoff Position Assignment**\n"
        "   - For each candidate unit cell, enumerate possible Wyckoff position combinations\n"
        "   - Consider space group symmetry constraints\n"
        "   - Filter based on composition and density constraints\n\n"

        "2. **Structure Generation**\n"
        "   - Generate initial atomic positions using symmetry operations\n"
        "   - Validate structural geometry and atomic overlaps\n"
        "   - Generate multiple random configurations per Wyckoff assignment\n\n"

        "3. **Geometry Optimization**\n"
        "   - Relax atomic positions using ASE with MACE force field\n"
        "   - Track energy minimization to identify stable configurations\n\n"
        "   - Apply stress constraints (max stress < 0.5 GPa initially)\n"

        "4. **XRD Pattern Matching**\n"
        "   - Calculate theoretical XRD patterns for optimized structures\n"
        "   - Compare with experimental data using similarity metrics\n"
        "   - Track best matches (similarity > 0.90)\n\n"

        "5. **Rietveld Refinement**\n"
        "   - Perform full-pattern refinement using GSAS-II for promising candidates\n"
        "   - Calculate fit metrics: Rwp, R², χ²\n"
        "   - Accept solutions with R² > 0.95 or χ² < 0.12\n\n"

        "**Search Strategy:**\n"
        "- Test top 5 unit cells (ranked by missing peaks)\n"
        "- Evaluate up to 20 Wyckoff position combinations per cell\n"
        "- Generate 3×DOF + 1 random structures per combination (max DOF=9)\n"
        "- Stop immediately when R² > 0.95 or χ² < 0.12 is achieved\n\n"

        "**Quality Criteria:**\n"
        "- Structural validity (no atomic overlaps)\n"
        "- Converged geometry (stress < 0.5 GPa)\n"
        "- Low potential energy per atom\n"
        "- High XRD pattern similarity (> 0.90)\n"
        "- Excellent Rietveld fit (R² > 0.95 or χ² < 0.12)\n\n"

        "**Expected Input (from previous agents):**\n"
        "- Space group number\n"
        "- Chemical formula and composition\n"
        "- List of indexed unit cells with dimensions\n"
        "- Experimental PXRD data (x1, y1, peaks)\n"
        "- Density constraints (min, max)\n"
        "- X-ray wavelength\n\n"

        "**Output Format:**\n"
        "Report the following:\n"
        "1. Number of cells tested\n"
        "2. Total structures generated and optimized\n"
        "3. Best similarity score and energy achieved\n"
        "4. Final Rietveld refinement metrics (Rwp, R², χ²)\n"
        "5. Whether a satisfactory solution was found (R² > 0.95)\n"
        "6. Paths to saved structure file (.cif) and XRD comparison plot (.png)\n\n"

        "Always use the WyckoffSolverTool to perform these computationally intensive tasks.\n"
        "This tool may take several minutes to hours depending on complexity.\n"
        "Report progress updates and immediately notify when a satisfactory solution is found.\n"
        "If no solution meets the R² threshold after exhausting the search space, "
        "recommend adjustments to constraints or suggest alternative space groups."
    )
)

builder = GraphBuilder()
builder.add_node(DataPreprocessAgent, "DataPreprocessorAgent")
builder.add_node(CellManagerAgent, "CellSolverAgent")
builder.add_node(WyckoffSolverAgent, "WyckoffSolverAgent")
builder.add_edge("DataPreprocessorAgent", "CellSolverAgent")
builder.add_edge("CellSolverAgent", "WyckoffSolverAgent")
builder.set_entry_point("DataPreprocessorAgent")
graph = builder.build()

force_fallback, strands_version = _startup_runtime_mode()

try:
    if force_fallback:
        print("Starting pipeline in fallback mode.")
        result = _run_pipeline_fallback(share_state)
        print("Pipeline completed successfully via fallback execution!")
    else:
        result = graph(INPUT_PROMPT,
                       invocation_state=share_state)
        print("Pipeline completed successfully!")
except KeyboardInterrupt:
    print("Process interrupted by user")
except Exception as exc:
    if _is_strands_gemini_stream_bug(exc):
        print("Strands Gemini streaming error detected; retrying with fallback execution.")
        result = _run_pipeline_fallback(share_state)
        print("Pipeline completed successfully via fallback execution!")
    else:
        raise
print("Exiting main thread")

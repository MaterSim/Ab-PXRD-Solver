# PXRD-Agent: Automated Crystal Determination from Powder X-Ray Diffraction

## Overview

PXRD-Agent is an agentic pipeline that automates the *ab initio* crystal structure determination workflow from experimental Powder X-Ray Diffraction (PXRD) data. Given a measured diffraction pattern and the chemical formula of the material, the system autonomously:

1. Preprocesses and cleans the raw diffraction pattern to get peaks and space group symmetry (a pretrained CNN model)
2. Predicts a physical density range from the given chemical formula (a pretrained ML ensemble model).
3. Indexes the peaks to candidate unit cells (tools.solver)
4. Samples Wyckoff position combinations and generates trial crystal structures (`PyXtal` or others)
5. Relaxes structures with a neural-network force field (Foundational `MACE-MLFF`)
6. Matches simulated patterns against the experiment and refines promising solutions with Rietveld refinement (`GSAS2` utility)

The pipeline is implemented with the [Strands](https://github.com/strands-agents/sdk-python) agentic framework and uses a **Gemini 2.5 Pro** LLM as the reasoning backbone for each specialist agent.

## Recent Search Improvements

- **Top-k inferred space-group iteration:** `--infer-spg` can now try ranked space groups up to `--spg-top-k` (supports `3`, `5`, `10`, `20`) and stop early on accepted solutions.
- **Cell reuse across inferred SGs:** after obtaining candidate cells from the first viable inferred SG, the same cells are reused while testing other inferred space groups during Wyckoff/structure generation.
- **Wyckoff reranking + adaptive expansion:** candidates are reranked (`score_wp_candidate`) and explored progressively (`top-3 → top-5 → top-10 → up to N2`) to reduce unnecessary expensive relaxations.
- **Supercell-aware budgeting:** likely near-integer supercells are still checked but with reduced per-cell search effort (`N2/N3`), prioritizing primitive/smaller cells first.
- **Composite refinement trigger:** refinement uses similarity plus relative energy (`eng_rel = eng - eng_best`) rather than similarity alone.
- **Transparent skip diagnostics:** when refinement is skipped despite promising similarity, logs now include the explicit reason and `eng_rel`.

---

## System Architecture

The pipeline follows a **linear directed acyclic graph (DAG)** of three specialist agents, each wrapping one Python tool:

```
┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│  DataPreprocessAgent │ ───► │   CellManagerAgent   │ ───► │  WyckoffSolverAgent  │
│  (DataPreprocessor)  │      │   (CellSolverTool)   │      │  (WyckoffSolverTool) │
└──────────────────────┘      └──────────────────────┘      └──────────────────────┘
```

All three agents share a single mutable `invocation_state` dictionary (`share_state`) that carries every intermediate result from one stage to the next without re-serialising through the LLM context.

### Runtime Robustness (new)

The default execution path is still the Strands graph, but `single_agent.py` now includes a **deterministic fallback pipeline** for runtime reliability:

- It first attempts graph execution.
- If a known Strands Gemini streaming bug is detected (the `candidate` `UnboundLocalError`), it automatically falls back to sequential stage execution:
  1. data preprocessing,
  2. cell solving,
  3. Wyckoff/structure search.

The script also prints startup runtime-control flags and selected mode for reproducibility.

#### Environment flags

- `STRANDS_FORCE_FALLBACK=1`: always run deterministic fallback mode.
- `STRANDS_ALLOW_GRAPH_WITH_KNOWN_BUG=1`: allow graph mode even if vulnerable Strands code is detected.

### Quick Start / Run

Run from the repository root:

```bash
python single_agent.py
```

The script prints startup flags, selected runtime mode (`graph` or `fallback`), and progress logs.

#### Force deterministic fallback mode

```bash
STRANDS_FORCE_FALLBACK=1 python single_agent.py
```

#### Force graph mode even when known bug signature is detected

```bash
STRANDS_ALLOW_GRAPH_WITH_KNOWN_BUG=1 python single_agent.py
```

#### Optional: show both runtime flags explicitly

```bash
STRANDS_FORCE_FALLBACK=0 STRANDS_ALLOW_GRAPH_WITH_KNOWN_BUG=0 python single_agent.py
```

#### Infer space group from PXRD model instead of filename

By default, the space group is read from the filename convention (`PXRD_<formula>_<spg>.csv`). Pass `--infer-spg` to instead predict it from the diffraction pattern using the pretrained space group classifier:

```bash
# Infer top-5 space groups (default) and try each until a solution is found
python single_agent.py --infer-spg --input-csv Examples/PXRD_Ba4NaBi_216.csv

# Try up to 20 predicted space groups
python single_agent.py --infer-spg --spg-top-k 20 --input-csv Examples/PXRD_Ce3Si8Ni2_65.csv
```

Available `--spg-top-k` values: `3`, `5` (default), `10`, `20`.

You can additionally constrain inferred candidates by crystal system:

```bash
# Auto: infer crystal system from filename SPG and keep only matching inferred SGs
python single_agent.py --infer-spg --lattice-symmetry auto --input-csv Examples/PXRD_Ce3Si8Ni2_65.csv

# Explicit: only test cubic inferred SG candidates
python single_agent.py --infer-spg --lattice-symmetry cubic --spg-top-k 20 --input-csv Examples/PXRD_Ba4NaBi_216.csv
```

`--lattice-symmetry` choices: `auto`, `any`, `triclinic`, `monoclinic`, `orthorhombic`, `tetragonal`, `trigonal`, `hexagonal`, `cubic`.

When `--infer-spg` is active, the pipeline:
1. Detects peaks with the CNN peak finder.
2. Reconstructs a Gaussian profile from those peaks.
3. Feeds the profile + formula into the space group classifier to rank candidates.
4. Solves cells once from the first inferred SG that yields valid cells, then reuses those cells across top-k SG candidates for Wyckoff/structure search.
5. Iterates over inferred SG candidates, stopping as soon as an accepted solution is found.

This infer-SPG behavior is consistent in deterministic fallback mode and graph-consistent deterministic mode.

#### Improve stability for stochastic search (recommended)

The Wyckoff/structure stage is stochastic. `single_agent.py` now supports adaptive multi-attempt search with deterministic seeds:

- `PXRD_MULTI_ATTEMPTS` (default `3`, minimum `1`): number of independent attempts.
- `PXRD_SEED_BASE` (default `20260315`): base seed used to derive per-attempt seeds.

Examples:

```bash
# More robust, slower
PXRD_MULTI_ATTEMPTS=5 python single_agent.py

# Fully reproducible run with a fixed seed base
PXRD_MULTI_ATTEMPTS=4 PXRD_SEED_BASE=123456 python single_agent.py
```

Recommended settings (speed vs stability):

| `PXRD_MULTI_ATTEMPTS` | Runtime Cost | Result Stability | Recommended Use |
|---|---|---|---|
| `1` | Lowest | Lowest | Fast debugging and smoke tests |
| `3` (default) | Moderate | Good | Daily development runs |
| `5` | High | Better | Important runs where robustness matters |
| `7+` | Very high | Best-effort | Hard cases and final confirmation |

### Shared State (`share_state`)

| Key | Type | Populated by | Description |
|-----|------|-------------|-------------|
| `pxrd_csv` | `str` | User | Path to the input CSV file |
| `formula` | `str` | DataPreprocessAgent | Chemical formula parsed from filename |
| `composition` | `dict` | DataPreprocessAgent | Element → count mapping |
| `x1`, `y1` | `list[float]` | DataPreprocessAgent | 2θ array and intensity array |
| `peaks` | `list[int]` | DataPreprocessAgent | Peak indices in the 2θ grid |
| `peak_positions` | `list[float]` | DataPreprocessAgent | 2θ values of detected peaks |
| `spg` | `int` | DataPreprocessAgent | Space group number (from filename) |
| `density_min/max` | `float` | DataPreprocessAgent | ML-predicted density bounds (g cm⁻³) |
| `min_volume` | `float` | DataPreprocessAgent | Minimum unit cell volume (Å³) |
| `cells` | `list[CellManager]` | CellManagerAgent | Ranked candidate unit cells |
| `wavelength` | `float` | Default | Cu-Kα₁ wavelength, 1.54184 Å |
| `min_r2` | `float` | Default | Minimum acceptable R² (0.95) |
| `max_chi2` | `float` | Default | Maximum acceptable χ² (0.12) |
| `max_cells` | `int` | Default | Maximum candidate cells to carry forward (10) |

---

## Stage 1 — Data Preprocessing (`DataPreprocessAgent`)

**Class / tool:** `DataPreprocessor` (in `single_agent.py`)  
**Core logic:** `tools/manager.py` → `RawDataManager`  
**ML component:** `tools/peak_prediction.py` → `_predict_peaks`  
**ML component:** `tools/density.py` → `DensityEnsemblePredictor` / `predict_density_ensemble`

### 1.1 Filename Parsing

The chemical formula and space group number are encoded in the input filename using the convention `PXRD_<formula>_<spg>.csv`. These are extracted with string splitting:

```
PXRD_PrYMg2_123.csv  →  formula = "PrYMg2",  spg = 123
```

The formula is further parsed into a composition dictionary (e.g., `{'Pr': 1, 'Y': 1, 'Mg': 2}`) by `parse_formula()` in `tools/utils.py`.

### 1.2 Raw Data Loading and Smoothing

The CSV file contains two columns: 2θ (degrees) and intensity (arbitrary units). The data are loaded with `pandas` and passed to `RawDataManager`, which optionally:

- Performs **background subtraction** via asymmetric least-squares polynomial fitting (`polyfit` order 6, 50 iterations). Points above the fitted baseline are down-weighted with asymmetry parameter `asym = 0.01`.
- Applies **Savitzky-Golay smoothing** (window 4, polynomial order 3) to reduce noise while preserving peak shapes.

### 1.3 Initial Peak Detection (SciPy)

`get_peaks_from_scipy()` calls `scipy.signal.find_peaks` with conservative thresholds (`height=1.0`, `distance=5`, `prominence=1.5`) to capture even weak peaks that would otherwise be missed. This intentional over-detection is followed by ML-based filtering.

### 1.4 ML-Based Peak Filtering

`filter_peaks_by_ml()` normalises the intensity profile to [0, 1] and passes it to a pretrained CNN/transformer peak-detection model (`tools/peak_finder/`) via `_predict_peaks`. Each data point receives a peak probability score. Candidate peaks from Step 1.3 are **discarded** only when both conditions hold:

- Model probability < `threshold` (default 0.8)
- Raw intensity < `min_height` (default 3.0 normalised units)

This conjunction prevents the removal of low-probability but clearly intense peaks.

### 1.5 Density Range Prediction

`predict_density_ensemble()` in `tools/density.py` loads a **Roost ensemble** (multiple checkpoint files matching `models/density/checkpoint-r*.pth.tar`). Roost is a message-passing neural network that ingests element embeddings and stoichiometry. For the given formula, all models perform inference and their predictions are aggregated. The output is a `(mean ± sigma·std)` interval, defining `density_min` and `density_max` (default `sigma = 2.5`).

### 1.6 Minimum Volume Constraint

The minimum unit cell volume is derived from the maximum density bound:

$$V_\text{min} = \frac{M_\text{formula}}{d_\text{max} \cdot N_A} \times 10^{24} \quad (\text{Å}^3)$$

where $M_\text{formula}$ is the formula-unit molecular weight in g mol⁻¹ and $N_A$ is Avogadro's number. This hard lower bound prevents physically unreasonable cells from entering the indexing stage.

### 1.7 Output

All extracted values are written back to `invocation_state` and returned as a structured dictionary for the LLM to reason about. The agent reports any data-quality issues before passing control downstream.

---

## Stage 2 — Unit Cell Indexing (`CellManagerAgent`)

**Class / tool:** `CellSolverTool` (in `single_agent.py`)  
**Core logic:** `tools/solver.py` → `CellSolver`, `tools/manager.py` → `CellManager`

### 2.1 Bravais Lattice

The Bravais lattice type is looked up from the space group number using `pyxtal.symmetry.get_bravais_lattice`. This determines how many independent cell parameters must be solved for (1 for cubic, 2 for hexagonal/tetragonal, 3 for orthorhombic, 4 for monoclinic, 6 for triclinic).

### 2.2 hkl Enumeration

All **(h k l)** triples up to `hkl_max = (2, 5, 6)` that are **systematically allowed** by the space group are generated. The solver then considers combinations of these up to the number needed for a linearly determined system (e.g., 2 peaks for tetragonal, 3 for orthorhombic, etc.).

### 2.3 Cell Parameter Solving

For each hkl combination, the Bragg equation

$$d_{hkl} = \frac{\lambda}{2 \sin\theta}$$

is combined with the lattice-metric formula to form a linear system. For example, for **tetragonal** symmetry:

$$\frac{1}{d^2} = \frac{h^2 + k^2}{a^2} + \frac{l^2}{c^2}$$

Each system is solved analytically (via `numpy.linalg.solve`) and the result is filtered against:

- $a, b, c > $ `min_abc` = 2.0 Å
- $a, b, c < $ `max_abc`
- $V > V_\text{min}$ from Stage 1

### 2.4 Mismatch Scoring and Tolerances

The solver re-indexes all detected peaks against the trial cell using a series of angular tolerances `theta_tols = [0.1°, 0.15°, 0.5°]`. The **mismatch score** is the number of experimentally observed peaks that cannot be assigned to any allowed reflection within the tightest tolerance (`max_mismatch = 12`). A `chi2` score quantifies the mean squared residual between observed and predicted peak positions.

### 2.5 Cell Consolidation

`CellManager.consolidate()` merges cells that are crystallographically equivalent (within a fractional tolerance of 5 % on each parameter) and retains the top `max_cells = 10` solutions ranked primarily by the number of missing peaks (fewer is better) and secondarily by chi2.

### 2.6 Output

The tool returns a list of `CellManager` objects, each carrying:

- `dims`: lattice parameters $(a, b, c, \alpha, \beta, \gamma)$
- `missing`: number of unindexed peaks

These are stored in `invocation_state["cells"]` for Stage 3.

---

## Stage 3 — Crystal Structure Solution (`WyckoffSolverAgent`)

**Class / tool:** `WyckoffSolverTool` (in `single_agent.py`)  
**Core logic:** `tools/solver.py` → `search_solution`  
**Supporting logic:** `tools/manager.py` → `WPManager`, `XtalManager`  
**Force field:** ASE + MACE (via `tools/ase_opt.py`)  
**Refinement:** GSAS-II (via `tools/gsas.py`)

This is the computationally intensive stage. For each candidate cell, the solver systematically explores the space of atomic arrangements consistent with the crystal symmetry.

### 3.1 Wyckoff Position Enumeration (`WPManager`)

For the given space group and chemical composition, `WPManager` enumerates all **Wyckoff position (WP) assignments** that:

- Place each element on a set of Wyckoff sites whose multiplicities sum to the element count in the formula.
- Produce unit cell contents consistent with the density bounds `(density_min, density_max)`.

`WPManager` first enumerates all valid assignments, then `search_solution()` reranks them with `score_wp_candidate()` to prioritize cheaper/more plausible candidates (lower DOF, higher combinatorial support, fewer/less fragmented sites).

Evaluation then uses adaptive expansion (`top-3 → top-5 → top-10 → up to N2`) rather than spending full effort on every candidate from the start.

### 3.2 Trial Structure Generation (`XtalManager`)

For each WP assignment, `XtalManager` generates random atomic coordinates consistent with the site symmetry using **PyXtal**. The number of random trials per WP assignment is `3 × DOF + 1`.

`search_solution()` also applies a **supercell-aware per-cell budget**: if a candidate cell volume is a near-integer multiple of the smallest tested cell (likely supercell), that cell is still evaluated but with reduced `N2/N3` effort.

### 3.3 Geometry Optimisation (ASE + MACE)

Each trial structure is relaxed using `relax_structure()` in `tools/utils.py`, which calls the MACE universal neural-network force field via ASE:

1. A first relaxation with larger steps (`10 × DOF` steps) loosens the initial geometry.
2. The structure is discarded if the mean diagonal stress exceeds 5 GPa after step 1.
3. A second finer relaxation (`5 × DOF` steps, `fmax = 0.1` eV Å⁻¹) converges atomic positions.

The potential energy per atom after relaxation is tracked. Only structures satisfying `max_force ≤ 0.5` eV Å⁻¹ and `max_stress ≤ 0.3` GPa proceed to XRD comparison.

### 3.4 XRD Pattern Similarity Screening

For each relaxed structure, a theoretical PXRD pattern is computed with `tools/XRD.py` using the experimental wavelength, 2θ range `[10°, 80°]`, and step size 0.02°. The similarity to the experimental pattern is evaluated with a cosine-like metric (`Similarity` class).

Candidates are sent to refinement when either:

- `sim ≥ sim_max - refine_margin` (default `0.90 - 0.02 = 0.88`), or
- `sim ≥ refine_sim_min` **and** `eng_rel ≤ refine_eng_window`, where
  - `eng_rel = eng - eng_best` and `eng_best` is tracked from valid (stress/force-passing) structures only.

When similarity is promising but energy-window filtering rejects refinement, the solver logs an explicit skip reason with `eng_rel`.

### 3.5 Rietveld Refinement (GSAS-II)

Structures that satisfy the composite trigger above are submitted to full-pattern Rietveld refinement via `tools/gsas.py`, which wraps GSAS-II:

- Refines lattice parameters, atomic positions, thermal parameters, and profile parameters.
- Computes the standard crystallographic fit metrics:
  - **Rwp** (weighted profile R-factor)
  - **R²** (coefficient of determination)
  - **χ²** (goodness of fit, normalised by degrees of freedom)

A solution is **accepted** when:

$$R^2 \geq 0.95 \quad \text{or} \quad \chi^2 \leq 0.12$$

The refined structure is saved as a CIF file and a comparison plot (observed vs. simulated pattern) is written to `Results/`.

### 3.6 Search Strategy Summary

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `N1` | 5 | Top cells tested |
| `N2` | 20 | Max WP combinations per cell before adaptive/supercell reduction |
| `N3` | 9 | Max DOF per WP combination before adaptive/supercell reduction |
| `max cells` | 10 | Cells retained after indexing |
| Refinement trigger 1 | `sim ≥ 0.88` | Near-threshold similarity gate (`sim_max - refine_margin`) |
| Refinement trigger 2 | `sim ≥ 0.70` and `eng_rel ≤ 0.5` | Composite similarity+relative-energy gate |
| Acceptance: R² | ≥ 0.95 | Rietveld quality gate |
| Acceptance: χ² | ≤ 0.12 | Rietveld quality gate |

The solver exits **immediately** upon finding any structure meeting the acceptance criteria. Because WP exploration now uses reranking + adaptive expansion + supercell-aware budget reduction, practical search cost is usually lower than the fixed-budget worst-case bound.

---

## Machine Learning Models

| Model | Location | Framework | Task |
|-------|----------|-----------|------|
| Peak detector | `tools/models/peaks/best_model.pth` | Custom CNN/transformer | Assign peak probability to each 2θ point |
| Space group predictor | `tools/models/spacegroup/best_model.pth` | `ImprovedXRDNetWithFormula` (203 classes) | Predict ranked space groups from reconstructed PXRD profile + formula |
| Density ensemble | `tools/models/density/checkpoint-r*.pth.tar` | Roost (PyTorch) | Predict density mean and uncertainty from composition |

---

## Model Evaluation

### Space Group Prediction Quality

`evaluate_spacegroup_prediction.py` measures how well the pretrained space group classifier performs across a set of labelled PXRD files. It uses the same reconstructed-profile pipeline as the main agent.

#### Basic usage

```bash
# Evaluate all CSV files in the Examples/ directory
python evaluate_spacegroup_prediction.py --input Examples/ --top-k 5

# Evaluate a specific pattern
python evaluate_spacegroup_prediction.py --input Examples/PXRD_Ba4NaBi_216.csv

# Report top-10 predictions; save per-file results to CSV
python evaluate_spacegroup_prediction.py --input Examples/ --top-k 10 --output-csv results_spg.csv

# Limit to the first 5 files (quick smoke test)
python evaluate_spacegroup_prediction.py --input Examples/ --max-files 5
```

#### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `Examples/` | Directory of CSV files or path to a single CSV |
| `--pattern` | `PXRD_*.csv` | Glob pattern when `--input` is a directory |
| `--top-k` | `5` | Number of ranked predictions to display and score (choices: `3`, `5`, `10`) |
| `--no-normalization` | off | Pass raw intensity directly to the classifier (skips reconstructed-profile step) |
| `--max-files` | ∞ | Stop after this many files |
| `--output-csv` | — | Write per-file predictions to a CSV file |

#### Output

For each file the script prints the true space group, whether it appeared in top-1/3/5, and the ranked predictions with probabilities:

```
Examples/PXRD_Ba4NaBi_216.csv | true_spg=216 | top1=✓ | preds=[216:42.1%, 225:18.3%, 221:9.7%, ...]
```

Summary metrics printed at the end:

```
=== Summary (N=20) ===
Top-1 Accuracy : 0.650
Top-3 Accuracy : 0.800
Top-5 Accuracy : 0.900
MRR            : 0.742
```

- **Top-k Accuracy**: fraction of files where the true space group appears in the top-k predictions.
- **MRR** (Mean Reciprocal Rank): average of $1/\text{rank}$ over all files (higher is better).

---

## Input / Output

### Input

- A CSV file with two columns: `2theta` (degrees) and `intensity` (a.u.)
- Filename convention: `PXRD_<formula>_<spg>.csv`  
  Example: `Examples/PXRD_PrYMg2_123.csv`

### Output (written to `Results/`)

| File | Description |
|------|-------------|
| `Match_<formula>_<spg>.cif` | Best refined crystal structure in CIF format |
| `Match_<formula>_<spg>.png` | Observed vs. simulated PXRD pattern comparison |
| `single_agent.log` | Detailed run log with per-peak, per-cell, and per-structure diagnostics |

Notes:

- `Results/` is auto-created if missing.
- `tmp/` (for GSAS project/log intermediates) is auto-created if missing.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `strands` | Agentic framework (agent, tool, graph builder) |
| `google-generativeai` | Gemini 2.5 Pro LLM backend |
| `pyxtal` | Space group symmetry, Wyckoff positions, structure generation |
| `ase` | Atomic simulation environment for geometry relaxation |
| `mace-torch` | Universal neural-network force field (MACE) |
| `torch` | PyTorch — ML model inference |
| `scipy` | Peak detection, signal smoothing |
| `pandas` / `numpy` | Data I/O and numerical operations |
| GSAS-II | Full-pattern Rietveld refinement |

---

## Logging

All output from the pipeline — including print statements from library code — is intercepted by `StreamToLogger` and routed through the Python `logging` module. Both `single_agent.log` and the console receive identical `INFO`-level output. The Strands multiagent logger is silenced at `ERROR` level to reduce noise.

---

## Extending the Pipeline

- **Space group inference from PXRD:** Already implemented via `--infer-spg`. The pretrained classifier (`models/spacegroup/best_model.pth`) ranks up to `--spg-top-k` candidates; the pipeline iterates through them and stops at the first accepted solution.
- **Multi-space-group search:** The graph can be extended with a fan-out node that spawns parallel `CellManagerAgent` + `WyckoffSolverAgent` sub-graphs for each candidate space group.
- **Alternative force fields:** `relax_structure()` in `tools/utils.py` and `tools/ase_opt.py` can be swapped to use any ASE-compatible calculator.
- **Batch processing:** The entry point loop can be extended to iterate over all CSVs in the `Examples/` directory, updating `share_state["pxrd_csv"]` before each run.

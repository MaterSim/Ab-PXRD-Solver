# Ab-PXRD-Solver: Ab Initio Powder X-Ray Diffraction Structure Solver

## Overview

Ab-PXRD-Solver is a fully automated *ab initio* crystal structure determination pipeline. Given an experimental Powder X-Ray Diffraction (PXRD) pattern and a chemical formula, it autonomously:

1. **Preprocesses** the diffraction pattern with adaptive background subtraction, Savitzky-Golay smoothing, and ML-guided peak detection.
2. **Predicts density** bounds from the chemical formula using a pretrained Roost ensemble model.
3. **Indexes peaks** to candidate unit cells via `CellSolver` (known SPG) or `SmartCellSolver` (unknown SPG).
4. **Enumerates Wyckoff positions** compatible with the composition and density range.
5. **Generates trial structures** using `PyXtal` with Quasi-Random Sampling (Sobol or Halton).
6. **Relaxes structures** with the `MACE` universal neural-network force field via `ASE`.
7. **Screens candidates** by cosine similarity of simulated vs. experimental PXRD patterns.
8. **Refines** promising structures with full-pattern Rietveld refinement via `GSAS-II`.


## Pipeline Flowchart

```
Input: PXRD CSV + formula
          │
          ▼
┌─────────────────────────┐
│   Data Preprocessing    │  adaptive background subtraction → smoothing
│   (RawDataManager)      │  SciPy peaks → ML peak filter
│                         │  Roost density ensemble → density bounds
└────────────┬────────────┘
             │  peaks, density_min/max, formula, composition
             ▼
    ┌────────────────────┐
    │  Space Group Mode? │
    └──┬─────────────────┘
       │
       ├── Known SPG (filename or --spg)
       │        │
       │        ▼
       │   ┌──────────────────────────────────────┐
       │   │  CellSolver                          │
       │   │  hkl enumeration → linear solve →    │
       │   │  mismatch scoring → consolidation    │
       │   └────────────────┬─────────────────────┘
       │                    │
       └── --infer-spg ─────┤
           (model backend)  │
                │           │
                ▼           │
           CNN classifier   │
           top-k SPG list   │
                │           │
                ▼           │
       SmartCellSolver ─────┘
       (smart-cell backend: jointly ranks SPG + cell)
                │
                ▼
┌──────────────────────────────────────────────────────────┐
│  For each (cell, SPG) pair, ordered by estimated cost:   │
│                                                          │
│   WPManager: enumerate valid Wyckoff assignments         │
│   XtalManager: generate trial structures (PyXtal)        │
│       ↓  Quasi-Random Sampling (QRS)                     │
│   MACE force field: geometry relaxation (ASE)            │
│   XRD.py: simulate pattern → Autocorrelation             │
│       ↓  if sim ≥ threshold or (sim + energy gate)       │
│   GSAS-II: Rietveld refinement                           │
│       ↓  if R² ≥ 0.95 or χ² ≤ 0.12                       │
│   ✓ ACCEPTED — save CIF + plot, exit immediately         │
└──────────────────────────────────────────────────────────┘
          │
          ▼
Output: Results/cifs/Match_<formula>_<spg>.cif
        Results/logs/<run>.log
        Results/summary.csv
```

---

## Module Structure

```
PXRD-Agent/
├── PXRD_solve.py              # Main entry point (deterministic pipeline)
├── pxrd_app/
│   ├── cli.py                 # Argument parsing, batch dispatch, parallel workers
│   ├── constants.py           # DEFAULT_STATE — all tunable hyperparameters
│   ├── core.py                # Pipeline stages: run_data_preprocessor,
│   │                          #   run_cell_solver, run_wyckoff_solver
│   ├── inference.py           # SPG inference backends, SmartCellSolver ranking
│   ├── runtime.py             # Results CSV writing, timing summary
│   └── tools/
│       ├── manager.py         # RawDataManager, CellManager, WPManager, XtalManager
│       │                      #   generate_qrs_grid (Sobol / Halton)
│       ├── solver.py          # CellSolver, SmartCellSolver, search_solution,
│       │                      #   enumerate_wyckoff, get_adaptive_wp_limits
│       ├── density.py         # Roost ensemble density predictor
│       ├── peak_prediction.py # CNN peak detector + space group classifier
│       ├── XRD.py             # Pattern simulation, Similarity (cosine) metric
│       ├── gsas.py            # GSAS-II Rietveld refinement wrapper
│       ├── ase_opt.py         # MACE + ASE structure relaxation
│       └── utils.py           # parse_formula, relax_structure, volume helpers
├── Examples/                  # Sample PXRD CSV files
├── GSAS_PXRD/                 # Larger benchmark dataset
├── data/                      # CIF reference files, run lists
└── environment.yml            # Conda environment spec
```

---

## Stage 1 — Data Preprocessing

**Code:** `pxrd_app/core.py → run_data_preprocessor`  
**Key class:** `pxrd_app/tools/manager.py → RawDataManager`

### 1.1 Filename Parsing

The chemical formula and space group are parsed from the filename convention `PXRD_<formula>_<spg>.csv`:

```
PXRD_PrYMg2_123.csv  →  formula = "PrYMg2",  spg = 123
```

Use `--formula` to override, or `--infer-spg` to ignore the filename SPG entirely. Hyphen-separated names are also supported.

### 1.2 Adaptive Background Subtraction

If `min(intensity) > 2.5`, background subtraction is applied via **asymmetric least-squares polynomial fitting** (order 6, 50 iterations, asymmetry `asym = 0.01`). The corrected pattern is saved as `<name>_bg_subtracted.csv` for downstream use. A **Savitzky-Golay filter** (window 4, polynomial order 3) then smooths noise while preserving peak shapes.

### 1.3 Peak Detection

`RawDataManager.get_peaks_from_scipy()` calls `scipy.signal.find_peaks` with conservative thresholds to over-detect peaks. `filter_peaks_by_ml()` then filters with a pretrained CNN model — a peak is **removed** only when **both** of these hold:

- Model peak probability < 0.8
- Intensity < `min_height` (3.0–7.5, depending on background mode)

### 1.4 Density Prediction

`predict_density_ensemble()` runs a **Roost** message-passing neural network ensemble on the composition. Predictions are aggregated as `mean ± 2.5·std`, yielding `density_min` and `density_max` (g cm⁻³). The minimum cell volume bound is:

$$V_{\text{min}} = \frac{M_{\text{formula}}}{d_{\text{max}} \cdot N_A} \times 10^{24} \;\; (\text{Å}^3)$$

---

## Stage 2 — Space Group Inference and Cell Indexing

### 2.1 Space Group Modes

| Mode | Flag | How SPG is obtained |
|------|------|---------------------|
| **Filename** | *(default)* | Parsed from `_<spg>.csv` suffix |
| **Override** | `--spg N` | Fixed to space group N |
| **SmartCellSolver** | `--infer-spg --spg-backend smart-cell` | Jointly enumerates SPGs and cells, ranked by indexing quality |


### 2.2 CellSolver (known SPG)

`CellSolver` in `tools/solver.py`:

1. **hkl enumeration** — all symmetry-allowed (h k l) triples up to `hkl_max = (2, 5, 6)`.
2. **Linear solve** — Bragg equation + lattice metric form a linear system solved by `numpy.linalg.solve`. For tetragonal:

$$\frac{1}{d^2} = \frac{h^2+k^2}{a^2} + \frac{l^2}{c^2}$$

3. **Mismatch scoring** — peaks re-indexed against trial cell at tolerances `[0.1°, 0.15°, 0.5°]`.

For orthorhombic SPGs, all six axis permutations are tried to handle axis-setting ambiguity.

### 2.3 SmartCellSolver (unknown SPG)

`SmartCellSolver` sweeps through space groups from **highest to lowest symmetry** (cubic → triclinic), simultaneously solving for unit cells under each SPG. Solutions are ranked jointly by mismatch, χ², and volume. This is the recommended mode when the SPG is unknown. The solver stops early once an ideal-mismatch solution is found for a high-symmetry system.

### 2.4 Cell Consolidation

`CellManager.consolidate()` merges equivalent cells (within 5% on each parameter) and retains the top `max_cells = 10` solutions ranked by (missing peaks, χ²).

---

## Stage 3 — Crystal Structure Solution

**Code:** `pxrd_app/core.py → run_wyckoff_solver`  
**Key function:** `pxrd_app/tools/solver.py → search_solution`

### 3.1 Wyckoff Position Enumeration

`WPManager` lists all valid Wyckoff position assignments where each element's site multiplicities sum to its count in the formula and the resulting density is within `(density_min, density_max)`. Assignments are reranked by `score_wp_candidate()` to prefer smaller-volume, lower-mismatch, lower-DOF, fewer-site configurations.

### 3.2 Trial Structure Generation and QRS

`XtalManager` uses **PyXtal** to place atoms on Wyckoff sites with **Quasi-Random Sampling:** Fractional coordinates are drawn from a **Sobol** or **Halton** low-discrepancy sequence (`generate_qrs_grid` in `tools/manager.py`) instead of pseudo-random numbers. This provides better uniform coverage of coordinate space with fewer trials.

### 3.3 Geometry Relaxation (MACE + ASE)

Each trial structure is relaxed with the **MACE** universal force field via ASE:

1. Coarse relaxation (`10 × DOF` steps).
2. Stress check — structures with diagonal stress > 5 GPa are discarded.
3. Fine relaxation (`5 × DOF` steps, `fmax = 0.1` eV Å⁻¹).

Only structures with `max_force ≤ 0.5` eV Å⁻¹ and `max_stress ≤ 0.3` GPa pass to screening. The global minimum energy per atom `eng_best` is tracked across all valid structures.

### 3.4 Pattern Similarity Screening

A theoretical PXRD pattern is simulated (2θ: 10°–80°, step 0.02°, Cu-Kα₁ λ = 1.54184 Å) and compared to the experiment using a cosine-like `Similarity` metric. A structure proceeds to refinement when **either** condition holds:

| Trigger | Condition |
|---------|-----------|
| Similarity gate | `sim ≥ sim_max − 0.02` (default: `sim ≥ 0.88`) |
| Energy + similarity gate | `sim ≥ 0.70` **and** `eng − eng_best ≤ max_eng_rel` |

`sim_max` (default 0.9) is automatically lowered for light-element compositions (Z ≤ 6) or sparse peak sets (≤ 4 peaks).

### 3.5 Rietveld Refinement (GSAS-II)

`pxrd_app/tools/gsas.py` wraps GSAS-II for full-pattern Rietveld refinement. A per-refinement wall-time limit of 60 s prevents hangs; GSAS-II is recycled every 30 calls to avoid memory leaks.

A solution is **accepted** when:

$$R^2 \geq 0.95 \quad \text{or} \quad \chi^2 \leq 0.12$$

The pipeline exits immediately on the first accepted solution.

### 3.6 Key Search Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `max_wp` | 18 | Max Wyckoff sites per assignment |
| `max_dof` | 25 | Max degrees of freedom per WP combination |
| `max_Z` | 24 | Max formula units per cell |
| `max_eng_rel` | 0.1 eV/atom | Energy window above best for refinement trigger 2 |
| `max_force` | 0.5 eV/Å | Max per-atom force after relaxation |
| `max_stress` | 0.3 GPa | Max diagonal stress after relaxation |
| `min_r2` | 0.95 | Rietveld acceptance: R² |
| `max_chi2` | 0.12 | Rietveld acceptance: χ² |
| `gsas_refine_timeout` | 60 s | Per-refinement wall-time limit |

---

## Machine Learning Models

| Model | Location | Task |
|-------|----------|------|
| Peak detector | `pxrd_app/tools/peak_finder/` | Assign peak probability to each 2θ point (CNN/transformer) |
| Space group predictor | `pxrd_app/tools/spacegroup/` | Rank space groups from PXRD profile + formula (`ImprovedXRDNetWithFormula`) |
| Density ensemble | `pxrd_app/tools/aviary/` | Predict density mean + uncertainty from composition (Roost, PyTorch) |

---

## Usage

### Quick Start

```bash
# Single file, SPG from filename
python PXRD_solve.py --input Examples/PXRD_PrYMg2_123.csv
```
This run will generate a list of trial solutions as follows.
```
Pair  WPs   SPG   Volume(Å³)  Chi2    EstTrials Missing  BalScore  Dims
--------------------------------------------------------------------------------------------------------
1     12    123   112.2       0.0004   42        7        0.034        3.818     7.701
2     2     123   448.9       0.0003   30        26       0.100        5.399    15.401
3     7     123   224.5       0.0004   91        22       0.119        5.399     7.701
4     2     123   448.9       0.0004   42        25       0.122        7.635     7.701
5     2     123   449.4       0.0030   18        11       0.127        3.818    30.832
6     7     123   224.5       0.0004   195       13       0.136        3.818    15.401
7     7     123   224.8       0.0167   39        17       0.319        7.645     3.846
8     2     123   449.9       0.0312   60        19       0.759       10.821     3.843
9     1     123   782.6       0.0353   156       28       2.041        7.652    13.367
```
The solver goes through the list and finds an excellent fit for the first cell which is the energy minimum with a high R2 value, and then stops the search.
<figure>
  <img src="Figs/EnergyR2_PrYMg2_123_single.png" width="600">
  <figcaption>Figure 1. Solution when SPG is known.</figcaption>
</figure>

```bash
# Single file, infer SPG with SmartCellSolver 
python PXRD_solve.py --input GSAS_PXRD/Ag2Hg5_127.csv --infer-spg

Pair  WPs   SPG   Volume(Å³)  Chi2    EstTrials Missing  BalScore  Dims
--------------------------------------------------------------------------------------------------------
1     3     115   112.2       0.0004   21        7        0.031        3.818     7.701
2     12    123   112.2       0.0004   42        7        0.045        3.818     7.701
3     2     99    112.2       0.0004   49        7        0.048        3.818     7.701
...
90    3     137   898.8       0.0046   234       29       1.586        7.635    15.416
91    2     113   898.8       0.0046   278       29       1.729        7.635    15.416
92    3     82    899.1       0.0167   284       11       1.857       10.811     7.692
```
When the space group is unknown, the solver will generate a larger list and then goes through the list. It takes a longer time to find the excellent fit  which is the energy minimum with a high R2 value.
<figure>
  <img src="Figs/EnergyR2_PrYMg2_123_auto.png" width="600">
  <figcaption>Figure 2. Solution when SPG is unknown.</figcaption>
</figure>
```bash
# Batch: file list (SLURM-style array job)
python PXRD_solve.py --use-list --input data/test.txt --infer-spg --workers 48 
```

### All CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input PATH` | `Examples/PXRD_PrYMg2_123.csv` | CSV file, directory of CSVs, or (with `--use-list`) a text file of paths |
| `--use-list` | off | Treat `--input` as a text file with one CSV path per line |
| `--output DIR` | `Results` | Output directory for CIFs, logs, plots, and summary CSV |
| `--formula STR` | *(from filename)* | Override formula instead of parsing from filename |
| `--spg N` | *(from filename)* | Fix or filter to a single space group (1–230) |
| `--infer-spg` | off | Infer space group from data instead of reading from filename |
| `--max-volume V` | 1500.0 | Maximum allowed unit-cell volume (Å³) |
| `--max-wp N` | 18 | Max Wyckoff sites per assignment |
| `--max-dof N` | 25 | Max degrees of freedom per WP combination |
| `--max-z N` | 24 | Max Z (formula units per cell) |
| `--max-sim S` | 0.9 | Similarity threshold for refinement trigger 1 |
| `--max-eng-rel E` | 0.1 | Energy-above-best (eV/atom) threshold for refinement trigger 2 |
| `--qrs-method {sobol,halton}` | `sobol` | QRS sampler type |
| `--workers N` | 1 | Parallel CSV workers (batch mode) |
| `--list-wp-only` | off | List Wyckoff candidates only, skip structure generation |
| `--ase-logfile PATH` | *(none)* | Write ASE FIRE optimizer logs to this file |
| `--wp-csv-path PATH` | `pxrd_app/tools/spg_comp_wp.csv` | Precomputed WP count table for cost estimation |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PXRD_SUPPRESS_TORCH_LOAD_FUTUREWARNING` | `1` | Suppress PyTorch FutureWarning on checkpoint load |

---

## Input / Output

### Input Format

A two-column CSV file:

```
2theta,intensity
10.02,12.3
10.04,14.1
...
```

**Filename convention:** `PXRD_<formula>_<spg>.csv` (e.g., `PXRD_PrYMg2_123.csv`). The SPG suffix is used when `--infer-spg` is not set. The formula is always parsed from the filename unless `--formula` overrides it.

### Output

All results are written to `--output` (default: `Results/`):

| Path | Description |
|------|-------------|
| `cifs/Match_<formula>_<spg>.cif` | Best refined crystal structure (CIF) |
| `logs/<name>.log` | Per-system run log with full diagnostics |
| `summary.csv` | One row per input file: runtime, R², χ², Rwp, SPG, Wyckoff, cell |

`tmp/` (GSAS-II intermediates) is created under the output directory and can be deleted after a run.

### Summary CSV Columns

`csv_file_name`, `Runtime`, `N_struc`, `N_attempts`, `N_est`, `Status`, `E`, `dE`, `R2`, `Chi2`, `Rwp`, `SPG`, `Wyckoff`, `Cell`, `WP_qrs_id`

---

## Shared Pipeline State

All stages communicate through a `run_state` dictionary (`pxrd_app/constants.py → DEFAULT_STATE`). Key fields:

| Key | Type | Description |
|-----|------|-------------|
| `pxrd_csv` | `str` | Path to input CSV |
| `formula` | `str` | Chemical formula |
| `composition` | `dict` | Element → count |
| `x1`, `y1` | `list[float]` | 2θ and intensity arrays |
| `peaks` | `list[int]` | Peak indices in 2θ grid |
| `peak_positions` | `list[float]` | 2θ values of detected peaks |
| `spg` | `int` | Space group number |
| `spg_predictions` | `list[int]` | Ranked SPG candidates (infer mode) |
| `density_min/max` | `float` | ML-predicted density bounds (g cm⁻³) |
| `min_volume` | `float` | Minimum unit-cell volume (Å³) |
| `cells` | `list[CellManager]` | Ranked candidate unit cells |
| `wavelength` | `float` | X-ray wavelength (default: 1.54184 Å) |
| `qrs_method` | `str` | `"sobol"` or `"halton"` |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `python >= 3.11` | Language runtime |
| `pyxtal` | Wyckoff positions, space group symmetry, structure generation |
| `ase` | Geometry relaxation environment |
| `mace-torch` | Universal MACE neural-network force field |
| `torch` | PyTorch — ML model inference |
| `scipy` | Peak detection, smoothing, QMC samplers (Sobol/Halton) |
| `pandas` / `numpy` | Data I/O and numerics |
| `pymatgen` | Structure handling, CIF I/O |
| `spglib` | Space group detection |
| `gsas2pkg` | Full-pattern Rietveld refinement (GSAS-II) |

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate pxrd-agent
```

For GPU-accelerated MACE, install a CUDA-enabled PyTorch:

```bash
conda install -c pytorch -c nvidia pytorch-cuda=12.1
```

---

## Logging

All pipeline output is routed through the Python `logging` module (`pxrd_agent` logger). A per-system log file is written to `Results/logs/` alongside `PXRD_solver.log` in the working directory. Print statements from libraries are intercepted by `StreamToLogger` and emitted at `INFO` level.

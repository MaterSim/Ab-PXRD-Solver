# Ab-PXRD-Solver: Ab Initio Powder X-Ray Diffraction Structure Solver

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Module Structure](#module-structure)
- [Pipeline Stages](#pipeline-stages)
- [Machine Learning Models](#machine-learning-models)
- [Usage](#usage)
- [Input / Output](#input--output)
- [Dependencies & Setup](#dependencies--setup)
- [Citation and Online Database](#citation-and-online-database)

---

## Overview

Ab-PXRD-Solver is a fully automated *ab initio* crystal structure determination pipeline. Given an experimental Powder X-Ray Diffraction (PXRD) pattern and a chemical formula, it autonomously:

1. **Preprocesses** the pattern (background subtraction, smoothing, ML peak detection).
2. **Predicts density** bounds via a Roost ensemble model.
3. **Indexes peaks** to candidate unit cells (`CellSolver` or `SmartCellSolver`).
4. **Enumerates Wyckoff positions** matching composition and density.
5. **Generates trial structures** with PyXtal and Quasi-Random Sampling.
6. **Relaxes structures** with the MACE force field (ASE).
7. **Screens candidates** by simulated vs. experimental PXRD similarity.
8. **Refines** promising structures with GSAS-II Rietveld refinement.

---

## Pipeline

```
Input: PXRD CSV + formula
          │
          ▼
┌─────────────────────────┐
│   Data Preprocessing    │  background → smoothing → peaks → density bounds
└────────────┬────────────┘
             ▼
    Known SPG? ──yes──► CellSolver
         │
         no (--infer-spg)
         ▼
    SmartCellSolver (ranks SPG + cell jointly)
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│  For each (cell, SPG) pair, ordered by estimated cost:   │
│   Wyckoff enumeration → PyXtal trials (QRS)              │
│   → MACE relaxation → pattern similarity screening       │
│   → GSAS-II refinement (accept if R² ≥ 0.95, χ² ≤ 0.12) │
└──────────────────────────────────────────────────────────┘
          │
          ▼
Output: Results/cifs/, Results/logs/, Results/summary.csv
```

---

## Module Structure

```
Ab-PXRD-Solver/
├── PXRD_solve.py          # Main entry point
├── pxrd_app/
│   ├── cli.py             # Argument parsing, batch dispatch
│   ├── core.py            # Pipeline stages
│   ├── inference.py       # SPG inference backends
│   └── tools/
│       ├── manager.py     # RawDataManager, CellManager, WPManager, XtalManager
│       ├── solver.py      # CellSolver, SmartCellSolver, search_solution
│       ├── density.py     # Roost density predictor
│       ├── peak_prediction.py, XRD.py, gsas.py, ase_opt.py
├── Examples/              # Sample PXRD CSV files
└── environment.yml
```

---

## Pipeline Stages

### Stage 1 — Data Preprocessing

`RawDataManager` (`pxrd_app/tools/manager.py`) parses `PXRD_<formula>_<spg>.csv`, subtracts background (asymmetric least-squares + Savitzky-Golay smoothing), detects peaks (SciPy + CNN filter), and predicts density bounds (Roost ensemble, mean ± 2.5σ).

### Stage 2 — Cell Indexing

| Mode | Flag | SPG source |
|------|------|------------|
| Filename | *(default)* | `_<spg>.csv` suffix |
| Override | `--spg N` | Fixed space group |
| Infer | `--infer-spg` | `SmartCellSolver` ranks SPG + cell jointly |

`CellSolver` enumerates hkl triples, solves the Bragg linear system, and scores mismatch. `SmartCellSolver` sweeps SPGs from high to low symmetry when SPG is unknown. Top 10 consolidated cells are retained.

### Stage 3 — Structure Solution

`search_solution` enumerates Wyckoff assignments, generates trial structures (PyXtal + Sobol/Halton QRS), relaxes with MACE/ASE, screens by pattern similarity, and refines with GSAS-II. The pipeline exits on the first solution with R² ≥ 0.95 and χ² ≤ 0.12.

Key defaults: `max_wp=18`, `max_dof=25`, `max_Z=24`, similarity gate `≥0.88`, energy window `0.1 eV/atom`.

---

## Machine Learning Models

| Model | Location | Task |
|-------|----------|------|
| Peak detector | `pxrd_app/tools/peak_finder/` | Peak probability per 2θ point |
| Space group predictor | `pxrd_app/tools/spacegroup/` | Rank SPGs from profile + formula |
| Density ensemble | `pxrd_app/tools/aviary/` | Density mean + uncertainty (Roost) |

---

## Usage

### Quick Start

```bash
# SPG from filename
python PXRD_solve.py --input Examples/PXRD_PrYMg2_123.csv

# Infer SPG with SmartCellSolver
python PXRD_solve.py --input Examples/PXRD_PrYMg2_123.csv --infer-spg

# Batch run (SLURM-style)
python PXRD_solve.py --use-list --input data/test.txt --infer-spg --workers 48
```

With a known SPG, the solver ranks (cell, SPG) pairs and typically finds an accepted solution quickly. With `--infer-spg`, it explores more candidates across space groups before converging.

<figure>
  <img src="Figs/EnergyR2_PrYMg2_123_single.png" width="600">
  <figcaption>Figure 1. Solution when SPG is known.</figcaption>
</figure>

<figure>
  <img src="Figs/EnergyR2_PrYMg2_123_auto.png" width="600">
  <figcaption>Figure 2. Solution when SPG is unknown.</figcaption>
</figure>

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input PATH` | `Examples/PXRD_PrYMg2_123.csv` | CSV file, directory, or path list (with `--use-list`) |
| `--use-list` | off | Treat `--input` as a text file of CSV paths |
| `--output DIR` | `Results` | Output directory |
| `--formula STR` | *(from filename)* | Override parsed formula |
| `--spg N` | *(from filename)* | Fix space group (1–230) |
| `--infer-spg` | off | Infer space group from data |
| `--max-vol V` | 1500.0 | Max unit-cell volume (Å³) |
| `--max-wp N` | 18 | Max Wyckoff sites per assignment |
| `--max-dof N` | 25 | Max degrees of freedom per WP combination |
| `--max-z N` | 24 | Max formula units per cell |
| `--max-sim S` | 0.9 | Similarity threshold for refinement |
| `--max-eng E` | 0.1 | Energy-above-best threshold (eV/atom) |
| `--qrs` | `halton` | QRS sampler (`sobol` or `halton`) |
| `--workers N` | 1 | Parallel workers (batch mode) |
| `--list-wp-only` | off | List Wyckoff candidates only |

Run `python PXRD_solve.py --help` for the full argument list. Tunable hyperparameters live in `pxrd_app/constants.py` (`DEFAULT_STATE`).

---

## Input / Output

**Input:** two-column CSV (`2theta,intensity`). Filename convention `PXRD_<formula>_<spg>.csv`; override with `--formula` or ignore SPG with `--infer-spg`.

**Output** (under `--output`, default `Results/`):

| Path | Description |
|------|-------------|
| `cifs/Match_<formula>_<spg>.cif` | Best refined structure |
| `logs/<name>.log` | Per-run diagnostics |
| `summary.csv` | Runtime, R², χ², Rwp, SPG, Wyckoff, cell |

`tmp/` (GSAS-II intermediates) is created under the output directory and can be deleted after a run.

---

## Dependencies & Setup

**Key packages:** Python ≥ 3.11, PyXtal, ASE, mace-torch, PyTorch, SciPy, pandas/numpy, pymatgen, spglib, GSAS-II.

```bash
conda env create -f environment.yml
conda activate ab-pxrd-solver
```

If Conda's `libmamba` solver is broken: `conda config --set solver classic`.

Logs are written to `Results/logs/` and `PXRD_solver.log` in the working directory.

---

## Citation and Online Database

If you use Ab-PXRD-Solver in your research, please cite:

```bibtex
@misc{su2026abinitiocrystalstructuredetermination,
      title={Ab-initio Crystal Structure Determination from Powder X-Ray Diffraction},
      author={Kaixiang Su and Osman Goni Ridwan and Hongfei Xue and Qiang Zhu},
      year={2026},
      eprint={2605.24594},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2605.24594},
}
```

Systematic results on 1000+ systems are available at [https://mmi.charlotte.edu/ab_pxrd_solver](https://mmi.charlotte.edu/ab_pxrd_solver).

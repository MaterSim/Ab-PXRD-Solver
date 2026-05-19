import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pxrd_app.tools.gsas import refine_pxrd
from pyxtal import pyxtal

N = 76
SPG = (139, 'I4/mmm', "4e 4e 2b 8g 4e 2a")
Relaxed = np.array([
    [-6.359, 0],
    [-8.402, 0],
    [-7.774, 0],
    [-8.257, 0],
    [-6.103, 0],
    [-7.341, 0],
    [-7.219, 0],
    [-8.403, 0]])
Refined = np.array([
    [-9.054, 0.002, 0.072, 3.559, 0.999, 0.021],
    [-9.055, 0.002, 0.096, 3.559, 0.999, 0.021],
    [-6.718, 0.192, 0.076, 57.009, 0.662, 5.378],
    [-7.360, 0.181, 0.091, 63.302, 0.583, 6.630],
    [-9.054, 0.003, 0.083, 3.559, 0.999, 0.021],
    [-6.531, 0.192, 0.078, 10.450, 0.989, 0.181],
    [-6.917, 0.266, 0.066, 50.736, 0.732, 4.259],
    [-7.079, 0.169, 0.072, 7.166, 0.995, 0.085],
    [-7.742, 0.198, 0.058, 53.911, 0.698, 4.809],
    [-7.742, 0.198, 0.084, 54.013, 0.697, 4.827]])


def add_structure_inset(ax, cif_path: str, loc: str = "upper right") -> None:
    xtal = pyxtal()
    xtal.from_seed(cif_path)
    ase_atoms = xtal.to_ase()

    inset = inset_axes(ax, width="32%", height="32%", loc=loc, borderpad=1.0)
    plot_atoms(ase_atoms, ax=inset, radii=0.35, rotation=("20x,25y,0z"))
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_visible(False)


def main() -> None:
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax1.scatter(Relaxed[:, 0], Relaxed[:, 1], c="steelblue", s=50, alpha=0.5, label=f"Relaxed only (N={len(Relaxed)})")
    refined_scatter = ax1.scatter(Refined[:, 0], Refined[:, -2], c=Refined[:, -2], cmap="Reds", vmin=0, vmax=1, marker='*', s=100, label=f'Refined (N={len(Refined)})')
    colorbar = fig.colorbar(refined_scatter, ax=ax1, pad=0.02)
    colorbar.set_label("R² value")
    ax1.legend(loc=2, fontsize=12)
    ax1.set_xlabel("Energy (eV/atom)", fontsize=12)
    ax1.set_ylabel("R² value", fontsize=12)
    ax1.set_title("(a) Energy vs R²", fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.1, 1.2)
    ax1.set_yticks(np.linspace(0, 1, 6))

    inst_file = "pxrd_app/tools/INST_XRY.PRM"
    pxrd_csv = 'GSAS_PXRD/Al2Eu3O7_139.csv'
    good_match_cif = 'Examples/Match_Al2Eu3O7_139.cif'
    wr, r2, chi2, cif, elapsed = refine_pxrd(pxrd_csv, good_match_cif, inst_file, ax=ax2)
    ax2.set_title("(b) True match: -9.054 eV/atom", fontsize=14, fontweight='bold')
    ax2.text(0.1, 0.5, f"R²: {r2:.3f}\nχ²: {chi2:.3f}\n$R_{{wp}}$: {wr:.2f}", transform=ax2.transAxes,
             fontsize=12, verticalalignment='center', horizontalalignment='center')
    #add_structure_inset(ax2, good_match_cif)
    ax2.legend(loc=2, fontsize=12)
    ax2.set_xlabel("2θ (degrees)", fontsize=12)
    ax2.set_ylabel("Intensity (a.u.)", fontsize=12)

    bad_match_cif = 'data/Match_Al2Eu3O7_bad.cif'
    wr, r2, chi2, cif, elapsed = refine_pxrd(pxrd_csv, bad_match_cif, inst_file, ax=ax3)
    ax3.set_title("(c) False match: -7.079 eV/atom", fontsize=14, fontweight='bold')
    ax3.text(0.1, 0.5, f"R²: {r2:.3f}\nχ²: {chi2:.3f}\n$R_{{wp}}$: {wr:.2f}", transform=ax3.transAxes,
             fontsize=12, verticalalignment='center', horizontalalignment='center')
    #add_structure_inset(ax3, bad_match_cif)
    ax3.legend(loc=2, fontsize=12)
    ax3.set_xlabel("2θ (degrees)", fontsize=12)
    ax3.set_ylabel("Intensity (a.u.)", fontsize=12)
    plt.tight_layout()
    plt.savefig("Fig3.pdf", dpi=300)


if __name__ == "__main__":
    main()

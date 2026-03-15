import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import re
from pathlib import Path
from typing import Mapping, Sequence, Any
from .ase_opt import ASE_relax
from pymatgen.core.periodic_table import Element


def plot_XRD(x_obs: Sequence[float], y_obs: Sequence[float],
             x_sim: Sequence[float], y_sim: Sequence[float],
             x_peak: Sequence[float], y_peak: Sequence[float],
             title: str, filename: str) -> None:
    """
    Plot observed and simulated XRD patterns along with peak positions.
    """
    plt.figure(figsize=(8, 3))
    plt.title(title)
    plt.plot(x_obs, y_obs, color='green', alpha=0.5, label='Observed')
    plt.plot(x_sim, y_sim, color='blue', alpha=0.5, label='Simulated')
    plt.plot(x_peak, y_peak, "x", c='orange')

    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()


def parse_formula(formula: str) -> dict[str, int]:
    """
    Parse chemical formula string to dictionary of element counts.

    Args:
        formula: Chemical formula string (e.g., 'PrYMg2')

    Returns:
        Dictionary mapping elements to counts (e.g., {'Pr': 1, 'Y': 1, 'Mg': 2})
    """
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)

    result = {}
    for element, count in matches:
        if element:  # Skip empty matches
            count = int(count) if count else 1
            result[element] = result.get(element, 0) + count

    return result

def get_volume_from_density(composition: Mapping[str, int], density: float) -> float:
    """
    Calculate minimum volume from maximum density.

    Args:
        composition: Dictionary mapping elements to counts.
        density: Density in g/cm³.

    Returns:
        Minimum volume in Å³.
    """

    molecular_weight = 0.0
    for elem, count in composition.items():
        molecular_weight += Element(elem).atomic_mass * count

    N_A = 6.022e23  # Avogadro's number
    min_volume = molecular_weight / density / N_A * 1e24  # in Å³
    return min_volume

def relax_structure(atoms: Any, dof: int) -> Any | None:
    if dof > 0:
        atoms = ASE_relax(atoms, opt_lat=False, step=10 * dof, logfile='ase.log')
        final_step = 5 * dof
        final_fmax = 0.1
    else:
        atoms = ASE_relax(atoms, opt_lat=False, step=5, logfile='ase.log')
        final_step = None
        final_fmax = None

    if atoms is None or abs(atoms.get_stress()[:3].mean()) > 5.0:
        return None

    if final_step is not None:
        atoms = ASE_relax(atoms, opt_lat=False, step=final_step, fmax=final_fmax,
                          logfile='ase.log')
    return atoms

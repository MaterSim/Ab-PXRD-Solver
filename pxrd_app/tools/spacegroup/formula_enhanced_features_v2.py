import torch
import numpy as np
from pymatgen.core import Composition
from tqdm import tqdm
from math import gcd
from functools import reduce


def reduce_formula(formula_str):
    """
    Reduce chemical formula to simplest form

    Examples:
        Fe2O3 → Fe2O3 (already simplified)
        Fe4O6 → Fe2O3 (divide by 2)
        Fe6O9 → Fe2O3 (divide by 3)
        BaTiO3 → BaTiO3 (cannot simplify)

    Args:
        formula_str: Original formula string

    Returns:
        reduced_formula: Simplified formula string
    """
    try:
        comp = Composition(formula_str)

        # Get all elements and their coefficients
        element_amounts = {}
        for element, amount in comp.items():
            element_amounts[element] = amount

        # Extract all coefficients
        amounts = list(element_amounts.values())

        # If all coefficients are integers, calculate GCD
        try:
            int_amounts = []
            for amt in amounts:
                # Handle floating point precision: treat near-integers as integers
                if abs(amt - round(amt)) < 1e-6:
                    int_amounts.append(int(round(amt)))
                else:
                    # If non-integer coefficient exists, return pymatgen's reduced_formula
                    return comp.reduced_formula

            # Calculate GCD of all coefficients
            if len(int_amounts) > 0:
                common_gcd = reduce(gcd, int_amounts)

                # If GCD > 1, perform reduction
                if common_gcd > 1:
                    reduced_comp = {}
                    for element, amount in element_amounts.items():
                        reduced_comp[element] = int(round(amount)) // common_gcd

                    # Reassemble formula string
                    # Sort by element symbol for consistency
                    sorted_elements = sorted(reduced_comp.keys(), key=lambda e: e.symbol)
                    reduced_formula = ""
                    for element in sorted_elements:
                        count = reduced_comp[element]
                        if count == 1:
                            reduced_formula += element.symbol
                        else:
                            reduced_formula += f"{element.symbol}{count}"

                    return reduced_formula
                else:
                    # GCD = 1, already in simplest form
                    return comp.reduced_formula
            else:
                return comp.reduced_formula

        except Exception as e:
            # If reduction fails, return pymatgen's default reduced_formula
            print(f"Reduction failed '{formula_str}': {e}, using default reduced_formula")
            return comp.reduced_formula

    except Exception as e:
        print(f"Formula parsing failed '{formula_str}': {e}")
        return formula_str  # Return original string

def extract_advanced_chemical_context(formula_str):
    """
    Extract advanced chemical context features (30-dim) - Fixed NoneType error

    Includes:
      1. Periodic table position features (10-dim)
      2. Chemical bond type features (10-dim)
      3. Crystal structure tendency features (10-dim)

    Returns:
        context_features: (30,) numpy array
    """
    try:
        formula_str = reduce_formula(formula_str)
        comp = Composition(formula_str)
        features = []

        # ========== 1. Periodic table position statistics (10-dim) ==========
        periods = []  # Period
        groups = []   # Group
        for element in comp.elements:
            periods.append(element.row)
            groups.append(element.group)

        features.extend([
            np.mean(periods) / 7.0,                           # [0] Average period
            np.std(periods) / 3.0,                            # [1] Period std dev
            np.max(periods) / 7.0,                            # [2] Max period
            np.min(periods) / 7.0,                            # [3] Min period
            np.mean(groups) / 18.0,                           # [4] Average group
            np.std(groups) / 9.0,                             # [5] Group std dev
            len(set(periods)) / 7.0,                          # [6] Period diversity
            len(set(groups)) / 18.0,                          # [7] Group diversity
            (np.max(groups) - np.min(groups)) / 18.0,         # [8] Group span
            1.0 if any(e.is_actinoid or e.is_lanthanoid for e in comp.elements) else 0.0  # [9] Contains f-block elements
        ])

        # ========== 2. Chemical bond type tendency (10-dim) ==========
        # Pauling electronegativity difference → bond type
        electronegativities = [e.X for e in comp.elements if e.X is not None]
        if len(electronegativities) > 1:
            en_diff = max(electronegativities) - min(electronegativities)
            features.extend([
                en_diff / 4.0,                                # [10] Electronegativity diff (normalized)
                1.0 if en_diff > 1.7 else 0.0,                # [11] Ionic bond tendency
                1.0 if en_diff < 0.5 else 0.0,                # [12] Metallic bond tendency
                1.0 if 0.5 <= en_diff <= 1.7 else 0.0,        # [13] Covalent bond tendency
            ])
        else:
            features.extend([0.0] * 4)

        # Valence electron statistics
        valence_electrons = []
        for element in comp.elements:
            # Simplified valence electron calculation
            if element.group <= 12:
                ve = element.group
            else:
                ve = element.group - 10
            valence_electrons.append(ve)

        features.extend([
            np.mean(valence_electrons) / 8.0,                 # [14] Average valence electrons
            np.std(valence_electrons) / 4.0,                  # [15] Valence electron std dev
            sum([comp.get_atomic_fraction(e) * ve for e, ve in zip(comp.elements, valence_electrons)]) / 8.0,  # [16] Weighted valence electrons
            1.0 if any(e.is_transition_metal for e in comp.elements) else 0.0,  # [17] Contains transition metal
            1.0 if any(e.is_noble_gas for e in comp.elements) else 0.0,         # [18] Contains noble gas
            1.0 if any(e.is_halogen for e in comp.elements) else 0.0,           # [19] Contains halogen
        ])

        # ========== 3. Crystal structure tendency features (10-dim) ==========
        # Fix: filter out elements with None atomic_radius
        ionic_radii = []
        for element in comp.elements:
            if element.atomic_radius is not None:
                # Approximate ionic radius
                if element.is_metal:
                    ionic_radii.append(element.atomic_radius * 0.6)  # Cation
                else:
                    ionic_radii.append(element.atomic_radius * 1.4)  # Anion

        if len(ionic_radii) >= 2:
            radius_ratio = min(ionic_radii) / max(ionic_radii)
            features.extend([
                radius_ratio,                                                    # [20] Ionic radius ratio
                1.0 if 0.414 < radius_ratio < 0.732 else 0.0,                   # [21] Octahedral coordination tendency
                1.0 if 0.225 < radius_ratio < 0.414 else 0.0,                   # [22] Tetrahedral coordination tendency
                1.0 if radius_ratio > 0.732 else 0.0,                           # [23] Cubic coordination tendency
            ])
        else:
            # If insufficient valid radii, fill with default values
            features.extend([0.5, 0.0, 0.0, 0.0])  # Default radius ratio 0.5, no coordination tendency

        # Atomic radius statistics
        atomic_radii = [e.atomic_radius for e in comp.elements if e.atomic_radius is not None]
        if atomic_radii:
            features.extend([
                np.mean(atomic_radii) / 300.0,                                  # [24] Average atomic radius
                np.std(atomic_radii) / 100.0,                                   # [25] Atomic radius std dev
                (max(atomic_radii) - min(atomic_radii)) / 300.0,                # [26] Atomic radius span
            ])
        else:
            features.extend([0.0] * 3)

        # Atomic mass statistics
        atomic_masses = [e.atomic_mass for e in comp.elements]
        features.extend([
            np.mean(atomic_masses) / 200.0,                                     # [27] Average atomic mass
            np.std(atomic_masses) / 100.0,                                      # [28] Atomic mass std dev
            (max(atomic_masses) - min(atomic_masses)) / 200.0,                  # [29] Atomic mass span
        ])

        # Ensure 30 dimensions
        if len(features) < 30:
            features.extend([0.0] * (30 - len(features)))

        return np.array(features[:30], dtype=np.float32)

    except Exception as e:
        print(f"Advanced feature extraction failed '{formula_str}': {e}")
        return np.zeros(30, dtype=np.float32)


def extract_bag_of_elements_130(formula_str):
    """
    Extract 130-dim Bag-of-Elements enhanced features (fixed version)

    Includes:
      - 100-dim: Bag-of-Elements (atomic fractions)
      - 8-dim: Average element properties (atomic number, electronegativity, etc.)
      - 8-dim: Weighted element properties
      - 14-dim: Statistical features (element count, molecular weight, etc.)

    Returns:
        bag_features: (130,) numpy array
    """
    try:
        formula_str = reduce_formula(formula_str)
        comp = Composition(formula_str)

        # ===== 1. Bag-of-Elements (100-dim) =====
        bag_vector = np.zeros(100, dtype=np.float32)
        for element in comp.elements:
            atomic_num = element.Z
            if 1 <= atomic_num <= 100:
                bag_vector[atomic_num - 1] = comp.get_atomic_fraction(element)

        # ===== 2. Average element properties (8-dim) =====
        atomic_nums = []
        electronegativities = []
        atomic_radii = []
        atomic_masses = []

        for element in comp.elements:
            atomic_nums.append(element.Z)
            if element.X is not None:
                electronegativities.append(element.X)
            if element.atomic_radius is not None:
                atomic_radii.append(element.atomic_radius)
            atomic_masses.append(element.atomic_mass)

        avg_features = [
            np.mean(atomic_nums) / 100.0 if atomic_nums else 0.0,
            np.mean(electronegativities) / 4.0 if electronegativities else 0.0,
            np.mean(atomic_radii) / 300.0 if atomic_radii else 0.0,
            np.mean(atomic_masses) / 200.0 if atomic_masses else 0.0,
            np.std(atomic_nums) / 50.0 if len(atomic_nums) > 1 else 0.0,
            np.std(electronegativities) / 2.0 if len(electronegativities) > 1 else 0.0,
            np.std(atomic_radii) / 150.0 if len(atomic_radii) > 1 else 0.0,
            np.std(atomic_masses) / 100.0 if len(atomic_masses) > 1 else 0.0,
        ]

        # ===== 3. Weighted element properties (8-dim) =====
        weighted_Z = sum([e.Z * comp.get_atomic_fraction(e) for e in comp.elements]) / 100.0
        weighted_X = sum([e.X * comp.get_atomic_fraction(e) for e in comp.elements if e.X is not None]) / 4.0 if any(e.X is not None for e in comp.elements) else 0.0
        weighted_radius = sum([e.atomic_radius * comp.get_atomic_fraction(e) for e in comp.elements if e.atomic_radius is not None]) / 300.0 if any(e.atomic_radius is not None for e in comp.elements) else 0.0
        weighted_mass = sum([e.atomic_mass * comp.get_atomic_fraction(e) for e in comp.elements]) / 200.0

        weighted_features = [
            weighted_Z,
            weighted_X,
            weighted_radius,
            weighted_mass,
            abs(weighted_Z - avg_features[0]),  # Z deviation
            abs(weighted_X - avg_features[1]),  # X deviation
            abs(weighted_radius - avg_features[2]),  # Radius deviation
            abs(weighted_mass - avg_features[3]),  # Mass deviation
        ]

        # ===== 4. Statistical features (14-dim) =====
        num_elements = len(comp.elements)
        total_atoms = sum([comp[e] for e in comp.elements])
        molecular_weight = comp.weight

        # Element type statistics
        num_metals = sum([1 for e in comp.elements if e.is_metal])
        num_nonmetals = num_elements - num_metals
        metal_fraction = num_metals / num_elements if num_elements > 0 else 0.0

        # Oxidation state statistics
        max_oxidation = max([max(e.common_oxidation_states) if e.common_oxidation_states else 0 for e in comp.elements])
        min_oxidation = min([min(e.common_oxidation_states) if e.common_oxidation_states else 0 for e in comp.elements])

        stat_features = [
            num_elements / 10.0,
            total_atoms / 50.0,
            molecular_weight / 1000.0,
            metal_fraction,
            num_metals / 10.0,
            num_nonmetals / 10.0,
            max_oxidation / 8.0,
            min_oxidation / 8.0 if min_oxidation != 0 else 0.0,
            np.max(list(bag_vector)) if bag_vector.max() > 0 else 0.0,  # Max atomic fraction
            np.min([f for f in bag_vector if f > 0]) if (bag_vector > 0).any() else 0.0,  # Min non-zero atomic fraction
            np.std([f for f in bag_vector if f > 0]) if (bag_vector > 0).sum() > 1 else 0.0,  # Atomic fraction std dev
            1.0 if any(e.is_transition_metal for e in comp.elements) else 0.0,
            1.0 if any(e.is_lanthanoid or e.is_actinoid for e in comp.elements) else 0.0,
            1.0 if any(e.is_alkali for e in comp.elements) else 0.0,
        ]

        # ===== Concatenate all features (130-dim) =====
        all_features = np.concatenate([
            bag_vector,           # 100-dim
            avg_features,         # 8-dim
            weighted_features,    # 8-dim
            stat_features         # 14-dim
        ])

        return all_features.astype(np.float32)

    except Exception as e:
        print(f"Bag-130 feature extraction failed '{formula_str}': {e}")
        return np.zeros(130, dtype=np.float32)


def batch_preprocess_formulas_enhanced_v2(formulas, verbose=False):
    """
    Batch process chemical formulas into 160-dim enhanced features (130-dim Bag + 30-dim chemical context)

    Args:
        formulas: List of formula strings
        verbose: Whether to print detailed information

    Returns:
        enhanced_features: (N, 160) torch tensor
    """
    bag_features_130_list = []
    context_features_30_list = []

    if verbose:
        print(f"🔧 Extracting 160-dim enhanced formula features...")
        iterator = tqdm(formulas, desc="Processing formulas")
    else:
        iterator = formulas

    for formula in iterator:
        # 1. Extract 130-dim Bag features
        bag_130 = extract_bag_of_elements_130(formula)
        bag_features_130_list.append(bag_130)

        # 2. Extract 30-dim chemical context features
        context_30 = extract_advanced_chemical_context(formula)
        context_features_30_list.append(context_30)

    # Optimization: convert to numpy array first then tensor (avoid warnings)
    bag_features_130 = torch.from_numpy(np.array(bag_features_130_list, dtype=np.float32))
    context_features_30 = torch.from_numpy(np.array(context_features_30_list, dtype=np.float32))

    # Concatenate
    enhanced_features_160 = torch.cat([bag_features_130, context_features_30], dim=1)

    if verbose:
        print(f"160-dim enhanced feature extraction completed:")
        print(f"   Shape: {enhanced_features_160.shape}")
        print(f"   Bag features (130-dim): Avg non-zero elements {(bag_features_130[:, :100] > 0).sum(dim=1).float().mean():.2f}")
        print(f"   Chemical context (30-dim): Range [{context_features_30.min():.3f}, {context_features_30.max():.3f}]")

        # Check for NaN and Inf
        if torch.isnan(enhanced_features_160).any():
            num_nan = torch.isnan(enhanced_features_160).sum()
            print(f"   Warning: Found {num_nan} NaN values, auto-filled with 0")
            enhanced_features_160 = torch.nan_to_num(enhanced_features_160, nan=0.0)

        if torch.isinf(enhanced_features_160).any():
            num_inf = torch.isinf(enhanced_features_160).sum()
            print(f"   Warning: Found {num_inf} Inf values, auto-filled with 0")
            enhanced_features_160 = torch.nan_to_num(enhanced_features_160, posinf=0.0, neginf=0.0)

        # Feature distribution statistics
        print(f"\nFeature statistics:")
        print(f"   Periodic table features: Avg period {context_features_30[:, 0].mean():.3f}")
        print(f"   Bond types: Ionic tendency {context_features_30[:, 11].mean():.3f}, Covalent tendency {context_features_30[:, 13].mean():.3f}")
        print(f"   Coordination tendencies: Octahedral {context_features_30[:, 21].mean():.3f}, Tetrahedral {context_features_30[:, 22].mean():.3f}")

    return enhanced_features_160


# =================================================================
# Test function
# =================================================================
if __name__ == "__main__":

    test_cases = [
        ("Fe2O3", "Fe2O3"),      # Cannot simplify
        ("Fe4O6", "Fe2O3"),      # Divide by 2
        ("Fe6O9", "Fe2O3"),      # Divide by 3
        ("BaTiO3", "BaTiO3"),    # Cannot simplify
        ("Al2O3", "Al2O3"),      # Cannot simplify
        ("Ca2O2", "CaO"),        # Divide by 2
        ("Li4Fe4O8", "LiFeO2"),  # Divide by 4
    ]

    print("Testing chemical formula reduction:")
    for original, expected in test_cases:
        result = reduce_formula(original)
        status = "True" if result == expected else "False"
        print(f"{status} {original} → {result} (Expected: {expected})")

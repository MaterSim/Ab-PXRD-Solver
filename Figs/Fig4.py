import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import csv
import matplotlib.pyplot as plt
from collections import Counter
from pxrd_app.inference import infer_formula_spg, spg_to_crystal_system


def collect_lattice_types(list_path: str) -> list[str]:
    lattice_types: list[str] = []
    with open(list_path, "r") as f:
        for line in f:
            csv_path = line.strip()
            if csv_path.startswith("#"): continue
            _, spg = infer_formula_spg(csv_path)
            lattice_type = spg_to_crystal_system(spg)
            lattice_types.append(lattice_type)
    return lattice_types


def collect_success_lattice_types(csv_path: str) -> list[str]:
    lattice_types: list[str] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("Status") or "").strip()
            if status.endswith("Failure"):
                continue

            spg_raw = (row.get("csv_file_name") or "").strip()
            spg_raw = spg_raw.split("_")[-1].split(".")[0]

            try:
                spg = int(spg_raw)
            except ValueError:
                continue

            lattice_type = spg_to_crystal_system(spg)
            if lattice_type is not None:
                lattice_types.append(lattice_type)
    return lattice_types


def collect_success_n_struc(csv_path: str) -> list[int]:
    n_struc_values: list[int] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("Status") or "").strip()
            if not status.endswith("Success") or status.startswith("C-Success"):
                continue

            n_struc_raw = (row.get("N_struc") or "").strip()
            try:
                n_struc_values.append(int(n_struc_raw))
            except ValueError:
                continue
    return n_struc_values


def main():
    lattice_types = collect_lattice_types("data/test.txt"); print(f"Collected {len(lattice_types)} lattice types from test.txt")
    lattice_types.extend(collect_lattice_types("data/mono.txt"))
    counter = Counter(lattice_types)
    print("Lattice type counts:", counter)
    success_lattice_types = collect_success_lattice_types("data/test.csv")
    success_lattice_types.extend(collect_success_lattice_types("data/mono.csv"))
    success_counter = Counter(success_lattice_types)
    success_n_struc = collect_success_n_struc("data/test.csv")
    n_struc_counter = Counter(success_n_struc)
    print("Successful lattice type counts:", success_counter)

    lattice_order = ["Cubic", "Hexagonal", "Trigonal", "Tetragonal", "Orthorhombic", "Monoclinic"]
    counts = [counter.get(lattice_type, 0) for lattice_type in lattice_order]
    counts1 = [success_counter.get(lattice_type, 0) for lattice_type in lattice_order]
    success_rates = [success / total * 100 if total else 0 for total, success in zip(counts, counts1)]
    n_struc_order = sorted(n_struc_counter)
    n_struc_counts = [n_struc_counter[value] for value in n_struc_order]
    total_color = "#cbd5e1"
    success_color = "#0f766e"
    success_edge_color = "#134e4a"
    distribution_color = "#f59e0b"
    distribution_edge_color = "#b45309"
    annotation_color = "#1f2937"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(lattice_order, counts, label=f"Total: {sum(counts)}", color=total_color, edgecolor="white", linewidth=1.2)
    ax1.bar(lattice_order, counts1, label=f"Success: {sum(counts1)}", color=success_color, edgecolor=success_edge_color, linewidth=1.2)

    y_offset = max(counts + counts1) * 0.02 if counts or counts1 else 0.5
    for x, total, rate in zip(lattice_order, counts, success_rates):
        ax1.text(x, total + y_offset, f"{rate:.1f}%", ha="center",
                 va="bottom", fontsize=14, color=annotation_color)
    ax1.set_ylabel("Number of Structures", fontsize=14)
    ax1.set_title("(a) Success Rate Distribution", fontsize=16, fontweight="bold")
    ax1.tick_params(labelsize=12)
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(fontsize=14)
    ax1.set_ylim(0, 300)
    #ax1.set_facecolor("#f8fafc")

    ax2.bar(n_struc_order, n_struc_counts, color=distribution_color, edgecolor=distribution_edge_color, linewidth=1.0)
    ax2.set_xlabel("Number of Valid Structures", fontsize=14)
    ax2.set_ylabel("Occurrence", fontsize=14)
    ax2.set_title("(b) Sampling Cost Distribution", fontsize=16, fontweight="bold")
    ax2.tick_params(labelsize=14)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    if n_struc_order:
        ax2.set_xlim(min(n_struc_order), max(n_struc_order))
    #ax2.set_facecolor("#fff7ed")

    fig.tight_layout()
    fig.savefig("Fig4.pdf", dpi=300)

if __name__ == "__main__":
    main()

import matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tools.gsas import refine_pxrd

def plot_energy_vs_r2(
    structure_log: list,
    best_state: dict,
    output_png: str,
    timing_breakdown_seconds: dict,
) -> None:
    """
    Scatter plot of energy-per-atom vs R² for every relaxed structure explored.
    Structures that were never refined receive R²=0.
    """
    formula = best_state.get("formula", "Unknown Formula")
    status = best_state.get("status")
    engs = [e["eng"] for e in structure_log]
    r2s  = [e["r2"]  for e in structure_log]
    mask = [e.get("refined", False) for e in structure_log]
    unref = [(e, r) for e, r, m in zip(engs, r2s, mask) if not m]
    ref   = [(e, r) for e, r, m in zip(engs, r2s, mask) if m]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 1]})
    # --- Upper plot: Energy vs R2 ---
    if unref:
        ue, ur = zip(*unref)
        ur = [0.0 for _ in ur]
        ax1.scatter(ue, ur, c="steelblue", s=50, alpha=0.5,
                   label=f"Relaxed only (N={len(ue)})")
    if ref:
        re, rr = zip(*ref)
        scatter = ax1.scatter(
            re, rr, c=rr, cmap="Reds", marker="*", s=100, alpha=0.5,
            label=f"Refined (N={len(re)})"
        )
        cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
        cbar.set_label("R² value")
        scatter.set_clim(0, 1)

    if engs:
        x_min = min(float(e) for e in engs)
        x_max = max(float(e) for e in engs)
        ax1.set_xlim(x_min - 0.03, x_max + 0.09)

    ax1.set_xlabel("Energy per atom (eV)")
    ax1.set_ylabel("R² score  (0 = not refined)")
    ax1.set_ylim(-0.1, 1.1)

    def _fmt_breakdown(seconds: float) -> str:
        if seconds is None or seconds <= 0: return "N/A"
        total_minutes = int(seconds // 60)
        seconds_remain = seconds - (60 * total_minutes)
        if total_minutes >= 60:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m {seconds_remain:04.1f}s"
        return f"{total_minutes}m {seconds_remain:04.1f}s"

    total_seconds = timing_breakdown_seconds.get("total")
    spg_cell_seconds = timing_breakdown_seconds.get("spg_and_cell")
    structure_seconds = timing_breakdown_seconds.get("structure_inference")

    time_text = _fmt_breakdown(total_seconds)
    breakdown_text = (
        f"SPG+Cell: {_fmt_breakdown(spg_cell_seconds)} | "
        f"Structure: {_fmt_breakdown(structure_seconds)}"
    )
    ax1.set_title(
        f"{formula} ({len(structure_log)} structures)  "
        f"[{status}]  [Time: {time_text}]"
        + (f"\n[{breakdown_text}]"))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Lower plot: Best fit PXRD ---
    # plot best fit from best state if available, otherwise show placeholder text
    if best_state is not None and best_state.get("best_result") is not None:
        pxrd_csv = best_state.get("pxrd_csv")
        INST_FILE = best_state.get("INST_FILE")
        match_cif = f"tmp/{formula}_best_state.cif"
        best_state['best_result']['xtal'].to_file(match_cif)
        wr, r2, chi2, cif = refine_pxrd(pxrd_csv, match_cif, INST_FILE, ax=ax2, remove=True)
        spg = best_state['best_result']['spg']
        wp_labels_text = best_state['best_result'].get('wp_labels') or best_state.get('wp_labels') or "n/a"
        spg_str = f"SPG: {spg}" if spg else ""
        ax2.set_title(f"Best Fit: R²={r2:.3f}, Chi²={chi2:.3f} | {spg_str}: {wp_labels_text}")
        ax2.set_xlabel("2θ (deg)")
        ax2.set_ylabel("Intensity (a.u.)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No best fit data available", ha="center", va="center", fontsize=12)
        ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)

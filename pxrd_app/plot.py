import matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
        seconds_remain = int(seconds - (60 * total_minutes))
        if total_minutes >= 60:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m {seconds_remain}s"
        return f"{total_minutes}m {seconds_remain}s"

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
    # Plot observed data vs simulated pattern from the best CIF.
    # We do NOT re-run GSAS here (too slow); instead we use the XRD simulator
    # and the stored metrics from the search refinement.
    if best_state is not None and best_state.get("best_result") is not None:
        best_result = best_state['best_result']
        wr   = best_result.get('wr')
        r2   = best_result.get('r2')
        chi2 = best_result.get('chi2')
        spg  = best_result.get('spg')
        wp_labels_text = best_result.get('wp_labels') or best_state.get('wp_labels') or "n/a"
        spg_str = f"SPG: {spg}" if spg else ""

        # Plot observed pattern
        pxrd_csv = best_state.get("pxrd_csv")
        try:
            import pandas as _pd
            _df = _pd.read_csv(pxrd_csv)
            x_obs = _df.iloc[:, 0].values
            y_obs = _df.iloc[:, 1].values
            ax2.plot(x_obs, y_obs, 'k.', markersize=2, label='Observed')
        except Exception:
            pass

        # Plot simulated pattern from best xtal using XRD simulator (no GSAS)
        try:
            from pxrd_app.tools.XRD import XRD
            _xtal = best_result.get('xtal')
            if _xtal is not None:
                _atoms = _xtal.to_ase(resort=False) if hasattr(_xtal, 'to_ase') else None
                if _atoms is not None:
                    _wavelength = best_state.get('wavelength', 1.54184)
                    _thetas = best_state.get('thetas', [10, 80])
                    _res = best_state.get('resolution', 0.02)
                    _tol = best_state.get('SCALED_INTENSITY_TOL', 0.01)
                    _xrd = XRD(_atoms, wavelength=_wavelength, thetas=_thetas,
                               res=_res, SCALED_INTENSITY_TOL=_tol)
                    x_calc, y_calc = _xrd.get_plot_gsas2(
                        U=0.1, V=-0.1, W=0.5, X=0.1, Y=0.1,
                        bg_ratio=0.0, mix_ratio=0.0)
                    # Normalise to observed peak
                    if y_obs is not None and len(y_obs) > 0 and max(y_calc) > 0:
                        y_calc = np.array(y_calc) / max(y_calc) * max(y_obs)
                    ax2.plot(x_calc, y_calc, 'r-', linewidth=0.8, label='Calculated')
        except Exception:
            pass

        r2_str   = f"{r2:.3f}"   if r2   is not None else "n/a"
        chi2_str = f"{chi2:.2f}" if chi2 is not None else "n/a"
        wr_str   = f"{wr:.2f}"   if wr   is not None else "n/a"
        ax2.set_title(f"Best Fit: R²={r2_str}, Chi²={chi2_str}, Rwp={wr_str} | {spg_str}: {wp_labels_text}")
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

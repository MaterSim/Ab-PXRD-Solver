import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_energy_vs_r2(
    structure_log: list,
    formula: str,
    spg: int,
    output_png: str,
    status: str = "Failure",
    elapsed_seconds: float | None = None,
    timing_breakdown_seconds: dict | None = None,
) -> None:
    """
    Scatter plot of energy-per-atom vs R² for every relaxed structure explored.
    Structures that were never refined receive R²=0.
    """
    engs = [e["eng"] for e in structure_log]
    r2s  = [e["r2"]  for e in structure_log]
    mask = [e.get("refined", False) for e in structure_log]

    unref = [(e, r) for e, r, m in zip(engs, r2s, mask) if not m]
    ref   = [(e, r) for e, r, m in zip(engs, r2s, mask) if m]

    fig, ax = plt.subplots(figsize=(8, 5))
    if unref:
        ue, ur = zip(*unref)
        ax.scatter(ue, ur, c="steelblue", s=25, alpha=0.5,
                   label=f"Relaxed only (N={len(ue)})")
    if ref:
        re, rr = zip(*ref)
        ax.scatter(re, rr, c="crimson", marker="*", s=140, alpha=0.7,
                   label=f"Refined (N={len(re)})")

    if engs:
        x_min = min(float(e) for e in engs)
        x_max = max(float(e) for e in engs)
        if (x_max - x_min) < 0.1:
            ax.set_xlim(x_min - 0.01, x_max + 0.09)

    ax.set_xlabel("Energy per atom (eV)")
    ax.set_ylabel("R² score  (0 = not refined)")
    ax.set_ylim(-0.2, 1.1)
    if timing_breakdown_seconds and "total" in timing_breakdown_seconds:
        total_seconds = max(0.0, float(timing_breakdown_seconds.get("total", 0.0)))
    elif elapsed_seconds is not None:
        total_seconds = max(0.0, float(elapsed_seconds))
        total_minutes = int(total_seconds // 60)
        seconds_remain = total_seconds - (60 * total_minutes)
        if total_minutes >= 60:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            time_text = f"{hours}h {minutes}m {seconds_remain:04.1f}s"
        else:
            time_text = f"{total_minutes}m {seconds_remain:04.1f}s"
    else:
        time_text = "n/a"
    if timing_breakdown_seconds and "total" in timing_breakdown_seconds:
        total_minutes = int(total_seconds // 60)
        seconds_remain = total_seconds - (60 * total_minutes)
        if total_minutes >= 60:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            time_text = f"{hours}h {minutes}m {seconds_remain:04.1f}s"
        else:
            time_text = f"{total_minutes}m {seconds_remain:04.1f}s"
    breakdown_text = None
    if timing_breakdown_seconds:
        spg_cell_seconds = max(0.0, float(timing_breakdown_seconds.get("spg_and_cell", 0.0)))
        structure_seconds = max(0.0, float(timing_breakdown_seconds.get("structure_inference", 0.0)))

        def _fmt_breakdown(seconds: float) -> str:
            total_minutes = int(seconds // 60)
            seconds_remain = seconds - (60 * total_minutes)
            if total_minutes >= 60:
                hours = total_minutes // 60
                minutes = total_minutes % 60
                return f"{hours}h {minutes}m {seconds_remain:04.1f}s"
            return f"{total_minutes}m {seconds_remain:04.1f}s"

        breakdown_text = (
            f"SPG+Cell: {_fmt_breakdown(spg_cell_seconds)} | "
            f"Structure: {_fmt_breakdown(structure_seconds)}"
        )
    ax.set_title(
        f"{formula}  SPG {spg} — Energy vs R²  ({len(structure_log)} structures)  "
        f"[{status}]  [Time: {time_text}]"
        + (f"\n[{breakdown_text}]" if breakdown_text else "")
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pxrd_app.tools.gsas import simulate_pxrd
import matplotlib.pyplot as plt

INST_FILE = "pxrd_app/tools/INST_XRY.PRM"
fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.5))
ax1, ax2, ax3, ax4 = axes.ravel()
ax1.set_title("(a) Similar Structures → Different PXRDs", fontsize=15, fontweight='bold')
x, y =simulate_pxrd("data/Tl3Au.cif",  Tmax=80.0, iparams=INST_FILE, 
                    U=1.0, V=-0.1, W=10.0, noise_level=0.1, Tstep=0.1,
                    X=10.0, Y=10.0, bg_ratio=0.00, grainsize=2.0)
ax1.plot(x, y, label='$Pm\overline{3}n$ Tl$_3$Au', linewidth=2.5, color='#1f77b4')
ax1.legend(loc=2, fontsize=15, frameon=False)
ax1.set_xlabel("2θ (degrees)", fontsize=15)
ax1.set_ylabel("Intensity (a.u.)", fontsize=15)
#ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=12)
ax1.set_yticks(range(0, 101, 20))
x, y =simulate_pxrd("data/Ti3Pd.cif",  Tmax=80.0, iparams=INST_FILE, 
                    U=1.0, V=-0.1, W=10.0, noise_level=0.1, Tstep=0.1,
                    X=10.0, Y=10.0, bg_ratio=0.00, grainsize=2.0)
ax3.plot(x, y, label='$Pm\overline{3}n$ Ti$_3$Pd', linewidth=2.5, color='#ff7f0e')
ax3.legend(loc=2, fontsize=15, frameon=False)
ax3.set_xlabel("2θ (degrees)", fontsize=15)
ax3.set_ylabel("Intensity (a.u.)", fontsize=15)
ax3.tick_params(labelsize=12)
ax3.set_yticks(range(0, 101, 20))

ax2.set_title("(b) Different Structures → Similar PXRDs", fontsize=15, fontweight='bold')
x, y =simulate_pxrd("data/2124726.cif",  Tmax=50.0, iparams=INST_FILE, 
                    U=1.0, V=-0.1, W=10.0, noise_level=0.1, Tstep=0.1,
                    X=10.0, Y=10.0, bg_ratio=0.00, grainsize=2.0)
ax2.plot(x, y, label='$P$2$_1$/$c$ di­fluoro­quinacridone #A', linewidth=2.5, color='#2ca02c')
ax2.legend(loc=1, fontsize=15, frameon=False)
ax2.set_xlabel("2θ (degrees)", fontsize=15)
#ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=12)
ax2.set_yticks(range(0, 101, 20))

x, y =simulate_pxrd("data/2124729.cif",  Tmax=50.0, iparams=INST_FILE, 
                    U=1.0, V=-0.1, W=10.0, noise_level=0.1, Tstep=0.1,
                    X=10.0, Y=10.0, bg_ratio=0.00, grainsize=2.0)
ax4.plot(x, y, label='$P$2$_1$/$c$ di­fluoro­quinacridone #D', linewidth=2.5, color='#d62728')
ax4.legend(loc=1, fontsize=15, frameon=False)
ax4.set_xlabel("2θ (degrees)", fontsize=15)
#ax4.grid(True, alpha=0.3, linestyle='--')
ax4.tick_params(labelsize=12)
ax4.set_yticks(range(0, 101, 20))
ax1.set_yticks(range(0, 101, 20))
ax2.set_yticks(range(0, 101, 20))
ax3.set_yticks(range(0, 101, 20))
ax1.set_ylim(-2, 125)
ax2.set_ylim(-2, 125)
ax3.set_ylim(-2, 125)
ax4.set_ylim(-2, 125)

fig.tight_layout()
fig.savefig("Fig1.pdf", dpi=300)

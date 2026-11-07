"""Dataset and broadening illustration figures (independent of model results)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path.home() / ".claude/skills/sci-journal-paper/scripts"))
sys.path.insert(0, str(Path(__file__).parent))
import journal_figures as jf
from broadening import broaden, scherrer_fwhm_deg, peak_features, TTH
from run_experiments import load, SYSTEMS, DATA

FIG = Path(__file__).resolve().parents[1] / "figures"
RES = Path(__file__).resolve().parents[1] / "results"
jf._style()
d = load()

# Fig: crystal-system counts (train/test) and bandgap histogram
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ctr, cte = np.bincount(d["ctr"], minlength=7), np.bincount(d["cte"], minlength=7)
x = np.arange(7)
ax[0].bar(x - 0.2, ctr, 0.4, label="Train (n = 2,499)", color="#2F5D8A")
ax[0].bar(x + 0.2, cte, 0.4, label="Test (n = 1,072)", color="#B5651D")
ax[0].set_xticks(x, [s.capitalize() for s in SYSTEMS], rotation=35, ha="right")
ax[0].set_ylabel("Number of compounds"); ax[0].legend(frameon=False); ax[0].set_title("(a)", loc="left")
y = np.concatenate([d["ytr"], d["yte"]]); semi = y[y > 0.01]
ax[1].hist(semi, bins=np.arange(0, 12.1, 0.25), color="#2A7F75", edgecolor="white")
ax[1].set_xlabel("Experimental bandgap (eV)"); ax[1].set_ylabel("Number of semiconductors")
ax[1].set_title("(b)", loc="left")
fig.tight_layout(); jf._save(fig, str(FIG / "dataset_distribution.png"))
pd.DataFrame(dict(system=SYSTEMS, train=ctr, test=cte)).to_csv(RES / "dataset_crystal_systems.csv", index=False)
print("metal train/test", (d["ytr"] <= 0.01).sum(), (d["yte"] <= 0.01).sum(), "semi", len(semi),
      "semi stats", semi.mean().round(3), np.median(semi).round(3), semi.min(), semi.max())

# Fig: Scherrer FWHM vs 2theta and an example compound at several sizes
names = list(d["names_tr"]); i = names.index("ZnO") if "ZnO" in names else 0
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1, 1.6]))
for L in [100, 50, 20, 10, 5, 3]:
    ax[0].plot(TTH, scherrer_fwhm_deg(TTH, L), label=f"{L} nm")
ax[0].set_yscale("log"); ax[0].set_xlabel(r"2$\theta$ (deg)"); ax[0].set_ylabel("Size-induced FWHM (deg)")
ax[0].legend(frameon=False, ncol=2, fontsize=8); ax[0].set_title("(a)", loc="left")
sizes = [None, 50, 10, 3]
for k, L in enumerate(sizes):
    p = broaden(d["Xtr"][i:i + 1], L)[0]
    ax[1].plot(TTH, p + 1.1 * (len(sizes) - 1 - k), lw=0.9, color=plt.cm.viridis(k / len(sizes)))
    ax[1].text(88, 1.1 * (len(sizes) - 1 - k) + 0.35, "bulk" if L is None else f"{L} nm", ha="right", fontsize=9)
ax[1].set_xlabel(r"2$\theta$ (deg)"); ax[1].set_ylabel("Normalised intensity (offset)"); ax[1].set_yticks([])
ax[1].set_xlim(5, 90); ax[1].set_title(f"(b) {names[i]}", loc="left")
fig.tight_layout(); jf._save(fig, str(FIG / "broadening_example.png"))

# Fig: resolvable peaks and pattern similarity vs crystallite size (test set)
rows = []
bulk = broaden(d["Xte"], None)
for L in [None, 100, 50, 30, 20, 10, 5, 3]:
    B = broaden(d["Xte"], L); P = peak_features(B)
    corr = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(bulk, B)])
    rows.append(dict(size="bulk" if L is None else L, n_peaks_mean=P[:, 0].mean(), n_peaks_sd=P[:, 0].std(),
                     mean_intensity=B.mean(), corr_to_bulk=corr.mean()))
df = pd.DataFrame(rows); df.to_csv(RES / "pattern_statistics.csv", index=False); print(df.round(3))

"""Result figures and markdown tables from the CSV files in ../results."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.metrics import f1_score, confusion_matrix
sys.path.insert(0, str(Path.home() / ".claude/skills/sci-journal-paper/scripts"))
import journal_figures as jf

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, TAB = ROOT / "results", ROOT / "figures", ROOT / "results" / "tables"
TAB.mkdir(exist_ok=True)
jf._style()
SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]
ORDER = ["bulk", "100", "50", "30", "20", "10", "5", "3"]
XLAB = ["Bulk", "100", "50", "30", "20", "10", "5", "3"]
REG_STYLE = {"bulk": ("#9E3B3B", "o", "Bulk-trained"), "aug": ("#2F5D8A", "s", "Size-augmented"), "matched": ("#4B7F2A", "^", "Size-matched")}
MODELS = [("CNN-XRD", "1D-CNN (profile)"), ("RF-Profile", "RF (binned profile)"), ("RF-Peaks", "RF (peak list)")]


def save(fig, name):
    fig.tight_layout(); jf._save(fig, str(FIG / name)); plt.close(fig)


def pm(m, s, k=2):
    return f"{m:.{k}f} ± {s:.{k}f}"


allres = pd.concat([pd.read_csv(RES / "cs_rf.csv"), pd.read_csv(RES / "cs_cnn.csv")[["model", "regime", "size", "seed", "ACC", "BACC", "F1"]]])
allres["size"] = allres["size"].astype(str)
for c in ["ACC", "BACC", "F1"]:
    allres[c] *= 100
agg = allres.groupby(["model", "regime", "size"])[["ACC", "BACC", "F1"]].agg(["mean", "std"])
agg.to_csv(RES / "summary_by_size.csv")

# ---- Fig: accuracy and macro-F1 vs size, one column per model
fig, ax = plt.subplots(2, 3, figsize=(12, 6.4), sharey="row")
x = np.arange(len(ORDER))
for j, (m, title) in enumerate(MODELS):
    for r, (col, mk, lab) in REG_STYLE.items():
        for i, met in enumerate(["ACC", "F1"]):
            mu = [agg.loc[(m, r, s), (met, "mean")] for s in ORDER]
            sd = [agg.loc[(m, r, s), (met, "std")] for s in ORDER]
            ax[i, j].errorbar(x, mu, yerr=sd, color=col, marker=mk, ms=5, capsize=3, label=lab)
    ax[0, j].set_title(f"({'abc'[j]}) {title}", loc="left", fontsize=11)
    ax[1, j].set_title(f"({'def'[j]}) {title}", loc="left", fontsize=11)
    for i in range(2):
        ax[i, j].set_xticks(x, XLAB); ax[i, j].set_ylim(0, 75)
    ax[1, j].set_xlabel("Test crystallite size (nm)")
ax[0, 0].set_ylabel("Accuracy (%)"); ax[1, 0].set_ylabel("Macro-F1 (%)")
ax[1, 1].legend(frameon=False, fontsize=9, loc="center", bbox_to_anchor=(0.5, 0.3))
save(fig, "metrics_vs_size.png")

# ---- Table: main results (macro-F1 and accuracy, mean ± SD) at all sizes
lines = ["| Model | Regime | " + " | ".join(XLAB) + " |", "|" + "---|" * (2 + len(ORDER))]
for met in ["F1", "ACC"]:
    for m, title in MODELS:
        for r in ["bulk", "aug", "matched"]:
            vals = [pm(agg.loc[(m, r, s), (met, "mean")], agg.loc[(m, r, s), (met, "std")], 1) for s in ORDER]
            lines.append(f"| {title} | {REG_STYLE[r][2]} | " + " | ".join(vals) + " |")
    lines.append("|" + "|".join([""] * (3 + len(ORDER))) + "|")
(TAB / "main_results.md").write_text("\n".join(lines))

# ---- Fig: pattern statistics vs size
ps = pd.read_csv(RES / "pattern_statistics.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].errorbar(x, ps.n_peaks_mean, yerr=ps.n_peaks_sd, marker="o", color="#2F5D8A", capsize=3)
ax[0].set_ylabel("Detected peaks per pattern"); ax[0].set_title("(a)", loc="left")
ax[1].plot(x, ps.corr_to_bulk, marker="s", color="#B5651D"); ax[1].set_ylabel("Correlation with bulk pattern, $r_L$")
ax[1].set_title("(b)", loc="left"); ax[1].set_ylim(0, 1.05)
for a in ax:
    a.set_xticks(x, XLAB); a.set_xlabel("Crystallite size (nm)")
save(fig, "pattern_statistics.png")

# ---- per-sample predictions (seed 0): confusion matrices, predicted-class shift, per-class F1, McNemar
pr = pd.read_csv(RES / "cs_predictions_seed0.csv")
y = pr.y_true.to_numpy()
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
short = ["Tric.", "Mono.", "Orth.", "Tetr.", "Trig.", "Hex.", "Cub."]
for a, (col, ttl) in zip(ax, [("CNN_bulk_bulk", "(a) Bulk-trained, bulk test"), ("CNN_bulk_5", "(b) Bulk-trained, 5 nm test"), ("CNN_aug_5", "(c) Size-augmented, 5 nm test")]):
    cm = confusion_matrix(y, pr[col], labels=range(7)).astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            if cm[i, j] > 0:
                a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.55 else "black")
    a.set_xticks(range(7), short, rotation=45, ha="right", fontsize=8); a.set_yticks(range(7), short, fontsize=8)
    a.set_xlabel("Predicted"); a.set_title(ttl, loc="left", fontsize=10); a.grid(False)
ax[0].set_ylabel("True")
save(fig, "confusion_cnn.png")

# predicted-class distribution of bulk-trained models as size shrinks
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
colors = plt.cm.tab10(np.arange(7))
for a, (pref, ttl) in zip(ax, [("CNN_bulk_", "(a) Bulk-trained 1D-CNN"), ("RFPeaks_bulk_", "(b) Bulk-trained RF (peak list)")]):
    bottom = np.zeros(len(ORDER) + 1)
    fr = np.array([[np.mean(y == c) for c in range(7)]] + [[np.mean(pr[pref + s] == c) for c in range(7)] for s in ORDER])
    for c in range(7):
        a.bar(np.arange(len(ORDER) + 1), fr[:, c] * 100, bottom=bottom, color=colors[c], label=SYSTEMS[c].capitalize(), width=0.75)
        bottom += fr[:, c] * 100
    a.set_xticks(np.arange(len(ORDER) + 1), ["True"] + XLAB, rotation=0, fontsize=9)
    a.set_xlabel("Test crystallite size (nm)"); a.set_title(ttl, loc="left", fontsize=10); a.grid(False)
ax[0].set_ylabel("Share of predictions (%)")
ax[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
save(fig, "prediction_shift.png")
shift = {pref: {s: float(np.mean(pr[pref + s] == 0) * 100) for s in ORDER} for pref in ["CNN_bulk_", "RFPeaks_bulk_"]}
pd.DataFrame(shift).to_csv(RES / "triclinic_share_bulk_models.csv")

# per-class F1 of the size-augmented CNN vs size
pc = pd.DataFrame({s: f1_score(y, pr["CNN_aug_" + s], average=None, labels=range(7)) * 100 for s in ORDER}, index=[c.capitalize() for c in SYSTEMS])
pc.to_csv(RES / "per_class_f1_cnn_aug_seed0.csv")
fig, ax = plt.subplots(figsize=(7.5, 4))
for k, row in enumerate(pc.itertuples()):
    ax.plot(x, row[1:], marker="o", color=colors[k], label=row[0])
ax.set_xticks(x, XLAB); ax.set_xlabel("Test crystallite size (nm)"); ax.set_ylabel("F1-score (%)")
ax.legend(frameon=False, fontsize=8, ncol=2); ax.set_ylim(0, 100)
save(fig, "per_class_f1.png")
sup = np.bincount(y, minlength=7)
lines = ["| Crystal system | Support | " + " | ".join(XLAB[i] for i in [0, 4, 5, 6, 7]) + " |", "|" + "---|" * 7]
for k, c in enumerate(pc.index):
    lines.append(f"| {c} | {sup[k]} | " + " | ".join(f"{pc.loc[c, ORDER[i]]:.1f}" for i in [0, 4, 5, 6, 7]) + " |")
(TAB / "per_class.md").write_text("\n".join(lines))


def mcnemar(a, b):
    n01 = int(np.sum((a == y) & (b != y))); n10 = int(np.sum((a != y) & (b == y)))
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    return n01, n10, stat, chi2.sf(stat, 1)


rows = []
for s in ORDER:
    for (a, b, lab) in [("CNN_aug_", "CNN_bulk_", "Aug. CNN vs bulk CNN"), ("CNN_aug_", "RFPeaks_aug_", "Aug. CNN vs aug. RF (peaks)")]:
        n01, n10, st, p = mcnemar(pr[a + s].to_numpy(), pr[b + s].to_numpy())
        rows.append(dict(comparison=lab, size=s, acc_a=np.mean(pr[a + s] == y) * 100, acc_b=np.mean(pr[b + s] == y) * 100, n01=n01, n10=n10, chi2=st, p=p))
mc = pd.DataFrame(rows); mc.to_csv(RES / "mcnemar_seed0.csv", index=False)

# ---- training curves
h = pd.read_csv(RES / "cnn_history_seed0.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for r in ["bulk", "aug"]:
    hh = h[h.regime == r]; col, _, lab = REG_STYLE[r]
    ax[0].plot(hh.epoch, hh.train_loss, color=col, label=f"{lab}, training")
    ax[0].plot(hh.epoch, hh.val_loss, color=col, ls="--", label=f"{lab}, validation")
    ax[1].plot(hh.epoch, hh.val_f1 * 100, color=col, label=lab)
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("(a)", loc="left")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Validation macro-F1 (%)"); ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("(b)", loc="left")
save(fig, "training_curves.png")

# ---- ablation
ab = pd.read_csv(RES / "ablation.csv"); ab["size"] = ab["size"].astype(str)
ab["F1"] *= 100; ab["ACC"] *= 100
sel = ["bulk", "20", "10", "5", "3"]
abm = ab[ab["size"].isin(sel)].groupby(["variant", "size"]).F1.agg(["mean", "std"])
variants = list(dict.fromkeys(ab.variant))
fig, ax = plt.subplots(figsize=(10, 4))
wbar = 0.8 / len(variants)
for k, v in enumerate(variants):
    ax.bar(np.arange(len(sel)) + k * wbar - 0.4 + wbar / 2, [abm.loc[(v, s), "mean"] for s in sel], wbar,
           yerr=[abm.loc[(v, s), "std"] for s in sel], capsize=2, label=v, color=plt.cm.Set2(k))
ax.set_xticks(np.arange(len(sel)), ["Bulk", "20", "10", "5", "3"]); ax.set_xlabel("Test crystallite size (nm)")
ax.set_ylabel("Macro-F1 (%)"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
save(fig, "ablation.png")
full = abm.xs("Full (size-aug CNN)")
lines = ["| Variant | " + " | ".join(["Bulk", "20 nm", "10 nm", "5 nm", "3 nm"]) + " | Mean Δ |", "|" + "---|" * 7]
for v in variants:
    d = np.mean([abm.loc[(v, s), "mean"] - full.loc[s, "mean"] for s in sel])
    lines.append(f"| {v} | " + " | ".join(pm(abm.loc[(v, s), "mean"], abm.loc[(v, s), "std"], 1) for s in sel) + f" | {d:+.1f} |")
(TAB / "ablation.md").write_text("\n".join(lines))

# ---- profile-shape sensitivity
pf = pd.read_csv(RES / "profile_sensitivity.csv"); pf["F1"] *= 100; pf["ACC"] *= 100
pfa = pf.groupby(["eta", "size"]).F1.agg(["mean", "std"])
sz = [100, 50, 20, 10, 5, 3]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for eta, col, lab in [(0.0, "#2A7F75", "Gaussian ($\\eta$ = 0)"), (0.5, "#2F5D8A", "Pseudo-Voigt ($\\eta$ = 0.5, training)"), (1.0, "#B5651D", "Lorentzian ($\\eta$ = 1)")]:
    ax.errorbar(range(len(sz)), [pfa.loc[(eta, s), "mean"] for s in sz], yerr=[pfa.loc[(eta, s), "std"] for s in sz], marker="o", capsize=3, color=col, label=lab)
ax.set_xticks(range(len(sz)), [str(s) for s in sz]); ax.set_xlabel("Test crystallite size (nm)"); ax.set_ylabel("Macro-F1 (%)")
ax.legend(frameon=False, fontsize=8)
save(fig, "profile_sensitivity.png")
lines = ["| Test profile | " + " | ".join(f"{s} nm" for s in sz) + " |", "|" + "---|" * (1 + len(sz))]
for eta, lab in [(0.0, "Gaussian (η = 0)"), (0.5, "Pseudo-Voigt (η = 0.5)"), (1.0, "Lorentzian (η = 1)")]:
    lines.append(f"| {lab} | " + " | ".join(pm(pfa.loc[(eta, s), "mean"], pfa.loc[(eta, s), "std"], 1) for s in sz) + " |")
(TAB / "profile.md").write_text("\n".join(lines))

# ---- learning curve (100 % = full ablation variant)
lc = pd.read_csv(RES / "learning_curve.csv"); lc["size"] = lc["size"].astype(str); lc["F1"] *= 100
full_rows = ab[(ab.variant == "Full (size-aug CNN)") & ab["size"].isin(["bulk", "10", "5", "3"])].assign(frac=1.0, n_train=2499)
lc = pd.concat([lc, full_rows[lc.columns.intersection(full_rows.columns)]])
lca = lc.groupby(["frac", "size"]).agg(F1=("F1", "mean"), sd=("F1", "std"), n=("n_train", "first"))
lca.to_csv(RES / "learning_curve_summary.csv")
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for s, col in zip(["bulk", "10", "5", "3"], ["#4B7F2A", "#2F5D8A", "#B5651D", "#9E3B3B"]):
    sub = lca.xs(s, level="size")
    ax.errorbar(sub.n, sub.F1, yerr=sub.sd, marker="o", capsize=3, color=col, label="Bulk test" if s == "bulk" else f"{s} nm test")
ax.set_xlabel("Training compounds"); ax.set_ylabel("Macro-F1 (%)"); ax.legend(frameon=False, fontsize=8)
save(fig, "learning_curve.png")

# ---- timing table
tm = pd.read_csv(RES / "timing.csv"); tm.to_csv(RES / "timing_summary.csv", index=False)
print(mc.round(4).to_string()); print(pc.round(1)); print(lca.round(2)); print(tm)

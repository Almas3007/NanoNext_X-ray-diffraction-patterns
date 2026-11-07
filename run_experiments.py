"""Crystal-system identification and bandgap prediction for nanocrystals from
crystallite-size-broadened XRD patterns.

Task A: 7-class crystal-system identification from XRD, full 70/30 split (split_70_30.csv).
Task B: bandgap regression for the semiconductors of the same split, from composition,
        XRD, and composition + CNN-predicted crystal-system probabilities.

Usage: python3 run_experiments.py {cs_rf|cs_cnn|profile|bandgap|all}
Results go to ../results/.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             f1_score, balanced_accuracy_score, confusion_matrix)

sys.path.insert(0, str(Path(__file__).parent))
from broadening import broaden, broadening_matrix, peak_features

DATA = Path(__file__).resolve().parents[2] / "exp_bandgap_dataset"
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
SIZES = [None, 100, 50, 30, 20, 10, 5, 3]          # None = bulk (instrumental width only)
AUG_BANK = np.geomspace(3, 100, 16)
SEEDS = [0, 1, 2]
SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]
METAL_EG = 0.01
torch.set_num_threads(8)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def size_label(s):
    return "bulk" if s is None else f"{s:g}"


def load():
    xrd = pd.read_csv(DATA / "xrd_patterns.csv").drop(columns=["bandgap"]).set_index("compound")
    mag = pd.read_csv(DATA / "magpie_features.csv").drop(columns=["bandgap"]).set_index("compound_name")
    sg = pd.read_csv(DATA / "spacegroup_labels.csv").set_index("compound")
    sp = pd.read_csv(DATA / "split_70_30.csv")
    tr, te = sp[sp.split == "train"], sp[sp.split == "test"]
    get = lambda d, s: d.loc[s.compound].to_numpy(np.float32)
    mag_tr, mag_te = get(mag, tr), get(mag, te)
    med = np.nanmedian(mag_tr, axis=0)
    mag_tr, mag_te = np.where(np.isnan(mag_tr), med, mag_tr), np.where(np.isnan(mag_te), med, mag_te)
    cs = lambda s: sg.loc[s.compound].crystal_system.map(SYSTEMS.index).to_numpy()
    return dict(Xtr=get(xrd, tr), Xte=get(xrd, te), Mtr=mag_tr, Mte=mag_te,
                ctr=cs(tr), cte=cs(te), ytr=tr.bandgap.to_numpy(np.float32), yte=te.bandgap.to_numpy(np.float32),
                names_tr=tr.compound.to_numpy(), names_te=te.compound.to_numpy())


def binned(X, f=2):
    """0.1 deg bins (850 features) for tree models."""
    n = X.shape[1] // f * f
    return X[:, :n].reshape(len(X), -1, f).mean(axis=2)


def augment_set(X, y, rng, copies=4):
    """Bulk copy plus `copies` copies broadened with sizes drawn from the 3-100 nm bank."""
    Xs, ys = [broaden(X, None)], [y]
    for _ in range(copies):
        sizes = rng.choice(AUG_BANK, len(X))
        Xa = np.empty_like(X)
        for s in np.unique(sizes):
            m = sizes == s
            Xa[m] = broaden(X[m], s)
        Xs.append(Xa); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def cls_metrics(y, p):
    return dict(ACC=accuracy_score(y, p), BACC=balanced_accuracy_score(y, p), F1=f1_score(y, p, average="macro"))


def reg_metrics(y, p):
    return dict(MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5, R2=r2_score(y, p))


# ---------------------------------------------------------------- 1D CNN
class XRDNet(nn.Module):
    def __init__(self, n_out=len(SYSTEMS)):
        super().__init__()
        def block(i, o, k, s=1):
            return nn.Sequential(nn.Conv1d(i, o, k, stride=s, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(block(1, 16, 15, 2), block(16, 32, 9), block(32, 64, 7), block(64, 64, 5))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_out))

    def forward(self, x):
        h = self.features(x.unsqueeze(1))
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(h)


def train_cnn(X, y, regime, seed, fixed_size=None, epochs=150, patience=40, max_shift=4):
    """Crystal-system CNN. regime: 'bulk', 'aug' (random size per batch, 20% bulk) or 'matched'."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); nv = int(0.15 * len(X))
    va, tr = idx[:nv], idx[nv:]
    kernels = {s: torch.from_numpy(broadening_matrix(s).T).to(DEV) for s in AUG_BANK} if regime == "aug" else {}
    base = broaden(X, fixed_size) if regime == "matched" else broaden(X, None)
    if regime == "aug":   # validation patterns spread over the same size range
        vsz = rng.choice(list(AUG_BANK) + [None] * 4, len(va))
        Xv = np.vstack([broaden(X[i:i + 1], s) for i, s in zip(va, vsz)])
    else:
        Xv = base[va]
    Xt, yt = torch.from_numpy(base).to(DEV), torch.from_numpy(y.astype(np.int64)).to(DEV)
    Xv = torch.from_numpy(Xv).to(DEV)
    w = torch.tensor(len(tr) / (len(SYSTEMS) * np.bincount(y[tr], minlength=len(SYSTEMS))), dtype=torch.float32, device=DEV)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    model = XRDNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = tr[rng.permutation(len(tr))]
        for b in range(0, len(perm), 64):
            bi = perm[b:b + 64]; xb = Xt[bi]
            if regime == "aug":   # independent random size per pattern, 20% left at bulk width
                xb = xb.clone(); sz = rng.integers(-1, len(AUG_BANK), len(bi))
                sz[rng.random(len(bi)) < 0.2] = -1
                for k in np.unique(sz[sz >= 0]):
                    m = torch.from_numpy(np.where(sz == k)[0]).to(DEV)
                    xk = xb[m] @ kernels[AUG_BANK[k]]
                    xb[m] = xk / xk.amax(dim=1, keepdim=True).clamp_min(1e-8)
            if max_shift:   # small zero-shift / lattice-strain jitter of +-0.2 deg
                xb = torch.roll(xb, int(rng.integers(-max_shift, max_shift + 1)), dims=1)
            opt.zero_grad(); loss = loss_fn(model(xb), yt[bi]); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).argmax(1).cpu().numpy()
        vl = -f1_score(y[va], pv, average="macro")      # early stopping on validation macro-F1
        if vl < best - 1e-4:
            best, bad = vl, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); model.eval()

    def predict_proba(Xn):
        with torch.no_grad():
            out = [torch.softmax(model(torch.from_numpy(Xn[i:i + 512]).to(DEV)), 1).cpu().numpy() for i in range(0, len(Xn), 512)]
        return np.vstack(out)
    return predict_proba, ep + 1


# ---------------------------------------------------------------- Task A: crystal system
def run_cs_rf():
    d = load(); rows = []
    test_sets = {s: broaden(d["Xte"], s) for s in SIZES}
    test_peaks = {s: peak_features(T) for s, T in test_sets.items()}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        rf = lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed, class_weight="balanced")
        Xb = broaden(d["Xtr"], None)
        m_prof, m_peak = rf().fit(binned(Xb), d["ctr"]), rf().fit(peak_features(Xb), d["ctr"])
        Xa, ca = augment_set(d["Xtr"], d["ctr"], rng)
        m_prof_a, m_peak_a = rf().fit(binned(Xa), ca), rf().fit(peak_features(Xa), ca)
        for s in SIZES:
            T, P, sl = test_sets[s], test_peaks[s], size_label(s)
            add = lambda model, regime, p: rows.append(dict(model=model, regime=regime, size=sl, seed=seed, **cls_metrics(d["cte"], p)))
            add("RF-Profile", "bulk", m_prof.predict(binned(T)))
            add("RF-Peaks", "bulk", m_peak.predict(P))
            add("RF-Profile", "aug", m_prof_a.predict(binned(T)))
            add("RF-Peaks", "aug", m_peak_a.predict(P))
            if s is not None:
                Xm = broaden(d["Xtr"], s)
                add("RF-Profile", "matched", rf().fit(binned(Xm), d["ctr"]).predict(binned(T)))
                add("RF-Peaks", "matched", rf().fit(peak_features(Xm), d["ctr"]).predict(P))
        print("cs_rf seed", seed, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "cs_rf.csv", index=False)
    df = pd.DataFrame(rows)   # bulk test of the bulk-trained model is also the matched result at "bulk"
    df = pd.concat([df, df[(df.regime == "bulk") & (df["size"] == "bulk")].assign(regime="matched")])
    df.to_csv(OUT / "cs_rf.csv", index=False)


def run_cs_cnn():
    d = load(); rows, cms, probs = [], {}, {}
    test_sets = {s: broaden(d["Xte"], s) for s in SIZES}
    for seed in SEEDS:
        for regime in ["bulk", "aug"]:
            t = time.time(); pp, ne = train_cnn(d["Xtr"], d["ctr"], regime, seed)
            for s in SIZES:
                pr = pp(test_sets[s]); p = pr.argmax(1)
                rows.append(dict(model="CNN-XRD", regime=regime, size=size_label(s), seed=seed, epochs=ne, train_s=time.time() - t, **cls_metrics(d["cte"], p)))
                if seed == 0:
                    cms[f"{regime}_{size_label(s)}"] = confusion_matrix(d["cte"], p, labels=range(len(SYSTEMS)))
            print("cs_cnn", regime, seed, ne, round(time.time() - t), flush=True)
            pd.DataFrame(rows).to_csv(OUT / "cs_cnn.csv", index=False)
        for s in SIZES[1:]:
            t = time.time(); pp, ne = train_cnn(d["Xtr"], d["ctr"], "matched", seed, fixed_size=s)
            rows.append(dict(model="CNN-XRD", regime="matched", size=size_label(s), seed=seed, epochs=ne, train_s=time.time() - t, **cls_metrics(d["cte"], pp(test_sets[s]).argmax(1))))
            print("cs_cnn matched", s, seed, round(time.time() - t), flush=True)
            pd.DataFrame(rows).to_csv(OUT / "cs_cnn.csv", index=False)
    rows += [dict(r, regime="matched") for r in rows if r["regime"] == "bulk" and r["size"] == "bulk"]
    pd.DataFrame(rows).to_csv(OUT / "cs_cnn.csv", index=False)
    np.savez(OUT / "cs_cnn_confusion_seed0.npz", labels=np.array(SYSTEMS), **cms)


def run_profile():
    """Peak-shape sensitivity: aug-CNN (trained with eta = 0.5) tested on Gaussian (0) and Lorentzian (1) broadening."""
    d = load(); rows = []
    for seed in SEEDS:
        pp, _ = train_cnn(d["Xtr"], d["ctr"], "aug", seed)
        for eta in [0.0, 0.5, 1.0]:
            for s in [100, 50, 20, 10, 5, 3]:
                rows.append(dict(eta=eta, size=s, seed=seed, **cls_metrics(d["cte"], pp(broaden(d["Xte"], s, eta=eta)).argmax(1))))
        print("profile seed", seed, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "profile_sensitivity.csv", index=False)


# ---------------------------------------------------------------- Task B: bandgap
def run_bandgap():
    """Bandgap of the semiconductors (Eg > 0.01 eV) in split_70_30.

    Crystal-system probabilities for training compounds are 5-fold out-of-fold predictions of
    the size-augmented CNN, so the regressor never sees in-sample CNN outputs.
    """
    d = load(); rows = []
    str_, ste = d["ytr"] > METAL_EG, d["yte"] > METAL_EG
    Mtr, Mte, ytr, yte = d["Mtr"][str_], d["Mte"][ste], d["ytr"][str_], d["yte"][ste]
    onehot = lambda c: np.eye(len(SYSTEMS), dtype=np.float32)[c]
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        rf = lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed, min_samples_leaf=2)
        # out-of-fold CNN probabilities on the training set, at a random crystallite size per compound
        oof = np.zeros((len(d["Xtr"]), len(SYSTEMS)), np.float32)
        tr_sizes = rng.choice(list(AUG_BANK) + [None] * 4, len(d["Xtr"]))
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(d["Xtr"], d["ctr"])):
            pp, _ = train_cnn(d["Xtr"][a], d["ctr"][a], "aug", seed * 10 + k)
            oof[b] = np.vstack([pp(broaden(d["Xtr"][i:i + 1], tr_sizes[i])) for i in b])
            print("bandgap fold", k, flush=True)
        pp_full, _ = train_cnn(d["Xtr"], d["ctr"], "aug", seed)
        Xb_tr = broaden(d["Xtr"][str_], None)
        Xa_tr, ya_tr = augment_set(d["Xtr"][str_], ytr, rng)
        models = {
            "RF-Magpie": rf().fit(Mtr, ytr),
            "RF-Magpie+CS(true)": rf().fit(np.hstack([Mtr, onehot(d["ctr"][str_])]), ytr),
            "RF-Magpie+CS(CNN)": rf().fit(np.hstack([Mtr, oof[str_]]), ytr),
            "RF-Peaks(bulk)": rf().fit(peak_features(Xb_tr), ytr),
            "RF-Peaks(aug)": rf().fit(peak_features(Xa_tr), ya_tr),
            "RF-Profile(aug)": rf().fit(binned(Xa_tr), ya_tr),
        }
        for s in SIZES:
            T = broaden(d["Xte"][ste], s); P = peak_features(T); sl = size_label(s)
            preds = {
                "RF-Magpie": models["RF-Magpie"].predict(Mte),
                "RF-Magpie+CS(true)": models["RF-Magpie+CS(true)"].predict(np.hstack([Mte, onehot(d["cte"][ste])])),
                "RF-Magpie+CS(CNN)": models["RF-Magpie+CS(CNN)"].predict(np.hstack([Mte, pp_full(T)])),
                "RF-Peaks(bulk)": models["RF-Peaks(bulk)"].predict(P),
                "RF-Peaks(aug)": models["RF-Peaks(aug)"].predict(P),
                "RF-Profile(aug)": models["RF-Profile(aug)"].predict(binned(T)),
            }
            for name, p in preds.items():
                rows.append(dict(model=name, size=sl, seed=seed, **reg_metrics(yte, p)))
            if seed == 0 and s in (None, 5):
                pd.DataFrame(dict(compound=d["names_te"][ste], y_true=yte, **{k: v for k, v in preds.items()})).to_csv(OUT / f"bandgap_pred_seed0_{sl}.csv", index=False)
        rows.append(dict(model="Mean predictor", size="all", seed=seed, **reg_metrics(yte, np.full_like(yte, ytr.mean()))))
        imp = models["RF-Magpie+CS(CNN)"].feature_importances_
        names = list(pd.read_csv(DATA / "magpie_features.csv", nrows=0).drop(columns=["compound_name", "bandgap"]).columns) + [f"P({c})" for c in SYSTEMS]
        if seed == 0:
            pd.DataFrame(dict(feature=names, importance=imp)).sort_values("importance", ascending=False).to_csv(OUT / "bandgap_importance_seed0.csv", index=False)
        print("bandgap seed", seed, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "bandgap.csv", index=False)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("cs_rf", run_cs_rf), ("cs_cnn", run_cs_cnn), ("profile", run_profile), ("bandgap", run_bandgap)]:
        if stage in (name, "all"):
            t = time.time(); fn(); print(name, "finished in", round(time.time() - t), "s", flush=True)

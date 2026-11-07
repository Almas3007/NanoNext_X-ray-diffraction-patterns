"""Follow-up analyses for the crystal-system CNN.

  curves    seed-0 bulk/aug CNN with per-epoch history, per-sample test predictions and timing
  ablation  size-augmented CNN with one component removed at a time (3 seeds)
  lcurve    size-augmented CNN trained on 25 / 50 / 100 % of the training set (3 seeds)

Usage: python3 extra_cnn.py {curves|ablation|lcurve|all}
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))
from broadening import broaden, broadening_matrix, peak_features
from run_experiments import load, XRDNet, cls_metrics, augment_set, size_label, SIZES, AUG_BANK, SEEDS, SYSTEMS, DEV, OUT


def train(X, y, regime, seed, bank=AUG_BANK, shift=4, smoothing=0.1, class_weights=True, bulk_frac=0.2,
          epochs=150, patience=40):
    """Same procedure as run_experiments.train_cnn, with switches for the ablation and a history log."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); nv = int(0.15 * len(X))
    va, tr = idx[:nv], idx[nv:]
    kernels = {s: torch.from_numpy(broadening_matrix(s).T).to(DEV) for s in bank} if regime == "aug" else {}
    base = broaden(X, None)
    if regime == "aug":
        vsz = rng.choice(list(AUG_BANK) + [None] * 4, len(va))
        Xv = np.vstack([broaden(X[i:i + 1], s) for i, s in zip(va, vsz)])
    else:
        Xv = base[va]
    Xt, yt = torch.from_numpy(base).to(DEV), torch.from_numpy(y.astype(np.int64)).to(DEV)
    Xv = torch.from_numpy(Xv).to(DEV)
    w = None
    if class_weights:
        w = torch.tensor(len(tr) / (len(SYSTEMS) * np.maximum(np.bincount(y[tr], minlength=len(SYSTEMS)), 1)), dtype=torch.float32, device=DEV)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=smoothing)
    model = XRDNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad, hist = 1e9, None, 0, []
    for ep in range(epochs):
        model.train(); perm = tr[rng.permutation(len(tr))]; tl = []
        for b in range(0, len(perm), 64):
            bi = perm[b:b + 64]; xb = Xt[bi]
            if regime == "aug":
                xb = xb.clone(); sz = rng.integers(-1, len(bank), len(bi))
                sz[rng.random(len(bi)) < bulk_frac] = -1
                for k in np.unique(sz[sz >= 0]):
                    m = torch.from_numpy(np.where(sz == k)[0]).to(DEV)
                    xk = xb[m] @ kernels[bank[k]]
                    xb[m] = xk / xk.amax(dim=1, keepdim=True).clamp_min(1e-8)
            if shift:
                xb = torch.roll(xb, int(rng.integers(-shift, shift + 1)), dims=1)
            opt.zero_grad(); loss = loss_fn(model(xb), yt[bi]); loss.backward(); opt.step(); tl.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            out = model(Xv); vloss = loss_fn(out, yt[va]).item(); pv = out.argmax(1).cpu().numpy()
        vf1 = f1_score(y[va], pv, average="macro")
        hist.append(dict(epoch=ep + 1, train_loss=float(np.mean(tl)), val_loss=vloss, val_f1=vf1, val_acc=float((pv == y[va]).mean())))
        if -vf1 < best - 1e-4:
            best, bad = -vf1, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); model.eval()

    def predict_proba(Xn):
        with torch.no_grad():
            return np.vstack([torch.softmax(model(torch.from_numpy(Xn[i:i + 512]).to(DEV)), 1).cpu().numpy() for i in range(0, len(Xn), 512)])
    return predict_proba, hist, model


def run_curves():
    d = load(); test_sets = {s: broaden(d["Xte"], s) for s in SIZES}
    preds, hists, timing = {}, [], []
    for regime in ["bulk", "aug"]:
        t = time.time(); pp, hist, model = train(d["Xtr"], d["ctr"], regime, 0)
        train_s = time.time() - t
        hists += [dict(h, regime=regime) for h in hist]
        for s in SIZES:
            preds[f"CNN_{regime}_{size_label(s)}"] = pp(test_sets[s]).argmax(1)
        # inference latency: batch of 1,072 test patterns and single pattern, after warm-up
        pp(test_sets[None][:64])
        t = time.time(); [pp(test_sets[None]) for _ in range(5)]; batch_ms = (time.time() - t) / 5 * 1000
        t = time.time(); [pp(test_sets[None][i:i + 1]) for i in range(200)]; single_ms = (time.time() - t) / 200 * 1000
        timing.append(dict(model="CNN-XRD", regime=regime, params=sum(p.numel() for p in model.parameters()),
                           train_s=train_s, epochs=len(hist), batch_ms_per_pattern=batch_ms / len(d["Xte"]), single_ms=single_ms))
        print("curves", regime, len(hist), round(train_s), flush=True)
    # matching random-forest predictions (seed 0) for paired tests
    rng = np.random.default_rng(0)
    rf = lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0, class_weight="balanced")
    Xb = broaden(d["Xtr"], None); Xa, ca = augment_set(d["Xtr"], d["ctr"], rng)
    t = time.time(); m_peak_a = rf().fit(peak_features(Xa), ca); rf_train = time.time() - t
    m_peak_b = rf().fit(peak_features(Xb), d["ctr"])
    for s in SIZES:
        P = peak_features(test_sets[s])
        preds[f"RFPeaks_aug_{size_label(s)}"] = m_peak_a.predict(P)
        preds[f"RFPeaks_bulk_{size_label(s)}"] = m_peak_b.predict(P)
    t = time.time(); m_peak_a.predict(peak_features(test_sets[None])); rf_ms = (time.time() - t) * 1000 / len(d["Xte"])
    timing.append(dict(model="RF-Peaks", regime="aug", params=None, train_s=rf_train, epochs=None, batch_ms_per_pattern=rf_ms, single_ms=None))
    pr = pd.DataFrame(preds); pr.insert(0, "y_true", d["cte"]); pr.insert(0, "compound", d["names_te"])
    pr.to_csv(OUT / "cs_predictions_seed0.csv", index=False)
    pd.DataFrame(hists).to_csv(OUT / "cnn_history_seed0.csv", index=False)
    pd.DataFrame(timing).to_csv(OUT / "timing.csv", index=False)


ABLATIONS = {
    "Full (size-aug CNN)": {},
    "No peak-shift jitter": dict(shift=0),
    "No label smoothing": dict(smoothing=0.0),
    "No class weighting": dict(class_weights=False),
    "Sizes 10-100 nm only": dict(bank=AUG_BANK[AUG_BANK >= 10]),
}


def run_ablation():
    d = load(); test_sets = {s: broaden(d["Xte"], s) for s in SIZES}; rows = []
    for name, kw in ABLATIONS.items():
        for seed in SEEDS:
            pp, hist, _ = train(d["Xtr"], d["ctr"], "aug", seed, **kw)
            for s in SIZES:
                rows.append(dict(variant=name, size=size_label(s), seed=seed, **cls_metrics(d["cte"], pp(test_sets[s]).argmax(1))))
            print("ablation", name, seed, len(hist), flush=True)
            pd.DataFrame(rows).to_csv(OUT / "ablation.csv", index=False)


def run_lcurve():
    d = load(); test_sets = {s: broaden(d["Xte"], s) for s in [None, 10, 5, 3]}; rows = []
    for frac in [0.25, 0.5]:
        for seed in SEEDS:
            rng = np.random.default_rng(100 + seed)
            # stratified subsample of the training set
            sub = np.concatenate([rng.choice(np.where(d["ctr"] == c)[0], max(2, int(frac * (d["ctr"] == c).sum())), replace=False) for c in range(len(SYSTEMS))])
            pp, hist, _ = train(d["Xtr"][sub], d["ctr"][sub], "aug", seed)
            for s, T in test_sets.items():
                rows.append(dict(frac=frac, n_train=len(sub), size=size_label(s), seed=seed, **cls_metrics(d["cte"], pp(T).argmax(1))))
            print("lcurve", frac, seed, flush=True)
            pd.DataFrame(rows).to_csv(OUT / "learning_curve.csv", index=False)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("curves", run_curves), ("ablation", run_ablation), ("lcurve", run_lcurve)]:
        if stage in (name, "all"):
            t = time.time(); fn(); print(name, "finished in", round(time.time() - t), "s", flush=True)

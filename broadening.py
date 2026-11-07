"""Crystallite-size (Scherrer) broadening of simulated powder XRD patterns.

Patterns in exp_bandgap_dataset are sampled on 2theta = 5..90 deg (0.05 deg step),
Cu K-alpha1 (1.5406 A), with a narrow instrumental width (~0.1-0.15 deg FWHM).
Size broadening is added by convolving each pattern with a pseudo-Voigt kernel whose
FWHM follows the Scherrer relation  beta(2theta) = K * lambda / (L * cos(theta)).
"""
import numpy as np
from scipy.signal import find_peaks

WAVELENGTH_NM = 0.15406
SCHERRER_K = 0.9
TTH = np.round(np.arange(5.0, 90.0001, 0.05), 2)


def scherrer_fwhm_deg(tth_deg, size_nm, k=SCHERRER_K, lam=WAVELENGTH_NM):
    """FWHM (degrees 2theta) produced by crystallites of mean size size_nm."""
    theta = np.radians(tth_deg) / 2.0
    return np.degrees(k * lam / (size_nm * np.cos(theta)))


def pseudo_voigt(dx, fwhm, eta):
    g = np.sqrt(4 * np.log(2) / np.pi) / fwhm * np.exp(-4 * np.log(2) * (dx / fwhm) ** 2)
    l = (2 / (np.pi * fwhm)) / (1 + 4 * (dx / fwhm) ** 2)
    return eta * l + (1 - eta) * g


def broadening_matrix(size_nm, eta=0.5, tth=TTH):
    """Dense (n, n) operator K so that broadened = pattern @ K.T.

    Column i spreads the intensity at tth[i] with the Scherrer width of that angle;
    columns are normalised so integrated intensity is conserved.
    """
    fwhm = scherrer_fwhm_deg(tth, size_nm)                 # width per source angle
    dx = tth[:, None] - tth[None, :]                        # rows: output, cols: source
    k = pseudo_voigt(dx, fwhm[None, :], eta)
    k /= k.sum(axis=0, keepdims=True)
    return k.astype(np.float32)


def broaden(patterns, size_nm, eta=0.5):
    """Apply size broadening and re-normalise each pattern to unit maximum."""
    if size_nm is None:
        out = patterns.astype(np.float32).copy()
    else:
        out = patterns.astype(np.float32) @ broadening_matrix(size_nm, eta).T
    out /= np.maximum(out.max(axis=1, keepdims=True), 1e-8)
    return out


def peak_features(patterns, n_top=10, prominence=0.02, tth=TTH):
    """Conventional peak-list descriptors re-extracted from (broadened) patterns."""
    step = tth[1] - tth[0]
    feats = []
    for p in patterns:
        idx, prop = find_peaks(p, prominence=prominence, width=1)
        tth_p, inten = tth[idx], p[idx]
        widths = prop["widths"] * step if len(idx) else np.array([])
        d = WAVELENGTH_NM * 10 / (2 * np.sin(np.radians(tth_p) / 2)) if len(idx) else np.array([])
        order = np.argsort(-inten)[:n_top]
        row = [len(idx),
               d.mean() if len(d) else 0, d.std() if len(d) else 0,
               d.min() if len(d) else 0, d.max() if len(d) else 0,
               widths.mean() if len(widths) else 0]
        for j in range(n_top):
            if j < len(order):
                o = order[j]
                row += [tth_p[o], inten[o], d[o], widths[o]]
            else:
                row += [0, 0, 0, 0]
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)

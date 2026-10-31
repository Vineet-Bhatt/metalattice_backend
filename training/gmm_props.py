"""Turn GraphMetaMat data (truss graph + stress-strain curve) into effective lattice properties.

GraphMetaMat stores, per dataset directory:
    <split>/mapping.tsv         tab-separated "gid<TAB>cid" rows   (split = train, dev, test)
    <split>/graphs/<gid>.gpkl   pickled networkx graph, g.graph['rho'] = relative density
    <split>/curves/<cid>.pkl    pickled dict, ['curve'] = (L, 2) array [strain, stress], ['cid']

This module needs no torch / torch-geometric, so it can run anywhere the data is.

UNITS WARNING: the sign, scale and floor (-40) of the stress column in the released data could
not be checked here (the figshare data was not reachable). `stress_to_mpa`, `base_E_mpa` and
`base_sy_mpa` make the conversion explicit. If the curves are already relative to the solid
material, leave all three at 1.0 and E_rel / sy_rel come out directly.
"""
from __future__ import annotations

import csv
import os
import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class TrussProps:
    gid: int
    cid: int
    rho: float
    E_rel: float    # effective modulus / solid modulus
    sy_rel: float   # effective yield strength / solid yield strength


def initial_modulus(strain, stress, fit_strain: float = 0.02, min_pts: int = 4) -> float:
    """Least-squares slope through the origin-anchored initial segment (strain <= fit_strain)."""
    strain = np.asarray(strain, float)
    stress = np.asarray(stress, float)
    m = strain <= fit_strain
    if m.sum() < min_pts:                      # coarse curves: use the first min_pts points
        m = np.zeros_like(m)
        m[:min_pts] = True
    x, y = strain[m], stress[m]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def offset_yield(strain, stress, E: float, offset: float = 0.002) -> float:
    """0.2 %-offset yield: first crossing of stress = E*(strain - offset).

    If the curve never crosses (brittle / still rising), the peak stress is returned.
    """
    strain = np.asarray(strain, float)
    stress = np.asarray(stress, float)
    gap = stress - E * (strain - offset)       # positive while the curve is above the offset line
    idx = np.where(gap <= 0)[0]
    idx = idx[idx > 0]
    if idx.size == 0:
        return float(stress.max())
    i = idx[0]
    g0, g1 = gap[i - 1], gap[i]
    t = g0 / (g0 - g1) if g0 != g1 else 0.0
    return float(stress[i - 1] + t * (stress[i] - stress[i - 1]))


def curve_to_props(strain, stress, rho: float, stress_to_mpa: float = 1.0,
                   base_E_mpa: float = 1.0, base_sy_mpa: float = 1.0,
                   fit_strain: float = 0.02, offset: float = 0.002):
    """Return (E_rel, sy_rel) for one curve, or None if the curve is unusable."""
    s = np.asarray(stress, float) * stress_to_mpa
    e = np.asarray(strain, float)
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(e))) or e.size < 4:
        return None
    E = initial_modulus(e, s, fit_strain)
    if not np.isfinite(E) or E <= 0:
        return None
    sy = offset_yield(e, s, E, offset)
    if sy <= 0:
        return None
    return E / base_E_mpa, sy / base_sy_mpa


def _find(dirpath: str, sub: str, name: str):
    """GraphMetaMat layout puts files in <split>/graphs and <split>/curves; accept flat too."""
    for cand in (os.path.join(dirpath, sub, name), os.path.join(dirpath, name)):
        if os.path.exists(cand):
            return cand
    return None


def _split_dirs(root: str, splits):
    """Folders holding a mapping.tsv: root itself, else root/<split> for each split."""
    if os.path.exists(os.path.join(root, "mapping.tsv")):
        return [root]
    found = [os.path.join(root, s) for s in splits
             if os.path.exists(os.path.join(root, s, "mapping.tsv"))]
    if not found:
        raise FileNotFoundError(
            f"no mapping.tsv in {root} or in its {'/'.join(splits)} subfolders")
    return found


def export_property_table(dataset_dir: str, out_csv: str, splits=("train", "dev", "test"), **kw):
    """Read a GraphMetaMat dataset and write rho / E_rel / sy_rel rows to CSV.

    dataset_dir is either a single split folder or the parent holding train/dev/test, e.g.
    dataset/stress_strain/standard . kw is forwarded to curve_to_props
    (stress_to_mpa, base_E_mpa, base_sy_mpa, fit_strain, offset). Returns (rows, n_skipped).
    """
    rows, skipped, seen = [], 0, set()
    for d in _split_dirs(dataset_dir, splits):
        with open(os.path.join(d, "mapping.tsv")) as fp:
            pairs = [(int(g), int(c)) for g, c in csv.reader(fp, delimiter="\t")]
        for gid, cid in pairs:
            if (d, gid, cid) in seen:
                continue
            seen.add((d, gid, cid))
            gpn = _find(d, "graphs", f"{gid}.gpkl")
            cpn = _find(d, "curves", f"{cid}.pkl")
            if gpn is None or cpn is None:
                skipped += 1
                continue
            try:
                with open(gpn, "rb") as f:
                    rho = float(pickle.load(f).graph["rho"])
                with open(cpn, "rb") as f:
                    curve = np.asarray(pickle.load(f)["curve"], float)
            except (OSError, KeyError, pickle.UnpicklingError):
                skipped += 1
                continue
            res = curve_to_props(curve[:, 0], curve[:, 1], rho, **kw)
            if res is None or rho <= 0:
                skipped += 1
                continue
            rows.append(TrussProps(gid, cid, rho, *res))

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gid", "cid", "rho", "E_rel", "sy_rel"])
        for r in rows:
            w.writerow([r.gid, r.cid, r.rho, r.E_rel, r.sy_rel])
    return rows, skipped


def load_property_table(csv_path: str):
    """Load a table written by export_property_table -> dict of numpy arrays."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return {k: np.atleast_1d(data[k]) for k in data.dtype.names}

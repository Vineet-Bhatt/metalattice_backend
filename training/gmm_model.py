"""Topology-aware surrogate (graph + density -> E_rel, sy_rel) and inverse search.

Run from the project root:
    python -m training.gmm_model fit    <dataset_root> --out data/surrogate.pkl
    python -m training.gmm_model design --model data/surrogate.pkl --E 0.02 --sy 0.0008 -k 5
E and sy are in the same (relative) units as data/gmm_props.csv.
"""
import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from .gmm_props import _find, _split_dirs, curve_to_props

FEATURES = ["n_nodes", "n_edges", "mean_deg", "max_deg", "total_len", "rho"]


def topo_features(g):
    """Cheap, torch-free graph descriptors (NaN where the pickle lacks the attribute)."""
    deg = np.array([d for _, d in g.degree()], float) if g.number_of_nodes() else np.zeros(1)
    lens = [d["length"] for *_, d in g.edges(data=True) if "length" in d]
    return [g.number_of_nodes(), g.number_of_edges(), deg.mean(), deg.max(),
            float(np.sum(lens)) if lens else np.nan]


def build_dataset(dataset_root, splits=("train", "dev", "test"), **curve_kw):
    """Returns X (n, len(FEATURES)), Y (n, 2) = [E_rel, sy_rel], gids (n,)."""
    X, Y, G = [], [], []
    for d in _split_dirs(dataset_root, splits):
        with open(os.path.join(d, "mapping.tsv")) as fp:
            pairs = [(int(a), int(b)) for a, b in csv.reader(fp, delimiter="\t")]
        cache = {}
        for gid, cid in pairs:
            gpn, cpn = _find(d, "graphs", f"{gid}.gpkl"), _find(d, "curves", f"{cid}.pkl")
            if gpn is None or cpn is None:
                continue
            try:
                if gid not in cache:
                    with open(gpn, "rb") as f:
                        g = pickle.load(f)
                    cache[gid] = (float(g.graph["rho"]), topo_features(g))
                with open(cpn, "rb") as f:
                    curve = np.asarray(pickle.load(f)["curve"], float)
            except (OSError, KeyError, pickle.UnpicklingError):
                continue
            rho, tf = cache[gid]
            res = curve_to_props(curve[:, 0], curve[:, 1], rho, **curve_kw)
            if res is None or rho <= 0:
                continue
            X.append(tf + [rho]); Y.append(res); G.append(hash((d, gid)) % (2 ** 31))
    return np.array(X, float), np.array(Y, float), np.array(G)


class Ensemble:
    """Bootstrap ensemble of gradient-boosted trees on log targets (mean + std)."""

    def __init__(self, n_models=5, seed=0):
        self.n_models, self.seed, self.models = n_models, seed, []

    def _clean(self, X):
        X = np.array(X, float)
        X[:, self.dead] = 0.0           # columns the pickles never provided carry no signal
        return X

    def fit(self, X, Y):
        rng = np.random.default_rng(self.seed)
        self.dead = np.isnan(X).all(axis=0)
        X, L = self._clean(X), np.log(Y)
        self.models = []
        for k in range(self.n_models):
            idx = rng.integers(0, len(X), len(X))
            self.models.append([HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                              random_state=k).fit(X[idx], L[idx, j])
                                for j in range(2)])
        return self

    def predict(self, X):
        """-> (mean, std) of shape (n, 2) in log space."""
        X = self._clean(X)
        P = np.stack([[m.predict(X) for m in ms] for ms in self.models])   # (models, 2, n)
        return P.mean(0).T, P.std(0).T


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)


def fit_and_report(dataset_root, out, **curve_kw):
    X, Y, G = build_dataset(dataset_root, **curve_kw)
    if len(X) < 50:
        raise SystemExit(f"only {len(X)} usable rows; check the dataset path / units")
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, Y, G))
    ens = Ensemble().fit(X[tr], Y[tr])
    mu, sd = ens.predict(X[te])
    L = np.log(Y[te])
    print(f"rows {len(X)} | held-out graphs: {len(te)} rows")
    for j, nm in enumerate(("E_rel", "sy_rel")):
        inside = np.abs(L[:, j] - mu[:, j]) <= 2 * sd[:, j]
        print(f"  {nm}: held-out R2 (log) = {r2(L[:, j], mu[:, j]):.3f} | "
              f"MAPE = {100*np.mean(np.abs(np.exp(mu[:, j]) / Y[te][:, j] - 1)):.1f}% | "
              f"2-sigma coverage = {100*inside.mean():.0f}%")
    ens = Ensemble().fit(X, Y)                      # refit on everything for deployment
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(dict(ens=ens, X=X), f)
    print("saved", out)


def design(model_path, E_req, sy_req, k=5, kappa=1.0, rho_grid=None):
    """Lowest-density (graph, rho) whose lower-confidence property bound meets both targets.

    kappa = number of ensemble std-devs of safety margin (0 = trust the mean).
    """
    with open(model_path, "rb") as f:
        blob = pickle.load(f)
    ens, X = blob["ens"], blob["X"]
    topo = np.unique(X[:, :-1], axis=0)             # one row per distinct topology
    rho_grid = np.linspace(0.05, 0.6, 56) if rho_grid is None else np.asarray(rho_grid)
    cand = np.hstack([np.repeat(topo, len(rho_grid), 0), np.tile(rho_grid, len(topo))[:, None]])
    mu, sd = ens.predict(cand)
    ok = ((mu[:, 0] - kappa * sd[:, 0] >= np.log(E_req)) &
          (mu[:, 1] - kappa * sd[:, 1] >= np.log(sy_req)))
    if not ok.any():
        raise SystemExit("no candidate meets both targets in this density range")
    order = np.where(ok)[0][np.argsort(cand[ok, -1])][:k]
    return [dict(rho=float(cand[i, -1]), E_rel=float(np.exp(mu[i, 0])), sy_rel=float(np.exp(mu[i, 1])),
                 E_cv=float(sd[i, 0]), sy_cv=float(sd[i, 1]),
                 **dict(zip(FEATURES[:-1], map(float, cand[i, :-1])))) for i in order]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("dataset_root"); f.add_argument("--out", default="data/surrogate.pkl")
    f.add_argument("--stress-to-mpa", type=float, default=1.0)
    f.add_argument("--base-E", type=float, default=1.0); f.add_argument("--base-sy", type=float, default=1.0)
    d = sub.add_parser("design"); d.add_argument("--model", default="data/surrogate.pkl")
    d.add_argument("--E", type=float, required=True); d.add_argument("--sy", type=float, required=True)
    d.add_argument("-k", type=int, default=5); d.add_argument("--kappa", type=float, default=1.0)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit_and_report(a.dataset_root, a.out, stress_to_mpa=a.stress_to_mpa,
                       base_E_mpa=a.base_E, base_sy_mpa=a.base_sy)
    else:
        for r in design(a.model, a.E, a.sy, a.k, a.kappa):
            print({k: round(v, 5) for k, v in r.items()})


if __name__ == "__main__":
    main()
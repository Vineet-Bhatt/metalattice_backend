import os, pickle, sys, tempfile
import numpy as np, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.gmm_model import build_dataset, Ensemble, fit_and_report, design

def make(root, n=400, seed=0):
    rng = np.random.default_rng(seed)
    for split, lo, hi in (("train", 0, int(n*.8)), ("test", int(n*.8), n)):
        d = os.path.join(root, split); os.makedirs(d + "/graphs"); os.makedirs(d + "/curves")
        with open(d + "/mapping.tsv", "w") as m:
            for i in range(lo, hi):
                nn = int(rng.integers(6, 30)); g = nx.gnm_random_graph(nn, int(nn * rng.uniform(1.2, 3)), seed=i)
                rho = float(rng.uniform(0.05, 0.5)); conn = g.number_of_edges() / nn
                g.graph["rho"] = rho
                E = 0.5 * conn / 3 * rho ** 2; sy = 0.02 * rho ** 1.5          # topology-dependent stiffness
                eps = np.linspace(0, 0.5, 256)
                pickle.dump(g, open(f"{d}/graphs/{i}.gpkl", "wb"))
                pickle.dump({"curve": np.stack([eps, np.minimum(E * eps, sy)], 1), "cid": i}, open(f"{d}/curves/{i}.pkl", "wb"))
                m.write(f"{i}\t{i}\n")

def test_fit_and_design():
    with tempfile.TemporaryDirectory() as r:
        make(r)
        X, Y, G = build_dataset(r)
        assert X.shape == (400, 6) and not np.isnan(X[:, [0, 1, 2, 3, 5]]).any()
        out = os.path.join(r, "s.pkl"); fit_and_report(r, out)
        res = design(out, E_req=0.01, sy_req=0.002, k=3)
        assert len(res) == 3 and all(x["E_rel"] > 0.009 and x["sy_rel"] > 0.0018 for x in res)
        assert res[0]["rho"] <= res[-1]["rho"]
if __name__ == "__main__":
    test_fit_and_design(); print("ok")

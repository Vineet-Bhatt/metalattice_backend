import os, sys, itertools
import numpy as np, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.constraints import (MechSpec, target_curve, check_candidate, strut_report,
                                  zone_requirements, cell_size_window)

def test_target_curve_roundtrip():
    eps, sig = target_curve(0.05, 0.002)
    r = check_candidate(eps, sig, 0.05, 0.002)
    assert r["ok"] and abs(r["E_ratio"] - 1) < 0.02 and abs(r["sy_ratio"] - 1) < 0.05, r
    weak = check_candidate(eps, 0.8 * sig, 0.05, 0.002)
    assert not weak["ok"]

def cubic_cell():
    g = nx.Graph()
    for c in itertools.product((-1, 1), repeat=3):
        g.add_node(c, coord=np.array(c, float))
    for a, b in itertools.combinations(list(g.nodes), 2):
        if sum(x != y for x, y in zip(a, b)) == 1:
            g.add_edge(a, b)
    return g

def test_strut_report_hand_check():
    g = cubic_cell()                                   # 12 struts of length 2
    spec = MechSpec(min_strut_diam_mm=0.5)
    rep = strut_report(g, rho=0.1, E_s=114000, sy_s=880, sigma_macro=10, sy_lattice=20, spec=spec)
    r = np.sqrt(0.1 * 8 / (np.pi * 24))                # r = sqrt(rho L^3/(pi sum len))
    assert abs(rep["r_unit"] - r) < 1e-12
    assert abs(rep["cell_min_mm"] - 0.5 * 2 / (2 * r)) < 1e-9      # diam = 2 r cell / 2
    lam = 0.7 * 2 / (r / 2)
    assert abs(rep["max_slenderness"] - lam) < 1e-9
    strut = 880 * 0.5
    assert abs(rep["buckling_margin"] - (np.pi**2 * 114000 / lam**2) / strut) < 1e-6

def test_zones_cover_all_elements_once():
    rng = np.random.default_rng(0); vm = rng.gamma(2, 20, 500); is_lat = rng.random(500) < 0.7
    z = zone_requirements(None, vm, is_lat, MechSpec(sf=2, knockdown=0.5), n_zones=4)
    cat = np.concatenate([q["elements"] for q in z])
    assert len(cat) == len(set(cat)) == is_lat.sum()
    assert all(a["required_stress"] <= b["required_stress"] for a, b in zip(z, z[1:]))
    assert abs(z[0]["required_stress"] - 2 * z[0]["vm_pct"] / 0.5) < 1e-9

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)

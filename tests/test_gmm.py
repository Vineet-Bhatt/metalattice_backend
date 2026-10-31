import os
import pickle
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.gmm_props import (curve_to_props, export_property_table,  # noqa: E402
                                initial_modulus, load_property_table, offset_yield)
from training.gmm_surrogate import fit_lattice_model, rho_for_strength  # noqa: E402

C_E, N_E, C_SY, N_SY = 0.8, 2.0, 0.03, 1.5   # yield strain ~ 0.04-0.17 over rho 0.05-0.5


def _curve(rho, rng, noise=0.0):
    """Elastic-perfectly-plastic curve with known E and yield for a given density."""
    E = C_E * rho ** N_E * (1 + noise * rng.standard_normal())
    sy = C_SY * rho ** N_SY * (1 + noise * rng.standard_normal())
    eps = np.linspace(0, 0.5, 512)
    return np.stack([eps, np.minimum(E * eps, sy)], axis=1), E, sy


def _make_dataset(d, n=60, noise=0.0, seed=0):
    import networkx as nx
    rng = np.random.default_rng(seed)
    with open(os.path.join(d, "mapping.tsv"), "w") as m:
        for i in range(n):
            rho = float(rng.uniform(0.05, 0.5))
            g = nx.Graph()
            g.graph["rho"] = rho
            curve, _, _ = _curve(rho, rng, noise)
            with open(os.path.join(d, f"{i}.gpkl"), "wb") as f:
                pickle.dump(g, f)
            with open(os.path.join(d, f"{i}.pkl"), "wb") as f:
                pickle.dump({"curve": curve, "cid": i}, f)
            m.write(f"{i}\t{i}\n")


def test_curve_extraction_exact():
    rng = np.random.default_rng(0)
    curve, E, sy = _curve(0.3, rng)
    assert abs(initial_modulus(curve[:, 0], curve[:, 1]) - E) / E < 1e-6
    got = offset_yield(curve[:, 0], curve[:, 1], E)
    assert abs(got - sy) / sy < 0.02          # 0.2% offset sits just under the plateau
    E_rel, sy_rel = curve_to_props(curve[:, 0], curve[:, 1], 0.3)
    assert abs(E_rel - E) / E < 1e-6


def test_unusable_curves_rejected():
    assert curve_to_props([0, 1, 2], [0, 1, 2], 0.1) is None                    # too short
    assert curve_to_props(np.linspace(0, 1, 10), -np.linspace(0, 1, 10), 0.1) is None  # negative E


def test_export_and_fit_recovers_exponents():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, n=80, noise=0.02)
        out = os.path.join(d, "props.csv")
        rows, skipped = export_property_table(d, out)
        assert len(rows) == 80 and skipped == 0
        lat, diag = fit_lattice_model(load_property_table(out))
        assert abs(lat.n_E - N_E) < 0.15 and abs(lat.n_sy - N_SY) < 0.15, (lat, diag)
        assert abs(lat.c_E - C_E) / C_E < 0.15 and abs(lat.c_sy - C_SY) / C_SY < 0.15
        assert diag["r2_E"] > 0.98 and diag["r2_sy"] > 0.98
        assert diag["frac_E_above_voigt"] == 0.0


def test_export_split_layout():
    """Real layout: root/<split>/mapping.tsv with graphs/ and curves/ subfolders."""
    with tempfile.TemporaryDirectory() as root:
        for k, split in enumerate(("train", "dev", "test")):
            d = os.path.join(root, split)
            os.makedirs(os.path.join(d, "graphs")), os.makedirs(os.path.join(d, "curves"))
            flat = tempfile.mkdtemp()
            _make_dataset(flat, n=10, seed=k)
            for i in range(10):
                os.rename(os.path.join(flat, f"{i}.gpkl"), os.path.join(d, "graphs", f"{i}.gpkl"))
                os.rename(os.path.join(flat, f"{i}.pkl"), os.path.join(d, "curves", f"{i}.pkl"))
            os.rename(os.path.join(flat, "mapping.tsv"), os.path.join(d, "mapping.tsv"))
        rows, skipped = export_property_table(root, os.path.join(root, "p.csv"))
        assert len(rows) == 30 and skipped == 0
        try:
            export_property_table(os.path.join(root, "nope"), os.path.join(root, "q.csv"))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


def test_export_skips_bad_files():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, n=10)
        os.remove(os.path.join(d, "3.pkl"))
        rows, skipped = export_property_table(d, os.path.join(d, "p.csv"))
        assert len(rows) == 9 and skipped == 1


def test_fit_needs_enough_rows():
    try:
        fit_lattice_model({"rho": [0.1, 0.2], "E_rel": [0.01, 0.04], "sy_rel": [0.01, 0.02]})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_rho_for_strength_inverts_law():
    from training.config import LatticeModel
    lat = LatticeModel(c_sy=C_SY, n_sy=N_SY)
    rho = 0.25
    stress = C_SY * 880.0 * rho ** N_SY
    assert abs(rho_for_strength(lat, 880.0, stress) - rho) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

"""Graded lattice density driven by the FEA stress field (verification loop, stage 6).

Each lattice element gets its own relative density so that
    sigma_y(rho_i) >= SF_target * vm_i
then FEA is re-run and the densities are relaxed until the safety-factor target is met.
Densities are clipped to [rho_min, rho_max] (rho_min is the manufacturability floor) and
optionally smoothed over neighbouring elements to avoid abrupt cell-size jumps.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from .config import Case, Material, LatticeModel
from .fea import solve_static
from .gmm_surrogate import rho_for_strength
from .lattice import assign_materials, element_volumes


def _smooth(cen, rho, is_lat, k=8):
    idx = np.where(is_lat)[0]
    tree = KDTree(cen[idx])
    _, nb = tree.query(cen[idx], k=min(k, idx.size))
    out = rho.copy()
    out[idx] = rho[idx][nb].mean(axis=1)
    return out


def graded_design(mesh, mat: Material, lat: LatticeModel, case: Case, sf_target: float,
                  rho_min: float = 0.10, rho_max: float = 0.60, iters: int = 6,
                  relax: float = 0.6, smooth_k: int = 8, tol: float = 0.02):
    """Iterate FEA <-> per-element density. Returns dict with the final rho field and metrics.

    Result keys: rho (per-element, 1.0 in solid regions), is_lat, history (list of dicts),
    converged (bool: every lattice element reached sf_target within tol), mass_g, res (last FEA).
    """
    vol = element_volumes(mesh)
    cen = mesh.p[:, mesh.t].mean(axis=1).T
    _, _, _, _, is_lat = assign_materials(mesh, mat, lat, case)
    if not is_lat.any():
        raise RuntimeError("no lattice region: shell/boss thickness leaves no interior elements")

    rho_field = np.where(is_lat, lat.rel_density, 1.0)
    history, converged = [], False
    for it in range(iters):
        rho_solved = rho_field.copy()          # the field this iteration's FEA actually used
        E, nu, sy, rho, _ = assign_materials(mesh, mat, lat, case, rho_field=rho_field)
        res = solve_static(mesh, E, nu, case)
        sf = sy / np.maximum(res["vm"], 1e-12)
        sf_min_lat = float(np.percentile(sf[is_lat], 1))      # 1st pct: ignores point-load spikes
        history.append(dict(iter=it, mass_g=float((vol * rho).sum() * mat.density),
                            sf_p1_lattice=sf_min_lat,
                            rho_mean_lattice=float(rho_field[is_lat].mean())))
        if sf_min_lat >= sf_target * (1 - tol) and it > 0:
            converged = True
            break
        # density that would exactly hit the target SF at the current stress
        want = rho_for_strength(lat, mat.yield_strength, sf_target * res["vm"])
        new = np.clip(want, rho_min, rho_max)
        rho_field = np.where(is_lat, (1 - relax) * rho_field + relax * new, 1.0)
        if smooth_k:
            rho_field = np.where(is_lat, _smooth(cen, rho_field, is_lat, smooth_k), 1.0)
    return dict(rho=rho_solved, is_lat=is_lat, history=history, converged=converged,
                mass_g=history[-1]["mass_g"], res=res)


def _filter_matrix(cen_lat, radius):
    """Row-normalised linear-hat neighbour filter among lattice elements (sparse)."""
    import scipy.sparse as sp
    tree = KDTree(cen_lat)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    n = len(cen_lat)
    d = np.linalg.norm(cen_lat[pairs[:, 0]] - cen_lat[pairs[:, 1]], axis=1)
    w = 1.0 - d / radius
    H = sp.coo_matrix((np.r_[w, w, np.ones(n)], (np.r_[pairs[:, 0], pairs[:, 1], np.arange(n)],
                                                np.r_[pairs[:, 1], pairs[:, 0], np.arange(n)])),
                      shape=(n, n)).tocsr()
    return sp.diags(1.0 / np.asarray(H.sum(axis=1)).ravel()) @ H


def oc_design(mesh, mat: Material, lat: LatticeModel, case: Case, mass_frac: float,
              rho_min: float = 0.12, rho_max: float = 0.90, iters: int = 40, move: float = 0.1,
              eta: float = 0.5, radius: float | None = None, tol: float = 2e-3):
    """Minimum-compliance grading of the lattice region at a fixed lattice mass fraction.

    Optimality-criteria update with the lattice law E = c_E*E_s*rho^n_E, so the element
    sensitivity is n_E * U_e / rho_e. `mass_frac` is the mean relative density of the lattice
    region (e.g. 0.25). Densities stay inside [rho_min, rho_max]; rho_min is the
    manufacturability floor. Returns dict(rho, is_lat, history, res).
    """
    vol = element_volumes(mesh)
    cen = mesh.p[:, mesh.t].mean(axis=1).T
    _, _, _, _, is_lat = assign_materials(mesh, mat, lat, case)
    idx = np.where(is_lat)[0]
    if idx.size == 0:
        raise RuntimeError("no lattice region: shell/boss thickness leaves no interior elements")
    if not (rho_min <= mass_frac <= rho_max):
        raise ValueError(f"mass_frac {mass_frac} must lie in [{rho_min}, {rho_max}]")
    V = vol[idx]
    target = mass_frac * V.sum()
    H = _filter_matrix(cen[idx], radius or 2.0 * case.mesh_size)

    rho = np.ones(mesh.t.shape[1])
    rho[idx] = mass_frac
    history = []
    for it in range(iters):
        E, nu, sy, _, _ = assign_materials(mesh, mat, lat, case, rho_field=rho)
        res = solve_static(mesh, E, nu, case)
        r = rho[idx]
        B = lat.n_E * (H @ (res["energy"][idx] / V)) / np.maximum(r, 1e-9)
        lo, hi = 1e-30, 1e30
        for _ in range(200):                                  # bisection on the volume multiplier
            lam = np.sqrt(lo * hi)
            new = np.clip(r * (B / lam) ** eta, np.maximum(r - move, rho_min), np.minimum(r + move, rho_max))
            if (new * V).sum() > target:
                lo = lam
            else:
                hi = lam
            if hi / lo < 1 + 1e-6:
                break
        change = float(np.abs(new - r).max())
        history.append(dict(iter=it, compliance=res["compliance"], max_change=change,
                            lattice_mass_frac=float((r * V).sum() / V.sum())))
        rho[idx] = new
        if change < tol:
            break
    E, nu, sy, _, _ = assign_materials(mesh, mat, lat, case, rho_field=rho)
    res = solve_static(mesh, E, nu, case)
    return dict(rho=rho, is_lat=is_lat, history=history, res=res, sy=sy)

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

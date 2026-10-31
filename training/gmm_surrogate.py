"""Fit the lattice property law from GraphMetaMat-derived data and use it for graded density.

The fitted constants drop straight into config.LatticeModel, which lattice.assign_materials
already consumes:  E = c_E * E_s * rho^n_E ,  sigma_y = c_sy * sy_s * rho^n_sy .
"""
from __future__ import annotations

import numpy as np

from .config import LatticeModel


def _loglog_fit(rho, y):
    """OLS of log y = log c + n log rho. Returns (c, n, r2)."""
    x, z = np.log(rho), np.log(y)
    A = np.vstack([x, np.ones_like(x)]).T
    (n, logc), *_ = np.linalg.lstsq(A, z, rcond=None)
    pred = A @ np.array([n, logc])
    ss_res = float(((z - pred) ** 2).sum())
    ss_tot = float(((z - z.mean()) ** 2).sum()) or 1.0
    return float(np.exp(logc)), float(n), 1.0 - ss_res / ss_tot


def fit_lattice_model(table: dict, name: str = "GraphMetaMat", rho_range=None,
                      nu: float = 0.33, rel_density: float = 0.30):
    """Fit c_E, n_E, c_sy, n_sy from a property table (see gmm_props.load_property_table).

    rho_range=(lo, hi) restricts the fit to the density band you intend to use, which matters
    because the exponent drifts with density (bending- vs stretching-dominated regimes).
    Returns (LatticeModel, diagnostics dict).
    """
    rho = np.asarray(table["rho"], float)
    E = np.asarray(table["E_rel"], float)
    sy = np.asarray(table["sy_rel"], float)
    keep = (rho > 0) & (E > 0) & (sy > 0)
    if rho_range is not None:
        keep &= (rho >= rho_range[0]) & (rho <= rho_range[1])
    if keep.sum() < 5:
        raise ValueError(f"only {int(keep.sum())} usable rows after filtering; need at least 5")
    c_E, n_E, r2_E = _loglog_fit(rho[keep], E[keep])
    c_sy, n_sy, r2_sy = _loglog_fit(rho[keep], sy[keep])
    lat = LatticeModel(name=name, rel_density=rel_density, c_E=c_E, n_E=n_E,
                       c_sy=c_sy, n_sy=n_sy, nu=nu)
    # Unit sanity: no lattice can be stiffer than the Voigt bound E_rel <= rho. A sizeable
    # fraction above it means the curves are not normalised by the solid modulus.
    frac_voigt = float((E[keep] > rho[keep] * 1.05).mean())
    diag = dict(n_rows=int(keep.sum()), r2_E=r2_E, r2_sy=r2_sy,
                frac_E_above_voigt=frac_voigt,
                median_yield_strain=float(np.median(sy[keep] / E[keep])),
                rho_min=float(rho[keep].min()), rho_max=float(rho[keep].max()))
    return lat, diag


def rho_for_strength(lat: LatticeModel, sy_solid: float, required_stress):
    """Smallest relative density whose yield strength covers `required_stress` (MPa)."""
    req = np.maximum(np.asarray(required_stress, float), 1e-12)
    return (req / (lat.c_sy * sy_solid)) ** (1.0 / lat.n_sy)


def rho_for_stiffness(lat: LatticeModel, E_solid: float, required_E):
    """Smallest relative density whose modulus reaches `required_E` (MPa)."""
    req = np.maximum(np.asarray(required_E, float), 1e-12)
    return (req / (lat.c_E * E_solid)) ** (1.0 / lat.n_E)

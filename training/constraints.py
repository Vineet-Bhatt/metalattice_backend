"""Mechanical + manufacturing constraints for the lattice stage.

Flow:  FEA stress -> zones -> per-zone target curve -> GraphMetaMat inverse model
       -> predicted curve/graph -> check_candidate / strut_report -> FEA verify_design.

Stiffness is NOT a per-zone constraint: lattice E_rel <= rho, so no lattice zone can match solid
stiffness locally. It is a global constraint, handled by the density distribution (oc_design)
and checked in verify_design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Case, LatticeModel, Material
from .fea import solve_static
from .gmm_props import curve_to_props
from .lattice import assign_materials, element_volumes


@dataclass
class MechSpec:
    sf: float = 2.0                  # required safety factor on yield
    knockdown: float = 0.7           # AM defect knockdown applied to lattice strength (0-1)
    k_ratio_min: float = 0.6         # global stiffness >= k_ratio_min * solid baseline
    max_disp_mm: float | None = None
    rho_min: float = 0.10
    rho_max: float = 0.60
    min_strut_diam_mm: float = 0.5   # process limit (powder bed fusion ~0.4-0.5)
    min_cells: int = 3               # unit cells across the thinnest lattice dimension
    buckling_K: float = 0.7          # effective length factor for rigidly jointed struts
    buckling_margin: float = 1.5     # sigma_cr / strut stress must exceed this


# ---------------------------------------------------------------- zones -> target curves
def zone_requirements(mesh, vm, is_lat, spec: MechSpec, n_zones: int = 4, pct: float = 95.0):
    """Split lattice elements into n_zones by von Mises quantile; return the strength each needs.

    required_stress (MPa) = sf * pct-percentile stress / knockdown. Using a high percentile
    rather than the max avoids designing every zone around a point-load spike.
    """
    idx = np.where(is_lat)[0]
    edges = np.quantile(vm[idx], np.linspace(0, 1, n_zones + 1))
    zones = []
    for z in range(n_zones):
        hi = edges[z + 1] if z < n_zones - 1 else np.inf
        sel = idx[(vm[idx] >= edges[z]) & (vm[idx] < hi)]
        if sel.size == 0:
            continue
        zones.append(dict(zone=z, elements=sel,
                          vm_pct=float(np.percentile(vm[sel], pct)),
                          required_stress=float(spec.sf * np.percentile(vm[sel], pct) / spec.knockdown)))
    return zones


def zone_density_field(zones, nel, lat: LatticeModel, mat: Material, spec: MechSpec):
    """Per-element density that just meets each zone's strength requirement.

    required rho = (required_stress / (c_sy * sy_s))^(1/n_sy). A zone whose requirement exceeds
    what the densest allowed lattice (rho_max) can give is returned as SOLID (rho = 1): that is
    the mechanical constraint turned into a keep-out region.
    Returns (rho_field, zones) with each zone annotated by rho and must_be_solid.
    """
    rho = np.ones(nel)
    for z in zones:
        need = (z["required_stress"] / (lat.c_sy * mat.yield_strength)) ** (1.0 / lat.n_sy)
        z["rho_needed"] = float(need)
        z["must_be_solid"] = bool(need > spec.rho_max)
        z["rho"] = 1.0 if z["must_be_solid"] else float(max(need, spec.rho_min))
        rho[z["elements"]] = z["rho"]
    return rho, zones


def target_curve(E_rel, sy_rel, n=256, eps_max=0.2966, hardening=0.02):
    """Elastic -> yield -> mild-hardening target in the dataset's curve format.

    Returns (strain, stress). Units follow the table (stress relative to the solid), so pass
    E_rel / sy_rel in the same units as data/gmm_props.csv. Default grid matches the dataset
    (256 points up to strain 0.2966).
    """
    eps = np.linspace(0.0, eps_max, n)
    ey = sy_rel / E_rel
    sig = np.where(eps <= ey, E_rel * eps, sy_rel + hardening * E_rel * (eps - ey))
    return eps, sig


# ---------------------------------------------------------------- candidate checks
def check_candidate(strain, stress_pred, E_rel_req, sy_rel_req, tol=1e-9):
    """Does the model's predicted curve meet the zone's stiffness and strength targets?"""
    res = curve_to_props(strain, stress_pred, rho=1.0)
    if res is None:
        return dict(ok=False, reason="unusable predicted curve")
    E, sy = res
    return dict(ok=bool(E >= E_rel_req * (1 - tol) and sy >= sy_rel_req * (1 - tol)),
                E_rel=E, sy_rel=sy, E_ratio=E / E_rel_req, sy_ratio=sy / sy_rel_req)


def strut_report(g, rho, E_s, sy_s, sigma_macro, sy_lattice, spec: MechSpec, l_scale=2.0):
    """Geometry checks on a generated truss graph (coords assumed in [-1, 1], cell edge = 2).

    Strut radius from density (same formula GraphMetaMat uses): r = sqrt(rho*L^3/(pi*sum(len))).
    Returns the smallest unit-cell edge (mm) that satisfies the minimum strut diameter and a
    buckling margin: struts are assumed to carry sy_s when the macro stress equals the lattice
    yield, so strut stress = sy_s * sigma_macro / sy_lattice (conservative axial estimate).
    """
    lens = np.array([np.linalg.norm(np.asarray(g.nodes[u]["coord"], float)
                                    - np.asarray(g.nodes[v]["coord"], float)) for u, v in g.edges()])
    r_unit = float(np.sqrt(rho * l_scale ** 3 / (np.pi * lens.sum())))
    cell_min_mm = spec.min_strut_diam_mm * l_scale / (2.0 * r_unit)                       # diam = 2*r_unit*cell/l_scale
    slender = spec.buckling_K * lens / (r_unit / 2.0)                    # K*L / radius_of_gyration
    sigma_cr = np.pi ** 2 * E_s / slender ** 2
    strut_stress = sy_s * min(sigma_macro / sy_lattice, 1.0)
    margin = float(sigma_cr.min() / max(strut_stress, 1e-12))
    return dict(r_unit=r_unit, cell_min_mm=float(cell_min_mm), n_struts=int(len(lens)),
                max_slenderness=float(slender.max()), buckling_margin=margin,
                buckling_ok=bool(margin >= spec.buckling_margin))


def cell_size_window(mesh, is_lat, cell_min_mm, spec: MechSpec):
    """Allowed unit-cell edge: >= process minimum, <= thinnest lattice extent / min_cells."""
    cen = mesh.p[:, mesh.t].mean(axis=1).T[is_lat]
    thin = float(np.min(cen.max(axis=0) - cen.min(axis=0)))
    cell_max = thin / spec.min_cells
    return dict(cell_min_mm=cell_min_mm, cell_max_mm=cell_max, feasible=bool(cell_min_mm <= cell_max))


# ---------------------------------------------------------------- final FEA verification
def verify_design(mesh, mat: Material, lat: LatticeModel, case: Case, rho_field,
                  base: dict, spec: MechSpec):
    """Re-solve with the final density field and check every mechanical constraint.

    `base` is the solid-part result from main.evaluate (needs mass_g, stiffness_N_per_mm).
    Returns (checks, ok) where checks maps name -> dict(value, limit, ok).
    """
    vol = element_volumes(mesh)
    E, nu, sy, rho, is_lat = assign_materials(mesh, mat, lat, case, rho_field=rho_field)
    res = solve_static(mesh, E, nu, case)
    sf = (sy * np.where(is_lat, spec.knockdown, 1.0)) / np.maximum(res["vm"], 1e-12)
    k = np.linalg.norm(case.force) / res["max_disp"]
    pure = is_lat & (rho < 0.999)                    # forced-solid elements are not "lattice"
    checks = {
        "safety_factor_lattice_p1": dict(value=float(np.percentile(sf[pure], 1)), limit=spec.sf,
                                          ok=bool(np.percentile(sf[pure], 1) >= spec.sf)),
        "safety_factor_solid_min": dict(value=float(sf[~is_lat].min()), limit=spec.sf,
                                         ok=bool(sf[~is_lat].min() >= spec.sf)),
        "stiffness_ratio": dict(value=float(k / base["stiffness_N_per_mm"]), limit=spec.k_ratio_min,
                                ok=bool(k / base["stiffness_N_per_mm"] >= spec.k_ratio_min)),
        "rho_band": dict(value=(float(rho[pure].min()), float(rho[pure].max())) if pure.any() else None,
                         limit=(spec.rho_min, spec.rho_max),
                         ok=bool(not pure.any() or (rho[pure].min() >= spec.rho_min - 1e-9
                                                    and rho[pure].max() <= spec.rho_max + 1e-9))),
        "solid_fraction_of_lattice_region": dict(value=float(vol[is_lat & ~pure].sum() / vol[is_lat].sum()),
                                                 limit=None, ok=True),
        "mass_g": dict(value=float((vol * rho).sum() * mat.density), limit=base["mass_g"], ok=True),
    }
    if spec.max_disp_mm is not None:
        checks["max_disp_mm"] = dict(value=res["max_disp"], limit=spec.max_disp_mm,
                                     ok=bool(res["max_disp"] <= spec.max_disp_mm))
    return checks, all(c["ok"] for c in checks.values())

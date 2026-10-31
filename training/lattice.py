import numpy as np
from scipy.spatial import KDTree

from .config import Case, Material, LatticeModel


def element_volumes(mesh):
    p, t = mesh.p, mesh.t
    a, b, c, d = (p[:, t[i]] for i in range(4))
    return np.abs(np.einsum("ij,ij->j", np.cross(b - a, c - a, axis=0), d - a)) / 6.0


def _surface_points(mesh, n=6):
    """Dense point sampling of the boundary triangles (for distance-to-surface)."""
    bf = mesh.boundary_facets()
    tri = mesh.p[:, mesh.facets[:, bf]]                # (3,3,nb)
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            l1, l2 = i / n, j / n
            l0 = 1 - l1 - l2
            pts.append(l0 * tri[:, 0] + l1 * tri[:, 1] + l2 * tri[:, 2])
    return np.hstack(pts).T


def classify(mesh, case: Case):
    cen = mesh.p[:, mesh.t].mean(axis=1).T             # (nelems,3)
    d_surf, _ = KDTree(_surface_points(mesh)).query(cen)
    lattice = d_surf > case.shell_thickness
    for h in case.holes:
        rr = np.hypot(cen[:, 1] - h.y, cen[:, 2] - h.z)
        lattice &= ~(rr < h.r + case.boss_thickness)
    return lattice


def assign_materials(mesh, mat: Material, lat: LatticeModel | None, case: Case,
                     rho_field=None):
    """rho_field: optional per-element relative density (length nelems) for graded lattices.
    Only entries inside the lattice region are used; solid regions stay at density 1."""
    nel = mesh.t.shape[1]
    E = np.full(nel, mat.E)
    nu = np.full(nel, mat.nu)
    sy = np.full(nel, mat.yield_strength)
    rho = np.ones(nel)
    is_lat = np.zeros(nel, dtype=bool)
    if lat is not None:
        is_lat = classify(mesh, case)
        if rho_field is None:
            r = np.full(nel, lat.rel_density)
        else:
            r = np.asarray(rho_field, float)
            if r.shape != (nel,):
                raise ValueError(f"rho_field must have shape ({nel},), got {r.shape}")
        E[is_lat] = lat.c_E * mat.E * r[is_lat] ** lat.n_E
        nu[is_lat] = lat.nu
        sy[is_lat] = lat.c_sy * mat.yield_strength * r[is_lat] ** lat.n_sy
        rho[is_lat] = r[is_lat]
    return E, nu, sy, rho, is_lat

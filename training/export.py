"""Turn a per-element density field into a printable graded-lattice STL / 3MF.

Geometry = periodic BCC strut lattice (8 struts from the cell centre to its corners) whose strut
radius follows the local density, clipped to the original part. Everything is an implicit
function sampled on a voxel grid and meshed with marching cubes:

    f(x) = min( r(x) - dist_to_struts(x),  signed_distance_to_part_surface(x) )

Elements with density >= 0.97 (skin, bosses, forced-solid zones) are fully solid.
Strut radius:  r = cell * sqrt(rho / (pi * sum_len/cell))  with sum_len = 4*sqrt(3)*cell  (BCC).
"""
from __future__ import annotations

import itertools
import os

import numpy as np
from scipy.spatial import KDTree

from .lattice import _surface_points

BCC_SUM_LEN = 4.0 * np.sqrt(3.0)                    # total strut length per unit cell edge


def strut_radius(rho, cell):
    """BCC strut radius for relative density rho (ignores node overlap, which makes the printed
    density slightly lower than rho)."""
    return cell * np.sqrt(np.asarray(rho, float) / (np.pi * BCC_SUM_LEN))


def min_cell_for_strut(min_diam_mm, rho_min):
    """Smallest cell edge that keeps the thinnest strut (at rho_min) above min_diam_mm."""
    return min_diam_mm / (2.0 * np.sqrt(rho_min / (np.pi * BCC_SUM_LEN)))


def surface_samples(mesh, n=6):
    """Boundary sample points and their OUTWARD normals (for an inside/outside sign)."""
    bf = mesh.boundary_facets()
    tri = mesh.p[:, mesh.facets[:, bf]]                                   # (3, 3, nb)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], axis=0).T  # (nb, 3)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    fc = tri.mean(axis=1).T
    tc = mesh.p[:, mesh.t[:, mesh.f2t[0, bf]]].mean(axis=1).T             # adjacent tet centroid
    nrm *= np.where(((fc - tc) * nrm).sum(1) > 0, 1.0, -1.0)[:, None]     # orient outward
    pts, nn = [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            l1, l2 = i / n, j / n
            pts.append((1 - l1 - l2) * tri[:, 0] + l1 * tri[:, 1] + l2 * tri[:, 2])
            nn.append(nrm)
    return np.hstack(pts).T, np.vstack(nn)


def _bcc_distance(pts, cell, chunk=200_000):
    """Distance from points to the periodic BCC strut network."""
    c = np.full(3, 0.5 * cell, np.float32)
    corners = np.array(list(itertools.product((0.0, 1.0), repeat=3)), np.float32) * np.float32(cell)
    out = np.empty(len(pts), np.float32)
    for s in range(0, len(pts), chunk):
        q = np.mod(pts[s:s + chunk].astype(np.float32), np.float32(cell))
        best = np.full(len(q), np.inf, np.float32)
        for k in corners:
            e = k - c
            t = np.clip(((q - c) @ e) / (e @ e), 0.0, 1.0)
            d = np.linalg.norm(q - (c + t[:, None] * e), axis=1)
            np.minimum(best, d, out=best)
        out[s:s + chunk] = best
    return out


def _tet_volume(mesh):
    from .lattice import element_volumes
    return element_volumes(mesh).sum()


def export_lattice(mesh, rho_el, cell_mm, out_path, voxels_per_cell=10, solid_above=0.97,
                   density_g_mm3=None):
    """Write the graded lattice to out_path (.stl or .3mf). Returns a report dict.

    rho_el: per-element relative density (1.0 = solid), length mesh.t.shape[1].
    """
    import trimesh
    from skimage.measure import marching_cubes

    vox = cell_mm / voxels_per_cell
    # offset the grid by an irrational fraction of a voxel so no sample sits exactly on a flat CAD face
    lo, hi = mesh.p.min(axis=1) - 2.371 * vox, mesh.p.max(axis=1) + 2 * vox
    axes = [np.arange(lo[i], hi[i] + vox, vox, dtype=np.float32) for i in range(3)]
    shape = tuple(len(a) for a in axes)
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)

    # signed distance to the part surface (positive inside): nearest boundary sample + outward normal
    sp, sn = surface_samples(mesh)
    d_surf, j = KDTree(sp).query(pts)
    inside = ((pts - sp[j]) * sn[j]).sum(axis=1) < 0
    vol_grid, vol_mesh = inside.sum() * vox ** 3, float(_tet_volume(mesh))
    if abs(vol_grid - vol_mesh) / vol_mesh > 0.03:
        raise RuntimeError(f"voxel/mesh volume mismatch ({vol_grid:.0f} vs {vol_mesh:.0f} mm^3): "
                           "inside test failed or voxels are too coarse; lower cell_mm or raise voxels_per_cell")
    sd = np.where(inside, d_surf, -d_surf).astype(np.float32)
    _, ce = KDTree(mesh.p[:, mesh.t].mean(axis=1).T).query(pts)           # density of the nearest tet
    rho = np.where(inside, np.asarray(rho_el)[ce], 0.0)

    f_lat = strut_radius(np.clip(rho, 1e-6, None), cell_mm).astype(np.float32) - _bcc_distance(pts, cell_mm)
    f_lat = np.where(rho >= solid_above, np.float32(cell_mm), f_lat)
    f = np.minimum(f_lat, sd).reshape(shape)

    verts, faces, _, _ = marching_cubes(f, level=0.0, spacing=(vox, vox, vox))
    verts += lo.astype(np.float32)
    m = trimesh.Trimesh(verts, faces, process=True)
    trimesh.repair.fix_normals(m)
    parts = m.split(only_watertight=False)          # drop floating specks from the voxel grid
    if len(parts) > 1:
        vols = [abs(p.volume) for p in parts]
        keep = [p for p, v in zip(parts, vols) if v > 0.01 * max(vols)]
        m = trimesh.util.concatenate(keep)
    m.export(out_path)
    rep = dict(path=out_path, faces=int(len(m.faces)), watertight=bool(m.is_watertight),
               volume_mm3=float(abs(m.volume)), voxel_mm=float(vox),
               n_components=int(len(parts)))
    if density_g_mm3 is not None:
        rep["mass_g"] = rep["volume_mm3"] * density_g_mm3
    return rep

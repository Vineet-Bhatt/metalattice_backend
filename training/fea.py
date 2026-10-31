import numpy as np
from scipy.sparse.linalg import splu
from skfem import (Basis, FacetBasis, ElementTetP1, ElementVector, BilinearForm,
                   LinearForm, asm, condense)
from skfem.helpers import sym_grad, ddot, trace, eye

from .config import Case


def _fixed_nodes(mesh, case: Case, tol: float = 0.25):
    p = mesh.p
    sel = np.zeros(p.shape[1], dtype=bool)
    for h in case.holes:
        rr = np.hypot(p[1] - h.y, p[2] - h.z)
        sel |= (rr <= h.r + tol) & (p[0] >= h.x_min) & (p[0] <= h.x_max)
    return np.where(sel)[0]


def solve_static(mesh, E_e, nu_e, case: Case):
    elem = ElementVector(ElementTetP1())
    basis = Basis(mesh, elem)
    nq = basis.X.shape[1]
    lam_e = E_e * nu_e / ((1 + nu_e) * (1 - 2 * nu_e))
    mu_e = E_e / (2 * (1 + nu_e))
    lam_q = np.repeat(lam_e[:, None], nq, axis=1)
    mu_q = np.repeat(mu_e[:, None], nq, axis=1)

    @BilinearForm
    def stiff(u, v, w):
        e_u, e_v = sym_grad(u), sym_grad(v)
        return w.lam * trace(e_u) * trace(e_v) + 2 * w.mu * ddot(e_u, e_v)

    K = asm(stiff, basis, lam=lam_q, mu=mu_q)

    # traction on the loaded face
    lf = mesh.facets_satisfying(lambda x: np.isclose(x[0], case.load_x, atol=1e-3))
    if lf.size == 0:
        raise RuntimeError(f"No boundary facets found on plane x={case.load_x}")
    fb = FacetBasis(mesh, elem, facets=lf)
    area = fb.dx.sum()
    t = np.array(case.force) / area

    @LinearForm
    def load(v, w):
        return t[0] * v.value[0] + t[1] * v.value[1] + t[2] * v.value[2]

    f = asm(load, fb)

    fixed_nodes = _fixed_nodes(mesh, case)
    if fixed_nodes.size == 0:
        raise RuntimeError("No fixed nodes found (check hole definitions vs. the CAD).")
    D = basis.nodal_dofs[:, fixed_nodes].ravel()
    u = np.zeros(K.shape[0])
    free = np.setdiff1d(np.arange(K.shape[0]), D)
    lu = splu(K[free][:, free].tocsc())
    u[free] = lu.solve(f[free])

    # element stresses (P1 -> constant per element)
    grad = basis.interpolate(u).grad[..., 0]          # (3,3,nelems)
    eps = 0.5 * (grad + grad.transpose(1, 0, 2))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    sig = 2 * mu_e * eps
    for i in range(3):
        sig[i, i] += lam_e * tr
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[1, 1]) ** 2 + (sig[1, 1] - sig[2, 2]) ** 2
                        + (sig[2, 2] - sig[0, 0]) ** 2)
                 + 3 * (sig[0, 1] ** 2 + sig[1, 2] ** 2 + sig[0, 2] ** 2))

    disp = u[basis.nodal_dofs].T                      # (nnodes,3)
    return dict(u=u, disp=disp, vm=vm, compliance=float(f @ u),
                max_disp=float(np.linalg.norm(disp, axis=1).max()),
                n_fixed_nodes=int(fixed_nodes.size), load_area=float(area))

"""STEP import + tetrahedral meshing (gmsh), and a demo-part generator."""
import numpy as np
import gmsh
from skfem import MeshTet

from .config import Case


def make_demo_bracket(step_path: str, case: Case | None = None):
    """Wall-mount bracket: 15 mm mounting plate + 95x40x25 mm cantilever arm, 4 bolt holes.
    Used only so the pipeline can run before a real customer STEP file is available."""
    case = case or Case()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("bracket")
        occ = gmsh.model.occ
        plate = occ.addBox(0, 0, 0, 15, 80, 80)
        arm = occ.addBox(15, 20, 0, 95, 40, 25)
        fused, _ = occ.fuse([(3, plate)], [(3, arm)])
        cyls = [(3, occ.addCylinder(h.x_min, h.y, h.z, h.x_max - h.x_min, 0, 0, h.r))
                for h in case.holes]
        occ.cut(fused, cyls)
        occ.synchronize()
        gmsh.write(step_path)
    finally:
        gmsh.finalize()


def mesh_step(step_path: str, h: float) -> MeshTet:
    """Import a STEP file and return a linear-tet scikit-fem mesh. h = target element size (mm)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("part")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h / 3)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.model.mesh.generate(3)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = coords.reshape(-1, 3)
        _, elem_tags, node_tags = gmsh.model.mesh.getElements(3)
        tets = np.asarray(node_tags[0], dtype=np.int64).reshape(-1, 4)

        remap = {t: i for i, t in enumerate(tags)}
        tets = np.vectorize(remap.get)(tets)
    finally:
        gmsh.finalize()

    # drop any nodes not used by tets
    used = np.unique(tets)
    new_idx = -np.ones(coords.shape[0], dtype=np.int64)
    new_idx[used] = np.arange(used.size)
    mesh = MeshTet(coords[used].T.copy(), new_idx[tets].T.copy())
    return mesh

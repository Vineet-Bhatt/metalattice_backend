"""STEP / STL / 3MF import + tetrahedral meshing (gmsh), and a demo-part generator."""
import os

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


def _extract_tets() -> MeshTet:
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    _, _, node_tags = gmsh.model.mesh.getElements(3)
    tets = np.asarray(node_tags[0], dtype=np.int64).reshape(-1, 4)
    remap = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    remap[tags.astype(np.int64)] = np.arange(len(tags))
    tets = remap[tets]
    used = np.unique(tets)                           # drop nodes not used by tets
    new_idx = -np.ones(coords.shape[0], dtype=np.int64)
    new_idx[used] = np.arange(used.size)
    return MeshTet(coords[used].T.copy(), new_idx[tets].T.copy())


SURFACE_EXT = (".stl", ".3mf", ".obj", ".ply", ".off")
STEP_EXT = (".step", ".stp")


def mesh_cad(path: str, h: float, scale: float = 1.0) -> MeshTet:
    """Tet-mesh a STEP / STL / 3MF / OBJ / PLY file. h = target element size (mm).

    scale converts file units to mm (STL has no units; 3MF units are applied by trimesh).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in STEP_EXT:
        return mesh_step(path, h)
    if ext in SURFACE_EXT:
        return mesh_surface(path, h, scale)
    raise ValueError(f"unsupported CAD format '{ext}'. Use one of {STEP_EXT + SURFACE_EXT}")


def load_clean_surface(path: str, scale: float = 1.0):
    """Load a surface file as ONE watertight trimesh, repairing what can be repaired."""
    import trimesh
    m = trimesh.load(path, force="mesh")           # 3MF/OBJ scenes are concatenated
    if scale != 1.0:
        m.apply_scale(scale)
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_inversion(m)
    trimesh.repair.fix_normals(m)
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    if not m.is_watertight:
        raise ValueError(
            f"{os.path.basename(path)} is not watertight after repair (open edges / non-manifold). "
            "Fix it in your CAD tool or a mesh repair tool, or export STEP instead.")
    if m.volume < 0:
        m.invert()
    return m


def _decimate_to_size(m, h: float, factor: float = 0.8):
    """Quadric-decimate a fine STL to ~ (factor*h) triangle edge length so the tet mesh stays small.

    Keeps the result watertight; falls back to the original surface if decimation breaks it.
    """
    target = int(m.area / (0.433 * (factor * h) ** 2))
    if target >= len(m.faces):
        return m
    try:
        d = m.simplify_quadric_decimation(face_count=max(target, 200))
        import trimesh
        d.merge_vertices()
        d.update_faces(d.nondegenerate_faces())       # decimation leaves a few zero-area faces
        d.remove_unreferenced_vertices()
        if not d.is_watertight:
            trimesh.repair.fill_holes(d)
        trimesh.repair.fix_normals(d)
        if d.is_watertight and abs(d.volume - m.volume) / abs(m.volume) < 0.02:
            return d
    except Exception:
        pass
    return m


def mesh_surface(path: str, h: float, scale: float = 1.0) -> MeshTet:
    import tempfile
    m = load_clean_surface(path, scale)
    m = _decimate_to_size(m, h)
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    tmp.close()
    try:
        m.export(tmp.name)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.model.add("part")
            gmsh.merge(tmp.name)
            # classify the facets into smooth patches, build a parametrisation so gmsh can
            # REMESH the surface at size h instead of keeping the (often huge) STL triangulation
            gmsh.model.mesh.classifySurfaces(np.deg2rad(40), True, True)
            gmsh.model.mesh.createGeometry()
            surfs = [e[1] for e in gmsh.model.getEntities(2)]
            loop = gmsh.model.geo.addSurfaceLoop(surfs)
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", h)
            gmsh.option.setNumber("Mesh.MeshSizeMin", h / 3)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
            gmsh.model.mesh.generate(3)
            mesh = _extract_tets()
        finally:
            gmsh.finalize()
    finally:
        os.unlink(tmp.name)
    return mesh


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
        mesh = _extract_tets()
    finally:
        gmsh.finalize()
    return mesh

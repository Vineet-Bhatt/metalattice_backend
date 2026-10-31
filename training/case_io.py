"""Supports and loads for an arbitrary part: JSON file or a quick preset.

bc.json example (all lengths mm, force N; boxes are [xmin, xmax, ymin, ymax, zmin, zmax]):
{
  "fixed_boxes": [[0, 3, -100, 100, -100, 100]],
  "load_box":    [99, 101, -100, 100, -100, 100],
  "force":       [0, 0, -5000],
  "shell_thickness": 2.0, "boss_thickness": 3.0, "mesh_size": 3.0,
  "holes": []
}
"""
import json

import numpy as np

from .config import Case, Hole


def load_case(path: str) -> Case:
    d = json.load(open(path))
    holes = [Hole(**h) for h in d.pop("holes", [])]
    d["force"] = tuple(d.get("force", (0.0, 0.0, -1000.0)))
    if d.get("load_box") is not None:
        d["load_box"] = tuple(d["load_box"])
    d["fixed_boxes"] = [tuple(b) for b in d.get("fixed_boxes", [])]
    case = Case(holes=holes, **d)
    if not case.fixed_boxes and not case.holes:
        raise ValueError("bc.json defines no supports (fixed_boxes or holes)")
    return case


def preset_cantilever(mesh, force_n: float, base: Case | None = None, direction=None) -> Case:
    """Clamp one end of the part's longest axis, load the opposite end face.

    The force acts along the shortest bounding-box axis (the natural bending direction) unless
    `direction` (a 3-vector) is given. This is a stand-in for real loads: results are only as
    good as the loads you give it.
    """
    base = base or Case(holes=[])
    lo, hi = mesh.p.min(axis=1), mesh.p.max(axis=1)
    ext = hi - lo
    a = int(np.argmax(ext))                         # span axis
    t = int(np.argmin(ext))                         # bending axis
    clamp = max(1.5 * base.mesh_size, 0.03 * ext[a])
    face = 0.02 * base.mesh_size + 1e-3
    fixed = lo.copy(), hi.copy()
    fb, lb = np.empty((3, 2)), np.empty((3, 2))
    for i in range(3):
        fb[i] = (lo[i] - 1e-3, hi[i] + 1e-3)
        lb[i] = (lo[i] - 1e-3, hi[i] + 1e-3)
    fb[a] = (lo[a] - 1e-3, lo[a] + clamp)
    lb[a] = (hi[a] - face, hi[a] + 1e-3)
    if direction is None:
        direction = np.zeros(3)
        direction[t] = -1.0
    direction = np.asarray(direction, float)
    base.fixed_boxes = [tuple(fb.ravel())]
    base.load_box = tuple(lb.ravel())
    base.force = tuple(force_n * direction / np.linalg.norm(direction))
    base.holes = []
    return base

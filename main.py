import argparse, json, os, time
import numpy as np

from training.config import Case, Material, LatticeModel
from training.geometry import make_demo_bracket, mesh_step
from training.lattice import assign_materials, element_volumes
from training.fea import solve_static
from training.gmm_props import load_property_table
from training.gmm_surrogate import fit_lattice_model
from training.graded import graded_design, oc_design


def evaluate(mesh, vol, mat, lat, case, rho_field=None):
    E, nu, sy, rho, is_lat = assign_materials(mesh, mat, lat, case, rho_field=rho_field)
    res = solve_static(mesh, E, nu, case)
    mass = float((vol * rho).sum() * mat.density)               # grams
    sf = sy / np.maximum(res["vm"], 1e-12)                      # per-element safety factor
    k = np.linalg.norm(case.force) / res["max_disp"]            # N/mm, secant stiffness
    return dict(
        mass_g=mass,
        lattice_volume_fraction=float(vol[is_lat].sum() / vol.sum()),
        max_disp_mm=res["max_disp"],
        compliance_Nmm=res["compliance"],
        stiffness_N_per_mm=float(k),
        specific_stiffness=float(k / mass),
        min_safety_factor=float(sf.min()),
        min_sf_p1pct=float(np.percentile(sf, 1)),   # robust to point-load/BC stress spikes
        min_sf_solid=float(sf[~is_lat].min()),
        min_sf_lattice=float(sf[is_lat].min()) if is_lat.any() else None,
        n_fixed_nodes=res["n_fixed_nodes"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default=None)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--mesh", type=float, default=None)
    ap.add_argument("--out", default="data/m1_report.json")
    ap.add_argument("--gmm-table", default=None,
                    help="CSV from export_gmm.py; fits c_E/n_E/c_sy/n_sy from GraphMetaMat data")
    ap.add_argument("--rho-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="restrict the property fit to this relative-density band")
    ap.add_argument("--graded", action="store_true",
                    help="grade lattice density per element from the FEA stress field")
    ap.add_argument("--sf-target", type=float, default=None,
                    help="safety-factor target for --graded (default: baseline 1st-percentile SF)")
    ap.add_argument("--oc", type=float, default=None, metavar="MASS_FRAC",
                    help="min-compliance grading at this mean lattice relative density (e.g. 0.25)")
    ap.add_argument("--rho-min", type=float, default=0.10)
    ap.add_argument("--rho-max", type=float, default=0.60)
    a = ap.parse_args()

    case, mat, lat = Case(), Material(), LatticeModel()
    if a.gmm_table:
        lat, diag = fit_lattice_model(load_property_table(a.gmm_table), rho_range=a.rho_range)
        print(f"[fit] GraphMetaMat table: {diag['n_rows']} rows, rho {diag['rho_min']:.3f}-"
              f"{diag['rho_max']:.3f} | E~{lat.c_E:.3g}*rho^{lat.n_E:.2f} (R2={diag['r2_E']:.3f}) | "
              f"sy~{lat.c_sy:.3g}*rho^{lat.n_sy:.2f} (R2={diag['r2_sy']:.3f})")
        print(f"[fit] unit check: {100*diag['frac_E_above_voigt']:.1f}% of rows have E_rel above the "
              f"Voigt bound (should be ~0); median sy/E strain = {diag['median_yield_strain']:.3g}")
    if a.density: lat.rel_density = a.density
    if a.mesh: case.mesh_size = a.mesh

    step = a.step
    if step is None:
        step = "data/demo_bracket.step"
        os.makedirs("data", exist_ok=True)
        make_demo_bracket(step, case)
        print(f"[1/4] demo STEP written: {step}")
    else:
        print(f"[1/4] using STEP: {step}")

    t0 = time.time()
    mesh = mesh_step(step, case.mesh_size)
    vol = element_volumes(mesh)
    print(f"[2/4] mesh: {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} tets, "
          f"volume {vol.sum():.0f} mm^3 ({time.time()-t0:.1f}s)")

    t0 = time.time()
    base = evaluate(mesh, vol, mat, None, case)
    print(f"[3/4] baseline (solid) solved ({time.time()-t0:.1f}s)")
    t0 = time.time()
    latt = evaluate(mesh, vol, mat, lat, case)
    print(f"[4/4] lattice-infilled solved ({time.time()-t0:.1f}s)")

    graded = None
    if a.graded:
        sf_t = a.sf_target or base["min_sf_p1pct"]
        t0 = time.time()
        graded = graded_design(mesh, mat, lat, case, sf_target=sf_t,
                               rho_min=a.rho_min, rho_max=a.rho_max)
        gl = evaluate(mesh, vol, mat, lat, case, rho_field=graded["rho"])
        print(f"[graded] SF target {sf_t:.3g}, {len(graded['history'])} FEA iterations, "
              f"converged={graded['converged']} ({time.time()-t0:.1f}s)")
        print(f"[graded] mass {gl['mass_g']:.4g} g ({100*(1-gl['mass_g']/base['mass_g']):.1f}% below solid), "
              f"stiffness {100*(gl['stiffness_N_per_mm']/base['stiffness_N_per_mm']-1):+.1f}%, "
              f"SF p1 {gl['min_sf_p1pct']:.3g}")
        graded["metrics"] = gl

    if a.oc:
        t0 = time.time()
        oc = oc_design(mesh, mat, lat, case, mass_frac=a.oc, rho_min=max(a.rho_min, 0.05), rho_max=a.rho_max)
        uni = lat.rel_density
        lat_u = type(lat)(**{**lat.__dict__, "rel_density": a.oc})
        un = evaluate(mesh, vol, mat, lat_u, case)                 # uniform lattice, same mass
        og = evaluate(mesh, vol, mat, lat, case, rho_field=oc["rho"])
        print(f"[oc] {len(oc['history'])} iterations ({time.time()-t0:.1f}s), "
              f"final change {oc['history'][-1]['max_change']:.3g}")
        print(f"{'':<20}{'solid':>12}{'uniform':>12}{'OC-graded':>12}")
        for lbl, key in (("mass (g)", "mass_g"), ("stiffness (N/mm)", "stiffness_N_per_mm"),
                         ("SF 1st pct", "min_sf_p1pct"), ("SF min lattice", "min_sf_lattice")):
            print(f"{lbl:<20}{base[key]:>12.4g}{(un[key] or 0):>12.4g}{(og[key] or 0):>12.4g}")
        print(f"[oc] vs uniform at equal mass: stiffness {100*(og['stiffness_N_per_mm']/un['stiffness_N_per_mm']-1):+.1f}%; "
              f"vs solid: mass -{100*(1-og['mass_g']/base['mass_g']):.1f}%, "
              f"stiffness {100*(og['stiffness_N_per_mm']/base['stiffness_N_per_mm']-1):+.1f}%")

    rows = [("mass (g)", "mass_g"), ("lattice vol. fraction", "lattice_volume_fraction"),
            ("max displacement (mm)", "max_disp_mm"), ("stiffness (N/mm)", "stiffness_N_per_mm"),
            ("specific stiffness (N/mm/g)", "specific_stiffness"),
            ("min safety factor (any elem)", "min_safety_factor"),
            ("safety factor, 1st percentile", "min_sf_p1pct"),
            ("min SF, solid regions", "min_sf_solid"), ("min SF, lattice regions", "min_sf_lattice")]
    print(f"\nMaterial: {mat.name} | Lattice: {lat.name}, rel. density {lat.rel_density}")
    print(f"Load: {case.force} N on x={case.load_x} | mesh h={case.mesh_size} mm\n")
    print(f"{'metric':<32}{'baseline':>14}{'lattice':>14}")
    for label, key in rows:
        b, l = base[key], latt[key]
        fmt = lambda v: "-" if v is None else f"{v:.4g}"
        print(f"{label:<32}{fmt(b):>14}{fmt(l):>14}")
    dm = 100 * (1 - latt["mass_g"] / base["mass_g"])
    ds = 100 * (latt["stiffness_N_per_mm"] / base["stiffness_N_per_mm"] - 1)
    dss = 100 * (latt["specific_stiffness"] / base["specific_stiffness"] - 1)
    print(f"\nmass change: -{dm:.1f}% | stiffness change: {ds:+.1f}% | specific stiffness change: {dss:+.1f}%")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    report = dict(baseline=base, lattice=latt, lattice_density=lat.rel_density,
                  mesh_size=case.mesh_size,
                  lattice_model=dict(name=lat.name, c_E=lat.c_E, n_E=lat.n_E,
                                     c_sy=lat.c_sy, n_sy=lat.n_sy))
    if graded is not None:
        report["graded"] = dict(metrics=graded["metrics"], history=graded["history"],
                                converged=graded["converged"])
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"report saved: {a.out}")


if __name__ == "__main__":
    main()

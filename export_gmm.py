"""Export a GraphMetaMat dataset directory to a rho / E_rel / sy_rel CSV (no torch needed).

    python export_gmm.py /path/to/gmm/dataset/stress_strain/standard data/gmm_props.csv \
        --stress-to-mpa 1 --base-E 1 --base-sy 1

Check the units first: see the UNITS WARNING in training/gmm_props.py.
"""
import argparse
from training.gmm_props import export_property_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--stress-to-mpa", type=float, default=1.0)
    ap.add_argument("--base-E", type=float, default=1.0, help="solid modulus the curves refer to (MPa)")
    ap.add_argument("--base-sy", type=float, default=1.0, help="solid yield the curves refer to (MPa)")
    ap.add_argument("--fit-strain", type=float, default=0.02)
    a = ap.parse_args()
    rows, skipped = export_property_table(
        a.dataset_dir, a.out_csv, stress_to_mpa=a.stress_to_mpa, base_E_mpa=a.base_E,
        base_sy_mpa=a.base_sy, fit_strain=a.fit_strain)
    print(f"wrote {len(rows)} rows to {a.out_csv} ({skipped} skipped)")


if __name__ == "__main__":
    main()

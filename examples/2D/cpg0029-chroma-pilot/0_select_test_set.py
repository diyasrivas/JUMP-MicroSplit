#!/usr/bin/env python3
"""
Draw the cpg0029 test FOVs once; every dye set predicts exactly these FOVs.

All five plates share one platemap, so a (well, site) pair is the same compound
on every plate. Draws, with one seed:
  n_compounds   random compounds out of the 90 treatments, all 4 wells each
  n_dmso_wells  random DMSO wells out of 24
  n_sites       random sites out of 9 per test well

Writes test_fovs.csv next to this script (kept in git).

Usage:
    python 0_select_test_set.py
    python 0_select_test_set.py --n_compounds 30 --n_dmso_wells 12 --n_sites 4 --seed 42
"""

import argparse
from pathlib import Path

from cpg0029_shared.split import select_test_fovs


def main() -> None:
    ap = argparse.ArgumentParser(description="Select the cpg0029 test FOVs shared by all dye sets")
    ap.add_argument("--n_compounds", type=int, default=30)
    ap.add_argument("--n_dmso_wells", type=int, default=12)
    ap.add_argument("--n_sites", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "test_fovs.csv"))
    args = ap.parse_args()

    fovs = select_test_fovs(args.n_compounds, args.n_dmso_wells, args.n_sites, args.seed)
    fovs.to_csv(args.out, index=False)
    trt = fovs[fovs.control_type == ""]
    print(f"{args.out}: {len(fovs)} FOVs on {fovs.well.nunique()} wells "
          f"({trt.broad_sample.nunique()} compounds, {trt.moa.nunique()} MoAs, "
          f"{fovs[fovs.control_type == 'negcon'].well.nunique()} DMSO wells)")


if __name__ == "__main__":
    main()

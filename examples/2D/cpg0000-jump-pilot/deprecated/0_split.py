#!/usr/bin/env python3
"""
Step 0 — build the gene-anchored train / test split manifest for cpg0000-jump-pilot.

Design (see docs/cpg0000_dataset_design.md):
  * Benchmark plates only: batch 2020_11_04_CPJUMP1, density 100, no antibiotics,
    compound plates in parental cells (the 4 Cas9-cell compound plates are excluded).
    That is 20 plates per cell line in 6 groups (compound/CRISPR/ORF × short/long).
  * Test set is chosen by GENE: G genes covered by all three modalities. Every compound
    targeting a test gene, both CRISPR guides and the ORF for it become test wells on
    every plate of their modality, at both time points → replicates stay complete.
  * Half of the negative controls are test negcons (fixed wells, same on every plate),
    so replicability mAP can use same-plate negatives exactly like the paper.
  * Test FOVs = test wells × a fixed subset of sites (default 2,4,6,8).
  * Training FOVs = N wells per plate drawn from the remaining wells, one random site.

Outputs (per cell line):
  {out_dir}/test_genes.csv                       shared gene list
  {out_dir}/{cell_line}/split_manifest.csv       one row per (plate, well, site, role)
  {out_dir}/{cell_line}/split_summary.json       counts + overlap checks

Usage:
    python 0_split.py --cell_line A549
    python 0_split.py --cell_line U2OS --n_genes 10 --seed 42
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BATCH = "2020_11_04_CPJUMP1"
CHANNEL_MAPPING = {"DNA": 5, "RNA": 3, "ER": 4, "AGP": 2, "Mito": 1}

DEFAULT_DATA_DIR  = "/project/cell_paint_mono"
DEFAULT_PAPER_DIR = "/project/cell_paint_mono/2024_Chandrasekaran_NatureMethods"
DEFAULT_OUT_DIR   = "/project/cell_paint_mono/splits/cpg0000-jump-pilot"

MANIFEST_COLUMNS = [
    "cell_line", "batch", "plate_dir", "barcode", "plate_map", "modality",
    "time_h", "timepoint", "well", "site", "role", "broad_sample", "gene",
    "control_type", "pert_type", "n_replicate_wells", "image_prefix",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def well_to_prefix(well: str, site: int) -> str:
    row = ord(well[0]) - 64
    col = int(well[1:])
    return f"r{row:02d}c{col:02d}f{site:02d}p01"


def benchmark_plates(tsv: str, cell_line: str) -> pd.DataFrame:
    """The plates the paper's Figures 4–5 use, for one cell line."""
    df = pd.read_csv(tsv, sep="\t")
    df = df[
        (df.Batch == BATCH)
        & (df.Density == 100)
        & (df.Antibiotics == "absent")
        & (df.Cell_type == cell_line)
    ]
    df = df[~((df.Perturbation == "compound") & (df.Cell_line == "Cas9"))].copy()
    df["modality"] = df.Perturbation.str.lower()
    df["time_h"] = df.Time.astype(int)
    # short/long per modality, as in the paper's utils.time_point
    df["timepoint"] = "short"
    for mod, g in df.groupby("modality"):
        long_t = g.time_h.max()
        df.loc[(df.modality == mod) & (df.time_h == long_t), "timepoint"] = "long"
    return df[["Assay_Plate_Barcode", "Plate_Map_Name", "modality", "time_h", "timepoint"]] \
        .rename(columns={"Assay_Plate_Barcode": "barcode", "Plate_Map_Name": "plate_map"}) \
        .sort_values(["modality", "time_h", "barcode"]).reset_index(drop=True)


def resolve_plate_dirs(data_dir: Path, barcodes) -> dict:
    images_root = data_dir / BATCH / "images"
    found = {}
    for d in images_root.iterdir():
        if d.is_dir():
            bc = d.name.split("__")[0]
            if bc in set(barcodes):
                found[bc] = d.name
    missing = [b for b in barcodes if b not in found]
    if missing:
        raise FileNotFoundError(f"Plate directories not found under {images_root}: {missing}")
    return found


def load_platemaps(paper_dir: Path) -> dict:
    pm_dir = paper_dir / "metadata" / "platemaps" / BATCH / "platemap"
    return {
        "compound": pd.read_csv(pm_dir / "JUMP-Target-1_compound_platemap.txt", sep="\t"),
        "crispr":   pd.read_csv(pm_dir / "JUMP-Target-1_crispr_platemap.txt",   sep="\t"),
        "orf":      pd.read_csv(pm_dir / "JUMP-Target-1_orf_platemap.txt",      sep="\t"),
    }


def load_annotations(paper_dir: Path) -> dict:
    ext = paper_dir / "metadata" / "external_metadata"
    comp = pd.read_csv(
        paper_dir / "benchmark" / "input" / "JUMP-Target-1_compound_metadata_additional_annotations.tsv",
        sep="\t",
    )
    comp["targets"] = comp.target_list.fillna("").apply(lambda s: [t for t in s.split("|") if t])
    crispr = pd.read_csv(ext / "JUMP-Target-1_crispr_metadata.tsv", sep="\t")
    orf    = pd.read_csv(ext / "JUMP-Target-1_orf_metadata.tsv",    sep="\t")
    rep    = pd.read_csv(paper_dir / "benchmark" / "output" / "cellprofiler_replicability_map.csv")
    return {"compound": comp, "crispr": crispr, "orf": orf, "replicability": rep}


def choose_test_genes(ann: dict, n_genes: int, seed: int, min_compounds: int = 2) -> pd.DataFrame:
    """Genes covered by all three modalities, ranked by how replicable their
    reagents were in the paper, then sampled from the top half with a fixed seed."""
    comp, crispr, orf, rep = ann["compound"], ann["crispr"], ann["orf"], ann["replicability"]

    crispr_trt = crispr[crispr.pert_type == "trt"]
    guides_per_gene = crispr_trt.groupby("gene").broad_sample.nunique()
    genes_2guides = set(guides_per_gene[guides_per_gene == 2].index)
    orf_genes = set(orf[orf.pert_type == "trt"].gene.dropna())

    comp_trt = comp[comp.pert_type == "trt"]
    comp_ex = comp_trt[["broad_sample", "targets"]].explode("targets").dropna()
    comps_per_gene = comp_ex.groupby("targets").broad_sample.nunique()
    comp_genes = set(comps_per_gene[comps_per_gene >= min_compounds].index)

    core = sorted(genes_2guides & orf_genes & comp_genes)
    if len(core) < n_genes:
        raise ValueError(f"Only {len(core)} core genes available, asked for {n_genes}")

    # replicability score: fraction of a gene's reagents (all groups) above q threshold
    rep_ok = rep.groupby("Metadata_broad_sample").above_q_threshold.mean()
    guide_of = crispr_trt.groupby("gene").broad_sample.apply(list)
    orf_of   = orf[orf.pert_type == "trt"].groupby("gene").broad_sample.apply(list)
    comp_of  = comp_ex.groupby("targets").broad_sample.apply(list)
    rows = []
    for g in core:
        reagents = guide_of.get(g, []) + orf_of.get(g, []) + comp_of.get(g, [])
        scores = [rep_ok[r] for r in reagents if r in rep_ok.index]
        rows.append({
            "gene": g,
            "n_compounds": len(comp_of.get(g, [])),
            "n_guides": len(guide_of.get(g, [])),
            "n_orfs": len(orf_of.get(g, [])),
            "replicability_score": float(np.mean(scores)) if scores else np.nan,
        })
    genes = pd.DataFrame(rows).sort_values(["replicability_score", "gene"], ascending=[False, True])
    pool = genes.head(max(n_genes, len(genes) // 2))
    rng = np.random.default_rng(seed)
    pick = rng.choice(pool.gene.values, size=n_genes, replace=False)
    chosen = genes[genes.gene.isin(pick)].sort_values("gene").reset_index(drop=True)
    chosen["seed"] = seed
    return chosen


def test_reagents(ann: dict, genes) -> dict:
    genes = set(genes)
    comp = ann["compound"]
    comp_hit = comp[comp.targets.apply(lambda ts: any(t in genes for t in ts))]
    crispr = ann["crispr"]
    orf = ann["orf"]
    return {
        "compound": sorted(comp_hit.broad_sample.unique()),
        "crispr": sorted(crispr[(crispr.pert_type == "trt") & crispr.gene.isin(genes)].broad_sample.unique()),
        "orf": sorted(orf[(orf.pert_type == "trt") & orf.gene.isin(genes)].broad_sample.unique()),
    }


def negcon_reagents(ann: dict, platemaps: dict, n_dmso: int, n_crispr: int, n_orf: int, seed: int) -> dict:
    """Fixed negative-control wells / reagents, same on every plate of a modality."""
    rng = np.random.default_rng(seed + 1)
    # compound plates: DMSO wells have broad_sample NaN in the platemap
    dmso_wells = sorted(platemaps["compound"][platemaps["compound"].broad_sample.isna()].well_position)
    dmso_pick = sorted(rng.choice(dmso_wells, size=n_dmso, replace=False))
    cr_neg = sorted(ann["crispr"][ann["crispr"].control_type == "negcon"].broad_sample)
    cr_pick = sorted(rng.choice(cr_neg, size=n_crispr, replace=False))
    orf_neg = sorted(ann["orf"][ann["orf"].control_type == "negcon"].broad_sample)
    orf_pick = sorted(rng.choice(orf_neg, size=n_orf, replace=False))
    return {"dmso_wells": dmso_pick, "crispr": cr_pick, "orf": orf_pick}


def annotate(modality: str, ann: dict, broad_sample):
    """(gene, control_type, pert_type) for a reagent."""
    if broad_sample is None or (isinstance(broad_sample, float) and np.isnan(broad_sample)):
        return ("", "negcon", "control") if modality == "compound" else ("", "empty", "empty")
    if modality == "compound":
        r = ann["compound"][ann["compound"].broad_sample == broad_sample]
        if r.empty:
            return ("", "", "")
        r = r.iloc[0]
        return ("|".join(r.targets), r.control_type if isinstance(r.control_type, str) else "", r.pert_type)
    tbl = ann[modality]
    r = tbl[tbl.broad_sample == broad_sample]
    if r.empty:
        return ("", "", "")
    r = r.iloc[0]
    return (r.gene if isinstance(r.gene, str) else "",
            r.control_type if isinstance(r.control_type, str) else "",
            r.pert_type)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build cpg0000 gene-anchored split manifest")
    ap.add_argument("--cell_line", required=True, choices=["A549", "U2OS"])
    ap.add_argument("--n_genes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_dmso", type=int, default=16, help="test DMSO wells per compound plate")
    ap.add_argument("--n_negcon_crispr", type=int, default=4, help="test negcon guides per CRISPR plate (2 wells each)")
    ap.add_argument("--n_negcon_orf", type=int, default=2, help="test negcon ORFs per ORF plate (4 wells each)")
    ap.add_argument("--test_sites", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--train_fovs_per_plate", type=int, default=15)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--paper_dir", default=DEFAULT_PAPER_DIR)
    ap.add_argument("--metadata_file", default=str(Path(__file__).parent / "experiment-metadata.tsv"))
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--no_verify", action="store_true", help="skip checking that every DNA tiff exists")
    args = ap.parse_args()

    data_dir, paper_dir, out_dir = Path(args.data_dir), Path(args.paper_dir), Path(args.out_dir)
    cl_dir = out_dir / args.cell_line
    cl_dir.mkdir(parents=True, exist_ok=True)

    plates = benchmark_plates(args.metadata_file, args.cell_line)
    plate_dirs = resolve_plate_dirs(data_dir, plates.barcode)
    plates["plate_dir"] = plates.barcode.map(plate_dirs)
    n_plates_in_group = plates.groupby(["modality", "timepoint"]).barcode.transform("count")
    plates["n_plates_in_group"] = n_plates_in_group
    print(f"{args.cell_line}: {len(plates)} benchmark plates")
    print(plates.groupby(["modality", "timepoint"]).barcode.apply(list).to_string())

    platemaps = load_platemaps(paper_dir)
    ann = load_annotations(paper_dir)

    # Test genes: reuse the shared file if it exists with the same settings
    genes_path = out_dir / "test_genes.csv"
    if genes_path.exists():
        genes = pd.read_csv(genes_path)
        if len(genes) != args.n_genes or int(genes.seed.iloc[0]) != args.seed:
            print(f"WARNING: {genes_path} has n={len(genes)} seed={genes.seed.iloc[0]}; regenerating")
            genes = choose_test_genes(ann, args.n_genes, args.seed)
            genes.to_csv(genes_path, index=False)
    else:
        genes = choose_test_genes(ann, args.n_genes, args.seed)
        genes.to_csv(genes_path, index=False)
    test_genes = list(genes.gene)
    reagents = test_reagents(ann, test_genes)
    negcons = negcon_reagents(ann, platemaps, args.n_dmso, args.n_negcon_crispr, args.n_negcon_orf, args.seed)
    print(f"Test genes ({len(test_genes)}): {test_genes}")
    print(f"Test reagents: {len(reagents['compound'])} compounds, {len(reagents['crispr'])} guides, "
          f"{len(reagents['orf'])} ORFs")

    rng = np.random.default_rng(args.seed + 2)
    rows = []
    for _, p in plates.iterrows():
        pm = platemaps[p.modality]
        mod = p.modality
        is_test_pert = pm.broad_sample.isin(reagents[mod])
        if mod == "compound":
            is_test_neg = pm.well_position.isin(negcons["dmso_wells"])
            is_empty = pd.Series(False, index=pm.index)
        else:
            is_test_neg = pm.broad_sample.isin(negcons[mod])
            is_empty = pm.broad_sample.isna()

        def add(pm_row, site, role):
            gene, ctype, ptype = annotate(mod, ann, pm_row.broad_sample)
            bs = pm_row.broad_sample if isinstance(pm_row.broad_sample, str) else ("DMSO" if mod == "compound" else "")
            copies = int((pm.broad_sample == pm_row.broad_sample).sum()) if isinstance(pm_row.broad_sample, str) else 1
            rows.append({
                "cell_line": args.cell_line, "batch": BATCH, "plate_dir": p.plate_dir,
                "barcode": p.barcode, "plate_map": p.plate_map, "modality": mod,
                "time_h": int(p.time_h), "timepoint": p.timepoint,
                "well": pm_row.well_position, "site": int(site), "role": role,
                "broad_sample": bs, "gene": gene, "control_type": ctype, "pert_type": ptype,
                "n_replicate_wells": copies * int(p.n_plates_in_group),
                "image_prefix": well_to_prefix(pm_row.well_position, site),
            })

        for _, r in pm[is_test_pert].iterrows():
            for s in args.test_sites:
                add(r, s, "test_perturbation")
        for _, r in pm[is_test_neg & ~is_test_pert].iterrows():
            for s in args.test_sites:
                add(r, s, "test_negcon")

        pool = pm[~is_test_pert & ~is_test_neg & ~is_empty]
        pick = rng.choice(pool.index.values, size=args.train_fovs_per_plate, replace=False)
        for idx in pick:
            add(pm.loc[idx], int(rng.integers(1, 10)), "train")

    man = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    man = man.sort_values(["role", "modality", "time_h", "barcode", "well", "site"]).reset_index(drop=True)

    # ---- checks -----------------------------------------------------------
    test = man[man.role != "train"]
    train = man[man.role == "train"]
    overlap_wells = set(zip(test.plate_dir, test.well)) & set(zip(train.plate_dir, train.well))
    overlap_samples = set(test[test.role == "test_perturbation"].broad_sample) & set(train.broad_sample)
    assert not overlap_wells, f"train/test well overlap: {list(overlap_wells)[:5]}"
    assert not overlap_samples, f"train/test perturbation overlap: {list(overlap_samples)[:5]}"
    # every test perturbation present on every plate of its modality
    for mod in ["compound", "crispr", "orf"]:
        n_pl = (plates.modality == mod).sum()
        per = test[(test.role == "test_perturbation") & (test.modality == mod)].groupby("broad_sample").plate_dir.nunique()
        bad = per[per != n_pl]
        assert bad.empty, f"{mod}: test perturbations missing on some plates: {bad.to_dict()}"

    missing = []
    if not args.no_verify:
        for _, r in man.iterrows():
            f = data_dir / BATCH / "images" / r.plate_dir / "Images" / f"{r.image_prefix}-ch{CHANNEL_MAPPING['DNA']}sk1fk1fl1.tiff"
            if not f.exists():
                missing.append(str(f))
        if missing:
            print(f"WARNING: {len(missing)} manifest FOVs have no DNA tiff on disk, e.g. {missing[:3]}")

    man_path = cl_dir / "split_manifest.csv"
    man.to_csv(man_path, index=False)

    def counts(df):
        return {
            "fovs": int(len(df)),
            "wells": int(df.groupby(["plate_dir", "well"]).ngroups),
            "by_modality": df.groupby("modality").size().to_dict(),
            "by_timepoint": df.groupby("timepoint").size().to_dict(),
            "by_plate": df.groupby("barcode").size().to_dict(),
        }
    summary = {
        "cell_line": args.cell_line,
        "batch": BATCH,
        "benchmark_plates": plates[["barcode", "plate_dir", "modality", "time_h", "timepoint"]].to_dict("records"),
        "args": vars(args),
        "test_genes": test_genes,
        "test_reagents": reagents,
        "test_negcons": negcons,
        "train": counts(train),
        "test_perturbation": counts(test[test.role == "test_perturbation"]),
        "test_negcon": counts(test[test.role == "test_negcon"]),
        "test_total_fovs": int(len(test)),
        "test_negcon_fraction": float(len(test[test.role == "test_negcon"]) / max(len(test), 1)),
        "benchmark_fovs_total": int(len(plates) * 384 * 9),
        "train_fraction_of_benchmark": float(len(train) / (len(plates) * 384 * 9)),
        "test_fraction_of_benchmark": float(len(test) / (len(plates) * 384 * 9)),
        "overlap_checks": {"well_overlap": 0, "perturbation_overlap": 0, "missing_files": len(missing)},
        "script_sha1": hashlib.sha1(open(__file__, "rb").read()).hexdigest(),
    }
    with open(cl_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nManifest: {man_path}  ({len(man)} rows)")
    print(f"  train: {summary['train']['fovs']} FOVs on {summary['train']['wells']} wells")
    print(f"  test:  {summary['test_total_fovs']} FOVs on "
          f"{summary['test_perturbation']['wells'] + summary['test_negcon']['wells']} wells "
          f"({summary['test_negcon_fraction']:.0%} negcon)")
    print(f"  train share of benchmark FOVs: {summary['train_fraction_of_benchmark']:.2%}; "
          f"test share: {summary['test_fraction_of_benchmark']:.2%}")


if __name__ == "__main__":
    main()

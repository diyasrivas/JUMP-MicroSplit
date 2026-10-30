#!/usr/bin/env python3
"""
Step 0: build the train / test split manifest for cpg0000-jump-pilot.

Only the 20 benchmark plates per cell line used in Figures 4 and 5 of
Chandrasekaran et al. (2024) are used: batch 2020_11_04_CPJUMP1, density 100,
no antibiotics, compound plates in parental cells. Every modality has one
platemap shared by all its plates, so a well position holds the same
perturbation on every replicate plate.

Test wells are chosen by well position and are test wells on every plate of
their modality, so all replicates of a test perturbation are kept. Two designs:

  gene_subset   genes drawn uniformly at random from those targeted in all three
                modalities (two CRISPR guides, one ORF, at least two compounds);
                every compound, guide and ORF of a drawn gene is a test perturbation.
  random_wells  a fraction of each modality's treatment perturbations drawn
                uniformly at random; all wells of a drawn perturbation are test wells.

Both designs add a fixed set of negative-control well positions per modality,
predict the same fixed sites of every test well, and sample training FOVs from
the remaining wells (distinct wells, one random site each, spread evenly over
the plates).

Outputs (per design and cell line):
  {out_dir}/{design}/{cell_line}/split_manifest.csv   one row per (plate, well, site, role)
  {out_dir}/{design}/{cell_line}/split_summary.json   counts and overlap checks

Usage:
    python 0_split.py --design gene_subset --cell_line A549
    python 0_split.py --design random_wells --cell_line U2OS --fraction 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BATCH = "2020_11_04_CPJUMP1"
MODALITIES = ("compound", "crispr", "orf")
DESIGNS = ("gene_subset", "random_wells")
SITES_PER_WELL = 9
DNA_CHANNEL_INDEX = 5

DEFAULT_DATA_DIR = "/project/cell_paint_mono"
DEFAULT_PAPER_DIR = "/project/cell_paint_mono/2024_Chandrasekaran_NatureMethods"
DEFAULT_OUT_DIR = "/project/cell_paint_mono/splits/cpg0000-jump-pilot"

MANIFEST_COLUMNS = [
    "cell_line", "batch", "plate_dir", "barcode", "plate_map", "modality",
    "time_h", "timepoint", "well", "site", "role", "broad_sample", "gene",
    "control_type", "pert_type", "n_replicate_wells", "image_prefix",
]


def well_to_prefix(well: str, site: int) -> str:
    """Opera Phenix file prefix of one FOV, e.g. ``r01c02f03p01``."""
    return f"r{ord(well[0]) - 64:02d}c{int(well[1:]):02d}f{site:02d}p01"


def benchmark_plates(metadata_file: str, cell_line: str) -> pd.DataFrame:
    """The plates of the paper's Figures 4 and 5 for one cell line.

    Parameters
    ----------
    metadata_file : str
        The paper's ``experiment-metadata.tsv``.
    cell_line : str
        ``"A549"`` or ``"U2OS"``.

    Returns
    -------
    pandas.DataFrame
        One row per plate: barcode, plate_map, modality, time_h, timepoint.
    """
    df = pd.read_csv(metadata_file, sep="\t")
    df = df[
        (df.Batch == BATCH)
        & (df.Density == 100)
        & (df.Antibiotics == "absent")
        & (df.Cell_type == cell_line)
    ]
    df = df[~((df.Perturbation == "compound") & (df.Cell_line == "Cas9"))].copy()
    df["modality"] = df.Perturbation.str.lower()
    df["time_h"] = df.Time.astype(int)
    long_time = df.groupby("modality").time_h.transform("max")
    df["timepoint"] = np.where(df.time_h == long_time, "long", "short")
    df = df.rename(columns={"Assay_Plate_Barcode": "barcode", "Plate_Map_Name": "plate_map"})
    columns = ["barcode", "plate_map", "modality", "time_h", "timepoint"]
    return df[columns].sort_values(["modality", "time_h", "barcode"]).reset_index(drop=True)


def resolve_plate_dirs(data_dir: Path, barcodes: list[str]) -> dict[str, str]:
    """Barcode -> plate directory name under ``{data_dir}/{BATCH}/images/``."""
    images_root = data_dir / BATCH / "images"
    found = {d.name.split("__")[0]: d.name for d in images_root.iterdir() if d.is_dir()}
    missing = [b for b in barcodes if b not in found]
    if missing:
        raise FileNotFoundError(f"Plate directories not found under {images_root}: {missing}")
    return {b: found[b] for b in barcodes}


def load_platemaps(paper_dir: Path) -> dict[str, pd.DataFrame]:
    """Platemap of each modality, one row per well."""
    pm_dir = paper_dir / "metadata" / "platemaps" / BATCH / "platemap"
    return {
        m: pd.read_csv(pm_dir / f"JUMP-Target-1_{m}_platemap.txt", sep="\t")
        for m in MODALITIES
    }


def load_annotations(paper_dir: Path) -> dict[str, pd.DataFrame]:
    """Perturbation annotations of each modality (targets, control and perturbation type)."""
    ext = paper_dir / "metadata" / "external_metadata"
    compound = pd.read_csv(
        paper_dir / "benchmark" / "input"
        / "JUMP-Target-1_compound_metadata_additional_annotations.tsv",
        sep="\t",
    )
    compound["targets"] = compound.target_list.fillna("").str.split("|").apply(
        lambda ts: [t for t in ts if t]
    )
    return {
        "compound": compound,
        "crispr": pd.read_csv(ext / "JUMP-Target-1_crispr_metadata.tsv", sep="\t"),
        "orf": pd.read_csv(ext / "JUMP-Target-1_orf_metadata.tsv", sep="\t"),
    }


def annotate_platemap(platemap: pd.DataFrame, annotation: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Add gene, control_type and pert_type to every well of a platemap.

    Compound DMSO wells have no ``broad_sample`` and become negative controls;
    empty CRISPR / ORF wells get ``pert_type == "empty"``.
    """
    pm = platemap.rename(columns={"well_position": "well"})[["well", "broad_sample"]].copy()
    ann = annotation.drop_duplicates("broad_sample").set_index("broad_sample")
    if modality == "compound":
        gene = ann.targets.apply("|".join)
    else:
        gene = ann.gene
    pm["gene"] = pm.broad_sample.map(gene).fillna("")
    pm["control_type"] = pm.broad_sample.map(ann.control_type).fillna("")
    pm["pert_type"] = pm.broad_sample.map(ann.pert_type).fillna("")
    no_sample = pm.broad_sample.isna()
    if modality == "compound":
        pm.loc[no_sample, ["broad_sample", "control_type", "pert_type"]] = ["DMSO", "negcon", "control"]
    else:
        pm.loc[no_sample, ["broad_sample", "pert_type"]] = ["", "empty"]
    pm["copies"] = pm.groupby("broad_sample").well.transform("count")
    return pm


def eligible_genes(annotations: dict[str, pd.DataFrame]) -> list[str]:
    """Genes with two CRISPR guides, an ORF and at least two compounds."""
    crispr = annotations["crispr"]
    guides = crispr[crispr.pert_type == "trt"].groupby("gene").broad_sample.nunique()
    orf = annotations["orf"]
    orf_genes = set(orf[orf.pert_type == "trt"].gene.dropna())
    compound = annotations["compound"]
    targets = compound[compound.pert_type == "trt"][["broad_sample", "targets"]].explode("targets")
    compounds = targets.dropna().groupby("targets").broad_sample.nunique()
    return sorted(set(guides[guides == 2].index) & orf_genes & set(compounds[compounds >= 2].index))


def test_perturbations_gene_subset(
    platemaps: dict[str, pd.DataFrame], genes: list[str]
) -> dict[str, set[str]]:
    """Treatment perturbations of each modality that target one of ``genes``."""
    genes = set(genes)
    out = {}
    for modality, pm in platemaps.items():
        trt = pm[pm.pert_type == "trt"]
        hits = trt.gene.str.split("|").apply(lambda gs: bool(genes.intersection(gs)))
        out[modality] = set(trt[hits].broad_sample)
    return out


def test_perturbations_random(
    platemaps: dict[str, pd.DataFrame], fraction: float, rng: np.random.Generator
) -> dict[str, set[str]]:
    """A random ``fraction`` of each modality's treatment perturbations."""
    out = {}
    for modality, pm in platemaps.items():
        samples = np.sort(pm[pm.pert_type == "trt"].broad_sample.unique())
        n = max(1, round(fraction * len(samples)))
        out[modality] = set(rng.choice(samples, size=n, replace=False))
    return out


def negcon_wells(
    platemaps: dict[str, pd.DataFrame], n_negcon: int, rng: np.random.Generator
) -> dict[str, list[str]]:
    """``n_negcon`` negative-control well positions per modality, the same on every plate."""
    out = {}
    for modality, pm in platemaps.items():
        wells = np.sort(pm[pm.control_type == "negcon"].well.values)
        out[modality] = sorted(rng.choice(wells, size=n_negcon, replace=False))
    return out


def build_manifest(
    plates: pd.DataFrame,
    platemaps: dict[str, pd.DataFrame],
    test_samples: dict[str, set[str]],
    test_negcons: dict[str, list[str]],
    test_sites: list[int],
    n_train: int,
    cell_line: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One manifest row per test FOV and per training FOV.

    Training FOVs are drawn from wells that are neither test perturbations nor
    test negative controls nor empty, ``n_train`` in total spread evenly over
    the plates, distinct wells, one random site each.
    """
    per_plate = np.full(len(plates), n_train // len(plates))
    per_plate[rng.choice(len(plates), size=n_train % len(plates), replace=False)] += 1

    rows = []
    for (_, plate), n_plate in zip(plates.iterrows(), per_plate):
        pm = platemaps[plate.modality]
        is_test = pm.broad_sample.isin(test_samples[plate.modality])
        is_negcon = pm.well.isin(test_negcons[plate.modality]) & ~is_test
        is_pool = ~is_test & ~is_negcon & (pm.pert_type != "empty")
        plate_fields = {
            "cell_line": cell_line, "batch": BATCH, "plate_dir": plate.plate_dir,
            "barcode": plate.barcode, "plate_map": plate.plate_map,
            "modality": plate.modality, "time_h": int(plate.time_h),
            "timepoint": plate.timepoint,
        }

        def add(well_row: pd.Series, site: int, role: str) -> None:
            rows.append({
                **plate_fields,
                "well": well_row.well, "site": int(site), "role": role,
                "broad_sample": well_row.broad_sample, "gene": well_row.gene,
                "control_type": well_row.control_type, "pert_type": well_row.pert_type,
                "n_replicate_wells": int(well_row.copies * plate.n_plates_in_group),
                "image_prefix": well_to_prefix(well_row.well, site),
            })

        for role, mask in (("test_perturbation", is_test), ("test_negcon", is_negcon)):
            for _, well_row in pm[mask].iterrows():
                for site in test_sites:
                    add(well_row, site, role)
        pool = pm[is_pool]
        for idx in rng.choice(pool.index.values, size=n_plate, replace=False):
            add(pm.loc[idx], int(rng.integers(1, SITES_PER_WELL + 1)), "train")

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return manifest.sort_values(
        ["role", "modality", "time_h", "barcode", "well", "site"]
    ).reset_index(drop=True)


def check_manifest(manifest: pd.DataFrame, plates: pd.DataFrame) -> None:
    """Assert that no training well or perturbation is in the test set and that
    every test perturbation is present on every plate of its modality."""
    test = manifest[manifest.role != "train"]
    train = manifest[manifest.role == "train"]
    overlap_wells = set(zip(test.plate_dir, test.well)) & set(zip(train.plate_dir, train.well))
    assert not overlap_wells, f"train/test well overlap: {sorted(overlap_wells)[:5]}"
    test_samples = set(test[test.role == "test_perturbation"].broad_sample)
    overlap_samples = test_samples & set(train.broad_sample)
    assert not overlap_samples, f"train/test perturbation overlap: {sorted(overlap_samples)[:5]}"
    for modality in MODALITIES:
        n_plates = (plates.modality == modality).sum()
        per = test[(test.role == "test_perturbation") & (test.modality == modality)]
        incomplete = per.groupby("broad_sample").plate_dir.nunique().loc[lambda s: s != n_plates]
        assert incomplete.empty, f"{modality}: test perturbations missing on some plates"


def missing_images(manifest: pd.DataFrame, data_dir: Path) -> list[str]:
    """Manifest FOVs without a DNA image on disk."""
    images = data_dir / BATCH / "images"
    paths = (
        images / manifest.plate_dir / "Images"
        / (manifest.image_prefix + f"-ch{DNA_CHANNEL_INDEX}sk1fk1fl1.tiff")
    )
    return [str(p) for p in paths if not p.exists()]


def summarize(manifest: pd.DataFrame, args: argparse.Namespace, test_info: dict, n_missing: int) -> dict:
    """Counts behind a split, written next to the manifest."""
    def counts(df: pd.DataFrame) -> dict:
        return {
            "fovs": len(df),
            "wells": df.groupby(["plate_dir", "well"]).ngroups,
            "by_modality": df.groupby("modality").size().to_dict(),
        }

    test = manifest[manifest.role != "train"]
    return {
        "design": args.design,
        "cell_line": args.cell_line,
        "args": {k: v for k, v in vars(args).items() if k not in ("data_dir", "paper_dir", "out_dir")},
        **test_info,
        "train": counts(manifest[manifest.role == "train"]),
        "test_perturbation": counts(test[test.role == "test_perturbation"]),
        "test_negcon": counts(test[test.role == "test_negcon"]),
        "test_fovs": len(test),
        "missing_images": n_missing,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the cpg0000 train / test split manifest")
    ap.add_argument("--design", required=True, choices=DESIGNS)
    ap.add_argument("--cell_line", required=True, choices=["A549", "U2OS"])
    ap.add_argument("--n_genes", type=int, default=5, help="gene_subset: number of test genes")
    ap.add_argument("--fraction", type=float, default=0.05,
                    help="random_wells: fraction of treatment perturbations per modality")
    ap.add_argument("--n_negcon", type=int, default=8,
                    help="test negative-control well positions per modality")
    ap.add_argument("--test_sites", type=int, nargs="+", default=[3, 7])
    ap.add_argument("--n_train", type=int, default=300, help="training FOVs per cell line")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--paper_dir", default=DEFAULT_PAPER_DIR)
    ap.add_argument("--metadata_file", default=str(Path(__file__).parent / "experiment-metadata.tsv"))
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir, paper_dir = Path(args.data_dir), Path(args.paper_dir)
    out_dir = Path(args.out_dir) / args.design / args.cell_line
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    plates = benchmark_plates(args.metadata_file, args.cell_line)
    plates["plate_dir"] = plates.barcode.map(resolve_plate_dirs(data_dir, list(plates.barcode)))
    plates["n_plates_in_group"] = plates.groupby(["modality", "timepoint"]).barcode.transform("count")
    annotations = load_annotations(paper_dir)
    platemaps = {
        m: annotate_platemap(pm, annotations[m], m) for m, pm in load_platemaps(paper_dir).items()
    }

    if args.design == "gene_subset":
        genes = sorted(rng.choice(eligible_genes(annotations), size=args.n_genes, replace=False))
        test_samples = test_perturbations_gene_subset(platemaps, genes)
        test_info = {"test_genes": genes}
    else:
        test_samples = test_perturbations_random(platemaps, args.fraction, rng)
        test_info = {}
    test_negcons = negcon_wells(platemaps, args.n_negcon, rng)
    test_info["test_perturbations"] = {m: sorted(s) for m, s in test_samples.items()}
    test_info["test_negcon_wells"] = test_negcons

    manifest = build_manifest(
        plates, platemaps, test_samples, test_negcons, args.test_sites,
        args.n_train, args.cell_line, rng,
    )
    check_manifest(manifest, plates)
    missing = missing_images(manifest, data_dir)
    if missing:
        print(f"WARNING: {len(missing)} manifest FOVs have no DNA image on disk, e.g. {missing[:3]}")

    manifest.to_csv(out_dir / "split_manifest.csv", index=False)
    summary = summarize(manifest, args, test_info, len(missing))
    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"{args.design} / {args.cell_line}: {len(plates)} plates -> {out_dir}")
    if "test_genes" in test_info:
        print(f"  test genes: {', '.join(test_info['test_genes'])}")
    for role in ("test_perturbation", "test_negcon", "train"):
        c = summary[role]
        print(f"  {role:18s} {c['fovs']:5d} FOVs on {c['wells']:4d} wells {c['by_modality']}")


if __name__ == "__main__":
    main()

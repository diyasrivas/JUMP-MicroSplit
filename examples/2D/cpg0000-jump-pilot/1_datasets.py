#!/usr/bin/env python3
"""
Training dataset generation for cpg0000-jump-pilot (JUMP Cell Painting Pilot).

Reads locally downloaded pilot plates, samples FOVs stratified by cell line,
perturbation type, and experimental group, then writes a MicroSplit dataset:

  training_dataset_{cell_line}/
      combined/        float32 weighted sum of DNA+RNA+ER+AGP+Mito
      DNA/             uint16 individual channels
      RNA/
      ER/
      AGP/
      Mito/
      metadata.csv

Channel mapping (cpg0000-jump-pilot, Opera Phenix):
  ch1=Mito, ch2=AGP, ch3=RNA, ch4=ER, ch5=DNA

MicroSplit targets: DNA(ch5), RNA(ch3), ER(ch4), AGP(ch2), Mito(ch1)

Usage:
    python 1_datasets.py --cell_line A549
    python 1_datasets.py --cell_line U2OS --samples 3500 --seed 42
    python 1_datasets.py --cell_line A549 --config config_A549.yaml
"""

import argparse
import sys
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from microsplit_reproducibility.workflows.cellpainting import (
    get_available_sites,
    build_dataset_from_samples,
)

# ---------------------------------------------------------------------------
# Channel configuration
# ---------------------------------------------------------------------------

CHANNEL_NAMES   = ["DNA", "RNA", "ER", "AGP", "Mito"]
CHANNEL_MAPPING = {"DNA": 5, "RNA": 3, "ER": 4, "AGP": 2, "Mito": 1}

# ---------------------------------------------------------------------------
# Plate metadata
# ---------------------------------------------------------------------------

PLATE_RANGES = [
    (116991, 116994, "U2OS",  "compound", "short"),
    (116995, 117007, "A549",  "compound", "short"),
    (117008, 117011, "U2OS",  "compound", "long"),
    (117012, 117024, "A549",  "compound", "long"),
    (117025, 117033, "U2OS",  "crispr",   "short"),
    (117034, 117041, "A549",  "crispr",   "short"),
    (117042, 117049, "U2OS",  "crispr",   "long"),
    (117050, 117057, "A549",  "crispr",   "long"),
    (117058, 117061, "U2OS",  "orf",      "short"),
    (117062, 117065, "A549",  "orf",      "short"),
    (117066, 117069, "U2OS",  "orf",      "long"),
    (118039, 118050, "A549",  "orf",      "long"),
]


@dataclass
class PlateMetadata:
    plate_id: str
    batch:    str
    cell_line: str
    perturbation_type: str
    timepoint: str
    experimental_group: str

    @property
    def stratum(self) -> str:
        return f"{self.perturbation_type}_{self.timepoint}_{self.experimental_group}"


@dataclass
class SamplingConfig:
    cell_line:     str
    target_samples: int  = 3500
    seed:           int  = 42
    metadata_file:  str  = "experiment-metadata.tsv"
    data_dir:       str  = "/project/cell_paint_mono/cpg0000-jump-pilot"
    output_dir:     Optional[str] = None
    exclude_plates: List[str] = field(default_factory=list)
    perturbation_weights: Dict[str, float] = field(
        default_factory=lambda: {"compound": 0.45, "crispr": 0.30, "orf": 0.25}
    )
    experimental_group_weights: Dict[str, float] = field(
        default_factory=lambda: {"primary": 0.75, "secondary": 0.25}
    )

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f"training_dataset_{self.cell_line}"

    @classmethod
    def from_yaml(cls, filepath: str):
        with open(filepath) as f:
            return cls(**yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Plate discovery / metadata loading
# ---------------------------------------------------------------------------

def _parse_plate_number(plate_dir: str) -> int:
    plate_id = plate_dir.split("__")[0].rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if plate_id.startswith("BR"):
        return int(plate_id[2:8])
    return 0


def _infer_plate_metadata(plate_dir: str, batch: str) -> Optional[PlateMetadata]:
    plate_num = _parse_plate_number(plate_dir)
    exp_group = (
        "secondary"
        if any(x in batch for x in ["TimepointDay", "WeeksTimePoint", "Bleaching"])
        else "primary"
    )
    for min_id, max_id, cell_line, pert_type, timepoint in PLATE_RANGES:
        if min_id <= plate_num <= max_id:
            return PlateMetadata(
                plate_id=plate_dir, batch=batch,
                cell_line=cell_line, perturbation_type=pert_type,
                timepoint=timepoint, experimental_group=exp_group,
            )
    return None


def load_plate_metadata_from_tsv(tsv_path: str) -> Dict[str, PlateMetadata]:
    """Load plate metadata from the experiment-metadata.tsv file."""
    import csv
    metadata = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            plate_id   = row["Assay_Plate_Barcode"]
            batch      = row["Batch"]
            cell_type  = row["Cell_type"]
            pert       = row["Perturbation"].lower()
            time_hrs   = int(row["Time"])
            timepoint  = "short" if time_hrs <= 48 else "long"
            exp_group  = (
                "secondary"
                if any(x in batch for x in ["TimepointDay", "WeeksTimePoint", "Bleaching"])
                else "primary"
            )
            key = f"{batch}_{plate_id}"
            metadata[key] = PlateMetadata(
                plate_id=plate_id, batch=batch, cell_line=cell_type,
                perturbation_type=pert, timepoint=timepoint,
                experimental_group=exp_group,
            )
    return metadata


def scan_plates(data_dir: Path, metadata_file: str) -> Dict[str, PlateMetadata]:
    """Scan local plate directories and match to metadata."""
    tsv_metadata = load_plate_metadata_from_tsv(metadata_file)
    print(f"Metadata loaded from TSV: {len(tsv_metadata)} entries")

    found = {}
    for batch_dir in data_dir.iterdir():
        if not batch_dir.is_dir():
            continue
        images_dir = batch_dir / "images"
        if not images_dir.exists():
            continue
        for plate_dir in images_dir.iterdir():
            if not plate_dir.is_dir():
                continue
            plate_name = plate_dir.name
            batch      = batch_dir.name
            key        = f"{batch}_{plate_name}"
            if key in tsv_metadata:
                found[key] = tsv_metadata[key]
                continue
            # Fuzzy match: try base ID (strip suffix)
            base_id = plate_name.split("__")[0].rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            for meta_key, meta in tsv_metadata.items():
                if meta.batch == batch and meta.plate_id == base_id:
                    found[key] = PlateMetadata(
                        plate_id=plate_name, batch=batch,
                        cell_line=meta.cell_line,
                        perturbation_type=meta.perturbation_type,
                        timepoint=meta.timepoint,
                        experimental_group=meta.experimental_group,
                    )
                    break

    print(f"Matched plates on disk: {len(found)}")
    if not found:
        raise ValueError("No plates matched. Check data_dir and metadata_file.")
    return found


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def generate_sample_plan(
    plate_metadata: Dict[str, PlateMetadata],
    config: SamplingConfig,
    data_dir: Path,
) -> List[Tuple[str, str, str, int]]:
    """Generate a stratified list of (batch, plate_dir, well, site) to sample."""
    rng = np.random.default_rng(config.seed)

    # Filter by cell line and optional plate exclusions
    filtered = {
        k: v for k, v in plate_metadata.items()
        if v.cell_line == config.cell_line
        and not any(excl in v.plate_id for excl in config.exclude_plates)
    }

    print(f"\n{'=' * 70}")
    print(f"Cell line: {config.cell_line} | Available plates: {len(filtered)}")
    print(f"{'=' * 70}")

    if not filtered:
        raise ValueError(
            f"No plates found for cell line '{config.cell_line}'. "
            "Check TSV metadata and data_dir."
        )

    # Group by stratum
    strata: Dict[str, List[str]] = defaultdict(list)
    for key, meta in filtered.items():
        strata[meta.stratum].append(key)

    print("Strata:")
    for s, plates in strata.items():
        print(f"  {s}: {len(plates)} plates")

    # Allocate samples across strata
    allocation = {}
    for stratum, plates in strata.items():
        pert_type, timepoint, exp_group = stratum.split("_", 2)
        weight = (
            config.perturbation_weights.get(pert_type, 0.33)
            * config.experimental_group_weights.get(exp_group, 0.5)
        )
        allocation[stratum] = int(weight * config.target_samples)

    deficit = config.target_samples - sum(allocation.values())
    if deficit > 0:
        allocation[max(allocation, key=allocation.get)] += deficit

    print("\nAllocation:")
    for s, n in allocation.items():
        print(f"  {s}: {n}")

    # Sample FOVs from each stratum
    all_samples: List[Tuple[str, str, str, int]] = []
    for stratum, n_samples in allocation.items():
        plates = strata[stratum]
        per_plate = max(1, n_samples // len(plates))

        for plate_key in plates:
            meta = plate_metadata[plate_key]
            images_dir = data_dir / meta.batch / "images" / meta.plate_id / "Images"
            available  = get_available_sites(images_dir)
            if not available:
                continue
            n_select   = min(per_plate, len(available))
            idxs       = rng.choice(len(available), size=n_select, replace=False)
            for i in idxs:
                well, site = available[int(i)]
                all_samples.append((meta.batch, meta.plate_id, well, site))
            if len(all_samples) >= sum(allocation.values()):
                break

    return all_samples[: config.target_samples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build cpg0000-jump-pilot training dataset for MicroSplit"
    )
    parser.add_argument("--config",     type=str,
                        help="YAML config file (overrides all other args)")
    parser.add_argument("--cell_line",  type=str, choices=["A549", "U2OS"],
                        help="Cell line to build dataset for")
    parser.add_argument("--samples",    type=int, default=3500)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--data_dir",   type=str,
                        default="/project/cell_paint_mono/cpg0000-jump-pilot",
                        help="Root of the locally downloaded pilot data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dataset directory "
                             "(default: training_dataset_{cell_line})")
    parser.add_argument("--metadata_file", type=str,
                        default="experiment-metadata.tsv",
                        help="Path to the experiment-metadata.tsv file")
    parser.add_argument("--exclude_plates", nargs="*", default=[],
                        help="Plate IDs (or substrings) to exclude")
    args = parser.parse_args()

    if args.config:
        config = SamplingConfig.from_yaml(args.config)
    else:
        if not args.cell_line:
            parser.error("--cell_line is required (unless --config is given)")
        config = SamplingConfig(
            cell_line=args.cell_line,
            target_samples=args.samples,
            seed=args.seed,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            metadata_file=args.metadata_file,
            exclude_plates=args.exclude_plates or [],
        )

    data_dir   = Path(config.data_dir)
    output_dir = Path(config.output_dir)

    print(f"Cell line:    {config.cell_line}")
    print(f"Data dir:     {data_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Target FOVs:  {config.target_samples}")

    plate_metadata = scan_plates(data_dir, config.metadata_file)
    samples        = generate_sample_plan(plate_metadata, config, data_dir)
    print(f"\nSample plan: {len(samples)} FOVs")

    # Build extra metadata for each sample
    plate_meta_lookup = {
        (v.batch, v.plate_id): v for v in plate_metadata.values()
    }
    extra = []
    for batch, plate, well, site in samples:
        meta = plate_meta_lookup.get((batch, plate))
        extra.append({
            "cell_line":        config.cell_line,
            "perturbation_type": meta.perturbation_type if meta else "unknown",
            "timepoint":         meta.timepoint         if meta else "unknown",
            "experimental_group": meta.experimental_group if meta else "unknown",
        })

    build_dataset_from_samples(
        samples=samples,
        data_dir=data_dir,
        output_dir=output_dir,
        channel_names=CHANNEL_NAMES,
        channel_mapping=CHANNEL_MAPPING,
        extra_metadata=extra,
    )


if __name__ == "__main__":
    main()

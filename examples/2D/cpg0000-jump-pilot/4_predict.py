#!/usr/bin/env python3
"""
Step 4 — MicroSplit prediction for cpg0000-jump-pilot, one plate per job.

With --manifest, only the test FOVs of that plate (roles test_perturbation and
test_negcon in the split manifest) are predicted; without it, every FOV of the
plate is predicted (legacy behaviour).

For each FOV:
  1. Reads the 5 fluorescence channels from the raw plate images
  2. Combined input = average of the channels (as in training)
  3. Tiles (patch 64, stride 32, inner tiling) with 3 lateral-context scales
  4. Runs the model mmse_count times → MMSE image; posterior samples are NOT
     saved unless --num_posterior_samples > 0
  5. Writes uint16 TIFFs and APPENDS one row to metadata.csv immediately
     (a killed job leaves a usable partial result; rerunning resumes)

At the end writes metrics_summary.csv and two CellProfiler LoadData files in the
JUMP load_data_with_illum layout: one pointing the fluorescence channels at the
MMSE images (load_data_with_illum.csv) and one at the raw originals
(load_data_with_illum_original.csv); brightfield channels and illumination
functions point at the raw plate data in both.

Output:
  <output_dir>/<cell_line>/<plate_dir>/
      mmse/mmse_{well}_s{site:02d}_{channel}.tif
      metadata.csv, metrics_summary.csv
      load_data_with_illum.csv, load_data_with_illum_original.csv

Usage:
    python 4_predict.py --plate_dir /project/cell_paint_mono/2020_11_04_CPJUMP1/images/BR00117004__... \\
        --training_dir /project/cell_paint_mono/training_datasets/training_dataset_cpg0000_A549 \\
        --manifest /project/cell_paint_mono/splits/cpg0000-jump-pilot/A549/split_manifest.csv \\
        --output_dir /project/cell_paint_mono/predictions/cpg0000-jump-pilot --autocast
"""

import argparse
import gc
import os
import sys
import time

import pandas as pd
import torch

from microsplit_reproducibility.workflows.cellpainting import (
    discover_fovs,
    read_fov_channels,
    predict_fov,
    save_fov_predictions,
    compute_metrics,
    find_checkpoint,
    load_model_and_stats,
    load_manifest,
    manifest_rows_for_plate,
    append_metadata_row,
    completed_fovs,
    write_cellprofiler_load_data,
)

CHANNELS = ["DNA", "RNA", "ER", "AGP", "Mito"]
CHANNEL_MAPPING = {"DNA": 5, "RNA": 3, "ER": 4, "AGP": 2, "Mito": 1}
# Brightfield channels are not unmixed; CellProfiler reads them from the raw plate.
# Mapping verified against the paper's load_data_with_illum.csv.
BRIGHTFIELD_MAPPING = {"HighZBF": 6, "LowZBF": 7, "Brightfield": 8}
MANIFEST_META_KEYS = ["role", "modality", "timepoint", "time_h", "broad_sample", "gene",
                      "control_type", "pert_type", "cell_line"]


def parse_args():
    p = argparse.ArgumentParser(description="MicroSplit plate prediction for cpg0000-jump-pilot")
    p.add_argument("--plate_dir", required=True, help="Plate directory containing Images/")
    p.add_argument("--training_dir", required=True,
                   help="Training dataset dir (checkpoints/, noise_models/, training_stats.npz)")
    p.add_argument("--output_dir", required=True,
                   help="Root output dir; {cell_line}/{plate_dir} is created below it")
    p.add_argument("--manifest", default=None, help="split_manifest.csv from 0_split.py")
    p.add_argument("--cell_line", default=None,
                   help="Output subdirectory (default: from manifest; else none)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--mmse_count", type=int, default=50)
    p.add_argument("--num_posterior_samples", type=int, default=0)
    p.add_argument("--posterior_seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--batch_size", type=int, default=256, help="tiles per forward pass")
    p.add_argument("--autocast", action="store_true", help="bfloat16 autocast on CUDA")
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--max_fovs", type=int, default=0, help="Process only the first N FOVs (0 = all)")
    p.add_argument("--no_resume", action="store_true", help="Recompute FOVs that already exist")
    p.add_argument("--save_float32", action="store_true")
    p.add_argument("--metadata_file", default=None, help="experiment-metadata.tsv (plate-level fields)")
    p.add_argument("--batch", default=None, help="Batch name (default: from manifest)")
    p.add_argument("--data_dir", default="/project/cell_paint_mono",
                   help="Root containing {batch}/illum/{barcode}/ for the load_data CSVs")
    return p.parse_args()


def load_plate_metadata(metadata_file, batch, barcode):
    if not metadata_file or not os.path.isfile(metadata_file) or not batch:
        return {}
    df = pd.read_csv(metadata_file, sep="\t")
    m = df[(df["Batch"] == batch) & (df["Assay_Plate_Barcode"] == barcode)]
    if m.empty:
        return {}
    return {f"plate_{k}": v for k, v in m.iloc[0].to_dict().items()}


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    plate_dir = args.plate_dir.rstrip("/")
    plate_images_dir = os.path.join(plate_dir, "Images")
    if not os.path.isdir(plate_images_dir):
        print(f"ERROR: Images/ not found in {plate_dir}", file=sys.stderr)
        sys.exit(1)
    plate_name = os.path.basename(plate_dir)
    barcode = plate_name.split("__")[0]

    # ---- FOV list ---------------------------------------------------------
    man_rows = None
    batch = args.batch
    cell_line = args.cell_line
    if args.manifest:
        man = load_manifest(args.manifest)
        man_rows = manifest_rows_for_plate(man, plate_name)
        if man_rows.empty:
            print(f"ERROR: no test rows for {plate_name} in {args.manifest}", file=sys.stderr)
            sys.exit(1)
        batch = batch or man_rows.batch.iloc[0]
        cell_line = cell_line or man_rows.cell_line.iloc[0]
        fovs = [((r.well, int(r.site)), r.image_prefix) for r in man_rows.itertuples()]
        row_info = {(r.well, int(r.site)): r._asdict() for r in man_rows.itertuples()}
    else:
        fovs = discover_fovs(plate_images_dir)
        row_info = {}
    if not batch:
        batch = os.path.basename(os.path.dirname(os.path.dirname(plate_dir)))

    output_dir = os.path.join(args.output_dir, cell_line, plate_name) if cell_line \
        else os.path.join(args.output_dir, plate_name)
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.csv")

    plate_meta = load_plate_metadata(args.metadata_file, batch, barcode)
    plate_meta.setdefault("plate_name", plate_name)
    plate_meta.setdefault("barcode", barcode)
    plate_meta.setdefault("batch", batch)

    total = len(fovs)
    if args.max_fovs > 0:
        fovs = fovs[: args.max_fovs]
    done = set() if args.no_resume else completed_fovs(meta_path)
    if done:
        mmse_dir = os.path.join(output_dir, "mmse")
        def _complete(ws):
            return all(os.path.isfile(os.path.join(mmse_dir, f"mmse_{ws[0]}_s{ws[1]:02d}_{ch}.tif")) for ch in CHANNELS)
        done = {ws for ws in done if _complete(ws)}
    todo = [(ws, pre) for ws, pre in fovs if ws not in done]

    print(f"Plate:      {plate_name}  (batch {batch}, cell line {cell_line})")
    print(f"Output:     {output_dir}")
    print(f"Manifest:   {args.manifest or 'none (whole plate)'}")
    print(f"MMSE:       {args.mmse_count}  posterior samples saved: {args.num_posterior_samples}")
    print(f"Tiles/batch:{args.batch_size}  autocast: {args.autocast}")
    print(f"FOVs:       {total} listed, {len(done)} already done, {len(todo)} to process")

    if not todo:
        print("Nothing to do.")
    else:
        checkpoint = args.checkpoint or find_checkpoint(args.training_dir)
        print(f"Checkpoint: {checkpoint}")
        model, stats = load_model_and_stats(args.training_dir, checkpoint, channel_names=CHANNELS)

    t0 = time.time()
    for i, ((well, site), prefix) in enumerate(todo):
        ft0 = time.time()
        try:
            channel_images = read_fov_channels(plate_images_dir, prefix, CHANNEL_MAPPING)
        except Exception as e:
            print(f"  SKIP {well} s{site}: {e}")
            continue

        mmse_pred, posteriors = predict_fov(
            model=model, channel_images=channel_images, stats=stats, channel_names=CHANNELS,
            image_size=64, grid_size=args.grid_size, multiscale_lowres_count=3,
            mmse_count=args.mmse_count, num_posterior_samples=args.num_posterior_samples,
            posterior_seeds=tuple(args.posterior_seeds[: args.num_posterior_samples]),
            batch_size=args.batch_size, use_autocast=args.autocast,
        )
        save_fov_predictions(output_dir, well, site, mmse_pred, posteriors,
                             channel_names=CHANNELS, save_uint16=not args.save_float32)
        metrics = compute_metrics(channel_images, mmse_pred, CHANNELS)

        row = {**plate_meta, "well": well, "site": site, "image_prefix": prefix}
        info = row_info.get((well, site), {})
        for k in MANIFEST_META_KEYS:
            row[k] = info.get(k, "")
        row["mmse_files"] = ",".join(f"mmse/mmse_{well}_s{site:02d}_{ch}.tif" for ch in CHANNELS)
        row.update(metrics)
        row["seconds"] = round(time.time() - ft0, 1)
        append_metadata_row(meta_path, row)

        el = time.time() - t0
        eta = (len(todo) - i - 1) * el / (i + 1)
        psnr = " ".join(f"{ch}={metrics[f'psnr_{ch}']:.1f}" for ch in CHANNELS)
        print(f"[{i + 1}/{len(todo)}] {well} s{site} {row['seconds']:.0f}s | ETA {eta / 3600:.1f}h | {psnr}")
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # ---- summary + CellProfiler load_data ------------------------------------
    if os.path.isfile(meta_path) and os.path.getsize(meta_path) > 0:
        meta = pd.read_csv(meta_path)
        summary = [{"channel": ch, "mean_psnr": meta[f"psnr_{ch}"].mean(), "mean_ssim": meta[f"ssim_{ch}"].mean()}
                   for ch in CHANNELS]
        summary.append({"channel": "all_channels",
                        "mean_psnr": meta[[f"psnr_{ch}" for ch in CHANNELS]].values.mean(),
                        "mean_ssim": meta[[f"ssim_{ch}" for ch in CHANNELS]].values.mean()})
        pd.DataFrame(summary).to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)

        illum_dir = os.path.join(args.data_dir, batch, "illum", barcode)
        if not os.path.isdir(illum_dir):
            print(f"WARNING: illum dir {illum_dir} not found; load_data CSVs written without Illum columns")
            illum_dir = None
        rows = meta.to_dict("records")
        n1 = write_cellprofiler_load_data(rows, os.path.join(output_dir, "load_data_with_illum.csv"),
                                          barcode, plate_images_dir, CHANNEL_MAPPING, illum_dir=illum_dir,
                                          predicted_dir=output_dir, extra_raw_channels=BRIGHTFIELD_MAPPING,
                                          metadata_keys=MANIFEST_META_KEYS)
        n2 = write_cellprofiler_load_data(rows, os.path.join(output_dir, "load_data_with_illum_original.csv"),
                                          barcode, plate_images_dir, CHANNEL_MAPPING, illum_dir=illum_dir,
                                          predicted_dir=None, extra_raw_channels=BRIGHTFIELD_MAPPING,
                                          metadata_keys=MANIFEST_META_KEYS)
        print(f"metadata.csv: {len(meta)} FOVs | load_data rows: predicted {n1}, original {n2}")
    print(f"Done in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()

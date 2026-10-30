#!/usr/bin/env python3
"""
Create the cpg0039 train / validation / test datasets used by the MicroSplit
experiment, without MicroSplit, by downloading the images from the Cell Painting
Gallery (https://broadinstitute.github.io/cellpainting-gallery/).

The images in every split are exactly those of the MicroSplit workflow:

  train   FOVs the MicroSplit model is trained on
  val     FOVs used for validation during MicroSplit training
  test    FOVs held out and predicted (DILI and AgNP test sets)
  unused  training-pool FOVs that MicroSplit training sets aside and never uses

Only the images in these splits are downloaded (about 2,700 files, 6 GB), plus
the authors' illumination functions of the test plates (12 files, 50 MB), over
plain HTTPS from the public bucket; no AWS account or AWS CLI is needed. A rerun
skips files that are already complete, so an interrupted download can be resumed.

Only numpy, pandas and tifffile are needed. The script must stay in this folder:
it imports 0_split.py and reads test_wells_agnp.csv from here.

Output:
    {out_dir}/{split}/AO_Green/{index:06d}.tif     image taken with the GFP filter (DNA, RNA, cytoplasm)
    {out_dir}/{split}/AO_Red/{index:06d}.tif       image taken with the Propidium Iodide filter (acidic organelles)
    {out_dir}/{split}/combined/{index:06d}.tif     channel average, the MicroSplit input (skip with --no_combined)
    {out_dir}/test/illum/{plate}/*_Illum{AOGFP,AOPI}.npy   illumination functions of the test plates
    {out_dir}/metadata.csv                         one row per FOV, see below
    {out_dir}/split_manifest.csv                   the split manifest the datasets were built from
    {out_dir}/gallery/                             the downloaded platemaps

The image files carry no information beyond their pixels. metadata.csv is the
only link from an image to its experimental condition: which plate, well and
field it shows, how the cells were treated, which split it belongs to and which
original gallery file it is. One row per FOV (both channels of a FOV share the row):

    split            train, val, test or unused
    index            file number within the split: {split}/{channel}/{index:06d}.tif
    role             train, test_perturbation (treated well) or test_negcon (negative-control well)
    plate_id         plate barcode, e.g. 220607_092050_Plate_1
    cell_line        Huh7 for every image
    test_set         DILI or AgNP for test images, empty otherwise
    batch            experiment the plate belongs to (2022_06_07_DILI, 2021_10_08_AgNPViability,
                     2022_03_24_Acidification_AO)
    well             well on the 96-well plate, e.g. B10
    site             field of view within the well (1 to 25 for DILI, 1 to 16 otherwise)
    compound         treatment from the authors' platemap (e.g. Etoposide, AgNP, Non-treated)
    np_size_nm       silver nanoparticle diameter in nm, 40 or 100 (AgNP plates only; 0 for
                     Non-treated wells, empty on the other plates)
    concentration    dose from the authors' platemap. Units differ: AgNP plates in uM;
                     Acidification plates in mM for 3-methyladenine and uM for Bafilomycin A and
                     Chloroquine; the DILI platemap gives no unit
    control_type     negcon for negative-control wells, empty otherwise
    source_AO_Green  URL of the original gallery file of the AO_Green image
    source_AO_Red    URL of the original gallery file of the AO_Red image
    illum_AO_Green   test images only: the plate's illumination function for AO_Green, relative
                     to out_dir (deposited by the authors; CellProfiler divides the image by it)
    illum_AO_Red     the same for AO_Red

Usage:
    python datasets_only.py                          # writes to OUT_DIR below
    python datasets_only.py --out_dir /path/to/output
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Edit this: the directory the datasets are written to (needs about 12 GB: 6 GB of
# downloaded images, 6 GB of combined images; about 6 GB with --no_combined).
OUT_DIR = "/path/to/cpg0039_datasets"

HERE = Path(__file__).resolve().parent
CHANNELS = ["AO_Green", "AO_Red"]
BATCHES = ["2022_06_07_DILI", "2021_10_08_AgNPViability", "2022_03_24_Acidification_AO"]
BUCKET_URL = "https://cellpainting-gallery.s3.amazonaws.com"
GALLERY_PREFIX = "cpg0039-garcia-fossa-livecellpainting/unicamp"
ILLUM_SUFFIXES = {"AO_Green": "_IllumAOGFP.npy", "AO_Red": "_IllumAOPI.npy"}

# Settings of the MicroSplit workflow. Changing any of them gives different datasets.
SEED = 42                     # 0_split.py --seed
N_TRAIN = 300                 # 0_split.py --n_train
DILI_FIELDS = [1, 9, 17, 25]  # 0_split.py --dili_fields
VAL_FRACTION = 0.1            # 3_train.py --val_fraction
TEST_FRACTION = 0.1           # 3_train.py --test_fraction
MAX_VAL_FRAMES = 30           # 3_train.py --max_val_frames


def load_split_module():
    """Import 0_split.py, so the train / test split is computed by the same code as the workflow."""
    # The file name starts with a digit, so it cannot be imported with a plain import statement.
    spec = importlib.util.spec_from_file_location("split", HERE / "0_split.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_gallery(prefix: str) -> dict[str, int]:
    """All object keys under ``{GALLERY_PREFIX}/{prefix}`` with their sizes in bytes."""
    objects, token = {}, None
    while True:
        query = {"list-type": "2", "prefix": f"{GALLERY_PREFIX}/{prefix}"}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET_URL}/?{urllib.parse.urlencode(query)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        ns = {"s3": root.tag.split("}")[0].strip("{")}
        for item in root.findall("s3:Contents", ns):
            objects[item.find("s3:Key", ns).text] = int(item.find("s3:Size", ns).text)
        # The bucket returns at most 1,000 keys per request.
        token_node = root.find("s3:NextContinuationToken", ns)
        if token_node is None:
            return objects
        token = token_node.text


def download(key: str, target: Path, size: int | None = None, retries: int = 3) -> None:
    """Download one gallery object to ``target``, skipping it if a complete copy exists."""
    if target.exists() and (size is None or target.stat().st_size == size):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    url = f"{BUCKET_URL}/{urllib.parse.quote(key)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(partial, "wb") as f:
                shutil.copyfileobj(r, f)
            # Written to a .part file first, so an interrupted download is never mistaken for a complete one.
            os.replace(partial, target)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(5)


def fetch_gallery_metadata(source_dir: Path) -> tuple[dict, dict[str, int]]:
    """Download the platemaps and list the images of every batch.

    Returns the image file names per batch and plate (the input 0_split.py needs)
    and the size of every image object.
    """
    image_names, sizes = {}, {}
    for batch in BATCHES:
        for key in list_gallery(f"workspace/metadata/platemaps/{batch}/"):
            download(key, source_dir / key.removeprefix(f"{GALLERY_PREFIX}/"))
        objects = list_gallery(f"images/{batch}/images/")
        sizes.update(objects)
        plates: dict[str, list[str]] = {}
        for key in objects:
            plate, name = key.split("/")[-2:]
            plates.setdefault(plate, []).append(name)
        image_names[batch] = plates
        print(f"  {batch}: {len(plates)} plates, {len(objects)} files")
    return image_names, sizes


def training_frame_split(
    n_frames: int, val_fraction: float, test_fraction: float
) -> tuple[list[int], list[int], list[int]]:
    """Positions of the training dataset that MicroSplit trains, validates and does not use.

    Re-implements ``careamics.lvae_training.dataset.utils.data_utils.get_datasplit_tuples``
    (called by ``3_train.py``), so the result is identical without installing careamics.
    The training images are ordered as written by ``1_datasets.py``; the first
    ``(val_fraction + test_fraction) * n_frames`` of them are dealt out in chunks of
    up to 3, alternately to validation and to an internal test set that is never used.
    """
    test_val_size = int((val_fraction + test_fraction) * n_frames)
    train = list(range(test_val_size, n_frames))
    if test_val_size == 0:
        return train, [], []

    chunk = max(1, min(3, test_val_size // 2))
    val, test = [], []
    start = 0
    for i in range(test_val_size // chunk):
        (val if i % 2 == 0 else test).extend(range(start, start + chunk))
        start += chunk
    # The remainder after the last full chunk.
    if i % 2 == 0:
        test.extend(range(start, test_val_size))
    else:
        half = (test_val_size - start) // 2
        test.extend(range(start, start + half))
        val.extend(range(start + half, test_val_size))

    # balances the two sets 
    if val_fraction == 0:
        test, val = test + val, []
    elif test_fraction == 0:
        val, test = val + test, []
    elif test_fraction > val_fraction:
        n_move = int((test_fraction - val_fraction) * n_frames / 2)
        val = list(np.random.RandomState(seed=955).permutation(val))
        test, val = test + val[:n_move], val[n_move:]
    elif val_fraction > test_fraction:
        n_move = int((val_fraction - test_fraction) * n_frames / 2)
        test = list(np.random.RandomState(seed=955).permutation(test))
        val, test = val + test[:n_move], test[n_move:]
    return train, [int(v) for v in val], [int(t) for t in test]


def assign_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add the split of every manifest row: train, val, unused or test."""
    # 1_datasets.py writes the training rows in manifest order, as images 000000, 000001, ...
    # and 3_train.py splits them by that position.
    train_rows = manifest[manifest.role == "train"].reset_index(drop=True)
    train_pos, val_pos, _ = training_frame_split(len(train_rows), VAL_FRACTION, TEST_FRACTION)
    # 3_train.py validates on the first MAX_VAL_FRAMES validation images only.
    val_pos = val_pos[:MAX_VAL_FRAMES]

    # Everything not trained or validated on stays "unused".
    split = pd.Series("unused", index=train_rows.index)
    split[train_pos] = "train"
    split[val_pos] = "val"
    train_rows["split"] = split

    test_rows = manifest[manifest.role != "train"].assign(split="test")
    return pd.concat([train_rows, test_rows], ignore_index=True)


def write_datasets(
    rows: pd.DataFrame, out_dir: Path, sizes: dict[str, int], save_combined: bool, workers: int
) -> pd.DataFrame:
    """Download every FOV's images into its split folder and return the metadata table."""
    jobs, records = [], []
    for split, group in rows.groupby("split", sort=False):
        for index, row in enumerate(group.to_dict("records")):
            # Source files as paths relative to the dataset root on the gallery.
            sources = {ch: row[f"path_{ch}"] for ch in CHANNELS}
            for channel, source in sources.items():
                key = f"{GALLERY_PREFIX}/{source}"
                jobs.append((key, out_dir / split / channel / f"{index:06d}.tif", sizes[key]))
            records.append({
                "split": split, "index": index,
                **{k: v for k, v in row.items() if not k.startswith("path_") and k != "split"},
                **{f"source_{ch}": f"{BUCKET_URL}/{GALLERY_PREFIX}/{s}" for ch, s in sources.items()},
            })
        print(f"  {split:6s} {len(group):5d} FOVs")

    print(f"Downloading {len(jobs)} files ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(lambda job: download(*job), jobs), start=1):
            if i % 250 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")

    metadata = pd.DataFrame(records)
    if save_combined:
        for r in metadata.itertuples():
            # The MicroSplit input: the average of the two channels, float32.
            images = [tifffile.imread(out_dir / r.split / ch / f"{r.index:06d}.tif").astype(np.float32)
                      for ch in CHANNELS]
            target = out_dir / r.split / "combined" / f"{r.index:06d}.tif"
            target.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(target, np.mean(images, axis=0).astype(np.float32))
    return metadata


def download_illumination(test_rows: pd.DataFrame, out_dir: Path) -> dict[tuple[str, str], str]:
    """Download the authors' illumination functions of every test plate.

    CellProfiler divides each image by its plate's illumination function before
    segmentation. The authors deposited one function per plate and channel for the
    DILI and AgNP plates, i.e. every test plate. They are saved under their
    original names in ``{out_dir}/test/illum/{plate}/``.

    Returns
    -------
    dict
        (plate, channel) -> file path relative to ``out_dir``.
    """
    illum = {}
    for batch, plate in sorted(set(zip(test_rows.batch, test_rows.plate_id))):
        keys = list_gallery(f"images/{batch}/illum/{plate}/")
        for channel, suffix in ILLUM_SUFFIXES.items():
            # DILI files are named {plate}_IllumAOGFP.npy, AgNP files {plate}_Huh7_IllumAOGFP.npy.
            matches = [k for k in keys if k.endswith(suffix)]
            if len(matches) != 1:
                raise FileNotFoundError(f"expected one *{suffix} for plate {plate}, found {matches}")
            relative = Path("test") / "illum" / plate / matches[0].split("/")[-1]
            download(matches[0], out_dir / relative, keys[matches[0]])
            illum[(plate, channel)] = str(relative)
    return illum


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download and create the cpg0039 train / val / test datasets")
    ap.add_argument("--out_dir", default=OUT_DIR, help=f"Output directory (default: {OUT_DIR})")
    ap.add_argument("--no_combined", action="store_true",
                    help="Do not write the channel average (the MicroSplit input)")
    ap.add_argument("--workers", type=int, default=8, help="Parallel downloads")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.out_dir == "/path/to/cpg0039_datasets":
        raise SystemExit("Set OUT_DIR at the top of this script, or pass --out_dir.")

    # The platemaps go to {out_dir}/gallery, laid out as on the gallery, because 0_split.py reads them from there.
    source_dir = out_dir / "gallery"
    print("Listing the gallery and downloading platemaps ...")
    image_names, sizes = fetch_gallery_metadata(source_dir)

    split = load_split_module()
    manifest = split.build_manifest(
        source_dir,
        pd.read_csv(HERE / "test_wells_agnp.csv"),
        DILI_FIELDS,
        N_TRAIN,
        np.random.default_rng(SEED),
        image_names=image_names,
    )
    split.check_manifest(manifest)
    # Same row order as the split_manifest.csv written by 0_split.py, with paths relative to the gallery root.
    manifest = manifest.reset_index(drop=True)
    for ch in CHANNELS:
        manifest[f"path_{ch}"] = [os.path.relpath(p, source_dir) for p in manifest[f"path_{ch}"]]
    manifest.to_csv(out_dir / "split_manifest.csv", index=False)

    print(f"Writing datasets to {out_dir}")
    metadata = write_datasets(assign_splits(manifest), out_dir, sizes, not args.no_combined, args.workers)

    print("Downloading illumination functions of the test plates ...")
    illum = download_illumination(manifest[manifest.role != "train"], out_dir)
    is_test = metadata.split == "test"
    for ch in CHANNELS:
        metadata[f"illum_{ch}"] = ""
        metadata.loc[is_test, f"illum_{ch}"] = [illum[(p, ch)] for p in metadata.loc[is_test, "plate_id"]]
    metadata.to_csv(out_dir / "metadata.csv", index=False)


if __name__ == "__main__":
    main()

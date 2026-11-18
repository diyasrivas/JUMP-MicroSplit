"""
Steps 0 to 4 of the cpg0029 MicroSplit workflow, parameterised by dye set.

Each dye-set folder calls these functions through thin numbered scripts, so the
logic lives in one place and every dye set runs exactly the same code.
"""

import argparse
import datetime
import glob
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from .dye_sets import CELL_LINE, get_dye_set
from .split import build_manifest, summarize

TEST_FOVS_CSV = Path(__file__).resolve().parents[1] / "test_fovs.csv"
LOAD_DATA_META_KEYS = ("broad_sample", "compound", "moa", "control_type")
WANDB_ENTITY = "juglab"
WANDB_PROJECT = "JUMP-MicroSplit"


def _add_wandb_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--wandb_entity", default=WANDB_ENTITY)
    p.add_argument("--wandb_project", default=WANDB_PROJECT)
    p.add_argument("--no_wandb", action="store_true")


def _wandb_init(args, ds, job_type: str, name: str, run_dir: str, config: dict):
    """Start a W&B run in the ``cpg0029`` group, or return None with --no_wandb."""
    if args.no_wandb:
        return None
    import wandb

    os.makedirs(run_dir, exist_ok=True)
    return wandb.init(
        entity=args.wandb_entity, project=args.wandb_project, group="cpg0029", job_type=job_type,
        name=f"cpg0029_{ds.name}_{name}_{datetime.date.today():%Y%m%d}", dir=run_dir,
        config={"dye_set": ds.name, "barcode": ds.barcode, "channels": ds.channels, **config},
    )


# ---------------------------------------------------------------------------
# Step 0: split manifest
# ---------------------------------------------------------------------------

def split_main(name: str) -> None:
    ds = get_dye_set(name)
    ap = argparse.ArgumentParser(description=f"cpg0029 {ds.label}: split manifest")
    ap.add_argument("--test_fovs", default=str(TEST_FOVS_CSV))
    ap.add_argument("--n_train", type=int, default=300, help="training FOVs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=str(ds.split_dir))
    ap.add_argument("--no_verify", action="store_true", help="skip opening every selected TIFF")
    args = ap.parse_args()

    test_fovs = pd.read_csv(args.test_fovs)
    manifest, checks = build_manifest(ds, test_fovs, args.n_train, args.seed, verify=not args.no_verify)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "split_manifest.csv", index=False)
    summary = summarize(manifest, checks, ds, {"n_train": args.n_train, "seed": args.seed,
                                               "test_fovs": args.test_fovs})
    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"{ds.label} ({ds.barcode}) -> {out_dir}")
    for role in ("test_perturbation", "test_negcon", "train"):
        c = summary[role]
        print(f"  {role:18s} {c['fovs']:5d} FOVs on {c['wells']:3d} wells")
    for key in ("unreadable_fovs_dropped", "test_fovs_missing_on_plate"):
        if checks[key]:
            print(f"  WARNING {key}: {checks[key]}")


# ---------------------------------------------------------------------------
# Step 1: training dataset
# ---------------------------------------------------------------------------

def datasets_main(name: str) -> None:
    from microsplit_reproducibility.workflows.cellpainting import build_dataset_from_manifest

    ds = get_dye_set(name)
    ap = argparse.ArgumentParser(description=f"cpg0029 {ds.label}: training dataset")
    ap.add_argument("--manifest", default=str(ds.manifest_path))
    ap.add_argument("--output_dir", default=str(ds.training_dir))
    ap.add_argument("--max_fovs", type=int, default=0, help="only the first N training rows (testing)")
    args = ap.parse_args()

    print(f"Dye set:  {ds.label} ({ds.barcode})\nChannels: {ds.channels}")
    print(f"Manifest: {args.manifest}\nOutput:   {args.output_dir}")
    build_dataset_from_manifest(Path(args.manifest), CELL_LINE, ds.channels,
                                Path(args.output_dir), max_fovs=args.max_fovs)


# ---------------------------------------------------------------------------
# Step 2: noise model, one channel per call
# ---------------------------------------------------------------------------

def noisemodel_main(name: str) -> None:
    from microsplit_reproducibility.workflows.noise_model import train_noise_model_for_channel

    ds = get_dye_set(name)
    ap = argparse.ArgumentParser(description=f"cpg0029 {ds.label}: per-channel noise model")
    ap.add_argument("--channel", required=True, choices=ds.channels)
    ap.add_argument("--dataset_dir", default=str(ds.training_dir))
    ap.add_argument("--output_dir", default=None, help="default: {dataset_dir}/noise_models")
    _add_wandb_args(ap)
    args = ap.parse_args()

    output_dir = args.output_dir or os.path.join(args.dataset_dir, "noise_models")
    run = _wandb_init(args, ds, "noise_model", f"nm_{args.channel}", args.dataset_dir,
                      {"channel": args.channel, "dataset_dir": args.dataset_dir,
                       "output_dir": output_dir})
    t0 = time.time()
    train_noise_model_for_channel(channel=args.channel, dataset_dir=args.dataset_dir,
                                  output_dir=output_dir)
    if run is not None:
        run.summary["minutes"] = round((time.time() - t0) / 60, 1)
        run.summary["noise_model_file"] = os.path.join(output_dir, f"noise_model_{args.channel}.npz")
        run.finish()


# ---------------------------------------------------------------------------
# Step 3: training
# ---------------------------------------------------------------------------

def _train_args(ds) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"cpg0029 {ds.label}: MicroSplit training")
    p.add_argument("--dataset_dir", default=str(ds.training_dir))
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--check_val_every_n_epoch", type=int, default=1)
    p.add_argument("--early_stopping_patience", type=int, default=50)
    p.add_argument("--test_mode", action="store_true", help="2 epochs on 2 frames")
    p.add_argument("--compile", action="store_true", help="torch.compile the inner LadderVAE")
    p.add_argument("--train_grid_size", type=int, default=64)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--test_fraction", type=float, default=0.1)
    p.add_argument("--cache_size", type=int, default=64, help="LRU cache size in frames")
    p.add_argument("--max_patches_per_epoch", type=int, default=45000,
                   help="patches sampled per epoch with replacement; 0 = full grid")
    p.add_argument("--max_val_frames", type=int, default=30, help="0 = all")
    p.add_argument("--wandb_run_name", default=None,
                   help="default: {dataset_dir_basename}_{YYYYMMDD}")
    _add_wandb_args(p)
    return p.parse_args()


def train_main(name: str) -> None:
    import numpy as np
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from torch.utils.data import DataLoader, RandomSampler
    from careamics.lightning import VAEModule

    from microsplit_reproducibility.configs.data.JUMP import get_data_configs
    from microsplit_reproducibility.configs.factory import (
        create_algorithm_config, get_likelihood_config, get_loss_config, get_lr_scheduler_config,
        get_model_config, get_optimizer_config, get_training_config,
    )
    from microsplit_reproducibility.configs.parameters.JUMP import get_microsplit_parameters
    from microsplit_reproducibility.datasets.common import create_lazy_datasets
    from microsplit_reproducibility.utils.callbacks import get_callbacks

    ds = get_dye_set(name)
    args = _train_args(ds)
    channels = ds.channels
    torch.set_float32_matmul_precision("high")

    dataset_dir = args.dataset_dir
    noise_models_dir = os.path.join(dataset_dir, "noise_models")
    checkpoint_dir = os.path.join(dataset_dir, "checkpoints")

    missing = [f"no TIFFs in {os.path.join(dataset_dir, ch)}" for ch in channels + ["combined"]
               if not glob.glob(os.path.join(dataset_dir, ch, "*.tif*"))]
    missing += [f"missing noise model for {ch}" for ch in channels
                if not os.path.isfile(os.path.join(noise_models_dir, f"noise_model_{ch}.npz"))]
    if missing:
        for m in missing:
            print(f"ERROR: {m}", file=sys.stderr)
        sys.exit(1)

    num_epochs = 2 if args.test_mode else args.num_epochs
    check_val = 1 if args.test_mode else args.check_val_every_n_epoch
    print(f"Dye set:  {ds.label} ({ds.barcode})\nChannels: {channels}\nDataset:  {dataset_dir}")
    print(f"Epochs: {num_epochs}  batch: {args.batch_size}  grid: {args.train_grid_size}  "
          f"test_mode: {args.test_mode}")

    train_data_config, _, _ = get_data_configs(channel_idx_list=channels,
                                               train_grid_size=args.train_grid_size)
    experiment_params = get_microsplit_parameters(nm_path=noise_models_dir, channel_idx_list=channels,
                                                  batch_size=args.batch_size)
    train_dset, val_dset, _, data_stats = create_lazy_datasets(
        datapath=dataset_dir, channel_names=channels, train_grid_size=args.train_grid_size,
        val_grid_size=32, image_size=64,
        multiscale_lowres_count=train_data_config.multiscale_lowres_count,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        cache_size=args.cache_size, enable_rotation=train_data_config.train_aug_rotate,
    )

    mean_dict, std_dict = train_dset.get_mean_std()
    stats_path = os.path.join(dataset_dir, "training_stats.npz")
    np.savez(stats_path,
             mean_input=np.array(mean_dict["input"]), std_input=np.array(std_dict["input"]),
             mean_target=np.array(mean_dict["target"]), std_target=np.array(std_dict["target"]),
             max_val=np.array(train_dset.get_max_val()))
    print(f"Training stats saved: {stats_path}")

    if args.test_mode:
        train_dset.reduce_data([0, 1])
        val_dset.reduce_data([0])
    elif args.max_val_frames > 0 and val_dset.get_num_frames() > args.max_val_frames:
        val_dset.reduce_data(list(range(args.max_val_frames)))
    print(f"Train frames: {train_dset.get_num_frames()}  Val frames: {val_dset.get_num_frames()}")

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                         persistent_workers=args.num_workers > 0,
                         prefetch_factor=2 if args.num_workers > 0 else None)
    if args.max_patches_per_epoch > 0 and not args.test_mode:
        sampler = RandomSampler(train_dset, replacement=True, num_samples=args.max_patches_per_epoch)
        train_loader = DataLoader(train_dset, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dset, shuffle=False, **loader_kwargs)

    experiment_params["data_stats"] = data_stats
    gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(**experiment_params)
    training_config = get_training_config(**experiment_params)
    training_config.num_epochs = num_epochs
    experiment_config = create_algorithm_config(
        algorithm=experiment_params["algorithm"],
        loss_config=get_loss_config(**experiment_params),
        model_config=get_model_config(**experiment_params),
        gaussian_lik_config=gaussian_lik_config,
        nm_config=noise_model_config,
        nm_lik_config=nm_lik_config,
        lr_scheduler_config=get_lr_scheduler_config(**experiment_params),
        optimizer_config=get_optimizer_config(**experiment_params),
    )
    model = VAEModule(algorithm_config=experiment_config)
    if args.compile:
        model.model = torch.compile(model.model)

    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = get_callbacks(checkpoint_dir)
    for cb in callbacks:
        if hasattr(cb, "patience"):
            cb.patience = args.early_stopping_patience

    logger = True
    if not args.no_wandb:
        run_name = args.wandb_run_name or (
            f"{os.path.basename(os.path.normpath(dataset_dir))}_{datetime.date.today():%Y%m%d}")
        logger = WandbLogger(entity=args.wandb_entity, project=args.wandb_project, name=run_name,
                             save_dir=dataset_dir, group="cpg0029", job_type="train",
                             config={"dye_set": ds.name, "barcode": ds.barcode, "channels": channels})
        print(f"W&B run: {args.wandb_project}/{run_name}")

    trainer = Trainer(
        max_epochs=num_epochs, accelerator="gpu", enable_progress_bar=True, logger=logger,
        callbacks=callbacks, precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        gradient_clip_algorithm=training_config.gradient_clip_algorithm,
        check_val_every_n_epoch=check_val, default_root_dir=dataset_dir,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Training complete. Checkpoints: {checkpoint_dir}")


# ---------------------------------------------------------------------------
# Step 4: prediction + CellProfiler load_data
# ---------------------------------------------------------------------------

def write_load_data(ds, rows: list, output_dir: str) -> None:
    """Two LoadData CSVs over the predicted FOVs, in the authors' channel naming.

    ``load_data_with_illum.csv``          unmixed channels from the MMSE predictions
    ``load_data_with_illum_original.csv`` unmixed channels from the raw images
    Raw-only channels (brightfield, phase, ...) point at the raw images in both.
    Illum columns are added only if the plate's illumination functions are on disk.
    """
    from microsplit_reproducibility.workflows.cellpainting import write_cellprofiler_load_data

    illum_dir = str(ds.illum_dir) if ds.illum_dir.is_dir() else None
    if illum_dir is None:
        print(f"WARNING: {ds.illum_dir} not found; load_data CSVs written without Illum columns")
    for fname, predicted in (("load_data_with_illum.csv", output_dir),
                             ("load_data_with_illum_original.csv", None)):
        n = write_cellprofiler_load_data(
            rows, os.path.join(output_dir, fname), ds.barcode, str(ds.images_dir), ds.unmixed,
            illum_dir=illum_dir, predicted_dir=predicted, extra_raw_channels=ds.raw_only,
            metadata_keys=LOAD_DATA_META_KEYS,
        )
        print(f"{fname}: {n} rows")


def _log_prediction(run, meta, channels: list, hours: float) -> None:
    """Per-FOV PSNR / SSIM table and per-channel means of the whole plate."""
    import wandb

    if meta is not None and len(meta):
        metric_cols = [f"{m}_{ch}" for ch in channels for m in ("psnr", "ssim")]
        cols = [c for c in ["well", "site", "compound", "control_type", *metric_cols, "seconds"]
                if c in meta.columns]
        run.log({"per_fov_metrics": wandb.Table(dataframe=meta[cols])})
        for ch in channels:
            run.summary[f"mean_psnr_{ch}"] = float(meta[f"psnr_{ch}"].mean())
            run.summary[f"mean_ssim_{ch}"] = float(meta[f"ssim_{ch}"].mean())
        run.summary["fovs_predicted"] = len(meta)
    run.summary["hours_this_job"] = round(hours, 2)
    run.finish()


def predict_main(name: str) -> None:
    import torch
    from microsplit_reproducibility.workflows.cellpainting import (
        find_checkpoint, load_manifest, load_model_and_stats, manifest_rows_for_plate,
        predict_manifest_fovs,
    )

    ds = get_dye_set(name)
    p = argparse.ArgumentParser(description=f"cpg0029 {ds.label}: MicroSplit prediction")
    p.add_argument("--manifest", default=str(ds.manifest_path))
    p.add_argument("--training_dir", default=str(ds.training_dir))
    p.add_argument("--output_dir", default=str(ds.prediction_dir))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--mmse_count", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256, help="tiles per forward pass")
    p.add_argument("--autocast", action="store_true", help="bfloat16 autocast on CUDA")
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--max_fovs", type=int, default=0, help="only the first N FOVs (0 = all)")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--save_float32", action="store_true")
    p.add_argument("--load_data_only", action="store_true",
                   help="skip prediction, only (re)write the load_data CSVs")
    _add_wandb_args(p)
    args = p.parse_args()
    if args.load_data_only:
        args.no_wandb = True
    torch.set_float32_matmul_precision("high")

    rows = manifest_rows_for_plate(load_manifest(args.manifest), ds.barcode, plate_column="plate_id")
    if rows.empty:
        print(f"ERROR: no test rows for {ds.barcode} in {args.manifest}", file=sys.stderr)
        sys.exit(1)
    if args.max_fovs > 0:
        rows = rows.head(args.max_fovs)

    if not args.load_data_only:
        checkpoint = args.checkpoint or find_checkpoint(args.training_dir)
        print(f"Dye set:    {ds.label} ({ds.barcode})\nChannels:   {ds.channels}")
        print(f"Output:     {args.output_dir}\nCheckpoint: {checkpoint}")
        print(f"MMSE: {args.mmse_count}  tiles/batch: {args.batch_size}  autocast: {args.autocast}")
        model, stats = load_model_and_stats(args.training_dir, checkpoint, channel_names=ds.channels)
        run = _wandb_init(args, ds, "predict", "predict", args.output_dir, {
            "checkpoint": checkpoint, "mmse_count": args.mmse_count, "batch_size": args.batch_size,
            "autocast": args.autocast, "grid_size": args.grid_size, "test_fovs": len(rows),
            "output_dir": args.output_dir,
        })
        t0 = time.time()
        meta = predict_manifest_fovs(
            rows, model, stats, ds.channels, args.output_dir,
            mmse_count=args.mmse_count, batch_size=args.batch_size, use_autocast=args.autocast,
            grid_size=args.grid_size, save_float32=args.save_float32, resume=not args.no_resume,
        )
        hours = (time.time() - t0) / 3600
        print(f"Prediction done in {hours:.2f} h")
        if run is not None:
            _log_prediction(run, meta, ds.channels, hours)

    meta_path = os.path.join(args.output_dir, "metadata.csv")
    if not os.path.isfile(meta_path):
        print(f"ERROR: {meta_path} not found", file=sys.stderr)
        sys.exit(1)
    done = pd.read_csv(meta_path, usecols=["well", "site"])
    done = set(zip(done.well.astype(str), done.site.astype(int)))
    predicted_rows = [r for r in rows.to_dict("records") if (str(r["well"]), int(r["site"])) in done]
    if len(predicted_rows) < len(rows):
        print(f"WARNING: {len(rows) - len(predicted_rows)} of {len(rows)} test FOVs not predicted yet")
    write_load_data(ds, predicted_rows, args.output_dir)

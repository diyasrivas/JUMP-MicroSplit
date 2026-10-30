# cpg0036-EU-OS-bioactives

MicroSplit channel-unmixing experiment for the EU-OPENSCREEN Bioactive compound
Cell Painting dataset (Wolff et al., *iScience* 2025).

## Dataset details

| Property | Value |
|----------|-------|
| Paper | Wolff et al., *iScience* 28, 112445 (2025) |
| Compounds | 2,464 EU-OPENSCREEN Bioactive compounds @ 10 μM |
| Cell lines | **Hep G2** (primary, 4 imaging sites) and **U-2 OS** (FMP site only) |
| Plates | 7 × 384-well plates, 4 replicates |
| Fields | 9 per well |
| Microscope | Opera Phenix™ (spinning disk, 20× water, NA 1.0, 2× binning) |

## Channel layout — 4 channels (not standard 5-channel Cell Painting)

Two channels each capture **two spectrally-overlapping dyes** — this is the
MicroSplit unmixing task for this dataset.

| Channel | TIFF index | Wavelength | Stains |
|---------|-----------|------------|--------|
| Mito | ch1 | 650–760 nm | MitoTracker Deep Red |
| Actin_AGP | ch2 | 570–630 nm | WGA Alexa 555 (Golgi/PM) **+** Phalloidin Alexa 568 (F-actin) |
| ER_RNA | ch3 | 500–550 nm | Concanavalin A Alexa 488 (ER) **+** SYTO 14 (RNA) |
| DNA | ch4 | 435–480 nm | Hoechst 33342 (nucleus) |

Channel indices follow the Opera Phenix convention (longest → shortest emission
wavelength, same as cpg0000). Verify against actual TIFF filenames before the
first run if in doubt.

> **Key point:** There is no separate RNA or AGP channel. RNA signal is
> co-acquired with ER in the 500–550 nm window; actin/AGP signals are
> co-acquired in the 570–630 nm window. MicroSplit unmixes these 4 spectral
> channels from their pixel-sum (combined) image.

## Cell lines — train separately!

Hep G2 and U-2 OS respond differently to compounds. Train one MicroSplit model
per cell line; predict only on plates imaged with the same cell line.

**Site / cell-line mapping (from the paper):**
- **FMP** (Berlin): imaged both **Hep G2** and **U-2 OS**
- **IMTM**, **MEDINA**, **USC**: imaged **Hep G2** only

## Workflow

```
1_datasets.py / datasets.sh      ← build dataset  (one job per cell line)
2_noisemodels.py / noise_models.sh ← 4 noise models in parallel
3_train.py / train.sh            ← train MicroSplit model
4_predict.py / predict.sh        ← plate-level prediction
```

### HPC quick start

```bash
# Step 1: Build datasets for both cell lines
DS_HEPG2=$(./datasets.sh --cell_line HepG2 --slurm)
DS_U2OS=$(./datasets.sh  --cell_line U2OS  --slurm)

# Step 2: Train noise models (4 parallel jobs per cell line)
NM_HEPG2=$(./noise_models.sh \
    --dataset_dir .../training_dataset_cpg0036_HepG2 \
    --slurm --dependency "${DS_HEPG2}")
NM_U2OS=$(./noise_models.sh \
    --dataset_dir .../training_dataset_cpg0036_U2OS \
    --slurm --dependency "${DS_U2OS}")

# Step 3: Train models
TR_HEPG2=$(./train.sh --dataset_dir .../training_dataset_cpg0036_HepG2 \
    --slurm --dependency "${NM_HEPG2}")
TR_U2OS=$(./train.sh  --dataset_dir .../training_dataset_cpg0036_U2OS  \
    --slurm --dependency "${NM_U2OS}")

# Step 4: Predict (always use the matching cell-line model)
./predict.sh --batch 2021_XX_XX_Batch1 \
    --training_dir .../training_dataset_cpg0036_HepG2 \
    --slurm --dependency "${TR_HEPG2}"
```

### Isolating plates by cell line

If both cell lines share a single `data_dir`, use `--plate_filter` with a
substring present in the plate name to restrict sampling:

```bash
./datasets.sh --cell_line HepG2 --plate_filter FMP_HepG2
./datasets.sh --cell_line U2OS  --plate_filter FMP_U2OS
```

## Output structure

```
training_dataset_cpg0036_{cell_line}/
    combined/        float32 sum of all 4 channels
    DNA/ ER_RNA/ Actin_AGP/ Mito/    uint16 per-channel images
    noise_models/    noise_model_{channel}.npz  (4 files)
    checkpoints/     model_*.ckpt + last.ckpt
    training_stats.npz
    metadata.csv

predictions/cpg0036-EU-OS-bioactives/{plate}/
    mmse/            mmse_{well}_s{site:02d}_{channel}.tif
    sample_0/ ...    posterior samples
    metadata.csv
    metrics_summary.csv
```

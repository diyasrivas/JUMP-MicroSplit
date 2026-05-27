# cpg0039-garcia-fossa-livecellpainting

MicroSplit channel-unmixing experiment for the Live Cell Painting dataset of
Garcia-Fossa et al. (2025) using Acridine Orange.

## Dataset details

| Property | Value |
|----------|-------|
| Paper | Garcia-Fossa et al. (2025) |
| Dye | Acridine Orange (AO) — single metachromatic dye |
| Cell lines | **Huh-7, MCF-7, PNT1A, PC-3** (train separately!) |
| Plates | 96-well plates, 16 sites per well |
| Microscope | Cytation 5 BioTek (20× objective, NA 0.45) |
| Image size | 1224 × 904 pixels, 16-bit |

## Channel layout — 2 channels only (NOT standard Cell Painting)

This dataset uses a single Acridine Orange (AO) metachromatic dye, which
produces **2 spectrally distinct signals** depending on the cellular
environment. There are **no DNA/RNA/ER/AGP/Mito channels**.

| Channel | TIFF index | Filter cube | Ex/Em | Biology |
|---------|-----------|-------------|-------|---------|
| AO_Red | ch1 | PI | 531/647 nm | Acidic organelles (lysosomes, lipid droplets) |
| AO_Green | ch2 | GFP | 469/525 nm | DNA + RNA + cytoplasm |

Channel indices follow longest-to-shortest convention. **Verify against actual
TIFF files before running** — the Cytation 5 BioTek file naming and channel
numbering differs from Opera Phenix.

> **Key point:** The two AO channels are spectrally distinct emissions from a
> single dye. MicroSplit learns to unmix the combined AO signal into the
> individual spectral components.

## Cell lines — train separately!

Huh-7, MCF-7, PNT1A, and PC-3 cells have different morphologies and AO staining
patterns. Train one MicroSplit model per cell line and predict only on plates
from the same cell line.

## File naming — Cytation 5 BioTek

The Cytation 5 Gen5 software uses a different file naming convention than
Opera Phenix. `1_datasets.py` contains two auto-detected patterns; if neither
matches the actual cpg0039 TIFF files, adjust `discover_fovs_cpg0039()` in
`1_datasets.py` to match the actual naming. Check a sample `Images/` directory:

```bash
ls /project/cell_paint_mono/cpg0039-garcia-fossa-livecellpainting/{batch}/images/{plate}/Images/ | head -20
```

## Workflow

```
1_datasets.py / datasets.sh      ← build dataset  (one job per cell line)
2_noisemodels.py / noise_models.sh ← 2 noise models in parallel
3_train.py / train.sh            ← train MicroSplit model
4_predict.py / predict.sh        ← plate-level prediction
```

### HPC quick start

```bash
# Step 1: Build datasets for all 4 cell lines
DS_HUH7=$(./datasets.sh  --cell_line Huh_7  --slurm)
DS_MCF7=$(./datasets.sh  --cell_line MCF_7  --slurm)
DS_PNT1A=$(./datasets.sh --cell_line PNT1A  --slurm)
DS_PC3=$(./datasets.sh   --cell_line PC_3   --slurm)

# Step 2: Train noise models (2 parallel jobs per cell line)
NM_HUH7=$(./noise_models.sh \
    --dataset_dir .../training_dataset_cpg0039_Huh_7 \
    --slurm --dependency "${DS_HUH7}")

# Step 3: Train model
TR_HUH7=$(./train.sh \
    --dataset_dir .../training_dataset_cpg0039_Huh_7 \
    --slurm --dependency "${NM_HUH7}")

# Step 4: Predict (use matching cell-line model)
./predict.sh --batch 2021_XX_Batch1 \
    --training_dir .../training_dataset_cpg0039_Huh_7 \
    --slurm --dependency "${TR_HUH7}"
```

## Output structure

```
training_dataset_cpg0039_{cell_line}/
    combined/        float32 sum of AO_Green + AO_Red
    AO_Green/        uint16 individual channel images
    AO_Red/
    noise_models/    noise_model_AO_Green.npz, noise_model_AO_Red.npz
    checkpoints/     model_*.ckpt + last.ckpt
    training_stats.npz
    metadata.csv

predictions/cpg0039-garcia-fossa-livecellpainting/{plate}/
    mmse/            mmse_{well}_s{site:02d}_{channel}.tif
    sample_0/ ...    posterior samples
    metadata.csv
    metrics_summary.csv
```

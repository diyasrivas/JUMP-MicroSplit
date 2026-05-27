# cpg0000-jump-pilot

MicroSplit workflow for the **JUMP-CP pilot dataset** (cpg0000), covering two cell lines:
- **A549** — lung carcinoma
- **U2OS** — osteosarcoma

## Dataset

- Source: [Cell Painting Gallery cpg0000-jump-pilot](https://github.com/broadinstitute/cellpainting-gallery/blob/main/cpg0000-jump-pilot/README.md)
- Local path: `/project/cell_paint_mono/cpg0000-jump-pilot/`
- Channel mapping (5-channel Opera Phenix):

| Channel | File index |
|---------|-----------|
| DNA     | ch5       |
| RNA     | ch3       |
| ER      | ch4       |
| AGP     | ch2       |
| Mito    | ch1       |

## Workflow

Run steps in order. Each step can be submitted as a SLURM job (add `--slurm`).

### Step 1 — Build training dataset
```bash
./datasets.sh --cell_lines "A549 U2OS" --slurm
# or for a single cell line:
./datasets.sh --cell_lines "A549" --slurm
```

Reads plate images from `experiment-metadata.tsv`, stratifies by cell line and plate,
samples ~3500 FOVs per cell line, writes to:
```
/project/cell_paint_mono/training_datasets/training_dataset_{A549,U2OS}/
    combined/        float32 pixel-wise sum
    DNA/ RNA/ ER/ AGP/ Mito/   uint16 per-channel images
    metadata.csv     per-FOV metadata
```

### Step 2 — Train noise models (5 channels in parallel)
```bash
./noise_models.sh --cell_lines "A549 U2OS" --slurm --dependency "job1:job2"
```
Trains a Noise2Void + GaussianMixture noise model per channel, saved as
`noise_models/noise_model_{channel}.npz`.

### Step 3 — Train MicroSplit model
```bash
./train.sh --cell_lines "A549 U2OS" --slurm --dependency "nm_job1:nm_job2"
# Quick sanity check (2 epochs):
./train.sh --cell_lines "A549" --test
```
Trains the LadderVAE with GMM noise model. Saves checkpoints and `training_stats.npz`.

### Step 4 — Predict (per plate)
```bash
./predict.sh --cell_line A549 --batch 2020_11_04_CPJUMP1 --slurm
# Single plate:
./predict.sh --cell_line A549 --plate_dir /project/cell_paint_mono/cpg0000-jump-pilot/.../plate_dir
```
Tiles each FOV, runs MMSE prediction + 3 posterior samples, writes uint16 TIFFs.

## Output structure
```
/project/cell_paint_mono/predictions/cpg0000-jump-pilot/{plate_name}/
    mmse/                  MMSE predictions (uint16 .tif)
    sample_0/              Posterior sample 0
    sample_1/              Posterior sample 1
    sample_2/              Posterior sample 2
    metadata.csv           Per-FOV metrics (PSNR/SSIM per channel)
    metrics_summary.csv    Mean PSNR/SSIM per channel
```

## Dependencies

All heavy jobs use the `microsplit_jobs` conda environment on the HPC cluster.
PYTHONPATH is set automatically in the bash scripts to use the local source.

## SLURM chaining example

```bash
# 1. Build datasets (CPU)
DS_JOBS=$(./datasets.sh --slurm)

# 2. Noise models (GPU, after datasets)
NM_JOBS=$(./noise_models.sh --slurm --dependency "${DS_JOBS}")

# 3. Train (GPU, after noise models)
TRAIN_JOBS=$(./train.sh --slurm --dependency "${NM_JOBS}")

# 4. Predict A549 (GPU, after training)
./predict.sh --cell_line A549 --batch 2020_11_04_CPJUMP1 --slurm --dependency "${TRAIN_JOBS}"
```

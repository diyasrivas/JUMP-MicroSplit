# cpg0029-chroma-pilot

MicroSplit workflow for the **cpg0029 ChromaLive pilot dataset** — fixed U2OS cells imaged
with 3 dye conditions to characterise ChromaLive reagents.

## Dataset

- Source: [Cell Painting Gallery cpg0029](https://github.com/broadinstitute/cellpainting-gallery)
- Local path: `/project/cell_paint_mono/cpg0029-chroma-pilot/images/`
- 7 plates × 384 wells × 9 fields = ~24,192 FOVs
- Channel mapping (8-channel Opera Phenix):

| Channel    | File index | Notes                  |
|------------|-----------|------------------------|
| Phase      | ch1       | not used               |
| Brightfield| ch2       | not used               |
| RNA        | ch3       | SYTO 14                |
| ER         | ch4       | Concanavalin A         |
| AGP        | ch5       | WGA + Phalloidin       |
| Mito       | ch6       | MitoTracker            |
| DNA        | ch7       | Hoechst 33342          |
| Actin      | ch8       | PhenoVue 400LS (new)   |

## Dye conditions

| Plate barcode | Dye condition     |
|---------------|------------------|
| BR00122244    | standard_cp       |
| BR00122246    | standard_cp       |
| BR00122250    | standard_cp       |
| BR00122247    | alt_mito          |
| BR00122245    | post_chromalive   |
| BR00122248    | post_chromalive   |
| BR00122249    | post_chromalive   |

## Workflow

Run steps in order. Each step can be submitted as a SLURM job (add `--slurm`).

### Step 1 — Build training dataset
```bash
./datasets.sh --slurm
```

Stratified-samples ~3500 FOVs across all 7 plates, holds out 30% of wells per plate
for evaluation. Writes to:
```
/project/cell_paint_mono/training_datasets/training_dataset_cpg0029/
    combined/        float32 pixel-wise sum of 5 channels
    DNA/ RNA/ ER/ AGP/ Mito/   uint16 per-channel images
    metadata.csv     per-FOV metadata (includes dye_condition)
    test_wells.csv   held-out wells for evaluation
```

### Step 2 — Train noise models (5 channels in parallel)
```bash
./noise_models.sh --slurm --dependency "DATASET_JOB_ID"
```

### Step 3 — Train MicroSplit model
```bash
./train.sh --slurm --dependency "NM_JOB1:NM_JOB2:..."
# Quick sanity check (2 epochs, 1h):
./train.sh --test
```

### Step 4 — Predict (per plate)
```bash
./predict.sh --plate_dir /project/cell_paint_mono/cpg0029-chroma-pilot/images/2023_05_15_Batch1/BR00122246__2023-04-01T03_13_00-Measurement1 --slurm
```

## Output structure
```
/project/cell_paint_mono/predictions/cpg0029-chroma-pilot/{plate_name}/
    mmse/                  MMSE predictions (uint16 .tif)
    sample_0/              Posterior sample 0
    sample_1/              Posterior sample 1
    sample_2/              Posterior sample 2
    metadata.csv           Per-FOV metrics (PSNR/SSIM per channel, dye_condition)
    metrics_summary.csv    Mean PSNR/SSIM per channel
```

## SLURM chaining example

```bash
DS_JOB=$(./datasets.sh --slurm)
NM_JOBS=$(./noise_models.sh --slurm --dependency "${DS_JOB}")
TRAIN_JOB=$(./train.sh --slurm --dependency "${NM_JOBS}")
./predict.sh --batch 2023_05_15_Batch1 --slurm --dependency "${TRAIN_JOB}"
```

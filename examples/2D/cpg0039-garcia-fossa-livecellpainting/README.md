# cpg0039-garcia-fossa-livecellpainting

MicroSplit on the Live Cell Painting dataset of Garcia-Fossa et al. (*Mol. Biol. Cell* 2025). Huh-7 cells are stained with a single dye, Acridine Orange (AO), and imaged in two channels. The two channels are averaged into one image and MicroSplit predicts the two channels back. One model.

## Data

- Raw images: `/project/cell_paint_mono/cpg0039-garcia-fossa-livecellpainting/unicamp/images/{batch}/images/{plate}/`. Cytation 5, 20x 0.45 NA, 904 x 1224 pixels, 96-well plates.
- File names: `{well}_{n}_{channel_index}_{field}_{filter}_001.tif`. Channels are read by filter name:

| Channel | Filter | Signal |
|---|---|---|
| AO_Green | `GFP` | DNA, RNA, cytoplasm |
| AO_Red | `Propidium Iodide` | acidic organelles |

| Batch | Plates | Fields per well | Use |
|---|---|---|---|
| `2022_06_07_DILI` | 3 | 25 | test (Figure 4) |
| `2021_10_08_AgNPViability` | 3 | 16 | test (Ag100 and Non-treated wells, Figure 3) and train (other wells) |
| `2022_03_24_Acidification_AO` | 4 | 16 | train (AO-stained wells) |

Only the three 2022 DILI plates have deposited images; the paper's Figure 4 also includes two 2023 plates.

## Split

| | Wells | Fields | FOVs |
|---|---|---|---|
| Test, DILI | every imaged well (177) | 1, 9, 17, 25 | 708 |
| Test, AgNP | `test_wells_agnp.csv`: all Ag100 wells (8 doses x 9) and Non-treated wells (18) | 1, 6, 11, 16 | 360 |
| Train | the other 90 AgNP wells and the 69 AO wells of the Acidification_AO plates | one or two random fields per well | 300 |

The DILI experiment is never used for training, and `0_split.py` asserts that no training well is a test well.

Outputs:

```
/project/cell_paint_mono/splits/cpg0039-garcia-fossa-livecellpainting/Huh7/split_manifest.csv, split_summary.json
/project/cell_paint_mono/training_datasets/training_dataset_cpg0039_Huh7/
/project/cell_paint_mono/predictions/cpg0039-garcia-fossa-livecellpainting/Huh7/{plate_id}/
```

## Workflow

```bash
# 0+1  split manifest + training dataset (CPU, minutes)
./datasets.sh

# 2    noise models, one GPU job per channel
D=/project/cell_paint_mono/training_datasets/training_dataset_cpg0039_Huh7
./noise_models.sh --dataset_dir ${D} --slurm

# 3    training (100 epochs of 45,000 patches)
./train.sh --dataset_dir ${D} --slurm

# 4a   timing benchmark: 1 FOV interactively on a GPU node
./predict.sh --benchmark

# 4b   prediction of the test FOVs, one job per plate, MMSE-50, no posterior samples
./predict.sh --slurm
```

Prediction output per plate: `mmse/mmse_{well}_s{site:02d}_{channel}.tif`, `metadata.csv` (one row per FOV with the manifest fields, raw image paths and PSNR / SSIM; appended as each FOV finishes, so rerunning a plate resumes) and `metrics_summary.csv`.

## Scripts

| File | Role |
|---|---|
| `0_split.py` | builds the split manifest from the image files, the platemaps and `test_wells_agnp.csv` |
| `1_datasets.py` | writes the training dataset from the `train` rows of the manifest |
| `2_noisemodels.py` | one N2V + GMM noise model per channel |
| `3_train.py` | LadderVAE training; `--max_patches_per_epoch` (45,000) caps the epoch, `--max_val_frames` (30) caps validation |
| `4_predict.py` | per-plate prediction of the test rows of the manifest |
| `*.sh` | SLURM launchers for each step |
| `test_wells_agnp.csv` | the AgNP test wells |
| `deprecated/` | earlier versions of the scripts, kept for reference |

# cpg0036-EU-OS-bioactives

MicroSplit on the EU-OPENSCREEN Bioactives Cell Painting dataset (Wolff et al., *iScience* 2025). Hep G2 was imaged at four partner sites (FMP, IMTM, MEDINA, USC) and U-2 OS at FMP only. One model per cell line; the Hep G2 model is trained on all four sites.

## Data

- Raw images: `/project/cell_paint_mono/cpg0036-EU-OS-bioactives/{FMP,IMTM,MEDINA,USC}/images/`. Seven library plates (B1001 to B1007) in four replicates (R1 to R4), 384 wells, 9 fields per well.
- Four channels. Two of them hold two co-acquired dyes each:

| Channel | Dyes |
|---|---|
| DNA | Hoechst 33342 |
| ER_RNA | Concanavalin A Alexa 488 + SYTO 14 |
| Actin_AGP | WGA Alexa 555 + Phalloidin Alexa 568 |
| Mito | MitoTracker Deep Red |

The four channels are averaged into one image and MicroSplit predicts the four channels back.

The TIFF channel order differs between sites and between the two FMP cell lines, and IMTM uses its own file naming (1000 x 1000 pixels; the other sites are 1080 x 1080). Image paths are therefore read from the authors' CellProfiler `load_data.csv` files (`{site}/workspace/load_data_csv/`), which list the file of every channel for every field.

## Split

| | Wells | Fields | FOVs |
|---|---|---|---|
| Test | `test_wells.csv`: library plate B1001, R1 to R4, every site and cell line; 40 compounds (stratified over the paper's toxic / active / low-active classes), all Tetrandrine and Nocodazole wells, 12 DMSO wells per plate | 2, 4, 6, 8 | 3,563 Hep G2, 852 U-2 OS |
| Train | library plates B1002 to B1007; positive-control wells excluded | one random field per well | 300 per cell line |

Hep G2 training FOVs are split equally over the four sites (75 each) and evenly over each site's 24 plates. B1001 is never used for training, and `0_split.py` asserts that no training well is a test well. One test field (IMTM B1001_R4, well P08, field 2) is not in the deposited data and is skipped.

Training images are centre-cropped to 1000 x 1000 pixels so that all sites share one image size. Prediction runs on the full images.

Outputs:

```
/project/cell_paint_mono/splits/cpg0036-EU-OS-bioactives/{cell_line}/split_manifest.csv, split_summary.json
/project/cell_paint_mono/training_datasets/training_dataset_cpg0036_allsites_{cell_line}/
/project/cell_paint_mono/predictions/cpg0036-EU-OS-bioactives/{cell_line}/{plate_id}/
```

## Workflow

```bash
# 0+1  split manifests + training datasets (CPU)
./datasets.sh --cell_lines "HepG2 U2OS"

# 2    noise models, one GPU job per channel
D=/project/cell_paint_mono/training_datasets
./noise_models.sh --dataset_dir ${D}/training_dataset_cpg0036_allsites_HepG2 --slurm
./noise_models.sh --dataset_dir ${D}/training_dataset_cpg0036_allsites_U2OS --slurm

# 3    training (100 epochs of 45,000 patches)
./train.sh --dataset_dir ${D}/training_dataset_cpg0036_allsites_HepG2 --slurm
./train.sh --dataset_dir ${D}/training_dataset_cpg0036_allsites_U2OS --slurm

# 4a   timing benchmark: 1 FOV interactively on a GPU node
./predict.sh --cell_line HepG2 --benchmark

# 4b   prediction of the test FOVs, one job per plate, MMSE-50, no posterior samples
./predict.sh --cell_line HepG2 --slurm
./predict.sh --cell_line U2OS --slurm
```

Prediction output per plate: `mmse/mmse_{well}_s{site:02d}_{channel}.tif`, `metadata.csv` (one row per FOV with the manifest fields, raw image paths and PSNR / SSIM; appended as each FOV finishes, so rerunning a plate resumes) and `metrics_summary.csv`.

## Scripts

| File | Role |
|---|---|
| `0_split.py` | builds the split manifests from `test_wells.csv`, the platemaps and the authors' `load_data.csv` files |
| `1_datasets.py` | writes the training dataset from the `train` rows of the manifest |
| `2_noisemodels.py` | one N2V + GMM noise model per channel |
| `3_train.py` | LadderVAE training; `--max_patches_per_epoch` (45,000) caps the epoch, `--max_val_frames` (30) caps validation |
| `4_predict.py` | per-plate prediction of the test rows of the manifest |
| `*.sh` | SLURM launchers for each step |
| `test_wells.csv` | the test wells |
| `deprecated/` | earlier versions of the scripts, kept for reference |

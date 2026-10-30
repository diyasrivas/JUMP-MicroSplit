# cpg0000-jump-pilot

MicroSplit on the **JUMP-CP pilot** (Chandrasekaran et al., Nat. Methods 2024), two cell lines (A549, U2OS), one model per cell line. Design rationale and all numbers: [`docs/cpg0000_dataset_design.md`](../../../docs/cpg0000_dataset_design.md).

## Data

- Raw plates: `/project/cell_paint_mono/2020_11_04_CPJUMP1/images/{plate_dir}/Images/`, 8 channels (ch1 Mito, ch2 AGP, ch3 RNA, ch4 ER, ch5 DNA; ch6–8 brightfield, not unmixed). Illumination functions in `…/2020_11_04_CPJUMP1/illum/{barcode}/`.
- Paper repo (platemaps, target annotations, published profiles, CellProfiler pipeline): `/project/cell_paint_mono/2024_Chandrasekaran_NatureMethods/`.
- Only the 40 benchmark plates of the paper's Figures 4–5 are used (20 per cell line: compound 24/48 h, CRISPR 96/144 h, ORF 48/96 h).

## Split design (one line)

Test = 10 genes → every compound / both guides / the ORF targeting them, all replicate wells on all 20 plates, plus 16 DMSO, 8 negcon-guide and 8 negcon-ORF wells per plate, at sites 2, 4, 6, 8. Train = 15 FOVs per plate from the remaining wells, 1 random site each (300 per cell line). The manifest is the single source of truth:

```
/project/cell_paint_mono/splits/cpg0000-jump-pilot/
    test_genes.csv
    {A549,U2OS}/split_manifest.csv      one row per (plate, well, site, role)
    {A549,U2OS}/split_summary.json      counts, shares, overlap checks
```

## Workflow

```bash
# 0+1  manifest + training dataset (CPU, minutes)           → training_datasets/training_dataset_cpg0000_{CL}/
./datasets.sh --cell_lines "A549 U2OS"            # add --slurm to submit; --skip_split to reuse manifests

# 2    noise models, 5 GPU jobs per cell line (~1.5 h)      → …/noise_models/noise_model_{ch}.npz
./noise_models.sh --cell_lines "A549 U2OS" --slurm

# 3    training (~1 day per cell line, 64 GB, 50 epochs × 45k patches)  → …/checkpoints/, training_stats.npz
./train.sh --cell_lines "A549 U2OS" --slurm

# 4a   timing benchmark: 1 FOV interactively on a GPU node, report s/FOV
./predict.sh --cell_line A549 --benchmark

# 4b   prediction, one 12 h job per plate, test FOVs only, MMSE-50, no posteriors
./predict.sh --cell_line A549 --slurm                       → predictions/cpg0000-jump-pilot/A549/{plate_dir}/
```

Prediction output per plate: `mmse/mmse_{well}_s{site:02d}_{channel}.tif`, `metadata.csv` (appended per FOV; rerunning a plate resumes), `metrics_summary.csv`, and two CellProfiler inputs in the JUMP `load_data_with_illum` layout: `load_data_with_illum.csv` (fluorescence → MMSE images) and `load_data_with_illum_original.csv` (fluorescence → raw images). Brightfield and illumination columns point at the raw plate in both.

## Scripts

| File | Role |
|---|---|
| `0_split.py` | builds the manifest from `experiment-metadata.tsv`, the paper's platemaps and target annotations |
| `1_datasets.py` | writes the training dataset from `role == train` rows |
| `2_noisemodels.py` | one N2V + GMM noise model per channel |
| `3_train.py` | LadderVAE training; `--max_patches_per_epoch` (default 45,000) caps the epoch, `--max_val_frames` (30) caps validation |
| `4_predict.py` | per-plate prediction from the manifest; `--batch_size 256 --autocast` are throughput settings only |
| `*.sh` | SLURM launchers for each step |

Fixed MicroSplit settings: patch 64, 3 lateral-context scales, stride 32 (inner tiling), MMSE-50, `--num_posterior_samples 0`.

## Next step after prediction

Run the paper's CellProfiler pipeline (`2024_Chandrasekaran_NatureMethods/pipelines/2020_11_04_CPJUMP1/CPJUMP1_analysis_without_batchfile_406.cppipe`, container `cellprofiler.sif`) on both load_data files of every plate, then pycytominer with the paper's feature list, then the copairs calls of `benchmark/1.0.calculate-map-cp.ipynb` on the two subset tables. A `5_cellprofiler.sh` for this is not written yet.

## Older runs

`experiment_run1_predictions/` and `training_dataset_{A549,U2OS}*` under `/project/cell_paint_mono/` are the March 2026 run1 (200-FOV sampling-strategy ablation, plate-level hold-out). They are not touched by these scripts.

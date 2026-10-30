# cpg0000-jump-pilot

MicroSplit on the JUMP Cell Painting pilot (Chandrasekaran et al., *Nat. Methods* 2024). Two cell lines (A549, U2OS), one model per cell line. The five fluorescence channels are averaged into one image and MicroSplit predicts the five channels back.

## Data

- Raw plates: `/project/cell_paint_mono/2020_11_04_CPJUMP1/images/{plate_dir}/Images/` (ch1 Mito, ch2 AGP, ch3 RNA, ch4 ER, ch5 DNA; ch6 to ch8 are brightfield and are not used). Illumination functions: `/project/cell_paint_mono/2020_11_04_CPJUMP1/illum/{barcode}/`.
- Paper repository (platemaps, target annotations, published profiles, CellProfiler pipeline): `/project/cell_paint_mono/2024_Chandrasekaran_NatureMethods/`.
- Only the 40 benchmark plates of the paper's Figures 4 and 5 are used: 20 per cell line (compound 24/48 h, CRISPR 96/144 h, ORF 48/96 h).

## Split designs

Each modality has one platemap shared by all its plates, so a test well position holds the same perturbation on every replicate plate and all replicates of a test perturbation are predicted. Two designs are available and are run side by side:

| Design | Test perturbations |
|---|---|
| `gene_subset` | 5 genes drawn uniformly at random from the genes targeted in all three modalities (two CRISPR guides, one ORF, at least two compounds); every compound, guide and ORF of these genes |
| `random_wells` | 5 % of each modality's treatment perturbations, drawn uniformly at random; all wells of each drawn perturbation |

In both designs:

- 8 fixed negative-control well positions per modality are added to the test set (DMSO, negative-control guides, negative-control ORFs).
- Sites 3 and 7 of every test well are predicted.
- Training uses 300 FOVs per cell line from the remaining wells, spread evenly over the 20 plates, one random site per well.
- `0_split.py` asserts that no training well and no test perturbation appears in both sets.

Per cell line this gives about 740 (`gene_subset`) and 820 (`random_wells`) test FOVs. All numbers are command-line options of `datasets.sh`.

Outputs:

```
/project/cell_paint_mono/splits/cpg0000-jump-pilot/{design}/{cell_line}/
    split_manifest.csv      one row per (plate, well, site, role)
    split_summary.json      test genes / perturbations, counts, image check
/project/cell_paint_mono/training_datasets/training_dataset_cpg0000_{design}_{cell_line}/
/project/cell_paint_mono/predictions/cpg0000-jump-pilot/{design}/{cell_line}/{plate_dir}/
```

## Workflow

Every launcher takes `--design gene_subset` or `--design random_wells`.

```bash
# 0+1  split manifest + training dataset (CPU, minutes)
./datasets.sh --design gene_subset --cell_lines "A549 U2OS"

# 2    noise models, one GPU job per channel and cell line
./noise_models.sh --design gene_subset --cell_lines "A549 U2OS" --slurm

# 3    training (100 epochs of 45,000 patches, about 1.5 days on one GPU)
./train.sh --design gene_subset --cell_lines "A549 U2OS" --slurm

# 4a   timing benchmark: 1 FOV interactively on a GPU node
./predict.sh --design gene_subset --cell_line A549 --benchmark

# 4b   prediction of the test FOVs, one job per plate, MMSE-50, no posterior samples
./predict.sh --design gene_subset --cell_line A549 --slurm
```

Prediction output per plate: `mmse/mmse_{well}_s{site:02d}_{channel}.tif`, `metadata.csv` (one row per FOV, appended as it finishes; rerunning a plate resumes), `metrics_summary.csv`, and two CellProfiler inputs in the JUMP `load_data_with_illum` layout: `load_data_with_illum.csv` (fluorescence channels from the MMSE images) and `load_data_with_illum_original.csv` (fluorescence channels from the raw images). Brightfield and illumination columns point at the raw plate in both.

## Scripts

| File | Role |
|---|---|
| `0_split.py` | builds the split manifest from `experiment-metadata.tsv` and the paper's platemaps and annotations |
| `1_datasets.py` | writes the training dataset from the `train` rows of the manifest |
| `2_noisemodels.py` | one N2V + GMM noise model per channel |
| `3_train.py` | LadderVAE training; `--max_patches_per_epoch` (45,000) caps the epoch, `--max_val_frames` (30) caps validation |
| `4_predict.py` | per-plate prediction of the test rows of the manifest |
| `*.sh` | SLURM launchers for each step |
| `deprecated/` | earlier versions of the scripts, kept for reference |

Fixed MicroSplit settings: patch 64, 3 lateral-context scales, stride 32 (inner tiling), MMSE-50, no posterior samples.

## After prediction

Run the paper's CellProfiler pipeline (`2024_Chandrasekaran_NatureMethods/pipelines/2020_11_04_CPJUMP1/CPJUMP1_analysis_without_batchfile_406.cppipe`) on both `load_data` files of every plate, then pycytominer with the paper's feature list, then the copairs calls of `benchmark/1.0.calculate-map-cp.ipynb` on the two subset tables.

# MicroSplit Examples with Cell Painting data

MicroSplit channel-unmixing experiments for Cell Painting data.

## Experiments

| Folder | Dataset | Cell line(s) | Status |
|--------|---------|-------------|--------|
| [`cpg0000-jump-pilot/`](cpg0000-jump-pilot/) | JUMP-CP pilot | A549, U2OS |
| [`cpg0029-chroma-pilot/`](cpg0029-chroma-pilot/) | ChromaLive pilot | U2OS | 
| [`cpg0036-EU-OS-bioactives/`](cpg0036-EU-OS-bioactives/) | EU-OS bioactives screen | HepG2, U2OS |
| [`cpg0039-garcia-fossa-livecellpainting/`](cpg0039-garcia-fossa-livecellpainting/) | Live Cell Painting | Huh-7 |
| [`example_notebooks/`](example_notebooks/) | cpg0000 small subset | A549 | Demo |
| [`HT_LIF24/`](HT_LIF24/) | HeLa/HCT116 HT | HeLa, HCT116 | Baseline |

## Workflow overview

Each experiment folder contains numbered Python scripts and matching bash launchers:

```
0_split.py                         ← train / test split manifest (run by datasets.sh; cpg0000, cpg0036, cpg0039)
1_datasets.py / datasets.sh        ← build training dataset from the manifest's training rows
2_noisemodels.py / noise_models.sh ← train per-channel Noise2Void + GMM models
3_train.py / train.sh              ← train LadderVAE MicroSplit model
4_predict.py / predict.sh          ← predict the manifest's test FOVs (tiling + stitching)
```

In cpg0000, cpg0036 and cpg0039 the split manifest lists every training and test FOV, and no test well is used for training.
Earlier versions of the scripts are kept in each experiment's `deprecated/` folder.

All data is read from `/project/cell_paint_mono/` on the HPC cluster.

## Quick start (HPC)

```bash
cd cpg0000-jump-pilot

# 0+1. Split manifests and training datasets for both cell lines
DS_JOBS=$(./datasets.sh --design gene_subset --cell_lines "A549 U2OS" --slurm)

# 2. Train noise models (5 parallel jobs per cell line)
NM_JOBS=$(./noise_models.sh --design gene_subset --cell_lines "A549 U2OS" --slurm --dependency "${DS_JOBS}")

# 3. Train model
TRAIN_JOBS=$(./train.sh --design gene_subset --cell_lines "A549 U2OS" --slurm --dependency "${NM_JOBS}")

# 4. Predict the test FOVs (one job per plate)
./predict.sh --design gene_subset --cell_line A549 --slurm --dependency "${TRAIN_JOBS}"
```

## Interactive notebook walkthrough

See [`example_notebooks/`](example_notebooks/) for a step-by-step Jupyter notebook
demonstration using a small downloaded subset from the Cell Painting Gallery.

## Channel mappings

| Dataset | Channels | Mapping |
|---------|----------|---------|
| cpg0000 | 5 (standard CP) | DNA=ch5, RNA=ch3, ER=ch4, AGP=ch2, Mito=ch1 |
| cpg0029 | 5 (alt-dye CP) | DNA=ch7, RNA=ch3, ER=ch4, AGP=ch5, Mito=ch6; ch1=Phase, ch2=BF, ch8=Actin ignored |
| cpg0036 | **4** (EU-OS CP) | DNA, ER_RNA\*, Actin_AGP\*, Mito; channel order differs per site, read from the authors' `load_data.csv` |
| cpg0039 | **2** (Live AO) | AO_Green = `GFP` filter, AO_Red = `Propidium Iodide` filter, read by file name |

\* cpg0036 ER_RNA channel mixes ConA Alexa 488 (ER) + SYTO 14 (RNA) at 500–550 nm;
Actin_AGP channel mixes WGA Alexa 555 (Golgi/PM) + Phalloidin Alexa 568 (F-actin) at 570–630 nm.
These spectral mixtures are the MicroSplit unmixing targets for cpg0036.

## Library code

All shared functions (image I/O, dataset building, tiling, prediction, noise model
training, metrics) live in:

```
src/microsplit_reproducibility/workflows/
    cellpainting/
        __init__.py        ← exports
        image_io.py        ← discover_fovs, read_fov_channels, combine_channels
        dataset_builder.py ← build_dataset_from_samples
        prediction.py      ← predict_fov, save_fov_predictions
        metrics.py         ← compute_metrics
        model_io.py        ← find_checkpoint, load_model_and_stats
    noise_model/
        training.py        ← train_noise_model_for_channel, load_channel_data
```

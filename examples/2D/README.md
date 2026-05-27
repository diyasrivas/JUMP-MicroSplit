# 2D MicroSplit Examples

MicroSplit channel-unmixing experiments for 2D Cell Painting data.

## Experiments

| Folder | Dataset | Cell line(s) | Status |
|--------|---------|-------------|--------|
| [`cpg0000-jump-pilot/`](cpg0000-jump-pilot/) | JUMP-CP pilot | A549, U2OS | Complete |
| [`cpg0029-chroma-pilot/`](cpg0029-chroma-pilot/) | ChromaLive pilot | U2OS | Complete |
| [`cpg0036-EU-OS-bioactives/`](cpg0036-EU-OS-bioactives/) | EU-OS bioactives screen | U2OS | In progress |
| [`cpg0039-garcia-fossa-livecellpainting/`](cpg0039-garcia-fossa-livecellpainting/) | Live Cell Painting | TBD | In progress |
| [`example_notebooks/`](example_notebooks/) | cpg0000 small subset | A549 | Demo |
| [`HT_LIF24/`](HT_LIF24/) | HeLa/HCT116 HT | HeLa, HCT116 | Baseline |

## Workflow overview

Each experiment folder contains 4 numbered Python scripts and matching bash launchers:

```
1_datasets.py / datasets.sh      ← build training dataset from local TIFFs
2_noisemodels.py / noise_models.sh ← train per-channel Noise2Void + GMM models
3_train.py / train.sh            ← train LadderVAE MicroSplit model
4_predict.py / predict.sh        ← run plate-level prediction (tiling + stitching)
```

All data is read from `/project/cell_paint_mono/` on the HPC cluster.

## Quick start (HPC)

```bash
cd cpg0000-jump-pilot

# 1. Build datasets for both cell lines
DS_JOBS=$(./datasets.sh --cell_lines "A549 U2OS" --slurm)

# 2. Train noise models (5 parallel jobs per cell line)
NM_JOBS=$(./noise_models.sh --cell_lines "A549 U2OS" --slurm --dependency "${DS_JOBS}")

# 3. Train model
TRAIN_JOBS=$(./train.sh --cell_lines "A549 U2OS" --slurm --dependency "${NM_JOBS}")

# 4. Predict (per batch × plate)
./predict.sh --cell_line A549 --batch 2020_11_04_CPJUMP1 --slurm --dependency "${TRAIN_JOBS}"
```

## Interactive notebook walkthrough

See [`example_notebooks/`](example_notebooks/) for a step-by-step Jupyter notebook
demonstration using a small downloaded subset from the Cell Painting Gallery.

## Channel mappings

| Dataset | Channels | Mapping |
|---------|----------|---------|
| cpg0000 | 5 (standard CP) | DNA=ch5, RNA=ch3, ER=ch4, AGP=ch2, Mito=ch1 |
| cpg0029 | 5 (alt-dye CP) | DNA=ch7, RNA=ch3, ER=ch4, AGP=ch5, Mito=ch6; ch1=Phase, ch2=BF, ch8=Actin ignored |
| cpg0036 | **4** (EU-OS CP) | DNA=ch4, ER_RNA=ch3\*, Actin_AGP=ch2\*, Mito=ch1 |
| cpg0039 | **2** (Live AO) | AO_Green=ch2, AO_Red=ch1 † |

\* cpg0036 ER_RNA channel mixes ConA Alexa 488 (ER) + SYTO 14 (RNA) at 500–550 nm;
Actin_AGP channel mixes WGA Alexa 555 (Golgi/PM) + Phalloidin Alexa 568 (F-actin) at 570–630 nm.
These spectral mixtures are the MicroSplit unmixing targets for cpg0036.

† cpg0039: Acridine Orange single-dye assay. Verify channel indices against actual
Cytation 5 TIFF files — may differ from Opera Phenix convention.

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

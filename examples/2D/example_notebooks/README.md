# MicroSplit Example Notebooks

End-to-end walkthrough of the MicroSplit workflow using a **small subset** of
the cpg0000-jump-pilot dataset. Designed to run on a laptop/workstation with
moderate GPU (or CPU for sanity checking).

## Context: JUMP Cell Painting datasets

These notebooks use **cpg0000** (CPJUMP1 pilot, Chandrasekaran et al. *Nature
Methods* 2024), which is small enough to download and demonstrate interactively.

The full-scale JUMP Cell Painting dataset is **cpg0016** (Chandrasekaran et al.
preprint 2023), which contains:
- ~116,000 unique compounds at a single site
- 12,602 ORF over-expression reagents
- 7,975 CRISPR-Cas9 knockouts (7,975 genes)
- Acquired across 12 pharmaceutical and academic partner sites
- Cell line: **U2OS** only (settled on U2OS based on cpg0000 pilot comparison)
- Standard 5-channel Cell Painting (same channel mapping as cpg0000)

For production-scale unmixing on cpg0016 data, use the HPC scripts in
`../cpg0000-jump-pilot/` (same channel mapping applies: DNA=ch5, RNA=ch3,
ER=ch4, AGP=ch2, Mito=ch1) and train on U2OS plates.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `00_datasets.ipynb` | Download images from Cell Painting Gallery and build a MicroSplit dataset |
| `01_noisemodels.ipynb` | Train per-channel Noise2Void + GaussianMixture noise models |
| `02_train.ipynb` | Train the LadderVAE MicroSplit model (5 epochs demo) |
| `03_predict.ipynb` | Run prediction, visualise channel unmixing, compute PSNR/SSIM |

## Setup

```bash
# Activate the MicroSplit conda environment
conda activate microsplit_jobs

# Install additional demo dependencies
pip install jump-portrait matplotlib

# Run notebooks in order
jupyter lab
```

## How to download from the Cell Painting Gallery

`00_datasets.ipynb` uses [`jump_portrait`](https://github.com/jump-cellpainting/jump-portrait)
to download images directly from the Cell Painting Gallery S3 bucket:

```python
from jump_portrait.fetch import get_jump_image

img = get_jump_image(
    source='cpg0000-jump-pilot',
    batch='2020_11_04_CPJUMP1',
    plate='BR00117015',
    well='B02',
    site='1',
    channel='DNA',   # channel name string, not an integer index
)
```

## Relation to HPC scripts

The notebooks use the **same library functions** as the HPC scripts:

```
src/microsplit_reproducibility/workflows/
    cellpainting/    ← shared I/O, prediction, metrics
    noise_model/     ← shared noise model training
```

The HPC scripts simply wrap these functions with SLURM job submission,
larger datasets, and longer training runs.

## Expected outputs

All outputs go to `./cpg0000_demo/` (gitignored — delete to start fresh):

```
cpg0000_demo/
    images/            raw downloaded TIFFs from Cell Painting Gallery
    dataset/
        combined/      float32 combined images
        DNA/ RNA/ ER/ AGP/ Mito/   uint16 per-channel images
        noise_models/  noise_model_{channel}.npz
        checkpoints/   model_*.ckpt + last.ckpt
        training_stats.npz
        metadata.csv
    predictions/
        mmse/          MMSE predictions (uint16 .tif)
        sample_0/      Posterior sample (uint16 .tif)
```

# MicroSplit Example Notebooks

End-to-end walkthrough of the MicroSplit workflow using a **small subset** of
cpg0016-jump (source_4, Batch1, 3 wells × 3 sites, U2OS cells). Designed to
run on a laptop/workstation with a moderate GPU (or CPU for sanity checking).

## Context: JUMP Cell Painting datasets

These notebooks use **cpg0016** (Chandrasekaran et al. preprint 2023), the
production-scale JUMP Cell Painting dataset:
- ~116,000 unique compounds at a single site
- 12,602 ORF over-expression reagents
- 7,975 CRISPR-Cas9 knockouts (7,975 genes)
- Acquired across 12 pharmaceutical and academic partner sites
- Cell line: **U2OS** only
- Standard 5-channel Cell Painting: DNA=ch5, RNA=ch3, ER=ch4, AGP=ch2, Mito=ch1

The demo downloads 9 FOVs (3 wells × 3 sites) from source_4/Batch1/BR00117035
directly from the public Cell Painting Gallery S3 bucket using anonymous boto3
access — no AWS credentials required.

For production-scale unmixing, use the HPC scripts in
`../cpg0016-jump/` (same channel mapping applies).

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

`00_datasets.ipynb` downloads images directly from the public S3 bucket using
`boto3` with anonymous (unsigned) access — no credentials required:

```python
import boto3, io, tifffile
from botocore import UNSIGNED
from botocore.config import Config

s3  = boto3.client('s3', config=Config(signature_version=UNSIGNED))
buf = io.BytesIO()
s3.download_fileobj('cellpainting-gallery', key, buf)
img = tifffile.imread(buf)
```

Images are stored as `r{row:02d}c{col:02d}f{site:02d}p01-ch{ch_idx}sk1fk1fl1.tiff`
under `cpg0016-jump/{source}/images/{batch}/images/{plate_folder}/Images/`.

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

All outputs go to `./cpg0016_demo/` (gitignored — delete to start fresh):

```
cpg0016_demo/
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

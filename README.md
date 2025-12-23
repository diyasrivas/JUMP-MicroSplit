# MicroSplit

[![Library](https://img.shields.io/badge/Library-CAREamics-orange)](https://careamics.github.io)
[![License](https://img.shields.io/pypi/l/careamics.svg?color=green)](https://github.com/CAREamics/MicroSplit-reproducibility/blob/main/LICENSE)
[![Image.sc](https://img.shields.io/badge/Got%20a%20question%3F-Image.sc-blue)](https://forum.image.sc/)

## Overview

This repository implements MicroSplit for [Cell Painting](https://jump-cellpainting.broadinstitute.org/) data, providing modular workflows optimized for large-scale image analysis and HPC environments. It extends the original [MicroSplit-reproducibility](https://github.com/CAREamics/MicroSplit-reproducibility) with data handling, metadata integration, and scalable processing pipelines for cell painting.

## What is MicroSplit

MicroSplit is a deep learning-based computational multiplexing technique that enables imaging of multiple cellular structures within a single fluorescent channel, allowing increased throughput, reduced acquisition time, and lower light exposure. The method uses a hierarchical variational auto-encoder (LVAE) with lateral context.
MicroSplit is implemented in the [CAREamics library](https://careamics.github.io). Using MicroSplit, cell painting assays can be revised to enable multiple cellular structures to be imaged in a single fluorescent channel and then computationally unmixed before image analysis steps.

## Workflow Architecture

### Phase 1: Dataset Creation
**Location:** `src/microsplit_reproducibility/workflows/dataset_creation/`
Fetches raw images from JUMP data sources, combines channels, and structures data for training:
- `JUMPDatasetBuilder`: Main interface for dataset creation
- Dataset-specific functions: `create_orf_dataset()`, `create_crispr_dataset()`, `create_compound_dataset()`, `create_pilot_dataset()`
- Handles: AWS S3 integration via `jump-portrait`, metadata parsing, channel normalization, TIFF export

### Phase 2: Noise Model Creation
**Location:** `src/microsplit_reproducibility/workflows/noise_model/`
Loads data for CAREamics Noise2Void training:
- `load_data_for_noise_model()`: Single function to prepare data from Phase 1 output
- Integrates directly with CAREamics N2V workflow

### Phase 3: Training
**Location:** `src/microsplit_reproducibility/datasets/JUMP.py`
- `get_train_val_data()`: Loads Phase 1 TIFFs for model training
- Configuration via config factories
- Training handled by CAREamics + PyTorch Lightning

### Phase 4: Prediction
**Location:** `src/microsplit_reproducibility/workflows/prediction/`, `examples/2D/JUMP/HPC/`
Applies trained models to generate predictions:
- Metadata mapping and result storage
- Preparing the output for image analysis with cellprofiler

## Installation

> [!IMPORTANT]  
> A GPU is required for training. Pre-trained models can be used for inference without GPU access.

Installation takes 5-10 minutes with an existing Conda/Mamba setup.

### Environment Setup

1. Create a Python environment (Python 3.10 recommended):
    ```bash
    mamba create -n microsplit python=3.10
    mamba activate microsplit
    ```

> [!TIP]  
> For Apple Silicon (M1/M2/M3), use:
> ```bash
> CONDA_SUBDIR=osx-arm64 conda create -n microsplit python=3.9
> conda activate microsplit
> conda config --env --set subdir osx-arm64
> ```

2. Install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) for your system.

3. Verify GPU access:
    ```bash
    # NVIDIA CUDA
    python -c "import torch; print([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())])"
    
    # Apple Silicon
    python -c "import torch; import platform; print(platform.processor() in ('arm', 'arm64') and torch.backends.mps.is_available())"
    ```

4. Install this repository:
    ```bash
    git clone https://github.com/[your-username]/JUMP-MicroSplit.git
    cd JUMP-MicroSplit
    pip install .
    ```

## Quick Start
### Phase 1: Create Dataset
```python
genes = ["TP53", "KRAS", "EGFR", "BRAF", "MYC"]
channels = ["DNA", "RNA", "ER", "AGP", "Mito"]

builder = JUMPDatasetBuilder(
    dataset_type="crispr",
    channels=channels,
    output_dir="./crispr_5gene_data"
)
profiles = load_crispr_profiles(source="source_4")
for gene in genes:
    crispr_ids = select_crispr_by_gene(profiles, gene)
    create_crispr_dataset(
        crispr_ids=crispr_ids,
        channels=[Channel[ch] for ch in channels],
        output_dir=Path(f"./crispr_5gene_data/{gene}"),
        images_per_perturbation=10
    )
```

### Phase 2: Create Noise Models
```python
channels = ["DNA", "RNA", "ER", "AGP", "Mito"]

for channel in channels:
    # Load single-channel data for N2V training
    noise_data = load_data_for_noise_model(
        dataset_dir="./crispr_5gene_data",
        channels=[channel],  
        max_images=None 
    )
    config = create_n2v_configuration(
        experiment_name=f"crispr_{channel.lower()}_noise_model",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=64,
        num_epochs=10
    )
    
    careamist = CAREamist(source=config)
    careamist.train(train_source=noise_data)
    careamist.save(f"./noise_models/{channel.lower()}_n2v_model")
```

### Phase 3: Train MicroSplit Model
```python
target_channels = ["DNA", "RNA", "ER", "AGP", "Mito"]
dataset_dir = "./crispr_5gene_data"
noise_models_dir = "./noise_models"

train_data_config, val_data_config, test_data_config = get_data_configs(
    channel_idx_list=target_channels
)

experiment_params = get_microsplit_parameters(
    nm_path=noise_models_dir,
    channel_idx_list=target_channels,
    batch_size=8
)

train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
    datapath=dataset_dir,
    train_config=train_data_config,
    val_config=val_data_config,
    test_config=test_data_config,
    load_data_func=get_train_val_data
)

train_dloader = DataLoader(train_dset, batch_size=8, shuffle=True)
val_dloader = DataLoader(val_dset, batch_size=8, shuffle=False)

train_microsplit_model(
    train_dloader=train_dloader,
    val_dloader=val_dloader,
    data_stats=data_stats,
    experiment_params=experiment_params,
    num_epochs=50,
    checkpoint_dir="./checkpoints/crispr_5gene_data"
)
```

### Phase 4: Prediction and metadata mapping
```python
predict_and_evaluate(
    dataset_dir="./crispr_5gene_data",
    checkpoint_dir="./checkpoints/crispr_5gene_data", 
    prediction_dir="./predictions/crispr_5gene_data",
    noise_models_dir="./noise_models",
    channels=["DNA", "RNA", "ER", "AGP", "Mito"],
    mmse_count=50
)

original_metadata = dataset_dir/"original_metadata.csv"
metadata_df = create_test_metadata_mapping(
    original_metadata_csv=original_metadata,
    prediction_dir=prediction_dir,
    channels=channels,
    output_csv="metadata_mapping.csv"
)

generate_cellprofiler_loaddata_csv(
    metadata_mapping_df=metadata_df,
    prediction_dir=prediction_dir,
    channels=channels,
    output_csv="cellprofiler_input.csv"
)
```

### HPC Batch Processing (Optional)

We **strongly recommend** to run training and prediction via HPC for processing multiple channel combinations or large datasets.
Remember to configure job arrays and resource allocation in SLURM scripts based on your HPC environment.

**Training:**
```bash
cd examples/2D/JUMP/HPC
sbatch train_all_combinations.sh
sbatch 5channels_predictions.sh
```

## [CellPaintMONO](https://github.com/juglab/CellPaintMONO)
After running MicroSplit on the cell painting data, we can evualate how the Microsplit-predicted cell painting data performs in comparison to the original data from pre-processing to profile creation. We use [CellProfiler](https://cellprofiler.org/) v.4.2.8 to run two analysis pipelines, adapted from the [cpg0000-jump-pilot experiment pipelines](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1/tree/56845c7d4dc322652952783d91dae0ffef47829f/pipelines/2020_11_04_CPJUMP1) and then perform some downstream analysis tasks on the profiles. The tools for analysis and comparison of cell painting data, as well as the entire image analysis workflow can be found in the CellPaintMONO repository.

## Tested Systems

#### System 1
- **OS:** Red Hat Enterprise Linux 8.10
- **GPU:** NVIDIA A40-16Q, 16GB
- **CUDA:** 12.4

#### System 2
- **OS:** macOS 14.1
- **GPU:** Apple M3, 16GB

#### System 3
- **OS:** Windows 10 Enterprise
- **GPU:** NVIDIA RTX A3000, 6GB
- **CUDA:** 12.3

## Troubleshooting

**Problem:** NVIDIA driver version error  
**Solution:** Downgrade PyTorch:
```bash
pip3 install torch==2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Problem:** Mac Silicon GPU test returns False  
**Solution:** Install PyTorch via pip (not conda) and ensure macOS-arm64 Anaconda/Mamba release.

## Resources
- [Original MicroSplit Repository](https://github.com/CAREamics/MicroSplit-reproducibility)
- [CAREamics Documentation](https://careamics.github.io)
- [Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery)
- [JUMP Cell Painting Consortium](https://jump-cellpainting.broadinstitute.org/)
- [JUMP Hub](https://broadinstitute.github.io/jump_hub/)

## License

BSD-3-Clause License - see [LICENSE](LICENSE) for details.

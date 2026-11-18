# cpg0029 chroma pilot: MicroSplit per dye set

Data: Cell Painting Gallery `cpg0029-chroma-pilot` (Sivagurunathan et al., *SLAS Discovery* 2025).
U2OS, 90 compounds x 4 wells + 24 DMSO wells, one 384-well plate per dye set, all with the
same platemap. The goal is to reproduce Figs 4 and 5 (phenotypic activity and
distinctiveness, scored within each plate) on the MicroSplit predictions.

Each dye set uses different dyes in the same TIFF channels, so **each dye set has its own
MicroSplit model**, trained on the non-test wells of its own plate.

| Folder | Plate | Dye set | Channels unmixed | Raw only (never mixed) |
|---|---|---|---|---|
| `standard_cp/` | BR00122250 | Standard CP | DNA, RNA, ER, AGP, Mito | Brightfield, Phase, Actin (ch8; no 400LS dye on this plate, features dropped by the authors) |
| `mitobrilliant/` | BR00122246 | CP + MitoBrilliant | DNA, RNA, ER, AGP, Mito | Brightfield, Phase, Actin (ch8; no 400LS dye on this plate, features dropped by the authors) |
| `phalloidin_400ls/` | BR00122247 | CP + Phenovue phalloidin 400LS | DNA, RNA, ER, GP (WGA), Mito, Actin (ch8) | Brightfield, Phase |
| `post_chromalive_cp/` | BR00122248 | Standard CP after live ChromaLive | DNA, RNA, ER, AGP, Mito | Brightfield, Phase, Actin (ch8; no 400LS dye on this plate, features dropped by the authors) |
| `chromalive/` | BR00122249 (48 h) | ChromaLive + Hoechst | CL488R, CL488Y, CL561YE, DNA (Hoechst) | Brightfield, Phase, 647 |

Channel names and TIFF indices come from the authors' `load_data.csv` of each plate and live
in one place: `cpg0029_shared/dye_sets.py`. Raw-only channels are loaded by the authors'
CellProfiler pipelines, so the `load_data` files written after prediction point them at the
original images in both series.

## Test set (shared by all dye sets)

`test_fovs.csv`, written once by `0_select_test_set.py` (seed 42): 30 compounds drawn at
random from the 90 (all 4 wells each), 12 random DMSO wells out of 24, and 4 random sites
out of 9 per well, which gives 528 FOVs. Because the platemap is identical, every plate
predicts the same (well, site) pairs. Training FOVs (300 per dye set by default) are drawn
from the other wells of the same plate.

## Layout

```
0_select_test_set.py, test_fovs.csv   test set, shared
cpg0029_shared/                       all logic: dye_sets.py, split.py, steps.py, launchers/
<dye_set>/                            thin entry points for one dye set
    0_split.py 1_datasets.py 2_noisemodels.py 3_train.py 4_predict.py
    datasets.sh noise_models.sh train.sh predict.sh
deprecated/                           previous scripts (wrong plate map, 30 % well hold-out)
```

Outputs (per dye set):

| Step | Path |
|---|---|
| split | `/project/cell_paint_mono/splits/cpg0029-chroma-pilot/<dye_set>/split_manifest.csv`, `split_summary.json` |
| dataset, noise models, checkpoints, logs | `/project/cell_paint_mono/training_datasets/training_dataset_cpg0029_<dye_set>/` |
| predictions | `/project/cell_paint_mono/predictions/cpg0029-chroma-pilot/<dye_set>/<barcode>/` with `mmse/`, `metadata.csv`, `metrics_summary.csv`, `load_data_with_illum.csv` (MicroSplit series), `load_data_with_illum_original.csv` (Original series) |

## Running one dye set

From the dye-set folder, on the cluster login node:

```bash
cd phalloidin_400ls
./datasets.sh --slurm                                   # steps 0+1 (CPU)
NM=$(./noise_models.sh --dependency <datasets_job>)     # step 2, one GPU job per channel
./train.sh --test --dependency "$NM"                    # optional 2-epoch sanity check
TR=$(./train.sh --dependency "$NM")                     # step 3, 100 epochs, W&B
./predict.sh --benchmark --dependency "$TR"             # optional: time 1 FOV
./predict.sh --dependency "$TR"                         # step 4, MMSE 50 + load_data
```

Each `.sh` has `--help`. Python steps can also be run directly, e.g.
`python phalloidin_400ls/0_split.py --help`.

## Notes

- Noise models, training and prediction all log to W&B (entity `juglab`, project
  `JUMP-MicroSplit`, group `cpg0029`, job types `noise_model`, `train`, `predict`). Prediction logs
  a per-FOV PSNR / SSIM table and per-channel means. Local W&B files go to the dataset or
  prediction folder on `/project`. Use `--no_wandb` to turn it off.
- `phalloidin_400ls` unmixes 6 channels, the maximum the configs support; not tried before.
- The illumination functions (`images/<batch>/illum/<barcode>/`) are not downloaded yet; the
  `load_data` files are written without `Illum` columns until they are. Rerun
  `4_predict.py --load_data_only` afterwards.
- BR00122246 well O01 has 8 of 9 sites; the split only uses FOVs whose files exist and open.

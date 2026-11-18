"""
The five cpg0029 dye sets of Figs 4 and 5 (Sivagurunathan et al., SLAS Discovery 2025).

Each dye set is one 384-well plate with the same platemap. Channel names and TIFF
indices are copied from the authors' ``load_data.csv`` of each plate, so the
CellProfiler ``load_data`` files written after prediction match the authors'
pipelines.

``unmixed``   fluorescent organelle channels that are averaged into the combined
              input and split back by MicroSplit.
``raw_only``  channels the plate's CellProfiler pipeline loads but MicroSplit never
              mixes (brightfield, phase, and fluorescent channels absent from the
              published profiles). Both series read them from the original images.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

DATA_DIR = Path("/project/cell_paint_mono/cpg0029-chroma-pilot")
SPLITS_DIR = Path("/project/cell_paint_mono/splits/cpg0029-chroma-pilot")
TRAINING_BASE = Path("/project/cell_paint_mono/training_datasets")
PREDICTIONS_DIR = Path("/project/cell_paint_mono/predictions/cpg0029-chroma-pilot")
CELL_LINE = "U2OS"

CP_UNMIXED = {"DNA": 7, "RNA": 3, "ER": 4, "AGP": 5, "Mito": 6}
CP_RAW_ONLY = {"Brightfield": 2, "Phase": 1, "Actin": 8}


@dataclass(frozen=True)
class DyeSet:
    name: str
    label: str
    barcode: str
    batch: str
    plate_dir: str
    unmixed: Dict[str, int]
    raw_only: Dict[str, int]

    @property
    def channels(self) -> list:
        """MicroSplit channel order."""
        return list(self.unmixed)

    @property
    def images_dir(self) -> Path:
        return DATA_DIR / "images" / self.batch / "images" / self.plate_dir / "Images"

    @property
    def illum_dir(self) -> Path:
        return DATA_DIR / "images" / self.batch / "illum" / self.barcode

    @property
    def split_dir(self) -> Path:
        return SPLITS_DIR / self.name

    @property
    def manifest_path(self) -> Path:
        return self.split_dir / "split_manifest.csv"

    @property
    def training_dir(self) -> Path:
        return TRAINING_BASE / f"training_dataset_cpg0029_{self.name}"

    @property
    def prediction_dir(self) -> Path:
        return PREDICTIONS_DIR / self.name / self.barcode


DYE_SETS: Dict[str, DyeSet] = {
    d.name: d for d in [
        DyeSet(
            name="standard_cp", label="Standard CP", barcode="BR00122250",
            batch="2023_05_15_Batch1", plate_dir="BR00122250__2023-04-01T05_43_06-Measurement1",
            unmixed=CP_UNMIXED, raw_only=CP_RAW_ONLY,
        ),
        DyeSet(
            name="mitobrilliant", label="CP + MitoBrilliant", barcode="BR00122246",
            batch="2023_05_15_Batch1", plate_dir="BR00122246__2023-04-01T03_13_00-Measurement1",
            unmixed=CP_UNMIXED, raw_only=CP_RAW_ONLY,
        ),
        # WGA alone stays in ch5 (named GP by the authors); phalloidin moves to ch8.
        DyeSet(
            name="phalloidin_400ls", label="CP + Phenovue phalloidin 400LS", barcode="BR00122247",
            batch="2023_05_17_Batch2", plate_dir="BR00122247__2023-04-01T08_12_58-Measurement1",
            unmixed={"DNA": 7, "RNA": 3, "ER": 4, "GP": 5, "Mito": 6, "Actin": 8},
            raw_only={"Brightfield": 2, "Phase": 1},
        ),
        # Fixed and Cell Painted after live ChromaLive + DRAQ7 + Cas3/7 imaging.
        # The live time points of the same barcode (2023_08_15_Batch6) are not used.
        DyeSet(
            name="post_chromalive_cp", label="Standard CP (exposed to ChromaLive)", barcode="BR00122248",
            batch="2023_05_17_Batch3", plate_dir="BR00122248__2023-04-01T10_42_56-Measurement1",
            unmixed=CP_UNMIXED, raw_only=CP_RAW_ONLY,
        ),
        # 48 h live plate used in Figs 4 and 5. The 4 h / 24 h plates (2024_02_02_Batch8)
        # are not used. Ch6 (647, dye not named) is absent from the published profile.
        DyeSet(
            name="chromalive", label="ChromaLive + Hoechst", barcode="BR00122249",
            batch="2023_08_02_Batch5", plate_dir="BR00122249__2023-03-25T00_14_17-Measurement2",
            unmixed={"CL488R": 3, "CL488Y": 4, "CL561YE": 5, "DNA": 7},
            raw_only={"Brightfield": 2, "Phase": 1, "647": 6},
        ),
    ]
}


def get_dye_set(name: str) -> DyeSet:
    if name not in DYE_SETS:
        raise KeyError(f"unknown dye set {name!r}; choose from {sorted(DYE_SETS)}")
    return DYE_SETS[name]

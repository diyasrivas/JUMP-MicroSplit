from typing import List, Optional, Dict
import polars as pl
import requests
from jump_portrait.fetch import get_jump_image, get_item_location_info


MANIFEST_URL = "https://raw.githubusercontent.com/jump-cellpainting/datasets/v0.11.0/manifests/profile_index.json"


def _load_manifest() -> List[Dict]:
    response = requests.get(MANIFEST_URL)
    response.raise_for_status()
    return response.json()


def _get_dataset_url(subset: str) -> str:
    manifest = _load_manifest()
    for dataset in manifest:
        if dataset['subset'] == subset:
            return dataset['url']
    raise ValueError(f"Dataset subset '{subset}' not found in manifest")


def load_pilot_metadata(source: str = "source_4") -> pl.LazyFrame:
    return pl.scan_parquet(
        f"https://cellpainting-gallery.s3.amazonaws.com/cpg0000-jump-pilot/{source}/workspace/metadata/*/*.parquet"
    )


def load_orf_profiles(source: str = "source_4") -> pl.LazyFrame:
    orf_url = _get_dataset_url('orf')
    return pl.scan_parquet(orf_url)


def load_compound_profiles(source: str = "source_4") -> pl.LazyFrame:
    compound_url = _get_dataset_url('compound')
    return pl.scan_parquet(compound_url)


def load_crispr_profiles(source: str = "source_4") -> pl.LazyFrame:
    crispr_url = _get_dataset_url('crispr')
    return pl.scan_parquet(crispr_url)


def get_location_info(perturbation_id: str, gene_symbol: Optional[str] = None) -> pl.DataFrame:
    """
    Get location info for a perturbation.
    
    For ORF data, gene_symbol must be provided because get_item_location_info()
    expects gene symbols, not JCP2022 IDs.
    """
    try:
        query_id = gene_symbol if gene_symbol else perturbation_id
        location_df = get_item_location_info(query_id)
        return location_df
    except Exception as e:
        raise RuntimeError(
            f"Failed to get location info for {perturbation_id} "
            f"(gene_symbol: {gene_symbol})"
        ) from e


def filter_by_batch(metadata: pl.LazyFrame, batch: str) -> pl.DataFrame:
    return metadata.filter(pl.col("Metadata_Batch") == batch).collect()


def get_plates_in_batch(metadata: pl.LazyFrame, batch: str) -> List[str]:
    filtered = filter_by_batch(metadata, batch)
    plates = filtered.select("Metadata_Plate").unique().to_series().to_list()
    return sorted(plates)


def get_wells_in_plate(
    metadata: pl.LazyFrame,
    batch: str,
    plate: str
) -> List[str]:
    
    filtered = metadata.filter(
        (pl.col("Metadata_Batch") == batch) &
        (pl.col("Metadata_Plate") == plate)
    ).collect()
    
    wells = filtered.select("Metadata_Well").unique().to_series().to_list()
    return sorted(wells)


def sample_locations_from_plate(
    metadata: pl.LazyFrame,
    batch: str,
    plate: str,
    num_samples: int,
    seed: int = 42
) -> pl.DataFrame:
    
    filtered = metadata.filter(
        (pl.col("Metadata_Batch") == batch) &
        (pl.col("Metadata_Plate") == plate)
    ).collect()
    
    if len(filtered) < num_samples:
        raise ValueError(
            f"Requested {num_samples} samples but only {len(filtered)} available"
        )
    
    return filtered.sample(n=num_samples, seed=seed)


def extract_location_from_row(row: dict) -> tuple[str, str, str, str, int]:
    source = row["Metadata_Source"]
    batch = row["Metadata_Batch"]
    plate = row["Metadata_Plate"]
    well = row["Metadata_Well"]
    site = row["Metadata_Site"]
    
    return source, batch, plate, well, site
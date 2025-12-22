from typing import List, Optional, Dict
import polars as pl
import pandas as pd
import numpy as np
import requests
import gzip
from io import StringIO


def load_orf_gene_mapping() -> Dict[str, str]:
    """
    Load ORF metadata from GitHub to get JCP2022 -> Gene Symbol mapping.
    This is required because get_item_location_info() expects gene symbols, not JCP IDs.
    """
    url = "https://raw.githubusercontent.com/jump-cellpainting/datasets/main/metadata/orf.csv.gz"
    response = requests.get(url)
    response.raise_for_status()
    
    content = gzip.decompress(response.content).decode('utf-8')
    orf_metadata_df = pd.read_csv(StringIO(content))
    
    mapping = dict(zip(
        orf_metadata_df['Metadata_JCP2022'],
        orf_metadata_df['Metadata_Symbol']
    ))
    
    return mapping


def get_orf_gene_mapping(profiles: pl.LazyFrame) -> Dict[str, str]:
    """
    Get ORF gene mapping from GitHub metadata.
    Returns dict mapping JCP2022 IDs to gene symbols.
    """
    return load_orf_gene_mapping()


def select_orfs_by_gene(
    profiles: pl.LazyFrame,
    gene_symbols: List[str]
) -> List[str]:
    
    orf_data = profiles.filter(
        pl.col("Metadata_Symbol").is_in(gene_symbols)
    ).select("Metadata_JCP2022").unique().collect()
    
    return orf_data["Metadata_JCP2022"].to_list()


def select_random_orfs(
    profiles: pl.LazyFrame,
    num_orfs: int,
    seed: int = 42
) -> List[str]:
    
    orf_data = profiles.filter(
        pl.col("Metadata_JCP2022").str.starts_with("JCP2022_")
    ).select("Metadata_JCP2022").unique().collect()
    
    np.random.seed(seed)
    all_orfs = orf_data["Metadata_JCP2022"].to_list()
    
    if len(all_orfs) < num_orfs:
        raise ValueError(
            f"Requested {num_orfs} ORFs but only {len(all_orfs)} available"
        )
    
    selected = np.random.choice(all_orfs, size=num_orfs, replace=False)
    return selected.tolist()


def select_crispr_by_gene(
    profiles: pl.LazyFrame,
    gene_symbols: List[str]
) -> List[str]:
    
    crispr_data = profiles.filter(
        pl.col("Metadata_Symbol").is_in(gene_symbols) &
        pl.col("Metadata_pert_type").str.contains("CRISPR")
    ).select("Metadata_JCP2022").unique().collect()
    
    return crispr_data["Metadata_JCP2022"].to_list()


def select_random_crispr(
    profiles: pl.LazyFrame,
    num_perturbations: int,
    seed: int = 42
) -> List[str]:
    
    crispr_data = profiles.filter(
        pl.col("Metadata_pert_type").str.contains("CRISPR")
    ).select("Metadata_JCP2022").unique().collect()
    
    np.random.seed(seed)
    all_crispr = crispr_data["Metadata_JCP2022"].to_list()
    
    if len(all_crispr) < num_perturbations:
        raise ValueError(
            f"Requested {num_perturbations} CRISPR but only {len(all_crispr)} available"
        )
    
    selected = np.random.choice(all_crispr, size=num_perturbations, replace=False)
    return selected.tolist()


def select_compounds_by_id(
    profiles: pl.LazyFrame,
    compound_ids: List[str]
) -> List[str]:
    
    compound_data = profiles.filter(
        pl.col("Metadata_JCP2022").is_in(compound_ids) &
        pl.col("Metadata_pert_type").str.contains("compound")
    ).select("Metadata_JCP2022").unique().collect()
    
    return compound_data["Metadata_JCP2022"].to_list()


def select_random_compounds(
    profiles: pl.LazyFrame,
    num_compounds: int,
    seed: int = 42
) -> List[str]:
    
    compound_data = profiles.filter(
        pl.col("Metadata_pert_type").str.contains("compound")
    ).select("Metadata_JCP2022").unique().collect()
    
    np.random.seed(seed)
    all_compounds = compound_data["Metadata_JCP2022"].to_list()
    
    if len(all_compounds) < num_compounds:
        raise ValueError(
            f"Requested {num_compounds} compounds but only {len(all_compounds)} available"
        )
    
    selected = np.random.choice(all_compounds, size=num_compounds, replace=False)
    return selected.tolist()


def get_perturbation_type(perturbation_id: str) -> str:
    
    if "JCP2022_" in perturbation_id:
        return "orf"
    elif "CRISPR" in perturbation_id.upper():
        return "crispr"
    else:
        return "compound"
from pathlib import Path
from typing import List, Optional, Dict
from .jump_dataset_creation import Channel, JUMPDataset, ImageMetadata
from .jump_pilot_dataset import create_pilot_dataset
from .jump_orf_dataset import create_orf_dataset
from .jump_crispr_dataset import create_crispr_dataset
from .jump_compound_dataset import create_compound_dataset
from .jump_perturbation_selection import (
    select_random_orfs,
    select_orfs_by_gene,
    select_random_crispr,
    select_crispr_by_gene,
    select_random_compounds,
    select_compounds_by_id,
    get_orf_gene_mapping
)
from .jump_metadata import (
    load_pilot_metadata,
    load_orf_profiles,
    load_crispr_profiles,
    load_compound_profiles,
    get_plates_in_batch
)
from .jump_dataset_visualization import (
    visualize_single_image,
    visualize_dataset_sample,
    verify_dataset_structure
)


class JUMPDatasetBuilder:
    
    def __init__(
        self,
        dataset_type: str,
        channels: List[str],
        output_dir: str,
        seed: int = 42,
        source: str = "source_4"
    ):
        self.dataset_type = dataset_type.lower()
        self.channels = [Channel[ch.upper()] for ch in channels]
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.source = source
        self.metadata_list: Optional[List[ImageMetadata]] = None
        
        self._validate_dataset_type()
    
    def _validate_dataset_type(self) -> None:
        valid_types = ["pilot", "orf", "crispr", "compound"]
        if self.dataset_type not in valid_types:
            raise ValueError(
                f"Invalid dataset_type: {self.dataset_type}. "
                f"Must be one of {valid_types}"
            )
    
    def create_pilot_dataset(
        self,
        batch: str,
        samples_per_plate: int = 5,
        plates: Optional[List[str]] = None,
        normalize: bool = False,
        weights: Optional[List[float]] = None
    ) -> List[ImageMetadata]:
        
        if self.dataset_type != "pilot":
            raise ValueError("Dataset type must be 'pilot'")
        
        self.metadata_list = create_pilot_dataset(
            batch=batch,
            channels=self.channels,
            output_dir=self.output_dir,
            samples_per_plate=samples_per_plate,
            plates=plates,
            normalize=normalize,
            weights=weights,
            seed=self.seed,
            source=self.source
        )
        
        return self.metadata_list
    
    def create_orf_dataset_random(
        self,
        num_orfs: int,
        images_per_orf: int = 5,
        normalize: bool = False,
        weights: Optional[List[float]] = None
    ) -> List[ImageMetadata]:
        
        if self.dataset_type != "orf":
            raise ValueError("Dataset type must be 'orf'")
        
        profiles = load_orf_profiles(self.source)
        orf_ids = select_random_orfs(profiles, num_orfs, seed=self.seed)
        gene_mapping = get_orf_gene_mapping(profiles)
        
        self.metadata_list = create_orf_dataset(
            orf_ids=orf_ids,
            channels=self.channels,
            output_dir=self.output_dir,
            images_per_orf=images_per_orf,
            normalize=normalize,
            weights=weights,
            seed=self.seed,
            source=self.source,
            gene_mapping=gene_mapping
        )
        
        return self.metadata_list
    
    def create_orf_dataset_by_genes(
        self,
        gene_symbols: List[str],
        images_per_orf: int = 5,
        normalize: bool = False,
        weights: Optional[List[float]] = None
    ) -> List[ImageMetadata]:
        
        if self.dataset_type != "orf":
            raise ValueError("Dataset type must be 'orf'")
        
        profiles = load_orf_profiles(self.source)
        orf_ids = select_orfs_by_gene(profiles, gene_symbols)
        gene_mapping = get_orf_gene_mapping(profiles)
        
        print(f"Found {len(orf_ids)} ORFs for genes: {', '.join(gene_symbols)}")
        
        self.metadata_list = create_orf_dataset(
            orf_ids=orf_ids,
            channels=self.channels,
            output_dir=self.output_dir,
            images_per_orf=images_per_orf,
            normalize=normalize,
            weights=weights,
            seed=self.seed,
            source=self.source,
            gene_mapping=gene_mapping
        )
        
        return self.metadata_list
    
    def create_crispr_dataset_random(
        self,
        num_perturbations: int,
        images_per_perturbation: int = 5,
        normalize: bool = False,
        weights: Optional[List[float]] = None
    ) -> List[ImageMetadata]:
        
        if self.dataset_type != "crispr":
            raise ValueError("Dataset type must be 'crispr'")
        
        profiles = load_crispr_profiles(self.source)
        crispr_ids = select_random_crispr(profiles, num_perturbations, seed=self.seed)
        
        self.metadata_list = create_crispr_dataset(
            crispr_ids=crispr_ids,
            channels=self.channels,
            output_dir=self.output_dir,
            images_per_perturbation=images_per_perturbation,
            normalize=normalize,
            weights=weights,
            seed=self.seed,
            source=self.source
        )
        
        return self.metadata_list
    
    def create_compound_dataset_random(
        self,
        num_compounds: int,
        images_per_compound: int = 5,
        normalize: bool = False,
        weights: Optional[List[float]] = None
    ) -> List[ImageMetadata]:
        
        if self.dataset_type != "compound":
            raise ValueError("Dataset type must be 'compound'")
        
        profiles = load_compound_profiles(self.source)
        compound_ids = select_random_compounds(profiles, num_compounds, seed=self.seed)
        
        self.metadata_list = create_compound_dataset(
            compound_ids=compound_ids,
            channels=self.channels,
            output_dir=self.output_dir,
            images_per_compound=images_per_compound,
            normalize=normalize,
            weights=weights,
            seed=self.seed,
            source=self.source
        )
        
        return self.metadata_list
    
    def visualize_sample(
        self,
        num_samples: int = 4,
        figsize: tuple = (20, 5)
    ):
        
        if not self.output_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {self.output_dir}")
        
        channel_names = [ch.value for ch in self.channels]
        
        return visualize_dataset_sample(
            self.output_dir,
            channel_names,
            num_samples=num_samples,
            seed=self.seed,
            figsize=figsize
        )
    
    def visualize_image(
        self,
        image_id: int,
        figsize: tuple = (15, 5)
    ):
        
        if not self.output_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {self.output_dir}")
        
        channel_names = [ch.value for ch in self.channels]
        
        return visualize_single_image(
            self.output_dir,
            image_id,
            channel_names,
            figsize=figsize
        )
    
    def verify_dataset(self) -> Dict:
        
        if not self.output_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {self.output_dir}")
        
        channel_names = [ch.value for ch in self.channels]
        
        results = verify_dataset_structure(self.output_dir, channel_names)
        
        print("\n=== Dataset Verification ===")
        print(f"Valid: {results['valid']}")
        
        if results['issues']:
            print("\nIssues found:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        print("\nStatistics:")
        for key, value in results['statistics'].items():
            print(f"  {key}: {value}")
        
        return results


def list_available_batches(dataset_type: str = "pilot", source: str = "source_4") -> List[str]:
    
    if dataset_type.lower() == "pilot":
        metadata = load_pilot_metadata(source)
        batches = metadata.select("Metadata_Batch").unique().collect()
        return sorted(batches["Metadata_Batch"].to_list())
    else:
        raise NotImplementedError(f"Batch listing not implemented for {dataset_type}")


def get_batch_info(batch: str, source: str = "source_4") -> Dict:
    
    metadata = load_pilot_metadata(source)
    plates = get_plates_in_batch(metadata, batch)
    
    return {
        "batch": batch,
        "num_plates": len(plates),
        "plates": plates
    }

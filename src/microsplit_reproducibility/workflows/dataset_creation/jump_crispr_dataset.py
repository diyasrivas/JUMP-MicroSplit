from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from .jump_dataset_creation import (
    ImageMetadata,
    Channel,
    validate_channels,
    validate_weights,
    create_directory_structure,
    combine_channels,
    save_images,
    save_metadata,
    print_dataset_summary
)
from .jump_metadata import (
    load_crispr_profiles,
    get_location_info,
    extract_location_from_row
)
from .jump_image_fetch import fetch_all_channels


def create_crispr_dataset(
    crispr_ids: List[str],
    channels: List[Channel],
    output_dir: Path,
    images_per_perturbation: int = 5,
    normalize: bool = False,
    weights: Optional[List[float]] = None,
    seed: int = 42,
    source: str = "source_4",
    gene_mapping: Optional[Dict[str, str]] = None
) -> List[ImageMetadata]:
    
    validate_channels(channels)
    channel_names = [ch.value for ch in channels]
    weights = validate_weights(weights, len(channels))
    
    create_directory_structure(output_dir, channels)
    
    metadata_list = []
    image_counter = 0
    
    total_target = len(crispr_ids) * images_per_perturbation
    print(f"Creating CRISPR dataset: {len(crispr_ids)} perturbations × {images_per_perturbation} images = {total_target} total images")
    print(f"Output directory: {output_dir}")
    
    np.random.seed(seed)
    
    for crispr_idx, crispr_id in enumerate(crispr_ids, 1):
        gene_symbol = gene_mapping.get(crispr_id, "Unknown") if gene_mapping else None
        print(f"\nProcessing CRISPR {crispr_idx}/{len(crispr_ids)}: {crispr_id}")
        
        if gene_symbol:
            print(f"Gene symbol: {gene_symbol}")
        
        try:
            location_info = get_location_info(crispr_id)
        except RuntimeError as e:
            print(f"Skipping {crispr_id}: {e}")
            continue
        
        if len(location_info) == 0:
            print(f"No images found for {crispr_id}")
            continue
        
        available_images = len(location_info)
        num_to_sample = min(images_per_perturbation, available_images)
        
        if num_to_sample < images_per_perturbation:
            print(f"Warning: Only {available_images} images available, sampling {num_to_sample}")
        
        sampled_indices = np.random.choice(
            available_images,
            size=num_to_sample,
            replace=False
        )
        
        images_processed = 0
        
        for sample_idx in sampled_indices:
            row = location_info.row(int(sample_idx), named=True)
            source_str, batch, plate, well, site = extract_location_from_row(row)
            
            print(f"Processing image {images_processed + 1}/{num_to_sample}: {source_str}/{batch}/{plate}/{well}/site_{site}")
            
            try:
                channel_images = fetch_all_channels(
                    source=source_str,
                    batch=batch,
                    plate=plate,
                    well=well,
                    site=site,
                    channels=channel_names
                )
                
                combined_image, stats = combine_channels(
                    channel_images,
                    channel_names,
                    weights,
                    normalize
                )
                
                save_images(
                    combined_image,
                    channel_images,
                    image_counter,
                    output_dir,
                    channel_names
                )
                
                metadata = ImageMetadata(
                    image_id=image_counter,
                    perturbation_id=crispr_id,
                    perturbation_type="crispr",
                    source=source_str,
                    batch=batch,
                    plate=plate,
                    well=well,
                    site=site,
                    channels=channel_names,
                    gene_symbol=gene_symbol
                )
                metadata_list.append(metadata)
                
                image_counter += 1
                images_processed += 1
                
            except RuntimeError as e:
                print(f"Failed to process image: {e}")
                continue
        
        print(f"Successfully processed {images_processed}/{num_to_sample} images for {crispr_id}")
    
    save_metadata(metadata_list, output_dir)
    print_dataset_summary(metadata_list, output_dir, channel_names)
    
    return metadata_list

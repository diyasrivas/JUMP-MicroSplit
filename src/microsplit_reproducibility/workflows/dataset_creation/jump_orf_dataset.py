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
    load_orf_profiles,
    get_location_info,
    extract_location_from_row
)
from .jump_image_fetch import fetch_all_channels
from .jump_perturbation_selection import get_orf_gene_mapping


def create_orf_dataset(
    orf_ids: List[str],
    channels: List[Channel],
    output_dir: Path,
    images_per_orf: int = 5,
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
    
    if gene_mapping is None:
        profiles = load_orf_profiles(source)
        gene_mapping = get_orf_gene_mapping(profiles)
    
    metadata_list = []
    image_counter = 0
    
    total_target = len(orf_ids) * images_per_orf
    print(f"Creating ORF dataset: {len(orf_ids)} ORFs × {images_per_orf} images = {total_target} total images")
    print(f"Output directory: {output_dir}")
    
    np.random.seed(seed)
    
    for orf_idx, orf_id in enumerate(orf_ids, 1):
        gene_symbol = gene_mapping.get(orf_id, None)
        
        if not gene_symbol:
            print(f"\nSkipping ORF {orf_idx}/{len(orf_ids)}: {orf_id} (no gene symbol found)")
            continue
            
        print(f"\nProcessing ORF {orf_idx}/{len(orf_ids)}: {orf_id}")
        print(f"Gene symbol: {gene_symbol}")
        
        try:
            location_info = get_location_info(orf_id, gene_symbol=gene_symbol)
        except RuntimeError as e:
            print(f"Skipping {orf_id}: {e}")
            continue
        
        if len(location_info) == 0:
            print(f"No images found for {gene_symbol}")
            continue
        
        available_images = len(location_info)
        num_to_sample = min(images_per_orf, available_images)
        
        if num_to_sample < images_per_orf:
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
                    perturbation_id=orf_id,
                    perturbation_type="orf",
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
        
        print(f"Successfully processed {images_processed}/{num_to_sample} images for {orf_id}")
    
    save_metadata(metadata_list, output_dir)
    print_dataset_summary(metadata_list, output_dir, channel_names)
    
    return metadata_list
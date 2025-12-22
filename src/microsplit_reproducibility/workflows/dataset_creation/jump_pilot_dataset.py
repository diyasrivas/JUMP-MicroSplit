from pathlib import Path
from typing import List, Optional
import numpy as np
from .jump_dataset_creation import (
    DatasetConfig,
    ImageMetadata,
    Channel,
    JUMPDataset,
    validate_channels,
    validate_weights,
    create_directory_structure,
    combine_channels,
    save_images,
    save_metadata,
    print_dataset_summary
)
from .jump_metadata import (
    load_pilot_metadata,
    filter_by_batch,
    get_plates_in_batch,
    sample_locations_from_plate,
    extract_location_from_row
)
from .jump_image_fetch import fetch_all_channels


def create_pilot_dataset(
    batch: str,
    channels: List[Channel],
    output_dir: Path,
    samples_per_plate: int = 5,
    plates: Optional[List[str]] = None,
    normalize: bool = False,
    weights: Optional[List[float]] = None,
    seed: int = 42,
    source: str = "source_4"
) -> List[ImageMetadata]:
    
    validate_channels(channels)
    channel_names = [ch.value for ch in channels]
    weights = validate_weights(weights, len(channels))
    
    create_directory_structure(output_dir, channels)
    
    metadata_list = []
    image_counter = 0
    
    pilot_metadata = load_pilot_metadata(source)
    
    if plates is None:
        plates = get_plates_in_batch(pilot_metadata, batch)
    
    print(f"Creating dataset from batch: {batch}")
    print(f"Found {len(plates)} plates")
    print(f"Target: {samples_per_plate} samples per plate ({len(plates) * samples_per_plate} total)")
    print(f"Output directory: {output_dir}")
    
    for plate_idx, plate in enumerate(plates, 1):
        print(f"\nProcessing plate {plate_idx}/{len(plates)}: {plate}")
        print(f"Target samples for this plate: {samples_per_plate}")
        
        try:
            samples = sample_locations_from_plate(
                pilot_metadata,
                batch,
                plate,
                samples_per_plate,
                seed=seed + plate_idx
            )
        except ValueError as e:
            print(f"Skipping plate {plate}: {e}")
            continue
        
        samples_processed = 0
        
        for sample_idx, row in enumerate(samples.iter_rows(named=True), 1):
            source, batch_name, plate_name, well, site = extract_location_from_row(row)
            
            print(f"Processing sample {sample_idx}/{samples_per_plate}: Well {well}, Site {site}")
            
            try:
                channel_images = fetch_all_channels(
                    source=source,
                    batch=batch_name,
                    plate=plate_name,
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
                    perturbation_id=f"{batch}_{plate}_{well}_{site}",
                    perturbation_type="pilot",
                    source=source,
                    batch=batch_name,
                    plate=plate_name,
                    well=well,
                    site=site,
                    channels=channel_names
                )
                metadata_list.append(metadata)
                
                image_counter += 1
                samples_processed += 1
                
            except RuntimeError as e:
                print(f"Failed to process sample: {e}")
                continue
        
        print(f"Successfully processed {samples_processed}/{samples_per_plate} samples from this plate")
    
    save_metadata(metadata_list, output_dir)
    print_dataset_summary(metadata_list, output_dir, channel_names)
    
    return metadata_list

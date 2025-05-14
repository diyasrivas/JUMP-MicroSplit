import os
import tifffile
import numpy as np

# Directory paths
base_dir = "/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP"
original_dataset_dir = f"{base_dir}/experiments/2_channels/er_mito"  

# Check original metadata
sample_file = os.path.join(original_dataset_dir, "ER", "img_00000_ER.tif")

if os.path.exists(sample_file):
    with tifffile.TiffFile(sample_file) as tif:
        metadata = tif.pages[0].tags
        print("Available TIFF tags:")
        for tag in metadata:
            print(f"  {tag}")

        important_tags = ['ImageDescription', 'XResolution', 'YResolution']
        for tag in important_tags:
            if tag in metadata:
                print(f"\n{tag}: {metadata[tag].value}")
        
    # Check filenames
    filename = os.path.basename(sample_file)
    print(f"\nFilename pattern: {filename}")
    print("Metadata extractable from filename: image index and channel only")
    
    print("\nConclusion: Original files lack embedded metadata for CellProfiler.")
    print("We need to use the experiment_info data to recover full metadata.")
else:
    print(f"Sample file not found: {sample_file}")

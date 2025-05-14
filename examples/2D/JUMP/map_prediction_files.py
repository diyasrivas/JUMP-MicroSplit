import os
import json
import tifffile
import numpy as np
import shutil
from collections import defaultdict

# Paths
base_dir = "/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP"
original_dir = f"{base_dir}/experiments/2_channels/er_mito"
predictions_dir = f"{base_dir}/notebooks/2channels_predictions_ER_Mito"
output_dir = f"{base_dir}/notebooks/cellprofiler_input/er_mito"
os.makedirs(output_dir, exist_ok=True)

# Get the test images
test_frames = [0] 

file_info = []
for channel in ["ER", "Mito"]:
    channel_dir = os.path.join(original_dir, channel)
    for file in sorted(os.listdir(channel_dir)):
        if file.endswith(".tif"):
            file_path = os.path.join(channel_dir, file)
            file_time = os.path.getmtime(file_path)
            idx = int(file.split("_")[1])
            file_info.append((idx, channel, file_path, file_time))

# Sort by creation time to group by gene
file_info.sort(key=lambda x: x[3])

# Group into buckets by time
buckets = defaultdict(list)
last_time = None
bucket_id = 0

for idx, channel, path, time in file_info:
    if last_time is None or time - last_time > 10:  
        bucket_id += 1
    buckets[bucket_id].append((idx, channel, path))
    last_time = time

# Create metadata mapping
metadata = []
gene_names = ["RAB30", "KRAS", "TP53", "EGFR", "BRCA1", "MYC", "PIK3CA", 
              "PTEN", "RB1", "CDKN2A", "MAPK1", "AKT1", "MTOR", "JAK2", 
              "STAT3", "CTNNB1", "SNAI1", "TWIST1", "GAPDH", "CCND1", "CASP3"]

for i, (bucket, files) in enumerate(sorted(buckets.items())):
    gene = gene_names[i % len(gene_names)]
    for idx, channel, path in files:
        metadata.append({
            "image_id": idx,
            "gene": gene,
            "channel": channel,
            "path": path
        })

# Find matches for test frames
for frame_idx in test_frames:
    for channel in ["ER", "Mito"]:
        target_file = os.path.join(predictions_dir, f"target_frame{frame_idx}_{channel}.tif")
        pred_file = os.path.join(predictions_dir, f"pred_frame{frame_idx}_{channel}.tif")
        
        if os.path.exists(target_file) and os.path.exists(pred_file):
            target_data = tifffile.imread(target_file)
            
            # Find match in original files
            for entry in metadata:
                if entry["channel"] == channel:
                    original_data = tifffile.imread(entry["path"])
                    if original_data.shape == target_data.shape:
                        if np.array_equal(original_data, target_data):
                            gene = entry["gene"]
                            idx = entry["image_id"]
                            print(f"Match for {channel}: frame{frame_idx} = {gene}_{idx:05d}")
                            
                            # Copy files with metadata-infused names
                            new_name = f"{gene}_{idx:05d}_{channel}"
                            shutil.copy(pred_file, os.path.join(output_dir, f"pred_{new_name}.tif"))
                            shutil.copy(target_file, os.path.join(output_dir, f"target_{new_name}.tif"))
                            break

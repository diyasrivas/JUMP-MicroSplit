#!/usr/bin/env python3
"""
Map bio-rand prediction files to original test dataset metadata
Flexible version that works for both CRISPR and ORF JUMP-MicroSplit experiments
"""

import pandas as pd
from pathlib import Path
import sys

# Configuration - Update these paths as needed
BASE_DIR = "/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_datasets"
PREDICTION_DIR = f"{BASE_DIR}/bio-rand_predictions_v2"
DATASET_METADATA_PATH = f"{BASE_DIR}/test_data/5_channels/dna_rna_er_agp_mito/dataset_metadata.csv"
CHANNELS = ["DNA", "RNA", "ER", "AGP", "Mito"]

# Column mapping for different dataset types
# Maps standard column names to possible alternatives
COLUMN_MAPPINGS = {
    'perturbation': ['gene_perturbation', 'pert_name', 'perturbation'],
    'pert_type': ['pert_type', 'perturbation_type'],
    'source': ['source', 'Source'],
    'batch': ['batch', 'Batch'],
    'plate': ['plate', 'Plate'],
    'well': ['well', 'Well'],
    'site': ['site', 'Site'],
    'experiment': ['experiment', 'Experiment'],
    'channel_combination': ['channel_combination', 'combined_channels'],
    'image_id': ['image_id', 'ImageID', 'image_number']
}

def detect_column_name(df, standard_name):
    """
    Detect which actual column name to use from the available alternatives.
    Returns the first matching column name found, or None if not found.
    """
    possible_names = COLUMN_MAPPINGS.get(standard_name, [standard_name])
    
    for name in possible_names:
        if name in df.columns:
            return name
    
    return None

def get_column_value(row, df, standard_name, default='N/A'):
    """
    Safely get a column value, handling missing columns gracefully.
    """
    actual_column = detect_column_name(df, standard_name)
    
    if actual_column:
        return row[actual_column]
    else:
        return default

def create_test_metadata_mapping():
    """
    Create mapping between test predictions and original dataset metadata.
    Flexible version that adapts to different metadata column structures.
    """
    
    # Load original metadata
    print(f"Loading metadata from: {DATASET_METADATA_PATH}")
    original_metadata = pd.read_csv(DATASET_METADATA_PATH)
    print(f"Loaded metadata: {len(original_metadata)} images")
    
    # Detect dataset type and available columns
    print(f"\n{'='*70}")
    print("Detected columns in metadata:")
    print(f"{'='*70}")
    
    column_detection = {}
    for standard_name, possible_names in COLUMN_MAPPINGS.items():
        detected = detect_column_name(original_metadata, standard_name)
        column_detection[standard_name] = detected
        status = f"✅ '{detected}'" if detected else "❌ Not found"
        print(f"{standard_name:20s}: {status}")
    
    # Verify we have the expected number of images
    num_images = len(original_metadata)
    print(f"\nDataset contains {num_images} images (expecting 10)")
    
    if num_images != 10:
        print(f"⚠️  WARNING: Expected 10 images but found {num_images}")
        print("   Proceeding anyway, but verify this is correct!")
    
    # Determine dataset type
    if 'dataset' in original_metadata.columns:
        dataset_type = original_metadata['dataset'].iloc[0] if len(original_metadata) > 0 else 'unknown'
        print(f"\nDataset type: {dataset_type}")
    else:
        dataset_type = 'unknown'
        print(f"\nDataset type: Could not determine (no 'dataset' column)")
    
    # Create mapping for each test image
    test_mapping = []
    
    for test_idx in range(num_images):
        row = original_metadata.iloc[test_idx]
        
        # Generate file lists
        prediction_files = ",".join([f"test_pred_frame{test_idx}_{ch}.tif" for ch in CHANNELS])
        target_files = ",".join([f"target_frame{test_idx}_{ch}.tif" for ch in CHANNELS])
        input_file = f"input_frame{test_idx}_combined.tif"
        
        # Create mapping record with flexible column detection
        mapping_entry = {
            'test_index': test_idx,
            'original_image_id': get_column_value(row, original_metadata, 'image_id', test_idx),
            'perturbation': get_column_value(row, original_metadata, 'perturbation', 'unknown'),
            'pert_type': get_column_value(row, original_metadata, 'pert_type', 'unknown'),
            'source': get_column_value(row, original_metadata, 'source', 'unknown'),
            'batch': get_column_value(row, original_metadata, 'batch', 'unknown'),
            'plate': get_column_value(row, original_metadata, 'plate', 'unknown'),
            'well': get_column_value(row, original_metadata, 'well', 'unknown'),
            'site': get_column_value(row, original_metadata, 'site', 'unknown'),
            'experiment': get_column_value(row, original_metadata, 'experiment', 'unknown'),
            'channel_combination': get_column_value(row, original_metadata, 'channel_combination', 'DNA_RNA_ER_AGP_Mito'),
            'prediction_files': prediction_files,
            'target_files': target_files,
            'input_file': input_file,
        }
        
        # Add any additional columns that exist in the original metadata
        # This preserves extra information without breaking the script
        additional_cols = ['dataset', 'normalized', 'weights']
        for col in additional_cols:
            if col in original_metadata.columns:
                mapping_entry[col] = row[col]
        
        test_mapping.append(mapping_entry)
    
    # Create DataFrame
    mapping_df = pd.DataFrame(test_mapping)
    
    # Save mapping
    output_path = f"{PREDICTION_DIR}/test_images_metadata.csv"
    mapping_df.to_csv(output_path, index=False)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"✅ Metadata mapping created: {len(mapping_df)} entries")
    print(f"✅ Output saved to: {output_path}")
    print(f"{'='*70}")
    
    print(f"\nFirst 3 rows of mapping:")
    print(mapping_df.head(3).to_string(index=False))
    
    # Generate summary statistics
    print(f"\n{'='*70}")
    print("Test Set Composition Summary:")
    print(f"{'='*70}")
    print(f"Unique perturbations: {mapping_df['perturbation'].nunique()}")
    print(f"Unique perturbation types: {mapping_df['pert_type'].nunique()}")
    print(f"Unique sources: {mapping_df['source'].nunique()}")
    print(f"Unique plates: {mapping_df['plate'].nunique()}")
    print(f"Unique wells: {mapping_df['well'].nunique()}")
    
    print(f"\nPerturbation distribution:")
    print(mapping_df['perturbation'].value_counts())
    
    print(f"\nPerturbation type distribution:")
    print(mapping_df['pert_type'].value_counts())
    
    # Verify file existence
    print(f"\n{'='*70}")
    print("Verifying prediction files exist...")
    print(f"{'='*70}")
    
    prediction_path = Path(PREDICTION_DIR)
    missing_files = []
    found_files = []
    
    for idx, row in mapping_df.iterrows():
        test_idx = row['test_index']
        
        # Check all channel files
        for channel in CHANNELS:
            pred_file = prediction_path / f"test_pred_frame{test_idx}_{channel}.tif"
            target_file = prediction_path / f"target_frame{test_idx}_{channel}.tif"
            
            if pred_file.exists():
                found_files.append(str(pred_file))
            else:
                missing_files.append(f"Missing prediction: {pred_file.name}")
            
            if target_file.exists():
                found_files.append(str(target_file))
            else:
                missing_files.append(f"Missing target: {target_file.name}")
        
        # Check input file
        input_file = prediction_path / f"input_frame{test_idx}_combined.tif"
        if input_file.exists():
            found_files.append(str(input_file))
        else:
            missing_files.append(f"Missing input: {input_file.name}")
    
    total_expected = len(mapping_df) * (len(CHANNELS) * 2 + 1)  # predictions + targets + inputs
    print(f"Expected files: {total_expected}")
    print(f"Found files: {len(found_files)}")
    print(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n⚠️  Missing files detected:")
        for missing in missing_files[:10]:  # Show first 10
            print(f"   - {missing}")
        if len(missing_files) > 10:
            print(f"   ... and {len(missing_files) - 10} more")
    else:
        print(f"\n✅ All expected files found!")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print("1. Review the generated metadata file:")
    print(f"   {output_path}")
    print("2. Use this file as input to generate_csv_cpipe1.py")
    print("3. This will create the CellProfiler load_data CSV files")
    
    return mapping_df

if __name__ == "__main__":
    print("Flexible Test Metadata Mapping Generator")
    print("Supports both CRISPR and ORF JUMP datasets")
    print("=" * 70)
    
    # Allow command-line arguments to override paths
    if len(sys.argv) > 1:
        BASE_DIR = sys.argv[1]
        PREDICTION_DIR = sys.argv[2] if len(sys.argv) > 2 else f"{BASE_DIR}/predictions"
        DATASET_METADATA_PATH = sys.argv[3] if len(sys.argv) > 3 else f"{BASE_DIR}/test_data/dataset_metadata.csv"
    
    print(f"Base directory: {BASE_DIR}")
    print(f"Prediction directory: {PREDICTION_DIR}")
    print(f"Metadata file: {DATASET_METADATA_PATH}")
    print("=" * 70)
    
    try:
        mapping_df = create_test_metadata_mapping()
        print(f"\n🎉 SUCCESS: Metadata mapping completed!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        print("   Check that all paths are correct.")
        print("\nUsage:")
        print("  python map_biorand_test_metadata_flexible.py [BASE_DIR] [PREDICTION_DIR] [METADATA_PATH]")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

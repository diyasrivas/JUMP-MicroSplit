from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


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


def detect_column_name(df: pd.DataFrame, standard_name: str) -> Optional[str]:
    
    possible_names = COLUMN_MAPPINGS.get(standard_name, [standard_name])
    
    for name in possible_names:
        if name in df.columns:
            return name
    
    return None


def get_column_value(row: pd.Series, df: pd.DataFrame, standard_name: str, default: str = 'N/A') -> str:
    
    actual_column = detect_column_name(df, standard_name)
    return row[actual_column] if actual_column else default


def create_test_metadata_mapping(
    original_metadata_csv: Path,
    prediction_dir: Path,
    channels: List[str],
    output_csv: Optional[Path] = None
) -> pd.DataFrame:
    
    original_metadata_csv = Path(original_metadata_csv)
    prediction_dir = Path(prediction_dir)
    
    original_metadata = pd.read_csv(original_metadata_csv)
    num_images = len(original_metadata)
    
    test_mapping = []
    
    for test_idx in range(num_images):
        row = original_metadata.iloc[test_idx]
        
        prediction_files = ",".join([f"test_pred_frame{test_idx}_{ch}.tif" for ch in channels])
        target_files = ",".join([f"target_frame{test_idx}_{ch}.tif" for ch in channels])
        input_file = f"input_frame{test_idx}_combined.tif"
        
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
            'channel_combination': get_column_value(row, original_metadata, 'channel_combination', 'N/A'),
            'prediction_files': prediction_files,
            'target_files': target_files,
            'input_file': input_file,
        }
        
        for col in ['dataset', 'normalized', 'weights']:
            if col in original_metadata.columns:
                mapping_entry[col] = row[col]
        
        test_mapping.append(mapping_entry)
    
    mapping_df = pd.DataFrame(test_mapping)
    
    if output_csv:
        output_csv = Path(output_csv)
        mapping_df.to_csv(output_csv, index=False)
    
    return mapping_df


def verify_prediction_files_exist(
    prediction_dir: Path,
    metadata_df: pd.DataFrame,
    channels: List[str]
) -> Tuple[List[str], List[str]]:
    
    prediction_dir = Path(prediction_dir)
    found = []
    missing = []
    
    for idx, row in metadata_df.iterrows():
        test_idx = row['test_index']
        
        for channel in channels:
            pred_file = prediction_dir / f"test_pred_frame{test_idx}_{channel}.tif"
            target_file = prediction_dir / f"target_frame{test_idx}_{channel}.tif"
            
            if pred_file.exists():
                found.append(str(pred_file.name))
            else:
                missing.append(str(pred_file.name))
            
            if target_file.exists():
                found.append(str(target_file.name))
            else:
                missing.append(str(target_file.name))
        
        input_file = prediction_dir / f"input_frame{test_idx}_combined.tif"
        if input_file.exists():
            found.append(str(input_file.name))
        else:
            missing.append(str(input_file.name))
    
    return found, missing


def generate_cellprofiler_loaddata_csv(
    metadata_mapping_df: pd.DataFrame,
    prediction_dir: Path,
    channels: List[str],
    output_csv: Path
) -> None:
    
    prediction_dir = Path(prediction_dir)
    output_csv = Path(output_csv)
    
    cellprofiler_rows = []
    
    for idx, row in metadata_mapping_df.iterrows():
        test_idx = row['test_index']
        
        cp_row = {
            'Metadata_ImageNumber': test_idx,
            'Metadata_Perturbation': row['perturbation'],
            'Metadata_PertType': row['pert_type'],
            'Metadata_Source': row['source'],
            'Metadata_Batch': row['batch'],
            'Metadata_Plate': row['plate'],
            'Metadata_Well': row['well'],
            'Metadata_Site': row['site'],
        }
        
        for channel in channels:
            pred_path = prediction_dir / f"test_pred_frame{test_idx}_{channel}.tif"
            target_path = prediction_dir / f"target_frame{test_idx}_{channel}.tif"
            
            cp_row[f'FileName_Pred_{channel}'] = pred_path.name
            cp_row[f'PathName_Pred_{channel}'] = str(pred_path.parent.absolute())
            cp_row[f'FileName_Target_{channel}'] = target_path.name
            cp_row[f'PathName_Target_{channel}'] = str(target_path.parent.absolute())
        
        input_path = prediction_dir / f"input_frame{test_idx}_combined.tif"
        cp_row['FileName_Input'] = input_path.name
        cp_row['PathName_Input'] = str(input_path.parent.absolute())
        
        cellprofiler_rows.append(cp_row)
    
    cp_df = pd.DataFrame(cellprofiler_rows)
    cp_df.to_csv(output_csv, index=False)


def get_test_set_summary(metadata_df: pd.DataFrame) -> Dict:
    
    return {
        'total_images': len(metadata_df),
        'unique_perturbations': metadata_df['perturbation'].nunique(),
        'unique_pert_types': metadata_df['pert_type'].nunique(),
        'unique_sources': metadata_df['source'].nunique(),
        'unique_plates': metadata_df['plate'].nunique(),
        'unique_wells': metadata_df['well'].nunique(),
        'perturbation_distribution': metadata_df['perturbation'].value_counts().to_dict(),
        'pert_type_distribution': metadata_df['pert_type'].value_counts().to_dict(),
    }

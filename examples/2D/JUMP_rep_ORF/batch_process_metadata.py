#!/usr/bin/env python3
"""
Batch process multiple prediction directories
Useful for running the metadata mapping on multiple experimental runs
"""

import subprocess
import sys
from pathlib import Path

# Configuration for your 6 ORF directories
EXPERIMENTS = [
    {
        'name': 'ORF Rand 1',
        'base_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF',
        'prediction_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/orf_predictions_rand1',
        'metadata_path': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/test_data/dataset_metadata.csv',
    },
    {
        'name': 'ORF Rand 2',
        'base_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF',
        'prediction_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/orf_predictions_rand2',
        'metadata_path': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/test_data/dataset_metadata.csv',
    },
    {
        'name': 'ORF Rand 3',
        'base_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF',
        'prediction_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/orf_predictions_rand3',
        'metadata_path': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/test_data/dataset_metadata.csv',
    },
    {
        'name': 'ORF Rand 4',
        'base_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF',
        'prediction_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/orf_predictions_rand4',
        'metadata_path': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/test_data/dataset_metadata.csv',
    },
    {
        'name': 'ORF Rand 5',
        'base_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF',
        'prediction_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/orf_predictions_rand5',
        'metadata_path': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/test_data/dataset_metadata.csv',
    },
    {
        'name': 'ORF Rand 6',
        'base_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF',
        'prediction_dir': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/orf_predictions_rand6',
        'metadata_path': '/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_ORF/test_data/dataset_metadata.csv',
    },
]

def run_mapping(script_path, base_dir, prediction_dir, metadata_path):
    """
    Run the mapping script for a single experiment
    """
    cmd = [
        'python',
        script_path,
        base_dir,
        prediction_dir,
        metadata_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def main():
    # Get the path to the flexible mapping script
    script_dir = Path(__file__).parent
    mapping_script = script_dir / 'map_biorand_test_metadata_flexible.py'
    
    if not mapping_script.exists():
        print(f"❌ ERROR: Mapping script not found at {mapping_script}")
        print("   Make sure map_biorand_test_metadata_flexible.py is in the same directory")
        sys.exit(1)
    
    print("=" * 70)
    print("BATCH METADATA MAPPING")
    print("=" * 70)
    print(f"Processing {len(EXPERIMENTS)} experiments\n")
    
    results = []
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(EXPERIMENTS)}] Processing: {exp['name']}")
        print(f"{'='*70}")
        print(f"Prediction dir: {exp['prediction_dir']}")
        
        # Check if prediction directory exists
        if not Path(exp['prediction_dir']).exists():
            print(f"⚠️  WARNING: Prediction directory does not exist, skipping...")
            results.append({'name': exp['name'], 'status': 'SKIPPED', 'reason': 'Directory not found'})
            continue
        
        # Check if metadata file exists
        if not Path(exp['metadata_path']).exists():
            print(f"⚠️  WARNING: Metadata file does not exist, skipping...")
            results.append({'name': exp['name'], 'status': 'SKIPPED', 'reason': 'Metadata not found'})
            continue
        
        # Run the mapping
        result = run_mapping(
            str(mapping_script),
            exp['base_dir'],
            exp['prediction_dir'],
            exp['metadata_path']
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCESS")
            results.append({'name': exp['name'], 'status': 'SUCCESS'})
        else:
            print(f"❌ FAILED")
            print(f"\nError output:")
            print(result.stderr)
            results.append({'name': exp['name'], 'status': 'FAILED', 'error': result.stderr[:200]})
        
        # Show abbreviated output
        output_lines = result.stdout.split('\n')
        # Show first 5 and last 5 lines
        if len(output_lines) > 20:
            print('\n'.join(output_lines[:5]))
            print(f"\n... ({len(output_lines) - 10} lines omitted) ...\n")
            print('\n'.join(output_lines[-5:]))
        else:
            print(result.stdout)
    
    # Print summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed_count = sum(1 for r in results if r['status'] == 'FAILED')
    skipped_count = sum(1 for r in results if r['status'] == 'SKIPPED')
    
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"  ✅ Successful: {success_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  ⚠️  Skipped: {skipped_count}")
    
    print(f"\nDetailed results:")
    for r in results:
        status_symbol = '✅' if r['status'] == 'SUCCESS' else '❌' if r['status'] == 'FAILED' else '⚠️'
        print(f"  {status_symbol} {r['name']:20s} - {r['status']}")
        if r['status'] == 'SKIPPED':
            print(f"      Reason: {r.get('reason', 'Unknown')}")
    
    print(f"\n{'='*70}")
    
    if failed_count > 0:
        print("\n⚠️  Some experiments failed. Review the error messages above.")
        sys.exit(1)
    else:
        print("\n🎉 All experiments processed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()

"""
TorchXRayVision Embedding Chunk Merger

Merges chunked TorchXRayVision embeddings into format compatible with downstream
linear probing pipeline.

=============================================================================
OUTPUT FILES (matching downstream expectations)
=============================================================================

Creates organized folder structure:
- xrv_d/xrv_drall_dicom.npz  (DenseNet: 1024×7×7 spatial features)
- xrv_r/xrv_rrall_dicom.npz  (ResNet: 2048-dim pooled features)

Each file structure:
├── filenames: (N,) string array (dicom_ids)
└── embeddings: (N,) object array
    └── DenseNet: each element (1024, 7, 7)
    └── ResNet: each element (2048,)

=============================================================================
"""

import glob
import numpy as np
import os
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge TorchXRayVision embedding chunks for downstream pipeline'
    )
    
    parser.add_argument('--chunk_pattern', type=str, required=True,
                       help='Glob pattern for chunk files')
    parser.add_argument('--output_base_dir', type=str, required=True,
                       help='Base directory for organized output')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['densenet', 'resnet'],
                       help='Model type (determines output folder name)')
    parser.add_argument('--verify_dimensions', action='store_true',
                       help='Verify all embeddings have consistent dimensions')
    
    return parser.parse_args()


def load_chunks(chunk_files):
    """Load all chunk files and concatenate embeddings."""
    all_filenames = []
    all_embeddings = []
    
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        data = np.load(chunk_file, allow_pickle=True)
        
        if 'filenames' not in data or 'embeddings' not in data:
            raise KeyError(f"Chunk {chunk_file} missing required keys")
        
        all_filenames.extend(data['filenames'])
        all_embeddings.extend(data['embeddings'])
    
    filenames = np.array(all_filenames)
    
    embeddings_obj = np.empty(len(all_embeddings), dtype=object)
    for i, emb in enumerate(all_embeddings):
        embeddings_obj[i] = emb
    
    return filenames, embeddings_obj


def save_organized_output(output_base_dir, filenames, embeddings_obj, model_type):
    """
    Save in organized folder structure matching downstream expectations.
    
    Structure:
    output_base_dir/
    ├── xrv_d/xrv_drall_dicom.npz  (DenseNet)
    └── xrv_r/xrv_rrall_dicom.npz  (ResNet)
    """
    
    if model_type == 'densenet':
        folder_name = 'xrv_d'
        filename = 'xrv_drall_dicom.npz'
    else:  # resnet
        folder_name = 'xrv_r'
        filename = 'xrv_rrall_dicom.npz'
    
    folder_path = os.path.join(output_base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    output_path = os.path.join(folder_path, filename)
    np.savez(output_path, filenames=filenames, embeddings=embeddings_obj)
    
    file_size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"Saved: {folder_name}/{filename} ({file_size_mb:.1f} MB)")
    print(f"  Sample embedding shape: {embeddings_obj[0].shape}")


def verify_dimensions(embeddings_obj, model_type):
    """Verify that all embeddings have consistent dimensions."""
    expected_shape = (1024, 7, 7) if model_type == 'densenet' else (2048,)
    
    print(f"\nVerifying dimensions (expected: {expected_shape})...")
    
    for i in range(len(embeddings_obj)):
        if embeddings_obj[i].shape != expected_shape:
            raise ValueError(f"Embedding {i} has shape {embeddings_obj[i].shape}, "
                           f"expected {expected_shape}")
    
    print("Dimension verification passed")


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("TorchXRayVision Embedding Chunk Merger (Downstream Compatible)")
    print("="*80)
    print(f"Model type: {args.model_type}")
    print(f"Chunk pattern: {args.chunk_pattern}")
    print(f"Output base directory: {args.output_base_dir}")
    print("="*80 + "\n")
    
    chunk_files = sorted(glob.glob(args.chunk_pattern))
    
    if len(chunk_files) == 0:
        raise FileNotFoundError(f"No chunks found matching pattern: {args.chunk_pattern}")
    
    print(f"Found {len(chunk_files)} chunk files\n")
    
    filenames, embeddings_obj = load_chunks(chunk_files)
    
    print(f"Total images: {len(filenames)}")
    print(f"Sample embedding shape: {embeddings_obj[0].shape}")
    
    if args.verify_dimensions:
        verify_dimensions(embeddings_obj, args.model_type)
    
    save_organized_output(args.output_base_dir, filenames, embeddings_obj, args.model_type)
    
    print("\n" + "="*80)
    print("Merge complete! Files ready for downstream pipeline.")
    print("="*80)


if __name__ == "__main__":
    main()
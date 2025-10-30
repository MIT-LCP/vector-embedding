"""
MedSigLIP Embedding Chunk Merger

Merges chunked MedSigLIP embeddings into format compatible with downstream
linear probing pipeline.
"""

import glob
import numpy as np
import os
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge MedSigLIP embedding chunks for downstream pipeline'
    )
    
    parser.add_argument('--chunk_pattern', type=str, required=True)
    parser.add_argument('--output_base_dir', type=str, required=True)
    parser.add_argument('--verify_dimensions', action='store_true')
    
    return parser.parse_args()


def load_chunks(chunk_files):
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


def save_organized_output(output_base_dir, filenames, embeddings_obj):
    """Save in organized folder structure: medsiglip/medsiglip_all.npz"""
    
    folder_path = os.path.join(output_base_dir, 'medsiglip')
    os.makedirs(folder_path, exist_ok=True)
    
    output_path = os.path.join(folder_path, 'medsiglip_all.npz')
    np.savez(output_path, filenames=filenames, embeddings=embeddings_obj)
    
    file_size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"Saved: medsiglip/medsiglip_all.npz ({file_size_mb:.1f} MB)")
    print(f"  Embedding dimension: {embeddings_obj[0].shape[0]}")


def verify_dimensions(embeddings_obj, expected_dim=1152):
    print(f"\nVerifying dimensions (expected: {expected_dim})...")
    
    for i in range(len(embeddings_obj)):
        if embeddings_obj[i].shape[0] != expected_dim:
            raise ValueError(f"Embedding {i} has dimension {embeddings_obj[i].shape[0]}")
    
    print("Dimension verification passed")


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("MedSigLIP Embedding Chunk Merger (Downstream Compatible)")
    print("="*80)
    
    chunk_files = sorted(glob.glob(args.chunk_pattern))
    
    if len(chunk_files) == 0:
        raise FileNotFoundError(f"No chunks found")
    
    print(f"Found {len(chunk_files)} chunk files\n")
    
    filenames, embeddings_obj = load_chunks(chunk_files)
    
    print(f"Total images: {len(filenames)}")
    print(f"Sample embedding shape: {embeddings_obj[0].shape}")
    
    if args.verify_dimensions:
        verify_dimensions(embeddings_obj)
    
    save_organized_output(args.output_base_dir, filenames, embeddings_obj)
    
    print("\n" + "="*80)
    print("Merge complete!")
    print("="*80)


if __name__ == "__main__":
    main()
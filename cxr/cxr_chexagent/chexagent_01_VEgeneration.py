"""
CheXagent Batch Embedding Extraction Script

This script processes large collections of chest X-ray DICOM files and generates 
embeddings using Stanford AIMI's CheXagent-8b model. Includes batch processing 
and resume capability.

=============================================================================
REFERENCES
=============================================================================

**Primary Citation:**
Chen, Z., et al. (2023). "CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation"
arXiv preprint arXiv:2401.12208
https://arxiv.org/abs/2401.12208

**Model Repository:**
- HuggingFace: https://huggingface.co/StanfordAIMI/CheXagent-8b
- GitHub: https://github.com/Stanford-AIMI/CheXagent

=============================================================================
MODEL DOWNLOAD & REQUIREMENTS
=============================================================================

**Download HugginFace Model to Model Directory** (Size: ~33GB (7 safetensors files + configuration))

**Python Environment Setup:**
```bash
# Create and activate virtual environment
python -m venv chexagent_env
source chexagent_env/bin/activate  # Mac/Linux
# OR: chexagent_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

=============================================================================
INPUT DATA REQUIREMENTS
=============================================================================

**Required Input:**
- CSV file with pre-filtered AP/PA DICOMs:
  - `dicom_path`: Full path to DICOM file on local system
  - `dicom_id`: Unique identifier for each DICOM
- CheXagent model directory (local path)
- Output directory for NPZ files

**Important:** This script processes ALL rows in the CSV without filtering by 
ViewPosition. Ensure your CSV is pre-filtered to include only PA (Posterior-Anterior) 
or AP (Anterior-Posterior) views before running.

**For MIMIC-CXR preprocessing:**
Use the metadata and record list files to filter by ViewPosition and verify DICOM paths:
- Metadata: https://physionet.org/content/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz
- Record List: https://physionet.org/content/mimic-cxr/2.1.0/cxr-record-list.csv.gz

=============================================================================
USAGE EXAMPLES
=============================================================================

**Basic usage:**
```bash
python run_single_file_extraction.py \
  --csv_path verified_dicom_paths.csv \
  --model_dir ./CheXagent-8b \
  --output_dir ./embeddings_output/
```

**Large-scale processing:**
```bash
python run_single_file_extraction.py \
  --csv_path /data/mimic-cxr/verified_ap_pa_dicoms.csv \
  --model_dir /models/CheXagent-8b \
  --output_dir /output/embeddings/ \
  --batch_size 200 \
  --device cuda
```

**Resume interrupted processing:**
```bash
python run_single_file_extraction.py \
  --csv_path verified_dicom_paths.csv \
  --model_dir ./CheXagent-8b \
  --output_dir ./embeddings_output/ \
  --start_idx 5000
```

=============================================================================
OUTPUT STRUCTURE
=============================================================================

**Output Format:** NPZ files (batch_000.npz, batch_001.npz, ...)

Each NPZ file contains:
- `embeddings`: NumPy array of shape (N, 1, 128, 768)
  - N: Number of successfully processed DICOMs in batch
  - 1: Singleton dimension for compatibility
  - 128: Number of Q-Former query tokens
  - 768: Embedding dimension per token
  
- `filenames`: NumPy array of shape (N,) containing dicom_id values

**Why Q-Former embeddings?**
- Fixed-size representation (128 tokens) regardless of image size
- Semantic compression of visual features
- Better generalization for downstream tasks
- Computational efficiency

=============================================================================
"""

import os
import gc
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from extract_embeddings import CheXagentWithEmbeddingExtraction


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate CheXagent embeddings with batch processing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--csv_path', type=str, required=True,
                       help='CSV file with columns: dicom_path, dicom_id')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Path to local CheXagent model directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for batch NPZ files')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Number of DICOMs per output file (default: 100)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'],
                       help='Device for inference (default: auto-select)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index to resume processing (default: 0)')
    
    return parser.parse_args()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_existing_batches(output_dir):
    """
    Identify already-processed batches for resume functionality.
    
    Returns:
        set: Set of batch IDs that have been processed
    """
    if not os.path.exists(output_dir):
        return set()
    
    existing_batches = set()
    for filename in os.listdir(output_dir):
        if filename.startswith("batch_") and filename.endswith(".npz"):
            try:
                batch_num = int(filename.split("_")[1].split(".")[0])
                existing_batches.add(batch_num)
            except (ValueError, IndexError):
                continue
    
    return existing_batches


def calculate_batch_range(total_files, start_idx, batch_size):
    """
    Calculate range of batches to process.
    
    Returns:
        tuple: (start_batch_id, end_batch_id)
    """
    start_batch = start_idx // batch_size
    end_batch = (total_files - 1) // batch_size
    return start_batch, end_batch


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def run_batch_extraction(csv_path, output_dir, model_dir, batch_size=100, device=None, start_idx=0):
    """
    Main batch processing function for CheXagent embeddings.
    
    Processing Steps:
    1. Load input CSV and validate
    2. Auto-select device if not specified
    3. Calculate batch range and identify existing batches
    4. Initialize CheXagent model
    5. Process each batch sequentially
    6. Save results with error handling
    """
    
    # STEP 1: Device selection
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("[INFO] Auto-selected device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"[INFO] Auto-selected device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            device = "cpu"
            print("[INFO] Auto-selected device: CPU")
    else:
        print(f"[INFO] Using device: {device}")

    # STEP 2: Load and validate CSV
    print(f"[INFO] Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    total_files = len(df)
    print(f"[INFO] Total files: {total_files}")
    
    if 'dicom_path' not in df.columns or 'dicom_id' not in df.columns:
        raise ValueError("CSV must contain 'dicom_path' and 'dicom_id' columns")
    
    # STEP 3: Calculate batch range
    start_batch, end_batch = calculate_batch_range(total_files, start_idx, batch_size)
    print(f"[INFO] Processing batches {start_batch} to {end_batch} (batch_size={batch_size})")
    
    # STEP 4: Setup output and identify existing batches
    os.makedirs(output_dir, exist_ok=True)
    existing_batches = get_existing_batches(output_dir)
    if existing_batches:
        print(f"[INFO] Found {len(existing_batches)} existing batches")
    
    batches_to_process = [b for b in range(start_batch, end_batch + 1) if b not in existing_batches]
    
    if not batches_to_process:
        print(f"[INFO] All batches already processed. Delete NPZ files from {output_dir} to reprocess.")
        return
    
    print(f"[INFO] Processing {len(batches_to_process)} new batches")
    
    # STEP 5: Initialize model
    print(f"[INFO] Loading CheXagent model...")
    model = CheXagentWithEmbeddingExtraction(model_dir, device=device)
    
    # STEP 6: Process each batch
    for batch_id in tqdm(batches_to_process, desc="Processing Batches"):
        batch_start_idx = batch_id * batch_size
        batch_end_idx = min((batch_id + 1) * batch_size - 1, total_files - 1)
        
        batch_df = df.iloc[batch_start_idx:batch_end_idx + 1]
        dicom_paths = batch_df["dicom_path"].tolist()
        dicom_ids = batch_df["dicom_id"].tolist()
        
        save_path = os.path.join(output_dir, f"batch_{batch_id:03d}.npz")
        print(f"\n[BATCH {batch_id}] Processing {len(dicom_paths)} files (idx {batch_start_idx}-{batch_end_idx})")
        
        all_embeddings = []
        all_ids = []
        
        # Process each DICOM
        for i, (dicom_path, dicom_id) in enumerate(zip(dicom_paths, dicom_ids)):
            try:
                pil_img = model.dicom_to_rgb_pil(dicom_path)
                with torch.no_grad():
                    embedding = model.extract_embeddings_from_image(pil_img)
                all_embeddings.append(embedding["qformer_embedding"])
                all_ids.append(dicom_id)
            except Exception as e:
                global_idx = batch_start_idx + i
                print(f"[ERROR] idx {global_idx} ({dicom_id}): {e}")
                continue
        
        # Save batch
        if all_embeddings:
            try:
                qformer_emb = np.stack(all_embeddings)
                if qformer_emb.ndim == 3:
                    qformer_emb = np.expand_dims(qformer_emb, axis=1)
                
                np.savez(save_path, filenames=np.array(all_ids), embeddings=qformer_emb)
                print(f"[SUCCESS] {save_path} ({len(all_ids)}/{len(dicom_paths)} successful)")
                
                del all_embeddings, qformer_emb
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[ERROR] Saving batch {batch_id}: {e}")
        else:
            print(f"[WARNING] Batch {batch_id} produced no valid embeddings")
    
    # Summary
    print(f"\n[COMPLETE] Processed batches {start_batch} to {end_batch}")
    print(f"[OUTPUT] {output_dir}")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    try:
        run_batch_extraction(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            device=args.device,
            start_idx=args.start_idx
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Use --start_idx to resume")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
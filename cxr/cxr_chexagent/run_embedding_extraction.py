# this script gets qformer embeddings from CheXagent with shape [N, 1, 128, 768]

# The csv file should contain dicom path and dicom id of the images you want to extract embeddings from
# To run this script, you need to have the CheXagent-8b model from hugginface downloaded and the path to the model directory

import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from extract_embeddings import CheXagentWithEmbeddingExtraction

def run_single_file_extraction(
    csv_path,
    output_dir,
    model_dir,
    batch_size=20,  # This is now how many results to save in one file
    device=None,
    start_idx=0
):
    # Auto-select device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"[INFO] Using device: {device}")

    # Load CSV of DICOM paths and IDs
    df = pd.read_csv(csv_path)
    dicom_paths = df["dicom_path"].tolist()
    dicom_ids = df["dicom_id"].tolist()

    os.makedirs(output_dir, exist_ok=True)

    # Initialize model once
    model = CheXagentWithEmbeddingExtraction(model_dir, device=device)

    total = len(dicom_paths)
    all_embeddings = []
    all_ids = []
    
    # Process each DICOM file individually
    for idx in tqdm(range(start_idx, total), desc="Processing DICOMs"):
        dicom_path = dicom_paths[idx]
        dicom_id = dicom_ids[idx]
        
        # Save every batch_size embeddings to a file
        batch_id = idx // batch_size
        save_path = os.path.join(output_dir, f"batch_{batch_id:03d}.npz")
        
        # Skip if this batch file already exists
        if os.path.exists(save_path) and (idx % batch_size) == 0:
            print(f"[SKIP] Batch {batch_id:03d} already exists")
            continue
        
        try:
            # Load and process the DICOM
#            print(f"[PROCESS] Image {idx}: {dicom_path}")
            pil_img = model.dicom_to_rgb_pil(dicom_path)
#            print(f"[DEBUG] Image mode={pil_img.mode}, size={pil_img.size}")
            
            # Extract embedding
            with torch.no_grad():
                embedding = model.extract_embeddings_from_image(pil_img)
            
            # Save the embedding and ID
            all_embeddings.append(embedding["qformer_embedding"])
            all_ids.append(dicom_id)
            
#            print(f"[SUCCESS] Processed image {idx}")
        except Exception as e:
            print(f"[ERROR] Failed on image {idx}: {e}")
            continue
        
        # Save batch of embeddings when we reach batch_size
        if (idx + 1) % batch_size == 0 or idx == total - 1:
            if all_embeddings:
                try:
                    # Stack all collected embeddings
                    qformer_emb = np.stack(all_embeddings)
                    
                    # Ensure correct shape [N, 1, 128, 768]
                    if qformer_emb.ndim == 3:  # [N, 128, 768]
                        qformer_emb = np.expand_dims(qformer_emb, axis=1)
                    
                    # Save the batch
                    np.savez(save_path, filenames=np.array(all_ids), embeddings=qformer_emb)
                    print(f"[SAVED] Batch {batch_id:03d} → {save_path} ({len(all_ids)} images)")
                    
                    # Clear the lists for the next batch
                    all_embeddings = []
                    all_ids = []
                    
                    # Clean up memory
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[ERROR] Failed to save batch {batch_id}: {e}")
    
    print(f"[DONE] Processed {total} DICOM files")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single-file embedding extraction for CheXagent")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file with dicom_path and dicom_id")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store output .npz files")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to local CheXagent model")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of results to save in one file")
    parser.add_argument("--device", type=str, default=None, help="Device to use: mps, cuda, cpu")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index to resume from")

    args = parser.parse_args()
    run_single_file_extraction(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        device=args.device,
        start_idx=args.start_idx
    )
    
    
    
# python run_embedding_extraction.py \
#   --csv_path /Volumes/code/my_project/mimiccxr/scripts/verified_dicom_paths.csv \
#   --model_dir /Volumes/code/my_project/mimiccxr_chexagent/CheXagent-8b \
#   --output_dir /Volumes/code/my_project/mimiccxr_chexagent/data/ \
#   --batch_size 100

#   --start_idx 1000 ...




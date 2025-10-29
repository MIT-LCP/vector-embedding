"""
PanEcho Vector Embedding Extraction for Transfer Learning
Main script for extracting embeddings from DICOM echocardiogram data using PanEcho model.
For VS Code Interactive Window execution with CSV output.
"""

import os, sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, "vector-embedding/echo/echo_panecho")
from utils_panecho import (
    DICOMProcessor, 
    PanEchoEmbeddingExtractor)

"""
to use this script, set the model path in utils_panecho.py and the DICOMPATH_CSV and VE_OUTPUT_PATH variables below.
"""
DICOMPATH_CSV = "**/**.csv" # file containing dicom_path column as "dicom_path"
VE_OUTPUT_PATH = "**/ve_panecho.csv"

def main(dicom_paths,output_path):
    """Main function for embedding extraction with CSV output."""
    
    # ========== parameters ==========   
    num_frames = 16 
    image_size = 224 

    model_type = 'backbone'                 # 'backbone' or 'image_encoder'
    device = 'auto'                         # 'auto', 'cpu', 'cuda:0'
    
    # ========== 処理開始 ==========
    
    print("=== PanEcho Embedding Extraction (CSV Output) ===")
    print(f"Input DICOM files: {len(dicom_paths)} files")
    print(f"Output CSV: {output_path}")
    print(f"Model type: {model_type}")
    print(f"Frames: {num_frames}, Image size: {image_size}")
    
    print("Initializing DICOM processor...")
    processor = DICOMProcessor()
    
    print("Loading PanEcho model...")
    extractor = PanEchoEmbeddingExtractor(
        model_type=model_type,
        device=device
    )
    
    print(f"Using device: {extractor.device}")

    print("Writing CSV header...")
    with open(output_path, 'w') as f:
        ve_columns = [f"ve{str(i+1).zfill(3)}" for i in range(768)]
        header = ["dicom_path"] + ve_columns
        f.write(",".join(header) + "\n")
    
    print("Processing DICOM files...")
    processed_count = 0
    failed_count = 0
    
    for i, file_path in enumerate(tqdm(dicom_paths, desc="Processing DICOMs")):
        if file_path.endswith('.dcm'):
            try:
                tensor_data, _ = processor.process_dicom(
                    file_path, 
                    num_frames=num_frames,
                    image_size=image_size
                )
                
                if tensor_data is not None:
                    echocardiogram_tensor = tensor_data.unsqueeze(0)  
                    with torch.no_grad():
                        echocardiogram_tensor = echocardiogram_tensor.to(extractor.device)
                        embeddings = extractor.model(echocardiogram_tensor).cpu().numpy()
                    
                    with open(output_path, 'a') as f:
                        embedding_str = ",".join([f"{x:.16e}" for x in embeddings.flatten()])
                        row_data = f"{file_path},{embedding_str}\n"
                        f.write(row_data)
                    
                    processed_count += 1
                else:
                    failed_count += 1
                    print(f"Failed to process: {file_path}")
                    
            except Exception as e:
                failed_count += 1
                print(f"Error processing {file_path}: {str(e)}")

    print(f"\n=== Processing completed ===")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Output saved to: {output_path}")
    
    return output_path, processed_count, failed_count


if __name__ == '__main__':
    test_file = DICOMPATH_CSV
    test_df = pd.read_csv(test_file)
    dicom_paths = test_df['dicom_path'].tolist()
    output_path = VE_OUTPUT_PATH
    main(dicom_paths, output_path)
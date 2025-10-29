import sys
sys.path.insert(0, "vector-embedding/echo/echo_echoprime")

import torch
import torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
import utils_echoprime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
to use this script, the ECHO_PRIME_ENCODER_PATH, DICOMPATH_CSV and VE_OUTPUT_PATH variables below.
"""
ECHO_PRIME_ENCODER_PATH = "**/echo_prime_encoder.pt"
DICOMPATH_CSV = "**/**.csv" # file containing dicom_path column as "dicom_path"
VE_OUTPUT_PATH = "**/ve_echoprime.csv"


def load_trained_model( model_file ):
    checkpoint = torch.load(model_file, 
                           map_location=device)
    echo_encoder = torchvision.models.video.mvit_v2_s()
    echo_encoder.head[-1] = torch.nn.Linear(echo_encoder.head[-1].in_features, 512)
    echo_encoder.load_state_dict(checkpoint)
    
    echo_encoder.eval().to(device)
    torch.set_grad_enabled(False)  
    
    return echo_encoder


def main(model_file, input_file, output_file):
    print("Loading trained model...")
    model = load_trained_model(model_file = model_file)

    input_df = pd.read_csv(input_file)
    dicom_paths = input_df['dicom_path'].tolist()
    
    print(f"Processing {len(dicom_paths)} DICOM files...")

    with open(output_file, 'w') as f:
        ve_columns = [f"ve{str(i+1).zfill(3)}" for i in range(512)]
        header = ["dicom_path"] + ve_columns
        f.write(",".join(header) + "\n")
    

    batch_size = 8
    for i in tqdm(range(0, len(dicom_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = dicom_paths[i:i+batch_size]
        batch_videos = []
        valid_paths = []

        for path in batch_paths:
            try:
                video = utils_echoprime.process_dicom_echoprime(path)
                batch_videos.append(video)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if not batch_videos:
            continue
            
        try:
            batch_tensor = torch.cat(batch_videos, dim=0).to(device)
            embeddings = model(batch_tensor).cpu().numpy()
            
            with open(output_file, 'a') as f:
                for path, embedding in zip(valid_paths, embeddings):
                    row_data = [path] + embedding.tolist()
                    f.write(",".join(map(str, row_data)) + "\n")
                    
        except Exception as e:
            print(f"Error in batch processing: {e}")
            continue
    
    print(f"✅ Embeddings saved to: {output_file}")

if __name__ == "__main__":
    main(
        model_file = ECHO_PRIME_ENCODER_PATH, 
        input_file = DICOMPATH_CSV,
        output_file = VE_OUTPUT_PATH )

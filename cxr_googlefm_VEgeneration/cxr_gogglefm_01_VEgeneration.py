### Script usage ###
# To setup google/cxr-foundation of Hugging Face (https://huggingface.co/google/cxr-foundation) to generate vector embeddings for MIMIC CXR
# Original codes modified from https://colab.research.google.com/github/google-health/cxr-foundation/blob/master/notebooks/quick_start_with_hugging_face.py

### 250210 AH


### 1.Download the model (If haven't done previously) 

from huggingface_hub.utils import HfFolder
from huggingface_hub import notebook_login

if HfFolder.get_token() is None:
    from huggingface_hub import notebook_login
    notebook_login()

# Install Required Dependencies - needed to downgrade tensorflow and tensorflow-text due to difference in version
!pip install huggingface_hub tensorflow tensorflow-text pydicom hcls_imaging_ml_toolkit retrying pypng
!pip install tensorflow-hub
!pip install pydicom

# Download the model
from huggingface_hub import snapshot_download
model_id = "google/cxr-foundation"
local_model_path = snapshot_download(
    repo_id=model_id,
    cache_dir="/home/sagemaker-user/.cache/huggingface"  # or another path you prefer
)
print(f"Model downloaded to: {local_model_path}")
# Model downloaded to: /home/sagemaker-user/.cache/huggingface/models--google--cxr-foundation/snapshots/fcef78915362e661d0e478dc460750727c615e96

from huggingface_hub import list_repo_refs
refs = list_repo_refs("google/cxr-foundation")
print("Available versions:", refs)



### 2. VE generation of MIMIC CXR

# upload MIMIC CXR to S3 bucket using https://github.com/MIT-LCP/vector-embedding/blob/main/cxr_data_preprocess/data_upload.py
# current usecase uses 15 dicom files in subdirectories of s3://resresearcher-lcp-takeshi-group-767397697436/ah_trial/

import os
import tensorflow as tf
import tensorflow_text  # Required for text operations
import numpy as np
import pydicom
import png
import io
from pathlib import Path
import boto3
import tempfile
from datetime import datetime

local_model_path = "/home/sagemaker-user/.cache/huggingface/models--google--cxr-foundation/snapshots/fcef78915362e661d0e478dc460750727c615e96"

def dicom_to_serialized_tfexample(dicom_path):
    """
    Converts a DICOM file to a serialized TensorFlow Example.
    This function follows the same preprocessing steps as the GitHub example:
    1. Reads the DICOM file
    2. Normalizes the image
    3. Handles both 8-bit and 16-bit images appropriately
    4. Converts to PNG format
    5. Creates a TensorFlow Example with the required features
    """
    # Read DICOM file
    dicom = pydicom.dcmread(dicom_path)
    image_array = dicom.pixel_array.astype(np.float32)

    # Normalize image by shifting minimum to zero
    image_array -= image_array.min()

    # Handle bit depth appropriately
    if dicom.BitsAllocated == 8:
        pixel_array = image_array.astype(np.uint8)
        bitdepth = 8
    else:
        # Scale to full 16-bit range for better precision
        max_val = image_array.max()
        if max_val > 0:
            image_array *= 65535 / max_val
        pixel_array = image_array.astype(np.uint16)
        bitdepth = 16

    # Convert to PNG format in memory
    output = io.BytesIO()
    png.Writer(
        width=pixel_array.shape[1],
        height=pixel_array.shape[0],
        greyscale=True,
        bitdepth=bitdepth
    ).write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    # Create TensorFlow Example with required features
    example = tf.train.Example()
    features = example.features.feature
    features['image/encoded'].bytes_list.value.append(png_bytes)
    features['image/format'].bytes_list.value.append(b'png')

    return example.SerializeToString()

def process_dicom_files(bucket_name, prefix, model_base_path):
    """
    Processes all DICOM files from S3 in a single batch.
    
    Args:
        bucket_name: S3 bucket name.
        prefix: S3 prefix where DICOM files are stored.
        model_base_path: Base path where models are stored.
    """

    elixrc_model = tf.saved_model.load(os.path.join(local_model_path, "elixr-c-v2-pooled"))
    elixrc_infer = elixrc_model.signatures['serving_default']  # Add this line
    qformer_model = tf.saved_model.load(os.path.join(local_model_path, "pax-elixr-b-text"))


    # Initialize S3 client
    s3_client = boto3.client('s3')
    results = []
    
    # List all DICOM files in S3
    print(f"Finding DICOM files in s3://{bucket_name}/{prefix}")
    paginator = s3_client.get_paginator('list_objects_v2')
    dicom_files = []
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            if obj['Key'].lower().endswith('.dcm'):
                dicom_files.append(obj['Key'])

    print(f"Found {len(dicom_files)} DICOM files")

    # Process all files at once
    all_paths = []
    all_elixrc_embeddings = []
    all_elixrb_embeddings = []
    
    for dicom_key in dicom_files:
        try:
            print(f"\nProcessing {dicom_key}")
            
            # Download DICOM file to a temporary location
            with tempfile.NamedTemporaryFile(suffix='.dcm') as temp_file:
                s3_client.download_file(bucket_name, dicom_key, temp_file.name)
                
                # Convert DICOM to TensorFlow Example
                serialized_example = dicom_to_serialized_tfexample(temp_file.name)
                
                # Step 1: Get ELIXR-C embeddings (interim embeddings)
                print("Getting ELIXR-C embeddings...")
                elixrc_output = elixrc_infer(input_example=tf.constant([serialized_example]))
                elixrc_embedding = elixrc_output['feature_maps_0'].numpy()
                print(f"ELIXR-C embedding shape: {elixrc_embedding.shape}")
                
                # Step 2: Get QFormer embeddings (final embeddings)
                print("Getting QFormer embeddings...")
                qformer_input = {
                    'image_feature': elixrc_embedding.tolist(),
                    'ids': np.zeros((1, 1, 128), dtype=np.int32).tolist(),
                    'paddings': np.zeros((1, 1, 128), dtype=np.float32).tolist(),
                }
                
                qformer_output = qformer_model.signatures['serving_default'](**qformer_input)
                elixrb_embedding = qformer_output['all_contrastive_img_emb'].numpy()
                print(f"ELIXR-B embedding shape: {elixrb_embedding.shape}")
                
                # Store results
                all_paths.append(f"s3://{bucket_name}/{dicom_key}")
                all_elixrc_embeddings.append(elixrc_embedding)
                all_elixrb_embeddings.append(elixrb_embedding)
                
        except Exception as e:
            print(f"Error processing {dicom_key}: {str(e)}")
            continue
    
    # Save all results in a `.npz` file
    if all_paths:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = "embeddings_all"
        os.makedirs(output_dir, exist_ok=True)
        
        np.savez(
            os.path.join(output_dir, 'embeddings_all.npz'),
            paths=np.array(all_paths),
            elixrc_embeddings=np.array(all_elixrc_embeddings),
            elixrb_embeddings=np.array(all_elixrb_embeddings)
        )
        
        print(f"Saved all embeddings to {output_dir}/embeddings_all.npz")

    return all_paths  # Return the processed file paths

# Run processing
if __name__ == "__main__":
    bucket_name = "resresearcher-lcp-takeshi-group-767397697436"
    input_prefix = "ah_trial"
    model_base_path = "/home/sagemaker-user/cxr_foundation/script/cxr_model/models--google--cxr-foundation"
    
    # Process all files at once
    results = process_dicom_files(bucket_name, input_prefix, model_base_path)
    print(f"\nProcessed {len(results)} files successfully")


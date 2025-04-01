### Script usage ###
# Uses https://github.com/MIT-LCP/vector-embedding/blob/main/cxr_data_preprocess/data_upload.py
# to upload published VE of MIMIC CXR to AWS S3

# 250210 AH

import os
import subprocess
import boto3
from pathlib import Path
import shutil
import requests
import pandas as pd
import io
import json
from requests.auth import HTTPBasicAuth
import argparse

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Script to download and upload entire subdirectories to S3.')
    parser.add_argument('--credential_path', type=str, required=True, help='Physionet Credential json file')
    parser.add_argument('--bucket_name', type=str, required=True, help='S3 bucket for data uploading')
    parser.add_argument('--s3_dir', type=str, required=True, help='S3 directory name for data uploading')
    parser.add_argument('--upload_num', type=int, required=False, help='Number of directories to process')
    return parser.parse_args()

def load_credentials(credential_path):
    """Load credentials from the specified JSON file."""
    with open(credential_path, 'r') as file:
        return json.load(file)

def fetch_study_list(mimic_cxr_path, username, password):
    """Fetch study list from Physionet and return it as a DataFrame."""
    url = mimic_cxr_path + 'cxr-study-list.csv.gz'
    headers = {'User-Agent': 'Wget/1.21.3 (linux-gnu)'}
    response = requests.get(url, auth=HTTPBasicAuth(username, password), headers=headers)
    file_content = io.BytesIO(response.content)
    return pd.read_csv(file_content, compression='gzip', encoding='utf-8')

def get_embedding_paths(path_df, mimic_cxr_path, embedding_base_path):
    """Generate subdirectory paths for downloading (entire subdirectories)."""
    return [embedding_base_path + path + '/' for path in path_df['path'].apply(lambda x: os.sep.join(x.split(os.sep)[:3])).drop_duplicates().tolist()]

def download_and_upload_data(urls, temp_dir, s3_client, bucket_name, s3_dir, username, password):
    """Download entire subdirectories using wget and upload them to S3."""
    for url in urls:
        # Create temporary directory for downloads
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Wget command to download **entire** subdirectories
        wget_command = [
            'wget',
            '--recursive',
            '--no-parent',
            '--no-host-directories',
            '--cut-dirs=4',
            '--no-check-certificate',
            '--user', username,
            '--password', password,
            '--directory-prefix=' + str(temp_dir),
            url
        ]
        
        # Run wget and capture output
        print(f"\nDownloading from: {url}")
        result = subprocess.run(wget_command, capture_output=True, text=True)
        
        if result.stderr:
            print("Wget stderr:", result.stderr)
            
        # Upload downloaded files to S3
        files_found = False
        for path in temp_dir.rglob('*'):
            if path.is_file():
                files_found = True
                relative_path = path.relative_to(temp_dir)
                s3_key = f"{s3_dir}/{relative_path}"
                print(f"Found file: {relative_path}")
                
                try:
                    print(f"Uploading to s3://{bucket_name}/{s3_key}")
                    with open(path, 'rb') as file_data:
                        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=file_data)
                    print(f"Successfully uploaded: {s3_key}")
                except Exception as e:
                    print(f"Error uploading {relative_path}: {str(e)}")
        
        if not files_found:
            print("Warning: No files found in temporary directory after download")
            
        # Clean up before next download
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

def main():
    """Main function to drive the script."""
    args = parse_arguments()
    
    # Load credentials
    credentials = load_credentials(args.credential_path)
    username = credentials.get('username')
    password = credentials.get('password')
    
    # Initialize S3 client and verify access
    s3_client = boto3.client('s3')
    try:
        s3_client.head_bucket(Bucket=args.bucket_name)
        print(f"Successfully connected to bucket: {args.bucket_name}")
    except Exception as e:
        print(f"Error accessing bucket: {str(e)}")
        raise
    
    # PhysioNet base paths
    mimic_cxr_path = 'https://physionet.org/files/mimic-cxr/2.1.0/'  # Old MIMIC-CXR path
    embedding_base_path = 'https://physionet.org/files/image-embeddings-mimic-cxr/1.0/'  # Image embeddings path

    # Fetch study list
    path_df = fetch_study_list(mimic_cxr_path, username, password)
    
    # Generate subdirectory paths for the image embeddings dataset
    embedding_paths = get_embedding_paths(path_df, mimic_cxr_path, embedding_base_path)
    urls = embedding_paths[:args.upload_num] if args.upload_num else embedding_paths
    
    # Create temporary directory
    temp_dir = Path('./temp_download')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    
    # Download and upload entire subdirectories
    download_and_upload_data(urls, temp_dir, s3_client, args.bucket_name, args.s3_dir, username, password)

if __name__ == '__main__':
    main()


# download 10 subdirectories
# python upload_s3_ah.py --credential_path /home/sagemaker-user/cxr_foundation/data/physionet_credential.json \
#   --bucket_name resresearcher-lcp-takeshi-group-767397697436 \
#   --s3_dir ah_trial/mimic_cxr_tfrecords \
#   --upload_num 10
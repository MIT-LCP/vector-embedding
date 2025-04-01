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
    parser = argparse.ArgumentParser(description='Script to download and upload data to S3.')
    parser.add_argument('--credential_path', type=str, required=True, help='Physionet Credential json file')
    parser.add_argument('--bucket_name', type=str, required=True, help='S3 bucket for data uploading')
    parser.add_argument('--s3_dir', type=str, required=True, help='S3 directory name for data uploading')
    parser.add_argument('--upload_num', type=int, required=False, help='Number of downloads to process')
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

def get_patient_paths(path_df, mimic_cxr_path):
    """Generate patient paths for downloading."""
    return [mimic_cxr_path + path + '/' for path in path_df['path'].apply(lambda x: os.sep.join(x.split(os.sep)[:3])).drop_duplicates().tolist()]

def download_and_upload_data(urls, temp_dir, s3_client, bucket_name, s3_dir, username, password):
    """Download files using wget and upload them to S3."""
    for url in urls:
        # Create temporary directory for downloads
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Run the wget command to download the data
        wget_command = [
            'wget', '-r', '-N', '-c', '-np', '--user', username, '--password', password,
            '-nH', '--cut-dirs=1', url
        ]
        os.chdir(temp_dir)
        subprocess.run(wget_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.chdir('..')

        # Upload downloaded files to S3
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file():
                s3_key = os.path.join(s3_dir, file_path.relative_to(temp_dir))
                with open(file_path, 'rb') as file_data:
                    s3_client.put_object(Bucket=bucket_name, Key=str(s3_key), Body=file_data)

        # Cleanup temporary directory
        shutil.rmtree(temp_dir)
        print(f"Uploaded to S3 bucket: {url}")

def main():
    """Main function to drive the script."""
    # Parse command-line arguments
    args = parse_arguments()

    # Load credentials
    credentials = load_credentials(args.credential_path)
    username = credentials.get('username')
    password = credentials.get('password')

    # Physionet MIMIC-CXR path
    mimic_cxr_path = 'https://physionet.org/files/mimic-cxr/2.1.0/'

    # Fetch study list from Physionet
    path_df = fetch_study_list(mimic_cxr_path, username, password)

    # Generate patient paths
    patient_paths = get_patient_paths(path_df, mimic_cxr_path)

    # Determine the number of URLs to download
    urls = patient_paths if args.upload_num is None else patient_paths[:args.upload_num]

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Create a temporary directory for downloads
    temp_dir = Path('./temp_download')

    # Download and upload data
    download_and_upload_data(urls, temp_dir, s3_client, args.bucket_name, args.s3_dir, username, password)

if __name__ == '__main__':
    main()

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

# Argments
parser = argparse.ArgumentParser(description='This script demonstrates how to use command-line arguments.')
parser.add_argument('--credential_path', type=str, required=True, help='Physionet Credential json file')
parser.add_argument('--bucket_name', type=int, required=True, help='s3 bucket for data uploading')
parser.add_argument('--s3_dir', type=int, required=True, help='s3 dir name for data uploading')
parser.add_argument('--download_num', type=int, required=False, help='Number for downloding')

args = parser.parse_args()

# Configure the S3 bucket
# args.bucket_name = 'resresearcher-lcp-takeshi-group-767397697436'
# args.s3_dir = 'rawdata/'

# Open the credentials.json file
with open(args.credential_path, 'r') as file:
    credentials = json.load(file)

# Retrieve authentication information
username = credentials.get('username')
password = credentials.get('password')

# mimic-cxr path
mimic_cxr_path = 'https://physionet.org/files/mimic-cxr/2.1.0/'

# download file path csv
url = mimic_cxr_path + 'cxr-study-list.csv.gz'
headers = {
    'User-Agent': 'Wget/1.21.3 (linux-gnu)',
}
response = requests.get(url, auth=HTTPBasicAuth(username, password), headers=headers) 
file_content = io.BytesIO(response.content)
path_df = pd.read_csv(file_content, compression='gzip', encoding='utf-8')

# URL of the source for download
patient_paths = [ mimic_cxr_path + path + '/'  for path in path_df['path'].apply(lambda x: os.sep.join(x.split(os.sep)[:3])).drop_duplicates().tolist()]

if args.download_num is None:
    urls = patient_paths
else:
    urls = patient_paths[:args.download_num]

for url in urls:
    # Create a temporary directory
    temp_dir = Path('./temp_download')
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Execute the wget command
    wget_command = [
        'wget',
        '-r', '-N', '-c', '-np',
        '--user', username,
        '--password', password,
        '-nH', '--cut-dirs=1', 
        url
    ]

    os.chdir(temp_dir)
    # run the wget command
    subprocess.run(wget_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.chdir('..')
    # S3 client
    s3_client = boto3.client('s3')

    # Upload the downloaded files to s3 bucket
    for file_path in temp_dir.rglob('*'):
        if file_path.is_file():
            s3_key = os.path.join(args.s3_dir, file_path.relative_to(temp_dir))
            with open(file_path, 'rb') as file_data:
                s3_client.put_object(Bucket=args.bucket_name, Key=str(s3_key), Body=file_data)


    # eliminate temp dir
    shutil.rmtree(temp_dir)
    print(f"Uploaded to s3 bucket: {url}")


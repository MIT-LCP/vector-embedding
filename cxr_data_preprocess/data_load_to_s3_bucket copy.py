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

# Configure the S3 bucket
bucket_name = 'resresearcher-lcp-takeshi-group-767397697436'
s3_key_prefix = 'rawdata/'

# Open the credentials.json file
with open('/home/sagemaker-user/vector-embedding/physionet_credential.json', 'r') as file:
    # JSONデータを読み込む
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
upload_num = 3
urls = patient_paths[:upload_num]


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
    # wgetコマンドの実行
    subprocess.run(wget_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.chdir('..')
    # S3クライアントの作成
    s3_client = boto3.client('s3')

    # ダウンロードしたファイルをS3にアップロード
    for file_path in temp_dir.rglob('*'):
        if file_path.is_file():
            s3_key = os.path.join(s3_key_prefix, file_path.relative_to(temp_dir))
            with open(file_path, 'rb') as file_data:
                s3_client.put_object(Bucket=bucket_name, Key=str(s3_key), Body=file_data)


    # 一時ディレクトリの削除
    shutil.rmtree(temp_dir)
    print(f"ファイルをS3にアップロードしました: {url}")


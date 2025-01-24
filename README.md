# Uploading CXR Files to S3 Bucket
1. Set your PhysioNet credentials (username and password) in the physionet_credential.json file.
2. Run the following Python script:
```bash
python cxr_data_preprocess/data_upload.py --credential_path physionet_credential.json --bucket_name 'resresearcher-lcp-takeshi-group-767397697436' --s3_dir 'rawdata/' --download_num 5
Parameters:
--credential_path: Path to your physionet_credential.json file, which contains your PhysioNet username and password.
--bucket_name: The name of your S3 bucket (e.g., 'resresearcher-lcp-takeshi-group-767397697436').
--s3_dir: The directory in your S3 bucket where the files will be uploaded (e.g., 'rawdata/').
--download_num: The number of files to download (e.g., 5).
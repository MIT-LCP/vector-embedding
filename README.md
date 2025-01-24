# for uploading CXR files to s3 bucket
1. set your physionet credeintialing username and password on physionet_credential.json file.
2. run the following python file. \n
```bash
$python cxr_data_preprocess.data_upload.py --credential_path physionet_credential.json --bucket_name 'resresearcher-lcp-takeshi-group-767397697436' --s3_dir 'rawdata/' --download_num 5

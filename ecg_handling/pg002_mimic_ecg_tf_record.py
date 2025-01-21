import numpy as np
import joblib, tqdm, os
import pandas as pd
import tensorflow as tf
import wfdb
from multiprocessing import Pool


def CreateTensorflowReadFile(df, out_file):
    with tf.io.TFRecordWriter(out_file) as writer:
        csv_paths = df['path_wo_ext'].to_list()
        ecg = np.empty((len(csv_paths), 5000, 12, 1), dtype=np.float32)
       
        for i, (item_path) in enumerate(csv_paths):
            signals, _ = wfdb.rdsamp(item_path)        
            ecg[i, :] = signals[:, :, np.newaxis].astype(np.float32)
        
        # Convert data to binary file
        example = tf.train.Example(features=tf.train.Features(feature={
            "ecg": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ecg.tobytes()])),
            }))

        # Write
        writer.write(example.SerializeToString())


# Load pandas DataFrame
mimic_iv_ecg_data_df = joblib.load("/mnt/s/Workfolder/vectorembedding/df/mimic_iv_ecg_data_df")


# # To avoid running out of memory, create TensorFlow data for each record.
def process_row(i):
    # Avoid unnecessary dataframe copy
    select_df = mimic_iv_ecg_data_df.iloc[[i]]
    CreateTensorflowReadFile(select_df, "/mnt/s/Workfolder/vectorembedding/tfrecord/mimic_iv_ecg/" + mimic_iv_ecg_data_df["subject_id"][i] + '_' + mimic_iv_ecg_data_df["ecg_id"][i])

if __name__ == "__main__":
    # Increasing chunk size for better parallelization
    chunk_size = 10
    with Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.imap(process_row, range(len(mimic_iv_ecg_data_df)), chunksize=chunk_size), total=len(mimic_iv_ecg_data_df)))

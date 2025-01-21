# load library
import numpy as np
import pandas as pd
import wfdb
import joblib
import glob
from IPython.display import display
from joblib import Parallel, delayed
import wfdb
from tqdm import tqdm
import h5py
import os
import tensorflow as tf
from multiprocessing import Pool


ptb_xl_df = joblib.load("/mnt/s/Workfolder/vectorembedding/df/ptb_xl_df")


def process_row(i):
    id = ptb_xl_df['ecg_id'][i]
    ecg = np.empty((1, 5000, 12, 1), dtype=np.float32)
    signals, _ = wfdb.rdsamp("/mnt/s/Workfolder/Physionet/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/" + ptb_xl_df['filename_hr'][i])        
    ecg[0, :] = signals[:, :, np.newaxis].astype(np.float32)

    with tf.io.TFRecordWriter('/mnt/s/Workfolder/vectorembedding/tfrecord/ptb_xl/id_' + str(id).zfill(10)) as writer:
        example = tf.train.Example(features=tf.train.Features(feature={
            "ecg": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ecg.tobytes()])),
            }))
        writer.write(example.SerializeToString())

def CreateTensorflowReadFile():
    chunk_size = 10
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(process_row, range(len(ptb_xl_df)), chunksize=chunk_size), total=len(ptb_xl_df)))

CreateTensorflowReadFile()
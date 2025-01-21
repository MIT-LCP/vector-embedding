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


hdf_paths = glob.glob('/mnt/s/Workfolder/CODE15%/ECG_wave/*.hdf5')
out_dir = "/mnt/s/Workfolder/vectorembedding/tfrecord/code_15_ecg"


def process_row(i):
    id = f['exam_id'][i]
    ecg = f['tracings'][i][:, :, np.newaxis]
    with tf.io.TFRecordWriter(out_dir + '/id_' + str(id).zfill(10)) as writer:
        example = tf.train.Example(features=tf.train.Features(feature={
            "ecg": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ecg.tobytes()])),
            }))
        writer.write(example.SerializeToString())

def CreateTensorflowReadFile(hdf):
    chunk_size = 10
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(process_row, range(len(f['exam_id'])), chunksize=chunk_size), total=len(f['exam_id'])))


for hdf in hdf_paths:
    f = h5py.File(hdf)
    CreateTensorflowReadFile(hdf)


code15_df = joblib.load('/mnt/s/Workfolder/vectorembedding/df/code15_df')

code15_paths = pd.DataFrame(glob.glob("/mnt/s/Workfolder/vectorembedding/tfrecord/code_15_ecg/*"),columns=['tfrecord_path'])

code15_path_df = pd.merge(code15_df,code15_paths,how='inner',on=['tfrecord_path'])

joblib.dump(code15_path_df,'/mnt/s/Workfolder/vectorembedding/df/code15_path_df')


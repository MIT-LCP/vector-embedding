import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import wfdb
import joblib
import tqdm
import shutil
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled")
    except RuntimeError as e:
        print(e)


def parse(example):
    features = tf.io.parse_single_example(
        example,
        features={"ecg": tf.io.FixedLenFeature([], tf.string),
                  })
    ecg_ = tf.io.decode_raw(features["ecg"], tf.float32) 
    ecg = tf.reshape(ecg_, tf.stack([4096, 12, 1]))    
    return ecg


# Load pandas DataFrame
code_15_ecg_data_df = joblib.load("/mnt/s/Workfolder/vectorembedding/df/code15_path_df")


tf_dataset = tf.data.TFRecordDataset(code_15_ecg_data_df['tfrecord_path']).map(parse).repeat(1).batch(256).prefetch(tf.data.experimental.AUTOTUNE)



def tfrecord_check(batch):
    # 時間軸 (5000) に沿って平均を計算
    means = tf.reduce_mean(batch, axis=1)  # (None, 12, 1)
    # 平均に 0 が含まれるかを判定 (True/False)
    has_zero = tf.reduce_any(tf.equal(means, 0.0), axis=[1, 2])  # 各データごとに判定
    # データ全体に NaN が含まれるかを判定 (True/False)
    has_nan = tf.reduce_any(tf.math.is_nan(batch), axis=[1, 2, 3])  # 各データごとに判定
    # 微分値を計算 (時間軸 5000 に沿って)
    diff = tf.experimental.numpy.diff(batch, axis=1)  # 微分: (None, 4999, 12, 1)
    # 微分値に 0 が含まれる割合を計算
    diff_zero_ratio = tf.reduce_mean(tf.cast(tf.equal(diff, 0.0), tf.float32), axis=[1, 2, 3])
    # 0 に近い値に対応させたい場合は、`tf.abs(diff) < epsilon` を使用可能
    # True/False を 1/0 に変換
    zero_flag = tf.cast(has_zero, tf.int32)
    nan_flag = tf.cast(has_nan, tf.int32)
    # 結果を結合して返す
    # 各データごとに [has_zero, has_nan, diff_zero_ratio] を出力
    return tf.concat([tf.cast(zero_flag[:, tf.newaxis], tf.float32),
                      tf.cast(nan_flag[:, tf.newaxis], tf.float32),
                      diff_zero_ratio[:, tf.newaxis]], axis=1)

# データセットに適用
tfrecord_check_flags = tf_dataset.map(tfrecord_check)
tfrecord_check_flags = tf.concat(list(tfrecord_check_flags.as_numpy_iterator()), axis=0)
tfrecord_check_flags_df = pd.DataFrame(tfrecord_check_flags,columns=['zero_frag','nan_frag','diff'])
tfrecord_check_flags_df['diff'].hist(bins=100)

code_15_ecg_data_frag_df = pd.concat([code_15_ecg_data_df,tfrecord_check_flags_df],axis=1)
joblib.dump(code_15_ecg_data_frag_df,"/mnt/s/Workfolder/vectorembedding/df/code15_frag_df")





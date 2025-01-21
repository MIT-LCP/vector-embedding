import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import wfdb
import joblib
import tqdm
import shutil
import os

import matplotlib.pyplot as plt



ecg_df = joblib.load("/mnt/s/Workfolder/vectorembedding/df/mimic_iv_ecg_data_frag_df")





ecg_df_sel19 = ecg_df[(ecg_df['diff']>0.91) & (ecg_df['diff']<0.95)].reset_index(drop=True)[:30]

ecg_df_sel3 = ecg_df[(ecg_df['diff']>0.29) & (ecg_df['diff']<0.31)].reset_index(drop=True)[:30]

def ecg_plot(dir, ecg):
    # ECGデータの読み込み
    ecg_dict = ecg.to_dict()

    file_ids = ecg_dict['subject_id'] +"_" + ecg_dict['ecg_id']
    signals, _ = wfdb.rdsamp(ecg_dict['path_wo_ext'])
    
    # 図の作成とサイズ設定（幅10インチ、高さ6インチ）
    plt.figure(figsize=(10, 8))
    
    # 上段のサブプロット
    plt.subplot(2, 1, 1)
    plt.plot(signals[:, 1])
    plt.ylim(-1, 1)
    plt.ylabel('II/mV')  # Y軸ラベル
    
    # 下段のサブプロット
    plt.subplot(2, 1, 2)
    plt.plot(signals[:, 6])
    plt.ylim(-1, 1)
    plt.xlabel('Sample Number')  # X軸ラベル
    plt.ylabel('V1/mV')  # Y軸ラベル
    # グラフの表示
    plt.tight_layout()
    plt.savefig(f"{dir}/ecg_"+file_ids+'.jpg')
    plt.show()


for i in range(30):
    ecg_plot('/mnt/s/OneDrive/Project/vectorembedding/ecg_data_review/ecg_fixed_axis/low_reso',ecg_df_sel19.iloc[i,:])

    ecg_plot('/mnt/s/OneDrive/Project/vectorembedding/ecg_data_review/ecg_fixed_axis/high_reso',ecg_df_sel3.iloc[i,:])




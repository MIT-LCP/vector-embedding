import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


ptb_xl_frag_df = joblib.load("/mnt/s/Workfolder/vectorembedding/df/ptb_xl_frag_df")
code15_frag_df = joblib.load("/mnt/s/Workfolder/vectorembedding/df/code15_frag_df")
mimic_iv_ecg_data_frag_df = joblib.load("/mnt/s/Workfolder/vectorembedding/df/mimic_iv_ecg_data_frag_df")

# 21799 -> 20997
ptb_xl_sel = ptb_xl_frag_df.copy()[(ptb_xl_frag_df["zero_frag"]==0) &
                                   (ptb_xl_frag_df["nan_frag"]==0) &
                                   (ptb_xl_frag_df["diff"]<0.5)]
# 345779 -> 342405
code15_sel = code15_frag_df.copy()[(code15_frag_df["zero_frag"]==0) &
                                   (code15_frag_df["nan_frag"]==0) &
                                   (code15_frag_df["diff"]<0.5)]
# 800035 -> 681827
mimic_sel = mimic_iv_ecg_data_frag_df.copy()[(mimic_iv_ecg_data_frag_df["zero_frag"]==0) &
                                            (mimic_iv_ecg_data_frag_df["nan_frag"]==0) &
                                            (mimic_iv_ecg_data_frag_df["diff"]<0.5)]


mimic_sel.loc[(mimic_sel["subject_id"].str[1:].astype(int)<14000000), 'dataset'] = 'train'
mimic_sel.loc[(mimic_sel["subject_id"].str[1:].astype(int)>=14000000)&
               (mimic_sel["subject_id"].str[1:].astype(int)<15000000), 'dataset'] = 'val'
mimic_sel.loc[mimic_sel["subject_id"].str[1:].astype(int)>=15000000, 'dataset'] = 'test'


code15_sel.loc[(code15_sel["patient_id"]<600000), 'dataset'] = 'train'
code15_sel.loc[(code15_sel["patient_id"]>=600000)&
               (code15_sel["patient_id"]<750000	), 'dataset'] = 'val'
code15_sel.loc[code15_sel["patient_id"]>=750000, 'dataset'] = 'test'

ptb_xl_sel['dataset'] = 'test'


mimic_sel.dataset.value_counts(normalize=True)
# test     0.500099
# train    0.398713
# val      0.101188
code15_sel.dataset.value_counts(normalize=True)
# test     0.512171
# train    0.414208
# val      0.073620
ptb_xl_sel.dataset.value_counts(normalize=True)
# test    1.0


joblib.dump(mimic_sel,"/mnt/s/Workfolder/vectorembedding/df/mimic_sel")
joblib.dump(code15_sel,"/mnt/s/Workfolder/vectorembedding/df/code15_sel")
joblib.dump(ptb_xl_sel,"/mnt/s/Workfolder/vectorembedding/df/ptb_xl_sel")





# mimic_sel mV 


# tf_dataset = tf.data.TFRecordDataset(ptb_xl_sel['tfrecord_path'])

# # tf_dataset = tf.data.TFRecordDataset(code15_sel['tfrecord_path'][100])

# def parse(example, ecg_length = 5000):
#     features = tf.io.parse_single_example(
#         example,
#         features={"ecg": tf.io.FixedLenFeature([], tf.string),
#                   })
#     ecg_ = tf.io.decode_raw(features["ecg"], tf.float32) 
#     ecg = tf.reshape(ecg_, tf.stack([ecg_length, 12, 1]))   
#     return ecg
# dataset = tf_dataset.map(parse).batch(1024)


# # dataset = tf.data.TFRecordDataset(filenames).map(parse).batch(1)

# # 初期化
# global_min = np.inf
# global_max = -np.inf

# # データセットの反復処理
# for batch in dataset:
#     # バッチの最小値と最大値を計算
#     batch_min = tf.reduce_min(batch).numpy()
#     batch_max = tf.reduce_max(batch).numpy()
    
#     # 全体の最小値と最大値を更新
#     if batch_min < global_min:
#         global_min = batch_min
#     if batch_max > global_max:
#         global_max = batch_max

# print(f"ECG データの最小値: {global_min}")
# print(f"ECG データの最大値: {global_max}")
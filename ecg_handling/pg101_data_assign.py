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


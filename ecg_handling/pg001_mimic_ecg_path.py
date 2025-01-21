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


# listing the ECG paths
ecg_path_list = glob.glob('/mnt/s/Workfolder/Physionet/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/*/*/*/*.dat')

print("ECG data count: ", len(ecg_path_list))


#### Check ECG data #####
# remove extention
ecg_data_df = pd.DataFrame(ecg_path_list,columns=['path'])
ecg_data_df['path_wo_ext'] = ecg_data_df['path'].str[:-4]

sig_name_list = []
units_list = []
fields_list = []
min_v_list = []
max_v_list = []
del_ecg_list = []
del_ecg_zero_list = []
del_ecg_nan_list = []
del_ecg_inf_list = []
for path in tqdm(ecg_data_df['path_wo_ext']):
    signals, fields = wfdb.rdsamp(path)
    sig_name_list.append(fields["sig_name"])
    units_list.append(fields["units"])
    # print(np.min(signals))
    # print(np.max(signals))
    min_v_list.append(np.min(signals))
    max_v_list.append(np.max(signals))

    if np.any(np.std(signals,axis=0)==0) or np.any(np.isnan(signals)) or np.any(np.isinf(signals)):
        del_ecg_list.append(path)
       
        if np.any(np.std(signals,axis=0)==0):
            #  print("del zero!! ")
             del_ecg_zero_list.append(path)
        if np.any(np.isnan(signals)):
            #  print("del nan!! ")
             del_ecg_nan_list.append(path)
        if np.any(np.isinf(signals)):
            #  print("del inf!! ")     
             del_ecg_inf_list.append(path)    
             
    del fields["sig_name"]
    del fields["units"]
    fields_list.append(fields)


print(len(del_ecg_list)) # 12358
print(len(del_ecg_zero_list)) # 1951
print(len(del_ecg_nan_list)) # 10554
print(len(del_ecg_inf_list)) # 0

pd.DataFrame(min_v_list).hist()
    
fields_df =  pd.DataFrame(fields_list)
sig_name_df =  pd.DataFrame(sig_name_list)
units_df =  pd.DataFrame(units_list)

display('variation of sampling frequency : ',fields_df.fs.value_counts())
display('variation of signal_length : ',fields_df.sig_len.value_counts())
display('variation of n_signal : ',fields_df.n_sig.value_counts())
display('variation of sig_name : ',sig_name_df.value_counts())
display('variation of units : ',units_df.value_counts())

ecg_data_df = pd.concat([ecg_data_df,fields_df],axis=1)

"""
'variation of sampling frequency : 'fs
500    800035
Name: count, dtype: int64'variation of signal_length : 'sig_len
5000    800035
Name: count, dtype: int64'variation of n_signal : 'n_sig
12    800035
Name: count, dtype: int64'variation of sig_name : '0  1   2    3    4    5    6   7   8   9   10  11
I  II  III  aVR  aVF  aVL  V1  V2  V3  V4  V5  V6    800035
Name: count, dtype: int64'variation of units : '0   1   2   3   4   5   6   7   8   9   10  11
mV  mV  mV  mV  mV  mV  mV  mV  mV  mV  mV  mV    800035
Name: count, dtype: int64
"""
ecg_data_sel_df = ecg_data_df.copy()[~ecg_data_df["path_wo_ext"].isin(del_ecg_list)]

ecg_data_df['del_ecg'] = ecg_data_df["path_wo_ext"].isin(del_ecg_list)
ecg_data_df['del_ecg_zero'] = ecg_data_df["path_wo_ext"].isin(del_ecg_zero_list)
ecg_data_df['del_ecg_nan_ecg'] = ecg_data_df["path_wo_ext"].isin(del_ecg_nan_list)

ecg_data_df["subject_id"] = ecg_data_df.path_wo_ext.apply(lambda x:os.path.basename(os.path.dirname(os.path.dirname(x))))
ecg_data_df["ecg_id"] = ecg_data_df.path_wo_ext.apply(lambda x:os.path.basename(os.path.dirname(x)))


joblib.dump(ecg_data_df,"/mnt/s/Workfolder/vectorembedding/df/mimic_iv_ecg_data_df")




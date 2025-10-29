import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "vector-embedding/echo/module/veecho")
from evaluation_utils import patient_based_cv

# データ読み込み・前処理（既存のコードと同じ）
ve_csv = pd.read_csv("/mnt/s/Workfolder/vector_embedding_echo/vedata/ve_lora_adv_sel.csv")
outcome_csv = pd.read_csv('/mnt/s/Workfolder/vector_embedding_echo/dataset/datasplit/test_sel_ds.csv')
ve_echo_df = pd.merge(ve_csv, outcome_csv, how='inner', on="dicom_path")



# 前処理
ve_echo_df["Sex"] = ve_echo_df["Sex"].replace(
    {"F": 1, "M": 0})
ve_echo_df["Language"] = ve_echo_df["Language"].replace(
    {"Non-engslish": 1, "English": 0})
ve_echo_df["Race"] = ve_echo_df["Race"].replace(
    {"White": 0, "Black": 1, "Hispanic": 1, "Asian": 1, "Other": 1, "Unknown": np.nan})
ve_echo_df["Medicaid"] = ve_echo_df["Medicaid"].replace(
    {"Medicaid": 1, "Non-medicaid": 0})
ve_echo_df["Medicare"] = ve_echo_df["Medicare"].replace(
    {"Medicare": 1, "Non-medicare": 0})
ve_echo_df["Private"] = ve_echo_df["Private"].replace(
    {"Private": 1, "Non-private": 0})

# 特徴量・目的変数設定
feature_cols = [f've{str(i).zfill(3)}' for i in range(1, 513)]

binary_targets = ['ab_Ao', 'ab_LVD', 'ab_LAVI',
       'ab_LVMI', 'rEF', 'RV_func', 'mildAR', 'mildAS', 'mildMR', 'mildTR',
       'modAR', 'modAS', 'modMR', 'modTR', 'PHT', 'ab_E_e', 'IVC',
       'Echo_quality', 'Dilated_cardiompyopthy', 'Hypertrophic_cardiomyopathy',
       'Sex', 'Race', 'Medicaid', 'Private', 'Medicare', 'Language']
binary_targets = ['rEF','mildMR', 'mildTR', 'Race']

# 患者IDベース評価（patient_id列があることを前提）
model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)

results = {}
for target in binary_targets:
    result = patient_based_cv(
    df=ve_echo_df,
    features=feature_cols,
    target_col=target,
    patient_col="subject_id",
    model_class=lambda: LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
    )
    results[target] = result
    print(f'{target} - Patient-based CV AUC: {result["auc_mean"]:.4f} ± {result["auc_std"]:.4f} (n_folds={result["n_folds"]})')

# 結果保存
results_df = pd.DataFrame(results).T
results_df.to_csv('/mnt/s/Workfolder/vector_embedding_echo/results/downstreamtask_results.csv')
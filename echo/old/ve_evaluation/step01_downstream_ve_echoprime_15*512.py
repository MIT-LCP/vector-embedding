import pandas as pd
import os
import numpy as np


ve_a4c_rep = pd.read_csv("/mnt/s/Workfolder/MIMIC-IV-ECHO/ve_files/all_ve_echoprime_7680.csv")

echo_df = pd.read_csv("/mnt/s/OneDrive/Project/EchoPrime/mimic_echo_ve/echo_df.csv")
icd_df = pd.read_csv("/mnt/s/Workfolder/MIMIC-IV-ECHO/csv/mimic_icd_dx.csv")
demog_df = pd.read_csv("/mnt/s/Workfolder/MIMIC-IV-ECHO/csv/mimic_demog.csv")

display(ve_a4c_rep.head())
display(echo_df.head())
display(icd_df.head())


icd_df_ = pd.merge(echo_df,icd_df,how='inner',on='subject_id')
icd_df_filtered = icd_df_[
    (pd.to_datetime(icd_df_["study_datetime"]) >= (pd.to_datetime(icd_df_["admittime"]) - pd.Timedelta(days=1))) &
    (pd.to_datetime(icd_df_["study_datetime"]) <= pd.to_datetime(icd_df_["dischtime"]))
].copy()[['study_id', 'Cardiac_amyloidosis','Sarcoid_myocarditis', 'Myocardial_infarction', 'Pulmonary_embolism','Dilated_cardiompyopthy', 'Hypertrophic_cardiomyopathy']]

ve_a4c_rep["study_id"] = ve_a4c_rep["study_id"].apply(lambda x: os.path.basename(x)).str[1:].astype(int)



ve_echo_df = pd.merge(echo_df,ve_a4c_rep,how='inner',on="study_id")
ve_echo_df = pd.merge(ve_echo_df,demog_df,how='inner',on="study_id")
ve_echo_df = ve_echo_df.merge(icd_df_filtered,how='left',on="study_id")


ve_echo_df["Sex"] = ve_echo_df["Sex"].replace({"F":1, "M":0})
ve_echo_df["Language"] = ve_echo_df["Language"].replace({"Non-engslish":1, "English":0})
ve_echo_df["Race"] = ve_echo_df["Race"].replace({"White":0, "Black":1,"Hispanic":1,"Asian":1, "Other":1, "Unknown":np.nan})

ve_echo_df["Medicaid"] = ve_echo_df["Medicaid"].replace({"Medicaid":1, "Non-medicaid":0})
ve_echo_df["Medicare"] = ve_echo_df["Medicare"].replace({"Medicare":1, "Non-medicare":0})
ve_echo_df["Private"] = ve_echo_df["Private"].replace({"Private":1, "Non-private":0})

ve_echo_df.loc[ve_echo_df['LVEF']>50, 'rEF'] = 0
ve_echo_df.loc[ve_echo_df['LVEF']<=50, 'rEF'] = 1
ve_echo_df.head()





from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd

# 説明変数
feature_cols = [f've{str(i).zfill(4)}' for i in range(1, 7680)]
X = ve_echo_df[feature_cols]


# 目的変数
continuous_targets = ['LA_4C', 'RA_4C', 'LVDD', 'LVDS', 'LVPW', 'IVS', 'AV_Vmax', 'TR_PG', 'LVEF']
binary_targets = ['AR', 'MR', 'TR', 'rEF','RV_func',
                  'Cardiac_amyloidosis','Myocardial_infarction', 'Pulmonary_embolism','Dilated_cardiompyopthy', 'Hypertrophic_cardiomyopathy',
                   "Sex", "Language","Race",
                   'Medicaid',"Medicare",'Private']

# 出力格納用
results = {}

# 回帰タスク（L2回帰）
for target in continuous_targets:
    y = ve_echo_df[target]
    Xy = pd.concat([X, y], axis=1).dropna()
    X_clean = Xy[feature_cols].values
    y_clean = Xy[target].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []
    coef_list = []

    for train_index, test_index in kf.split(X_clean):
        X_train, X_test = X_clean[train_index], X_clean[test_index]
        y_train, y_test = y_clean[train_index], y_clean[test_index]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)
        coef_list.append(model.coef_)

    results[target] = {
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'coef_mean': np.mean(coef_list, axis=0)
    }

    print(f'{target} - Ridge RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}')

# 分類タスク（ロジスティック回帰：L2正則化）
for target in binary_targets:
    y = ve_echo_df[target]
    Xy = pd.concat([X, y], axis=1).dropna()
    X_clean = Xy[feature_cols].values
    y_clean = Xy[target].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_list = []
    coef_list = []

    for train_index, test_index in skf.split(X_clean, y_clean):
        X_train, X_test = X_clean[train_index], X_clean[test_index]
        y_train, y_test = y_clean[train_index], y_clean[test_index]

        model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_list.append(auc)
        coef_list.append(model.coef_[0])

    results[target] = {
        'auc_mean': np.mean(auc_list),
        'auc_std': np.std(auc_list),
        'coef_mean': np.mean(coef_list, axis=0)
    }

    print(f'{target} - Logistic Regression AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}')




"""
LA_4C - Ridge RMSE: 0.6268 ± 0.0101
RA_4C - Ridge RMSE: 0.5978 ± 0.0093
LVDD - Ridge RMSE: 0.5370 ± 0.0127
LVDS - Ridge RMSE: 0.4663 ± 0.0053
LVPW - Ridge RMSE: 0.2403 ± 0.1164
IVS - Ridge RMSE: 0.1897 ± 0.0146
AV_Vmax - Ridge RMSE: 0.3975 ± 0.0129
TR_PG - Ridge RMSE: 9.9171 ± 0.5103
LVEF - Ridge RMSE: 9.4102 ± 0.4010
AR - Logistic Regression AUC: 0.8524 ± 0.0156
MR - Logistic Regression AUC: 0.7781 ± 0.0137
TR - Logistic Regression AUC: 0.7592 ± 0.0122
rEF - Logistic Regression AUC: 0.9382 ± 0.0158
RV_func - Logistic Regression AUC: 0.8910 ± 0.0152
Cardiac_amyloidosis - Logistic Regression AUC: 0.8835 ± 0.0476
Myocardial_infarction - Logistic Regression AUC: 0.7108 ± 0.0260
Pulmonary_embolism - Logistic Regression AUC: 0.7133 ± 0.0409
Dilated_cardiompyopthy - Logistic Regression AUC: 0.8871 ± 0.0639
Hypertrophic_cardiomyopathy - Logistic Regression AUC: 0.7140 ± 0.0795
Sex - Logistic Regression AUC: 0.9368 ± 0.0064
Language - Logistic Regression AUC: 0.6822 ± 0.0185
Race - Logistic Regression AUC: 0.7167 ± 0.0220
Medicaid - Logistic Regression AUC: 0.6839 ± 0.0270
Medicare - Logistic Regression AUC: 0.7702 ± 0.0132
Private - Logistic Regression AUC: 0.7496 ± 0.0201
"""
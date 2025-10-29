import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def patient_based_cv(df, features, target_col, patient_col, model_class, n_splits=5, random_state=42):
    """
    患者IDベースでバイナリ分類のクロスバリデーション評価（完全版）
    """
    # 1. データ準備
    required_cols = [patient_col] + features + [target_col]
    clean_df = df[required_cols].dropna().copy()
    
    if len(clean_df) == 0:
        return {'auc_mean': np.nan, 'auc_std': np.nan, 'n_folds': 0}
    
    # 2. 患者レベルでのラベル決定
    patient_target = clean_df.groupby(patient_col)[target_col].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    )
    
    if patient_target.nunique() < 2:
        return {'auc_mean': np.nan, 'auc_std': np.nan, 'n_folds': 0}
    
    # 3. 患者レベルでStratified分割
    unique_patients = patient_target.index.values
    patient_labels = patient_target.values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores = []
    
    # 4. 各foldで評価
    for train_patients_idx, val_patients_idx in skf.split(unique_patients, patient_labels):
        # 患者IDから実際のデータ行を取得
        train_patients = unique_patients[train_patients_idx]
        val_patients = unique_patients[val_patients_idx]
        
        train_mask = clean_df[patient_col].isin(train_patients)
        val_mask = clean_df[patient_col].isin(val_patients)
        
        X_train = clean_df.loc[train_mask, features].values
        X_val = clean_df.loc[val_mask, features].values
        y_train = clean_df.loc[train_mask, target_col].values
        y_val = clean_df.loc[val_mask, target_col].values
        
        # 検証セットが単一クラスの場合スキップ
        if len(np.unique(y_val)) < 2:
            continue
        
        # モデル訓練・予測
        model = model_class()
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_pred_proba)
        auc_scores.append(auc)
    
    return {
        'auc_mean': np.mean(auc_scores) if auc_scores else np.nan,
        'auc_std': np.std(auc_scores) if auc_scores else np.nan,
        'n_folds': len(auc_scores)
    }
    
def evaluate_binary_cv(df, features, target, patient_col, model_class, n_splits=5, random_state=42):
    """
    患者IDベースでバイナリ分類のクロスバリデーション評価
    
    Args:
        df: データフレーム
        features: 特徴量列のリスト
        target: 目的変数列名
        patient_col: 患者ID列名
        model_class: sklearn分類器クラス（LogisticRegressionなど）
        
    Returns:
        dict: {'auc_mean', 'auc_std', 'n_folds'}
    """
    from sklearn.metrics import roc_auc_score
    
    # 欠損値除去
    clean_df = df[[patient_col] + features + [target]].dropna()
    
    if len(clean_df) == 0 or clean_df[target].nunique() < 2:
        return {'auc_mean': np.nan, 'auc_std': np.nan, 'n_folds': 0}
    
    X = clean_df[features].values
    y = clean_df[target].values
    auc_scores = []
    
    for train_idx, val_idx in patient_based_split(clean_df, patient_col, target, n_splits, random_state):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 検証セットが単一クラスの場合スキップ
        if len(np.unique(y_val)) < 2:
            continue
            
        model = model_class()
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, y_pred))
    
    return {
        'auc_mean': np.mean(auc_scores) if auc_scores else np.nan,
        'auc_std': np.std(auc_scores) if auc_scores else np.nan,
        'n_folds': len(auc_scores)
    }
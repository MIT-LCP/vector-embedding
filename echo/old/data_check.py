import pandas as pd
import numpy as np

def diagnose_dataset(csv_path):
    """データセットの問題を診断"""
    print(f"\n=== データセット診断: {csv_path} ===")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ CSVファイル読み込み成功")
        print(f"  - 総サンプル数: {len(df)}")
        print(f"  - 列名: {list(df.columns)}")
        
        # 必須カラムの確認
        required_cols = ['subject_id', 'dicom_path', 'Sex', 'Race']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 必須カラムが不足: {missing_cols}")
            return False
        else:
            print(f"✓ 必須カラムが揃っています")
        
        # subject_idの分析
        subject_groups = df.groupby('subject_id')
        subject_counts = subject_groups.size()
        print(f"\n--- subject_id分析 ---")
        print(f"  - ユニークsubject_id数: {len(subject_groups)}")
        print(f"  - subject_id別サンプル数統計:")
        print(f"    平均: {subject_counts.mean():.2f}")
        print(f"    最小: {subject_counts.min()}")
        print(f"    最大: {subject_counts.max()}")
        print(f"    1サンプルのみのsubject数: {(subject_counts == 1).sum()}")
        print(f"    2サンプル以上のsubject数: {(subject_counts >= 2).sum()}")
        
        # トリプレット生成可能性の確認
        viable_subjects = subject_counts[subject_counts >= 2]
        if len(viable_subjects) < 2:
            print(f"❌ トリプレット生成不可能: 2サンプル以上のsubjectが{len(viable_subjects)}個しかありません")
            print("   - トリプレット生成には最低2つのsubject（各2サンプル以上）が必要です")
            return False
        else:
            print(f"✓ トリプレット生成可能: {len(viable_subjects)}個のsubjectが利用可能")
        
        # Sex/Race分布の確認
        print(f"\n--- 保護属性分析 ---")
        print(f"Sex分布:")
        print(df['Sex'].value_counts())
        print(f"\nRace分布:")
        print(df['Race'].value_counts())
        
        # dicom_pathの確認
        print(f"\n--- ファイルパス分析 ---")
        print(f"  - ユニークファイル数: {df['dicom_path'].nunique()}")
        print(f"  - 重複ファイル数: {len(df) - df['dicom_path'].nunique()}")
        
        # サンプルデータ表示
        print(f"\n--- サンプルデータ (最初の5行) ---")
        print(df.head())
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_triplet_generation(csv_path, buffer_size=100):
    """実際にトリプレット生成をテスト"""
    print(f"\n=== トリプレット生成テスト ===")
    
    try:
        df = pd.read_csv(csv_path)
        subject_groups = df.groupby('subject_id')
        subject_ids = list(subject_groups.groups.keys())
        
        triplets = []
        triplets_per_subject = min(50, buffer_size // len(subject_ids)) if len(subject_ids) > 0 else 50
        
        print(f"  - subject数: {len(subject_ids)}")
        print(f"  - subject当たりのトリプレット目標数: {triplets_per_subject}")
        
        for subject_id in subject_ids:
            subject_samples = subject_groups.get_group(subject_id)
            
            # 同一subject内での組み合わせをチェック
            if len(subject_samples) < 2:
                print(f"  - Subject {subject_id}: サンプル数不足 ({len(subject_samples)})")
                continue
                
            # 他のsubjectがあるかチェック
            other_subjects = [sid for sid in subject_ids if sid != subject_id]
            if len(other_subjects) == 0:
                print(f"  - Subject {subject_id}: 他のsubjectが存在しない")
                continue
            
            current_triplets = 0
            for idx, anchor_row in subject_samples.iterrows():
                if current_triplets >= triplets_per_subject:
                    break
                
                # Positive候補
                pos_candidates = subject_samples[subject_samples.index != idx]
                if len(pos_candidates) == 0:
                    continue
                
                # Negative候補
                neg_subject = np.random.choice(other_subjects)
                neg_samples = subject_groups.get_group(neg_subject)
                
                pos_row = pos_candidates.sample(1).iloc[0]
                neg_row = neg_samples.sample(1).iloc[0]
                
                # 重複チェック
                if (anchor_row['dicom_path'] != pos_row['dicom_path'] and 
                    anchor_row['dicom_path'] != neg_row['dicom_path'] and 
                    pos_row['dicom_path'] != neg_row['dicom_path']):
                    
                    triplets.append({
                        'anchor_idx': idx,
                        'positive_idx': pos_row.name,
                        'negative_idx': neg_row.name,
                        'anchor_path': anchor_row['dicom_path']
                    })
                    current_triplets += 1
                
                if len(triplets) >= buffer_size:
                    break
            
            print(f"  - Subject {subject_id}: {current_triplets}個のトリプレット生成")
            
            if len(triplets) >= buffer_size:
                break
        
        print(f"\n✓ 総トリプレット数: {len(triplets)}")
        return len(triplets)
        
    except Exception as e:
        print(f"❌ トリプレット生成エラー: {e}")
        return 0

# 使用例
if __name__ == "__main__":
    dataset_path = "/mnt/s/Workfolder/vector_embedding_echo/dataset/datasplit/"
    
    # 両方のデータセットを診断
    train_valid = diagnose_dataset(dataset_path + "train_sel_ds.csv")
    val_valid = diagnose_dataset(dataset_path + "val_sel_ds.csv")
    
    if train_valid:
        train_triplets = test_triplet_generation(dataset_path + "train_sel_ds.csv")
    
    if val_valid:
        val_triplets = test_triplet_generation(dataset_path + "val_sel_ds.csv")
    
    print(f"\n=== 総合診断結果 ===")
    print(f"訓練データセット: {'有効' if train_valid else '無効'}")
    print(f"検証データセット: {'有効' if val_valid else '無効'}")
    
    if train_valid and val_valid:
        print("✓ 両方のデータセットが使用可能です")
    else:
        print("❌ データセットに問題があります。修正が必要です")
# step02_ve_extraction_quickfix.py
# 最小限の修正版

import torch
import torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# ローカルモジュール
sys.path.insert(0, "vector-embedding/echo/module/veecho")
from model_utils import EchoEmbeddingModel, TrainingConfig
import data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(finetuned_model=True, model_file="best_echo_model.pth"):
    # Step01と同じ設定
    config = TrainingConfig(
        use_adversarial=True, 
        adversarial_attributes=['Sex', 'Race'],
        lambda_adv=1.0, 
        dynamic_lambda=True, 
        use_lora=True, 
        lora_r=8
    )
    
    # ベースモデル読み込み
    checkpoint = torch.load("/mnt/s/Workfolder/vector_embedding_echo/model/echoprime/echo_prime_encoder.pt", 
                           map_location=device)
    echo_encoder = torchvision.models.video.mvit_v2_s()
    echo_encoder.head[-1] = torch.nn.Linear(echo_encoder.head[-1].in_features, 512)
    echo_encoder.load_state_dict(checkpoint)
    
    # 訓練済みモデル作成
    model = EchoEmbeddingModel(echo_encoder, config)
    
    # 訓練済み重みの読み込み判定
    if finetuned_model:
        try:
            model_path = '/mnt/s/Workfolder/vector_embedding_echo/model/domain_adapted_model/'+ model_file
            trained_checkpoint = torch.load(model_path, map_location=device)
            
            if isinstance(trained_checkpoint, dict) and 'model_state_dict' in trained_checkpoint:
                model.load_state_dict(trained_checkpoint['model_state_dict'], strict=False)
                print("✅ Step01 trained model loaded successfully")
            else:
                model.load_state_dict(trained_checkpoint, strict=False)
                print("✅ Trained model loaded successfully")
                
        except Exception as e:
            print(f"⚠️ Warning: Failed to load trained weights, using base model: {e}")
            print("📋 Using base model only")
    else:
        print("📋 Using base model only (as requested)")
    
    model.eval().to(device)
    torch.set_grad_enabled(False)  # 全体で勾配を無効化
    
    return model

def main(finetuned_model, model_file, test_file, out_file, batch_size=4):  # ← batch_size削減
    print("Loading trained model...")
    model = load_trained_model(finetuned_model=finetuned_model, model_file=model_file)
    
    # 対象データ読み込み
    test_df = pd.read_csv("/mnt/s/Workfolder/vector_embedding_echo/dataset/datasplit/" + test_file)
    dicom_paths = test_df['dicom_path'].tolist()
    
    print(f"Processing {len(dicom_paths)} DICOM files with batch_size={batch_size}")
    
    # 結果保存用
    output_file = "/mnt/s/Workfolder/vector_embedding_echo/vedata/" + out_file + ".csv"
    
    # ヘッダー書き込み
    with open(output_file, 'w') as f:
        ve_columns = [f"ve{str(i+1).zfill(3)}" for i in range(512)]
        header = ["dicom_path"] + ve_columns
        f.write(",".join(header) + "\n")
    
    # バッチ処理でembedding抽出（最適化版）
    successful_count = 0
    error_count = 0
    
    for i in tqdm(range(0, len(dicom_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = dicom_paths[i:i+batch_size]
        batch_videos = []
        valid_paths = []
        
        # バッチデータ準備
        for path in batch_paths:
            try:
                video = data_utils.process_dicom(path)
                if video is not None:
                    batch_videos.append(video)
                    valid_paths.append(path)
                else:
                    error_count += 1
            except Exception as e:
                print(f"Error processing {path}: {e}")
                error_count += 1
                continue
        
        if not batch_videos:
            continue
            
        # バッチ推論
        try:
            batch_tensor = torch.cat(batch_videos, dim=0).to(device)
            embeddings = model.base_encoder(batch_tensor).cpu().numpy()
            
            # 結果をファイルに追記
            with open(output_file, 'a') as f:
                for path, embedding in zip(valid_paths, embeddings):
                    row_data = [path] + embedding.tolist()
                    f.write(",".join(map(str, row_data)) + "\n")
            
            successful_count += len(valid_paths)
                    
        except Exception as e:
            print(f"Error in batch processing: {e}")
            error_count += len(valid_paths)
            continue
    
    # 結果サマリー
    total_files = successful_count + error_count
    success_rate = (successful_count / total_files * 100) if total_files > 0 else 0
    
    print(f"\n📊 Extraction Summary:")
    print(f"✅ Successful: {successful_count}/{total_files} ({success_rate:.1f}%)")
    print(f"❌ Errors: {error_count}")
    print(f"💾 Output: {output_file}")

if __name__ == "__main__":
    # 小規模テスト（高速設定）
    print("🔬 Quick Fix Test (Small batches)")
    main(finetuned_model=True, model_file="echomodel_lora_quickfix.pth", 
         test_file="test_sel_ds.csv", out_file="ve_lora_quickfix", batch_size=2)
    
    main(finetuned_model=True, model_file="echomodel_lora_opt.pth", 
         test_file="test_sel_ds.csv", out_file="ve_lora_opt", batch_size=4)
    
    main(finetuned_model=False, model_file="echo_prime_encoder.pt", 
         test_file="test_sel_ds.csv", out_file="ve_echoprime_quickfix", batch_size=6)
    
    # 本番実行（コメントアウト解除して使用）
    # print("\n🏭 Production Extraction")
    # main(finetuned_model=True, model_file="echomodel_lora_production.pth", 
    #      test_file="test_ds.csv", out_file="ve_lora_production", batch_size=8)
    
    # main(finetuned_model=True, model_file="echomodel_lora_adv_production.pth", 
    #      test_file="test_ds.csv", out_file="ve_lora_adv_production", batch_size=8)
    
    # main(finetuned_model=False, model_file="echo_prime_encoder.pt", 
    #      test_file="test_ds.csv", out_file="ve_echoprime_production", batch_size=8)
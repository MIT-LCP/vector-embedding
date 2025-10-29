# 大幅簡略化されたstep02_embedding_extraction.py

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


def load_trained_model(finetuned_model=True, model_file = "best_echo_model.pth"):
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


def main(finetuned_model, model_file, test_file, out_file):
    print("Loading trained model...")
    model = load_trained_model(finetuned_model=finetuned_model, model_file = model_file)
    
    # 対象データ読み込み
    test_df = pd.read_csv("/mnt/s/Workfolder/vector_embedding_echo/dataset/datasplit/" + test_file)
    dicom_paths = test_df['dicom_path'].tolist()
    
    print(f"Processing {len(dicom_paths)} DICOM files...")
    
    # 結果保存用
    output_file = "/mnt/s/Workfolder/vector_embedding_echo/vedata/" + out_file + ".csv"
    
    # ヘッダー書き込み
    with open(output_file, 'w') as f:
        ve_columns = [f"ve{str(i+1).zfill(3)}" for i in range(512)]
        header = ["dicom_path"] + ve_columns
        f.write(",".join(header) + "\n")
    
    # バッチ処理でembedding抽出
    batch_size = 8
    for i in tqdm(range(0, len(dicom_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = dicom_paths[i:i+batch_size]
        batch_videos = []
        valid_paths = []
        
        # バッチデータ準備
        for path in batch_paths:
            try:
                video = data_utils.process_dicom(path)
                batch_videos.append(video)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
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
                    
        except Exception as e:
            print(f"Error in batch processing: {e}")
            continue
    
    print(f"✅ Embeddings saved to: {output_file}")

if __name__ == "__main__":
    main(finetuned_model=True, model_file = "echomodel_lora_sub.pth", 
         test_file = "test_sel_ds.csv", out_file = "ve_lora_sel" )
    main(finetuned_model=True, model_file = "echomodel_lora_adv_sub.pth", 
         test_file = "test_sel_ds.csv", out_file = "ve_lora_adv_sel" )
    main(finetuned_model=False, model_file = "echo_prime_encoder.pt", 
         test_file = "test_sel_ds.csv", out_file = "ve_echoprime_sel" )
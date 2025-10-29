# step01_ssl_lora_grl_minimal.py
# 既存コードの最小限修正（data_utils.pyの一行だけ変更）

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import pandas as pd
import sys

# ローカルモジュール
sys.path.insert(0, "vector-embedding/echo/module/veecho")
from model_utils import EchoEmbeddingModel, TrainingConfig
import data_utils, train_utils

def main(train_set_csv,
         val_set_csv,
         model_name,
         use_adversarial=True,
         epochs=10):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 設定
    config = TrainingConfig(
        use_adversarial=use_adversarial,
        adversarial_attributes=['Sex', 'Race'],
        lambda_adv=1.0,
        dynamic_lambda=True,
        use_lora=True,
        lora_r=8
    )
    
    print(f"Configuration:")
    print(f"  - Adversarial Learning: {config.use_adversarial}")
    if config.use_adversarial:
        print(f"  - Adversarial Attributes: {config.adversarial_attributes}")
        print(f"  - Lambda: {config.lambda_adv}")
        print(f"  - Dynamic Lambda: {config.dynamic_lambda}")
    print(f"  - LoRA: {config.use_lora}")
    
    # ベースモデル読み込み
    checkpoint = torch.load("/mnt/s/Workfolder/vector_embedding_echo/model/echoprime/echo_prime_encoder.pt", map_location=device)
    echo_encoder = torchvision.models.video.mvit_v2_s()
    echo_encoder.head[-1] = nn.Linear(echo_encoder.head[-1].in_features, 512)
    echo_encoder.load_state_dict(checkpoint)
    
    # モデル作成
    model = EchoEmbeddingModel(echo_encoder, config)
    model.to(device)
    
    # Adversarial headsもデバイスに移動
    if config.use_adversarial:
        for head in model.adversarial_heads.values():
            head.to(device)
    
    # 損失関数
    criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion_adv = nn.CrossEntropyLoss()
    
    # 最適化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # 🚀 最適化されたデータローダー作成
    print("Creating optimized dataloaders...")
    
    train_loader = data_utils.create_dataloader(
        csv_path=train_set_csv,
        config=config,
        is_validation=False,
        batch_size=6,           # 8→6に削減
        buffer_size=20,         # 100→20に削減
        seed=42
    )
    
    val_loader = data_utils.create_dataloader(
        csv_path=val_set_csv,
        config=config,
        is_validation=True,
        batch_size=4,           # 8→4に削減
        seed=42
    )
    
    print("✅ Dataloaders ready!")
    
    # トレーナー作成
    trainer = train_utils.create_trainer(model, log_dir="/mnt/s/Workfolder/vector_embedding_echo/logs/"+ model_name)
    
    # 🚀 トレーニング開始！   
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion_triplet=criterion_triplet,
        criterion_adv=criterion_adv,
        device=device,
        config=config,
        epochs=epochs,
        scheduler=scheduler,
        save_best_path='/mnt/s/Workfolder/vector_embedding_echo/model/domain_adapted_model/' + model_name + '.pth'
    )
    
    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    dataset_path = "/mnt/s/Workfolder/vector_embedding_echo/dataset/datasplit/"

    print("🔬 Minimal Modification Test")
    print("Only changing batch_size and buffer_size parameters")
    
    # 小規模テスト
    main(dataset_path + "train_sel_ds.csv", dataset_path + "val_sel_ds.csv",
         "echomodel_lora_minimal", use_adversarial=False, epochs=2)
    
    main(dataset_path + "train_sel_ds.csv", dataset_path + "val_sel_ds.csv",
         "echomodel_lora_adv_minimal", use_adversarial=True, epochs=2)
    
    # 本番実行（コメントアウト解除して使用）
    # main(dataset_path + "train_ds.csv", dataset_path + "val_ds.csv",
    #      "echomodel_lora_production", use_adversarial=False, epochs=10)
    
    # main(dataset_path + "train_ds.csv", dataset_path + "val_ds.csv",
    #      "echomodel_lora_adv_production", use_adversarial=True, epochs=10)
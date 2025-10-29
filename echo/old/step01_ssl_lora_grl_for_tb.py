# step01_fixed_large_scale.py - マルチプロセス問題修正版

import torch
import torch.nn as nn
import torchvision
import sys
import os
import psutil
import time
import gc
from pathlib import Path

# ローカルモジュール
sys.path.insert(0, "vector-embedding/echo/module/veecho")
from model_utils import EchoEmbeddingModel, TrainingConfig
from large_scale_data_utils import create_fixed_dataloader

def get_optimal_settings():
    """システムに応じた最適設定"""
    # メモリ情報
    memory = psutil.virtual_memory()
    available_gb = memory.available / 1e9
    
    # GPU情報
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb >= 24:
            batch_size = 6
            max_samples = 3000
        elif gpu_memory_gb >= 16:
            batch_size = 4
            max_samples = 2000
        else:
            batch_size = 2
            max_samples = 1000
    else:
        batch_size = 2
        max_samples = 500
    
    # キャッシュサイズ
    memory_cache_gb = min(available_gb * 0.25, 12)
    disk_cache_gb = min(100, available_gb * 2)  # 最大100GB
    
    return {
        'batch_size': batch_size,
        'max_samples_per_epoch': max_samples,
        'memory_cache_gb': max(memory_cache_gb, 2),
        'disk_cache_gb': max(disk_cache_gb, 20),
        'num_workers': 0  # マルチプロセス問題回避
    }

def main_fixed(train_set_csv, val_set_csv, model_name,
               use_adversarial=True, epochs=10, cache_dir=None):
    
    print("🚀 修正版大規模トレーニング開始")
    print("=" * 60)
    
    # システム情報
    memory = psutil.virtual_memory()
    print(f"💻 システム情報:")
    print(f"   - CPU: {psutil.cpu_count()}コア")
    print(f"   - RAM: {memory.total/1e9:.1f}GB (利用可能: {memory.available/1e9:.1f}GB)")
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"   - GPU: {gpu_props.name} ({gpu_props.total_memory/1e9:.1f}GB)")
    
    # 最適設定
    settings = get_optimal_settings()
    print(f"⚙️  最適化設定:")
    for key, value in settings.items():
        print(f"   - {key}: {value}")
    
    # キャッシュディレクトリ
    if cache_dir is None:
        cache_dir = Path(f"./cache_fixed/{model_name}")
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 キャッシュディレクトリ: {cache_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 計算デバイス: {device}")
    
    # PyTorch最適化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    # モデル設定
    config = TrainingConfig(
        use_adversarial=use_adversarial,
        adversarial_attributes=['Sex', 'Race'] if use_adversarial else [],
        lambda_adv=1.0,
        dynamic_lambda=True,
        use_lora=True,
        lora_r=8
    )
    
    print(f"🔧 モデル設定:")
    print(f"   - Adversarial Learning: {config.use_adversarial}")
    print(f"   - LoRA rank: {config.lora_r}")
    
    # ベースモデル読み込み
    print("🔄 ベースモデル読み込み...")
    try:
        checkpoint_path = "/mnt/s/Workfolder/vector_embedding_echo/model/echoprime/echo_prime_encoder.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        echo_encoder = torchvision.models.video.mvit_v2_s()
        echo_encoder.head[-1] = nn.Linear(echo_encoder.head[-1].in_features, 512)
        echo_encoder.load_state_dict(checkpoint)
        
        print("✅ ベースモデル読み込み完了")
    except Exception as e:
        print(f"❌ ベースモデル読み込みエラー: {e}")
        raise
    
    # モデル作成
    model = EchoEmbeddingModel(echo_encoder, config)
    model.to(device)
    
    if config.use_adversarial:
        for head in model.adversarial_heads.values():
            head.to(device)
    
    print("✅ モデル作成完了")
    
    # データローダー作成
    print("🔄 データローダー作成...")
    
    train_loader, train_dataset, train_dicom_manager = create_fixed_dataloader(
        csv_path=train_set_csv,
        config=config,
        cache_dir=cache_dir / "train",
        max_memory_cache_gb=settings['memory_cache_gb'] * 0.7,
        max_disk_cache_gb=settings['disk_cache_gb'] * 0.7,
        max_samples_per_epoch=settings['max_samples_per_epoch'],
        batch_size=settings['batch_size'],
        num_workers=settings['num_workers'],
        is_validation=False
    )
    
    val_loader, val_dataset, val_dicom_manager = create_fixed_dataloader(
        csv_path=val_set_csv,
        config=config,
        cache_dir=cache_dir / "val",
        max_memory_cache_gb=settings['memory_cache_gb'] * 0.3,
        max_disk_cache_gb=settings['disk_cache_gb'] * 0.3,
        max_samples_per_epoch=settings['max_samples_per_epoch'] // 2,
        batch_size=settings['batch_size'],
        num_workers=settings['num_workers'],
        is_validation=True
    )
    
    print("✅ データローダー作成完了")
    
    # 最適化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        weight_decay=1e-5,
        eps=1e-8
    )
    
    # スケジューラ
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs * len(train_loader),
        eta_min=1e-6
    )
    
    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # 損失関数
    criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion_adv = nn.CrossEntropyLoss()
    
    # 保存パス
    save_dir = Path("/mnt/s/Workfolder/vector_embedding_echo/model/fixed_large_scale")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{model_name}.pth"
    
    print(f"💾 モデル保存先: {save_path}")
    
    # トレーニング開始
    print("\n🚀 修正版トレーニング開始!")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n📅 エポック {epoch+1}/{epochs}")
        
        # エポック設定
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        
        # Dynamic lambda
        if config.use_adversarial and config.dynamic_lambda:
            p = float(epoch) / epochs
            lambda_dynamic = 2. / (1. + torch.exp(torch.tensor(-10 * p))) - 1
            lambda_final = config.lambda_adv * lambda_dynamic.item()
            
            for head in model.adversarial_heads.values():
                head.set_lambda(lambda_final)
            
            print(f"   Adversarial lambda: {lambda_final:.4f}")
        
        # トレーニング
        model.train()
        train_loss = 0.0
        train_triplet_loss = 0.0
        train_adv_loss = 0.0
        train_batches = 0
        
        try:
            for batch_idx, batch in enumerate(train_loader):
                batch_start_time = time.time()
                
                # データをGPUに転送
                anchor = batch['anchor'].to(device, non_blocking=True)
                positive = batch['positive'].to(device, non_blocking=True)
                negative = batch['negative'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Forward + Backward
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        anchor_features = model(anchor)
                        pos_features = model(positive)
                        neg_features = model(negative)
                        
                        # Triplet loss
                        triplet_loss = criterion_triplet(anchor_features, pos_features, neg_features)
                        total_batch_loss = triplet_loss
                        current_adv_loss = 0.0
                        
                        # Adversarial losses
                        if config.use_adversarial:
                            for attr in config.adversarial_attributes:
                                if attr in batch:
                                    attr_labels = batch[attr].to(device, non_blocking=True)
                                    attr_pred = model.adversarial_heads[attr](anchor_features)
                                    attr_loss = criterion_adv(attr_pred, attr_labels)
                                    total_batch_loss += attr_loss
                                    current_adv_loss += attr_loss.item()
                    
                    # Backward
                    scaler.scale(total_batch_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision
                    anchor_features = model(anchor)
                    pos_features = model(positive)
                    neg_features = model(negative)
                    
                    triplet_loss = criterion_triplet(anchor_features, pos_features, neg_features)
                    total_batch_loss = triplet_loss
                    current_adv_loss = 0.0
                    
                    if config.use_adversarial:
                        for attr in config.adversarial_attributes:
                            if attr in batch:
                                attr_labels = batch[attr].to(device, non_blocking=True)
                                attr_pred = model.adversarial_heads[attr](anchor_features)
                                attr_loss = criterion_adv(attr_pred, attr_labels)
                                total_batch_loss += attr_loss
                                current_adv_loss += attr_loss.item()
                    
                    total_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                
                # 統計更新
                train_loss += total_batch_loss.item()
                train_triplet_loss += triplet_loss.item()
                train_adv_loss += current_adv_loss
                train_batches += 1
                
                batch_time = time.time() - batch_start_time
                
                # プログレス表示
                if batch_idx % 20 == 0:
                    current_loss = train_loss / train_batches
                    current_triplet = train_triplet_loss / train_batches
                    current_adv = train_adv_loss / train_batches if config.use_adversarial else 0
                    lr = optimizer.param_groups[0]['lr']
                    
                    # メモリ統計
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1e9
                    else:
                        gpu_memory = 0
                    
                    print(f"   Batch {batch_idx:3d}: Loss {current_loss:.4f} "
                          f"(T:{current_triplet:.4f}, A:{current_adv:.4f}), "
                          f"LR {lr:.2e}, Time {batch_time:.2f}s, GPU {gpu_memory:.1f}GB")
                
                # 定期的なメモリクリア
                if batch_idx % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        
        except Exception as e:
            print(f"⚠️ トレーニングエラー: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
        
        # 検証
        print("   🔍 検証開始...")
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            try:
                for batch_idx, batch in enumerate(val_loader):
                    if val_batches >= 50:  # 検証は制限
                        break
                    
                    anchor = batch['anchor'].to(device, non_blocking=True)
                    positive = batch['positive'].to(device, non_blocking=True)
                    negative = batch['negative'].to(device, non_blocking=True)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            anchor_features = model(anchor)
                            pos_features = model(positive)
                            neg_features = model(negative)
                            triplet_loss = criterion_triplet(anchor_features, pos_features, neg_features)
                    else:
                        anchor_features = model(anchor)
                        pos_features = model(positive)
                        neg_features = model(negative)
                        triplet_loss = criterion_triplet(anchor_features, pos_features, neg_features)
                    
                    val_loss += triplet_loss.item()
                    val_batches += 1
            
            except Exception as e:
                print(f"⚠️ 検証エラー: {e}")
        
        # エポック統計
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_triplet_loss = train_triplet_loss / max(train_batches, 1)
        avg_adv_loss = train_adv_loss / max(train_batches, 1) if config.use_adversarial else 0
        avg_val_loss = val_loss / max(val_batches, 1)
        
        print(f"\n📊 エポック {epoch+1} 結果:")
        print(f"   - 時間: {epoch_time:.1f}秒 ({train_batches}訓練バッチ, {val_batches}検証バッチ)")
        print(f"   - 訓練Loss: {avg_train_loss:.4f} (Triplet: {avg_triplet_loss:.4f}, Adv: {avg_adv_loss:.4f})")
        print(f"   - 検証Loss: {avg_val_loss:.4f}")
        print(f"   - 学習率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # キャッシュ統計
        train_stats = train_dicom_manager.get_stats()
        val_stats = val_dicom_manager.get_stats()
        
        print(f"   - キャッシュヒット率: Train {train_stats['cache_hit_rate']:.1%}, Val {val_stats['cache_hit_rate']:.1%}")
        print(f"   - メモリ使用量: Train {train_stats['memory_usage_mb']:.0f}MB, Val {val_stats['memory_usage_mb']:.0f}MB")
        print(f"   - ディスク使用量: Train {train_stats['disk_usage_gb']:.1f}GB, Val {val_stats['disk_usage_gb']:.1f}GB")
        
        # ベストモデル保存
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config.__dict__,
                'settings': settings
            }
            
            torch.save(checkpoint, save_path)
            print(f"   💾 ベストモデル保存: Loss {best_loss:.4f}")
        
        # メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # 最終統計
    print("\n🎉 修正版トレーニング完了!")
    print("=" * 60)
    
    final_train_stats = train_dicom_manager.get_stats()
    final_val_stats = val_dicom_manager.get_stats()
    
    print(f"📊 最終統計:")
    print(f"   - 最終ベストロス: {best_loss:.4f}")
    print(f"   - 訓練キャッシュ: {final_train_stats['memory_cache_items']}アイテム")
    print(f"   - 検証キャッシュ: {final_val_stats['memory_cache_items']}アイテム")
    print(f"   - 総ディスク使用量: {final_train_stats['disk_usage_gb'] + final_val_stats['disk_usage_gb']:.1f}GB")
    
    # リソース解放
    train_dicom_manager.shutdown()
    val_dicom_manager.shutdown()


if __name__ == "__main__":
    dataset_path = "/mnt/s/Workfolder/vector_embedding_echo/dataset/datasplit/"
    
    print("🔧 修正版1TB DICOMデータトレーニング")
    print("=" * 60)
    
    # デバッグ用（小規模テスト）
    # main_fixed(
    #     dataset_path + "train_debag_ds.csv",
    #     dataset_path + "val_debag_ds.csv", 
    #     "echo_fixed_debug",
    #     use_adversarial=False,
    #     epochs=2
    # )
    
    # 本番用（LoRAのみ）
    main_fixed(
        dataset_path + "train_ds.csv",
        dataset_path + "val_ds.csv",
        "echo_fixed_lora",
        use_adversarial=False,
        epochs=10
    )
    
    # 本番用（LoRA + Adversarial）
    # main_fixed(
    #     dataset_path + "train_ds.csv", 
    #     dataset_path + "val_ds.csv",
    #     "echo_fixed_lora_adv",
    #     use_adversarial=True,
    #     epochs=10
    # )
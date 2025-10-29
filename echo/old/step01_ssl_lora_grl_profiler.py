# step01_ssl_lora_grl_profiled.py - プロファイリング機能付きバージョン

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import pandas as pd
import sys
import time
import psutil
import numpy as np
from collections import defaultdict

# ローカルモジュール
sys.path.insert(0, "vector-embedding/echo/module/veecho")
from model_utils import EchoEmbeddingModel, TrainingConfig
import data_utils, train_utils


# step01の冒頭に追加して実行
from video_loading_diagnosis import run_diagnosis

# 詳細診断実行
dataset_path = "/mnt/s/Workfolder/vector_embedding_echo/dataset/datasplit/"
diagnosis_result = run_diagnosis(dataset_path + "train_sel_ds.csv")


class QuickProfiler:
    """軽量プロファイラー"""
    
    def __init__(self):
        self.stats = defaultdict(list)
    
    def profile_quick_batch_analysis(self, train_loader, model, optimizer, 
                                   criterion_triplet, criterion_adv, device, config, 
                                   num_batches=5):
        """高速バッチ分析"""
        
        print("🔍 Quick profiling (5 batches)...")
        model.train()
        
        times = {'total': [], 'data_load': [], 'forward': [], 'backward': []}
        
        loader_iter = iter(train_loader)
        
        for i in range(min(num_batches, len(train_loader))):
            try:
                batch_start = time.time()
                
                # データロード時間
                data_start = time.time()
                batch = next(loader_iter)
                anchor = batch['anchor'].to(device, non_blocking=True)
                positive = batch['positive'].to(device, non_blocking=True)
                negative = batch['negative'].to(device, non_blocking=True)
                data_time = time.time() - data_start
                
                optimizer.zero_grad()
                
                # Forward時間
                forward_start = time.time()
                anchor_features = model(anchor)
                pos_features = model(positive)
                neg_features = model(negative)
                triplet_loss = criterion_triplet(anchor_features, pos_features, neg_features)
                total_loss = triplet_loss
                
                if config.use_adversarial:
                    for attr in config.adversarial_attributes:
                        if attr in batch:
                            attr_labels = batch[attr].to(device, non_blocking=True)
                            attr_pred = model.adversarial_heads[attr](anchor_features)
                            attr_loss = criterion_adv(attr_pred, attr_labels)
                            total_loss += attr_loss
                
                forward_time = time.time() - forward_start
                
                # Backward時間
                backward_start = time.time()
                total_loss.backward()
                optimizer.step()
                backward_time = time.time() - backward_start
                
                total_time = time.time() - batch_start
                
                times['total'].append(total_time)
                times['data_load'].append(data_time)
                times['forward'].append(forward_time)
                times['backward'].append(backward_time)
                
                print(f"  Batch {i+1}: {total_time:.3f}s (data: {data_time:.3f}s, "
                      f"forward: {forward_time:.3f}s, backward: {backward_time:.3f}s)")
                
            except StopIteration:
                break
            except Exception as e:
                print(f"  Batch {i+1}: Error - {e}")
                break
        
        if times['total']:
            avg_stats = {
                'avg_total': np.mean(times['total']),
                'avg_data_load': np.mean(times['data_load']),
                'avg_forward': np.mean(times['forward']),
                'avg_backward': np.mean(times['backward'])
            }
            
            data_ratio = avg_stats['avg_data_load'] / avg_stats['avg_total']
            forward_ratio = avg_stats['avg_forward'] / avg_stats['avg_total']
            backward_ratio = avg_stats['avg_backward'] / avg_stats['avg_total']
            
            print(f"\n📊 Quick Analysis Results:")
            print(f"  Average batch time: {avg_stats['avg_total']:.3f}s")
            print(f"  Data loading: {avg_stats['avg_data_load']:.3f}s ({data_ratio*100:.1f}%)")
            print(f"  Forward pass: {avg_stats['avg_forward']:.3f}s ({forward_ratio*100:.1f}%)")
            print(f"  Backward pass: {avg_stats['avg_backward']:.3f}s ({backward_ratio*100:.1f}%)")
            
            # ボトルネック特定
            print(f"\n💡 Quick Bottleneck Analysis:")
            if data_ratio > 0.3:
                print(f"  ❌ Data loading is slow ({data_ratio*100:.1f}% of batch time)")
                print(f"     → Try: increase num_workers, use pin_memory=True")
            else:
                print(f"  ✅ Data loading is efficient ({data_ratio*100:.1f}%)")
            
            if forward_ratio > 0.5:
                print(f"  ❌ Forward pass is slow ({forward_ratio*100:.1f}% of batch time)")
                print(f"     → Try: mixed precision, reduce batch size, optimize model")
            else:
                print(f"  ✅ Forward pass is efficient ({forward_ratio*100:.1f}%)")
            
            if backward_ratio > 0.3:
                print(f"  ❌ Backward pass is slow ({backward_ratio*100:.1f}% of batch time)")
                print(f"     → Try: gradient accumulation, check optimizer settings")
            else:
                print(f"  ✅ Backward pass is efficient ({backward_ratio*100:.1f}%)")
            
            return avg_stats
        else:
            print("❌ No valid batches processed")
            return None
    
    def check_system_resources(self):
        """システムリソースチェック"""
        print(f"\n🖥️  System Resources:")
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        print(f"  CPU: {cpu_percent:.1f}% ({cpu_count} cores)")
        
        # メモリ
        memory = psutil.virtual_memory()
        print(f"  RAM: {memory.percent:.1f}% ({memory.available / 1024**3:.1f}GB available)")
        
        # GPU
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i} ({gpu_name}): {memory_allocated:.1f}GB/{memory_total:.1f}GB allocated")
        else:
            print("  GPU: Not available")
    
    def analyze_dataloader(self, train_loader, num_samples=10):
        """データローダー分析"""
        print(f"\n📁 DataLoader Analysis:")
        
        load_times = []
        batch_sizes = []
        
        try:
            start_time = time.time()
            for i, batch in enumerate(train_loader):
                if i >= num_samples:
                    break
                
                batch_start = time.time()
                batch_size = batch['anchor'].size(0)
                load_time = time.time() - batch_start
                
                load_times.append(load_time)
                batch_sizes.append(batch_size)
                
                if i < 3:  # 最初の3バッチのみ詳細表示
                    print(f"  Batch {i+1}: {load_time:.3f}s, size={batch_size}")
            
            total_time = time.time() - start_time
            
            if load_times:
                avg_load_time = np.mean(load_times)
                avg_batch_size = np.mean(batch_sizes)
                throughput = sum(batch_sizes) / total_time
                
                print(f"  Average load time: {avg_load_time:.3f}s")
                print(f"  Average batch size: {avg_batch_size:.1f}")
                print(f"  Throughput: {throughput:.1f} samples/sec")
                
                return {
                    'avg_load_time': avg_load_time,
                    'avg_batch_size': avg_batch_size,
                    'throughput': throughput
                }
        except Exception as e:
            print(f"  Error analyzing dataloader: {e}")
            return None
    
    def analyze_model_params(self, model):
        """モデルパラメータ分析"""
        print(f"\n🧠 Model Analysis:")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # メモリ推定
        param_memory_mb = trainable_params * 4 / 1024**2  # float32
        print(f"  Estimated parameter memory: {param_memory_mb:.1f} MB")
        
        # LoRA分析
        lora_params = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_params += param.numel()
        
        if lora_params > 0:
            print(f"  LoRA parameters: {lora_params:,} ({lora_params/trainable_params*100:.2f}% of trainable)")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'param_memory_mb': param_memory_mb,
            'lora_params': lora_params
        }

def main(train_set_csv,
         val_set_csv,
         model_name,
         use_adversarial=True,
         epochs=10,
         run_profiling=True):
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
    
    # プロファイラー初期化
    profiler = QuickProfiler()
    
    # システムリソースチェック
    if run_profiling:
        profiler.check_system_resources()
    
    # ベースモデル読み込み
    print(f"\n📥 Loading base model...")
    model_load_start = time.time()
    
    checkpoint = torch.load("/mnt/s/Workfolder/vector_embedding_echo/model/echoprime/echo_prime_encoder.pt", map_location=device)
    echo_encoder = torchvision.models.video.mvit_v2_s()
    echo_encoder.head[-1] = nn.Linear(echo_encoder.head[-1].in_features, 512)
    echo_encoder.load_state_dict(checkpoint)
    
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")
    
    # モデル作成
    print(f"\n🏗️  Creating model with LoRA and adversarial heads...")
    model_create_start = time.time()
    
    model = EchoEmbeddingModel(echo_encoder, config)
    model.to(device)
    
    # Adversarial headsもデバイスに移動
    if config.use_adversarial:
        for head in model.adversarial_heads.values():
            head.to(device)
    
    model_create_time = time.time() - model_create_start
    print(f"Model created in {model_create_time:.2f}s")
    
    # モデル分析
    if run_profiling:
        profiler.analyze_model_params(model)
    
    # 損失関数と最適化器
    criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion_adv = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # データローダー作成
    print(f"\n📊 Creating data loaders...")
    dataloader_start = time.time()
    
    train_loader = data_utils.create_dataloader(
        csv_path=train_set_csv,
        config=config,
        is_validation=False,
        batch_size=8,
        buffer_size=100,
        seed=42
    )
    
    val_loader = data_utils.create_dataloader(
        csv_path=val_set_csv,
        config=config,
        is_validation=True,
        batch_size=8,
        seed=42
    )
    
    dataloader_time = time.time() - dataloader_start
    print(f"Data loaders created in {dataloader_time:.2f}s")
    
    # データローダー分析
    if run_profiling:
        profiler.analyze_dataloader(train_loader, num_samples=5)
    
    # 🔍 詳細プロファイリング実行
    if run_profiling:
        print(f"\n" + "="*60)
        print("🔍 PERFORMANCE PROFILING")
        print("="*60)
        
        batch_stats = profiler.profile_quick_batch_analysis(
            train_loader, model, optimizer, criterion_triplet, 
            criterion_adv, device, config, num_batches=5
        )
        
        # 最適化提案
        print(f"\n🚀 Optimization Suggestions:")
        
        if batch_stats:
            data_ratio = batch_stats['avg_data_load'] / batch_stats['avg_total']
            forward_ratio = batch_stats['avg_forward'] / batch_stats['avg_total']
            
            suggestions = []
            
            if data_ratio > 0.25:
                suggestions.extend([
                    "• Increase DataLoader num_workers (current: default)",
                    "• Add pin_memory=True to DataLoader",
                    "• Use non_blocking=True for .to(device) calls",
                    "• Consider data preprocessing optimization"
                ])
            
            if forward_ratio > 0.4:
                suggestions.extend([
                    "• Enable mixed precision training (torch.cuda.amp)",
                    "• Consider gradient checkpointing",
                    "• Optimize batch size vs memory usage",
                    "• Profile individual model components"
                ])
            
            if batch_stats['avg_total'] > 2.0:
                suggestions.extend([
                    "• Overall batch time is slow - consider:",
                    "  - Reducing model complexity",
                    "  - Using smaller input resolution",
                    "  - Implementing gradient accumulation with smaller batches"
                ])
            
            # LoRA最適化
            model_stats = profiler.analyze_model_params(model)
            if model_stats and model_stats['lora_params'] > 0:
                total_trainable = model_stats['trainable_params']
                lora_ratio = model_stats['lora_params'] / total_trainable
                if lora_ratio < 0.5:
                    suggestions.append("• Consider increasing LoRA rank for better model capacity")
            
            if suggestions:
                for suggestion in suggestions:
                    print(f"  {suggestion}")
            else:
                print("  ✅ Training pipeline appears well optimized!")
        
        print(f"\n" + "="*60)
    
    # トレーナー作成
    trainer = train_utils.create_trainer(model, log_dir="/mnt/s/Workfolder/vector_embedding_echo/logs/"+ model_name)
    
    # 🚀 トレーニング開始
    print(f"\n🚀 Starting training for {epochs} epochs...")
    training_start = time.time()
    
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
    
    total_training_time = time.time() - training_start
    print(f"\n🎉 Training completed in {total_training_time:.1f}s ({total_training_time/60:.1f} minutes)")
    
    # 最終統計
    if run_profiling and batch_stats:
        estimated_time_per_epoch = batch_stats['avg_total'] * len(train_loader)
        print(f"\n📈 Performance Summary:")
        print(f"  Average batch time: {batch_stats['avg_total']:.3f}s")
        print(f"  Estimated time per epoch: {estimated_time_per_epoch:.1f}s ({estimated_time_per_epoch/60:.1f} minutes)")
        print(f"  Actual time per epoch: {total_training_time/epochs:.1f}s ({total_training_time/epochs/60:.1f} minutes)")

def run_optimized_dataloader_test():
    """最適化されたDataLoaderのテスト"""
    print("🧪 Testing optimized DataLoader configurations...")
    
    dataset_path = "/mnt/s/Workfolder/vector_embedding_echo/dataset/datasplit/"
    train_csv = dataset_path + "train_sel_ds.csv"
    
    config = TrainingConfig(use_adversarial=True, use_lora=True)
    
    # 異なる設定でテスト
    test_configs = [
        {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
        {"num_workers": 2, "pin_memory": True, "persistent_workers": False},
        {"num_workers": 4, "pin_memory": True, "persistent_workers": True},
        {"num_workers": 6, "pin_memory": True, "persistent_workers": True},
    ]
    
    for i, test_config in enumerate(test_configs):
        print(f"\n📊 Test {i+1}: {test_config}")
        
        try:
            # DataLoaderを作成（data_utils.pyの関数を使用）
            loader = data_utils.create_dataloader(
                csv_path=train_csv,
                config=config,
                is_validation=False,
                batch_size=8,
                buffer_size=100,
                seed=42,
                **test_config  # 追加の設定を渡す
            )
            
            # 速度テスト
            load_times = []
            start_time = time.time()
            
            for j, batch in enumerate(loader):
                if j >= 10:  # 10バッチのみテスト
                    break
                batch_time = time.time()
                _ = batch['anchor'].shape  # データアクセス
                load_times.append(time.time() - batch_time)
            
            avg_time = sum(load_times) / len(load_times) if load_times else float('inf')
            total_time = time.time() - start_time
            
            print(f"  Average batch load: {avg_time:.4f}s")
            print(f"  Total time (10 batches): {total_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    dataset_path = "/mnt/s/Workfolder/vector_embedding_echo/dataset/datasplit/"
    
    # データローダー最適化テスト（オプション）
    # run_optimized_dataloader_test()
    
    # プロファイリング付きで実行
    print("🔍 Running with profiling enabled...")
    
    main(dataset_path + "train_sel_ds.csv", dataset_path + "val_sel_ds.csv",
         "echomodel_lora_sub_profiled", use_adversarial=False, epochs=2, run_profiling=True)
    
    main(dataset_path + "train_sel_ds.csv", dataset_path + "val_sel_ds.csv",
         "echomodel_lora_adv_sub_profiled", use_adversarial=True, epochs=2, run_profiling=True)
    
    # フルデータセットでの実行（コメントアウト）
    # main(dataset_path + "train_ds.csv", dataset_path + "val_ds.csv",
    #      "echomodel_lora_profiled", use_adversarial=False, epochs=10, run_profiling=True)
    # main(dataset_path + "train_ds.csv", dataset_path + "val_ds.csv",
    #      "echomodel_lora_adv_profiled", use_adversarial=True, epochs=10, run_profiling=True)
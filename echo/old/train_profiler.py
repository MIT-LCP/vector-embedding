# training_profiler.py - 訓練速度のボトルネック分析ツール

import torch
import torch.profiler
import time
import psutil
import GPUtil
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

class TrainingProfiler:
    """訓練速度のボトルネック分析クラス"""
    
    def __init__(self, log_dir="./profile_logs"):
        self.log_dir = log_dir
        self.timing_stats = defaultdict(list)
        self.memory_stats = []
        self.gpu_stats = []
        
    def profile_batch_processing(self, train_loader, model, optimizer, 
                                criterion_triplet, criterion_adv, device, config, 
                                num_batches=10):
        """バッチ処理の詳細プロファイリング"""
        
        print("🔍 Starting detailed batch profiling...")
        model.train()
        
        batch_times = []
        data_load_times = []
        forward_times = []
        backward_times = []
        
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
                
            batch_start = time.time()
            
            # データ転送時間
            data_start = time.time()
            anchor = batch['anchor'].to(device, non_blocking=True)
            positive = batch['positive'].to(device, non_blocking=True)
            negative = batch['negative'].to(device, non_blocking=True)
            data_load_time = time.time() - data_start
            
            optimizer.zero_grad()
            
            # Forward pass時間
            forward_start = time.time()
            anchor_features = model(anchor)
            pos_features = model(positive)
            neg_features = model(negative)
            
            # 損失計算
            triplet_loss = criterion_triplet(anchor_features, pos_features, neg_features)
            total_loss = triplet_loss
            
            # Adversarial losses
            if config.use_adversarial:
                for attr in config.adversarial_attributes:
                    if attr in batch:
                        attr_labels = batch[attr].to(device, non_blocking=True)
                        attr_pred = model.adversarial_heads[attr](anchor_features)
                        attr_loss = criterion_adv(attr_pred, attr_labels)
                        total_loss += attr_loss
            
            forward_time = time.time() - forward_start
            
            # Backward pass時間
            backward_start = time.time()
            total_loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start
            
            batch_time = time.time() - batch_start
            
            # 統計記録
            batch_times.append(batch_time)
            data_load_times.append(data_load_time)
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            
            # メモリ使用量記録
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(device) / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
                self.memory_stats.append({
                    'batch': i,
                    'memory_used_gb': memory_used,
                    'memory_cached_gb': memory_cached
                })
            
            # GPU使用率記録
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.gpu_stats.append({
                        'batch': i,
                        'gpu_util': gpu.load * 100,
                        'memory_util': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature
                    })
            except:
                pass
            
            if i % 5 == 0:
                print(f"Batch {i+1}: Total={batch_time:.3f}s, "
                      f"Data={data_load_time:.3f}s, "
                      f"Forward={forward_time:.3f}s, "
                      f"Backward={backward_time:.3f}s")
        
        # 統計サマリー
        self.timing_stats['batch_times'] = batch_times
        self.timing_stats['data_load_times'] = data_load_times
        self.timing_stats['forward_times'] = forward_times
        self.timing_stats['backward_times'] = backward_times
        
        return {
            'avg_batch_time': np.mean(batch_times),
            'avg_data_load_time': np.mean(data_load_times),
            'avg_forward_time': np.mean(forward_times),
            'avg_backward_time': np.mean(backward_times),
            'data_load_ratio': np.mean(data_load_times) / np.mean(batch_times),
            'forward_ratio': np.mean(forward_times) / np.mean(batch_times),
            'backward_ratio': np.mean(backward_times) / np.mean(batch_times)
        }
    
    def profile_dataloader(self, train_loader, num_batches=20):
        """データローダーのプロファイリング"""
        
        print("🔍 Profiling data loader...")
        
        load_times = []
        batch_sizes = []
        
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            
            # バッチサイズ記録
            batch_sizes.append(batch['anchor'].size(0))
            
            # データをCPUで処理する時間を測定
            _ = batch['anchor'].numpy() if hasattr(batch['anchor'], 'numpy') else None
            
            load_time = time.time() - batch_start
            load_times.append(load_time)
            
            if i % 10 == 0:
                print(f"Batch {i+1}: Load time={load_time:.3f}s, Size={batch_sizes[-1]}")
        
        total_time = time.time() - start_time
        
        return {
            'avg_load_time': np.mean(load_times),
            'total_time': total_time,
            'avg_batch_size': np.mean(batch_sizes),
            'throughput_samples_per_sec': sum(batch_sizes) / total_time
        }
    
    def analyze_model_complexity(self, model):
        """モデルの計算複雑度分析"""
        
        print("🔍 Analyzing model complexity...")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # レイヤー別パラメータ数
        layer_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_params[name] = param.numel()
        
        # メモリ使用量推定
        param_memory_mb = sum(p.numel() * 4 for p in model.parameters() if p.requires_grad) / 1024**2  # 4 bytes per float32
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / total_params,
            'param_memory_mb': param_memory_mb,
            'layer_params': layer_params
        }
    
    def pytorch_profiler_analysis(self, train_loader, model, optimizer, 
                                 criterion_triplet, criterion_adv, device, config):
        """PyTorch Profilerを使用した詳細分析"""
        
        print("🔍 Running PyTorch Profiler...")
        
        def trace_handler(prof):
            print(f"Profiler step {prof.step_num}")
            if prof.step_num == 5:  # 5回目のステップで結果を出力
                prof.export_chrome_trace(f"{self.log_dir}/trace.json")
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        
        model.train()
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            
            for step, batch in enumerate(train_loader):
                if step >= 10:  # 10バッチで終了
                    break
                
                anchor = batch['anchor'].to(device, non_blocking=True)
                positive = batch['positive'].to(device, non_blocking=True)
                negative = batch['negative'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
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
                
                total_loss.backward()
                optimizer.step()
                
                prof.step()
    
    def system_resource_monitor(self, duration_minutes=2):
        """システムリソースモニタリング"""
        
        print(f"🔍 Monitoring system resources for {duration_minutes} minutes...")
        
        cpu_usage = []
        memory_usage = []
        timestamps = []
        
        start_time = time.time()
        end_time = start_time + duration_minutes * 60
        
        while time.time() < end_time:
            current_time = time.time() - start_time
            timestamps.append(current_time)
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage.append(cpu_percent)
            
            # メモリ使用率
            memory = psutil.virtual_memory()
            memory_usage.append(memory.percent)
            
            print(f"Time: {current_time:.1f}s, CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")
        
        return {
            'timestamps': timestamps,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'avg_cpu': np.mean(cpu_usage),
            'avg_memory': np.mean(memory_usage)
        }
    
    def generate_report(self, batch_stats, dataloader_stats, model_stats):
        """プロファイリング結果のレポート生成"""
        
        print("\n" + "="*80)
        print("🔍 TRAINING PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        
        print("\n📊 BATCH PROCESSING ANALYSIS")
        print(f"Average batch time: {batch_stats['avg_batch_time']:.3f}s")
        print(f"Data loading time: {batch_stats['avg_data_load_time']:.3f}s ({batch_stats['data_load_ratio']*100:.1f}%)")
        print(f"Forward pass time: {batch_stats['avg_forward_time']:.3f}s ({batch_stats['forward_ratio']*100:.1f}%)")
        print(f"Backward pass time: {batch_stats['avg_backward_time']:.3f}s ({batch_stats['backward_ratio']*100:.1f}%)")
        
        print("\n📈 DATA LOADER ANALYSIS")
        print(f"Average load time per batch: {dataloader_stats['avg_load_time']:.3f}s")
        print(f"Average batch size: {dataloader_stats['avg_batch_size']:.1f}")
        print(f"Throughput: {dataloader_stats['throughput_samples_per_sec']:.1f} samples/sec")
        
        print("\n🧠 MODEL COMPLEXITY ANALYSIS")
        print(f"Total parameters: {model_stats['total_params']:,}")
        print(f"Trainable parameters: {model_stats['trainable_params']:,} ({model_stats['trainable_ratio']*100:.2f}%)")
        print(f"Parameter memory: {model_stats['param_memory_mb']:.1f} MB")
        
        print("\n💡 BOTTLENECK IDENTIFICATION")
        bottlenecks = []
        
        if batch_stats['data_load_ratio'] > 0.3:
            bottlenecks.append("❌ Data loading is slow (>30% of batch time)")
        else:
            bottlenecks.append("✅ Data loading is efficient")
            
        if batch_stats['forward_ratio'] > 0.5:
            bottlenecks.append("❌ Forward pass is slow (>50% of batch time)")
        else:
            bottlenecks.append("✅ Forward pass is efficient")
            
        if batch_stats['backward_ratio'] > 0.3:
            bottlenecks.append("❌ Backward pass is slow (>30% of batch time)")
        else:
            bottlenecks.append("✅ Backward pass is efficient")
        
        for bottleneck in bottlenecks:
            print(f"  {bottleneck}")
        
        print("\n🔧 OPTIMIZATION RECOMMENDATIONS")
        recommendations = []
        
        if batch_stats['data_load_ratio'] > 0.3:
            recommendations.extend([
                "• Increase num_workers in DataLoader",
                "• Use pin_memory=True for GPU training",
                "• Consider data preprocessing optimization",
                "• Use non_blocking=True for data transfer"
            ])
        
        if batch_stats['forward_ratio'] > 0.5:
            recommendations.extend([
                "• Consider mixed precision training (AMP)",
                "• Reduce batch size if memory allows more frequent updates",
                "• Optimize model architecture",
                "• Use gradient checkpointing for memory-compute tradeoff"
            ])
        
        if model_stats['trainable_ratio'] < 0.1:
            recommendations.append("• Consider increasing LoRA rank for better capacity")
        
        if not recommendations:
            recommendations.append("• Training pipeline seems well optimized!")
        
        for rec in recommendations:
            print(f"  {rec}")
        
        print("\n" + "="*80)
    
    def plot_profiling_results(self):
        """プロファイリング結果の可視化"""
        
        if not self.timing_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # バッチ処理時間の内訳
        ax = axes[0, 0]
        times = ['Data Load', 'Forward', 'Backward']
        avg_times = [
            np.mean(self.timing_stats['data_load_times']),
            np.mean(self.timing_stats['forward_times']),
            np.mean(self.timing_stats['backward_times'])
        ]
        ax.bar(times, avg_times, color=['lightcoral', 'lightblue', 'lightgreen'])
        ax.set_title('Average Time per Phase')
        ax.set_ylabel('Time (seconds)')
        
        # バッチ時間の推移
        ax = axes[0, 1]
        batches = range(len(self.timing_stats['batch_times']))
        ax.plot(batches, self.timing_stats['batch_times'], 'b-', alpha=0.7)
        ax.set_title('Batch Processing Time')
        ax.set_xlabel('Batch Number')
        ax.set_ylabel('Time (seconds)')
        
        # メモリ使用량
        if self.memory_stats:
            ax = axes[1, 0]
            memory_df = pd.DataFrame(self.memory_stats)
            ax.plot(memory_df['batch'], memory_df['memory_used_gb'], 'r-', label='Used')
            ax.plot(memory_df['batch'], memory_df['memory_cached_gb'], 'b--', label='Cached')
            ax.set_title('GPU Memory Usage')
            ax.set_xlabel('Batch Number')
            ax.set_ylabel('Memory (GB)')
            ax.legend()
        
        # GPU使用率
        if self.gpu_stats:
            ax = axes[1, 1]
            gpu_df = pd.DataFrame(self.gpu_stats)
            ax.plot(gpu_df['batch'], gpu_df['gpu_util'], 'g-', label='GPU Util')
            ax.plot(gpu_df['batch'], gpu_df['memory_util'], 'r-', label='Memory Util')
            ax.set_title('GPU Utilization')
            ax.set_xlabel('Batch Number')
            ax.set_ylabel('Utilization (%)')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/profiling_results.png', dpi=150, bbox_inches='tight')
        plt.show()

# 使用例
def run_comprehensive_profiling(train_loader, model, optimizer, criterion_triplet, 
                               criterion_adv, device, config):
    """包括的なプロファイリング実行"""
    
    profiler = TrainingProfiler()
    
    # 1. バッチ処理プロファイリング
    batch_stats = profiler.profile_batch_processing(
        train_loader, model, optimizer, criterion_triplet, 
        criterion_adv, device, config, num_batches=10
    )
    
    # 2. データローダープロファイリング
    dataloader_stats = profiler.profile_dataloader(train_loader, num_batches=20)
    
    # 3. モデル複雑度分析
    model_stats = profiler.analyze_model_complexity(model)
    
    # 4. PyTorch Profiler（オプション）
    # profiler.pytorch_profiler_analysis(
    #     train_loader, model, optimizer, criterion_triplet, criterion_adv, device, config
    # )
    
    # 5. レポート生成
    profiler.generate_report(batch_stats, dataloader_stats, model_stats)
    
    # 6. 結果可視化
    profiler.plot_profiling_results()
    
    return profiler

# step01.pyに統合するための修正関数
def add_profiling_to_main():
    """step01.pyに追加するプロファイリングコード"""
    profiling_code = '''
    # プロファイリング実行（訓練前に追加）
    print("\\n🔍 Running performance profiling...")
    from training_profiler import run_comprehensive_profiling
    
    profiler = run_comprehensive_profiling(
        train_loader, model, optimizer, criterion_triplet, 
        criterion_adv, device, config
    )
    
    # 通常の訓練は続行...
    '''
    return profiling_code
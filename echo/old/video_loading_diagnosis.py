# video_loading_diagnosis.py - ビデオロード問題の詳細診断

import pandas as pd
import cv2
import os
import time
import torch
import numpy as np
from pathlib import Path
import concurrent.futures
from collections import defaultdict

class VideoLoadingDiagnostic:
    """ビデオロード問題の詳細診断"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        print(f"📊 Loaded dataset: {len(self.df)} samples")
        
    def analyze_file_system(self):
        """ファイルシステムとパス分析"""
        print("\n🗂️  FILE SYSTEM ANALYSIS")
        print("="*50)
        
        # パス分析
        sample_paths = []
        for col in ['dicom_path']:
            if col in self.df.columns:
                sample_paths.extend(self.df[col].dropna().head(10).tolist())
        
        for i, path in enumerate(sample_paths[:5]):
            print(f"Sample path {i+1}: {path}")
            
            # ファイル存在確認
            exists = os.path.exists(path)
            
            if exists:
                # ファイルサイズ
                size_mb = os.path.getsize(path) / (1024*1024)
                
                # ディスクタイプ推定（簡易）
                disk_type = "Unknown"
                if "/mnt/" in path:
                    disk_type = "Mounted drive (possibly network/external)"
                elif "SSD" in path.upper() or "NVME" in path.upper():
                    disk_type = "SSD (estimated)"
                elif "HDD" in path.upper():
                    disk_type = "HDD (estimated)"
                
                print(f"  ✅ Exists: {size_mb:.1f}MB ({disk_type})")
            else:
                print(f"  ❌ Missing file")
        
        # パスの共通パターン
        common_dirs = defaultdict(int)
        for path in sample_paths:
            parent_dir = str(Path(path).parent)
            common_dirs[parent_dir] += 1
        
        print(f"\nCommon directories:")
        for dir_path, count in sorted(common_dirs.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {dir_path}: {count} files")
    
    def benchmark_single_video_loading(self, num_videos=10):
        """単一ビデオ読み込みのベンチマーク"""
        print(f"\n📹 SINGLE VIDEO LOADING BENCHMARK")
        print("="*50)
        
        # サンプルビデオパスを取得
        test_paths = []
        for col in ['dicom_path']:
            if col in self.df.columns:
                test_paths.extend(self.df[col].dropna().head(num_videos//3).tolist())
        
        test_paths = test_paths[:num_videos]
        
        load_times = []
        file_sizes = []
        error_count = 0
        
        for i, video_path in enumerate(test_paths):
            print(f"\nTesting video {i+1}: {os.path.basename(video_path)}")
            
            if not os.path.exists(video_path):
                print(f"  ❌ File not found")
                error_count += 1
                continue
            
            # ファイルサイズ
            file_size_mb = os.path.getsize(video_path) / (1024*1024)
            file_sizes.append(file_size_mb)
            print(f"  📁 Size: {file_size_mb:.1f}MB")
            
            # ロード時間測定
            start_time = time.time()
            
            try:
                # OpenCVでビデオ読み込み
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    print(f"  ❌ Cannot open with OpenCV")
                    error_count += 1
                    continue
                
                # ビデオ情報取得
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"  📊 Props: {width}x{height}, {frame_count} frames, {fps:.1f}fps")
                
                # 16フレーム読み込み（実際の処理をシミュレート）
                frames = []
                frame_interval = max(1, frame_count // 16)
                
                for frame_idx in range(0, frame_count, frame_interval):
                    if len(frames) >= 16:
                        break
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        # リサイズと正規化（実際の処理）
                        frame = cv2.resize(frame, (224, 224))
                        frame = frame.astype(np.float32) / 255.0
                        frames.append(frame)
                
                cap.release()
                
                load_time = time.time() - start_time
                load_times.append(load_time)
                
                print(f"  ⏱️  Load time: {load_time:.3f}s ({file_size_mb/load_time:.1f}MB/s)")
                
                # 異常に遅い場合の警告
                if load_time > 10:
                    print(f"  🚨 VERY SLOW: {load_time:.1f}s is abnormally slow!")
                elif load_time > 5:
                    print(f"  ⚠️  SLOW: {load_time:.1f}s is slower than expected")
                elif load_time < 1:
                    print(f"  ✅ FAST: Good performance")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                error_count += 1
        
        # 統計サマリー
        if load_times:
            avg_load_time = np.mean(load_times)
            max_load_time = np.max(load_times)
            min_load_time = np.min(load_times)
            avg_file_size = np.mean(file_sizes) if file_sizes else 0
            
            print(f"\n📈 LOADING STATISTICS:")
            print(f"  Average load time: {avg_load_time:.3f}s")
            print(f"  Min/Max load time: {min_load_time:.3f}s / {max_load_time:.3f}s")
            print(f"  Average file size: {avg_file_size:.1f}MB")
            print(f"  Average throughput: {avg_file_size/avg_load_time:.1f}MB/s")
            print(f"  Error rate: {error_count}/{len(test_paths)} ({error_count/len(test_paths)*100:.1f}%)")
            
            # 問題診断
            print(f"\n🔍 DIAGNOSIS:")
            if avg_load_time > 5:
                print(f"  🚨 CRITICAL: Average load time ({avg_load_time:.1f}s) is extremely slow")
                print(f"     → Likely causes: slow disk, network storage, large files")
            elif avg_load_time > 2:
                print(f"  ⚠️  WARNING: Load time ({avg_load_time:.1f}s) is slower than optimal")
                print(f"     → Consider: SSD upgrade, file compression, preprocessing")
            else:
                print(f"  ✅ Load time ({avg_load_time:.1f}s) seems reasonable")
            
            if max_load_time > avg_load_time * 3:
                print(f"  ⚠️  Large variance: Some files are much slower than others")
                print(f"     → Check: file corruption, different formats, network issues")
        
        return {
            'avg_load_time': np.mean(load_times) if load_times else float('inf'),
            'max_load_time': np.max(load_times) if load_times else float('inf'),
            'error_rate': error_count / len(test_paths) if test_paths else 1.0,
            'avg_file_size': np.mean(file_sizes) if file_sizes else 0
        }
    
    def test_parallel_loading(self, num_workers=4, num_videos=12):
        """並列ロードのテスト"""
        print(f"\n🔄 PARALLEL LOADING TEST ({num_workers} workers)")
        print("="*50)
        
        # テスト用ビデオパス
        test_paths = []
        for col in ['dicom_path']:
            if col in self.df.columns:
                test_paths.extend(self.df[col].dropna().head(num_videos//3).tolist())
        
        test_paths = test_paths[:num_videos]
        existing_paths = [p for p in test_paths if os.path.exists(p)]
        
        if not existing_paths:
            print("❌ No valid video files found for testing")
            return
        
        # シーケンシャルロード
        print("Testing sequential loading...")
        start_time = time.time()
        for path in existing_paths:
            self._load_video_simple(path)
        sequential_time = time.time() - start_time
        
        # 並列ロード
        print(f"Testing parallel loading with {num_workers} workers...")
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._load_video_simple, path) for path in existing_paths]
            concurrent.futures.wait(futures)
        parallel_time = time.time() - start_time
        
        # 結果
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        
        print(f"\n📊 PARALLEL LOADING RESULTS:")
        print(f"  Sequential time: {sequential_time:.2f}s")
        print(f"  Parallel time: {parallel_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup < 1.5:
            print(f"  ⚠️  Poor parallelization - likely I/O bound")
            print(f"     → Bottleneck: disk speed, not CPU")
        elif speedup > 2:
            print(f"  ✅ Good parallelization benefit")
        
        return speedup
    
    def _load_video_simple(self, video_path):
        """簡単なビデオロード（テスト用）"""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in range(min(16, frame_count)):
                    ret, frame = cap.read()
                    if not ret:
                        break
                cap.release()
            return True
        except:
            return False
    
    def check_disk_performance(self):
        """ディスク性能の簡易チェック"""
        print(f"\n💾 DISK PERFORMANCE CHECK")
        print("="*50)
        
        # サンプルファイルパス
        sample_paths = self.df['dicom_path'].dropna().head(5).tolist()
        
        for path in sample_paths:
            if os.path.exists(path):
                parent_dir = os.path.dirname(path)
                
                # ディスク読み込み速度テスト
                print(f"Testing directory: {parent_dir}")
                
                try:
                    # ファイル情報
                    file_size = os.path.getsize(path)
                    
                    # ファイル読み込み速度（RAW）
                    start_time = time.time()
                    with open(path, 'rb') as f:
                        chunk_size = 1024 * 1024  # 1MB chunks
                        total_read = 0
                        while total_read < file_size:
                            chunk = f.read(min(chunk_size, file_size - total_read))
                            if not chunk:
                                break
                            total_read += len(chunk)
                    
                    read_time = time.time() - start_time
                    throughput_mbps = (file_size / (1024*1024)) / read_time
                    
                    print(f"  File size: {file_size/(1024*1024):.1f}MB")
                    print(f"  Raw read time: {read_time:.3f}s")
                    print(f"  Disk throughput: {throughput_mbps:.1f}MB/s")
                    
                    # 性能評価
                    if throughput_mbps < 50:
                        print(f"  🚨 VERY SLOW disk (<50MB/s) - likely HDD or network")
                    elif throughput_mbps < 200:
                        print(f"  ⚠️  SLOW disk (<200MB/s) - consider SSD upgrade")
                    else:
                        print(f"  ✅ GOOD disk speed (>200MB/s)")
                
                except Exception as e:
                    print(f"  ❌ Error testing disk: {e}")
                
                break  # 1つのディスクのみテスト
    
    def generate_optimization_report(self):
        """最適化レポート生成"""
        print(f"\n📋 OPTIMIZATION RECOMMENDATIONS")
        print("="*50)
        
        # ファイルシステム分析
        self.analyze_file_system()
        
        # ビデオロード性能テスト
        video_stats = self.benchmark_single_video_loading(num_videos=5)
        
        # 並列化テスト
        parallel_speedup = self.test_parallel_loading(num_workers=4, num_videos=8)
        
        # ディスク性能チェック
        self.check_disk_performance()
        
        # 総合提案
        print(f"\n🚀 RECOMMENDED OPTIMIZATIONS")
        print("="*50)
        
        recommendations = []
        
        if video_stats['avg_load_time'] > 5:
            recommendations.extend([
                "🔥 CRITICAL: Video loading is extremely slow",
                "   → Move videos to SSD if on HDD",
                "   → Check if videos are on network storage",
                "   → Consider video preprocessing/compression",
                "   → Implement video caching strategy"
            ])
        
        if video_stats['error_rate'] > 0.1:
            recommendations.extend([
                "⚠️  High error rate in video loading",
                "   → Check file corruption",
                "   → Verify file permissions",
                "   → Handle missing files gracefully"
            ])
        
        if parallel_speedup < 1.5:
            recommendations.extend([
                "📊 Poor parallelization suggests I/O bottleneck",
                "   → Upgrade to faster storage (NVMe SSD)",
                "   → Reduce video resolution/length",
                "   → Implement smart caching"
            ])
        
        if not recommendations:
            recommendations.append("✅ Video loading performance seems acceptable")
        
        for rec in recommendations:
            print(rec)
        
        return {
            'video_stats': video_stats,
            'parallel_speedup': parallel_speedup,
            'recommendations': recommendations
        }

def run_diagnosis(csv_path):
    """診断実行"""
    diagnostics = VideoLoadingDiagnostic(csv_path)
    return diagnostics.generate_optimization_report()

if __name__ == "__main__":
    # 使用例
    dataset_path = "/mnt/s/Workfolder/vector_embedding_echo/dataset/datasplit/"
    csv_path = dataset_path + "train_sel_ds.csv"
    
    print("🔍 DIAGNOSING VIDEO LOADING PERFORMANCE")
    print("="*60)
    
    run_diagnosis(csv_path)
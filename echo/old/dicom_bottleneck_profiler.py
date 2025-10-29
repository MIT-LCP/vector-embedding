# dicom_bottleneck_profiler.py - 既存DICOM処理のボトルネック特定

import pydicom as dicom
import time
import numpy as np
import torch
import cv2
import os
from functools import wraps

def profile_function(func_name):
    """関数の実行時間を測定するデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"  {func_name}: {end_time - start_time:.4f}s")
            return result, end_time - start_time
        return wrapper
    return decorator

class DICOMProcessingProfiler:
    """DICOM処理の詳細プロファイリング"""
    
    def __init__(self):
        self.timing_stats = {}
    
    def profile_dicom_processing_steps(self, dicom_path, target_frames=16, target_size=(224, 224)):
        """DICOM処理の各ステップを詳細にプロファイル"""
        
        print(f"\n🔍 PROFILING DICOM PROCESSING: {os.path.basename(dicom_path)}")
        print("-" * 60)
        
        total_start = time.time()
        
        # Step 1: ファイル読み込み
        @profile_function("1. dicom.dcmread()")
        def read_dicom():
            return dicom.dcmread(dicom_path)
        
        dcm, read_time = read_dicom()
        
        # Step 2: pixel_array取得
        @profile_function("2. .pixel_array access")
        def get_pixel_array():
            return dcm.pixel_array
        
        pixel_array, pixel_time = get_pixel_array()
        
        print(f"  └─ Original shape: {pixel_array.shape}")
        print(f"  └─ Data type: {pixel_array.dtype}")
        print(f"  └─ Value range: {pixel_array.min():.1f} - {pixel_array.max():.1f}")
        
        # Step 3: データ型変換
        @profile_function("3. Data type conversion")
        def convert_dtype():
            return pixel_array.astype(np.float32)
        
        pixel_float, convert_time = convert_dtype()
        
        # Step 4: 正規化
        @profile_function("4. Normalization")
        def normalize():
            if pixel_float.max() > 1:
                return (pixel_float - pixel_float.min()) / (pixel_float.max() - pixel_float.min())
            return pixel_float
        
        normalized, norm_time = normalize()
        
        # Step 5: 次元処理（フレーム軸の処理）
        @profile_function("5. Frame dimension handling")
        def handle_frames():
            if normalized.ndim == 2:
                # 2D -> 3D (静止画を動画に)
                return np.stack([normalized] * target_frames, axis=0)
            elif normalized.ndim == 3:
                # 既に3D
                if normalized.shape[0] > target_frames:
                    # フレーム数削減
                    indices = np.linspace(0, normalized.shape[0]-1, target_frames, dtype=int)
                    return normalized[indices]
                elif normalized.shape[0] < target_frames:
                    # フレーム数増加
                    padding = target_frames - normalized.shape[0]
                    last_frame = normalized[-1:]
                    padding_frames = np.repeat(last_frame, padding, axis=0)
                    return np.concatenate([normalized, padding_frames], axis=0)
                else:
                    return normalized
            else:
                return normalized[:target_frames] if normalized.shape[0] >= target_frames else normalized
        
        frames_processed, frame_time = handle_frames()
        print(f"  └─ After frame processing: {frames_processed.shape}")
        
        # Step 6: リサイズ処理
        @profile_function("6. Resize processing")
        def resize_frames():
            resized_frames = []
            for i, frame in enumerate(frames_processed):
                if len(frame.shape) == 2:
                    # グレースケール -> RGB
                    frame_rgb = np.stack([frame] * 3, axis=-1)
                else:
                    frame_rgb = frame
                
                # リサイズ
                frame_resized = cv2.resize(frame_rgb, target_size)
                resized_frames.append(frame_resized)
            
            return np.stack(resized_frames, axis=0)
        
        resized_array, resize_time = resize_frames()
        print(f"  └─ After resize: {resized_array.shape}")
        
        # Step 7: テンソル変換
        @profile_function("7. Tensor conversion")
        def convert_to_tensor():
            # (T, H, W, C) -> (C, T, H, W)
            return torch.from_numpy(resized_array).permute(3, 0, 1, 2).float()
        
        final_tensor, tensor_time = convert_to_tensor()
        print(f"  └─ Final tensor: {final_tensor.shape}")
        
        total_time = time.time() - total_start
        
        # 統計サマリー
        steps_times = {
            'read_dicom': read_time,
            'pixel_array': pixel_time,
            'dtype_convert': convert_time,
            'normalize': norm_time,
            'frame_handling': frame_time,
            'resize': resize_time,
            'tensor_convert': tensor_time
        }
        
        print(f"\n📊 TIMING BREAKDOWN:")
        print(f"  Total time: {total_time:.4f}s")
        print("-" * 40)
        
        for step, step_time in steps_times.items():
            percentage = (step_time / total_time) * 100
            print(f"  {step:15s}: {step_time:.4f}s ({percentage:5.1f}%)")
        
        # ボトルネック特定
        max_time_step = max(steps_times.items(), key=lambda x: x[1])
        print(f"\n🔥 BOTTLENECK: {max_time_step[0]} ({max_time_step[1]:.4f}s)")
        
        return {
            'total_time': total_time,
            'steps': steps_times,
            'bottleneck': max_time_step,
            'final_tensor': final_tensor
        }
    
    def compare_optimization_strategies(self, dicom_path):
        """最適化戦略の比較"""
        
        print(f"\n🚀 OPTIMIZATION STRATEGIES COMPARISON")
        print("=" * 60)
        
        # 戦略1: 現在の方法（詳細プロファイル）
        print(f"\n1️⃣ Current method (detailed profiling):")
        current_result = self.profile_dicom_processing_steps(dicom_path)
        
        # 戦略2: 最適化された正規化
        print(f"\n2️⃣ Optimized normalization:")
        start_time = time.time()
        
        dcm = dicom.dcmread(dicom_path)
        pixel_array = dcm.pixel_array
        
        # 最適化: 一度に正規化
        if pixel_array.dtype != np.float32:
            pixel_array = pixel_array.astype(np.float32)
        
        # Min/Maxを一度だけ計算
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > 1:
            pixel_array = (pixel_array - pmin) / (pmax - pmin)
        
        opt_norm_time = time.time() - start_time
        print(f"  Optimized normalization: {opt_norm_time:.4f}s")
        
        # 戦略3: バッチリサイズ
        print(f"\n3️⃣ Batch resize strategy:")
        start_time = time.time()
        
        # フレーム処理
        if pixel_array.ndim == 2:
            frames = np.stack([pixel_array] * 16, axis=0)
        else:
            frames = pixel_array[:16] if pixel_array.shape[0] >= 16 else pixel_array
        
        # バッチでリサイズ（より効率的）
        if len(frames.shape) == 3:
            # (T, H, W) -> (T, H, W, 3)
            frames_rgb = np.stack([frames] * 3, axis=-1)
        else:
            frames_rgb = frames
        
        # 一度にすべてのフレームをリサイズ
        resized_batch = []
        for frame in frames_rgb:
            resized_batch.append(cv2.resize(frame, (224, 224)))
        
        batch_resize_time = time.time() - start_time
        print(f"  Batch resize: {batch_resize_time:.4f}s")
        
        # 戦略4: メモリ効率的な処理
        print(f"\n4️⃣ Memory efficient processing:")
        start_time = time.time()
        
        dcm = dicom.dcmread(dicom_path)
        
        # インプレース操作でメモリ節約
        pixel_array = dcm.pixel_array
        
        # データ型変換とリサイズを組み合わせ
        if pixel_array.ndim == 2:
            # 直接リサイズ
            resized = cv2.resize(pixel_array.astype(np.float32), (224, 224))
            # 正規化
            if resized.max() > 1:
                resized = (resized - resized.min()) / (resized.max() - resized.min())
            # RGB化とフレーム複製
            frames = np.stack([np.stack([resized] * 3, axis=-1)] * 16, axis=0)
        
        # テンソル変換
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        
        memory_eff_time = time.time() - start_time
        print(f"  Memory efficient: {memory_eff_time:.4f}s")
        
        # 結果比較
        print(f"\n📈 COMPARISON RESULTS:")
        print("-" * 40)
        
        methods = [
            ("Current method", current_result['total_time']),
            ("Optimized normalization", opt_norm_time),
            ("Batch resize", batch_resize_time),
            ("Memory efficient", memory_eff_time)
        ]
        
        best_time = min(time for _, time in methods)
        
        for name, time_val in methods:
            speedup = current_result['total_time'] / time_val
            improvement = "📈" if time_val < current_result['total_time'] else "📉"
            print(f"  {name:20s}: {time_val:.4f}s (×{speedup:.1f}) {improvement}")
        
        return {
            'current': current_result['total_time'],
            'optimized_norm': opt_norm_time,
            'batch_resize': batch_resize_time,
            'memory_efficient': memory_eff_time
        }
    
    def analyze_memory_usage(self, dicom_path):
        """メモリ使用量分析"""
        print(f"\n💾 MEMORY USAGE ANALYSIS")
        print("=" * 40)
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory: {initial_memory:.1f}MB")
        
        # Step 1: DICOM読み込み
        dcm = dicom.dcmread(dicom_path)
        after_read = process.memory_info().rss / 1024 / 1024
        print(f"After dcmread: {after_read:.1f}MB (+{after_read-initial_memory:.1f}MB)")
        
        # Step 2: pixel_array取得
        pixel_array = dcm.pixel_array
        after_pixel = process.memory_info().rss / 1024 / 1024
        print(f"After pixel_array: {after_pixel:.1f}MB (+{after_pixel-after_read:.1f}MB)")
        print(f"  Array size: {pixel_array.nbytes / 1024 / 1024:.1f}MB")
        
        # Step 3: 処理後
        if pixel_array.ndim == 2:
            frames = np.stack([pixel_array.astype(np.float32)] * 16, axis=0)
        
        after_process = process.memory_info().rss / 1024 / 1024
        print(f"After processing: {after_process:.1f}MB (+{after_process-after_pixel:.1f}MB)")
        
        # メモリクリーンアップ
        del dcm, pixel_array
        gc.collect()
        
        after_cleanup = process.memory_info().rss / 1024 / 1024
        print(f"After cleanup: {after_cleanup:.1f}MB ({after_cleanup-initial_memory:+.1f}MB)")

def generate_optimized_dicom_loader():
    """最適化されたDICOMローダーコード生成"""
    
    code = '''
def load_dicom_optimized(dicom_path, target_frames=16, target_size=(224, 224)):
    """最適化されたDICOM読み込み"""
    
    # 1. 高速読み込み
    dcm = dicom.dcmread(dicom_path)
    pixel_array = dcm.pixel_array
    
    # 2. 効率的な型変換と正規化
    if pixel_array.dtype != np.float32:
        pixel_array = pixel_array.astype(np.float32)
    
    # Min/Maxを一度だけ計算
    if pixel_array.max() > 1:
        pmin, pmax = pixel_array.min(), pixel_array.max()
        pixel_array = (pixel_array - pmin) / (pmax - pmin)
    
    # 3. 効率的なフレーム処理
    if pixel_array.ndim == 2:
        # 2D: 直接リサイズしてから複製
        resized_frame = cv2.resize(pixel_array, target_size)
        frames_rgb = np.stack([resized_frame] * 3, axis=-1)  # RGB化
        video_array = np.stack([frames_rgb] * target_frames, axis=0)  # フレーム複製
    
    elif pixel_array.ndim == 3:
        # 3D: フレーム数調整
        if pixel_array.shape[0] != target_frames:
            if pixel_array.shape[0] > target_frames:
                indices = np.linspace(0, pixel_array.shape[0]-1, target_frames, dtype=int)
                pixel_array = pixel_array[indices]
            else:
                # パディング
                padding = target_frames - pixel_array.shape[0]
                last_frames = np.repeat(pixel_array[-1:], padding, axis=0)
                pixel_array = np.concatenate([pixel_array, last_frames], axis=0)
        
        # バッチリサイズ
        resized_frames = []
        for frame in pixel_array:
            frame_rgb = np.stack([frame] * 3, axis=-1) if len(frame.shape) == 2 else frame
            resized_frames.append(cv2.resize(frame_rgb, target_size))
        
        video_array = np.stack(resized_frames, axis=0)
    
    # 4. テンソル変換
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).float()
    
    return video_tensor
'''
    
    return code

def run_comprehensive_dicom_analysis(dicom_path):
    """包括的なDICOM分析"""
    
    profiler = DICOMProcessingProfiler()
    
    print("🔍 COMPREHENSIVE DICOM PROCESSING ANALYSIS")
    print("=" * 70)
    
    # 1. 詳細ステップ分析
    step_analysis = profiler.profile_dicom_processing_steps(dicom_path)
    
    # 2. 最適化戦略比較
    optimization_results = profiler.compare_optimization_strategies(dicom_path)
    
    # 3. メモリ分析
    profiler.analyze_memory_usage(dicom_path)
    
    # 4. 最適化コード生成
    optimized_code = generate_optimized_dicom_loader()
    
    # 5. 推奨事項
    print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS:")
    print("=" * 50)
    
    bottleneck_step = step_analysis['bottleneck'][0]
    bottleneck_time = step_analysis['bottleneck'][1]
    
    recommendations = []
    
    if 'resize' in bottleneck_step and bottleneck_time > 0.01:
        recommendations.append("🔥 Resize is the bottleneck - consider batch processing")
    
    if 'pixel_array' in bottleneck_step:
        recommendations.append("🔥 Pixel array access is slow - DICOM file may be compressed")
    
    if 'normalize' in bottleneck_step:
        recommendations.append("🔥 Normalization is slow - optimize min/max computation")
    
    if step_analysis['total_time'] > 1.0:
        recommendations.append("🚨 Overall processing is very slow - multiple optimizations needed")
    
    # 最大の改善効果を特定
    best_optimization = min(optimization_results.items(), key=lambda x: x[1])
    improvement = optimization_results['current'] / best_optimization[1]
    
    recommendations.append(f"✅ Best optimization: {best_optimization[0]} ({improvement:.1f}x speedup)")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    return {
        'step_analysis': step_analysis,
        'optimization_results': optimization_results,
        'optimized_code': optimized_code,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    # サンプルDICOMファイル
    sample_dicom = "/mnt/s/Workfolder/Physionet/mimic-iv-echo/0.1/p10/p10002221/s94106955/94106955_0001.dcm"
    
    if os.path.exists(sample_dicom):
        results = run_comprehensive_dicom_analysis(sample_dicom)
        
        print(f"\n📝 OPTIMIZED CODE:")
        print("=" * 40)
        print(results['optimized_code'])
        
    else:
        print("❌ Sample DICOM file not found")
        print("Please update the path to your DICOM file")
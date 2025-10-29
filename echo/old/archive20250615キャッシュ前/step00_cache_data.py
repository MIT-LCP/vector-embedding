# =============================================================================
# 最低限並列キャッシュシステム
# =============================================================================

import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path
import torch
import numpy as np
import cv2
import pydicom as dicom
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# 設定
DATASET_PATH = "/mnt/s/Workfolder/vector_embedding_echo/dataset/datasplit/"
CACHE_BASE_DIR = "/mnt/s/Workfolder/vector_embedding_echo/cache"

class HalfCache:
    def __init__(self, cache_subdir):
        self.cache_dir = Path(CACHE_BASE_DIR) / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.csv"
        self.index_df = pd.read_csv(self.index_file) if self.index_file.exists() else pd.DataFrame(columns=['path', 'cache', 'size'])
    
    def _cache_name(self, path):
        parts = Path(path).parts[-3:]
        clean = "_".join(re.sub(r'[^\w\-_]', '_', p) for p in parts)
        hash_short = hashlib.md5(str(path).encode()).hexdigest()[:6]
        return f"{clean[:-4]}_{hash_short}.pt"
    

    def _cache_path(self, path):
        existing = self.index_df[self.index_df['path'] == str(path)]
        cache_name = existing.iloc[0]['cache'] if not existing.empty else self._cache_name(path)
        return self.cache_dir / cache_name
    
    def exists(self, path):
        return self._cache_path(path).exists()
    
    def save(self, path, tensor):
        cache_path = self._cache_path(path)
        torch.save(tensor.half(), cache_path)
        
        cache_name = cache_path.name
        size = cache_path.stat().st_size / (1024**2)
        
        mask = self.index_df['path'] == str(path)
        if mask.any():
            self.index_df.loc[mask, ['cache', 'size']] = [cache_name, size]
        else:
            new_row = pd.DataFrame({'path': [str(path)], 'cache': [cache_name], 'size': [size]})
            self.index_df = pd.concat([self.index_df, new_row], ignore_index=True)
    
    def load(self, path):
        return torch.load(self._cache_path(path)).float()
    
    def save_index(self):
        self.index_df.to_csv(self.index_file, index=False)
    
    def stats(self):
        files = list(self.cache_dir.glob("*.pt"))
        size_gb = sum(f.stat().st_size for f in files) / (1024**3)
        return len(files), size_gb

def process_dicom(path):
    try:
        dcm = dicom.dcmread(path)
        pixels = dcm.pixel_array
        
        if pixels.ndim == 3:
            pixels = np.stack([pixels] * 3, axis=-1)
        
        target_frames, target_size = 16, 224
        if len(pixels) != target_frames:
            if len(pixels) > target_frames:
                indices = np.linspace(0, len(pixels) - 1, target_frames, dtype=int)
                pixels = pixels[indices]
            else:
                pixels = np.tile(pixels, ((target_frames + len(pixels) - 1) // len(pixels), 1, 1, 1))[:target_frames]
        
        resized = np.array([cv2.resize(frame, (target_size, target_size)) for frame in pixels], dtype=np.float32)
        tensor = torch.from_numpy(resized).permute(3, 0, 1, 2) / 255.0
        return tensor.unsqueeze(0).contiguous()
    except:
        return torch.zeros(1, 3, 16, 224, 224, dtype=torch.float32)

def process_batch(batch_args):
    batch_paths, cache_subdir = batch_args
    cache = HalfCache(cache_subdir)
    
    for path in batch_paths:
        if cache.exists(path):
            continue
        if not os.path.exists(path):
            continue
        tensor = process_dicom(path)
        cache.save(path, tensor)
    
    cache.save_index()
    return len(batch_paths)

def create_cache(csv_file, max_files=None):
    # データセット種別判定
    if 'train' in csv_file:
        cache_subdir = 'train'
    elif 'val' in csv_file:
        cache_subdir = 'val'
    elif 'test' in csv_file:
        cache_subdir = 'test'
    else:
        cache_subdir = 'other'
    
    cache = HalfCache(cache_subdir)
    
    df = pd.read_csv(DATASET_PATH + csv_file)
    paths = df['dicom_path'].tolist()[:max_files] if max_files else df['dicom_path'].tolist()
    
    todo = [p for p in paths if not cache.exists(p) and os.path.exists(p)]
    
    if not todo:
        files, size = cache.stats()
        print(f"{cache_subdir}: {files} files, {size:.1f} GB")
        return
    
    # 並列処理
    workers = min(cpu_count()-4, 6)
    batch_size = max(1, len(todo) // workers)
    batches = [todo[i:i+batch_size] for i in range(0, len(todo), batch_size)]
    batch_args = [(batch, cache_subdir) for batch in batches]
    
    print(f"{cache_subdir}: Processing {len(todo)} files ({workers} threads)...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(process_batch, batch_args), total=len(batch_args)))
    
    files, size = cache.stats()
    print(f"{cache_subdir}: Done. {files} files, {size:.1f} GB")

# 実行関数
def sel_datasets():
    for csv in ["test_sel_ds.csv", "val_sel_ds.csv", "train_sel_ds.csv"]:
        print(f"\n{csv}:")
        create_cache(csv)

def all_datasets():
    for csv in ["test_ds.csv", "val_ds.csv", "train_ds.csv"]:
        print(f"\n{csv}:")
        create_cache(csv)
        
def info():
    total_files = 0
    total_size = 0
    
    for subdir in ['train', 'val', 'test']:
        cache_path = Path(CACHE_BASE_DIR) / subdir
        if cache_path.exists():
            cache = HalfCache(subdir)
            files, size = cache.stats()
            if files > 0:
                print(f"{subdir}: {files} files, {size:.1f} GB")
                total_files += files
                total_size += size
    
    if total_files > 0:
        print(f"Total: {total_files} files, {total_size:.1f} GB")

def clear():
    import shutil
    if Path(CACHE_BASE_DIR).exists():
        shutil.rmtree(CACHE_BASE_DIR)
    print("All cache cleared")

def clear_dataset(dataset):
    import shutil
    cache_path = Path(CACHE_BASE_DIR) / dataset
    if cache_path.exists():
        shutil.rmtree(cache_path)
    print(f"{dataset} cache cleared")

# 自動実行
print("Parallel cache system ready")

sel_datasets()
# all_datasets()
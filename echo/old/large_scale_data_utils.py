# fixed_large_scale_data_utils.py - デーモンプロセス問題修正版

import os
import sqlite3
import numpy as np
import torch
import pickle
import lz4.frame
import hashlib
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil
import cv2
import pydicom as dicom
from torch.utils.data import Dataset
import pandas as pd
import random
from collections import defaultdict, deque
import gc
import logging

class FixedDICOMProcessor:
    """修正版DICOM処理器（マルチプロセス問題を回避）"""
    
    def __init__(self, cache_dir, max_memory_cache_gb=8, max_disk_cache_gb=100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # メモリ制限
        self.max_memory_cache = max_memory_cache_gb * 1024**3
        self.max_disk_cache = max_disk_cache_gb * 1024**3
        self.current_memory_usage = 0
        
        # SQLiteベースのメタデータ管理
        self.db_path = self.cache_dir / "metadata.db"
        self.init_database()
        
        # 階層キャッシュ
        self.memory_cache = {}  # Hot data
        self.compressed_cache = {}  # Warm data (compressed)
        self.access_history = deque(maxlen=5000)  # LRU tracking
        self.cache_lock = threading.RLock()
        
        # スレッドプールのみ使用（プロセスプール回避）
        self.io_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dicom_io")
        
        # 統計
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'disk_reads': 0,
            'compressions': 0,
            'decompressions': 0,
            'processing_errors': 0
        }
        
        print(f"🏗️  修正版DICOM処理器初期化")
        print(f"   - メモリキャッシュ上限: {max_memory_cache_gb}GB")
        print(f"   - ディスクキャッシュ上限: {max_disk_cache_gb}GB")
        print(f"   - I/Oスレッド: {self.io_pool._max_workers}")
    
    def init_database(self):
        """メタデータ用SQLite DB初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dicom_metadata (
                    file_path TEXT PRIMARY KEY,
                    cache_key TEXT,
                    file_size INTEGER,
                    last_modified REAL,
                    last_accessed REAL,
                    compressed_size INTEGER,
                    processing_time REAL,
                    access_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON dicom_metadata(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON dicom_metadata(access_count)")
            conn.commit()
    
    def get_cache_key(self, dicom_path, params=None):
        """高速キャッシュキー生成"""
        if params is None:
            params = {'n_frames': 32, 'frame_stride': 2, 'video_size': 224}
        
        try:
            stat = os.stat(dicom_path)
            file_info = f"{dicom_path}_{stat.st_size}_{stat.st_mtime_ns}"
        except OSError:
            file_info = str(dicom_path)
        
        param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        cache_string = f"{file_info}_{param_str}"
        
        return hashlib.blake2b(cache_string.encode(), digest_size=16).hexdigest()
    
    def process_dicom_single(self, dicom_path, params=None):
        """単一DICOM処理（エラー時は例外を上げる）"""
        if params is None:
            params = {'n_frames': 32, 'frame_stride': 2, 'video_size': 224}
        
        cache_key = self.get_cache_key(dicom_path, params)
        
        # キャッシュチェック
        cached_data = self._get_from_cache(cache_key, dicom_path)
        if cached_data is not None:
            self.stats['cache_hits'] += 1
            return cached_data
        
        # 新規処理
        self.stats['cache_misses'] += 1
        start_time = time.time()
        
        # エラー時は例外を上げる（ゼロテンソル返却を削除）
        result = self._process_dicom_optimized(dicom_path, params)
        processing_time = time.time() - start_time
        
        # 非同期でキャッシュ保存
        self.io_pool.submit(self._save_to_cache, cache_key, dicom_path, result, processing_time)
        
        return result
    
    def process_dicom_batch_sequential(self, dicom_paths, params=None):
        """バッチDICOM処理（順次処理版）"""
        results = {}
        
        for path in dicom_paths:
            try:
                result = self.process_dicom_single(path, params)
                results[path] = result
            except Exception as e:
                print(f"⚠️ バッチ処理エラー {path}: {e}")
                if params is None:
                    params = {'n_frames': 32, 'frame_stride': 2, 'video_size': 224}
                results[path] = torch.zeros(3, params['n_frames']//params['frame_stride'], 
                                         params['video_size'], params['video_size'])
        
        return results
    
    def _get_from_cache(self, cache_key, file_path):
        """階層キャッシュから取得"""
        current_time = time.time()
        
        with self.cache_lock:
            # 1. メモリキャッシュ
            if cache_key in self.memory_cache:
                self.access_history.append((cache_key, current_time))
                self._update_access_record_async(file_path)
                return self.memory_cache[cache_key].clone()
            
            # 2. 圧縮メモリキャッシュ
            if cache_key in self.compressed_cache:
                try:
                    compressed_data = self.compressed_cache[cache_key]
                    decompressed = lz4.frame.decompress(compressed_data)
                    tensor_data = pickle.loads(decompressed)
                    
                    # ホットキャッシュに昇格
                    self._manage_memory_cache(tensor_data.nbytes)
                    self.memory_cache[cache_key] = tensor_data.clone()
                    
                    self.stats['decompressions'] += 1
                    return tensor_data
                except Exception as e:
                    print(f"圧縮キャッシュ展開エラー: {e}")
                    del self.compressed_cache[cache_key]
        
        # 3. ディスクキャッシュ
        cache_file = self.cache_dir / f"{cache_key}.lz4"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    compressed_data = f.read()
                
                decompressed = lz4.frame.decompress(compressed_data)
                tensor_data = pickle.loads(decompressed)
                
                with self.cache_lock:
                    # メモリキャッシュに読み込み
                    self._manage_memory_cache(tensor_data.nbytes)
                    self.memory_cache[cache_key] = tensor_data.clone()
                
                self.stats['disk_reads'] += 1
                self._update_access_record_async(file_path)
                return tensor_data
            except Exception as e:
                print(f"ディスクキャッシュ読み込みエラー: {e}")
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def _save_to_cache(self, cache_key, file_path, tensor_data, processing_time=0):
        """階層キャッシュに保存"""
        try:
            with self.cache_lock:
                data_size = tensor_data.nbytes
                
                # メモリキャッシュに保存
                self._manage_memory_cache(data_size)
                self.memory_cache[cache_key] = tensor_data.clone()
                self.current_memory_usage += data_size
            
            # 圧縮してディスクに保存
            pickled_data = pickle.dumps(tensor_data.cpu(), protocol=pickle.HIGHEST_PROTOCOL)
            compressed_data = lz4.frame.compress(pickled_data, compression_level=1)
            
            cache_file = self.cache_dir / f"{cache_key}.lz4"
            with open(cache_file, 'wb') as f:
                f.write(compressed_data)
            
            # メタデータ更新
            self._update_metadata(file_path, cache_key, len(compressed_data), processing_time)
            
            self.stats['compressions'] += 1
            
        except Exception as e:
            print(f"キャッシュ保存エラー: {e}")
    
    def _manage_memory_cache(self, new_data_size):
        """メモリキャッシュ管理（LRU）"""
        while (self.current_memory_usage + new_data_size > self.max_memory_cache 
               and self.memory_cache):
            
            # 最も古いアクセスのアイテムを削除
            if self.access_history:
                # アクセス履歴から削除候補選択
                for cache_key, _ in list(self.access_history)[:50]:
                    if cache_key in self.memory_cache:
                        tensor_data = self.memory_cache.pop(cache_key)
                        self.current_memory_usage -= tensor_data.nbytes
                        
                        # 圧縮キャッシュに降格
                        try:
                            pickled = pickle.dumps(tensor_data.cpu(), protocol=pickle.HIGHEST_PROTOCOL)
                            compressed = lz4.frame.compress(pickled, compression_level=1)
                            self.compressed_cache[cache_key] = compressed
                        except Exception:
                            pass  # 圧縮失敗時は破棄
                        
                        if self.current_memory_usage + new_data_size <= self.max_memory_cache:
                            break
            else:
                # フォールバック: 最初のアイテムを削除
                oldest_key = next(iter(self.memory_cache))
                removed_data = self.memory_cache.pop(oldest_key)
                self.current_memory_usage -= removed_data.nbytes
    
    def _process_dicom_optimized(self, dicom_path, params):
        """最適化されたDICOM処理（フレーム数統一）"""
        n_frames = params['n_frames']
        frame_stride = params['frame_stride'] 
        video_size = params['video_size']
        target_frames = n_frames // frame_stride  # 最終的な目標フレーム数
        
        # 高速DICOM読み込み
        dcm = dicom.dcmread(dicom_path, defer_size="1KB")
        pixels = dcm.pixel_array
        
        if pixels is None or pixels.size == 0:
            raise ValueError(f"DICOMファイルが空です: {dicom_path}")
        
        # 効率的な前処理
        if pixels.ndim == 3:
            pixels = np.repeat(pixels[..., None], 3, axis=3)
        
        # フレーム数制限（メモリ効率）
        if pixels.shape[0] > n_frames * 2:
            step = max(1, pixels.shape[0] // n_frames)
            pixels = pixels[::step][:n_frames]
        
        # 簡略マスキング
        if len(pixels) >= 2:
            try:
                first = cv2.cvtColor(pixels[0].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                last = cv2.cvtColor(pixels[-1].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                diff = cv2.absdiff(first, last)
                _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
                
                # 全フレームにマスク適用
                for i in range(len(pixels)):
                    if pixels[i].ndim == 3:
                        pixels[i] = cv2.bitwise_and(pixels[i].astype(np.uint8), 
                                                  pixels[i].astype(np.uint8), mask=mask)
            except Exception:
                pass  # マスキング失敗時はスキップ
        
        # リサイズ処理
        processed_frames = []
        for frame in pixels:
            resized = cv2.resize(frame, (video_size, video_size), interpolation=cv2.INTER_LINEAR)
            processed_frames.append(resized)
        
        if not processed_frames:
            raise ValueError(f"処理可能なフレームがありません: {dicom_path}")
        
        # テンソル変換
        x = torch.tensor(np.array(processed_frames), dtype=torch.float32).permute(3, 0, 1, 2)
        
        # 正規化（EchoPrime統計）
        mean = torch.tensor([29.110628, 28.076836, 29.096405]).view(3, 1, 1, 1)
        std = torch.tensor([47.989223, 46.456997, 47.20083]).view(3, 1, 1, 1)
        x = (x - mean) / std
        
        # フレーム数を目標数に統一（重要な修正）
        current_frames = x.shape[1]
        
        if current_frames < target_frames:
            # 不足分をパディング
            padding = torch.zeros(3, target_frames - current_frames, video_size, video_size)
            x = torch.cat([x, padding], dim=1)
        elif current_frames > target_frames:
            # ストライドまたは切り取り
            if current_frames >= target_frames * frame_stride:
                x = x[:, ::frame_stride][:, :target_frames]
            else:
                x = x[:, :target_frames]
        
        # 最終確認：必ず target_frames になるようにする
        if x.shape[1] != target_frames:
            if x.shape[1] < target_frames:
                padding = torch.zeros(3, target_frames - x.shape[1], video_size, video_size)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :target_frames]
        
        return x
    
    def _update_metadata(self, file_path, cache_key, compressed_size, processing_time=0):
        """メタデータ更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stat = os.stat(file_path)
                current_time = time.time()
                
                conn.execute("""
                    INSERT OR REPLACE INTO dicom_metadata 
                    (file_path, cache_key, file_size, last_modified, last_accessed, 
                     compressed_size, processing_time, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 
                            COALESCE((SELECT access_count FROM dicom_metadata WHERE file_path = ?), 0) + 1)
                """, (file_path, cache_key, stat.st_size, stat.st_mtime, 
                      current_time, compressed_size, processing_time, file_path))
                conn.commit()
        except Exception as e:
            print(f"メタデータ更新エラー: {e}")
    
    def _update_access_record_async(self, file_path):
        """アクセス記録更新（非同期）"""
        def update_db():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE dicom_metadata 
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE file_path = ?
                    """, (time.time(), file_path))
                    conn.commit()
            except Exception:
                pass
        
        self.io_pool.submit(update_db)
    
    def get_stats(self):
        """統計情報取得"""
        with self.cache_lock:
            memory_usage_mb = self.current_memory_usage / 1e6
            
        try:
            disk_usage_gb = sum(f.stat().st_size for f in self.cache_dir.glob("*.lz4")) / 1e9
        except Exception:
            disk_usage_gb = 0
            
        hit_rate = self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
        
        return {
            'memory_cache_items': len(self.memory_cache),
            'compressed_cache_items': len(self.compressed_cache),
            'memory_usage_mb': memory_usage_mb,
            'disk_usage_gb': disk_usage_gb,
            'cache_hit_rate': hit_rate,
            **self.stats
        }
    
    def clear_cache(self):
        """キャッシュクリア"""
        with self.cache_lock:
            self.memory_cache.clear()
            self.compressed_cache.clear()
            self.current_memory_usage = 0
            self.access_history.clear()
        gc.collect()
    
    def shutdown(self):
        """リソース解放"""
        self.io_pool.shutdown(wait=True)


class SimplifiedEchoDataset(Dataset):
    """簡略化されたEchoデータセット（マルチプロセス問題回避）"""
    
    def __init__(self, csv_path, config, dicom_processor, 
                 max_samples_per_epoch=2000, is_validation=False, seed=42):
        self.csv_path = csv_path
        self.config = config
        self.dicom_processor = dicom_processor
        self.max_samples_per_epoch = max_samples_per_epoch
        self.is_validation = is_validation
        self.seed = seed
        self.epoch = 0
        
        # データ読み込み
        self.df = pd.read_csv(csv_path)
        self.total_samples = len(self.df)
        
        # 保護属性準備
        self.protected_attrs = self._prepare_protected_attributes()
        
        # 初期トリプレット生成
        self.regenerate_triplets()
        
        print(f"📊 簡略化データセット:")
        print(f"   - 総サンプル: {self.total_samples:,}")
        print(f"   - エポック毎最大トリプレット: {max_samples_per_epoch:,}")
        print(f"   - 現在のトリプレット数: {len(self.triplets):,}")
    
    def _prepare_protected_attributes(self):
        """保護属性準備"""
        protected_attrs = {}
        sex_mapping = {'M': 0, 'F': 1}
        race_mapping = {'White': 0, 'Black': 1, 'Hispanic': 2, 'Asian': 2, 'Unknown': 2}
        
        for _, row in self.df.iterrows():
            protected_attrs[row['dicom_path']] = {
                'Sex': sex_mapping.get(row['Sex'], 0),
                'Race': race_mapping.get(row['Race'], 0),
            }
        return protected_attrs
    
    def set_epoch(self, epoch):
        """エポック設定"""
        self.epoch = epoch
        self.regenerate_triplets()
    
    def regenerate_triplets(self):
        """トリプレット再生成"""
        print(f"🔄 エポック{self.epoch}: トリプレット生成中...")
        
        # シード設定
        random.seed(self.seed + self.epoch * 1000)
        np.random.seed(self.seed + self.epoch * 1000)
        
        # サンプル数制限
        if not self.is_validation and len(self.df) > self.max_samples_per_epoch:
            # ランダムサンプリング
            sampled_df = self.df.sample(n=self.max_samples_per_epoch, random_state=self.seed + self.epoch)
        else:
            sampled_df = self.df.copy()
        
        # トリプレット生成
        self.triplets = self._generate_triplets(sampled_df)
        
        print(f"   生成完了: {len(self.triplets)}個のトリプレット")
    
    def _generate_triplets(self, df):
        """効率的なトリプレット生成"""
        triplets = []
        
        # view別にグループ化
        view_groups = df.groupby('view')
        
        for view, view_df in view_groups:
            # subject_id別にグループ化
            subject_groups = view_df.groupby('subject_id')
            
            subject_ids = list(subject_groups.groups.keys())
            if len(subject_ids) < 2:
                continue
            
            # 各被験者のサンプルを取得
            subject_samples = {sid: group.to_dict('records') 
                             for sid, group in subject_groups}
            
            # トリプレット生成
            for subject_id in subject_ids:
                samples = subject_samples[subject_id]
                if len(samples) < 2:
                    continue
                
                # 他の被験者のサンプル
                other_samples = []
                for other_sid in subject_ids:
                    if other_sid != subject_id:
                        other_samples.extend(subject_samples[other_sid])
                
                if not other_samples:
                    continue
                
                # 各サンプルをanchorとして使用
                max_triplets_per_anchor = 1 if self.is_validation else 2
                
                for anchor_sample in samples:
                    pos_candidates = [s for s in samples if s['dicom_path'] != anchor_sample['dicom_path']]
                    if not pos_candidates:
                        continue
                    
                    # トリプレット生成（制限付き）
                    for _ in range(min(max_triplets_per_anchor, len(pos_candidates), len(other_samples))):
                        pos_sample = random.choice(pos_candidates)
                        neg_sample = random.choice(other_samples)
                        
                        triplet = {
                            'anchor_path': anchor_sample['dicom_path'],
                            'positive_path': pos_sample['dicom_path'],
                            'negative_path': neg_sample['dicom_path'],
                        }
                        
                        # Adversarial属性
                        if self.config and self.config.use_adversarial:
                            for attr in self.config.adversarial_attributes:
                                attr_value = self.protected_attrs.get(anchor_sample['dicom_path'], {}).get(attr, 0)
                                triplet[f'{attr}_value'] = attr_value
                        
                        triplets.append(triplet)
        
        random.shuffle(triplets)
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """アイテム取得（エラー時は次のアイテムを試行）"""
        max_retries = 10  # 最大リトライ回数
        
        for retry in range(max_retries):
            try:
                # インデックス調整
                current_idx = (idx + retry) % len(self.triplets)
                triplet = self.triplets[current_idx]
                
                # DICOM処理
                anchor = self.dicom_processor.process_dicom_single(triplet['anchor_path'])
                positive = self.dicom_processor.process_dicom_single(triplet['positive_path'])
                negative = self.dicom_processor.process_dicom_single(triplet['negative_path'])
                
                # ゼロテンソルチェック（処理失敗の検出）
                if (anchor.sum() == 0 or positive.sum() == 0 or negative.sum() == 0):
                    if retry < max_retries - 1:
                        continue  # 次のアイテムを試行
                    else:
                        raise ValueError("有効なデータが見つかりません")
                
                # サンプル作成
                sample = {
                    'anchor': anchor,
                    'positive': positive,
                    'negative': negative
                }
                
                # Adversarial属性
                if self.config and self.config.use_adversarial:
                    for attr in self.config.adversarial_attributes:
                        attr_value = triplet.get(f'{attr}_value', 0)
                        sample[attr] = torch.tensor(attr_value, dtype=torch.long)
                
                return sample
                
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"⚠️ データ読み込みエラー (retry {retry+1}/{max_retries}): {e}")
                    continue  # 次のアイテムを試行
                else:
                    print(f"❌ 最大リトライ回数に達しました。スキップします: {e}")
                    # 最後の手段として次のインデックスを返す
                    return self.__getitem__((idx + max_retries) % len(self.triplets))


def create_fixed_dataloader(csv_path, config, cache_dir, 
                           max_memory_cache_gb=8, max_disk_cache_gb=100,
                           max_samples_per_epoch=2000, batch_size=4, 
                           num_workers=0, is_validation=False, **kwargs):
    """修正版データローダー作成（マルチプロセス問題回避）"""
    
    # DICOM処理器
    dicom_processor = FixedDICOMProcessor(
        cache_dir=cache_dir,
        max_memory_cache_gb=max_memory_cache_gb,
        max_disk_cache_gb=max_disk_cache_gb
    )
    
    # 簡略化データセット
    dataset = SimplifiedEchoDataset(
        csv_path=csv_path,
        config=config,
        dicom_processor=dicom_processor,
        max_samples_per_epoch=max_samples_per_epoch,
        is_validation=is_validation,
        **kwargs
    )
    
    # DataLoader（num_workers=0でマルチプロセス回避）
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,  # 通常は0を推奨
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=not is_validation
    )
    
    return dataloader, dataset, dicom_processor
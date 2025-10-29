import cv2
import pydicom as dicom
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import random
import hashlib
import os
from torch.utils.data import IterableDataset

# ========== 改良版キャッシュクラス ==========
class OptimizedHalfCache:
    """最適化されたキャッシュクラス - メモリキャッシュ併用"""
    
    def __init__(self, cache_dir, memory_cache_size=200):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # メモリキャッシュ（LRU風）
        self.memory_cache = {}
        self.memory_cache_order = []
        self.max_memory_cache = memory_cache_size
        
        print(f"📁 Optimized cache ready: {self.cache_dir}")
        print(f"🧠 Memory cache size: {memory_cache_size} tensors")
    
    def _get_cache_path(self, dicom_path):
        path_hash = hashlib.md5(str(dicom_path).encode()).hexdigest()
        return self.cache_dir / f"{path_hash}_half.pt"
    
    def exists(self, dicom_path):
        # メモリキャッシュ優先チェック
        if dicom_path in self.memory_cache:
            return True
        return self._get_cache_path(dicom_path).exists()
    
    def load(self, dicom_path):
        """メモリキャッシュ優先ロード"""
        # メモリキャッシュから取得
        if dicom_path in self.memory_cache:
            # LRU更新
            self.memory_cache_order.remove(dicom_path)
            self.memory_cache_order.append(dicom_path)
            return self.memory_cache[dicom_path].clone()
        
        # ディスクキャッシュから取得
        cache_path = self._get_cache_path(dicom_path)
        if cache_path.exists():
            half_tensor = torch.load(cache_path, map_location='cpu')
            tensor = half_tensor.float()
            
            # メモリキャッシュに追加
            self._add_to_memory_cache(dicom_path, tensor)
            return tensor.clone()
        
        raise FileNotFoundError(f"Cache not found: {dicom_path}")
    
    def save(self, dicom_path, tensor):
        """ディスクとメモリの両方に保存"""
        # ディスクに保存
        cache_path = self._get_cache_path(dicom_path)
        half_tensor = tensor.half()
        torch.save(half_tensor, cache_path)
        
        # メモリキャッシュに追加
        self._add_to_memory_cache(dicom_path, tensor)
    
    def _add_to_memory_cache(self, dicom_path, tensor):
        """メモリキャッシュに追加（LRU管理）"""
        # 既存のものは削除
        if dicom_path in self.memory_cache:
            self.memory_cache_order.remove(dicom_path)
        
        # 容量チェック・古いものを削除
        while len(self.memory_cache) >= self.max_memory_cache:
            old_path = self.memory_cache_order.pop(0)
            del self.memory_cache[old_path]
        
        # 新しいテンソルを追加
        self.memory_cache[dicom_path] = tensor.clone()
        self.memory_cache_order.append(dicom_path)
    
    def get_memory_stats(self):
        """メモリキャッシュ統計"""
        if not self.memory_cache:
            return {"memory_cached": 0, "memory_size_mb": 0}
        
        total_size = sum(t.element_size() * t.numel() for t in self.memory_cache.values())
        return {
            "memory_cached": len(self.memory_cache),
            "memory_size_mb": total_size / (1024**2)
        }


# ========== 既存関数（変更なし） ==========
def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    """画像のクロップとスケーリング"""
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)
    return img

def downsample_and_crop(testarray):
    # 既存のコードをそのまま使用
    try:
        ##################### CREATE MASK #####################
        # Sum all the frames
        frame_sum = testarray[0] # Start off the frameSum with the first frame<<
        # Convert color profile b/c cv2 messes up colors when it reads it in
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_BGR2GRAY)
        original = frame_sum
        frame_sum = np.where(frame_sum>0,1,0) # make all non-zero values 1
        frames = testarray.shape[0]
        for i in range(frames): # Go through every frame
            frame = testarray[i, :, :, :]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.where(frame > 0, 1, 0) # make all non-zero values 1
            frame_sum = np.add(frame_sum, frame)
        # Dilate
        kernel = np.ones((3,3), np.uint8)
        frame_sum = cv2.dilate(np.uint8(frame_sum), kernel, iterations=10)
        # Make binary
        frame_overlap = np.where(frame_sum>0,1,0)                

        ###### Center and Square both Mask and Video ########        
        # Center image by finding center x of the image
        # Pick first 300 y-values
        center = frame_overlap[0:300, :]
        # compress along y axis
        center = np.mean(center, axis=0)
        try:
            center = np.where(center > 0, 1, 0) # make binary
        except:
            return
        # find index where first goes from 0 to 1 and goes from 1 to 0
        try:
            indexL = np.where(center>0)[0][0]
            indexR = center.shape[0]-np.where(np.flip(center)>0)[0][0]
            center_index = int((indexL + indexR) / 2)
        except:
            return
        # Cut off x on one side so that it's centered on x axis
        left_margin = center_index
        right_margin = center.shape[0] - center_index
        if left_margin > right_margin:
            frame_overlap = frame_overlap[:, (left_margin - right_margin):]
            testarray = testarray[:, :, (left_margin - right_margin):, :]
        else:
            frame_overlap = frame_overlap[: , :(center_index + left_margin)]
            testarray = testarray[:, :, :(center_index + left_margin), :]   

        #Make image square by cutting
        height = frame_overlap.shape[0]
        width = frame_overlap.shape[1]
        #Trim by 1 pixel if a dimension has an odd number of pixels
        if (height % 2) != 0:
            frame_overlap = frame_overlap[0:height - 1, :]
            testarray = testarray[:, 0:height - 1, :, :]
        if (width % 2) != 0:
            frame_overlap = frame_overlap[:, 0:width - 1]
            testarray = testarray[:, :, 0:width - 1, :]
        height = frame_overlap.shape[0]
        width = frame_overlap.shape[1]
        bias = int(abs(height - width) / 2)
        if height > width:
            frame_overlap = frame_overlap[bias:height-bias, :]
            testarray = testarray[:, bias:height-bias, :, :]
        else:
            frame_overlap = frame_overlap[:,bias:width-bias]
            testarray = testarray[:, :, bias:width-bias, :]
        return testarray
    except Exception as e:
        print(f"downsample_and_crop failed: {e}")
        return testarray

def mask_outside_ultrasound(original_pixels: np.array) -> np.array:
    # 既存のコードをそのまま使用
    try:
        testarray=np.copy(original_pixels)
        vid=np.copy(original_pixels)
        ##################### CREATE MASK #####################
        # Sum all the frames
        frame_sum = testarray[0].astype(np.float32)  # Start off the frameSum with the first frame
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_YUV2RGB)
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_RGB2GRAY)
        frame_sum = np.where(frame_sum > 0, 1, 0) # make all non-zero values 1
        frames = testarray.shape[0]
        for i in range(frames): # Go through every frame
            frame = testarray[i, :, :, :].astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.where(frame>0,1,0) # make all non-zero values 1
            frame_sum = np.add(frame_sum,frame)

        # Erode to get rid of the EKG tracing
        kernel = np.ones((3,3), np.uint8)
        frame_sum = cv2.erode(np.uint8(frame_sum), kernel, iterations=10)

        # Make binary
        frame_sum = np.where(frame_sum > 0, 1, 0)

        # Make the difference frame fr difference between 1st and last frame
        # This gets rid of static elements
        frame0 = testarray[0].astype(np.uint8)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_YUV2RGB)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        frame_last = testarray[testarray.shape[0] - 1].astype(np.uint8)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_YUV2RGB)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_RGB2GRAY)
        frame_diff = abs(np.subtract(frame0, frame_last))
        frame_diff = np.where(frame_diff > 0, 1, 0)

        # Ensure the upper left hand corner 20x20 box all 0s.
        # There is a weird dot that appears here some frames on Stanford echoes
        frame_diff[0:20, 0:20] = np.zeros([20, 20])

        # Take the overlap of the sum frame and the difference frame
        frame_overlap = np.add(frame_sum,frame_diff)
        frame_overlap = np.where(frame_overlap > 1, 1, 0)

        # Dilate
        kernel = np.ones((3,3), np.uint8)
        frame_overlap = cv2.dilate(np.uint8(frame_overlap), kernel, iterations=10).astype(np.uint8)

        # Fill everything that's outside the mask sector with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.where(frame_overlap!=100,255,0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(frame_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours[0] has shape (445, 1, 2). 445 coordinates. each coord is 1 row, 2 numbers
        # Find the convex hull
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(frame_overlap, [hull], -1, (255, 0, 0), 3)
        frame_overlap = np.where(frame_overlap > 0, 1, 0).astype(np.uint8) #make all non-0 values 1
        # Fill everything that's outside hull with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.array(np.where(frame_overlap != 100, 255, 0),dtype=bool)
        ################## Create your .avi file and apply mask ##################
        # Store the dimension values

        # Apply the mask to every frame and channel (changing in place)
        for i in range(len(vid)):
            frame = vid[i, :, :, :].astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            frame = cv2.bitwise_and(frame, frame, mask = frame_overlap.astype(np.uint8))
            vid[i,:,:,:]=frame
        return vid
    except Exception as e:
        print("Error masking returned as is.")
        return vid

def prepare_protected_attributes(df):
    protected_attrs = {}
    sex_mapping = {'M': 0, 'F': 1}
    race_mapping = {'White': 0, 'Black': 1, 'Hispanic': 2, 'Asian': 2, 'Unknown': 2}
    
    for _, row in df.iterrows():
        protected_attrs[row['dicom_path']] = {
            'Sex': sex_mapping.get(row['Sex'], 0),
            'Race': race_mapping.get(row['Race'], 0),
        }
    return protected_attrs


# ========== 最適化されたデータセット ==========
class OptimizedEchoTripletDataset(IterableDataset):
    """キャッシュを活用した最適化データセット"""
    
    def __init__(self, csv_path, config=None, is_validation=False, 
                 buffer_size=100, seed=None, cache_dir=None):
        self.csv_path = csv_path
        self.config = config
        self.is_validation = is_validation
        self.buffer_size = buffer_size if not is_validation else 1
        self.seed = seed
        
        # キャッシュの初期化
        if cache_dir and os.path.exists(cache_dir):
            self.cache = OptimizedHalfCache(cache_dir, memory_cache_size=200)
            self.use_cache = True
            print(f"✅ Cache enabled: {cache_dir}")
        else:
            self.cache = None
            self.use_cache = False
            print("⚠️ Cache disabled - processing DICOMs on-the-fly")
        
        # ビデオ処理設定
        self.video_config = {
            'n_frames': 16,
            'res': (224, 224),
            'fps': None,
            'out_fps': None,
            'zoom': 0.1,
            'interpolation': cv2.INTER_CUBIC,
            'sample_period': 1,
            'frame_interpolation': True,
            'apply_mask': True,
            'downsample_and_crop': True,
        }

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
    
    def __iter__(self):
        if self.is_validation:
            yield from self._iterate_validation()
        else:
            yield from self._iterate_training()
    
    def _iterate_validation(self):
        df = pd.read_csv(self.csv_path)
        protected_attrs = prepare_protected_attributes(df)
        
        for idx, anchor_row in df.iterrows():
            sample = self._create_triplet_sample(df, anchor_row, idx, protected_attrs)
            if sample is not None:
                yield sample
    
    def _iterate_training(self):
        buffer = []
        
        for chunk in pd.read_csv(self.csv_path, chunksize=1000):
            chunk = chunk.reset_index(drop=True)
            protected_attrs = prepare_protected_attributes(chunk)
            
            for idx, anchor_row in chunk.iterrows():
                sample = self._create_triplet_sample(chunk, anchor_row, idx, protected_attrs)
                if sample is None:
                    continue
                
                buffer.append(sample)
                
                # バッファシャッフル
                if len(buffer) >= self.buffer_size:
                    yield buffer.pop(random.randint(0, len(buffer) - 1))
        
        # 残りのバッファをランダムに排出
        while buffer:
            yield buffer.pop(random.randint(0, len(buffer) - 1))
    
    def _create_triplet_sample(self, df, anchor_row, anchor_idx, protected_attrs):
        """トリプレットサンプルの作成"""
        # Positive/Negative候補の選択
        pos_candidates = df[(df['subject_id'] == anchor_row['subject_id']) & (df.index != anchor_idx)]
        neg_candidates = df[df['subject_id'] != anchor_row['subject_id']]
        
        if len(pos_candidates) == 0 or len(neg_candidates) == 0:
            return None
        
        # ランダムサンプリング
        seed_state = self.seed if self.is_validation else None
        pos_row = pos_candidates.sample(1, random_state=seed_state).iloc[0]
        neg_row = neg_candidates.sample(1, random_state=seed_state).iloc[0]
        
        # ビデオ読み込み（キャッシュ優先）
        try:
            anchor = self._load_video_cached(anchor_row['dicom_path'])
            positive = self._load_video_cached(pos_row['dicom_path'])
            negative = self._load_video_cached(neg_row['dicom_path'])
        except Exception as e:
            print(f"Video load error: {e}")
            return None
        
        # サンプル作成
        sample = {
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        }
        
        # Adversarial属性の追加
        if self.config and self.config.use_adversarial:
            for attr in self.config.adversarial_attributes:
                attr_value = protected_attrs.get(anchor_row['dicom_path'], {}).get(attr, 0)
                sample[attr] = torch.tensor(attr_value, dtype=torch.long)
        
        return sample
    
    def _load_video_cached(self, video_path):
        """キャッシュ優先でビデオロード"""
        # キャッシュから読み込み試行
        if self.use_cache and self.cache.exists(video_path):
            try:
                return self.cache.load(video_path)
            except Exception as e:
                print(f"⚠️ Cache load failed for {video_path}: {e}")
                # キャッシュ失敗時はフォールバック
        
        # オンザフライ処理（キャッシュなしまたは失敗時）
        return self._load_video_direct(video_path)
    
    def _load_video_direct(self, video_path):
        """直接DICOM処理（既存ロジック）"""
        try:
            # DICOM読み込み
            dcm = dicom.dcmread(video_path)
            pixels = dcm.pixel_array
            
            # 基本的な前処理
            if pixels.ndim == 3:  # [T, H, W] -> [T, H, W, 3]
                pixels = np.stack([pixels] * 3, axis=-1)
            
            # 超音波領域外マスキング（既存と同じ）
            if self.video_config['apply_mask']:
                pixels = mask_outside_ultrasound(pixels)
            
            # ダウンサンプル＆クロッピング（既存と同じ）
            if self.video_config['downsample_and_crop']:
                processed_pixels = downsample_and_crop(pixels)
                if processed_pixels is not None:
                    pixels = processed_pixels
            
            # フレーム数調整
            target_frames = self.video_config['n_frames']
            if len(pixels) >= target_frames:
                indices = np.linspace(0, len(pixels) - 1, target_frames, dtype=int)
                pixels = pixels[indices]
            else:
                # 不足分は繰り返し
                repeat_factor = target_frames // len(pixels) + 1
                pixels = np.tile(pixels, (repeat_factor, 1, 1, 1))[:target_frames]
            
            # リサイズ
            resized_frames = []
            for frame in pixels:
                resized_frame = crop_and_scale(
                    frame, 
                    res=self.video_config['res'],
                    interpolation=self.video_config['interpolation'],
                    zoom=self.video_config['zoom']
                )
                resized_frames.append(resized_frame)
            pixels = np.array(resized_frames, dtype=np.float32)
            
            # PyTorchテンソルに変換 [T, H, W, C] -> [C, T, H, W]
            video_tensor = torch.from_numpy(pixels).permute(3, 0, 1, 2)
            video_tensor = video_tensor / 255.0  # 正規化
            
            return video_tensor.contiguous()
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # ダミーテンソル
            dummy_shape = (3, self.video_config['n_frames'], 
                          self.video_config['res'][1], self.video_config['res'][0])
            return torch.zeros(dummy_shape, dtype=torch.float32)


# ========== Step02用のDICOM処理（変更なし） ==========
def process_dicom(dicom_path):
    """Step02用のDICOM処理（既存と同じパイプライン）"""
    try:
        dcm = dicom.dcmread(dicom_path)
        pixels = dcm.pixel_array
        
        # 基本的な前処理
        if pixels.ndim == 3:
            pixels = np.repeat(pixels[..., None], 3, axis=3)
        
        # 既存の処理パイプライン
        pixels = mask_outside_ultrasound(pixels)
        
        # 簡易前処理
        frames_to_take, frame_stride, video_size = 32, 2, 224
        mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)
        
        x = np.zeros((len(pixels), video_size, video_size, 3))
        for i in range(len(x)):
            x[i] = crop_and_scale(pixels[i])
        
        x = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
        x.sub_(mean).div_(std)

        if x.shape[1] < frames_to_take:
            padding = torch.zeros((3, frames_to_take - x.shape[1], video_size, video_size), dtype=torch.float)
            x = torch.cat((x, padding), dim=1)
            
        start = 0
        stack_of_video = x[:, start:(start + frames_to_take):frame_stride, :, :]
        return stack_of_video.unsqueeze(0)
        
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")
        return torch.zeros(1, 3, 16, 224, 224)


# ========== ヘルパー関数 ==========

def create_dataloader(csv_path, config, is_validation=False, batch_size=8, **kwargs):
    """データローダー作成のヘルパー関数"""
    from torch.utils.data import DataLoader
    
    # OptimizedEchoTripletDataset または EchoTripletDataset を使用
    try:
        dataset = OptimizedEchoTripletDataset(
            csv_path=csv_path,
            config=config,
            is_validation=is_validation,
            **kwargs
        )
    except NameError:
        dataset = EchoTripletDataset(
            csv_path=csv_path,
            config=config,
            is_validation=is_validation,
            **kwargs
        )
    
    # 🚀 最適化された設定
    if is_validation:
        num_workers = 0          # 2→0（検証時はシンプルに）
        prefetch_factor = None
        pin_memory = False
    else:
        num_workers = 1          # 4→1（大幅削減）
        prefetch_factor = 2      # 4→2（半減）
        pin_memory = True
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor
    )

# ========== キャッシュベンチマーク関数 ==========
def benchmark_cache_performance(csv_path, cache_dir, num_samples=50):
    """キャッシュ性能のベンチマーク"""
    import time
    
    print("🔬 キャッシュ性能テスト開始...")
    
    # データ準備
    df = pd.read_csv(csv_path)
    test_paths = df['dicom_path'].tolist()[:num_samples]
    
    cache = OptimizedHalfCache(cache_dir)
    
    # 1. キャッシュからの読み込み速度テスト
    cached_paths = [p for p in test_paths if cache.exists(p)]
    
    if len(cached_paths) < 10:
        print("❌ キャッシュファイルが不足しています（最低10ファイル必要）")
        return
    
    print(f"📁 テスト対象: {len(cached_paths)} キャッシュファイル")
    
    # キャッシュ読み込み速度測定
    start_time = time.time()
    for path in cached_paths:
        tensor = cache.load(path)
    cache_time = time.time() - start_time
    
    print(f"⚡ キャッシュ読み込み: {cache_time:.2f}秒 ({cache_time/len(cached_paths)*1000:.1f}ms/file)")
    
    # メモリキャッシュ統計
    memory_stats = cache.get_memory_stats()
    print(f"🧠 メモリキャッシュ: {memory_stats['memory_cached']}ファイル "
          f"({memory_stats['memory_size_mb']:.1f}MB)")
    
    # 2. 直接処理との比較（数ファイルのみ）
    direct_test_paths = test_paths[:5]  # 5ファイルのみで比較
    
    # 直接処理時間
    dataset_direct = OptimizedEchoTripletDataset(csv_path, cache_dir=None)
    start_time = time.time()
    for path in direct_test_paths:
        tensor = dataset_direct._load_video_direct(path)
    direct_time = time.time() - start_time
    
    print(f"🐌 直接処理: {direct_time:.2f}秒 ({direct_time/len(direct_test_paths)*1000:.1f}ms/file)")
    
    # 速度比較
    if cache_time > 0:
        speedup = (direct_time/len(direct_test_paths)) / (cache_time/len(cached_paths))
        print(f"🚀 高速化倍率: {speedup:.1f}x")
    
    return {
        "cache_time_per_file": cache_time/len(cached_paths),
        "direct_time_per_file": direct_time/len(direct_test_paths),
        "speedup": speedup if cache_time > 0 else 0,
        "memory_stats": memory_stats
    }
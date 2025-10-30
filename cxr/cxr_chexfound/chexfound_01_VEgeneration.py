"""
CheXFound Embedding Extraction Script

This script generates embeddings for chest X-ray DICOM images using CheXFound, 
a chest X-ray foundation model based on DINOv2 architecture. Includes chunked 
saving, auto-resume capability, and memory monitoring for large-scale datasets.

=============================================================================
REFERENCES
=============================================================================

**Primary Citation:**
Yang, Z., Xu, X., Zhang, J., Wang, G., Kalra, M. K., & Yan, P. (2025). Chest 
X-ray Foundation Model with Global and Local Representations Integration. ArXiv. 
https://arxiv.org/abs/2502.05142

https://github.com/RPIDIAL/CheXFound

**Model Architecture:**
It is pretrained in a self-supervised DINO-style teacher-student framework, 
combining masked image modeling and global feature alignment. 
- Backbone: Vision Transformer (ViT-Large)
- Patch size: 16x16
- Image size: 512x512
- Embedding dimension: 1024

**Model Repository:**
- GitHub: https://github.com/RPIDIAL/CheXFound

=============================================================================
MODEL DOWNLOAD & REQUIREMENTS
=============================================================================

**Download Model & Code:**

# Clone CheXFound repository with
git clone https://github.com/RPIDIAL/CheXFound.git

# Download pretrained weights
# Follow instructions in the repository README to download:
- teacher_checkpoint.pth
- config.yaml

**Important Notes:**
- This script requires the CheXFound repository to be cloned and accessible
- Set `chexfound_path` variable to point to your CheXFound directory
- For CPU/MPS (non-CUDA) devices, you may need to modify `eval/setup.py`:
  ```python
  # Inside build_model_for_eval(...) function
  model.eval()
  import torch
  if torch.cuda.is_available():
      model.cuda()
  return model
  ```

=============================================================================
INPUT DATA REQUIREMENTS
=============================================================================

**CSV File Format:**
Required columns in metadata CSV:
1. `dicom_id`: Unique identifier for each DICOM
2. `subject_id`: MIMIC-CXR subject ID (e.g., 10000032)
3. `study_id`: MIMIC-CXR study ID (e.g., 50414267)
4. `ViewPosition`: View position (PA, AP, LATERAL, etc.)

**Example Metadata CSV:**
```csv
dicom_id,subject_id,study_id,ViewPosition
02aa804e-bde0afdd,10000032,50414267,PA
174e3ec5-1bdea825,10000764,53189527,AP
```

**Important:** This script filters by ViewPosition during processing. By default, 
it processes only PA (Posterior-Anterior) and AP (Anterior-Posterior) views. 
Unlike other foundation models, CheXFound is also trained on the lateral view,
so lateral images can be used. However, the script currently generates embeddings 
for AP/PA views to maintain compatibility with other models in the embedding library.

**For MIMIC-CXR Dataset:**
- Metadata source: https://physionet.org/content/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz
- Script automatically filters to PA/AP views
- DICOM paths constructed as: `base_dir/p{subject_id}/s{study_id}/{dicom_id}.dcm`

=============================================================================
USAGE EXAMPLES
=============================================================================

**Extraction modes:**
- `'single_cls'`: CLS token only [1024-dim]
- `'single_pooled'`: Pooled patch tokens [1024-dim]
- `'single_combined'`: Both CLS and pooled patches [RECOMMENDED]
- `'multilayer_full'`: Multi-layer CLS + all patches [4096-dim CLS, full spatial]

# Example command to run the script:
```bash
python chexfound_embedding_extraction.py \
  --config_file /path/to/config.yaml \
  --pretrained_weights /path/to/teacher_checkpoint.pth \
  --base_dir /data/mimic-cxr/ \
  --metadata_path /path/to/metadata/file.csv \
  --output_path ./embeddings/chexfound_embeddings.npz \
  --extraction_mode single_combined \
  --batch_size 64 \
  --save_every 5000 \
  --filter_views PA AP \
  --device cuda \
  --memory_warning_threshold 75
```

=============================================================================
OUTPUT STRUCTURE
=============================================================================

**Output Format:** NPZ files with chunked saving (part000.npz, part001.npz, ...)

**For 'single_combined' mode (RECOMMENDED):**
Each NPZ file contains:
- `filenames`: NumPy array of shape (N,) containing dicom_id values
- `cls_embeddings`: NumPy array of shape (N, 1024)
  - CLS token from last layer
  - dtype: float16 (for storage efficiency)
- `pooled_patch_embeddings`: NumPy array of shape (N, 1024)
  - Mean-pooled patch tokens from last layer
  - dtype: float16 (for storage efficiency)

**For 'single_cls' mode:**
- `filenames`: Array of dicom_id values
- `embeddings`: Shape (N, 1024), dtype: float16

**For 'single_pooled' mode:**
- `filenames`: Array of dicom_id values
- `embeddings`: Shape (N, 1024), dtype: float16

**For 'multilayer_full' mode:**
- `filenames`: Array of dicom_id values
- `cls_multilayer`: Object array of (N,), each element shape (1, 4096)
- `patches_multilayer`: Object array of (N,), each element shape (1024, 4096)

**Features:**
1. **Chunked Saving**: Saves every N images (default 10,000) to prevent memory issues
2. **Auto-Resume**: Automatically resumes from last saved chunk if interrupted
3. **Memory Monitoring**: Tracks RAM usage and warns when exceeding threshold
4. **Failed Files Tracking**: Saves list of files that failed processing

**Progress Tracking:**
- Progress state saved in `*_progress.json`
- Automatically resumes from last checkpoint
- Failed files saved to `*_failed.txt`

=============================================================================
"""


import os
import sys
import numpy as np
import torch
import pydicom
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm
import gzip
import argparse
import psutil
import json
from datetime import datetime


# =============================================================================
# CHEXFOUND REPOSITORY PATH CONFIGURATION
# =============================================================================

# Update this path to point to your local CheXFound repository
chexfound_path = "/Volumes/code/my_project/mimiccxr_chexfound/CheXFound"
sys.path.append(chexfound_path)

from chexfound.eval.setup import setup_and_build_model
from chexfound.data.transforms import make_classification_eval_transform


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """
    Parse command line arguments for flexible script execution.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate CheXFound embeddings with chunked saving and auto-resume',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required Arguments
    parser.add_argument('--config_file', type=str, required=True,
                       help='Path to CheXFound config.yaml file')
    parser.add_argument('--pretrained_weights', type=str, required=True,
                       help='Path to CheXFound teacher_checkpoint.pth file')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing DICOM files')
    parser.add_argument('--metadata_path', type=str, required=True,
                       help='Path to metadata CSV file (can be gzipped)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for embeddings (will create chunk files)')
    
    # Model Configuration
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size (default: 512)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'],
                       help='Device for inference (default: auto-select)')
    
    # Extraction Configuration
    parser.add_argument('--extraction_mode', type=str, default='single_combined',
                       choices=['single_cls', 'single_pooled', 'single_combined', 'multilayer_full'],
                       help='Embedding extraction mode (default: single_combined)')
    parser.add_argument('--filter_views', type=str, nargs='+', default=['PA', 'AP'],
                       help='ViewPosition values to include (default: PA AP)')
    
    # Processing Parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Number of images to process at once (default: 32)')
    parser.add_argument('--save_every', type=int, default=10000,
                       help='Save intermediate chunks every N images (default: 10000)')
    
    # Execution Options
    parser.add_argument('--auto_resume', action='store_true', default=True,
                       help='Resume from previous run (default: True)')
    parser.add_argument('--no_auto_resume', dest='auto_resume', action='store_false',
                       help='Disable auto-resume feature')
    parser.add_argument('--memory_warning_threshold', type=int, default=80,
                       help='Warn when RAM usage exceeds this percentage (default: 80)')
    
    return parser.parse_args()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _parse_cls_and_patches_from_features(features):
    """
    Parse CLS and patch tokens from CheXFound model output.
    
    CheXFound returns features in different formats depending on version:
    - Tuple format: (patches, cls) where patches [B, T, D], cls [B, D]
    - Dict format: {'x_norm_clstoken', 'x_norm_patchtokens'}
    - Tensor format: [B, T+1, D] (DINOv2-like)
    
    Returns:
        tuple: (cls [B, D], patches [B, T, D])
    """
    f0 = features[0]
    
    if isinstance(f0, dict):
        cls = f0['x_norm_clstoken'].squeeze(1)
        patches = f0['x_norm_patchtokens']
        return cls, patches
    
    elif isinstance(f0, tuple) and len(f0) == 2:
        patch_tokens, cls_token = f0
        if torch.is_tensor(cls_token) and torch.is_tensor(patch_tokens):
            return cls_token, patch_tokens
    
    elif torch.is_tensor(f0):
        cls = f0[:, 0, :]
        patches = f0[:, 1:, :]
        return cls, patches
    
    raise TypeError(f"Unsupported feature type: {type(f0)}")


# =============================================================================
# CHEXFOUND EMBEDDER CLASS
# =============================================================================

class CheXFound_Embedder:
    """
    CheXFound embedding extractor for chest X-ray DICOM images.
    
    Provides interface for extracting embeddings from CheXFound model with:
    - Built-in DICOM preprocessing
    - Chunked saving for large datasets
    - Auto-resume capability
    - Memory monitoring
    """
    
    def __init__(self, config_file, pretrained_weights, image_size=512, device=None):
        """
        Initialize CheXFound model for embedding extraction.
        
        Args:
            config_file: Path to config.yaml
            pretrained_weights: Path to teacher_checkpoint.pth
            image_size: Input image size (default: 512)
            device: Device for inference ('cuda', 'mps', or 'cpu')
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")

        # Patch distributed init for non-CUDA devices
        if not torch.cuda.is_available():
            import chexfound.distributed as dist
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            
            if hasattr(dist, "enable"):
                dist.enable = lambda *a, **k: None
            if hasattr(dist, "is_enabled"):
                dist.is_enabled = lambda: False
            if hasattr(dist, "get_global_rank"):
                dist.get_global_rank = lambda: 0
            if hasattr(dist, "get_world_size"):
                dist.get_world_size = lambda: 1
        
        # Setup argparse for CheXFound
        parser = argparse.ArgumentParser()
        parser.set_defaults(
            config_file=config_file,
            pretrained_weights=None,
            output_dir='./temp',
            opts=[],
            image_size=image_size,
            patch_size=16,
            n_register_tokens=4,
        )
        args, _ = parser.parse_known_args()
        
        # Load model
        print("Loading CheXFound model...")
        self.model, self.autocast_dtype = setup_and_build_model(args)
        
        # Load pretrained weights
        print(f"Loading weights from {pretrained_weights}...")
        state_dict = torch.load(pretrained_weights, map_location='cpu')['teacher']
        
        # Clean state dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone'):
                ls = k.split('.')
                if 'blocks' in k:
                    new_k = '.'.join([ls[1], *ls[3:]])
                else:
                    new_k = '.'.join(ls[1:])
            else:
                new_k = k
            new_state_dict.update({new_k: v})
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        print("CheXFound model loaded successfully!")
        print(f"Model embedding dimension: 1024 (ViT-Large)")
        
        # Setup transform
        self.transform = make_classification_eval_transform(
            resize_size=image_size, 
            crop_size=image_size
        )
    
    def dicom_to_image(self, dicom_path):
        """
        Convert DICOM file to PIL Image with clinical preprocessing.
        
        Preprocessing steps:
        1. Apply rescale slope/intercept if present
        2. Handle MONOCHROME1 photometric interpretation (invert)
        3. Percentile-based intensity clipping (0.5% - 99.5%)
        4. Normalize to [0, 1] and convert to 8-bit
        5. Convert to RGB
        
        Args:
            dicom_path: Full path to DICOM file
            
        Returns:
            PIL.Image: RGB image ready for model inference, or None if failed
        """
        try:
            dcm = pydicom.dcmread(dicom_path, force=True)
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            
            if hasattr(dcm, 'PhotometricInterpretation'):
                if dcm.PhotometricInterpretation == 'MONOCHROME1':
                    pixel_array = np.max(pixel_array) - pixel_array
            
            pixel_array = np.nan_to_num(pixel_array, nan=0.0, posinf=255.0, neginf=0.0)
            lower = np.percentile(pixel_array, 0.5)
            upper = np.percentile(pixel_array, 99.5)
            pixel_array = np.clip(pixel_array, lower, upper)
            
            if pixel_array.max() > pixel_array.min():
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
                pixel_array = (pixel_array * 255).astype(np.uint8)
            else:
                pixel_array = np.zeros_like(pixel_array).astype(np.uint8)
            
            img = Image.fromarray(pixel_array).convert('RGB')
            return img
            
        except Exception as e:
            print(f"Error processing DICOM {dicom_path}: {e}")
            return None
    
    def extract_batch_single_layer_cls(self, img_tensors):
        """Extract single-layer CLS tokens for a batch."""
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(img_tensors, n=1, return_class_token=True)
            cls, _ = _parse_cls_and_patches_from_features(feats)
            return cls.detach().to('cpu').numpy()
    
    def extract_batch_single_layer_pooled_patches(self, img_tensors):
        """Extract single-layer pooled patches for a batch."""
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(img_tensors, n=1, return_class_token=True)
            _, patches = _parse_cls_and_patches_from_features(feats)
            pooled = patches.mean(dim=1)
            return pooled.detach().to('cpu').numpy()
    
    def extract_batch_single_layer_combined(self, img_tensors):
        """Extract both single-layer CLS and pooled patches for a batch."""
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(img_tensors, n=1, return_class_token=True)
            cls, patches = _parse_cls_and_patches_from_features(feats)
            pooled = patches.mean(dim=1)
            return {
                'cls': cls.detach().to('cpu').numpy(),
                'pooled_patches': pooled.detach().to('cpu').numpy()
            }
        
    def extract_batch_multilayer_full(self, img_tensors, n_layers=4):
        """Extract multi-layer CLS and full patches for a batch."""
        with torch.no_grad():
            features = self.model.get_intermediate_layers(
                img_tensors, n=n_layers, return_class_token=True
            )
            
            cls_list = [f['x_norm_clstoken'] for f in features]
            cls_multilayer = torch.cat(cls_list, dim=-1)
            
            patches_list = [f['x_norm_patchtokens'] for f in features]
            patches_multilayer = torch.cat(patches_list, dim=-1)
            
            return {
                'cls_multilayer': cls_multilayer.cpu().numpy(),
                'patches_multilayer': patches_multilayer.cpu().numpy()
            }
    
    def _load_progress_state(self, progress_file):
        """Load progress state from JSON file for auto-resume."""
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                state = json.load(f)
            print(f"📂 Resuming from previous run: {state['last_chunk_saved']} chunks saved, {state['images_processed']} images processed")
            return state
        return None
    
    def _save_progress_state(self, progress_file, file_counter, images_processed, processed_ids):
        """Save progress state to JSON file for auto-resume."""
        state = {
            'last_chunk_saved': file_counter,
            'images_processed': images_processed,
            'processed_ids': list(processed_ids),
            'timestamp': datetime.now().isoformat()
        }
        with open(progress_file, 'w') as f:
            json.dump(state, f)
    
    def _get_memory_info(self):
        """Get current memory usage info."""
        mem = psutil.virtual_memory()
        return {
            'percent': mem.percent,
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'total_gb': mem.total / (1024**3)
        }
    
    def process_filtered_dicoms(self, base_dir, metadata_path, output_path, 
                                filter_views=('PA', 'AP'), extraction_mode='single_combined',
                                batch_size=32, save_every=10000, auto_resume=True,
                                memory_warning_threshold=80):
        """
        Process DICOM files with chunked saving, auto-resume, and memory monitoring.
        
        Processing pipeline:
        1. Load metadata and filter by ViewPosition
        2. Initialize or load progress state
        3. Process images in batches
        4. Extract embeddings using specified mode
        5. Save chunks periodically
        6. Track failed files
        
        Args:
            base_dir: Base directory containing DICOM files
            metadata_path: Path to metadata CSV file (can be gzipped)
            output_path: Path to save embeddings (creates multiple chunk files)
            filter_views: Tuple of ViewPosition values to include
            extraction_mode: 'single_cls', 'single_pooled', 'single_combined', or 'multilayer_full'
            batch_size: Number of images to process at once
            save_every: Save intermediate chunks every N images
            auto_resume: If True, resume from last saved chunk
            memory_warning_threshold: Warn when RAM usage exceeds this percentage
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Setup progress tracking
        base_name = os.path.splitext(output_path)[0]
        progress_file = f"{base_name}_progress.json"
        
        processed_ids_set = set()
        file_counter = 0
        images_processed = 0
        
        if auto_resume:
            state = self._load_progress_state(progress_file)
            if state:
                processed_ids_set = set(state['processed_ids'])
                file_counter = state['last_chunk_saved'] + 1
                images_processed = state['images_processed']
        
        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        if metadata_path.endswith('.gz'):
            with gzip.open(metadata_path, 'rt') as f:
                metadata_df = pd.read_csv(f)
        else:
            metadata_df = pd.read_csv(metadata_path)
        
        # Filter by ViewPosition
        filtered_df = metadata_df[metadata_df['ViewPosition'].isin(filter_views)]
        
        if auto_resume and processed_ids_set:
            filtered_df = filtered_df[~filtered_df['dicom_id'].isin(processed_ids_set)]
            print(f"🔄 Auto-resume: Skipping {len(processed_ids_set)} already processed images")
        
        print(f"Found {len(filtered_df)} DICOM files to process with ViewPosition in {filter_views}")
        print(f"Extraction mode: {extraction_mode}")
        print(f"Batch size: {batch_size}, Save every: {save_every} images")
        
        mem_info = self._get_memory_info()
        print(f"💾 Initial RAM: {mem_info['percent']:.1f}% ({mem_info['used_gb']:.2f}/{mem_info['total_gb']:.2f} GB)")
        
        # Initialize collections
        filenames = []
        failed_files = []
        
        if extraction_mode == 'single_cls':
            all_embeddings = []
        elif extraction_mode == 'single_pooled':
            all_embeddings = []
        elif extraction_mode == 'single_combined':
            all_cls_embeddings = []
            all_pooled_embeddings = []
        elif extraction_mode == 'multilayer_full':
            all_cls_multilayer = []
            all_patches_multilayer = []
        else:
            raise ValueError(f"Invalid extraction_mode: {extraction_mode}")
        
        # Process in batches
        batch_images = []
        batch_ids = []
        batch_paths = []
        
        for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), 
                            desc=f"Extracting {extraction_mode} embeddings"):
            dicom_id = row['dicom_id']
            subject_id = row['subject_id'] if 'subject_id' in row.index else ""
            study_id = row['study_id'] if 'study_id' in row.index else ""
            
            if not subject_id or not study_id:
                continue
            
            dicom_path = os.path.join(base_dir, f"p{subject_id}", f"s{study_id}", f"{dicom_id}.dcm")
            
            if not os.path.exists(dicom_path):
                failed_files.append(dicom_path)
                continue
            
            img = self.dicom_to_image(dicom_path)
            if img is None:
                failed_files.append(dicom_path)
                continue
            
            try:
                img_tensor = self.transform(img)
                batch_images.append(img_tensor)
                batch_ids.append(dicom_id)
                batch_paths.append(dicom_path)
            except Exception as e:
                print(f"Error transforming {dicom_path}: {e}")
                failed_files.append(dicom_path)
                continue
            
            # Process batch when full
            if len(batch_images) >= batch_size:
                self._process_batch(
                    batch_images, batch_ids, batch_paths, extraction_mode, filenames, failed_files,
                    all_embeddings if extraction_mode in ['single_cls', 'single_pooled'] else None,
                    all_cls_embeddings if extraction_mode == 'single_combined' else None,
                    all_pooled_embeddings if extraction_mode == 'single_combined' else None,
                    all_cls_multilayer if extraction_mode == 'multilayer_full' else None,
                    all_patches_multilayer if extraction_mode == 'multilayer_full' else None,
                    memory_warning_threshold
                )
                
                images_processed += len(batch_ids)
                
                # Save intermediate chunks
                if len(filenames) >= save_every:
                    if extraction_mode == 'single_cls':
                        self._save_single_chunk(output_path, file_counter, filenames, all_embeddings, "CLS", 1024)
                        all_embeddings = []
                    elif extraction_mode == 'single_pooled':
                        self._save_single_chunk(output_path, file_counter, filenames, all_embeddings, "pooled patch", 1024)
                        all_embeddings = []
                    elif extraction_mode == 'single_combined':
                        self._save_combined_chunk(output_path, file_counter, filenames, all_cls_embeddings, all_pooled_embeddings)
                        all_cls_embeddings = []
                        all_pooled_embeddings = []
                    elif extraction_mode == 'multilayer_full':
                        self._save_multilayer_chunk(output_path, file_counter, filenames, all_cls_multilayer, all_patches_multilayer)
                        all_cls_multilayer = []
                        all_patches_multilayer = []
                    
                    processed_ids_set.update(filenames)
                    self._save_progress_state(progress_file, file_counter, images_processed, processed_ids_set)
                    
                    file_counter += 1
                    filenames = []
                    
                    mem_info = self._get_memory_info()
                    print(f"💾 RAM after save: {mem_info['percent']:.1f}% ({mem_info['used_gb']:.2f}/{mem_info['total_gb']:.2f} GB)")
                
                batch_images = []
                batch_ids = []
                batch_paths = []
        
        # Process remaining batch
        if len(batch_images) > 0:
            self._process_batch(
                batch_images, batch_ids, batch_paths, extraction_mode, filenames, failed_files,
                all_embeddings if extraction_mode in ['single_cls', 'single_pooled'] else None,
                all_cls_embeddings if extraction_mode == 'single_combined' else None,
                all_pooled_embeddings if extraction_mode == 'single_combined' else None,
                all_cls_multilayer if extraction_mode == 'multilayer_full' else None,
                all_patches_multilayer if extraction_mode == 'multilayer_full' else None,
                memory_warning_threshold
            )
            images_processed += len(batch_ids)
        
        # Save final chunk
        if len(filenames) > 0:
            if extraction_mode == 'single_cls':
                self._save_single_chunk(output_path, file_counter, filenames, all_embeddings, "CLS", 1024)
            elif extraction_mode == 'single_pooled':
                self._save_single_chunk(output_path, file_counter, filenames, all_embeddings, "pooled patch", 1024)
            elif extraction_mode == 'single_combined':
                self._save_combined_chunk(output_path, file_counter, filenames, all_cls_embeddings, all_pooled_embeddings)
            elif extraction_mode == 'multilayer_full':
                self._save_multilayer_chunk(output_path, file_counter, filenames, all_cls_multilayer, all_patches_multilayer)
            
            processed_ids_set.update(filenames)
            self._save_progress_state(progress_file, file_counter, images_processed, processed_ids_set)
            
            print(f"\n✅ Created {file_counter + 1} chunk files")
        
        # Save failed files list
        if failed_files:
            failed_path = os.path.splitext(output_path)[0] + '_failed.txt'
            with open(failed_path, 'w') as f:
                for file in failed_files:
                    f.write(f"{file}\n")
            print(f"⚠️  Saved list of {len(failed_files)} failed files to {failed_path}")
        
        # Clean up progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("🧹 Removed progress file (extraction complete)")
        
        print("\n✅ Extraction complete!")
        mem_info = self._get_memory_info()
        print(f"💾 Final RAM: {mem_info['percent']:.1f}% ({mem_info['used_gb']:.2f}/{mem_info['total_gb']:.2f} GB)")
    
    def _process_batch(self, batch_images, batch_ids, batch_paths, extraction_mode, 
                      filenames, failed_files, all_embeddings, all_cls_embeddings, 
                      all_pooled_embeddings, all_cls_multilayer, all_patches_multilayer, 
                      memory_warning_threshold=80):
        """Process a batch of images with aggressive memory management."""
        import gc
        embeddings = None
        batch_tensor = None
        
        try:
            batch_tensor = torch.stack(batch_images).to(self.device, non_blocking=True)

            if extraction_mode == 'single_cls':
                embeddings = self.extract_batch_single_layer_cls(batch_tensor)
                for i, emb in enumerate(embeddings):
                    all_embeddings.append(emb)
                    filenames.append(batch_ids[i])

            elif extraction_mode == 'single_pooled':
                embeddings = self.extract_batch_single_layer_pooled_patches(batch_tensor)
                for i, emb in enumerate(embeddings):
                    all_embeddings.append(emb)
                    filenames.append(batch_ids[i])

            elif extraction_mode == 'single_combined':
                embeddings = self.extract_batch_single_layer_combined(batch_tensor)
                cls_batch = embeddings['cls']
                pooled_batch = embeddings['pooled_patches']
                for i in range(len(batch_ids)):
                    all_cls_embeddings.append(cls_batch[i])
                    all_pooled_embeddings.append(pooled_batch[i])
                    filenames.append(batch_ids[i])

            elif extraction_mode == 'multilayer_full':
                embeddings = self.extract_batch_multilayer_full(batch_tensor)
                cls_ml = embeddings['cls_multilayer']
                patches_ml = embeddings['patches_multilayer']
                for i in range(len(batch_ids)):
                    all_cls_multilayer.append(cls_ml[i])
                    all_patches_multilayer.append(patches_ml[i])
                    filenames.append(batch_ids[i])

        except Exception as e:
            print(f"\n❌ Error processing batch: {e}")
            failed_files.extend(batch_paths)

        finally:
            # Aggressive memory cleanup
            if batch_tensor is not None:
                del batch_tensor
            if embeddings is not None:
                if isinstance(embeddings, dict):
                    for k in list(embeddings.keys()):
                        del embeddings[k]
                del embeddings
            
            batch_images.clear()
            batch_ids.clear()
            batch_paths.clear()
            
            gc.collect()
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            mem_after = self._get_memory_info()
            if mem_after['percent'] > memory_warning_threshold:
                print(f"\n⚠️  HIGH MEMORY WARNING: {mem_after['percent']:.1f}% RAM used ({mem_after['used_gb']:.2f} GB)")
    
    def _save_single_chunk(self, base_output_path, file_counter, filenames, embeddings, emb_type, dim):
        """Save a chunk of single embeddings."""
        base, ext = os.path.splitext(base_output_path)
        output_path = f"{base}_part{file_counter:03d}{ext}"
        
        arr = np.stack(embeddings).astype('float16')
        np.savez(output_path, filenames=np.array(filenames), embeddings=arr)
        print(f"\n💾 Saved chunk {file_counter}: {len(filenames)} {emb_type} embeddings")
        print(f"   File: {os.path.basename(output_path)} ({os.path.getsize(output_path)/(1024**2):.1f} MB)")
    
    def _save_combined_chunk(self, base_output_path, file_counter, filenames, cls_embeddings, pooled_embeddings):
        """Save a chunk of combined embeddings."""
        base, ext = os.path.splitext(base_output_path)
        output_path = f"{base}_part{file_counter:03d}{ext}"
        
        cls_arr = np.stack(cls_embeddings).astype('float16')
        pooled_arr = np.stack(pooled_embeddings).astype('float16')
        np.savez(output_path, filenames=np.array(filenames), 
                cls_embeddings=cls_arr, pooled_patch_embeddings=pooled_arr)
        print(f"\n💾 Saved chunk {file_counter}: {len(filenames)} combined embeddings")
        print(f"   File: {os.path.basename(output_path)} ({os.path.getsize(output_path)/(1024**2):.1f} MB)")
    
    def _save_multilayer_chunk(self, base_output_path, file_counter, filenames, 
                               cls_multilayer, patches_multilayer):
        """Save a chunk of multilayer embeddings."""
        base, ext = os.path.splitext(base_output_path)
        output_path = f"{base}_part{file_counter:03d}{ext}"
        
        cls_array = np.empty(len(filenames), dtype=object)
        patches_array = np.empty(len(filenames), dtype=object)
        for i in range(len(filenames)):
            cls_array[i] = cls_multilayer[i]
            patches_array[i] = patches_multilayer[i]
        
        np.savez(output_path, filenames=np.array(filenames),
                cls_multilayer=cls_array, patches_multilayer=patches_array)
        print(f"\n💾 Saved chunk {file_counter}: {len(filenames)} multilayer embeddings")
        print(f"   File: {os.path.basename(output_path)} ({os.path.getsize(output_path)/(1024**2):.1f} MB)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main script execution with command-line argument parsing.
    
    Example usage:
    
    Basic execution:
    python chexfound_embedding_extraction.py \
      --config_file /path/to/config.yaml \
      --pretrained_weights /path/to/teacher_checkpoint.pth \
      --base_dir /data/mimic-cxr/ \
      --metadata_path /data/metadata.csv.gz \
      --output_path ./embeddings/chexfound_embeddings.npz
    
    With custom settings:
    python chexfound_embedding_extraction.py \
      --config_file /path/to/config.yaml \
      --pretrained_weights /path/to/teacher_checkpoint.pth \
      --base_dir /data/mimic-cxr/ \
      --metadata_path /data/metadata.csv.gz \
      --output_path ./embeddings/chexfound_embeddings.npz \
      --extraction_mode single_combined \
      --batch_size 64 \
      --save_every 5000 \
      --filter_views PA AP \
      --device cuda
    """
    
    # Parse command line arguments
    args = parse_args()
    
    print("\n" + "="*80)
    print("CheXFound Embedding Extraction")
    print("="*80)
    print(f"Config file: {args.config_file}")
    print(f"Pretrained weights: {args.pretrained_weights}")
    print(f"Base directory: {args.base_dir}")
    print(f"Metadata: {args.metadata_path}")
    print(f"Output path: {args.output_path}")
    print(f"Extraction mode: {args.extraction_mode}")
    print(f"Filter views: {args.filter_views}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save every: {args.save_every} images")
    print(f"Auto-resume: {args.auto_resume}")
    print("="*80 + "\n")
    
    # Initialize embedder
    embedder = CheXFound_Embedder(
        config_file=args.config_file,
        pretrained_weights=args.pretrained_weights,
        image_size=args.image_size,
        device=args.device
    )
    
    # Extract embeddings
    embedder.process_filtered_dicoms(
        base_dir=args.base_dir,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        filter_views=tuple(args.filter_views),
        extraction_mode=args.extraction_mode,
        batch_size=args.batch_size,
        save_every=args.save_every,
        auto_resume=args.auto_resume,
        memory_warning_threshold=args.memory_warning_threshold
    )
    
    print("\n" + "="*80)
    print("All extractions complete!")
    print("="*80)
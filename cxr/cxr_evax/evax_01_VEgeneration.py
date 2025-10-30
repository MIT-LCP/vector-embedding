"""
EVA-X Embedding Extraction Script

This script generates embeddings for chest X-ray DICOM images using EVA-X,
a foundation model for general chest X-ray analysis with self-supervised learning.
Includes chunked saving, auto-resume capability, and memory monitoring for 
large-scale datasets.

=============================================================================
REFERENCES
=============================================================================

**Primary Citation:**
Yao, J., Wang, X., Song, Y., Zhao, H., Ma, J., Chen, Y., Liu, W., & Wang, B. (2024).
EVA-X: A foundation model for general chest X-ray analysis with self-supervised learning.
https://github.com/hustvl/EVA-X

**Model Architecture:**
- Self-supervised learning framework capturing semantic and geometric information
- Backbone: Vision Transformer (ViT)
- Patch size: 16x16
- Image size: 224x224
- Embedding dimensions: 
  * Tiny: 192-dim
  * Small: 384-dim
  * Base: 768-dim

**Model Repository:**
- GitHub: https://github.com/hustvl/EVA-X
- Huggingface: https://huggingface.co/MapleF/eva_x

=============================================================================
MODEL DOWNLOAD & REQUIREMENTS
=============================================================================

**Download Model & Code:**

# Clone EVA-X repository
git clone https://github.com/hustvl/EVA-X.git

# Download pretrained weights from Huggingface
- eva_x_base_patch16_merged520k_mim.pt (Current script tested with this model)
- eva_x_tiny_patch16_merged520k_mim.pt
- eva_x_small_patch16_merged520k_mim.pt

** Environment Requirements:**
- github repository provides requirements.txt for EVA-X dependencies

**Important Notes:**
- This script requires the EVA-X repository to be cloned and accessible
- Set `eva_x_path` variable to point to your EVA-X directory
- Script includes automatic patching for EVA_X class position embedding compatibility

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

**For MIMIC-CXR Dataset:**
- Metadata source: https://physionet.org/content/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz
- Script automatically filters to PA/AP views by default
- DICOM paths constructed as: `base_dir/p{subject_id}/s{study_id}/{dicom_id}.dcm`

=============================================================================
USAGE EXAMPLES
=============================================================================

**Extraction modes:**
- `'single_cls'`: CLS token only [192/384/768-dim depending on model size]
- `'single_pooled'`: Global Average Pooled patch tokens [768-dim for base model]
- `'single_combined'`: Both CLS and pooled patches [RECOMMENDED]

**Example command:**
```bash
python evax_embedding_extraction.py \
  --model_size base \
  --pretrained_weights /path/to/eva_x_base_patch16_merged520k_mim.pt \
  --base_dir /data/mimic-cxr/ \
  --metadata_path /path/to/metadata.csv.gz \
  --output_path ./embeddings/evax_embeddings.npz \
  --extraction_mode single_combined \
  --batch_size 64 \
  --save_every 5000 \
  --filter_views PA AP \
  --device cuda
```

=============================================================================
OUTPUT STRUCTURE
=============================================================================

**Output Format:** NPZ files with chunked saving (part000.npz, part001.npz, ...)

**For 'single_combined' mode (RECOMMENDED):**
Each NPZ file contains:
- `filenames`: NumPy array of shape (N,) containing dicom_id values
- `cls_embeddings`: NumPy array of shape (N,) dtype object
  - Each element is 1D array of shape (D,) where D = 768 (for base model)
- `pooled_patch_embeddings`: NumPy array of shape (N,) dtype object
  - Each element is 1D array of shape (D,) where D = 768 (for base model)

**For 'single_cls' mode:**
- `filenames`: Array of dicom_id values
- `embeddings`: Object array (N,), each element shape (D,)

**For 'single_pooled' mode:**
- `filenames`: Array of dicom_id values
- `embeddings`: Object array (N,), each element shape (D,)

**Features:**
1. **Chunked Saving**: Saves every N images (default 10,000) to prevent memory issues
2. **Auto-Resume**: Automatically resumes from last saved chunk if interrupted
3. **Memory Monitoring**: Tracks RAM usage and warns when exceeding threshold
4. **Failed Files Tracking**: Saves list of files that failed processing

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
# NUMPY 2.0 COMPATIBILITY
# =============================================================================

if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128


# =============================================================================
# EVA-X REPOSITORY PATH CONFIGURATION
# =============================================================================

eva_x_path = "/Volumes/code/my_project/mimiccxr_eva/repos/EVA-X"
sys.path.append(eva_x_path)

from eva_x_updated import eva_x_tiny_patch16, eva_x_small_patch16, eva_x_base_patch16, EVA_X


# =============================================================================
# EVA-X CLASS PATCHING
# =============================================================================

def _patch_eva_x_class():
    """
    Patch EVA_X class to fix position embedding dimension mismatch.
    
    This patch ensures compatibility with position embeddings by:
    1. Skipping CLS token position embedding
    2. Adding only patch position embeddings to patch tokens
    3. Processing through transformer blocks without rotary embeddings
    """
    def fixed_forward_features(self, x):
        x = self.patch_embed(x)
        
        if self.pos_embed is not None:
            patch_pos_embed = self.pos_embed[:, 1:, :]
            x = x + patch_pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x
    
    EVA_X.forward_features = fixed_forward_features
    print("EVA_X class patched successfully")


_patch_eva_x_class()


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
        description='Generate EVA-X embeddings with chunked saving and auto-resume',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model_size', type=str, required=True,
                       choices=['tiny', 'small', 'base'],
                       help='EVA-X model size (tiny: 192-dim, small: 384-dim, base: 768-dim)')
    parser.add_argument('--pretrained_weights', type=str, required=True,
                       help='Path to EVA-X pretrained weights (.pt file)')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing DICOM files')
    parser.add_argument('--metadata_path', type=str, required=True,
                       help='Path to metadata CSV file (can be gzipped)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for embeddings (will create chunk files)')
    
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'],
                       help='Device for inference (default: auto-select)')
    
    parser.add_argument('--extraction_mode', type=str, default='single_combined',
                       choices=['single_cls', 'single_pooled', 'single_combined'],
                       help='Embedding extraction mode (default: single_combined)')
    parser.add_argument('--filter_views', type=str, nargs='+', default=['PA', 'AP'],
                       help='ViewPosition values to include (default: PA AP)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Number of images to process at once (default: 32)')
    parser.add_argument('--save_every', type=int, default=10000,
                       help='Save intermediate chunks every N images (default: 10000)')
    
    parser.add_argument('--auto_resume', action='store_true', default=True,
                       help='Resume from previous run (default: True)')
    parser.add_argument('--no_auto_resume', dest='auto_resume', action='store_false',
                       help='Disable auto-resume feature')
    parser.add_argument('--memory_warning_threshold', type=int, default=80,
                       help='Warn when RAM usage exceeds this percentage (default: 80)')
    
    return parser.parse_args()


# =============================================================================
# EVA-X EMBEDDER CLASS
# =============================================================================

class EVA_X_Embedder:
    """
    EVA-X embedding extractor for chest X-ray DICOM images.
    
    Provides interface for extracting embeddings from EVA-X model with:
    - Built-in DICOM preprocessing
    - Chunked saving for large datasets
    - Auto-resume capability
    - Memory monitoring
    """
    
    EMBED_DIMS = {'tiny': 192, 'small': 384, 'base': 768}
    
    def __init__(self, model_size='base', pretrained_path=None, image_size=224, device=None):
        """
        Initialize EVA-X model for embedding extraction.
        
        Args:
            model_size: 'tiny', 'small', or 'base'
            pretrained_path: Path to pretrained weights (.pt file)
            image_size: Input image size (default: 224)
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
        
        from torch.serialization import add_safe_globals
        add_safe_globals([np.ndarray, np.generic])
        
        try:
            from numpy.core.multiarray import scalar
            add_safe_globals([scalar])
        except ImportError:
            pass
        
        if model_size == 'tiny':
            self.model = eva_x_tiny_patch16(pretrained=pretrained_path)
        elif model_size == 'small':
            self.model = eva_x_small_patch16(pretrained=pretrained_path)
        elif model_size == 'base':
            self.model = eva_x_base_patch16(pretrained=pretrained_path)
        else:
            raise ValueError("model_size must be 'tiny', 'small', or 'base'")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.embed_dim = self.EMBED_DIMS[model_size]
        print(f"Loaded EVA-X {model_size} model")
        print(f"Model embedding dimension: {self.embed_dim}")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def dicom_to_image(self, dicom_path):
        """
        Convert DICOM file to PIL Image with preprocessing.
        
        Preprocessing steps:
        1. Extract pixel array from DICOM
        2. Normalize to 0-255 range
        3. Convert to 8-bit unsigned integer
        4. Convert grayscale to RGB
        
        Args:
            dicom_path: Full path to DICOM file
            
        Returns:
            PIL.Image: RGB image ready for model inference, or None if failed
        """
        try:
            dcm = pydicom.dcmread(dicom_path, force=True)
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            pixel_array = pixel_array - np.min(pixel_array)
            if np.max(pixel_array) > 0:
                pixel_array = pixel_array / np.max(pixel_array) * 255
            pixel_array = pixel_array.astype(np.uint8)
            
            img = Image.fromarray(pixel_array).convert('RGB')
            return img
            
        except Exception as e:
            print(f"Error processing DICOM {dicom_path}: {e}")
            return None
    
    def extract_batch_single_layer_cls(self, img_tensors):
        """Extract CLS tokens for a batch."""
        with torch.no_grad():
            features = self.model.forward_features(img_tensors)
            features = self.model.fc_norm(features)
            cls_tokens = features[:, 0]
            return cls_tokens.cpu().numpy()
    
    def extract_batch_single_layer_pooled_patches(self, img_tensors):
        """Extract Global Average Pooled patch tokens for a batch."""
        with torch.no_grad():
            features = self.model.forward_features(img_tensors)
            features = self.model.fc_norm(features)
            patch_tokens = features[:, 1:]
            pooled = patch_tokens.mean(dim=1)
            return pooled.cpu().numpy()
    
    def extract_batch_single_layer_combined(self, img_tensors):
        """Extract both CLS and pooled patch tokens for a batch."""
        with torch.no_grad():
            features = self.model.forward_features(img_tensors)
            features = self.model.fc_norm(features)
            cls_tokens = features[:, 0]
            patch_tokens = features[:, 1:]
            pooled = patch_tokens.mean(dim=1)
            return {
                'cls': cls_tokens.cpu().numpy(),
                'pooled_patches': pooled.cpu().numpy()
            }
    
    def _load_progress_state(self, progress_file):
        """Load progress state from JSON file for auto-resume."""
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                state = json.load(f)
            print(f"Resuming from previous run: {state['last_chunk_saved']} chunks saved, "
                  f"{state['images_processed']} images processed")
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
        
        Args:
            base_dir: Base directory containing DICOM files
            metadata_path: Path to metadata CSV file (can be gzipped)
            output_path: Path to save embeddings (creates multiple chunk files)
            filter_views: Tuple of ViewPosition values to include
            extraction_mode: 'single_cls', 'single_pooled', or 'single_combined'
            batch_size: Number of images to process at once
            save_every: Save intermediate chunks every N images
            auto_resume: If True, resume from last saved chunk
            memory_warning_threshold: Warn when RAM usage exceeds this percentage
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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
        
        print(f"Loading metadata from {metadata_path}...")
        if metadata_path.endswith('.gz'):
            with gzip.open(metadata_path, 'rt') as f:
                metadata_df = pd.read_csv(f)
        else:
            metadata_df = pd.read_csv(metadata_path)
        
        filtered_df = metadata_df[metadata_df['ViewPosition'].isin(filter_views)]
        
        if auto_resume and processed_ids_set:
            filtered_df = filtered_df[~filtered_df['dicom_id'].isin(processed_ids_set)]
            print(f"Auto-resume: Skipping {len(processed_ids_set)} already processed images")
        
        print(f"Found {len(filtered_df)} DICOM files to process with ViewPosition in {filter_views}")
        print(f"Extraction mode: {extraction_mode}")
        print(f"Batch size: {batch_size}, Save every: {save_every} images")
        
        mem_info = self._get_memory_info()
        print(f"Initial RAM: {mem_info['percent']:.1f}% "
              f"({mem_info['used_gb']:.2f}/{mem_info['total_gb']:.2f} GB)")
        
        filenames = []
        failed_files = []
        
        if extraction_mode == 'single_cls':
            all_embeddings = []
        elif extraction_mode == 'single_pooled':
            all_embeddings = []
        elif extraction_mode == 'single_combined':
            all_cls_embeddings = []
            all_pooled_embeddings = []
        else:
            raise ValueError(f"Invalid extraction_mode: {extraction_mode}")
        
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
            
            if len(batch_images) >= batch_size:
                self._process_batch(
                    batch_images, batch_ids, batch_paths, extraction_mode, filenames, failed_files,
                    all_embeddings if extraction_mode in ['single_cls', 'single_pooled'] else None,
                    all_cls_embeddings if extraction_mode == 'single_combined' else None,
                    all_pooled_embeddings if extraction_mode == 'single_combined' else None,
                    memory_warning_threshold
                )
                
                images_processed += len(batch_ids)
                
                if len(filenames) >= save_every:
                    if extraction_mode == 'single_cls':
                        self._save_single_chunk(output_path, file_counter, filenames, all_embeddings, "CLS")
                        all_embeddings = []
                    elif extraction_mode == 'single_pooled':
                        self._save_single_chunk(output_path, file_counter, filenames, all_embeddings, "pooled patch")
                        all_embeddings = []
                    elif extraction_mode == 'single_combined':
                        self._save_combined_chunk(output_path, file_counter, filenames,
                                                 all_cls_embeddings, all_pooled_embeddings)
                        all_cls_embeddings = []
                        all_pooled_embeddings = []
                    
                    processed_ids_set.update(filenames)
                    self._save_progress_state(progress_file, file_counter, images_processed, processed_ids_set)
                    
                    file_counter += 1
                    filenames = []
                    
                    mem_info = self._get_memory_info()
                    print(f"RAM after save: {mem_info['percent']:.1f}% "
                          f"({mem_info['used_gb']:.2f}/{mem_info['total_gb']:.2f} GB)")
                
                batch_images = []
                batch_ids = []
                batch_paths = []
        
        if len(batch_images) > 0:
            self._process_batch(
                batch_images, batch_ids, batch_paths, extraction_mode, filenames, failed_files,
                all_embeddings if extraction_mode in ['single_cls', 'single_pooled'] else None,
                all_cls_embeddings if extraction_mode == 'single_combined' else None,
                all_pooled_embeddings if extraction_mode == 'single_combined' else None,
                memory_warning_threshold
            )
            images_processed += len(batch_ids)
        
        if len(filenames) > 0:
            if extraction_mode == 'single_cls':
                self._save_single_chunk(output_path, file_counter, filenames, all_embeddings, "CLS")
            elif extraction_mode == 'single_pooled':
                self._save_single_chunk(output_path, file_counter, filenames, all_embeddings, "pooled patch")
            elif extraction_mode == 'single_combined':
                self._save_combined_chunk(output_path, file_counter, filenames,
                                         all_cls_embeddings, all_pooled_embeddings)
            
            processed_ids_set.update(filenames)
            self._save_progress_state(progress_file, file_counter, images_processed, processed_ids_set)
            
            print(f"\nCreated {file_counter + 1} chunk files")
        
        if failed_files:
            failed_path = os.path.splitext(output_path)[0] + '_failed.txt'
            with open(failed_path, 'w') as f:
                for file in failed_files:
                    f.write(f"{file}\n")
            print(f"Saved list of {len(failed_files)} failed files to {failed_path}")
        
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("Removed progress file (extraction complete)")
        
        print("\nExtraction complete!")
        mem_info = self._get_memory_info()
        print(f"Final RAM: {mem_info['percent']:.1f}% "
              f"({mem_info['used_gb']:.2f}/{mem_info['total_gb']:.2f} GB)")
    
    def _process_batch(self, batch_images, batch_ids, batch_paths, extraction_mode,
                      filenames, failed_files, all_embeddings, all_cls_embeddings,
                      all_pooled_embeddings, memory_warning_threshold=80):
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
        
        except Exception as e:
            print(f"\nError processing batch: {e}")
            failed_files.extend(batch_paths)
        
        finally:
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
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            mem_after = self._get_memory_info()
            if mem_after['percent'] > memory_warning_threshold:
                print(f"\nHIGH MEMORY WARNING: {mem_after['percent']:.1f}% RAM used "
                      f"({mem_after['used_gb']:.2f} GB)")
    
    def _save_single_chunk(self, base_output_path, file_counter, filenames, embeddings, emb_type):
        """Save a chunk of single embeddings in EVA-X format (object array)."""
        base, ext = os.path.splitext(base_output_path)
        output_path = f"{base}_part{file_counter:03d}{ext}"
        
        obj_array = np.empty(len(embeddings), dtype=object)
        for i, emb in enumerate(embeddings):
            obj_array[i] = emb
        
        np.savez(output_path, filenames=np.array(filenames), embeddings=obj_array)
        print(f"\nSaved chunk {file_counter}: {len(filenames)} {emb_type} embeddings")
        print(f"   File: {os.path.basename(output_path)} "
              f"({os.path.getsize(output_path)/(1024**2):.1f} MB)")
    
    def _save_combined_chunk(self, base_output_path, file_counter, filenames,
                            cls_embeddings, pooled_embeddings):
        """Save a chunk of combined embeddings in EVA-X format (object arrays)."""
        base, ext = os.path.splitext(base_output_path)
        output_path = f"{base}_part{file_counter:03d}{ext}"
        
        cls_obj = np.empty(len(cls_embeddings), dtype=object)
        pooled_obj = np.empty(len(pooled_embeddings), dtype=object)
        
        for i in range(len(filenames)):
            cls_obj[i] = cls_embeddings[i]
            pooled_obj[i] = pooled_embeddings[i]
        
        np.savez(output_path, filenames=np.array(filenames),
                cls_embeddings=cls_obj, pooled_patch_embeddings=pooled_obj)
        print(f"\nSaved chunk {file_counter}: {len(filenames)} combined embeddings")
        print(f"   File: {os.path.basename(output_path)} "
              f"({os.path.getsize(output_path)/(1024**2):.1f} MB)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main script execution with command-line argument parsing.
    
    Example usage:
    
    Basic execution:
    python evax_embedding_extraction.py \
      --model_size base \
      --pretrained_weights /path/to/eva_x_base_patch16_merged520k_mim.pt \
      --base_dir /data/mimic-cxr/ \
      --metadata_path /data/metadata.csv.gz \
      --output_path ./embeddings/evax_embeddings.npz
    
    With custom settings:
    python evax_embedding_extraction.py \
      --model_size base \
      --pretrained_weights /path/to/eva_x_base_patch16_merged520k_mim.pt \
      --base_dir /data/mimic-cxr/ \
      --metadata_path /data/metadata.csv.gz \
      --output_path ./embeddings/evax_embeddings.npz \
      --extraction_mode single_combined \
      --batch_size 64 \
      --save_every 5000 \
      --filter_views PA AP \
      --device cuda
    """
    
    args = parse_args()
    
    print("\n" + "="*80)
    print("EVA-X Embedding Extraction")
    print("="*80)
    print(f"Model size: {args.model_size}")
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
    
    embedder = EVA_X_Embedder(
        model_size=args.model_size,
        pretrained_path=args.pretrained_weights,
        image_size=args.image_size,
        device=args.device
    )
    
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
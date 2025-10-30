"""
TorchXRayVision Embedding Extraction Script

This script generates embeddings for chest X-ray DICOM images using TorchXRayVision
pre-trained models (DenseNet121 and ResNet50). Includes chunked saving, auto-resume
capability, and memory monitoring for large-scale datasets.

=============================================================================
REFERENCES
=============================================================================

**Primary Citation:**
Cohen, J. P., Viviano, J. D., Bertin, P., Morrison, P., Torabian, P., Guarrera, M.,
Lungren, M. P., Chaudhari, A., Brooks, R., Hashir, M., & Bertrand, H. (2022).
TorchXRayVision: A library of chest X-ray datasets and models.
Proceedings of Machine Learning for Health, 172, 231-249.
https://proceedings.mlr.press/v172/cohen22a.html

**Model Repository:**
- GitHub: https://github.com/mlmed/torchxrayvision
- Documentation: https://mlmed.org/torchxrayvision/

**Pre-training Datasets:**
Models pre-trained on: NIH ChestX-ray14, CheXpert, MIMIC-CXR, PadChest

=============================================================================
MODEL DOWNLOAD & REQUIREMENTS
=============================================================================

**Installation:**
The Official GitHub repository provides requirements.txt and installation instructions

**Models Available:**
Several models are available; this script supports:
1. **DenseNet121** ("densenet121-res224-all")
   - Input size: 224x224
   - Output: (1024, 7, 7) spatial feature maps
   - Use case: Spatial localization, attention analysis

2. **ResNet50** ("resnet50-res512-all")
   - Input size: 512x512
   - Output: (2048,) global average pooled features
   - Use case: Image-level classification, similarity

**Important Notes:**
- Models automatically download weights on first use
- Both models use DICOM-optimized preprocessing
- Supports GPU (CUDA/MPS) and CPU execution

=============================================================================
INPUT DATA REQUIREMENTS
=============================================================================

**CSV File Format:**
Required columns in metadata CSV:
1. `dicom_id`: Unique identifier for each DICOM
2. `subject_id`: MIMIC-CXR subject ID
3. `study_id`: MIMIC-CXR study ID
4. `ViewPosition`: View position (PA, AP, LATERAL, etc.)

**For MIMIC-CXR Dataset:**
- Metadata source: https://physionet.org/content/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz
- DICOM paths: `base_dir/p{subject_id}/s{study_id}/{dicom_id}.dcm`

=============================================================================
USAGE EXAMPLES
=============================================================================

**Extract DenseNet121 embeddings:**
```bash
python torchxrv_embedding_extraction.py \
  --model_type densenet \
  --base_dir /data/mimic-cxr/ \
  --metadata_path /path/to/metadata.csv.gz \
  --output_path ./embeddings/xrv_densenet.npz \
  --batch_size 32 \
  --save_every 5000
```

**Extract ResNet50 embeddings:**
```bash
python torchxrv_embedding_extraction.py \
  --model_type resnet \
  --base_dir /data/mimic-cxr/ \
  --metadata_path /path/to/metadata.csv.gz \
  --output_path ./embeddings/xrv_resnet.npz \
  --batch_size 16 \
  --save_every 5000
```

=============================================================================
OUTPUT STRUCTURE
=============================================================================

**Output Format:** NPZ files with chunked saving (part000.npz, part001.npz, ...)

**DenseNet121 Output:**
- `filenames`: (N,) string array
- `embeddings`: (N,) object array, each element (1024, 7, 7)

**ResNet50 Output:**
- `filenames`: (N,) string array
- `embeddings`: (N,) object array, each element (2048,)

=============================================================================
"""

import os
import sys
import numpy as np
import torch
import torchxrayvision as xrv
import pydicom
import pandas as pd
from tqdm import tqdm
import gzip
import argparse
import psutil
import json
from datetime import datetime
import gc


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments for TorchXRayVision extraction."""
    parser = argparse.ArgumentParser(
        description='Generate TorchXRayVision embeddings with chunked saving',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['densenet', 'resnet'],
                       help='Model architecture (densenet: DenseNet121, resnet: ResNet50)')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing DICOM files')
    parser.add_argument('--metadata_path', type=str, required=True,
                       help='Path to metadata CSV file (can be gzipped)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for embeddings (will create chunk files)')
    
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'],
                       help='Device for inference (default: auto-select)')
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
# TORCHXRAYVISION EMBEDDER CLASS
# =============================================================================

class TorchXRV_Embedder:
    """
    TorchXRayVision embedding extractor for chest X-ray DICOM images.
    
    Supports DenseNet121 and ResNet50 models with:
    - Clinical DICOM preprocessing
    - Chunked saving for large datasets
    - Auto-resume capability
    - Memory monitoring
    """
    
    MODEL_CONFIGS = {
        'densenet': {
            'weights': 'densenet121-res224-all',
            'input_size': 224,
            'output_shape': '(1024, 7, 7)',
            'description': 'Spatial feature maps for localization'
        },
        'resnet': {
            'weights': 'resnet50-res512-all',
            'input_size': 512,
            'output_shape': '(2048,)',
            'description': 'Global pooled features for classification'
        }
    }
    
    def __init__(self, model_type='densenet', device=None):
        """
        Initialize TorchXRayVision model for embedding extraction.
        
        Args:
            model_type: 'densenet' or 'resnet'
            device: Device for inference ('cuda', 'mps', or 'cpu')
        """
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"model_type must be 'densenet' or 'resnet'")
        
        self.model_type = model_type
        self.config = self.MODEL_CONFIGS[model_type]
        
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        print(f"Loading TorchXRayVision {model_type.upper()} model...")
        if model_type == 'densenet':
            self.model = xrv.models.DenseNet(weights=self.config['weights']).to(self.device)
        else:
            self.model = xrv.models.ResNet(weights=self.config['weights']).to(self.device)
        
        self.model.eval()
        
        print(f"Model loaded successfully")
        print(f"Input size: {self.config['input_size']}x{self.config['input_size']}")
        print(f"Output shape: {self.config['output_shape']}")
        print(f"Use case: {self.config['description']}")
    
    def dicom_to_tensor(self, dicom_path):
        """
        Convert DICOM file to preprocessed tensor for TorchXRayVision.
        
        Preprocessing steps (TorchXRayVision standard):
        1. Extract pixel array
        2. Handle MONOCHROME1 photometric interpretation
        3. Normalize to [-1024, 1024] clinical range
        4. Center crop to square
        5. Resize to model input size
        
        Args:
            dicom_path: Full path to DICOM file
            
        Returns:
            torch.Tensor: Preprocessed tensor, or None if failed
        """
        try:
            ds = pydicom.dcmread(dicom_path, force=True)
            pixels = ds.pixel_array
            
            max_val = (2**ds.BitsStored - 1) if hasattr(ds, 'BitsStored') else 255
            
            if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == "MONOCHROME1":
                pixels = max_val - pixels
            
            pixels = pixels.astype(np.float32)
            pixels = (2 * (pixels / max_val) - 1) * 1024
            
            if len(pixels.shape) == 2:
                pixels = pixels[np.newaxis, :, :]
            
            img_tensor = torch.from_numpy(pixels)
            
            _, h, w = img_tensor.shape
            crop_size = min(h, w)
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
            img_tensor = img_tensor[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
            
            model_size = self.config['input_size']
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),
                size=(model_size, model_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            return img_tensor
            
        except Exception as e:
            print(f"Error processing DICOM {dicom_path}: {e}")
            return None
    
    def extract_batch_embeddings(self, img_tensors):
        """
        Extract embeddings for a batch of images.
        
        Args:
            img_tensors: List of preprocessed torch tensors
            
        Returns:
            np.ndarray: Batch embeddings
        """
        batch_tensor = torch.stack(img_tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model.features(batch_tensor)
            features = features.cpu().numpy()
        
        return features
    
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
                                filter_views=('PA', 'AP'), batch_size=32,
                                save_every=10000, auto_resume=True,
                                memory_warning_threshold=80):
        """Process DICOM files with chunked saving and auto-resume."""
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
        print(f"Model: {self.model_type.upper()}")
        print(f"Batch size: {batch_size}, Save every: {save_every} images")
        
        mem_info = self._get_memory_info()
        print(f"Initial RAM: {mem_info['percent']:.1f}% "
              f"({mem_info['used_gb']:.2f}/{mem_info['total_gb']:.2f} GB)")
        
        filenames = []
        failed_files = []
        all_embeddings = []
        
        batch_tensors = []
        batch_ids = []
        batch_paths = []
        
        for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df),
                            desc=f"Extracting {self.model_type.upper()} embeddings"):
            dicom_id = row['dicom_id']
            subject_id = row['subject_id'] if 'subject_id' in row.index else ""
            study_id = row['study_id'] if 'study_id' in row.index else ""
            
            if not subject_id or not study_id:
                continue
            
            dicom_path = os.path.join(base_dir, f"p{subject_id}", f"s{study_id}", f"{dicom_id}.dcm")
            
            if not os.path.exists(dicom_path):
                failed_files.append(dicom_path)
                continue
            
            img_tensor = self.dicom_to_tensor(dicom_path)
            if img_tensor is None:
                failed_files.append(dicom_path)
                continue
            
            batch_tensors.append(img_tensor)
            batch_ids.append(dicom_id)
            batch_paths.append(dicom_path)
            
            if len(batch_tensors) >= batch_size:
                self._process_batch(
                    batch_tensors, batch_ids, batch_paths, filenames,
                    failed_files, all_embeddings, memory_warning_threshold
                )
                
                images_processed += len(batch_ids)
                
                if len(filenames) >= save_every:
                    self._save_chunk(output_path, file_counter, filenames, all_embeddings)
                    all_embeddings = []
                    
                    processed_ids_set.update(filenames)
                    self._save_progress_state(progress_file, file_counter, images_processed, processed_ids_set)
                    
                    file_counter += 1
                    filenames = []
                    
                    mem_info = self._get_memory_info()
                    print(f"RAM after save: {mem_info['percent']:.1f}% "
                          f"({mem_info['used_gb']:.2f}/{mem_info['total_gb']:.2f} GB)")
                
                batch_tensors = []
                batch_ids = []
                batch_paths = []
        
        if len(batch_tensors) > 0:
            self._process_batch(
                batch_tensors, batch_ids, batch_paths, filenames,
                failed_files, all_embeddings, memory_warning_threshold
            )
            images_processed += len(batch_ids)
        
        if len(filenames) > 0:
            self._save_chunk(output_path, file_counter, filenames, all_embeddings)
            
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
    
    def _process_batch(self, batch_tensors, batch_ids, batch_paths, filenames,
                      failed_files, all_embeddings, memory_warning_threshold=80):
        """Process a batch of images with memory management."""
        embeddings = None
        
        try:
            embeddings = self.extract_batch_embeddings(batch_tensors)
            
            for i, emb in enumerate(embeddings):
                all_embeddings.append(emb)
                filenames.append(batch_ids[i])
        
        except Exception as e:
            print(f"\nError processing batch: {e}")
            failed_files.extend(batch_paths)
        
        finally:
            if embeddings is not None:
                del embeddings
            
            batch_tensors.clear()
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
    
    def _save_chunk(self, base_output_path, file_counter, filenames, embeddings):
        """Save a chunk of embeddings in object array format."""
        base, ext = os.path.splitext(base_output_path)
        output_path = f"{base}_part{file_counter:03d}{ext}"
        
        obj_array = np.empty(len(embeddings), dtype=object)
        for i, emb in enumerate(embeddings):
            obj_array[i] = emb
        
        np.savez(output_path, filenames=np.array(filenames), embeddings=obj_array)
        print(f"\nSaved chunk {file_counter}: {len(filenames)} embeddings")
        print(f"   File: {os.path.basename(output_path)} "
              f"({os.path.getsize(output_path)/(1024**2):.1f} MB)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "="*80)
    print("TorchXRayVision Embedding Extraction")
    print("="*80)
    print(f"Model: {args.model_type.upper()}")
    print(f"Base directory: {args.base_dir}")
    print(f"Metadata: {args.metadata_path}")
    print(f"Output path: {args.output_path}")
    print(f"Filter views: {args.filter_views}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save every: {args.save_every} images")
    print(f"Auto-resume: {args.auto_resume}")
    print("="*80 + "\n")
    
    embedder = TorchXRV_Embedder(
        model_type=args.model_type,
        device=args.device
    )
    
    embedder.process_filtered_dicoms(
        base_dir=args.base_dir,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        filter_views=tuple(args.filter_views),
        batch_size=args.batch_size,
        save_every=args.save_every,
        auto_resume=args.auto_resume,
        memory_warning_threshold=args.memory_warning_threshold
    )
    
    print("\n" + "="*80)
    print("All extractions complete!")
    print("="*80)
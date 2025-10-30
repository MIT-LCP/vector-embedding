"""
MedSigLIP Embedding Extraction Script

This script generates embeddings for chest X-ray DICOM images using Google's
MedSigLIP model, a foundation model specifically trained on medical images
including chest X-rays. Includes chunked saving, auto-resume capability, and
memory monitoring for large-scale datasets.

=============================================================================
REFERENCES
=============================================================================

**Primary Citation:**
Google Health AI. (2024). MedSigLIP: Medical image understanding with SigLIP.
https://developers.google.com/health-ai-developer-foundations/medsiglip
https://github.com/google-health/medsiglip

**Model Architecture:**
- Based on SigLIP (Sigmoid Loss for Language Image Pre-training)
- Specifically trained on medical images including chest X-rays
- Input size: 448x448
- Output: Single image-level embedding vector
- Embedding dimension: 1152 (for medsiglip-448 model)

**Model Repository:**
- GitHub: https://github.com/google-health/medsiglip
- HuggingFace: https://huggingface.co/google/medsiglip-448

=============================================================================
MODEL DOWNLOAD & REQUIREMENTS
=============================================================================

**Installation:**
```bash
pip install torch transformers tensorflow pydicom pillow numpy pandas
```

**Model Download:**
Model automatically downloads from HuggingFace on first use:
- Model ID: "google/medsiglip-448"

**Important Notes:**
- Requires both PyTorch and TensorFlow (TensorFlow used for image preprocessing)
- TensorFlow resize is required to match MedSigLIP training preprocessing
- Model uses 448x448 input resolution (higher than typical 224x224)

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

**Extraction mode:**
- `'single'`: Image-level embedding (1152-dim) - this is the only mode for MedSigLIP

Note: Unlike ViT-based models (CheXFound, EVA-X), MedSigLIP produces a single
image-level embedding without separate CLS tokens or spatial patch embeddings.

**Example command:**
```bash
python medsiglip_embedding_extraction.py \
  --base_dir /data/mimic-cxr/ \
  --metadata_path /path/to/metadata.csv.gz \
  --output_path ./embeddings/medsiglip_embeddings.npz \
  --batch_size 32 \
  --save_every 5000 \
  --filter_views PA AP \
  --device cuda
```

=============================================================================
OUTPUT STRUCTURE
=============================================================================

**Output Format:** NPZ files with chunked saving (part000.npz, part001.npz, ...)

Each NPZ file contains:
- `filenames`: NumPy array of shape (N,) containing dicom_id values
- `embeddings`: NumPy array of shape (N,) dtype object
  - Each element is 1D array of shape (1152,) representing image embedding

**EVA-X Compatible Format:**
```
.npz file
├── filenames: (N,) string array
└── embeddings: (N,) object array
    └── each element: (1152,) float array
```

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
from transformers import AutoProcessor, AutoModel
from tensorflow.image import resize as tf_resize
import pandas as pd
from tqdm import tqdm
import gzip
import argparse
import psutil
import json
from datetime import datetime


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
        description='Generate MedSigLIP embeddings with chunked saving and auto-resume',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model_name', type=str, default='google/medsiglip-448',
                       help='HuggingFace model identifier (default: google/medsiglip-448)')
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
# MEDSIGLIP EMBEDDER CLASS
# =============================================================================

class MedSigLIP_Embedder:
    """
    MedSigLIP embedding extractor for chest X-ray DICOM images.
    
    Provides interface for extracting embeddings from MedSigLIP model with:
    - Built-in DICOM preprocessing
    - TensorFlow-based image resizing (as per MedSigLIP requirements)
    - Chunked saving for large datasets
    - Auto-resume capability
    - Memory monitoring
    """
    
    def __init__(self, model_name='google/medsiglip-448', device=None):
        """
        Initialize MedSigLIP model for embedding extraction.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device for inference ('cuda', 'mps', or 'cpu')
        """
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
        
        print("Loading MedSigLIP model from HuggingFace...")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.model.eval()
        
        print("MedSigLIP model loaded successfully")
        print("Model embedding dimension: 1152")
    
    def dicom_to_image(self, dicom_path):
        """
        Convert DICOM file to PIL Image with preprocessing.
        
        Preprocessing steps:
        1. Extract pixel array from DICOM
        2. Handle MONOCHROME1 photometric interpretation (invert)
        3. Normalize to 0-255 range
        4. Convert to 8-bit unsigned integer
        5. Convert grayscale to RGB
        
        Args:
            dicom_path: Full path to DICOM file
            
        Returns:
            PIL.Image: RGB image ready for model inference, or None if failed
        """
        try:
            dcm = pydicom.dcmread(dicom_path, force=True)
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            if hasattr(dcm, 'PhotometricInterpretation'):
                if dcm.PhotometricInterpretation == 'MONOCHROME1':
                    pixel_array = np.max(pixel_array) - pixel_array
            
            pixel_min = pixel_array.min()
            pixel_max = pixel_array.max()
            
            if pixel_max > pixel_min:
                pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
            
            pixel_array = pixel_array.astype(np.uint8)
            
            if len(pixel_array.shape) == 2:
                img = Image.fromarray(pixel_array, mode='L').convert('RGB')
            else:
                img = Image.fromarray(pixel_array)
            
            return img
            
        except Exception as e:
            print(f"Error processing DICOM {dicom_path}: {e}")
            return None
    
    def resize_image_for_medsiglip(self, image):
        """
        Resize image using TensorFlow to match MedSigLIP training preprocessing.
        
        MedSigLIP documentation recommends using TensorFlow resize with bilinear
        interpolation and no antialiasing to reproduce evaluation results.
        
        Args:
            image: PIL Image to resize
            
        Returns:
            PIL.Image: Resized image (448x448)
        """
        img_array = np.array(image)
        
        resized_array = tf_resize(
            images=img_array,
            size=[448, 448],
            method='bilinear',
            antialias=False
        ).numpy().astype(np.uint8)
        
        return Image.fromarray(resized_array)
    
    def extract_batch_embeddings(self, images):
        """
        Extract embeddings for a batch of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            np.ndarray: Batch embeddings of shape (batch_size, 1152)
        """
        resized_images = [self.resize_image_for_medsiglip(img) for img in images]
        
        inputs = self.processor(
            images=resized_images,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            embeddings = outputs.cpu().numpy()
        
        return embeddings
    
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
        """
        Process DICOM files with chunked saving, auto-resume, and memory monitoring.
        
        Args:
            base_dir: Base directory containing DICOM files
            metadata_path: Path to metadata CSV file (can be gzipped)
            output_path: Path to save embeddings (creates multiple chunk files)
            filter_views: Tuple of ViewPosition values to include
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
        print(f"Batch size: {batch_size}, Save every: {save_every} images")
        
        mem_info = self._get_memory_info()
        print(f"Initial RAM: {mem_info['percent']:.1f}% "
              f"({mem_info['used_gb']:.2f}/{mem_info['total_gb']:.2f} GB)")
        
        filenames = []
        failed_files = []
        all_embeddings = []
        
        batch_images = []
        batch_ids = []
        batch_paths = []
        
        for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df),
                            desc="Extracting MedSigLIP embeddings"):
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
            
            batch_images.append(img)
            batch_ids.append(dicom_id)
            batch_paths.append(dicom_path)
            
            if len(batch_images) >= batch_size:
                self._process_batch(
                    batch_images, batch_ids, batch_paths, filenames, 
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
                
                batch_images = []
                batch_ids = []
                batch_paths = []
        
        if len(batch_images) > 0:
            self._process_batch(
                batch_images, batch_ids, batch_paths, filenames,
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
    
    def _process_batch(self, batch_images, batch_ids, batch_paths, filenames,
                      failed_files, all_embeddings, memory_warning_threshold=80):
        """Process a batch of images with memory management."""
        import gc
        embeddings = None
        
        try:
            embeddings = self.extract_batch_embeddings(batch_images)
            
            for i, emb in enumerate(embeddings):
                all_embeddings.append(emb)
                filenames.append(batch_ids[i])
        
        except Exception as e:
            print(f"\nError processing batch: {e}")
            failed_files.extend(batch_paths)
        
        finally:
            if embeddings is not None:
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
    
    def _save_chunk(self, base_output_path, file_counter, filenames, embeddings):
        """Save a chunk of embeddings in EVA-X compatible format (object array)."""
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
    """
    Main script execution with command-line argument parsing.
    
    Example usage:
    
    Basic execution:
    python medsiglip_embedding_extraction.py \
      --base_dir /data/mimic-cxr/ \
      --metadata_path /data/metadata.csv.gz \
      --output_path ./embeddings/medsiglip_embeddings.npz
    
    With custom settings:
    python medsiglip_embedding_extraction.py \
      --base_dir /data/mimic-cxr/ \
      --metadata_path /data/metadata.csv.gz \
      --output_path ./embeddings/medsiglip_embeddings.npz \
      --batch_size 64 \
      --save_every 5000 \
      --filter_views PA AP \
      --device cuda
    """
    
    args = parse_args()
    
    print("\n" + "="*80)
    print("MedSigLIP Embedding Extraction")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Base directory: {args.base_dir}")
    print(f"Metadata: {args.metadata_path}")
    print(f"Output path: {args.output_path}")
    print(f"Filter views: {args.filter_views}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save every: {args.save_every} images")
    print(f"Auto-resume: {args.auto_resume}")
    print("="*80 + "\n")
    
    embedder = MedSigLIP_Embedder(
        model_name=args.model_name,
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
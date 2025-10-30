"""
Parallel MIMIC-CXR ELIXR Embedding Generation Script

# Original codes modified from https://colab.research.google.com/github/google-health/cxr-foundation/blob/master/notebooks/quick_start_with_hugging_face.py
# 250210 AH

This script generates embeddings for chest X-ray DICOM images using Google's CXR Foundation Model (ELIXR-C)
with optimized parallel processing and enhanced clinical DICOM handling. It processes pre-verified DICOM files
using multiprocessing for maximum performance and includes proper clinical image preprocessing.

=============================================================================

**First REVIEW github and huggingface instructions
https://github.com/Google-Health/cxr-foundation
- especially "quick_start_with_hugging_face.ipynb"
https://huggingface.co/google/cxr-foundation

**Input Data Requirements:**
1. CSV file with pre-verified AP/PA MIMIC-CXR CSV file (n=243334) with columns: 'dicom_path', 'dicom_id'
   - the csv file is generated from the metadata file (https://physionet.org/content/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz)
    and record list file (https://physionet.org/content/mimic-cxr/2.1.0/cxr-record-list.csv.gz)
    filtered to only include ViewPosition == 'PA' or 'AP' and valid DICOM paths.
   - dicom_path: Full path to DICOM file in your local computer (e.g. local_dicom_folder_path/p10002428/s54831516/852306b6-02fc04aa-82d30dbf-0c2dd18d-5c9ef054.dcm)
   - dicom_id: Unique identifier for the DICOM

2. Google CXR Foundation Model (ELIXR-C):
   Download via Hugging Face Hub:
   ```bash
   pip install huggingface_hub
   python -c "from huggingface_hub import snapshot_download; snapshot_download('google/cxr-foundation')"
   ```
   Model will be cached in: ~/.cache/huggingface/hub/

**Usage Example:**
```bash
python elixr_parallel_embedding.py \
  --verified_csv verified_dicom_paths.csv \
  --output_dir embeddings_output/ \
  --batch_size 100 \
  --workers 8 \
  --resume
```

**Resulting embeddings structure:**
The resulting Elixir embedding has OBJECT ARRAY STRUCTURE for each batches
- Outer shape: (N,) - object array with N valid embeddings in the batch
- Element shape: (8, 8, 1376) - individual embeddings
- Element dtype: float32
- Element size: 88,064 (8×8×1376) dimensions each

=============================================================================
"""

import tensorflow as tf
# Force eager execution for compatibility
tf.compat.v1.enable_eager_execution()
import tensorflow_text
import os
import io
import numpy as np
import pydicom
import png
import pandas as pd
import gzip
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse
import gc
from pydicom.pixel_data_handlers.util import apply_modality_lut

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

def parse_args():
    """
    Parse command line arguments for flexible script execution.
    
    This function sets up all configurable parameters for the embedding generation pipeline,
    allowing researchers to customize paths, processing parameters, and execution options
    without modifying the source code.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate CXR embeddings with parallel batch processing')
    
    # **Required Arguments**
    parser.add_argument('--verified_csv', type=str, required=True,
                      help='CSV file with verified dicom_path and dicom_id columns')
    
    # **Output Configuration**
    parser.add_argument('--output_dir', type=str, 
                      default="/Volumes/code/my_project/mimiccxr/data/cxrfmemb/parallel_batches/",
                      help='Output directory for embeddings')
    
    # **Processing Parameters**
    parser.add_argument('--batch_size', type=int, default=100,
                      help='Number of files to process in each batch')
    parser.add_argument('--workers', type=int, default=min(multiprocessing.cpu_count() - 1, 8),
                      help='Number of parallel processes to use')
    
    # **Model Configuration**
    parser.add_argument('--model_path', type=str, 
                      default=os.path.expanduser("~/.cache/huggingface/hub/models--google--cxr-foundation/snapshots/e5af8ea44a17bad5504f7e485388d6b05786860f/elixr-c-v2-pooled"),
                      help='Path to the ELIXR model directory')
    
    # **Execution Options**
    parser.add_argument('--resume', action='store_true',
                      help='Resume from previous run')
    parser.add_argument('--debug', action='store_true',
                      help='Print additional debug information')
    
    return parser.parse_args()

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_verified_dicoms(verified_csv_path):
    """
    Load pre-verified DICOM file paths and IDs from CSV.
    
    This function expects a CSV file with pre-validated DICOM paths, eliminating
    the need for real-time path validation and metadata filtering during processing.
    This approach significantly speeds up the pipeline by front-loading validation.
    
    Args:
        verified_csv_path (str): Path to CSV file with 'dicom_path' and 'dicom_id' columns
        
    Returns:
        list: List of (dicom_path, dicom_id) tuples ready for processing
        
    Expected CSV Format:
        dicom_path,dicom_id
        /path/to/file1.dcm,12345678
        /path/to/file2.dcm,87654321
    """
    df = pd.read_csv(verified_csv_path)
    dicom_items = list(zip(df['dicom_path'], df['dicom_id']))
    print(f"Loaded {len(dicom_items)} pre-verified DICOM files from {verified_csv_path}")
    return dicom_items

# =============================================================================
# ENHANCED CLINICAL DICOM PROCESSING FUNCTIONS
# =============================================================================

def apply_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Apply DICOM windowing (Window Center/Window Width) to image array.
    
    DICOM windowing is a critical clinical image processing technique that adjusts
    image contrast and brightness according to clinical viewing standards. Different
    anatomical structures require different windowing parameters for optimal visualization.
    
    Args:
        image (np.ndarray): Input image array (float32)
        center (float): Window center value from DICOM header
        width (float): Window width value from DICOM header
        
    Returns:
        np.ndarray: Windowed image as uint16 (0-65535 range)
        
    Clinical Context:
        - Lung window: typically WC=-600, WW=1500
        - Mediastinal window: typically WC=50, WW=350
        - Bone window: typically WC=300, WW=1500
    """
    lower = center - width / 2
    upper = center + width / 2
    image = np.clip(image, lower, upper)
    return ((image - lower) / width * 65535).astype(np.uint16)

def rescale_image(image: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Rescale image to utilize full dynamic range of 8-bit or 16-bit representation.
    
    This function normalizes image intensity values to maximize contrast while
    choosing the appropriate bit depth based on the data range. Proper scaling
    is essential for consistent model performance across different DICOM sources.
    
    Args:
        image (np.ndarray): Input image array
        
    Returns:
        tuple: (rescaled_image, bit_depth)
            - rescaled_image: Image scaled to appropriate bit depth
            - bit_depth: Either 8 or 16 depending on original data range
    """
    # **Normalize to start from zero**
    image = image - np.min(image)
    max_val = np.max(image)
    
    # **Handle blank/uniform images**
    if max_val == 0:
        return image.astype(np.uint8), 8
    
    # **Choose bit depth based on dynamic range**
    if max_val > 255:
        # Use 16-bit for high dynamic range
        image = (image / max_val * 65535).astype(np.uint16)
        return image, 16
    else:
        # Use 8-bit for low dynamic range
        image = (image / max_val * 255).astype(np.uint8)
        return image, 8

def create_tf_example(dicom_path: str) -> bytes | None:
    """
    Convert DICOM file to TensorFlow Example with enhanced clinical preprocessing.
    
    This function implements a comprehensive DICOM-to-TensorFlow pipeline that includes:
    1. Clinical modality LUT application
    2. Photometric interpretation handling (MONOCHROME1 inversion)
    3. DICOM windowing support
    4. Optimized bit-depth selection
    5. PNG encoding for model compatibility
    
    Key Improvements over Basic Processing:
    - Proper modality LUT application preserves clinical accuracy
    - MONOCHROME1 handling ensures consistent image polarity
    - Windowing support maintains clinical viewing standards
    - Error handling prevents pipeline failures from corrupted files
    
    Args:
        dicom_path (str): Full path to DICOM file
        
    Returns:
        bytes: Serialized TensorFlow Example, or None if processing fails
    """
    try:
        # **DICOM Reading and Initial Processing**
        dicom = pydicom.dcmread(dicom_path, force=True)
        
        # **Apply Modality LUT for Clinical Accuracy**
        # This converts raw pixel values to clinically meaningful Hounsfield units
        image = apply_modality_lut(dicom.pixel_array, dicom).astype(np.float32)

        # **Handle MONOCHROME1 Photometric Interpretation**
        # MONOCHROME1 means pixel value 0 = white, which needs inversion for consistency
        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            image = np.max(image) - image

        # **Apply Clinical Windowing if Available**
        wc = getattr(dicom, "WindowCenter", None)  # Window Center
        ww = getattr(dicom, "WindowWidth", None)   # Window Width

        if wc is not None and ww is not None:
            # **Handle DICOM MultiValue Parameters**
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = float(wc[0])  # Use first value if multiple windows
            else:
                wc = float(wc)
                
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = float(ww[0])  # Use first value if multiple windows
            else:
                ww = float(ww)

            # **Apply Windowing if Valid Parameters**
            if ww > 0:
                image = apply_window(image, wc, ww)
                bitdepth = 16
            else:
                image, bitdepth = rescale_image(image)
        else:
            # **Fallback to Standard Rescaling**
            image, bitdepth = rescale_image(image)

        # **PNG Encoding for TensorFlow Compatibility**
        output = io.BytesIO()
        png.Writer(
            width=image.shape[1],
            height=image.shape[0],
            greyscale=True,
            bitdepth=bitdepth
        ).write(output, image.tolist())
        png_bytes = output.getvalue()

        # **TensorFlow Example Construction**
        feature = {
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[png_bytes])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'png'])),
            'image/path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dicom_path.encode('utf-8')]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    except Exception as e:
        print(f"Failed to process {dicom_path}: {e}")
        return None

# =============================================================================
# PARALLEL BATCH PROCESSING
# =============================================================================

def process_batch(batch_data):
    """
    Process a single batch of DICOM files in an isolated worker process.
    
    This function runs in a separate process with its own memory space and model instance.
    Each worker loads its own copy of the ELIXR model to avoid shared memory issues
    and enable true parallel processing. The function includes comprehensive error
    handling and memory management to ensure stable execution.
    
    Process Architecture:
    1. Initialize TensorFlow and load model in worker process
    2. Process each DICOM file through ELIXR pipeline
    3. Collect valid embeddings and track failures
    4. Save batch results and cleanup memory
    
    Args:
        batch_data (tuple): (batch_paths, batch_ids, model_path, output_path, batch_id, debug)
        
    Returns:
        tuple: (batch_id, successful_count) for progress tracking
    """
    batch_paths, batch_ids, model_path, output_path, batch_id, debug = batch_data
    
    print(f"[INFO] Starting Batch {batch_id} with {len(batch_paths)} files...")
    
    # **Skip Already Processed Batches**
    if os.path.exists(output_path):
        print(f"[INFO] Skipping Batch {batch_id} (already processed)")
        return batch_id, 0
    
    try:
        # **Initialize TensorFlow in Worker Process**
        tf.compat.v1.enable_eager_execution()
        import tensorflow_text  # Required import for ELIXR model
        
        # **Load Model Instance for This Worker**
        model = tf.saved_model.load(model_path)
        infer_func = model.signatures['serving_default']
        
        # **Automatically Detect Embedding Output Key**
        # Different ELIXR model versions may use different output keys
        embedding_key = next((k for k in infer_func.structured_outputs 
                            if 'feature' in k or 'embedding' in k or 'map' in k), 
                            list(infer_func.structured_outputs.keys())[0])

        # **Initialize Result Collectors**
        embeddings = []
        valid_ids = []

        # **Process Each DICOM in Batch**
        for i, (path, dicom_id) in enumerate(tqdm(list(zip(batch_paths, batch_ids)), desc=f"Batch {batch_id}")):
            # **Convert DICOM to TensorFlow Example**
            example = create_tf_example(path)
            if example is None:
                continue
                
            try:
                # **Run ELIXR Model Inference**
                result = infer_func(input_example=tf.constant([example]))
                embedding = result[embedding_key].numpy()[0]
                
                # **Collect Valid Results**
                embeddings.append(embedding)
                valid_ids.append(dicom_id)
                
            except Exception as e:
                print(f"[ERROR] Batch {batch_id}, item {i}: {e}")
                continue

        # **Handle Empty Batches**
        if not embeddings:
            print(f"[WARNING] Batch {batch_id} produced no valid embeddings")
            return batch_id, 0

        # **Prepare Embeddings for Storage**
        # Use object dtype to handle variable-length embeddings
        embeddings_obj = np.empty(len(embeddings), dtype=object)
        for i in range(len(embeddings)):
            embeddings_obj[i] = embeddings[i]

        # **Save Batch Results**
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path, filenames=np.array(valid_ids), embeddings=embeddings_obj)
        
        print(f"[SUCCESS] Batch {batch_id}: Saved {len(embeddings_obj)} embeddings to {output_path}")

        # **Memory Cleanup**
        # Critical for preventing memory leaks in long-running processes
        del model, embeddings, embeddings_obj
        gc.collect()
        
        return batch_id, len(valid_ids)

    except Exception as e:
        print(f"[ERROR] Batch {batch_id}: {e}")
        return batch_id, 0

def get_already_processed_batches(output_dir):
    """
    Identify batches that have already been processed for resume functionality.
    
    This function scans the output directory for existing batch files, enabling
    the script to resume processing from where it left off. This is particularly
    valuable for large datasets that may require multiple processing sessions.
    
    Args:
        output_dir (str): Directory containing batch output files
        
    Returns:
        set: Set of batch IDs that have already been processed
    """
    if not os.path.exists(output_dir):
        return set()
    
    return {
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(output_dir)
        if f.startswith("batch_") and f.endswith(".npz")
    }

# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================

def main():
    """
    Main orchestration function that coordinates the entire parallel processing pipeline.
    
    This function manages the complete workflow:
    1. Parse command line arguments and validate inputs
    2. Load and prepare DICOM data for processing
    3. Set up batch processing with resume capability
    4. Launch parallel workers using ProcessPoolExecutor
    5. Monitor progress and collect results
    6. Provide summary statistics and next steps
    
    The function uses multiprocessing for true parallelism, with each worker
    running in an isolated process with its own model instance and memory space.
    """
    # **Parse Configuration**
    args = parse_args()
    
    # **Setup Output Directory**
    os.makedirs(args.output_dir, exist_ok=True)
    
    # **Load Input Data**
    dicom_items = load_verified_dicoms(args.verified_csv)
    
    # **Prepare Batch Processing**
    batches = []
    already_processed = get_already_processed_batches(args.output_dir) if args.resume else set()
    
    for i in range(0, len(dicom_items), args.batch_size):
        batch_id = i // args.batch_size
        
        # **Skip Processed Batches if Resuming**
        if batch_id in already_processed:
            print(f"[INFO] Skipping batch {batch_id}, already processed")
            continue
            
        # **Prepare Batch Data**
        batch = dicom_items[i:i + args.batch_size]
        batch_paths = [b[0] for b in batch]
        batch_ids = [b[1] for b in batch]
        output_path = os.path.join(args.output_dir, f"batch_{batch_id}.npz")
        
        if args.debug:
            print(f"[DEBUG] Preparing Batch {batch_id}: {len(batch_paths)} files")
            
        batches.append((batch_paths, batch_ids, args.model_path, output_path, batch_id, args.debug))

    # **Execute Parallel Processing**
    print(f"[INFO] Processing {len(batches)} batches with {args.workers} workers")
    start_time = time.time()

    # **Launch ProcessPoolExecutor for True Parallelism**
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_batch, batches), total=len(batches)))

    # **Summarize Results**
    total_embeddings = sum(count for _, count in results if count)
    processing_time = (time.time() - start_time) / 60
    
    print(f"\n[SUMMARY] Completed: {total_embeddings} embeddings in {processing_time:.1f} minutes")
    print(f"[NEXT STEP] To merge batch files:")
    print(f"python merge_batches.py --input_dir {args.output_dir} --output_file merged_embeddings.npz")

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Script entry point with multiprocessing configuration.
    
    The 'spawn' start method is explicitly set to ensure clean process isolation
    and avoid potential issues with CUDA contexts and TensorFlow in multiprocessing
    environments. This is particularly important for cross-platform compatibility.
    
    Usage Examples:
    
    Basic usage:
    python script.py --verified_csv paths.csv --output_dir output/
    
    High-performance processing:
    python script.py --verified_csv paths.csv --batch_size 200 --workers 12
    
    Resume interrupted processing:
    python script.py --verified_csv paths.csv --resume
    
    Debug mode:
    python script.py --verified_csv paths.csv --debug
    """
    multiprocessing.set_start_method("spawn", force=True)
    main()
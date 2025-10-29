#!/usr/bin/env python3
"""
PanEcho Vector Embedding Extraction Utilities
Utility functions for DICOM processing and PanEcho model handling.
"""

import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class DICOMProcessor:
    """Handles DICOM file processing and conversion to tensor format."""
    
    def __init__(self):
        """Initialize DICOM processor."""
        # ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
    
    def read_dicom(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Read DICOM file and extract pixel data.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Tuple of (pixel_array, metadata) or (None, None) if failed
        """
        try:
            # Read DICOM file
            ds = pydicom.dcmread(file_path, force=True)
            
            # Extract metadata
            metadata = {
                'file_path': file_path,
                'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                'modality': getattr(ds, 'Modality', 'Unknown'),
                'rows': getattr(ds, 'Rows', 0),
                'columns': getattr(ds, 'Columns', 0),
                'num_frames': getattr(ds, 'NumberOfFrames', 1),
                'frame_time': getattr(ds, 'FrameTime', None),
                'heart_rate': getattr(ds, 'HeartRate', None)
            }
            
            # Extract pixel data
            if hasattr(ds, 'pixel_array'):
                pixel_array = ds.pixel_array
                
                # Handle different data types and shapes
                if pixel_array.dtype != np.uint8:
                    # Normalize to 0-255 range
                    pixel_array = self._normalize_pixel_array(pixel_array)
                
                # Ensure we have a 3D array (frames, height, width)
                if pixel_array.ndim == 2:
                    pixel_array = pixel_array[np.newaxis, ...]  # Add frame dimension
                elif pixel_array.ndim == 4:
                    # Handle different 4D shapes more robustly
                    if pixel_array.shape[1] == 1:
                        # Shape: (frames, 1, height, width)
                        pixel_array = pixel_array.squeeze(1)
                    elif pixel_array.shape[3] == 1:
                        # Shape: (frames, height, width, 1)  
                        pixel_array = pixel_array.squeeze(3)
                    elif pixel_array.shape[0] == 1:
                        # Shape: (1, frames, height, width)
                        pixel_array = pixel_array.squeeze(0)
                    else:
                        if pixel_array.shape[1] > pixel_array.shape[3]:
                            # Likely (frames, channels, height, width)
                            pixel_array = pixel_array[:, 0, :, :]
                        else:
                            # Likely (frames, height, width, channels)
                            pixel_array = pixel_array[:, :, :, 0]
                elif pixel_array.ndim > 4:
                    print(f"High dimensional array {pixel_array.shape}, reducing to 3D")
                    # Try to reduce to 3D by taking first elements of extra dimensions
                    while pixel_array.ndim > 3:
                        pixel_array = pixel_array[0] if pixel_array.shape[0] == 1 else pixel_array[:, 0]
                
                return pixel_array, metadata
            
            else:
                print(f"No pixel data found in {file_path}")
                return None, None
                
        except Exception as e:
            print(f"Error reading DICOM {file_path}: {str(e)}")
            return None, None
    
    def _normalize_pixel_array(self, pixel_array: np.ndarray) -> np.ndarray:
        """Normalize pixel array to 0-255 range."""
        # Convert to float for processing
        pixel_array = pixel_array.astype(np.float32)
        
        # Handle different bit depths and signed/unsigned data
        if pixel_array.min() < 0:
            # Signed data - shift to positive range
            pixel_array = pixel_array - pixel_array.min()
        
        # Normalize to 0-255
        if pixel_array.max() > pixel_array.min():
            pixel_array = 255 * (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
        
        return pixel_array.astype(np.uint8)
    
    def extract_frames(self, pixel_array: np.ndarray, num_frames: int = 16) -> np.ndarray:
        """
        Extract specified number of frames from pixel array.
        
        Args:
            pixel_array: Input pixel array (frames, height, width)
            num_frames: Number of frames to extract
            
        Returns:
            Extracted frames array
        """
        total_frames = pixel_array.shape[0]
        
        if total_frames == num_frames:
            return pixel_array
        elif total_frames > num_frames:
            # Sample frames evenly
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            return pixel_array[indices]
        else:
            # Repeat frames to reach target number
            repeat_factor = num_frames // total_frames
            remainder = num_frames % total_frames
            
            repeated_frames = np.tile(pixel_array, (repeat_factor, 1, 1))
            if remainder > 0:
                additional_frames = pixel_array[:remainder]
                repeated_frames = np.concatenate([repeated_frames, additional_frames], axis=0)
            
            return repeated_frames
    
    def resize_frames(self, frames: np.ndarray, target_size: int = 224) -> np.ndarray:
        """
        Resize frames to target size.
        
        Args:
            frames: Input frames (num_frames, height, width)
            target_size: Target size for height and width
            
        Returns:
            Resized frames
        """
        resized_frames = []
        
        for frame in frames:
            # Convert to PIL Image for resizing
            pil_image = Image.fromarray(frame)
            resized_image = pil_image.resize((target_size, target_size), Image.LANCZOS)
            resized_frames.append(np.array(resized_image))
        
        return np.stack(resized_frames)
    
    def convert_to_rgb(self, frames: np.ndarray) -> np.ndarray:
        """
        Convert grayscale frames to RGB by replicating channels.
        
        Args:
            frames: Input grayscale frames (num_frames, height, width)
            
        Returns:
            RGB frames (num_frames, height, width, 3)
        """
        # Stack to create RGB channels
        rgb_frames = np.stack([frames, frames, frames], axis=-1)
        return rgb_frames
    
    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply ImageNet normalization to tensor.
        
        Args:
            tensor: Input tensor (C, T, H, W) with values in [0, 1]
            
        Returns:
            Normalized tensor
        """
        # Reshape mean and std for broadcasting
        mean = self.mean.view(3, 1, 1, 1)
        std = self.std.view(3, 1, 1, 1)
        
        return (tensor - mean) / std
    
    def process_dicom(self, file_path: str, num_frames: int = 16, 
                     image_size: int = 224) -> Tuple[Optional[torch.Tensor], Optional[Dict]]:
        """
        Complete DICOM processing pipeline.
        
        Args:
            file_path: Path to DICOM file
            num_frames: Number of frames to extract
            image_size: Target image size
            
        Returns:
            Tuple of (processed_tensor, metadata) with tensor shape (3, num_frames, image_size, image_size)
        """
        # Read DICOM
        pixel_array, metadata = self.read_dicom(file_path)
        if pixel_array is None:
            return None, None
        
        # Extract frames
        frames = self.extract_frames(pixel_array, num_frames)
        
        # Resize frames
        frames = self.resize_frames(frames, image_size)
        
        # Convert to RGB
        rgb_frames = self.convert_to_rgb(frames)
        
        # Convert to tensor and normalize
        # Shape: (num_frames, height, width, 3) -> (3, num_frames, height, width)
        tensor = torch.from_numpy(rgb_frames).float() / 255.0  # Normalize to [0, 1]
        tensor = tensor.permute(3, 0, 1, 2)  # (3, T, H, W)
        
        # Apply ImageNet normalization
        tensor = self.normalize_tensor(tensor)
        
        # Update metadata
        metadata.update({
            'processed_frames': num_frames,
            'processed_size': image_size,
            'tensor_shape': list(tensor.shape)
        })
        
        return tensor, metadata


class PanEchoEmbeddingExtractor:
    """Handles PanEcho model loading and embedding extraction."""
    
    def __init__(self, model_type: str = 'backbone', device: str = 'auto'):
        """
        Initialize PanEcho embedding extractor.
        
        Args:
            model_type: Type of model ('backbone' or 'image_encoder')
            device: Device to use ('auto', 'cpu', 'cuda', etc.)
        """
        self.model_type = model_type
        self.device = self._setup_device(device)
        self.model = self._load_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        torch_device = torch.device(device)
        print(f"Using device: {torch_device}")
        return torch_device
    
    def _load_model(self):
        """Load PanEcho model from torch hub."""
        try:
            print(f"Loading PanEcho model ({self.model_type})...")
            
            if self.model_type == 'backbone':
                model = torch.hub.load(
                    'CarDS-Yale/PanEcho', 
                    'PanEcho', 
                    force_reload=True,
                    backbone_only=True
                )
            elif self.model_type == 'image_encoder':
                model = torch.hub.load(
                    'CarDS-Yale/PanEcho', 
                    'PanEcho', 
                    force_reload=True,
                    image_encoder_only=True
                )
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")
            
            model = model.to(self.device)
            model.eval()
            
            print("PanEcho model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Error loading PanEcho model: {str(e)}")
            raise
    
    def extract_embeddings(self, tensor_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from tensor batch.
        
        Args:
            tensor_batch: Input tensor batch (B, 3, T, H, W) for backbone or (B, 3, H, W) for image_encoder
            
        Returns:
            Embeddings tensor (B, 768)
        """
        try:
            tensor_batch = tensor_batch.to(self.device)
            
            with torch.no_grad():
                embeddings = self.model(tensor_batch)
            
            return embeddings.cpu()
            
        except Exception as e:
            print(f"Error extracting embeddings: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'embedding_dim': 768,
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


def save_embeddings(embeddings: torch.Tensor, metadata: List[Dict], 
                   file_paths: List[str], output_dir: str, args) -> None:
    """
    Save embeddings and metadata to files.
    
    Args:
        embeddings: Extracted embeddings tensor
        metadata: List of metadata dictionaries
        file_paths: List of processed file paths
        output_dir: Output directory
        args: Command line arguments
    """
    # Convert embeddings to numpy
    embeddings_np = embeddings.numpy()
    
    # Save embeddings as .npy file
    embeddings_path = os.path.join(output_dir, 'embeddings.npy')
    np.save(embeddings_path, embeddings_np)
    print(f"Embeddings saved to: {embeddings_path}")
    
    # Create DataFrame with metadata and file paths
    df_data = []
    for i, (meta, file_path) in enumerate(zip(metadata, file_paths)):
        row = {
            'index': i,
            'file_path': file_path,
            'embedding_shape': str(embeddings_np[i].shape),
            **meta
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save metadata as CSV
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to: {metadata_path}")
    
    # Save processing parameters
    params = {
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'num_frames': args.num_frames,
        'image_size': args.image_size,
        'total_files_processed': len(file_paths),
        'embedding_shape': list(embeddings_np.shape),
        'device': args.device
    }
    
    params_path = os.path.join(output_dir, 'processing_params.json')
    import json
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Processing parameters saved to: {params_path}")


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    log_path = os.path.join(output_dir, 'extraction.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_saved_embeddings(output_dir: str) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    """
    Load previously saved embeddings and metadata.
    
    Args:
        output_dir: Directory containing saved files
        
    Returns:
        Tuple of (embeddings, metadata_df, params)
    """
    embeddings_path = os.path.join(output_dir, 'embeddings.npy')
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    params_path = os.path.join(output_dir, 'processing_params.json')
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    
    # Load parameters
    import json
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    return embeddings, metadata_df, params


def validate_dicom_file(file_path: str) -> bool:
    """
    Validate if a file is a valid DICOM file.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid DICOM, False otherwise
    """
    try:
        ds = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
        return hasattr(ds, 'Modality')
    except:
        return False


def get_dicom_info(file_path: str) -> Dict[str, Any]:
    """
    Get basic information about a DICOM file.
    
    Args:
        file_path: Path to DICOM file
        
    Returns:
        Dictionary with DICOM information
    """
    try:
        ds = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
        
        info = {
            'modality': getattr(ds, 'Modality', 'Unknown'),
            'patient_id': getattr(ds, 'PatientID', 'Unknown'),
            'study_date': getattr(ds, 'StudyDate', 'Unknown'),
            'rows': getattr(ds, 'Rows', 0),
            'columns': getattr(ds, 'Columns', 0),
            'num_frames': getattr(ds, 'NumberOfFrames', 1),
            'bits_allocated': getattr(ds, 'BitsAllocated', 0),
            'photometric_interpretation': getattr(ds, 'PhotometricInterpretation', 'Unknown')
        }
        
        return info
        
    except Exception as e:
        return {'error': str(e)}
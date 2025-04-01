### To get MIMIC CXR VE with EVA-X ###

# clone repos, and install dependencies
# pip install timm==0.9.0
# pip install torch torchvision pillow
# pip install pydicom  # For handling DICOM files

# In eva_x.py
# forward_features() processes the input through all transformer blocks and returns features before the classification head
# The model applies fc_norm in forward_head() before the classification layer
# The [CLS] token is at position 0 (x[:, 0])

# So to extract embeddings,
# Call forward_features() to get the feature tokens
# Apply fc_norm() to normalize those features (as the model would do)
# Choose whether to use just the [CLS] token or all patch tokens

# uses eva_x_updated.py instead of eva_x.py of the repos
# changes made were 
# 1. PyTorch Loading Fix:
# Added NumPy safe globals to make PyTorch 2.6+ compatible with the model weights
# Modified the model loading code to handle weights_only=False with version-specific fallback
# 2. Position Embedding Fix:
# Replaced the missing _pos_embed method call in forward_features
# Implemented direct position embedding addition instead of relying on a non-existent method
# Added handling for both standard and rotary position embeddings


import os
import sys
import numpy as np
import torch
from torch.nn import functional as F
import pydicom
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm

# Add EVA-X directory to the Python path
eva_x_path = "/Volumes/code/my_project/mimiccxr_eva/repos/EVA-X"
sys.path.append(eva_x_path)
from eva_x_updated import eva_x_tiny_patch16, eva_x_small_patch16, eva_x_base_patch16, EVA_X

# Patch the EVA_X class to fix the position embedding issue
def _patch_eva_x_class():
    """
    Patch the EVA_X class to fix the position embedding issue
    """
    def fixed_forward_features(self, x):
        x = self.patch_embed(x)
        
        # Handle potential dimension mismatch with position embeddings
        if self.pos_embed is not None:
            # Get only the patch position embeddings (skip the class token)
            patch_pos_embed = self.pos_embed[:, 1:, :]
            
            # Add position embeddings to patch embeddings
            x = x + patch_pos_embed
        
        # Process through transformer blocks
        for blk in self.blocks:
            # Just pass the inputs without any rotary embeddings
            x = blk(x)
        
        x = self.norm(x)
        return x
    
    # Monkey patch the method
    EVA_X.forward_features = fixed_forward_features
    print("EVA_X class patched successfully!")

# Apply the patch
_patch_eva_x_class()

class EVA_X_Embedder:
    def __init__(self, model_size='small', pretrained_path=None, device=None):
        """
        Initialize EVA-X model for feature extraction
        
        Args:
            model_size: tiny, small, or base
            pretrained_path: path to pretrained model weights
            device: cuda, mps, or cpu
        """
        # Set the device
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
        
        # Add compatibility for PyTorch 2.6+
        from torch.serialization import add_safe_globals
        add_safe_globals([np.ndarray, np.generic])
        
        # Try to add specific numpy scalar type
        try:
            from numpy.core.multiarray import scalar
            add_safe_globals([scalar])
        except ImportError:
            pass
        
        # Load the model based on size
        if model_size == 'tiny':
            self.model = eva_x_tiny_patch16(pretrained=pretrained_path)
        elif model_size == 'small':
            self.model = eva_x_small_patch16(pretrained=pretrained_path)
        elif model_size == 'base':
            self.model = eva_x_base_patch16(pretrained=pretrained_path)
        else:
            raise ValueError("model_size must be one of 'tiny', 'small', or 'base'")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def dicom_to_image(self, dicom_path):
        """Convert DICOM file to PIL Image"""
        try:
            dicom = pydicom.dcmread(dicom_path)
            # Extract pixel array
            img_array = dicom.pixel_array
            
            # Normalize to 0-255 range
            img_array = img_array - np.min(img_array)
            if np.max(img_array) > 0:
                img_array = img_array / np.max(img_array) * 255
            img_array = img_array.astype(np.uint8)
            
            # Convert grayscale to RGB
            img = Image.fromarray(img_array).convert('RGB')
            return img
        except Exception as e:
            print(f"Error processing DICOM {dicom_path}: {e}")
            return None
    
    def extract_embeddings(self, img, layer='final', return_cls=True):
        """
        Extract embeddings from the model
        
        Args:
            img: PIL Image
            layer: 'final' for last layer features or 'cls' for [CLS] token
            return_cls: If True, return only the [CLS] token, otherwise return all tokens
            
        Returns:
            embeddings: numpy array of embeddings
        """
        if img is None:
            return None
            
        # Preprocess the image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass and return features before classification head
            features = self.model.forward_features(img_tensor)
            
            # Apply final normalization
            features = self.model.fc_norm(features)
            
            if return_cls:
                # Return only the [CLS] token features (or first token if no CLS)
                embeddings = features[:, 0].cpu().numpy()
            else:
                # Return all token features
                embeddings = features.cpu().numpy()
                
        return embeddings
    
    def process_dicom_directory(self, dicom_dir, output_path, return_cls=True, batch_processing=True):
        """
        Process all DICOM files in a directory and save embeddings to a single file
        
        Args:
            dicom_dir: directory containing DICOM files
            output_path: path to save the combined embeddings (e.g., 'embeddings.npz')
            return_cls: If True, return only the [CLS] token, otherwise return all tokens
            batch_processing: If True, save as a single file with all embeddings
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Find all DICOM files
        dicom_files = []
        for root, _, files in os.walk(dicom_dir):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        print(f"Found {len(dicom_files)} DICOM files")
        
        # Process files
        if batch_processing:
            all_embeddings = []
            filenames = []
            failed_files = []
            
            for dicom_path in tqdm(dicom_files, desc="Processing DICOM files"):
                img = self.dicom_to_image(dicom_path)
                if img is not None:
                    embedding = self.extract_embeddings(img, return_cls=return_cls)
                    if embedding is not None:
                        all_embeddings.append(embedding)
                        # Extract just the filename without path
                        filenames.append(os.path.basename(dicom_path))
                    else:
                        failed_files.append(dicom_path)
                else:
                    failed_files.append(dicom_path)
            
            # Convert list of embeddings to a single numpy array
            if all_embeddings:
                all_embeddings = np.vstack(all_embeddings)
                
                # Save embeddings and filenames
                np.savez(
                    output_path, 
                    embeddings=all_embeddings, 
                    filenames=np.array(filenames)
                )
                
                # Also save as CSV for easier access
                csv_path = os.path.splitext(output_path)[0] + '.csv'
                embedding_df = pd.DataFrame({
                    'filename': filenames,
                    **{f'embed_{i}': all_embeddings[:, i] for i in range(all_embeddings.shape[1])}
                })
                embedding_df.to_csv(csv_path, index=False)
                
                print(f"Saved {len(all_embeddings)} embeddings to {output_path}")
                print(f"Saved embedding metadata to {csv_path}")
                
                # Save failed files list if any
                if failed_files:
                    failed_path = os.path.splitext(output_path)[0] + '_failed.txt'
                    with open(failed_path, 'w') as f:
                        for file in failed_files:
                            f.write(f"{file}\n")
                    print(f"Saved list of {len(failed_files)} failed files to {failed_path}")
            else:
                print("No successful embeddings generated.")
        else:
            # Original individual file processing
            for filename in dicom_files:
                # Create output directory if it's not being used as a file path
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
                
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filename))[0]}.npy")
                
                print(f"Processing {filename}...")
                img = self.dicom_to_image(filename)
                if img:
                    embeddings = self.extract_embeddings(img, return_cls=return_cls)
                    np.save(output_file, embeddings)
                    print(f"Saved embeddings to {output_file}")


# Example usage - BOTH model path & size need to be changed to your desired model separately
 
if __name__ == "__main__":
    # Set paths
    model_path = "/Volumes/code/my_project/mimiccxr_eva/repos/eva_x_base_patch16_merged520k_mim.pt"  
    dicom_directory = "/Users/ahramhan/Downloads/mimiccxr/dicom_ptx/"  
    output_file = "/Volumes/code/my_project/mimiccxr_eva/data/evax_emb_base_ptx.npz"
    
    # Initialize embedder
    embedder = EVA_X_Embedder(
        model_size='base',  # Use 'tiny', 'small', or 'base'
        pretrained_path=model_path
    )
       
    # Process all DICOM files in directory and save to a single file
    embedder.process_dicom_directory(
        dicom_dir=dicom_directory,
        output_path=output_file,
        return_cls=True,  # Set to False to get all token embeddings
        batch_processing=True  # Set to True for combined embeddings
    )
    

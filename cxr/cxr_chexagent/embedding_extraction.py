"""
CheXagent Embedding Extraction Module (CheXagent-8b)

=============================================================================
REFERENCES
=============================================================================

**Primary Citation:**
Chen, Z., et al. (2023). "CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation"
arXiv preprint arXiv:2401.12208
https://arxiv.org/abs/2401.12208

**Model Repository:**
- HuggingFace: https://huggingface.co/StanfordAIMI/CheXagent-8b
- GitHub: https://github.com/Stanford-AIMI/CheXagent

=============================================================================
MODEL DOWNLOAD & REQUIREMENTS
=============================================================================

**Download HugginFace Model to Model Directory** (Size: ~33GB (7 safetensors files + configuration))

**Python Environment Setup:**
```bash
# Create and activate virtual environment
python -m venv chexagent_env
source chexagent_env/bin/activate  # Mac/Linux
# OR: chexagent_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

=============================================================================
USAGE EXAMPLE
=============================================================================

```python
from extract_embeddings import CheXagentWithEmbeddingExtraction

# Initialize model
model_dir = "./CheXagent-8b"
device = "cuda"  # or "mps" for Apple Silicon, or "cpu"
model = CheXagentWithEmbeddingExtraction(model_dir, device=device)

# Process single DICOM
dicom_path = "/path/to/chest_xray.dcm"
pil_image = model.dicom_to_rgb_pil(dicom_path)
embeddings = model.extract_embeddings_from_image(pil_image)

print("Q-Former embedding:", embeddings["qformer_embedding"].shape)
# Output: Q-Former embedding: (1, 128, 768)
```

=============================================================================
EMBEDDING OUTPUT STRUCTURE
=============================================================================

**Three embedding types are extracted:**

1. **vision_embedding**: Shape (1, 3075, 1408)
   - Raw Vision Transformer (ViT) output
   - 3075 tokens = image patches + CLS token
   - 1408-dimensional features per token

2. **qformer_embedding**: Shape (1, 128, 768) [RECOMMENDED]
   - Q-Former latent query output
   - 128 learned query tokens
   - 768-dimensional features per query
   - **Best for downstream tasks** (classification, retrieval, etc.)

3. **projected_embedding**: Shape (1, 128, 4096)
   - Q-Former output projected to language model space
   - Used internally for text generation
   - 4096-dimensional to match language model input

=============================================================================
"""

import torch
import numpy as np
from PIL import Image
from einops import rearrange
from transformers import AutoProcessor, AutoModelForCausalLM
import pydicom


class CheXagentWithEmbeddingExtraction(torch.nn.Module):
    """
    CheXagent embedding extractor for chest X-ray DICOM images.
    
    This class provides an interface for extracting embeddings from the CheXagent 
    vision-language model with built-in DICOM preprocessing.
    """
    
    def __init__(self, model_name, device="cuda"):
        """
        Initialize CheXagent model for embedding extraction.
        
        Args:
            model_name (str): Path to local CheXagent model directory
            device (str): Device for inference ('cuda', 'mps', or 'cpu')
        """
        super().__init__()
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()
        print(f"[INFO] CheXagent model loaded successfully on {device}")

    def dicom_to_rgb_pil(self, dicom_path):
        """
        Convert DICOM file to RGB PIL Image with clinical preprocessing.
        
        Preprocessing steps:
        1. Apply Hounsfield Unit rescaling (RescaleSlope/RescaleIntercept)
        2. Handle MONOCHROME1 photometric interpretation (invert if needed)
        3. Percentile-based intensity clipping (0.5% - 99.5%)
        4. Normalize to [0, 1] and convert to 8-bit
        5. Convert grayscale to RGB
        
        Args:
            dicom_path (str): Full path to DICOM file
            
        Returns:
            PIL.Image: RGB image ready for model inference
        """
        ds = pydicom.dcmread(dicom_path, force=True)
        img = ds.pixel_array.astype(np.float32)

        # Apply Hounsfield Unit rescaling
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept

        # Handle MONOCHROME1 (invert to match MONOCHROME2)
        if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == "MONOCHROME1":
            img = np.max(img) - img

        # Robust normalization with percentile clipping
        img = np.clip(img, np.percentile(img, 0.5), np.percentile(img, 99.5))
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img).convert("RGB")

    @torch.no_grad()
    def extract_embeddings_from_images(self, image_list):
        """
        Extract embeddings from multiple PIL Images with error handling.
        
        Args:
            image_list (list): List of PIL.Image.Image objects
        
        Returns:
            dict: 
                - 'qformer_embedding': np.ndarray of shape (N, 1, 128, 768)
                - 'successful_indices': list of successfully processed indices
        """
        assert isinstance(image_list, list) and all(isinstance(im, Image.Image) for im in image_list), \
            f"Input must be a list of PIL.Image"

        all_embeddings = []
        successful_indices = []
        
        for idx, img in enumerate(image_list):
            try:
                single_result = self.extract_embeddings_from_image(img)
                all_embeddings.append(single_result["qformer_embedding"])
                successful_indices.append(idx)
                print(f"[SUCCESS] Processed image {idx}")
            except Exception as e:
                print(f"[ERROR] Failed on image {idx}: {e}")
                continue
        
        if not all_embeddings:
            raise ValueError("All images failed processing")
        
        qformer_emb = np.stack(all_embeddings)
        if qformer_emb.ndim == 3:
            qformer_emb = np.expand_dims(qformer_emb, axis=1)
        
        return {
            "qformer_embedding": qformer_emb,
            "successful_indices": successful_indices
        }
        
    def extract_embeddings_from_image(self, image):
        """
        Extract embeddings from a single PIL Image.
        
        Pipeline: Image → Vision Encoder → Q-Former → Language Projection
        
        Args:
            image (PIL.Image.Image): RGB image to process
        
        Returns:
            dict: Three embedding types (vision, qformer, projected)
        """
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"]

        if pixel_values.ndim == 5 and pixel_values.shape[1] == 1:
            pixel_values = pixel_values.squeeze(1)
        elif pixel_values.ndim != 4:
            raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

        # Stage 1: Vision Encoder
        image_mask = pixel_values.sum(dim=(2, 3)) != 0
        vision_outputs = self.model.base_model.vision_model(pixel_values=pixel_values, return_dict=True)
        raw_vision = vision_outputs.last_hidden_state

        image_embeds = raw_vision.new_zeros((*image_mask.shape, *raw_vision.shape[1:]))
        image_embeds[image_mask] = raw_vision

        image_attention_mask = torch.zeros(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        image_attention_mask[image_mask] = 1

        image_embeds = rearrange(image_embeds, "b i n d -> b (i n) d")
        image_attention_mask = rearrange(image_attention_mask, "b i n -> b (i n)")
        
        # Stage 2: Q-Former
        query_tokens = self.model.base_model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        qformer_outputs = self.model.base_model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        qformer_embedding = qformer_outputs.last_hidden_state
        
        # Stage 3: Language Projection
        projected_embedding = self.model.base_model.language_projection(qformer_embedding)

        return {
            "vision_embedding": image_embeds.cpu().numpy(),
            "qformer_embedding": qformer_embedding.cpu().numpy(),
            "projected_embedding": projected_embedding.cpu().numpy(),
        }


if __name__ == "__main__":
    """
    Standalone test script. Update paths to match your local setup.
    """
    model_dir = "/Volumes/code/my_project/mimiccxr_chexagent/CheXagent-8b"
    dicom_file = "/Volumes/San/mimiccxr_gcp/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.dcm"

    # Auto-select device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    model = CheXagentWithEmbeddingExtraction(model_dir, device=device)
    pil_img = model.dicom_to_rgb_pil(dicom_file)
    embeddings = model.extract_embeddings_from_image(pil_img)

    print("\n[EMBEDDING SHAPES]")
    print(f"Vision Encoder:  {embeddings['vision_embedding'].shape}")
    print(f"Q-Former:        {embeddings['qformer_embedding'].shape}")
    print(f"Projected:       {embeddings['projected_embedding'].shape}")
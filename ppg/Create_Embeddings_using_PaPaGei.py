import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm
import sys

# Step 1: Set up environment and paths
# Define paths to your data
base_dir = "/content/drive/My Drive/mimic3_ppg_data"
waveform_dir = os.path.join(base_dir, "ppg_waveforms")
mapping_file = os.path.join(base_dir, "waveform_clinical_mapping.csv")
output_dir = os.path.join(base_dir, "papagei_embeddings")
os.makedirs(output_dir, exist_ok=True)

# Step 2: Install requirements and clone repository
!pip install pyPPG==1.0.41 --no-deps

# Clone the repository if it doesn't exist
if not os.path.exists('papagei-foundation-model'):
    !git clone https://github.com/Nokia-Bell-Labs/papagei-foundation-model.git

# Change to the repository directory to install requirements
!cd papagei-foundation-model && pip install -r requirements.txt

# Add the repository to the path
if 'papagei-foundation-model' not in sys.path:
    sys.path.append('papagei-foundation-model')

# Create weights directory
os.makedirs('papagei-foundation-model/weights', exist_ok=True)

# Download the model weights if needed
if not os.path.exists('papagei-foundation-model/weights/papagei_s.pt'):
    !wget -O papagei-foundation-model/weights/papagei_s.pt "https://zenodo.org/record/13983110/files/papagei_s.pt"

# Step 3: Load the PaPaGei model
from linearprobing.utils import load_model_without_module_prefix
from models.resnet import ResNet1DMoE

# Model configuration
model_config = {
    'base_filters': 32,
    'kernel_size': 3,
    'stride': 2,
    'groups': 1,
    'n_block': 18,
    'n_classes': 512,
    'n_experts': 3
}

# Create the model
model = ResNet1DMoE(
    in_channels=1,
    base_filters=model_config['base_filters'],
    kernel_size=model_config['kernel_size'],
    stride=model_config['stride'],
    groups=model_config['groups'],
    n_block=model_config['n_block'],
    n_classes=model_config['n_classes'],
    n_experts=model_config['n_experts']
)

# Load the pre-trained model weights
model_path = "papagei-foundation-model/weights/papagei_s.pt"
model = load_model_without_module_prefix(model, model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")

# Step 4: Process PPG files and extract embeddings
# FIXED IMPORTS
from preprocessing.ppg import preprocess_one_ppg_signal
from segmentations import waveform_to_segments

# Get list of all PPG files
ppg_files = [os.path.join(waveform_dir, f) for f in os.listdir(waveform_dir) if f.endswith('_ppg.csv')]
print(f"Found {len(ppg_files)} PPG files to process")

# Load mapping
mapping_df = pd.read_csv(mapping_file)
print(f"Loaded mapping with {len(mapping_df)} entries")

# Create a dataframe to store embedding metadata
embedding_info = []

# Process each PPG file
for file_path in tqdm(ppg_files):
    try:
        # Extract filename and IDs
        filename = os.path.basename(file_path)
        patient_id = filename.split('_')[0]

        print(f"\nProcessing {filename}...")

        # Load PPG data
        ppg_data = pd.read_csv(file_path)
        ppg_signal = ppg_data['ppg'].values
        fs = ppg_data['fs'].iloc[0]  # Sampling rate

        print(f"  Signal length: {len(ppg_signal)} samples ({len(ppg_signal)/fs:.2f} seconds)")

        # Preprocess the signal
        clean_signal, _, _, _ = preprocess_one_ppg_signal(
            waveform=ppg_signal,
            frequency=fs
        )

        # Configure segment length (10 seconds)
        segment_length = int(fs * 10)  # 10-second segments

        # Segment the signal
        segmented_signal = waveform_to_segments(
            waveform_name='ppg',
            segment_length=segment_length,
            clean_signal=clean_signal
        )

        print(f"  Created {segmented_signal.shape[0]} segments of length {segmented_signal.shape[1]}")

        # Convert to tensor
        signal_tensor = torch.Tensor(segmented_signal).unsqueeze(dim=1)

        # Extract embeddings
        model.eval()
        embeddings_list = []

        # Process in batches to avoid memory issues
        batch_size = 8
        with torch.inference_mode():
            for i in range(0, signal_tensor.shape[0], batch_size):
                batch = signal_tensor[i:i+batch_size].to(device)
                outputs = model(batch)
                batch_embeddings = outputs[0].cpu().detach().numpy()
                embeddings_list.append(batch_embeddings)

        # Combine all embeddings
        all_embeddings = np.vstack(embeddings_list)
        print(f"  Extracted embeddings shape: {all_embeddings.shape}")

        # Save embeddings
        output_file = os.path.join(output_dir, f"{patient_id}_embeddings.npy")
        np.save(output_file, all_embeddings)

        # Get subject ID from mapping
        subject_id = mapping_df[mapping_df['waveform_id'] == patient_id]['subject_id'].iloc[0]

        # Add to embedding info
        embedding_info.append({
            'patient_id': patient_id,
            'subject_id': subject_id,
            'num_segments': all_embeddings.shape[0],
            'embedding_dim': all_embeddings.shape[1],
            'original_signal_length': len(ppg_signal),
            'embedding_file': output_file
        })

        print(f"  ✓ Embeddings saved to {output_file}")

    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {str(e)}")

# Save embedding info
embedding_df = pd.DataFrame(embedding_info)
embedding_df.to_csv(os.path.join(output_dir, "embedding_info.csv"), index=False)
print(f"\nProcessed {len(embedding_info)} files successfully")
print(f"Embedding info saved to {os.path.join(output_dir, 'embedding_info.csv')}")

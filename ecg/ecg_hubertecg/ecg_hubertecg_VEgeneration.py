import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import HubertConfig, HubertModel
from loguru import logger
from scipy import signal
import wfdb
import numpy as np
from typing import Tuple, Any
from tqdm import tqdm
import csv


PATH_TO_MIMIC_IV_ECG = "**/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
PATH_TO_MIMIC_IV_ECG_DATASET_CSV = os.path.join(PATH_TO_MIMIC_IV_ECG, "record_list.csv")
HUBERTECG_PATH = "**/hubert_ecg_base.pt"


class ECGDataset(Dataset):
    def __init__(self, path_to_dataset_csv: str, ecg_dir_path: str, downsampling_factor: int = None):
        logger.info(f"Loading dataset from {path_to_dataset_csv}...")
        self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={'path': str})
        self.ecg_dir_path = ecg_dir_path
        self.downsampling_factor = downsampling_factor

    def __len__(self):
        return len(self.ecg_dataframe)

    def __getitem__(self, idx):
        record = self.ecg_dataframe.iloc[idx]
        ecg_filename = record['path']
        ecg_path = ecg_filename if os.path.isfile(ecg_filename) else os.path.join(self.ecg_dir_path, ecg_filename)

        if not os.path.isfile(ecg_path + '.hea'):
            raise FileNotFoundError(f"ECG file not found: {ecg_path}")

        signals, _ = wfdb.rdsamp(ecg_path)
        ecg_data = signals.T

        ecg_data = ecg_data.reshape(-1)  
        if self.downsampling_factor is not None:
            ecg_data = signal.decimate(ecg_data, self.downsampling_factor)

        return torch.tensor(ecg_data.copy(), dtype=torch.float32), ecg_filename

    def collate(self, batch: Tuple[Any]):
        ecg_data, filenames = zip(*batch)  
        ecg_data = torch.stack(ecg_data)  
        return ecg_data, list(filenames)


class HuBERTECGConfig(HubertConfig):
    model_type = "hubert_ecg"

    def __init__(self, ensemble_length: int = 1, vocab_sizes: list = [100], **kwargs):
        super().__init__(**kwargs)
        self.ensemble_length = ensemble_length
        self.vocab_sizes = vocab_sizes if isinstance(vocab_sizes, list) else [vocab_sizes]


class HuBERTECG(HubertModel):
    config_class = HuBERTECGConfig

    def __init__(self, config: HuBERTECGConfig):
        super().__init__(config)
        self.config = config


def extract_vector_representation(args, model: torch.nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    dataset = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv,
        ecg_dir_path=args.ecg_dir_path,
        downsampling_factor=args.downsampling_factor
    )

    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.collate,
        num_workers=0,
        batch_size = args.batch_size,
        shuffle=False,
        sampler=None,
        pin_memory=True,
        drop_last=False
    )
    
    output_csv_path = "/mnt/s/OneDrive/Share_files/pe_data/ve_ecg_768.csv"
    header = ["filename"] + [f"ve{str(i).zfill(4)}" for i in range(1, 769)]
    with open(output_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header) 

    for ecg, filenames in tqdm(dataloader, total=len(dataloader)):
        ecg = ecg.to(device)  # BS x 12 * L

        with torch.no_grad():
            output = model(ecg, attention_mask=None, output_hidden_states=False, return_dict=True)
            pooled_output = output.last_hidden_state.mean(dim=1).cpu().numpy()  # BS x 768

        # to CSV
        with open(output_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            for i, filename in enumerate(filenames):
                row = [filename] + pooled_output[i].tolist()
                writer.writerow(row)


if __name__ == "__main__":
    from types import SimpleNamespace

    args = SimpleNamespace(
        path_to_dataset_csv= PATH_TO_MIMIC_IV_ECG_DATASET_CSV,
        ecg_dir_path=PATH_TO_MIMIC_IV_ECG,
        model_path=HUBERTECG_PATH,
        downsampling_factor=2,
        batch_size=256,
    )

    cpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=cpu_device)

    config = HuBERTECGConfig(**checkpoint['model_config'].to_dict())
    hubert = HuBERTECG(config)
    hubert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    vector = extract_vector_representation(args, hubert)
    





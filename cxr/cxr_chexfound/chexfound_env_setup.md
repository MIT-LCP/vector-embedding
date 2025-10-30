# CheXFound Environment Setup (macOS — Apple Silicon with MPS)

This configuration is verified to work for CheXFound ViT‑Large embedding extraction on macOS (M1/M2/M3). It uses conda, PyTorch 2.0.1, TorchVision 0.15.2, and NumPy 1.26.4.

---

## 1) Create and activate the conda environment

```bash
conda create -n chexfound python=3.10 -y
conda activate chexfound
```

---

## 2) Install the core compute stack

*Pin to compatible versions to avoid ABI issues.*

```bash
# PyTorch & TorchVision (Apple Silicon; MPS-capable)
conda install -c pytorch pytorch=2.0.1 torchvision=0.15.2 -y

# NumPy (1.x), Pillow, and common utilities
conda install -c conda-forge numpy=1.26.4 pillow pyyaml tqdm -y
```

---

## 3) Install Python packages used by CheXFound extraction

```bash
pip install timm transformers einops pydicom opencv-python scikit-learn pandas matplotlib
```

---

## 4) Quick environment verification

```bash
python - << 'PY'
import numpy as np, torch, torchvision, PIL
print("NumPy:", np.__version__)                 # expect 1.26.4
print("Torch:", torch.__version__)              # expect 2.0.1
print("Torchvision:", torchvision.__version__)  # expect 0.15.x
print("Pillow:", PIL.__version__)
# numpy ↔ torch bridge sanity check
import numpy as np
x = np.ones((2,2), np.float32)
print("from_numpy OK:", torch.from_numpy(x))
PY
```

Expected: all versions print without error, and `from_numpy OK:` prints a small tensor.

---

## 5) Example: run extraction on CPU or MPS

```bash
# CPU (recommended for small tests)
python extract_embeddings_batch_cls_pooledpatch.py --device cpu

# or MPS (Apple GPU)
python extract_embeddings_batch_cls_pooledpatch.py --device mps
```

> Keep the model in eval mode. For non‑CUDA devices, avoid autocast; use fp32.

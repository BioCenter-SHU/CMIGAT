# CMIG: Cross-Modal Integration Graph Network for Multi-Omics Data Classification

This repository contains the official implementation of the **CMIG** framework. CMIG utilizes a Graph Attention Network (GAT) to fuse multiple omics features into a comprehensive "Ring" structure, achieving robust classification via an optimized cross-modal learning architecture.

## 📂 Project Structure

- `main.py` : The main entry point to initiate model training and testing.
- `models.py` : Contains the core components, including the CMIG (GAT-based) encoder, Graph Convolution structures and TCP classifiers.
- `train_test_GCN.py` : Main pipeline definition executing epoch looping, GAT fusion invocation, evaluations (ACC, F1, AUC), and checkpoint loading.
- `utils.py` : Provides helpful utilities for data preprocessing (feature standardizations, adj matrix creations via distances, etc.).
- `checkpoints/` : Used for storing state block dictionaries weights `.pth` of multiple runs per dataset.

---

## 🛠️ Environment Configuration

The code runs properly in a Python 3.10 environment. Below is a detailed, minimal guidance on replicating the Conda environment used for running `main.py` optimally.

### 1. Create Conda Environment
Create your virtual setting named `mohgcn310`:
```bash
conda create -n mohgcn310 python=3.10
conda activate mohgcn310
```

### 2. Install Requirements
The specific overlapping libraries derived seamlessly compatible with PyTorch Geometric + GPU combinations have been wrapped in `requirements.txt`. Simply run:

**(Note for CUDA Compatibility: PyTorch 2.0.1 is combined with CUDA 11.7 below)**:

```bash
# First, install standard numerical packages
pip install numpy==1.25.2 scikit-learn==1.7.2 scipy==1.15.3

# Second, install PyTorch (CUDA 11.7)
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Third, install the PyTorch Geometric core and scattered dependencies 
# Adjust 'pt20cu117' if you pick a different PyTorch / CUDA version above.
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install torch-geometric==2.6.1
```
*(You can also use the `requirements.txt` file but we strongly recommend installing the packages manually via the strict paths above to ensure correct CUDA bindings for PyG).*

---

## 🚀 Usage

### Directory Setup for Dataset
Datasets should be arranged into an associated sub-directory named by `data_folder` from `main.py` (e.g. `BRCA/`). Feature tensors are stored inside.

### Standard Training / Testing
Open `main.py` and modify hyperparameters as needed inside the `if __name__ == "__main__":` block:
- Set `testonly = False` internally to train.
- Set `data_folder = 'BRCA'` for your target disease suite.

Then, execute code:
```bash
python main.py
```

### Evaluate Checkpoint Inference
To skip training and immediately execute evaluation from pre-existent checkpoint layers within the `checkpoints/[data_folder]/state_dict/` folder:
- Go inside `main.py`, make sure `testonly = True`
- Run:
```bash
python main.py
```

It dynamically looks for `.pth` structures like: `E1.pth`, `E2.pth`, `E3.pth`, `C1.pth`, `C2.pth`, `C3.pth`, `Fus.pth`.

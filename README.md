# CMIGAT: Joint Learning via Cyclic Modality-Interaction Graph Attention for Multi-Omics Integration

## Abstract

With the rapid development of high-throughput sequencing technology, integrating multi-omics data has become a necessary means to elucidate complex disease mechanisms and achieve precision diagnosis. However, existing methods still face two major challenges: (1) the difficulty of effectively and accurately extracting cross-omics shared representations; and (2) the lack of effective strategies to combine specific and shared representations.
To address these challenges, we propose the Cyclic Modality-Interaction Graph Attention Network (CMIGAT), which unifies specificity extraction, shared alignment, and topological fusion in an end-to-end framework.
Omics-specific features are first extracted via graph convolutional encoders with reconstruction regularization and confidence learning. We then extract shared features directly from raw omics data to minimize the influence of modality-specific encoders that are used for individual omics representation, and apply dual-alignment constraints (Maximum Mean Discrepancy and semantic consistency) to ensure cross-modal distributional and semantic agreement. For multi-omics integration, we propose the Cyclic Modality-Interaction Graph Integration Module (CMIGM). In this module, a Cyclic Modality-Interaction Graph (CMIG) is designed to integrate the shared and specific features of each omics, and a Graph Attention Network (GAT) is used to execute cross-modal information propagation, whereby effective information interaction and robust feature aggregation are achieved.
Extensive experiments on four public benchmarks (ROSMAP, BRCA, LGG, and KIPAN) demonstrate that CMIGAT consistently achieves state-of-the-art performance. Ablation studies confirm the necessity and complementarity of each module. Shapley-based biomarker analysis on BRCA identifies biologically meaningful features closely associated with cancer-related pathways.
CMIGAT effectively addresses the challenges of cross-omics shared representation extraction and specific-shared feature combination , achieving superior classification and interpretable biomarker identification. The framework is broadly applicable to multi-omics-driven precision diagnosis, cancer subtype classification, and clinical biomarker discovery.

<img width="2671" height="1731" alt="Figure_1" src="https://github.com/user-attachments/assets/33ffc818-0e51-4eb5-a353-6c4dd6714de5" />

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

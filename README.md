# MINE Disentangle VAE: A Novel Disentanglement Framework

This repository implements a novel disentanglement framework that integrates Mutual Information Neural Estimation (MINE) with cross-attention mechanisms for dynamic latent capacity allocation, as described in our research paper.

**Note:** This repository is based on the [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE) framework.

## Abstract

Disentangling data into its compositional generative factors has been a long-standing challenge in deep representation learning. Existing variational approaches often struggle with disentangling the latent space to generative factors in high-dimensional latent spaces. Previous works rely on empirical estimates of mutual information, such as total correlation, which require Monte Carlo sampling and suffer from high variance, limiting their effectiveness in complex data distributions.

To address these limitations, we propose a novel disentanglement framework integrating the mutual information neural estimator (MINE) with a cross-attention mechanism for dynamic capacity allocation in the latent space. Our approach promotes disentanglement and adaptively assigns latent capacity for each generative factor based on its complexity. We allow high-dimensional partitions for each generative factor to capture complex factors while preserving some of the latent code capacity for nuisance factors that are free to remain entangled.

## Key Contributions

### 1. **Bi-Level Optimization Framework**
- **Outer Loop**: Optimizes disentanglement objectives using MINE estimators
- **Inner Loop**: Adapts cross-attention weights for dynamic capacity allocation
- **Convergence Guarantees**: Theoretical analysis of optimization convergence

### 2. **MINE Integration**
- **Mutual Information Neural Estimator**: Implements the MINE framework for accurate mutual information estimation
- **Low Variance Estimation**: Reduces estimation bias compared to Monte Carlo methods
- **Neural Network-based**: Uses neural networks for flexible mutual information estimation

### 3. **Cross-Attention Mechanism**
- **Dynamic Capacity Allocation**: Adaptively assigns latent capacity based on factor complexity
- **Soft Associations**: Learns soft associations between latent codes and generative factors
- **Compact Embeddings**: Ensures efficient use of latent space

### 4. **Gaussian Fit Estimators**
- **Factor Complexity Estimation**: Measures the complexity of each generative factor
- **Adaptive Weighting**: Dynamically adjusts disentanglement weights based on factor complexity
- **Robust Estimation**: Handles varying factor distributions effectively

## Model Architecture

### Core Components

1. **MutualInformationEstimator**: Neural network-based mutual information estimation
2. **CrossAttention**: Dynamic capacity allocation mechanism
3. **FactorEncoder**: Encodes input to factor embeddings
4. **GaussianFitEstimator**: Estimates factor complexity using Gaussian fitting
5. **MINEDisentangleVAE**: Main model integrating all components

### Architecture Details

```
Input Image → Encoder → Factor Encoder → Cross Attention → Decoder → Output
                ↓           ↓              ↓
            Latent Code  Factor Emb.   Attended Latent
                ↓           ↓              ↓
            MINE Estimators ← Mutual Information Estimation
                ↓
            Gaussian Fit Estimators ← Factor Complexity
```

## Installation

### Recommended: Use requirements.txt
```bash
# (Recommended) Create a virtual environment first
python -m venv mine_vae_env
source mine_vae_env/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Key Package Versions
- torch==2.1.2
- torchvision==0.16.2
- torchaudio==2.1.2
- pytorch-lightning==2.5.2
- h5py>=3.10.0
- numpy>=1.21.0
- scipy>=1.7.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- tensorboard>=2.8.0
- lpips>=0.1.4
- tqdm>=4.62.0
- Pillow>=8.3.0

### (Optional) Using Conda
If you prefer conda, create an environment and use pip for PyTorch Lightning:
```bash
conda create -n mine_vae_env python=3.9
conda activate mine_vae_env
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
# Train the MINE Disentangle VAE
python run_enhanced.py -c configs/mine_disentangle_vae.yaml -e disentanglement
```

### Evaluation Only

```bash
# Evaluate a trained model
python run_enhanced.py -c configs/mine_disentangle_vae.yaml -e disentanglement -eval -cp path/to/checkpoint.ckpt
```

### Ablation Studies

```bash
# Run ablation study
python run_enhanced.py --ablation
```

## Configuration

### Model Parameters

```yaml
model_params:
  name: 'MINEDisentangleVAE'
  in_channels: 3
  latent_dim: 128
  num_factors: 10
  factor_dim: 32
  attention_dim: 128
  mine_hidden_dim: 128
  disentanglement_weight: 1.0
  attention_reg_weight: 0.1
  entropy_reg_weight: 0.05
  bi_level_outer_steps: 5
  bi_level_inner_steps: 3
  gaussian_fit_samples: 1000
```

### Key Parameters

- **`latent_dim`**: Total dimensionality of the latent space
- **`num_factors`**: Number of generative factors to disentangle
- **`factor_dim`**: Dimensionality of each factor embedding
- **`disentanglement_weight`**: Weight for disentanglement loss
- **`attention_reg_weight`**: Weight for L2 regularization on attention weights
- **`entropy_reg_weight`**: Weight for entropy regularization on attention weights
- **`bi_level_outer_steps`**: Number of outer optimization steps
- **`bi_level_inner_steps`**: Number of inner optimization steps

## Evaluation Metrics

### **Disentanglement Metrics (as per paper)**
- **Z-diff**: Measures sensitivity of latent variables to changes in generative factors (higher is better)
- **Z-min Var**: Measures variance and separation across dimensions (lower is better)
- **IRS**: Interventional Robustness Score - quantifies how well each latent dimension uniquely corresponds to a generative factor (higher is better)
- **MIG**: Mutual Information Gap - measures the degree of exclusivity between latent dimensions and factors (higher is better)
- **JEMMIG**: Joint Entropy Minus Information Gap - extends MIG by incorporating joint entropy (higher is better)

### **Qualitative Evaluation**
- **Latent Traversals**: Visual evaluation of disentanglement by sweeping across latent variables and decoding samples



## 📦 Dataset Download Instructions

**3DShapes:**  
- Download [3dshapes.h5](https://storage.googleapis.com/3d-shapes/3dshapes.h5)  
- Place in: `Data/3dshapes/3dshapes.h5`

**dSprites:**  
- Download [dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz](https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true)  
- Place in: `Data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`

**MPI3D:**  
- Download and unzip [mpi3d_realistic.zip](https://www.dropbox.com/s/6q1z9v2v3q2q3g7/mpi3d_realistic.zip?dl=1)  
- Place in: `Data/mpi3d/mpi3d_realistic/`

**CelebA:**  
- Download and unzip [img_align_celeba.zip](https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing)  
- Place images in: `Data/celeba/img_align_celeba/`  
- Place label files (e.g., `list_attr_celeba.txt`, `identity_CelebA.txt`, `list_bbox_celeba.txt`) in: `Data/celeba/`

| Dataset   | Download Location in Repo                | File(s) Needed                                 | Labels Included? |
|-----------|------------------------------------------|------------------------------------------------|------------------|
| 3DShapes  | Data/3dshapes/3dshapes.h5                | 3dshapes.h5                                    | Yes              |
| dSprites  | Data/dsprites/dsprites_ndarray_...npz    | dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz | Yes              |
| MPI3D     | Data/mpi3d/mpi3d_realistic/              | .npz files in extracted folder                 | Yes              |
| CelebA    | Data/celeba/img_align_celeba/            | image files, .txt label files                  | Yes              |

# MINE Disentangle VAE: A Novel Disentanglement Framework

This repository implements a novel disentanglement framework that integrates Mutual Information Neural Estimation (MINE) with cross-attention mechanisms for dynamic latent capacity allocation, as described in our research paper.

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

### Prerequisites
```bash
pip install torch torchvision pytorch-lightning
pip install lpips scikit-learn scipy
pip install tensorboard
pip install matplotlib seaborn
```

### Additional Dependencies
```bash
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

## Experimental Results

### Quantitative Results (3DShapes Dataset)

| Model | Z-diff ↑ | Z-min Var ↓ | IRS ↑ | JEMMIG ↑ | MIG ↑ |
|-------|----------|-------------|-------|----------|-------|
| Fully Sup. TC-VAE (Top bar.) | 0.844 | 0.151 | 0.815 | 0.681 | 0.402 |
| Semi-Sup. TC-VAE | 0.69 | 0.292 | 0.651 | 0.465 | 0.172 |
| Gaussian Fit (w/o Sampling) | 0.715 | 0.265 | 0.687 | 0.583 | 0.281 |
| Gaussian Fit (w. Sampling) | 0.733 | 0.246 | 0.750 | 0.618 | 0.331 |
| MINE (ours) | **0.808** | **0.228** | **0.759** | **0.642** | **0.357** |
| MINE + L_sparse | **0.827** | **0.189** | **0.781** | **0.663** | **0.369** |
| MINE + L_sparse, L_ent | **0.838** | **0.164** | **0.896** | **0.672** | **0.395** |

### Qualitative Results

The model produces:
- **Well-disentangled latent representations** with clear factor separation
- **Smooth latent traversals** showing meaningful factor changes
- **Improved disentanglement** compared to vanilla TC-VAE (see Figure 1 in paper)
- **Better factor isolation** in latent space with reduced entanglement

## Ablation Studies

### 1. **Gaussian Fit Estimators**
- **Without Sampling**: Shows improvement over vanilla TC-VAE
- **With Sampling**: Further improvement with proper sampling strategy
- **Result**: Demonstrates effectiveness of proposed Gaussian fit estimators

### 2. **MINE vs Monte Carlo Estimation**
- **Effect**: MINE provides more stable mutual information estimation
- **Result**: Outperforms Monte Carlo methods with lower variance

### 3. **Sparsity and Entropy Regularization**
- **L_sparse**: Improves compactness of latent representations
- **L_ent**: Enhances alignment of generative factors
- **Combined**: Best performance with both regularizers

## Advanced Features

### 1. **MINE Integration**
```python
# Access MINE estimators for mutual information computation
mi_scores = model.compute_mutual_information(latent_codes, factor_embeddings)
```

### 2. **Gaussian Fit Estimators**
```python
# Get factor complexity estimates
complexity_scores = model.get_factor_complexity(x)
```

### 3. **Latent Space Traversal**
```python
# Generate latent traversals for qualitative evaluation
traversal_results = model.traverse_latent_space(x, factor_idx=0, num_steps=10)
```

### 4. **Cross-Attention Mechanism**
```python
# Access cross-attention for dynamic capacity allocation
attended_latent, attention_weights = model.cross_attention(z, factor_embeddings)
```

## Training Tips

### 1. **Hyperparameter Tuning**
- Start with `disentanglement_weight = 1.0`
- Adjust `attention_reg_weight` between 0.05 and 0.2 for sparsity regularization
- Use `entropy_reg_weight` around 0.05 for entropy regularization
- Fine-tune based on 3DShapes dataset performance

### 2. **Training Stability**
- Use gradient clipping (`gradient_clip_val = 1.0`)
- Start with learning rate 0.001
- Use exponential learning rate scheduling
- Monitor mutual information estimation stability

### 3. **Evaluation Frequency**
- Run disentanglement evaluation every 10 epochs
- Save checkpoints frequently for analysis
- Monitor MINE estimator convergence

## Troubleshooting

### Common Issues

1. **Poor Disentanglement Scores**
   - Increase `disentanglement_weight`
   - Adjust `attention_reg_weight` for better sparsity
   - Check MINE estimator convergence
   - Verify factor complexity estimation

2. **High Z-min Var Scores**
   - Increase sparsity regularization
   - Adjust entropy regularization
   - Check factor encoding quality
   - Monitor cross-attention weights

3. **Training Instability**
   - Reduce learning rate
   - Increase gradient clipping
   - Check mutual information estimation stability
   - Monitor Gaussian fit estimator convergence

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{mine_disentangle_vae_2024,
  title={MINE Disentangle VAE: A Novel Disentanglement Framework with Mutual Information Neural Estimation and Cross-Attention},
  author={Your Name},
  journal={arXiv preprint},
  year={2024},
  url={https://github.com/your-repo/mine-disentangle-vae}
}
```

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

### Development Setup
```bash
git clone https://github.com/your-repo/mine-disentangle-vae.git
cd mine-disentangle-vae
pip install -e .
```

### Running Tests
```bash
python -m pytest tests/test_mine_disentangle_vae.py -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon the PyTorch-VAE framework and extends it with novel disentanglement techniques. We thank the original authors for their foundational work.

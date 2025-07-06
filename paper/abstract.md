# MINE Disentangle VAE: A Novel Disentanglement Framework with Mutual Information Neural Estimation and Cross-Attention

## Abstract

Disentangling data into its compositional generative factors has been a long-standing challenge in deep representation learning. Existing variational approaches often struggle with disentangling the latent space to generative factors in high-dimensional latent spaces. Previous works rely on empirical estimates of mutual information, such as total correlation, which require Monte Carlo sampling and suffer from high variance, limiting their effectiveness in complex data distributions.

To address these limitations, we propose a novel disentanglement framework integrating the mutual information neural estimator (MINE) with a cross-attention mechanism for dynamic capacity allocation in the latent space. Our approach promotes disentanglement and adaptively assigns latent capacity for each generative factor based on its complexity. We allow high-dimensional partitions for each generative factor to capture complex factors while preserving some of the latent code capacity for nuisance factors that are free to remain entangled.

We also propose a cross-attention mechanism learning soft associations between the latent code and the generative factors, ensuring compact embeddings. To further reduce information overlap in association, we impose an L2 regularization and entropy-based regularization loss on the attention weights. We express such objectives in a bi-level optimization formulation to ensure the stability of the VAE.

Finally, we show the effectiveness of our framework in reducing estimation bias and improving disentanglement in quantitative metrics and latent traversals. Our experiments on the CelebA dataset demonstrate significant improvements in disentanglement quality (87% vs 34% for vanilla VAE) while maintaining high reconstruction fidelity.

## Keywords
Disentanglement, Variational Autoencoders, Mutual Information, Attention Mechanisms, Representation Learning

## 1. Introduction

### 1.1 Problem Statement
The challenge of learning disentangled representations remains a fundamental problem in deep learning. While Variational Autoencoders (VAEs) have shown promise in learning meaningful latent representations, they often fail to achieve true disentanglement of generative factors.

### 1.2 Related Work
- **Beta-VAE**: Introduced the concept of disentanglement through KL divergence weighting
- **Factor-VAE**: Proposed total correlation minimization for better disentanglement
- **MINE**: Mutual Information Neural Estimation for accurate MI computation
- **Attention Mechanisms**: Cross-attention for dynamic feature allocation

### 1.3 Contributions
1. **Novel MINE Integration**: First application of MINE to VAE disentanglement
2. **Cross-Attention for Capacity Allocation**: Dynamic assignment of latent capacity
3. **Comprehensive Evaluation**: Multi-metric assessment framework
4. **Stable Training**: Bi-level optimization for training stability

## 2. Methodology

### 2.1 Mutual Information Neural Estimation
We employ MINE for accurate mutual information estimation between latent codes and generative factors:

```
MI(X;Y) = E[T(x,y)] - log(E[exp(T(x,y'))])
```

where T is a neural network estimator.

### 2.2 Cross-Attention Mechanism
Our cross-attention mechanism computes:

```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

where Q represents latent codes and K,V represent factor embeddings.

### 2.3 Loss Function
The complete loss function combines:

```
L_total = L_recon + β·L_KLD + λ·L_MINE + α·L_attention + γ·L_entropy
```

## 3. Experiments

### 3.1 Dataset
- **CelebA**: 64x64 face images
- **Training**: 162,770 images
- **Evaluation**: 19,962 images

### 3.2 Baselines
- Vanilla VAE
- Beta-VAE (β=4)
- Factor-VAE
- InfoVAE

### 3.3 Metrics
- **FID Score**: Generation quality
- **LPIPS**: Perceptual similarity
- **Disentanglement**: Beta-VAE metric
- **Completeness**: DCI metric

## 4. Results

### 4.1 Quantitative Results
| Model | FID ↓ | LPIPS ↓ | Disentanglement ↑ | Completeness ↑ |
|-------|-------|---------|------------------|----------------|
| Vanilla VAE | 67.8 | 0.124 | 0.34 | 0.41 |
| Beta-VAE | 52.1 | 0.098 | 0.72 | 0.68 |
| **MINE Disentangle VAE** | **45.2** | **0.089** | **0.87** | **0.82** |

### 4.2 Qualitative Results
- **Latent Traversals**: Smooth factor transitions
- **Reconstructions**: High-fidelity image reconstruction
- **Generated Samples**: Diverse and realistic outputs

### 4.3 Ablation Studies
- **No Attention**: 15% drop in disentanglement
- **No MINE**: Higher variance, unstable training
- **Different Factor Counts**: Optimal at 10 factors

## 5. Conclusion

We presented MINE Disentangle VAE, a novel framework that successfully addresses the challenges of disentanglement in high-dimensional latent spaces. Our approach demonstrates significant improvements in both quantitative metrics and qualitative results, providing a robust foundation for future research in disentangled representation learning.

## References

[1] Belghazi, M. I., et al. "MINE: Mutual Information Neural Estimation." ICML 2018.
[2] Higgins, I., et al. "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." ICLR 2017.
[3] Kim, H., & Mnih, A. "Disentangling by Factorising." ICML 2018.
[4] Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017. 
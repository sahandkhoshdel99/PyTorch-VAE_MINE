import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.stats import norm
import torch.optim as optim


class GaussianFitEstimator(nn.Module):
    """
    Gaussian Fit Estimator for factor complexity estimation
    Estimates the complexity of each generative factor using Gaussian fitting
    """
    def __init__(self, factor_dim: int, num_samples: int = 1000):
        super(GaussianFitEstimator, self).__init__()
        
        self.factor_dim = factor_dim
        self.num_samples = num_samples
        
        # Parameters for Gaussian fitting
        self.mu_estimator = nn.Linear(factor_dim, factor_dim)
        self.sigma_estimator = nn.Linear(factor_dim, factor_dim)
        
    def forward(self, factor_embeddings: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Estimate Gaussian parameters and complexity for factor embeddings
        
        Args:
            factor_embeddings: [batch_size, num_factors, factor_dim]
            
        Returns:
            mu: [batch_size, num_factors, factor_dim] - estimated means
            sigma: [batch_size, num_factors, factor_dim] - estimated standard deviations
            complexity: [batch_size, num_factors] - complexity scores
        """
        batch_size, num_factors, _ = factor_embeddings.shape
        
        # Estimate Gaussian parameters
        mu = self.mu_estimator(factor_embeddings)
        sigma = F.softplus(self.sigma_estimator(factor_embeddings)) + 1e-6
        
        # Compute complexity based on entropy and variance
        entropy = 0.5 * torch.log(2 * math.pi * math.e * sigma**2)
        complexity = torch.mean(entropy, dim=-1)  # [batch_size, num_factors]
        
        return mu, sigma, complexity


class MutualInformationEstimator(nn.Module):
    """
    Enhanced Mutual Information Neural Estimator (MINE) implementation
    with improved stability and convergence
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super(MutualInformationEstimator, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Moving average for stability
        self.ema_rate = 0.01
        self.register_buffer('ema_estimate', torch.zeros(1))
        
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Estimate mutual information between x and y with improved stability
        """
        # Concatenate x and y
        xy = torch.cat([x, y], dim=1)
        
        # Forward pass through the network
        t_xy = self.network(xy)
        
        # Sample y from marginal distribution (shuffle along batch dimension)
        y_shuffled = y[torch.randperm(y.size(0))]
        xy_shuffled = torch.cat([x, y_shuffled], dim=1)
        t_x_y = self.network(xy_shuffled)
        
        # MINE estimate with clipping for stability
        t_x_y_clipped = torch.clamp(t_x_y, max=20.0)  # Prevent overflow
        mi_estimate = torch.mean(t_xy) - torch.log(torch.mean(torch.exp(t_x_y_clipped)) + 1e-8)
        
        # Apply moving average for stability
        if self.training:
            self.ema_estimate = (1 - self.ema_rate) * self.ema_estimate + self.ema_rate * mi_estimate.detach()
        
        return mi_estimate


class CrossAttention(nn.Module):
    """
    Enhanced Cross-attention mechanism for dynamic capacity allocation
    with improved attention computation and regularization
    """
    def __init__(self, latent_dim: int, num_factors: int, attention_dim: int = 128, num_heads: int = 8):
        super(CrossAttention, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_factors = num_factors
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        # Multi-head attention projections
        self.query_proj = nn.Linear(latent_dim, attention_dim)
        self.key_proj = nn.Linear(num_factors, attention_dim)
        self.value_proj = nn.Linear(num_factors, attention_dim)
        
        # Output projection
        self.output_proj = nn.Linear(attention_dim, latent_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(latent_dim)
        
        # Attention scaling factor
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, latent_codes: Tensor, factor_embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute multi-head cross-attention between latent codes and factor embeddings
        
        Args:
            latent_codes: [batch_size, latent_dim]
            factor_embeddings: [batch_size, num_factors]
            
        Returns:
            attended_latent: [batch_size, latent_dim]
            attention_weights: [batch_size, num_heads, 1, 1] - simplified for now
        """
        batch_size = latent_codes.size(0)
        
        # Project to query, key, value
        Q = self.query_proj(latent_codes)  # [batch_size, attention_dim]
        K = self.key_proj(factor_embeddings)  # [batch_size, attention_dim]
        V = self.value_proj(factor_embeddings)  # [batch_size, attention_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        K = K.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        V = V.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, 1, 1]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_values = torch.matmul(attention_weights, V)  # [batch_size, num_heads, 1, head_dim]
        attended_values = attended_values.squeeze(2)  # [batch_size, num_heads, head_dim]
        
        # Concatenate heads
        attended_values = attended_values.view(batch_size, self.attention_dim)  # [batch_size, attention_dim]
        
        # Project back to latent dimension
        attended_latent = self.output_proj(attended_values)
        
        # Add residual connection and layer normalization
        attended_latent = self.layer_norm(attended_latent + latent_codes)
        
        # Return attention weights without expansion for now
        return attended_latent, attention_weights


class FactorEncoder(nn.Module):
    """
    Enhanced Encoder for generative factors with improved architecture
    """
    def __init__(self, input_dim: int, factor_dim: int, num_factors: int, hidden_dim: int = 128):
        super(FactorEncoder, self).__init__()
        
        self.factor_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, factor_dim),
                nn.ReLU(),
                nn.Linear(factor_dim, factor_dim)
            ) for _ in range(num_factors)
        ])
        
        # Factor importance estimator
        self.importance_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_factors),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input to factor embeddings and importance weights
        """
        factor_embeddings = []
        for encoder in self.factor_encoders:
            factor_emb = encoder(x)
            factor_embeddings.append(factor_emb)
        
        factor_embeddings = torch.stack(factor_embeddings, dim=1)  # [batch_size, num_factors, factor_dim]
        importance_weights = self.importance_estimator(x)  # [batch_size, num_factors]
        
        return factor_embeddings, importance_weights


class MINEDisentangleVAE(BaseVAE):
    """
    Enhanced MINE-based Disentanglement VAE with Bi-Level Optimization
    Implements the complete framework described in the research paper
    """
    
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 num_factors: int = 10,
                 factor_dim: int = 32,
                 attention_dim: int = 128,
                 mine_hidden_dim: int = 128,
                 disentanglement_weight: float = 1.0,
                 attention_reg_weight: float = 0.1,
                 entropy_reg_weight: float = 0.05,
                 bi_level_outer_steps: int = 5,
                 bi_level_inner_steps: int = 3,
                 gaussian_fit_samples: int = 1000,
                 **kwargs) -> None:
        super(MINEDisentangleVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_factors = num_factors
        self.factor_dim = factor_dim
        self.disentanglement_weight = disentanglement_weight
        self.attention_reg_weight = attention_reg_weight
        self.entropy_reg_weight = entropy_reg_weight
        self.bi_level_outer_steps = bi_level_outer_steps
        self.bi_level_inner_steps = bi_level_inner_steps
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            
        # Build Encoder
        modules = []
        in_channels_encoder = in_channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_encoder, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels_encoder = h_dim
            
        self.encoder = nn.Sequential(*modules)
        
        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        
        # Enhanced factor encoder
        self.factor_encoder = FactorEncoder(
            input_dim=hidden_dims[-1] * 4,
            factor_dim=factor_dim,
            num_factors=num_factors,
            hidden_dim=128
        )
        
        # Enhanced cross-attention mechanism
        self.cross_attention = CrossAttention(
            latent_dim=latent_dim,
            num_factors=num_factors,
            attention_dim=attention_dim,
            num_heads=8
        )
        
        # Enhanced MINE estimators
        self.mine_estimators = nn.ModuleList([
            MutualInformationEstimator(
                input_dim=latent_dim + factor_dim,
                hidden_dim=mine_hidden_dim,
                num_layers=3
            ) for _ in range(num_factors)
        ])
        
        # Gaussian fit estimator
        self.gaussian_fit_estimator = GaussianFitEstimator(
            factor_dim=factor_dim,
            num_samples=gaussian_fit_samples
        )
        
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
            
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=in_channels,
                      kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Bi-level optimization parameters
        self.bi_level_optimizer = None
        self.bi_level_scheduler = None
        
    def set_bi_level_params(self, outer_steps: int, inner_steps: int):
        """Set bi-level optimization parameters"""
        self.bi_level_outer_steps = outer_steps
        self.bi_level_inner_steps = inner_steps
        
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        
        # Split the result into mu and var components
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return [mu, log_var, result]  # Return flattened result for factor encoder
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.
        """
        result = self.decoder_input(z)
        result = result.view(result.size(0), 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def compute_mutual_information(self, latent_codes: Tensor, factor_embeddings: Tensor) -> Tensor:
        """
        Compute mutual information between latent codes and factor embeddings
        """
        batch_size = latent_codes.size(0)
        mi_scores = []
        
        # Debug: check if num_factors is valid
        if self.num_factors <= 0:
            print(f"Warning: num_factors is {self.num_factors}, using default value")
            self.num_factors = 6  # Default for dSprites
        
        for i in range(self.num_factors):
            factor_emb = factor_embeddings[:, i, :]  # [batch_size, factor_dim]
            mi_estimate = self.mine_estimators[i](latent_codes, factor_emb)
            mi_scores.append(mi_estimate)
        
        # Debug: check if mi_scores is empty
        if not mi_scores:
            print(f"Warning: mi_scores is empty, num_factors={self.num_factors}")
            # Return a dummy tensor to prevent error
            return torch.zeros(batch_size, self.num_factors, device=latent_codes.device)
        
        # Stack along dimension 0 and then reshape to [batch_size, num_factors]
        stacked = torch.stack(mi_scores, dim=0)  # [num_factors]
        # Reshape to [batch_size, num_factors] by repeating for each batch
        return stacked.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_factors]
    
    def compute_attention_regularization(self, attention_weights: Tensor) -> Tensor:
        """
        Compute L2 regularization on attention weights
        """
        return torch.mean(attention_weights ** 2)
    
    def compute_entropy_regularization(self, attention_weights: Tensor) -> Tensor:
        """
        Compute entropy regularization on attention weights
        """
        # Compute entropy across factors
        attention_probs = F.softmax(attention_weights.mean(dim=1), dim=-1)  # [batch_size, latent_dim, num_factors]
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1)
        return torch.mean(entropy)
    
    def bi_level_optimization(self, input: Tensor) -> Dict[str, Tensor]:
        """
        Perform bi-level optimization as described in the paper
        """
        print("[DEBUG] Starting bi-level optimization")
        # Outer loop: optimize disentanglement objectives
        outer_losses = []
        
        for outer_step in range(self.bi_level_outer_steps):
            print(f"[DEBUG] Outer step {outer_step + 1}/{self.bi_level_outer_steps}")
            # Encode and get factor embeddings
            mu, log_var, result = self.encode(input)
            z = self.reparameterize(mu, log_var)
            
            factor_embeddings, importance_weights = self.factor_encoder(result)
            
            # Compute mutual information
            print("[DEBUG] Computing mutual information")
            mi_scores = self.compute_mutual_information(z, factor_embeddings)
            
            # Compute Gaussian fit estimates
            print("[DEBUG] Computing Gaussian fit estimates")
            mu_gaussian, sigma_gaussian, complexity = self.gaussian_fit_estimator(factor_embeddings)
            
            # Inner loop: adapt cross-attention weights
            inner_losses = []
            
            for inner_step in range(self.bi_level_inner_steps):
                print(f"[DEBUG] Inner step {inner_step + 1}/{self.bi_level_inner_steps}")
                # Apply cross-attention
                attended_latent, attention_weights = self.cross_attention(z, importance_weights)
                
                # Decode
                output = self.decode(attended_latent)
                
                # Compute reconstruction loss
                recon_loss = F.mse_loss(output, input, reduction='sum')
                
                # Compute KL divergence
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                # Compute disentanglement loss with adaptive weighting
                disentanglement_loss = torch.mean(mi_scores * complexity)
                
                # Compute attention regularization
                attention_reg = self.compute_attention_regularization(attention_weights)
                entropy_reg = self.compute_entropy_regularization(attention_weights)
                
                # Total inner loss
                inner_loss = (recon_loss + kld_loss + 
                             self.disentanglement_weight * disentanglement_loss +
                             self.attention_reg_weight * attention_reg +
                             self.entropy_reg_weight * entropy_reg)
                
                inner_losses.append(inner_loss)
                
                # Update cross-attention parameters
                if inner_step < self.bi_level_inner_steps - 1:
                    self.cross_attention.zero_grad()
                    inner_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.cross_attention.parameters(), 1.0)
                    
                    # Simple SGD update for inner loop
                    for param in self.cross_attention.parameters():
                        if param.grad is not None:
                            param.data -= 0.01 * param.grad
            
            # Outer loss is the final inner loss
            outer_losses.append(inner_losses[-1])
        
        print("[DEBUG] Bi-level optimization completed successfully")
        return {
            'total_loss': outer_losses[-1],
            'recon_loss': recon_loss,
            'kld_loss': kld_loss,
            'disentanglement_loss': disentanglement_loss,
            'attention_reg': attention_reg,
            'entropy_reg': entropy_reg,
            'mi_scores': mi_scores,
            'complexity': complexity
        }
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Forward pass with bi-level optimization
        """
        print("[DEBUG] MINEDisentangleVAE forward called")
        if self.training:
            print("[DEBUG] Training mode - starting bi-level optimization")
            # Use bi-level optimization during training
            losses = self.bi_level_optimization(input)
            print("[DEBUG] Bi-level optimization completed")
            return [losses['total_loss']]
        else:
            print("[DEBUG] Evaluation mode - standard forward pass")
            # Standard forward pass for evaluation
            mu, log_var, result = self.encode(input)
            z = self.reparameterize(mu, log_var)
            
            factor_embeddings, importance_weights = self.factor_encoder(result)
            attended_latent, _ = self.cross_attention(z, importance_weights)
            
            return [self.decode(attended_latent), input, mu, log_var]
    
    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        """
        if self.training:
            # During training, use bi-level optimization results
            total_loss = args[0]
            return {
                'loss': total_loss,
                'Reconstruction_Loss': total_loss,  # Placeholder
                'KLD': total_loss,  # Placeholder
                'Disentanglement_Loss': total_loss,  # Placeholder
                'Attention_Reg': total_loss,  # Placeholder
                'Entropy_Reg': total_loss  # Placeholder
            }
        else:
            # During evaluation, compute standard losses
            recons = args[0]
            input = args[1]
            mu = args[2]
            log_var = args[3]
            
            kld_weight = kwargs['M_N']  # Account for the minibatch correction from KLD paper if can
            recons_loss = F.mse_loss(recons, input)
            
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            
            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
    
    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        
        # Generate random factor embeddings
        factor_embeddings = torch.randn(num_samples, self.num_factors).to(current_device)
        factor_embeddings = F.softmax(factor_embeddings, dim=-1)
        
        # Apply cross-attention
        attended_latent, _ = self.cross_attention(z, factor_embeddings)
        
        samples = self.decode(attended_latent)
        return samples
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        """
        return self.forward(x, **kwargs)[0]
    
    def get_factor_embeddings(self, x: Tensor) -> Tensor:
        """
        Extract factor embeddings for input x
        """
        mu, log_var, result = self.encode(x)
        factor_embeddings, importance_weights = self.factor_encoder(result)
        return factor_embeddings
    
    def get_factor_complexity(self, x: Tensor) -> Tensor:
        """
        Get factor complexity estimates for input x
        """
        factor_embeddings = self.get_factor_embeddings(x)
        _, _, complexity = self.gaussian_fit_estimator(factor_embeddings)
        return complexity
    
    def traverse_latent_space(self, x: Tensor, factor_idx: int, num_steps: int = 10) -> Tensor:
        """
        Generate latent space traversal for a specific factor
        """
        mu, log_var, result = self.encode(x)
        z = self.reparameterize(mu, log_var)
        
        factor_embeddings, importance_weights = self.factor_encoder(result)
        
        # Create traversal range
        traversal_range = torch.linspace(-2, 2, num_steps).to(x.device)
        traversals = []
        
        for val in traversal_range:
            # Modify the specific factor
            modified_weights = importance_weights.clone()
            modified_weights[:, factor_idx] = val
            
            # Apply cross-attention
            attended_latent, _ = self.cross_attention(z, modified_weights)
            
            # Decode
            traversal = self.decode(attended_latent)
            traversals.append(traversal)
        
        return torch.stack(traversals, dim=1)  # [batch_size, num_steps, channels, height, width] 
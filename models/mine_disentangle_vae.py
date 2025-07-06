import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math
import numpy as np
from typing import Tuple, Optional


class MutualInformationEstimator(nn.Module):
    """
    Mutual Information Neural Estimator (MINE) implementation
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(MutualInformationEstimator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Estimate mutual information between x and y
        """
        # Concatenate x and y
        xy = torch.cat([x, y], dim=1)
        
        # Forward pass through the network
        t_xy = self.network(xy)
        
        # Sample y from marginal distribution (shuffle along batch dimension)
        y_shuffled = y[torch.randperm(y.size(0))]
        xy_shuffled = torch.cat([x, y_shuffled], dim=1)
        t_x_y = self.network(xy_shuffled)
        
        # MINE estimate: E[t_xy] - log(E[exp(t_x_y)])
        mi_estimate = torch.mean(t_xy) - torch.log(torch.mean(torch.exp(t_x_y)))
        
        return mi_estimate


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for dynamic capacity allocation
    """
    def __init__(self, latent_dim: int, num_factors: int, attention_dim: int = 128):
        super(CrossAttention, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_factors = num_factors
        self.attention_dim = attention_dim
        
        # Query, Key, Value projections
        self.query_proj = nn.Linear(latent_dim, attention_dim)
        self.key_proj = nn.Linear(num_factors, attention_dim)
        self.value_proj = nn.Linear(num_factors, attention_dim)
        
        # Output projection
        self.output_proj = nn.Linear(attention_dim, latent_dim)
        
        # Attention scaling factor
        self.scale = math.sqrt(attention_dim)
        
    def forward(self, latent_codes: Tensor, factor_embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute cross-attention between latent codes and factor embeddings
        
        Args:
            latent_codes: [batch_size, latent_dim]
            factor_embeddings: [batch_size, num_factors]
            
        Returns:
            attended_latent: [batch_size, latent_dim]
            attention_weights: [batch_size, latent_dim, num_factors]
        """
        batch_size = latent_codes.size(0)
        
        # Project to query, key, value
        Q = self.query_proj(latent_codes)  # [batch_size, attention_dim]
        K = self.key_proj(factor_embeddings)  # [batch_size, attention_dim]
        V = self.value_proj(factor_embeddings)  # [batch_size, attention_dim]
        
        # Reshape for attention computation
        Q = Q.unsqueeze(1)  # [batch_size, 1, attention_dim]
        K = K.unsqueeze(1)  # [batch_size, 1, attention_dim]
        V = V.unsqueeze(1)  # [batch_size, 1, attention_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, 1, 1]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_values = torch.matmul(attention_weights, V)  # [batch_size, 1, attention_dim]
        attended_values = attended_values.squeeze(1)  # [batch_size, attention_dim]
        
        # Project back to latent dimension
        attended_latent = self.output_proj(attended_values)
        
        # Expand attention weights for regularization
        attention_weights = attention_weights.expand(batch_size, self.latent_dim, self.num_factors)
        
        return attended_latent, attention_weights


class FactorEncoder(nn.Module):
    """
    Encoder for generative factors
    """
    def __init__(self, input_dim: int, factor_dim: int, num_factors: int):
        super(FactorEncoder, self).__init__()
        
        self.factor_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, factor_dim),
                nn.ReLU(),
                nn.Linear(factor_dim, factor_dim)
            ) for _ in range(num_factors)
        ])
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode input to factor embeddings
        """
        factor_embeddings = []
        for encoder in self.factor_encoders:
            factor_emb = encoder(x)
            factor_embeddings.append(factor_emb)
        
        return torch.stack(factor_embeddings, dim=1)  # [batch_size, num_factors, factor_dim]


class MINEDisentangleVAE(BaseVAE):
    """
    MINE-based Disentanglement VAE with Cross-Attention
    Implements the framework described in the abstract
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
                 **kwargs) -> None:
        super(MINEDisentangleVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_factors = num_factors
        self.factor_dim = factor_dim
        self.disentanglement_weight = disentanglement_weight
        self.attention_reg_weight = attention_reg_weight
        self.entropy_reg_weight = entropy_reg_weight
        
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
        
        # Factor encoder
        self.factor_encoder = FactorEncoder(
            input_dim=hidden_dims[-1] * 4,
            factor_dim=factor_dim,
            num_factors=num_factors
        )
        
        # Cross-attention mechanism
        self.cross_attention = CrossAttention(
            latent_dim=latent_dim,
            num_factors=num_factors,
            attention_dim=attention_dim
        )
        
        # MINE estimators for each factor
        self.mine_estimators = nn.ModuleList([
            MutualInformationEstimator(
                input_dim=latent_dim + factor_dim,
                hidden_dim=mine_hidden_dim
            ) for _ in range(num_factors)
        ])
        
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
            
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encode input to latent representation and factor embeddings
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        
        # Encode to latent space
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        # Encode to factor embeddings
        factor_embeddings = self.factor_encoder(result)  # [batch_size, num_factors, factor_dim]
        
        return [mu, log_var, factor_embeddings]
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representation to image
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def compute_mutual_information(self, latent_codes: Tensor, factor_embeddings: Tensor) -> Tensor:
        """
        Compute mutual information between latent codes and each factor
        """
        batch_size = latent_codes.size(0)
        mi_scores = []
        
        for i in range(self.num_factors):
            # Get factor embedding for this factor
            factor_emb = factor_embeddings[:, i, :]  # [batch_size, factor_dim]
            
            # Compute mutual information using MINE
            mi_score = self.mine_estimators[i](latent_codes, factor_emb)
            mi_scores.append(mi_score)
        
        return torch.stack(mi_scores)  # [num_factors]
    
    def compute_attention_regularization(self, attention_weights: Tensor) -> Tensor:
        """
        Compute L2 regularization on attention weights
        """
        return torch.mean(torch.norm(attention_weights, p=2, dim=-1))
    
    def compute_entropy_regularization(self, attention_weights: Tensor) -> Tensor:
        """
        Compute entropy-based regularization on attention weights
        """
        # Compute entropy of attention distribution
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        return torch.mean(entropy)
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # Encode
        mu, log_var, factor_embeddings = self.encode(input)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Apply cross-attention for dynamic capacity allocation
        attended_z, attention_weights = self.cross_attention(z, factor_embeddings.mean(dim=-1))
        
        # Decode
        recons = self.decode(attended_z)
        
        return [recons, input, mu, log_var, factor_embeddings, attention_weights]
    
    def loss_function(self, *args, **kwargs) -> dict:
        """
        Compute the complete loss function including disentanglement objectives
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        factor_embeddings = args[4]
        attention_weights = args[5]
        
        kld_weight = kwargs['M_N']
        
        # Reconstruction loss
        recons_loss = F.mse_loss(recons, input)
        
        # KL divergence loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        # Mutual information estimation for disentanglement
        z = self.reparameterize(mu, log_var)
        mi_scores = self.compute_mutual_information(z, factor_embeddings)
        
        # Disentanglement loss: minimize mutual information between factors
        disentanglement_loss = torch.mean(mi_scores)
        
        # Attention regularization
        attention_l2_reg = self.compute_attention_regularization(attention_weights)
        attention_entropy_reg = self.compute_entropy_regularization(attention_weights)
        
        # Total loss
        total_loss = (recons_loss + 
                     kld_weight * kld_loss + 
                     self.disentanglement_weight * disentanglement_loss +
                     self.attention_reg_weight * attention_l2_reg +
                     self.entropy_reg_weight * attention_entropy_reg)
        
        return {
            'loss': total_loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach(),
            'Disentanglement_Loss': disentanglement_loss.detach(),
            'Attention_L2_Reg': attention_l2_reg.detach(),
            'Attention_Entropy_Reg': attention_entropy_reg.detach(),
            'MI_Scores': mi_scores.detach()
        }
    
    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Sample from the disentangled latent space
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        
        # Create dummy factor embeddings for sampling
        dummy_factors = torch.randn(num_samples, self.num_factors).to(current_device)
        
        # Apply cross-attention
        attended_z, _ = self.cross_attention(z, dummy_factors)
        
        samples = self.decode(attended_z)
        return samples
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Generate reconstruction
        """
        return self.forward(x)[0]
    
    def get_factor_embeddings(self, x: Tensor) -> Tensor:
        """
        Extract factor embeddings for analysis
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        return self.factor_encoder(result)
    
    def traverse_latent_space(self, x: Tensor, factor_idx: int, num_steps: int = 10) -> Tensor:
        """
        Perform latent space traversal for a specific factor
        """
        mu, log_var, factor_embeddings = self.encode(x)
        z = self.reparameterize(mu, log_var)
        
        # Create traversal range
        z_min, z_max = z.min(dim=0)[0], z.max(dim=0)[0]
        traversal_range = torch.linspace(z_min[factor_idx], z_max[factor_idx], num_steps)
        
        traversals = []
        for val in traversal_range:
            z_traversed = z.clone()
            z_traversed[:, factor_idx] = val
            
            # Apply cross-attention
            attended_z, _ = self.cross_attention(z_traversed, factor_embeddings.mean(dim=-1))
            
            # Decode
            sample = self.decode(attended_z)
            traversals.append(sample)
        
        return torch.cat(traversals, dim=0) 
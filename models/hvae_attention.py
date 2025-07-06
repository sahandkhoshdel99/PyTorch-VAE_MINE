import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attention_output)
        return output, attention_weights


class HierarchicalVAEAttention(BaseVAE):
    """
    Hierarchical VAE with Attention mechanism
    Implements a multi-level hierarchical structure with attention for better feature learning
    """
    
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 num_hierarchical_levels: int = 3,
                 attention_heads: int = 8,
                 attention_dim: int = 256,
                 beta: float = 1.0,
                 **kwargs) -> None:
        super(HierarchicalVAEAttention, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_hierarchical_levels = num_hierarchical_levels
        self.beta = beta
        
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
                    nn.LeakyReLU(),
                    nn.Dropout2d(0.1)
                )
            )
            in_channels_encoder = h_dim
            
        self.encoder = nn.Sequential(*modules)
        
        # Attention mechanism for feature aggregation
        self.attention = MultiHeadAttention(
            d_model=hidden_dims[-1] * 4,
            num_heads=attention_heads,
            dropout=0.1
        )
        
        # Hierarchical latent variables
        self.hierarchical_mus = nn.ModuleList([
            nn.Linear(hidden_dims[-1] * 4, latent_dim // num_hierarchical_levels)
            for _ in range(num_hierarchical_levels)
        ])
        
        self.hierarchical_log_vars = nn.ModuleList([
            nn.Linear(hidden_dims[-1] * 4, latent_dim // num_hierarchical_levels)
            for _ in range(num_hierarchical_levels)
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
                    nn.LeakyReLU(),
                    nn.Dropout2d(0.1)
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
        
        # Additional attention for decoder
        self.decoder_attention = MultiHeadAttention(
            d_model=hidden_dims[-1] * 4,
            num_heads=attention_heads,
            dropout=0.1
        )
        
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input through hierarchical levels with attention
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        
        # Apply attention to features
        result_reshaped = result.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.attention(result_reshaped, result_reshaped, result_reshaped)
        attended_features = attended_features.squeeze(1)
        
        # Hierarchical encoding
        mus = []
        log_vars = []
        
        for i in range(self.num_hierarchical_levels):
            mu = self.hierarchical_mus[i](attended_features)
            log_var = self.hierarchical_log_vars[i](attended_features)
            mus.append(mu)
            log_vars.append(log_var)
            
        return mus + log_vars
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Decodes the hierarchical latent representation
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        
        # Apply attention in decoder
        result_flat = torch.flatten(result, start_dim=1)
        result_reshaped = result_flat.unsqueeze(1)
        attended_result, _ = self.decoder_attention(result_reshaped, result_reshaped, result_reshaped)
        attended_result = attended_result.squeeze(1)
        result = attended_result.view(-1, 512, 2, 2)
        
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
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # Encode to get hierarchical parameters
        encoded = self.encode(input)
        num_levels = self.num_hierarchical_levels
        
        mus = encoded[:num_levels]
        log_vars = encoded[num_levels:]
        
        # Reparameterize each level
        zs = []
        for mu, log_var in zip(mus, log_vars):
            z = self.reparameterize(mu, log_var)
            zs.append(z)
        
        # Concatenate all hierarchical latents
        z = torch.cat(zs, dim=1)
        
        # Decode
        recons = self.decode(z)
        
        return [recons, input] + mus + log_vars
    
    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the hierarchical VAE loss with attention regularization
        """
        recons = args[0]
        input = args[1]
        mus = args[2:2+self.num_hierarchical_levels]
        log_vars = args[2+self.num_hierarchical_levels:]
        
        kld_weight = kwargs['M_N']
        
        # Reconstruction loss
        recons_loss = F.mse_loss(recons, input)
        
        # Hierarchical KL divergence
        kld_loss = 0
        for mu, log_var in zip(mus, log_vars):
            kld_loss += torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        # Attention regularization (encourage diverse attention patterns)
        attention_reg = 0.1 * torch.mean(torch.abs(self.attention.w_q.weight)) + \
                       0.1 * torch.mean(torch.abs(self.attention.w_k.weight))
        
        # Total loss
        loss = recons_loss + self.beta * kld_weight * kld_loss + attention_reg
        
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach(),
            'Attention_Reg': attention_reg.detach()
        }
    
    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the hierarchical latent space
        """
        # Sample from each hierarchical level
        zs = []
        for i in range(self.num_hierarchical_levels):
            z_level = torch.randn(num_samples, self.latent_dim // self.num_hierarchical_levels)
            zs.append(z_level)
        
        z = torch.cat(zs, dim=1)
        z = z.to(current_device)
        
        samples = self.decode(z)
        return samples
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        """
        return self.forward(x)[0]
    
    def get_hierarchical_features(self, x: Tensor) -> List[Tensor]:
        """
        Extract hierarchical features for analysis
        """
        encoded = self.encode(x)
        num_levels = self.num_hierarchical_levels
        mus = encoded[:num_levels]
        log_vars = encoded[num_levels:]
        
        features = []
        for mu, log_var in zip(mus, log_vars):
            z = self.reparameterize(mu, log_var)
            features.append(z)
            
        return features 
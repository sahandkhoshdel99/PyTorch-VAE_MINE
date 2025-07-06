import torch
import torch.nn as nn
import numpy as np
import pytest
from models.mine_disentangle_vae import MINEDisentangleVAE, MutualInformationEstimator, CrossAttention, FactorEncoder


class TestMINEDisentangleVAE:
    """Test cases for MINE Disentangle VAE"""
    
    @pytest.fixture
    def model_params(self):
        return {
            'in_channels': 3,
            'latent_dim': 64,
            'num_factors': 5,
            'factor_dim': 16,
            'attention_dim': 64,
            'mine_hidden_dim': 64,
            'disentanglement_weight': 1.0,
            'attention_reg_weight': 0.1,
            'entropy_reg_weight': 0.05
        }
    
    @pytest.fixture
    def model(self, model_params):
        return MINEDisentangleVAE(**model_params)
    
    @pytest.fixture
    def sample_batch(self):
        batch_size = 8
        return torch.randn(batch_size, 3, 64, 64)
    
    def test_model_initialization(self, model_params):
        """Test that the model initializes correctly"""
        model = MINEDisentangleVAE(**model_params)
        assert model is not None
        assert model.latent_dim == model_params['latent_dim']
        assert model.num_factors == model_params['num_factors']
    
    def test_encoder_output(self, model, sample_batch):
        """Test encoder output shape"""
        mu, log_var, factor_embeddings = model.encode(sample_batch)
        
        batch_size = sample_batch.size(0)
        expected_latent_dim = model.latent_dim
        expected_factor_dim = model.factor_dim
        expected_num_factors = model.num_factors
        
        assert mu.shape == (batch_size, expected_latent_dim)
        assert log_var.shape == (batch_size, expected_latent_dim)
        assert factor_embeddings.shape == (batch_size, expected_num_factors, expected_factor_dim)
    
    def test_decoder_output(self, model, sample_batch):
        """Test decoder output shape"""
        # Create a sample latent code
        batch_size = sample_batch.size(0)
        z = torch.randn(batch_size, model.latent_dim)
        
        output = model.decode(z)
        
        assert output.shape == (batch_size, 3, 64, 64)
        assert torch.all(output >= -1) and torch.all(output <= 1)  # Tanh output
    
    def test_forward_pass(self, model, sample_batch):
        """Test complete forward pass"""
        results = model.forward(sample_batch)
        
        # Check output structure
        assert len(results) == 6  # [recons, input, mu, log_var, factor_embeddings, attention_weights]
        
        recons, input_data, mu, log_var, factor_embeddings, attention_weights = results
        
        batch_size = sample_batch.size(0)
        
        assert recons.shape == (batch_size, 3, 64, 64)
        assert input_data.shape == (batch_size, 3, 64, 64)
        assert mu.shape == (batch_size, model.latent_dim)
        assert log_var.shape == (batch_size, model.latent_dim)
        assert factor_embeddings.shape == (batch_size, model.num_factors, model.factor_dim)
        assert attention_weights.shape == (batch_size, model.latent_dim, model.num_factors)
    
    def test_loss_function(self, model, sample_batch):
        """Test loss function computation"""
        results = model.forward(sample_batch)
        loss_dict = model.loss_function(*results, M_N=1.0)
        
        # Check that all expected loss components are present
        expected_keys = ['loss', 'Reconstruction_Loss', 'KLD', 'Disentanglement_Loss', 
                        'Attention_L2_Reg', 'Attention_Entropy_Reg', 'MI_Scores']
        
        for key in expected_keys:
            assert key in loss_dict
            assert isinstance(loss_dict[key], torch.Tensor)
        
        # Check that total loss is a scalar
        assert loss_dict['loss'].dim() == 0
    
    def test_sampling(self, model):
        """Test sampling from the model"""
        num_samples = 16
        device = next(model.parameters()).device
        
        samples = model.sample(num_samples, device)
        
        assert samples.shape == (num_samples, 3, 64, 64)
        assert torch.all(samples >= -1) and torch.all(samples <= 1)
    
    def test_generate(self, model, sample_batch):
        """Test generation/reconstruction"""
        recons = model.generate(sample_batch)
        
        assert recons.shape == sample_batch.shape
        assert torch.all(recons >= -1) and torch.all(recons <= 1)
    
    def test_hierarchical_features(self, model, sample_batch):
        """Test hierarchical feature extraction"""
        features = model.get_hierarchical_features(sample_batch)
        
        batch_size = sample_batch.size(0)
        expected_features_per_level = model.latent_dim // model.num_hierarchical_levels
        
        assert len(features) == model.num_hierarchical_levels
        for feature in features:
            assert feature.shape == (batch_size, expected_features_per_level)


class TestMutualInformationEstimator:
    """Test cases for Mutual Information Estimator"""
    
    @pytest.fixture
    def mine_estimator(self):
        return MutualInformationEstimator(input_dim=32, hidden_dim=64)
    
    def test_mine_initialization(self, mine_estimator):
        """Test MINE estimator initialization"""
        assert mine_estimator is not None
        assert len(mine_estimator.network) > 0
    
    def test_mine_forward(self, mine_estimator):
        """Test MINE forward pass"""
        batch_size = 16
        x = torch.randn(batch_size, 16)  # 16 + 16 = 32 input dim
        y = torch.randn(batch_size, 16)
        
        mi_estimate = mine_estimator(x, y)
        
        assert isinstance(mi_estimate, torch.Tensor)
        assert mi_estimate.dim() == 0  # Scalar output


class TestCrossAttention:
    """Test cases for Cross Attention mechanism"""
    
    @pytest.fixture
    def cross_attention(self):
        return CrossAttention(latent_dim=64, num_factors=5, attention_dim=32)
    
    def test_cross_attention_initialization(self, cross_attention):
        """Test cross attention initialization"""
        assert cross_attention is not None
        assert cross_attention.latent_dim == 64
        assert cross_attention.num_factors == 5
        assert cross_attention.attention_dim == 32
    
    def test_cross_attention_forward(self, cross_attention):
        """Test cross attention forward pass"""
        batch_size = 8
        latent_codes = torch.randn(batch_size, 64)
        factor_embeddings = torch.randn(batch_size, 5)
        
        attended_latent, attention_weights = cross_attention(latent_codes, factor_embeddings)
        
        assert attended_latent.shape == (batch_size, 64)
        assert attention_weights.shape == (batch_size, 64, 5)
        
        # Check that attention weights sum to 1
        attention_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-6)


class TestFactorEncoder:
    """Test cases for Factor Encoder"""
    
    @pytest.fixture
    def factor_encoder(self):
        return FactorEncoder(input_dim=128, factor_dim=16, num_factors=5)
    
    def test_factor_encoder_initialization(self, factor_encoder):
        """Test factor encoder initialization"""
        assert factor_encoder is not None
        assert len(factor_encoder.factor_encoders) == 5
    
    def test_factor_encoder_forward(self, factor_encoder):
        """Test factor encoder forward pass"""
        batch_size = 8
        x = torch.randn(batch_size, 128)
        
        factor_embeddings = factor_encoder(x)
        
        assert factor_embeddings.shape == (batch_size, 5, 16)


def test_model_integration():
    """Integration test for the complete model"""
    # Create model
    model_params = {
        'in_channels': 3,
        'latent_dim': 64,
        'num_factors': 5,
        'factor_dim': 16,
        'attention_dim': 64,
        'mine_hidden_dim': 64,
        'disentanglement_weight': 1.0,
        'attention_reg_weight': 0.1,
        'entropy_reg_weight': 0.05
    }
    
    model = MINEDisentangleVAE(**model_params)
    
    # Create sample data
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    
    # Test complete pipeline
    model.train()
    results = model.forward(x)
    loss_dict = model.loss_function(*results, M_N=1.0)
    
    # Verify outputs
    assert loss_dict['loss'].requires_grad
    assert loss_dict['loss'].item() > 0
    
    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        recons = model.generate(x)
        assert recons.shape == x.shape


if __name__ == "__main__":
    # Run basic tests
    print("Running MINE Disentangle VAE tests...")
    
    # Test model initialization
    model_params = {
        'in_channels': 3,
        'latent_dim': 64,
        'num_factors': 5,
        'factor_dim': 16,
        'attention_dim': 64,
        'mine_hidden_dim': 64,
        'disentanglement_weight': 1.0,
        'attention_reg_weight': 0.1,
        'entropy_reg_weight': 0.05
    }
    
    model = MINEDisentangleVAE(**model_params)
    print("✓ Model initialization successful")
    
    # Test forward pass
    x = torch.randn(4, 3, 64, 64)
    results = model.forward(x)
    print("✓ Forward pass successful")
    
    # Test loss function
    loss_dict = model.loss_function(*results, M_N=1.0)
    print("✓ Loss function computation successful")
    print(f"  - Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  - Reconstruction loss: {loss_dict['Reconstruction_Loss'].item():.4f}")
    print(f"  - KLD: {loss_dict['KLD'].item():.4f}")
    print(f"  - Disentanglement loss: {loss_dict['Disentanglement_Loss'].item():.4f}")
    
    # Test sampling
    samples = model.sample(4, 0)
    print("✓ Sampling successful")
    
    print("All tests passed!") 
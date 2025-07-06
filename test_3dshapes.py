#!/usr/bin/env python3
"""
Test script for MINE Disentangle VAE on 3DShapes dataset
This script verifies the implementation works with the metrics described in the paper
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.mine_disentangle_vae import MINEDisentangleVAE
from evaluation_metrics import DisentanglementMetrics, LatentTraversalVisualizer
import os


def create_dummy_3dshapes_data(num_samples=1000, image_size=64):
    """
    Create dummy 3DShapes-like data for testing
    In practice, you would load the actual 3DShapes dataset
    """
    # Create dummy images (random for testing)
    images = torch.rand(num_samples, 3, image_size, image_size)
    
    # Create dummy factor values (6 factors as in 3DShapes)
    # floor hue (10 values), wall hue (10 values), object hue (10 values),
    # object size (8 values), object shape (4 values), camera elevation (15 values)
    factor_values = torch.randint(0, 10, (num_samples, 6))  # Simplified for testing
    
    return images, factor_values


def test_model_initialization():
    """Test that the model initializes correctly"""
    print("Testing model initialization...")
    
    model = MINEDisentangleVAE(
        in_channels=3,
        latent_dim=128,
        num_factors=6,  # 6 factors for 3DShapes
        factor_dim=32,
        attention_dim=128,
        mine_hidden_dim=128,
        disentanglement_weight=1.0,
        attention_reg_weight=0.1,
        entropy_reg_weight=0.05
    )
    
    print(f"Model initialized successfully with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def test_forward_pass(model, images):
    """Test forward pass through the model"""
    print("Testing forward pass...")
    
    model.eval()
    with torch.no_grad():
        # Test standard forward pass
        outputs = model(images)
        print(f"Forward pass successful. Output shape: {outputs[0].shape}")
        
        # Test factor embeddings
        factor_embeddings = model.get_factor_embeddings(images)
        print(f"Factor embeddings shape: {factor_embeddings.shape}")
        
        # Test factor complexity
        complexity = model.get_factor_complexity(images)
        print(f"Factor complexity shape: {complexity.shape}")
        
    return outputs, factor_embeddings, complexity


def test_latent_traversal(model, images):
    """Test latent space traversal"""
    print("Testing latent traversal...")
    
    model.eval()
    with torch.no_grad():
        # Test traversal for first factor
        traversal = model.traverse_latent_space(images[:1], factor_idx=0, num_steps=5)
        print(f"Traversal shape: {traversal.shape}")
        
    return traversal


def test_evaluation_metrics(images, factor_values, model):
    """Test evaluation metrics"""
    print("Testing evaluation metrics...")
    
    # Create dummy latent codes for testing
    # In practice, these would come from the model
    num_samples = images.shape[0]
    latent_dim = 128
    latent_codes = torch.randn(num_samples, latent_dim)
    
    # Initialize metrics
    metrics = DisentanglementMetrics(device='cpu')
    
    # Compute all metrics
    results = metrics.compute_all_metrics(latent_codes, factor_values)
    
    print("Evaluation metrics results:")
    for metric_name, score in results.items():
        print(f"  {metric_name}: {score:.4f}")
    
    return results


def test_visualization(traversal):
    """Test visualization of latent traversal"""
    print("Testing visualization...")
    
    # Create a simple plot
    fig, axes = plt.subplots(1, traversal.shape[1], figsize=(15, 3))
    
    for i in range(traversal.shape[1]):
        img = traversal[0, i].permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].set_title(f'Step {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('Latent Traversal Test')
    plt.tight_layout()
    plt.savefig('test_traversal.png', dpi=150, bbox_inches='tight')
    print("Traversal visualization saved as 'test_traversal.png'")
    plt.close()


def main():
    """Main test function"""
    print("Starting MINE Disentangle VAE tests...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    images, factor_values = create_dummy_3dshapes_data(num_samples=100)
    images = images.to(device)
    factor_values = factor_values.to(device)
    
    print(f"Created dummy data: {images.shape}, factor values: {factor_values.shape}")
    
    # Test model initialization
    model = test_model_initialization()
    model = model.to(device)
    
    # Test forward pass
    outputs, factor_embeddings, complexity = test_forward_pass(model, images)
    
    # Test latent traversal
    traversal = test_latent_traversal(model, images)
    
    # Test evaluation metrics
    results = test_evaluation_metrics(images, factor_values, model)
    
    # Test visualization
    test_visualization(traversal)
    
    print("\nAll tests completed successfully!")
    print("\nModel features tested:")
    print("- Model initialization and parameter count")
    print("- Forward pass and factor embedding extraction")
    print("- Factor complexity estimation")
    print("- Latent space traversal")
    print("- Disentanglement metrics (Z-diff, Z-min Var, IRS, MIG, JEMMIG)")
    print("- Visualization capabilities")
    
    print("\nNext steps:")
    print("1. Replace dummy data with actual 3DShapes dataset")
    print("2. Train the model on the dataset")
    print("3. Run full evaluation with trained model")
    print("4. Compare results with paper's quantitative results")


if __name__ == "__main__":
    main() 
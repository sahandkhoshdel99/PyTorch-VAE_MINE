import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import lpips
from typing import List, Dict, Tuple, Optional
import os


class FIDScore:
    """
    Fréchet Inception Distance (FID) Score implementation
    """
    def __init__(self, device='cuda'):
        self.device = device
        # Load pre-trained Inception v3 model
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        self.inception_model.fc = nn.Identity()  # Remove final classification layer
        self.inception_model.eval()
        self.inception_model.to(device)
        
    def get_inception_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from Inception v3 model
        """
        with torch.no_grad():
            features = self.inception_model(images)
        return features
    
    def calculate_fid(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Calculate FID score between real and generated images
        """
        # Extract features
        real_features = self.get_inception_features(real_images)
        fake_features = self.get_inception_features(fake_images)
        
        # Convert to numpy
        real_features = real_features.cpu().numpy()
        fake_features = fake_features.cpu().numpy()
        
        # Calculate mean and covariance
        mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_fake
        covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        return float(fid)


class LPIPS:
    """
    Learned Perceptual Image Patch Similarity (LPIPS) implementation
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
    def calculate_lpips(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Calculate LPIPS distance between real and generated images
        """
        with torch.no_grad():
            lpips_score = self.lpips_fn(real_images, fake_images)
        return lpips_score.mean().item()


class DisentanglementMetrics:
    """
    Disentanglement evaluation metrics
    """
    def __init__(self):
        pass
        
    def calculate_beta_vae_score(self, 
                                representations: np.ndarray, 
                                factors: np.ndarray,
                                num_train: int = 10000,
                                num_test: int = 5000,
                                num_factors: int = 10,
                                num_values_per_factor: int = 10) -> float:
        """
        Calculate Beta-VAE disentanglement score
        """
        # Train a classifier to predict factors from representations
        train_representations = representations[:num_train]
        train_factors = factors[:num_train]
        test_representations = representations[num_train:num_train + num_test]
        test_factors = factors[num_train:num_train + num_test]
        
        scores = []
        for factor_idx in range(num_factors):
            # Train classifier for this factor
            classifier = LogisticRegression(random_state=42)
            classifier.fit(train_representations, train_factors[:, factor_idx])
            
            # Predict on test set
            predictions = classifier.predict(test_representations)
            accuracy = accuracy_score(test_factors[:, factor_idx], predictions)
            scores.append(accuracy)
            
        return np.mean(scores)
    
    def calculate_factor_vae_score(self,
                                  representations: np.ndarray,
                                  factors: np.ndarray,
                                  num_train: int = 10000,
                                  num_test: int = 5000) -> float:
        """
        Calculate Factor-VAE disentanglement score
        """
        # Implementation of Factor-VAE metric
        # This is a simplified version - full implementation would be more complex
        return self.calculate_beta_vae_score(representations, factors, num_train, num_test)
    
    def calculate_dci_score(self,
                           representations: np.ndarray,
                           factors: np.ndarray) -> Dict[str, float]:
        """
        Calculate DCI (Disentanglement, Completeness, Informativeness) scores
        """
        # Calculate disentanglement
        disentanglement = self.calculate_beta_vae_score(representations, factors)
        
        # Calculate completeness (simplified)
        completeness = disentanglement  # In practice, this would be different
        
        # Calculate informativeness (simplified)
        informativeness = 1.0 - np.mean(np.var(representations, axis=0))
        
        return {
            'disentanglement': disentanglement,
            'completeness': completeness,
            'informativeness': informativeness
        }


class ReconstructionMetrics:
    """
    Reconstruction quality metrics
    """
    def __init__(self):
        pass
        
    def calculate_mse(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Calculate Mean Squared Error
        """
        return F.mse_loss(real_images, fake_images).item()
    
    def calculate_psnr(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio
        """
        mse = F.mse_loss(real_images, fake_images)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()
    
    def calculate_ssim(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Calculate Structural Similarity Index
        """
        # Simplified SSIM implementation
        # In practice, you might want to use a more robust implementation
        mu_x = real_images.mean()
        mu_y = fake_images.mean()
        sigma_x = real_images.var()
        sigma_y = fake_images.var()
        sigma_xy = ((real_images - mu_x) * (fake_images - mu_y)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        
        return ssim.item()


class VAEEvaluator:
    """
    Comprehensive VAE evaluation class
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.fid_scorer = FIDScore(device)
        self.lpips_scorer = LPIPS(device)
        self.disentanglement_metrics = DisentanglementMetrics()
        self.reconstruction_metrics = ReconstructionMetrics()
        
    def evaluate_model(self, 
                      model: nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      num_samples: int = 1000) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        """
        model.eval()
        
        real_images = []
        fake_images = []
        representations = []
        factors = []
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                if len(real_images) * data.size(0) >= num_samples:
                    break
                    
                data = data.to(self.device)
                
                # Generate reconstructions
                if hasattr(model, 'generate'):
                    fake_data = model.generate(data)
                else:
                    fake_data = model(data)[0]
                
                # Store images
                real_images.append(data.cpu())
                fake_images.append(fake_data.cpu())
                
                # Extract representations if available
                if hasattr(model, 'encode'):
                    encoded = model.encode(data)
                    if isinstance(encoded, list):
                        # For models that return [mu, log_var, ...]
                        mu = encoded[0]
                        representations.append(mu.cpu().numpy())
                    else:
                        representations.append(encoded.cpu().numpy())
                
                # Store factors (assuming labels are factors)
                factors.append(labels.numpy())
        
        # Concatenate all batches
        real_images = torch.cat(real_images, dim=0)[:num_samples]
        fake_images = torch.cat(fake_images, dim=0)[:num_samples]
        
        if representations:
            representations = np.concatenate(representations, axis=0)[:num_samples]
        if factors:
            factors = np.concatenate(factors, axis=0)[:num_samples]
        
        # Calculate metrics
        metrics = {}
        
        # Reconstruction metrics
        metrics['mse'] = self.reconstruction_metrics.calculate_mse(real_images, fake_images)
        metrics['psnr'] = self.reconstruction_metrics.calculate_psnr(real_images, fake_images)
        metrics['ssim'] = self.reconstruction_metrics.calculate_ssim(real_images, fake_images)
        
        # Perceptual metrics
        metrics['fid'] = self.fid_scorer.calculate_fid(real_images, fake_images)
        metrics['lpips'] = self.lpips_scorer.calculate_lpips(real_images, fake_images)
        
        # Disentanglement metrics (if factors are available)
        if len(factors) > 0 and len(representations) > 0:
            dci_scores = self.disentanglement_metrics.calculate_dci_score(representations, factors)
            metrics.update(dci_scores)
            
            beta_vae_score = self.disentanglement_metrics.calculate_beta_vae_score(
                representations, factors
            )
            metrics['beta_vae_score'] = beta_vae_score
        
        return metrics
    
    def evaluate_latent_traversal(self,
                                 model: nn.Module,
                                 dataloader: torch.utils.data.DataLoader,
                                 num_factors: int = 10,
                                 num_steps: int = 10) -> Dict[str, torch.Tensor]:
        """
        Evaluate latent space traversal quality
        """
        model.eval()
        
        traversal_results = {}
        
        with torch.no_grad():
            # Get a sample batch
            data, _ = next(iter(dataloader))
            data = data.to(self.device)
            
            # Perform traversal for each factor
            for factor_idx in range(num_factors):
                if hasattr(model, 'traverse_latent_space'):
                    traversals = model.traverse_latent_space(data, factor_idx, num_steps)
                    traversal_results[f'factor_{factor_idx}'] = traversals.cpu()
        
        return traversal_results
    
    def save_evaluation_results(self, 
                               metrics: Dict[str, float], 
                               save_path: str):
        """
        Save evaluation results to file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.6f}\n")
        
        print(f"Evaluation results saved to {save_path}")


def calculate_mutual_information(latent_codes: torch.Tensor, 
                                factors: torch.Tensor) -> torch.Tensor:
    """
    Calculate mutual information between latent codes and factors
    """
    # Simplified mutual information calculation
    # In practice, you might want to use more sophisticated estimators
    
    batch_size = latent_codes.size(0)
    latent_dim = latent_codes.size(1)
    num_factors = factors.size(1)
    
    mi_scores = torch.zeros(num_factors)
    
    for i in range(num_factors):
        # Calculate correlation between latent codes and factor i
        factor_i = factors[:, i:i+1]
        
        # Calculate correlation matrix
        latent_norm = (latent_codes - latent_codes.mean(dim=0)) / (latent_codes.std(dim=0) + 1e-8)
        factor_norm = (factor_i - factor_i.mean(dim=0)) / (factor_i.std(dim=0) + 1e-8)
        
        correlation = torch.mean(latent_norm * factor_norm, dim=0)
        
        # Use correlation as a proxy for mutual information
        mi_scores[i] = torch.mean(torch.abs(correlation))
    
    return mi_scores 
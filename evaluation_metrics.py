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
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
import math
import matplotlib.pyplot as plt
import seaborn as sns


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
    Implementation of disentanglement metrics as described in the research paper:
    - Z-diff: Measures sensitivity of latent variables to changes in generative factors
    - Z-min Var: Measures variance and separation across dimensions
    - IRS: Interventional Robustness Score
    - MIG: Mutual Information Gap
    - JEMMIG: Joint Entropy Minus Information Gap
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def compute_z_diff(self, latent_codes: torch.Tensor, factor_values: torch.Tensor) -> float:
        """
        Compute Z-diff metric: measures sensitivity of latent variables to changes in generative factors
        
        Args:
            latent_codes: [num_samples, latent_dim]
            factor_values: [num_samples, num_factors]
            
        Returns:
            z_diff_score: Higher is better
        """
        num_samples, latent_dim = latent_codes.shape
        num_factors = factor_values.shape[1]
        
        # Compute differences in latent space for each factor
        z_diff_scores = []
        
        for factor_idx in range(num_factors):
            # Get unique values for this factor
            unique_values = torch.unique(factor_values[:, factor_idx])
            
            factor_diffs = []
            for val in unique_values:
                # Find samples with this factor value
                mask = factor_values[:, factor_idx] == val
                if torch.sum(mask) > 1:
                    # Compute variance across latent dimensions for this factor value
                    latent_var = torch.var(latent_codes[mask], dim=0)
                    factor_diffs.append(torch.mean(latent_var))
            
            if factor_diffs:
                z_diff_scores.append(torch.mean(torch.stack(factor_diffs)))
        
        return torch.mean(torch.stack(z_diff_scores)).item()
    
    def compute_z_min_var(self, latent_codes: torch.Tensor, factor_values: torch.Tensor) -> float:
        """
        Compute Z-min Var metric: measures variance and separation across dimensions
        
        Args:
            latent_codes: [num_samples, latent_dim]
            factor_values: [num_samples, num_factors]
            
        Returns:
            z_min_var_score: Lower is better
        """
        num_samples, latent_dim = latent_codes.shape
        num_factors = factor_values.shape[1]
        
        # For each factor, find the latent dimension with minimum variance
        min_variances = []
        
        for factor_idx in range(num_factors):
            # Get unique values for this factor
            unique_values = torch.unique(factor_values[:, factor_idx])
            
            factor_variances = []
            for val in unique_values:
                # Find samples with this factor value
                mask = factor_values[:, factor_idx] == val
                if torch.sum(mask) > 1:
                    # Compute variance across latent dimensions for this factor value
                    latent_var = torch.var(latent_codes[mask], dim=0)
                    factor_variances.append(torch.min(latent_var))
            
            if factor_variances:
                min_variances.append(torch.mean(torch.stack(factor_variances)))
        
        return torch.mean(torch.stack(min_variances)).item()
    
    def compute_irs(self, latent_codes: torch.Tensor, factor_values: torch.Tensor) -> float:
        """
        Compute Interventional Robustness Score (IRS)
        Quantifies how well each latent dimension uniquely corresponds to a generative factor
        
        Args:
            latent_codes: [num_samples, latent_dim]
            factor_values: [num_samples, num_factors]
            
        Returns:
            irs_score: Higher is better
        """
        num_samples, latent_dim = latent_codes.shape
        num_factors = factor_values.shape[1]
        
        # Train classifiers to predict factors from latent codes
        irs_scores = []
        
        for factor_idx in range(num_factors):
            # Convert factor values to discrete classes
            unique_values = torch.unique(factor_values[:, factor_idx])
            factor_classes = torch.zeros(num_samples, dtype=torch.long)
            
            for i, val in enumerate(unique_values):
                mask = factor_values[:, factor_idx] == val
                factor_classes[mask] = i
            
            # Train random forest classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(latent_codes.cpu().numpy(), factor_classes.cpu().numpy())
            
            # Compute accuracy
            predictions = clf.predict(latent_codes.cpu().numpy())
            accuracy = accuracy_score(factor_classes.cpu().numpy(), predictions)
            irs_scores.append(accuracy)
        
        return np.mean(irs_scores)
    
    def compute_mig(self, latent_codes: torch.Tensor, factor_values: torch.Tensor) -> float:
        """
        Compute Mutual Information Gap (MIG)
        Measures the degree of exclusivity between latent dimensions and factors
        
        Args:
            latent_codes: [num_samples, latent_dim]
            factor_values: [num_samples, num_factors]
            
        Returns:
            mig_score: Higher is better
        """
        num_samples, latent_dim = latent_codes.shape
        num_factors = factor_values.shape[1]
        
        # Compute mutual information between each latent dimension and each factor
        mi_matrix = torch.zeros(latent_dim, num_factors)
        
        for i in range(latent_dim):
            for j in range(num_factors):
                # Compute mutual information using histogram-based estimation
                mi_val = self._compute_mutual_information(
                    latent_codes[:, i], factor_values[:, j]
                )
                mi_matrix[i, j] = mi_val
        
        # For each factor, find the two highest MI values
        mig_scores = []
        for factor_idx in range(num_factors):
            factor_mi = mi_matrix[:, factor_idx]
            sorted_mi, _ = torch.sort(factor_mi, descending=True)
            
            if len(sorted_mi) >= 2:
                # MIG = (MI_1 - MI_2) / H(factor)
                mi_gap = sorted_mi[0] - sorted_mi[1]
                
                # Compute factor entropy
                factor_entropy = self._compute_entropy(factor_values[:, factor_idx])
                
                if factor_entropy > 0:
                    mig_scores.append(mi_gap / factor_entropy)
        
        return torch.mean(torch.stack(mig_scores)).item() if mig_scores else 0.0
    
    def compute_jemmig(self, latent_codes: torch.Tensor, factor_values: torch.Tensor) -> float:
        """
        Compute Joint Entropy Minus Information Gap (JEMMIG)
        Extends MIG by incorporating joint entropy
        
        Args:
            latent_codes: [num_samples, latent_dim]
            factor_values: [num_samples, num_factors]
            
        Returns:
            jemmig_score: Higher is better
        """
        num_samples, latent_dim = latent_codes.shape
        num_factors = factor_values.shape[1]
        
        # Compute mutual information matrix
        mi_matrix = torch.zeros(latent_dim, num_factors)
        
        for i in range(latent_dim):
            for j in range(num_factors):
                mi_val = self._compute_mutual_information(
                    latent_codes[:, i], factor_values[:, j]
                )
                mi_matrix[i, j] = mi_val
        
        # Compute joint entropy of all factors
        joint_entropy = self._compute_joint_entropy(factor_values)
        
        # Compute MIG component
        mig_scores = []
        for factor_idx in range(num_factors):
            factor_mi = mi_matrix[:, factor_idx]
            sorted_mi, _ = torch.sort(factor_mi, descending=True)
            
            if len(sorted_mi) >= 2:
                mi_gap = sorted_mi[0] - sorted_mi[1]
                factor_entropy = self._compute_entropy(factor_values[:, factor_idx])
                
                if factor_entropy > 0:
                    mig_scores.append(mi_gap / factor_entropy)
        
        mig_component = torch.mean(torch.stack(mig_scores)) if mig_scores else 0.0
        
        # JEMMIG = MIG - joint_entropy
        jemmig_score = mig_component - joint_entropy
        
        return jemmig_score.item()
    
    def _compute_mutual_information(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute mutual information between two variables using histogram-based estimation
        """
        # Convert to numpy for easier processing
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Create histograms
        x_bins = np.histogram_bin_edges(x_np, bins='auto')
        y_bins = np.histogram_bin_edges(y_np, bins='auto')
        
        # Compute joint histogram
        joint_hist, _, _ = np.histogram2d(x_np, y_np, bins=[x_bins, y_bins])
        x_hist, _ = np.histogram(x_np, bins=x_bins)
        y_hist, _ = np.histogram(y_np, bins=y_bins)
        
        # Normalize to probabilities
        joint_prob = joint_hist / np.sum(joint_hist)
        x_prob = x_hist / np.sum(x_hist)
        y_prob = y_hist / np.sum(y_hist)
        
        # Compute mutual information
        mi = 0.0
        for i in range(joint_prob.shape[0]):
            for j in range(joint_prob.shape[1]):
                if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (x_prob[i] * y_prob[j]))
        
        return mi
    
    def _compute_entropy(self, x: torch.Tensor) -> float:
        """
        Compute entropy of a variable
        """
        x_np = x.cpu().numpy()
        hist, _ = np.histogram(x_np, bins='auto')
        prob = hist / np.sum(hist)
        
        entropy = 0.0
        for p in prob:
            if p > 0:
                entropy -= p * np.log(p)
        
        return entropy
    
    def _compute_joint_entropy(self, factors: torch.Tensor) -> float:
        """
        Compute joint entropy of all factors
        """
        factors_np = factors.cpu().numpy()
        
        # Create multi-dimensional histogram
        hist, _ = np.histogramdd(factors_np, bins='auto')
        prob = hist / np.sum(hist)
        
        joint_entropy = 0.0
        for p in prob.flatten():
            if p > 0:
                joint_entropy -= p * np.log(p)
        
        return joint_entropy
    
    def compute_all_metrics(self, latent_codes: torch.Tensor, factor_values: torch.Tensor) -> Dict[str, float]:
        """
        Compute all disentanglement metrics
        
        Args:
            latent_codes: [num_samples, latent_dim]
            factor_values: [num_samples, num_factors]
            
        Returns:
            Dictionary containing all metric scores
        """
        metrics = {}
        
        metrics['Z-diff'] = self.compute_z_diff(latent_codes, factor_values)
        metrics['Z-min Var'] = self.compute_z_min_var(latent_codes, factor_values)
        metrics['IRS'] = self.compute_irs(latent_codes, factor_values)
        metrics['MIG'] = self.compute_mig(latent_codes, factor_values)
        metrics['JEMMIG'] = self.compute_jemmig(latent_codes, factor_values)
        
        return metrics
    
    def plot_metrics_comparison(self, results: Dict[str, Dict[str, float]], save_path: str = None):
        """
        Plot comparison of metrics across different models
        
        Args:
            results: Dictionary with model names as keys and metric dictionaries as values
            save_path: Path to save the plot
        """
        metrics = ['Z-diff', 'Z-min Var', 'IRS', 'MIG', 'JEMMIG']
        models = list(results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = [results[model][metric] for model in models]
                
                bars = axes[i].bar(models, values)
                axes[i].set_title(f'{metric}')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Color bars based on performance
                for j, bar in enumerate(bars):
                    if metric == 'Z-min Var':  # Lower is better
                        if values[j] == min(values):
                            bar.set_color('green')
                        else:
                            bar.set_color('red')
                    else:  # Higher is better
                        if values[j] == max(values):
                            bar.set_color('green')
                        else:
                            bar.set_color('red')
        
        # Remove extra subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


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
        self.disentanglement_metrics = DisentanglementMetrics(device)
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
            dci_scores = self.disentanglement_metrics.compute_all_metrics(representations, factors)
            metrics.update(dci_scores)
        
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


class LatentTraversalVisualizer:
    """
    Generate latent traversals for qualitative evaluation of disentanglement
    """
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        
    def generate_traversal(self, x: torch.Tensor, factor_idx: int, num_steps: int = 10) -> torch.Tensor:
        """
        Generate latent traversal for a specific factor
        
        Args:
            x: Input image [batch_size, channels, height, width]
            factor_idx: Index of factor to traverse
            num_steps: Number of traversal steps
            
        Returns:
            Traversal images [batch_size, num_steps, channels, height, width]
        """
        self.model.eval()
        with torch.no_grad():
            traversal = self.model.traverse_latent_space(x, factor_idx, num_steps)
        return traversal
    
    def plot_traversal(self, traversal: torch.Tensor, factor_name: str, save_path: str = None):
        """
        Plot latent traversal
        
        Args:
            traversal: Traversal images [batch_size, num_steps, channels, height, width]
            factor_name: Name of the factor being traversed
            save_path: Path to save the plot
        """
        batch_size, num_steps, channels, height, width = traversal.shape
        
        # Take first sample from batch
        traversal = traversal[0]  # [num_steps, channels, height, width]
        
        # Convert to numpy and transpose for plotting
        traversal_np = traversal.cpu().numpy()
        traversal_np = np.transpose(traversal_np, (0, 2, 3, 1))  # [num_steps, height, width, channels]
        
        # Create subplot grid
        fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2))
        
        for i in range(num_steps):
            if num_steps == 1:
                ax = axes
            else:
                ax = axes[i]
            
            ax.imshow(traversal_np[i])
            ax.set_title(f'Step {i+1}')
            ax.axis('off')
        
        plt.suptitle(f'Latent Traversal: {factor_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_all_traversals(self, x: torch.Tensor, factor_names: List[str], 
                               num_steps: int = 10, save_dir: str = None) -> Dict[str, torch.Tensor]:
        """
        Generate traversals for all factors
        
        Args:
            x: Input image
            factor_names: Names of all factors
            num_steps: Number of traversal steps
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of traversals for each factor
        """
        traversals = {}
        
        for i, factor_name in enumerate(factor_names):
            traversal = self.generate_traversal(x, i, num_steps)
            traversals[factor_name] = traversal
            
            if save_dir:
                save_path = f"{save_dir}/traversal_{factor_name}.png"
                self.plot_traversal(traversal, factor_name, save_path)
        
        return traversals


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
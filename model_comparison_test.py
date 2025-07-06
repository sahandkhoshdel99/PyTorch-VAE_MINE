#!/usr/bin/env python3
"""
Comprehensive model comparison test script
Compares MINE Disentangle VAE with all other models in the models folder
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
import importlib
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ModelComparator:
    def __init__(self, base_config_path: str = "configs/mine_disentangle_vae.yaml"):
        self.base_config_path = base_config_path
        self.results_dir = Path("model_comparison_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Get all available models
        self.models = self.get_available_models()
        
    def get_available_models(self) -> List[Dict]:
        """Get all available models from the models folder"""
        
        models_dir = Path("models")
        available_models = []
        
        # Define model configurations
        model_configs = {
            'vanilla_vae': {
                'name': 'VanillaVAE',
                'config_file': 'configs/vae.yaml',
                'description': 'Standard VAE baseline'
            },
            'beta_vae': {
                'name': 'BetaVAE',
                'config_file': 'configs/bhvae.yaml',
                'description': 'Beta-VAE with KL weighting'
            },
            'betatc_vae': {
                'name': 'BetaTCVAE',
                'config_file': 'configs/betatc_vae.yaml',
                'description': 'Beta-TC-VAE with total correlation penalty'
            },
            'dfc_vae': {
                'name': 'DFCVAE',
                'config_file': 'configs/dfc_vae.yaml',
                'description': 'Deep Feature Consistent VAE'
            },
            'dip_vae': {
                'name': 'DIPVAE',
                'config_file': 'configs/dip_vae.yaml',
                'description': 'DIP-VAE with covariance regularization'
            },
            'factor_vae': {
                'name': 'FactorVAE',
                'config_file': 'configs/factorvae.yaml',
                'description': 'Factor-VAE with factorized prior'
            },
            'gamma_vae': {
                'name': 'GammaVAE',
                'config_file': 'configs/gammavae.yaml',
                'description': 'Gamma-VAE with gamma prior'
            },
            'hvae': {
                'name': 'HVAE',
                'config_file': 'configs/hvae.yaml',
                'description': 'Hierarchical VAE'
            },
            'info_vae': {
                'name': 'InfoVAE',
                'config_file': 'configs/infovae.yaml',
                'description': 'Info-VAE with mutual information'
            },
            'iwae': {
                'name': 'IWAE',
                'config_file': 'configs/iwae.yaml',
                'description': 'Importance Weighted Autoencoder'
            },
            'joint_vae': {
                'name': 'JointVAE',
                'config_file': 'configs/joint_vae.yaml',
                'description': 'Joint VAE with continuous and discrete latents'
            },
            'logcosh_vae': {
                'name': 'LogCoshVAE',
                'config_file': 'configs/logcosh_vae.yaml',
                'description': 'Log-Cosh VAE'
            },
            'lvae': {
                'name': 'LVAE',
                'config_file': 'configs/lvae.yaml',
                'description': 'Ladder VAE'
            },
            'miwae': {
                'name': 'MIWAE',
                'config_file': 'configs/miwae.yaml',
                'description': 'Missing Data VAE'
            },
            'mssim_vae': {
                'name': 'MSSIMVAE',
                'config_file': 'configs/mssim_vae.yaml',
                'description': 'MSSIM-VAE with structural similarity'
            },
            'swae': {
                'name': 'SWAE',
                'config_file': 'configs/swae.yaml',
                'description': 'Sliced Wasserstein Autoencoder'
            },
            'vampvae': {
                'name': 'VampVAE',
                'config_file': 'configs/vampvae.yaml',
                'description': 'VampPrior VAE'
            },
            'vq_vae': {
                'name': 'VQVAE',
                'config_file': 'configs/vq_vae.yaml',
                'description': 'Vector Quantized VAE'
            },
            'wae_mmd': {
                'name': 'WAE_MMD',
                'config_file': 'configs/wae_mmd_rbf.yaml',
                'description': 'Wasserstein Autoencoder with MMD'
            }
        }
        
        # Check which models are actually available
        for model_key, config in model_configs.items():
            config_path = Path(config['config_file'])
            if config_path.exists():
                available_models.append({
                    'key': model_key,
                    'name': config['name'],
                    'config_file': config['config_file'],
                    'description': config['description']
                })
        
        # Add MINE Disentangle VAE
        available_models.append({
            'key': 'mine_disentangle_vae',
            'name': 'MINEDisentangleVAE',
            'config_file': 'configs/mine_disentangle_vae.yaml',
            'description': 'MINE Disentangle VAE (Our Method)'
        })
        
        print(f"📋 Found {len(available_models)} available models:")
        for model in available_models:
            print(f"   - {model['name']}: {model['description']}")
        
        return available_models
    
    def create_model_config(self, model_info: Dict, dataset_name: str) -> str:
        """Create a config file for a specific model and dataset"""
        
        config_path = Path(model_info['config_file'])
        
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return None
        
        # Load model-specific config
        with open(config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Update data parameters
        model_config['data_params'].update({
            'data_path': "Data/",
            'dataset_name': dataset_name,
            'train_batch_size': 64,
            'val_batch_size': 64,
            'patch_size': 64,
            'num_workers': 4
        })
        
        # Update trainer parameters for faster testing
        model_config['trainer_params'].update({
            'max_epochs': 10,  # Reduced for comparison
            'gpus': [0] if torch.cuda.is_available() else [0]
        })
        
        # Update logging parameters
        model_config['logging_params'].update({
            'name': f"{model_info['name']}_{dataset_name}_comparison"
        })
        
        # Save modified config
        output_config_path = self.results_dir / f"config_{model_info['key']}_{dataset_name}.yaml"
        with open(output_config_path, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)
        
        return str(output_config_path)
    
    def run_model_test(self, model_info: Dict, dataset_name: str) -> Dict:
        """Run a single model test"""
        
        print(f"\n{'='*60}")
        print(f"Testing {model_info['name']} on {dataset_name}")
        print(f"Description: {model_info['description']}")
        print(f"{'='*60}")
        
        # Create config file
        config_path = self.create_model_config(model_info, dataset_name)
        
        if not config_path:
            return {
                'model_name': model_info['name'],
                'model_key': model_info['key'],
                'dataset_name': dataset_name,
                'success': False,
                'error': 'Config file not found'
            }
        
        # Run the experiment
        try:
            result = subprocess.run([
                'python', 'run_enhanced.py',
                '--config', config_path
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            # Parse results
            metrics = self.parse_experiment_output(result.stdout, result.stderr)
            
            # Add experiment metadata
            metrics.update({
                'model_name': model_info['name'],
                'model_key': model_info['key'],
                'dataset_name': dataset_name,
                'config_path': config_path,
                'success': result.returncode == 0,
                'stdout': result.stdout[-1000:],  # Last 1000 chars
                'stderr': result.stderr[-1000:]   # Last 1000 chars
            })
            
            return metrics
            
        except subprocess.TimeoutExpired:
            print(f"❌ Model {model_info['name']} timed out")
            return {
                'model_name': model_info['name'],
                'model_key': model_info['key'],
                'dataset_name': dataset_name,
                'config_path': config_path,
                'success': False,
                'error': 'timeout'
            }
        except Exception as e:
            print(f"❌ Model {model_info['name']} failed: {e}")
            return {
                'model_name': model_info['name'],
                'model_key': model_info['key'],
                'dataset_name': dataset_name,
                'config_path': config_path,
                'success': False,
                'error': str(e)
            }
    
    def parse_experiment_output(self, stdout: str, stderr: str) -> Dict:
        """Parse experiment output to extract metrics"""
        
        metrics = {
            'dci_disentanglement': None,
            'dci_completeness': None,
            'dci_informativeness': None,
            'mig': None,
            'irs': None,
            'z_diff': None,
            'z_min_var': None,
            'jemmig': None,
            'final_loss': None,
            'final_reconstruction_loss': None,
            'final_kl_loss': None
        }
        
        # Parse stdout for metrics
        lines = stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for metric patterns
            if 'DCI-Disentanglement:' in line:
                try:
                    metrics['dci_disentanglement'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            elif 'DCI-Completeness:' in line:
                try:
                    metrics['dci_completeness'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            elif 'DCI-Informativeness:' in line:
                try:
                    metrics['dci_informativeness'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            elif 'MIG:' in line:
                try:
                    metrics['mig'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            elif 'IRS:' in line:
                try:
                    metrics['irs'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            elif 'Z-diff:' in line:
                try:
                    metrics['z_diff'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            elif 'Z-min Var:' in line:
                try:
                    metrics['z_min_var'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            elif 'JEMMIG:' in line:
                try:
                    metrics['jemmig'] = float(line.split(':')[1].strip())
                except:
                    pass
        
        return metrics
    
    def run_comparison(self, datasets: List[str] = None):
        """Run comparison tests on all models"""
        
        if datasets is None:
            datasets = ['3dshapes', 'dsprites']
        
        all_results = []
        
        for dataset_name in datasets:
            print(f"\n{'='*80}")
            print(f"Starting model comparison for {dataset_name}")
            print(f"{'='*80}")
            
            dataset_results = []
            
            for model_info in self.models:
                # Run model test
                result = self.run_model_test(model_info, dataset_name)
                
                dataset_results.append(result)
                all_results.append(result)
                
                # Save intermediate results
                self.save_results(dataset_results, f"{dataset_name}_model_comparison.json")
                
                print(f"✅ {model_info['name']} completed")
                if result.get('dci_disentanglement'):
                    print(f"   DCI-D: {result['dci_disentanglement']:.4f}")
        
        # Save all results
        self.save_results(all_results, "all_model_comparison.json")
        
        # Generate analysis
        self.analyze_comparison_results(all_results)
    
    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file"""
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def analyze_comparison_results(self, results: List[Dict]):
        """Analyze and visualize comparison results"""
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON ANALYSIS")
        print(f"{'='*80}")
        
        # Filter successful experiments
        successful_results = [r for r in results if r.get('success', False) and r.get('dci_disentanglement') is not None]
        
        if not successful_results:
            print("❌ No successful experiments found")
            return
        
        print(f"✅ {len(successful_results)} successful experiments analyzed")
        
        # Create visualizations
        self.create_comparison_visualizations(successful_results)
        
        # Generate ranking report
        self.generate_ranking_report(successful_results)
    
    def create_comparison_visualizations(self, results: List[Dict]):
        """Create comparison visualizations"""
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Create output directory
            viz_dir = self.results_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # 1. Model ranking by DCI-D
            self.plot_model_ranking(df, viz_dir)
            
            # 2. Dataset comparison
            self.plot_dataset_comparison(df, viz_dir)
            
            # 3. Metric comparison
            self.plot_metric_comparison(df, viz_dir)
            
            # 4. Performance heatmap
            self.plot_performance_heatmap(df, viz_dir)
            
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
    
    def plot_model_ranking(self, df: pd.DataFrame, viz_dir: Path):
        """Plot model ranking by DCI-D"""
        
        # Group by dataset and get best models
        datasets = df['dataset_name'].unique()
        
        fig, axes = plt.subplots(1, len(datasets), figsize=(15, 8))
        if len(datasets) == 1:
            axes = [axes]
        
        for i, dataset in enumerate(datasets):
            dataset_df = df[df['dataset_name'] == dataset]
            
            # Sort by DCI-D
            sorted_df = dataset_df.sort_values('dci_disentanglement', ascending=True)
            
            # Create horizontal bar plot
            bars = axes[i].barh(range(len(sorted_df)), sorted_df['dci_disentanglement'])
            
            # Color MINE Disentangle VAE differently
            colors = ['red' if 'MINE' in name else 'steelblue' for name in sorted_df['model_name']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            axes[i].set_yticks(range(len(sorted_df)))
            axes[i].set_yticklabels(sorted_df['model_name'], fontsize=10)
            axes[i].set_xlabel('DCI-Disentanglement')
            axes[i].set_title(f'Model Ranking - {dataset.upper()}')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels
            for j, (_, row) in enumerate(sorted_df.iterrows()):
                axes[i].text(row['dci_disentanglement'], j, f'{row["dci_disentanglement"]:.3f}', 
                           va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'model_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Model ranking plot saved")
    
    def plot_dataset_comparison(self, df: pd.DataFrame, viz_dir: Path):
        """Plot dataset comparison for each model"""
        
        # Get unique models
        models = df['model_name'].unique()
        
        # Create comparison plot
        metrics = ['dci_disentanglement', 'dci_completeness', 'dci_informativeness', 'mig', 'irs']
        metric_labels = ['DCI-D', 'DCI-C', 'DCI-I', 'MIG', 'IRS']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            if metric not in df.columns:
                continue
                
            ax = axes[i]
            
            # Pivot data for plotting
            pivot_data = df.pivot(index='model_name', columns='dataset_name', values=metric)
            
            # Create bar plot
            pivot_data.plot(kind='bar', ax=ax, alpha=0.7)
            
            ax.set_title(f'{label} Comparison')
            ax.set_ylabel('Score')
            ax.set_xlabel('Models')
            ax.legend(title='Dataset')
            ax.tick_params(axis='x', rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Dataset comparison plot saved")
    
    def plot_metric_comparison(self, df: pd.DataFrame, viz_dir: Path):
        """Plot metric comparison for each model"""
        
        # Select metrics to compare
        metrics = ['dci_disentanglement', 'dci_completeness', 'dci_informativeness', 'mig', 'irs']
        metric_labels = ['DCI-D', 'DCI-C', 'DCI-I', 'MIG', 'IRS']
        
        # Calculate mean scores across datasets
        mean_scores = df.groupby('model_name')[metrics].mean()
        
        # Create radar/spider plot
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
        
        # Number of metrics
        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(mean_scores)))
        
        for i, (model_name, scores) in enumerate(mean_scores.iterrows()):
            values = scores.values.tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison (All Metrics)', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("🕷️  Metric comparison plot saved")
    
    def plot_performance_heatmap(self, df: pd.DataFrame, viz_dir: Path):
        """Plot performance heatmap"""
        
        # Select metrics
        metrics = ['dci_disentanglement', 'dci_completeness', 'dci_informativeness', 'mig', 'irs']
        
        # Create pivot table
        pivot_data = df.pivot(index='model_name', columns='dataset_name', values='dci_disentanglement')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': 'DCI-Disentanglement'})
        
        plt.title('Model Performance Heatmap (DCI-Disentanglement)')
        plt.xlabel('Dataset')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("🔥 Performance heatmap saved")
    
    def generate_ranking_report(self, results: List[Dict]):
        """Generate ranking report"""
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        report = []
        report.append("=" * 80)
        report.append("MODEL COMPARISON RANKING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall ranking
        report.append("🏆 OVERALL RANKING BY DCI-DISENTANGLEMENT")
        report.append("-" * 50)
        
        overall_ranking = df.groupby('model_name')['dci_disentanglement'].mean().sort_values(ascending=False)
        
        for i, (model, score) in enumerate(overall_ranking.items(), 1):
            report.append(f"{i:2d}. {model:25s} DCI-D: {score:.4f}")
        
        report.append("")
        
        # Dataset-specific rankings
        for dataset in df['dataset_name'].unique():
            dataset_df = df[df['dataset_name'] == dataset]
            dataset_ranking = dataset_df.sort_values('dci_disentanglement', ascending=False)
            
            report.append(f"📊 RANKING FOR {dataset.upper()}")
            report.append("-" * 40)
            
            for i, (_, row) in enumerate(dataset_ranking.iterrows(), 1):
                report.append(f"{i:2d}. {row['model_name']:25s} DCI-D: {row['dci_disentanglement']:.4f}")
                if 'mig' in row and pd.notna(row['mig']):
                    report.append(f"    {'':25s} MIG: {row['mig']:.4f}")
            
            report.append("")
        
        # MINE Disentangle VAE performance
        mine_results = df[df['model_name'].str.contains('MINE')]
        if not mine_results.empty:
            report.append("🎯 MINE DISENTANGLE VAE PERFORMANCE")
            report.append("-" * 40)
            
            for _, row in mine_results.iterrows():
                report.append(f"Dataset: {row['dataset_name']}")
                report.append(f"DCI-D: {row['dci_disentanglement']:.4f}")
                if 'mig' in row and pd.notna(row['mig']):
                    report.append(f"MIG: {row['mig']:.4f}")
                if 'irs' in row and pd.notna(row['irs']):
                    report.append(f"IRS: {row['irs']:.4f}")
                report.append("")
        
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        with open(self.results_dir / 'ranking_report.txt', 'w') as f:
            f.write(report_text)
        
        # Print report
        print(report_text)
        
        print(f"\n📄 Ranking report saved to: {self.results_dir / 'ranking_report.txt'}")


def main():
    """Main function"""
    print("🔬 Starting Model Comparison Test")
    print("Comparing MINE Disentangle VAE with all available models")
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Run comparison on confirmed datasets
    datasets = ['3dshapes', 'dsprites']
    
    print(f"Datasets: {datasets}")
    print(f"Results will be saved to: {comparator.results_dir}")
    
    # Run comparison
    comparator.run_comparison(datasets)
    
    print(f"\n🎉 Model comparison completed!")
    print(f"Check results in: {comparator.results_dir}")


if __name__ == "__main__":
    main() 
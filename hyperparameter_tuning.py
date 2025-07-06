#!/usr/bin/env python3
"""
Hyperparameter tuning script for MINE Disentangle VAE
Focuses on DCI-D disentanglement metric optimization
"""

import os
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import itertools
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class HyperparameterTuner:
    def __init__(self, base_config_path: str = "configs/mine_disentangle_vae.yaml"):
        self.base_config_path = base_config_path
        self.results_dir = Path("hyperparameter_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def generate_hyperparameter_combinations(self) -> List[Dict]:
        """Generate hyperparameter combinations for tuning"""
        
        # Define hyperparameter search space
        hyperparams = {
            'disentanglement_weight': [0.1, 0.5, 1.0, 2.0, 5.0],
            'attention_reg_weight': [0.01, 0.05, 0.1, 0.2, 0.5],
            'entropy_reg_weight': [0.01, 0.05, 0.1, 0.2, 0.5],
            'latent_dim': [64, 128, 256],
            'factor_dim': [16, 32, 64],
            'attention_dim': [64, 128, 256],
            'mine_hidden_dim': [64, 128, 256],
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
            'batch_size': [32, 64, 128]
        }
        
        # Generate combinations (limit to reasonable number)
        combinations = []
        
        # Focus on most important parameters first
        key_params = {
            'disentanglement_weight': [0.1, 0.5, 1.0, 2.0, 5.0],
            'attention_reg_weight': [0.01, 0.1, 0.5],
            'entropy_reg_weight': [0.01, 0.1, 0.5],
            'latent_dim': [64, 128, 256],
            'learning_rate': [1e-4, 1e-3, 5e-3]
        }
        
        # Generate combinations
        param_names = list(key_params.keys())
        param_values = list(key_params.values())
        
        for combination in itertools.product(*param_values):
            config = dict(zip(param_names, combination))
            combinations.append(config)
        
        print(f"Generated {len(combinations)} hyperparameter combinations")
        return combinations
    
    def create_config_file(self, config: Dict, dataset_name: str, experiment_id: str) -> str:
        """Create a config file for a specific experiment"""
        
        # Create experiment-specific config
        experiment_config = self.base_config.copy()
        
        # Update model parameters
        experiment_config['model_params'].update({
            'latent_dim': config.get('latent_dim', 128),
            'factor_dim': config.get('factor_dim', 32),
            'attention_dim': config.get('attention_dim', 128),
            'mine_hidden_dim': config.get('mine_hidden_dim', 128),
            'disentanglement_weight': config.get('disentanglement_weight', 1.0),
            'attention_reg_weight': config.get('attention_reg_weight', 0.1),
            'entropy_reg_weight': config.get('entropy_reg_weight', 0.05)
        })
        
        # Update data parameters
        experiment_config['data_params'].update({
            'dataset_name': dataset_name,
            'train_batch_size': config.get('batch_size', 64),
            'val_batch_size': config.get('batch_size', 64)
        })
        
        # Update experiment parameters
        experiment_config['exp_params'].update({
            'LR': config.get('learning_rate', 0.001)
        })
        
        # Update logging parameters
        experiment_config['logging_params'].update({
            'name': f"MINE_Disentangle_{dataset_name}_{experiment_id}"
        })
        
        # Save config file
        config_path = self.results_dir / f"config_{dataset_name}_{experiment_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(experiment_config, f, default_flow_style=False)
        
        return str(config_path)
    
    def run_experiment(self, config_path: str, dataset_name: str, experiment_id: str) -> Dict:
        """Run a single experiment"""
        
        print(f"\n{'='*60}")
        print(f"Running experiment {experiment_id} on {dataset_name}")
        print(f"Config: {config_path}")
        print(f"{'='*60}")
        
        # Run the experiment
        try:
            result = subprocess.run([
                'python', 'run_enhanced.py',
                '--config', config_path,
                '--max_epochs', '10',  # Reduced for faster tuning
                '--gpus', '0' if torch.cuda.is_available() else '0'
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            # Parse results
            metrics = self.parse_experiment_output(result.stdout, result.stderr)
            
            # Add experiment metadata
            metrics.update({
                'experiment_id': experiment_id,
                'dataset_name': dataset_name,
                'config_path': config_path,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            })
            
            return metrics
            
        except subprocess.TimeoutExpired:
            print(f"❌ Experiment {experiment_id} timed out")
            return {
                'experiment_id': experiment_id,
                'dataset_name': dataset_name,
                'config_path': config_path,
                'success': False,
                'error': 'timeout'
            }
        except Exception as e:
            print(f"❌ Experiment {experiment_id} failed: {e}")
            return {
                'experiment_id': experiment_id,
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
            
            elif 'Final Loss:' in line:
                try:
                    metrics['final_loss'] = float(line.split(':')[1].strip())
                except:
                    pass
        
        return metrics
    
    def run_hyperparameter_tuning(self, datasets: List[str] = None):
        """Run hyperparameter tuning on specified datasets"""
        
        if datasets is None:
            datasets = ['3dshapes', 'dsprites']
        
        all_results = []
        
        for dataset_name in datasets:
            print(f"\n{'='*80}")
            print(f"Starting hyperparameter tuning for {dataset_name}")
            print(f"{'='*80}")
            
            # Generate hyperparameter combinations
            combinations = self.generate_hyperparameter_combinations()
            
            # Limit to first 20 combinations for initial tuning
            combinations = combinations[:20]
            
            dataset_results = []
            
            for i, config in enumerate(combinations):
                experiment_id = f"{dataset_name}_{i:03d}"
                
                # Create config file
                config_path = self.create_config_file(config, dataset_name, experiment_id)
                
                # Run experiment
                result = self.run_experiment(config_path, dataset_name, experiment_id)
                
                # Add hyperparameters to result
                result.update(config)
                
                dataset_results.append(result)
                all_results.append(result)
                
                # Save intermediate results
                self.save_results(dataset_results, f"{dataset_name}_results.json")
                
                print(f"✅ Experiment {experiment_id} completed")
                if result.get('dci_disentanglement'):
                    print(f"   DCI-D: {result['dci_disentanglement']:.4f}")
        
        # Save all results
        self.save_results(all_results, "all_results.json")
        
        # Generate analysis
        self.analyze_results(all_results)
    
    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file"""
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def analyze_results(self, results: List[Dict]):
        """Analyze and visualize results"""
        
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING ANALYSIS")
        print(f"{'='*80}")
        
        # Filter successful experiments
        successful_results = [r for r in results if r.get('success', False) and r.get('dci_disentanglement') is not None]
        
        if not successful_results:
            print("❌ No successful experiments found")
            return
        
        print(f"✅ {len(successful_results)} successful experiments analyzed")
        
        # Find best configurations
        self.find_best_configurations(successful_results)
        
        # Create visualizations
        self.create_visualizations(successful_results)
    
    def find_best_configurations(self, results: List[Dict]):
        """Find best configurations for each dataset"""
        
        datasets = set(r['dataset_name'] for r in results)
        
        for dataset in datasets:
            dataset_results = [r for r in results if r['dataset_name'] == dataset]
            
            # Sort by DCI-D disentanglement
            dataset_results.sort(key=lambda x: x.get('dci_disentanglement', 0), reverse=True)
            
            print(f"\n📊 Best configurations for {dataset}:")
            print("-" * 60)
            
            for i, result in enumerate(dataset_results[:5]):
                print(f"{i+1}. DCI-D: {result.get('dci_disentanglement', 0):.4f}")
                print(f"   Config: disentanglement_weight={result.get('disentanglement_weight')}, "
                      f"attention_reg_weight={result.get('attention_reg_weight')}, "
                      f"entropy_reg_weight={result.get('entropy_reg_weight')}, "
                      f"latent_dim={result.get('latent_dim')}, "
                      f"learning_rate={result.get('learning_rate')}")
                print(f"   Other metrics: MIG={result.get('mig', 0):.4f}, "
                      f"IRS={result.get('irs', 0):.4f}, "
                      f"Z-diff={result.get('z_diff', 0):.4f}")
                print()
    
    def create_visualizations(self, results: List[Dict]):
        """Create visualization plots"""
        
        try:
            # Create correlation heatmap
            self.plot_correlation_heatmap(results)
            
            # Create parameter importance plot
            self.plot_parameter_importance(results)
            
            # Create best results comparison
            self.plot_best_results_comparison(results)
            
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
    
    def plot_correlation_heatmap(self, results: List[Dict]):
        """Plot correlation heatmap between parameters and DCI-D"""
        
        # Extract numerical parameters and DCI-D
        param_names = ['disentanglement_weight', 'attention_reg_weight', 'entropy_reg_weight', 
                      'latent_dim', 'learning_rate']
        
        data = []
        for result in results:
            row = [result.get(param, 0) for param in param_names]
            row.append(result.get('dci_disentanglement', 0))
            data.append(row)
        
        if not data:
            return
        
        # Create correlation matrix
        param_names.append('dci_disentanglement')
        df = np.array(data)
        
        # Calculate correlations
        corr_matrix = np.corrcoef(df.T)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=param_names, yticklabels=param_names)
        plt.title('Parameter Correlation with DCI-D Disentanglement')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Correlation heatmap saved")
    
    def plot_parameter_importance(self, results: List[Dict]):
        """Plot parameter importance based on DCI-D variation"""
        
        param_names = ['disentanglement_weight', 'attention_reg_weight', 'entropy_reg_weight', 
                      'latent_dim', 'learning_rate']
        
        importance_scores = []
        
        for param in param_names:
            # Group by parameter value and calculate DCI-D variance
            param_values = {}
            for result in results:
                value = result.get(param, 0)
                dci_d = result.get('dci_disentanglement', 0)
                
                if value not in param_values:
                    param_values[value] = []
                param_values[value].append(dci_d)
            
            # Calculate variance across parameter values
            dci_d_values = list(param_values.values())
            if len(dci_d_values) > 1:
                variance = np.var([np.mean(values) for values in dci_d_values])
                importance_scores.append(variance)
            else:
                importance_scores.append(0)
        
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(param_names, importance_scores)
        plt.title('Parameter Importance for DCI-D Disentanglement')
        plt.ylabel('Variance in DCI-D across parameter values')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, importance_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'parameter_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Parameter importance plot saved")
    
    def plot_best_results_comparison(self, results: List[Dict]):
        """Plot comparison of best results across datasets"""
        
        datasets = set(r['dataset_name'] for r in results)
        
        # Get best result for each dataset
        best_results = {}
        for dataset in datasets:
            dataset_results = [r for r in results if r['dataset_name'] == dataset]
            best_result = max(dataset_results, key=lambda x: x.get('dci_disentanglement', 0))
            best_results[dataset] = best_result
        
        # Plot metrics comparison
        metrics = ['dci_disentanglement', 'dci_completeness', 'dci_informativeness', 'mig', 'irs']
        metric_labels = ['DCI-D', 'DCI-C', 'DCI-I', 'MIG', 'IRS']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [best_results[dataset].get(metric, 0) for dataset in datasets]
            
            axes[i].bar(datasets, values)
            axes[i].set_title(f'{label}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Best Results Comparison Across Datasets')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'best_results_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Best results comparison saved")


def main():
    """Main function"""
    print("🚀 Starting Hyperparameter Tuning for MINE Disentangle VAE")
    print("Focus: DCI-D Disentanglement Metric Optimization")
    
    # Initialize tuner
    tuner = HyperparameterTuner()
    
    # Run tuning on confirmed datasets
    datasets = ['3dshapes', 'dsprites']
    
    print(f"Datasets: {datasets}")
    print(f"Results will be saved to: {tuner.results_dir}")
    
    # Run hyperparameter tuning
    tuner.run_hyperparameter_tuning(datasets)
    
    print(f"\n🎉 Hyperparameter tuning completed!")
    print(f"Check results in: {tuner.results_dir}")


if __name__ == "__main__":
    main() 
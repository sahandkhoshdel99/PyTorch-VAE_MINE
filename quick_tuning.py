#!/usr/bin/env python3
"""
Quick hyperparameter tuning script for testing
"""

import os
import yaml
import json
import torch
import numpy as np
from pathlib import Path
import subprocess
import itertools
from typing import Dict, List


class QuickTuner:
    def __init__(self, base_config_path: str = "configs/mine_disentangle_vae.yaml"):
        self.base_config_path = base_config_path
        self.results_dir = Path("quick_tuning_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def generate_quick_combinations(self) -> List[Dict]:
        """Generate a small set of hyperparameter combinations for quick testing"""
        
        # Focus on most important parameters with fewer values
        key_params = {
            'disentanglement_weight': [0.5, 1.0, 2.0],
            'attention_reg_weight': [0.01, 0.1],
            'entropy_reg_weight': [0.01, 0.1],
            'latent_dim': [64, 128],
            'learning_rate': [1e-3, 5e-3]
        }
        
        # Generate combinations
        param_names = list(key_params.keys())
        param_values = list(key_params.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            config = dict(zip(param_names, combination))
            combinations.append(config)
        
        print(f"Generated {len(combinations)} quick hyperparameter combinations")
        return combinations
    
    def create_config_file(self, config: Dict, dataset_name: str, experiment_id: str) -> str:
        """Create a config file for a specific experiment"""
        
        # Create experiment-specific config
        experiment_config = self.base_config.copy()
        
        # Update model parameters
        experiment_config['model_params'].update({
            'latent_dim': config.get('latent_dim', 128),
            'factor_dim': 32,  # Keep fixed
            'attention_dim': 128,  # Keep fixed
            'mine_hidden_dim': 128,  # Keep fixed
            'disentanglement_weight': config.get('disentanglement_weight', 1.0),
            'attention_reg_weight': config.get('attention_reg_weight', 0.1),
            'entropy_reg_weight': config.get('entropy_reg_weight', 0.05)
        })
        
        # Update data parameters
        experiment_config['data_params'].update({
            'dataset_name': dataset_name,
            'train_batch_size': 64,
            'val_batch_size': 64
        })
        
        # Update experiment parameters
        experiment_config['exp_params'].update({
            'LR': config.get('learning_rate', 0.001)
        })
        
        # Update logging parameters
        experiment_config['logging_params'].update({
            'name': f"Quick_MINE_{dataset_name}_{experiment_id}"
        })
        
        # Reduce epochs for quick testing
        experiment_config['trainer_params']['max_epochs'] = 5
        
        # Save config file
        config_path = self.results_dir / f"quick_config_{dataset_name}_{experiment_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(experiment_config, f, default_flow_style=False)
        
        return str(config_path)
    
    def run_experiment(self, config_path: str, dataset_name: str, experiment_id: str) -> Dict:
        """Run a single experiment"""
        
        print(f"\n{'='*50}")
        print(f"Running quick experiment {experiment_id} on {dataset_name}")
        print(f"Config: {config_path}")
        print(f"{'='*50}")
        
        # Run the experiment
        try:
            result = subprocess.run([
                'python', 'run_enhanced.py',
                '--config', config_path
            ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            # Parse results
            metrics = self.parse_experiment_output(result.stdout, result.stderr)
            
            # Add experiment metadata
            metrics.update({
                'experiment_id': experiment_id,
                'dataset_name': dataset_name,
                'config_path': config_path,
                'success': result.returncode == 0,
                'stdout': result.stdout[-1000:],  # Last 1000 chars
                'stderr': result.stderr[-1000:]   # Last 1000 chars
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
            'final_loss': None
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
    
    def run_quick_tuning(self, datasets: List[str] = None):
        """Run quick hyperparameter tuning"""
        
        if datasets is None:
            datasets = ['3dshapes', 'dsprites']
        
        all_results = []
        
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Starting quick tuning for {dataset_name}")
            print(f"{'='*60}")
            
            # Generate combinations
            combinations = self.generate_quick_combinations()
            
            # Limit to first 5 combinations for quick test
            combinations = combinations[:5]
            
            dataset_results = []
            
            for i, config in enumerate(combinations):
                experiment_id = f"{dataset_name}_{i:02d}"
                
                # Create config file
                config_path = self.create_config_file(config, dataset_name, experiment_id)
                
                # Run experiment
                result = self.run_experiment(config_path, dataset_name, experiment_id)
                
                # Add hyperparameters to result
                result.update(config)
                
                dataset_results.append(result)
                all_results.append(result)
                
                # Save intermediate results
                self.save_results(dataset_results, f"quick_{dataset_name}_results.json")
                
                print(f"✅ Quick experiment {experiment_id} completed")
                if result.get('dci_disentanglement'):
                    print(f"   DCI-D: {result['dci_disentanglement']:.4f}")
        
        # Save all results
        self.save_results(all_results, "quick_all_results.json")
        
        # Show best results
        self.show_best_results(all_results)
    
    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file"""
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def show_best_results(self, results: List[Dict]):
        """Show best results for each dataset"""
        
        print(f"\n{'='*60}")
        print("QUICK TUNING RESULTS")
        print(f"{'='*60}")
        
        # Filter successful experiments
        successful_results = [r for r in results if r.get('success', False) and r.get('dci_disentanglement') is not None]
        
        if not successful_results:
            print("❌ No successful experiments found")
            return
        
        print(f"✅ {len(successful_results)} successful experiments")
        
        # Group by dataset
        datasets = set(r['dataset_name'] for r in successful_results)
        
        for dataset in datasets:
            dataset_results = [r for r in successful_results if r['dataset_name'] == dataset]
            
            # Sort by DCI-D disentanglement
            dataset_results.sort(key=lambda x: x.get('dci_disentanglement', 0), reverse=True)
            
            print(f"\n📊 Best results for {dataset}:")
            print("-" * 50)
            
            for i, result in enumerate(dataset_results[:3]):
                print(f"{i+1}. DCI-D: {result.get('dci_disentanglement', 0):.4f}")
                print(f"   Config: disentanglement_weight={result.get('disentanglement_weight')}, "
                      f"attention_reg_weight={result.get('attention_reg_weight')}, "
                      f"entropy_reg_weight={result.get('entropy_reg_weight')}, "
                      f"latent_dim={result.get('latent_dim')}, "
                      f"learning_rate={result.get('learning_rate')}")
                if result.get('mig'):
                    print(f"   MIG: {result.get('mig', 0):.4f}")
                print()


def main():
    """Main function"""
    print("🚀 Starting Quick Hyperparameter Tuning")
    print("Focus: DCI-D Disentanglement Metric (Quick Test)")
    
    # Initialize tuner
    tuner = QuickTuner()
    
    # Run quick tuning on confirmed datasets
    datasets = ['3dshapes', 'dsprites']
    
    print(f"Datasets: {datasets}")
    print(f"Results will be saved to: {tuner.results_dir}")
    
    # Run quick tuning
    tuner.run_quick_tuning(datasets)
    
    print(f"\n🎉 Quick tuning completed!")
    print(f"Check results in: {tuner.results_dir}")


if __name__ == "__main__":
    main() 
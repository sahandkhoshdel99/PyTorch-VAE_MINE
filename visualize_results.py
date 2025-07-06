#!/usr/bin/env python3
"""
Comprehensive visualization script for hyperparameter tuning results
Includes sensitivity analysis, performance comparison, and top configurations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResultsVisualizer:
    def __init__(self, results_dir: str = "quick_tuning_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("visualization_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_results(self, filename: str = "quick_all_results.json") -> pd.DataFrame:
        """Load results from JSON file"""
        filepath = self.results_dir / filename
        
        if not filepath.exists():
            print(f"❌ Results file not found: {filepath}")
            return pd.DataFrame()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Filter successful experiments
        df_success = df[df['success'] == True].copy()
        
        print(f"📊 Loaded {len(df)} total experiments")
        print(f"✅ {len(df_success)} successful experiments")
        
        return df_success
    
    def plot_hyperparameter_sensitivity(self, df: pd.DataFrame):
        """Plot hyperparameter sensitivity analysis"""
        
        if df.empty:
            print("❌ No successful experiments to visualize")
            return
        
        # Define parameters to analyze
        params = ['disentanglement_weight', 'attention_reg_weight', 'entropy_reg_weight', 
                 'latent_dim', 'learning_rate']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, param in enumerate(params):
            if param not in df.columns:
                continue
                
            ax = axes[i]
            
            # Group by parameter value and calculate mean DCI-D
            param_data = df.groupby(param)['dci_disentanglement'].agg(['mean', 'std', 'count']).reset_index()
            
            # Plot mean DCI-D vs parameter
            ax.errorbar(param_data[param], param_data['mean'], 
                       yerr=param_data['std'], marker='o', capsize=5, capthick=2)
            
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('DCI-Disentanglement')
            ax.set_title(f'{param.replace("_", " ").title()} Sensitivity')
            ax.grid(True, alpha=0.3)
            
            # Add count annotations
            for _, row in param_data.iterrows():
                ax.annotate(f"n={row['count']}", 
                           (row[param], row['mean']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Remove extra subplot
        if len(params) < 6:
            axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Hyperparameter sensitivity plot saved")
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap between parameters and metrics"""
        
        if df.empty:
            return
        
        # Select numerical columns
        numerical_cols = ['disentanglement_weight', 'attention_reg_weight', 'entropy_reg_weight', 
                         'latent_dim', 'learning_rate', 'dci_disentanglement', 'dci_completeness', 
                         'dci_informativeness', 'mig', 'irs', 'z_diff', 'z_min_var', 'jemmig']
        
        # Filter columns that exist in dataframe
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        if len(available_cols) < 2:
            print("❌ Not enough numerical columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Parameter and Metric Correlations', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Correlation heatmap saved")
    
    def plot_top_performances(self, df: pd.DataFrame):
        """Plot top performing configurations"""
        
        if df.empty:
            return
        
        # Get top 10 configurations by DCI-D
        top_configs = df.nlargest(10, 'dci_disentanglement')
        
        # Create subplots for different metrics
        metrics = ['dci_disentanglement', 'dci_completeness', 'dci_informativeness', 'mig', 'irs']
        metric_labels = ['DCI-D', 'DCI-C', 'DCI-I', 'MIG', 'IRS']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            if metric not in df.columns:
                continue
                
            ax = axes[i]
            
            # Sort by metric
            sorted_data = top_configs.sort_values(metric, ascending=True)
            
            # Create bar plot
            bars = ax.barh(range(len(sorted_data)), sorted_data[metric])
            
            # Color bars by dataset
            colors = ['skyblue' if '3dshapes' in name else 'lightcoral' 
                     for name in sorted_data['experiment_id']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_yticks(range(len(sorted_data)))
            ax.set_yticklabels([f"{row['experiment_id']}\n{row['dataset_name']}" 
                               for _, row in sorted_data.iterrows()], fontsize=8)
            ax.set_xlabel(label)
            ax.set_title(f'Top 10 by {label}')
            
            # Add value labels
            for j, (_, row) in enumerate(sorted_data.iterrows()):
                ax.text(row[metric], j, f'{row[metric]:.3f}', 
                       va='center', ha='left', fontsize=8)
        
        # Remove extra subplot
        axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_performances.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("🏆 Top performances plot saved")
    
    def plot_dataset_comparison(self, df: pd.DataFrame):
        """Plot comparison between datasets"""
        
        if df.empty:
            return
        
        # Group by dataset
        dataset_stats = df.groupby('dataset_name').agg({
            'dci_disentanglement': ['mean', 'std', 'max'],
            'dci_completeness': ['mean', 'std', 'max'],
            'dci_informativeness': ['mean', 'std', 'max'],
            'mig': ['mean', 'std', 'max'],
            'irs': ['mean', 'std', 'max']
        }).round(4)
        
        # Flatten column names
        dataset_stats.columns = ['_'.join(col).strip() for col in dataset_stats.columns]
        
        # Create comparison plot
        metrics = ['dci_disentanglement', 'dci_completeness', 'dci_informativeness', 'mig', 'irs']
        metric_labels = ['DCI-D', 'DCI-C', 'DCI-I', 'MIG', 'IRS']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]
            
            # Get mean and std for each dataset
            datasets = dataset_stats.index.tolist()
            means = [dataset_stats.loc[ds, f'{metric}_mean'] for ds in datasets]
            stds = [dataset_stats.loc[ds, f'{metric}_std'] for ds in datasets]
            
            # Create bar plot with error bars
            bars = ax.bar(datasets, means, yerr=stds, capsize=5, capthick=2)
            
            # Color bars
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_title(f'{label} Comparison')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for j, (mean, std) in enumerate(zip(means, stds)):
                ax.text(j, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Dataset comparison plot saved")
    
    def plot_parameter_importance(self, df: pd.DataFrame):
        """Plot parameter importance based on variance analysis"""
        
        if df.empty:
            return
        
        # Define parameters
        params = ['disentanglement_weight', 'attention_reg_weight', 'entropy_reg_weight', 
                 'latent_dim', 'learning_rate']
        
        importance_scores = {}
        
        for param in params:
            if param not in df.columns:
                continue
                
            # Calculate variance in DCI-D across parameter values
            param_values = df.groupby(param)['dci_disentanglement'].mean()
            variance = param_values.var()
            importance_scores[param] = variance
        
        if not importance_scores:
            return
        
        # Sort by importance
        sorted_params = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        param_names = [p.replace('_', ' ').title() for p, _ in sorted_params]
        scores = [s for _, s in sorted_params]
        
        bars = plt.bar(param_names, scores, color='steelblue', alpha=0.7)
        
        plt.title('Parameter Importance for DCI-D Disentanglement', fontsize=16, pad=20)
        plt.ylabel('Variance in DCI-D across parameter values')
        plt.xlabel('Parameters')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Parameter importance plot saved")
    
    def generate_summary_report(self, df: pd.DataFrame):
        """Generate a comprehensive summary report"""
        
        if df.empty:
            print("❌ No data for summary report")
            return
        
        report = []
        report.append("=" * 60)
        report.append("HYPERPARAMETER TUNING SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        report.append("📊 OVERALL STATISTICS")
        report.append("-" * 30)
        report.append(f"Total successful experiments: {len(df)}")
        report.append(f"Datasets tested: {df['dataset_name'].nunique()}")
        report.append(f"Datasets: {', '.join(df['dataset_name'].unique())}")
        report.append("")
        
        # Best configurations by dataset
        for dataset in df['dataset_name'].unique():
            dataset_df = df[df['dataset_name'] == dataset]
            
            report.append(f"🏆 BEST CONFIGURATIONS FOR {dataset.upper()}")
            report.append("-" * 40)
            
            # Top 3 by DCI-D
            top_3 = dataset_df.nlargest(3, 'dci_disentanglement')
            
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                report.append(f"{i}. DCI-D: {row['dci_disentanglement']:.4f}")
                report.append(f"   Config: disentanglement_weight={row['disentanglement_weight']}, "
                            f"attention_reg_weight={row['attention_reg_weight']}, "
                            f"entropy_reg_weight={row['entropy_reg_weight']}, "
                            f"latent_dim={row['latent_dim']}, "
                            f"learning_rate={row['learning_rate']}")
                
                if 'mig' in row and pd.notna(row['mig']):
                    report.append(f"   Other metrics: MIG={row['mig']:.4f}")
                if 'irs' in row and pd.notna(row['irs']):
                    report.append(f"   IRS={row['irs']:.4f}")
                report.append("")
        
        # Parameter insights
        report.append("🔍 PARAMETER INSIGHTS")
        report.append("-" * 25)
        
        params = ['disentanglement_weight', 'attention_reg_weight', 'entropy_reg_weight', 
                 'latent_dim', 'learning_rate']
        
        for param in params:
            if param in df.columns:
                best_value = df.loc[df['dci_disentanglement'].idxmax(), param]
                report.append(f"Best {param}: {best_value}")
        
        report.append("")
        report.append("=" * 60)
        
        # Save report
        report_text = "\n".join(report)
        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        # Print report
        print(report_text)
        
        print(f"\n📄 Summary report saved to: {self.output_dir / 'summary_report.txt'}")
    
    def create_all_visualizations(self, filename: str = "quick_all_results.json"):
        """Create all visualizations"""
        
        print("🎨 Creating comprehensive visualizations...")
        
        # Load data
        df = self.load_results(filename)
        
        if df.empty:
            print("❌ No data to visualize")
            return
        
        # Create all plots
        self.plot_hyperparameter_sensitivity(df)
        self.plot_correlation_heatmap(df)
        self.plot_top_performances(df)
        self.plot_dataset_comparison(df)
        self.plot_parameter_importance(df)
        self.generate_summary_report(df)
        
        print(f"\n🎉 All visualizations saved to: {self.output_dir}")
        print("📁 Files created:")
        for file in self.output_dir.glob("*.png"):
            print(f"   - {file.name}")
        print(f"   - summary_report.txt")


def main():
    """Main function"""
    print("🎨 Starting Results Visualization")
    
    # Initialize visualizer
    visualizer = ResultsVisualizer()
    
    # Create all visualizations
    visualizer.create_all_visualizations()
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main() 
import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from enhanced_experiment import create_experiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin
from evaluation_metrics import VAEEvaluator


def main():
    parser = argparse.ArgumentParser(description='Enhanced runner for VAE models with comprehensive evaluation')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/mine_disentangle_vae.yaml')
    parser.add_argument('--experiment-type', '-e',
                        choices=['standard', 'disentanglement'],
                        default='disentanglement',
                        help='type of experiment to run')
    parser.add_argument('--evaluation-only', '-eval',
                        action='store_true',
                        help='run evaluation only on a trained model')
    parser.add_argument('--checkpoint-path', '-cp',
                        type=str,
                        help='path to model checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return
    
    # Set up logging
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['model_params']['name'])
    
    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    
    # Create model
    model = vae_models[config['model_params']['name']](**config['model_params'])
    
    # Create experiment
    experiment = create_experiment(model, config['exp_params'], args.experiment_type)
    
    # Create data module
    data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    data.setup()
    
    # Create trainer
    runner = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=3,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True
            ),
        ],
        strategy=DDPPlugin(find_unused_parameters=False),
        **config['trainer_params']
    )
    
    # Create directories
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Traversals").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Evaluations").mkdir(exist_ok=True, parents=True)
    
    if args.evaluation_only:
        # Run evaluation only
        if args.checkpoint_path is None:
            print("Error: checkpoint path required for evaluation-only mode")
            return
        
        print(f"Loading model from {args.checkpoint_path}")
        experiment = experiment.load_from_checkpoint(args.checkpoint_path, model=model, params=config['exp_params'])
        
        print("Running comprehensive evaluation...")
        evaluator = VAEEvaluator(device='cuda' if len(config['trainer_params']['gpus']) > 0 else 'cpu')
        
        # Run evaluation
        metrics = evaluator.evaluate_model(
            model=experiment.model,
            dataloader=data.test_dataloader(),
            num_samples=5000
        )
        
        # Save results
        eval_save_path = os.path.join(tb_logger.log_dir, "final_evaluation_results.json")
        evaluator.save_evaluation_results(metrics, eval_save_path)
        
        # Print results
        print("\n=== Final Evaluation Results ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.6f}")
        
        # Run latent traversal if it's a disentanglement model
        if args.experiment_type == 'disentanglement':
            print("\nRunning latent traversal evaluation...")
            traversal_results = evaluator.evaluate_latent_traversal(
                model=experiment.model,
                dataloader=data.test_dataloader(),
                num_factors=10,
                num_steps=10
            )
            
            # Save traversal results
            for factor_name, traversals in traversal_results.items():
                save_path = os.path.join(tb_logger.log_dir, "Traversals", f"{factor_name}_final.png")
                import torchvision.utils as vutils
                vutils.save_image(traversals, save_path, normalize=True, nrow=10)
            
            print("Latent traversal evaluation completed.")
        
    else:
        # Train the model
        print(f"======= Training {config['model_params']['name']} =======")
        print(f"Experiment type: {args.experiment_type}")
        print(f"Model parameters: {config['model_params']}")
        
        runner.fit(experiment, datamodule=data)
        
        print("Training completed successfully!")


def run_ablation_study():
    """
    Run ablation study for the MINE Disentangle VAE
    """
    print("Running ablation study...")
    
    # Base configuration
    base_config = {
        'model_params': {
            'name': 'MINEDisentangleVAE',
            'in_channels': 3,
            'latent_dim': 128,
            'num_factors': 10,
            'factor_dim': 32,
            'attention_dim': 128,
            'mine_hidden_dim': 128,
            'disentanglement_weight': 1.0,
            'attention_reg_weight': 0.1,
            'entropy_reg_weight': 0.05
        },
        'data_params': {
            'data_path': "Data/",
            'train_batch_size': 64,
            'val_batch_size': 64,
            'patch_size': 64,
            'num_workers': 4
        },
        'exp_params': {
            'LR': 0.001,
            'weight_decay': 0.0001,
            'scheduler_gamma': 0.95,
            'kld_weight': 0.00025,
            'manual_seed': 1265
        },
        'trainer_params': {
            'gpus': [1],
            'max_epochs': 50,  # Shorter for ablation study
            'gradient_clip_val': 1.0
        },
        'logging_params': {
            'save_dir': "logs/",
            'name': "MINEDisentangleVAE"
        }
    }
    
    # Ablation configurations
    ablation_configs = [
        {
            'name': 'no_attention',
            'attention_reg_weight': 0.0,
            'entropy_reg_weight': 0.0
        },
        {
            'name': 'no_mine',
            'disentanglement_weight': 0.0
        },
        {
            'name': 'high_disentanglement',
            'disentanglement_weight': 2.0
        },
        {
            'name': 'low_factors',
            'num_factors': 5
        },
        {
            'name': 'high_factors',
            'num_factors': 15
        }
    ]
    
    results = {}
    
    for ablation_config in ablation_configs:
        print(f"\nRunning ablation: {ablation_config['name']}")
        
        # Update base config
        config = base_config.copy()
        config['logging_params']['name'] = f"MINEDisentangleVAE_{ablation_config['name']}"
        
        # Update model params
        for key, value in ablation_config.items():
            if key != 'name':
                config['model_params'][key] = value
        
        # Save config
        config_path = f"configs/ablation_{ablation_config['name']}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run training
        try:
            # This would normally run the training
            # For now, just print the configuration
            print(f"Configuration for {ablation_config['name']}:")
            print(f"  - Disentanglement weight: {config['model_params']['disentanglement_weight']}")
            print(f"  - Attention reg weight: {config['model_params']['attention_reg_weight']}")
            print(f"  - Number of factors: {config['model_params']['num_factors']}")
            
            results[ablation_config['name']] = "Configuration prepared"
            
        except Exception as e:
            print(f"Error in ablation {ablation_config['name']}: {e}")
            results[ablation_config['name']] = f"Error: {e}"
    
    # Save ablation results
    with open("logs/ablation_study_results.json", 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    print("\nAblation study completed!")


if __name__ == "__main__":
    main() 
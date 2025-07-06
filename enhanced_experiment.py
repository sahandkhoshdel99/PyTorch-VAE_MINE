import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from evaluation_metrics import VAEEvaluator
import numpy as np
from typing import Dict, List
import json


class EnhancedVAEXperiment(pl.LightningModule):
    """
    Enhanced VAE experiment with comprehensive evaluation metrics
    """

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict,
                 evaluation_frequency: int = 10) -> None:
        super(EnhancedVAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.evaluation_frequency = evaluation_frequency
        
        # Initialize evaluator
        self.evaluator = VAEEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store metrics history
        self.metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }
        
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        # Log all loss components
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        
        # Store metrics for analysis
        if batch_idx % 100 == 0:  # Store every 100 batches
            self.metrics_history['train'].append({
                'epoch': self.current_epoch,
                'batch': batch_idx,
                **{key: val.item() for key, val in train_loss.items()}
            })

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
        # Store validation metrics
        self.metrics_history['val'].append({
            'epoch': self.current_epoch,
            'batch': batch_idx,
            **{key: val.item() for key, val in val_loss.items()}
        })

    def on_validation_end(self) -> None:
        self.sample_images()
        
        # Run comprehensive evaluation every N epochs
        if self.current_epoch % self.evaluation_frequency == 0:
            self.run_comprehensive_evaluation()

    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir, 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir, 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def run_comprehensive_evaluation(self):
        """
        Run comprehensive evaluation including FID, LPIPS, and disentanglement metrics
        """
        print(f"Running comprehensive evaluation at epoch {self.current_epoch}")
        
        # Get test dataloader
        test_dataloader = self.trainer.datamodule.test_dataloader()
        
        # Run evaluation
        metrics = self.evaluator.evaluate_model(
            model=self.model,
            dataloader=test_dataloader,
            num_samples=1000
        )
        
        # Log metrics
        for metric_name, value in metrics.items():
            self.log(f"eval_{metric_name}", value, sync_dist=True)
        
        # Store evaluation metrics
        self.metrics_history['test'].append({
            'epoch': self.current_epoch,
            **metrics
        })
        
        # Save evaluation results
        eval_save_path = os.path.join(self.logger.log_dir, 
                                     f"evaluation_epoch_{self.current_epoch}.json")
        self.save_evaluation_results(metrics, eval_save_path)
        
        print(f"Evaluation completed. FID: {metrics.get('fid', 'N/A'):.4f}, "
              f"LPIPS: {metrics.get('lpips', 'N/A'):.4f}")

    def save_evaluation_results(self, metrics: Dict[str, float], save_path: str):
        """
        Save evaluation results to JSON file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Evaluation results saved to {save_path}")

    def on_train_end(self):
        """
        Final evaluation and cleanup
        """
        print("Training completed. Running final evaluation...")
        
        # Run final comprehensive evaluation
        test_dataloader = self.trainer.datamodule.test_dataloader()
        final_metrics = self.evaluator.evaluate_model(
            model=self.model,
            dataloader=test_dataloader,
            num_samples=5000
        )
        
        # Save final results
        final_save_path = os.path.join(self.logger.log_dir, "final_evaluation.json")
        self.save_evaluation_results(final_metrics, final_save_path)
        
        # Save metrics history
        history_save_path = os.path.join(self.logger.log_dir, "metrics_history.json")
        with open(history_save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print("Final evaluation completed and saved.")

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims


class DisentanglementExperiment(EnhancedVAEXperiment):
    """
    Specialized experiment for disentanglement-focused models
    """
    
    def __init__(self, vae_model: BaseVAE, params: dict, evaluation_frequency: int = 5):
        super(DisentanglementExperiment, self).__init__(vae_model, params, evaluation_frequency)
        
    def run_latent_traversal_evaluation(self):
        """
        Run latent space traversal evaluation for disentanglement analysis
        """
        print(f"Running latent traversal evaluation at epoch {self.current_epoch}")
        
        test_dataloader = self.trainer.datamodule.test_dataloader()
        
        # Run latent traversal evaluation
        traversal_results = self.evaluator.evaluate_latent_traversal(
            model=self.model,
            dataloader=test_dataloader,
            num_factors=10,
            num_steps=10
        )
        
        # Save traversal results
        for factor_name, traversals in traversal_results.items():
            save_path = os.path.join(self.logger.log_dir, 
                                   "Traversals", 
                                   f"{factor_name}_epoch_{self.current_epoch}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            vutils.save_image(traversals,
                              save_path,
                              normalize=True,
                              nrow=10)
        
        print("Latent traversal evaluation completed.")
    
    def on_validation_end(self) -> None:
        super().on_validation_end()
        
        # Run latent traversal evaluation more frequently for disentanglement models
        if self.current_epoch % (self.evaluation_frequency // 2) == 0:
            self.run_latent_traversal_evaluation()


def create_experiment(model: BaseVAE, params: dict, experiment_type: str = 'standard') -> pl.LightningModule:
    """
    Factory function to create appropriate experiment type
    """
    if experiment_type == 'disentanglement':
        return DisentanglementExperiment(model, params)
    else:
        return EnhancedVAEXperiment(model, params) 
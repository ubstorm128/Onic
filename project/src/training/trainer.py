"""
Comprehensive PyTorch training engine with early stopping,
learning rate scheduling, checkpointing, and evaluation tracking.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_metrics
from .losses import get_loss_function
from ..utils.logging import get_logger


class ModelTrainer:
    """
    Handles model training, early stopping, LR scheduling, and checkpoint persistence.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 15,
        min_delta: float = 5e-4,
        lr_scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        class_weights: Optional[np.ndarray] = None,
        model_name: str = "model",
        checkpoint_dir: str | Path = "artifacts/models",
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = get_loss_function(class_weights=class_weights, device=str(self.device))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        if lr_scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_epochs, eta_min=1e-6
            )
        else:
            self.scheduler = None
            
        self.logger = get_logger(f"trainer_{model_name}")
        self.best_val_loss = float("inf")
        self.best_model_weights = None
        self.history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Batch formats:
            # For multimodal: (modality_tensors_dict, mask, labels, sample_ids)
            # For single-modality: (x_tensor, labels, sample_ids)
            if len(batch) == 4:
                mod_tensors, mask, labels, _ = batch
                mod_tensors = {k: v.to(self.device) for k, v in mod_tensors.items()}
                mask = mask.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(mod_tensors, mask)
                # If model returns (logits, attn_weights)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = self.criterion(logits, labels)
            else:
                x, labels, _ = batch
                x = x.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, labels)
                
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        if self.scheduler is not None:
            self.scheduler.step()
            
        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Evaluate model on validation or test set.
        Returns:
            avg_loss: float
            metrics: dict of metrics (macro_f1, balanced_accuracy, etc.)
            all_preds: 1D numpy array of predicted labels
            all_targets: 1D numpy array of true labels
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []
        
        for batch in data_loader:
            if len(batch) == 4:
                mod_tensors, mask, labels, _ = batch
                mod_tensors = {k: v.to(self.device) for k, v in mod_tensors.items()}
                mask = mask.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(mod_tensors, mask)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = self.criterion(logits, labels)
            else:
                x, labels, _ = batch
                x = x.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(x)
                loss = self.criterion(logits, labels)
                
            total_loss += loss.item()
            num_batches += 1
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / max(1, num_batches)
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        metrics = compute_metrics(y_true, y_pred)
        
        return avg_loss, metrics, y_pred, y_true

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping on validation loss.
        """
        epochs_no_improve = 0
        best_epoch = 0
        
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics, _, _ = self.evaluate(val_loader)
            val_f1 = val_metrics["macro_f1"]
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_macro_f1"].append(val_f1)
            
            # Check for improvement
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
                
                if save_checkpoint:
                    checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.best_model_weights,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                    }, checkpoint_path)
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= self.patience:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch: {best_epoch} with val_loss={self.best_val_loss:.4f}"
                )
                break
                
        # Restore best weights
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
            
        return {
            "best_epoch": best_epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

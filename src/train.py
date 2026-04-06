import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import numpy as np

# Import from src directory
from src.dataset import PanNukeDataset
from src.model import get_model, get_optimizer
from src.utils import calculate_dice, save_checkpoint, load_checkpoint


class CombinedLoss(nn.Module):
    """
    Combined CrossEntropy and Dice Loss for segmentation.
    This loss function handles class imbalance and ignores border pixels.
    """
    def __init__(self, num_classes=2, class_weights=None, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    
    def forward(self, inputs, targets, smooth=1e-6):
        """
        Args:
            inputs: Model predictions (logits) [Batch, Classes, H, W]
            targets: Ground truth masks [Batch, H, W]
            smooth: Small constant to avoid division by zero
        
        Returns:
            combined_loss: CE loss + Dice loss
        """
        # 1. Cross Entropy Loss
        ce_loss = self.ce(inputs, targets)
        
        # 2. Dice Loss for foreground classes
        probs = torch.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Create mask for valid pixels (ignore border pixels)
        valid_mask = (targets != self.ignore_index).float().unsqueeze(1)
        
        dice_loss = 0.0
        num_foreground = 0
        
        # Compute Dice for each foreground class
        for cls in range(1, self.num_classes):
            pred_cls = probs[:, cls:cls+1, :, :] * valid_mask
            target_cls = targets_one_hot[:, cls:cls+1, :, :] * valid_mask
            
            intersection = (pred_cls * target_cls).sum(dim=(2, 3))
            union = pred_cls.sum(dim=(2, 3)) + target_cls.sum(dim=(2, 3))
            
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_loss += (1 - dice.mean())
            num_foreground += 1
        
        if num_foreground > 0:
            dice_loss = dice_loss / num_foreground
        else:
            dice_loss = 0.0
        
        # Combine losses (weighted equally)
        return ce_loss + dice_loss


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Executes one training epoch.
    
    Args:
        model: Segmentation model
        loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on (cuda/cpu)
        epoch: Current epoch number (for logging)
    
    Returns:
        avg_loss: Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/Training", leave=False)
    
    for images, masks in pbar:
        # Move data to device
        images, masks = images.to(device), masks.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)['out']
        
        # Compute loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device, num_classes, epoch=None):
    """
    Executes one validation epoch.
    
    Args:
        model: Segmentation model
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on
        num_classes: Number of output classes
        epoch: Current epoch number (optional, for debugging)
    
    Returns:
        val_loss: Average validation loss
        val_dice: Average Dice score
    """
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)['out']
            
            # Compute metrics
            loss = criterion(outputs, masks)
            dice = calculate_dice(outputs, masks, num_classes=num_classes)
            
            val_loss += loss.item()
            val_dice += dice
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})
    
    return val_loss / len(loader), val_dice / len(loader)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_classes,
    epochs=30,
    checkpoint_dir="checkpoints",
    history=None
):
    """
    Main training loop with checkpointing and history tracking.
    
    Args:
        model: Segmentation model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run on
        num_classes: Number of output classes
        epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints
        history: Optional existing history dict to continue from
    
    Returns:
        history: Dictionary containing training metrics
    """
    # Initialize history tracking
    if history is None:
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'lr': []
        }
    
    best_dice = max(history['val_dice']) if history['val_dice'] else 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"🏁 Starting training on {device}")
    print(f"   Epochs: {epochs} | Classes: {num_classes}")
    print(f"   Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training phase
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validation phase
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device, num_classes, epoch)
        
        # Update learning rate based on validation Dice
        if scheduler:
            scheduler.step(val_dice)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)
        
        # Print summary
        print(f"\n📊 Summary Epoch [{epoch+1}/{epochs}]")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Val Dice: {val_dice:.4f} | LR: {current_lr:.6f}")
        
        # Save checkpoint if improved
        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            print(f"💾 Best checkpoint saved! Dice: {best_dice:.4f}")
        
        print("-" * 50)
    
    print(f"✅ Training completed! Best Dice: {best_dice:.4f}")
    return history


def get_class_weights(dataset, num_classes=2, device='cuda'):
    """
    Compute class weights to handle class imbalance.
    
    Args:
        dataset: PanNukeDataset instance
        num_classes: Number of classes (2 for binary, 6 for multiclass)
        device: Device to put weights on
    
    Returns:
        weights: Tensor of class weights
    """
    class_counts = dataset.get_class_distribution()
    
    # For binary mode, we need to combine classes 1-5
    if num_classes == 2:
        background_pixels = class_counts[0]
        nucleus_pixels = sum(class_counts[i] for i in range(1, 6))
        total_pixels = background_pixels + nucleus_pixels
        
        # Inverse frequency weighting
        weights = torch.tensor([
            total_pixels / (num_classes * background_pixels),
            total_pixels / (num_classes * nucleus_pixels)
        ]).to(device)
    else:
        # Multiclass mode: compute weights for classes 0-5
        total_pixels = sum(class_counts.values())
        weights = []
        for cls in range(num_classes):
            count = class_counts.get(cls, 1)  # Avoid division by zero
            weight = total_pixels / (num_classes * count)
            weights.append(weight)
        weights = torch.tensor(weights).to(device)
    
    print(f"📊 Class weights: {weights.cpu().numpy()}")
    return weights


if __name__ == "__main__":
    """
    Quick test script to verify training functions.
    Run with: python src/train.py
    """
    print("🚀 Training module ready.")
    print("   Import these functions in your Jupyter notebook.")
    print("   Example usage:")
    print("   from src.train import train_model, CombinedLoss")
    print("   history = train_model(model, train_loader, val_loader, ...)")
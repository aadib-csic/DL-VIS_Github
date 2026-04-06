import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from PIL import Image

def calculate_dice(preds, targets, num_classes=2, ignore_index=255, smooth=1e-6):
    """
    Computes the Dice Coefficient for segmentation evaluation.
    
    Args:
        preds: Model predictions (logits) of shape [Batch, Classes, H, W]
        targets: Ground truth masks of shape [Batch, H, W]
        num_classes: Number of output classes
        ignore_index: Index to ignore in evaluation (e.g., 255 for borders)
        smooth: Small constant to avoid division by zero
    
    Returns:
        dice_score: Mean Dice score for all foreground classes
    """
    # Convert logits to class predictions
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)
    
    dice_scores = []
    
    # Iterate through foreground classes (skip background)
    for cls in range(1, num_classes):
        p = (preds == cls).float()
        t = (targets == cls).float()
        
        # Create mask for valid pixels (ignore border pixels)
        valid_mask = (targets != ignore_index).float()
        p = p * valid_mask
        t = t * valid_mask
        
        # Compute intersection and union
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        
        # Dice calculation with smoothing
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    
    # Return average of foreground classes
    if not dice_scores:
        return 0.0
    
    return torch.stack(dice_scores).mean().item()


def calculate_iou(preds, targets, num_classes=2, ignore_index=255, smooth=1e-6):
    """
    Computes Intersection over Union (IoU/Jaccard Index) for segmentation evaluation.
    
    Args:
        preds: Model predictions (logits) of shape [Batch, Classes, H, W]
        targets: Ground truth masks of shape [Batch, H, W]
        num_classes: Number of output classes
        ignore_index: Index to ignore in evaluation
        smooth: Small constant to avoid division by zero
    
    Returns:
        iou_score: Mean IoU for all foreground classes
    """
    # Convert logits to class predictions
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)
    
    iou_scores = []
    
    # Iterate through foreground classes
    for cls in range(1, num_classes):
        p = (preds == cls).float()
        t = (targets == cls).float()
        
        # Create mask for valid pixels
        valid_mask = (targets != ignore_index).float()
        p = p * valid_mask
        t = t * valid_mask
        
        # Compute intersection and union
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        
        # IoU calculation
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou)
    
    if not iou_scores:
        return 0.0
    
    return torch.stack(iou_scores).mean().item()


def save_checkpoint(state, filename="checkpoints/best_model.pth"):
    """
    Saves model checkpoint to disk.
    
    Args:
        state: Dictionary containing model state, optimizer state, epoch, etc.
        filename: Path where checkpoint will be saved
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"💾 Checkpoint saved to: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoints/best_model.pth", device='cuda'):
    """
    Loads model checkpoint from disk.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        filename: Path to checkpoint file
        device: Device to load model to
    
    Returns:
        epoch: Last epoch number
        best_dice: Best Dice score achieved
    """
    if not os.path.exists(filename):
        print(f"⚠️ Checkpoint not found: {filename}")
        return 0, 0.0
    
    checkpoint = torch.load(filename, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_dice = checkpoint.get('dice', 0.0)
    
    print(f"✅ Checkpoint loaded from: {filename}")
    print(f"   Epoch: {epoch}, Best Dice: {best_dice:.4f}")
    
    return epoch, best_dice


def plot_learning_curves(history, output_path="results/learning_curves.png"):
    """
    Visualizes training/validation loss and Dice scores.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'val_dice' lists
        output_path: Path where plot will be saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 5))

    # Plot Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Training & Validation Loss', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Dice score
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice', color='green', linewidth=2)
    plt.title('Segmentation Performance (Dice Score)', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line for reference at 0.7
    if max(history['val_dice']) < 0.7:
        plt.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Target (0.7)')
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 Learning curves saved to: {output_path}")


def save_experiment_results(history, config, folder_name="results"):
    """
    Exports training history and configuration to JSON for reproducibility.
    
    Args:
        history: Dictionary containing training metrics
        config: Dictionary containing hyperparameters and configuration
        folder_name: Directory where results will be saved
    """
    os.makedirs(folder_name, exist_ok=True)
    
    # Save training history
    results_path = os.path.join(folder_name, "training_history.json")
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save configuration
    config_path = os.path.join(folder_name, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Results exported to {folder_name}/")
    print(f"   History: {results_path}")
    print(f"   Config: {config_path}")


def visualize_prediction(image, mask, prediction, class_names=None, save_path=None):
    """
    Visualizes a single prediction with ground truth overlay.
    
    Args:
        image: Input image tensor [C, H, W] or numpy array
        mask: Ground truth mask [H, W]
        prediction: Model prediction [H, W]
        class_names: List of class names for legend
        save_path: If provided, saves figure instead of showing
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert image to display format
    if isinstance(image, torch.Tensor):
        img_display = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize if needed (approximate)
        if img_display.min() < 0:
            img_display = img_display * 0.2 + 0.5
        img_display = img_display.clip(0, 1)
    else:
        img_display = image
    
    # Plot input image
    axes[0].imshow(img_display)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis('off')
    
    # Plot ground truth
    im1 = axes[1].imshow(mask, cmap='jet', vmin=0, vmax=5)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis('off')
    
    # Plot prediction
    im2 = axes[2].imshow(prediction, cmap='jet', vmin=0, vmax=5)
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis('off')
    
    # Add colorbar
    plt.colorbar(im2, ax=axes[2], ticks=range(6))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📸 Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
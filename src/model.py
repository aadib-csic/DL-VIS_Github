import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def get_model(num_classes=2, pretrain=True, freeze_backbone=False):
    """
    Initializes DeepLabV3 with ResNet50 backbone.
    
    Args:
        num_classes: Number of output classes (2 for binary, 6 for multiclass)
        pretrain: If True, load ImageNet pretrained weights
        freeze_backbone: If True, freeze backbone layers (useful for fine-tuning)
    
    Returns:
        model: DeepLabV3 segmentation model
    """
    # 1. Load pretrained DeepLabV3 if requested
    if pretrain:
        weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        model = models.segmentation.deeplabv3_resnet50(weights=weights)
    else:
        model = models.segmentation.deeplabv3_resnet50(weights=None)

    # 2. Replace main classifier (ASPP head)
    # ResNet50 backbone outputs 2048 channels, we adapt to num_classes
    model.classifier = DeepLabHead(2048, num_classes)

    # 3. Fix auxiliary classifier (if it exists)
    # The error was here: ResNet50 auxiliary features have 256 channels, not 1024
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        # Auxiliary classifier structure: [Conv2d, BatchNorm, ReLU, Conv2d]
        # We need to replace the last Conv2d to match num_classes
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    
    # 4. Optionally freeze backbone layers
    if freeze_backbone:
        print("🔒 Freezing backbone layers...")
        for param in model.backbone.parameters():
            param.requires_grad = False
        # Keep classifier and aux_classifier trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
            for param in model.aux_classifier.parameters():
                param.requires_grad = True
    
    return model

def get_optimizer(model, base_lr=1e-4, backbone_lr=1e-5, weight_decay=1e-4):
    """
    Creates optimizer with different learning rates for backbone and head.
    This is crucial for fine-tuning pretrained models effectively.
    
    Args:
        model: The DeepLabV3 model
        base_lr: Learning rate for new layers (classifier heads)
        backbone_lr: Learning rate for pretrained backbone (usually lower)
        weight_decay: Weight decay for regularization
    
    Returns:
        optimizer: AdamW optimizer with parameter groups
    """
    # Separate parameters by whether they belong to backbone
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': base_lr, 'weight_decay': weight_decay}
    ])
    
    return optimizer

def count_parameters(model):
    """Utility function to count trainable parameters in the model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    return trainable_params, total_params
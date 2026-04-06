import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

class PanNukeDataset(Dataset):
    """
    Corrected Dataset for PanNuke.
    
    Proper mask handling for this specific dataset:
    - Values 0 = Background
    - Values 255 = Nucleus (foreground)
    - No border pixels in this encoding
    
    For binary segmentation: 255 -> 1 (nucleus), 0 -> 0 (background)
    """
    def __init__(self, root_dir, split='train', transform=None, binary_mode=True):
        """
        Args:
            root_dir: Root directory of the dataset
            split: 'train', 'validate', or 'test'
            transform: Transformations to apply to images
            binary_mode: If True, converts to binary (nucleus vs background)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.binary_mode = binary_mode

        self.images_dir = os.path.join(root_dir, split, "images")
        self.masks_dir = os.path.join(root_dir, split, "masks")

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"❌ Directory not found: {self.images_dir}")
        
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"❌ Directory not found: {self.masks_dir}")

        # Get all image files
        all_images = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        all_masks = sorted([f for f in os.listdir(self.masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Extract base names without extension for matching
        def get_basename(filename):
            return os.path.splitext(filename)[0]
        
        image_dict = {get_basename(f): f for f in all_images}
        mask_dict = {get_basename(f): f for f in all_masks}
        
        # Find common basenames
        common_basenames = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
        
        print(f"📁 Loading {split} split:")
        print(f"   Total images found: {len(all_images)}")
        print(f"   Total masks found: {len(all_masks)}")
        print(f"   Matching pairs: {len(common_basenames)}")
        
        # Create paired lists
        self.images_fps = []
        self.masks_fps = []
        
        for basename in common_basenames:
            self.images_fps.append(os.path.join(self.images_dir, image_dict[basename]))
            self.masks_fps.append(os.path.join(self.masks_dir, mask_dict[basename]))
        
        print(f"   Final dataset size: {len(self.images_fps)} samples")
        
        # Show sample of mismatched files for debugging
        if len(common_basenames) < len(all_images):
            extra_images = set(image_dict.keys()) - set(mask_dict.keys())
            if extra_images:
                print(f"   ⚠️ {len(extra_images)} images without masks (first 3): {list(extra_images)[:3]}")
        
        if len(common_basenames) < len(all_masks):
            extra_masks = set(mask_dict.keys()) - set(image_dict.keys())
            if extra_masks:
                print(f"   ⚠️ {len(extra_masks)} masks without images (first 3): {list(extra_masks)[:3]}")
        
        # Raise error if no matching pairs found
        if len(self.images_fps) == 0:
            raise ValueError(f"No matching image-mask pairs found in {split} split")

    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, idx):
        img_path = self.images_fps[idx]
        mask_path = self.masks_fps[idx]

        try:
            # 1. Load RGB image
            image = Image.open(img_path).convert("RGB")
            
            # 2. Load mask
            mask_pil = Image.open(mask_path)
            mask_np = np.array(mask_pil)
            
            # 3. Process mask according to mode
            if self.binary_mode:
                # BINARY MODE: Convert 255 (nucleus) to 1, background (0) stays 0
                # This matches our diagnostic: masks contain 0 and 255
                mask_processed = np.zeros_like(mask_np)
                mask_processed[mask_np == 255] = 1  # Nucleus
                # Background remains 0
            else:
                # MULTICLASS MODE: Not implemented for this encoding
                # Keep original values (0 and 255)
                mask_processed = mask_np.copy()
            
            # 4. Apply transformations to image
            if self.transform:
                image = self.transform(image)
            else:
                # Default transformation: just ToTensor
                image = transforms.ToTensor()(image)
            
            # 5. Convert mask to tensor
            mask = torch.from_numpy(mask_processed).long()
            
            return image, mask

        except Exception as e:
            print(f"⚠️ Error loading {os.path.basename(img_path)}: {e}")
            # Return a random different sample on error
            return self.__getitem__((idx + 1) % len(self))

    def get_class_distribution(self):
        """
        Utility method to analyze class distribution.
        """
        class_counts = {0: 0, 255: 0}
        
        for mask_path in self.masks_fps:
            mask = np.array(Image.open(mask_path))
            unique, counts = np.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                class_counts[int(u)] = class_counts.get(int(u), 0) + c
        
        return class_counts
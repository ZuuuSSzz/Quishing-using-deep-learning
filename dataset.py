"""
Custom Dataset class for QR Code Phishing Detection
Loads images from benign and malicious folders and assigns labels accordingly.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class QRCodeDataset(Dataset):
    """
    Custom Dataset for QR Code images.
    
    Args:
        benign_dir: Path to directory containing benign QR code images
        malicious_dir: Path to directory containing malicious QR code images
        transform: Optional transform to be applied on images
        sample_size: Optional number of images to sample per class (None = use all)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        benign_dir: str,
        malicious_dir: str,
        transform: Optional[transforms.Compose] = None,
        sample_size: Optional[int] = None,
        seed: int = 42
    ):
        self.benign_dir = Path(benign_dir)
        self.malicious_dir = Path(malicious_dir)
        self.transform = transform
        self.sample_size = sample_size
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Collect all image paths
        self.image_paths: List[Tuple[str, int]] = []
        
        # Load benign images (label = 0)
        benign_images = self._get_image_paths(self.benign_dir)
        if sample_size and len(benign_images) > sample_size:
            benign_images = random.sample(benign_images, sample_size)
        self.image_paths.extend([(str(img), 0) for img in benign_images])
        
        # Load malicious images (label = 1)
        malicious_images = self._get_image_paths(self.malicious_dir)
        if sample_size and len(malicious_images) > sample_size:
            malicious_images = random.sample(malicious_images, sample_size)
        self.image_paths.extend([(str(img), 1) for img in malicious_images])
        
        # Shuffle the dataset
        random.shuffle(self.image_paths)
        
        print(f"Dataset initialized:")
        print(f"  - Benign images: {len(benign_images)}")
        print(f"  - Malicious images: {len(malicious_images)}")
        print(f"  - Total images: {len(self.image_paths)}")
    
    def _get_image_paths(self, directory: Path) -> List[Path]:
        """Get all image file paths from a directory."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_paths = []
        
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        for file_path in directory.iterdir():
            if file_path.suffix in image_extensions:
                image_paths.append(file_path)
        
        return sorted(image_paths)
    
    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label by index.
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in the dataset."""
        labels = [label for _, label in self.image_paths]
        benign_count = labels.count(0)
        malicious_count = labels.count(1)
        
        return {
            'benign': benign_count,
            'malicious': malicious_count,
            'total': len(labels)
        }


def get_transforms(image_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Get data transforms for training or validation/test.
    
    Args:
        image_size: Target image size (default: 224)
        is_training: If True, apply data augmentation; if False, only normalization
        
    Returns:
        Compose transform object
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),  # Reduced from 0.5 (QR codes are symmetric)
            transforms.RandomRotation(degrees=5),  # Reduced from 10 (QR codes need to be readable)
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Added for QR code variation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


"""
Data utilities for creating train/val/test splits and DataLoaders.
Uses Option A: Sampling strategy (sample first, then split).
"""

import torch
from torch.utils.data import DataLoader, random_split
from dataset import QRCodeDataset, get_transforms


def create_data_splits(
    benign_dir: str,
    malicious_dir: str,
    sample_size: int = 5000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    image_size: int = 224,
    seed: int = 42
):
    """
    Create train/val/test splits using Option A (sampling strategy).
    
    Args:
        benign_dir: Path to benign QR code images
        malicious_dir: Path to malicious QR code images
        sample_size: Number of images to sample per class (Option A)
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        image_size: Target image size (default: 224)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    print("=" * 60)
    print("Creating Data Splits (Option A: Sampling Strategy)")
    print("=" * 60)
    print(f"Sampling {sample_size} images per class...")
    print(f"Total images after sampling: {sample_size * 2}")
    print(f"Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    print()
    
    # Step 1: Create full dataset with sampling (Option A)
    # This automatically samples sample_size images per class
    full_dataset = QRCodeDataset(
        benign_dir=benign_dir,
        malicious_dir=malicious_dir,
        transform=None,  # We'll apply transforms per split
        sample_size=sample_size,
        seed=seed
    )
    
    # Step 2: Split the sampled dataset into train/val/test
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Handle rounding
    
    print(f"Splitting {total_size} images:")
    print(f"  - Train: {train_size} images")
    print(f"  - Val:   {val_size} images")
    print(f"  - Test:  {test_size} images")
    print()
    
    # Create splits
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Step 3: Apply appropriate transforms to each split
    train_transform = get_transforms(image_size=image_size, is_training=True)
    val_transform = get_transforms(image_size=image_size, is_training=False)
    test_transform = get_transforms(image_size=image_size, is_training=False)
    
    # Apply transforms by wrapping datasets
    train_dataset = TransformedDataset(train_dataset, train_transform)
    val_dataset = TransformedDataset(val_dataset, val_transform)
    test_dataset = TransformedDataset(test_dataset, test_transform)
    
    print("✅ Data splits created successfully!")
    print("=" * 60)
    print()
    
    return train_dataset, val_dataset, test_dataset


class TransformedDataset(torch.utils.data.Dataset):
    """
    Wrapper to apply transforms to a dataset.
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    Create DataLoaders for train/val/test datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (faster GPU transfer)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("Creating DataLoaders...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num workers: {num_workers}")
    print()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Keep last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print("✅ DataLoaders created successfully!")
    print()
    
    return train_loader, val_loader, test_loader


def get_data_info(dataloader, name: str = "Dataset"):
    """
    Print information about a DataLoader.
    
    Args:
        dataloader: DataLoader to analyze
        name: Name of the dataset
    """
    total_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    
    # Count class distribution
    labels = []
    for _, label_batch in dataloader:
        labels.extend(label_batch.tolist())
    
    benign_count = labels.count(0)
    malicious_count = labels.count(1)
    
    print(f"{name} Info:")
    print(f"  - Total samples: {total_samples}")
    print(f"  - Number of batches: {num_batches}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Class distribution:")
    print(f"    * Benign (0): {benign_count} ({benign_count/total_samples:.1%})")
    print(f"    * Malicious (1): {malicious_count} ({malicious_count/total_samples:.1%})")
    print()


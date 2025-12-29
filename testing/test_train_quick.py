"""
Quick test script to verify training setup works (without full training).
Tests data loading, model creation, and one training step.
"""

import torch
import yaml
from data_utils import create_data_splits, create_dataloaders
from model import create_model, create_optimizer, create_scheduler
from train import train_epoch, validate

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Testing Training Setup")
print("=" * 60)

# Use smaller sample for quick test
config['data']['sample_size'] = 50  # Very small for quick test

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

# Create data splits (small sample)
print("Creating data splits...")
train_dataset, val_dataset, test_dataset = create_data_splits(
    benign_dir=config['data']['benign_dir'],
    malicious_dir=config['data']['malicious_dir'],
    sample_size=config['data']['sample_size'],
    train_ratio=config['data']['train_ratio'],
    val_ratio=config['data']['val_ratio'],
    test_ratio=config['data']['test_ratio'],
    image_size=config['data']['image_size'],
    seed=config['data']['seed']
)

# Create DataLoaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=8,  # Small batch for testing
    num_workers=2,
    pin_memory=False
)

# Create model
print("\nCreating model...")
model = create_model(
    num_classes=config['model']['num_classes'],
    dropout=config['model']['dropout'],
    device=device
)

# Create loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = create_optimizer(
    model,
    learning_rate=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

print("\nTesting training step...")
# Test one training epoch (should be very quick with small data)
train_loss, train_acc = train_epoch(
    model, train_loader, criterion, optimizer, device
)
print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

print("\nTesting validation step...")
val_loss, val_acc = validate(
    model, val_loader, criterion, device
)
print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

print("\n✅ Training setup test passed!")
print("=" * 60)


"""
Debug script to check if data is being loaded correctly and model is working.
"""

import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from data_utils import create_data_splits, create_dataloaders
from model import create_model

# Load config
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=" * 60)
print("DEBUGGING TRAINING ISSUES")
print("=" * 60)

# Create small dataset for debugging
print("\n1. Checking Data Loading...")
train_dataset, val_dataset, test_dataset = create_data_splits(
    benign_dir=config['data']['benign_dir'],
    malicious_dir=config['data']['malicious_dir'],
    sample_size=100,  # Small sample for debugging
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    image_size=config['data']['image_size'],
    seed=config['data']['seed']
)

# Check a few samples
print("\n2. Checking Sample Data...")
for i in range(5):
    image, label = train_dataset[i]
    print(f"Sample {i}: Image shape: {image.shape}, Label: {label} ({'Benign' if label == 0 else 'Malicious'})")
    print(f"  Image min: {image.min():.3f}, max: {image.max():.3f}, mean: {image.mean():.3f}")

# Check class distribution
print("\n3. Checking Class Distribution...")
train_loader, _, _ = create_dataloaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=16,
    num_workers=2,
    pin_memory=False
)

labels_list = []
for images, labels in train_loader:
    labels_list.extend(labels.tolist())

benign_count = labels_list.count(0)
malicious_count = labels_list.count(1)
print(f"Train set - Benign: {benign_count}, Malicious: {malicious_count}")
print(f"Balance: {benign_count/(benign_count+malicious_count)*100:.1f}% / {malicious_count/(benign_count+malicious_count)*100:.1f}%")

# Check model output
print("\n4. Checking Model Output...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(num_classes=2, dropout=0.5, device=device)
model.eval()

# Test with a batch
sample_batch = next(iter(train_loader))
images, labels = sample_batch
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(outputs, dim=1)

print(f"Model output shape: {outputs.shape}")
print(f"Sample outputs (first 5):")
for i in range(min(5, len(outputs))):
    print(f"  Sample {i}: Logits: {outputs[i].cpu().numpy()}, Probs: {probs[i].cpu().numpy()}, Pred: {predictions[i].item()}, True: {labels[i].item()}")

# Check if model is learning (gradients)
print("\n5. Checking Gradients...")
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

# One training step
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()

# Check gradient norms
total_norm = 0
param_count = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        param_count += 1
        if param_count <= 3:  # Print first 3
            print(f"  {name}: grad norm = {param_norm.item():.6f}")

total_norm = total_norm ** (1. / 2)
print(f"Total gradient norm: {total_norm:.6f}")

# Check loss value
print(f"\n6. Loss Analysis...")
print(f"Loss value: {loss.item():.4f}")
print(f"Expected random loss (ln(2)): {np.log(2):.4f}")
print(f"Difference: {abs(loss.item() - np.log(2)):.4f}")

if abs(loss.item() - np.log(2)) < 0.01:
    print("⚠️  WARNING: Loss is exactly random! Model is not learning.")
    print("   Possible causes:")
    print("   1. Model always outputs same prediction")
    print("   2. Data labels might be wrong")
    print("   3. Model architecture issue")
    print("   4. Data preprocessing issue")

print("\n" + "=" * 60)


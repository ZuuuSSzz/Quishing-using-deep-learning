"""
Test script to verify the model architecture works correctly.
"""

import torch
from model import create_model, create_optimizer, create_scheduler

print("Testing Model Architecture")
print("=" * 60)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

model = create_model(num_classes=2, dropout=0.5, device=device)

# Print model architecture
print("Model Architecture:")
print("-" * 60)
print(model)
print()

# Count parameters
num_params = model.count_parameters()
model_size_mb = model.get_model_size_mb()

print(f"Model Statistics:")
print(f"  - Total parameters: {num_params:,}")
print(f"  - Model size: {model_size_mb:.2f} MB")
print()

# Test forward pass
print("Testing forward pass...")
batch_size = 4
dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

# Forward pass
model.eval()
with torch.no_grad():
    output = model(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
print()

# Test optimizer
print("Testing optimizer (AdamW)...")
optimizer = create_optimizer(model, learning_rate=0.001, weight_decay=0.01)
print(f"Optimizer: {type(optimizer).__name__}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
print(f"Weight decay: {optimizer.param_groups[0]['weight_decay']}")
print()

# Test scheduler
print("Testing scheduler...")
scheduler = create_scheduler(optimizer, mode='reduce_on_plateau')
print(f"Scheduler: {type(scheduler).__name__}")
print()

# Test training step
print("Testing training step...")
model.train()
criterion = torch.nn.CrossEntropyLoss()

# Simulate a training step
optimizer.zero_grad()
output = model(dummy_input)
labels = torch.randint(0, 2, (batch_size,)).to(device)
loss = criterion(output, labels)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
print(f"Gradients computed: {any(p.grad is not None for p in model.parameters())}")
print()

print("✅ Model test passed!")
print("=" * 60)


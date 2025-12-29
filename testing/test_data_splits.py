"""
Test script to verify data splitting (Steps 2.2, 2.3, 2.4) works correctly.
"""

from data_utils import create_data_splits, create_dataloaders, get_data_info

# Paths to your data
BENIGN_DIR = "data/QR_All_benign/QR_All_benign/qrs"
MALICIOUS_DIR = "data/QR_All_benign/QR_All_Malicious/qrs"

print("Testing Data Splitting (Option A: Sampling Strategy)")
print("=" * 60)
print()

# Step 2.2, 2.3: Create data splits with sampling (Option A)
# Using small sample for quick testing
train_dataset, val_dataset, test_dataset = create_data_splits(
    benign_dir=BENIGN_DIR,
    malicious_dir=MALICIOUS_DIR,
    sample_size=100,  # Small sample for testing (100 per class = 200 total)
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    image_size=224,
    seed=42
)

# Step 2.4: Create DataLoaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=16,
    num_workers=2,  # Reduced for testing
    pin_memory=False  # Not needed for CPU
)

# Verify the splits
print("Verifying Data Splits:")
print("=" * 60)
get_data_info(train_loader, "Train")
get_data_info(val_loader, "Validation")
get_data_info(test_loader, "Test")

# Test loading a batch
print("Testing batch loading...")
train_batch = next(iter(train_loader))
images, labels = train_batch

print(f"Batch shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Image dtype: {images.dtype}")
print(f"Label dtype: {labels.dtype}")
print(f"Sample labels: {labels[:5].tolist()}")

print()
print("✅ All data splitting tests passed!")
print("=" * 60)


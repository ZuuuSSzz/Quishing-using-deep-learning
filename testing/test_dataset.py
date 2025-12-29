"""
Quick test script to verify the dataset is working correctly.
Run this after installing dependencies to check if data loading works.
"""

from dataset import QRCodeDataset, get_transforms

# Paths to your data
BENIGN_DIR = "data/QR_All_benign/QR_All_benign/qrs"
MALICIOUS_DIR = "data/QR_All_benign/QR_All_Malicious/qrs"

# Test with a small sample (10 images per class for quick testing)
print("Testing dataset loading...")
print("=" * 50)

# Create dataset with small sample for testing
train_transform = get_transforms(image_size=224, is_training=True)
test_dataset = QRCodeDataset(
    benign_dir=BENIGN_DIR,
    malicious_dir=MALICIOUS_DIR,
    transform=train_transform,
    sample_size=10,  # Just 10 images per class for testing
    seed=42
)

# Check dataset length
print(f"\nDataset length: {len(test_dataset)}")

# Check class distribution
dist = test_dataset.get_class_distribution()
print(f"Class distribution: {dist}")

# Test getting a sample
print("\nTesting __getitem__...")
image, label = test_dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label: {label} ({'Benign' if label == 0 else 'Malicious'})")
print(f"Image dtype: {image.dtype}")
print(f"Image min/max: {image.min():.3f} / {image.max():.3f}")

print("\n✅ Dataset test passed!")


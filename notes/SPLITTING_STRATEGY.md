# Data Splitting Strategy Explanation

## Your Dataset Size
- **Benign images**: ~430,000
- **Malicious images**: ~576,000
- **Total**: ~1,006,000 images

## Option A: Sampling Strategy (Recommended for CPU + Tight Deadline)

### How it works:
1. **First**: Sample a manageable subset (e.g., 5,000 images per class)
2. **Then**: Split that subset into train/val/test

### Example with sample_size=5000:
```
Step 1: Sampling
├── Benign folder (430K images) → Randomly pick 5,000
└── Malicious folder (576K images) → Randomly pick 5,000
    ↓
Subset created: 10,000 images total

Step 2: Splitting (70/15/15)
├── Train: 7,000 images
├── Val:   1,500 images
└── Test:  1,500 images
```

### Code Example:
```python
# Create dataset with sampling
dataset = QRCodeDataset(
    benign_dir="...",
    malicious_dir="...",
    sample_size=5000,  # Only use 5K per class
    ...
)

# Then split this dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)
```

### Pros:
- ✅ Fast training (2-4 hours on CPU)
- ✅ Good for prototyping and testing
- ✅ Fits in memory easily
- ✅ Perfect for tight deadlines

### Cons:
- ❌ Not using full dataset
- ❌ Might miss some patterns (but 10K is usually enough)

---

## Option B: Full Dataset Strategy (Traditional Approach)

### How it works:
1. **Load ALL images** from both folders
2. **Then**: Split the entire dataset into train/val/test

### Example:
```
Step 1: Load Everything
├── Benign folder → All 430,000 images
└── Malicious folder → All 576,000 images
    ↓
Full Dataset: 1,006,000 images

Step 2: Splitting (70/15/15)
├── Train: ~704,200 images
├── Val:   ~150,900 images
└── Test:  ~150,900 images
```

### Code Example:
```python
# Create dataset WITHOUT sampling (sample_size=None)
dataset = QRCodeDataset(
    benign_dir="...",
    malicious_dir="...",
    sample_size=None,  # Use ALL images
    ...
)

# Then split this full dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)
```

### Pros:
- ✅ Uses all available data
- ✅ Maximum model performance potential
- ✅ Better for production/final models

### Cons:
- ❌ Very slow training (2-3 days on CPU)
- ❌ High memory usage
- ❌ Not practical for tight deadlines

---

## Recommendation for Your Deadline (Tomorrow)

**Use Option A with sample_size=5000-7000 per class**

This gives you:
- 10,000-14,000 total images
- Train: 7,000-9,800 images
- Val: 1,500-2,100 images
- Test: 1,500-2,100 images
- Training time: 2-4 hours on CPU

**You can always increase sample_size later if you have more time!**

---

## How to Switch Between Options

In your code, just change the `sample_size` parameter:

```python
# Option A: Sampling (fast)
dataset = QRCodeDataset(..., sample_size=5000, ...)

# Option B: Full dataset (slow)
dataset = QRCodeDataset(..., sample_size=None, ...)
```

That's it! The rest of the code stays the same.


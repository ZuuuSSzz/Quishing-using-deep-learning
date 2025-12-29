# Step 3: Model Architecture - Complete ✅

## 🎯 Model Architecture: QRCodeCNN

### Architecture Overview

```
Input: (batch_size, 3, 224, 224) RGB images
    ↓
Conv Block 1: Conv2d(3→32) + ReLU + MaxPool
    ↓ (batch, 32, 112, 112)
Conv Block 2: Conv2d(32→64) + ReLU + MaxPool
    ↓ (batch, 64, 56, 56)
Conv Block 3: Conv2d(64→128) + ReLU + MaxPool
    ↓ (batch, 128, 28, 28)
Flatten
    ↓ (batch, 100352)
FC1: Linear(100352 → 512) + ReLU + Dropout(0.5)
    ↓ (batch, 512)
FC2: Linear(512 → 128) + ReLU + Dropout(0.5)
    ↓ (batch, 128)
FC3: Linear(128 → 2)
    ↓
Output: (batch, 2) - Binary classification logits
```

### Key Features

1. **3 Convolutional Blocks**: Extract hierarchical features
2. **MaxPooling**: Reduce spatial dimensions (224→112→56→28)
3. **Dropout (0.5)**: Prevent overfitting
4. **Kaiming Initialization**: Better weight initialization for ReLU
5. **Binary Classification**: 2 output classes (Benign=0, Malicious=1)

### Model Statistics

- **Total Parameters**: ~51M parameters
- **Model Size**: ~200 MB (when saved)
- **Input Size**: 224x224 RGB images
- **Output**: 2 logits (before softmax)

---

## 🚀 Optimizer: AdamW (Why Better Than Adam?)

### AdamW Advantages

1. **Decoupled Weight Decay**:
   - Adam: Weight decay is coupled with gradient update
   - AdamW: Weight decay is decoupled (applied separately)
   - **Result**: Better regularization and generalization

2. **Better Generalization**:
   - Often achieves better test accuracy
   - More stable training
   - Less prone to overfitting

3. **PyTorch Recommendation**:
   - PyTorch recommends AdamW as default
   - Used in modern architectures (Transformers, etc.)

4. **Hyperparameter Settings**:
   ```python
   AdamW(
       lr=0.001,
       weight_decay=0.01,  # Decoupled L2 regularization
       betas=(0.9, 0.999),
       eps=1e-8
   )
   ```

### Comparison: Adam vs AdamW

| Feature | Adam | AdamW |
|---------|------|-------|
| Weight Decay | Coupled | **Decoupled** ✅ |
| Generalization | Good | **Better** ✅ |
| Stability | Good | **More Stable** ✅ |
| Modern Usage | Legacy | **Recommended** ✅ |

---

## 📁 Files Created

1. **`model.py`**
   - `QRCodeCNN`: Main CNN architecture
   - `create_model()`: Model factory function
   - `create_optimizer()`: Creates AdamW optimizer
   - `create_scheduler()`: Creates LR scheduler
   - Helper functions for parameter counting

2. **`test_model.py`**
   - Test script to verify model works
   - Tests forward pass, optimizer, scheduler

3. **`config.yaml`** (Updated)
   - Added optimizer settings
   - Added scheduler settings
   - Added dropout parameter

---

## 🧪 Testing

Run the test script:
```bash
uv run test_model.py
```

This will:
- Create the model
- Print architecture
- Count parameters
- Test forward pass
- Test optimizer (AdamW)
- Test scheduler
- Test training step

---

## 📊 Model Details

### Layer Breakdown

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| Conv1 | (3, 224, 224) | (32, 112, 112) | 896 |
| Conv2 | (32, 112, 112) | (64, 56, 56) | 18,496 |
| Conv3 | (64, 56, 56) | (128, 28, 28) | 73,856 |
| FC1 | 100,352 | 512 | 51,380,736 |
| FC2 | 512 | 128 | 65,664 |
| FC3 | 128 | 2 | 258 |
| **Total** | | | **~51M** |

### Loss Function

- **CrossEntropyLoss**: Standard for multi-class classification
  - Includes softmax internally
  - Handles class imbalance
  - Works with logits (no softmax needed in forward pass)

### Learning Rate Scheduler

- **ReduceLROnPlateau**: Reduces LR when validation loss plateaus
  - Factor: 0.5 (halve LR)
  - Patience: 3 epochs
  - Monitors validation loss

---

## 🎯 Next Steps

Ready for:
- **Step 4**: Training Loop (`train.py`)
- **Step 5**: Evaluation (`test.py`)

---

## 💡 Why This Architecture?

1. **Simple but Effective**: 3 conv layers sufficient for QR codes
2. **Not Too Deep**: Avoids overfitting on limited data
3. **Dropout**: Regularization for better generalization
4. **AdamW**: Modern optimizer for better results
5. **Binary Classification**: Perfect for benign vs malicious

The model is designed to be:
- ✅ Fast to train (2-4 hours on CPU)
- ✅ Good accuracy (expected 85-95%)
- ✅ Reasonable size (~200 MB)
- ✅ Easy to understand and modify


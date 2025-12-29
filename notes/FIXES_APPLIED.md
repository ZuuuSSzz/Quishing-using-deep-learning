# Fixes Applied to Improve Model Training

## 🔧 Changes Made

### 1. **Learning Rate Reduction**
**Before:** `learning_rate: 0.001`  
**After:** `learning_rate: 0.0005`  
**Why:** Lower learning rate prevents overshooting and allows more stable convergence. The original 0.001 was too high, causing the model to not learn properly.

### 2. **Weight Decay Reduction**
**Before:** `weight_decay: 0.01`  
**After:** `weight_decay: 0.001`  
**Why:** 0.01 was too aggressive, preventing the model from learning. Reduced to 0.001 for better balance between regularization and learning.

### 3. **Gradient Clipping Added**
**Before:** No gradient clipping  
**After:** `gradient_clip: 1.0`  
**Why:** Prevents exploding gradients which can cause training instability. Clips gradients at norm 1.0.

### 4. **More Training Epochs**
**Before:** `epochs: 10`  
**After:** `epochs: 20`  
**Why:** 10 epochs wasn't enough for the model to learn. Increased to 20 to give more time for convergence.

### 5. **Scheduler Patience Increased**
**Before:** `scheduler_patience: 3`  
**After:** `scheduler_patience: 5`  
**Why:** More patience before reducing learning rate gives the model more time to improve before LR reduction.

### 6. **Model Architecture Simplified**
**Before:**
- FC1: 100352 → 512
- FC2: 512 → 128
- FC3: 128 → 2

**After:**
- FC1: 100352 → 256
- FC2: 256 → 64
- FC3: 64 → 2

**Why:** Smaller model is easier to train, reduces overfitting risk, and trains faster. The original model was too large for the dataset size.

### 7. **Better Weight Initialization**
**Before:** Kaiming init for all layers  
**After:** 
- Conv2d: Kaiming init (good for ReLU)
- Linear: Xavier init (better for final layers)
- BatchNorm: Proper initialization

**Why:** Different initialization strategies for different layer types improves training stability.

---

## 📊 Expected Improvements

### Before Fixes:
- ❌ Validation Accuracy: ~50% (random)
- ❌ Validation Loss: ~0.693 (stuck, not decreasing)
- ❌ Training Loss: Drops to 0 immediately (bug/issue)
- ❌ Model not learning

### After Fixes:
- ✅ Validation Accuracy: Should improve above 50% (target: 70-90%)
- ✅ Validation Loss: Should decrease over epochs
- ✅ Training Loss: Should decrease gradually (not drop to 0 immediately)
- ✅ Model should learn patterns

---

## 🎯 Key Differences Summary

| Aspect | Before | After | Impact |
|--------|--------|-------|--------|
| **Learning Rate** | 0.001 | 0.0005 | More stable training |
| **Weight Decay** | 0.01 | 0.001 | Less aggressive regularization |
| **Gradient Clipping** | None | 1.0 | Prevents exploding gradients |
| **Epochs** | 10 | 20 | More time to learn |
| **Scheduler Patience** | 3 | 5 | More time before LR reduction |
| **Model Size** | Large (512→128) | Smaller (256→64) | Easier to train |
| **Initialization** | Kaiming only | Mixed (Kaiming+Xavier) | Better stability |

---

## 🚀 Next Steps

1. **Retrain the model** with these fixes:
   ```bash
   uv run train.py
   ```

2. **Monitor improvements**:
   - Validation accuracy should increase above 50%
   - Validation loss should decrease
   - Training should be more stable

3. **If still not working**, check:
   - Data loading (verify images and labels)
   - Data balance (should be ~50/50)
   - Image quality (QR codes should be readable)

---

## 💡 Why These Fixes Work

1. **Lower LR**: Prevents the optimizer from overshooting optimal weights
2. **Lower Weight Decay**: Allows model to learn without being over-regularized
3. **Gradient Clipping**: Prevents training instability from large gradients
4. **More Epochs**: Gives model time to learn complex patterns
5. **Smaller Model**: Easier to train, less prone to overfitting
6. **Better Init**: Starts training from a better point

These changes address the core issues: learning rate too high, model too complex, and lack of training stability mechanisms.


# Complete Project Run Flow

## 🎯 Step-by-Step Guide to Run the Project

### Prerequisites Check
```bash
# 1. Navigate to project directory
cd /home/zuss/pytorch/quishing-with-ml

# 2. Verify dependencies are installed
uv sync

# 3. Check if wandb is available (optional)
python -c "import wandb; print('Wandb available')" || echo "Wandb not installed"
```

---

## 📋 Phase 1: Quick Tests (5-10 minutes)

**Purpose**: Verify all components work before full training

### Test 1: Dataset Loading
```bash
uv run test_dataset.py
```
**Expected Output**: 
- ✅ Dataset initialized with images
- ✅ Image shape: torch.Size([3, 224, 224])
- ✅ Labels working correctly

### Test 2: Data Splitting
```bash
uv run test_data_splits.py
```
**Expected Output**:
- ✅ Train/Val/Test splits created
- ✅ DataLoaders working
- ✅ Batch loading successful

### Test 3: Model Architecture
```bash
uv run test_model.py
```
**Expected Output**:
- ✅ Model created successfully
- ✅ Forward pass works
- ✅ Optimizer (AdamW) initialized
- ✅ Scheduler working

### Test 4: Training Setup (Quick)
```bash
uv run test_train_quick.py
```
**Expected Output**:
- ✅ Training step works
- ✅ Validation step works
- ✅ Loss computed correctly

**If all tests pass → Proceed to Phase 2**

---

## 🚀 Phase 2: Full Training (2-4 hours on CPU)

### Step 1: Setup Wandb (Optional but Recommended)

```bash
# Login to wandb (first time only)
wandb login
# Enter your API key from https://wandb.ai/authorize
```

### Step 2: Configure Wandb in config.yaml

Edit `config.yaml`:
```yaml
logging:
  use_wandb: true  # Change from false to true
  wandb_project: "qr-phishing-detection"
  wandb_entity: null  # Or your username
```

### Step 3: Review Training Configuration

Check `config.yaml` settings:
```yaml
data:
  sample_size: 5000  # Images per class (10K total)
  
training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
```

### Step 4: Start Training

```bash
uv run train.py
```

### What Happens During Training:

1. **Data Loading** (1-2 minutes)
   - Samples 5,000 images per class
   - Creates train/val/test splits
   - Creates DataLoaders

2. **Model Creation** (< 1 minute)
   - Creates CNN model
   - Initializes optimizer (AdamW)
   - Sets up scheduler

3. **Wandb Initialization** (if enabled)
   - Logs: "✓ Weights & Biases initialized"
   - Starts tracking run

4. **Training Loop** (10 epochs × ~10-15 min = 2-2.5 hours)
   ```
   For each epoch:uv 
   ├── Training Phase (~10-15 min)
   │   ├── Forward pass
   │   ├── Loss computation
   │   ├── Backward pass
   │   └── Weight update
   │
   ├── Validation Phase (~2-3 min)
   │   ├── Forward pass (no gradients)
   │   └── Metrics computation
   │
   ├── Logging to Wandb (if enabled)
   │   ├── Train/Val Loss
   │   ├── Train/Val Accuracy
   │   ├── Learning Rate
   │   └── Best metrics
   │
   └── Model Saving
       └── Saves if validation loss improved
   ```

5. **Training Complete**
   - Best model saved to `models/best_model.pth`
   - Training plots saved to `models/training_history.png`
   - Wandb run completed

### Expected Console Output:
```
Using device: cpu
============================================================
Loading data...
Creating Data Splits (Option A: Sampling Strategy)
Sampling 5000 images per class...
...

Creating model...
Model parameters: 51,539,906
Model size: 196.61 MB

✓ Weights & Biases initialized  # If wandb enabled

============================================================
Starting Training
============================================================
Epoch 1/10
------------------------------------------------------------
Training: 100%|████████| 219/219 [10:23<00:00, 1.65s/it]
Validating: 100%|██████| 47/47 [02:15<00:00, 2.88s/it]
Train Loss: 0.6234 | Train Acc: 65.23%
Val Loss:   0.5891 | Val Acc:   68.45%
LR: 0.001000
✓ Saved best model (Val Loss: 0.5891)

Epoch 2/10
...

============================================================
Training Complete!
============================================================
Total training time: 125.34 minutes
Best validation loss: 0.2345
Best validation accuracy: 89.23%
Model saved to: models/best_model.pth
Training plots saved to: models/training_history.png
✓ Wandb run completed
```

### Verify Wandb Logging:

1. **During Training**: Check console for:
   - "✓ Weights & Biases initialized"
   - No errors about wandb

2. **After Training**: Check console for:
   - "✓ Wandb run completed"

3. **Online Dashboard**: 
   - Go to https://wandb.ai
   - Navigate to project: `qr-phishing-detection`
   - See your run with all metrics!

---

## 📊 Phase 3: Evaluation (5-10 minutes)

### Step 1: Run Evaluation

```bash
uv run test.py
```

### What Happens:

1. **Load Model** (< 1 minute)
   - Loads `models/best_model.pth`
   - Displays model info

2. **Load Test Data** (< 1 minute)
   - Uses same data splits as training

3. **Evaluate** (2-5 minutes)
   - Forward pass on test set
   - Computes all metrics

4. **Generate Reports** (< 1 minute)
   - Confusion matrix plot
   - Classification report

### Expected Output:
```
Using device: cpu
============================================================
Loading test data...
Loading model...
Model loaded from: models/best_model.pth
Model was trained for 10 epochs
Best validation accuracy: 89.23%

============================================================
Evaluating on Test Set
============================================================
Evaluating: 100%|████████| 47/47 [XX:XX<00:00, X.XXit/s]

============================================================
EVALUATION METRICS
============================================================
Overall Metrics:
  Accuracy:  0.8923 (89.23%)
  Precision: 0.8934 (89.34%)
  Recall:    0.8923 (89.23%)
  F1-Score:  0.8928 (89.28%)

...

Confusion matrix saved to: models/confusion_matrix.png
```

---

## 🔍 Verification Checklist

### ✅ After Phase 1 (Tests):
- [ ] All 4 test scripts pass
- [ ] No errors in console

### ✅ After Phase 2 (Training):
- [ ] Training completes without errors
- [ ] Model saved: `models/best_model.pth` exists
- [ ] Plot saved: `models/training_history.png` exists
- [ ] Wandb: Run visible at wandb.ai (if enabled)
- [ ] Wandb: Metrics logged (loss, accuracy, LR)
- [ ] Console shows: "✓ Wandb run completed"

### ✅ After Phase 3 (Evaluation):
- [ ] Evaluation completes successfully
- [ ] All metrics printed
- [ ] Confusion matrix saved: `models/confusion_matrix.png`
- [ ] Test accuracy reported

---

## 🐛 Troubleshooting

### Wandb Not Logging?

1. **Check if enabled**:
   ```bash
   grep "use_wandb" config.yaml
   # Should show: use_wandb: true
   ```

2. **Check if logged in**:
   ```bash
   wandb login
   # Should say: Successfully logged in
   ```

3. **Check console output**:
   - Should see: "✓ Weights & Biases initialized"
   - Should see: "✓ Wandb run completed"

4. **Check wandb.ai**:
   - Go to https://wandb.ai
   - Check project: `qr-phishing-detection`
   - Should see your run

### Training Too Slow?

- Reduce `sample_size` in config.yaml (e.g., 2000 instead of 5000)
- Reduce `batch_size` if memory issues
- Reduce `epochs` for quick test

### Out of Memory?

- Reduce `batch_size` in config.yaml (e.g., 16 instead of 32)
- Reduce `sample_size` (fewer images)
- Reduce `image_size` (e.g., 128 instead of 224)

---

## 📈 Complete Flow Summary

```
1. Quick Tests (5-10 min)
   ├── test_dataset.py ✅
   ├── test_data_splits.py ✅
   ├── test_model.py ✅
   └── test_train_quick.py ✅

2. Full Training (2-4 hours)
   ├── Setup wandb (login + config)
   ├── Run: uv run train.py
   ├── Monitor console output
   ├── Check wandb.ai dashboard
   └── Verify model saved

3. Evaluation (5-10 min)
   ├── Run: uv run test.py
   ├── Review metrics
   └── Check plots generated
```

---

## 🎯 Quick Start (Minimal)

If you just want to verify everything works quickly:

```bash
# 1. Quick tests
uv run test_dataset.py
uv run test_data_splits.py
uv run test_model.py
uv run test_train_quick.py

# 2. Quick training (small sample, few epochs)
# Edit config.yaml: sample_size: 100, epochs: 2
uv run train.py

# 3. Evaluate
uv run test.py
```

This will take ~15-20 minutes instead of 2-4 hours!


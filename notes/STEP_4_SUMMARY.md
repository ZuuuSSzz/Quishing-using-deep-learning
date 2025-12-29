# Step 4: Training Loop - Complete ✅

## 🎯 Training Script Features

### Core Components

1. **Training Loop** (`train_epoch`)
   - Forward pass
   - Loss computation
   - Backward pass
   - Optimizer step
   - Accuracy tracking

2. **Validation Loop** (`validate`)
   - Evaluation mode (no gradients)
   - Loss and accuracy computation
   - Used to monitor overfitting

3. **Model Saving**
   - Saves best model based on validation loss
   - Includes model state, optimizer state, and training history
   - Saves to `models/best_model.pth`

4. **Learning Rate Scheduling**
   - ReduceLROnPlateau: Reduces LR when validation loss plateaus
   - Automatically adjusts learning rate during training

5. **Training History**
   - Tracks train/val loss and accuracy per epoch
   - Plots saved automatically as `training_history.png`

## 📁 Files Created

1. **`train.py`** - Main training script
   - `train()`: Main training function
   - `train_epoch()`: One training epoch
   - `validate()`: Validation function
   - `plot_training_history()`: Plot loss/accuracy curves

2. **`test_train_quick.py`** - Quick test script
   - Tests training setup without full training
   - Verifies data loading, model, and one training step

## 🚀 Usage

### Full Training
```bash
# Train with default config.yaml
uv run train.py

# Train with custom config
uv run train.py --config my_config.yaml

# Train without saving plots
uv run train.py --no-plots
```

### Quick Test
```bash
# Test training setup (quick, uses small sample)
uv run test_train_quick.py
```

## 📊 Training Process

### What Happens During Training

1. **Data Loading**
   - Loads train/val/test splits (Option A: sampled)
   - Creates DataLoaders with appropriate batch size

2. **Model Initialization**
   - Creates CNN model
   - Initializes optimizer (AdamW)
   - Sets up learning rate scheduler

3. **Training Loop** (for each epoch)
   ```
   For each epoch:
     ├── Train Phase:
     │   ├── Forward pass
     │   ├── Compute loss
     │   ├── Backward pass
     │   ├── Update weights
     │   └── Track metrics
     │
     ├── Validation Phase:
     │   ├── Forward pass (no gradients)
     │   ├── Compute loss & accuracy
     │   └── Track metrics
     │
     ├── Update Learning Rate:
     │   └── Scheduler step (if validation loss plateaus)
     │
     └── Save Best Model:
         └── If validation loss improved
   ```

4. **After Training**
   - Saves best model checkpoint
   - Generates loss/accuracy plots
   - Prints training summary

## 📈 Output Files

After training, you'll have:

1. **`models/best_model.pth`**
   - Model weights
   - Optimizer state
   - Training history
   - Best validation metrics

2. **`models/training_history.png`**
   - Loss curves (train vs val)
   - Accuracy curves (train vs val)

## 🎛️ Configuration

All training settings in `config.yaml`:

```yaml
training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.01
  scheduler: "reduce_on_plateau"
  save_best: true
```

## 📊 Expected Output

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

============================================================
Starting Training
============================================================
Epoch 1/10
------------------------------------------------------------
Training: 100%|████████| 219/219 [XX:XX<00:00, X.XXit/s]
Validating: 100%|██████| 47/47 [XX:XX<00:00, X.XXit/s]
Train Loss: 0.6234 | Train Acc: 65.23%
Val Loss:   0.5891 | Val Acc:   68.45%
LR: 0.001000
✓ Saved best model (Val Loss: 0.5891)

...

============================================================
Training Complete!
============================================================
Total training time: 125.34 minutes
Best validation loss: 0.2345
Best validation accuracy: 89.23%
Model saved to: models/best_model.pth
Training plots saved to: models/training_history.png
```

## ⏱️ Training Time Estimates (CPU)

With `sample_size=5000` (10K total images):
- **Per epoch**: ~10-15 minutes
- **10 epochs**: ~2-2.5 hours
- **20 epochs**: ~4-5 hours

## ✨ Key Features

1. **Automatic Model Saving**: Saves best model based on validation loss
2. **Progress Tracking**: tqdm progress bars for each phase
3. **Learning Rate Scheduling**: Automatically adjusts LR
4. **History Plotting**: Automatic loss/accuracy curve generation
5. **Reproducible**: Uses config file for all settings
6. **Flexible**: Easy to modify hyperparameters

## 🎯 Next Steps

Ready for:
- **Step 5**: Evaluation (`test.py`)
  - Load best model
  - Evaluate on test set
  - Compute metrics (accuracy, precision, recall, F1)
  - Generate confusion matrix
  - Measure efficiency metrics

## 💡 Tips

1. **Monitor Training**: Watch for overfitting (val loss increasing while train loss decreases)
2. **Early Stopping**: Can add early stopping if validation loss doesn't improve
3. **Adjust Batch Size**: If OOM, reduce batch_size in config.yaml
4. **Reduce Sample Size**: For faster iteration, reduce `sample_size` in config.yaml
5. **Check Plots**: Review `training_history.png` to see training progress


# Understanding `use_wandb: true`

## 📝 What is `use_wandb`?

It's a **switch** in your `config.yaml` file that turns Wandb tracking ON or OFF.

---

## 🔄 Current Setting

**File**: `config.yaml`  
**Line**: 64  
**Current value**: `false`

```yaml
logging:
  use_wandb: false  # ← Currently OFF
```

---

## ✅ How to Enable (Change to `true`)

### Step 1: Open `config.yaml`
```bash
# In your editor, open:
config.yaml
```

### Step 2: Find the line
Look for line 64, under `logging:` section:
```yaml
logging:
  # Experiment tracking
  use_wandb: false  # ← This line
```

### Step 3: Change `false` to `true`
```yaml
logging:
  # Experiment tracking
  use_wandb: true  # ← Changed to true
```

### Step 4: Save the file

---

## 🎯 What Happens When You Change It?

### Before (`use_wandb: false`):
```
Training starts...
Epoch 1/10
Train Loss: 0.6234 | Train Acc: 65.23%
Val Loss:   0.5891 | Val Acc:   68.45%
(No wandb messages)
```

### After (`use_wandb: true`):
```
Training starts...
✓ Weights & Biases initialized  ← NEW!
Epoch 1/10
Train Loss: 0.6234 | Train Acc: 65.23%
Val Loss:   0.5891 | Val Acc:   68.45%
(All metrics logged to wandb.ai)
...
✓ Wandb run completed  ← NEW!
```

---

## 📊 What Gets Logged to Wandb?

When `use_wandb: true`, these are automatically logged:

1. **Hyperparameters**:
   - Batch size, epochs, learning rate
   - Model parameters, model size

2. **Training Metrics** (per epoch):
   - Train Loss
   - Train Accuracy
   - Validation Loss
   - Validation Accuracy
   - Learning Rate

3. **Best Metrics**:
   - Best validation loss
   - Best validation accuracy

4. **Training Time**:
   - Total training time

5. **Plots**:
   - Loss curves
   - Accuracy curves

---

## 🔍 How to Verify It's Working

### Check 1: Console Output
When training starts, you should see:
```
✓ Weights & Biases initialized
```

### Check 2: Wandb Dashboard
1. Go to: https://wandb.ai
2. Navigate to project: `qr-phishing-detection`
3. See your training run with all metrics!

### Check 3: End of Training
You should see:
```
✓ Wandb run completed
```

---

## ⚙️ Complete Config Section

Here's what the `logging` section looks like:

```yaml
logging:
  # Experiment tracking
  use_wandb: true              # ← Enable/disable wandb
  wandb_project: "qr-phishing-detection"  # Project name
  wandb_entity: null           # Your username (optional)
  wandb_api_key: null          # Optional (or use 'wandb login')
```

---

## 💡 Quick Summary

| Setting | What It Does |
|---------|-------------|
| `use_wandb: false` | Training runs, NO wandb logging |
| `use_wandb: true` | Training runs, WITH wandb logging |

**To enable**: Change `false` → `true` in `config.yaml` line 64

---

## 🚀 After Changing

1. **Save** `config.yaml`
2. **Run training**:
   ```bash
   uv run train.py
   ```
3. **Check wandb.ai** for your metrics!

---

## ❓ Common Questions

**Q: Do I need to change anything else?**  
A: No! Just change `false` to `true`. Make sure you're logged in (`uv run wandb login`).

**Q: What if I don't want wandb?**  
A: Keep it as `false`. Training works perfectly without it.

**Q: Can I turn it on/off anytime?**  
A: Yes! Change the setting and restart training.


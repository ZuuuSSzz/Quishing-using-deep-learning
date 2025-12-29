# Weights & Biases (Wandb) Integration

## ✅ Wandb is Now Integrated!

Your training script now supports Weights & Biases for experiment tracking and visualization, just like the example `IDS_CICIDS2017.py`.

## 🚀 How to Enable Wandb

### Step 1: Install Wandb (if not already installed)
```bash
# Already in pyproject.toml, so just run:
uv sync
```

### Step 2: Login to Wandb
```bash
wandb login
```
Enter your API key when prompted. You can get it from: https://wandb.ai/authorize

### Step 3: Enable in Config
Edit `config.yaml`:
```yaml
logging:
  use_wandb: true  # Change from false to true
  wandb_project: "qr-phishing-detection"  # Your project name
  wandb_entity: null  # Your username (optional, null = default)
```

### Step 4: Run Training
```bash
uv run train.py
```

## 📊 What Gets Logged to Wandb

### During Training (per epoch):
- ✅ **Train Loss**: Training loss
- ✅ **Train Acc**: Training accuracy
- ✅ **Val Loss**: Validation loss
- ✅ **Val Acc**: Validation accuracy
- ✅ **Learning Rate**: Current learning rate
- ✅ **Best Val Loss**: Best validation loss so far
- ✅ **Best Val Acc**: Best validation accuracy so far

### At Training Start:
- ✅ **Hyperparameters**: batch_size, epochs, learning_rate, etc.
- ✅ **Model Info**: parameters count, model size
- ✅ **Data Info**: sample_size, image_size

### At Training End:
- ✅ **Total Training Time**: Time in minutes
- ✅ **Final Best Metrics**: Best validation loss and accuracy
- ✅ **Training History Plot**: Loss/accuracy curves

### Model Watching:
- ✅ **Gradients**: Model gradients are logged
- ✅ **Parameters**: Model parameters are tracked

## 🎯 Visualizations in Wandb Dashboard

Once training starts, you can view:

1. **Loss Curves**: Train vs Validation loss over epochs
2. **Accuracy Curves**: Train vs Validation accuracy over epochs
3. **Learning Rate**: LR schedule over time
4. **System Metrics**: CPU/GPU usage, memory (if available)
5. **Model Architecture**: Model graph visualization
6. **Hyperparameters**: All config values in one place

## 📝 Example Usage

### Basic Training with Wandb:
```bash
# Enable wandb in config.yaml first
uv run train.py
```

### View Results:
1. Go to https://wandb.ai
2. Navigate to your project: `qr-phishing-detection`
3. See all your training runs with metrics and plots!

## 🔧 Configuration Options

In `config.yaml`:
```yaml
logging:
  use_wandb: true                    # Enable/disable wandb
  wandb_project: "qr-phishing-detection"  # Project name
  wandb_entity: null                  # Your username (optional)
```

## 💡 Tips

1. **Compare Runs**: Run multiple experiments and compare them in Wandb
2. **Hyperparameter Tuning**: Use Wandb Sweeps for automated tuning
3. **Team Collaboration**: Share your project with teammates
4. **Experiment Tracking**: Keep track of all your model versions

## 🎨 What You'll See in Wandb

Similar to your `IDS_CICIDS2017.py` example, you'll see:
- Real-time loss and accuracy plots
- Learning rate schedule
- Model performance metrics
- Training time and efficiency
- All hyperparameters in one place

## ⚠️ Note

- Wandb is **optional** - if `use_wandb: false`, training works normally without it
- If wandb is not installed, training will continue without it (graceful fallback)
- You can enable/disable it anytime in the config file

## 🚀 Quick Start

```bash
# 1. Login to wandb
wandb login

# 2. Enable in config.yaml
# Set use_wandb: true

# 3. Run training
uv run train.py

# 4. View results at wandb.ai
```

Enjoy visualizing your training! 🎉


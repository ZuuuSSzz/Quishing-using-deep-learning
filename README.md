# QR Code Phishing Detection using Deep Learning

A PyTorch-based deep learning project for detecting phishing attempts in QR codes using Convolutional Neural Networks (CNN) and Transfer Learning.

## 📋 Project Overview

This project implements a binary classification system to distinguish between **benign** and **malicious** QR codes. The system supports both **custom CNN architecture** and **transfer learning** with pre-trained models (ResNet, EfficientNet, MobileNet).

### Key Features

- **Dual Model Support**: Custom CNN or Transfer Learning (ResNet, EfficientNet, MobileNet)
- **Transfer Learning**: Pre-trained ResNet18 achieving **78.34% validation accuracy**
- **Data Augmentation**: Random flips, rotations, and color jittering
- **Experiment Tracking**: Integrated with Weights & Biases (wandb)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Efficiency Metrics**: Inference time, throughput, FLOPS, memory usage, model size
- **Modular Design**: Separate modules for dataset, model, training, and evaluation
- **GPU Optimized**: Supports CUDA for faster training

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- `uv` package manager (for dependency management)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/user/quishing-with-ml
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up Weights & Biases (Optional but recommended):**
   ```bash
   uv run wandb login
   ```
   This will use your currently logged-in account. No need to hardcode API keys!

### Dataset

**The dataset is not included in this repository due to size (~4GB).**

#### Download Dataset

Download the dataset from one of the following sources:

- **CIC-Trap4Phish 2025 Dataset**: [Download from UNB CIC](https://www.unb.ca/cic/datasets/trap4phish2025.html)

**Or use your own dataset** by placing QR code images in the following structure:

```
data/
├── QR_All_benign/
│   └── QR_All_benign/
│       └── qrs/          # Benign QR code images
└── QR_All_benign/
    └── QR_All_Malicious/
        └── qrs/          # Malicious QR code images
```

**Note:** The dataset paths can be configured in `config.yaml`

## 📖 Usage

### 1. Configuration

Edit `config.yaml` to adjust hyperparameters:

```yaml
data:
  sample_size: 100000     # Images per class (100K = 200K total)
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

model:
  model_type: "transfer"  # "cnn" or "transfer"
  model_name: "resnet18"  # Pre-trained model name

training:
  epochs: 10
  batch_size: 64          # Optimized for GPU
  learning_rate: 0.00001
  weight_decay: 0.0005
```

### 2. Training

**Option A: Using Python script**
```bash
uv run python train.py
```

**Option B: Using Jupyter Notebook**
```bash
uv run jupyter notebook notebook.ipynb
```
Then run all cells sequentially.

### 2.1 Switching Between Models

**Use Custom CNN:**
```yaml
# config.yaml
model:
  model_type: "cnn"
```

**Use Transfer Learning (Recommended):**
```yaml
# config.yaml
model:
  model_type: "transfer"
  model_name: "resnet18"  # or "resnet34", "efficientnet_b0", etc.
```

No code changes needed - just modify `config.yaml` and run training!

### 3. Evaluation

After training, evaluate the model:

```bash
uv run python test.py
```

Or specify a custom model path:
```bash
uv run python test.py --model path/to/model.pth
```

## 📁 Project Structure

```
quishing-with-ml/
├── config.yaml              # Configuration file
├── dataset.py               # Custom Dataset class
├── data_utils.py            # Data splitting and loading utilities
├── model.py                 # Model architecture (CNN + Transfer Learning)
├── train.py                 # Training script
├── test.py                  # Evaluation script
├── main.py                  # Entry point
├── README.md                # This file
├── REPORT.md                # Project report
├── COMPARISON.md            # CNN vs Transfer Learning comparison
├── ASSIGNMENT_CHECKLIST.md  # Assignment requirements checklist
├── pyproject.toml           # Dependencies
└── models/                  # Saved models (created after training)
    ├── best_model.pth
    └── training_history.png
```

## 🏗️ Model Architecture

The project supports two model architectures:

### Option 1: Custom CNN (Default)
- **Convolutional Layers**: 3 blocks (32 → 64 → 128 channels)
- **Pooling**: MaxPool2d after each conv block
- **Fully Connected**: 512 → 256 → 2 (output classes)
- **Dropout**: 0.3 for regularization
- **Activation**: ReLU (except output layer)
- **Total Parameters**: ~2.5M
- **Model Size**: ~10 MB

### Option 2: Transfer Learning (Recommended) ⭐
- **Pre-trained Models**: ResNet18, ResNet34, ResNet50, EfficientNet, MobileNet
- **Currently Used**: ResNet18
- **Total Parameters**: ~11.2M (ResNet18)
- **Model Size**: ~42.7 MB (ResNet18)
- **Performance**: 78.34% validation accuracy
- **Training Time**: 25.15 minutes (10 epochs, 200K images, GPU)

**Switch between models** by changing `model_type` in `config.yaml`:
- `model_type: "cnn"` - Use custom CNN
- `model_type: "transfer"` - Use pre-trained model

## 📊 Evaluation Metrics

The model is evaluated using:

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed per-class performance

### Efficiency Metrics
- **Inference Time**: Average time per sample (ms)
- **Throughput**: Samples processed per second
- **Model Size**: Disk storage (MB)
- **Parameters**: Total trainable parameters
- **FLOPs**: Floating Point Operations (computational complexity)
- **Memory Usage**: Runtime memory consumption (GPU/CPU)
- **Training Time**: Total training duration

## 🔧 Configuration Options

### Data Configuration

- `sample_size`: Number of images to sample per class (None = use all)
- `train_ratio`, `val_ratio`, `test_ratio`: Data split ratios
- `image_size`: Input image size (default: 224x224)
- `seed`: Random seed for reproducibility

### Training Configuration

- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Initial learning rate
- `weight_decay`: L2 regularization strength
- `gradient_clip`: Gradient clipping value (prevents exploding gradients)
- `scheduler`: Learning rate scheduler type
- `scheduler_patience`: Patience for ReduceLROnPlateau

### Model Configuration

- `model_type`: Model type - `"cnn"` (custom) or `"transfer"` (pre-trained)
- `model_name`: Pre-trained model name (if using transfer learning)
  - Options: `"resnet18"`, `"resnet34"`, `"resnet50"`, `"efficientnet_b0"`, `"efficientnet_b1"`, `"mobilenet_v3_small"`, `"mobilenet_v3_large"`
- `num_classes`: Number of output classes (2 for binary)
- `dropout`: Dropout probability

### Logging Configuration

- `use_wandb`: Enable/disable Weights & Biases tracking
- `wandb_project`: WandB project name
- `wandb_entity`: WandB entity/username (optional, can use `WANDB_ENTITY` env var)

## 📈 Weights & Biases Integration

The project integrates with Weights & Biases for experiment tracking:

1. **Login to wandb**:
   ```bash
   uv run wandb login
   ```
   This will use your currently logged-in account. No API keys need to be hardcoded!

2. **Enable in config**:
   ```yaml
   logging:
     use_wandb: true
     wandb_project: "qr-phishing-detection"
   ```

3. **Switch accounts without modifying code** (recommended):
   - **Method 1**: Use `wandb login` to switch accounts
     ```bash
     uv run wandb login --relogin
     ```
   
   - **Method 2**: Use environment variables (for temporary switching)
     ```bash
     export WANDB_ENTITY="your-entity-name"
     export WANDB_API_KEY="your-api-key"  # Optional if already logged in
     export WANDB_PROJECT="your-project"   # Optional, overrides config
     ```
   
   - **Method 3**: Set entity in config.yaml (permanent)
     ```yaml
     logging:
       wandb_entity: "your-entity-name"
     ```

4. **View results**: Training metrics are automatically logged to your WandB dashboard.

## 🧪 Testing Individual Components

The project includes test scripts for individual components:

```bash
# Test dataset loading
uv run python testing/test_dataset.py

# Test data splits
uv run python testing/test_data_splits.py

# Test model architecture
uv run python testing/test_model.py

# Quick training test
uv run python testing/test_train_quick.py
```

## 📝 Key Files

- **`notebook.ipynb`**: Main deliverable - Complete pipeline in Jupyter notebook
- **`train.py`**: Training script with validation monitoring
- **`test.py`**: Comprehensive evaluation script
- **`config.yaml`**: Centralized configuration
- **`REPORT.md`**: Detailed project report

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or `sample_size` in `config.yaml`
2. **Slow Training**: Reduce `sample_size` or `epochs`
3. **WandB not syncing**: Run `uv run wandb sync wandb/run-*` manually
4. **Import Errors**: Ensure you're using `uv run python` or activate the virtual environment

### Performance Tips

- **CPU Training**: Use smaller `sample_size` (e.g., 5000-10000 per class), `batch_size: 32`
- **GPU Training**: Increase `batch_size` to 64 or 128, use larger `sample_size` (100K+)
- **Faster Iteration**: Reduce `epochs` for testing
- **Transfer Learning**: Faster convergence - fewer epochs needed (5-8 vs 10+)
- **Memory Issues**: Reduce `batch_size` if GPU runs out of memory

## 📚 Dependencies

All dependencies are listed in `pyproject.toml`:

- `torch`, `torchvision`, `torchaudio`: PyTorch ecosystem
- `timm`: Pre-trained models for transfer learning
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization
- `scikit-learn`: Metrics
- `pillow`: Image processing
- `wandb`: Experiment tracking
- `tqdm`: Progress bars
- `pyyaml`: Configuration management

## 📄 License

This project is part of an educational assignment.

## 👤 Author

Created as part of the Deep Learning with PyTorch workshop assignment.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- Weights & Biases for experiment tracking tools
- Dataset providers for the QR code images

---

## 🎯 Current Results

**Transfer Learning (ResNet18) Performance:**
- ✅ **Validation Accuracy**: 78.34%
- ✅ **Training Time**: 25.15 minutes (10 epochs, 200K images, GPU)
- ✅ **Model**: ResNet18 (11.2M parameters, 42.7 MB)
- ✅ **Dataset**: 200,000 images (100K per class)

**To get complete metrics**, run:
```bash
uv run python test.py
```

This will output:
- Test set accuracy, precision, recall, F1-score
- Confusion matrix
- Efficiency metrics (FLOPS, memory usage, inference time)

---

**Note**: This project supports both custom CNN and transfer learning. Transfer learning (ResNet18) is currently configured and showing excellent results. For production use, consider ensemble methods or hybrid approaches combining visual and URL features.

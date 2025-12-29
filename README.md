# QR Code Phishing Detection using Deep Learning

A PyTorch-based deep learning project for detecting phishing attempts in QR codes using Convolutional Neural Networks (CNN).

## 📋 Project Overview

This project implements a binary classification system to distinguish between **benign** and **malicious** QR codes. The model uses a custom CNN architecture trained on a large dataset of QR code images.

### Key Features

- **Custom CNN Architecture**: 3 convolutional layers + 2 fully connected layers
- **Data Augmentation**: Random flips, rotations, and color jittering
- **Experiment Tracking**: Integrated with Weights & Biases (wandb)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Efficiency Metrics**: Inference time and throughput measurements
- **Modular Design**: Separate modules for dataset, model, training, and evaluation

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- `uv` package manager (for dependency management)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/zuss/pytorch/quishing-with-ml
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up Weights & Biases (Optional but recommended):**
   ```bash
   uv run wandb login
   ```
   Or add your API key to `config.yaml` under `logging.wandb_api_key`.

### Dataset Structure

The project expects the following directory structure:

```
data/
├── QR_All_benign/
│   └── QR_All_benign/
│       └── qrs/          # Benign QR code images
└── QR_All_benign/
    └── QR_All_Malicious/
        └── qrs/          # Malicious QR code images
```

**Note:** The dataset paths can be configured in `config.yaml`.

## 📖 Usage

### 1. Configuration

Edit `config.yaml` to adjust hyperparameters:

```yaml
data:
  sample_size: 20000      # Images per class (None = use all)
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.001
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
├── model.py                 # CNN model architecture
├── train.py                 # Training script
├── test.py                  # Evaluation script
├── notebook.ipynb           # Jupyter notebook (main deliverable)
├── README.md                # This file
├── REPORT.md                # Project report
├── pyproject.toml           # Dependencies
└── models/                  # Saved models (created after training)
    └── best_model.pth
```

## 🏗️ Model Architecture

The CNN architecture consists of:

- **Convolutional Layers**: 3 blocks (32 → 64 → 128 channels)
- **Pooling**: MaxPool2d after each conv block
- **Fully Connected**: 512 → 256 → 2 (output classes)
- **Dropout**: 0.5 for regularization
- **Activation**: ReLU (except output layer)

**Total Parameters**: ~2.5M  
**Model Size**: ~10 MB

## 📊 Evaluation Metrics

The model is evaluated using:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed per-class performance
- **Inference Time**: Average time per sample
- **Throughput**: Samples processed per second

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

- `num_classes`: Number of output classes (2 for binary)
- `dropout`: Dropout probability

### Logging Configuration

- `use_wandb`: Enable/disable Weights & Biases tracking
- `wandb_project`: WandB project name
- `wandb_entity`: WandB entity/username
- `wandb_api_key`: WandB API key (optional, can use `wandb login`)

## 📈 Weights & Biases Integration

The project integrates with Weights & Biases for experiment tracking:

1. **Login** (if not using API key in config):
   ```bash
   uv run wandb login
   ```

2. **Enable in config**:
   ```yaml
   logging:
     use_wandb: true
     wandb_project: "qr-phishing-detection"
   ```

3. **View results**: Training metrics are automatically logged to your WandB dashboard.

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

- **CPU Training**: Use smaller `sample_size` (e.g., 5000-10000 per class)
- **GPU Training**: Increase `batch_size` to 64 or 128
- **Faster Iteration**: Reduce `epochs` for testing

## 📚 Dependencies

All dependencies are listed in `pyproject.toml`:

- `torch`, `torchvision`, `torchaudio`: PyTorch ecosystem
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

**Note**: This project is designed for educational purposes. For production use, consider additional improvements such as transfer learning, ensemble methods, or hybrid approaches combining visual and URL features.


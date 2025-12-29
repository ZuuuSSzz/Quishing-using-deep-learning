"""
CNN Model Architecture for QR Code Phishing Detection
Binary Classification: Benign (0) vs Malicious (1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QRCodeCNN(nn.Module):
    """
    Convolutional Neural Network for QR Code classification.
    
    Architecture:
    - 3 Convolutional blocks (Conv2d + ReLU + MaxPool)
    - 2 Fully connected layers with dropout
    - Output: 2 classes (binary classification)
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of output classes (default: 2 for binary)
            dropout: Dropout probability (default: 0.5)
        """
        super(QRCodeCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Convolutional layers
        # Input: (batch_size, 3, 224, 224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        # Input: 224x224
        # After conv1 + pool: 112x112
        # After conv2 + pool: 56x56
        # After conv3 + pool: 28x28
        # So: 128 * 28 * 28 = 100352
        self.fc1_input_size = 128 * 28 * 28
        
        # Fully connected layers (balanced size)
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional block 1
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 112, 112)
        
        # Convolutional block 2
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 56, 56)
        
        # Convolutional block 3
        x = self.pool(F.relu(self.conv3(x)))  # (batch, 128, 28, 28)
        
        # Flatten
        x = x.view(-1, self.fc1_input_size)  # (batch, 100352)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_layer(x)
        
        # Output layer (no activation, will use CrossEntropyLoss which includes Softmax)
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self):
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        """Get the model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
        return total_size


def create_model(num_classes: int = 2, dropout: float = 0.5, device: str = 'cpu'):
    """
    Create and initialize the model.
    
    Args:
        num_classes: Number of output classes
        dropout: Dropout probability
        device: Device to place model on ('cpu' or 'cuda')
        
    Returns:
        Initialized model
    """
    model = QRCodeCNN(num_classes=num_classes, dropout=dropout)
    model = model.to(device)
    
    # Initialize weights (Xavier/He initialization with better scaling)
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Use smaller initialization for final layer to prevent bias
            if m is model.fc3:
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            else:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    return model


def create_optimizer(model, learning_rate: float = 0.001, weight_decay: float = 0.01):
    """
    Create optimizer (AdamW) for the model.
    
    Args:
        model: The model to optimize
        learning_rate: Learning rate (default: 0.001)
        weight_decay: Weight decay for regularization (default: 0.01)
        
    Returns:
        AdamW optimizer
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return optimizer


def create_scheduler(optimizer, mode: str = 'reduce_on_plateau', patience: int = 5, factor: float = 0.5, **kwargs):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        mode: Scheduler type ('reduce_on_plateau' or 'step')
        patience: Patience for ReduceLROnPlateau (default: 5)
        factor: Factor to reduce LR (default: 0.5)
        **kwargs: Additional arguments for scheduler
        
    Returns:
        Learning rate scheduler
    """
    if mode == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            **kwargs
        )
    elif mode == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.1,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")
    
    return scheduler


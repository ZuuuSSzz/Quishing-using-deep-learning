"""
Training script for QR Code Phishing Detection model.
Implements full training loop with validation monitoring.
"""

import os
import time
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils import create_data_splits, create_dataloaders
from model import create_model, create_optimizer, create_scheduler

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip=None):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        gradient_clip: Gradient clipping value (None to disable)
        
    Returns:
        Average training loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train(
    config_path: str = "config.yaml",
    save_plots: bool = True
):
    """
    Main training function.
    
    Args:
        config_path: Path to config file
        save_plots: Whether to save loss/accuracy plots
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)
    
    # Create data splits
    print("Loading data...")
    train_dataset, val_dataset, test_dataset = create_data_splits(
        benign_dir=config['data']['benign_dir'],
        malicious_dir=config['data']['malicious_dir'],
        sample_size=config['data']['sample_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        image_size=config['data']['image_size'],
        seed=config['data']['seed']
    )
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        device=device,
        model_type=config['model'].get('model_type', 'cnn'),
        model_name=config['model'].get('model_name', 'resnet18')
    )
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler with configurable patience
    scheduler = create_scheduler(
        optimizer,
        mode=config['training']['scheduler'],
        patience=config['training'].get('scheduler_patience', 5),
        factor=config['training'].get('scheduler_factor', 0.5)
    )
    
    # Training setup
    num_epochs = config['training']['epochs']
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    # Create save directory
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / config['model']['save_path']
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Initialize wandb if enabled
    use_wandb = config['logging'].get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        # Get wandb configuration (prioritize environment variables over config file)
        # This allows switching accounts without modifying code/config
        api_key = os.environ.get('WANDB_API_KEY') or config['logging'].get('wandb_api_key')
        entity = os.environ.get('WANDB_ENTITY') or config['logging'].get('wandb_entity')
        project = os.environ.get('WANDB_PROJECT') or config['logging'].get('wandb_project', 'qr-phishing-detection')
        
        # Login to wandb (if API key provided, otherwise uses existing login from ~/.netrc)
        if api_key:
            wandb.login(key=api_key)
        
        # Display which account/entity is being used
        if entity:
            print(f"Using wandb entity: {entity}")
        else:
            # Try to get current logged-in user
            try:
                api = wandb.Api()
                current_user = api.viewer.username if hasattr(api.viewer, 'username') else 'default'
                print(f"Using wandb account: {current_user} (default entity)")
            except:
                print("Using wandb default account")
        
        wandb.init(
            project=project,
            entity=entity,
            config={
                'batch_size': config['training']['batch_size'],
                'epochs': num_epochs,
                'learning_rate': config['training']['learning_rate'],
                'weight_decay': config['training']['weight_decay'],
                'optimizer': config['training'].get('optimizer', 'adamw'),
                'scheduler': config['training']['scheduler'],
                'sample_size': config['data']['sample_size'],
                'image_size': config['data']['image_size'],
                'dropout': config['model']['dropout'],
                'model_parameters': model.count_parameters(),
                'model_size_mb': model.get_model_size_mb()
            }
        )
        wandb.watch(model, log='all', log_freq=10)
        print("✓ Weights & Biases initialized")
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    if use_wandb:
        print(f"Wandb tracking: Enabled (project: {config['logging'].get('wandb_project', 'qr-phishing-detection')})")
    print("=" * 60)
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip=config['training'].get('gradient_clip', None)
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate (for ReduceLROnPlateau)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': current_lr
            })
        
        # Save best model
        if config['training']['save_best']:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'history': history
                }, model_path)
                print(f"✓ Saved best model (Val Loss: {val_loss:.4f})")
                
                # Log best model to wandb
                if use_wandb:
                    wandb.log({'best_val_loss': best_val_loss, 'best_val_acc': best_val_acc})
    
    total_time = time.time() - start_time
    
    # Log final metrics to wandb
    if use_wandb:
        wandb.log({
            'total_training_time_minutes': total_time / 60,
            'final_best_val_loss': best_val_loss,
            'final_best_val_acc': best_val_acc
        })
        wandb.finish()
        print("✓ Wandb run completed")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_path}")
    
    # Plot training history
    if save_plots:
        plot_training_history(history, save_dir)
        # Optionally log plot to wandb
        if use_wandb:
            try:
                wandb.log({"training_history": wandb.Image(str(save_dir / 'training_history.png'))})
            except:
                pass  # If wandb already finished, skip
    
    return model, history


def plot_training_history(history, save_dir):
    """
    Plot and save training history (loss and accuracy curves).
    
    Args:
        history: Dictionary with training history
        save_dir: Directory to save plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = save_dir / 'training_history.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training plots saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train QR Code Phishing Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip saving plots')
    
    args = parser.parse_args()
    
    train(config_path=args.config, save_plots=not args.no_plots)


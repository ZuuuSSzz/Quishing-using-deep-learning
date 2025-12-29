"""
Evaluation script for QR Code Phishing Detection model.
Computes metrics and efficiency measurements on test set.
"""

import os
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from data_utils import create_data_splits, create_dataloaders
from model import create_model


def evaluate_model(
    model,
    test_loader,
    device,
    criterion=None
):
    """
    Evaluate model on test set and compute metrics.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Device to run on
        criterion: Loss function (optional)
        
    Returns:
        Dictionary with metrics and predictions
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss if criterion provided
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Average loss
    avg_loss = running_loss / len(test_loader) if criterion else None
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'loss': avg_loss
    }
    
    return results


def measure_inference_time(model, test_loader, device, num_batches=10):
    """
    Measure inference time per sample and per batch.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Device to run on
        num_batches: Number of batches to measure
        
    Returns:
        Dictionary with timing metrics
    """
    model.eval()
    
    batch_times = []
    sample_times = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            
            images = images.to(device)
            batch_size = images.size(0)
            
            # Warmup
            if i == 0:
                _ = model(images)
            
            # Measure time
            start_time = time.time()
            _ = model(images)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            batch_time = end_time - start_time
            sample_time = batch_time / batch_size
            
            batch_times.append(batch_time)
            sample_times.append(sample_time)
    
    return {
        'avg_batch_time_ms': np.mean(batch_times) * 1000,
        'std_batch_time_ms': np.std(batch_times) * 1000,
        'avg_sample_time_ms': np.mean(sample_times) * 1000,
        'std_sample_time_ms': np.std(sample_times) * 1000,
        'samples_per_second': 1.0 / np.mean(sample_times)
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def print_metrics(results, class_names):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        results: Dictionary with metrics
        class_names: List of class names
    """
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1-Score:  {results['f1']:.4f} ({results['f1']*100:.2f}%)")
    
    if results['loss']:
        print(f"  Loss:      {results['loss']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {results['precision_per_class'][i]:.4f}")
        print(f"    Recall:    {results['recall_per_class'][i]:.4f}")
        print(f"    F1-Score:  {results['f1_per_class'][i]:.4f}")
    
    print("\n" + "=" * 60)


def evaluate(
    config_path: str = "config.yaml",
    model_path: str = None,
    save_plots: bool = True
):
    """
    Main evaluation function.
    
    Args:
        config_path: Path to config file
        model_path: Path to saved model (if None, uses config)
        save_plots: Whether to save confusion matrix plot
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)
    
    # Determine model path
    if model_path is None:
        save_dir = Path(config['training']['save_dir'])
        model_path = save_dir / config['model']['save_path']
    
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first using train.py"
        )
    
    # Create data splits (same as training)
    print("Loading test data...")
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
    
    # Create test DataLoader only
    _, _, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config['evaluation']['inference_batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Create model
    print("\nLoading model...")
    model = create_model(
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        device=device
    )
    
    # Load saved model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from: {model_path}")
    print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    if 'val_acc' in checkpoint:
        print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Model statistics
    num_params = model.count_parameters()
    model_size_mb = model.get_model_size_mb()
    
    print(f"\nModel Statistics:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Size: {model_size_mb:.2f} MB")
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    results = evaluate_model(model, test_loader, device, criterion)
    
    # Print metrics
    class_names = ['Benign', 'Malicious']
    print_metrics(results, class_names)
    
    # Measure inference time
    if config['evaluation']['measure_inference_time']:
        print("\n" + "=" * 60)
        print("EFFICIENCY METRICS")
        print("=" * 60)
        
        print("\nMeasuring inference time...")
        timing_results = measure_inference_time(
            model, test_loader, device, num_batches=20
        )
        
        print(f"\nInference Performance:")
        print(f"  Avg batch time: {timing_results['avg_batch_time_ms']:.2f} ms "
              f"(±{timing_results['std_batch_time_ms']:.2f} ms)")
        print(f"  Avg sample time: {timing_results['avg_sample_time_ms']:.2f} ms "
              f"(±{timing_results['std_sample_time_ms']:.2f} ms)")
        print(f"  Throughput: {timing_results['samples_per_second']:.2f} samples/sec")
        
        results['inference_time'] = timing_results
    
    # Save confusion matrix
    if save_plots:
        save_dir = Path(config['training']['save_dir'])
        save_dir.mkdir(exist_ok=True)
        cm_path = save_dir / 'confusion_matrix.png'
        plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)
    
    # Print classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        results['labels'],
        results['predictions'],
        target_names=class_names,
        digits=4
    ))
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Test F1-Score: {results['f1']*100:.2f}%")
    print(f"Model Parameters: {num_params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    if 'inference_time' in results:
        print(f"Inference Time: {results['inference_time']['avg_sample_time_ms']:.2f} ms/sample")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate QR Code Phishing Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint (default: from config)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip saving plots')
    
    args = parser.parse_args()
    
    evaluate(config_path=args.config, model_path=args.model, save_plots=not args.no_plots)


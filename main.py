"""
Main entry point for QR Code Phishing Detection project.

This script provides a simple interface to run training and evaluation.
"""

import argparse
import sys
from pathlib import Path

from train import train
from test import evaluate


def main():
    """Main function to run training or evaluation."""
    parser = argparse.ArgumentParser(
        description='QR Code Phishing Detection - Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run training
  python main.py --train

  # Run evaluation
  python main.py --eval

  # Run both training and evaluation
  python main.py --train --eval

  # Specify custom config
  python main.py --train --config custom_config.yaml

  # Evaluate with specific model
  python main.py --eval --model models/custom_model.pth
        """
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run training'
    )
    
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Run evaluation'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model checkpoint for evaluation (default: from config)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip saving plots'
    )
    
    args = parser.parse_args()
    
    # Check if at least one action is specified
    if not args.train and not args.eval:
        parser.print_help()
        print("\nError: Please specify --train, --eval, or both.")
        sys.exit(1)
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run training
    if args.train:
        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        try:
            model, history = train(
                config_path=args.config,
                save_plots=not args.no_plots
            )
            print("\n✓ Training completed successfully!")
        except Exception as e:
            print(f"\n✗ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Run evaluation
    if args.eval:
        print("\n" + "=" * 60)
        print("STARTING EVALUATION")
        print("=" * 60)
        try:
            results = evaluate(
                config_path=args.config,
                model_path=args.model,
                save_plots=not args.no_plots
            )
            print("\n✓ Evaluation completed successfully!")
        except Exception as e:
            print(f"\n✗ Evaluation failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("ALL OPERATIONS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import logging
import json
from typing import Dict
# Set project root directory relative to this script file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Add project root to Python path to allow importing modules
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# Now import project modules
try:
    # Ensure imports happen after path modification
    from data_processor import process_data
    from train import train
    from predict import predict
except ImportError as e:
     # Provide more context on import error
     print(f"Error: Failed to import project modules.")
     print(f"PROJECT_ROOT={PROJECT_ROOT}")
     print(f"sys.path={sys.path}")
     print(f"Error details: {e}")
     sys.exit(1)


# Setup basic logging for the main script orchestrator
# Use a more detailed format for the main orchestrator
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s [%(module)s:%(funcName)s:%(lineno)d] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Log to stdout by default
logger = logging.getLogger(__name__) # Get logger for this module


def load_config(config_path: str) -> Dict:
    """Loads YAML configuration file."""
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        logger.debug(f"Config content: {json.dumps(config, indent=2)}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
         logger.error(f"An unexpected error occurred loading config {config_path}: {e}", exc_info=True)
         sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='ESM-Flex Temperature-Aware: Protein Flexibility (RMSF) Prediction Pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(
        dest='command',
        help='Select the command: process, train, or predict.',
        required=True
    )

    # === Process data command ===
    process_parser = subparsers.add_parser(
        'process',
        help='Process aggregated RMSF/Temperature CSV data into standardized splits.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    process_parser.add_argument('--config', type=str, default='config.yaml', help='Path to the main YAML configuration file.')
    # Allow overriding config file paths via CLI for flexibility
    process_parser.add_argument('--csv', type=str, default=None, help='Override path to the input aggregated CSV file.')
    process_parser.add_argument('--output', type=str, default=None, help='Override output directory for processed data.')
    process_parser.add_argument('--train_ratio', type=float, default=None, help='Override fraction for training set topology split.')
    process_parser.add_argument('--val_ratio', type=float, default=None, help='Override fraction for validation set topology split.')
    process_parser.add_argument('--seed', type=int, default=None, help='Override random seed for splitting.')

    # === Train command ===
    train_parser = subparsers.add_parser(
        'train',
        help='Train the Temperature-Aware ESM Regression model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')

    # === Predict command ===
    predict_parser = subparsers.add_parser(
        'predict',
        help='Predict RMSF for sequences at a specific temperature using a trained model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Prediction doesn't use the main config file directly, takes specific inputs
    predict_parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pt file).')
    predict_parser.add_argument('--fasta_path', type=str, required=True, help='Path to the input FASTA file.')
    # #!#!# REFACTOR NOTE: Added required temperature argument
    predict_parser.add_argument('--temperature', type=float, required=True, help='Target temperature (in Kelvin) for prediction.')
    predict_parser.add_argument('--output_dir', type=str, default='predictions', help='Base directory to save prediction results.')
    predict_parser.add_argument('--batch_size', type=int, default=8, help='Batch size for prediction.')
    predict_parser.add_argument('--max_length', type=int, default=None, help='Optional: Max sequence length filter.')
    predict_parser.add_argument('--plot_predictions', action=argparse.BooleanOptionalAction, default=True, help='Generate plots.')
    predict_parser.add_argument('--smoothing_window', type=int, default=1, help='Smoothing window for plots (1=none).')


    # Parse arguments
    args = parser.parse_args()
    logger.info(f"Executing command: {args.command}")

    # === Execute Command ===
    if args.command == 'process':
        logger.info(f"Loading config for 'process' command from: {args.config}")
        config = load_config(args.config)

        # Override config values if provided via CLI
        csv_path = args.csv if args.csv is not None else config.get('data', {}).get('raw_csv_path')
        output_dir = args.output if args.output is not None else config.get('data', {}).get('data_dir', 'data/processed')
        train_ratio = args.train_ratio if args.train_ratio is not None else config.get('training', {}).get('train_ratio', 0.7)
        val_ratio = args.val_ratio if args.val_ratio is not None else config.get('training', {}).get('val_ratio', 0.15)
        seed = args.seed if args.seed is not None else config.get('training', {}).get('seed', 42)
        scaling_file = config.get('data', {}).get('temp_scaling_filename', 'temp_scaling_params.json')

        if not csv_path:
             logger.error("Raw CSV data path ('data.raw_csv_path') not found in config or provided via --csv.")
             sys.exit(1)

        logger.info(f"Starting data processing...")
        process_data(
            csv_path=csv_path,
            output_dir=output_dir,
            temp_scaling_filename=scaling_file, #!#!# Pass filename
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed
        )
        logger.info("Data processing finished.")

    elif args.command == 'train':
        logger.info(f"Loading config for 'train' command from: {args.config}")
        config = load_config(args.config)
        logger.info("Starting model training...")
        try:
            train(config) # Pass the loaded config dictionary
            logger.info("Training finished.")
        except Exception as e:
             # Catch potential errors during the train function execution
             logger.error(f"An unexpected error occurred during training execution: {e}", exc_info=True)
             sys.exit(1)

    elif args.command == 'predict':
        logger.info(f"Starting prediction...")
        # Prepare the configuration dictionary directly from args for the predict function
        predict_config = {
            'model_checkpoint': args.model_checkpoint,
            'fasta_path': args.fasta_path,
            'temperature': args.temperature, #!#!# Pass temperature
            'output_dir': args.output_dir,
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'plot_predictions': args.plot_predictions,
            'smoothing_window': args.smoothing_window
        }
        try:
            predict(predict_config)
            logger.info("Prediction finished.")
        except Exception as e:
             logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
             sys.exit(1)

    else:
        # Should be unreachable due to 'required=True'
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

import os
import torch
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import logging
from pathlib import Path
import time
import json #!#!# For loading scaling params
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict

# #!#!# REFACTOR NOTE: Import the renamed model
from model import TemperatureAwareESMRegressionModel
from dataset import load_sequences_from_fasta # Keep for loading input fasta
from train import log_gpu_memory, get_temperature_scaler #!#!# Import scaler function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# #!#!# REFACTOR NOTE: Updated loading function for the new model class
def load_model_for_prediction(checkpoint_path: str, device: torch.device) -> Tuple[Optional[TemperatureAwareESMRegressionModel], Optional[Dict]]:
    """Load a trained TemperatureAwareESMRegressionModel from checkpoint."""
    logger.info(f"Loading model checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return None, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
         logger.error(f"Failed to load checkpoint file {checkpoint_path}: {e}", exc_info=True)
         return None, None

    required_keys = ['config', 'model_state_dict', 'epoch']
    if not all(key in checkpoint for key in required_keys):
         missing = [k for k in required_keys if k not in checkpoint]
         logger.error(f"Checkpoint {checkpoint_path} is missing required keys: {missing}. Found: {list(checkpoint.keys())}")
         return None, None

    config_from_ckpt = checkpoint['config']
    logger.info("Config loaded from checkpoint.")
    logger.debug(f"Checkpoint Config: {json.dumps(config_from_ckpt, indent=2)}")


    try:
        logger.info(f"Recreating model architecture:")
        model_config = config_from_ckpt.get('model', {})
        esm_version = model_config.get('esm_version', 'esmc_600m') # Default if missing
        regr_config = model_config.get('regression', {})
        hidden_dim = regr_config.get('hidden_dim', 64) # Default if missing
        dropout = regr_config.get('dropout', 0.1) # Default if missing

        logger.info(f"  ESM Version: {esm_version}")
        logger.info(f"  Regression Hidden Dim: {hidden_dim}")
        logger.info(f"  Regression Dropout: {dropout}")

        # #!#!# REFACTOR NOTE: Instantiate the correct model class
        model = TemperatureAwareESMRegressionModel(
            esm_model_name=esm_version,
            regression_hidden_dim=hidden_dim,
            regression_dropout=dropout
        )
        logger.info("Model instance created.")
    except KeyError as e:
         logger.error(f"Missing expected key in checkpoint config structure: {e}")
         return None, None
    except Exception as e:
         logger.error(f"Error creating model instance from config: {e}", exc_info=True)
         return None, None

    # --- Load model state dictionary ---
    try:
        # Allow flexibility for minor architecture changes if needed, but log warnings
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
             # Filter out potential frozen ESM base model keys if they appear missing (shouldn't happen if frozen)
             trainable_head_keys = {name for name, param in model.regression_head.named_parameters()}
             head_missing = [k for k in missing_keys if any(k.startswith(f"regression_head.{n}") for n in trainable_head_keys)]
             if head_missing:
                  logger.warning(f"State dict potentially missing keys in regression head: {head_missing}")
             else:
                   logger.info(f"State dict missing non-trainable (likely ESM base) keys: {missing_keys}") # Less critical
        if unexpected_keys:
             logger.warning(f"State dict has unexpected keys (may indicate architecture mismatch): {unexpected_keys}")

        logger.info(f"Model weights loaded successfully.")
    except Exception as e:
         logger.error(f"Error loading state_dict into model: {e}", exc_info=True)
         return None, None

    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded to {device} and set to eval mode.")
    logger.info(f"  Trained for {checkpoint['epoch']+1} epochs.")
    if 'val_corr' in checkpoint:
        logger.info(f"  Best Validation Corr at save time: {checkpoint.get('val_corr', 'N/A'):.6f}")

    return model, config_from_ckpt

def group_sequences_by_length(sequences: Dict[str, str], batch_size: int, bucket_size: int = 50) -> List[List[Tuple[str, str]]]:
    """Groups sequences by length into batches for efficient prediction."""
    if not sequences: return []
    length_buckets = defaultdict(list)
    for seq_id, seq in sequences.items():
        bucket_idx = len(seq) // bucket_size
        length_buckets[bucket_idx].append((seq_id, seq))

    all_batches = []
    # Process shortest first generally helps memory management
    for bucket_idx in sorted(length_buckets.keys()):
        bucket_items = length_buckets[bucket_idx]
        for i in range(0, len(bucket_items), batch_size):
            batch = bucket_items[i : i + batch_size]
            all_batches.append(batch)
    logger.info(f"Grouped {len(sequences)} sequences into {len(all_batches)} batches for prediction.")
    return all_batches

# #!#!# REFACTOR NOTE: Updated predict_rmsf to take single target temp and scaler
def predict_rmsf_at_temperature(
    model: TemperatureAwareESMRegressionModel,
    sequences: Dict[str, str],
    target_temperature: float,
    temp_scaler: Callable[[float], float], # Expects the scaling function
    batch_size: int,
    device: torch.device,
    use_amp: bool = True
) -> Dict[str, np.ndarray]:
    """
    Predict RMSF values for sequences at a specific target temperature.

    Args:
        model: The trained TemperatureAwareESMRegressionModel.
        sequences: Dictionary mapping sequence IDs to sequences.
        target_temperature: The single temperature (raw, unscaled) to predict at.
        temp_scaler: The function to scale the raw target temperature.
        batch_size: Batch size for inference.
        device: Device ('cuda' or 'cpu').
        use_amp: Whether to use Automatic Mixed Precision (GPU only).

    Returns:
        Dictionary mapping sequence IDs to predicted RMSF values (NumPy array).
    """
    model.eval()
    if not sequences: return {}

    # #!#!# REFACTOR NOTE: Scale the single target temperature ONCE
    scaled_target_temp = temp_scaler(target_temperature)
    logger.info(f"Predicting for raw temperature {target_temperature:.1f}K (scaled: {scaled_target_temp:.4f})")

    # Prepare batches based on length
    batches = group_sequences_by_length(sequences, batch_size)
    results = {}
    prediction_start_time = time.time()
    autocast_device_type = device.type
    amp_enabled = (device.type == 'cuda' and use_amp)

    with torch.no_grad():
        for batch_data in tqdm(batches, desc=f"Predicting @ {target_temperature:.0f}K", leave=False):
            batch_ids = [item[0] for item in batch_data]
            batch_seqs = [item[1] for item in batch_data]
            current_batch_size = len(batch_ids)

            # #!#!# REFACTOR NOTE: Create tensor of the *same* scaled temperature for the whole batch
            scaled_temps_batch = torch.tensor([scaled_target_temp] * current_batch_size,
                                              device=device, dtype=torch.float32)

            try:
                # Forward pass with optional AMP, using model.predict method
                with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                    # #!#!# REFACTOR NOTE: Pass sequences and the repeating scaled temp tensor
                    batch_predictions_np = model.predict(batch_seqs, scaled_temps_batch)

                # Store results (already numpy arrays from model.predict)
                if len(batch_predictions_np) == len(batch_ids):
                    for seq_id, pred_np in zip(batch_ids, batch_predictions_np):
                        results[seq_id] = pred_np
                else:
                    logger.error(f"Prediction output length mismatch: {len(batch_predictions_np)} preds vs {len(batch_ids)} IDs.")
                    # Handle partial assignment or error as needed

            except Exception as e:
                 logger.error(f"Error predicting batch starting with {batch_ids[0]}: {e}", exc_info=True)
                 # Add placeholder or skip IDs in this batch
                 for seq_id in batch_ids: results[seq_id] = np.array([]) # Example: empty array on error

            # Optional: Periodic GPU cache clearing
            if device.type == 'cuda' and len(results) % (10 * batch_size) == 0:
                 torch.cuda.empty_cache()

    prediction_duration = time.time() - prediction_start_time
    num_predicted = len(results)
    logger.info(f"Prediction completed for {num_predicted} sequences in {prediction_duration:.2f}s.")
    if num_predicted > 0: logger.info(f"Avg time per sequence: {prediction_duration / num_predicted:.4f}s")

    return results

# #!#!# REFACTOR NOTE: Added temperature to plot title
def plot_rmsf(
    sequence: str,
    predictions: np.ndarray,
    title: str, # Should include domain_id and temperature
    output_path: str,
    window_size: int = 1,
    figsize: Tuple[int, int] = (15, 6)
):
    # """Plot predicted RMSF values against residue position."""
    # if predictions is None or len(predictions) == 0:
    #     logger.warning(f"No prediction data to plot for '{title}'. Skipping plot.")
    #     return

    # plt.style.use('seaborn-v0_8-whitegrid')
    # fig, ax = plt.subplots(figsize=figsize)

    # pred_len = len(predictions)
    # residue_indices = np.arange(1, pred_len + 1)

    # if window_size > 1:
    #     s = pd.Series(predictions)
    #     plot_data = s.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    #     plot_label = f'RMSF Prediction (Smoothed, win={window_size})'
    # else:
    #     plot_data = predictions
    #     plot_label = 'RMSF Prediction'

    # ax.plot(residue_indices, plot_data, '-', color='dodgerblue', linewidth=1.5, label=plot_label)

    # ax.set_xlabel('Residue Position')
    # ax.set_ylabel('Predicted RMSF')
    # ax.set_title(f'Predicted RMSF for {title} (Length: {pred_len})') # Title now includes Temp
    # ax.set_xlim(0, pred_len + 1)
    # ax.grid(True, linestyle=':', alpha=0.7)

    # # Add stats text box
    # mean_rmsf = np.mean(predictions)
    # median_rmsf = np.median(predictions)
    # stats_text = (f'Mean: {mean_rmsf:.3f}\n'
    #               f'Median: {median_rmsf:.3f}')
    # ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
    #         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.5))

    # ax.legend(loc='upper right')
    # plt.tight_layout()

    # try:
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     plt.savefig(output_path, dpi=100, bbox_inches='tight') # Lower DPI for potentially many plots
    # except Exception as e:
    #     logger.error(f"Failed to save plot to {output_path}: {e}")
    # finally:
    #     plt.close(fig)
    return

# #!#!# REFACTOR NOTE: Added temperature to output CSV filename and title
def save_predictions(predictions: Dict[str, np.ndarray], output_path: str, target_temperature: float):
    """Save predictions to a CSV file."""
    if not predictions: return
    data_to_save = []
    for domain_id, rmsf_values in predictions.items():
        if rmsf_values is None or len(rmsf_values) == 0: continue
        for i, rmsf in enumerate(rmsf_values):
            data_to_save.append({
                'domain_id': domain_id,
                'resid': i + 1,
                'rmsf_pred': rmsf,
                'predicted_at_temp': target_temperature #!#!# Add temp info
            })
    if not data_to_save:
        logger.warning("No valid prediction data points found to save in CSV.")
        return

    try:
        df = pd.DataFrame(data_to_save)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, float_format='%.6f')
        logger.info(f"Predictions for T={target_temperature:.0f}K saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions DataFrame to {output_path}: {e}")

# #!#!# REFACTOR NOTE: Major changes for temperature handling
def predict(config: Dict[str, Any]):
    """Main prediction function."""
    predict_start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda': log_gpu_memory()

    # --- Get Required Config ---
    model_checkpoint = config.get('model_checkpoint')
    fasta_path = config.get('fasta_path')
    output_dir = config.get('output_dir', 'predictions')
    # #!#!# REFACTOR NOTE: Get target temperature from config (passed from main.py)
    target_temperature = config.get('temperature')

    if not model_checkpoint or not fasta_path or target_temperature is None:
        logger.critical("Missing required config: 'model_checkpoint', 'fasta_path', or 'temperature'.")
        return

    # --- Output Dir & Logging ---
    # Include temperature in output subdir for organization
    temp_str = f"{target_temperature:.0f}K"
    output_dir_temp = os.path.join(output_dir, temp_str)
    os.makedirs(output_dir_temp, exist_ok=True)

    log_path = os.path.join(output_dir_temp, 'prediction.log')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler) # Add handler for this run
    logger.info(f"--- Starting Prediction Run for T={target_temperature:.1f}K ---")
    logger.info(f"Prediction Config: {json.dumps(config, indent=2)}")
    logger.info(f"Saving results to: {output_dir_temp}")


    # --- Load Model ---
    model, model_config_from_ckpt = load_model_for_prediction(model_checkpoint, device)
    if model is None: return
    if device.type == 'cuda': log_gpu_memory()

    # --- Load Temperature Scaler ---
    # #!#!# REFACTOR NOTE: Assume scaler params are in the same dir as the checkpoint OR specified elsewhere
    # Default: Look next to checkpoint
    checkpoint_dir = os.path.dirname(model_checkpoint)
    scaling_filename = model_config_from_ckpt.get('data', {}).get('temp_scaling_filename', 'temp_scaling_params.json')
    temp_scaling_path = os.path.join(checkpoint_dir, scaling_filename)
    # Alternative: Check model_dir from config if path above doesn't exist
    if not os.path.exists(temp_scaling_path):
         logger.warning(f"Scaling file not found next to checkpoint ({temp_scaling_path}).")
         # Try path relative to model_dir specified in config (might be different from checkpoint dir)
         config_model_dir = model_config_from_ckpt.get('output',{}).get('model_dir')
         if config_model_dir:
              alt_path = os.path.join(config_model_dir, scaling_filename)
              if os.path.exists(alt_path):
                   temp_scaling_path = alt_path
                   logger.info(f"Found scaling file in config model_dir: {temp_scaling_path}")
              else:
                   logger.error(f"Scaling file also not found in config model_dir: {alt_path}")
                   logger.error("Cannot proceed without temperature scaling parameters.")
                   return
         else:
              logger.error("Cannot find scaling file and no model_dir specified in checkpoint config.")
              return

    try:
        temp_scaler = get_temperature_scaler(temp_scaling_path)
    except Exception:
         logger.error("Failed to load temperature scaler. Aborting prediction.")
         return

    # --- Load Sequences ---
    try:
        sequences = load_sequences_from_fasta(fasta_path)
        if not sequences: raise ValueError("No sequences found in FASTA file.")
        logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    except Exception as e:
         logger.critical(f"Error loading sequences from {fasta_path}: {e}", exc_info=True)
         return

    # --- Filter Sequences by Max Length (Optional) ---
    max_length = config.get('max_length')
    if max_length is not None and max_length > 0:
        original_count = len(sequences)
        sequences = {sid: seq for sid, seq in sequences.items() if len(seq) <= max_length}
        filtered_count = len(sequences)
        if filtered_count < original_count:
            logger.info(f"Filtered out {original_count - filtered_count} sequences longer than {max_length}.")
        if not sequences:
            logger.critical(f"No sequences remaining after filtering. Aborting.")
            return

    # --- Predict RMSF ---
    predictions = predict_rmsf_at_temperature(
        model,
        sequences,
        target_temperature, # Pass raw temperature
        temp_scaler,        # Pass the loaded scaler function
        config.get('batch_size', 8),
        device,
        use_amp=(device.type == 'cuda') # Use AMP on GPU by default
    )

    # --- Save & Plot Results ---
    if predictions:
         output_csv_path = os.path.join(output_dir_temp, f'predictions_{temp_str}.csv')
         save_predictions(predictions, output_csv_path, target_temperature)

         if config.get('plot_predictions', True):
             plots_dir = os.path.join(output_dir_temp, 'plots')
             os.makedirs(plots_dir, exist_ok=True)
             smoothing = config.get('smoothing_window', 1)
             logger.info(f"Generating plots (smoothing={smoothing})...")
             plot_count = 0
             for domain_id, pred_array in tqdm(predictions.items(), desc="Plotting", leave=False):
                  if domain_id in sequences and pred_array.size > 0 :
                      try:
                          # Add temperature to plot title
                          plot_title = f"{domain_id} @ {target_temperature:.0f}K"
                          plot_rmsf(
                              sequence=sequences[domain_id],
                              predictions=pred_array,
                              title=plot_title, #!#!# Updated title
                              output_path=os.path.join(plots_dir, f'{domain_id}_{temp_str}.png'),
                              window_size=smoothing
                          )
                          plot_count += 1
                      except Exception as e:
                          logger.error(f"Failed to generate plot for {domain_id}: {e}")
                  elif pred_array.size == 0:
                       logger.debug(f"Skipping plot for {domain_id} - no prediction data.")
                  else:
                       logger.warning(f"Cannot plot for {domain_id}: Original sequence not found.")
             logger.info(f"Generated {plot_count} plots.")
    else:
         logger.warning("Prediction resulted in no output.")

    # --- Finalize ---
    predict_end_time = time.time()
    logger.info(f"--- Prediction Run Finished (T={target_temperature:.1f}K) ---")
    logger.info(f"Total prediction time: {predict_end_time - predict_start_time:.2f} seconds.")
    logger.info(f"Results saved in: {output_dir_temp}")
    # Remove the file handler specific to this run
    logger.removeHandler(file_handler)
    file_handler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict RMSF using a trained Temperature-Aware ESM Regression model')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pt file)')
    parser.add_argument('--fasta_path', type=str, required=True, help='Path to the input FASTA file')
    # #!#!# REFACTOR NOTE: Added required temperature argument
    parser.add_argument('--temperature', type=float, required=True, help='Target temperature (in Kelvin) for prediction')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Base directory to save prediction results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for prediction')
    parser.add_argument('--max_length', type=int, default=None, help='Optional: Max sequence length filter')
    parser.add_argument('--plot_predictions', action=argparse.BooleanOptionalAction, default=True, help='Generate plots')
    parser.add_argument('--smoothing_window', type=int, default=1, help='Smoothing window for plots (1=none)')
    # Optional: Explicit path for scaling params if not stored with model
    # parser.add_argument('--temp_scaling_path', type=str, default=None, help='Path to temp_scaling_params.json (if not found near checkpoint)')

    args = parser.parse_args()
    config_dict = vars(args)
    predict(config_dict)


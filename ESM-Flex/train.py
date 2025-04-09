import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import logging
import time
import json #!#!# For loading scaling params
from typing import Dict, Any, Optional, Callable

# #!#!# REFACTOR NOTE: Import the renamed model
from model import TemperatureAwareESMRegressionModel
from dataset import create_length_batched_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Optional: Force deterministic algorithms
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")

def log_gpu_memory(detail=False):
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.debug(f"GPU Memory: Allocated={allocated:.1f} MB, Reserved={reserved:.1f} MB")
        if detail:
            logger.info(torch.cuda.memory_summary())

# #!#!# REFACTOR NOTE: Added temperature scaling function
def get_temperature_scaler(params_path: str) -> Callable[[float], float]:
    """Loads scaling parameters and returns a scaling function."""
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        temp_min = params['temp_min']
        temp_max = params['temp_max']
        logger.info(f"Loaded temperature scaling parameters from {params_path}: Min={temp_min}, Max={temp_max}")

        # Handle case where min and max are the same (avoid division by zero)
        temp_range = temp_max - temp_min
        if abs(temp_range) < 1e-6:
            logger.warning("Temperature min and max are identical in scaling parameters. Scaling will return 0.5.")
            # Return a function that outputs a constant (e.g., 0.5 for midpoint)
            return lambda t: 0.5
        else:
            # Standard Min-Max scaling function
            # Adding epsilon to range for safety, though checked above
            return lambda t: (float(t) - temp_min) / (temp_range + 1e-8)

    except FileNotFoundError:
        logger.error(f"Temperature scaling file not found: {params_path}")
        raise
    except KeyError as e:
        logger.error(f"Missing key {e} in temperature scaling file: {params_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading or parsing temperature scaling file {params_path}: {e}")
        raise

# #!#!# REFACTOR NOTE: Added temp_scaler argument
def train_epoch(model, dataloader, optimizer, device, temp_scaler, accumulation_steps=1, max_gradient_norm=1.0):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_corr = 0.0
    num_samples_processed = 0
    num_residues_processed = 0

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    autocast_device_type = device.type

    optimizer.zero_grad(set_to_none=True)

    batch_iterator = tqdm(dataloader, desc="Training", leave=False)
    for i, batch in enumerate(batch_iterator):
        sequences = batch['sequences']
        # #!#!# REFACTOR NOTE: Get raw temperatures from batch
        raw_temperatures = batch['temperatures'] # This is a tensor [batch_size]
        # Targets are a list of tensors [seq_len]
        target_rmsf_values = batch['rmsf_values']

        current_batch_size = len(sequences)
        if current_batch_size == 0: continue

        try:
            # #!#!# REFACTOR NOTE: Scale temperatures before passing to model
            # Apply scaler (which operates on floats) and convert back to tensor
            # Requires moving tensor to CPU for scaling function, then back to device.
            scaled_temps_list = [temp_scaler(t.item()) for t in raw_temperatures]
            scaled_temps_tensor = torch.tensor(scaled_temps_list, device=device, dtype=torch.float32)

            # Forward pass with AMP
            with torch.amp.autocast(device_type=autocast_device_type, enabled=(scaler is not None)):
                # #!#!# REFACTOR NOTE: Pass scaled temperatures to model
                outputs = model(sequences=sequences,
                                temperatures=scaled_temps_tensor,
                                target_rmsf_values=target_rmsf_values)
                loss = outputs['loss']

            # Basic loss check
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Batch {i}: Invalid loss ({loss}). Skipping gradient update.")
                # Clear gradients manually if we skip optimizer step
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                     optimizer.zero_grad(set_to_none=True)
                continue

            loss_value = loss.item() # Store for logging before scaling

            # Normalize loss for accumulation BEFORE backward pass
            loss = loss / accumulation_steps

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient Accumulation & Optimizer Step
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                if scaler is not None:
                    if max_gradient_norm > 0:
                        scaler.unscale_(optimizer) # Unscale before clipping
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in model.parameters() if p.requires_grad),
                            max_gradient_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else: # No AMP
                    if max_gradient_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in model.parameters() if p.requires_grad),
                            max_gradient_norm
                        )
                    optimizer.step()

                # IMPORTANT: Zero gradients AFTER optimizer step and gradient clipping
                optimizer.zero_grad(set_to_none=True)


            # Update cumulative metrics using the original (unscaled) loss value
            # Weight loss/corr by number of residues for a more stable average if lengths vary significantly
            batch_residues = sum(len(p) for p in outputs['predictions'] if p.numel() > 0)
            if batch_residues > 0:
                total_loss += loss_value * batch_residues # Use original loss here
                correlation = outputs['metrics'].get('pearson_correlation', 0.0)
                # Ensure correlation is float and not nan
                if isinstance(correlation, torch.Tensor): correlation = correlation.item()
                if not np.isnan(correlation):
                    total_corr += correlation * batch_residues
                num_samples_processed += current_batch_size
                num_residues_processed += batch_residues

            # Update progress bar (using residue-weighted averages)
            avg_loss = total_loss / num_residues_processed if num_residues_processed > 0 else 0.0
            avg_corr = total_corr / num_residues_processed if num_residues_processed > 0 else 0.0
            batch_iterator.set_postfix(loss=f"{avg_loss:.4f}", corr=f"{avg_corr:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")


        except Exception as e:
             logger.error(f"Error during training batch {i}: {e}", exc_info=True)
             logger.warning("Skipping batch due to error. Attempting to clear gradients.")
             optimizer.zero_grad(set_to_none=True) # Attempt to reset state
             if device.type == 'cuda': torch.cuda.empty_cache()
             continue

        # Optional: Periodic memory logging
        # if i > 0 and i % 100 == 0 and device.type == 'cuda': log_gpu_memory()


    # Calculate final epoch averages (residue-weighted)
    final_avg_loss = total_loss / num_residues_processed if num_residues_processed > 0 else 0.0
    final_avg_corr = total_corr / num_residues_processed if num_residues_processed > 0 else 0.0

    logger.info(f"Processed {num_samples_processed} samples ({num_residues_processed} residues) in training epoch.")
    return final_avg_loss, final_avg_corr


# #!#!# REFACTOR NOTE: Added temp_scaler argument
def validate(model, dataloader, device, temp_scaler):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_corr = 0.0
    num_samples_processed = 0
    num_residues_processed = 0
    domain_correlations = {} # Per-domain/sample correlation

    autocast_device_type = device.type

    batch_iterator = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in batch_iterator:
            sequences = batch['sequences']
            domain_ids = batch['domain_ids']
            # #!#!# REFACTOR NOTE: Get raw temperatures from batch
            raw_temperatures = batch['temperatures']
            target_rmsf_values = batch['rmsf_values']

            current_batch_size = len(sequences)
            if current_batch_size == 0: continue

            try:
                # #!#!# REFACTOR NOTE: Scale temperatures
                scaled_temps_list = [temp_scaler(t.item()) for t in raw_temperatures]
                scaled_temps_tensor = torch.tensor(scaled_temps_list, device=device, dtype=torch.float32)

                # Forward pass with AMP
                with torch.amp.autocast(device_type=autocast_device_type, enabled=(device.type == 'cuda')):
                     # #!#!# REFACTOR NOTE: Pass scaled temperatures to model
                     outputs = model(sequences=sequences,
                                     temperatures=scaled_temps_tensor,
                                     target_rmsf_values=target_rmsf_values)
                     loss = outputs['loss'] # Batch average loss

                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Validation: Invalid loss ({loss}) for batch starting with {domain_ids[0]}. Skipping.")
                    continue

                loss_value = loss.item()

                # Calculate batch residues for weighting metrics
                predictions_list = outputs['predictions'] # List of tensors
                batch_residues = sum(len(p) for p in predictions_list if p.numel() > 0)

                if batch_residues > 0:
                    total_loss += loss_value * batch_residues
                    # Use batch correlation if available (already averaged over samples in batch)
                    batch_avg_corr = outputs['metrics'].get('pearson_correlation', 0.0)
                    if isinstance(batch_avg_corr, torch.Tensor): batch_avg_corr = batch_avg_corr.item()
                    if not np.isnan(batch_avg_corr):
                         total_corr += batch_avg_corr * batch_residues

                    num_samples_processed += current_batch_size
                    num_residues_processed += batch_residues

                    # Store per-domain/sample correlation (calculated within model forward now)
                    # Note: This requires the model's forward pass metric calculation to be per-sample before averaging.
                    # If model's 'pearson_correlation' is already batch-averaged, this part needs adjustment
                    # Let's assume the model's forward calculates it correctly for now.
                    # Rerun correlation calc here if model doesn't provide per-sample:
                    for i, domain_id in enumerate(domain_ids):
                         if i < len(predictions_list) and predictions_list[i].numel() > 1:
                              pred_tensor = predictions_list[i]
                              true_tensor = target_rmsf_values[i].to(device)
                              min_len = min(len(pred_tensor), len(true_tensor))
                              if min_len > 1:
                                   corr_val = TemperatureAwareESMRegressionModel.safe_pearson_correlation(
                                        pred_tensor[:min_len], true_tensor[:min_len]
                                   ).item()
                                   domain_correlations[domain_id] = corr_val
                              else: domain_correlations[domain_id] = 0.0


                # Update progress bar (residue-weighted averages)
                avg_loss = total_loss / num_residues_processed if num_residues_processed > 0 else 0.0
                avg_corr = total_corr / num_residues_processed if num_residues_processed > 0 else 0.0
                batch_iterator.set_postfix(loss=f"{avg_loss:.4f}", corr=f"{avg_corr:.4f}")

            except Exception as e:
                logger.error(f"Error during validation batch starting with {domain_ids[0]}: {e}", exc_info=True)
                continue

    # Log detailed correlation statistics from this epoch
    if domain_correlations:
         correlations = np.array(list(domain_correlations.values()))
         correlations = correlations[~np.isnan(correlations)] # Clean NaNs
         if len(correlations) > 0:
              logger.info(f"Per-Sample Validation Correlation stats (n={len(correlations)}): "
                          f"Mean={np.mean(correlations):.4f}, Median={np.median(correlations):.4f}, "
                          f"Std={np.std(correlations):.4f}, Min={np.min(correlations):.4f}, Max={np.max(correlations):.4f}")
         else: logger.warning("No valid per-sample correlations calculated during validation.")
    else: logger.warning("No per-sample correlations were calculated during validation.")

    # Calculate final epoch averages (residue-weighted)
    final_avg_loss = total_loss / num_residues_processed if num_residues_processed > 0 else 0.0
    final_avg_corr = total_corr / num_residues_processed if num_residues_processed > 0 else 0.0

    logger.info(f"Processed {num_samples_processed} samples ({num_residues_processed} residues) in validation epoch.")
    return final_avg_loss, final_avg_corr


def save_model(model, optimizer, epoch, val_loss, val_corr, config, save_path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_corr': val_corr,
        'config': config # Include config used for this run
    }
    try:
        torch.save(checkpoint, save_path)
        logger.info(f"Model checkpoint saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint to {save_path}: {e}")

def plot_metrics(train_losses, val_losses, train_corrs, val_corrs, save_dir, lr_values=None):
    """Plot training and validation metrics."""
    epochs = range(1, len(train_losses) + 1)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Loss Plot
    axes[0].plot(epochs, train_losses, 'o-', color='royalblue', label='Train Loss', markersize=4)
    axes[0].plot(epochs, val_losses, 's-', color='orangered', label='Validation Loss', markersize=4)
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        axes[0].axvline(best_epoch, linestyle='--', color='gray', alpha=0.7, label=f'Best Val Loss Epoch ({best_epoch})')
        axes[0].legend() # Update legend

    # Correlation Plot
    axes[1].plot(epochs, train_corrs, 'o-', color='royalblue', label='Train Correlation', markersize=4)
    axes[1].plot(epochs, val_corrs, 's-', color='orangered', label='Validation Correlation', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('Training and Validation Correlation')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    if val_corrs:
        best_epoch = np.argmax(val_corrs) + 1
        axes[1].axvline(best_epoch, linestyle='--', color='gray', alpha=0.7, label=f'Best Val Corr Epoch ({best_epoch})')

    # Optional Learning Rate Plot on Correlation Axis
    if lr_values:
        ax_lr = axes[1].twinx()
        ax_lr.plot(epochs, lr_values, 'd--', color='green', label='Learning Rate', markersize=3, alpha=0.6)
        ax_lr.set_ylabel('Learning Rate', color='green')
        ax_lr.tick_params(axis='y', labelcolor='green')
        if len(set(lr_values)) > 2: ax_lr.set_yscale('log')
        # Combine legends
        lines, labels = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax_lr.get_legend_handles_labels()
        axes[1].legend(lines + lines2, labels + labels2, loc='lower left')
    else:
         axes[1].legend(loc='lower left')


    plt.tight_layout(pad=2.0)
    plot_path = os.path.join(save_dir, 'training_metrics.png')
    try:
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Metrics plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Error saving metrics plot to {plot_path}: {e}")
    plt.close(fig)

def train(config: Dict[str, Any]):
    """Main training function."""
    start_time_train = time.time()

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda': log_gpu_memory()

    seed = config['training']['seed']
    set_seed(seed)

    # Directories
    model_save_dir = config['output']['model_dir']
    data_dir = config['data']['data_dir'] # Processed data lives here
    log_dir = os.path.join(model_save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True) # Ensure model dir exists

    # File logging
    log_path = os.path.join(log_dir, 'training.log')
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"--- Starting Temperature-Aware Training Run (Seed: {seed}) ---")
    logger.info(f"Full Configuration: {json.dumps(config, indent=2)}")


    # --- Load Temperature Scaler ---
    # #!#!# REFACTOR NOTE: Load scaling params from data_dir where process_data saved them
    temp_scaling_path = os.path.join(data_dir, config['data']['temp_scaling_filename'])
    try:
        temp_scaler = get_temperature_scaler(temp_scaling_path)
    except Exception:
         logger.error("Failed to load temperature scaler. Aborting training.")
         return


    # --- Data Loaders ---
    logger.info("Creating data loaders...")
    try:
        train_dataloader = create_length_batched_dataloader(
            data_dir, 'train', config['training']['batch_size'],
            shuffle=True, max_length=config['training'].get('max_length'),
            length_bucket_size=config['training'].get('length_bucket_size', 50)
        )
        val_dataloader = create_length_batched_dataloader(
            data_dir, 'val', config['training']['batch_size'],
            shuffle=False, max_length=config['training'].get('max_length'),
            length_bucket_size=config['training'].get('length_bucket_size', 50)
        )
        if not train_dataloader or not val_dataloader:
            raise RuntimeError("Failed to create one or both dataloaders.")
        logger.info(f"Train samples: {len(train_dataloader.dataset)}, Val samples: {len(val_dataloader.dataset)}")
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}", exc_info=True)
        return

    # --- Model ---
    logger.info("Creating model...")
    try:
        # #!#!# REFACTOR NOTE: Use the new TemperatureAware class
        model = TemperatureAwareESMRegressionModel(
            esm_model_name=config['model']['esm_version'],
            regression_hidden_dim=config['model']['regression']['hidden_dim'],
            regression_dropout=config['model']['regression']['dropout']
        )
        model = model.to(device)
        if device.type == 'cuda': log_gpu_memory()
    except Exception as e:
        logger.error(f"Error creating model: {e}", exc_info=True)
        return

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
         logger.error("Model has no trainable parameters! Check model initialization and freezing logic.")
         return
    logger.info(f"Optimizing {len(trainable_params)} parameter tensors in the regression head.")

    optimizer = optim.AdamW(
        trainable_params,
        lr=float(config['training']['learning_rate']),
        eps=float(config['training'].get('adam_epsilon', 1e-8)),
        weight_decay=float(config['training']['weight_decay'])
    )
    logger.info(f"Optimizer: AdamW (LR={config['training']['learning_rate']:.2e}, WD={config['training']['weight_decay']})")

    # --- Scheduler ---
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', # Maximize validation correlation
        factor=0.5, # Reduce LR by half
        patience=config['training']['scheduler_patience'],
        verbose=True,
        threshold=0.001 # Minimum improvement to consider significant
    )
    logger.info(f"Scheduler: ReduceLROnPlateau on Val Correlation (Patience={config['training']['scheduler_patience']})")

    # --- Training Loop ---
    logger.info("--- Starting Training Loop ---")
    best_val_corr = -float('inf')
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    train_losses, val_losses, train_corrs, val_corrs, lr_values = [], [], [], [], []

    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f"--- Epoch {epoch+1}/{num_epochs} ---")

        # Train
        train_loss, train_corr = train_epoch(
            model, train_dataloader, optimizer, device, temp_scaler, # Pass scaler
            config['training']['accumulation_steps'],
            config['training'].get('max_gradient_norm', 1.0)
        )
        train_losses.append(train_loss)
        train_corrs.append(train_corr)

        # Validate
        val_loss, val_corr = validate(model, val_dataloader, device, temp_scaler) # Pass scaler
        val_losses.append(val_loss)
        val_corrs.append(val_corr)

        current_lr = optimizer.param_groups[0]['lr']
        lr_values.append(current_lr)

        epoch_duration = time.time() - epoch_start_time

        logger.info(f"Epoch {epoch+1} Summary (Duration: {epoch_duration:.1f}s):")
        logger.info(f"  Train Loss: {train_loss:.6f} | Train Corr: {train_corr:.6f}")
        logger.info(f"  Val Loss:   {val_loss:.6f} | Val Corr:   {val_corr:.6f} {'*' if val_corr > best_val_corr else ''}")
        logger.info(f"  Learning Rate: {current_lr:.3e}")


        # Checkpointing & Early Stopping based on Validation Correlation
        scheduler.step(val_corr)

        is_best = val_corr > best_val_corr
        if is_best:
            improvement = val_corr - best_val_corr
            logger.info(f"  New best validation correlation! Improvement: +{improvement:.6f}")
            best_val_corr = val_corr
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            save_model(model, optimizer, epoch, val_loss, val_corr, config, os.path.join(model_save_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            logger.info(f"  Validation correlation did not improve. Patience: {patience_counter}/{config['training']['early_stopping_patience']}")

        # Save latest model state
        save_model(model, optimizer, epoch, val_loss, val_corr, config, os.path.join(model_save_dir, 'latest_model.pt'))

        # Save periodic checkpoint if enabled
        checkpoint_interval = config['training'].get('checkpoint_interval', 0)
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            save_model(model, optimizer, epoch, val_loss, val_corr, config, os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        # Update metrics plot
        plot_metrics(train_losses, val_losses, train_corrs, val_corrs, model_save_dir, lr_values)

        # Check for early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch+1} epochs due to lack of improvement in validation correlation.")
            break

        if device.type == 'cuda': torch.cuda.empty_cache()

    # --- End of Training ---
    total_training_time = time.time() - start_time_train
    logger.info("--- Training Finished ---")
    logger.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.1f} minutes)")
    logger.info(f"Best validation correlation: {best_val_corr:.6f} at epoch {best_epoch}")
    logger.info(f"Corresponding validation loss: {best_val_loss:.6f}")
    logger.info(f"Best model saved to: {os.path.join(model_save_dir, 'best_model.pt')}")
    logger.info(f"Training logs and plots saved in: {model_save_dir}")


import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import LogitsConfig, ESMProtein
except ImportError:
    logger.error("Failed to import from 'esm' library. Please install `pip install esm`.", exc_info=True)
    raise

class TemperatureAwareESMRegressionModel(nn.Module):
    """
    ESM-C based model for Temperature-Aware RMSF prediction.

    Uses ESM-C embeddings and incorporates a temperature feature
    into the regression head.
    """
    def __init__(self,
                 esm_model_name: str = "esmc_600m",
                 regression_hidden_dim: int = 64, # Increased default
                 regression_dropout: float = 0.1):
        super().__init__()

        logger.info(f"Initializing TemperatureAwareESMRegressionModel...")
        logger.info(f"Loading base ESM-C Model: {esm_model_name}")
        try:
            # Load the base ESMC model object
            self.esm_model = ESMC.from_pretrained(esm_model_name)
        except Exception as e:
            logger.error(f"Failed to load ESM-C model '{esm_model_name}'. Error: {e}")
            raise

        self.esm_model.eval() # Set base model to evaluation mode
        self.esm_model_name = esm_model_name

        # --- Freeze ESM-C parameters ---
        logger.info("Freezing ESM-C model parameters...")
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # --- Detect embedding dimension ---
        # Do a dummy forward pass on CPU first to avoid moving large model prematurely
        temp_cpu_model = ESMC.from_pretrained(esm_model_name)
        embedding_dim = -1
        try:
            with torch.no_grad():
                test_protein = ESMProtein(sequence="A") # Minimal sequence
                encoded = temp_cpu_model.encode(test_protein)
                logits_output = temp_cpu_model.logits(
                    encoded, LogitsConfig(sequence=True, return_embeddings=True)
                )
                embedding_dim = logits_output.embeddings.size(-1)
                logger.info(f"Detected ESM embedding dimension: {embedding_dim}")
        except Exception as e:
            logger.error(f"Error during embedding dimension detection: {e}")
            raise ValueError(f"Could not determine embedding dimension for {esm_model_name}.")
        finally:
            del temp_cpu_model # Clean up temporary model

        if embedding_dim <= 0:
             raise ValueError("Failed to detect a valid embedding dimension.")

        self.esm_hidden_dim = embedding_dim

        # --- Create Temperature-Aware Regression Head ---
        # #!#!# REFACTOR NOTE: Input dimension includes ESM embedding + 1 for temperature
        regression_input_dim = self.esm_hidden_dim + 1

        logger.info(f"Creating regression head. Input dimension: {regression_input_dim} (ESM: {self.esm_hidden_dim} + Temp: 1)")

        if regression_hidden_dim > 0:
            self.regression_head = nn.Sequential(
                nn.LayerNorm(regression_input_dim), # Normalize combined input
                nn.Linear(regression_input_dim, regression_hidden_dim),
                nn.GELU(),
                nn.Dropout(regression_dropout),
                nn.Linear(regression_hidden_dim, 1) # Output single RMSF value
            )
            logger.info(f"Using MLP regression head (LayerNorm -> Linear({regression_input_dim},{regression_hidden_dim}) -> GELU -> Dropout -> Linear({regression_hidden_dim},1))")
        else: # Direct linear layer after LayerNorm
            self.regression_head = nn.Sequential(
                nn.LayerNorm(regression_input_dim),
                nn.Dropout(regression_dropout), # Apply dropout even in linear case
                nn.Linear(regression_input_dim, 1)
            )
            logger.info(f"Using Linear regression head (LayerNorm -> Dropout -> Linear({regression_input_dim},1))")

        self._log_parameter_counts()
        logger.info("TemperatureAwareESMRegressionModel initialized successfully.")

    def _log_parameter_counts(self):
        total_params = sum(p.numel() for p in self.parameters())
        esm_params = sum(p.numel() for p in self.esm_model.parameters())
        trainable_params = sum(p.numel() for p in self.regression_head.parameters())

        logger.info(f"Parameter Counts:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  ESM-C parameters (frozen): {esm_params:,}")
        logger.info(f"  Trainable regression head parameters: {trainable_params:,}")
        if total_params > 0:
            logger.info(f"  Trainable percentage: {trainable_params/total_params:.4%}")

    # #!#!# REFACTOR NOTE: Added 'temperatures' argument and processing logic
    def forward(self,
                sequences: List[str],
                temperatures: torch.Tensor, # Expecting a BATCHED tensor of SCALED temperatures
                target_rmsf_values: Optional[List[torch.Tensor]] = None # Renamed for clarity
                ) -> Dict[str, Any]:
        """
        Forward pass incorporating sequence embeddings and temperature.

        Args:
            sequences: List of amino acid sequence strings (batch_size).
            temperatures: Tensor of SCALED temperature values for each sequence,
                          shape [batch_size]. MUST be pre-scaled.
            target_rmsf_values: Optional list of target RMSF tensors for loss calculation.

        Returns:
            Dictionary containing 'predictions', 'loss', 'metrics'.
        """
        # --- Basic Input Checks ---
        if len(sequences) != len(temperatures):
             msg = f"Batch size mismatch: {len(sequences)} sequences vs {len(temperatures)} temperatures."
             logger.error(msg)
             raise ValueError(msg)
        if target_rmsf_values is not None and len(sequences) != len(target_rmsf_values):
             msg = f"Batch size mismatch: {len(sequences)} sequences vs {len(target_rmsf_values)} target RMSF values."
             logger.error(msg)
             raise ValueError(msg)

        # --- Setup Device ---
        # Infer device from regression head parameters (guaranteed to exist)
        device = next(self.regression_head.parameters()).device
        # Ensure ESM base model is on the correct device
        # Note: Moving large models can be slow, ideally done once outside the loop if possible.
        # However, checking here ensures correctness if device changes.
        if next(self.esm_model.parameters()).device != device:
             self.esm_model.to(device)

        # --- Prepare ESMProtein objects ---
        proteins = []
        original_indices = [] # Store original batch index for each valid protein
        skipped_indices = []

        for i, seq_str in enumerate(sequences):
            if not seq_str or len(seq_str) == 0:
                 logger.debug(f"Skipping empty sequence at original batch index {i}.")
                 skipped_indices.append(i)
                 continue
            try:
                 proteins.append(ESMProtein(sequence=seq_str))
                 original_indices.append(i)
            except Exception as e_prot:
                 logger.warning(f"Could not create ESMProtein for sequence at index {i}. Error: {e_prot}. Skipping.")
                 skipped_indices.append(i)

        if not proteins:
            logger.warning("No valid sequences found in the batch to process.")
            # Return structure consistent with successful run but empty preds/zero loss
            return {
                'predictions': [torch.tensor([], device=device) for _ in sequences], # Match input batch size
                'loss': torch.tensor(0.0, device=device, requires_grad=True if self.training else False),
                'metrics': {'pearson_correlation': 0.0}
            }

        # --- Process Proteins Individually (Batched processing with ESMProtein API is tricky) ---
        all_predictions = [] # Store final per-residue predictions for each protein
        processed_indices_map = {} # Map index in `all_predictions` back to original batch index

        try:
            for protein_idx, protein in enumerate(proteins):
                original_batch_idx = original_indices[protein_idx]
                current_temp = temperatures[original_batch_idx] # Get the SCALED temperature for this protein

                try:
                    # 1. Get ESM Embeddings (No Gradients for ESM part)
                    with torch.no_grad():
                        encoded_protein = self.esm_model.encode(protein)
                        logits_output = self.esm_model.logits(
                            encoded_protein,
                            LogitsConfig(sequence=True, return_embeddings=True)
                        )

                    if logits_output.embeddings is None:
                        logger.warning(f"No embeddings returned for protein {original_batch_idx}. Skipping.")
                        continue

                    # Embeddings shape: [1, seq_len_with_tokens, hidden_dim]
                    embeddings = logits_output.embeddings.to(device)
                    # Remove batch dimension: [seq_len_with_tokens, hidden_dim]
                    embeddings_tokens = embeddings.squeeze(0)

                    # 2. #!#!# REFACTOR NOTE: Prepare Temperature Feature
                    # Temperature is already scaled and on the correct device (from input tensor)
                    # Expand temperature to match the sequence length dimension of embeddings
                    seq_len_tokens = embeddings_tokens.size(0)
                    temp_feature = current_temp.view(1, 1).expand(seq_len_tokens, 1)
                    # Result shape: [seq_len_with_tokens, 1]

                    # 3. #!#!# REFACTOR NOTE: Concatenate Embeddings and Temperature
                    # Concatenate along the feature dimension (dim=1)
                    combined_features = torch.cat((embeddings_tokens, temp_feature), dim=1)
                    # Result shape: [seq_len_with_tokens, hidden_dim + 1]

                    # 4. Pass through Regression Head (Gradients ARE required here)
                    # Ensure head is in correct mode (train/eval) based on model state
                    self.regression_head.train(self.training)
                    # Get per-token predictions
                    token_predictions = self.regression_head(combined_features).squeeze(-1)
                    # Result shape: [seq_len_with_tokens]

                    # 5. Extract Residue-Level Predictions (Remove BOS/EOS)
                    original_seq_len = len(protein.sequence) # Length of the actual AA sequence
                    expected_tokens = original_seq_len + 2 # Assuming BOS and EOS tokens

                    if len(token_predictions) >= expected_tokens:
                         # Slice: Start after BOS (index 1), end before EOS (index expected_tokens - 1)
                         residue_predictions = token_predictions[1:expected_tokens-1]
                         # Final check - ensure sliced length matches original sequence length
                         if len(residue_predictions) != original_seq_len:
                              logger.warning(f"Length mismatch AFTER slicing for seq {original_batch_idx}. "
                                             f"Expected {original_seq_len}, got {len(residue_predictions)}. "
                                             f"Using this prediction, but check data/tokenization.")
                         # Store the valid residue predictions
                         all_predictions.append(residue_predictions)
                         processed_indices_map[len(all_predictions)-1] = original_batch_idx

                    else: # Token prediction tensor too short
                         logger.warning(f"Prediction tensor length ({len(token_predictions)}) is shorter than "
                                      f"expected seq+BOS+EOS ({expected_tokens}) for original sequence {original_batch_idx}. "
                                      "Cannot reliably slice BOS/EOS. Skipping this sequence.")

                except Exception as e_inner:
                    logger.error(f"Error processing protein at original batch index {original_batch_idx}: {e_inner}", exc_info=True)
                    # Continue to the next protein in the batch

        except Exception as e_outer:
            logger.error(f"Error during main forward loop: {e_outer}", exc_info=True)
            # Return empty/zero structure if outer loop fails catastrophically
            return {
                'predictions': [torch.tensor([], device=device) for _ in sequences],
                'loss': torch.tensor(0.0, device=device, requires_grad=True if self.training else False),
                'metrics': {'pearson_correlation': 0.0}
            }

        # --- Loss Calculation (Optional) ---
        loss = None
        metrics = {'pearson_correlation': 0.0} # Default metrics

        if target_rmsf_values is not None:
            valid_losses = []
            valid_correlations = []
            num_valid_pairs = 0

            # Iterate through the predictions we successfully generated
            for pred_idx, prediction_tensor in enumerate(all_predictions):
                original_batch_idx = processed_indices_map[pred_idx]
                target_tensor = target_rmsf_values[original_batch_idx].to(device)

                # Align lengths (prediction might be slightly off if slicing warning occurred)
                min_len = min(len(prediction_tensor), len(target_tensor))
                if min_len <= 1: # Need at least 2 points for correlation
                    continue

                pred_aligned = prediction_tensor[:min_len]
                target_aligned = target_tensor[:min_len]

                # Calculate MSE Loss
                mse = F.mse_loss(pred_aligned, target_aligned, reduction='mean')
                if not torch.isnan(mse) and not torch.isinf(mse):
                    valid_losses.append(mse)

                # Calculate Pearson Correlation safely
                pearson_corr = self.safe_pearson_correlation(pred_aligned, target_aligned)
                if not torch.isnan(pearson_corr):
                    valid_correlations.append(pearson_corr)

                num_valid_pairs += 1

            # Average loss and correlation over valid pairs in the batch
            if valid_losses:
                # Average the MSE loss across samples in the batch
                loss = torch.stack(valid_losses).mean()
                if torch.isnan(loss): # Handle potential NaN if all losses were somehow NaN
                    loss = torch.tensor(0.0, device=device, requires_grad=True if self.training else False)
            else: # No valid pairs, set loss to 0
                loss = torch.tensor(0.0, device=device, requires_grad=True if self.training else False)

            if valid_correlations:
                # Average correlation across samples
                metrics['pearson_correlation'] = torch.stack(valid_correlations).mean().item()
            else: # No valid correlations calculated
                metrics['pearson_correlation'] = 0.0 # Keep as float

        # Ensure loss is always a tensor, required by training loop
        if loss is None:
             loss = torch.tensor(0.0, device=device, requires_grad=True if self.training else False)


        # --- Reconstruct Output List ---
        # Create a list of tensors matching the original batch size, filling with
        # predictions where available and empty tensors otherwise.
        final_predictions_list = [torch.tensor([], device=device) for _ in sequences]
        for pred_idx, pred_tensor in enumerate(all_predictions):
             original_batch_idx = processed_indices_map[pred_idx]
             final_predictions_list[original_batch_idx] = pred_tensor

        return {'predictions': final_predictions_list, 'loss': loss, 'metrics': metrics}

    @staticmethod
    def safe_pearson_correlation(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Calculate Pearson correlation safely, returning 0 for std dev near zero or len < 2."""
        # Ensure float type
        x = x.float()
        y = y.float()

        # Check for conditions where correlation is undefined or unstable
        if len(x) < 2 or torch.std(x) < epsilon or torch.std(y) < epsilon:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        # Use matrix multiplication for covariance calculation for efficiency if needed,
        # but direct sum is fine for typical sequence lengths here.
        cov = torch.sum(vx * vy)
        sx = torch.sqrt(torch.sum(vx ** 2))
        sy = torch.sqrt(torch.sum(vy ** 2))
        denominator = sx * sy

        # Check for near-zero denominator
        if denominator < epsilon:
             return torch.tensor(0.0, device=x.device, dtype=torch.float32)

        corr = cov / denominator
        # Clamp to handle potential floating point inaccuracies near +/- 1
        corr = torch.clamp(corr, -1.0, 1.0)

        # Final NaN check (should be rare after previous checks, but just in case)
        if torch.isnan(corr):
            logger.warning("NaN detected during Pearson Correlation calculation despite checks. Returning 0.")
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)

        return corr


    # #!#!# REFACTOR NOTE: predict method now requires SCALED temperatures
    @torch.no_grad()
    def predict(self,
                sequences: List[str],
                scaled_temperatures: torch.Tensor # Expecting tensor shape [batch_size]
               ) -> List[np.ndarray]:
        """
        Predict RMSF values for sequences at given SCALED temperatures.

        Args:
            sequences: List of amino acid sequences.
            scaled_temperatures: Tensor of SCALED temperatures (one per sequence).

        Returns:
            List of NumPy arrays containing predicted RMSF values for each sequence.
        """
        self.eval() # Ensure evaluation mode

        if len(sequences) != len(scaled_temperatures):
             raise ValueError("Number of sequences must match number of temperatures for prediction.")

        # Pass sequences and scaled temperatures to the forward method
        # target_rmsf_values is None during prediction
        outputs = self.forward(sequences=sequences,
                               temperatures=scaled_temperatures.to(next(self.parameters()).device), # Ensure temp on correct device
                               target_rmsf_values=None)

        # Convert predictions tensor list to list of numpy arrays
        np_predictions = []
        for pred_tensor in outputs['predictions']:
            if pred_tensor is not None and pred_tensor.numel() > 0:
                 np_predictions.append(pred_tensor.cpu().numpy())
            else: # Handle cases where prediction failed for a sequence
                 np_predictions.append(np.array([], dtype=np.float32))

        return np_predictions

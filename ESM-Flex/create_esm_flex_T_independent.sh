#!/bin/bash

echo "Creating ESM-Flex Temperature-Aware Project Structure..."

# Create directories
mkdir -p data/raw data/processed models predictions utils

# === Create config.yaml ===
echo "Creating config.yaml..."
cat << 'EOF' > config.yaml
# Configuration for ESM-Flex Temperature-Aware Project

data:
  # Path to the directory containing processed data splits
  data_dir: data/processed
  # #!#!# REFACTOR NOTE: Update this path to your *NEW* aggregated CSV file!
  raw_csv_path: /path/to/your/aggregated_rmsf_data.csv
  # Path where temperature scaling parameters (min/max) will be saved/loaded from data_dir
  temp_scaling_filename: temp_scaling_params.json

model:
  # Identifier for the ESM-C model from the 'esm' library.
  esm_version: "esmc_600m"

  # Regression head configuration
  regression:
    # Hidden dimension for the MLP head. Set to 0 for a direct Linear layer.
    hidden_dim: 64 #!#!# REFACTOR NOTE: Slightly increased default head size due to added temperature input
    # Dropout rate for the regression head
    dropout: 0.1

training:
  # Number of training epochs
  num_epochs: 50 #!#!# REFACTOR NOTE: Increased default epochs for potentially more complex task
  # Batch size (adjust based on GPU memory and model size)
  batch_size: 4
  # Learning rate for the optimizer
  learning_rate: 1.0e-4
  # Weight decay for the optimizer (applied only to non-bias/norm params)
  weight_decay: 0.01
  # AdamW epsilon parameter
  adam_epsilon: 1.0e-8
  # Gradient accumulation steps (effective_batch_size = batch_size * accumulation_steps)
  accumulation_steps: 4
  # Max gradient norm for clipping (0 to disable)
  max_gradient_norm: 1.0
  # Learning rate scheduler patience (epochs) based on validation correlation
  scheduler_patience: 5
  # Early stopping patience (epochs) based on validation correlation
  early_stopping_patience: 10
  # Random seed for reproducibility
  seed: 42
  # Optional: Maximum sequence length to process (helps manage memory)
  # max_length: 1024
  # Size of length buckets for grouping similar-length sequences in dataloader
  length_bucket_size: 50
  # Frequency (in epochs) to save intermediate checkpoints
  checkpoint_interval: 5

output:
  # Directory to save trained models, logs, plots, and scaling parameters
  model_dir: models

# Prediction settings (used by predict.py if called via main.py)
prediction:
  batch_size: 8 # Can often be larger than training batch size
  plot_predictions: true
  smoothing_window: 1 # Window size for smoothing plots (1 = no smoothing)
  # Optional: max_length for prediction if different from training
  # max_length: 1024

EOF

# === Create data_processor.py ===
echo "Creating data_processor.py..."
cat << 'EOF' > data_processor.py
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import random
from typing import Dict, List, Tuple, Set, Optional
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard 1-letter amino acid codes
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # Include common Histidine variants if they weren't fixed by fix_data_.py
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H'
}

def load_data(csv_path: str) -> Optional[pd.DataFrame]:
    """Load data from the aggregated CSV file."""
    if not os.path.exists(csv_path):
        logger.error(f"Data file not found: {csv_path}")
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # #!#!# REFACTOR NOTE: Updated required columns for temp-aware model
        required_cols = ['domain_id', 'resid', 'resname', 'temperature_feature', 'target_rmsf']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"CSV missing required columns: {missing_cols}. Cannot proceed.")
            return None
        # Check for NaN in essential columns
        nan_check_cols = ['domain_id', 'resid', 'resname', 'temperature_feature', 'target_rmsf']
        nan_counts = df[nan_check_cols].isnull().sum()
        if nan_counts.sum() > 0:
             logger.warning(f"Found NaN values in essential columns:\n{nan_counts[nan_counts > 0]}")
             logger.warning("Attempting to drop rows with NaNs in these essential columns...")
             df.dropna(subset=nan_check_cols, inplace=True)
             logger.info(f"{len(df)} rows remaining after dropping NaNs.")
             if len(df) == 0:
                  logger.error("No valid rows remaining after dropping NaNs. Cannot proceed.")
                  return None

        # Convert temperature to numeric, coercing errors
        df['temperature_feature'] = pd.to_numeric(df['temperature_feature'], errors='coerce')
        df['target_rmsf'] = pd.to_numeric(df['target_rmsf'], errors='coerce')
        df.dropna(subset=['temperature_feature', 'target_rmsf'], inplace=True) # Drop if conversion failed
        logger.info(f"{len(df)} rows remaining after ensuring numeric temperature and RMSF.")

        if len(df) == 0:
             logger.error("No valid numeric temperature/RMSF rows found. Cannot proceed.")
             return None

        return df
    except Exception as e:
        logger.error(f"Error loading or performing initial validation on CSV file {csv_path}: {e}", exc_info=True)
        raise

def group_by_domain(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group data by domain_id."""
    domains = {}
    required_cols = ['domain_id', 'resid', 'temperature_feature']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"DataFrame missing one or more required columns for grouping: {required_cols}.")
        return domains

    # Group by domain_id first
    grouped = df.groupby('domain_id')
    skipped_inconsistent_temp = 0

    for domain_id, group in grouped:
        domain_id_str = str(domain_id)
        # #!#!# REFACTOR NOTE: Check for consistent temperature within the domain group
        unique_temps = group['temperature_feature'].unique()
        if len(unique_temps) > 1:
            logger.warning(f"Domain ID {domain_id_str} has multiple temperatures: {unique_temps}. Skipping this domain.")
            skipped_inconsistent_temp += 1
            continue
        # Ensure residues are sorted
        domains[domain_id_str] = group.sort_values('resid')

    logger.info(f"Grouped data into {len(domains)} unique domains with consistent temperature.")
    if skipped_inconsistent_temp > 0:
        logger.warning(f"Skipped {skipped_inconsistent_temp} domains due to inconsistent temperatures within the group.")
    return domains

def extract_sequence_rmsf_temp(domains: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Extract amino acid sequence, target RMSF values, and temperature for each domain."""
    processed_data = {}
    processed_count = 0
    skipped_residues = defaultdict(int)
    skipped_domains_missing_cols = set()
    skipped_domains_length_mismatch = set()
    skipped_domains_no_sequence = set()

    for domain_id, domain_df in domains.items():
        # #!#!# REFACTOR NOTE: Check for new required columns: target_rmsf and temperature_feature
        required_cols = ['resname', 'target_rmsf', 'temperature_feature']
        if not all(col in domain_df.columns for col in required_cols):
            logger.warning(f"Skipping domain {domain_id} due to missing columns ({required_cols}). Found: {domain_df.columns.tolist()}")
            skipped_domains_missing_cols.add(domain_id)
            continue

        sequence = ''
        rmsf_values = []
        temperatures = domain_df['temperature_feature'].tolist() # Should be consistent, take list to verify later
        valid_domain = True

        for _, row in domain_df.iterrows():
            residue = str(row['resname']).upper().strip()
            if residue in AA_MAP:
                sequence += AA_MAP[residue]
                rmsf_values.append(row['target_rmsf'])
            else:
                # Log unknown residues but continue processing the domain for now
                skipped_residues[residue] += 1
                # logger.warning(f"Unknown residue '{residue}' found in domain {domain_id}.")
                # Decide if this should invalidate the domain
                # valid_domain = False
                # break

        if not valid_domain:
             continue

        if sequence:
            # Final check: sequence length vs RMSF length
            if len(sequence) == len(rmsf_values):
                 # Verify temperature consistency again (should be guaranteed by grouping step)
                 temp_val = temperatures[0] # Get the single temperature
                 if not all(t == temp_val for t in temperatures):
                      logger.error(f"Inconsistency detected: Temperatures within domain {domain_id} are not uniform after grouping: {temperatures}. This should not happen. Skipping.")
                      continue

                 processed_data[domain_id] = {
                     'sequence': sequence,
                     'rmsf': np.array(rmsf_values, dtype=np.float32),
                     'temperature': float(temp_val) # Store the single temperature value
                 }
                 processed_count += 1
            else:
                logger.warning(f"Length mismatch for domain {domain_id}: Sequence={len(sequence)}, RMSF={len(rmsf_values)}. Skipping.")
                skipped_domains_length_mismatch.add(domain_id)
        else:
             logger.warning(f"Domain {domain_id} resulted in an empty sequence. Skipping.")
             skipped_domains_no_sequence.add(domain_id)

    # Report skipped counts
    if skipped_residues:
        logger.warning(f"Encountered unknown residues (counts): {dict(skipped_residues)}")
    if skipped_domains_missing_cols:
         logger.warning(f"Skipped {len(skipped_domains_missing_cols)} domains due to missing columns.")
    if skipped_domains_length_mismatch:
         logger.warning(f"Skipped {len(skipped_domains_length_mismatch)} domains due to length mismatch.")
    if skipped_domains_no_sequence:
         logger.warning(f"Skipped {len(skipped_domains_no_sequence)} domains due to empty sequence.")

    logger.info(f"Successfully extracted sequence, RMSF, and temperature for {processed_count} domains.")
    return processed_data

def extract_topology(domain_id: str) -> str:
    """Extract topology identifier (e.g., PDB ID) from domain_id."""
    # Simple example: Assume first 4 chars are PDB ID
    if isinstance(domain_id, str) and len(domain_id) >= 4:
        # Handle cases like '1xyz' or '1xyz_A' or '1xyz.A' -> '1XYZ'
        pdb_id = domain_id[:4].upper()
        # Basic check if it looks like a PDB ID (e.g., digit + 3 alphanumeric)
        if len(pdb_id) == 4 and pdb_id[0].isdigit() and pdb_id[1:].isalnum():
            return pdb_id
        else:
            # Fallback if first 4 chars don't look like PDB ID - use a hash or more robust parsing
             logger.debug(f"Domain ID '{domain_id}' doesn't start with PDB pattern. Using fallback topology hash.")
             # Use a more stable hash part if IDs can vary slightly but belong together
             base_id = domain_id.split('_')[0].split('.')[0] # Example heuristic
             return f"topo_{hash(base_id)}"

    logger.warning(f"Could not reliably extract topology from domain_id: {domain_id}. Using fallback.")
    return f"unknown_{hash(domain_id)}"

def split_by_topology(data: Dict[str, Dict], train_ratio=0.7, val_ratio=0.15, seed=42) -> Tuple[Dict, Dict, Dict]:
    """Split data by topology to ensure no topology overlap between splits."""
    if not data:
        logger.warning("No data provided to split_by_topology. Returning empty splits.")
        return {}, {}, {}

    random.seed(seed)
    logger.info(f"Splitting {len(data)} domains by topology using seed {seed}")

    topology_groups = defaultdict(list)
    for domain_id in data.keys():
        topology = extract_topology(domain_id)
        topology_groups[topology].append(domain_id)

    logger.info(f"Found {len(topology_groups)} unique topologies.")

    topologies = list(topology_groups.keys())
    random.shuffle(topologies)

    n_topologies = len(topologies)
    if n_topologies < 3:
        logger.warning(f"Very few topologies ({n_topologies}). Splits might be skewed or empty.")

    train_idx = int(n_topologies * train_ratio)
    val_idx = train_idx + int(n_topologies * val_ratio)
    # Adjust indices to prevent empty splits if possible
    if n_topologies == 1:
         train_idx, val_idx = 1, 1 # Put all in train, val/test empty
    elif n_topologies == 2:
         train_idx, val_idx = 1, 1 # 1 for train, 1 for test, val empty
         if val_ratio > 0: # If user wants val, put 1 in val instead of test
              val_idx = 2
    elif train_idx == 0 and n_topologies > 0 : # Ensure train has at least one
         train_idx = 1
         val_idx = max(val_idx, train_idx) # Ensure val_idx >= train_idx
    elif train_idx == val_idx and val_idx < n_topologies: # Ensure val has at least one if possible
         val_idx += 1
    elif val_idx == n_topologies and train_idx < val_idx: # Ensure test has at least one if possible
         val_idx -= 1
         # Re-ensure val has at least one if train took everything before test
         if train_idx == val_idx:
              train_idx = max(0, train_idx -1)


    train_topologies = set(topologies[:train_idx])
    val_topologies = set(topologies[train_idx:val_idx])
    test_topologies = set(topologies[val_idx:])

    logger.info(f"Calculated split indices: Train end={train_idx}, Val end={val_idx}, Total={n_topologies}")
    logger.info(f"Split topologies: Train={len(train_topologies)}, Val={len(val_topologies)}, Test={len(test_topologies)}")
    # Log overlaps as a sanity check (should be empty)
    logger.debug(f"Train/Val overlap: {len(train_topologies.intersection(val_topologies))}")
    logger.debug(f"Train/Test overlap: {len(train_topologies.intersection(test_topologies))}")
    logger.debug(f"Val/Test overlap: {len(val_topologies.intersection(test_topologies))}")


    train_data, val_data, test_data = {}, {}, {}
    assigned_domains = 0
    unassigned_domains = []
    for domain_id, domain_info in data.items():
        topology = extract_topology(domain_id)
        assigned = False
        if topology in train_topologies:
            train_data[domain_id] = domain_info
            assigned = True
        elif topology in val_topologies:
            val_data[domain_id] = domain_info
            assigned = True
        elif topology in test_topologies:
            test_data[domain_id] = domain_info
            assigned = True

        if assigned:
            assigned_domains += 1
        else:
            logger.warning(f"Domain {domain_id} with topology {topology} was not assigned to any split!")
            unassigned_domains.append(domain_id)


    logger.info(f"Split domains: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    if assigned_domains != len(data):
         logger.warning(f"Mismatch in assigned domains ({assigned_domains}) vs total domains ({len(data)}). Unassigned: {len(unassigned_domains)}")
         # Log first few unassigned for debugging
         # logger.warning(f" Example unassigned: {unassigned_domains[:5]}")

    return train_data, val_data, test_data

def save_split_data(data: Dict[str, Dict], output_dir: str, split_name: str):
    """
    Save split data (domain list, FASTA, RMSF numpy, Temperature numpy) to disk.
    #!#!# REFACTOR NOTE: Added saving of temperatures.
    """
    if not data:
        logger.warning(f"No data to save for split '{split_name}'. Skipping save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    domain_ids = list(data.keys())

    # --- Save domain list ---
    domain_list_path = os.path.join(output_dir, f"{split_name}_domains.txt")
    try:
        with open(domain_list_path, 'w') as f:
            for domain_id in domain_ids:
                f.write(f"{domain_id}\n")
        logger.info(f"Saved {len(domain_ids)} domain IDs to {domain_list_path}")
    except IOError as e:
        logger.error(f"Error writing domain list {domain_list_path}: {e}")

    # --- Save sequences in FASTA format ---
    fasta_path = os.path.join(output_dir, f"{split_name}_sequences.fasta")
    try:
        with open(fasta_path, 'w') as f:
            for domain_id in domain_ids:
                if 'sequence' in data.get(domain_id, {}):
                    f.write(f">{domain_id}\n{data[domain_id]['sequence']}\n")
                else:
                    logger.warning(f"Missing 'sequence' key for domain {domain_id} when saving FASTA for split {split_name}.")
        logger.info(f"Saved sequences to {fasta_path}")
    except IOError as e:
        logger.error(f"Error writing FASTA file {fasta_path}: {e}")

    # --- Save RMSF values as a NumPy dictionary ---
    rmsf_path = os.path.join(output_dir, f"{split_name}_rmsf.npy")
    rmsf_data_to_save = {}
    for domain_id in domain_ids:
        domain_info = data.get(domain_id, {})
        if 'rmsf' in domain_info:
             rmsf_array = domain_info['rmsf']
             if not isinstance(rmsf_array, np.ndarray):
                  logger.warning(f"RMSF data for {domain_id} is not a numpy array (type: {type(rmsf_array)}). Attempting conversion.")
                  try:
                      rmsf_array = np.array(rmsf_array, dtype=np.float32)
                  except Exception as conv_err:
                       logger.error(f"Could not convert RMSF data for {domain_id} to numpy array: {conv_err}. Skipping RMSF.")
                       continue # Skip saving RMSF for this domain
             rmsf_data_to_save[domain_id] = rmsf_array
        else:
             logger.warning(f"Missing 'rmsf' key for domain {domain_id} when saving RMSF data for split {split_name}.")

    if rmsf_data_to_save:
        try:
            np.save(rmsf_path, rmsf_data_to_save, allow_pickle=True)
            logger.info(f"Saved RMSF data for {len(rmsf_data_to_save)} domains to {rmsf_path}")
        except Exception as e:
            logger.error(f"Error saving RMSF numpy file {rmsf_path}: {e}", exc_info=True)
    else:
        logger.warning(f"No valid RMSF data found to save for split {split_name}.")

    # --- #!#!# REFACTOR NOTE: Save Temperature values as a NumPy dictionary ---
    temp_path = os.path.join(output_dir, f"{split_name}_temperatures.npy")
    temp_data_to_save = {}
    for domain_id in domain_ids:
         domain_info = data.get(domain_id, {})
         if 'temperature' in domain_info:
              temp_val = domain_info['temperature']
              try:
                   # Ensure temperature is a float before saving
                   temp_data_to_save[domain_id] = float(temp_val)
              except (ValueError, TypeError) as temp_err:
                    logger.error(f"Could not convert temperature data for {domain_id} to float: {temp_val}. Error: {temp_err}. Skipping temperature.")
                    continue # Skip saving temp for this domain
         else:
              logger.warning(f"Missing 'temperature' key for domain {domain_id} when saving temperature data for split {split_name}.")

    if temp_data_to_save:
        try:
            np.save(temp_path, temp_data_to_save, allow_pickle=True)
            logger.info(f"Saved Temperature data for {len(temp_data_to_save)} domains to {temp_path}")
        except Exception as e:
            logger.error(f"Error saving Temperature numpy file {temp_path}: {e}", exc_info=True)
    else:
        logger.warning(f"No valid Temperature data found to save for split {split_name}.")


def calculate_and_save_temp_scaling(train_data: Dict[str, Dict], output_dir: str, filename: str):
    """
    Calculates min/max temperature from the training data and saves them.
    #!#!# REFACTOR NOTE: New function for temperature scaling parameters.
    """
    if not train_data:
        logger.error("No training data provided. Cannot calculate temperature scaling parameters.")
        return

    temps = [d['temperature'] for d in train_data.values() if 'temperature' in d]
    if not temps:
        logger.error("No temperature values found in training data. Cannot calculate scaling parameters.")
        return

    temp_min = float(np.min(temps))
    temp_max = float(np.max(temps))

    scaling_params = {'temp_min': temp_min, 'temp_max': temp_max}
    save_path = os.path.join(output_dir, filename)

    try:
        with open(save_path, 'w') as f:
            json.dump(scaling_params, f, indent=4)
        logger.info(f"Calculated temperature scaling params (Min={temp_min}, Max={temp_max}) using {len(temps)} training samples.")
        logger.info(f"Saved temperature scaling parameters to {save_path}")
    except IOError as e:
        logger.error(f"Error saving temperature scaling parameters to {save_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving temperature scaling parameters: {e}", exc_info=True)


def process_data(csv_path: str, output_dir: str, temp_scaling_filename: str, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Main function to process RMSF data, extract temp, create splits, and save scaling params."""
    logger.info(f"--- Starting Data Processing Pipeline (Temperature Aware) ---")
    logger.info(f"Input CSV: {csv_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Split Ratios: Train={train_ratio}, Val={val_ratio}, Test={1 - train_ratio - val_ratio:.2f}")
    logger.info(f"Random Seed: {seed}")
    logger.info(f"Temp Scaling Filename: {temp_scaling_filename}")

    try:
        # 1. Load Data
        df = load_data(csv_path)
        if df is None: raise ValueError("Failed to load data.")

        # 2. Group by Domain
        domains = group_by_domain(df)
        if not domains: raise ValueError("Failed to group data by domain.")

        # 3. Extract Sequence, RMSF, and Temperature
        # #!#!# REFACTOR NOTE: Using updated extraction function
        data = extract_sequence_rmsf_temp(domains)
        if not data: raise ValueError("No valid domain data extracted.")

        # 4. Split by Topology
        train_data, val_data, test_data = split_by_topology(data, train_ratio, val_ratio, seed)

        # 5. Save Splits (including temperature)
        save_split_data(train_data, output_dir, 'train')
        save_split_data(val_data, output_dir, 'val')
        save_split_data(test_data, output_dir, 'test')

        # 6. #!#!# REFACTOR NOTE: Calculate and Save Temperature Scaling Parameters (using TRAIN data only)
        calculate_and_save_temp_scaling(train_data, output_dir, temp_scaling_filename)

        logger.info("--- Data Processing Completed Successfully ---")
        return train_data, val_data, test_data # Return data for potential chaining

    except FileNotFoundError as e:
         logger.error(f"Processing failed: {e}")
         return None, None, None
    except ValueError as e:
         logger.error(f"Processing failed: {e}")
         return None, None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during data processing: {e}", exc_info=True)
        return None, None, None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process protein RMSF data, extract sequence/RMSF/temperature, split by topology, and save scaling parameters.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the input aggregated RMSF CSV file.')
    parser.add_argument('--output', type=str, default='data/processed', help='Directory to save the processed data splits and scaling info.')
    # #!#!# REFACTOR NOTE: Added argument for scaling param filename
    parser.add_argument('--scaling_file', type=str, default='temp_scaling_params.json', help='Filename for saving temperature scaling parameters (min/max).')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Fraction of topologies for the training set.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Fraction of topologies for the validation set.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling topologies.')
    args = parser.parse_args()

    process_data(
        csv_path=args.csv,
        output_dir=args.output,
        temp_scaling_filename=args.scaling_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
EOF

# === Create dataset.py ===
echo "Creating dataset.py..."
cat << 'EOF' > dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RMSFDataset(Dataset):
    """
    PyTorch Dataset for Temperature-Aware RMSF prediction.

    Handles loading sequences, target RMSF values, and temperatures.
    #!#!# REFACTOR NOTE: Modified to include temperature.
    """
    def __init__(self,
                 domain_ids: List[str],
                 sequences: Dict[str, str],
                 rmsf_values: Dict[str, np.ndarray],
                 temperatures: Dict[str, float]): #!#!# Added temperatures
        """
        Initialize the dataset.

        Args:
            domain_ids: Ordered list of domain IDs for this dataset split.
            sequences: Dictionary mapping domain IDs to amino acid sequences.
            rmsf_values: Dictionary mapping domain IDs to target RMSF values (NumPy arrays).
            temperatures: Dictionary mapping domain IDs to temperature values (float).
        """
        self.domain_ids = domain_ids
        self.sequences = sequences
        self.rmsf_values = rmsf_values # Target RMSF
        self.temperatures = temperatures #!#!# Store temperatures

        # Data Consistency Check
        valid_domain_ids = []
        removed_count = 0
        for did in list(self.domain_ids): # Iterate over a copy
            # #!#!# REFACTOR NOTE: Check for sequence, RMSF, AND temperature
            if did in self.sequences and did in self.rmsf_values and did in self.temperatures:
                 # Basic length check remains useful
                 if len(self.sequences[did]) != len(self.rmsf_values[did]):
                     logger.warning(f"Length mismatch for {did}: Seq={len(self.sequences[did])}, RMSF={len(self.rmsf_values[did])}. Removing.")
                     removed_count += 1
                 # Check if temperature is valid (e.g., not NaN, although should be caught earlier)
                 elif self.temperatures[did] is None or np.isnan(self.temperatures[did]):
                     logger.warning(f"Invalid temperature for {did}: {self.temperatures[did]}. Removing.")
                     removed_count += 1
                 else:
                     valid_domain_ids.append(did) # Keep if all checks pass
            else:
                logger.warning(f"Domain ID {did} missing sequence, RMSF, or temperature. Removing.")
                removed_count += 1

        if removed_count > 0:
             logger.info(f"Removed {removed_count} domain IDs from dataset due to missing/inconsistent data.")
             self.domain_ids = valid_domain_ids

        # Calculate and log dataset statistics
        self._log_stats()

    def _log_stats(self):
        """Log statistics about the loaded dataset."""
        if not self.domain_ids:
            logger.warning("Dataset created with 0 proteins.")
            return

        num_proteins = len(self.domain_ids)
        logger.info(f"Dataset created with {num_proteins} proteins.")
        try:
            seq_lengths = [len(self.sequences[did]) for did in self.domain_ids]
            rmsf_lengths = [len(self.rmsf_values[did]) for did in self.domain_ids]
            temp_values = [self.temperatures[did] for did in self.domain_ids]

            logger.info(f"  Sequence length stats: Min={min(seq_lengths)}, Max={max(seq_lengths)}, " +
                        f"Mean={np.mean(seq_lengths):.1f}, Median={np.median(seq_lengths):.1f}")
            logger.info(f"  RMSF length stats:     Min={min(rmsf_lengths)}, Max={max(rmsf_lengths)}, " +
                        f"Mean={np.mean(rmsf_lengths):.1f}, Median={np.median(rmsf_lengths):.1f}")
            logger.info(f"  Temperature stats:     Min={min(temp_values):.1f}, Max={max(temp_values):.1f}, " +
                        f"Mean={np.mean(temp_values):.1f}, Median={np.median(temp_values):.1f}")

            if np.mean(seq_lengths) != np.mean(rmsf_lengths):
                 logger.warning("Mean sequence length differs from mean RMSF length. Verify processing.")
        except Exception as e:
             logger.error(f"Error calculating dataset statistics: {e}", exc_info=True)


    def __len__(self) -> int:
        return len(self.domain_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get data for a single protein by index.

        Args:
            idx: Index of the protein.

        Returns:
            Dictionary containing:
              - 'domain_id': The domain identifier (string).
              - 'sequence': The amino acid sequence (string).
              - 'rmsf': The target RMSF values (NumPy array of float32).
              - 'temperature': The temperature value (float). #!#!# Added temperature
        """
        if idx < 0 or idx >= len(self.domain_ids):
             raise IndexError(f"Index {idx} out of bounds for dataset size {len(self.domain_ids)}")

        domain_id = self.domain_ids[idx]

        # Retrieve data, handling potential KeyError if consistency check failed unexpectedly
        try:
            sequence = self.sequences[domain_id]
            rmsf = self.rmsf_values[domain_id]
            temperature = self.temperatures[domain_id] #!#!# Get temperature
        except KeyError as e:
            logger.error(f"Data inconsistency: Cannot find '{e}' for domain ID {domain_id} at index {idx}. Was it filtered out?")
            # Decide how to handle: raise error, return None, return dummy? Raising is safest.
            raise RuntimeError(f"Inconsistent dataset state: Missing data for {domain_id}") from e


        # Ensure RMSF is float32
        if rmsf.dtype != np.float32:
             rmsf = rmsf.astype(np.float32)

        return {
            'domain_id': domain_id,
            'sequence': sequence,
            'rmsf': rmsf, # This is the TARGET RMSF
            'temperature': float(temperature) # Ensure float type
        }

def load_sequences_from_fasta(fasta_path: str) -> Dict[str, str]:
    """Loads sequences from a FASTA file."""
    sequences = {}
    current_id = None
    current_seq = ""
    try:
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if current_id is not None:
                        sequences[current_id] = current_seq
                    current_id = line[1:].split()[0] # Use ID before first space
                    current_seq = ""
                else:
                    current_seq += line.upper()
            if current_id is not None: # Add last sequence
                sequences[current_id] = current_seq
        logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    except FileNotFoundError:
        logger.error(f"FASTA file not found: {fasta_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading FASTA file {fasta_path}: {e}")
        raise
    return sequences

def load_numpy_dict(npy_path: str) -> Dict[str, Any]:
    """Loads a dictionary saved as a NumPy file."""
    if not os.path.exists(npy_path):
        logger.error(f"NumPy file not found: {npy_path}")
        raise FileNotFoundError(f"NumPy file not found: {npy_path}")
    try:
        # allow_pickle=True is required for loading dictionaries
        loaded_data = np.load(npy_path, allow_pickle=True).item()
        # Ensure keys are strings for consistency
        string_key_data = {str(k): v for k, v in loaded_data.items()}
        logger.info(f"Loaded {len(string_key_data)} entries from {npy_path}")
        return string_key_data
    except Exception as e:
        logger.error(f"Error loading or processing NumPy dictionary from {npy_path}: {e}")
        raise


# #!#!# REFACTOR NOTE: Updated function signature and logic to load temperature
def load_split_data(data_dir: str, split: str) -> Tuple[List[str], Dict[str, str], Dict[str, np.ndarray], Dict[str, float]]:
    """
    Load data (domain IDs, sequences, RMSF values, temperatures) for a specific split.

    Args:
        data_dir: Directory containing the processed data files.
        split: Split name ('train', 'val', or 'test').

    Returns:
        Tuple of (domain_ids, sequences, rmsf_values, temperatures).
        Returns ([], {}, {}, {}) if data loading fails for essential components.
    """
    logger.info(f"--- Loading {split} data from directory: {data_dir} ---")
    sequences, rmsf_values, temperatures = {}, {}, {}
    domain_ids = []

    try:
        # --- Load domain IDs (essential) ---
        domain_ids_path = os.path.join(data_dir, f"{split}_domains.txt")
        if not os.path.exists(domain_ids_path):
             logger.error(f"Domain ID file not found: {domain_ids_path}")
             return [], {}, {}, {}
        with open(domain_ids_path, 'r') as f:
            domain_ids = [line.strip() for line in f if line.strip()]
        if not domain_ids:
             logger.warning(f"Domain ID file is empty: {domain_ids_path}")
             # Allow proceeding but likely useless
        logger.info(f"Loaded {len(domain_ids)} domain IDs from {domain_ids_path}")

        # --- Load sequences (essential) ---
        sequences_path = os.path.join(data_dir, f"{split}_sequences.fasta")
        sequences = load_sequences_from_fasta(sequences_path)
        if not sequences:
             logger.error(f"No sequences loaded from required file: {sequences_path}")
             return [], {}, {}, {} # Treat as fatal if no sequences

        # --- Load RMSF values (essential) ---
        rmsf_path = os.path.join(data_dir, f"{split}_rmsf.npy")
        rmsf_dict = load_numpy_dict(rmsf_path)
        # Ensure values are float32 numpy arrays
        rmsf_values = {k: np.array(v, dtype=np.float32) for k, v in rmsf_dict.items()}
        if not rmsf_values:
             logger.error(f"No RMSF data loaded from required file: {rmsf_path}")
             return [], {}, {}, {} # Treat as fatal

        # --- Load Temperatures (essential) --- #!#!# REFACTOR NOTE: Load temperatures
        temperatures_path = os.path.join(data_dir, f"{split}_temperatures.npy")
        temp_dict = load_numpy_dict(temperatures_path)
        # Ensure values are floats
        temperatures = {k: float(v) for k, v in temp_dict.items()}
        if not temperatures:
             logger.error(f"No Temperature data loaded from required file: {temperatures_path}")
             return [], {}, {}, {} # Treat as fatal

    except FileNotFoundError as e:
        logger.error(f"Failed to load essential data file: {e}")
        return [], {}, {}, {}
    except Exception as e:
        logger.error(f"An error occurred during data loading for split '{split}': {e}", exc_info=True)
        return [], {}, {}, {}

    # --- Verify data consistency across all loaded components ---
    logger.info("Verifying data consistency for split '{}'...".format(split))
    original_domain_count = len(domain_ids)
    valid_domain_ids = []
    missing_data_counts = defaultdict(int)
    length_mismatches = 0

    for did in domain_ids:
        has_seq = did in sequences
        has_rmsf = did in rmsf_values
        has_temp = did in temperatures #!#!# Check temperature presence

        if has_seq and has_rmsf and has_temp:
            # Check sequence-RMSF length consistency
            seq_len = len(sequences[did])
            rmsf_len = len(rmsf_values[did])
            if seq_len == rmsf_len:
                # Check temperature validity (redundant if checked earlier, but safe)
                if temperatures[did] is not None and not np.isnan(temperatures[did]):
                    valid_domain_ids.append(did)
                else:
                     missing_data_counts['invalid_temp'] += 1
                     logger.debug(f"Invalid temperature for {did}. Removing.")
            else:
                length_mismatches += 1
                logger.debug(f"Length mismatch for {did}: seq={seq_len}, RMSF={rmsf_len}. Removing.")
        else:
            if not has_seq: missing_data_counts['sequence'] += 1; logger.debug(f"Missing sequence for {did}")
            if not has_rmsf: missing_data_counts['rmsf'] += 1; logger.debug(f"Missing RMSF for {did}")
            if not has_temp: missing_data_counts['temperature'] += 1; logger.debug(f"Missing temperature for {did}")

    logger.info(f"Initial domain IDs in list: {original_domain_count}")
    if sum(missing_data_counts.values()) > 0:
         logger.warning(f"Missing data counts: {dict(missing_data_counts)}")
    if length_mismatches > 0:
        logger.warning(f"Found {length_mismatches} domains with sequence-RMSF length mismatches.")

    final_domain_count = len(valid_domain_ids)
    if final_domain_count != original_domain_count:
        removed_count = original_domain_count - final_domain_count
        logger.info(f"Removed {removed_count} domains due to inconsistencies.")
        logger.info(f"Final number of valid, consistent domains for split '{split}': {final_domain_count}")

    # Filter all dictionaries to only include valid domains
    final_sequences = {did: sequences[did] for did in valid_domain_ids if did in sequences}
    final_rmsf_values = {did: rmsf_values[did] for did in valid_domain_ids if did in rmsf_values}
    final_temperatures = {did: temperatures[did] for did in valid_domain_ids if did in temperatures} #!#!# Filter temperatures

    if final_domain_count == 0:
         logger.error(f"No valid, consistent data found for split '{split}' after filtering. Please check the processed data files in {data_dir}.")
         # Return empty structures to avoid downstream errors
         return [], {}, {}, {}


    logger.info(f"--- Successfully loaded and verified {final_domain_count} samples for split '{split}' ---")
    return valid_domain_ids, final_sequences, final_rmsf_values, final_temperatures


# #!#!# REFACTOR NOTE: Updated collate function to handle temperature
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for the DataLoader.

    Batches domain IDs, sequences, RMSF values (as Tensors), and temperatures (as Tensors).
    Padding is NOT done here.
    """
    domain_ids = [item['domain_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    # Convert RMSF numpy arrays to tensors (target values)
    rmsf_values = [torch.tensor(item['rmsf'], dtype=torch.float32) for item in batch]
    # #!#!# REFACTOR NOTE: Extract and convert temperatures to a tensor
    temperatures = torch.tensor([item['temperature'] for item in batch], dtype=torch.float32)

    return {
        'domain_ids': domain_ids,
        'sequences': sequences,
        'rmsf_values': rmsf_values, # List of Tensors (targets)
        'temperatures': temperatures # Tensor of shape [batch_size] (input features)
    }


# #!#!# REFACTOR NOTE: Updated DataLoader creation function
def create_length_batched_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    max_length: Optional[int] = None,
    length_bucket_size: int = 50,
    num_workers: int = 0
) -> Optional[DataLoader]:
    """
    Creates a PyTorch DataLoader with length-based batching, including temperatures.

    Args:
        data_dir: Directory containing the processed data splits.
        split: Split name ('train', 'val', or 'test').
        batch_size: Target number of sequences per batch.
        shuffle: Whether to shuffle data.
        max_length: Optional maximum sequence length for filtering.
        length_bucket_size: Size of length ranges for grouping.
        num_workers: Number of worker processes.

    Returns:
        A PyTorch DataLoader instance, or None if data loading fails.
    """
    # 1. Load data (including temperatures)
    # #!#!# REFACTOR NOTE: Unpack temperatures from load_split_data
    domain_ids, sequences, rmsf_values, temperatures = load_split_data(data_dir, split)

    if not domain_ids:
        logger.error(f"Failed to load any valid data for split '{split}'. Cannot create DataLoader.")
        return None

    # 2. Filter by max length if specified
    if max_length is not None and max_length > 0:
        original_count = len(domain_ids)
        # Keep only IDs whose sequences are <= max_length
        filtered_domain_ids = [
            did for did in domain_ids if len(sequences.get(did, '')) <= max_length
        ]
        filtered_count = len(filtered_domain_ids)
        if filtered_count < original_count:
            logger.info(f"Filtered out {original_count - filtered_count} sequences " +
                        f"longer than {max_length} residues for split '{split}'.")
            domain_ids = filtered_domain_ids
            # Filter all dictionaries based on the remaining domain_ids
            sequences = {did: sequences[did] for did in domain_ids if did in sequences}
            rmsf_values = {did: rmsf_values[did] for did in domain_ids if did in rmsf_values}
            temperatures = {did: temperatures[did] for did in domain_ids if did in temperatures} #!#!# Filter temps

        if not domain_ids:
             logger.warning(f"No sequences remaining after filtering by max_length={max_length} for split '{split}'. Cannot create DataLoader.")
             return None

    # 3. Group domain IDs by length buckets
    length_buckets = defaultdict(list)
    for did in domain_ids:
        # Use sequence length for bucketing
        seq_len = len(sequences.get(did, ''))
        if seq_len > 0: # Avoid bucketing empty sequences if any slipped through
            bucket_idx = seq_len // length_bucket_size
            length_buckets[bucket_idx].append(did)
        else:
             logger.warning(f"Domain ID {did} has zero length sequence during bucketing. Skipping.")


    if not length_buckets:
         logger.error(f"No non-empty sequences found to create length buckets for split '{split}'. Cannot create DataLoader.")
         return None

    logger.info(f"Grouped {len(domain_ids)} sequences into {len(length_buckets)} length buckets.")

    # 4. Create batches within buckets
    all_batches = []
    sorted_bucket_indices = sorted(length_buckets.keys())

    for bucket_idx in sorted_bucket_indices:
        bucket_domain_ids = length_buckets[bucket_idx]
        if shuffle:
            random.shuffle(bucket_domain_ids)

        for i in range(0, len(bucket_domain_ids), batch_size):
            batch_domain_ids = bucket_domain_ids[i : i + batch_size]
            all_batches.append(batch_domain_ids)

    # 5. Shuffle the order of batches for training
    if shuffle:
        random.shuffle(all_batches)

    # 6. Flatten the batches to get the final ordered list of domain IDs for the epoch
    ordered_domain_ids = [did for batch in all_batches for did in batch]

    # 7. Create the Dataset with the final order and all required data dicts
    # #!#!# REFACTOR NOTE: Pass temperatures dict to RMSFDataset
    dataset = RMSFDataset(ordered_domain_ids, sequences, rmsf_values, temperatures)

    if len(dataset) == 0:
         logger.error(f"Final dataset for split '{split}' is empty after processing. Cannot create DataLoader.")
         return None

    # 8. Create the DataLoader
    logger.info(f"Creating DataLoader for {len(dataset)} samples for split '{split}' with batch size {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffling is handled by length batching strategy
        collate_fn=collate_fn, # Use updated collate function
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False # Keep all data
    )

# Example Usage (if script is run directly)
if __name__ == "__main__":
    logger.info("Testing Temperature-Aware DataLoader creation...")
    # Create dummy data for testing
    dummy_data_dir = "data/processed_dummy_temp"
    os.makedirs(dummy_data_dir, exist_ok=True)

    dummy_domains = [f"D{i:03d}" for i in range(100)]
    dummy_sequences = {}
    dummy_rmsf = {}
    dummy_temps = {} #!#!# Add dummy temps
    for i, did in enumerate(dummy_domains):
        length = random.randint(50, 250)
        dummy_sequences[did] = "A" * length
        dummy_rmsf[did] = np.random.rand(length).astype(np.float32) * 2.0
        dummy_temps[did] = random.choice([298.0, 310.0, 320.0, 330.0]) #!#!# Assign random temp

    # Save dummy data
    with open(os.path.join(dummy_data_dir, "train_domains.txt"), "w") as f: f.write("\n".join(dummy_domains))
    with open(os.path.join(dummy_data_dir, "train_sequences.fasta"), "w") as f:
        for did, seq in dummy_sequences.items(): f.write(f">{did}\n{seq}\n")
    np.save(os.path.join(dummy_data_dir, "train_rmsf.npy"), dummy_rmsf)
    np.save(os.path.join(dummy_data_dir, "train_temperatures.npy"), dummy_temps) #!#!# Save dummy temps

    # Test dataloader creation
    train_loader = create_length_batched_dataloader(
        data_dir=dummy_data_dir,
        split='train',
        batch_size=16,
        shuffle=True,
        max_length=200,
        length_bucket_size=25
    )

    if train_loader:
        logger.info("DataLoader created successfully. Iterating through a few batches...")
        batch_count = 0
        max_batches_to_show = 3
        for i, batch in enumerate(train_loader):
            if i >= max_batches_to_show: break
            logger.info(f"Batch {i+1}:")
            logger.info(f"  Domain IDs: {batch['domain_ids']}")
            logger.info(f"  Num sequences: {len(batch['sequences'])}")
            logger.info(f"  Seq lengths: {[len(s) for s in batch['sequences']]}")
            logger.info(f"  RMSF Tensors: {[t.shape for t in batch['rmsf_values']]}")
            logger.info(f"  Temperatures Tensor: {batch['temperatures']}") #!#!# Log temps tensor
            logger.info(f"  Temperatures Tensor Shape: {batch['temperatures'].shape}")
            batch_count += 1
        logger.info(f"Iterated through {batch_count} batches.")
    else:
        logger.error("Failed to create DataLoader.")

    # Clean up dummy data
    # import shutil
    # try:
    #      shutil.rmtree(dummy_data_dir)
    #      logger.info(f"Cleaned up dummy data directory: {dummy_data_dir}")
    # except OSError as e:
    #      logger.error(f"Error removing dummy directory {dummy_data_dir}: {e}")

EOF

# === Create model.py ===
echo "Creating model.py..."
cat << 'EOF' > model.py
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
    #!#!# REFACTOR NOTE: Renamed class and modified architecture.
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
EOF

# === Create train.py ===
echo "Creating train.py..."
cat << 'EOF' > train.py
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

EOF

# === Create predict.py ===
echo "Creating predict.py..."
cat << 'EOF' > predict.py
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
    """Plot predicted RMSF values against residue position."""
    if predictions is None or len(predictions) == 0:
        logger.warning(f"No prediction data to plot for '{title}'. Skipping plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    pred_len = len(predictions)
    residue_indices = np.arange(1, pred_len + 1)

    if window_size > 1:
        s = pd.Series(predictions)
        plot_data = s.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
        plot_label = f'RMSF Prediction (Smoothed, win={window_size})'
    else:
        plot_data = predictions
        plot_label = 'RMSF Prediction'

    ax.plot(residue_indices, plot_data, '-', color='dodgerblue', linewidth=1.5, label=plot_label)

    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Predicted RMSF')
    ax.set_title(f'Predicted RMSF for {title} (Length: {pred_len})') # Title now includes Temp
    ax.set_xlim(0, pred_len + 1)
    ax.grid(True, linestyle=':', alpha=0.7)

    # Add stats text box
    mean_rmsf = np.mean(predictions)
    median_rmsf = np.median(predictions)
    stats_text = (f'Mean: {mean_rmsf:.3f}\n'
                  f'Median: {median_rmsf:.3f}')
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.5))

    ax.legend(loc='upper right')
    plt.tight_layout()

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight') # Lower DPI for potentially many plots
    except Exception as e:
        logger.error(f"Failed to save plot to {output_path}: {e}")
    finally:
        plt.close(fig)

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

EOF


# === Create main.py ===
echo "Creating main.py..."
cat << 'EOF' > main.py
#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import logging
import json

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
EOF

# === Create requirements.txt ===
echo "Creating requirements.txt..."
cat << 'EOF' > requirements.txt
# Core ML/DL
torch>=1.10.0 #,<2.3.0 # Check compatibility with your CUDA version
esm>=2.0.0 # Native ESM library from Meta
# transformers, accelerate, tokenizers might be needed by esm internally
transformers>=4.30.0
accelerate>=0.20.0
tokenizers>=0.13.0

# Data Handling & Utilities
numpy>=1.20.0
pandas>=1.3.0
pyyaml>=5.4
tqdm>=4.60.0

# Plotting
matplotlib>=3.4.0
seaborn # Often used with matplotlib for nicer plots

# Optional: For hyperparameter tuning
# optuna
# ray[tune]
EOF

# === Create empty utils/__init__.py ===
echo "Creating utils/__init__.py..."
touch utils/__init__.py

# === Create dummy placeholder files in data/processed ===
echo "Creating placeholder files in data/processed..."
touch data/processed/train_domains.txt
touch data/processed/val_domains.txt
touch data/processed/test_domains.txt

# === Create empty data/raw/fix_data_.py (Keep original if needed) ===
# If the original fix_data_.py is useful, copy it here instead of creating empty.
# For now, creating an empty placeholder as it's not directly part of the core refactor.
echo "Creating placeholder data/raw/fix_data_.py..."
cat << 'EOF' > data/raw/fix_data_.py
# This script was originally used to standardize residue names (e.g., HIS variants).
# Keep or adapt if needed for your new aggregated dataset *before* running the main 'process' command.
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("This is a placeholder for the data fixing script.")
    logger.info("If your aggregated CSV requires preprocessing (like standardizing residue names),")
    logger.info("implement the logic here and run it before using 'main.py process'.")
    # Example:
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', required=True)
    # parser.add_argument('--output', required=True)
    # args = parser.parse_args()
    # logger.info(f"Processing {args.input} to {args.output}...")
    # # Add processing logic here
    # logger.info("Processing finished (placeholder).")

EOF


# Make Python scripts executable
chmod +x *.py

echo "--------------------------------------------------"
echo "ESM-Flex Temperature-Aware project created."
echo "ACTION REQUIRED:"
echo "1. Edit 'config.yaml' and set 'data.raw_csv_path' to your aggregated data file."
echo "2. Place your aggregated CSV file at the specified path."
echo "3. Ensure dependencies are installed: pip install -r requirements.txt"
echo "--------------------------------------------------"
echo "Workflow:"
echo "1. Run data processing: python main.py process [--config config.yaml]"
echo "   (This will create processed splits and 'temp_scaling_params.json' in data/processed/)"
echo "2. Run training: python main.py train [--config config.yaml]"
echo "   (Models and logs will be saved in 'models/')"
echo "3. Run prediction: python main.py predict --model_checkpoint models/best_model.pt --fasta_path your_input.fasta --temperature 310"
echo "   (Replace temp, checkpoint, fasta path as needed. Results in 'predictions/310K/')"
echo "--------------------------------------------------"


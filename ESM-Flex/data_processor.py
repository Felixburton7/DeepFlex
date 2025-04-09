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

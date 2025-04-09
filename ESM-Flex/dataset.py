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


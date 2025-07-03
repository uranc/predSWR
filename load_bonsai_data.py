import os
import warnings
import numpy as np
from typing import Dict, Union, List

def load_bonsai_data(
    file_path: str = 'recording.dat',
    dtype: Union[str, np.dtype] = np.uint16,
    channels: int = 32
) -> np.ndarray:
    """
    Load a raw bonsai binary file recorded via Bonsai into a Numpy array.
    The data is assumed to be stored in interleaved format (i.e., all channels for
    sample 1, then all channels for sample 2, and so on), which is equivalent
    to a column-major ordering of a (channels, num_samples) matrix.

    Args:
        file_path: Path to the raw Bonsai binary file (default 'amplifier-data_1.raw').
        dtype: NumPy dtype of each sample (default np.uint16).
        channels: Number of channels in the file (default 32).

    Returns:
        data: np.ndarray of shape (num_samples, channels).
    """
    # Warn user about assumed offset
    warnings.warn(
        "Offset is currently assumed to be 0 bytes. (No header)",
        UserWarning
    )
    # Read raw data via memmap for efficiency
    raw = np.memmap(file_path, dtype=dtype, mode='r', offset=0)
    total = raw.size
    # Ensure full frames
    if total % channels != 0:
        usable = (total // channels) * channels
        raw = raw[:usable]
        warnings.warn(
        "There's a mismatch between the number of samples and channels. "
        "Getting usable data, until the last full frame."
        "Check the data.",
        UserWarning
    )
    data = raw.reshape(-1, channels)
    return np.asarray(data)

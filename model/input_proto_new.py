# ----------------------------- Imports --------------------------------------
import os
import pdb
import numpy as np
import tensorflow as tf
from scipy.ndimage import binary_dilation
from tensorflow.keras.utils import timeseries_dataset_from_array

# cnn-ripple utils from your repo
from model.cnn_ripple_utils import (
    load_lab_data, process_LFP, split_data,
    # the following are imported elsewhere in the repo but unused here:
    # filter_LFP, load_info, load_raw_data
)

from numpy.lib.stride_tricks import sliding_window_view


def sliding_window_zscore(x32, win=1250, eps=1e-8):
    x64 = x32.astype(np.float64)                 # temp up-cast
    n, c = x64.shape

    cs   = np.cumsum(np.pad(x64, ((1,0),(0,0))), axis=0, dtype=np.float64)
    cs2  = np.cumsum(np.pad(x64**2, ((1,0),(0,0))), axis=0, dtype=np.float64)

    idx0 = np.clip(np.arange(n) - win + 1, 0, None)
    L    = (np.arange(n) - idx0 + 1).astype(np.float64)

    w_sum  = cs [1:] - cs [idx0]
    w_sum2 = cs2[1:] - cs2[idx0]

    mu  = w_sum / L[:, None]
    var = w_sum2 / L[:, None] - mu**2            # population variance
    sig = np.sqrt(np.maximum(var, 0.0))

    return ((x64 - mu) / (sig + eps)).astype(np.float32)

# ----------------- Triplet-sampling helpers ---------------------------------
def _build_window_catalogue_2T(labels, win_len, grace, anchor_thr, pos_thr, min_neg_gap, multiplier=2):
    """
    Build window catalogue for nT extraction where:
    - We extract multiplier*win_len samples starting at returned indices
    - Overlap is computed only on the last win_len samples (label region)
    - Safety is checked for the entire multiplier*win_len window

    Args:
        multiplier: Window size multiplier (2 for 2T, 3 for 3T, etc.)
    """
    total_win_len = multiplier * win_len

    # 1) mark grace period
    pos_mask = labels.astype(bool).copy()
    onsets = np.flatnonzero(np.diff(np.r_[0, labels]) == 1)
    for onset in onsets:
        pos_mask[max(0, onset - grace):onset] = True

    # 2) For each potential starting index, compute overlap on last portion
    max_start_idx = len(labels) - total_win_len
    if max_start_idx < 0:
        print(f"Warning: Data too short for {multiplier}T windows. Need {total_win_len}, got {len(labels)}")
        return np.array([]), np.array([]), np.array([])

    overlap_ratios = np.zeros(max_start_idx + 1)
    for i in range(max_start_idx + 1):
        # Overlap computed on last portion: [i+(multiplier-1)*win_len:i+multiplier*win_len]
        label_start = i + (multiplier - 1) * win_len
        label_end = i + multiplier * win_len
        label_region = pos_mask[label_start:label_end]
        overlap_ratios[i] = np.sum(label_region) / win_len

    # 3) Safe region for negatives - entire nT window must be safe
    safe_mask = ~binary_dilation(pos_mask, structure=np.ones(min_neg_gap * 2 + 1))
    safe_nT = np.zeros(max_start_idx + 1, dtype=bool)
    for i in range(max_start_idx + 1):
        safe_nT[i] = np.all(safe_mask[i:i + total_win_len])

    # 4) Build catalogues - FIX THE BUG HERE
    valid_indices = np.arange(max_start_idx + 1)
    anchor_idx = valid_indices[overlap_ratios >= anchor_thr]
    # BUG: This excludes anchors from positives!
    # positive_idx = valid_indices[(overlap_ratios >= pos_thr) & (overlap_ratios < anchor_thr)]
    # FIXED: Include all windows with sufficient overlap as positives
    positive_idx = valid_indices[overlap_ratios >= pos_thr]
    negative_idx = valid_indices[(overlap_ratios == 0) & safe_nT]

    return anchor_idx, positive_idx, negative_idx


def _build_window_catalogue(labels,
                            win_len,
                            grace,
                            anchor_thr,
                            pos_thr,
                            min_neg_gap,
                            overlap_offset=0):
    """
    Return three 1-D numpy arrays of window-start indices:

        anchor_idx   \u2013 overlap \u2265 anchor_thr
        positive_idx \u2013 pos_thr \u2264 overlap < anchor_thr
        negative_idx \u2013 0 overlap and \u2265 min_neg_gap away from any event sample

    The grace period (samples prior to onset) is also treated as positive.

    Args:
        labels: Binary label array
        win_len: Window length for overlap computation
        grace: Grace period before event onset
        anchor_thr: Minimum overlap ratio for anchors
        pos_thr: Minimum overlap ratio for positives
        min_neg_gap: Minimum gap from events for negatives
        overlap_offset: Offset to apply when computing overlap (for 2T windows,
                       use offset=win_len to compute overlap on second half only)
    """
    # 1) mark grace period    
    pos_mask = labels.astype(bool).copy()
    onsets   = np.flatnonzero(np.diff(np.r_[0, labels]) == 1)
    for onset in onsets:
        pos_mask[max(0, onset - grace):onset] = True
    
    # 2) windowed overlap ratio
    win_vec        = np.ones(win_len, dtype=int)
    win_pos_counts = np.convolve(pos_mask, win_vec, mode="valid")
    overlap_ratio  = win_pos_counts / win_len

    # If using overlap_offset, shift the indices to account for the offset
    # This handles cases where we extract 2T but only care about second T
    if overlap_offset > 0 and len(overlap_ratio) > overlap_offset:
        # Compute overlap for the offset region
        offset_pos_counts = np.convolve(pos_mask[overlap_offset:], win_vec, mode="valid")
        if len(offset_pos_counts) > 0:
            offset_overlap_ratio = offset_pos_counts / win_len
            # Pad to match original length for indexing consistency
            overlap_ratio = np.pad(offset_overlap_ratio, (overlap_offset,
                                  max(0, len(overlap_ratio) - len(offset_pos_counts) - overlap_offset)),
                                  mode='constant', constant_values=0)

    # 3) safe region for negatives
    safe_mask  = ~binary_dilation(pos_mask,
                                  structure=np.ones(min_neg_gap * 2 + 1))
    win_safe   = np.convolve(safe_mask, win_vec, mode="valid") == win_len

    # 4) catalogues
    anchor_idx   = np.where(overlap_ratio >= anchor_thr)[0]
    positive_idx = np.where((overlap_ratio >= pos_thr) &
                            (overlap_ratio < anchor_thr))[0]
    negative_idx = np.where((overlap_ratio == 0) & win_safe)[0]

    return anchor_idx, positive_idx, negative_idx

def create_triplet_dataset(data,
                           labels,
                           params):
    """
    Fixed version with memory-efficient sampling and proper error handling
    """
    # ---- parameters ---------------------------------------------------------
    B            = params['BATCH_SIZE']
    W            = params['NO_TIMEPOINTS']  # This is T (target length)
    window_multiplier = params.get('WINDOW_MULTIPLIER', 2)
    W_total      = window_multiplier * W
    
    # Simplified parameters to reduce memory usage
    jitter       = params.get('JITTER', min(W // 8, 8))  # Reduced jitter
    min_gap      = params.get('MIN_NEG_GAP', 100)
    grace        = params.get('GRACE', 10)    
    anchor_thr   = params.get('ANCHOR_THR', 0.80)
    pos_thr      = params.get('POS_THR', 0.60)
    
    # Simplified onset parameters
    onset_bias   = params.get('ONSET_BIAS', 0.2)  # Reduced from 0.4
    use_onset_weighting = params.get('USE_ONSET_WEIGHTING', False)
    
    C = 1 if data.ndim == 1 else data.shape[1]
    
    print(f"Creating triplet dataset: B={B}, T={W}, {window_multiplier}T={W_total}, C={C}")

    # Validate data dimensions
    if len(data) < W_total:
        raise ValueError(f"Data too short: need {W_total} samples, got {len(data)}")

    # ---- Build catalogues with memory-efficient approach -------------------
    print("Building window catalogues...")
    anchor_idx, pos_idx, neg_idx = _build_window_catalogue_2T(
        labels, W, grace, anchor_thr, pos_thr, min_gap, multiplier=window_multiplier
    )
    
    if len(anchor_idx) == 0 or len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(
            f"Not enough candidate windows. Found: {len(anchor_idx)} anchors, "
            f"{len(pos_idx)} positives, {len(neg_idx)} negatives"
        )
    
    print(f"Window catalogues: {len(anchor_idx)} anchors, {len(pos_idx)} positives, {len(neg_idx)} negatives")
    
    # Calculate meaningful steps per epoch based on catalogue sizes
    effective_batch_size = 3 * B  # Each batch has B triplets = 3B samples
    
    # Use the limiting factor (usually negatives) to estimate steps
    limiting_samples = min(len(anchor_idx), len(pos_idx), len(neg_idx))
    estimated_steps_per_epoch = max(100, limiting_samples // B)  # At least 100 steps
    
    # Cap at reasonable maximum to prevent overly long epochs
    max_steps_per_epoch = 2000
    steps_per_epoch = min(estimated_steps_per_epoch, max_steps_per_epoch)
    
    print(f"Estimated steps per epoch: {steps_per_epoch} (limiting factor: {limiting_samples} samples)")
    
    # Store in params for later use
    params['ESTIMATED_STEPS_PER_EPOCH'] = steps_per_epoch
    
    # Create simple onset weights if requested
    anchor_weights = None
    pos_weights = None

    if use_onset_weighting and onset_bias > 0:
        print("Creating enhanced onset-focused weights...")
        anchor_weights = _create_enhanced_onset_weights(labels, anchor_idx, W, window_multiplier, debug=True)
        pos_weights = _create_enhanced_onset_weights(labels, pos_idx, W, window_multiplier, debug=True)

    # Calculate valid index range
    max_valid_idx = len(data) - W_total
    
    # Filter indices to ensure they're all valid
    anchor_idx = anchor_idx[anchor_idx <= max_valid_idx]
    pos_idx = pos_idx[pos_idx <= max_valid_idx]
    neg_idx = neg_idx[neg_idx <= max_valid_idx]
    
    # Update weights if they exist
    if anchor_weights is not None:
        anchor_weights = anchor_weights[:len(anchor_idx)]
    if pos_weights is not None:
        pos_weights = pos_weights[:len(pos_idx)]

    def _safe_extract_window(start_idx, data_array, label_array, window_size, label_size):
        """Safely extract window with bounds checking"""
        if start_idx < 0 or start_idx + window_size > len(data_array):
            # Fallback to a safe index
            start_idx = max(0, min(start_idx, len(data_array) - window_size))
        
        # Extract data window
        data_window = data_array[start_idx:start_idx + window_size]
        
        # Extract label window (last portion)
        label_offset = window_size - label_size
        label_window = label_array[start_idx + label_offset:start_idx + window_size]
        
        return data_window, label_window

    def _sample_triplet(rng):
        """Sample a single triplet with proper error handling"""
        try:
            # Sample indices with weights if available
            if anchor_weights is not None and len(anchor_weights) > 0:
                stable_weights = np.clip(anchor_weights, 0.1, 10.0)
                stable_weights = stable_weights / np.sum(stable_weights)
                a = rng.choice(anchor_idx, p=stable_weights)
            else:
                a = rng.choice(anchor_idx)
                
            if pos_weights is not None and len(pos_weights) > 0:
                stable_weights = np.clip(pos_weights, 0.1, 10.0)
                stable_weights = stable_weights / np.sum(stable_weights)
                p = rng.choice(pos_idx, p=stable_weights)
            else:
                p = rng.choice(pos_idx)
                
            n = rng.choice(neg_idx)
            
            # Apply jittering with bounds checking
            if jitter > 0:
                actual_jitter = min(jitter, 2)  # Cap at 2 samples max
                a_jitter = rng.integers(-actual_jitter, actual_jitter + 1)
                p_jitter = rng.integers(-actual_jitter, actual_jitter + 1)
                n_jitter = rng.integers(-actual_jitter, actual_jitter + 1)
                
                a = max(0, min(a + a_jitter, max_valid_idx))
                p = max(0, min(p + p_jitter, max_valid_idx))
                n = max(0, min(n + n_jitter, max_valid_idx))
            
            # Extract windows safely
            anchor_data, anchor_lab = _safe_extract_window(a, data, labels, W_total, W)
            positive_data, positive_lab = _safe_extract_window(p, data, labels, W_total, W)
            negative_data, negative_lab = _safe_extract_window(n, data, labels, W_total, W)

            # Apply onset shifting if enabled
            if params.get('USE_ONSET_SHIFTED_LABELS', False):
                shift_samples = params.get('ONSET_SHIFT_SAMPLES', -8)
                extend_samples = params.get('ONSET_EXTEND_SAMPLES', 15)
                
                anchor_lab = _create_onset_shifted_labels(anchor_lab, shift_samples, extend_samples)
                positive_lab = _create_onset_shifted_labels(positive_lab, shift_samples, extend_samples)
                # Don't shift negative labels
                
            # Ensure correct shapes for channel dimension
            if anchor_data.ndim == 1:
                anchor_data = anchor_data[:, None]
                positive_data = positive_data[:, None]
                negative_data = negative_data[:, None]
                
            # Validate final shapes
            assert anchor_data.shape == (W_total, C), f"Anchor shape mismatch: {anchor_data.shape} vs ({W_total}, {C})"
            assert positive_data.shape == (W_total, C), f"Positive shape mismatch: {positive_data.shape} vs ({W_total}, {C})"
            assert negative_data.shape == (W_total, C), f"Negative shape mismatch: {negative_data.shape} vs ({W_total}, {C})"
            assert anchor_lab.shape == (W,), f"Anchor label shape mismatch: {anchor_lab.shape} vs ({W},)"
            assert positive_lab.shape == (W,), f"Positive label shape mismatch: {positive_lab.shape} vs ({W},)"
            assert negative_lab.shape == (W,), f"Negative label shape mismatch: {negative_lab.shape} vs ({W},)"
                
            return ((anchor_data.astype(np.float32), anchor_lab.astype(np.float32)),
                    (positive_data.astype(np.float32), positive_lab.astype(np.float32)),
                    (negative_data.astype(np.float32), negative_lab.astype(np.float32)))
                    
        except Exception as e:
            print(f"Warning: Error in triplet sampling, using fallback: {e}")
            # Simple fallback sampling
            a = rng.choice(anchor_idx)
            p = rng.choice(pos_idx)
            n = rng.choice(neg_idx)
            
            # Ensure indices are valid
            a = max(0, min(a, max_valid_idx))
            p = max(0, min(p, max_valid_idx))
            n = max(0, min(n, max_valid_idx))
            
            # Simple extraction
            anchor_data, anchor_lab = _safe_extract_window(a, data, labels, W_total, W)
            positive_data, positive_lab = _safe_extract_window(p, data, labels, W_total, W)
            negative_data, negative_lab = _safe_extract_window(n, data, labels, W_total, W)
            
            if anchor_data.ndim == 1:
                anchor_data = anchor_data[:, None]
                positive_data = positive_data[:, None]
                negative_data = negative_data[:, None]
                
            return ((anchor_data.astype(np.float32), anchor_lab.astype(np.float32)),
                    (positive_data.astype(np.float32), positive_lab.astype(np.float32)),
                    (negative_data.astype(np.float32), negative_lab.astype(np.float32)))
    
    # ---- Memory-efficient batch generator ----------------------------------
    def _batch_generator():
        rng = np.random.default_rng()
        
        while True:
            try:
                # Sample triplets
                triplets = [_sample_triplet(rng) for _ in range(B)]
                
                # Extract and stack
                A_data, A_lab = zip(*[t[0] for t in triplets])
                P_data, P_lab = zip(*[t[1] for t in triplets])
                N_data, N_lab = zip(*[t[2] for t in triplets])
                
                # Stack efficiently - MAINTAIN TRIPLET ORDER [A, P, N]
                data_batch = np.concatenate([
                    np.stack(A_data),
                    np.stack(P_data),
                    np.stack(N_data)
                ], axis=0)
                
                labels_batch = np.concatenate([
                    np.stack(A_lab)[:, :, None],
                    np.stack(P_lab)[:, :, None],
                    np.stack(N_lab)[:, :, None]
                ], axis=0)
                
                # Final shape validation
                expected_data_shape = (3*B, W_total, C)
                expected_label_shape = (3*B, W, 1)
                
                assert data_batch.shape == expected_data_shape, f"Data batch shape: {data_batch.shape} vs {expected_data_shape}"
                assert labels_batch.shape == expected_label_shape, f"Label batch shape: {labels_batch.shape} vs {expected_label_shape}"
                
                yield (data_batch, labels_batch)
                
            except Exception as e:
                print(f"Error in batch generation: {e}")
                # Create a dummy batch to prevent pipeline failure
                dummy_data = np.zeros((3*B, W_total, C), dtype=np.float32)
                dummy_labels = np.zeros((3*B, W, 1), dtype=np.float32)
                yield (dummy_data, dummy_labels)
    
    # Create dataset - REMOVE SHUFFLE to preserve triplet structure
    spec = (
        tf.TensorSpec(shape=(3*B, W_total, C), dtype=tf.float32),
        tf.TensorSpec(shape=(3*B, W, 1), dtype=tf.float32),
    )
    
    triplet_ds = tf.data.Dataset.from_generator(_batch_generator, output_signature=spec)
    return triplet_ds

# Replace _create_simple_onset_weights function around line 267:
def _create_enhanced_onset_weights(labels, window_indices, win_len, multiplier=2, debug=False):
    """
    Enhanced onset weighting with better numerical stability and stronger onset focus
    """
    if len(window_indices) == 0:
        return np.array([])
    
    # Find onsets with better edge handling
    padded_labels = np.concatenate(([0], labels, [0]))
    onsets = np.where(np.diff(padded_labels) == 1)[0]
    
    if len(onsets) == 0:
        if debug:
            print("No onsets found for weighting")
        return np.ones(len(window_indices), dtype=np.float32)
    
    weights = np.ones(len(window_indices), dtype=np.float32)
    label_offset = (multiplier - 1) * win_len
    
    onset_window_count = 0
    early_onset_count = 0
    
    for i, start_idx in enumerate(window_indices):
        label_start = start_idx + label_offset
        label_end = start_idx + multiplier * win_len
        
        # Find onsets in label region
        onsets_in_window = onsets[(onsets >= label_start) & (onsets < label_end)]
        
        if len(onsets_in_window) > 0:
            onset_window_count += 1
            base_weight = 4.0  # Strong base onset weight
            
            # Early onset bonus (first 25% of window gets extra weight)
            early_region_end = label_start + win_len // 4
            early_onsets = np.sum((onsets_in_window >= label_start) & 
                                (onsets_in_window < early_region_end))
            
            if early_onsets > 0:
                weights[i] = base_weight * 2.5  # Very strong weight for early onsets
                early_onset_count += 1
            else:
                weights[i] = base_weight
            
            # Multiple onset bonus
            if len(onsets_in_window) > 1:
                weights[i] *= 1.2
        else:
            # Check for partial event coverage (good for learning boundaries)
            event_samples = np.sum(labels[label_start:label_end])
            event_ratio = event_samples / win_len
            
            if 0.1 < event_ratio < 0.7:  # Partial events
                weights[i] = 2.0  # Moderate weight for partial events
    
    # Normalize to prevent extreme values while maintaining bias
    max_weight = np.max(weights)
    if max_weight > 10.0:
        weights = weights * (10.0 / max_weight)
    
    # Ensure minimum weight
    weights = np.maximum(weights, 0.5)
    
    if debug:
        print(f"Enhanced onset weights: {onset_window_count}/{len(weights)} windows with onsets")
        print(f"Early onset windows: {early_onset_count}")
        print(f"Weight range: [{weights.min():.2f}, {weights.max():.2f}], mean: {weights.mean():.2f}")
    
    return weights

# -------------------------- Data wrangling ----------------------------------
def rippleAI_prepare_training_data(train_LFPs, train_GTs,
                                   val_LFPs,   val_GTs,
                                   sf=1250, new_sf=1250,
                                   channels=np.arange(0, 8),
                                   zscore=True, process_online=False, use_band=None):
    """
    Down-sample, band-filter, z-score and concatenate training sessions;
    return processed training LFP, concatenated GT, list of processed
    validation sessions and their GT.
    """
    assert len(train_LFPs) == len(train_GTs)
    assert len(val_LFPs)   == len(val_GTs)
    assert len(train_LFPs) + len(val_LFPs) == len(sf)

    retrain_LFP, retrain_GT = [], []
    offset        = 0.0     # seconds
    counter_sf    = 0

    # ---- training sessions concatenated ------------------------------------
    for LFP, GT in zip(train_LFPs, train_GTs):
        print('Original training data shape:', LFP.shape,
              '| sf:', sf[counter_sf])

        if process_online:
            LFP = LFP[::int(sf[counter_sf] // new_sf), :].astype(np.float32)
            print('Sub-sampled data shape:', LFP.shape,
                  '| new sf:', new_sf)
            import matplotlib.pyplot as plt
            aux = sliding_window_zscore(LFP, win=new_sf, eps=1e-8)
            print('Sliding-window z-scored data shape:', aux.shape, '| running window:', new_sf)
        else:
            aux = process_LFP(LFP, ch=channels, sf=sf[counter_sf],
                            new_sf=new_sf, use_zscore=False, use_band=use_band)
            if zscore:
                aux = (aux - aux.mean(0)) / aux.std(0)

        # shift events by offset (seconds accumulated so far)
        GT = GT + offset

        if len(retrain_LFP) == 0:
            retrain_LFP = aux
            retrain_GT  = GT
        else:
            retrain_LFP = np.vstack([retrain_LFP, aux])
            retrain_GT  = np.vstack([retrain_GT, GT])

        offset     += len(aux) / new_sf
        counter_sf += 1

    # ---- validation sessions processed separately --------------------------
    norm_val_LFP = []
    for LFP in val_LFPs:
        print('Original validation data shape:', LFP.shape,
              '| sf:', sf[counter_sf])

        tmp = process_LFP(LFP, ch=channels, sf=sf[counter_sf],
                          new_sf=new_sf, use_zscore=False, use_band=use_band)
        if zscore:
            tmp = (tmp - tmp.mean(0)) / tmp.std(0)

        norm_val_LFP.append(tmp)
        counter_sf += 1

    return retrain_LFP, retrain_GT, norm_val_LFP, val_GTs


# -------------------------- Top-level loader --------------------------------
def rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=False, use_band=None):
    """
    Build train / test tf.data.Datasets of triplets.
    """
    
    onset_params = {
        # Core onset detection parameters
        'USE_ONSET_WEIGHTING': False,
        'ONSET_BIAS': 0.4,                    # 50% bias toward onset windows
        'USE_ONSET_SHIFTED_LABELS': False,     # Enable label shifting
        'ONSET_SHIFT_SAMPLES': -8,            # Start positive labels 8 samples before onset
        'ONSET_EXTEND_SAMPLES': 8,           # Extend positive region 15 samples after onset
        
        # Training stability
        'JITTER': 8,                          # Reduced jitter to preserve timing
        'MIN_NEG_GAP': 150,                   # Larger negative gap
        'ANCHOR_THR': 0.85,                   # Higher anchor threshold
        'POS_THR': 0.65,                      # Higher positive threshold
        'GRACE': 25,                          # Larger grace period
        
        # Debug
        'DEBUG_ONSETS': False                  # Enable onset analysis
    }

    # Update your existing params:
    params.update(onset_params)    
    # ---- sample-shift parsing for causal architectures ----------------------
    if 'Shift' in params['TYPE_ARCH']:
        shift_ms    = float(params['TYPE_ARCH'].split('Shift')[1][:2])
        sample_shift = int(shift_ms / 1000 * params['SRATE'])
        print('Using Shift:', sample_shift, 'samples')
    else:
        sample_shift = 0

    # ---- gather raw sessions ------------------------------------------------
    train_LFPs, train_GTs, all_SFs = [], [], []
    if mode == 'train':
        for mouse, fig_id in [('Amigo2', '16847521'), ('Som2', '16856137')]:
            path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/',
                                'Downloaded_data', mouse, f'figshare_{fig_id}')
            LFP, GT = load_lab_data(path)
            train_LFPs.append(LFP)
            train_GTs.append(GT)
            all_SFs.append(30000)

    val_LFPs, val_GTs = [], []
    if mode == 'test':
        for mouse, fig_id in [('Dlx1', '14959449'),
                              ('Thy7', '14960085'),
                              ('Calb20', '20072252')]:
            path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/',
                                'Downloaded_data', mouse, f'figshare_{fig_id}')
            LFP, GT = load_lab_data(path)
            val_LFPs.append(LFP)
            val_GTs.append(GT)
            all_SFs.append(30000)

    # ---- preprocess ---------------------------------------------------------
    train_data, train_GT, val_data, val_GT = rippleAI_prepare_training_data(
        train_LFPs, train_GTs, val_LFPs, val_GTs,
        sf=all_SFs, new_sf=params['SRATE'],
        zscore=preprocess, process_online=process_online, use_band=use_band
    )

    if mode == 'test':
        print('Returning validation data only')
        val_data = [v.astype('float32') for v in val_data]
        return val_data, val_GT

    # ---- train / test split -------------------------------------------------
    train_data = train_data.astype('float32')
    te_x, ev_test, tr_x, ev_train = split_data(
        train_data, train_GT,
        sf=params['SRATE'], split=0.7
    )

    sf          = params['SRATE']
    train_lab   = np.zeros(len(tr_x), dtype=np.float32)
    for e in ev_train:
        train_lab[int(sf * e[0]):int(sf * e[1]) + sample_shift] = 1

    test_lab    = np.zeros(len(te_x), dtype=np.float32)
    for e in ev_test:
        test_lab[int(sf * e[0]):int(sf * e[1]) + sample_shift] = 1

    label_ratio = train_lab.sum() / len(train_lab)    
    # ---- triplet datasets ---------------------------------------------------
    # train_ds = create_triplet_dataset(tr_x, train_lab, params)
    # train_ds = (train_ds.prefetch(tf.data.AUTOTUNE))    
    train_seq = TripletSequence(tr_x, train_lab, params)   # Sequence
    train_ds = train_seq        # keeps API identical to caller
    
    # test set
    sample_length = params['NO_TIMEPOINTS']*2
    label_length = sample_length/2
    label_skip = int(sample_length/2)    
    stride_step = 8#params['STRIDE_STEP'] if 'STRIDE_STEP' in params else 32
    
    # Create standard test dataset with T-length windows
    test_x = timeseries_dataset_from_array(
        te_x,
        None,
        sequence_length=sample_length,  # T timepoints
        sequence_stride=stride_step,
        batch_size=None,
        shuffle=False
    )
    
    test_y = timeseries_dataset_from_array(
        test_lab[label_skip:].reshape(-1,1),
        None,
        sequence_length=label_length,  # T timepoints
        sequence_stride=stride_step,
        batch_size=None,
        shuffle=False
    )
    
    # Simple validation dataset - drop incomplete batches to avoid splitting errors
    test_ds = tf.data.Dataset.zip((test_x, test_y))
    test_ds = test_ds.batch(params['BATCH_SIZE']*6, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    # Calculate validation steps for complete batches only
    total_windows = (len(te_x) - sample_length) // stride_step + 1
    val_steps = total_windows // (params['BATCH_SIZE'] * 6)  # Integer division - drops partial batch
    
    print(f"Validation: {len(te_x)} samples, {total_windows} windows")
    print(f"Validation steps: {val_steps} (dropping partial batch)")

    dataset_params = params.copy()
    dataset_params['VAL_STEPS'] = val_steps
    dataset_params['ESTIMATED_STEPS_PER_EPOCH'] = len(train_seq)
    return train_ds, test_ds, label_ratio, dataset_params


# --------------------- Convenience loader for Allen -------------------------
def load_allen(indices=np.int32(np.linspace(49, 62, 8))):
    """
    Example helper that loads Allen data and processes it exactly like the
    training recordings (down-sample, z-score, etc.).
    """
    raw = np.load(
        '/cs/projects/OWVinckSWR/Carmen/LFP_extracted/sanity_check/raw_lfp_fc.npy'
    )
    indices[::-1].sort()
    LFP = raw[:, indices]
    return process_LFP(LFP, sf=1250, channels=np.arange(0, 8))

def _enhance_anchor_sampling_for_long_events(labels, anchor_idx, win_len, multiplier=2):
    """
    Enhance anchor sampling to better handle events longer than the window.

    For events that are longer than the window, we want to sample different
    parts of the event to provide better training diversity.

    Args:
        labels: Binary label array
        anchor_idx: Original anchor indices
        win_len: Label window length (T)
        multiplier: Window multiplier for input size (nT)

    Returns:
        Enhanced anchor indices with better coverage of long events
    """
    total_win_len = multiplier * win_len

    # Find continuous event regions
    event_starts = np.where(np.diff(np.concatenate(([0], labels, [0]))) == 1)[0]
    event_ends = np.where(np.diff(np.concatenate(([0], labels, [0]))) == -1)[0]

    enhanced_anchors = list(anchor_idx)  # Start with original anchors

    for start, end in zip(event_starts, event_ends):
        event_length = end - start

        # For events significantly longer than window, add strategic positions
        if event_length > win_len * 1.5:  # Event is 1.5x longer than label window
            # Calculate positions that would place different parts of the event
            # in the label region (last win_len samples of the nT window)

            # Position 1: Event start in label region
            pos1 = max(0, start - (multiplier - 1) * win_len)

            # Position 2: Event middle in label region
            event_mid = (start + end) // 2
            pos2 = max(0, event_mid - (multiplier - 1) * win_len - win_len // 2)

            # Position 3: Event end in label region
            pos3 = max(0, end - total_win_len)

            # Add positions if they create valid windows and aren't already covered
            for pos in [pos1, pos2, pos3]:
                if (pos >= 0 and
                    pos <= len(labels) - total_win_len and
                    pos not in enhanced_anchors):

                    # Verify this position actually has event overlap in label region
                    label_start = pos + (multiplier - 1) * win_len
                    label_end = pos + total_win_len
                    if np.sum(labels[label_start:label_end]) > 0:
                        enhanced_anchors.append(pos)

    return np.unique(enhanced_anchors)

def _analyze_event_onsets(labels, window_len, debug=False):
    """
    Analyze event onsets and their distribution to understand onset characteristics.
    Returns onset positions and statistics for onset-focused sampling.
    """
    # Find event onsets and offsets
    diff_labels = np.diff(np.r_[0, labels, 0])
    onsets = np.flatnonzero(diff_labels == 1)
    offsets = np.flatnonzero(diff_labels == -1)

    if len(onsets) == 0:
        if debug:
            print("No events found in labels")
        return onsets, {}

    # Calculate event durations
    if len(offsets) >= len(onsets):
        event_durations = offsets[:len(onsets)] - onsets
    else:
        # Handle case where recording ends during an event
        event_durations = np.concatenate([
            offsets - onsets[:len(offsets)],
            [len(labels) - onsets[len(offsets):]]
        ]).flatten()

    # Onset statistics
    stats = {
        'n_events': len(onsets),
        'mean_duration': np.mean(event_durations) if len(event_durations) > 0 else 0,
        'median_duration': np.median(event_durations) if len(event_durations) > 0 else 0,
        'min_duration': np.min(event_durations) if len(event_durations) > 0 else 0,
        'max_duration': np.max(event_durations) if len(event_durations) > 0 else 0,
        'window_len': window_len,
        'onset_coverage_ratio': len(onsets) / (len(labels) // window_len) if len(labels) > 0 else 0
    }

    if debug:
        print(f"\nOnset Analysis:")
        print(f"  Events found: {stats['n_events']}")
        print(f"  Duration stats: mean={stats['mean_duration']:.1f}, median={stats['median_duration']:.1f}")
        print(f"  Duration range: [{stats['min_duration']}, {stats['max_duration']}]")
        print(f"  Onset coverage: {stats['onset_coverage_ratio']:.3f} onsets per window")

        # Check how many onsets would be captured with different window positions
        onset_in_window_count = 0
        total_possible_windows = max(1, len(labels) - window_len)
        for i in range(0, len(labels) - window_len + 1, window_len // 4):  # Sample every quarter window
            window_onsets = np.sum((onsets >= i) & (onsets < i + window_len))
            if window_onsets > 0:
                onset_in_window_count += 1

        print(f"  Windows with onsets: {onset_in_window_count}/{total_possible_windows // (window_len//4)} sampled windows")

    return onsets, stats


def _create_onset_focused_catalogue(labels, win_len, grace, anchor_thr, pos_thr,
                                   min_neg_gap, multiplier=2, onset_bias=0.3, debug=False):
    """
    Enhanced window catalogue that specifically biases toward event onsets.

    Args:
        onset_bias: Float [0,1] - fraction of windows that should preferentially include onsets
                   0.0 = no bias, 1.0 = maximum bias toward onsets
    """
    # Get standard catalogue first
    anchor_idx, positive_idx, negative_idx = _build_window_catalogue_2T(
        labels, win_len, grace, anchor_thr, pos_thr, min_neg_gap, multiplier
    )

    if onset_bias <= 0:
        return anchor_idx, positive_idx, negative_idx

    # Analyze onsets
    onsets, stats = _analyze_event_onsets(labels, win_len, debug=debug)

    if len(onsets) == 0:
        if debug:
            print("No onsets found - using standard catalogue")
        return anchor_idx, positive_idx, negative_idx

    # Calculate onset regions for biased sampling
    # For causal TCNs, we want onsets to appear in the LABEL region (last T samples)
    # So for a window starting at position i, the label region is [i+(multiplier-1)*win_len : i+multiplier*win_len]
    # We want onsets to fall in this region

    total_win_len = multiplier * win_len
    label_offset = (multiplier - 1) * win_len
    max_start_idx = len(labels) - total_win_len

    if max_start_idx < 0:
        return anchor_idx, positive_idx, negative_idx

    # Find windows where onsets fall in the label region
    onset_focused_anchors = []
    onset_focused_positives = []

    for start_idx in range(max_start_idx + 1):
        label_start = start_idx + label_offset
        label_end = start_idx + total_win_len

        # Count onsets in the label region
        onsets_in_label = np.sum((onsets >= label_start) & (onsets < label_end))

        if onsets_in_label > 0:
            # Calculate overlap for classification
            label_region = labels[label_start:label_end]
            overlap_ratio = np.sum(label_region) / win_len

            if overlap_ratio >= anchor_thr:
                onset_focused_anchors.append(start_idx)
            elif overlap_ratio >= pos_thr:
                onset_focused_positives.append(start_idx)

    onset_focused_anchors = np.array(onset_focused_anchors)
    onset_focused_positives = np.array(onset_focused_positives)

    if debug:
        print(f"\nOnset-focused sampling:")
        print(f"  Original anchors: {len(anchor_idx)}, onset-focused: {len(onset_focused_anchors)}")
        print(f"  Original positives: {len(positive_idx)}, onset-focused: {len(onset_focused_positives)}")
        print(f"  Onset bias factor: {onset_bias}")

    # Blend original catalogue with onset-focused catalogue
    n_onset_anchors = int(len(anchor_idx) * onset_bias)
    n_onset_positives = int(len(positive_idx) * onset_bias)

    # Combine catalogues with bias toward onset-focused windows
    if len(onset_focused_anchors) > 0 and n_onset_anchors > 0:
        # Sample onset-focused anchors (with replacement if needed)
        onset_anchor_sample = np.random.choice(
            onset_focused_anchors,
            size=min(n_onset_anchors, len(onset_focused_anchors) * 3),  # Allow some oversampling
            replace=len(onset_focused_anchors) < n_onset_anchors
        )
        # Sample remaining from original catalogue
        remaining_anchors = anchor_idx[~np.isin(anchor_idx, onset_focused_anchors)]
        if len(remaining_anchors) > 0:
            n_remaining_anchors = max(0, len(anchor_idx) - len(onset_anchor_sample))
            remaining_anchor_sample = np.random.choice(
                remaining_anchors,
                size=min(n_remaining_anchors, len(remaining_anchors)),
                replace=False
            )
            final_anchor_idx = np.concatenate([onset_anchor_sample, remaining_anchor_sample])
        else:
            final_anchor_idx = onset_anchor_sample
    else:
        final_anchor_idx = anchor_idx

    # Same for positives
    if len(onset_focused_positives) > 0 and n_onset_positives > 0:
        onset_pos_sample = np.random.choice(
            onset_focused_positives,
            size=min(n_onset_positives, len(onset_focused_positives) * 3),
            replace=len(onset_focused_positives) < n_onset_positives
        )
        remaining_positives = positive_idx[~np.isin(positive_idx, onset_focused_positives)]
        if len(remaining_positives) > 0:
            n_remaining_positives = max(0, len(positive_idx) - len(onset_pos_sample))
            remaining_pos_sample = np.random.choice(
                remaining_positives,
                size=min(n_remaining_positives, len(remaining_positives)),
                replace=False
            )
            final_positive_idx = np.concatenate([onset_pos_sample, remaining_pos_sample])
        else:
            final_positive_idx = onset_pos_sample
    else:
        final_positive_idx = positive_idx

    # Ensure we have unique indices
    final_anchor_idx = np.unique(final_anchor_idx)
    final_positive_idx = np.unique(final_positive_idx)

    # Add this function after the weight creation:
    def analyze_onset_coverage(anchor_idx, pos_idx, labels, win_len, multiplier=2):
        """
        Analyze how well the sampling covers event onsets
        """
        onsets = np.where(np.diff(np.concatenate(([0], labels, [0]))) == 1)[0]
        
        if len(onsets) == 0:
            print("No onsets found for coverage analysis")
            return
        
        label_offset = (multiplier - 1) * win_len
        
        # Check onset coverage
        anchor_onset_coverage = 0
        for idx in anchor_idx:
            label_start = idx + label_offset
            label_end = idx + multiplier * win_len
            if np.any((onsets >= label_start) & (onsets < label_end)):
                anchor_onset_coverage += 1
        
        pos_onset_coverage = 0
        for idx in pos_idx:
            label_start = idx + label_offset
            label_end = idx + multiplier * win_len
            if np.any((onsets >= label_start) & (onsets < label_end)):
                pos_onset_coverage += 1
        
        print(f"\nOnset Coverage Analysis:")
        print(f"  Total onsets: {len(onsets)}")
        print(f"  Anchor windows with onsets: {anchor_onset_coverage}/{len(anchor_idx)} "
            f"({100*anchor_onset_coverage/len(anchor_idx):.1f}%)")
        print(f"  Positive windows with onsets: {pos_onset_coverage}/{len(pos_idx)} "
            f"({100*pos_onset_coverage/len(pos_idx):.1f}%)")

    # Add this call after building catalogues (around line 175):
    if params.get('DEBUG_ONSETS', False):
        analyze_onset_coverage(anchor_idx, pos_idx, labels, W, window_multiplier)
        
    if debug:
        print(f"  Final anchors: {len(final_anchor_idx)}, positives: {len(final_positive_idx)}")

        # Analyze onset coverage in final catalogue
        onset_coverage_anchors = 0
        onset_coverage_positives = 0

        for idx in final_anchor_idx:
            label_start = idx + label_offset
            label_end = idx + total_win_len
            if np.sum((onsets >= label_start) & (onsets < label_end)) > 0:
                onset_coverage_anchors += 1

        for idx in final_positive_idx:
            label_start = idx + label_offset
            label_end = idx + total_win_len
            if np.sum((onsets >= label_start) & (onsets < label_end)) > 0:
                onset_coverage_positives += 1

        anchor_onset_ratio = onset_coverage_anchors / len(final_anchor_idx) if len(final_anchor_idx) > 0 else 0
        pos_onset_ratio = onset_coverage_positives / len(final_positive_idx) if len(final_positive_idx) > 0 else 0

        print(f"  Onset coverage: {anchor_onset_ratio:.2f} anchors, {pos_onset_ratio:.2f} positives")

    return final_anchor_idx, final_positive_idx, negative_idx


def _enhance_onset_sampling_weights(labels, window_indices, win_len, multiplier=2,
                                   onset_weight=3.0, early_weight=2.0):
    """
    Create sampling weights that favor windows containing event onsets,
    with extra weight for early onset detection.

    Args:
        labels: Binary label array
        window_indices: Array of window start indices
        win_len: Window length (T)
        multiplier: Window multiplier for total length
        onset_weight: Weight multiplier for windows containing onsets
        early_weight: Additional weight for windows where onset occurs early in label region

    Returns:
        weights: Sampling weights for each window index
    """
    if len(window_indices) == 0:
        return np.array([])

    # Find onsets
    onsets = np.flatnonzero(np.diff(np.r_[0, labels]) == 1)

    if len(onsets) == 0:
        return np.ones(len(window_indices))  # Uniform weights if no onsets

    weights = np.ones(len(window_indices))
    label_offset = (multiplier - 1) * win_len
    total_win_len = multiplier * win_len

    for i, start_idx in enumerate(window_indices):
        label_start = start_idx + label_offset
        label_end = start_idx + total_win_len

        # Find onsets in this window's label region
        window_onsets = onsets[(onsets >= label_start) & (onsets < label_end)]

        if len(window_onsets) > 0:
            # Base weight for containing onsets
            weights[i] *= onset_weight

            # Additional weight for early onsets (first quarter of label region)
            early_region_end = label_start + win_len // 4
            early_onsets = np.sum((window_onsets >= label_start) & (window_onsets < early_region_end))

            if early_onsets > 0:
                weights[i] *= early_weight

            # Slight additional weight for multiple onsets
            if len(window_onsets) > 1:
                weights[i] *= 1.2

    # Normalize weights to prevent extreme values
    weights = weights / np.max(weights)

    return weights


def _create_onset_weighted_labels(labels, onset_weight=5.0, onset_window=10,
                                decay_factor=0.8, debug=False):
    """
    Create onset-weighted labels that emphasize event onsets for early detection training.
    This can be used as sample weights or as modified target labels.

    Args:
        labels: Binary label array
        onset_weight: Weight multiplier for onset samples
        onset_window: Number of samples after onset to apply enhanced weighting
        decay_factor: How quickly weight decays after onset (0-1)
        debug: Whether to print debug information

    Returns:
        weights: Array of same length as labels with onset-focused weights
    """
    if len(labels) == 0:
        return np.array([])

    weights = np.ones_like(labels, dtype=np.float32)

    # Find event onsets
    onsets = np.flatnonzero(np.diff(np.r_[0, labels]) == 1)

    if len(onsets) == 0:
        if debug:
            print("No onsets found for weighting")
        return weights

    # Apply enhanced weighting around onsets
    for onset in onsets:
        end_idx = min(len(labels), onset + onset_window)

        # Create decaying weight profile
        for i in range(onset, end_idx):
            decay_steps = i - onset
            current_weight = onset_weight * (decay_factor ** decay_steps)
            weights[i] = max(weights[i], current_weight)

    # Also weight the actual event samples (but less than onsets)
    event_indices = np.where(labels > 0)[0]
    weights[event_indices] = np.maximum(weights[event_indices], 2.0)

    if debug:
        print(f"Onset weighting: {len(onsets)} onsets found")
        print(f"Weight statistics: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
        high_weight_samples = np.sum(weights > 2.0)
        print(f"High-weight samples: {high_weight_samples}/{len(weights)} ({100*high_weight_samples/len(weights):.1f}%)")

    return weights


def _apply_onset_focused_augmentation_hook(data, labels, rng, augment_params=None):
    """
    Hook for onset-focused data augmentation. This can be extended with various
    augmentation techniques that preserve or enhance onset detectability.

    Currently includes:
    - Onset-preserving noise addition
    - Onset-focused temporal shifts
    - Future: spectral augmentations, etc.

    Args:
        data: Input data array (T, C) or (T,)
        labels: Label array (T,)
        rng: Random number generator
        augment_params: Dictionary of augmentation parameters

    Returns:
        augmented_data: Augmented data
        augmented_labels: Corresponding labels (may include onset weighting)
    """
    if augment_params is None:
        return data, labels

    augmented_data = data.copy()
    augmented_labels = labels.copy()

    # Onset-preserving noise addition
    if augment_params.get('onset_noise', False):
        noise_std = augment_params.get('noise_std', 0.02)

        # Find onset regions to apply reduced noise
        onsets = np.flatnonzero(np.diff(np.r_[0, labels]) == 1)
        onset_mask = np.zeros_like(labels, dtype=bool)

        onset_protect_window = augment_params.get('onset_protect_window', 10)
        for onset in onsets:
            start_idx = max(0, onset - onset_protect_window // 2)
            end_idx = min(len(labels), onset + onset_protect_window // 2)
            onset_mask[start_idx:end_idx] = True

        # Apply reduced noise near onsets, full noise elsewhere
        noise = rng.normal(0, noise_std, augmented_data.shape)
        noise[onset_mask] *= 0.3  # Reduced noise near onsets

        augmented_data = augmented_data + noise

    # Onset-focused temporal jittering (micro-shifts that preserve onset timing)
    if augment_params.get('onset_jitter', False):
        max_jitter = augment_params.get('max_jitter', 2)  # Very small shifts

        # Apply small random shifts while preserving onset structure
        jitter = rng.integers(-max_jitter, max_jitter + 1)
        if jitter != 0:
            if jitter > 0:
                augmented_data = np.pad(augmented_data, ((jitter, 0), (0, 0)) if augmented_data.ndim > 1
                                      else (jitter, 0), mode='edge')[:-jitter]
                augmented_labels = np.pad(augmented_labels, (jitter, 0), mode='edge')[:-jitter]
            else:
                augmented_data = np.pad(augmented_data, ((0, -jitter), (0, 0)) if augmented_data.ndim > 1
                                      else (0, -jitter), mode='edge')[-jitter:]
                augmented_labels = np.pad(augmented_labels, (0, -jitter), mode='edge')[-jitter:]

    return augmented_data, augmented_labels


# Add this function after the enhanced weights function:
def _create_onset_shifted_labels(original_labels, shift_samples=-8, extend_samples=15):
    """
    Create labels that encourage earlier detection by extending positive regions backward
    """
    shifted_labels = original_labels.copy()
    
    # Find onsets
    onsets = np.where(np.diff(np.concatenate(([0], original_labels, [0]))) == 1)[0]
    
    if len(onsets) == 0:
        return shifted_labels
    
    for onset in onsets:
        # Extend positive region backward from onset
        new_start = max(0, onset + shift_samples)  # shift_samples is negative
        new_end = min(len(original_labels), onset + extend_samples)
        
        # Only extend if the region before onset is currently negative
        if new_start < onset and np.sum(original_labels[new_start:onset]) == 0:
            shifted_labels[new_start:new_end] = 1
    
    return shifted_labels

def _create_pre_onset_labels(labels, pre_onset_window=15, onset_ratio=0.3, debug=False):
    """
    Create additional labels for pre-onset detection. This creates a secondary
    label channel that marks regions immediately before event onsets.

    Args:
        labels: Binary label array
        pre_onset_window: How many samples before onset to mark as pre-onset
        onset_ratio: What fraction of pre-onset window to label as positive
        debug: Whether to print debug information

    Returns:
        pre_onset_labels: Binary array marking pre-onset regions
    """
    if len(labels) == 0:
        return np.zeros_like(labels)

    pre_onset_labels = np.zeros_like(labels)

    # Find event onsets
    onsets = np.flatnonzero(np.diff(np.r_[0, labels]) == 1)

    if len(onsets) == 0:
        if debug:
            print("No onsets found for pre-onset labeling")
        return pre_onset_labels

    # Mark pre-onset regions
    pre_onset_samples = int(pre_onset_window * onset_ratio)

    for onset in onsets:
        # Mark region immediately before onset
        start_idx = max(0, onset - pre_onset_samples)
        end_idx = onset

        # Only mark if there's no existing event in this region
        if np.sum(labels[start_idx:end_idx]) == 0:
            pre_onset_labels[start_idx:end_idx] = 1

    if debug:
        pre_onset_positive = np.sum(pre_onset_labels)
        print(f"Pre-onset labeling: {len(onsets)} onsets, "
              f"pre-onset positive samples: {pre_onset_positive}")

    return pre_onset_labels


def analyze_onset_detection_performance(predictions, true_labels, onset_tolerance=10,
                                      debug=True, sampling_rate=1250):
    """
    Analyze how well the model detects event onsets vs peaks.

    Args:
        predictions: Model predictions (T,) or (T, 1)
        true_labels: True binary labels (T,) or (T, 1)
        onset_tolerance: Tolerance window around onset for "early" detection
        debug: Whether to print detailed analysis
        sampling_rate: Sampling rate for time conversion

    Returns:
        dict: Performance metrics focused on onset detection
    """
    if predictions.ndim > 1:
        predictions = predictions.squeeze()
    if true_labels.ndim > 1:
        true_labels = true_labels.squeeze()

    # Find true onsets and event centers
    true_onsets = np.flatnonzero(np.diff(np.r_[0, true_labels]) == 1)
    true_offsets = np.flatnonzero(np.diff(np.r_[0, true_labels]) == -1)

    if len(true_onsets) == 0:
        if debug:
            print("No events found in true labels")
        return {'n_events': 0}

    # Calculate event centers
    if len(true_offsets) >= len(true_onsets):
        event_centers = (true_onsets + true_offsets[:len(true_onsets)]) // 2
    else:
        # Handle case where recording ends during event
        full_centers = (true_onsets[:len(true_offsets)] + true_offsets) // 2
        partial_centers = true_onsets[len(true_offsets):] + 10  # Estimate for incomplete events
        event_centers = np.concatenate([full_centers, partial_centers])

    # Find predicted onsets (using a threshold)
    threshold = 0.5
    pred_binary = (predictions > threshold).astype(int)
    pred_onsets = np.flatnonzero(np.diff(np.r_[0, pred_binary]) == 1)

    # Analyze detection timing
    onset_detections = 0
    peak_detections = 0
    early_detections = 0
    missed_detections = 0

    detection_delays = []

    for i, (true_onset, event_center) in enumerate(zip(true_onsets, event_centers)):
        # Find nearest predicted onset
        if len(pred_onsets) > 0:
            delays = pred_onsets - true_onset

            # Find predictions within reasonable range
            valid_preds = pred_onsets[np.abs(delays) <= onset_tolerance * 2]

            if len(valid_preds) > 0:
                # Take the earliest valid prediction
                earliest_pred = valid_preds[np.argmin(valid_preds - true_onset)]
                delay = earliest_pred - true_onset
                detection_delays.append(delay)

                if delay <= -onset_tolerance // 2:
                    early_detections += 1
                elif delay <= onset_tolerance:
                    onset_detections += 1
                else:
                    peak_detections += 1
            else:
                missed_detections += 1
        else:
            missed_detections += 1

    # Calculate metrics
    total_events = len(true_onsets)
    detected_events = total_events - missed_detections

    results = {
        'n_events': total_events,
        'n_detected': detected_events,
        'detection_rate': detected_events / total_events if total_events > 0 else 0,
        'onset_detections': onset_detections,
        'peak_detections': peak_detections,
        'early_detections': early_detections,
        'missed_detections': missed_detections,
        'onset_detection_rate': onset_detections / total_events if total_events > 0 else 0,
        'early_detection_rate': early_detections / total_events if total_events > 0 else 0,
        'mean_detection_delay': np.mean(detection_delays) if detection_delays else None,
        'median_detection_delay': np.median(detection_delays) if detection_delays else None,
        'detection_delays': detection_delays
    }

    if debug:
        print(f"\nOnset Detection Analysis:")
        print(f"  Total events: {total_events}")
        print(f"  Detected: {detected_events}/{total_events} ({100*results['detection_rate']:.1f}%)")
        print(f"  Early detections: {early_detections} ({100*results['early_detection_rate']:.1f}%)")
        print(f"  Onset detections: {onset_detections} ({100*results['onset_detection_rate']:.1f}%)")
        print(f"  Peak detections: {peak_detections}")
        print(f"  Missed: {missed_detections}")

        if detection_delays:
            delay_ms = np.array(detection_delays) * 1000 / sampling_rate
            print(f"  Detection delay: {np.mean(delay_ms):.1f}±{np.std(delay_ms):.1f}ms")
            print(f"  Delay range: [{np.min(delay_ms):.1f}, {np.max(delay_ms):.1f}]ms")

    return results


# -------------------------------------------------------
# TripletSequence with its own _sample_triplet ----------
# -------------------------------------------------------

class TripletSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence that:
      1) Rebuilds the negative/positive/anchor catalog each epoch
      2) Samples triplets on the fly (with jitter, onset-shift, etc.)
    """
    def __init__(self, data, labels, params):
        self.data   = data
        self.labels = labels
        self.p      = params
        self.B      = params['BATCH_SIZE']
        self.rng    = np.random.default_rng()
        self._rebuild_catalogue()

    def _safe_extract_window(self, idx, data, labels, win_total, win_label):
        # exactly as in your snippet
        if idx < 0 or idx + win_total > len(data):
            idx = max(0, min(idx, len(data) - win_total))
        dw = data[idx:idx+win_total]
        offset = win_total - win_label
        lw = labels[idx+offset:idx+win_total]
        return dw, lw

    def _sample_triplet(self):
        # unpack parameters
        W      = self.p['NO_TIMEPOINTS']
        mul    = self.p.get('WINDOW_MULTIPLIER', 2)
        Wtot   = W * mul
        jitter = self.p.get('JITTER', 8)
        # choose indices
        a = int(self.rng.choice(self.anchor_idx))
        p = int(self.rng.choice(self.pos_idx))
        n = int(self.rng.choice(self.neg_idx))
        # apply jitter
        if jitter > 0:
            j = lambda i: max(0, min(i + self.rng.integers(-jitter, jitter+1),
                                      len(self.data)-Wtot))
            a, p, n = j(a), j(p), j(n)
        # extract windows
        Ad, Al = self._safe_extract_window(a, self.data, self.labels, Wtot, W)
        Pd, Pl = self._safe_extract_window(p, self.data, self.labels, Wtot, W)
        Nd, Nl = self._safe_extract_window(n, self.data, self.labels, Wtot, W)
        # optional onset shift
        if self.p.get('USE_ONSET_SHIFTED_LABELS', False):
            shift, extend = self.p['ONSET_SHIFT_SAMPLES'], self.p['ONSET_EXTEND_SAMPLES']
            Al = _create_onset_shifted_labels(Al, shift, extend)
            Pl = _create_onset_shifted_labels(Pl, shift, extend)
        # ensure channel dim
        for arr in (Ad, Pd, Nd):
            if arr.ndim == 1:
                arr.resize((arr.shape[0], 1))
        return (Ad.astype(np.float32), Al.astype(np.float32)), \
               (Pd.astype(np.float32), Pl.astype(np.float32)), \
               (Nd.astype(np.float32), Nl.astype(np.float32))

    def _rebuild_catalogue(self):
        tf.print("Rebuilding triplet catalogue...")
        W   = self.p['NO_TIMEPOINTS']
        mul = self.p.get('WINDOW_MULTIPLIER', 2)
        self.anchor_idx, self.pos_idx, self.neg_idx = _build_window_catalogue_2T(
            self.labels, W,
            self.p['GRACE'], self.p['ANCHOR_THR'],
            self.p['POS_THR'], self.p['MIN_NEG_GAP'],
            multiplier=mul
        )
        self.steps = max(1, len(self.anchor_idx) // self.B)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        triplets = [self._sample_triplet() for _ in range(self.B)]
        A, P, N = zip(*triplets)
        Ad, Al = zip(*A)
        Pd, Pl = zip(*P)
        Nd, Nl = zip(*N)
        # stack in [A, P, N] order
        data_batch = np.concatenate([np.stack(Ad),
                                     np.stack(Pd),
                                     np.stack(Nd)], axis=0)
        labels_batch = np.concatenate([np.stack(Al)[:, :, None],
                                       np.stack(Pl)[:, :, None],
                                       np.stack(Nl)[:, :, None]], axis=0)
        return data_batch, labels_batch

    def on_epoch_end(self):
        # triggers a fresh negative sample catalog
        self._rebuild_catalogue()

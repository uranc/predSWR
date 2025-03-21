import os
import pdb
import random  # Add missing import
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.utils import timeseries_dataset_from_array
from model.cnn_ripple_utils import load_lab_data, process_LFP, filter_LFP, split_data, load_info, load_raw_data



def random_scaling(data, min_scale=0.8, max_scale=1.2):
    """
    Randomly scale the entire batch by a single scaling factor.

    Args:
        data: Input tensor of shape [batch_size, num_samples, num_channels].
        min_scale: Minimum scaling factor.
        max_scale: Maximum scaling factor.

    Returns:
        Scaled data with one scale factor applied uniformly across all channels for each batch.
    """
    # Generate a single random scaling factor per batch
    # scale_factor = tf.random.uniform([], min_scale, max_scale)
    scale_factors = tf.random.uniform((1, 1, 8), min_scale, max_scale)

    # Apply the scaling factor uniformly to all channels and samples within the batch
    return data * scale_factors

def random_shift(data, max_shift_ms=2, sampling_rate=1250):
    """Random temporal shift for each channel, applied per batch and converted from milliseconds to samples."""
    batch_size = tf.shape(data)[0]
    num_samples = tf.shape(data)[1]  # Time points
    num_channels = tf.shape(data)[2]

    # Convert max_shift from milliseconds to samples
    max_shift_samples = tf.cast((max_shift_ms / 1000.0) * tf.cast(sampling_rate, tf.float32), tf.int32)

    # Generate a different random shift for each channel
    shift_amounts = tf.random.uniform([batch_size, num_channels], -max_shift_samples, max_shift_samples, dtype=tf.int32)

    # Apply the shift to each channel in the data
    def shift_single_channel(args):
        data_batch, shift_per_channel = args
        shifted_data = tf.map_fn(lambda shift: tf.roll(data_batch, shift, axis=0), shift_per_channel, dtype=tf.float32)
        return shifted_data

    # Apply the channel-wise shifting across the whole batch
    data = tf.map_fn(shift_single_channel, (data, shift_amounts), fn_output_signature=tf.float32)

    return data

def add_varying_noise(data, noise_factor_range=(0.01, 0.1)):
    """
    Add varying levels of Gaussian noise per channel, scaled relative to the amplitude or variance of each channel.

    Args:
        data: Input tensor of shape [batch_size, num_samples, num_channels].
        noise_factor_range: Tuple (min, max) representing the range of noise factors to apply.

    Returns:
        data with varying noise added per channel.
    """
    batch_size = tf.shape(data)[0]
    num_channels = tf.shape(data)[2]

    # Compute the variance per channel in the batch
    channel_variance = tf.math.reduce_variance(data, axis=[0, 1], keepdims=True)  # Shape: [1, 1, num_channels]

    # Generate random noise factors for each channel
    noise_factors = tf.random.uniform([batch_size, num_channels], noise_factor_range[0], noise_factor_range[1])

    # Reshape noise factors to broadcast over time points
    noise_factors = tf.reshape(noise_factors, [batch_size, 1, num_channels])

    # Generate Gaussian noise, scaled by the variance of each channel
    noise = tf.random.normal(tf.shape(data), mean=0.0, stddev=1.0)

    # Scale the noise relative to the variance of each channel and noise factors
    scaled_noise = noise * noise_factors * tf.sqrt(channel_variance)

    return data + scaled_noise

def add_optimized_burst_noise(data, burst_prob=0.1, burst_duration_range=(1, 5), burst_frequency_range=(80, 500), max_amplitude=0.7, sampling_rate=1250, params=None):
    """
    Add optimized burst noise to simulate artifacts per channel, ensuring burst noise is applied independently for each channel.

    Args:
        data: Input tensor of shape [batch_size, num_samples, num_channels].
        burst_prob: Probability of a burst occurring at any time point.
        burst_duration_range: Range of burst durations in milliseconds.
        burst_frequency_range: Frequency range for the burst noise.
        max_amplitude: Maximum amplitude of burst noise.
        sampling_rate: Sampling rate of the signal (samples per second).

    Returns:
        Data with burst noise applied per channel.
    """
    batch_size = tf.shape(data)[0]
    num_samples = params['NO_TIMEPOINTS']*2
    num_channels = tf.shape(data)[2]

    # import pdb
    # pdb.set_trace()
    # Determine burst occurrence locations per channel
    burst_indices = tf.where(tf.random.uniform([batch_size, num_samples, num_channels]) < burst_prob)

    def apply_burst_noise(index):
        burst_length = tf.random.uniform([], minval=burst_duration_range[0], maxval=burst_duration_range[1], dtype=tf.int32)
        burst_frequency = tf.random.uniform([], minval=burst_frequency_range[0], maxval=burst_frequency_range[1])

        # Generate burst noise
        t = tf.range(burst_length, dtype=tf.float32) / sampling_rate
        burst_noise = tf.sin(2 * np.pi * burst_frequency * t) * tf.random.uniform([], 0.5, max_amplitude)

        # Reshape burst noise to apply it across the channel
        burst_noise = tf.tile(burst_noise[:, tf.newaxis], [1, num_channels])
        start_idx = index[1]

        # Ensure burst noise fits within the time series
        condition = tf.cast(start_idx, tf.int64) + tf.cast(burst_length, tf.int64) <= num_samples

        def scatter_burst():
            # Cast start_idx and burst_length to int64
            start_idx_int64 = tf.cast(start_idx, tf.int64)
            burst_length_int64 = tf.cast(burst_length, tf.int64)

            # Ensure burst noise fits within the time series
            indices = tf.expand_dims(tf.range(start_idx_int64, start_idx_int64 + burst_length_int64), axis=1)
            indices = tf.tile(indices, [1, num_channels])
            return tf.tensor_scatter_nd_add(data, indices, burst_noise)
        return tf.cond(condition, scatter_burst, lambda: data)

    return tf.map_fn(apply_burst_noise, burst_indices, fn_output_signature=tf.float32)

def add_pink_noise(data, noise_factor=0.1):
    """
    Add pink noise (low-frequency noise) to simulate slow drifts in the signal.

    Args:
    - data: The input data of shape [batch_size, num_samples, num_channels].
    - noise_factor: Scaling factor for the noise amplitude.

    Returns:
    - Data with pink noise added.
    """
    batch_size = tf.shape(data)[0]  # batch size
    num_samples = tf.shape(data)[1]  # time points (n_timepoints)
    num_channels = tf.shape(data)[2]  # number of channels
    num_generators = 16  # Number of generators used to create pink noise

    # Initial state for the pink noise generators
    state = tf.zeros([num_channels, num_generators], dtype=tf.float32)

    # Function to generate pink noise for a single time step
    def generate_pink_noise(i, state):
        random_values = tf.random.normal([num_channels, num_generators], dtype=tf.float32)
        state = state + random_values
        pink_noise = tf.reduce_mean(state, axis=-1)  # Average over generators for each channel
        return pink_noise, state

    # Generate pink noise over all time points
    def loop_body(i, pink_noise_accum, state):
        pink_noise, state = generate_pink_noise(i, state)
        pink_noise_accum = pink_noise_accum.write(i, pink_noise)  # Accumulate the noise over time steps
        return i + 1, pink_noise_accum, state

    pink_noise_accum = tf.TensorArray(dtype=tf.float32, size=num_samples)
    i0 = tf.constant(0)
    _, pink_noise_accum, _ = tf.while_loop(
        lambda i, *_: i < num_samples,
        loop_body,
        [i0, pink_noise_accum, state]
    )

    # Stack the accumulated pink noise across time steps
    pink_noise = pink_noise_accum.stack()  # Shape: [num_samples, num_channels]

    # Expand dimensions to match the input data shape
    pink_noise = tf.expand_dims(pink_noise, axis=0)  # Shape: [1, num_samples, num_channels]
    pink_noise = tf.tile(pink_noise, [batch_size, 1, 1])  # Tile across the batch dimension

    # Scale the noise and add it to the data
    pink_noise *= noise_factor
    return data + pink_noise

def random_channel_shuffle(data):
    """Shuffle channels to simulate random reordering of input data."""
    permuted = tf.random.shuffle(tf.range(tf.shape(data)[-1]))
    return tf.gather(data, permuted, axis=-1)

def random_channel_dropout(data, dropout_prob=0.1):
    """Randomly drop channels to simulate missing data."""
    mask = tf.cast(tf.random.uniform([tf.shape(data)[-1]]) > dropout_prob, tf.float32)
    return data * mask

def replace_channels_with_noise(data, noise_factor=0.1):
    """Replace some channels with noise to simulate faulty electrodes."""
    num_channels = tf.shape(data)[-1]  # Get the number of channels
    num_samples = tf.shape(data)[1]    # Time points
    batch_size = tf.shape(data)[0]     # Batch size

    # Randomly select channels to replace with noise
    noise_channel_indices = tf.random.shuffle(tf.range(num_channels))[:num_channels // 3]
    num_noise_channels = tf.shape(noise_channel_indices)[0]  # Get the number of channels to replace with noise

    # Generate noise to replace selected channels, matching the shape of the channels to be updated
    noise = tf.random.normal(shape=[batch_size, num_samples, num_noise_channels], stddev=noise_factor)

    # Create batch indices for updating
    batch_indices = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
    batch_indices = tf.tile(batch_indices, [1, num_samples, num_noise_channels])

    # Create time indices for updating
    time_indices = tf.range(num_samples)[tf.newaxis, :, tf.newaxis]
    time_indices = tf.tile(time_indices, [batch_size, 1, num_noise_channels])

    # Tile noise_channel_indices across the batch and time dimensions
    channel_indices = tf.tile(noise_channel_indices[tf.newaxis, tf.newaxis, :], [batch_size, num_samples, 1])

    # Combine batch, time, and channel indices into one tensor for scatter
    indices = tf.stack([batch_indices, time_indices, channel_indices], axis=-1)

    # Apply noise to selected channels using scatter update
    updated_data = tf.tensor_scatter_nd_update(data, indices, noise)

    return updated_data

def apply_frequency_masking(data, num_masks=1, mask_width=2):
    """Mask frequency bands in the signal."""
    fft_data = tf.signal.fft(tf.cast(data, tf.complex64))
    for _ in range(num_masks):
        start = tf.random.uniform([], 0, tf.shape(fft_data)[0] - mask_width, dtype=tf.int32)
        fft_data = tf.concat([fft_data[:start], tf.zeros([mask_width, tf.shape(fft_data)[1]], dtype=fft_data.dtype), fft_data[start+mask_width:]], axis=0)
    return tf.math.real(tf.signal.ifft(fft_data))

def event_dropout(data, event_indices, drop_prob=0.3):
    """Randomly zero out parts of the ground truth event data."""
    for idx in event_indices:
        if tf.random.uniform([]) < drop_prob:
            data = tf.tensor_scatter_nd_update(data, [[idx]], [0])
    return data

def misalign_channels(data, max_shift=3):
    """Randomly misalign channels to simulate electrode misplacement."""
    shifts = tf.random.uniform([tf.shape(data)[2]], -max_shift, max_shift, dtype=tf.int32)  # Ensure we use the correct dimension (channels)

    # Apply the shifts to each channel
    data = tf.map_fn(lambda i: tf.roll(data[:, :, i], shifts[i], axis=1), tf.range(tf.shape(data)[2]), fn_output_signature=tf.float32)

    return data

# Dynamic masking of weights for ripple events
def dynamic_mask_weights(train_labels, weights, params, random_gap_prob=0.5, noise_factor=0.05):
    """
    Dynamically modify train weights by masking out parts of the event onset with a random mask,
    adjusting for each batch during training.
    """
    onset_indices = tf.where(train_labels == 1)[:, 0]

    if params['TYPE_LOSS'].find('Gap') > -1:
        print('Using Dynamic Gap Before Onset')

        for onset in onset_indices:
            if tf.random.uniform([]) < random_gap_prob:
                gap_start = onset - 40  # Adjust this range as per your dataset
                gap_start = tf.maximum(0, gap_start)
                gap_end = onset

                weights[gap_start:gap_end] *= train_labels[gap_start:gap_end]
                random_noise = tf.random.normal(tf.shape(weights[gap_start:gap_end]), mean=1.0, stddev=noise_factor)
                weights[gap_start:gap_end] *= random_noise

    return weights[:tf.shape(train_labels)[0]]

def add_pre_onset_noise(data, onset_indices, noise_duration=50, noise_factor=0.05):
    """Add noise just before the ripple onset to simulate real-world noisy conditions."""
    num_samples, num_channels = tf.shape(data)[0], tf.shape(data)[1]

    for onset in onset_indices:
        pre_onset_start = tf.maximum(0, onset - noise_duration)
        noise = tf.random.normal([noise_duration, num_channels], stddev=noise_factor)
        data = tf.tensor_scatter_nd_add(data, tf.expand_dims(tf.range(pre_onset_start, onset), axis=1), noise)

    return data


# augment data
@tf.function
def augment_data(data, labels, event_indices=None, apply_mixup=False, mixup_data=None, params=None, sampling_rate=1250):
    """
    Full augmentation pipeline combining all strategies for ripple detection.
    Accepts data, labels, and applies augmentations.
    Returns augmented data, unaltered labels.
    """
    # Ensure data has 3 dimensions [batch, time, channels]
    if len(tf.shape(data)) < 3:
        data = tf.expand_dims(data, axis=-1)
    
    # Ensure labels has at least 2 dimensions [batch, time]
    if len(tf.shape(labels)) < 2:
        labels = tf.expand_dims(labels, axis=-1)
    
    # Apply random scaling 20% of the time
    if tf.random.uniform([]) < 0.2:
        try:
            data = random_scaling(data)
        except tf.errors.InvalidArgumentError:
            # Gracefully handle dimension errors
            pass
        
    # Return the augmented data and original labels
    return data, labels


# Optimized version of triplet sampling
def sample_triplets_for_batch(data, labels, window_size=256, min_negative_distance=50, batch_size=32):
    """Optimized version of triplet sampling"""
    # Ensure labels is a 1D numpy array
    labels = np.array(labels).squeeze()

    # --- Modified event extraction logic ---
    ones = np.where(labels == 1)[0]
    if len(ones) == 0:
        return None, None, None
    splits = np.split(ones, np.where(np.diff(ones) != 1)[0] + 1)
    events = [(group[0], group[-1]) for group in splits]
    # -------------------------------

    # Pre-calculate event centers and lengths
    events = np.array(events)
    event_centers = (events[:, 0] + events[:, 1]) // 2
    event_lengths = events[:, 1] - events[:, 0] + 1
    
    # Generate pools of anchors and positives
    anchors = []
    positives = []
    
    n_samples_per_event = 100  # Increased samples per event
    
    # Generate windows for each event
    for idx, (start, end) in enumerate(events):
        length = end - start + 1
        center = start + length // 2
        max_jit = min(window_size // 4, length // 2)
        max_jit = max(max_jit, 1)
        
        for _ in range(n_samples_per_event):
            # Anchor windows - highly overlapping with event
            jitter = np.random.randint(-max_jit, max_jit+1)
            window_center = center + jitter
            window_start = max(0, window_center - window_size // 2)
            window_end = min(len(labels), window_start + window_size)
            
            if window_end - window_start == window_size:
                overlap = min(end, window_end) - max(start, window_start) + 1
                if overlap / length >= 0.7:
                    anchors.append((window_start, window_end))

            # Additional positive windows with broader range
            pos_jitter = np.random.randint(-max_jit*2, max_jit*2+1)
            pos_center = center + pos_jitter
            pos_start = max(0, pos_center - window_size // 2)
            pos_end = min(len(labels), pos_start + window_size)
            
            if pos_end - pos_start == window_size:
                overlap = min(end, pos_end) - max(start, pos_start) + 1
                if 0.4 <= overlap / length:  # Broaden the condition for positive samples
                    positives.append((pos_start, pos_end))

    # Ensure more positives than anchors
    if len(positives) < len(anchors):
        num_additional_positives = len(anchors) - len(positives)
        for _ in range(num_additional_positives):
            # Sample from existing anchors and add jitter to create new positives
            a_start, a_end = random.choice(anchors)
            jitter = np.random.randint(-max_jit, max_jit+1)
            pos_start = max(0, a_start + jitter)
            pos_end = min(len(labels), pos_start + window_size)
            if pos_end - pos_start == window_size:
                positives.append((pos_start, pos_end))

    # Generate negative samples efficiently
    valid_regions = np.ones(len(labels), dtype=bool)
    for start, end in events:
        valid_regions[max(0, start - min_negative_distance):min(len(labels), end + min_negative_distance)] = False
    
    potential_negative_starts = np.where(valid_regions[:-window_size])[0]
    
    if len(potential_negative_starts) == 0:
        print("Warning: No potential negative samples found.")
        return None, None, None
        
    n_desired_negatives = len(anchors) * 10
    if len(potential_negative_starts) > n_desired_negatives:
        negative_starts = np.random.choice(potential_negative_starts, n_desired_negatives, replace=False)
    else:
        negative_starts = potential_negative_starts
        
    negative_regions = [(start, start + window_size) for start in negative_starts]

    print(f"Pool sizes - Anchors: {len(anchors)}, Positives: {len(positives)}, Negatives: {len(negative_regions)}")
    
    min_samples = min(len(anchors), len(positives), len(negative_regions))
    if min_samples < batch_size:
        print(f"Warning: Insufficient samples. Min pool size: {min_samples}")
        # Reduce batch size if possible, otherwise return None
        if min_samples > 0:
            batch_size = min_samples
            print(f"Adjusting batch size to {batch_size} to use available samples.")
        else:
            print("No samples available. Returning None.")
            return None, None, None

    sampled_anchors = random.sample(anchors, batch_size)
    sampled_positives = random.sample(positives, batch_size)
    sampled_negatives = random.sample(negative_regions, batch_size)

    anchor_samples = np.zeros((batch_size, window_size, data.shape[1]), dtype=np.float32)
    positive_samples = np.zeros_like(anchor_samples)
    negative_samples = np.zeros_like(anchor_samples)
    
    anchor_labels = np.zeros((batch_size, window_size, 1), dtype=np.float32)
    positive_labels = np.zeros_like(anchor_labels)
    negative_labels = np.zeros_like(anchor_labels)

    for i in range(batch_size):
        a_start, a_end = sampled_anchors[i]
        p_start, p_end = sampled_positives[i]
        n_start, n_end = sampled_negatives[i]

        anchor_samples[i] = data[a_start:a_end]
        positive_samples[i] = data[p_start:p_end]
        negative_samples[i] = data[n_start:n_end]

        anchor_labels[i, :, 0] = labels[a_start:a_end]
        positive_labels[i, :, 0] = labels[p_start:p_end]
        negative_labels[i, :, 0] = labels[n_start:n_end]

    return ((anchor_samples, positive_samples, negative_samples),
            (anchor_labels, positive_labels, negative_labels))

def create_triplet_dataset(data, labels, params):
    """
    Create a TensorFlow dataset for triplet loss training that works in graph mode.
    Ensures each epoch contains randomly sampled triplets with increased steps.
    """
    window_size = params['NO_TIMEPOINTS']
    batch_size = params['BATCH_SIZE']
    min_negative_distance = params.get('MIN_NEGATIVE_DISTANCE', 64)
    
    # Increase steps per epoch to achieve ~1500 steps
    steps_per_epoch = params.get('steps_per_epoch', 1000)
    print(f'Training steps increased to: {steps_per_epoch}')
    
    # Calculate samples needed for the entire epoch
    samples_needed = batch_size * (steps_per_epoch+1) # Generate more samples
    
    # Sample triplets for the first epoch to verify we can create a dataset
    print("Pre-generating triplet examples for dataset initialization...")
    initial_triplets = sample_triplets_for_batch(
        data, labels, 
        window_size=window_size, 
        min_negative_distance=min_negative_distance,
        batch_size=samples_needed  # Generate enough samples for the entire epoch
    )
    
    # Unpack triplets and ensure correct shapes
    (anchor_samples, positive_samples, negative_samples), (anchor_labels, positive_labels, negative_labels) = initial_triplets
    
    # Ensure labels have shape [batch, time, 1]
    anchor_labels = np.expand_dims(anchor_labels, axis=-1) if anchor_labels.ndim == 2 else anchor_labels
    positive_labels = np.expand_dims(positive_labels, axis=-1) if positive_labels.ndim == 2 else positive_labels
    negative_labels = np.expand_dims(negative_labels, axis=-1) if negative_labels.ndim == 2 else negative_labels
    
    # Ensure samples have shape [batch, time, channels]
    if anchor_samples.ndim == 2:
        anchor_samples = np.expand_dims(anchor_samples, axis=-1)
    if positive_samples.ndim == 2:
        positive_samples = np.expand_dims(positive_samples, axis=-1)
    if negative_samples.ndim == 2:
        negative_samples = np.expand_dims(negative_samples, axis=-1)
    
    num_samples = len(anchor_samples)
    print(f"Generated {num_samples} triplets for this epoch")
    print(f"Sample shapes - Anchor: {anchor_samples.shape}, Labels: {anchor_labels.shape}")
    
    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices(((anchor_samples, positive_samples, negative_samples),
                                                  (anchor_labels, positive_labels, negative_labels)))
    
    # Random shuffling for each epoch
    dataset = dataset.shuffle(buffer_size=num_samples)
    
    # Set batch size
    dataset = dataset.batch(batch_size, drop_remainder=True)  # Drop incomplete batches
    
    # concat batches
    @tf.autograph.experimental.do_not_convert
    def concat_lfps_labels(lfps, labels):
        return tf.concat(lfps, axis=0), tf.concat(labels, axis=0)
    dataset = dataset.map(lambda x, y: concat_lfps_labels(x, y))

    return dataset, params

def apply_augmentation_to_dataset(dataset, params=None, sampling_rate=1250):
    """
    Function to apply augmentations to a tf.data.Dataset with 3 arguments (data, labels, weights).
    """
    def augment_batch(data, labels):
        augmented_data, updated_labels = augment_data(
            data, labels, params=params, sampling_rate=sampling_rate)
        return augmented_data, updated_labels   

    # Apply augment_batch function to each batch using map
    return dataset.map(augment_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def rippleAI_prepare_training_data(train_LFPs,train_GTs,val_LFPs,val_GTs,sf=1250,new_sf=1250,channels=np.arange(0,8),zscore=True,use_band=None):
    '''
        Prepares data for training: subsamples, interpolates (if required), z-scores and concatenates
        the train/test data passed. Does the same for the validation data, but without concatenating
        inputs:
            train_LFPs:  (n_train_sessions) list with the raw LFP of n sessions that will be used to train
            train_GTs:   (n_train_sessions) list with the GT events of n sessions, in the format [ini end] in seconds
            (A): quizá se podría quitar esto, lo de formatear tambien las de validacion
            val_LFPs:    (n_val_sessions) list: with the raw LFP of the sessions that will be used in validation
            val_GTs:     (n_val_sessions) list: with the GT events of n validation sessions
            sf:          (int) original sampling frequency of the data TODO (consultar con Andrea): make it an array, so every session could have a different sf
            channels:    (n_channels) np.array. Channels that will be used to generate data. Check interpolate_channels for more information
        output:
            retrain_LFP: (n_samples x n_channels): sumbsampled, z-scored, interpolated and concatenated data from all the training sessions
            retrain_GT:  (n_events x 2): concatenation of all the events in the training sessions
            norm_val_GT: (n_val_sessions) list: list with the normalized LFP of all the val sessions
            val_GTs:     (n_val_sessions) list: Gt events of each val sessions
    A Rubio LCN 2023

    '''

    assert len(train_LFPs) == len(train_GTs), "The number of train LFPs doesn't match the number of train GTs"
    assert len(val_LFPs) == len(val_GTs), "The number of test LFPs doesn't match the number of test GTs"

    assert len(train_LFPs)+len(val_LFPs) == len(sf), "The number of sampling frequencies doesn't match the number of sessions"
    
    # All the training sessions data and GT will be concatenated in one data array and one GT array (2 x n events)
    counter_sf = 0
    retrain_LFP=[]
    retrain_GT=[]
    offset=0
    for LFP,GT in zip(train_LFPs,train_GTs):
        # pdb.set_trace()
        # 1st session in the array
        print('Original training data shape: ',LFP.shape)
        print('Sampling frequency: ',sf[counter_sf])
        aux_LFP = process_LFP(LFP, ch=channels, sf=sf[counter_sf], new_sf=new_sf, use_zscore=False, use_band=use_band)
        if zscore:
            aux_LFP = (aux_LFP - np.mean(aux_LFP, axis=0))/np.std(aux_LFP, axis=0)
        offset_sf = new_sf
        if offset_sf == 30000:
            assert(aux_LFP.shape[0] == LFP.shape[0])
        
        # shifting GT events
        GT = GT + offset
        
        if len(retrain_LFP)==0:
            retrain_LFP = aux_LFP
            retrain_GT=GT
        # Append the rest of the sessions, taking into account the length (in seconds)
        # of the previous sessions, to cocatenate the events' times
        else:
            retrain_LFP=np.vstack([retrain_LFP,aux_LFP])
            retrain_GT=np.vstack([retrain_GT,GT])
        offset+=len(aux_LFP)/offset_sf
        counter_sf += 1
    # Each validation session LFP will be normalized, etc and stored in an array
    #  the GT needs no further treatment

    norm_val_GT=[]
    for LFP in val_LFPs:
        print('Original validation data shape: ',LFP.shape)
        print('Sampling frequency: ',sf[counter_sf])
        tmpLFP = process_LFP(LFP, ch=channels, sf=sf[counter_sf], new_sf=new_sf, use_zscore=False, use_band=use_band)
        if zscore:
            tmpLFP = (tmpLFP - np.mean(tmpLFP, axis=0))/np.std(tmpLFP, axis=0)
        norm_val_GT.append(tmpLFP)
        counter_sf += 1
    return retrain_LFP, retrain_GT , norm_val_GT, val_GTs


def rippleAI_load_dataset(params, mode='train', preprocess=True, use_band=None):
    """
    Loads the dataset for the Ripple AI model.

    Returns:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
        params (dict): Updated parameters including dataset regenerators.
    """
    if params['TYPE_ARCH'].find('Shift')>-1:
        print('Using Shift')
        sample_shift = float(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Shift')+5:params['TYPE_ARCH'].find('Shift')+7])
        sample_shift = int(sample_shift/1000*params['SRATE'])
        print(sample_shift)
    else:
        sample_shift = 0

    # The training sessions will be appended together. Do the same with your training data
    train_LFPs = []
    train_GTs = []
    all_SFs = []

    # Amigo2
    path=os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Amigo2','figshare_16847521')
    LFP,GT=load_lab_data(path)
    train_LFPs.append(LFP)#1000
    train_GTs.append(GT)
    all_SFs.append(30000)

    # Som2
    path=os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Som2','figshare_16856137')
    LFP,GT=load_lab_data(path)
    train_LFPs.append(LFP)#1000
    train_GTs.append(GT)
    all_SFs.append(30000)

    # Append all your validation sessions
    val_LFPs=[]
    val_GTs=[]
    if mode == 'test':

        # Dlx1 Validation
        path=os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Dlx1','figshare_14959449')
        LFP,GT=load_lab_data(path)
        # LFP = LFP[0:30000*60, :]
        # GT = GT[GT[:,0]<30000*60, :]
        # val_LFPs.append(LFP/1000)
        val_LFPs.append(LFP)
        val_GTs.append(GT)
        # all_SFs.append(30000)
        all_SFs.append(30000)

        # Thy07 Validation
        path=os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Thy7','figshare_14960085')
        LFP,GT=load_lab_data(path)
        # LFP = LFP[0:30000*60, :]
        # GT = GT[GT[:,0]<30000*60, :]
        val_LFPs.append(LFP)
        # val_LFPs.append(LFP/1000)
        val_GTs.append(GT)
        all_SFs.append(30000)

        # Calb20 Validation
        path=os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Calb20','figshare_20072252')
        LFP,GT=load_lab_data(path)
        val_LFPs.append(LFP)
        val_GTs.append(GT)
        all_SFs.append(30000)

    # # ThyNpx Validation
    # path=os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','ThyNpx','figshare_19779337')
    # LFP,GT=load_lab_data(path)
    # val_LFPs.append(LFP)
    # val_GTs.append(GT)
    # all_SFs.append(2500)

    # RippleNet Test Dataset
    # f = h5py.File(os.path.join('/mnt/hpc/projects/MWNaturalPredict/DL/RippleNet/data/m4029_session1.h5'), 'r')
    # f["m4029_session1"]['lfp'][:]
    # f["m4029_session1"]['lfp'][:]


    # # pdb.set_trace()
    # downsampled_fs = 1250
    # with open(path+'/calb20_false_positives.csv') as csv_file:
    #     next(csv_file)

    #     all_FP_labels = []
    #     all_FP_times = []
    #     pop_iniends = []
    #     SWn_iniends = []
    #     SWr_iniends = []

    #     for line in csv_file:
    #         tokens = line.replace("\n", "").split(",")
    #         label = int(tokens[-1])
    #         all_FP_labels.append(label)
    #         all_FP_times.append([float(tokens[0]), float(tokens[1])])

    #         if label == 1:
    #             pop_iniends.append([float(tokens[0]), float(tokens[1])])
    #         elif label == 2:
    #             SWn_iniends.append([float(tokens[0]), float(tokens[1])])
    #         elif label == 3:
    #             SWr_iniends.append([float(tokens[0]), float(tokens[1])])

    #     all_FP_times = np.array(all_FP_times)
    #     pop_iniends = np.array(pop_iniends)
    #     SWn_iniends = np.array(SWn_iniends)
    #     SWr_iniends = np.array(SWr_iniends)

    # all_FP_indexes = np.array(all_FP_times * downsampled_fs, dtype=int)
    # pop_indexes = np.array(pop_iniends * downsampled_fs, dtype=int)
    # SWn_indexes = np.array(SWn_iniends * downsampled_fs, dtype=int)
    # SWr_indexes = np.array(SWr_iniends * downsampled_fs, dtype=int)
    # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/all_FPs_val{0}.npy'.format(2), all_FP_indexes)
    # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/all_pops_val{0}.npy'.format(2), pop_indexes)
    # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/all_SWRn_val{0}.npy'.format(2), SWn_indexes)
    # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/DL/predSWR/all_SWRr_val{0}.npy'.format(2), SWr_indexes)

    # pdb.set_trace()
    train_data, train_labels_vec, val_data, val_labels_vec = rippleAI_prepare_training_data(train_LFPs,
                                                                                            train_GTs,
                                                                                            val_LFPs,
                                                                                            val_GTs,
                                                                                            sf=all_SFs,
                                                                                            new_sf=params['SRATE'],
                                                                                            zscore=preprocess,
                                                                                            use_band=use_band)
    train_data = train_data.astype('float32')
    if mode == 'test':
        val_data = [k.astype('float32') for k in val_data]
        return val_data, val_labels_vec

    # Split data for training and testing
    test_examples, events_test, train_examples, events_train = split_data(train_data, train_labels_vec, sf=params['SRATE'], split=0.7)

    # fix labels
    sf = params['SRATE']
    
    # Generate binary labels for training and testing data
    y = np.zeros(shape=len(train_examples), dtype=np.float32)
    for event in events_train:
        y[int(sf*event[0]):int(sf*event[1])+sample_shift] = 1
    train_labels = y
    label_ratio = np.sum(train_labels)/len(train_labels)

    y = np.zeros(shape=len(test_examples), dtype=np.float32)
    for event in events_test:
        y[int(sf*event[0]):int(sf*event[1])+sample_shift] = 1
    test_labels = y

    # Create triplet datasets for training and testing
    print("Using triplet loss for training")
    regenerator = TripletDatasetRegenerator(train_examples, train_labels, params)
    train_triplet_dataset = regenerator.dataset
    test_triplet_dataset, _ = create_triplet_dataset(test_examples, test_labels, params)
    
    # Add regenerator to dataset params
    dataset_params = params.copy()
    dataset_params['triplet_regenerator'] = regenerator
    
    return train_triplet_dataset, test_triplet_dataset, label_ratio, dataset_params

def load_allen(indeces= np.int32(np.linspace(49,62,8))):
    loaded_data_raw = np.load('/cs/projects/OWVinckSWR/Carmen/LFP_extracted/sanity_check/raw_lfp_fc.npy')
    indeces[::-1].sort()
    LFP = loaded_data_raw[:,indeces]
    # Process LFP
    data = process_LFP(LFP, sf = 1250, channels=np.arange(0,8))
    return data

class TripletDatasetRegenerator:
    def __init__(self, data, labels, params):
        self.data = data
        self.labels = labels
        self.params = params
        self.dataset = None
        self.initialize()
    
    def initialize(self):
        """Initialize/reinitialize the dataset with new triplet samples"""
        self.dataset, _ = create_triplet_dataset(self.data, self.labels, self.params)
        
    def reinitialize(self):
        """Regenerate triplet samples"""
        self.initialize()

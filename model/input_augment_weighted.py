import tensorflow.keras.backend as K
from model.cnn_ripple_utils import load_lab_data, process_LFP, filter_LFP, split_data, load_info, load_raw_data
import tensorflow as tf
import numpy as np
import pdb
import os
from keras.utils import timeseries_dataset_from_array


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

def apply_realistic_augmentations(data, params=None):
    """
    Applies a combination of realistic augmentations to the input data.

    Args:
        data: Input tensor of shape [batch_size, time_points, channels]
        params: Optional parameters dictionary
    """
    # Random scaling (0.5 to 2.0)
    scale_factor = tf.random.uniform([tf.shape(data)[0], 1, tf.shape(data)[-1]], 0.5, 2.0)
    data = data * scale_factor

    # Random timepoint shift (±20 points)
    shifts = tf.random.uniform([tf.shape(data)[0]], -20, 20, dtype=tf.int32)
    data = tf.map_fn(lambda x: tf.roll(x[0], x[1], axis=0), (data, shifts), dtype=tf.float32)

    # DC shift (±0.5)
    dc_shift = tf.random.uniform([tf.shape(data)[0], 1, tf.shape(data)[-1]], -100, 100)
    data = data + dc_shift

    # Bandstop filter (2-85 Hz)
    # Using FFT to implement bandstop
    fft = tf.signal.fft(tf.cast(data, tf.complex64))
    freq_mask = tf.ones_like(fft)
    band_indices = tf.cast(tf.range(2, 85) / (params['SRATE']/2) * tf.shape(data)[1]/2, tf.int32)
    updates = tf.zeros_like(band_indices, dtype=tf.float32)
    freq_mask = tf.tensor_scatter_nd_update(freq_mask, tf.expand_dims(band_indices, 1), updates)
    data = tf.cast(tf.math.real(tf.signal.ifft(fft * tf.cast(freq_mask, tf.complex64))), tf.float32)

    # Gaussian noise (0-0.2)
    noise = tf.random.normal(tf.shape(data), 0, tf.random.uniform([], 0.0, 0.2))
    data = data + noise

    return data

def apply_artifact_augmentations(data):
    """
    Applies artifactual augmentations that the model should be robust against.
    """
    # High amplitude gaussian noise (10-100)
    noise_amp = tf.random.uniform([], 10.0, 100.0)
    noise = tf.random.normal(tf.shape(data), 0, noise_amp)
    data = data + noise

    # Random extreme value artifacts
    batch_size, time_points, channels = tf.shape(data)[0], tf.shape(data)[1], tf.shape(data)[2]

    # Generate random positions for artifacts
    num_artifacts = tf.random.uniform([], 1, 5, dtype=tf.int32)
    artifact_positions = tf.random.uniform([num_artifacts, 3],
                                         maxval=[batch_size, time_points, channels],
                                         dtype=tf.int32)

    # Generate extreme values (both positive and negative)
    extreme_values = tf.random.uniform([num_artifacts], -100.0, 100.0)
    data = tf.tensor_scatter_nd_update(data, artifact_positions, extreme_values)

    return data

def create_adversarial_short_events(data, labels, params):
    """
    Creates adversarial samples from positive/anchor events by creating short events.

    Args:
        data: Input tensor of shape [batch_size, time_points, channels]
        labels: Binary labels tensor
        params: Parameters dictionary containing ripple characteristics
    """
    batch_size = tf.shape(data)[0]

    # Find positive events
    positive_indices = tf.where(tf.reduce_max(labels, axis=1) > 0)

    # For each positive event
    def process_event(idx):
        event_data = data[idx]
        event_label = labels[idx]

        # Extract a short segment (1-5ms) from the ripple
        event_start = tf.random.uniform([], 0, tf.shape(event_data)[0]-10, dtype=tf.int32)
        segment_length = tf.random.uniform([], 5, 25, dtype=tf.int32)  # 4-20ms at 1250Hz
        short_segment = event_data[event_start:event_start+segment_length]

        # Create surrounding noise
        noise = tf.random.normal([tf.shape(event_data)[0], tf.shape(event_data)[1]], 0, 0.1)

        # Insert short segment into noise
        result = tf.tensor_scatter_nd_update(
            noise,
            tf.expand_dims(tf.range(event_start, event_start+segment_length), 1),
            short_segment
        )

        # Create corresponding short label
        short_label = tf.zeros_like(event_label)
        short_label = tf.tensor_scatter_nd_update(
            short_label,
            tf.expand_dims(tf.range(event_start, event_start+segment_length), 1),
            tf.ones([segment_length])
        )

        return result, short_label

    # Process selected positive events
    adversarial_data, adversarial_labels = tf.map_fn(
        process_event,
        positive_indices,
        fn_output_signature=(tf.float32, tf.float32)
    )

    return adversarial_data, adversarial_labels


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

    # # Apply realistic augmentations (20% chance)
    # if tf.random.uniform([]) < 0.0:
    #     data = apply_realistic_augmentations(data, params)

    # # Apply artifact augmentations (10% chance)
    # if tf.random.uniform([]) < 0.1:
    #     data = apply_artifact_augmentations(data)

    # # Create adversarial short events (5% chance)
    # if tf.random.uniform([]) < 0.05:
    #     adv_data, adv_labels = create_adversarial_short_events(data, labels, params)
    #     # Randomly replace some samples with adversarial ones
    #     mask = tf.random.uniform([tf.shape(data)[0]]) < 0.2
    #     data = tf.where(mask[:, tf.newaxis, tf.newaxis], adv_data, data)
    #     labels = tf.where(mask[:, tf.newaxis], adv_labels, labels)

    # Return the augmented data and original labels
    return data, labels


# Apply augmentations to tf.data.Dataset
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
    for LFP,GT in zip(train_LFPs,train_GTs):
        # pdb.set_trace()
        # 1st session in the array
        print('Original training data shape: ',LFP.shape)
        print('Sampling frequency: ',sf[counter_sf])
        if len(retrain_LFP)==0:
            retrain_LFP = process_LFP(LFP, ch=channels, sf=sf[counter_sf], new_sf=new_sf, use_zscore=False, use_band=use_band)
            if zscore:
                retrain_LFP = (retrain_LFP - np.mean(retrain_LFP, axis=0))/np.std(retrain_LFP, axis=0)
            offset_sf = new_sf
            if offset_sf == 30000:
                assert(retrain_LFP.shape[0] == LFP.shape[0])
            offset=len(retrain_LFP)/offset_sf # fix labels
            retrain_GT=GT

        # Append the rest of the sessions, taking into account the length (in seconds)
        # of the previous sessions, to cocatenate the events' times
        else:
            aux_LFP = process_LFP(LFP, ch=channels, sf=sf[counter_sf], new_sf=new_sf, use_zscore=False, use_band=use_band)
            if zscore:
                aux_LFP = (aux_LFP- np.mean(aux_LFP, axis=0))/np.std(aux_LFP, axis=0)
            retrain_LFP=np.vstack([retrain_LFP,aux_LFP])
            retrain_GT=np.vstack([retrain_GT,GT+offset])
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
    train_LFPs.append(LFP)#/1000
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
    # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/all_SWRr_val{0}.npy'.format(2), SWr_indexes)

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

    # test_examples, events_test, train_examples, events_train = split_data(train_data, train_labels_vec, sf=30000, split=0.7)
    test_examples, events_test, train_examples, events_train = split_data(train_data, train_labels_vec, sf=params['SRATE'], split=0.7)

    # fix labels
    sf = params['SRATE']
    # sf = 30000

    y = np.zeros(shape=len(train_examples), dtype=np.float32)
    for event in events_train:
        y[int(sf*event[0]):int(sf*event[1])+sample_shift] = 1
    train_labels = y
    label_ratio = np.sum(train_labels)/len(train_labels)

    y = np.zeros(shape=len(test_examples), dtype=np.float32)
    for event in events_test:
        y[int(sf*event[0]):int(sf*event[1])+sample_shift] = 1
    test_labels = y


    from scipy import signal
    from scipy.signal import butter, filtfilt
    M = round(51/1000*params['SRATE'])
    # onsets = np.diff(train_labels)==1
    onsets = np.hstack((0, np.diff(train_labels))).astype(np.uint32)==1
    # offsets = np.hstack((0, np.diff(train_labels))).astype(np.uint32)==-1
    assert(np.unique(train_labels[np.where(onsets)[0]])==1)
    assert(np.unique(train_labels[np.where(onsets)[0]]-1)==0)
    # assert(np.unique(train_labels[np.where(offsets)[0]])==0)
    # assert(np.unique(train_labels[np.where(offsets)[0]]-1)==1)

    if params['TYPE_LOSS'].find('Anchor')>-1:
        print('Using Anchor Weights')
        weights = signal.convolve(onsets, signal.windows.exponential(M, 0, 3, False))+0.01
    else:
        print('Using Else (Ones)')
        weights = np.ones(train_labels.shape, dtype=np.float32)
    weights /= np.max(weights)

    # make a gap in the weights
    if params['TYPE_LOSS'].find('Gap')>-1:
        print('Using Gap Before Onset')
        onset_indices = np.where(onsets)[0]
        for onset in onset_indices:
            # if np.any(train_labels[onset-40:onset]==1):
            weights[onset-40:onset] *= train_labels[onset-40:onset]
    weights = weights[:train_labels.shape[0]]

    # pdb.set_trace()
    if params['TYPE_LOSS'].find('Gap')>-1:
        assert(np.abs(np.unique(weights[np.where(onsets)[0]])-1)<0.0001)
        assert(np.unique(weights[np.where(onsets)[0]-1])<0.0001)

    # make batches
    if sample_shift>0:
        train_examples = train_examples[:-sample_shift, :]
        train_labels = train_labels[sample_shift:]
        weights = weights[sample_shift:]
        test_examples = test_examples[:-sample_shift, :]
        test_labels = test_labels[sample_shift:]

    # pdb.set_trace()
    print(train_examples.shape)
    print(train_labels.shape)
    print(weights.shape)
    print(test_examples.shape)
    print(test_labels.shape)

    weights = weights.astype('float32')
    # make datasets

    if params['TYPE_ARCH'].find('CAD')>-1:
        sample_length = params['NO_TIMEPOINTS']
        label_length = 1
        label_skip = sample_length
    elif params['TYPE_ARCH'].find('Patch')>-1:
        sample_length = params['NO_TIMEPOINTS']*2
        label_length = sample_length
        label_skip = 0
    else:
        sample_length = params['NO_TIMEPOINTS']*2
        label_length = sample_length/2
        label_skip = int(sample_length/2)
    stride_step = params['NO_STRIDES']

    # train
    train_x = timeseries_dataset_from_array(
        train_examples,
        None,
        sequence_length=sample_length,
        sequence_stride=stride_step,
        batch_size=None,#params["BATCH_SIZE"],
        shuffle=False
    )
    train_y = timeseries_dataset_from_array(
        train_labels[label_skip+sample_shift:].reshape(-1,1),
        None,
        sequence_length=label_length,
        sequence_stride=stride_step,
        batch_size=None,#params["BATCH_SIZE"],
        shuffle=False
    )
    train_w = timeseries_dataset_from_array(
        weights[label_skip+sample_shift:].reshape(-1,1),
        None,
        sequence_length=label_length,
        sequence_stride=stride_step,
        batch_size=None,#params["BATCH_SIZE"],
        shuffle=False
    )
    test_x = timeseries_dataset_from_array(
        test_examples,
        None,
        sequence_length=sample_length,
        sequence_stride=stride_step,
        batch_size=None,#params["BATCH_SIZE"],
        shuffle=False
    )
    test_y = timeseries_dataset_from_array(
        test_labels[label_skip+sample_shift:].reshape(-1,1),
        None,
        sequence_length=label_length,
        sequence_stride=stride_step,
        batch_size=None,#params["BATCH_SIZE"],
        shuffle=False
    )

    # if params['TYPE_ARCH'].find('CAD')>-1:
    #     print('Using CAD')
    #     def downsample_labels(targets):
    #         # Add a channel dimension for max pooling
    #         targets = tf.expand_dims(targets, axis=0)
    #         # Apply max pooling with a window size of 12 and stride of 12
    #         targets = tf.nn.max_pool1d(targets, ksize=12, strides=12, padding='VALID')
    #         # Remove the channel dimension after pooling
    #         targets = tf.squeeze(targets, axis=0)
    #         return targets
    #     train_y = train_y.map(lambda x: downsample_labels(x))
    #     train_w = train_w.map(lambda x: downsample_labels(x))
    #     test_y = test_y.map(lambda x: downsample_labels(x))

    # # Apply layer normalization to LFPs
    # def channel_normalize(data):
    #     mean, variance = tf.nn.moments(data, axes=[1], keepdims=True)
    #     normalized_data = (data - mean) / tf.sqrt(variance + 1e-6)
    #     return normalized_data

    # train_x = train_x.map(lambda x: (channel_normalize(x)))
    # test_x = test_x.map(lambda x: (channel_normalize(x)))
    if params['TYPE_ARCH'].find('Only')>-1:
        print('NOT PREDICTING LFPs')
        # Concatenate train_x and train_y per batch
        test_c = test_y #tf.data.Dataset.zip((test_xy, test_y))
        train_c = tf.data.Dataset.zip((train_y, train_w))

        @tf.autograph.experimental.do_not_convert
        def concat_lfps_labels_weights(labels, weights):
            return tf.concat([labels, weights], axis=-1)  # Concatenate along the last axis (channels)ZZ
        train_d = train_c.map(lambda x, y: concat_lfps_labels_weights(x, y))
        test_d = test_c
    else:
        # train_y = train_y.map(lambda x: tf.pad(x, [[0, 0], [50, 0], [0, 0]], 'CONSTANT', constant_values=0.0))
        if params['TYPE_ARCH'].find('Patch')>-1:
            print('Using Patch with Full Window')
            train_xy = train_x.map(lambda x: x)
            test_xy = test_x.map(lambda x: x)
        else:
            train_xy = train_x.map(lambda x: x[:, -params['NO_TIMEPOINTS']:, :])
            test_xy = test_x.map(lambda x: x[:, -params['NO_TIMEPOINTS']:, :])
            
        # Concatenate train_x and train_y per batch
        test_c = tf.data.Dataset.zip((test_xy, test_y))
        train_c = tf.data.Dataset.zip((train_xy, train_y, train_w))

        @tf.autograph.experimental.do_not_convert
        def concat_lfps_labels_weights(lfps, labels, weights):
            return tf.concat([lfps, labels, weights], axis=-1)  # Concatenate along the last axis (channels)ZZ
        train_d = train_c.map(lambda x, y, z: concat_lfps_labels_weights(x, y, z))

        @tf.autograph.experimental.do_not_convert
        def concat_lfps_labels(lfps, labels):
            return tf.concat([lfps, labels], axis=-1)  # Concatenate along the last axis (channels)ZZ
        test_d = test_c.map(lambda x, y: concat_lfps_labels(x, y))

    # Combine the dataset with weights
    train_dataset = tf.data.Dataset.zip((train_x, train_d))
    test_dataset = tf.data.Dataset.zip((test_x, test_d)).batch(params['BATCH_SIZE']).prefetch(tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.shuffle(params["SHUFFLE_BUFFER_SIZE"], reshuffle_each_iteration=True).batch(params['BATCH_SIZE']).prefetch(tf.data.experimental.AUTOTUNE)
    if params['TYPE_ARCH'].find('Aug')>-1:
        print('Using Augmentations:')
        train_dataset = apply_augmentation_to_dataset(train_dataset, params=params, sampling_rate=params['SRATE'])
    else:
        print('No augmentation')
    return train_dataset, test_dataset, label_ratio#, val_dataset

def load_allen(indeces= np.int32(np.linspace(49,62,8))):
    loaded_data_raw = np.load('/cs/projects/OWVinckSWR/Carmen/LFP_extracted/sanity_check/raw_lfp_fc.npy')
    indeces[::-1].sort()
    LFP = loaded_data_raw[:,indeces]
    # Process LFP
    data = process_LFP(LFP, sf = 1250, channels=np.arange(0,8))
    return data

def load_topological_dataset(batch_size=32, shuffle_buffer=1000):
    """
    Loads topological data from Juan Pablo and Alberto datasets and creates a TensorFlow dataset.

    Args:
        batch_size (int): Size of batches for training
        shuffle_buffer (int): Size of shuffle buffer

    Returns:
        tf.data.Dataset: Dataset containing batched and preprocessed data
    """
    import scipy.io as sio
    import h5py
    # Load both .mat files
    # jp_data = sio.loadmat('/cs/projects/OWVinckSWR/Dataset/TopologicalData/JuanPabloDB_struct.mat')
    # ab_data = sio.loadmat('/cs/projects/OWVinckSWR/Dataset/TopologicalData/AlbertoDB_struct.mat')

    # Load both .mat files using h5py
    jp_data = h5py.File('/cs/projects/OWVinckSWR/Dataset/TopologicalData/JuanPabloDB_struct.mat', 'r')
    ab_data = h5py.File('/cs/projects/OWVinckSWR/Dataset/TopologicalData/AlbertoDB_struct.mat', 'r')

    # Process and concatenate data from both sources
    def process_struct(data):
        ripples = data['ripples']  # nEvents x 127
        n_events = ripples.shape[1]

        # Reshape ripples to (nEvents, 127, 8) by splitting last dimension
        ripples_reshaped = np.tile(np.transpose(ripples, (1, 0))[:,:,np.newaxis], (1,1,8))

        # Collect other features
        features = np.column_stack([
            np.array(data['amplitude']).reshape(-1, 1),
            np.array(data['entropy']).reshape(-1, 1),
            np.array(data['duration']).reshape(-1, 1),
            np.array(data['frequency']).reshape(-1, 1)
        ])

        return ripples_reshaped, features

    # Process both datasets
    jp_ripples, jp_features = process_struct(jp_data)
    ab_ripples, ab_features = process_struct(ab_data)

    # Concatenate datasets
    all_ripples = np.concatenate([jp_ripples, ab_ripples], axis=0)
    all_features = np.concatenate([jp_features, ab_features], axis=0)

    # Convert to float32 for better performance
    all_ripples = all_ripples.astype(np.float32)
    all_features = all_features.astype(np.float32)

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        'ripples': all_ripples,
        'features': all_features
    })

    # Apply dataset transformations
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


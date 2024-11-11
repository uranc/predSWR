import tensorflow.keras.backend as K
from model.cnn_ripple_utils import load_lab_data, process_LFP, split_data, load_info, load_raw_data
import tensorflow as tf
import numpy as np
import pdb
import os
from keras.utils import timeseries_dataset_from_array


def random_scaling(data, min_scale=0.1, max_scale=10.0):
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
    scale_factor = tf.random.uniform([], min_scale, max_scale)

    # Apply the scaling factor uniformly to all channels and samples within the batch
    return data * scale_factor

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
def augment_data(data, labels, weights, event_indices=None, apply_mixup=False, mixup_data=None, params=None, sampling_rate=1250):
    """
    Full augmentation pipeline combining all strategies for ripple detection.
    Accepts data, labels, and weights and applies augmentations.
    Returns augmented data, unaltered labels, and updated weights.
    """
    # Ensure all operations are TensorFlow operations
    batch_size = tf.shape(data)[0]

    # Dynamic masking: Use tf.map_fn for dynamic masking across batches
    # def update_weights_per_sample(i):
    #     return dynamic_mask_weights(labels[i], weights[i], params)

    # # Use tf.map_fn to apply the masking across the entire batch (no Python loops)
    # weights = tf.map_fn(update_weights_per_sample, tf.range(batch_size), fn_output_signature=tf.float32)

    # # Add burst noise (simulate artifacts)
    # data = add_optimized_burst_noise(data, sampling_rate=sampling_rate, params=params)

    # Add pink noise (low-frequency noise)
    print(data.shape)
    
    # if tf.random.uniform([]) < 0.1:
    #     data = add_pink_noise(data)

    if tf.random.uniform([]) < 0.1:
        data = random_scaling(data)

    if tf.random.uniform([]) < 0.1:
        data = add_varying_noise(data)

    # if tf.random.uniform([]) < 0.1:
    #     data = random_channel_shuffle(data)

    # if tf.random.uniform([]) < 0.1:
    #     data = replace_channels_with_noise(data)
    # # data = apply_frequency_masking(data)

    # if event_indices is not None:
    #     data = event_dropout(data, event_indices)
    #     data = add_pre_onset_noise(data, event_indices)

    # data = misalign_channels(data)

    # if apply_mixup and mixup_data is not None:
    #     data = mixup(data, mixup_data)

    return data, labels, weights  # Return augmented data, unaltered labels, and modified weights


# Apply augmentations to tf.data.Dataset
def apply_augmentation_to_dataset(dataset, params=None, sampling_rate=1250):
    """
    Function to apply augmentations to a tf.data.Dataset with 3 arguments (data, labels, weights).
    """
    def augment_batch(data, labels, weights):
        augmented_data, updated_labels, updated_weights = augment_data(
            data, labels, weights, params=params, sampling_rate=sampling_rate)
        return augmented_data, updated_labels, updated_weights

    # Apply augment_batch function to each batch using map
    return dataset.map(augment_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def rippleAI_prepare_training_data(train_LFPs,train_GTs,val_LFPs,val_GTs,sf=1250,channels=np.arange(0,8), zscore=True, use_band=None):
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
    def filter_LFP(LFP, sf=1250, band='low'):
        """
        Filters the LFP data in a specified frequency band.

        Args:
            LFP: Input LFP data of shape [num_samples, num_channels].
            sf: Sampling frequency of the data.
            band: Frequency band to filter ('low' for below 100 Hz, 'high' for 100-300 Hz).

        Returns:
            Filtered LFP data.
        """
        from scipy import signal
        from scipy.signal import butter, filtfilt
        if band == 'low':
            print('Filtering low band')
            lowcut = 0.5
            highcut = 30.0
        elif band == 'high':
            print('Filtering high band')
            lowcut = 120.0
            highcut = 250.0
        else:
            raise ValueError("Invalid band. Choose 'low' or 'high'.")

        b, a = butter(4, [lowcut, highcut], btype='band', fs=sf)
        filtered_LFP = filtfilt(b, a, LFP, axis=0)
        return filtered_LFP    
    
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
            retrain_LFP = process_LFP(LFP,sf[counter_sf],channels,use_zscore=False)
            if use_band is not None:
                retrain_LFP = filter_LFP(retrain_LFP, sf=1250, band=use_band)
            if zscore:
                retrain_LFP = (retrain_LFP- np.mean(retrain_LFP, axis=0))/np.std(retrain_LFP, axis=0)
            if retrain_LFP.shape[0] != LFP.shape[0]:
                offset_sf = 1250
            else:
                offset_sf = 30000
            offset=len(retrain_LFP)/offset_sf # fix labels
            retrain_GT=GT
            
        # Append the rest of the sessions, taking into account the length (in seconds)
        # of the previous sessions, to cocatenate the events' times
        else:
            aux_LFP = process_LFP(LFP,sf[counter_sf],channels,use_zscore=False)
            if use_band is not None:
                aux_LFP = filter_LFP(aux_LFP, sf=1250, band=use_band)
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
        tmpLFP = process_LFP(LFP,sf[counter_sf],channels,use_zscore=False)
        if use_band is not None:
            tmpLFP = filter_LFP(tmpLFP, sf=1250, band=use_band)
        if zscore:
            tmpLFP = (tmpLFP - np.mean(tmpLFP, axis=0))/np.std(tmpLFP, axis=0)
        norm_val_GT.append(tmpLFP)
        counter_sf += 1
    return retrain_LFP, retrain_GT , norm_val_GT, val_GTs


def rippleAI_load_dataset(params, mode='train', preprocess=True, spatial_freq=1250, use_band=None):
    """
    Loads the dataset for the Ripple AI model.

    Returns:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
    """
    if params['TYPE_ARCH'].find('Shift')>-1:
        print('Using Shift')
        sample_shift = int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Shift')+5:params['TYPE_ARCH'].find('Shift')+7])
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
    train_data, train_labels_vec, val_data, val_labels_vec = rippleAI_prepare_training_data(train_LFPs,train_GTs,val_LFPs,val_GTs,sf=all_SFs,zscore=preprocess,use_band=use_band)
    train_data = train_data.astype('float32')
    if mode == 'test':
        val_data = [k.astype('float32') for k in val_data]
        return val_data, val_labels_vec

    # test_examples, events_test, train_examples, events_train = split_data(train_data, train_labels_vec, sf=30000, split=0.7)
    test_examples, events_test, train_examples, events_train = split_data(train_data, train_labels_vec, sf=1250, split=0.7)

    # fix labels
    sf = 1250
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
    M = 51
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
    sample_length = params['NO_TIMEPOINTS']*2
    stride_step = sample_length/params['NO_STRIDES']

    # train
    train_x = timeseries_dataset_from_array(
        train_examples,
        None,
        sequence_length=sample_length,
        sequence_stride=sample_length/stride_step,
        batch_size=params["BATCH_SIZE"]
    )
    train_y = timeseries_dataset_from_array(
        train_labels[int(sample_length/2)+sample_shift:].reshape(-1,1),
        None,
        sequence_length=sample_length/2,
        sequence_stride=sample_length/stride_step,
        batch_size=params["BATCH_SIZE"]
    )
    train_w = timeseries_dataset_from_array(
        weights[int(sample_length/2)+sample_shift:].reshape(-1,1),
        None,
        sequence_length=sample_length/2,
        sequence_stride=sample_length/stride_step,
        batch_size=params["BATCH_SIZE"]
    )

    test_x = timeseries_dataset_from_array(
        test_examples,
        None,
        sequence_length=sample_length,
        sequence_stride=sample_length/stride_step,
        batch_size=params["BATCH_SIZE"]
    )
    test_y = timeseries_dataset_from_array(
        test_labels[int(sample_length/2)+sample_shift:].reshape(-1,1),
        None,
        sequence_length=sample_length/2,
        sequence_stride=sample_length/stride_step,
        batch_size=params["BATCH_SIZE"]
    )

    # train_y = train_y.map(lambda x: tf.pad(x, [[0, 0], [50, 0], [0, 0]], 'CONSTANT', constant_values=0.0))
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
    test_dataset = tf.data.Dataset.zip((test_x, test_d)).prefetch(tf.data.experimental.AUTOTUNE)

    if params['TYPE_ARCH'].find('Aug')>-1:
        print('Using Augmentations:')
        train_dataset = apply_augmentation_to_dataset(train_dataset, params=params)
    else:
        print('No augmentation')

    train_dataset = train_dataset.shuffle(params["SHUFFLE_BUFFER_SIZE"], reshuffle_each_iteration=True).prefetch(tf.data.experimental.AUTOTUNE)#.batch(params["BATCH_SIZE"])



    return train_dataset, test_dataset, label_ratio#, val_dataset


def load_allen(indeces= np.int32(np.linspace(49,62,8))):
    loaded_data_raw = np.load('/cs/projects/OWVinckSWR/Carmen/LFP_extracted/sanity_check/raw_lfp_fc.npy')
    indeces[::-1].sort()
    LFP = loaded_data_raw[:,indeces]
    # Process LFP
    data = process_LFP(LFP, sf = 1250, channels=np.arange(0,8))
    return data
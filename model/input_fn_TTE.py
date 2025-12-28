import tensorflow.keras.backend as K
from model.cnn_ripple_utils import load_lab_data, process_LFP, filter_LFP, split_data, load_info, load_raw_data
import tensorflow as tf
import numpy as np
import pdb
import os
import h5py
from keras.utils import timeseries_dataset_from_array
from scipy import signal

# ==========================================
# 1. Augmentation & Helper Functions
# ==========================================

def random_scaling(data, min_scale=0.8, max_scale=1.2):
    """
    Randomly scale the entire batch by a single scaling factor.
    Args:
        data: Input tensor of shape [batch_size, num_samples, num_channels].
    """
    scale_factors = tf.random.uniform((1, 1, 8), min_scale, max_scale)
    return data * scale_factors

def random_shift(data, max_shift_ms=2, sampling_rate=1250):
    """Random temporal shift for each channel."""
    batch_size = tf.shape(data)[0]
    num_samples = tf.shape(data)[1]
    num_channels = tf.shape(data)[2]
    max_shift_samples = tf.cast((max_shift_ms / 1000.0) * tf.cast(sampling_rate, tf.float32), tf.int32)
    shift_amounts = tf.random.uniform([batch_size, num_channels], -max_shift_samples, max_shift_samples, dtype=tf.int32)

    def shift_single_channel(args):
        data_batch, shift_per_channel = args
        shifted_data = tf.map_fn(lambda shift: tf.roll(data_batch, shift, axis=0), shift_per_channel, dtype=tf.float32)
        return shifted_data

    data = tf.map_fn(shift_single_channel, (data, shift_amounts), fn_output_signature=tf.float32)
    return data

def add_varying_noise(data, noise_factor_range=(0.01, 0.1)):
    """Add varying levels of Gaussian noise per channel."""
    batch_size = tf.shape(data)[0]
    num_channels = tf.shape(data)[2]
    channel_variance = tf.math.reduce_variance(data, axis=[0, 1], keepdims=True)
    noise_factors = tf.random.uniform([batch_size, num_channels], noise_factor_range[0], noise_factor_range[1])
    noise_factors = tf.reshape(noise_factors, [batch_size, 1, num_channels])
    noise = tf.random.normal(tf.shape(data), mean=0.0, stddev=1.0)
    scaled_noise = noise * noise_factors * tf.sqrt(channel_variance)
    return data + scaled_noise

# [Keeping your original augmentation wrapper]
@tf.function
def augment_data(data, labels, event_indices=None, apply_mixup=False, mixup_data=None, params=None, sampling_rate=1250):
    """
    Full augmentation pipeline.
    """
    if len(tf.shape(data)) < 3:
        data = tf.expand_dims(data, axis=-1)
    if len(tf.shape(labels)) < 2:
        labels = tf.expand_dims(labels, axis=-1)

    # Apply random scaling 20% of the time
    if tf.random.uniform([]) < 0.2:
        try:
            data = random_scaling(data)
        except tf.errors.InvalidArgumentError:
            pass

    return data, labels

def apply_augmentation_to_dataset(dataset, params=None, sampling_rate=1250):
    """Apply augmentations to tf.data.Dataset."""
    def augment_batch(data, labels):
        augmented_data, updated_labels = augment_data(
            data, labels, params=params, sampling_rate=sampling_rate)
        return augmented_data, updated_labels

    return dataset.map(augment_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)


# ==========================================
# 2. Data Preparation Logic
# ==========================================

def rippleAI_prepare_training_data(train_LFPs, train_GTs,
                                   val_LFPs,   val_GTs,
                                   sf=1250, new_sf=1250,
                                   channels=np.arange(0, 8),
                                   zscore=True, process_online=False, use_band=None):    
    '''
    Prepares data: subsamples, interpolates, z-scores and concatenates.
    '''
    assert len(train_LFPs) == len(train_GTs), "Mismatch in train LFPs/GTs"
    assert len(val_LFPs) == len(val_GTs), "Mismatch in val LFPs/GTs"

    counter_sf = 0
    retrain_LFP=[]
    
    # Process Training Data (Concatenate all sessions)
    for LFP, GT in zip(train_LFPs, train_GTs):
        print(f'Processing Train Session. Shape: {LFP.shape}, SF: {sf[counter_sf]}')
        
        if len(retrain_LFP) == 0:
            retrain_LFP = process_LFP(LFP, ch=channels, sf=sf[counter_sf], new_sf=new_sf, use_zscore=False, use_band=use_band)
            if zscore:
                retrain_LFP = (retrain_LFP - np.mean(retrain_LFP, axis=0)) / np.std(retrain_LFP, axis=0)
            
            offset_sf = new_sf
            offset = len(retrain_LFP) / offset_sf
            retrain_GT = GT
        else:
            aux_LFP = process_LFP(LFP, ch=channels, sf=sf[counter_sf], new_sf=new_sf, use_zscore=False, use_band=use_band)
            if zscore:
                aux_LFP = (aux_LFP - np.mean(aux_LFP, axis=0)) / np.std(aux_LFP, axis=0)
            
            retrain_LFP = np.vstack([retrain_LFP, aux_LFP])
            retrain_GT = np.vstack([retrain_GT, GT + offset])
            offset += len(aux_LFP) / offset_sf
        counter_sf += 1

    # Process Validation Data (Keep separate list)
    norm_val_GT = []
    for LFP in val_LFPs:
        print(f'Processing Val Session. Shape: {LFP.shape}, SF: {sf[counter_sf]}')
        tmpLFP = process_LFP(LFP, ch=channels, sf=sf[counter_sf], new_sf=new_sf, use_zscore=False, use_band=use_band)
        if zscore:
            tmpLFP = (tmpLFP - np.mean(tmpLFP, axis=0)) / np.std(tmpLFP, axis=0)
        norm_val_GT.append(tmpLFP)
        counter_sf += 1
        
    return retrain_LFP, retrain_GT, norm_val_GT, val_GTs


# ==========================================
# 3. New Label Generation (Parametric)
# ==========================================

def create_parametric_labels(binary_labels, sampling_rate, params):
    """
    Generates Trapezoidal Regression Targets (0.0 to 1.0) for Onset Detection.
    
    Shape: Rise (Anticipation) -> Plateau (Certainty) -> Fall (Reset)
    
    Params keys used:
    - 'LABEL_RISE_MS': Duration of anticipation (0->1)
    - 'LABEL_PLATEAU_MS': Duration of certainty (1.0)
    - 'LABEL_FALL_MS': Duration of reset (1->0)
    - 'LABEL_RISE_POWER': Shape of rise (1=Linear, 2=Quadratic/SharpWave)
    """
    # Extract params with defaults tailored for SWR Onset
    rise_ms = params.get('LABEL_RISE_MS', 30)      # 30ms to see the Sharp Wave descent
    plateau_ms = params.get('LABEL_PLATEAU_MS', 12) # 12ms (~2 ripple cycles) of certainty
    fall_ms = params.get('LABEL_FALL_MS', 40)      # 40ms reset
    rise_power = params.get('LABEL_RISE_POWER', 2.0) # Quadratic ramp (fits sharp wave physics)
    
    # Convert to samples
    rise_samples = int((rise_ms / 1000) * sampling_rate)
    plateau_samples = int((plateau_ms / 1000) * sampling_rate)
    fall_samples = int((fall_ms / 1000) * sampling_rate)
    
    # Identify Onsets (0 -> 1 transitions)
    diffs = np.diff(binary_labels, prepend=0)
    onset_indices = np.where(diffs == 1)[0]
    
    targets = np.zeros_like(binary_labels, dtype=np.float32)
    n_points = len(targets)
    
    for onset in onset_indices:
        # 1. RISE PHASE (Anticipation)
        start_rise = max(0, onset - rise_samples)
        rise_len = onset - start_rise
        if rise_len > 0:
            t = np.linspace(0, 1, rise_len)
            # Parametric shape: t^power
            rise_curve = t ** rise_power
            targets[start_rise:onset] = np.maximum(targets[start_rise:onset], rise_curve)

        # 2. PLATEAU PHASE (Certainty)
        end_plateau = min(n_points, onset + plateau_samples)
        targets[onset:end_plateau] = 1.0

        # 3. FALL PHASE (Reset)
        start_fall = end_plateau
        end_fall = min(n_points, start_fall + fall_samples)
        fall_len = end_fall - start_fall
        if fall_len > 0:
            t = np.linspace(1, 0, fall_len) # Linear decay is standard
            targets[start_fall:end_fall] = np.maximum(targets[start_fall:end_fall], t)
            
    return targets


# ==========================================
# 4. Main Dataset Loader
# ==========================================

def rippleAI_load_dataset(params, mode='train', preprocess=True, use_band=None):
    """
    Loads dataset, generates parametric labels, and applies weighting.
    """
    # ---------------------------------------------------------
    # 1. Initialize & Load Raw Data
    # ---------------------------------------------------------
    
    # Extract Causal Shift if present in arch name (e.g., 'Shift20')
    sample_shift = 0
    if params['TYPE_ARCH'].find('Shift') > -1:
        try:
            shift_str = params['TYPE_ARCH'].split('Shift')[1][:2]
            shift_ms = int(shift_str)
            sample_shift = int((shift_ms / 1000) * params['SRATE'])
            print(f'Using Causal Shift: {shift_ms}ms ({sample_shift} samples)')
        except:
            print('Could not parse Shift value, using 0.')

    train_LFPs = []
    train_GTs = []
    all_SFs = []

    # -- Load Amigo2 --
    path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Amigo2','figshare_16847521')
    LFP, GT = load_lab_data(path)
    train_LFPs.append(LFP)
    train_GTs.append(GT)
    all_SFs.append(30000)

    # -- Load Som2 --
    path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Som2','figshare_16856137')
    LFP, GT = load_lab_data(path)
    train_LFPs.append(LFP)
    train_GTs.append(GT)
    all_SFs.append(30000)

    val_LFPs = []
    val_GTs = []
    
    if mode == 'test':
        # -- Load Dlx1 --
        path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Dlx1','figshare_14959449')
        LFP, GT = load_lab_data(path)
        val_LFPs.append(LFP)
        val_GTs.append(GT)
        all_SFs.append(30000)

        # -- Load Thy7 --
        path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Thy7','figshare_14960085')
        LFP, GT = load_lab_data(path)
        val_LFPs.append(LFP)
        val_GTs.append(GT)
        all_SFs.append(30000)

        # -- Load Calb20 --
        path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Calb20','figshare_20072252')
        LFP, GT = load_lab_data(path)
        val_LFPs.append(LFP)
        val_GTs.append(GT)
        all_SFs.append(30000)

    # ---------------------------------------------------------
    # 2. Process Raw Data
    # ---------------------------------------------------------
    train_data, train_labels_vec, val_data, val_labels_vec = rippleAI_prepare_training_data(
        train_LFPs, train_GTs, val_LFPs, val_GTs,
        sf=all_SFs, new_sf=params['SRATE'],
        zscore=preprocess, use_band=use_band
    )
    
    train_data = train_data.astype('float32')
    if mode == 'test':
        val_data = [k.astype('float32') for k in val_data]
        return val_data, val_labels_vec

    # Split into Train/Test chunks (e.g. 70/30 split)
    # Using 'sf' here as params['SRATE']
    test_examples, events_test, train_examples, events_train = split_data(
        train_data, train_labels_vec, sf=params['SRATE'], split=0.7
    )

    # ---------------------------------------------------------
    # 3. Generate Binary & Parametric Labels
    # ---------------------------------------------------------
    sf = params['SRATE']

    # Reconstruct Binary Ground Truth (Temporary, used for generation)
    y_train_binary = np.zeros(len(train_examples), dtype=np.float32)
    for event in events_train:
        y_train_binary[int(sf*event[0]):int(sf*event[1])] = 1

    y_test_binary = np.zeros(len(test_examples), dtype=np.float32)
    for event in events_test:
        y_test_binary[int(sf*event[0]):int(sf*event[1])] = 1

    # GENERATE PARAMETRIC TARGETS (Regression)
    print(f"Generating Parametric Targets...")
    print(f"  Rise: {params.get('LABEL_RISE_MS', 30)}ms (Power: {params.get('LABEL_RISE_POWER', 2)})")
    print(f"  Plateau: {params.get('LABEL_PLATEAU_MS', 12)}ms")
    
    train_labels = create_parametric_labels(y_train_binary, sf, params)
    test_labels  = create_parametric_labels(y_test_binary, sf, params)
    
    # ---------------------------------------------------------
    # 4. Generate Weights
    # ---------------------------------------------------------
    weights = np.ones_like(train_labels, dtype=np.float32)
    
    w_plateau = params.get('WEIGHT_PLATEAU', 20.0)    # Critical detection zone
    w_transition = params.get('WEIGHT_TRANSITION', 5.0) # Slope zone
    
    # Apply weights based on target value
    # Plateau (Target ~ 1.0)
    weights[train_labels > 0.95] = w_plateau
    # Transition (0 < Target < 0.95)
    weights[(train_labels > 0.0) & (train_labels <= 0.95)] = w_transition
    
    # Optional: Gap/Masking logic from original code can be added here if needed,
    # but for regression, continuous weighting is usually superior.

    # ---------------------------------------------------------
    # 5. Handle Causal Shift & Batching
    # ---------------------------------------------------------
    # If using Causal TCN, we shift X back or Y forward so the model predicts "future"
    if sample_shift > 0:
        train_examples = train_examples[:-sample_shift, :]
        train_labels = train_labels[sample_shift:]
        weights = weights[sample_shift:]
        test_examples = test_examples[:-sample_shift, :]
        test_labels = test_labels[sample_shift:]

    print('Train X shape:', train_examples.shape)
    print('Train Y (Parametric) shape:', train_labels.shape)
    print('Weights shape:', weights.shape)

    # Dataset Parameters
    sample_length = params['NO_TIMEPOINTS'] * 2
    stride_step = params['NO_STRIDES']
    
    # Logic for label sequence length (Architecture dependent)
    if params['TYPE_ARCH'].find('Patch') > -1:
        label_length = sample_length
        label_skip = 0
    else:
        # Standard TCN often predicts central point or last point
        label_length = sample_length // 2
        label_skip = int(sample_length / 2)

    # Helper to slice dataset
    def create_ts_dataset(data, seq_len, stride):
        return timeseries_dataset_from_array(
            data, None, sequence_length=seq_len, sequence_stride=stride,
            batch_size=None, shuffle=False
        )

    # Create Inputs (X)
    if params['TYPE_ARCH'].find('Patch') > -1:
        # Patch models usually take full windows
        train_x = create_ts_dataset(train_examples, sample_length, stride_step)
        test_x = create_ts_dataset(test_examples, sample_length, stride_step)
    else:
        # Standard models
        train_x = create_ts_dataset(train_examples, sample_length, stride_step)
        test_x = create_ts_dataset(test_examples, sample_length, stride_step)
        # Often mapped to take only recent history if causal, but TCN usually handles window internally
        train_x = train_x.map(lambda x: x[:, -params['NO_TIMEPOINTS']:, :])
        test_x  = test_x.map(lambda x: x[:, -params['NO_TIMEPOINTS']:, :])

    # Create Targets (Y)
    # We slice labels to align with the 'valid' output of the convolution
    y_start = label_skip
    
    train_y = create_ts_dataset(train_labels[y_start:].reshape(-1, 1), label_length, stride_step)
    test_y  = create_ts_dataset(test_labels[y_start:].reshape(-1, 1), label_length, stride_step)
    
    train_w = create_ts_dataset(weights[y_start:].reshape(-1, 1), label_length, stride_step)

    # ---------------------------------------------------------
    # 6. Combine & Finalize Datasets
    # ---------------------------------------------------------
    
    if params['TYPE_ARCH'].find('Only') > -1:
        # Not predicting LFPs, just labels
        train_c = tf.data.Dataset.zip((train_y, train_w))
        
        @tf.autograph.experimental.do_not_convert
        def concat_lfps_labels_weights(labels, weights):
            return tf.concat([labels, weights], axis=-1)
        
        train_d = train_c.map(lambda x, y: concat_lfps_labels_weights(x, y))
        test_d = test_y
        
    else:
        # Standard Path: Combine X, Y, W
        train_c = tf.data.Dataset.zip((train_x, train_y, train_w))

        @tf.autograph.experimental.do_not_convert
        def concat_inputs_labels_weights(inputs, labels, weights):
            # Input X is separate in the final tuple (X, Y)
            # This function prepares the Y part (Label + Weight)
            return tf.concat([labels, weights], axis=-1) 

        # Note: In your original code, train_d was concatenating x, y, z.
        # But typically Keras expects (x, y) or (x, y, w).
        # Assuming your model handles [Labels, Weights] as the target tensor 
        # via a custom loss function that unpacks them.
        train_d = train_c.map(lambda x, y, z: concat_inputs_labels_weights(x, y, z))
        
        test_c = tf.data.Dataset.zip((test_x, test_y))
        
        @tf.autograph.experimental.do_not_convert
        def concat_test_labels(inputs, labels):
             return labels # Just return labels for test (no weights needed for inference)
             
        test_d = test_c.map(lambda x, y: concat_test_labels(x, y))
        
        # Redefine train_dataset to be (X, Target)
        # where Target = [Labels, Weights]
        train_dataset = tf.data.Dataset.zip((train_x, train_d))
        test_dataset = tf.data.Dataset.zip((test_x, test_d))

    # Batching & Prefetching
    test_dataset = test_dataset.batch(params['BATCH_SIZE'], drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(params["SHUFFLE_BUFFER_SIZE"], reshuffle_each_iteration=True).batch(params['BATCH_SIZE']).prefetch(tf.data.experimental.AUTOTUNE)

    # Augmentations
    if params['TYPE_ARCH'].find('Aug') > -1:
        print('Using Augmentations...')
        train_dataset = apply_augmentation_to_dataset(train_dataset, params=params, sampling_rate=params['SRATE'])
    else:
        print('No augmentation')

    # Return ratio 0 because it's not relevant for regression
    return train_dataset, test_dataset, 0 

# ==========================================
# 5. Other Data Loaders (Allen / Topological)
# ==========================================

def load_allen(indeces=np.int32(np.linspace(49,62,8))):
    loaded_data_raw = np.load('/cs/projects/OWVinckSWR/Carmen/LFP_extracted/sanity_check/raw_lfp_fc.npy')
    indeces[::-1].sort()
    LFP = loaded_data_raw[:, indeces]
    data = process_LFP(LFP, sf=1250, channels=np.arange(0,8))
    return data

def load_topological_dataset(batch_size=32, shuffle_buffer=1000):
    import scipy.io as sio
    # Load both .mat files using h5py
    jp_data = h5py.File('/cs/projects/OWVinckSWR/Dataset/TopologicalData/JuanPabloDB_struct.mat', 'r')
    ab_data = h5py.File('/cs/projects/OWVinckSWR/Dataset/TopologicalData/AlbertoDB_struct.mat', 'r')

    def process_struct(data):
        ripples = data['ripples']  # nEvents x 127
        ripples_reshaped = np.tile(np.transpose(ripples, (1, 0))[:,:,np.newaxis], (1,1,8))
        features = np.column_stack([
            np.array(data['amplitude']).reshape(-1, 1),
            np.array(data['entropy']).reshape(-1, 1),
            np.array(data['duration']).reshape(-1, 1),
            np.array(data['frequency']).reshape(-1, 1)
        ])
        return ripples_reshaped, features

    jp_ripples, jp_features = process_struct(jp_data)
    ab_ripples, ab_features = process_struct(ab_data)

    all_ripples = np.concatenate([jp_ripples, ab_ripples], axis=0).astype(np.float32)
    all_features = np.concatenate([jp_features, ab_features], axis=0).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices({
        'ripples': all_ripples,
        'features': all_features
    })

    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
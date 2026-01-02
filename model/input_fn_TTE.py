import tensorflow.keras.backend as K
from model.cnn_ripple_utils import load_lab_data, process_LFP, filter_LFP, split_data, load_info, load_raw_data
import tensorflow as tf
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
from keras.utils import timeseries_dataset_from_array

# ==========================================
# 1. Helpers & Augmentation
# ==========================================

def compute_ttne(n_samples, events_seconds, sf):
    """
    Computes Time To Next Event (TTNE) in seconds.
    TTNE = Start Time of Next Event - Current Time
    Values are strictly positive and count down to 0.
    """
    # Default to 60s (infinity cap)
    ttne = np.full(n_samples, 60.0, dtype=np.float32) 
    t_vec = np.arange(n_samples) / sf
    
    if len(events_seconds) > 0:
        events_idx = (events_seconds * sf).astype(int)
        start_indices = events_idx[:, 0]
        end_indices = events_idx[:, 1]
        
        # 1. Before first event
        if start_indices[0] > 0:
            ttne[:start_indices[0]] = (start_indices[0] / sf) - t_vec[:start_indices[0]]
            
        # 2. Between events
        for i in range(len(start_indices) - 1):
            curr_end = end_indices[i]
            next_start = start_indices[i+1]
            
            # Only fill valid gaps
            if next_start > curr_end:
                # The countdown is: (Time of Next Start) - (Current Time)
                ttne[curr_end:next_start] = (next_start / sf) - t_vec[curr_end:next_start]

        # 3. Zero out during events (or keep 0)
        for i in range(len(start_indices)):
            s, e = start_indices[i], end_indices[i]
            if e > n_samples: e = n_samples
            ttne[s:e] = 0.0
            
    return np.maximum(ttne, 0).astype(np.float32)

def random_scaling(data, min_scale=0.8, max_scale=1.2):
    scale_factors = tf.random.uniform((1, 1, 8), min_scale, max_scale)
    return data * scale_factors

def augment_data(data, labels, event_indices=None, apply_mixup=False, mixup_data=None, params=None, sampling_rate=1250):
    if len(tf.shape(data)) < 3: data = tf.expand_dims(data, axis=-1)
    if len(tf.shape(labels)) < 2: labels = tf.expand_dims(labels, axis=-1)
    if tf.random.uniform([]) < 0.2:
        try: data = random_scaling(data)
        except: pass
    return data, labels

def apply_augmentation_to_dataset(dataset, params=None, sampling_rate=1250):
    def augment_batch(data, labels):
        augmented_data, updated_labels = augment_data(
            data, labels, params=params, sampling_rate=sampling_rate)
        return augmented_data, updated_labels
    return dataset.map(augment_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# ==========================================
# 2. Data Preparation
# ==========================================

def rippleAI_prepare_training_data(train_LFPs, train_GTs,
                                   val_LFPs,   val_GTs,
                                   sf=1250, new_sf=1250,
                                   channels=np.arange(0, 8),
                                   zscore=True, process_online=False, use_band=None):    
    assert len(train_LFPs) == len(train_GTs)
    assert len(val_LFPs) == len(val_GTs)

    counter_sf = 0
    retrain_LFP = []
    retrain_TTNE = [] 
    
    # Process Train
    for LFP, GT in zip(train_LFPs, train_GTs):
        print(f'Processing Train Session. Shape: {LFP.shape}')
        try:
             aux_LFP = process_LFP(LFP, ch=channels, sf=sf[counter_sf], new_sf=new_sf, 
                                   use_zscore=False, use_band=use_band, process_online=process_online)
        except TypeError:
             aux_LFP = process_LFP(LFP, ch=channels, sf=sf[counter_sf], new_sf=new_sf, 
                                   use_zscore=False, use_band=use_band)

        if zscore:
            aux_LFP = (aux_LFP - np.mean(aux_LFP, axis=0)) / np.std(aux_LFP, axis=0)
        
        # Compute TTNE
        aux_TTNE = compute_ttne(len(aux_LFP), GT, new_sf)

        if len(retrain_LFP) == 0:
            retrain_LFP = aux_LFP
            retrain_TTNE = aux_TTNE
            offset_sf = new_sf
            offset = len(retrain_LFP) / offset_sf
            retrain_GT = GT
        else:
            retrain_LFP = np.vstack([retrain_LFP, aux_LFP])
            retrain_TTNE = np.concatenate([retrain_TTNE, aux_TTNE])
            retrain_GT = np.vstack([retrain_GT, GT + offset])
            offset += len(aux_LFP) / offset_sf
            
        counter_sf += 1

    # Process Val
    norm_val_GT = []
    for LFP in val_LFPs:
        try:
             tmpLFP = process_LFP(LFP, ch=channels, sf=sf[counter_sf], new_sf=new_sf, 
                                  use_zscore=False, use_band=use_band, process_online=process_online)
        except TypeError:
             tmpLFP = process_LFP(LFP, ch=channels, sf=sf[counter_sf], new_sf=new_sf, 
                                  use_zscore=False, use_band=use_band)

        if zscore:
            tmpLFP = (tmpLFP - np.mean(tmpLFP, axis=0)) / np.std(tmpLFP, axis=0)
        norm_val_GT.append(tmpLFP)
        counter_sf += 1
        
    return retrain_LFP, retrain_GT, norm_val_GT, val_GTs, retrain_TTNE

# ==========================================
# 3. Label Logic
# ==========================================

def create_parametric_labels(binary_labels, sampling_rate, params):
    rise_ms = params.get('LABEL_RISE_MS', 30)
    plateau_ms = params.get('LABEL_PLATEAU_MS', 12)
    fall_ms = params.get('LABEL_FALL_MS', 40)
    rise_power = params.get('LABEL_RISE_POWER', 2.0)
    
    rise_samples = int((rise_ms / 1000) * sampling_rate)
    plateau_samples = int((plateau_ms / 1000) * sampling_rate)
    fall_samples = int((fall_ms / 1000) * sampling_rate)
    
    diffs = np.diff(binary_labels, prepend=0)
    onset_indices = np.where(diffs == 1)[0]
    
    targets = np.zeros_like(binary_labels, dtype=np.float32)
    n_points = len(targets)
    
    for onset in onset_indices:
        start_rise = max(0, onset - rise_samples)
        if onset > start_rise:
            t = np.linspace(0, 1, onset - start_rise)
            targets[start_rise:onset] = np.maximum(targets[start_rise:onset], t ** rise_power)
            
        end_plateau = min(n_points, onset + plateau_samples)
        targets[onset:end_plateau] = 1.0
        
        start_fall = end_plateau
        end_fall = min(n_points, start_fall + fall_samples)
        if end_fall > start_fall:
            t = np.linspace(1, 0, end_fall - start_fall)
            targets[start_fall:end_fall] = np.maximum(targets[start_fall:end_fall], t)
    return targets

def create_weights(labels, params):
    w_plateau = params.get('WEIGHT_PLATEAU', 20.0)
    w_transition = params.get('WEIGHT_TRANSITION', 5.0)
    weights = np.ones_like(labels, dtype=np.float32)
    weights[labels > 0.95] = w_plateau
    weights[(labels > 0.0) & (labels <= 0.95)] = w_transition
    return weights

# ==========================================
# 4. Main Loader
# ==========================================

def rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=False, use_band=None):
    if params['TYPE_ARCH'].find('Shift') > -1:
        try:
            shift_str = params['TYPE_ARCH'].split('Shift')[1][:2]
            sample_shift = int((int(shift_str)/1000)*params['SRATE'])
            print(f"Using Shift: {sample_shift}")
        except: sample_shift = 0
    else: sample_shift = 0

    # Load Data
    train_LFPs = []; train_GTs = []; all_SFs = []
    path1 = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Amigo2','figshare_16847521')
    LFP1, GT1 = load_lab_data(path1); train_LFPs.append(LFP1); train_GTs.append(GT1); all_SFs.append(30000)

    path2 = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Som2','figshare_16856137')
    LFP2, GT2 = load_lab_data(path2); train_LFPs.append(LFP2); train_GTs.append(GT2); all_SFs.append(30000)

    val_LFPs=[]; val_GTs=[]
    if mode == 'test':
        path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Dlx1','figshare_14959449')
        LFP, GT = load_lab_data(path); val_LFPs.append(LFP); val_GTs.append(GT); all_SFs.append(30000)
        
        path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Thy7','figshare_14960085')
        LFP, GT = load_lab_data(path); val_LFPs.append(LFP); val_GTs.append(GT); all_SFs.append(30000)

        path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Calb20','figshare_20072252')
        LFP, GT = load_lab_data(path); val_LFPs.append(LFP); val_GTs.append(GT); all_SFs.append(30000)

    train_data, train_labels_vec, val_data, val_labels_vec, train_ttne_full = rippleAI_prepare_training_data(
        train_LFPs, train_GTs, val_LFPs, val_GTs,
        sf=all_SFs, new_sf=params['SRATE'],
        zscore=preprocess, process_online=process_online, use_band=use_band
    )
    
    train_data = train_data.astype('float32')
    if mode == 'test':
        val_data = [k.astype('float32') for k in val_data]
        return val_data, val_labels_vec

    # -------------------------------------------------------------------------
    # VISUALIZATION (Whole Session & Distribution)
    # -------------------------------------------------------------------------
    print("Generating Visualizations...")
    sf = params['SRATE']
    
    # 1. Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(train_ttne_full[::100], bins=100, color='teal', alpha=0.7, log=True)
    plt.title('Distribution of Time-To-Next-Event (Training Set)')
    plt.xlabel('Seconds to Next Ripple'); plt.ylabel('Count (Log Scale)')
    plt.grid(True, alpha=0.3); plt.savefig('ttne_histogram.png'); print("Saved ttne_histogram.png")

    # 2. Whole Session Trace
    temp_binary = np.zeros(len(train_data), dtype=np.float32)
    for event in train_labels_vec:
        start = int(sf * event[0]); end = int(sf * event[1])
        if start < len(temp_binary): temp_binary[start:end] = 1.0
    temp_labels = create_parametric_labels(temp_binary, sf, params)
    temp_weights = create_weights(temp_labels, params)
    
    ds = 10 
    t_axis = np.arange(len(temp_binary))[::ds] / sf
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(t_axis, temp_labels[::ds], 'b', linewidth=1); axes[0].set_title('Target Label (Proximity)')
    axes[1].plot(t_axis, temp_weights[::ds], 'orange', linewidth=1); axes[1].set_title('Weights')
    axes[2].plot(t_axis, train_ttne_full[::ds], 'g', linewidth=1); axes[2].set_title('Time To Next Event (s)')
    axes[2].set_xlabel('Time (s)')
    pdb.set_trace()
    
    plt.tight_layout(); plt.savefig('whole_session_vis.png'); print("Saved whole_session_vis.png")
    # -------------------------------------------------------------------------

    # Split Data
    test_examples, events_test, train_examples, events_train = split_data(train_data, train_labels_vec, sf=params['SRATE'], split=0.7)
    test_ttne, _, train_ttne, _ = split_data(train_ttne_full.reshape(-1, 1), train_labels_vec, sf=params['SRATE'], split=0.7)

    # Generate Batched Labels
    y_train_binary = np.zeros(len(train_examples), dtype=np.float32)
    for event in events_train: y_train_binary[int(sf*event[0]):int(sf*event[1])] = 1
    
    y_test_binary = np.zeros(len(test_examples), dtype=np.float32)
    for event in events_test: y_test_binary[int(sf*event[0]):int(sf*event[1])] = 1

    train_labels = create_parametric_labels(y_train_binary, sf, params)
    test_labels  = create_parametric_labels(y_test_binary, sf, params)
    weights = create_weights(train_labels, params)

    if sample_shift > 0:
        train_examples = train_examples[:-sample_shift]; train_labels = train_labels[sample_shift:]
        train_ttne = train_ttne[sample_shift:]; weights = weights[sample_shift:]
        test_examples = test_examples[:-sample_shift]; test_labels = test_labels[sample_shift:]

    print('Train X:', train_examples.shape)
    print('Train Y:', train_labels.shape)
    print('Weights:', weights.shape)

    # Dataset Creation
    sample_length = params['NO_TIMEPOINTS'] * 2
    stride_step = params['NO_STRIDES']
    if params['TYPE_ARCH'].find('Patch') > -1: label_length = sample_length; label_skip = 0
    else: label_length = sample_length // 2; label_skip = int(sample_length / 2)

    def create_ds(data, seq_len):
        return timeseries_dataset_from_array(data, None, sequence_length=seq_len, sequence_stride=stride_step, batch_size=None, shuffle=False)

    train_x = create_ds(train_examples, sample_length)
    test_x = create_ds(test_examples, sample_length)
    
    y_start = label_skip
    train_y = create_ds(train_labels[y_start:].reshape(-1,1), label_length)
    test_y  = create_ds(test_labels[y_start:].reshape(-1,1), label_length)
    train_w = create_ds(weights[y_start:].reshape(-1,1), label_length)
    
    # Create TTNE Datasets
    train_t = create_ds(train_ttne[y_start:].reshape(-1,1), label_length)
    test_t  = create_ds(test_ttne[y_start:].reshape(-1,1), label_length)

    if params['TYPE_ARCH'].find('Only') > -1:
        test_d = test_y 
        # Train Y = [Label, Weight, TTNE]
        train_c = tf.data.Dataset.zip((train_y, train_w, train_t))
        train_d = train_c.map(lambda l, w, t: tf.concat([l, w, t], axis=-1))
        test_d = test_y
    else:
        # Standard
        train_xy = train_x; test_xy = test_x

        test_c = tf.data.Dataset.zip((test_xy, test_y))
        
        if params['TYPE_ARCH'].find('Patch') > -1:
             train_c = tf.data.Dataset.zip((train_xy, train_y))
             train_d = train_c.map(lambda x, y: tf.concat([x, y], axis=-1))
        else:
             train_c = tf.data.Dataset.zip((train_xy, train_y, train_w, train_t))
             # Target = [Label, Weight, TTNE]
             # We include TTNE so you can monitor regression error (in seconds) during training if you want
             train_d = train_c.map(lambda x, l, w, t: tf.concat([l, w, t], axis=-1))

        test_d = test_c.map(lambda x, y: y)

    train_dataset = tf.data.Dataset.zip((train_x, train_d))
    test_dataset = tf.data.Dataset.zip((test_x, test_d))

    train_dataset = train_dataset.shuffle(params["SHUFFLE_BUFFER_SIZE"], reshuffle_each_iteration=True).batch(params['BATCH_SIZE']).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(params['BATCH_SIZE'], drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    if params['TYPE_ARCH'].find('Aug') > -1:
        print('Using Augmentations...')
        train_dataset = apply_augmentation_to_dataset(train_dataset, params=params, sampling_rate=params['SRATE'])

    return train_dataset, test_dataset, 0
import tensorflow.keras.backend as K
from model.cnn_ripple_utils import load_lab_data, process_LFP, split_data
import tensorflow as tf
import numpy as np
import pdb
import os
from keras.utils import timeseries_dataset_from_array

def rippleAI_prepare_training_data(train_LFPs,train_GTs,val_LFPs,val_GTs,sf=30000,channels=np.arange(0,8)):
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

    # All the training sessions data and GT will be concatenated in one data array and one GT array (2 x n events)
    retrain_LFP=[]
    for LFP,GT in zip(train_LFPs,train_GTs):
        # pdb.set_trace()
        # 1st session in the array
        print('Original training data shape: ',LFP.shape)
        if len(retrain_LFP)==0:
            retrain_LFP=process_LFP(LFP,sf,channels)
            offset=len(retrain_LFP)/1250
            retrain_GT=GT
        # Append the rest of the sessions, taking into account the length (in seconds) 
        # of the previous sessions, to cocatenate the events' times
        else:
            aux_LFP=process_LFP(LFP,sf,channels)
            retrain_LFP=np.vstack([retrain_LFP,aux_LFP])
            retrain_GT=np.vstack([retrain_GT,GT+offset])
            offset+=len(aux_LFP)/1250
    # Each validation session LFP will be normalized, etc and stored in an array
    #  the GT needs no further treatment
    norm_val_GT=[]
    for LFP in val_LFPs:
        print('Original validation data shape: ',LFP.shape)
        norm_val_GT.append(process_LFP(LFP,sf,channels))
    return retrain_LFP, retrain_GT , norm_val_GT, val_GTs


def rippleAI_load_dataset(params, mode='train'):
    """
    Loads the dataset for the Ripple AI model.

    Returns:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
    """
    
    # The training sessions will be appended together. Do the same with your training data
    train_LFPs=[]
    train_GTs=[]
    
    # Amigo2
    path=os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Amigo2','figshare_16847521')
    LFP,GT=load_lab_data(path)
    train_LFPs.append(LFP)
    train_GTs.append(GT)
    
    # Som2
    path=os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Som2','figshare_16856137')
    LFP,GT=load_lab_data(path)
    train_LFPs.append(LFP)
    train_GTs.append(GT)
    
    ## Append all your validation sessions
    val_LFPs=[]
    val_GTs=[]
    
    # Dlx1 Validation
    path=os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Dlx1','figshare_14959449')
    LFP,GT=load_lab_data(path)
    val_LFPs.append(LFP)
    val_GTs.append(GT)
    
    # Thy07 Validation
    path=os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/','Downloaded_data','Thy7','figshare_14960085')
    LFP,GT=load_lab_data(path)
    val_LFPs.append(LFP)
    val_GTs.append(GT)
    
    train_data, train_labels_vec, val_data, val_labels_vec = rippleAI_prepare_training_data(train_LFPs,train_GTs,val_LFPs,val_GTs)
    
    if mode == 'test':
        return val_data, val_labels_vec
    
    test_examples, events_test, train_examples, events_train = split_data(train_data, train_labels_vec, split=0.7)

    # pdb.set_trace()
    # fix labels
    sf = 1250
    y = np.zeros(shape=len(train_examples))
    for event in events_train:
        y[int(sf*event[0]):int(sf*event[1])] = 1
    train_labels = y
    label_ratio = np.sum(train_labels)/len(train_labels)
    
    y = np.zeros(shape=len(test_examples))
    for event in events_test:
        y[int(sf*event[0]):int(sf*event[1])] = 1
    test_labels = y
    
    from scipy import signal
    M = 51
    # onsets = np.diff(train_labels)==1
    onsets = np.hstack((0, np.diff(train_labels))).astype(np.uint32)==1
    assert(np.unique(train_labels[np.where(onsets)[0]])==1)
    assert(np.unique(train_labels[np.where(onsets)[0]]-1)==0)
    
    
    if params['TYPE_LOSS'].find('AnchorNarrow')>-1: 
        print('Using AnchorNarrow Weights')
        weights = signal.convolve(onsets, signal.windows.exponential(M, 0, 3, False))+0.01
        # weights = signal.convolve(onsets, signal.windows.exponential(M, 0, 5, False))+0.1
    elif params['TYPE_LOSS'].find('AnchorWide')>-1:
        print('Using AnchorWide Weights')
        weights = signal.convolve(onsets, signal.windows.exponential(M, 0, 10, False))+0.5
    elif params['TYPE_LOSS'].find('AnchorWider')>-1:
        print('Using AnchorWider Weights')
        weights = signal.convolve(onsets, signal.windows.exponential(M, 0, 20, False))+0.8
    elif params['TYPE_LOSS'].find('Focal')>-1:
        print('Using Focal Weights (Ones)')
        weights = np.ones(train_labels.shape, dtype=np.float32)
    else:
        print('Using Else (Ones)')
        weights = np.ones(train_labels.shape, dtype=np.float32)
    weights /= np.max(weights)
    # pdb.set_trace()
    # make a gap in the weights
    if params['TYPE_LOSS'].find('Gap')>-1:
        print('Using Gap Before Onset')
        onset_indices = np.where(onsets)[0]
        for onset in onset_indices:
            # if np.any(train_labels[onset-40:onset]==1):
            weights[onset-40:onset] *= train_labels[onset-40:onset]
    weights = weights[:train_labels.shape[0]]

    assert(np.unique(weights[np.where(onsets)[0]])==1)
    assert(np.unique(weights[np.where(onsets)[0]]-1)==0)

    # make batches 
    # pdb.set_trace()
    if params['TYPE_ARCH'].find('Shift')>-1:
        print('Using Shift')
        sample_shift = int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Shift')+5:params['TYPE_ARCH'].find('Shift')+7])
        print(sample_shift)
        if sample_shift>0:
            train_examples = train_examples[:-sample_shift, :]
            train_labels = train_labels[sample_shift:]
            weights = weights[sample_shift:]
            test_examples = test_examples[:-sample_shift, :]
            test_labels = test_labels[sample_shift:]        
    else:
        sample_shift = 0

    # pdb.set_trace()
    print(train_examples.shape)
    print(train_labels.shape)
    print(weights.shape)
    print(test_examples.shape)
    print(test_labels.shape)
    
    # make datasets
    sample_length = params['NO_TIMEPOINTS']*2
    stride_step = sample_length/params['NO_STRIDES']
    train_x = timeseries_dataset_from_array(train_examples, None, sequence_length=sample_length, sequence_stride=sample_length/stride_step, batch_size=params["BATCH_SIZE"])
    train_y = timeseries_dataset_from_array(train_labels[int(sample_length/2)+sample_shift:].reshape(-1,1), None, sequence_length=sample_length/stride_step, sequence_stride=sample_length/2, batch_size=params["BATCH_SIZE"])
    train_w = timeseries_dataset_from_array(weights[int(sample_length/2)+sample_shift:].reshape(-1,1), None, sequence_length=sample_length/stride_step, sequence_stride=sample_length/2, batch_size=params["BATCH_SIZE"])
    
    test_x = timeseries_dataset_from_array(test_examples, None, sequence_length=sample_length, sequence_stride=sample_length/stride_step, batch_size=params["BATCH_SIZE"])
    test_y = timeseries_dataset_from_array(test_labels[int(sample_length/2)+sample_shift:].reshape(-1,1), None, sequence_length=sample_length/stride_step, sequence_stride=sample_length/2, batch_size=params["BATCH_SIZE"])
    
    
    # zip datasets
    train_dataset = tf.data.Dataset.zip((train_x, train_y, train_w))#.prefetch(8)#.batch(params["BATCH_SIZE"])
    train_dataset = tf.data.Dataset.zip((train_x, train_y))#.batch(params["BATCH_SIZE"])
    test_dataset = tf.data.Dataset.zip((test_x, test_y))#.prefetch(8)#.batch(params["BATCH_SIZE"])
    train_dataset = train_dataset.shuffle(params["SHUFFLE_BUFFER_SIZE"], reshuffle_each_iteration=True)#.prefetch(8)
    
    # pdb.set_trace()
    return train_dataset, test_dataset, label_ratio#, val_dataset


def load_allen(indeces= np.int32(np.linspace(49,62,8))):
    loaded_data_raw = np.load('/cs/projects/OWVinckSWR/Carmen/LFP_extracted/sanity_check/raw_lfp_fc.npy')
    indeces[::-1].sort()
    LFP = loaded_data_raw[:,indeces]
    # Process LFP
    data = process_LFP(LFP, sf = 1250, channels=np.arange(0,8))
    return data
import tensorflow.keras.backend as K
from model.cnn_ripple_utils import load_lab_data, process_LFP, split_data
import tensorflow as tf
import numpy as np
import pdb
import os


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
    onsets = np.diff(train_labels)==1
    # onsets = np.hstack((np.diff(train_labels), 0))==1
    weights = signal.convolve(onsets, signal.exponential(M, 0, 5, False))+0.1
    # weights = signal.convolve(onsets, signal.exponential(M, 0, 10, False))+0.1
    weights /= np.max(weights)
    weights = np.hstack((0, weights))
    # pdb.set_trace()
    onset_indices = np.where(onsets)[0]
    for onset in onset_indices:
        weights[onset-50:onset+1] = 0
        # weights[onset-50:onset+1] = 1e-3
    # import matplotlib.pyplot as plt
    # plt.plot(weights)
    # plt.plot(train_labels)
    # plt.show()
    # pdb.set_trace()
    # y_train=np.zeros(shape=[x_train.shape[0],1])
    # for i in range(y_train_aux.shape[0]):
    #     y_train[i]=1  if any (y_train_aux[i]==1) else 0
    # print("Train Input and Output dimension", x_train.shape,y_train.shape)
    
    # y_test=np.zeros(shape=[x_test.shape[0],1])
    # for i in range(y_test_aux.shape[0]):
    #     y_test[i]=1  if any (y_test_aux[i]==1) else 0
    # y = np.zeros(shape=len(val_data))
    # for event in val_labels_vec:
    #     y[int(sf*event[0]):int(sf*event[1])] = 1
    # val_labels = y
    n_cut = params["NO_TIMEPOINTS"]*params["NO_CHANNELS"]
    train_examples = train_examples[:len(train_examples)-len(train_examples)%params["NO_TIMEPOINTS"], :].reshape(-1,params["NO_TIMEPOINTS"],params["NO_CHANNELS"])
    test_examples = test_examples[:len(test_examples)-len(test_examples)%params["NO_TIMEPOINTS"], :].reshape(-1,params["NO_TIMEPOINTS"],params["NO_CHANNELS"])
    train_labels = train_labels[:len(train_labels)-len(train_labels)%params["NO_TIMEPOINTS"]].reshape(-1,params["NO_TIMEPOINTS"], 1)
    test_labels = test_labels[:len(test_labels)-len(test_labels)%params["NO_TIMEPOINTS"]].reshape(-1,params["NO_TIMEPOINTS"], 1)
    train_weights = weights[:len(weights)-len(weights)%params["NO_TIMEPOINTS"]].reshape(-1,params["NO_TIMEPOINTS"], 1)
    # train_labels = train_labels.max(axis=1)
    # test_labels = test_labels.max(axis=1)
    
    # pdb.set_trace()
    # make datasetstrain_weights, 
    # inputs = tf.data.Dataset.from_tensor_slices(train_examples)
    # weights = tf.data.Dataset.from_tensor_slices(train_weights)
    # X = tf.data.Dataset.zip((inputs, weights)).map(lambda x1, x2: {'inputs': x1, 'weights': x2})
    # train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
    # train_dataset = tf.data.Dataset.zip((X, train_labels))
    # pdb.set_trace()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels, train_weights))
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
    # val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    train_dataset = train_dataset.shuffle(params["SHUFFLE_BUFFER_SIZE"]).batch(params["BATCH_SIZE"])
    # train_dataset = train_dataset.map(lambda x: tf.reshape(x,axis=[-1,40,8]))
    test_dataset = test_dataset.batch(params["BATCH_SIZE"])
    return train_dataset, test_dataset, label_ratio#, val_dataset


def load_allen(indeces= np.int32(np.linspace(49,62,8))):
    loaded_data_raw = np.load('/cs/projects/OWVinckSWR/Carmen/LFP_extracted/sanity_check/raw_lfp_fc.npy')
    indeces[::-1].sort()
    LFP = loaded_data_raw[:,indeces]
    # Process LFP
    data = process_LFP(LFP, sf = 1250, channels=np.arange(0,8))
    return data
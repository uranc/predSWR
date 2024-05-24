import os
import pdb
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks as cb
from shutil import copyfile
import argparse
import copy
from os import path
import shutil
from scipy.stats import pearsonr 
from scipy.io import loadmat


parser = argparse.ArgumentParser(
    description='Example 3 - Local and Parallel Execution.')
# parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
parser.add_argument('--model', type=str, nargs=1,
                    help='model name ie. l9:experiments/l9', default='testSWR')
parser.add_argument('--mode', type=str, nargs=1,
                    help='mode training/predict', default='train')

args = parser.parse_args()
mode = args.mode[0]
model_name = args.model[0]

# Parameters
params = {'BATCH_SIZE': 64, 'SHUFFLE_BUFFER_SIZE': 4096, 
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-3, 'NO_EPOCHS': 200,
          'NO_TIMEPOINTS': 50, 'NO_CHANNELS': 8,
          'EXP_DIR': '/cs/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
          }

if mode == 'train':

    from model.model_fn import build_DBI_TCN
    # pdb.set_trace()
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()
    
    # input
    from model.input_fn import rippleAI_load_dataset
    train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params)
    train_size = len(list(train_dataset))
    
    params['RIPPLE_RATIO'] = label_ratio
    # model
    # from model.model_fn import build_Prida_LSTM
    # model = build_Prida_LSTM([params["NO_TIMEPOINTS"],params["NO_CHANNELS"]])
  
    
    # [traces, labels] = next(iter(train_dataset))
    # pdb.set_trace()
    # # for traces, labels in next(iter(train_dataset)):
    
    # train 
    from model.training import train_pred
    
    # pdb.set_trace()
    hist = train_pred(model, train_dataset, test_dataset, params['NO_EPOCHS'], params['EXP_DIR'])
elif mode == 'predict':

    # modelname
    model = args.model[0]
    model_name = model
    import importlib
    
    # get model
    a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
    build_DBI_TCN = getattr(a_model, 'build_DBI_TCN')
    # from model.model_fn import build_DBI_TCN
    
    params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()
        
    # from model.model_fn import build_DBI_TCN
    # # pdb.set_trace()
    # model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    # model.summary()

    
    # pdb.set_trace()
    # noise = np.random.rand(1024,8).reshape(int(1024/32),32,8)
    # probs = model.predict(noise)
    
    # get inputs
    # a_input = importlib.import_module('experiments.{0}.model.input_fn'.format(model))
    # rippleAI_load_dataset = getattr(a_input, 'rippleAI_load_dataset')
    from model.input_fn import rippleAI_load_dataset
    val_datasets, val_labels = rippleAI_load_dataset(params, mode='test')
    
    from model.cnn_ripple_utils import get_predictions_index, get_performance
    # get predictions
    val_pred = []
    th_arr=np.linspace(0.1,0.9,10)
    n_channels = params['NO_CHANNELS']
    timesteps = params['NO_TIMEPOINTS']
    # val_datasets[0] = val_datasets[0][89500:90500-500,:]
    # val_datasets[1] = val_datasets[1][:100,:]
    val_datasets = [val_datasets[0]]
    test_end = val_datasets[0].shape[0]
    val_batches = []
    for i in range(0, test_end - timesteps, 1):
        val_batches.append(val_datasets[0][np.arange(i, i + timesteps), :])
    val_batches = np.array(val_batches)
    for LFP in [val_datasets[0]]:
        # test_end = LFP.shape[0]
        # LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels)
        # aa = []
        # for i in range(0, test_end - timesteps, 1):
        windowed_signal = np.squeeze(model.predict(val_batches, verbose=1))
            # aa.append(windowed_signal)
        # probs = np.hstack(aa)
        probs = np.hstack(windowed_signal)
        val_pred.append(probs)

    pdb.set_trace()
    # Validation plot in the second ax
    all_pred_events = []
    F1_val=np.zeros(shape=(len(val_datasets),len(th_arr)))
    for j,pred in enumerate(val_pred):
        tmp_pred = []
        for i,th in enumerate(th_arr):
            pred_val_events=get_predictions_index(pred,th)/1250
            _,_,F1_val[j,i],_,_,_=get_performance(pred_val_events,val_labels[j],verbose=False)
            tmp_pred.append(pred_val_events)
        all_pred_events.append(tmp_pred)

    # pick model
    print(F1_val[0])
    # pdb.set_trace()
    mind = np.argmax(F1_val[0])
    # print(all_pred_events[0][mind])
    best_preds = all_pred_events[0][mind]
    pred_vec = np.zeros(val_datasets[0].shape[0])
    label_vec = np.zeros(val_datasets[0].shape[0])
        
    for pred in best_preds:
        pred_vec[int(pred[0]*1250):int(pred[1]*1250)] = 1
        
    for lab in val_labels[0]:
        label_vec[int(lab[0]*1250):int(lab[1]*1250)] = 0.9
    
    import matplotlib.pyplot as plt
    for pred in best_preds:
        rip_begin = int(pred[0]*1250)
        plt.plot(val_datasets[0][rip_begin-128:rip_begin+128, :])
        plt.plot(val_pred[0][rip_begin-128:rip_begin+128], 'k')
        plt.plot(val_pred[0][rip_begin-128:rip_begin+128]*pred_vec[rip_begin-128:rip_begin+128], 'r')
        plt.plot(label_vec[rip_begin-128:rip_begin+128], 'k')
        plt.show()
    
elif mode == 'predictSynth':
    
    # modelname
    model = args.model[0]
    model_name = model
    import importlib
    
    # # get model
    a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
    build_DBI_TCN = getattr(a_model, 'build_DBI_TCN')
    # from model.model_fn import build_DBI_TCN
    
    params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()
    
    synth = np.load('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/synth_stim.npy')
    synth = np.tile(synth, (1,8))
    
    train_end = synth.shape[0]
    timesteps = 50
    synth_batches = []
    for i in range(0, train_end - timesteps, 1):
        synth_batches.append(synth[np.arange(i, i + timesteps), :])
    synth_batches = np.array(synth_batches)
    # get predictions
    n_channels = params['NO_CHANNELS']
    timesteps = params['NO_TIMEPOINTS']
    # synth=synth[:len(synth)-len(synth)%timesteps,:].reshape(-1,timesteps,n_channels)
    # pdb.set_trace()
    # synth=np.expand_dims(synth, axis=0)
    windowed_signal = np.squeeze(model.predict(synth_batches,verbose=1))
    probs = np.hstack(windowed_signal)
    # pdb.set_trace()
    from scipy.signal import decimate
    import matplotlib.pyplot as plt
    synth = np.load('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/synth_stim.npy')
    synth = (synth-np.min(synth))/(np.max(synth)-np.min(synth))
    tt = np.arange(synth.shape[0])/1250
    # pdb.set_trace()
    # plt.plot(decimate(tt, 4), decimate(synth[:, 0], 4))
    plt.plot(tt, synth[:, 0])
    tt = np.arange(probs.shape[0])/1250
    # plt.plot(decimate(tt,4), decimate(probs, 4))
    plt.plot(tt, probs)
    plt.show()
    pdb.set_trace()
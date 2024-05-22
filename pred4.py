# Compare different TCNs
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
import matplotlib.pyplot as plt
import tensorflow.keras as kr


parser = argparse.ArgumentParser(
    description='Example 3 - Local and Parallel Execution.')
parser.add_argument('--model', type=str, nargs=3,
                    help='model name ie. l9:experiments/l9', default='testSWR')
parser.add_argument('--mode', type=str, nargs=1,
                    help='mode training/predict', default='train')

args = parser.parse_args()
mode = args.mode[0]
fs = 1250
models = []
model_names = []
save_prob = 'False'
number_of_models = 3
for i in range(number_of_models):
    models.append(args.model[i])
    model_names.append(args.model[i])


if mode == 'train':
    params = {'BATCH_SIZE': 64, 'SHUFFLE_BUFFER_SIZE': 4096, 
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-4, 'NO_EPOCHS': 500,
          'NO_TIMEPOINTS': 40, 'NO_CHANNELS': 8,
          }
    # input
    from model.input_fn import rippleAI_load_dataset
    train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params)
    train_size = len(list(train_dataset))
    
    params['RIPPLE_RATIO'] = label_ratio
    # model
    # from model.model_fn import build_Prida_LSTM
    # model = build_Prida_LSTM([params["NO_TIMEPOINTS"],params["NO_CHANNELS"]])
    from model.model_fn import build_DBI_TCN
    # pdb.set_trace()
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()
    
    # [traces, labels] = next(iter(train_dataset))
    # # for traces, labels in next(iter(train_dataset)):
    
    # train 
    from model.training import train_pred
    
    # pdb.set_trace()
    hist = train_pred(model, train_dataset, test_dataset, params['NO_EPOCHS'], params['EXP_DIR'])

elif mode == 'predict':
    import importlib
    colors = ['red', 'aqua', 'lime'] 
    # Parameters
    params = {'BATCH_SIZE': 64, 'SHUFFLE_BUFFER_SIZE': 4096, 
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-4, 'NO_EPOCHS': 500,
          'NO_TIMEPOINTS': 40, 'NO_CHANNELS': 8,
          }
    from model.input_fn import rippleAI_load_dataset
    from model.cnn_ripple_utils import get_predictions_index, get_performance
    val_datasets, val_labels = rippleAI_load_dataset(params, mode='test')
        
    aux_probs = []
    for model, model_name in zip(models, model_names):
        params = {'BATCH_SIZE': 64, 'SHUFFLE_BUFFER_SIZE': 4096, 
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-4, 'NO_EPOCHS': 500,
          'NO_TIMEPOINTS': 40, 'NO_CHANNELS': 8,
          'EXP_DIR': '/cs/projects/OWVinckSWR/DL/predSWR/experiments/' + model_name,
          }
        tf.keras.backend.clear_session()
        # # get model
        a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
        build_DBI_TCN = getattr(a_model, 'build_DBI_TCN')
        # from model.model_fn import build_DBI_TCN
        
        params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
        model.summary()
        

        # get predictions
        val_pred = []
        th_arr=np.linspace(0.1,0.9,19)
        n_channels = params['NO_CHANNELS']
        timesteps = params['NO_TIMEPOINTS']
        for LFP in val_datasets:
            LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels)
            windowed_signal = np.squeeze(model.predict(LFP,verbose=1))
            probs = np.hstack(windowed_signal)
            val_pred.append(probs)

        aux_probs.append(val_pred)
        pred_vec = np.zeros(val_datasets[0].shape[0])
        label_vec = np.zeros(val_datasets[0].shape[0])
            
        for lab in val_labels[0]:
            label_vec[int(lab[0]*fs):int(lab[1]*fs)] = 1 
                
        if save_prob == 'True' :
            # Saving the probabilities
            directory =  '/cs/projects/OWVinckSWR/DL/predSWR/experiments/probabilities/' 
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Saving the probabilities in a .npy
            for i, array in enumerate(val_pred):
                np.save(directory + 'prob_{0}_{1}.npy'.format(model_name, i), np.array(array))
        
            # Saving LABELS
            directory =  '/cs/projects/OWVinckSWR/DL/predSWR/experiments/probabilities/' 
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Saving the probabilities in a .npy
            np.save(directory + 'val_labels_0.npy', np.array(label_vec))
            label_vec_1 = np.zeros(val_datasets[1].shape[0])
            for lab in val_labels[1]:
                label_vec_1[int(lab[0]*fs):int(lab[1]*fs)] = 1 
            np.save(directory + 'val_labels_1.npy', np.array(label_vec))    

    
    # Saving images   
    directory =  '/cs/projects/OWVinckSWR/DL/predSWR/TCN_comparison/{0}/'.format(model_names)   
    if not os.path.exists(directory):
        os.makedirs(directory)
    # # Prediction centered in the detected ripple
    for pred in val_labels[0]:
        rip_begin = int(pred[0]*fs)
        plt.plot(val_datasets[0][rip_begin-128:rip_begin+128, :]) #LFP
        plt.plot(label_vec[rip_begin-128:rip_begin+128], 'k',  linewidth=2, label = 'GT') #Labels of DS
        for i,color in zip(range(number_of_models), colors):
            plt.plot(aux_probs[i][0][rip_begin-128:rip_begin+128], color, linewidth=1.5,label = 'prob_{0}'.format(model_names[i]))
        plot_filename = os.path.join(directory, f'plot_{rip_begin}.png')
        plt.legend(fontsize=6)
        plt.xlabel('Frequency samples')
        plt.title(f'{model_names}',fontsize=5)
        plt.savefig(plot_filename)
        plt.close()
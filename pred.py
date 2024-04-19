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
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-4, 'NO_EPOCHS': 250,
          'NO_TIMEPOINTS': 32, 'NO_CHANNELS': 8,
          'EXP_DIR': '/cs/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
          }

if mode == 'train':

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
    # print(traces.shape)
    # print(labels.shape)
    # print(labels)
    # pdb.set_trace()
    # train 
    from model.training import train_pred
    
    # pdb.set_trace()
    hist = train_pred(model, train_dataset, test_dataset, params['NO_EPOCHS'], params['EXP_DIR'])
elif mode == 'predict':

    # modelname
    model = args.model[0]
    model_name = model
    # import importlib
    
    # # get model
    # a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
    # build_DBI_TCN = getattr(a_model, 'build_DBI_TCN')
    from model.model_fn import build_DBI_TCN
    
    params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()
    
    # get inputs
    # a_input = importlib.import_module('experiments.{0}.model.input_fn'.format(model))
    # rippleAI_load_dataset = getattr(a_input, 'rippleAI_load_dataset')
    from model.input_fn import rippleAI_load_dataset
    val_datasets, val_labels = rippleAI_load_dataset(params, mode='test')
    
    from model.cnn_ripple_utils import get_predictions_index, get_performance
    # get predictions
    val_pred = []
    th_arr=np.linspace(0.1,0.9,19)
    n_channels = params['NO_CHANNELS']
    timesteps = params['NO_TIMEPOINTS']
    for LFP in val_datasets:
        LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels)
        # y_predict = np.zeros(shape=(LFP.shape[0]))
        windowed_signal = model.predict(LFP,verbose=1)
        # pdb.set_trace()
        probs = np.hstack(windowed_signal)
        # y_predict = probs.mean(axis=1)
        val_pred.append(probs)
        # for i,window in enumerate(windowed_signal):
        #     print(window.shape)
            # if len(window) != timesteps:
            #     pdb.set_trace()
            #     y_predict[i*timesteps:i*timesteps+len(window)]=window
            # else:
            #     y_predict[i*timesteps:(i+1)*timesteps]=window

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
    # pdb.set_trace()
    
    
    best_preds = all_pred_events[0][14]
    pdb.set_trace()
    pred_vec = np.zeros(val_datasets[0].shape[0])
    label_vec = np.zeros(val_datasets[0].shape[0])
        
    for pred in best_preds:
        pred_vec[int(pred[0]*1250):int(pred[1]*1250)] = 1
        
    for lab in val_labels[0]:
        label_vec[int(lab[0]*1250):int(lab[1]*1250)] = 0.9
    
    import matplotlib.pyplot as plt
    for pred in best_preds:
        rip_begin = int(pred[0]*1250)
        # ss = np.where(np.abs(val_labels[0][:,0]-pred[0])==np.min(np.abs(val_labels[0][:,0]-pred[0])))[0]
        # label_begin = val_labels[0][int(ss*1250), 0]
        
        plt.plot(val_datasets[0][rip_begin-128:rip_begin+128, :])
        plt.plot(pred_vec[rip_begin-128:rip_begin+128], 'r')
        plt.plot(label_vec[rip_begin-128:rip_begin+128], 'k')
        plt.show()
    
    # F1_train=np.empty(shape=len(th_arr))
    # F1_test=np.empty(shape=len(th_arr))
    # for i,th in enumerate(th_arr):
    #     pdb.set_trace()
    #     pred_train_events=get_predictions_index(y_pred_train,th)/1250
    #     pred_test_events=get_predictions_index(y_pred_test,th)/1250
    #     _,_,F1_train[i],_,_,_=get_performance(pred_train_events,events_train,verbose=False)
    #     _,_,F1_test[i],_,_,_=get_performance(pred_test_events,events_test,verbose=False)
    

    # fig,axs=plt.subplots(1,2,figsize=(12,5),sharey='all')
    # axs[0].plot(th_arr,F1_train,'k.-')
    # axs[0].plot(th_arr,F1_test,'b.-')
    # axs[0].legend(['Train','Test'])
    # axs[0].set_ylim([0 ,max(max(F1_train), max(F1_test)) + 0.1])
    # axs[0].set_title('F1 test and train')
    # axs[0].set_ylabel('F1')
    # axs[0].set_xlabel('Threshold')


    # # Validation plot in the second ax
    # F1_val=np.zeros(shape=(len(LFP_val),len(th_arr)))
    # for j,pred in enumerate(val_pred):
    #     for i,th in enumerate(th_arr):
    #         pred_val_events=get_predictions_index(pred,th)/1250
    #         _,_,F1_val[j,i],_,_,_=get_performance(pred_val_events,GT_val[j],verbose=False)

    # for i in range(len(LFP_val)):
    #     axs[1].plot(th_arr,F1_val[i])
    # axs[1].plot(th_arr,np.mean(F1_val,axis=0),'k.-')
    # axs[1].set_title('Validation F1')
    # axs[1].set_ylabel('F1')
    # axs[1].set_xlabel('Threshold')
       
    
    
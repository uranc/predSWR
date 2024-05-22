# Results using the allendataset, I think that it is overfited to their dataset 
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
# parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
parser.add_argument('--model', type=str, nargs=1,
                    help='model name ie. l9:experiments/l9', default='testSWR')
parser.add_argument('--mode', type=str, nargs=1,
                    help='mode training/predict', default='train')
parser.add_argument('--comparison', type=str, nargs=1,
                    help='True/False', default='True')
parser.add_argument('--rippAI_arch', type=str, nargs=1,
                    help='CNN1D, CNN2D, LSTM, SVM or XGBOOST', default='CNN1D')
args = parser.parse_args()
mode = args.mode[0]
model_name = args.model[0]
comparison = args.comparison[0]
arch = args.rippAI_arch[0]
fs = 1250
save_prob = 'False'


# Parameters
params = {'BATCH_SIZE': 64, 'SHUFFLE_BUFFER_SIZE': 4096, 
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-4, 'NO_EPOCHS': 500,
          'NO_TIMEPOINTS': 40, 'NO_CHANNELS': 8,
          'EXP_DIR': '/cs/projects/OWVinckSWR/DL/predSWR/experiments/' + model_name,
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
    
    # train 
    from model.training import train_pred
    
    # pdb.set_trace()
    hist = train_pred(model, train_dataset, test_dataset, params['NO_EPOCHS'], params['EXP_DIR'])
elif mode == 'predict':

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
    
    from model.input_fn import rippleAI_load_dataset, load_allen
    #val_datasets, val_labels = rippleAI_load_dataset(params, mode='test')
    val_datasets = load_allen()
    
    from model.cnn_ripple_utils import get_predictions_index, get_performance
    # get predictions
    val_pred = []
    th_arr=np.linspace(0.1,0.9,19)
    n_channels = params['NO_CHANNELS']
    timesteps = params['NO_TIMEPOINTS']
    #for LFP in val_datasets:
    LFP = val_datasets
    LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels)
    #### windowed_signal = model.predict(LFP,verbose=1)
    windowed_signal = np.squeeze(model.predict(LFP,verbose=1))
    
    probs = np.hstack(windowed_signal)
    val_pred.append(probs)

    # Validation plot in the second ax
    all_pred_events = []
    F1_val=np.zeros(shape=(len(val_datasets),len(th_arr)))
    for j,pred in enumerate(val_pred):
        tmp_pred = []
        for i,th in enumerate(th_arr):
            pred_val_events=get_predictions_index(pred,th)/fs
            #_,_,F1_val[j,i],_,_,_=get_performance(pred_val_events,val_labels[j],verbose=False)
            tmp_pred.append(pred_val_events)
        all_pred_events.append(tmp_pred)
    
    # pick model
    chosen_thr = 16 # 0.81
    #chosen_thr = np.argmax(F1_val[0])
    best_preds = all_pred_events[0][chosen_thr]
    pred_vec = np.zeros(val_datasets.shape[0])
    label_vec = np.zeros(val_datasets.shape[0])
        
    for pred in best_preds:
        pred_vec[int(pred[0]*fs):int(pred[1]*fs)] = 1
        
    # for lab in val_labels[0]:
    #     label_vec[int(lab[0]*fs):int(lab[1]*fs)] = 1 

    if save_prob == 'True' :
        # Saving the probabilities
        directory =  '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/probabilities/' 
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Saving the probabilities in a .npy
        for i, array in enumerate(val_pred):
            np.save(directory + 'prob_{0}_{1}.npy'.format(model_name, i), np.array(array))

 
    # # Saving images   
    # directory =  '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/images/TCN/{0}/'.format(model_name)   
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # # Prediction centered in the detected ripple
    
    # for pred in best_preds:
    #     rip_begin = int(pred[0]*fs)
    #     plt.plot(val_datasets[rip_begin-128:rip_begin+128, :]) #LFP
    #     #plt.plot(label_vec[rip_begin-128:rip_begin+128], 'k',  linewidth=2, label = 'GT') #Labels of DS
    #     plt.plot(val_pred[0][rip_begin-128:rip_begin+128], 'red', linewidth=2, label = 'prob') #Probabilities
    #     plt.plot(val_pred[0][rip_begin-128:rip_begin+128]*pred_vec[rip_begin-128:rip_begin+128], 'aqua', label = 'thr' ) #Bigger than threshold 
    #     plot_filename = os.path.join(directory, f'plot_{rip_begin}.png')
    #     plt.legend()
    #     plt.xlabel('Frequency samples')
    #     plt.title(f'{model_name}, thr={round(th_arr[chosen_thr],2)}',fontsize=7)
    #     plt.savefig(plot_filename)
    #     plt.close()

    
    if comparison == 'True' :

        ############
    #### RIPPL_AI_1D ####
        ############
        # path to the optimized models: /mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models/
        # this was changed in prediction_parser
        from model.cnn_ripple_utils import prediction_parser
        tf.keras.backend.clear_session()
        # n_channels = 8
        # n_timesteps = 32
        # timesteps = 16
        val_pred_rippAI = []
        #for LFP in val_datasets:
        LFP = val_datasets
        val_pred_rippAI.append(prediction_parser(LFP,arch = arch))
            #val_pred_rippAI.append(prediction_parser(LFP, n_channels = 8, n_timesteps = 16))
        # Getting intervals with different thresholds
        all_pred_events_rippAI = []
        for j,pred in enumerate(val_pred_rippAI):
            tmp_pred_rippAI = []
            for i,th in enumerate(th_arr):
                pred_val_events_ripplAI=get_predictions_index(pred,th)/fs
                #_,_,F1_val[j,i],_,_,_=get_performance(pred_val_events_ripplAI,val_labels[j],verbose=False)
                tmp_pred_rippAI.append(pred_val_events_ripplAI)
            all_pred_events_rippAI.append(tmp_pred_rippAI)
        
        best_preds_rippAI = all_pred_events_rippAI[0][14]
       
        print('Rippl_AI',best_preds_rippAI.shape)

        pred_vec_rippAI = np.zeros(val_datasets.shape[0])
            
        for pred in best_preds_rippAI:
            pred_vec_rippAI[int(pred[0]*fs):int(pred[1]*fs)] = 1

        if save_prob == 'True' :
            # Saving the probabilities
            directory =  '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/probabilities/' 
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Saving the probabilities in a .npy
            for i, array in enumerate(val_pred_rippAI):
                np.save(directory + 'prob_rippAI_{0}_{1}.npy'.format(arch, i), np.array(array))

        # directory =  '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/images/imagesrippAI/{0}/'.format(arch)   
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
            
        # for pred in best_preds_rippAI:
            
        #     rip_begin = int(pred[0]*fs)
        #     plt.plot(val_datasets[rip_begin-128:rip_begin+128, :]) #LFP
        #     #plt.plot(label_vec[rip_begin-128:rip_begin+128], 'k',  linewidth=2, label = 'GT') # Labels of DS
        #     plt.plot(val_pred_rippAI[0][rip_begin-128:rip_begin+128], 'red', linewidth=2, label = 'prob') #Probabilities
        #     plt.plot(val_pred_rippAI[0][rip_begin-128:rip_begin+128]*pred_vec_rippAI[rip_begin-128:rip_begin+128], 'aqua',  linewidth=2, label = 'thr') # Predicted by rippAI bigger than thr
        #     plot_filename = os.path.join(directory, f'plot_{rip_begin}.png')
        #     plt.xlabel('Frequency samples')
        #     plt.title(f'Ripp AI {arch}, thr={round(th_arr[14],2)}')
        #     plt.legend()
        #     plt.savefig(plot_filename)
        #     plt.close()

            ####################################
        ### Loading the band-power ripple times ###
            ####################################

        # Method 1: envelope 
        ripple_pred_times_M1 = np.load('/cs/projects/OWVinckSWR/Carmen/LFP_extracted/time_comparison/ripple_pred_times_M1.npy')
        # Method 2: wavelet
        ripple_pred_times_M2 = np.load('/cs/projects/OWVinckSWR/Carmen/LFP_extracted/time_comparison/ripple_pred_times_M2.npy')

        pred_vec_M1 = np.zeros(val_datasets.shape[0])
        pred_vec_M2 = np.zeros(val_datasets.shape[0])

        for pred in ripple_pred_times_M1:
            pred_vec_M1[int(pred[0]):int(pred[1])] = 1
            
        for pred in ripple_pred_times_M2:
            pred_vec_M2[int(pred[0]):int(pred[1])] = 1

            #################
        #### ALL PREDICTIONS ####
            #################
        #Ground-truth centered
        directory =  '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/images/all_predictions/best_preds_{0}_{1}/'.format(arch, model_name )   
        
        if not os.path.exists(directory):
            os.makedirs(directory)
         
        # for pred in val_labels[0]:


        
        for pred in best_preds:
            rip_begin = np.int32(pred[0])
            plt.plot(val_datasets[rip_begin-128:rip_begin+128, :]) #LFP
            plt.plot(pred_vec_M1[rip_begin-128:rip_begin+128], 'k',  linewidth=2.5, label = 'GT_M1') # 'Labels' M1
            plt.plot(pred_vec_M2[rip_begin-128:rip_begin+128], 'grey',  linewidth=2, label = 'GT_M2') # 'Labels' M2
            plt.plot(val_pred_rippAI[0][rip_begin-128:rip_begin+128], 'aqua', linewidth=2, label = 'rippAI') # Probabilities rippAI
            #plt.plot(pred_vec_rippAI[rip_begin-128:rip_begin+128], 'b',  linewidth=2, label = 'rippAI') # Predicted by rippAI bigger than threshold
            plt.plot(val_pred[0][rip_begin-128:rip_begin+128], 'red', linewidth=2, label = 'TCN') # Probabilities TCN
            #plt.plot(val_pred[0][rip_begin-128:rip_begin+128]*pred_vec[rip_begin-128:rip_begin+128], 'r') # Predicted by TCN bigger than threshold
            plot_filename = os.path.join(directory, f'plot_{rip_begin}.png')
            plt.xlabel('Frequency samples')
            plt.title(f'Ripp AI {arch} and {model_name}',fontsize=7)
            plt.legend()
            plt.savefig(plot_filename)
            plt.close()
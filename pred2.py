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
save_prob = 'True'

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
        windowed_signal = model.predict(LFP,verbose=1)
        probs = np.hstack(windowed_signal)
        val_pred.append(probs)

    # Validation plot in the second ax
    all_pred_events = []
    F1_val=np.zeros(shape=(len(val_datasets),len(th_arr)))
    for j,pred in enumerate(val_pred):
        tmp_pred = []
        for i,th in enumerate(th_arr):
            pred_val_events=get_predictions_index(pred,th)/fs
            _,_,F1_val[j,i],_,_,_=get_performance(pred_val_events,val_labels[j],verbose=False)
            tmp_pred.append(pred_val_events)
        all_pred_events.append(tmp_pred)
    
    # pick model
    best_preds = all_pred_events[0][5]
    pred_vec = np.zeros(val_datasets[0].shape[0])
    label_vec = np.zeros(val_datasets[0].shape[0])
        
    for pred in best_preds:
        pred_vec[int(pred[0]*fs):int(pred[1]*1250)] = 1
        
    for lab in val_labels[0]:
        label_vec[int(lab[0]*fs):int(lab[1]*1250)] = 1 
            
    if save_prob == 'True' :
        # Saving the probabilities
        directory =  '/cs/projects/OWVinckSWR/DL/predSWR/experiments/probabilities/' 
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Saving the probabilities in a .npy
        for i, array in enumerate(val_pred):
            np.save(directory + 'prob_{0}_{1}.npy'.format(model_name, i), np.array(array))

    # Saving images   
    directory =  '/cs/projects/OWVinckSWR/DL/predSWR/experiments/images/{0}'.format(model_name)   
    if not os.path.exists(directory):
        os.makedirs(directory)

    # # Prediction centered in the detected ripple
    # for pred in best_preds:
    #     rip_begin = int(pred[0]*fs)
    #     plt.plot(val_datasets[0][rip_begin-128:rip_begin+128, :]) #LFP
    #     plt.plot(label_vec[rip_begin-128:rip_begin+128], 'k',  linewidth=2, label = 'GT') #Labels of DS
    #     plt.plot(val_pred[0][rip_begin-128:rip_begin+128], 'red', label = 'prob') #Probabilities
    #     plt.plot(val_pred[0][rip_begin-128:rip_begin+128]*pred_vec[rip_begin-128:rip_begin+128], 'aqua', label = 'thr' ) #Bigger than threshold 
    #     plot_filename = os.path.join(directory, f'plot_{rip_begin}.png')
    #     plt.legend()
    #     plt.xlabel('Frequency samples')
    #     plt.title(f'{model_name}')
    #     plt.savefig(plot_filename)
    #     plt.show()
    #     plt.close()


    
    if comparison == 'True' :
        #####
    #### CNN ####
        #####
        tf.keras.backend.clear_session()
        # Preprocessing of the data: downsample, z-norm and overlapping windows 
        from model.cnn_ripple_utils import z_score_normalization, downsample_data
        from model.cnn_ripple_utils import generate_overlapping_windows, get_predictions_indexes, real_ripple_times
        overlapping = True
        window_size = 0.0128
        val_pred_cnn = []
        for LFP in val_datasets:
            if overlapping:
                stride = 0.0064
                # Separate the data into 12.8ms windows with 6.4ms overlapping
                X = generate_overlapping_windows(LFP, window_size, stride, fs)
            else:
                stride = window_size
                X = np.expand_dims(LFP, 0)

        # Loading the model 
            optimizer = kr.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
            model = kr.models.load_model("/cs/projects/OWVinckSWR/DL/predSWR/experiments/cnn/cnn_model", compile=False)
            model.compile(loss="binary_crossentropy", optimizer=optimizer)

        # Predictions 
            predictions = model.predict(X, verbose=True)
            probs_aux = np.hstack(predictions)
            probs = np.hstack(probs_aux)
            val_pred_cnn.append(predictions)
            #pdb.set_trace()
        # Thresholding the predictions
        th_arr=np.linspace(0.1,0.9,19)
        all_pred_events_cnn = []
        #pdb.set_trace()
        ###for j,(pred,LFP) in enumerate(zip(val_pred_cnn, val_datasets)):
        for j,pred in enumerate(val_pred_cnn):
            tmp_pred_cnn = []
            for i,th in enumerate(th_arr):
                ###pred_val_events = get_predictions_indexes(LFP, pred, window_size=window_size, stride=stride, fs=downsampled_fs, threshold=th)/downsampled_fs
                pred_val_events = get_predictions_indexes(val_datasets[j], pred, window_size=window_size, stride=stride, fs=fs, threshold=th)/fs
                tmp_pred_cnn.append(pred_val_events)
            all_pred_events_cnn.append(tmp_pred_cnn)

        print((all_pred_events_cnn[0][14]).shape)
        # real_ripple_times
        best_preds_cnn = real_ripple_times(all_pred_events_cnn[0][14])
        print('CNN predictions',best_preds_cnn.shape)
        #auxiliar_fs.append(best_preds_cnn)
        #pdb.set_trace()
        pred_vec_cnn = np.zeros(val_datasets[0].shape[0])
        
        for pred in best_preds_cnn:
            pred_vec_cnn[int(pred[0]*1250):int(pred[1]*1250)] = 1

        if save_prob == 'True' :
            # Saving the probabilities
            directory =  '/cs/projects/OWVinckSWR/DL/predSWR/experiments/probabilities/' 
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Saving the probabilities in a .npy
            for i, array in enumerate(val_pred_cnn):
                np.save(directory + 'prob_cnn_{0}.npy'.format(i), np.array(array))
            
        # #pdb.set_trace()    

        # directory =  '/cs/projects/OWVinckSWR/DL/predSWR/experiments/images/imagescnn'   
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # pdb.set_trace()
        # #Prediction centered in the cnn detected ripple
        # for pred in best_preds_cnn:
        #     rip_begin = int(pred[0]*1250)
        #     #pdb.set_trace()
        #     plt.plot(val_datasets[0][rip_begin-128:rip_begin+128, :]) #LFP
        #     #plt.plot(pred_vec_cnn[rip_begin-128:rip_begin+128], 'b',  linewidth=2, label = 'CNN') # Predicted by cnn
        
        #     plt.plot(label_vec[rip_begin-128:rip_begin+128], 'k',  linewidth=2, label = 'GT') # Labels of DS
        #     plt.plot((val_pred_cnn[0][rip_begin-128:rip_begin+128]).flatten(), 'red',linewidth=2, label = 'prob') #Probabilities
        #     plt.plot((val_pred_cnn[0][rip_begin-128:rip_begin+128]).flatten() * pred_vec_cnn[rip_begin-128:rip_begin+128], 'b', linewidth=2, label = 'thr') #Bigger than threshold 
        #     plot_filename = os.path.join(directory, f'plot_{rip_begin}.png')
        #     plt.legend()
        #     plt.savefig(plot_filename)
        #     plt.close()
        # PROBLEM with dimensions and plotting

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
        for LFP in val_datasets:
            val_pred_rippAI.append(prediction_parser(LFP,arch = arch))
            #val_pred_rippAI.append(prediction_parser(LFP, n_channels = 8, n_timesteps = 16))
        # Getting intervals with different thresholds
        all_pred_events_rippAI = []
        for j,pred in enumerate(val_pred_rippAI):
            tmp_pred_rippAI = []
            for i,th in enumerate(th_arr):
                pred_val_events_ripplAI=get_predictions_index(pred,th)/fs
                _,_,F1_val[j,i],_,_,_=get_performance(pred_val_events_ripplAI,val_labels[j],verbose=False)
                tmp_pred_rippAI.append(pred_val_events_ripplAI)
            all_pred_events_rippAI.append(tmp_pred_rippAI)
        
        best_preds_rippAI = all_pred_events_rippAI[0][14]
        
        print('Rippl_AI',best_preds_rippAI.shape)

        pred_vec_rippAI = np.zeros(val_datasets[0].shape[0])
            
        for pred in best_preds_rippAI:
            pred_vec_rippAI[int(pred[0]*fs):int(pred[1]*fs)] = 1

        if save_prob == 'True' :
            # Saving the probabilities
            directory =  '/cs/projects/OWVinckSWR/DL/predSWR/experiments/probabilities/' 
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Saving the probabilities in a .npy
            for i, array in enumerate(val_pred_rippAI):
                np.save(directory + 'prob_rippAI_{0}_{1}.npy'.format(arch, i), np.array(array))

        directory =  '/cs/projects/OWVinckSWR/DL/predSWR/experiments/images/imagesrippAI'   
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # for pred in best_preds_rippAI:
        #     rip_begin = int(pred[0]*fs)
        #     plt.plot(val_datasets[0][rip_begin-128:rip_begin+128, :]) #LFP
        #     plt.plot(label_vec[rip_begin-128:rip_begin+128], 'k',  linewidth=2, label = 'GT') # Labels of DS
        #     plt.plot(val_pred_rippAI[0][rip_begin-128:rip_begin+128], 'red', label = 'prob') #Probabilities
        #     plt.plot(pred_vec_rippAI[rip_begin-128:rip_begin+128], 'aqua',  linewidth=2, label = 'thr') # Predicted by rippAI bigger than thr
        #     plot_filename = os.path.join(directory, f'plot_{rip_begin}.png')
        #     plt.xlabel('Frequency samples')
        #     plt.title(f'Ripp AI {arch}')
        #     plt.legend()
        #     plt.savefig(plot_filename)
        #     plt.close()

            #################
        #### ALL PREDICTIONS ####
            #################
        #Ground-truth centered
        directory =  '/cs/projects/OWVinckSWR/DL/predSWR/experiments/images/all_predictions'   
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # for pred in val_labels[0]:
        #     rip_begin = np.int32(pred[0]*fs)
        #     plt.plot(val_datasets[0][rip_begin-128:rip_begin+128, :]) #LFP
        #     plt.plot(label_vec[rip_begin-128:rip_begin+128], 'k',  linewidth=2, label = 'GT') # Labels of DS
        #     plt.plot(val_pred_rippAI[0][rip_begin-128:rip_begin+128], 'aqua', label = 'rippAI') # Probabilities rippAI
        #     #plt.plot(pred_vec_rippAI[rip_begin-128:rip_begin+128], 'b',  linewidth=2, label = 'rippAI') # Predicted by rippAI bigger than threshold
        #     plt.plot(val_pred[0][rip_begin-128:rip_begin+128], 'red', label = 'TCN') # Probabilities TCN
        #     #plt.plot(val_pred[0][rip_begin-128:rip_begin+128]*pred_vec[rip_begin-128:rip_begin+128], 'r') # Predicted by TCN bigger than threshold
        #     plot_filename = os.path.join(directory, f'plot_{rip_begin}.png')
        #     plt.xlabel('Frequency samples')
        #     plt.title(f'Ripp AI {arch} and {model_name}')
        #     plt.legend()
        #     plt.savefig(plot_filename)
        #     plt.close()
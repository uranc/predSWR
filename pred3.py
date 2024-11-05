# Results using the allendataset-> I think that it is overfited to their dataset 
# Save the probs of the allen online applying both TCN and RippAI
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
def dataset_to_numpy_array(dataset):
    data = []
    for batch in dataset:
        for element in batch:
            data.append(element)
    return np.array(data)

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
save_prob = 'True' #'True' 'False'


# Parameters
params = {'BATCH_SIZE': 64, 'SHUFFLE_BUFFER_SIZE': 4096*5, 
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
    try:
        import tensorflow.keras as kr
        params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
        model = kr.models.load_model(params['WEIGHT_FILE'])
    except:
        # get model parameters
        print(model_name)
        param_lib = model_name.split('_')
        assert(len(param_lib)==12)
        params['TYPE_MODEL'] = param_lib[0]
        print(params['TYPE_MODEL'])
        assert(param_lib[1][0]=='K')
        params['NO_KERNELS'] = int(param_lib[1][1:])
        print(params['NO_KERNELS'])
        assert(param_lib[2][0]=='T')
        params['NO_TIMEPOINTS'] = int(param_lib[2][1:])
        print(params['NO_TIMEPOINTS'])
        assert(param_lib[3][0]=='D')
        params['NO_DILATIONS'] = int(param_lib[3][1:])
        print(params['NO_DILATIONS'])
        assert(param_lib[4][0]=='N')
        params['NO_FILTERS'] = int(param_lib[4][1:])
        print(params['NO_FILTERS'])
        assert(param_lib[5][0]=='L')
        params['LEARNING_RATE'] = (1e-1)**int(param_lib[5][1:])
        print(params['LEARNING_RATE'])
        assert(param_lib[6][0]=='E')
        params['NO_EPOCHS'] = int(param_lib[6][1:])
        print(params['NO_EPOCHS'])
        assert(param_lib[7][0]=='B')
        params['BATCH_SIZE'] = int(param_lib[7][1:])
        print(params['BATCH_SIZE'])
        assert(param_lib[8][0]=='S')
        params['NO_STRIDES'] = int(param_lib[8][1:])
        print(params['NO_STRIDES'])
        params['TYPE_LOSS'] = param_lib[9]
        print(params['TYPE_LOSS'])
        params['TYPE_REG'] = param_lib[10]
        print(params['TYPE_REG'])
        params['TYPE_ARCH'] = param_lib[11]
        print(params['TYPE_ARCH'])  

    params['BATCH_SIZE'] = 512*2 #8 
    # # get model
    #a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
    a_model = importlib.import_module('model.model_fn')
    build_DBI_TCN = getattr(a_model, 'build_DBI_TCN')
    # from model.model_fn import build_DBI_TCN
    
    params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()
    
    from model.input_fn import rippleAI_load_dataset, load_allen, load_bruce
    #val_datasets, val_labels = rippleAI_load_dataset(params, mode='test')

    # Change dataset HERE
    val_datasets = load_allen()
    #val_datasets = load_bruce(probe = 'probe1')
    
    from model.cnn_ripple_utils import get_predictions_index, get_performance, fcn_load_pickle
    # get predictions
    val_pred = []
    th_arr=np.linspace(0.1,0.9,19)
    n_channels = params['NO_CHANNELS']
    timesteps = params['NO_TIMEPOINTS']
    # No online version
    # LFP = val_datasets
    # LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels)
    # #### windowed_signal = model.predict(LFP,verbose=1)
    # windowed_signal = np.squeeze(model.predict(LFP,verbose=1))
    
    # probs = np.hstack(windowed_signal)
    # val_pred.append(probs)
    
    # Online version
    from keras.utils import timeseries_dataset_from_array
    sample_length = params['NO_TIMEPOINTS'] #*2
    #for LFP in val_datasets:
    LFP = val_datasets
   
    train_x = timeseries_dataset_from_array(LFP, None, sequence_length=sample_length, sequence_stride=1, batch_size=params["BATCH_SIZE"])
    windowed_signal = np.squeeze(model.predict(train_x, verbose=1))
    probs = np.hstack((np.zeros((50, 1)).flatten(),windowed_signal[0,:-1], windowed_signal[:, -1]))
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
    #chosen_thr = 16 # 0.81
    chosen_thr = np.nanargmax(F1_val[0])
    best_preds = all_pred_events[0][chosen_thr]
    pred_vec = np.zeros(val_datasets.shape[0])
    label_vec = np.zeros(val_datasets.shape[0])
        
    for pred in best_preds:
        pred_vec[int(pred[0]*fs):int(pred[1]*fs)] = 1
        
    # for lab in val_labels[0]:
    #     label_vec[int(lab[0]*fs):int(lab[1]*fs)] = 1 

    if save_prob == 'True' :
        # Saving the probabilities
        directory =  '/cs/projects/OWVinckSWR/DL/predSWR/bruce_dataset/probabilities_online/' 
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Saving the probabilities in a .npy
        for i, array in enumerate(val_pred):
            np.save(directory + 'online_prob_bruce_{0}_{1}.npy'.format(model_name, i), np.array(array))

 
    # # Saving images   
    # directory =  '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/online_images/TCN/{0}/'.format(model_name)   
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
        from tensorflow import keras

        val_pred_rippAI = []
        
        LFP = val_datasets
        # No online version
        #val_pred_rippAI.append(prediction_parser(LFP,arch = arch))
        # Online version
        model_number = 1
        for filename in os.listdir('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models/'):
            if f'{arch}_{model_number}' in filename:
                break
        print(filename)
        sp = filename.split('_')
        n_channels = int(sp[2][2]) # 8
        timesteps = int(sp[4][2:]) # 16
        from keras.utils import timeseries_dataset_from_array
        sample_length = timesteps
        

        input_len = LFP.shape[0]
        train_x = timeseries_dataset_from_array(LFP, None, sequence_length=sample_length, sequence_stride=1, batch_size=params["BATCH_SIZE"])
        if arch=='XGBOOST':
            from xgboost import XGBClassifier
            y_predict= np.zeros(shape=(input_len,1,1))
            xgb=XGBClassifier()
            xgb.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename))
            #windowed_signal=xgb.predict_proba(train_x)[:,1]
            windowed_signal = np.squeeze(xgb.predict_proba(train_x))

        elif arch == 'SVM':
            y_predict = np.zeros(shape=(input_len, 1, 1))
            clf = fcn_load_pickle(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models', filename))
            train_x_np = dataset_to_numpy_array(train_x)
            windowed_signal = np.squeeze(clf.predict_proba(train_x_np.reshape(-1, sample_length * 8)))
            print("sample_length:", sample_length)
            print("Shape of windowed_signal:", windowed_signal.shape)
            print("Shape of windowed_signal after SVM predict_proba:", windowed_signal.shape)
            # Ensure zero array matches the correct shape for concatenation
            if windowed_signal.ndim == 1:
                zero_array = np.zeros(sample_length - 1)
            else:
                zero_array = np.zeros((sample_length - 1, windowed_signal.shape[1]))
            print("Shape of zero_array:", zero_array.shape)
            probs = np.hstack((zero_array, windowed_signal))
            
        elif arch=='LSTM':
            # Model load
            model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename))
            #y_predict = model.predict(LFP,verbose=1)
            windowed_signal = np.squeeze(model.predict(train_x, verbose=1))
            # y_predict = windowed_signal
            # y_predict=y_predict.reshape(-1,1,1)
            # y_predict=np.append(y_predict,np.zeros(shape=(input_len%timesteps,1,1))) if (input_len%timesteps!=0) else y_predict 
            # They add 0s at the end of the prob array
            probs = np.hstack((windowed_signal[0,:-1], windowed_signal[:, -1]))

        elif arch=='CNN1D':
            optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
            model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename), compile=False)
            model.compile(loss="binary_crossentropy", optimizer=optimizer)
            #windowed_signal = model.predict(LFP, verbose=True)
            windowed_signal = np.squeeze(model.predict(train_x, verbose=1))
            probs = np.hstack((np.zeros(sample_length-1), windowed_signal))
        elif arch=='CNN2D':
            model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename))
            #windowed_signal= model.predict(LFP,verbose=1)
            windowed_signal = np.squeeze(model.predict(train_x, verbose=1))
            probs = np.hstack((np.zeros(sample_length-1), windowed_signal))
            
        else:
            raise ValueError(f'The introduced architecture -{arch}- does not match the existing ones.')
        
        val_pred_rippAI.append(probs)

        # F1_val_rippAI = np.zeros(shape=(len(val_datasets),len(th_arr)))
        # # Getting intervals with different thresholds
        # all_pred_events_rippAI = []
        # for j,pred in enumerate(val_pred_rippAI):
        #     tmp_pred_rippAI = []
        #     for i,th in enumerate(th_arr):
        #         pred_val_events_ripplAI=get_predictions_index(pred,th)/fs
        #         _,_,F1_val_rippAI[j,i],_,_,_=get_performance(pred_val_events_ripplAI,val_labels[j],verbose=False)
        #         tmp_pred_rippAI.append(pred_val_events_ripplAI)
        #     all_pred_events_rippAI.append(tmp_pred_rippAI)
        
        # #best_preds_rippAI = all_pred_events_rippAI[0][14]
        # max_ind = np.nanargmax(F1_val_rippAI[0,:])
        # best_preds_rippAI = all_pred_events_rippAI[0][max_ind]

        # print('Rippl_AI',best_preds_rippAI.shape)

        # pred_vec_rippAI = np.zeros(val_datasets.shape[0])
            
        # for pred in best_preds_rippAI:
        #     pred_vec_rippAI[int(pred[0]*fs):int(pred[1]*fs)] = 1

        if save_prob == 'True' :
            # Saving the probabilities
            directory =  '/cs/projects/OWVinckSWR/DL/predSWR/bruce_dataset/probabilities_online/' 
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Saving the probabilities in a .npy
            for i, array in enumerate(val_pred_rippAI):
                np.save(directory + 'online_prob_rippAI_bruce_{0}_{1}.npy'.format(arch, i), np.array(array))

        # directory =  '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/online_images/imagesrippAI/{0}/'.format(arch)   
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
        ripple_pred_times_M1 = np.load('/cs/projects/OWVinckSWR/DL/predSWR/bruce_dataset/LFP_sanity_check/M1_M2_methods/ripple_pred_times_M1_bruce_probe1.npy')
        # Method 2: wavelet
        ripple_pred_times_M2 = np.load('/cs/projects/OWVinckSWR/DL/predSWR/bruce_dataset/LFP_sanity_check/M1_M2_methods/ripple_pred_times_M2_bruce_probe1.npy')

        # pred_vec_M1 = np.zeros(val_datasets.shape[0])
        # pred_vec_M2 = np.zeros(val_datasets.shape[0])

        # for pred in ripple_pred_times_M1:
        #     pred_vec_M1[int(pred[0]):int(pred[1])] = 1
            
        # for pred in ripple_pred_times_M2:
        #     pred_vec_M2[int(pred[0]):int(pred[1])] = 1

            #################
        #### ALL PREDICTIONS ####
            #################
        #Ground-truth centered
        directory =  '/cs/projects/OWVinckSWR/DL/predSWR/bruce_dataset/online_images/all_predictions/best_preds_{0}_{1}/'.format(arch, model_name )   
        
        if not os.path.exists(directory):
            os.makedirs(directory)
         
        # for pred in val_labels[0]:


        
        for pred in best_preds:
            rip_begin = np.int32(pred[0])
            plt.plot(val_datasets[rip_begin-128:rip_begin+128, :]) #LFP
            plt.plot(ripple_pred_times_M1[rip_begin-128:rip_begin+128], 'k',  linewidth=2.5, label = 'GT_M1') # 'Labels' M1
            plt.plot(ripple_pred_times_M2[rip_begin-128:rip_begin+128], 'grey',  linewidth=2, label = 'GT_M2') # 'Labels' M2
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
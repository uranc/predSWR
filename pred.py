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
params = {'BATCH_SIZE': 32, 'SHUFFLE_BUFFER_SIZE': 4096*5, 
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-3, 'NO_EPOCHS': 200,
          'NO_TIMEPOINTS': 50, 'NO_CHANNELS': 8,
          'EXP_DIR': '/cs/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
          }

if mode == 'train':

    # update params
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
        
        # get model
        a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
        build_DBI_TCN = getattr(a_model, 'build_DBI_TCN')
        # from model.model_fn import build_DBI_TCN
        
        params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()
        
    # from model.model_fn import build_DBI_TCN
    # pdb.set_trace()
    # model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    # model.summary()
    params['BATCH_SIZE'] = 512*2
    
    # pdb.set_trace()
    # noise = np.random.rand(1024,8).reshape(int(1024/32),32,8)
    # probs = model.predict(noise)
    
    # get inputs
    a_input = importlib.import_module('experiments.{0}.model.input_fn'.format(model_name))
    rippleAI_load_dataset = getattr(a_input, 'rippleAI_load_dataset')
    # from model.input_fn import rippleAI_load_dataset
    val_datasets, val_labels = rippleAI_load_dataset(params, mode='test')
    
    # for j, labels in enumerate(val_labels):
    #     np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/labels_val{0}.npy'.format(j), labels)
    # pdb.set_trace()
    from model.cnn_ripple_utils import get_predictions_index, get_performance
    # get predictions
    val_pred = []
    th_arr=np.linspace(0.1,0.9,19)
    n_channels = params['NO_CHANNELS']
    timesteps = params['NO_TIMEPOINTS']
    
    from keras.utils import timeseries_dataset_from_array

    # get predictions
    sample_length = params['NO_TIMEPOINTS']*2
   
    # val_datasets[0] = val_datasets[0][89500:90500-500,:]
    # val_datasets[1] = val_datasets[1][:100,:]
    # val_datasets = [val_datasets[0]]
    # test_end = val_datasets[0].shape[0]
    # val_batches = []
    # for i in range(0, test_end - timesteps, 1):
    #     val_batches.append(val_datasets[0][np.arange(i, i + timesteps), :])
    # val_batches = np.array(val_batches)
    for LFP in val_datasets:
        # test_end = LFP.shape[0]
        # LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels)
        # aa = []
        train_x = timeseries_dataset_from_array(LFP, None, sequence_length=sample_length, sequence_stride=1, batch_size=params["BATCH_SIZE"])

        # for i in range(0, test_end - timesteps, 1):
        windowed_signal = np.squeeze(model.predict(train_x, verbose=1))
            # aa.append(windowed_signal)
        probs = np.hstack((np.zeros((50, 1)).flatten(),windowed_signal[0,:-1], windowed_signal[:, -1]))
        
        # probs = np.hstack(aa)
        # pdb.set_trace()
        # probs = np.hstack(windowed_signal)
        val_pred.append(probs)

    # pdb.set_trace()
    # Validation plot in the second ax
    all_pred_events = []
    precision=np.zeros(shape=(len(val_datasets),len(th_arr)))
    recall=np.zeros(shape=(len(val_datasets),len(th_arr)))
    F1_val=np.zeros(shape=(len(val_datasets),len(th_arr)))
    TP=np.zeros(shape=(len(val_datasets),len(th_arr)))    
    FN=np.zeros(shape=(len(val_datasets),len(th_arr)))
    IOU=np.zeros(shape=(len(val_datasets),len(th_arr)))
    for j,pred in enumerate(val_pred):
        tmp_pred = []
        # tmp_IOUs = []
        for i,th in enumerate(th_arr):
            pred_val_events=get_predictions_index(pred,th)/1250
            # pdb.set_trace()
            [precision[j,i], recall[j,i], F1_val[j,i], tmpTP, tmpFN, tmpIOU] = get_performance(pred_val_events,val_labels[j],verbose=False)
            TP[j,i] = tmpTP.sum()
            FN[j,i] = tmpFN.sum()
            IOU[j,i] = np.median(tmpIOU.sum(axis=0))
            tmp_pred.append(pred_val_events)
        # pdb.set_trace()
        # IOU.append(np.array(tmp_IOUs))
        all_pred_events.append(tmp_pred)
    # pdb.set_trace()
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
    
    
    # mean_median_IOU = np.stack((np.stack((np.nanmean(IOU[0], axis=1), np.nanmedian(IOU[0],axis=1)), axis=-1),
    #                             np.stack((np.nanmean(IOU[1], axis=1), np.nanmedian(IOU[1],axis=1)), axis=-1)),axis=0)
    # mean_median_IOU[:,:,0], mean_median_IOU[:,:,1]
    stats = np.stack((precision, recall, F1_val, TP, FN, IOU), axis=-1)
    # pdb.set_trace()
    
    for j,pred in enumerate(val_pred):
        np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/preds_val{0}_{1}.npy'.format(j, model_name), pred)
        np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/stats_val{0}_{1}.npy'.format(j, model_name), stats[j,])
        # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/areas_val{0}_{1}.npy'.format(j, model_name), IOU[j])

    # import matplotlib.pyplot as plt
    # for pred in val_labels[0]:
    #     rip_begin = int(pred[0]*1250)
    #     plt.plot(val_datasets[0][rip_begin-128:rip_begin+128, :])
    #     plt.plot(val_pred[0][rip_begin-128:rip_begin+128], 'k')
    #     plt.plot(val_pred[0][rip_begin-128:rip_begin+128]*pred_vec[rip_begin-128:rip_begin+128], 'r')
    #     plt.plot(label_vec[rip_begin-128:rip_begin+128], 'k')
    #     plt.show()
    
elif mode == 'predictSynth':
    
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
        
        # get model
        a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
        build_DBI_TCN = getattr(a_model, 'build_DBI_TCN')
        # from model.model_fn import build_DBI_TCN
        
        params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()

    synth = np.load('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/synth_stim.npy')
    synth = np.tile(synth, (1,8))
    
    # train_end = synth.shape[0]
    # timesteps = 50
    # synth_batches = []
    # for i in range(0, train_end - timesteps, 1):
    #     synth_batches.append(synth[np.arange(i, i + timesteps), :])
    # synth_batches = np.array(synth_batches)
    from keras.utils import timeseries_dataset_from_array

    # get predictions
    params["BATCH_SIZE"] = 512
    n_channels = params['NO_CHANNELS']
    timesteps = params['NO_TIMEPOINTS']
    sample_length = params['NO_TIMEPOINTS']*2
    train_x = timeseries_dataset_from_array(synth, None, sequence_length=sample_length, sequence_stride=1, batch_size=params["BATCH_SIZE"])
    # synth=synth[:len(synth)-len(synth)%timesteps,:].reshape(-1,timesteps,n_channels)
    # probs = np.squeeze(model.predict(train_x))
    probs = np.squeeze(model.predict(train_x))
    probs = np.hstack((probs[0,:-1], probs[:, -1]))
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    probs = moving_average(probs, 4)

    # pdb.set_trace()
    # synth=np.expand_dims(synth, axis=0)
    # pdb.set_trace()
    from scipy.signal import decimate
    import matplotlib.pyplot as plt
    synth = np.load('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/synth_stim.npy')
    synth = (synth-np.min(synth))/(np.max(synth)-np.min(synth))
    synth = synth[50:,:]
    tt = np.arange(synth.shape[0])/1250
    # pdb.set_trace()
    # plt.plot(decimate(tt, 4), decimate(synth[:, 0], 4))
    plt.plot(tt, synth[:, 0])
    tt = np.arange(probs.shape[0])/1250
    # plt.plot(decimate(tt,4), decimate(probs, 4))
    plt.plot(tt, probs)
    plt.show()
    pdb.set_trace()
    
elif mode == 'predictPlot':
    import matplotlib.pyplot as plt   
    import importlib 
    from model.cnn_ripple_utils import get_predictions_index, get_performance

    # modelname
    model = args.model[0]
    model_name = model
    
    # get inputs
    a_input = importlib.import_module('experiments.{0}.model.input_fn'.format(model))
    rippleAI_load_dataset = getattr(a_input, 'rippleAI_load_dataset')
    # from model.input_fn import rippleAI_load_dataset
    val_datasets, val_labels = rippleAI_load_dataset(params, mode='test')
    
    # probs 
    val_id = 0
    val_pred = np.load('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/preds_val{0}_{1}.npy'.format(val_id, model_name))
    # val_pred = np.hstack((np.zeros((50,1)).flatten(), val_pred))
    val_pred = [val_pred]
    
    val_datasets = [val_datasets[val_id]]

    # Validation plot in the second axs
    th_arr=np.linspace(0.1,0.9,19)
    all_pred_events = []
    F1_val=np.zeros(shape=(len(val_datasets),len(th_arr)))
    for j,pred in enumerate(val_pred):
        tmp_pred = []
        for i,th in enumerate(th_arr):
            pred_val_events=get_predictions_index(pred,th)/1250
            _,_,F1_val[j,i],_,_,_=get_performance(pred_val_events ,val_labels[j], verbose=False)
            # if i==12:
            #     [precision, recall, F1, TP, FN, IOU]=get_performance(pred_val_events ,val_labels[j], exclude_matched_trues=True, verbose=False)
            #     pdb.set_trace()
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
    # pdb.set_trace()
    for pred in val_labels[0]:
        rip_begin = int(pred[0]*1250)
        plt.plot(val_datasets[0][rip_begin-256:rip_begin+256, :]/3, 'gray')
        plt.plot(val_pred[0][rip_begin-256:rip_begin+256], 'k')
        plt.plot(val_pred[0][rip_begin-256:rip_begin+256]*pred_vec[rip_begin-256:rip_begin+256], 'r')
        plt.plot(label_vec[rip_begin-256:rip_begin+256], 'k')
        plt.show()        
        # pdb.set_trace()
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import tensorflow as tf
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
from keras.utils import timeseries_dataset_from_array


# loading the LFP data of the pyr layer 
dir = '/cs/projects/OWVinckSWR/DL/predSWR/experiments/probabilities/'
y_labels = np.load(dir + 'val_labels_0.npy')
val_datasets = np.load(dir + 'val_dataset_0.npy')
LFP = val_datasets[:,4]

parser = argparse.ArgumentParser(
    description='Example 3 - Local and Parallel Execution.')
# parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
parser.add_argument('--model', type=str, nargs=1,
                    help='model name ie. l9:experiments/l9', default='testSWR')
# parser.add_argument('--mode', type=str, nargs=1,
#                     help='mode training/predict', default='train')

args = parser.parse_args()
# mode = args.mode[0]
model_name = args.model[0]
# Parameters
params = {'BATCH_SIZE': 32, 'SHUFFLE_BUFFER_SIZE': 4096*5, 
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-3, 'NO_EPOCHS': 200,
          'NO_TIMEPOINTS': 50, 'NO_CHANNELS': 8,
          'EXP_DIR': '/cs/projects/OWVinckSWR/DL/predSWR/experiments/' + model_name,
          }
 
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
    pdb.set_trace()
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
pdb.set_trace()
# from model.model_fn import build_DBI_TCN
# pdb.set_trace()
# model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
# model.summary()
params['BATCH_SIZE'] = 512

# pdb.set_trace()
# noise = np.random.rand(1024,8).reshape(int(1024/32),32,8)
# probs = model.predict(noise)

# get inputs
a_input = importlib.import_module('experiments.{0}.model.input_fn'.format(model_name))
rippleAI_load_dataset = getattr(a_input, 'rippleAI_load_dataset')
# from model.input_fn import rippleAI_load_dataset
val_datasets, val_labels = rippleAI_load_dataset(params, mode='test')
from model.cnn_ripple_utils import get_predictions_index, get_performance


# get predictions
val_pred = []
th_arr=np.linspace(0.1,0.9,19)
n_channels = params['NO_CHANNELS']
timesteps = params['NO_TIMEPOINTS']



# get predictions
sample_length = params['NO_TIMEPOINTS']*2

# val_datasets[0] = val_datasets[0][89500:90500-500,:]
# val_datasets[1] = val_datasets[1][:100,:]
val_datasets = [val_datasets[0]]
# test_end = val_datasets[0].shape[0]
# val_batches = []
# for i in range(0, test_end - timesteps, 1):
#     val_batches.append(val_datasets[0][np.arange(i, i + timesteps), :])
# val_batches = np.array(val_batches)
for LFP in [val_datasets[0]]:
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

# pdb.set_trace()

for j,pred in enumerate(val_pred):
    np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/preds_val{0}_{1}.npy'.format(j, model_name), pred)

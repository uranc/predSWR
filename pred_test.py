import os
import pdb
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
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
import joblib
import json
import logging
import gc
import pandas as pd
import sys
import glob
import importlib
from tensorflow.keras import callbacks as cb
import pdb
import tensorflow.keras as kr


parser = argparse.ArgumentParser(
    description='Example 3 - Local and Parallel Execution.')
parser.add_argument('--model', type=str, nargs=1,
                    help='model name ie. l9:experiments/l9', default='testSWR')
parser.add_argument('--directory', type=str, nargs=1,
                    help='LFP directory ie. ONIX/Analysis/2ndExp/Exp_LFP_/', default='./ripple_BP_LFP.npy')
parser.add_argument('--channels', type=str, nargs=1,
                    help='Channels to apply the model', default='[0,1,2,3,4,5,6,7]')


args = parser.parse_args()
print(parser.parse_args())
LFP_directory = args.directory[0]
model_name = args.model[0]
channels = args.channels[0]
print(channels)
# pdb.set_trace()
# Parameters
params = {'BATCH_SIZE': 32, 'SHUFFLE_BUFFER_SIZE': 4096*2,
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-3, 'NO_EPOCHS': 200,
          'NO_TIMEPOINTS': 50, 'NO_CHANNELS': 8, 'SRATE': 2500,
          'EXP_DIR': '/mnt/hpc/projects/OWVinckSWR/DL/predSWR/experiments/' + model_name,
          'mode' : 'test',
          }




# modelname
model = args.model[0]
model_name = model

param_lib = model_name.split('_')
print(param_lib)
tag = param_lib[5]
study_num = param_lib[3]
param_dir = f"params_{tag}"
istune = param_lib[2]
params['BATCH_SIZE'] = 512*2


if istune.startswith('Tune'):
    # Extract study number from model name (e.g., 'Tune_45_' -> '45')
    
    print(f"Loading tuned model from study {study_num}")

    # params['SRATE'] = 2500
    # Find the study directory

    # study_dirs = glob.glob(f'studies_1/study_{study_num}_*')
    param_dir = f"params_{tag}"

    study_dirs = glob.glob(f'/mnt/hpc/projects/OWVinckSWR/DL/predSWR/experiments/studies/{param_dir}/study_{study_num}_*')
    # study_dirs = glob.glob(f'studies_CHECK_SIGNALS/{param_dir}/study_{study_num}_*')
    if not study_dirs:
        raise ValueError(f"No study directory found for study number {study_num}")
    study_dir = study_dirs[0]  # Take the first matching directory

    # Load trial info to get parameters
    with open(f"{study_dir}/trial_info.json", 'r') as f:
        trial_info = json.load(f)
        params.update(trial_info['parameters'])

    #pdb.set_trace()
    # Import required modules
    if 'MixerOnly' in params['TYPE_ARCH']:
        # from model.model_fn import build_DBI_TCN_MixerOnly as build_DBI_TCN
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        build_DBI_TCN = model_module.build_DBI_TCN_MixerOnly
    elif 'MixerHori' in params['TYPE_ARCH']:
        # from model.model_fn import build_DBI_TCN_HorizonMixer as build_DBI_TCN
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        build_DBI_TCN = model_module.build_DBI_TCN_HorizonMixer
    elif 'MixerDori' in params['TYPE_ARCH']:
        # from model.model_fn import build_DBI_TCN_DorizonMixer as build_DBI_TCN
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        build_DBI_TCN = model_module.build_DBI_TCN_DorizonMixer
    elif 'MixerCori' in params['TYPE_ARCH']:
        # from model.model_fn import build_DBI_TCN_CorizonMixer as build_DBI_TCN
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        build_DBI_TCN = model_module.build_DBI_TCN_CorizonMixer
    elif 'TripletOnly' in params['TYPE_ARCH']:
        # from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        build_DBI_TCN = model_module.build_DBI_TCN_TripletOnly

#model = kr.models.load_model(params['WEIGHT_FILE'])
weight_file = f"{study_dir}/max.weights.h5"
# weight_file = f"{study_dir}/robust.weights.h5"
print(f"Loading weights from: {weight_file}")
#model = kr.models.load_model(os.path.join(study_dir,/"max.weights.h5"))

pdb.set_trace()
model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
model.load_weights(weight_file)
# Load the LFP data from the directory, Ripple bandpass filter

LFP_load =  np.load(os.path.join(LFP_directory, 'ripple_BP_LFP.npy'))
print(LFP_load.shape)
LFP = LFP_load[:, channels]
print(LFP.shape)


# get predictions
th_arr=np.linspace(0.0,1.0,11)
n_channels = params['NO_CHANNELS']
timesteps = params['NO_TIMEPOINTS']
samp_freq = params['SRATE']

all_pred_events = []

from keras.utils import timeseries_dataset_from_array


sample_length = params['NO_TIMEPOINTS']
train_x = timeseries_dataset_from_array(LFP, None, sequence_length=sample_length, sequence_stride=1, batch_size=params["BATCH_SIZE"])
windowed_signal = np.squeeze(model.predict(train_x, verbose=1))

# different outputs
if model_name.find('Hori') != -1 or model_name.find('Dori') != -1 or model_name.find('Cori') != -1:
    if len(windowed_signal.shape) == 3:
        probs = np.hstack((windowed_signal[0,:-1,-1], windowed_signal[:, -1,-1]))
        horizon = np.vstack((windowed_signal[0,:-1,:-1], windowed_signal[:, -1,:-1]))
    else:
        probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal[:,-1]))
        horizon = np.vstack((np.zeros((sample_length-1, 8)), windowed_signal[:, :-1]))
elif  model_name.startswith('Tune') != -1:
    if 'Only' in params['TYPE_ARCH']:
        probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal))
    else:                
        probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal[:,-1]))
        horizon = np.vstack((np.zeros((sample_length-1, 8)), windowed_signal[:, :-1]))
elif model_name.find('Proto') != -1:
    probs = np.hstack((windowed_signal[0,:-1], windowed_signal[:, -1]))
elif model_name.find('Base_') != -1:
    probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal))
else:
    probs = np.hstack((windowed_signal[0,:-1], windowed_signal[:, -1]))

# Saving probabilities 
probs_dir = os.path.join(LFP_directory, 'probs')
os.makedirs(probs_dir, exist_ok=True)

np.save(os.path.join(probs_dir, f'preds_{model_name}_{channels}.npy'), probs)
# Saving horizon
if model_name.find('Hori') != -1 or model_name.find('Dori') != -1 or model_name.find('Cori') != -1 or model_name.startswith('Tune') != -1:
    if not ('Only' in params['TYPE_ARCH']):
        np.save(os.path.join(probs_dir, f'horis_{model_name}_{channels}.npy'), horizon)
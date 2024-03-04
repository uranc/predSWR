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
params = {'BATCH_SIZE': 40, 'SHUFFLE_BUFFER_SIZE': 4096, 
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-4, 'NO_EPOCHS': 200,
          'LSTM_TIMEPOINTS': 40, 'LSTM_CHANNELS': 8,
          'EXP_DIR': '/mnt/hpc/projects/OWVinckSWR/DL/predSWR/experiments/' + model_name,
          }

if mode == 'train':

    # input
    from model.input_fn import rippleAI_load_dataset
    train_dataset, test_dataset = rippleAI_load_dataset(params)
    train_size = len(list(train_dataset))
    
    # model
    from model.model_fn import build_Prida_LSTM
    model = build_Prida_LSTM([params["LSTM_TIMEPOINTS"],params["LSTM_CHANNELS"]])
    model.summary()
    
    # train 
    from model.training import train_pred
    # pdb.set_trace()
    hist = train_pred(model,train_dataset,test_dataset,params['NO_EPOCHS'],params['EXP_DIR'])
elif mode == 'predict':

    # modelname
    model = args.model[0]
    model_name = model
    import importlib
    
    # get model
    a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
    build_compile_model_pred = getattr(a_model, 'build_compile_model_pred_splitBlock')

    # get inputs
    a_input = importlib.import_module('experiments.{0}.model.input_pred'.format(model))
    train_inputs_fn_a = getattr(a_input, 'train_inputs_fn_g')
    test_inputs_fn_a = getattr(a_input, 'test_inputs_fn_g')
    
    # load
    params['weight_dir'] = 'experiments/' + model + '/weights.last.h5'
    model_name = model_base  + '_{0}'.format(i_step)
    K.clear_session()
    name = '/mnt/hpc/projects/MWNaturalPredict/DL/data/predBatch{0}_0_224_001.npy'.format(tagNP)
    test_inputs, test_size = get_numpy_dataset(fname, params['b_size'])
    test_steps = int(np.floor(test_size/params['b_size']))+1
    tests = test_inputs.take(test_steps)
    
    # load data
    images = []
    labels = []
    [images.append(test[0]) for test in tests]
    [labels.append(test[1]) for test in tests]
    images = np.vstack(images)
    ytrue = np.vstack(labels)
    
    # preprocess
    img = images + np.expand_dims([103.939, 116.779, 123.68], axis=0)
    img = img[:,:,:,::-1]
    img = np.round(img)
    img[img<0] = 0
    img[img>255] = 255
            
    # model
    model = build_compile_model_pred(mode, params)
    model.summary()
    tmp_pred = model.predict(images, steps=test_steps)

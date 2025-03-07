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
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.storages import RDBStorage
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

# import tensorflow_models as tfm


# tf.config.run_functions_eagerly(True)
parser = argparse.ArgumentParser(
    description='Example 3 - Local and Parallel Execution.')
# parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
parser.add_argument('--model', type=str, nargs=1,
                    help='model name ie. l9:experiments/l9', default='testSWR')
parser.add_argument('--mode', type=str, nargs=1,
                    help='mode training/predict', default='train')
parser.add_argument('--val', type=str, nargs=1,
                    help='val_id 0,1,2,3', default='10')
parser.add_argument('--tag', type=str, nargs=1,
                    help='tag for running multiple studies in parallel', default='base')

args = parser.parse_args()
print(parser.parse_args())
mode = args.mode[0]
model_name = args.model[0]
val_id = int(args.val[0])
tag = args.tag[0]
print(val_id)
# pdb.set_trace()
# Parameters
params = {'BATCH_SIZE': 32, 'SHUFFLE_BUFFER_SIZE': 4096*2,
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-3, 'NO_EPOCHS': 200,
          'NO_TIMEPOINTS': 50, 'NO_CHANNELS': 8, 'SRATE': 2500,
          'EXP_DIR': '/cs/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
          }
params['mode'] = mode

# Define objective function before mode selection
def create_study_name(trial):
    """Create unique study name based on trial parameters"""
    return f"study_{trial.number}_{trial.datetime_start.strftime('%Y%m%d_%H%M%S')}"

def objective(trial):
    """Objective function for Optuna optimization"""
    tf.compat.v1.reset_default_graph()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
    
    # Start with base parameters
    params = {'BATCH_SIZE': 32, 'SHUFFLE_BUFFER_SIZE': 4096*2,
            'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-3, 'NO_EPOCHS': 300,
            'NO_TIMEPOINTS': 50, 'NO_CHANNELS': 8, 'SRATE': 2500,
            'EXP_DIR': '/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
            'mode': 'train'
            }

    # Dynamic learning rate range
    # learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    learning_rate = 1e-3
    params['LEARNING_RATE'] = learning_rate

    # Optional batch size tuning
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    batch_size = 64
    params['BATCH_SIZE'] = batch_size
    # ...rest of existing objective function code...

    # Base parameters
    params['SRATE'] = 2500
    params['NO_EPOCHS'] = 250
    params['TYPE_MODEL'] = 'Base'

    arch_lib = ['MixerOnly', 'MixerHori', 'MixerDori', 'DualMixerDori', 'MixerCori', 'SingleCh']
    # Model architecture parameters - Fix the categorical suggestion
    # arch_ind = trial.suggest_int('IND_ARCH', 0, len(arch_lib)-1)
    arch_ind = 0
    params['TYPE_ARCH'] = arch_lib[arch_ind]
    # params['TYPE_ARCH'] = 'MixerHori'
    # pdb.set_trace()
    # Update model import based on architecture choice
    if 'MixerHori' in params['TYPE_ARCH']:
        from model.input_augment_weighted import rippleAI_load_dataset
        from model.model_fn import build_DBI_TCN_HorizonMixer as build_DBI_TCN
    elif 'MixerDori' in params['TYPE_ARCH']:
        from model.input_augment_weighted import rippleAI_load_dataset
        from model.model_fn import build_DBI_TCN_DorizonMixer as build_DBI_TCN
    elif 'MixerCori' in params['TYPE_ARCH']:
        from model.input_augment_weighted import rippleAI_load_dataset
        from model.model_fn import build_DBI_TCN_CorizonMixer as build_DBI_TCN
    elif 'MixerOnly' in params['TYPE_ARCH']:
        from model.input_augment_weighted import rippleAI_load_dataset
        from model.model_fn import build_DBI_TCN_MixerOnly as build_DBI_TCN

    # Timing parameters remain the same
    # params['NO_TIMEPOINTS'] = trial.suggest_categorical('NO_TIMEPOINTS', [128, 196, 384])
    params['NO_TIMEPOINTS'] = 128
    params['NO_STRIDES'] = int(params['NO_TIMEPOINTS'] // 2)

    # Timing parameters remain the same
    params['HORIZON_MS'] = trial.suggest_int('HORIZON_MS', 1, 10)
    params['SHIFT_MS'] = 0

    params['LOSS_WEIGHT'] = 7.5e-4

    params['HYPER_ENTROPY'] = trial.suggest_float('HYPER_ENTROPY', 0.000001, 10.0, log=True)
    params['HYPER_TMSE'] = trial.suggest_float('HYPER_TMSE', 0.000001, 10.0, log=True)
    # params['HYPER_BARLOW'] = 2e-5
    params['HYPER_BARLOW'] = trial.suggest_float('HYPER_BARLOW', 0.000001, 10.0, log=True)
    # params['HYPER_MONO'] = trial.suggest_float('HYPER_MONO', 0.000001, 10.0, log=True)
    params['HYPER_MONO'] = 0 #trial.suggest_float('HYPER_MONO', 0.000001, 10.0, log=True)

    # Model parameters matching training format
    # params['NO_KERNELS'] = trial.suggest_int('NO_KERNELS', 2, 6) # for kernels 2,3,4,5,6
    params['NO_KERNELS'] = 4
    if params['NO_TIMEPOINTS'] == 32:
        dil_lib = [4,3,2,2,2]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 64:
        dil_lib = [5,4,3,3,3]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 128:
        dil_lib = [6,5,4,4,4]           # for kernels 2,3,4,5,6]
    elif params['NO_TIMEPOINTS'] == 196:
        dil_lib = [7,6,5,5,5]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 384:
        dil_lib = [8,7,6,6,6]           # for kernels 2,3,4,5,6
    params['NO_DILATIONS'] = dil_lib[params['NO_KERNELS']-2]
    # params['NO_DILATIONS'] = trial.suggest_int('NO_DILATIONS', 2, 6)
    # params['NO_FILTERS'] = trial.suggest_categorical('NO_FILTERS', [32, 64, 128])
    params['NO_FILTERS'] = 64
    ax = 65#trial.suggest_int('AX', 1, 99)
    gx = 100#trial.suggest_int('GX', 50, 999)

    # Removed duplicate TYPE_ARCH suggestion that was causing the error
    params['TYPE_LOSS'] = 'FocalGapAx{:03d}Gx{:03d}EntropyTMSE'.format(ax, gx)

    # Remove the hardcoded use_freq and derive it from tag instead
    tag = args.tag[0]
    params['TYPE_LOSS'] += tag
    print(params['TYPE_LOSS'])
    # init_lib = ['He', 'Glo']
    # par_init = init_lib[trial.suggest_int('IND_INIT', 0, len(init_lib)-1)]
    par_init = 'He'
    # norm_lib = ['LN','BN','GN','WN']
    # par_norm = norm_lib[trial.suggest_int('IND_NORM', 0, len(norm_lib)-1)]
    par_norm = 'LN'
    # act_lib = ['RELU', 'ELU', 'GELU']
    # par_act = act_lib[trial.suggest_int('IND_ACT', 0, len(act_lib)-1)]
    par_act = 'ELU'
    
    opt_lib = ['Adam', 'AdamW', 'SGD']
    par_opt = opt_lib[trial.suggest_int('IND_OPT', 0, len(opt_lib)-1)]
    
    params['TYPE_REG'] = (f"{par_init}"f"{par_norm}"f"{par_act}"f"{par_opt}")
    # Build architecture string with timing parameters (adjust format)
    arch_str = (f"{params['TYPE_ARCH']}"  # Take first 4 chars: Hori/Dori/Cori
                f"{int(params['HORIZON_MS']):02d}")
    print(arch_str)
    params['TYPE_ARCH'] = arch_str

    # Use multiple binary flags for a combinatorial categorical parameter
    params['USE_ZNorm'] = trial.suggest_categorical('USE_ZNorm', [True, False])
    if params['USE_ZNorm']:
        params['TYPE_ARCH'] += 'ZNorm'
    # params['USE_L2N'] = trial.suggest_categorical('USE_L2N', [True, False])
    # if params['USE_L2N']:
    params['TYPE_ARCH'] += 'L2N'


    params['USE_Aug'] = trial.suggest_categorical('USE_Aug', [True, False])
    if params['USE_Aug']:
        params['TYPE_ARCH'] += 'Aug'
        
    if (not 'Cori' in params['TYPE_ARCH']) and  (not 'SingleCh' in params['TYPE_MODEL']):
        params['TYPE_LOSS'] += 'BarAug'
    # params['USE_L2Reg'] = trial.suggest_categorical('USE_L2Reg', [True])#, False
    # params['USE_CSD'] = trial.suggest_categorical('USE_CSD', [True, False])
    # if params['USE_CSD']:
    #     params['TYPE_ARCH'] += 'CSD'
    # params['Dropout'] = trial.suggest_int('Dropout', 0, 10)
    # params['USE_L2Reg'] = trial.suggest_categorical('USE_L2Reg', [True])#, False
    # if params['USE_L2Reg']:
    #     params['TYPE_LOSS'] += 'L2Reg'

    # params['Dropout'] = trial.suggest_int('Dropout', 0, 10)
    # params['TYPE_ARCH'] += f"Drop{params['Dropout']:02d}"
    params['TYPE_ARCH'] += f"Shift{int(params['SHIFT_MS']):02d}"

    # Build name in correct format
    run_name = (f"{params['TYPE_MODEL']}_" 
                f"K{params['NO_KERNELS']}_"
                f"T{params['NO_TIMEPOINTS']}_"
                f"D{params['NO_DILATIONS']}_"
                f"N{params['NO_FILTERS']}_"
                f"L{int(-np.log10(params['LEARNING_RATE']))}_"
                f"E{params['NO_EPOCHS']}_"
                f"B{params['BATCH_SIZE']}_"
                f"S{params['NO_STRIDES']}_"
                f"{params['TYPE_LOSS']}_"
                f"{params['TYPE_REG']}_"
                f"{params['TYPE_ARCH']}")
    # pdb.set_trace()
    params['NAME'] = run_name
    print(params['NAME'])

    # pdb.set_trace()
    tag = args.tag[0]
    param_dir = f"params_{tag}"
    study_dir = f"studies/{param_dir}/study_{trial.number}_{trial.datetime_start.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(study_dir, exist_ok=True)

    # Copy pred.py and model/ directory
    shutil.copy2('./pred.py', f"{study_dir}/pred.py")
    if os.path.exists(f"{study_dir}/model"):
        shutil.rmtree(f"{study_dir}/model")
    shutil.copytree('./model', f"{study_dir}/model")
    preproc = True
    # Load data and build model
    print(params['TYPE_LOSS'])
    if 'FiltL' in params['TYPE_LOSS']:
        train_dataset, val_dataset, label_ratio = rippleAI_load_dataset(params, use_band='low', preprocess=preproc)
    elif 'FiltH' in params['TYPE_LOSS']:
        train_dataset, val_dataset, label_ratio = rippleAI_load_dataset(params, use_band='high', preprocess=preproc)
    elif 'FiltM' in params['TYPE_LOSS']:
        train_dataset, val_dataset, label_ratio = rippleAI_load_dataset(params, use_band='muax', preprocess=preproc)
    else:
        train_dataset, val_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)

    # if params['TYPE_MODEL'] == 'SingleCh':
    #     model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params, input_chans=1)
    # else:
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    # Early stopping with tunable patience
    callbacks = []

    # Early stopping with tunable patience
    callbacks.append(cb.EarlyStopping(
        monitor='val_max_f1_metric_horizon',  # Change monitor
        patience=50,
        mode='max',
        verbose=1,
        restore_best_weights=True
    ))
    
    # Import the new multi-objective callback
    from model.training import WeightDecayCallback, lr_scheduler
    
    use_LR = trial.suggest_categorical('USE_LR', [True, False])
    if use_LR:
        params['TYPE_REG'] += 'LR'
        callbacks.append(cb.LearningRateScheduler(lr_scheduler, verbose=1))

    if par_opt == 'AdamW':
        params['TYPE_REG'] += 'WD'
        callbacks.append(WeightDecayCallback())
    
    callbacks.append(cb.TensorBoard(log_dir=f"{study_dir}/",
                                      write_graph=True,
                                      write_images=True,
                                      update_freq='epoch'))
    # Better model checkpoint
    callbacks.append(cb.ModelCheckpoint(
        f"{study_dir}/max.weights.h5",
        monitor='val_max_f1_metric_horizon',  # Change monitor
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='max'
    ))

    # Better model checkpoint
    callbacks.append(cb.ModelCheckpoint(
        f"{study_dir}/latency.weights.h5",
        monitor='val_latency_metric',  # Change monitor
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
    ))

    # Better model checkpoint
    callbacks.append(cb.ModelCheckpoint(
        f"{study_dir}/robust.weights.h5",
        monitor='val_robust_f1',  # Change monitor
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='max'
    ))

    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=params['NO_EPOCHS'],
            callbacks=callbacks,  # this includes your F1PruningCallback
            verbose=1
        )
    except optuna.TrialPruned as e:
        # Optionally extract intermediate metrics from history if available.
        # Note: In many cases, history might not be complete.
        intermediate_f1 = None
        intermediate_latency = None
        if history is not None and 'val_max_f1_metric_horizon' in history.history:
            intermediate_f1 = max(history.history['val_max_f1_metric_horizon'])
        if history is not None and 'val_latency_metric' in history.history:
            intermediate_latency = np.mean(history.history['val_latency_metric'])
        
        trial_info = {
            'parameters': params,
            'metrics': {
                'val_accuracy': intermediate_f1,
                'val_latency': intermediate_latency
            }
        }
        with open(f"{study_dir}/trial_info.json", 'w') as f:
            json.dump(trial_info, f, indent=4)
        # Re-raise to mark the trial as pruned.
        raise e

    # If the trial completes, compute the final metrics.
    final_f1 = (np.mean(history.history['val_robust_f1']) + 
                max(history.history['val_max_f1_metric_horizon'])) / 2
    final_latency = np.mean(history.history['val_latency_metric'])
    trial_info = {
        'parameters': params,
        'metrics': {
            'val_accuracy': final_f1,
            'val_latency': final_latency
        }
    }
    with open(f"{study_dir}/trial_info.json", 'w') as f:
        json.dump(trial_info, f, indent=4)
    
    return final_f1, final_latency

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
    # if params['LEARNING_RATE'] < 1e-3:
    #     params['LEARNING_RATE'] *= 5 # 5e-4 hack
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

    if params['TYPE_ARCH'].find('Loss')>-1:
        print('Using Loss Weight:')
        loss_weight = (params['TYPE_ARCH'][params['TYPE_ARCH'].find('Loss')+4:params['TYPE_ARCH'].find('Loss')+7])
        weight = 1 if int(loss_weight[0])==1 else -1
        loss_weight = float(loss_weight[1])*10**(weight*float(loss_weight[2]))
        print(loss_weight)
        params['LOSS_WEIGHT'] = loss_weight

    # get sampling rate # little dangerous assumes 4 digits
    if 'Samp' in params['TYPE_LOSS']:
        sind = params['TYPE_LOSS'].find('Samp')
        params['SRATE'] = int(params['TYPE_LOSS'][sind+4:sind+8])
    else:
        params['SRATE'] = 1250

    if model_name.find('MixerHori') != -1:
        from model.model_fn import build_DBI_TCN_HorizonMixer as build_DBI_TCN
        from model.input_augment_weighted import rippleAI_load_dataset
    elif model_name.find('MixerDori') != -1:
        from model.model_fn import build_DBI_TCN_DorizonMixer as build_DBI_TCN
        from model.input_augment_weighted import rippleAI_load_dataset
    elif model_name.find('MixerCori') != -1:
        from model.model_fn import build_DBI_TCN_CorizonMixer as build_DBI_TCN  
        from model.input_augment_weighted import rippleAI_load_dataset
    elif model_name.find('MixerOnly') != -1:
        print('Using MixerOnly')
        from model.model_fn import build_DBI_TCN_MixerOnly as build_DBI_TCN
        from model.input_augment_weighted import rippleAI_load_dataset       
    elif model_name.find('Barlow') != -1:
        from model.model_fn import build_DBI_TCN_HorizonBarlow
        from model.input_augment_weighted import rippleAI_load_dataset
        model, train_model = build_DBI_TCN_HorizonBarlow(input_timepoints=params['NO_TIMEPOINTS'], input_chans=8, params=params)
    elif model_name.find('Hori') != -1: # predict horizon, pred lfp
        from model.model_fn import build_DBI_TCN_Horizon as build_DBI_TCN
        from model.input_augment_weighted import rippleAI_load_dataset
    elif model_name.find('Dori') != -1: # predict horizon dual loss, pred lfp and lfp
        from model.model_fn import build_DBI_TCN_Dorizon as build_DBI_TCN
        from model.input_augment_weighted import rippleAI_load_dataset
    elif model_name.find('Cori') != -1: # predict lfp, csd and lfp
        from model.model_fn import build_DBI_TCN_Corizon as build_DBI_TCN
        from model.input_augment_weighted import rippleAI_load_dataset
    elif model_name.find('CSD') != -1:
        from model.model_fn import build_DBI_TCN_CSD as build_DBI_TCN
        from model.input_aug import rippleAI_load_dataset
    elif model_name.find('Proto') != -1:
        from model.model_fn import build_DBI_TCN_Horizon_Updated
        model, train_model = build_DBI_TCN_Horizon_Updated(input_timepoints=params['NO_TIMEPOINTS'], input_chans=8, embedding_dim=params['NO_FILTERS'], params=params)
        from model.input_proto import rippleAI_load_dataset
    else:
        from model.model_fn import build_DBI_TCN
        from model.input_aug import rippleAI_load_dataset
    # input
    if ('Proto' not in model_name) and ('Barlow' not in model_name):
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
        model.summary()

    # input
    if 'NOZ' in params['TYPE_LOSS']:
        print('Not pre-processing')
        preproc = False
    else:
        print('Preprocessing')
        preproc = True
    if 'FiltL' in params['TYPE_LOSS']:
        train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, use_band='low', preprocess=preproc)
    elif 'FiltH' in params['TYPE_LOSS']:
        train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, use_band='high', preprocess=preproc)
    elif 'FiltM' in params['TYPE_LOSS']:
        train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, use_band='muax', preprocess=preproc)
    else:
        train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)
    train_size = len(list(train_dataset))
    params['RIPPLE_RATIO'] = label_ratio

    # Calculate model FLOPs using TensorFlow Profiler
    import tensorflow as tf
    @tf.function
    def get_flops(model, batch_size=1):
        concrete_func = tf.function(lambda x: model(x))
        frozen_func = concrete_func.get_concrete_function(
            tf.TensorSpec([batch_size, params['NO_TIMEPOINTS'], params['NO_CHANNELS']], tf.float32))
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='scope', options=opts)
        return flops.total_float_ops

    try:
        flops = get_flops(model)
        print(f"Estimated FLOPs: {flops}")
        print(f"Parameter count: {model.count_params()}")
    except Exception as e:
        print("Could not calculate FLOPs:", str(e))
        print(f"Parameter count: {model.count_params()}")

    # train
    if ('Proto' in model_name) or ('Barlow' in model_name):
        print('Training model with custom training loop')
        train_model(train_dataset, test_dataset, params=params)
    else:
        print('Training model with keras')
        from model.training import train_pred
        if 'SigmoidFoc' in params['TYPE_LOSS']:
            hist = train_pred(model, train_dataset, test_dataset, params['NO_EPOCHS'], params['EXP_DIR'], checkpoint_metric='val_max_f1_metric_horizon_mixer')
        else:
            hist = train_pred(model, train_dataset, test_dataset, params['NO_EPOCHS'], params['EXP_DIR'])

elif mode == 'predict':
    # modelname
    model = args.model[0]
    model_name = model
    import importlib

    if model_name.startswith('Tune'):
        # Extract study number from model name (e.g., 'Tune_45_' -> '45')
        study_num = model_name.split('_')[1]
        print(f"Loading tuned model from study {study_num}")

        # params['SRATE'] = 2500
        # Find the study directory
        import glob
        # study_dirs = glob.glob(f'studies_1/study_{study_num}_*')
        tag = args.tag[0]
        param_dir = f"params_{tag}"
        study_dirs = glob.glob(f'studies/{param_dir}/study_{study_num}_*')
        if not study_dirs:
            raise ValueError(f"No study directory found for study number {study_num}")
        study_dir = study_dirs[0]  # Take the first matching directory

        # Load trial info to get parameters
        with open(f"{study_dir}/trial_info.json", 'r') as f:
            trial_info = json.load(f)
            params.update(trial_info['parameters'])

        # pdb.set_trace()
        # Import required modules
        if 'MixerHori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_HorizonMixer as build_DBI_TCN
        elif 'MixerDori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_DorizonMixer as build_DBI_TCN
        elif 'MixerCori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_CorizonMixer as build_DBI_TCN
        from model.model_fn import CSDLayer
        from tcn import TCN
        from keras.models import load_model
        # Load weights
        params['mode'] = 'predict'
        # weight_file = f"{study_dir}/last.weights.h5"
        weight_file = f"{study_dir}/max.weights.h5"
        # weight_file = f"{study_dir}/robust.weights.h5"
        print(f"Loading weights from: {weight_file}")
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
        model.load_weights(weight_file)
    elif model_name == 'RippleNet':
        import sys, pickle, keras, h5py
        # load info on best model (path, threhsold settings)
        sys.path.insert(0, '/cs/projects/OWVinckSWR/DL/RippleNet/')
        from ripplenet.common import *
        params['TYPE_ARCH'] = 'RippleNet'

        # load info on best model (path, threhsold settings)
        # load the 'best' performing model on the validation sets
        with open('/cs/projects/OWVinckSWR/DL/RippleNet/best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
            print(best_model)

        # load the 'best' performing model on the validation sets
        model = keras.models.load_model(best_model['model_file'])
    elif model_name == 'CNN1D':
        from tensorflow import keras
        new_model = None
        model_number = 1
        arch = model_name
        params['TYPE_ARCH'] = arch
        for filename in os.listdir('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models/'):
            if f'{arch}_{model_number}' in filename:
                break
        print(filename)
        sp=filename.split('_')
        n_channels=int(sp[2][2])
        timesteps=int(sp[4][2:])
        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        if new_model==None:
            model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename), compile=False)
        else:
            model=new_model
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
    else:
        # try:
        #     import tensorflow.keras as kr
        #     params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
        #     model = kr.models.load_model(params['WEIGHT_FILE'])
        # except:
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

        # get sampling rate # little dangerous assumes 4 digits
        if 'Samp' in params['TYPE_LOSS']:
            sind = params['TYPE_LOSS'].find('Samp')
            params['SRATE'] = int(params['TYPE_LOSS'][sind+4:sind+8])
        else:
            params['SRATE'] = 1250

        # tag = ''  # MUAX, LP,
        # get model
        a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
        if model.find('CSD') != -1:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_CSD')
        else:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN')
        # from model.model_fn import build_DBI_TCN

        params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()

    params['BATCH_SIZE'] = 512*2
    from model.input_augment_weighted import rippleAI_load_dataset
    # from model.input_aug import rippleAI_load_dataset
    # from model.input_fn import rippleAI_load_dataset
    preproc = False if model_name=='RippleNet' else True
    if 'FiltL' in params['NAME']:
        val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', use_band='low', preprocess=preproc)
    elif 'FiltH' in params['NAME']:
        val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', use_band='high', preprocess=preproc)
    elif 'FiltM' in params['NAME']:
        val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', use_band='muax', preprocess=preproc)
    else:
        val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', preprocess=preproc)
    print('val_id: ', val_id)
    LFP = val_datasets[val_id]
    labels = val_labels[val_id]
    print(LFP.shape)
    np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/labels_val{0}_sf{1}.npy'.format(val_id, params['SRATE']), labels)
    np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/signals{2}_val{0}_sf{1}.npy'.format(val_id, params['SRATE'], tag), LFP)
    # get predictions
    # import pdb
    # pdb.set_trace()
    from model.cnn_ripple_utils import get_predictions_index, get_performance
    th_arr=np.linspace(0.0,1.0,11)
    n_channels = params['NO_CHANNELS']
    timesteps = params['NO_TIMEPOINTS']
    samp_freq = params['SRATE']

    all_pred_events = []
    precision=np.zeros(shape=(1,len(th_arr)))
    recall=np.zeros(shape=(1,len(th_arr)))
    F1_val=np.zeros(shape=(1,len(th_arr)))
    TP=np.zeros(shape=(1,len(th_arr)))
    FN=np.zeros(shape=(1,len(th_arr)))
    IOU=np.zeros(shape=(1,len(th_arr)))
    from keras.utils import timeseries_dataset_from_array
    if model_name == 'RippleNet':
        sample_length = params['NO_TIMEPOINTS']
        all_probs = []
        for ich in range(LFP.shape[1]):
            train_x = timeseries_dataset_from_array(LFP[:,ich]/1000, None, sequence_length=sample_length, sequence_stride=1, batch_size=params["BATCH_SIZE"])
            windowed_signal = np.squeeze(model.predict(train_x, verbose=1))
            probs = np.hstack((windowed_signal[0,:-1], windowed_signal[:, -1]))
            all_probs.append(probs)
        probs = np.array(all_probs).mean(axis=0)
        # pdb.set_trace()
    elif model_name == 'CNN1D':
        sample_length = 16
        train_x = timeseries_dataset_from_array(LFP, None, sequence_length=sample_length, sequence_stride=1, batch_size=params["BATCH_SIZE"])
        windowed_signal = np.squeeze(model.predict(train_x, verbose=1))
        probs = np.hstack((np.zeros((15, 1)).flatten(), windowed_signal))
    else:
        sample_length = params['NO_TIMEPOINTS']
        train_x = timeseries_dataset_from_array(LFP, None, sequence_length=sample_length, sequence_stride=1, batch_size=params["BATCH_SIZE"])
        windowed_signal = np.squeeze(model.predict(train_x, verbose=1))

        if model_name.find('Hori') != -1 or model_name.find('Dori') != -1 or model_name.find('Cori') != -1:
            if len(windowed_signal.shape) == 3:
                probs = np.hstack((windowed_signal[0,:-1,-1], windowed_signal[:, -1,-1]))
                horizon = np.vstack((windowed_signal[0,:-1,:-1], windowed_signal[:, -1,:-1]))
            else:
                probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal[:,-1]))
                horizon = np.vstack((np.zeros((sample_length-1, 8)), windowed_signal[:, :-1]))
        elif  model_name.startswith('Tune') != -1:
            probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal[:,-1]))
            horizon = np.vstack((np.zeros((sample_length-1, 8)), windowed_signal[:, :-1]))
        elif model_name.find('Proto') != -1:
            probs = np.hstack((windowed_signal[0,:-1], windowed_signal[:, -1]))
        elif model_name.find('Base_') != -1:
            probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal))
        else:
            probs = np.hstack((windowed_signal[0,:-1], windowed_signal[:, -1]))
    np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/preds_val{0}_{1}_sf{2}.npy'.format(val_id, model_name, params['SRATE']), probs)
    # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/preds_val{0}_{1}_{3}_sf{2}.npy'.format(val_id, model_name, params['SRATE'], tag), probs)
    if model_name.find('Hori') != -1 or model_name.find('Dori') != -1 or model_name.find('Cori') != -1 or model_name.startswith('Tune') != -1:
        np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/horis_val{0}_{1}_sf{2}.npy'.format(val_id, model_name, params['SRATE']), horizon)
        # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/horis_val{0}_{1}_{3}_sf{2}.npy'.format(val_id, model_name, params['SRATE'], tag), horizon)

    for i,th in enumerate(th_arr):
        pred_val_events = get_predictions_index(probs,th)/samp_freq
        [precision[0,i], recall[0,i], F1_val[0,i], tmpTP, tmpFN, tmpIOU] = get_performance(pred_val_events,labels,verbose=False)
        TP[0,i] = tmpTP.sum()
        FN[0,i] = tmpFN.sum()
        IOU[0,i] = np.mean(tmpIOU.sum(axis=0))
    stats = np.stack((precision, recall, F1_val, TP, FN, IOU), axis=-1)
    np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/stats_val{0}_{1}_sf{2}.npy'.format(val_id, model_name, params['SRATE']), stats)
    # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/stats_val{0}_{1}_{3}_sf{2}.npy'.format(val_id, model_name, params['SRATE'], tag), stats)

elif mode == 'predictSynth':

    # modelname
    model = args.model[0]
    model_name = model
    import importlib

    if model_name == 'RippleNet':
        import sys, pickle, keras, h5py
        sys.path.insert(0, '/cs/projects/OWVinckSWR/DL/RippleNet/')
        from ripplenet.common import *
        params['TYPE_ARCH'] = 'RippleNet'

        # load info on best model (path, threhsold settings)
        with open('/cs/projects/OWVinckSWR/DL/RippleNet/best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
            print(best_model)

        # load the 'best' performing model on the validation sets
        model = keras.models.load_model(best_model['model_file'])
    elif model_name == 'CNN1D':
        from tensorflow import keras
        new_model = None
        model_number = 1
        arch = model_name
        params['TYPE_ARCH'] = arch

        for filename in os.listdir('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models/'):
            if f'{arch}_{model_number}' in filename:
                break
        print(filename)
        sp=filename.split('_')
        n_channels=int(sp[2][2])
        timesteps=int(sp[4][2:])

        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        if new_model==None:
            model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename), compile=False)
        else:
            model=new_model
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
    else:
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
            if model.find('CSD') != -1:
                build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_CSD')
            else:
                build_DBI_TCN = getattr(a_model, 'build_DBI_TCN')
            # from model.model_fn import build_DBI_TCN


            params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
            model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()

    # synth = np.load('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/synth_stim_30k.npy')
    synth = np.load('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/synth_stim.npy')
    if not (model_name == 'RippleNet'):
        synth = np.tile(synth, (1,8))#*5
    else:
        synth /= 2

    # synth = synth[9*1250:-1250*12]
    from keras.utils import timeseries_dataset_from_array

    # get predictions
    samp_freq = 30000
    params["BATCH_SIZE"] = 512*4
    n_channels = params['NO_CHANNELS']
    timesteps = params['NO_TIMEPOINTS']
    sample_length = params['NO_TIMEPOINTS']*2
    train_x = timeseries_dataset_from_array(synth, None, sequence_length=sample_length, sequence_stride=1, batch_size=params["BATCH_SIZE"])
    probs = np.squeeze(model.predict(train_x))
    # probs = np.hstack((probs[0,:-1], probs[:, -1]))

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    # probs = moving_average(probs, 4)

    from scipy.signal import decimate
    import matplotlib.pyplot as plt
    synth = np.load('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/synth_stim.npy')
    # synth = np.load('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/synth_stim_30k.npy')
    synth = (synth-np.min(synth))/(np.max(synth)-np.min(synth))
    # synth = synth[600:,:]
    # synth = synth[50+9*1250:-1250*12,:]
    synth = synth[50:,:]
    tt = np.arange(synth.shape[0])/samp_freq
    plt.plot(tt, synth[:, 0])
    tt = np.arange(probs.shape[0])/samp_freq
    plt.plot(tt, probs)
    plt.show()
    # pdb.set_trace()

elif mode == 'predictPlot':
    import matplotlib.pyplot as plt
    import importlib
    from model.cnn_ripple_utils import get_predictions_index, get_performance

    # modelname
    model = args.model[0]
    model_name = model
    import importlib

    if model_name == 'RippleNet':
        import sys, pickle, keras, h5py
        sys.path.insert(0, '/cs/projects/OWVinckSWR/DL/RippleNet/')
        from ripplenet.common import *
        params['TYPE_ARCH'] = 'RippleNet'

        # load info on best model (path, threhsold settings)
        with open('/cs/projects/OWVinckSWR/DL/RippleNet/best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
            print(best_model)

        # load the 'best' performing model on the validation sets
        model = keras.models.load_model(best_model['model_file'])
    elif model_name == 'CNN1D':
        from tensorflow import keras
        new_model = None
        model_number = 1
        arch = model_name
        params['TYPE_ARCH'] = arch

        for filename in os.listdir('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models/'):
            if f'{arch}_{model_number}' in filename:
                break
        print(filename)
        sp=filename.split('_')
        n_channels=int(sp[2][2])
        timesteps=int(sp[4][2:])

        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        if new_model==None:
            model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename), compile=False)
        else:
            model=new_model
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
    else:
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

            params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
            model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()

    # get inputs
    # a_input = importlib.import_module('experiments.{0}.model.input_fn'.format(model_name))
    # rippleAI_load_dataset = getattr(a_input, 'rippleAI_load_dataset')

    from model.input_fn import rippleAI_load_dataset
    val_datasets, val_labels = rippleAI_load_dataset(params, mode='test')

    # probs
    for val_id in range(3):
        val_pred = np.load('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/preds_val{0}_{1}.npy'.format(val_id, model_name))
        val_pred = [val_pred]

        val_datasets = [val_datasets[0]]

        # Validation plot in the second axs
        samp_freq = 1250
        th_arr=np.linspace(0.0,1.0,11)
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
                pred_val_events = get_predictions_index(pred,th)/samp_freq
                # pred_val_events = get_predictions_index(pred,th)/1250
                [precision[j,i], recall[j,i], F1_val[j,i], tmpTP, tmpFN, tmpIOU] = get_performance(pred_val_events,val_labels[j],verbose=False)
                TP[j,i] = tmpTP.sum()
                FN[j,i] = tmpFN.sum()
                IOU[j,i] = np.median(tmpIOU.sum(axis=0))
                tmp_pred.append(pred_val_events)
            all_pred_events.append(tmp_pred)

        # pdb.set_trace()
        # pick model
        # print(F1_val[0])
        # print(F1_val[1])
        # mind = np.argmax(F1_val[0])
        # best_preds = all_pred_events[0][mind]
        # pred_vec = np.zeros(val_datasets[0].shape[0])
        # label_vec = np.zeros(val_datasets[0].shape[0])

        # for pred in best_preds:
        #     pred_vec[int(pred[0]*1250):int(pred[1]*1250)] = 1

        # for lab in val_labels[0]:
        #     label_vec[int(lab[0]*1250):int(lab[1]*1250)] = 0.9

        stats = np.stack((precision, recall, F1_val, TP, FN, IOU), axis=-1)

        for j,pred in enumerate(val_pred):
            np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/preds_val{0}_{1}.npy'.format(val_id, model_name), pred)
            np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/stats_val{0}_{1}.npy'.format(val_id, model_name), stats[j,])

    # for pred in val_labels[0]:
    #     rip_begin = int(pred[0]*1250)
    #     plt.plot(val_datasets[0][rip_begin-256:rip_begin+256, :]/3, 'gray')
    #     plt.plot(val_pred[0][rip_begin-256:rip_begin+256], 'k')
    #     plt.plot(val_pred[0][rip_begin-256:rip_begin+256]*pred_vec[rip_begin-256:rip_begin+256], 'r')
    #     plt.plot(label_vec[rip_begin-256:rip_begin+256], 'k')
    #     plt.show()

elif mode=='export':
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers

    # modelname
    model = args.model[0]
    model_name = model
    import importlib

    # necessary !!!
    # tf.compat.v1.disable_eager_execution()

    if model_name == 'RippleNet':
        import sys, pickle, keras, h5py
        sys.path.insert(0, '/cs/projects/OWVinckSWR/DL/RippleNet/')
        from ripplenet.common import *
        params['TYPE_ARCH'] = 'RippleNet'

        # load info on best model (path, threhsold settings)
        with open('/cs/projects/OWVinckSWR/DL/RippleNet/best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
            print(best_model)

        # load the 'best' performing model on the validation sets
        model = keras.models.load_model(best_model['model_file'])
    elif model_name == 'CNN1D':
        from tensorflow import keras
        new_model = None
        model_number = 1
        arch = model_name
        params['TYPE_ARCH'] = arch

        for filename in os.listdir('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models/'):
            if f'{arch}_{model_number}' in filename:
                break
        print(filename)
        sp=filename.split('_')
        n_channels=int(sp[2][2])
        timesteps=int(sp[4][2:])

        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        if new_model==None:
            model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename), compile=False)
        else:
            model=new_model
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
    else:
        # try:
        #     import tensorflow.keras as kr
        #     params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
        #     model = kr.models.load_model(params['WEIGHT_FILE'])
        # except:
        # get model parameters
        print(model_name)
        param_lib = model_name.split('_')
        # pdb.set_trace()
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
        # a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
        a_model = importlib.import_module('model.model_fn')
        # if model.find('CSD') != -1:
        #     build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_CSD')
        if model.find('Hori') != -1:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_Horizon')
        elif model.find('Dori') != -1:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_Dorizon')
        elif model.find('Cori') != -1:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_Corizon')

        params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()

    # pdb.set_trace()
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    full_model = tf.function(lambda x: model(x))
    pdb.set_trace()
    full_model = full_model.get_concrete_function([tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="x")])
    frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
    frozen_func.graph.as_graph_def(add_shapes=True)
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    # print("Frozen model layers: ")
    # for layer in layers:
    #     print(layer)
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    os.mkdir('./frozen_models/{}'.format(model_name))
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir="./frozen_models/{}".format(model_name),
                    name="simple_frozen_graph.pb",
                    as_text=False)
    # model.save(
    #     './experiments/'+model_name+'/lucid_model.pb',
    #     image_shape=[W, W, 3],
    #     input_name='input_1',
    #     output_names=['flatten/Reshape'],
    #     image_value_range=[-117, 138],  # 114.799
    # )

    # from tensorflow import keras

    # from tensorflow.compat.v1 import graph_util
    # from tensorflow.python.keras import backend as K


    # # h5_path = '/path/to/model.h5'
    # # model = keras.models.load_model(h5_path)
    # # model.summary()
    # # save pb
    # with K.get_session() as sess:
    #     output_names = [out.op.name for out in model.outputs]
    #     input_graph_def = sess.graph.as_graph_def()
    #     for node in input_graph_def.node:
    #         node.device = ""
    #     graph = graph_util.remove_training_nodes(input_graph_def)
    #     # graph_frozen = graph_util.convert_variables_to_constants(sess, graph, output_names, lower_control_flow=False)
    #     tf.io.write_graph(graph_frozen, 'frozen_models/simple_frozen_graph.pb', as_text=False)

    # def frozen_keras_graph(model):
    #     from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    #     from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

    #     real_model = tf.function(model).get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    #     frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    #     input_tensors = [
    #         tensor for tensor in frozen_func.inputs
    #         if tensor.dtype != tf.resource
    #     ]
    #     output_tensors = frozen_func.outputs

    #     graph_def = run_graph_optimizations(
    #         graph_def,
    #         input_tensors,
    #         output_tensors,
    #         config=get_grappler_config(["constfold", "function"]),
    #         graph=frozen_func.graph)
    #     return graph_def

    # graph_def = frozen_keras_graph(model)

    # # frozen_func.graph.as_graph_def()
    # tf.io.write_graph(graph_def, './frozen_models', 'simple_frozen_graph.pb')
    # from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    # # Convert Keras model to ConcreteFunction
    # full_model = tf.function(lambda x: model(x))
    # full_model = full_model.get_concrete_function(
    #     x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # # Get frozen ConcreteFunction
    # frozen_func = convert_variables_to_constants_v2(full_model)#, lower_control_flow=False)
    # frozen_func.graph.as_graph_def()

    # # inspect the layers operations inside your frozen graph definition and see the name of its input and output tensors
    # layers = [op.name for op in frozen_func.graph.get_operations()]
    # print("-" * 50)
    # print("Frozen model layers: ")
    # for layer in layers:
    #     print(layer)

    # print("-" * 50)
    # print("Frozen model inputs: ")
    # print(frozen_func.inputs)
    # print("Frozen model outputs: ")
    # print(frozen_func.outputs)

    # # Save frozen graph from frozen ConcreteFunction to hard drive
    # # serialize the frozen graph and its text representation to disk.
    # tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
    #                 logdir="./frozen_models",
    #                 name="simple_frozen_graph.pb",
    #                 as_text=False)

    # #Optional
    # tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
    #                 logdir="./frozen_models",
    #                 name="simple_frozen_graph.pbtxt",
    #                 as_text=True)

    # model.summary()

    # # Save model in h5 format
    # model.save('export/model_{}.h5'.format(model_name))
    # # Save model in pb format
    # model.save('export/model_{}.pb'.format(model_name))

elif mode == 'embedding':

    # modelname
    model = args.model[0]
    model_name = model
    import importlib

    if model_name == 'RippleNet':
        import sys, pickle, keras, h5py
        sys.path.insert(0, '/cs/projects/OWVinckSWR/DL/RippleNet/')
        from ripplenet.common import *
        params['TYPE_ARCH'] = 'RippleNet'

        # load info on best model (path, threhsold settings)
        with open('/cs/projects/OWVinckSWR/DL/RippleNet/best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
            print(best_model)

        # load the 'best' performing model on the validation sets
        model = keras.models.load_model(best_model['model_file'])

    elif model_name == 'CNN1D':
        from tensorflow import keras
        new_model = None
        model_number = 1
        arch = model_name
        params['TYPE_ARCH'] = arch
        for filename in os.listdir('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models/'):
            if f'{arch}_{model_number}' in filename:
                break
        print(filename)
        sp=filename.split('_')
        n_channels=int(sp[2][2])
        timesteps=int(sp[4][2:])

        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        if new_model==None:
            model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename), compile=False)
        else:
            model=new_model
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
    else:
        # try:
        #     import tensorflow.keras as kr
        #     params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
        #     model = kr.models.load_model(params['WEIGHT_FILE'])
        # except:
        # get model parameters
        print(model_name)
        param_lib = model_name.split('_')
        # pdb.set_trace()
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

        # get sampling rate # little dangerous assumes 4 digits
        if 'Samp1250' in model_name:
            pro_data = np.array([decimate(pro_data[:,k], 2) for k in range(pro_data.shape[1])])
        elif 'Samp2500' in model_name:
            pro_data = np.transpose(pro_data, [1,0])
        else:
            pro_data = np.transpose(pro_data, [1,0])
        ripples = np.tile(np.expand_dims(pro_data, axis=-1), (1,1,8))



        params["BATCH_SIZE"] = 512*4
        n_channels = params['NO_CHANNELS']
        timesteps = params['NO_TIMEPOINTS']
        sample_length = params['NO_TIMEPOINTS']*2
        from keras.utils import timeseries_dataset_from_array

        test_ripples = tf.data.Dataset.from_tensor_slices(ripples[:,-sample_length:,:]).batch(params["BATCH_SIZE"])
        # test_ripples = timeseries_dataset_from_array(ripples, None, sequence_length=sample_length, sequence_stride=1, batch_size=params["BATCH_SIZE"])

        tmp_act = model_act.predict(test_ripples)

        # save activations
        for il in range(len(tmp_act)):
            np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/activations/{0}_act{1}.npy'.format(model_name, il), tmp_act[il])

    def get_embeddings(model, data):
        """Extract embeddings from the model's penultimate layer"""
        embedding_model = tf.keras.Model(inputs=model.input,
                                       outputs=model.get_layer('global_average_pooling1d').output)
        return embedding_model.predict(data)

    def extract_centered_events(data, labels, params):
        """Extract events centered exactly on SWR peaks"""
        window_size = params['NO_TIMEPOINTS']
        pos_events = []

        for start, end in labels:
            # Calculate the center point of the ripple
            center_time = (start + end) / 2
            center_idx = int(center_time * params['SRATE'])

            # Only include if we have enough padding on both sides
            if center_idx - window_size//2 >= 0 and center_idx + window_size//2 < data.shape[0]:
                event = data[center_idx - window_size//2:center_idx + window_size//2]
                pos_events.append(event)

        return np.array(pos_events)

    # Process each validation set
    for val_id in range(len(val_datasets)):
        LFP = val_datasets[val_id]
        labels = val_labels[val_id]

        # Extract centered ripple events
        ripple_events = extract_centered_events(LFP, labels, params)
        print(f"Extracted {len(ripple_events)} centered ripple events")

        # Get embeddings
        embeddings = get_embeddings(model, ripple_events)

        # Calculate pairwise distances between all ripple events
        from scipy.spatial.distance import pdist, squareform
        pairwise_distances = squareform(pdist(embeddings))

        # Basic statistics
        stats = {
            'mean_distance': np.mean(pairwise_distances),
            'std_distance': np.std(pairwise_distances),
            'min_distance': np.min(pairwise_distances[pairwise_distances > 0]),
            'max_distance': np.max(pairwise_distances),
            'embedding_mean': np.mean(embeddings, axis=0),
            'embedding_std': np.std(embeddings, axis=0),
        }

        # Dimensionality reduction for visualization
        import umap
        reducer = umap.UMAP(random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)

        # Clustering to identify potential subtypes
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(embeddings)

        # Save results
        np.savez(f'embeddings_val{val_id}_analysis.npz',
                 ripple_events=ripple_events,
                 embeddings=embeddings,
                 embedding_2d=embedding_2d,
                 clusters=clustering.labels_,
                 stats=stats)

        # Visualization
        plt.figure(figsize=(15, 5))

        # Plot UMAP embedding colored by cluster
        plt.subplot(131)
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                            c=clustering.labels_, cmap='Spectral')
        plt.colorbar(scatter)
        plt.title('UMAP Visualization\nColored by Cluster')

        # Plot average waveform with std
        plt.subplot(132)
        mean_waveform = np.mean(ripple_events, axis=0)
        std_waveform = np.std(ripple_events, axis=0)
        t = np.arange(mean_waveform.shape[0]) / params['SRATE']
        for ch in range(mean_waveform.shape[1]):
            plt.plot(t, mean_waveform[:, ch] + ch*0.5, 'k', alpha=0.8)
            plt.fill_between(t,
                           mean_waveform[:, ch] - std_waveform[:, ch] + ch*0.5,
                           mean_waveform[:, ch] + std_waveform[:, ch] + ch*0.5,
                           alpha=0.2)
        plt.title('Average Waveform\nwith Standard Deviation')

        # Plot distance distribution
        plt.subplot(133)
        plt.hist(pairwise_distances.flatten(), bins=50)
        plt.title('Distribution of\nPairwise Distances')

        plt.tight_layout()
        plt.savefig(f'embeddings_val{val_id}_analysis.png')
        plt.close()

        # Print statistics
        print(f"\nValidation set {val_id} embedding statistics:")
        print(f"Number of events: {len(ripple_events)}")
        print(f"Mean pairwise distance: {stats['mean_distance']:.4f}")
        print(f"Std pairwise distance: {stats['std_distance']:.4f}")
        print(f"Number of clusters: {len(np.unique(clustering.labels_))}")

elif mode == 'embedding_analysis':
    # modelname
    model = args.model[0]
    model_name = model
    import importlib

    if model_name.startswith('Tune'):
        # Extract study number from model name (e.g., 'Tune_45_' -> '45')
        study_num = model_name.split('_')[1]
        print(f"Loading tuned model from study {study_num}")

        # params['SRATE'] = 2500
        # Find the study directory
        import glob
        # study_dirs = glob.glob(f'studies_1/study_{study_num}_*')
        tag = args.tag[0]
        param_dir = f"params_{tag}"
        study_dirs = glob.glob(f'studies/{param_dir}/study_{study_num}_*')
        if not study_dirs:
            raise ValueError(f"No study directory found for study number {study_num}")
        study_dir = study_dirs[0]  # Take the first matching directory

        # Load trial info to get parameters
        with open(f"{study_dir}/trial_info.json", 'r') as f:
            trial_info = json.load(f)
            params.update(trial_info['parameters'])

        # pdb.set_trace()
        # Import required modules
        if 'MixerHori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_HorizonMixer as build_DBI_TCN
        elif 'MixerDori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_DorizonMixer as build_DBI_TCN
        elif 'MixerCori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_CorizonMixer as build_DBI_TCN
        from model.model_fn import CSDLayer
        from tcn import TCN
        from keras.models import load_model

        params['mode'] = 'predict'
        # Build model with trial parameters
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)

        # Load weights
        # weight_file = f"{study_dir}/last.weights.h5"
        weight_file = f"{study_dir}/max.weights.h5"
        print(f"Loading weights from: {weight_file}")
        model.load_weights(weight_file)

    elif model_name == 'RippleNet':
        import sys, pickle, keras, h5py
        sys.path.insert(0, '/cs/projects/OWVinckSWR/DL/RippleNet/')
        from ripplenet.common import *
        params['TYPE_ARCH'] = 'RippleNet'

        # load info on best model (path, threhsold settings)
        with open('/cs/projects/OWVinckSWR/DL/RippleNet/best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
            print(best_model)

        # load the 'best' performing model on the validation sets
        model = keras.models.load_model(best_model['model_file'])

    elif model_name == 'CNN1D':
        from tensorflow import keras
        new_model = None
        model_number = 1
        arch = model_name
        params['TYPE_ARCH'] = arch
        for filename in os.listdir('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models/'):
            if f'{arch}_{model_number}' in filename:
                break
        print(filename)
        sp=filename.split('_')
        n_channels=int(sp[2][2])
        timesteps=int(sp[4][2:])

        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        if new_model==None:
            model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename), compile=False)
        else:
            model=new_model
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
    else:
        # try:
        #     import tensorflow.keras as kr
        #     params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
        #     model = kr.models.load_model(params['WEIGHT_FILE'])
        # except:
        # get model parameters
        pdb.set_trace()
        print(model_name)
        param_lib = model_name.split('_')
        # pdb.set_trace()
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

        # get sampling rate # little dangerous assumes 4 digits
        if 'Samp' in params['TYPE_LOSS']:
            sind = params['TYPE_LOSS'].find('Samp')
            params['SRATE'] = int(params['TYPE_LOSS'][sind+4:sind+8])
        else:
            params['SRATE'] = 1250

        # tag = ''  # MUAX, LP,

        # get model
        # a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
        a_model = importlib.import_module('model.model_fn')
        # if model.find('CSD') != -1:
        #     build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_CSD')

        if model_name.find('Tune') != -1:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_HorizonMixer')
            from keras.utils import custom_object_scope
            from model.model_fn import CSDLayer
            from tcn import TCN
            from keras.models import load_model


            params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'best_f1_model.h5'
            print((params['WEIGHT_FILE']))
            with custom_object_scope({'CSDLayer': CSDLayer, 'TCN': TCN}):
                model = load_model(params['WEIGHT_FILE'])
        elif model_name.find('MixerHori') != -1:
            from model.model_fn import build_DBI_TCN_HorizonMixer as build_DBI_TCN
        elif model_name.find('MixerDori') != -1:
            from model.model_fn import build_DBI_TCN_DorizonMixer as build_DBI_TCN
        elif model_name.find('MixerCori') != -1:
            from model.model_fn import build_DBI_TCN_CorizonMixer as build_DBI_TCN
        elif model_name.find('Barlow') != -1:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_HorizonBarlow')
            from keras.utils import custom_object_scope
            from model.model_fn import CSDLayer
            from tcn import TCN
            from keras.models import load_model
            params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'best_f1_model.h5'
            print((params['WEIGHT_FILE']))
            with custom_object_scope({'CSDLayer': CSDLayer, 'TCN': TCN}):
                model = load_model(params['WEIGHT_FILE'])
            # model.summary()
        elif model.find('Hori') != -1:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_Horizon')
        elif model.find('Dori') != -1:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_Dorizon')
        elif model.find('Cori') != -1:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_Corizon')
        elif model_name.find('Proto') != -1:
            from keras.utils import custom_object_scope
            from model.model_fn import CSDLayer
            from tcn import TCN
            from keras.models import load_model
            # with custom_object_scope({'CSDLayer': CSDLayer, 'TCN': TCN}):
            params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'best_f1_model.h5'
            print((params['WEIGHT_FILE']))
            with custom_object_scope({'CSDLayer': CSDLayer, 'TCN': TCN}):
                model = load_model(params['WEIGHT_FILE'])
            # model.summary()
            # model.load('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/experiments/{0}/'.format(model_name)+'best_model.h5')
            # import pdb
            # build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_Horizon_Updated')
            # model, train_model = build_DBI_TCN_Horizon_Updated(input_timepoints=params['NO_TIMEPOINTS'], input_chans=8, embedding_dim=params['NO_FILTERS'], params=params)
        else:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN')
        # pdb.set_trace()
        # # from model.model_fn import build_DBI_TCN
        # # from model.model_fn import build_DBI_TCN
        # # from model.model_fn import build_DBI_TCN_CSD as build_DBI_TCN

        if (model_name.find('Proto') == -1) and (model_name.find('Barlow') == -1):
            params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
            # params['WEIGHT_FILE'] = ''
            model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
        # from keras.utils import custom_object_scope
        # from model.model_fn import CSDLayer
        # from tcn import TCN
        # from keras.models import load_model
        # with custom_object_scope({'CSDLayer': CSDLayer, 'TCN': TCN}):
        #     model = load_model(params['WEIGHT_FILE'])
    model.summary()

    params['BATCH_SIZE'] = 512*2
    from model.input_augment_weighted import rippleAI_load_dataset
    # from model.input_aug import rippleAI_load_dataset
    # from model.input_fn import rippleAI_load_dataset

    preproc = False if model_name=='RippleNet' else True
    if 'FiltL' in params['NAME']:
        val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', use_band='low', preprocess=preproc)
    elif 'FiltH' in params['NAME']:
        val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', use_band='high', preprocess=preproc)
    elif 'FiltM' in params['NAME']:
        val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', use_band='muax', preprocess=preproc)
    else:
        val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', preprocess=preproc)


    def extract_events(data, labels, params, window_size=128):
        """Extract positive and negative events with domain-specific considerations"""
        pos_events = []
        neg_events = []
        neg_far_events = []  # Events very far from SWRs
        neg_edge_events = [] # Events just outside SWR boundaries

        # Calculate ripple duration statistics for scaling
        ripple_durations = [(end - start) for start, end in labels]
        avg_duration = np.mean(ripple_durations)

        forbidden_windows = []  # Track regions we can't use for negative samples
        for start, end in labels:
            start_idx = int(start * params['SRATE'])
            end_idx = int(end * params['SRATE'])
            mid_idx = (start_idx + end_idx) // 2

            # Add ripple window to forbidden regions (with safety margin)
            margin = int(0.1 * params['SRATE'])  # 100ms margin
            forbidden_windows.append((start_idx - margin, end_idx + margin))

            # Extract positive event centered on ripple
            if mid_idx - window_size//2 >= 0 and mid_idx + window_size//2 < data.shape[0]:
                pos_events.append(data[mid_idx - window_size//2:mid_idx + window_size//2])

                # Get edge events (just outside ripple)
                pre_edge_start = start_idx - window_size - margin
                if pre_edge_start >= 0:
                    neg_edge_events.append(data[pre_edge_start:pre_edge_start + window_size])

                post_edge_start = end_idx + margin
                if post_edge_start + window_size < data.shape[0]:
                    neg_edge_events.append(data[post_edge_start:post_edge_start + window_size])

        # Get negative events far from any ripple (>2 seconds)
        far_margin = int(2.0 * params['SRATE'])
        potential_regions = []
        last_end = 0
        for start, _ in forbidden_windows:
            if start - last_end > window_size + 2*far_margin:
                potential_regions.append((last_end + far_margin, start - far_margin))
            last_end = max(last_end, start)

        # Randomly sample from allowed regions
        n_samples = len(pos_events)
        for region_start, region_end in potential_regions:
            if region_end - region_start > window_size:
                for _ in range(n_samples // len(potential_regions)):
                    start_idx = np.random.randint(region_start, region_end - window_size)
                    neg_far_events.append(data[start_idx:start_idx + window_size])

        return np.array(pos_events), np.array(neg_edge_events), np.array(neg_far_events)

    def create_variations(events, params):
        """Create physiologically relevant variations of events"""
        variations = []
        labels = []

        # Amplitude scaling (mimicking distance from probe variations)
        scales = np.linspace(0.5, 2.0, 5)
        for scale in scales:
            scaled = events * scale
            variations.append(scaled)
            labels.extend([f'amp_{scale:.1f}'] * len(events))

        # Frequency-specific noise (targeting different bands)
        from scipy import signal

        def add_band_noise(data, band, noise_level=0.2):
            nyq = params['SRATE'] / 2
            b, a = signal.butter(3, band/nyq, btype='bandpass')
            noise = np.random.normal(0, noise_level, data.shape)
            filtered_noise = signal.filtfilt(b, a, noise, axis=1)
            return data + filtered_noise

        # Add ripple band noise (150-250 Hz)
        variations.append(add_band_noise(events, [150, 250]))
        labels.extend(['ripple_noise'] * len(events))

        # Add gamma band noise (30-80 Hz)
        variations.append(add_band_noise(events, [30, 80]))
        labels.extend(['gamma_noise'] * len(events))

        # Add theta band noise (4-12 Hz)
        variations.append(add_band_noise(events, [4, 12]))
        labels.extend(['theta_noise'] * len(events))

        # Temporal jitter (small shifts)
        jitter_samples = [-4, -2, 2, 4]  # Small temporal shifts
        for shift in jitter_samples:
            if shift > 0:
                jittered = np.pad(events, ((0,0), (shift,0), (0,0)))[:, :-shift, :]
            else:
                jittered = np.pad(events, ((0,0), (0,-shift), (0,0)))[:, -shift:, :]
            variations.append(jittered)
            labels.extend([f'jitter_{shift}'] * len(events))

        # Channel dropout (simulating electrode issues)
        n_channels = events.shape[-1]
        for i in range(2):  # Drop 1-2 random channels
            channel_dropout = events.copy()
            drop_channels = np.random.choice(n_channels, i+1, replace=False)
            channel_dropout[:, :, drop_channels] = 0
            variations.append(channel_dropout)
            labels.extend([f'drop_{i+1}ch'] * len(events))

        return np.vstack(variations), labels

    # Process each validation set
    for val_id in range(len(val_datasets)):
        print(f"Processing validation set {val_id}")

        # Extract different types of events
        pos_events, neg_edge_events, neg_far_events = extract_events(
            val_datasets[val_id], val_labels[val_id], params)

        # Create variations
        pos_variations, pos_var_labels = create_variations(pos_events, params)
        neg_edge_variations, neg_edge_var_labels = create_variations(neg_edge_events, params)
        neg_far_variations, neg_far_var_labels = create_variations(neg_far_events, params)

        # Get embeddings
        pos_embeddings = get_embeddings(model, pos_variations)
        neg_edge_embeddings = get_embeddings(model, neg_edge_variations)
        neg_far_embeddings = get_embeddings(model, neg_far_variations)

        # Calculate embedding statistics
        stats = {
            'pos_centroid': np.mean(pos_embeddings, axis=0),
            'neg_edge_centroid': np.mean(neg_edge_embeddings, axis=0),
            'neg_far_centroid': np.mean(neg_far_embeddings, axis=0),
            'pos_variance': np.var(pos_embeddings, axis=0),
            'neg_edge_variance': np.var(neg_edge_embeddings, axis=0),
            'neg_far_variance': np.var(neg_far_embeddings, axis=0),
            'edge_distance': np.linalg.norm(
                np.mean(pos_embeddings, axis=0) - np.mean(neg_edge_embeddings, axis=0)),
            'far_distance': np.linalg.norm(
                np.mean(pos_embeddings, axis=0) - np.mean(neg_far_embeddings, axis=0))
        }

        # Save detailed analysis
        np.savez(f'embedding_analysis_val{val_id}.npz',
                 pos_embeddings=pos_embeddings,
                 neg_edge_embeddings=neg_edge_embeddings,
                 neg_far_embeddings=neg_far_embeddings,
                 pos_labels=pos_var_labels,
                 neg_edge_labels=neg_edge_var_labels,
                 neg_far_labels=neg_far_var_labels,
                 stats=stats)

        # Plotting
        plt.figure(figsize=(15, 10))

        # PCA visualization
        pca = PCA(n_components=2)
        all_embeddings = np.vstack([pos_embeddings, neg_edge_embeddings, neg_far_embeddings])
        all_labels = (['positive'] * len(pos_embeddings) +
                     ['negative_edge'] * len(neg_edge_embeddings) +
                     ['negative_far'] * len(neg_far_embeddings))
        all_variations = pos_var_labels + neg_edge_var_labels + neg_far_var_labels

        pca_result = pca.fit_transform(all_embeddings)

        plt.subplot(2, 2, 1)
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1],
                       hue=all_labels, style=all_variations)
        plt.title('PCA Visualization')

        # Print statistics
        print(f"\nValidation set {val_id} statistics:")
        print(f"Edge boundary distance: {stats['edge_distance']:.4f}")
        print(f"Far negative distance: {stats['far_distance']:.4f}")
        print(f"Positive variance: {np.mean(stats['pos_variance']):.4f}")
        print(f"Edge negative variance: {np.mean(stats['neg_edge_variance']):.4f}")
        print(f"Far negative variance: {np.mean(stats['neg_far_variance']):.4f}")

        plt.tight_layout()
        plt.savefig(f'embedding_analysis_val{val_id}.png')
        plt.close()


elif mode == 'tune_server':
    import optuna
    import logging
    import os
    from optuna.samplers import NSGAIISampler

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    tag = args.tag[0]
    param_dir = f'params_{tag}'

    if not os.path.exists(f'studies/{param_dir}'):
        os.makedirs(f'studies/{param_dir}')

    # Configure storage with more resilient settings
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///studies/{param_dir}/{param_dir}.db",
        heartbeat_interval=1,
        grace_period=600,  # Increased grace period
        failed_trial_callback=lambda study_id, trial_id: True,  # Auto-fail dead trials
        engine_kwargs={
            "connect_args": {
                "timeout": 300,  # Longer timeout
                "isolation_level": "IMMEDIATE"  # Better concurrent access
            }
        }
    )
    # study = optuna.create_study(
    #     study_name=param_dir,
    #     storage=storage,
    #     direction='maximize',
    #     load_if_exists=True,
    #     sampler=optuna.samplers.TPESampler(
    #         n_startup_trials=24,    # Increased for better exploration
    #         n_ei_candidates=24,     # More candidates for parallel optimization
    #         multivariate=True,      # Enable multivariate sampling
    #         seed=42,
    #         constant_liar=True      # Better parallel optimization
    #     ),
    #     pruner=optuna.pruners.MedianPruner(
    #         n_startup_trials=16,     # Allow more trials before pruning starts
    #         n_warmup_steps=30,       # Don't prune before 30 epochs
    #         interval_steps=5,         # Check more frequently for pruning
    #     )
    # )
    study = optuna.create_study(
    study_name=param_dir,
    storage=storage,
    directions=["maximize", "minimize"],  # Maximize F1, minimize latency
    load_if_exists=True,
    sampler=NSGAIISampler(
        population_size=50,  # Number of parallel solutions evolved
        crossover_prob=0.9,  # Probability of crossover between solutions
        mutation_prob=0.1,   # Probability of mutating a solution
        seed=42
    ),
    pruner=optuna.pruners.PatientPruner(
        optuna.pruners.MedianPruner(
            n_startup_trials=20,
            n_warmup_steps=50,
            interval_steps=10,
        ),
    patience=3,
    min_delta=0.0
    )
)

    print("Resilient async study server started.")
    print("Study name:", study.study_name)
    print("Storage:", study._storage)

elif mode == 'tune_worker':
    tag = args.tag[0]
    param_dir = f'params_{tag}'

    import json
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Configure worker with auto-retry
    study = optuna.load_study(
        study_name=param_dir,
        storage=f"sqlite:///studies/{param_dir}/{param_dir}.db"
    )

    # study.optimize(
    #     objective,
    #     n_trials=1000,
    #     gc_after_trial=True,
    #     show_progress_bar=True,
    #     callbacks=[
    #         # Auto-save callback
    #         lambda study, trial: logger.info(f"Trial {trial.number} finished")
    #     ]
    # )
    # Optimize for 1000 trials
    study.optimize(
        objective,
        n_trials=1000,
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[lambda study, trial: logger.info(f"Trial {trial.number} finished")]
    )

elif mode == 'tune_viz':
    tag = args.tag[0]
    param_dir = f'params_{tag}'

    # Load study for visualization
    study = optuna.load_study(
        study_name=param_dir,
        storage=f"sqlite:///studies/{param_dir}/{param_dir}.db",
    )

    # Create visualization directory under the param_dir
    os.makedirs(f"studies/{param_dir}/visualizations", exist_ok=True)

    # Update paths to store under param_dir
    fig = plot_optimization_history(study)
    fig.write_html(f"studies/{param_dir}/visualizations/optimization_history.html")

    fig = plot_param_importances(study)
    fig.write_html(f"studies/{param_dir}/visualizations/param_importances.html")

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(f"studies/{param_dir}/visualizations/parallel_coordinate.html")

    # Save study statistics
    stats = {
        "number_of_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
        "completed_trials": len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
        "pruned_trials": len(study.get_trials(states=[optuna.trial.TrialState.PRUNED])),
        "failed_trials": len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))
    }

    with open(f"studies/{param_dir}/visualizations/study_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Visualization results saved to studies/{param_dir}/visualizations/")
    print("\nStudy Statistics:")
    print(json.dumps(stats, indent=2))

    # Top N trials analysis
    N = 30
    trials = study.trials
    sorted_trials = sorted(trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
    top_trials = sorted_trials[:N]

    import pandas as pd
    data = []
    for trial in top_trials:
        if trial.values:
            row = {
                'trial_number': trial.number,
                'score': trial.values[0],
                **trial.params
            }
            data.append(row)

    df = pd.DataFrame(data)

    # Save analysis files under param_dir
    df.to_csv(f"studies/{param_dir}/visualizations/top_{N}_trials.csv", index=False)

    html_content = """
    <html>
    <head>
        <title>Top {} Trial Parameters</title>
        <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid black; padding: 8px; text-align: left; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h2>Top {} Trials by Performance</h2>
        {}
    </body>
    </html>
    """.format(N, N, df.to_html())

    with open(f"studies/{param_dir}/visualizations/top_{N}_trials.html", "w") as f:
        f.write(html_content)

    import matplotlib.pyplot as plt
    numerical_params = df.select_dtypes(include=['float64', 'int64']).columns
    fig, axes = plt.subplots(len(numerical_params), 1, figsize=(10, 4*len(numerical_params)))
    for i, param in enumerate(numerical_params):
        if param != 'score' and param != 'trial_number':
            ax = axes[i] if len(numerical_params) > 1 else axes
            df.plot(kind='scatter', x=param, y='score', ax=ax)
            ax.set_title(f'Score vs {param}')

    plt.tight_layout()
    plt.savefig(f"studies/{param_dir}/visualizations/top_{N}_parameter_distributions.png")

    print(f"\nTop {N} trials analysis saved to studies/{param_dir}/visualizations/")
    print("Check the following files:")
    print(f"- top_{N}_trials.csv")
    print(f"- top_{N}_trials.html")
    print(f"- top_{N}_parameter_distributions.png")

elif mode == 'tune_viz_multi':
    import pandas as pd
    from optuna.visualization import plot_pareto_front
    from optuna.visualization import plot_optimization_history
    tag = args.tag[0]
    param_dir = f'params_{tag}'

    # Load study for visualization
    study = optuna.load_study(
        study_name=param_dir,
        storage=f"sqlite:///studies/{param_dir}/{param_dir}.db",
    )

    # Create visualization directory under the param_dir
    os.makedirs(f"studies/{param_dir}/visualizations", exist_ok=True)

    # Plot optimization history for F1 Score
    fig = plot_optimization_history(study, target=lambda t: t.values[0], target_name="F1 Score")
    fig.write_html(f"studies/{param_dir}/visualizations/optimization_history_f1.html")

    # Plot optimization history for Latency
    fig = plot_optimization_history(study, target=lambda t: t.values[1], target_name="Latency")
    fig.write_html(f"studies/{param_dir}/visualizations/optimization_history_latency.html")

    # Plot Pareto Front
    fig = plot_pareto_front(study, target_names=["F1 Score", "Latency"])
    fig.write_html(f"studies/{param_dir}/visualizations/pareto_front.html")

    # Plot parameter importance for F1 Score
    fig = plot_param_importances(study, target=lambda t: t.values[0], target_name="F1 Score")
    fig.write_html(f"studies/{param_dir}/visualizations/param_importances_f1.html")

    # Plot parameter importance for Latency
    fig = plot_param_importances(study, target=lambda t: t.values[1], target_name="Latency")
    fig.write_html(f"studies/{param_dir}/visualizations/param_importances_latency.html")

    # Save study statistics
    stats = {
        "number_of_trials": len(study.trials),
        "completed_trials": len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
        "pruned_trials": len(study.get_trials(states=[optuna.trial.TrialState.PRUNED])),
        "failed_trials": len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))
    }

    with open(f"studies/{param_dir}/visualizations/study_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Visualization results saved to studies/{param_dir}/visualizations/")
    print("\nStudy Statistics:")
    print(json.dumps(stats, indent=2))

    # Top N trials analysis for F1 Score
    N = 30
    trials = study.trials
    sorted_trials_f1 = sorted(trials, key=lambda t: t.values[0] if t.values else float('-inf'), reverse=True)[:N]
    sorted_trials_latency = sorted(trials, key=lambda t: t.values[1] if t.values else float('inf'))[:N]

    data_f1 = [
        {"trial_number": trial.number, "f1_score": trial.values[0], "latency": trial.values[1], **trial.params}
        for trial in study.trials
        if trial.values is not None  # Filter out trials with None values
    ]
    data_latency = [
    {
        "trial_number": trial.number,
        "f1_score": trial.values[0],
        "latency": trial.values[1],
        **trial.params
    }
    for trial in sorted_trials_latency
    if trial.values is not None and len(trial.values) == 2  # Ensure both f1_score and latency are available
]
    df_f1 = pd.DataFrame(data_f1)
    df_latency = pd.DataFrame(data_latency)

    # Save CSVs for top trials
    df_f1.to_csv(f"studies/{param_dir}/visualizations/top_{N}_trials_f1.csv", index=False)
    df_latency.to_csv(f"studies/{param_dir}/visualizations/top_{N}_trials_latency.csv", index=False)

    # Generate HTML reports for top trials
    def generate_html_report(df, metric_name):
        html_content = f"""
        <html>
        <head>
            <title>Top {N} Trial Parameters by {metric_name}</title>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid black; padding: 8px; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h2>Top {N} Trials by {metric_name}</h2>
            {df.to_html(index=False)}
        </body>
        </html>
        """
        return html_content

    with open(f"studies/{param_dir}/visualizations/top_{N}_trials_f1.html", "w") as f:
        f.write(generate_html_report(df_f1, "F1 Score"))

    with open(f"studies/{param_dir}/visualizations/top_{N}_trials_latency.html", "w") as f:
        f.write(generate_html_report(df_latency, "Latency"))

    print(f"\nTop {N} trials analysis saved to studies/{param_dir}/visualizations/")
    print("Check the following files:")
    print(f"- top_{N}_trials_f1.csv")
    print(f"- top_{N}_trials_latency.csv")
    print(f"- top_{N}_trials_f1.html")
    print(f"- top_{N}_trials_latency.html")

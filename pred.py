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

def objective_only(trial):
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
    params['NO_EPOCHS'] = 400
    params['TYPE_MODEL'] = 'Base'

    arch_lib = ['MixerOnly', 'MixerHori', 
                'MixerDori', 'DualMixerDori', 'MixerCori', 
                'SingleCh', 'TripletOnly']    
    # Model architecture parameters - Fix the categorical suggestion
    # arch_ind = trial.suggest_int('IND_ARCH', 0, len(arch_lib)-1)
    # pdb.set_trace()
    tag = args.tag[0]
    arch_ind = np.where([(arch.lower() in tag.lower()) for arch in arch_lib])[0][0]
    # print(arch_ind)
    # pdb.set_trace()
    params['TYPE_ARCH'] = arch_lib[arch_ind]
    print(params['TYPE_ARCH'])
    # pdb.set_trace()
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
    elif 'TripletOnly' in params['TYPE_ARCH']:
        from model.input_proto import rippleAI_load_dataset
        from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
        
        # Set a reasonable steps_per_epoch value - much smaller than 1500 for initial testing
        params['steps_per_epoch'] = 250  # Increase gradually if training works
        
        # Add debug lines
        print("Setting up triplet dataset with steps_per_epoch:", params['steps_per_epoch'])
        print("Batch size:", params['BATCH_SIZE'])

    # pdb.set_trace()
    # Timing parameters remain the same
    # params['NO_TIMEPOINTS'] = trial.suggest_categorical('NO_TIMEPOINTS', [128, 196, 384])
    params['NO_TIMEPOINTS'] = 128
    params['NO_STRIDES'] = int(params['NO_TIMEPOINTS'] // 2)

    # Timing parameters remain the same
    params['HORIZON_MS'] = trial.suggest_int('HORIZON_MS', 1, 5)
    params['SHIFT_MS'] = 0

    params['LOSS_WEIGHT'] = trial.suggest_float('LOSS_WEIGHT', 0.000001, 10.0, log=True)
    # params['LOSS_WEIGHT'] = 7.5e-4

    entropyLib = [0, 0.5, 1, 3]
    entropy_ind = trial.suggest_categorical('HYPER_ENTROPY', [0,1,2,3])
    params['HYPER_ENTROPY'] = entropyLib[entropy_ind]
    
    # params['HYPER_TMSE'] = trial.suggest_float('HYPER_TMSE', 0.000001, 10.0, log=True)
    # params['HYPER_BARLOW'] = 2e-5
    # params['HYPER_BARLOW'] = trial.suggest_float('HYPER_BARLOW', 0.000001, 10.0, log=True)
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
    params['NO_FILTERS'] = trial.suggest_categorical('NO_FILTERS', [32, 64, 128])
    # params['NO_FILTERS'] = 64
    ax = 65#trial.suggest_int('AX', 1, 99)
    gx = 100#trial.suggest_int('GX', 50, 999)

    # Removed duplicate TYPE_ARCH suggestion that was causing the error
    params['TYPE_LOSS'] = 'FocalGapAx{:03d}Gx{:03d}Entropy'.format(ax, gx)

    # Remove the hardcoded use_freq and derive it from tag instead
    
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
    
    # opt_lib = ['Adam', 'AdamW', 'SGD']
    par_opt = 'Adam'
    # par_opt = opt_lib[trial.suggest_int('IND_OPT', 0, len(opt_lib)-1)]
    
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
    params['USE_L2N'] = trial.suggest_categorical('USE_L2N', [True, False])
    if params['USE_L2N']:
        params['TYPE_ARCH'] += 'L2N'


    params['USE_Aug'] = trial.suggest_categorical('USE_Aug', [True, False])
    if params['USE_Aug']:
        params['TYPE_ARCH'] += 'Aug'
        
    # if (not 'Cori' in params['TYPE_ARCH']) and  (not 'SingleCh' in params['TYPE_MODEL']):
    #     params['TYPE_LOSS'] += 'BarAug'
    # params['USE_L2Reg'] = trial.suggest_categorical('USE_L2Reg', [True])#, False
    # params['USE_CSD'] = trial.suggest_categorical('USE_CSD', [True, False])
    # if params['USE_CSD']:
    #     params['TYPE_ARCH'] += 'CSD'
    # params['Dropout'] = trial.suggest_int('Dropout', 0, 10)
    # params['USE_L2Reg'] = trial.suggest_categorical('USE_L2Reg', [True])#, False
    # if params['USE_L2Reg']:
    #     params['TYPE_LOSS'] += 'L2Reg'

    drop_lib = [0, 0.05, 0.1, 0.2, 0.5]
    drop_ind = trial.suggest_categorical('Dropout', [0,1,2,3,4])
    if drop_ind > 0:
        params['TYPE_ARCH'] += f"Drop{drop_lib[drop_ind]:02d}"
    
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

    # if 'TripletOnly' in params['TYPE_ARCH']:
    #         train_dataset = transform_dataset_for_training(train_dataset)
    #         val_dataset = transform_dataset_for_training(val_dataset)
    # if params['TYPE_MODEL'] == 'SingleCh':
    #     model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params, input_chans=1)
    # else:
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    # Early stopping with tunable patience
    callbacks = []

    # Early stopping with tunable patience
    callbacks.append(cb.EarlyStopping(
        monitor='val_max_f1_metric_horizon',  # Change monitor
        patience=30,
        mode='max',
        verbose=1,
        restore_best_weights=True
    ))
    
    # Import the new multi-objective callback
    from model.training import WeightDecayCallback, lr_scheduler
    
    # use_LR = trial.suggest_categorical('USE_LR', [True, False])
    # if use_LR:
    #     params['TYPE_REG'] += 'LR'
    #     callbacks.append(cb.LearningRateScheduler(lr_scheduler, verbose=1))

    # if par_opt == 'AdamW':
    #     params['TYPE_REG'] += 'WD'
    #     callbacks.append(WeightDecayCallback())
    
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


def objective_triplet(trial):
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
    params['NO_EPOCHS'] = 800
    params['TYPE_MODEL'] = 'Base'

    arch_lib = ['MixerOnly', 'MixerHori', 
                'MixerDori', 'DualMixerDori', 'MixerCori', 
                'SingleCh', 'TripletOnly']    
    # Model architecture parameters - Fix the categorical suggestion
    # arch_ind = trial.suggest_int('IND_ARCH', 0, len(arch_lib)-1)
    # pdb.set_trace()
    tag = args.tag[0]
    arch_ind = np.where([(arch.lower() in tag.lower()) for arch in arch_lib])[0][0]
    # print(arch_ind)
    # pdb.set_trace()
    params['TYPE_ARCH'] = arch_lib[arch_ind]
    print(params['TYPE_ARCH'])
    # pdb.set_trace()
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
    elif 'TripletOnly' in params['TYPE_ARCH']:
        from model.input_proto import rippleAI_load_dataset
        from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
        
        # Set a reasonable steps_per_epoch value - much smaller than 1500 for initial testing
        params['steps_per_epoch'] = 1200  # Increase gradually if training works
        
        # Add debug lines
        print("Setting up triplet dataset with steps_per_epoch:", params['steps_per_epoch'])
        print("Batch size:", params['BATCH_SIZE'])

    # pdb.set_trace()
    # Timing parameters remain the same
    # params['NO_TIMEPOINTS'] = trial.suggest_categorical('NO_TIMEPOINTS', [128, 196, 384])
    params['NO_TIMEPOINTS'] = 128
    params['NO_STRIDES'] = int(params['NO_TIMEPOINTS'] // 2)

    # Timing parameters remain the same
    params['HORIZON_MS'] = trial.suggest_int('HORIZON_MS', 1, 5)
    params['SHIFT_MS'] = 0

    params['LOSS_WEIGHT'] = trial.suggest_float('LOSS_WEIGHT', 0.000001, 10.0, log=True)
    # params['LOSS_WEIGHT'] = 7.5e-4


    ax = 65#trial.suggest_int('AX', 1, 99)
    gx = 100#trial.suggest_int('GX', 50, 999)

    # Removed duplicate TYPE_ARCH suggestion that was causing the error
    params['TYPE_LOSS'] = 'FocalGapAx{:03d}Gx{:03d}'.format(ax, gx)

    entropyLib = [0, 0.5, 1, 3]
    entropy_ind = trial.suggest_categorical('HYPER_ENTROPY', [0,1,2,3])
    if entropy_ind > 0:
        params['HYPER_ENTROPY'] = entropyLib[entropy_ind]
        params['TYPE_LOSS'] += 'Entropy'
    
    # params['HYPER_TMSE'] = trial.suggest_float('HYPER_TMSE', 0.000001, 10.0, log=True)
    # params['HYPER_BARLOW'] = 2e-5
    # params['HYPER_BARLOW'] = trial.suggest_float('HYPER_BARLOW', 0.000001, 10.0, log=True)
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
    params['NO_FILTERS'] = trial.suggest_categorical('NO_FILTERS', [32, 64, 128])
    # params['NO_FILTERS'] = 64

    # Remove the hardcoded use_freq and derive it from tag instead
    
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
    
    # opt_lib = ['Adam', 'AdamW', 'SGD']
    par_opt = 'Adam'
    # par_opt = opt_lib[trial.suggest_int('IND_OPT', 0, len(opt_lib)-1)]
    
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
    params['USE_L2N'] = trial.suggest_categorical('USE_L2N', [True, False])
    if params['USE_L2N']:
        params['TYPE_ARCH'] += 'L2N'


    params['USE_Aug'] = trial.suggest_categorical('USE_Aug', [True, False])
    if params['USE_Aug']:
        params['TYPE_ARCH'] += 'Aug'
        
    # if (not 'Cori' in params['TYPE_ARCH']) and  (not 'SingleCh' in params['TYPE_MODEL']):
    #     params['TYPE_LOSS'] += 'BarAug'
    # params['USE_L2Reg'] = trial.suggest_categorical('USE_L2Reg', [True])#, False
    # params['USE_CSD'] = trial.suggest_categorical('USE_CSD', [True, False])
    # if params['USE_CSD']:
    #     params['TYPE_ARCH'] += 'CSD'
    # params['Dropout'] = trial.suggest_int('Dropout', 0, 10)
    # params['USE_L2Reg'] = trial.suggest_categorical('USE_L2Reg', [True])#, False
    # if params['USE_L2Reg']:
    #     params['TYPE_LOSS'] += 'L2Reg'

    drop_lib = [0, 5, 10, 20, 50]
    drop_ind = trial.suggest_categorical('Dropout', [0,1,2,3,4])
    print('Dropout rate:', drop_lib[drop_ind])
    if drop_ind > 0:
        params['TYPE_ARCH'] += f"Drop{drop_lib[drop_ind]:02d}"
    
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
        if 'TripletOnly' in params['TYPE_ARCH']:
            params['steps_per_epoch'] = 1200
            train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(params, mode='train', preprocess=True)
        else:
            train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)        

    # if 'TripletOnly' in params['TYPE_ARCH']:
    #         train_dataset = transform_dataset_for_training(train_dataset)
    #         val_dataset = transform_dataset_for_training(val_dataset)
    # if params['TYPE_MODEL'] == 'SingleCh':
    #     model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params, input_chans=1)
    # else:
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    # Early stopping with tunable patience
   
    # Early stopping parameters
    best_metric = float('-inf')
    best_metric2 = float('-inf')
    patience = 50
    min_delta = 0.0001
    patience_counter = 0
    
    # Create a list to collect history from each epoch
    history_list = []
    
    # Setup callbacks including the verifier
    callbacks = [cb.TensorBoard(log_dir=f"{study_dir}/",
                                      write_graph=True,
                                      write_images=True,
                                      update_freq='epoch'),
        cb.ModelCheckpoint(f"{study_dir}/max.weights.h5",
                            monitor='val_max_f1_metric_horizon', 
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='max'),
        cb.ModelCheckpoint(
        f"{study_dir}/robust.weights.h5",
        monitor='val_robust_f1',  # Change monitor
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='max')
    ]
    
    # Loop through epochs manually
    n_epoch = params['NO_EPOCHS']
    for epoch in range(n_epoch):
        print(f"\nEpoch {epoch+1}/{n_epoch}")
        if dataset_params is not None and 'triplet_regenerator' in dataset_params:
            regenerating_dataset = dataset_params['triplet_regenerator']
            print(f"Regenerating triplet samples for epoch {epoch+1}")
            
            if epoch > 0:
                regenerating_dataset.reinitialize()
            train_data = regenerating_dataset.dataset if hasattr(regenerating_dataset, 'dataset') else regenerating_dataset

            steps = dataset_params.get('steps_per_epoch', 500)
            # pdb.set_trace()
            epoch_history = model.fit(train_data,
                steps_per_epoch=steps,
                initial_epoch=epoch,
                epochs=epoch+1,
                validation_data=test_dataset,
                callbacks=callbacks,
                verbose=1
            )

        # Collect history
        history_list.append(epoch_history.history)
                    
        # Early stopping check after each epoch
        current_metric = epoch_history.history.get('val_max_f1_metric_horizon', [float('-inf')])[0]
        current_metric2 = epoch_history.history.get('val_robust_f1', [float('-inf')])[0]
        if (current_metric > (best_metric + min_delta)) or (current_metric2 > (best_metric2 + min_delta)):
            if current_metric > best_metric:
                print(f"New best metric: {current_metric}")
                best_metric = current_metric
            else:
                best_metric2 = current_metric2
                print(f"New best metric2: {current_metric2}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            break
        
    # Combine histories from all epochs
    combined_history = {}
    for key in history_list[0].keys():
        combined_history[key] = []
        for h in history_list:
            combined_history[key].extend(h[key])

    # If the trial completes, compute the final metrics.
    # pdb.set_trace()
    final_f1 = (np.mean(combined_history['val_robust_f1']) + 
                max(combined_history['val_max_f1_metric_horizon'])) / 2
    final_latency = np.mean(combined_history['val_latency_metric'])
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
    elif 'TripletOnly' in params['TYPE_ARCH']:
        print('Using TripletOnly')
        from model.input_proto import rippleAI_load_dataset
        from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
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
    elif model_name.find('PatchAD') != -1:
        from model.model_fn import build_model_PatchAD as build_DBI_TCN
        from model.input_augment_weighted import rippleAI_load_dataset
    elif model_name.find('PatchTCN') != -1:
        from model.model_fn import build_model_TCN_PatchAD as build_DBI_TCN
        from model.input_augment_weighted import rippleAI_load_dataset
    else:
        from model.model_fn import build_DBI_TCN
        from model.input_aug import rippleAI_load_dataset
    # input

    if 'PatchAD' in model_name:
        # tf.config.run_functions_eagerly(True)
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], input_chans=8, patch_sizes=[2,4,8,16], d_model=256, num_layers=3)
    elif ('Proto' not in model_name) and ('Barlow' not in model_name):
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
        if 'TripletOnly' in params['TYPE_ARCH']:
            params['steps_per_epoch'] = 1000
            train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(params, mode='train', preprocess=True)
        else:
            train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)
    train_size = len(list(train_dataset))
    params['RIPPLE_RATIO'] = label_ratio

    # Calculate model FLOPs using TensorFlow Profiler
    @tf.function
    def get_flops(model, batch_size=1, params=params):
        concrete_func = tf.function(lambda x: model(x))
        if 'TripletOnly' in params['TYPE_ARCH']:
            tt = tf.TensorSpec([batch_size*3, params['NO_TIMEPOINTS'], params['NO_CHANNELS']], tf.float32)
            frozen_func = concrete_func.get_concrete_function(tt)
        else:
            frozen_func = concrete_func.get_concrete_function(
                tf.TensorSpec([batch_size, params['NO_TIMEPOINTS'], params['NO_CHANNELS']], tf.float32))
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='scope', options=opts)
        return flops.total_float_ops

    try:
        flops = get_flops(model, batch_size=1, params=params)
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
        elif 'TripletOnly' in params['TYPE_ARCH']:            
            hist = train_pred(model, train_dataset, test_dataset, params['NO_EPOCHS'], params['EXP_DIR'], dataset_params=dataset_params)
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
        if 'MixerOnly' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_MixerOnly as build_DBI_TCN
        elif 'MixerHori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_HorizonMixer as build_DBI_TCN
        elif 'MixerDori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_DorizonMixer as build_DBI_TCN
        elif 'MixerCori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_CorizonMixer as build_DBI_TCN
        elif 'TripletOnly' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
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
    np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/preds_val{0}_{1}_sf{2}.npy'.format(val_id, model_name, params['SRATE']), probs)
    # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/preds_val{0}_{1}_{3}_sf{2}.npy'.format(val_id, model_name, params['SRATE'], tag), probs)
    if model_name.find('Hori') != -1 or model_name.find('Dori') != -1 or model_name.find('Cori') != -1 or model_name.startswith('Tune') != -1:
        if not ('Only' in params['TYPE_ARCH']):
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

elif mode == 'embedding':

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
        # Import required modules
        if 'MixerOnly' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_MixerOnly as build_DBI_TCN
        elif 'MixerHori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_HorizonMixer as build_DBI_TCN
        elif 'MixerDori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_DorizonMixer as build_DBI_TCN
        elif 'MixerCori' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_CorizonMixer as build_DBI_TCN
        from model.model_fn import CSDLayer
        from tcn import TCN
        from keras.models import load_model
        # Load weights
        params['mode'] = mode
        weight_file = f"{study_dir}/max.weights.h5"
        # weight_file = f"{study_dir}/robust.weights.h5"
        print(f"Loading weights from: {weight_file}")
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
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
    pro_data = np.load('/cs/projects/OWVinckSWR/Dataset/TopologicalData/ripple_data.pkl', allow_pickle=True)['ripples']

    if model_name.startswith('Tune'):
        if 'Samp1250' in params['NAME']:
            pro_data = np.array([decimate(pro_data[:,k], 2) for k in range(pro_data.shape[1])])
        # elif 'Samp2500' in params['NAME']:
        #     pro_data = np.transpose(pro_data, [1,0])
        # else:
        #     pro_data = np.transpose(pro_data, [1,0])
    else:
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

    tmp_act = model.predict(test_ripples)

    # pdb.set_trace()
    np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/activations/{0}_act{1}.npy'.format(model_name, 'TCN'), tmp_act)
    # save activations
    # for il in range(len(tmp_act)):
    #     np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/activations/{0}_act{1}.npy'.format(model_name, il), tmp_act[il])

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
        population_size=30,  # Number of parallel solutions evolved
        crossover_prob=0.9,  # Probability of crossover between solutions
        mutation_prob=0.2,   # Probability of mutating a solution
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
    if 'TripletOnly'.lower() in tag.lower():
        objective = objective_triplet
    elif 'MixerOnly'.lower() in tag.lower():
        objective = objective_only
    else:
        objective = objective_only
    
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
    
    # Create a subdirectory for hyperparameter impact plots
    os.makedirs(f"studies/{param_dir}/visualizations/param_impact", exist_ok=True)

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

    # Collect all trial data with valid values for both metrics
    all_trial_data = [
        {"trial_number": trial.number, "f1_score": trial.values[0], "latency": trial.values[1], **trial.params}
        for trial in study.trials
        if trial.values is not None and len(trial.values) == 2
    ]
    all_df = pd.DataFrame(all_trial_data)
    
    # Save the complete dataset for further analysis
    all_df.to_csv(f"studies/{param_dir}/visualizations/all_trials_data.csv", index=False)
    
    # Generate hyperparameter impact plots
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get all hyperparameter names
    hyperparams = [p for p in all_df.columns if p not in ['trial_number', 'f1_score', 'latency']]
    
    # Identify top models for highlighting in plots
    N_TOP_HIGHLIGHT = 15  # Number of top models to highlight
    top_f1_trials = all_df.sort_values('f1_score', ascending=False).head(N_TOP_HIGHLIGHT)['trial_number'].tolist()
    
    # Function to check if a parameter should use log scale (if it spans multiple orders of magnitude)
    def should_use_log_scale(series):
        if not pd.api.types.is_numeric_dtype(series):
            return False
        if series.min() <= 0:  # Can't use log scale with zero or negative values
            return False
        return series.max() / max(series.min(), 1e-10) > 10  # Use log scale if range spans more than 1 order of magnitude
    
    # Create plots showing how each hyperparameter affects both metrics
    for param in hyperparams:
        plt.figure(figsize=(15, 10))
        use_log_scale = should_use_log_scale(all_df[param])
        
        # Create a subplot for F1 Score
        plt.subplot(2, 1, 1)
        if all_df[param].dtype == bool or all_df[param].nunique() < 10:  # Categorical parameter
            sns.boxplot(x=param, y='f1_score', data=all_df)
            # Add individual data points for better visibility
            sns.stripplot(x=param, y='f1_score', data=all_df, color='black', alpha=0.5, size=3)
            
            # Highlight top models with different color
            top_data = all_df[all_df['trial_number'].isin(top_f1_trials)]
            if not top_data.empty:
                sns.stripplot(x=param, y='f1_score', data=top_data, color='red', 
                             size=8, marker='D', label=f'Top {N_TOP_HIGHLIGHT} models')
                plt.legend()
        else:  # Numerical parameter
            # Regular models in blue
            sns.scatterplot(x=param, y='f1_score', data=all_df, alpha=0.6, label='All models')
            
            # Top models in red with larger markers
            sns.scatterplot(x=param, y='f1_score', data=all_df[all_df['trial_number'].isin(top_f1_trials)], 
                           color='red', s=100, label=f'Top {N_TOP_HIGHLIGHT} models')
            
            # Add a trend line
            sns.regplot(x=param, y='f1_score', data=all_df, scatter=False, color='darkblue')
            
            if use_log_scale:
                plt.xscale('log')
                plt.xlabel(f"{param} (log scale)")
            
        plt.title(f'Impact of {param} on F1 Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Create a subplot for Latency
        plt.subplot(2, 1, 2)
        if all_df[param].dtype == bool or all_df[param].nunique() < 10:  # Categorical parameter
            sns.boxplot(x=param, y='latency', data=all_df)
            # Add individual data points for better visibility
            sns.stripplot(x=param, y='latency', data=all_df, color='black', alpha=0.5, size=3)
            
            # Highlight top models with different color
            top_data = all_df[all_df['trial_number'].isin(top_f1_trials)]
            if not top_data.empty:
                sns.stripplot(x=param, y='latency', data=top_data, color='red', 
                             size=8, marker='D', label=f'Top {N_TOP_HIGHLIGHT} models by F1')
                plt.legend()
        else:  # Numerical parameter
            # Regular models in blue
            sns.scatterplot(x=param, y='latency', data=all_df, alpha=0.6, label='All models')
            
            # Top models in red with larger markers
            sns.scatterplot(x=param, y='latency', data=all_df[all_df['trial_number'].isin(top_f1_trials)], 
                           color='red', s=100, label=f'Top {N_TOP_HIGHLIGHT} models by F1')
            
            # Add a trend line
            sns.regplot(x=param, y='latency', data=all_df, scatter=False, color='darkblue')
            
            if use_log_scale:
                plt.xscale('log')
                plt.xlabel(f"{param} (log scale)")
            
        plt.title(f'Impact of {param} on Latency')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"studies/{param_dir}/visualizations/param_impact/{param}_impact.png")
        plt.close()
    
    # Also generate a correlation plot between hyperparameters and metrics
    plt.figure(figsize=(12, 10))
    corr_columns = [c for c in all_df.columns if c not in ['trial_number'] and pd.api.types.is_numeric_dtype(all_df[c])]
    corr_df = all_df[corr_columns].corr()
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Between Parameters and Metrics')
    plt.tight_layout()
    plt.savefig(f"studies/{param_dir}/visualizations/param_impact/correlation_heatmap.png")
    plt.close()
    
    # Create a summary HTML file that links to all hyperparameter impact plots
    impact_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hyperparameter Impact Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .gallery { display: flex; flex-wrap: wrap; }
            .param-card { margin: 10px; padding: 10px; border: 1px solid #ccc; width: 300px; }
            img { width: 100%; }
            .correlation { margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Hyperparameter Impact on F1 Score and Latency</h1>
        <p>Click on images to view full size. Red diamonds highlight the top 15 models by F1 score.</p>
        
        <div class="correlation">
            <h2>Parameter Correlation Analysis</h2>
            <a href="param_impact/correlation_heatmap.png" target="_blank">
                <img src="param_impact/correlation_heatmap.png" alt="Parameter Correlation Heatmap">
            </a>
        </div>
        
        <h2>Individual Parameter Analysis</h2>
        <div class="gallery">
    """
    
    for param in hyperparams:
        impact_html += f"""
        <div class="param-card">
            <h3>{param}</h3>
            <a href="param_impact/{param}_impact.png" target="_blank">
                <img src="param_impact/{param}_impact.png" alt="Impact of {param}">
            </a>
        </div>
        """
    
    impact_html += """
        </div>
    </body>
    </html>
    """
    
    with open(f"studies/{param_dir}/visualizations/hyperparameter_impact_analysis.html", "w") as f:
        f.write(impact_html)
    
    # Top N trials analysis for F1 Score
    N = 30
    trials = study.trials
    sorted_trials_f1 = sorted(trials, key=lambda t: t.values[0] if t.values else float('-inf'), reverse=True)[:N]
    sorted_trials_latency = sorted(trials, key=lambda t: t.values[1] if t.values else float('inf'))[:N]

    data_f1 = [
        {"trial_number": trial.number, "f1_score": trial.values[0], "latency": trial.values[1], **trial.params}
        for trial in sorted_trials_f1
        if trial.values is not None and len(trial.values) == 2  # Ensure both f1_score and latency are available
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
            <p><a href="hyperparameter_impact_analysis.html">View Hyperparameter Impact Analysis</a></p>
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
    print("- hyperparameter_impact_analysis.html (new detailed analysis of each parameter's impact)")

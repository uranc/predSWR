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

# Define objective function before mode selection
def create_study_name(trial):
    """Create unique study name based on trial parameters"""
    return f"study_{trial.number}_{trial.datetime_start.strftime('%Y%m%d_%H%M%S')}"


def objective_patch(trial):
    """Objective function for Optuna optimization"""
    tf.compat.v1.reset_default_graph()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()

    # Start with base parameters
    params = {'BATCH_SIZE': 32, 'SHUFFLE_BUFFER_SIZE': 4096*8,
            'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-3, 'NO_EPOCHS': 300,
            'NO_TIMEPOINTS': 50, 'NO_CHANNELS': 8, 'SRATE': 2500,
            'EXP_DIR': '/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/experiments/' + 'test',
            'mode': 'train'
            }

    # Dynamic learning rate range
    batch_size = 64
    params['LEARNING_RATE'] = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)# 1e-3

    params['BATCH_SIZE'] = batch_size

    # Base parameters
    params['SRATE'] = 2500
    params['NO_EPOCHS'] = 400
    params['TYPE_MODEL'] = 'Base'

    arch_lib = ['MixerOnly', 'MixerHori',
                'MixerDori', 'DualMixerDori', 'MixerCori',
                'SingleCh', 'TripletOnly', 'Patch']
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
    elif 'Patch' in params['TYPE_ARCH']:
        from model.input_augment_weighted import rippleAI_load_dataset
        from model.model_fn import build_DBI_TCN_PatchOnly as build_DBI_TCN

    # pdb.set_trace()
    # Timing parameters remain the same
    # params['NO_TIMEPOINTS'] = trial.suggest_categorical('NO_TIMEPOINTS', [128, 196, 384])
    params['NO_TIMEPOINTS'] = 64#*3
    # params['NO_STRIDES'] = int(params['NO_TIMEPOINTS'] // 2)
    params['NO_STRIDES'] = trial.suggest_int('NO_STRIDES', 32, 160, step=32)

    # Timing parameters remain the same
    params['HORIZON_MS'] = trial.suggest_int('HORIZON_MS', 0, 10, step=2)
    params['SHIFT_MS'] = 0

    params['LOSS_WEIGHT'] = 1#trial.suggest_float('LOSS_WEIGHT', 0.000001, 10.0, log=True)
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
    elif params['NO_TIMEPOINTS'] == 128*12:
        dil_lib = [12,12,8,12,12,12]          # for kernels 2,3,4,5,6]
    elif params['NO_TIMEPOINTS'] == 128*3:
        dil_lib = [12,12,6,12,12,12]          # for kernels 2,3,4,5,6]

    params['NO_DILATIONS'] = dil_lib[params['NO_KERNELS']-2]
    # params['NO_DILATIONS'] = trial.suggest_int('NO_DILATIONS', 2, 6)
    params['NO_FILTERS'] = trial.suggest_categorical('NO_FILTERS', [32, 64, 128, 256])
    # params['NO_FILTERS'] = 64
    ax = trial.suggest_int('AX', 25, 75, step=25)
    gx = trial.suggest_int('GX', 50, 200, step=50)

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
    params['USE_ZNorm'] = False#trial.suggest_categorical('USE_ZNorm', [True, False])
    if params['USE_ZNorm']:
        params['TYPE_ARCH'] += 'ZNorm'
    params['USE_L2N'] = True#trial.suggest_categorical('USE_L2N', [True, False])
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
    drop_ind = 0#trial.suggest_categorical('Dropout', [0,1,2,3,4])
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
    print(params)
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    # Setup callbacks including the verifier
    callbacks = [cb.TensorBoard(log_dir=f"{study_dir}/",
                                      write_graph=True,
                                      write_images=True,
                                      update_freq='epoch'),
                cb.EarlyStopping(monitor='val_event_f1_metric',  # Change monitor
                                patience=50,
                                mode='max',
                                verbose=1,
                                restore_best_weights=True),
                cb.ModelCheckpoint(f"{study_dir}/max.weights.h5",
                                    monitor='val_max_f1_metric_horizon',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max'),
                                    # cb.ModelCheckpoint(
                                    # f"{study_dir}/robust.weights.h5",
                                    # monitor='val_robust_f1',  # Change monitor
                                    # verbose=1,
                                    # save_best_only=True,
                                    # save_weights_only=True,
                                    # mode='max'),
                cb.ModelCheckpoint(f"{study_dir}/event.weights.h5",
                                    monitor='val_event_f1_metric',  # Change monitor
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max')
    ]

    # Train and evaluate
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=params['NO_EPOCHS'],
        callbacks=callbacks,
        verbose=1
    )
    val_accuracy = (max(history.history['val_event_f1_metric'])+max(history.history['val_max_f1_metric_horizon']))/2
    val_accuracy_mean = (np.mean(history.history['val_event_f1_metric'])+np.mean(history.history['val_max_f1_metric_horizon']))/2
    val_accuracy = (val_accuracy + val_accuracy_mean)/2
    val_latency = np.mean(history.history['val_event_fp_rate'])
    # Log results
    logger.info(f"Trial {trial.number} finished with val_accuracy: {val_accuracy:.4f}, val_fprate: {val_latency:.4f}")

    # Save trial information
    trial_info = {
        'parameters': params,
        'metrics': {
        'val_accuracy': val_accuracy,
        'val_latency': val_latency
        }
    }
    with open(f"{study_dir}/trial_info.json", 'w') as f:
        json.dump(trial_info, f, indent=4)

    # Proper cleanup after training
    del model
    gc.collect()
    tf.keras.backend.clear_session()

    return val_accuracy, val_latency

def objective_only(trial):
    """Objective function for Optuna optimization"""
    tf.compat.v1.reset_default_graph()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()

    # Start with base parameters
    params = {'BATCH_SIZE': 32, 'SHUFFLE_BUFFER_SIZE': 4096*8,
            'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-3, 'NO_EPOCHS': 300,
            'NO_TIMEPOINTS': 50, 'NO_CHANNELS': 8, 'SRATE': 2500,
            'EXP_DIR': '/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
            'mode': 'train'
            }

    # Dynamic learning rate range
    batch_size = 64
    params['LEARNING_RATE'] = 1e-3
    params['BATCH_SIZE'] = batch_size

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

    # pdb.set_trace()
    # Timing parameters remain the same
    # params['NO_TIMEPOINTS'] = trial.suggest_categorical('NO_TIMEPOINTS', [128, 196, 384])
    params['NO_TIMEPOINTS'] = 128#*3
    # params['NO_STRIDES'] = int(params['NO_TIMEPOINTS'] // 2)
    params['NO_STRIDES'] = trial.suggest_int('NO_STRIDES', 32, 160, step=32)

    # Timing parameters remain the same
    params['HORIZON_MS'] = trial.suggest_int('HORIZON_MS', 1, 5, step=2)
    params['SHIFT_MS'] = 0

    params['LOSS_WEIGHT'] = 1#trial.suggest_float('LOSS_WEIGHT', 0.000001, 10.0, log=True)
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
    elif params['NO_TIMEPOINTS'] == 128*12:
        dil_lib = [12,12,8,12,12,12]          # for kernels 2,3,4,5,6]
    elif params['NO_TIMEPOINTS'] == 128*3:
        dil_lib = [12,12,6,12,12,12]          # for kernels 2,3,4,5,6]

    params['NO_DILATIONS'] = dil_lib[params['NO_KERNELS']-2]
    # params['NO_DILATIONS'] = trial.suggest_int('NO_DILATIONS', 2, 6)
    params['NO_FILTERS'] = 128#trial.suggest_categorical('NO_FILTERS', [32, 64, 128])
    # params['NO_FILTERS'] = 64
    ax = trial.suggest_int('AX', 25, 75, step=10)
    gx = trial.suggest_int('GX', 50, 999, step=150)

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
    params['USE_ZNorm'] = False#trial.suggest_categorical('USE_ZNorm', [True, False])
    if params['USE_ZNorm']:
        params['TYPE_ARCH'] += 'ZNorm'
    params['USE_L2N'] = True#trial.suggest_categorical('USE_L2N', [True, False])
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
    drop_ind = 0#trial.suggest_categorical('Dropout', [0,1,2,3,4])
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
    print(params)
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    # Setup callbacks including the verifier
    callbacks = [cb.TensorBoard(log_dir=f"{study_dir}/",
                                      write_graph=True,
                                      write_images=True,
                                      update_freq='epoch'),
                cb.EarlyStopping(monitor='val_event_f1_metric',  # Change monitor
                                patience=50,
                                mode='max',
                                verbose=1,
                                restore_best_weights=True),
                cb.ModelCheckpoint(f"{study_dir}/max.weights.h5",
                                    monitor='val_max_f1_metric_horizon',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max'),
                                    # cb.ModelCheckpoint(
                                    # f"{study_dir}/robust.weights.h5",
                                    # monitor='val_robust_f1',  # Change monitor
                                    # verbose=1,
                                    # save_best_only=True,
                                    # save_weights_only=True,
                                    # mode='max'),
                cb.ModelCheckpoint(f"{study_dir}/event.weights.h5",
                                    monitor='val_event_f1_metric',  # Change monitor
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max')
    ]

    # Train and evaluate
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=params['NO_EPOCHS'],
        callbacks=callbacks,
        verbose=1
    )
    val_accuracy = (max(history.history['val_event_f1_metric'])+max(history.history['val_max_f1_metric_horizon']))/2
    val_accuracy_mean = (np.mean(history.history['val_event_f1_metric'])+np.mean(history.history['val_max_f1_metric_horizon']))/2
    val_accuracy = (val_accuracy + val_accuracy_mean)/2
    val_latency = np.mean(history.history['val_event_fp_rate'])
    # Log results
    logger.info(f"Trial {trial.number} finished with val_accuracy: {val_accuracy:.4f}, val_fprate: {val_latency:.4f}")

    # Save trial information
    trial_info = {
        'parameters': params,
        'metrics': {
        'val_accuracy': val_accuracy,
        'val_latency': val_latency
        }
    }
    with open(f"{study_dir}/trial_info.json", 'w') as f:
        json.dump(trial_info, f, indent=4)

    # Proper cleanup after training
    del model
    gc.collect()
    tf.keras.backend.clear_session()

    return val_accuracy, val_latency

def objective_only_30k(trial):
    """Objective function for Optuna optimization"""
    tf.compat.v1.reset_default_graph()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()

    # Start with base parameters
    params = {'BATCH_SIZE': 32, 'SHUFFLE_BUFFER_SIZE': 4096*8,
            'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-3, 'NO_EPOCHS': 300,
            'NO_TIMEPOINTS': 128, 'NO_CHANNELS': 8, 'SRATE': 30000,
            'EXP_DIR': '/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/experiments/' + 'test',
            'mode': 'train'
            }

    # Dynamic learning rate range
    batch_size = 64
    params['LEARNING_RATE'] = 5e-3
    # params['LEARNING_RATE'] = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    params['BATCH_SIZE'] = batch_size

    # Base parameters
    params['SRATE'] = 30000
    params['NO_EPOCHS'] = 400
    params['TYPE_MODEL'] = 'Base'

    arch_lib = ['MixerOnly', 'MixerHori',
                'MixerDori', 'DualMixerDori', 'MixerCori',
                'SingleCh', 'TripletOnly', 'CADOnly']
    # Model architecture parameters - Fix the categorical suggestion
    # arch_ind = trial.suggest_int('IND_ARCH', 0, len(arch_lib)-1)
    # pdb.set_trace()
    tag = args.tag[0]
    arch_ind = np.where([(arch.lower() in tag.lower()) for arch in arch_lib])[0][0]
    # print(arch_ind)
    params['TYPE_ARCH'] = arch_lib[arch_ind]
    print(params['TYPE_ARCH'])
    # pdb.set_trace()
    # params['TYPE_ARCH'] = 'MixerHori'
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
    elif 'CADOnly' in params['TYPE_ARCH']:
            from model.input_augment_weighted import rippleAI_load_dataset
            from model.model_fn import build_DBI_TCN_CADMixerOnly as build_DBI_TCN        
            pretrain_tag = 'params_mixerOnlyEvents2500'
            pretrain_num = 1414#958# 1414
            study_dir = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{pretrain_tag}/study_{pretrain_num}_*')
            if not study_dir:
                raise ValueError(f"No study directory found for study number {pretrain_num}")
            study_dir = study_dir[0]  # Take the first matching directory

            pretrained_params = copy.deepcopy(params)
            # Load trial info to get parameters
            with open(f"{study_dir}/trial_info.json", 'r') as f:
                trial_info = json.load(f)
                pretrained_params.update(trial_info['parameters'])

            # Check which weight file is most recent
            event_weights = f"{study_dir}/event.weights.h5"
            max_weights = f"{study_dir}/max.weights.h5"

            if os.path.exists(event_weights) and os.path.exists(max_weights):
                # Both files exist, select the most recently modified one
                event_mtime = os.path.getmtime(event_weights)
                max_mtime = os.path.getmtime(max_weights)

                if event_mtime > max_mtime:
                    weight_file = event_weights
                    print(f"Using event.weights.h5 (more recent, modified at {time.ctime(event_mtime)})")
                else:
                    weight_file = max_weights
                    print(f"Using max.weights.h5 (more recent, modified at {time.ctime(max_mtime)})")
            elif os.path.exists(event_weights):
                weight_file = event_weights
                print("Using event.weights.h5 (max.weights.h5 not found)")
            elif os.path.exists(max_weights):
                weight_file = max_weights
                print("Using max.weights.h5 (event.weights.h5 not found)")
            else:
                raise ValueError(f"Neither event.weights.h5 nor max.weights.h5 found in {study_dir}")
            print(f"Loading weights from: {weight_file}")

            pretrained_params["WEIGHT_FILE"] = weight_file
            print('pretrained_params', pretrained_params)
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            build_DBI_TCN_Pretrained = model_module.build_DBI_TCN_MixerOnly

            pretrained_tcn = build_DBI_TCN_Pretrained(params["NO_TIMEPOINTS"], params=pretrained_params)
            pretrained_tcn.load_weights(weight_file)
            pretrained_tcn.trainable = False
            pretrained_tcn.compile(optimizer='adam', loss='mse')
            # model = build_DBI_TCN(pretrained_params["NO_TIMEPOINTS"], params=params, pretrained_tcn=pretrained_tcn)

    # pdb.set_trace()
    # Timing parameters remain the same
    # params['NO_TIMEPOINTS'] = trial.suggest_categorical('NO_TIMEPOINTS', [128, 196, 384])
    params['NO_TIMEPOINTS'] = 1104 #128
    # params['NO_STRIDES'] = int(params['NO_TIMEPOINTS'])
    params['NO_STRIDES'] = trial.suggest_int('NO_STRIDES', params['NO_TIMEPOINTS']/16, params['NO_TIMEPOINTS'], step=params['NO_TIMEPOINTS']/8)

    # params['NO_STRIDES'] = int(params['NO_TIMEPOINTS'] // 2)
    # params['NO_STRIDES'] = trial.suggest_int('NO_STRIDES', 128*3, params['NO_TIMEPOINTS']*4*3, step=128*3)

    # Timing parameters remain the same
    # params['HORIZON_MS'] = trial.suggest_int('HORIZON_MS', 1, 9, step=2)
    params['HORIZON_MS'] = 0
    params['SHIFT_MS'] = 0

    params['LOSS_WEIGHT'] = 1#trial.suggest_float('LOSS_WEIGHT', 0.000001, 10.0, log=True)
    # params['LOSS_WEIGHT'] = 7.5e-4

    entropyLib = [0, 0.5, 1, 3]
    entropy_ind = 2#trial.suggest_categorical('HYPER_ENTROPY', [0,1,2,3])
    # entropy_ind = trial.suggest_categorical('HYPER_ENTROPY', [0,1,2,3])
    params['HYPER_ENTROPY'] = entropyLib[entropy_ind]
    # params['HYPER_ENTROPY'] = trial.suggest_float('HYPER_ENTROPY', 0.1, 100.0, log=True)
    # params['HYPER_TMSE'] = trial.suggest_float('HYPER_TMSE', 0.000001, 10.0, log=True)
    # params['HYPER_BARLOW'] = 2e-5
    # params['HYPER_BARLOW'] = trial.suggest_float('HYPER_BARLOW', 0.000001, 10.0, log=True)
    # params['HYPER_MONO'] = trial.suggest_float('HYPER_MONO', 0.000001, 10.0, log=True)
    params['HYPER_MONO'] = 0 #trial.suggest_float('HYPER_MONO', 0.000001, 10.0, log=True)

    # Model parameters matching training format
    # params['NO_KERNELS'] = trial.suggest_int('NO_KERNELS', 2, 6) # for kernels 2,3,4,5,6
    # params['NO_KERNELS'] = 10
    ####### FIX
    ####### FIX
    params['NO_KERNELS'] = 4

    if params['NO_TIMEPOINTS'] == 32:
        dil_lib = [4,3,2,2,2]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 64:
        dil_lib = [5,4,3,3,3]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 128:
        dil_lib = [6,5,4,4,4]           # for kernels 2,3,4,5,6]
    elif params['NO_TIMEPOINTS'] == 196:
        dil_lib = [7,6,5,5,5]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 288:
        dil_lib = [8,7,6,6,6]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 384:
        dil_lib = [8,7,6,6,6]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 128*12:
        dil_lib = [6,5,4,4,4]           # for kernels 2,3,4,5,6]
    elif params['NO_TIMEPOINTS'] == 92*12:
        dil_lib = [6,5,4,4,4]           # for kernels 2,3,4,5,6]


    params['NO_DILATIONS'] = dil_lib[params['NO_KERNELS']-2]
    # # params['NO_DILATIONS'] = 4 ####### FIX
    ####### FIX
    ####### FIX
    # params['NO_DILATIONS'] = trial.suggest_int('NO_DILATIONS', 2, 6)
    params['NO_FILTERS'] = trial.suggest_categorical('NO_FILTERS', [32, 64, 128])
    # params['NO_FILTERS'] = 64
    # ax = trial.suggest_int('AX', 25, 75, step=25)
    ax = 25
    # ax = trial.suggest_int('AX', 25, 25, step=0)
    # gx = 100
    # gx = trial.suggest_int('GX', 50, 200, step=50)
    gx = 150

    # Removed duplicate TYPE_ARCH suggestion that was causing the error
    params['TYPE_LOSS'] = 'Samp{}FocalGapAx{:03d}Gx{:03d}Entropy'.format(params['SRATE'], ax, gx)

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
    params['USE_ZNorm'] = False#trial.suggest_categorical('USE_ZNorm', [True, False])
    if params['USE_ZNorm']:
        params['TYPE_ARCH'] += 'ZNorm'
    params['USE_L2N'] = True#trial.suggest_categorical('USE_L2N', [True, False])
    if params['USE_L2N']:
        params['TYPE_ARCH'] += 'L2N'
    params['USE_Aug'] = False #trial.suggest_categorical('USE_Aug', [True, False])
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
    drop_ind = 0#trial.suggest_categorical('Dropout', [0,1,2,3,4])
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
    # pdb.set_trace()
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
    print(params)
    if 'CADOnly' in params['TYPE_ARCH']:
        model = build_DBI_TCN(pretrained_params["NO_TIMEPOINTS"], params=params, pretrained_tcn=pretrained_tcn)
    else:
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    # Setup callbacks including the verifier
    callbacks = [
                cb.TensorBoard(log_dir=f"{study_dir}/",
                                            write_graph=True,
                                            write_images=True,
                                            update_freq='epoch'),
                cb.EarlyStopping(monitor='val_max_f1_metric_horizon',  # Change monitor
                                patience=50,
                                mode='max',
                                verbose=1,
                                restore_best_weights=True),
                cb.ModelCheckpoint(f"{study_dir}/max.weights.h5",
                                    monitor='val_max_f1_metric_horizon',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max'),
                                    # cb.ModelCheckpoint(
                                    # f"{study_dir}/robust.weights.h5",
                                    # monitor='val_robust_f1',  # Change monitor
                                    # verbose=1,
                                    # save_best_only=True,
                                    # save_weights_only=True,
                                    # mode='max'),
                cb.ModelCheckpoint(f"{study_dir}/event.weights.h5",
                                    monitor='val_event_f1_metric',  # Change monitor
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max')
    ]

    # Train and evaluate
    # pdb.set_trace()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=params['NO_EPOCHS'],
        callbacks=callbacks,
        verbose=1
    )
    val_accuracy = (max(history.history['val_event_f1_metric'])+max(history.history['val_max_f1_metric_horizon']))/2
    val_accuracy_mean = (np.mean(history.history['val_event_f1_metric'])+np.mean(history.history['val_max_f1_metric_horizon']))/2
    val_accuracy = (val_accuracy + val_accuracy_mean)/2
    val_latency = np.mean(history.history['val_event_fp_rate'])
    # Log results
    logger.info(f"Trial {trial.number} finished with val_accuracy: {val_accuracy:.4f}, val_fprate: {val_latency:.4f}")

    # Save trial information
    trial_info = {
        'parameters': params,
        'metrics': {
        'val_accuracy': val_accuracy,
        'val_latency': val_latency
        }
    }
    with open(f"{study_dir}/trial_info.json", 'w') as f:
        json.dump(trial_info, f, indent=4)

    # Proper cleanup after training
    del model
    gc.collect()
    tf.keras.backend.clear_session()

    return val_accuracy, val_latency


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
    params['HORIZON_MS'] = 1#trial.suggest_int('HORIZON_MS', 1, 5)
    params['SHIFT_MS'] = 0

    params['LOSS_WEIGHT'] = trial.suggest_float('LOSS_WEIGHT', 0.000001, 10.0, log=True)

    params['LOSS_NEGATIVES'] = trial.suggest_float('LOSS_NEGATIVES', 1.0, 1000.0, log=True)
    # params['LOSS_WEIGHT'] = 7.5e-4

    ax = trial.suggest_int('AX', 1, 99)
    gx = trial.suggest_int('GX', 50, 999)

    # Removed duplicate TYPE_ARCH suggestion that was causing the error
    # params['TYPE_LOSS'] = 'FocalGapAx{:03d}Gx{:03d}'.format(ax, gx)
    params['TYPE_LOSS'] = 'FocalAx{:03d}Gx{:03d}'.format(ax, gx)

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
    params['NO_FILTERS'] = 128#trial.suggest_categorical('NO_FILTERS', [32, 64, 128])
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
    # params['USE_L2N'] = trial.suggest_categorical('USE_L2N', [True, False])
    # if params['USE_L2N']:
    params['TYPE_ARCH'] += 'L2N'


    # params['USE_Aug'] = trial.suggest_categorical('USE_Aug', [True, False])
    # if params['USE_Aug']:
    params['TYPE_ARCH'] += 'Aug'


    params['USE_StopGrad'] = trial.suggest_categorical('USE_StopGrad', [True, False])
    if params['USE_StopGrad']:
        print('Using Stop Gradient for Class. Branch')
        params['TYPE_ARCH'] += 'StopGrad'

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

    drop_lib = [0, 5, 10, 20, 50, 80]
    drop_ind = 0#trial.suggest_categorical('Dropout', [0,1,2,3,4, 5])
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
    best_metric3 = float('inf')
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
                            # cb.ModelCheckpoint(
                            # f"{study_dir}/robust.weights.h5",
                            # monitor='val_robust_f1',  # Change monitor
                            # verbose=1,
                            # save_best_only=True,
                            # save_weights_only=True,
                            # mode='max'),
                            cb.ModelCheckpoint(
                            f"{study_dir}/event.weights.h5",
                            monitor='val_event_f1_metric',  # Change monitor
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
        current_metric2 = epoch_history.history.get('val_event_f1_metric', [float('-inf')])[0]
        current_metric3 = epoch_history.history.get('val_event_fp_rate', [float('inf')])[0]

        if (current_metric > (best_metric + min_delta)) or (current_metric2 > (best_metric2 + min_delta)) or (current_metric3 < (best_metric3 - min_delta)):
            if current_metric > best_metric:
                print(f"New best metric: {current_metric}")
                best_metric = current_metric
            elif current_metric2 > best_metric2:
                best_metric2 = current_metric2
                print(f"New best metric2: {current_metric2}")
            elif current_metric3 < best_metric3:
                best_metric3 = current_metric3
                print(f"New best metric3: {current_metric3}")
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
    # final_f1 = (np.mean(combined_history['val_robust_f1']) +
    #             max(combined_history['val_max_f1_metric_horizon'])) / 2
    # final_f1 = max(combined_history['val_event_f1_metric'])
    # final_latency = np.mean(combined_history['val_latency_metric'])
    final_f1 = np.mean(combined_history['val_event_f1_metric'])
    final_fp_penalty = np.mean(combined_history['val_event_fp_rate'])  # Or your new FP-aware metric

    trial_info = {
        'parameters': params,
        'metrics': {
            'val_f1_accuracy': final_f1,
            'val_fp_penalty': final_fp_penalty
            # 'val_latency': final_latency
        }
    }
    with open(f"{study_dir}/trial_info.json", 'w') as f:
        json.dump(trial_info, f, indent=4)

    return final_f1, final_fp_penalty #, final_latency

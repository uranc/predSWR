import os, pdb, time, copy, shutil, argparse, sys, glob, importlib, random
import joblib, json, logging, gc, random, ctypes, h5py, datetime, json
from os import path
from shutil import copyfile
import numpy as np
from scipy.stats import pearsonr
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks as cb
from tensorflow.keras import callbacks as cb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.storages import RDBStorage

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


def objective_triplet_old(trial):
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

########################################
########################################
########################################
########################################
############# STUDY HypLat #############
########################################
########################################
########################################
########################################

def _to_serializable(obj):
    # make numpy / sets / arrays JSON-safe
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, (np.ndarray,)):            return obj.tolist()
    if isinstance(obj, (set,)):                    return list(obj)
    return obj

def _atomic_write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_to_serializable)
    os.replace(tmp, path)

def objective_triplet(trial, model_name, tag, logger):
    ctypes.CDLL("libcuda.so.1")
    """Objective function for Optuna optimization"""

    tf.keras.backend.clear_session()    
    gc.collect()
    tf.config.run_functions_eagerly(False)
    tf.random.set_seed(1337); np.random.seed(1337); random.seed(1337)
    # Start with base parameters
    params = {'BATCH_SIZE': 128, 'SHUFFLE_BUFFER_SIZE': 4096*2,
            'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-2, 'NO_EPOCHS': 300,
            'NO_TIMEPOINTS': 50, 'NO_CHANNELS': 8, 'SRATE': 2500,
            'EXP_DIR': '/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
            'mode': 'train'
            }

    # Dynamic learning rate range
    learning_rate = 1e-2
    params['LEARNING_RATE'] = learning_rate
    batch_size = 128
    params['BATCH_SIZE'] = batch_size
    params['SRATE'] = 2500
    params['NO_EPOCHS'] = 500
    params['TYPE_MODEL'] = 'Base'

    arch_lib = ['MixerOnly', 'MixerHori',
                'MixerDori', 'DualMixerDori', 'MixerCori',
                'SingleCh', 'TripletOnly']
    # Model architecture parameters - Fix the categorical suggestion
    arch_ind = np.where([(arch.lower() in tag.lower()) for arch in arch_lib])[0][0]
    params['TYPE_ARCH'] = arch_lib[arch_ind]
    print(params['TYPE_ARCH'])

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
        # from model.input_proto import rippleAI_load_dataset
        from model.input_proto_new import rippleAI_load_dataset
        # from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN

    # pdb.set_trace()
    # Timing parameters remain the same
    params['NO_TIMEPOINTS'] = 64
    params['NO_STRIDES'] = 32

    params.update({
        "SHIFT_MS": 0, "HORIZON_MS": 1,
        "CIRCLE_m": 0.32, "CIRCLE_gamma": 20.0,
        "LOSS_Circle": 60.0, "LOSS_SupCon": 0.0, "SUPCON_T": 0.1,
        "BCE_ANC_ALPHA": 2.0, "BCE_POS_ALPHA": 2.0,
        "LOSS_WEIGHT": 1.0, "LABEL_SMOOTHING": 0.0,
        "LOSS_NEGATIVES_MIN": 4.0, "LOSS_NEGATIVES": 26.0,
        "LOSS_TV": 0.30, "SMOOTH_TYPE": "tMSE", "SMOOTH_SPACE": "logit", "SMOOTH_TAU": 3.5,
        "CLF_SCALE": 0.30,
        # ramps — keep as you had; no tuning needed
        "RAMP_DELAY": 0.02 * params.get("TOTAL_STEPS", 100000),
        "RAMP_STEPS": 0.30 * params.get("TOTAL_STEPS", 100000),
        "NEG_RAMP_DELAY": 0.10 * params.get("TOTAL_STEPS", 100000),
        "NEG_RAMP_STEPS": 0.60 * params.get("TOTAL_STEPS", 100000),
        "TV_DELAY": 0.08 * params.get("TOTAL_STEPS", 100000),
        "TV_DUR":   0.35 * params.get("TOTAL_STEPS", 100000),
    })

    # ---- Metric: Circle + MPNTuple (time-averaged sims) ----
    # 1. Structure (High Weight, Tight Clusters)
    params['LOSS_TupMPN']   = trial.suggest_float('LOSS_TupMPN', 50.0, 180.0, log=True)
    params['MARGIN_WEAK']   = trial.suggest_float('MARGIN_WEAK', 0.05, 0.15, step=0.05) # Keep this low!

    # 2. Geometry (Circle Fix)
    params['LOSS_Circle']   = trial.suggest_int('LOSS_Circle', 1, 20, step=10)
    params['CIRCLE_m']      = trial.suggest_float('CIRCLE_m', 0.25, 0.40, step=0.05)
    params['CIRCLE_gamma']  = trial.suggest_categorical('CIRCLE_gamma', [32.0, 64.0])

    # 3. Classification (Soft & Robust)
    params['BCE_POS_ALPHA'] = trial.suggest_float('BCE_POS_ALPHA', 1.0, 3.0, step=0.5)
    params['LABEL_SMOOTHING'] = trial.suggest_float('LABEL_SMOOTHING', 0.04, 0.18) # Handles jitter
    params['LOSS_NEGATIVES']= trial.suggest_int('LOSS_NEGATIVES', 16, 30, step=2)

    # 4. General
    params['LEARNING_RATE'] = trial.suggest_float('LEARNING_RATE', 1e-4, 2e-2, log=True)
    params['LOSS_TV']       = trial.suggest_float('LOSS_TV', 0.05, 0.8, log=True)
    params['USE_StopGrad']  = trial.suggest_categorical('USE_StopGrad', [False, True])

    True # Essential for this architecture
    if params['USE_StopGrad']:
        print('Using Stop Gradient for Class. Branch')
        params['TYPE_ARCH'] += 'StopGrad'

    # =====================  FIXED RIDGE / CONSTANTS  =====================
    params["USE_LR_SCHEDULE"] = True
    params["LR_WARMUP_RATIO"] = 0.10
    params["LR_COOL_RATIO"] = 0.80
    params["LR_FINAL_SCALE"] = 0.08
    params["WEIGHT_DECAY"]  = 1e-4
    params["CLIP_NORM"]     = 1.5

    params['TYPE_LOSS'] = 'HybridV3_Sigmoid'
    params['HYPER_MONO'] = 0 #trial.suggest_float('HYPER_MONO', 0.000001, 10.0, log=True)

    # Model parameters matching training format
    # params['NO_KERNELS'] = trial.suggest_int('NO_KERNELS', 2, 6) # for kernels 2,3,4,5,6
    params['NO_KERNELS'] = 4
    if params['NO_TIMEPOINTS'] == 32:
        dil_lib = [4,3,2,2,2]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 64:
        dil_lib = [5,4,3,3,3]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 92:
        dil_lib = [6,5,4,4,4]           # for kernels 2,3,4,5,6]
    elif params['NO_TIMEPOINTS'] == 128:
        dil_lib = [6,5,4,4,4]           # for kernels 2,3,4,5,6]
    elif params['NO_TIMEPOINTS'] == 196:
        dil_lib = [7,6,5,5,5]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 384:
        dil_lib = [8,7,6,6,6]           # for kernels 2,3,4,5,6
    params['NO_DILATIONS'] = dil_lib[params['NO_KERNELS']-2]
    # params['NO_DILATIONS'] = trial.suggest_int('NO_DILATIONS', 2, 6)
    params['NO_FILTERS'] = 32#trial.suggest_categorical('NO_FILTERS', [24, 32, 48])
    # params['NO_FILTERS'] = 64

    # Remove the hardcoded use_freq and derive it from tag instead

    params['TYPE_LOSS'] += tag
    print(params['TYPE_LOSS'])

    # regularization
    par_init = 'He'
    par_norm = 'LN'

    # act_lib = ['ELU', 'GELU'] # 'RELU',
    # par_act = act_lib[trial.suggest_int('IND_ACT', 0, len(act_lib)-1)]
    par_act = 'GELU'
    par_opt = 'AdamWA'
    
    # reg_lib = ['LOne',  'None'] #'LTwo',
    # par_reg = reg_lib[trial.suggest_int('IND_REG', 0, len(reg_lib)-1)]
    par_reg = 'None'  # No regularization for now

    params['TYPE_REG'] = (f"{par_init}"f"{par_norm}"f"{par_act}"f"{par_opt}"f"{par_reg}")
    # Build architecture string with timing parameters (adjust format)
    arch_str = (f"{params['TYPE_ARCH']}"  # Take first 4 chars: Hori/Dori/Cori
                f"{int(params['HORIZON_MS']):02d}")
    print(arch_str)
    params['TYPE_ARCH'] = arch_str


    # params['USE_Aug'] = trial.suggest_categorical('USE_Aug', [True, False])
    # if params['USE_Aug']:
    #     params['TYPE_ARCH'] += 'Aug'
    # params['TYPE_ARCH'] += 'Aug'


    # params['USE_Attention'] = trial.suggest_categorical('USE_Attention', [True, False])
    # if params['USE_Attention']:
    #     print('Using Attention')
    #     params['TYPE_ARCH'] += 'Att'
    
    params['TYPE_ARCH'] += 'Online'

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
    param_dir = f"params_{tag}"
    study_dir = f"studies/{param_dir}/study_{trial.number}_{trial.datetime_start.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(study_dir, exist_ok=True)

    # from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
    import importlib.util
    model_dir = f"studies/{param_dir}/base_model"
    spec = importlib.util.spec_from_file_location("model_fn", f"{model_dir}/model_fn.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    build_DBI_TCN = model_module.build_DBI_TCN_TripletOnly
    
    # Copy pred.py and model/ directory
    shutil.copy2('./pred.py', f"{study_dir}/pred.py")
    # if os.path.exists(f"{study_dir}/model"):
    #     shutil.rmtree(f"{study_dir}/model")
    shutil.copytree(model_dir, f"{study_dir}/model")
    preproc = True
    
    # minimal trial_info skeleton so later code can safely update it
    trial_info = {
        "study_name": (study.study_name if "study" in globals() else param_dir),
        "trial_number": int(trial.number),
        "start_time": datetime.datetime.now().isoformat(),
        "study_dir": study_dir,
        "run_name": run_name,
        "parameters": dict(params),   # snapshot NOW, before any mutation
        "dataset": {},                # will fill after dataset is built
        "environment": {
            "python": sys.version,
            "tensorflow": tf.__version__,
            "optuna": optuna.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        },
        "selected_epoch": None,
        "selection_metric": None,
        "metrics": {}                 # will fill after training/selection
    }

    # write immediately so the trial is traceable even if it prunes/crashes
    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)

        
    # (optional but handy for dashboards)
    try:
        trial.set_user_attr("study_dir", study_dir)
        trial.set_user_attr("run_name", run_name)
    except Exception:
        pass
    
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
            params['steps_per_epoch'] = 1000
            flag_online = 'Online' in params['TYPE_ARCH']
            train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=flag_online)
        else:
            train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)

    params.update(dataset_params)   # merge, don't overwrite
    # Ramps (keep constant for stability this round)
    

    # total steps
    ts     = float(params['steps_per_epoch'] * int(params['NO_EPOCHS']) * 0.9)
    print('Total Steps: ', ts)
    params['TOTAL_STEPS'] = ts
    params["RAMP_DELAY"]     = 0.02 * ts
    params["RAMP_STEPS"]     = 0.30 * ts
    params["NEG_RAMP_DELAY"] = 0.1 * ts
    params["NEG_RAMP_STEPS"] = 0.6 * ts
    params["TV_DELAY"]       = 0.25 * ts
    params["TV_DUR"]         = 0.40 * ts
    params['CLF_RAMP_DELAY']  = params['RAMP_DELAY']
    params['CLF_RAMP_STEPS']  = params['RAMP_STEPS']

    params['GRACE_MS'] = 5
    params['ANCHOR_MIN_MS'] = 20
    params['POS_MIN_MS'] = 10
    params['POS_EXCLUDE_ANCHORS'] = True
    
    # after: train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(...)
    trial_info["dataset"] = {
        "estimated_steps_per_epoch": int(params.get("ESTIMATED_STEPS_PER_EPOCH",
                                params.get("steps_per_epoch", 0))),
        "val_steps": int(params.get("VAL_STEPS", 0)),
        "label_ratio": float(label_ratio) if "label_ratio" in locals() else None,
        "no_timepoints": int(params.get("NO_TIMEPOINTS", 0)),
        "stride": int(params.get("NO_STRIDES", 0)),
        "srate": int(params.get("SRATE", 0)),
    }    
    trial_info["parameters_final"] = dict(params)

    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)
    
    # load model
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    
    # Early stopping with tunable patience
    model.summary()

    callbacks = [
        cb.EarlyStopping(monitor='val_sample_pr_auc', patience=50,mode='max',verbose=1,restore_best_weights=True),        
                 
        cb.TensorBoard(log_dir=f"{study_dir}/", write_graph=True, write_images=True, update_freq='epoch'),

        # Save best by F1 (max)
        cb.ModelCheckpoint(f"{study_dir}/mcc.weights.h5", monitor="val_sample_max_mcc",
                        mode="max", save_best_only=True, save_weights_only=True, verbose=1),
        
        # Save best by F1 (max)
        cb.ModelCheckpoint(f"{study_dir}/max.weights.h5", monitor='val_sample_max_f1',
                        save_best_only=True, save_weights_only=True, mode='max', verbose=1),

        # Save best by PR-AUC (max)
        cb.ModelCheckpoint(f"{study_dir}/event.weights.h5", monitor='val_sample_pr_auc',
                        save_best_only=True, save_weights_only=True, mode='max', verbose=1),
        
        # optuna.integration.TFKerasPruningCallback(trial, 'val_sample_pr_auc'),

    ]
    val_steps = dataset_params['VAL_STEPS']

    # Train and evaluate
    history = model.fit(
        train_dataset,
        steps_per_epoch=dataset_params['ESTIMATED_STEPS_PER_EPOCH'],
        validation_data=test_dataset,
        validation_steps=val_steps,  # Explicitly set to avoid the partial batch
        epochs=params['NO_EPOCHS'],
        callbacks=callbacks,
        verbose=1,
        max_queue_size=8)       # how many batches to keep ready
    hist = history.history

    # --- helper: safe history picker (never crashes on missing/short arrays)
    def hist_pick(hist, key, idx, default=np.nan):
        arr = hist.get(key, None)
        if arr is None:
            return float(default)
        if idx is None:
            return float(default)
        # guard negative or out-of-bounds
        if idx < 0 or idx >= len(arr):
            return float(default)
        try:
            return float(arr[idx])
        except (TypeError, ValueError):
            return float(default)

    # ---- choose the epoch by YOUR selector (max PR-AUC)
    pr_list = np.asarray(hist.get("val_sample_pr_auc", []), dtype=np.float64)
    if pr_list.size == 0 or np.all(np.isnan(pr_list)):
        raise optuna.TrialPruned("No valid PR-AUC history; pruning trial.")

    idx = int(np.nanargmax(pr_list))
    epoch_sel = idx  # for logging

    # ---- pick all requested metrics from that epoch
    prauc_sel = hist_pick(hist, "val_sample_pr_auc",     idx)        # selector metric itself
    rec07_sel = hist_pick(hist, "val_recall_at_0p7",     idx)        # recall@0.7 (sample-level)
    fpmin_sel = hist_pick(hist, "val_fp_per_min",        idx)        # FP/min (batch-level, thresh=0.3 in your current FP metric)
    lat_sel   = hist_pick(hist, "val_latency_score",     idx)        # latency score (↑)

    # also: F1 / MCC at that epoch
    f1_sel    = hist_pick(hist, "val_sample_max_f1",     idx)
    mcc_sel   = hist_pick(hist, "val_sample_max_mcc",    idx)

    # ---- log both the across-epochs bests and the selected-epoch snapshot
    logger.info(
        f"Trial {trial.number} SELECTED@epoch[{epoch_sel}] — PRAUC: {prauc_sel:.4f} | "
        f"Recall@0.7: {rec07_sel:.4f} | FP/min@0.3: {fpmin_sel:.3f} | Latency: {lat_sel:.4f} | "
        f"MaxF1: {f1_sel:.4f} | MaxMCC: {mcc_sel:.4f}"
    )

    # ---- stash into trial attrs for downstream viz/filtering
    trial.set_user_attr("sel_epoch",           int(epoch_sel))
    trial.set_user_attr("sel_prauc",           float(prauc_sel))
    trial.set_user_attr("sel_recall_at_0p7",   float(rec07_sel))
    trial.set_user_attr("sel_fp_per_min",      float(fpmin_sel))
    trial.set_user_attr("sel_latency_score",   float(lat_sel))
    trial.set_user_attr("sel_max_f1",          float(f1_sel))
    trial.set_user_attr("sel_max_mcc",         float(mcc_sel))

    # ---- also expand your saved JSON
    trial_info["selected_epoch"] = int(epoch_sel)
    trial_info["selection_metric"] = "val_sample_pr_auc"
    trial_info["selected_epoch_metrics"] = {
        'val_sample_pr_auc':  float(prauc_sel),
        'val_recall_at_0p7':  float(rec07_sel),
        'val_fp_per_min':     float(fpmin_sel),
        'val_latency_score':  float(lat_sel),
        'val_sample_max_f1':  float(f1_sel),
        'val_sample_max_mcc': float(mcc_sel),
    }
    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)

    # ---- quick sanity filter to kill obviously bad/degenerate runs (same signatures you used)
    bad = (
        (not np.isfinite(prauc_sel)) or (prauc_sel <= 0.001) or
        (not np.isfinite(rec07_sel)) or (rec07_sel <= 0.001) or (rec07_sel > 0.999) or
        (not np.isfinite(fpmin_sel)) or (fpmin_sel <= 0.001) or   # fp≈0 → meaningless flat model
        (not np.isfinite(lat_sel))   or (lat_sel >= 0.999)      # latency≈1 → bug signature
    )
    if bad:
        raise optuna.TrialPruned("Bug signature at selected epoch (prauc/rec≈0, fp≈0, or latency≈1).")

    # ---- finally: report to Optuna as 2-objective (prauc, fp/min@0.3)
    return float(prauc_sel), float(lat_sel) #float(fpmin_sel)


def objective_proxy(trial, model_name, tag, logger):
    ctypes.CDLL("libcuda.so.1")
    """Objective function for Optuna optimization - Phase 1 Proxy Anchor"""

    tf.keras.backend.clear_session()    
    gc.collect()
    tf.config.run_functions_eagerly(False)
    tf.random.set_seed(1337); np.random.seed(1337); random.seed(1337)
    
    # ============================================================
    # 1. PARAMETERS & ARCHITECTURE
    # ============================================================
    params = {'BATCH_SIZE': 128, 'SHUFFLE_BUFFER_SIZE': 4096*2,
            'WEIGHT_FILE': '', 'NO_EPOCHS': 300,
            'NO_TIMEPOINTS': 64, 'NO_CHANNELS': 8, 'SRATE': 2500,
            'EXP_DIR': '/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
            'mode': 'train', 'TYPE_MODEL': 'Base', 'NO_STRIDES': 32
            }

    # Dynamic learning rate range
    learning_rate = 1e-2
    params['LEARNING_RATE'] = learning_rate
    batch_size = 128
    params['BATCH_SIZE'] = batch_size
    params['SRATE'] = 2500
    params['NO_EPOCHS'] = 500
    params['TYPE_MODEL'] = 'Base'

    # Architecture Selection
    arch_lib = ['MixerOnly', 'MixerHori', 'MixerDori', 'DualMixerDori', 'MixerCori', 'SingleCh', 'TripletOnly']
    arch_ind = np.where([(arch.lower() in tag.lower()) for arch in arch_lib])[0][0]
    params['TYPE_ARCH'] = arch_lib[arch_ind]
    print(f"Architecture: {params['TYPE_ARCH']}")

    # Dynamic Imports
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
        from model.input_proto_new import rippleAI_load_dataset
        # build_DBI_TCN loaded dynamically later

    params['NO_TIMEPOINTS'] = 64
    params['NO_STRIDES'] = 32

    # ============================================================
    # 2. HYPERPARAMETERS
    # ============================================================
    
    # --- A. Metric Learning (The Core) ---
    params['LOSS_PROXY']     = trial.suggest_float('LOSS_PROXY', 0.01, 4.0, log=True)
    params['NUM_SUBCENTERS'] = trial.suggest_int('NUM_SUBCENTERS', 2, 16, step=2)
    params['PROXY_ALPHA']    = trial.suggest_float('PROXY_ALPHA', 16.0, 64.0, step=8.0)
    params['PROXY_MARGIN']   = trial.suggest_float('PROXY_MARGIN', 0.001, 1.2, step=0.1)
    
    # --- B. Classification Head & Regularization ---
    params['LOSS_NEGATIVES']  = trial.suggest_float('LOSS_NEGATIVES', 15.0, 42.0, step=3.0)
    params['LABEL_SMOOTHING'] = trial.suggest_float('LABEL_SMOOTHING', 0.0, 0.0)
    params['LOSS_TV']         = trial.suggest_float('LOSS_TV', 0.001, 0.1, log=True)
    
    # Dropout (Categorical)
    drop_lib = [0.1, 0.2, 0.3, 0.4]
    params['DROP_RATE']       = drop_lib[trial.suggest_int('DROP_RATE', 0, len(drop_lib)-2)]
    # params['DROP_RATE']       = drop_lib[trial.suggest_int('DROP_RATE', 0, len(drop_lib)-1)]

    # --- C. Constants / Fixed ---
    params['BCE_POS_ALPHA'] = 1.0
    params['LEARNING_RATE'] = trial.suggest_float('LEARNING_RATE', 5e-5, 5e-3, log=True)
    
    params['USE_StopGrad'] = int(trial.suggest_int('USE_StopGrad', 1, 1)) == 1
    if params['USE_StopGrad']:
        print('Using Stop Gradient for Class. Branch')
        params['TYPE_ARCH'] += 'StopGrad'

    params['USE_Attention'] = int(trial.suggest_int('USE_Attention', 0, 1)) == 1
    if params['USE_Attention']:
        print('Using Attention')
        params['TYPE_ARCH'] += 'Att'
    # --- D. Derived / Fixed Params ---
    params.update({
        "SHIFT_MS": 0, "HORIZON_MS": 1,
        "LOSS_WEIGHT": 1.0, 
        "LOSS_NEGATIVES_MIN": 1.0, 
        "NUM_CLASSES": 2,
        "PROXY_SCALING": 1.0,
        # Optimizer defaults
        "USE_LR_SCHEDULE": True,
        "LR_WARMUP_RATIO": 0.10,
        "LR_COOL_RATIO": 0.80,
        "LR_FINAL_SCALE": 0.08,
        "WEIGHT_DECAY": 1e-4,
        "CLIP_NORM": 1.5,
    })

    # =====================  FIXED RIDGE / CONSTANTS  =====================
    params['TYPE_LOSS'] = 'ProxyPhase1'
    params['HYPER_MONO'] = 0 

    # TCN Configuration
    params['NO_KERNELS'] = 4
    if params['NO_TIMEPOINTS'] == 32:   dil_lib = [4,3,2,2,2]
    elif params['NO_TIMEPOINTS'] == 64: dil_lib = [5,4,3,3,3]
    elif params['NO_TIMEPOINTS'] == 92: dil_lib = [6,5,4,4,4]
    elif params['NO_TIMEPOINTS'] == 128:dil_lib = [6,5,4,4,4]
    elif params['NO_TIMEPOINTS'] == 196:dil_lib = [7,6,5,5,5]
    elif params['NO_TIMEPOINTS'] == 384:dil_lib = [8,7,6,6,6]
    params['NO_DILATIONS'] = dil_lib[params['NO_KERNELS']-2]
    params['NO_FILTERS'] = 32
    params['EMBEDDING_DIM'] = params['NO_FILTERS']

    # Set Loss Name (Strict)
    params['TYPE_LOSS'] += tag
    print(params['TYPE_LOSS'])

    # Regularization Name Construction (Strict)
    par_init = 'He'
    par_norm = 'LN'
    par_act = 'GELU'
    par_opt = 'AdamWA'
    par_reg = 'None' 

    params['TYPE_REG'] = (f"{par_init}"f"{par_norm}"f"{par_act}"f"{par_opt}"f"{par_reg}")
    
    # Build architecture string
    arch_str = (f"{params['TYPE_ARCH']}" f"{int(params['HORIZON_MS']):02d}")
    print(arch_str)
    params['TYPE_ARCH'] = arch_str
    
    params['TYPE_ARCH'] += 'Online'
    params['TYPE_ARCH'] += f"Shift{int(params['SHIFT_MS']):02d}"

    # Build Run Name (STRICTLY PRESERVED STRUCTURE)
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
    
    params['NAME'] = run_name
    print(params['NAME'])

    # ============================================================
    # 3. SETUP & LOADING
    # ============================================================
    param_dir = f"params_{tag}"
    study_dir = f"studies/{param_dir}/study_{trial.number}_{trial.datetime_start.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(study_dir, exist_ok=True)

    import importlib.util
    model_dir = f"studies/{param_dir}/base_model"
    spec = importlib.util.spec_from_file_location("model_fn", f"{model_dir}/model_fn.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    build_DBI_TCN = model_module.build_DBI_TCN_TripletOnly
    
    shutil.copy2('./pred.py', f"{study_dir}/pred.py")
    shutil.copytree(model_dir, f"{study_dir}/model")
    
    trial_info = {
        "study_name": (study.study_name if "study" in globals() else param_dir),
        "trial_number": int(trial.number),
        "start_time": datetime.datetime.now().isoformat(),
        "study_dir": study_dir,
        "run_name": run_name,
        "parameters": dict(params),
        "dataset": {},
        "environment": {
            "python": sys.version,
            "tensorflow": tf.__version__,
            "optuna": optuna.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        },
        "selected_epoch": None,
        "selection_metric": None,
        "metrics": {}
    }
    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)

    # Load Data
    preproc = True
    if 'TripletOnly' in params['TYPE_ARCH']:
        params['steps_per_epoch'] = 1000
        flag_online = 'Online' in params['TYPE_ARCH']
        train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=flag_online)
    else:
        train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)

    params.update(dataset_params)

    # Ramps
    ts = float(params['steps_per_epoch'] * int(params['NO_EPOCHS']) * 0.9)
    print('Total Steps: ', ts)
    params['TOTAL_STEPS'] = ts
    params["RAMP_DELAY"]     = 0.02 * ts
    params["RAMP_STEPS"]     = 0.30 * ts
    params["NEG_RAMP_DELAY"] = 0.10 * ts 
    params["NEG_RAMP_STEPS"] = 0.60 * ts
    params["TV_DELAY"]       = 0.08 * ts
    params["TV_DUR"]         = 0.35 * ts
    params['CLF_RAMP_DELAY']  = params['RAMP_DELAY']
    params['CLF_RAMP_STEPS']  = params['RAMP_STEPS']

    params['GRACE_MS'] = 5
    params['ANCHOR_MIN_MS'] = 20
    params['POS_MIN_MS'] = 10
    params['POS_EXCLUDE_ANCHORS'] = True
    
    trial_info["dataset"] = {
        "estimated_steps_per_epoch": int(params.get("ESTIMATED_STEPS_PER_EPOCH", params.get("steps_per_epoch", 0))),
        "val_steps": int(params.get("VAL_STEPS", 0)),
        "label_ratio": float(label_ratio) if "label_ratio" in locals() else None,
        "no_timepoints": int(params.get("NO_TIMEPOINTS", 0)),
        "stride": int(params.get("NO_STRIDES", 0)),
        "srate": int(params.get("SRATE", 0)),
    }    
    trial_info["parameters_final"] = dict(params)
    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)
    
    # ============================================================
    # 4. BUILD & TRAIN
    # ============================================================
    # Model is compiled inside build_DBI_TCN
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()
    
    val_steps = dataset_params['VAL_STEPS']

    # Callbacks (Early Stopping + Checkpoints)
    callbacks = [
        cb.EarlyStopping(monitor='val_sample_pr_auc', patience=50, mode='max', verbose=1, restore_best_weights=True),        
        cb.TensorBoard(log_dir=f"{study_dir}/", write_graph=True, write_images=True, update_freq='epoch'),
        cb.ModelCheckpoint(f"{study_dir}/mcc.weights.h5", monitor="val_sample_max_mcc", mode="max", save_best_only=True, save_weights_only=True, verbose=1),
        cb.ModelCheckpoint(f"{study_dir}/max.weights.h5", monitor='val_sample_max_f1', save_best_only=True, save_weights_only=True, mode='max', verbose=1),
        cb.ModelCheckpoint(f"{study_dir}/event.weights.h5", monitor='val_sample_pr_auc', save_best_only=True, save_weights_only=True, mode='max', verbose=1),
    ]

    history = model.fit(
        train_dataset,
        steps_per_epoch=dataset_params['ESTIMATED_STEPS_PER_EPOCH'],
        validation_data=test_dataset,
        validation_steps=val_steps,
        epochs=params['NO_EPOCHS'],
        callbacks=callbacks,
        verbose=1,
        max_queue_size=8
    )
    hist = history.history

    # ============================================================
    # 5. METRIC SELECTION & PRUNING
    # ============================================================
    def hist_pick(hist, key, idx, default=np.nan):
        arr = hist.get(key, None)
        if arr is None or idx is None or idx < 0 or idx >= len(arr): return float(default)
        return float(arr[idx])

    # Select by Max PR-AUC
    pr_list = np.asarray(hist.get("val_sample_pr_auc", []), dtype=np.float64)
    if pr_list.size == 0 or np.all(np.isnan(pr_list)):
        raise optuna.TrialPruned("No valid PR-AUC history; pruning trial.")

    idx = int(np.nanargmax(pr_list))
    epoch_sel = idx 

    # Pick metrics (Metrics defined in build_DBI_TCN)
    prauc_sel = hist_pick(hist, "val_sample_pr_auc",     idx)
    rec_sel   = hist_pick(hist, "val_mean_high_conf_recall", idx) 
    fpmin_sel = hist_pick(hist, "val_mean_low_conf_fp", idx)
    lat_sel   = hist_pick(hist, "val_latency_score_range",     idx)
    f1_sel    = hist_pick(hist, "val_sample_max_f1",     idx)
    mcc_sel   = hist_pick(hist, "val_sample_max_mcc",    idx)

    # Logging
    logger.info(
        f"Trial {trial.number} SELECTED@epoch[{epoch_sel}] — PRAUC: {prauc_sel:.4f} | "
        f"RecHigh: {rec_sel:.4f} | FPLow: {fpmin_sel:.3f} | Lat: {lat_sel:.4f} | "
        f"MaxF1: {f1_sel:.4f} | MaxMCC: {mcc_sel:.4f}"
    )

    # User Attributes
    trial.set_user_attr("sel_epoch",           int(epoch_sel))
    trial.set_user_attr("sel_prauc",           float(prauc_sel))
    trial.set_user_attr("sel_high_conf_rec",   float(rec_sel))
    trial.set_user_attr("sel_low_conf_fp",     float(fpmin_sel))
    trial.set_user_attr("sel_latency_range",   float(lat_sel))
    trial.set_user_attr("sel_max_f1",          float(f1_sel))
    trial.set_user_attr("sel_max_mcc",         float(mcc_sel))

    # JSON Update
    trial_info["selected_epoch"] = int(epoch_sel)
    trial_info["selection_metric"] = "val_sample_pr_auc"
    trial_info["selected_epoch_metrics"] = {
        'val_sample_pr_auc':         float(prauc_sel),
        'val_mean_high_conf_recall': float(rec_sel),
        'val_mean_low_conf_fp':      float(fpmin_sel),
        'val_latency_score_range':   float(lat_sel),
        'val_sample_max_f1':         float(f1_sel),
        'val_sample_max_mcc':        float(mcc_sel),
    }
    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)

    # Pruning (Restored EXACT Logic)
    bad = (
        (not np.isfinite(prauc_sel)) or (prauc_sel <= 0.001) or
        (not np.isfinite(rec_sel))   or (rec_sel <= 0.001) or (rec_sel > 0.999) or
        (not np.isfinite(fpmin_sel)) or (fpmin_sel <= 0.001) or   
        (not np.isfinite(lat_sel))   or (lat_sel >= 0.999)      
    )
    if bad:
        raise optuna.TrialPruned("Bug signature at selected epoch.")

    return float(prauc_sel), float(fpmin_sel)

def objective_proxy_finetune(trial, model_name, tag, logger):
    ctypes.CDLL("libcuda.so.1")
    """Objective function for Optuna optimization - Phase 1 Proxy Anchor"""

    tf.keras.backend.clear_session()    
    gc.collect()
    tf.config.run_functions_eagerly(False)
    tf.random.set_seed(1337); np.random.seed(1337); random.seed(1337)
    
    # ============================================================
    # 1. PARAMETERS & ARCHITECTURE
    # ============================================================
    params = {'BATCH_SIZE': 128, 'SHUFFLE_BUFFER_SIZE': 4096*2,
            'WEIGHT_FILE': '', 'NO_EPOCHS': 300,
            'NO_TIMEPOINTS': 64, 'NO_CHANNELS': 8, 'SRATE': 2500,
            'EXP_DIR': '/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
            'mode': 'train', 'TYPE_MODEL': 'Base', 'NO_STRIDES': 32
            }

    # Dynamic learning rate range
    learning_rate = 1e-2
    params['LEARNING_RATE'] = learning_rate
    batch_size = 128
    params['BATCH_SIZE'] = batch_size
    params['SRATE'] = 2500
    params['NO_EPOCHS'] = 500
    params['TYPE_MODEL'] = 'Base'

    # Architecture Selection
    arch_lib = ['MixerOnly', 'MixerHori', 'MixerDori', 'DualMixerDori', 'MixerCori', 'SingleCh', 'TripletOnly']
    arch_ind = np.where([(arch.lower() in tag.lower()) for arch in arch_lib])[0][0]
    params['TYPE_ARCH'] = arch_lib[arch_ind]
    print(f"Architecture: {params['TYPE_ARCH']}")

    # Dynamic Imports
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
        from model.input_proto_new import rippleAI_load_dataset
        # build_DBI_TCN loaded dynamically later

    params['NO_TIMEPOINTS'] = 64
    params['NO_STRIDES'] = 32

    # ============================================================
    # 2. HYPERPARAMETERS
    # ============================================================
    
    # --- A. Metric Learning (The Core) ---
    params['LOSS_PROXY']     = trial.suggest_float('LOSS_PROXY', 0.0001, 0.5, log=True)
    params['NUM_SUBCENTERS'] = trial.suggest_int('NUM_SUBCENTERS', 8, 12, step=2)
    params['PROXY_ALPHA']    = trial.suggest_float('PROXY_ALPHA', 16.0, 64.0, step=16.0)
    params['PROXY_MARGIN']   = trial.suggest_float('PROXY_MARGIN', 0.4, 1.2, step=0.1)
    
    # --- B. Classification Head & Regularization ---
    params['LOSS_NEGATIVES']  = trial.suggest_float('LOSS_NEGATIVES', 21.0, 60.0, step=3.0)
    params['LABEL_SMOOTHING'] = trial.suggest_float('LABEL_SMOOTHING', 0.0, 0.0)
    params['LOSS_TV']         = trial.suggest_float('LOSS_TV', 0.0001, 0.05, log=True)
    
    # Dropout (Categorical)
    drop_lib = [0.1, 0.2, 0.3, 0.4]
    params['DROP_RATE']       = drop_lib[trial.suggest_int('DROP_RATE', 0, len(drop_lib)-3)]
    # params['DROP_RATE']       = drop_lib[trial.suggest_int('DROP_RATE', 0, len(drop_lib)-1)]

    # --- C. Constants / Fixed ---
    params['BCE_POS_ALPHA'] = 1.0
    params['LEARNING_RATE'] = trial.suggest_float('LEARNING_RATE', 5e-5, 5e-3, log=True)
    
    params['USE_StopGrad'] = int(trial.suggest_int('USE_StopGrad', 1, 1)) == 1
    if params['USE_StopGrad']:
        print('Using Stop Gradient for Class. Branch')
        params['TYPE_ARCH'] += 'StopGrad'

    params['USE_Attention'] = int(trial.suggest_int('USE_Attention', 1, 1)) == 1
    if params['USE_Attention']:
        print('Using Attention')
        params['TYPE_ARCH'] += 'Att'
    # --- D. Derived / Fixed Params ---
    params.update({
        "SHIFT_MS": 0, "HORIZON_MS": 1,
        "LOSS_WEIGHT": 1.0, 
        "LOSS_NEGATIVES_MIN": 1.0, 
        "NUM_CLASSES": 2,
        "PROXY_SCALING": 1.0,
        # Optimizer defaults
        "USE_LR_SCHEDULE": True,
        "LR_WARMUP_RATIO": 0.10,
        "LR_COOL_RATIO": 0.80,
        "LR_FINAL_SCALE": 0.08,
        "WEIGHT_DECAY": 1e-4,
        "CLIP_NORM": 1.5,
    })

    # =====================  FIXED RIDGE / CONSTANTS  =====================
    params['TYPE_LOSS'] = 'ProxyPhase1'
    params['HYPER_MONO'] = 0 

    # TCN Configuration
    params['NO_KERNELS'] = 4
    if params['NO_TIMEPOINTS'] == 32:   dil_lib = [4,3,2,2,2]
    elif params['NO_TIMEPOINTS'] == 64: dil_lib = [5,4,3,3,3]
    elif params['NO_TIMEPOINTS'] == 92: dil_lib = [6,5,4,4,4]
    elif params['NO_TIMEPOINTS'] == 128:dil_lib = [6,5,4,4,4]
    elif params['NO_TIMEPOINTS'] == 196:dil_lib = [7,6,5,5,5]
    elif params['NO_TIMEPOINTS'] == 384:dil_lib = [8,7,6,6,6]
    params['NO_DILATIONS'] = dil_lib[params['NO_KERNELS']-2]
    params['NO_FILTERS'] = 32
    params['EMBEDDING_DIM'] = params['NO_FILTERS']

    # Set Loss Name (Strict)
    params['TYPE_LOSS'] += tag
    print(params['TYPE_LOSS'])

    # Regularization Name Construction (Strict)
    par_init = 'He'
    par_norm = 'LN'
    par_act = 'GELU'
    par_opt = 'AdamWA'
    par_reg = 'None' 

    params['TYPE_REG'] = (f"{par_init}"f"{par_norm}"f"{par_act}"f"{par_opt}"f"{par_reg}")
    
    # Build architecture string
    arch_str = (f"{params['TYPE_ARCH']}" f"{int(params['HORIZON_MS']):02d}")
    print(arch_str)
    params['TYPE_ARCH'] = arch_str
    
    params['TYPE_ARCH'] += 'Online'
    params['TYPE_ARCH'] += f"Shift{int(params['SHIFT_MS']):02d}"

    # Build Run Name (STRICTLY PRESERVED STRUCTURE)
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
    
    params['NAME'] = run_name
    print(params['NAME'])

    # ============================================================
    # 3. SETUP & LOADING
    # ============================================================
    param_dir = f"params_{tag}"
    study_dir = f"studies/{param_dir}/study_{trial.number}_{trial.datetime_start.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(study_dir, exist_ok=True)

    import importlib.util
    model_dir = f"studies/{param_dir}/base_model"
    spec = importlib.util.spec_from_file_location("model_fn", f"{model_dir}/model_fn.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    build_DBI_TCN = model_module.build_DBI_TCN_TripletOnly
    
    shutil.copy2('./pred.py', f"{study_dir}/pred.py")
    shutil.copytree(model_dir, f"{study_dir}/model")
    
    trial_info = {
        "study_name": (study.study_name if "study" in globals() else param_dir),
        "trial_number": int(trial.number),
        "start_time": datetime.datetime.now().isoformat(),
        "study_dir": study_dir,
        "run_name": run_name,
        "parameters": dict(params),
        "dataset": {},
        "environment": {
            "python": sys.version,
            "tensorflow": tf.__version__,
            "optuna": optuna.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        },
        "selected_epoch": None,
        "selection_metric": None,
        "metrics": {}
    }
    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)

    # Load Data
    preproc = True
    if 'TripletOnly' in params['TYPE_ARCH']:
        params['steps_per_epoch'] = 1000
        flag_online = 'Online' in params['TYPE_ARCH']
        train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=flag_online)
    else:
        train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)

    params.update(dataset_params)

    # Ramps
    ts = float(params['steps_per_epoch'] * int(params['NO_EPOCHS']) * 0.9)
    print('Total Steps: ', ts)
    params['TOTAL_STEPS'] = ts
    params["RAMP_DELAY"]     = 0.02 * ts
    params["RAMP_STEPS"]     = 0.30 * ts
    params["NEG_RAMP_DELAY"] = 0.10 * ts 
    params["NEG_RAMP_STEPS"] = 0.60 * ts
    params["TV_DELAY"]       = 0.08 * ts
    params["TV_DUR"]         = 0.35 * ts
    params['CLF_RAMP_DELAY']  = params['RAMP_DELAY']
    params['CLF_RAMP_STEPS']  = params['RAMP_STEPS']

    params['GRACE_MS'] = 5
    params['ANCHOR_MIN_MS'] = 20
    params['POS_MIN_MS'] = 10
    params['POS_EXCLUDE_ANCHORS'] = True
    
    trial_info["dataset"] = {
        "estimated_steps_per_epoch": int(params.get("ESTIMATED_STEPS_PER_EPOCH", params.get("steps_per_epoch", 0))),
        "val_steps": int(params.get("VAL_STEPS", 0)),
        "label_ratio": float(label_ratio) if "label_ratio" in locals() else None,
        "no_timepoints": int(params.get("NO_TIMEPOINTS", 0)),
        "stride": int(params.get("NO_STRIDES", 0)),
        "srate": int(params.get("SRATE", 0)),
    }    
    trial_info["parameters_final"] = dict(params)
    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)
    
    # ============================================================
    # 4. BUILD & TRAIN
    # ============================================================
    # Model is compiled inside build_DBI_TCN
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()
    
    val_steps = dataset_params['VAL_STEPS']

    # Callbacks (Early Stopping + Checkpoints)
    callbacks = [
        cb.EarlyStopping(monitor='val_sample_pr_auc', patience=50, mode='max', verbose=1, restore_best_weights=True),        
        cb.TensorBoard(log_dir=f"{study_dir}/", write_graph=True, write_images=True, update_freq='epoch'),
        cb.ModelCheckpoint(f"{study_dir}/mcc.weights.h5", monitor="val_sample_max_mcc", mode="max", save_best_only=True, save_weights_only=True, verbose=1),
        cb.ModelCheckpoint(f"{study_dir}/max.weights.h5", monitor='val_sample_max_f1', save_best_only=True, save_weights_only=True, mode='max', verbose=1),
        cb.ModelCheckpoint(f"{study_dir}/event.weights.h5", monitor='val_sample_pr_auc', save_best_only=True, save_weights_only=True, mode='max', verbose=1),
    ]

    history = model.fit(
        train_dataset,
        steps_per_epoch=dataset_params['ESTIMATED_STEPS_PER_EPOCH'],
        validation_data=test_dataset,
        validation_steps=val_steps,
        epochs=params['NO_EPOCHS'],
        callbacks=callbacks,
        verbose=1,
        max_queue_size=8
    )
    hist = history.history

    # ============================================================
    # 5. METRIC SELECTION & PRUNING
    # ============================================================
    def hist_pick(hist, key, idx, default=np.nan):
        arr = hist.get(key, None)
        if arr is None or idx is None or idx < 0 or idx >= len(arr): return float(default)
        return float(arr[idx])

    # Select by Max PR-AUC
    pr_list = np.asarray(hist.get("val_sample_pr_auc", []), dtype=np.float64)
    if pr_list.size == 0 or np.all(np.isnan(pr_list)):
        raise optuna.TrialPruned("No valid PR-AUC history; pruning trial.")

    idx = int(np.nanargmax(pr_list))
    epoch_sel = idx 

    # Pick metrics (Metrics defined in build_DBI_TCN)
    prauc_sel = hist_pick(hist, "val_sample_pr_auc",     idx)
    rec_sel   = hist_pick(hist, "val_mean_high_conf_recall", idx) 
    fpmin_sel = hist_pick(hist, "val_mean_low_conf_fp", idx)
    lat_sel   = hist_pick(hist, "val_latency_score_range",     idx)
    f1_sel    = hist_pick(hist, "val_sample_max_f1",     idx)
    mcc_sel   = hist_pick(hist, "val_sample_max_mcc",    idx)

    # Logging
    logger.info(
        f"Trial {trial.number} SELECTED@epoch[{epoch_sel}] — PRAUC: {prauc_sel:.4f} | "
        f"RecHigh: {rec_sel:.4f} | FPLow: {fpmin_sel:.3f} | Lat: {lat_sel:.4f} | "
        f"MaxF1: {f1_sel:.4f} | MaxMCC: {mcc_sel:.4f}"
    )

    # User Attributes
    trial.set_user_attr("sel_epoch",           int(epoch_sel))
    trial.set_user_attr("sel_prauc",           float(prauc_sel))
    trial.set_user_attr("sel_high_conf_rec",   float(rec_sel))
    trial.set_user_attr("sel_low_conf_fp",     float(fpmin_sel))
    trial.set_user_attr("sel_latency_range",   float(lat_sel))
    trial.set_user_attr("sel_max_f1",          float(f1_sel))
    trial.set_user_attr("sel_max_mcc",         float(mcc_sel))

    # JSON Update
    trial_info["selected_epoch"] = int(epoch_sel)
    trial_info["selection_metric"] = "val_sample_pr_auc"
    trial_info["selected_epoch_metrics"] = {
        'val_sample_pr_auc':         float(prauc_sel),
        'val_mean_high_conf_recall': float(rec_sel),
        'val_mean_low_conf_fp':      float(fpmin_sel),
        'val_latency_score_range':   float(lat_sel),
        'val_sample_max_f1':         float(f1_sel),
        'val_sample_max_mcc':        float(mcc_sel),
    }
    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)

    # Pruning (Restored EXACT Logic)
    bad = (
        (not np.isfinite(prauc_sel)) or (prauc_sel <= 0.001) or
        (not np.isfinite(rec_sel))   or (rec_sel <= 0.001) or (rec_sel > 0.999) or
        (not np.isfinite(fpmin_sel)) or (fpmin_sel <= 0.001) or   
        (not np.isfinite(lat_sel))   or (lat_sel >= 0.999)      
    )
    if bad:
        raise optuna.TrialPruned("Bug signature at selected epoch.")

    return float(prauc_sel), float(fpmin_sel)

def objective_time_to_event(trial, model_name, tag, logger):
    ctypes.CDLL("libcuda.so.1")
    """Objective function for Optuna optimization"""

    tf.keras.backend.clear_session()    
    gc.collect()
    tf.config.run_functions_eagerly(False)
    tf.random.set_seed(1337); np.random.seed(1337); random.seed(1337)
    # Start with base parameters
    params = {'BATCH_SIZE': 128, 'SHUFFLE_BUFFER_SIZE': 4096*2,
            'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-2, 'NO_EPOCHS': 300,
            'NO_TIMEPOINTS': 50, 'NO_CHANNELS': 8, 'SRATE': 2500,
            'EXP_DIR': '/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
            'mode': 'train'
            }

    # Dynamic learning rate range
    learning_rate = 1e-2
    params['LEARNING_RATE'] = learning_rate
    batch_size = 128
    params['BATCH_SIZE'] = batch_size
    params['SRATE'] = 2500
    params['NO_EPOCHS'] = 500
    params['TYPE_MODEL'] = 'Base'

    arch_lib = ['MixerOnly', 'MixerHori',
                'MixerDori', 'DualMixerDori', 'MixerCori',
                'SingleCh', 'TripletOnly']
    # Model architecture parameters - Fix the categorical suggestion
    arch_ind = np.where([(arch.lower() in tag.lower()) for arch in arch_lib])[0][0]
    params['TYPE_ARCH'] = arch_lib[arch_ind]
    print(params['TYPE_ARCH'])

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
        # from model.input_augment_weighted import rippleAI_load_dataset
        from model.input_fn_TTE import rippleAI_load_dataset
        from model.model_fn import build_DBI_TCN_MixerOnly as build_DBI_TCN
    elif 'TripletOnly' in params['TYPE_ARCH']:
        # from model.input_proto import rippleAI_load_dataset
        from model.input_proto_new import rippleAI_load_dataset
        # from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN

    # pdb.set_trace()
    # Timing parameters remain the same
    params['NO_TIMEPOINTS'] = 64
    params['NO_STRIDES'] = 32

    params.update({
        "SHIFT_MS": 0, "HORIZON_MS": 1,
        'BATCH_SIZE': 64,
        'NO_TIMEPOINTS': 64,   # Window size
        'NO_STRIDES': 32,
        
        # --- NEW REGRESSION PARAMS ---
        'LABEL_RISE_MS': 30,    # Anticipation window (Sharp Wave slope)
        'LABEL_PLATEAU_MS': 12, # Certainty window (SWR peak)
        'LABEL_FALL_MS': 40,    # Reset window
        'LABEL_RISE_POWER': 2.0,# Quadratic ramp (fits physics better than linear)
        
        'WEIGHT_PLATEAU': 20.0, # Punishment for missing the peak
        'WEIGHT_TRANSITION': 5.0 # Punishment for missing the slope        
        
    })


    params['TYPE_LOSS'] = 'TTE'
    params['HYPER_MONO'] = 0 #trial.suggest_float('HYPER_MONO', 0.000001, 10.0, log=True)

    # Model parameters matching training format
    # params['NO_KERNELS'] = trial.suggest_int('NO_KERNELS', 2, 6) # for kernels 2,3,4,5,6
    params['NO_KERNELS'] = 4
    if params['NO_TIMEPOINTS'] == 32:
        dil_lib = [4,3,2,2,2]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 64:
        dil_lib = [5,4,3,3,3]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 92:
        dil_lib = [6,5,4,4,4]           # for kernels 2,3,4,5,6]
    elif params['NO_TIMEPOINTS'] == 128:
        dil_lib = [6,5,4,4,4]           # for kernels 2,3,4,5,6]
    elif params['NO_TIMEPOINTS'] == 196:
        dil_lib = [7,6,5,5,5]           # for kernels 2,3,4,5,6
    elif params['NO_TIMEPOINTS'] == 384:
        dil_lib = [8,7,6,6,6]           # for kernels 2,3,4,5,6
    params['NO_DILATIONS'] = dil_lib[params['NO_KERNELS']-2]
    # params['NO_DILATIONS'] = trial.suggest_int('NO_DILATIONS', 2, 6)
    params['NO_FILTERS'] = 32#trial.suggest_categorical('NO_FILTERS', [24, 32, 48])
    # params['NO_FILTERS'] = 64

    # Remove the hardcoded use_freq and derive it from tag instead

    params['TYPE_LOSS'] += tag
    print(params['TYPE_LOSS'])

    # regularization
    par_init = 'He'
    par_norm = 'LN'

    # act_lib = ['ELU', 'GELU'] # 'RELU',
    # par_act = act_lib[trial.suggest_int('IND_ACT', 0, len(act_lib)-1)]
    par_act = 'ELU'
    par_opt = 'AdamMixer'
    
    # reg_lib = ['LOne',  'None'] #'LTwo',
    # par_reg = reg_lib[trial.suggest_int('IND_REG', 0, len(reg_lib)-1)]
    par_reg = 'LOne'  # No regularization for now

    params['TYPE_REG'] = (f"{par_init}"f"{par_norm}"f"{par_act}"f"{par_opt}"f"{par_reg}")
    # Build architecture string with timing parameters (adjust format)
    arch_str = (f"{params['TYPE_ARCH']}"  # Take first 4 chars: Hori/Dori/Cori
                f"{int(params['HORIZON_MS']):02d}")
    print(arch_str)
    params['TYPE_ARCH'] = arch_str
    
    params['TYPE_ARCH'] += 'Online'

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
    param_dir = f"params_{tag}"
    study_dir = f"studies/{param_dir}/study_{trial.number}_{trial.datetime_start.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(study_dir, exist_ok=True)

    # from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
    import importlib.util
    model_dir = f"studies/{param_dir}/base_model"
    spec = importlib.util.spec_from_file_location("model_fn", f"{model_dir}/model_fn.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    build_DBI_TCN = model_module.build_DBI_TCN_TripletOnly
    
    # Copy pred.py and model/ directory
    shutil.copy2('./pred.py', f"{study_dir}/pred.py")
    # if os.path.exists(f"{study_dir}/model"):
    #     shutil.rmtree(f"{study_dir}/model")
    shutil.copytree(model_dir, f"{study_dir}/model")
    preproc = True
    
    # minimal trial_info skeleton so later code can safely update it
    trial_info = {
        "study_name": (study.study_name if "study" in globals() else param_dir),
        "trial_number": int(trial.number),
        "start_time": datetime.datetime.now().isoformat(),
        "study_dir": study_dir,
        "run_name": run_name,
        "parameters": dict(params),   # snapshot NOW, before any mutation
        "dataset": {},                # will fill after dataset is built
        "environment": {
            "python": sys.version,
            "tensorflow": tf.__version__,
            "optuna": optuna.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        },
        "selected_epoch": None,
        "selection_metric": None,
        "metrics": {}                 # will fill after training/selection
    }

    # write immediately so the trial is traceable even if it prunes/crashes
    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)

        
    # (optional but handy for dashboards)
    try:
        trial.set_user_attr("study_dir", study_dir)
        trial.set_user_attr("run_name", run_name)
    except Exception:
        pass
    
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
            params['steps_per_epoch'] = 1000
            flag_online = 'Online' in params['TYPE_ARCH']
            train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=flag_online)
        elif 'MixerOnly' in params['TYPE_ARCH']:
            flag_online = 'Online' in params['TYPE_ARCH']
            train_dataset, test_dataset = rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=flag_online)
        else:
            train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)

    params.update(dataset_params)   # merge, don't overwrite
    # Ramps (keep constant for stability this round)

    # after: train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(...)
    trial_info["dataset"] = {
        "val_steps": int(params.get("VAL_STEPS", 0)),
        "label_ratio": float(label_ratio) if "label_ratio" in locals() else None,
        "no_timepoints": int(params.get("NO_TIMEPOINTS", 0)),
        "stride": int(params.get("NO_STRIDES", 0)),
        "srate": int(params.get("SRATE", 0)),
    }    
    trial_info["parameters_final"] = dict(params)

    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)
    
    # load model
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    
    # Early stopping with tunable patience
    model.summary()

    callbacks = [
        cb.EarlyStopping(monitor='val_sample_pr_auc', patience=100,mode='max',verbose=1,restore_best_weights=True),        
        cb.TensorBoard(log_dir=f"{study_dir}/", write_graph=True, write_images=True, update_freq='epoch'),
        # Save best by F1 (max)
        cb.ModelCheckpoint(f"{study_dir}/mcc.weights.h5", monitor="val_sample_max_mcc",
                        mode="max", save_best_only=True, save_weights_only=True, verbose=1),
        # Save best by F1 (max)
        cb.ModelCheckpoint(f"{study_dir}/max.weights.h5", monitor='val_sample_max_f1',
                        save_best_only=True, save_weights_only=True, mode='max', verbose=1),
        # Save best by PR-AUC (max)
        cb.ModelCheckpoint(f"{study_dir}/event.weights.h5", monitor='val_sample_pr_auc',
                        save_best_only=True, save_weights_only=True, mode='max', verbose=1),
    ]
    val_steps = dataset_params['VAL_STEPS']

    # Train and evaluate
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        validation_steps=val_steps,  # Explicitly set to avoid the partial batch
        epochs=params['NO_EPOCHS'],
        callbacks=callbacks,
        verbose=1,
        max_queue_size=8)       # how many batches to keep ready
    hist = history.history

    # --- helper: safe history picker (never crashes on missing/short arrays)
    def hist_pick(hist, key, idx, default=np.nan):
        arr = hist.get(key, None)
        if arr is None:
            return float(default)
        if idx is None:
            return float(default)
        # guard negative or out-of-bounds
        if idx < 0 or idx >= len(arr):
            return float(default)
        try:
            return float(arr[idx])
        except (TypeError, ValueError):
            return float(default)

    # ---- choose the epoch by YOUR selector (max PR-AUC)
    pr_list = np.asarray(hist.get("val_sample_pr_auc", []), dtype=np.float64)
    if pr_list.size == 0 or np.all(np.isnan(pr_list)):
        raise optuna.TrialPruned("No valid PR-AUC history; pruning trial.")

    idx = int(np.nanargmax(pr_list))
    epoch_sel = idx  # for logging

    # ---- pick all requested metrics from that epoch
    prauc_sel = hist_pick(hist, "val_sample_pr_auc",     idx)        # selector metric itself
    rec07_sel = hist_pick(hist, "val_recall_at_0p7",     idx)        # recall@0.7 (sample-level)
    fpmin_sel = hist_pick(hist, "val_fp_per_min",        idx)        # FP/min (batch-level, thresh=0.3 in your current FP metric)
    lat_sel   = hist_pick(hist, "val_latency_score",     idx)        # latency score (↑)

    # also: F1 / MCC at that epoch
    f1_sel    = hist_pick(hist, "val_sample_max_f1",     idx)
    mcc_sel   = hist_pick(hist, "val_sample_max_mcc",    idx)

    # ---- log both the across-epochs bests and the selected-epoch snapshot
    logger.info(
        f"Trial {trial.number} SELECTED@epoch[{epoch_sel}] — PRAUC: {prauc_sel:.4f} | "
        f"Recall@0.7: {rec07_sel:.4f} | FP/min@0.3: {fpmin_sel:.3f} | Latency: {lat_sel:.4f} | "
        f"MaxF1: {f1_sel:.4f} | MaxMCC: {mcc_sel:.4f}"
    )

    # ---- stash into trial attrs for downstream viz/filtering
    trial.set_user_attr("sel_epoch",           int(epoch_sel))
    trial.set_user_attr("sel_prauc",           float(prauc_sel))
    trial.set_user_attr("sel_recall_at_0p7",   float(rec07_sel))
    trial.set_user_attr("sel_fp_per_min",      float(fpmin_sel))
    trial.set_user_attr("sel_latency_score",   float(lat_sel))
    trial.set_user_attr("sel_max_f1",          float(f1_sel))
    trial.set_user_attr("sel_max_mcc",         float(mcc_sel))

    # ---- also expand your saved JSON
    trial_info["selected_epoch"] = int(epoch_sel)
    trial_info["selection_metric"] = "val_sample_pr_auc"
    trial_info["selected_epoch_metrics"] = {
        'val_sample_pr_auc':  float(prauc_sel),
        'val_recall_at_0p7':  float(rec07_sel),
        'val_fp_per_min':     float(fpmin_sel),
        'val_latency_score':  float(lat_sel),
        'val_sample_max_f1':  float(f1_sel),
        'val_sample_max_mcc': float(mcc_sel),
    }
    _atomic_write_json(os.path.join(study_dir, "trial_info.json"), trial_info)

    # ---- quick sanity filter to kill obviously bad/degenerate runs (same signatures you used)
    bad = (
        (not np.isfinite(prauc_sel)) or (prauc_sel <= 0.001) or
        (not np.isfinite(rec07_sel)) or (rec07_sel <= 0.001) or (rec07_sel > 0.999) or
        (not np.isfinite(fpmin_sel)) or (fpmin_sel <= 0.001) or   # fp≈0 → meaningless flat model
        (not np.isfinite(lat_sel))   or (lat_sel >= 0.999)      # latency≈1 → bug signature
    )
    if bad:
        raise optuna.TrialPruned("Bug signature at selected epoch (prauc/rec≈0, fp≈0, or latency≈1).")

    # ---- finally: report to Optuna as 2-objective (prauc, fp/min@0.3)
    return float(prauc_sel), float(lat_sel) #float(fpmin_sel)

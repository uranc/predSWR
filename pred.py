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

# tf.config.run_functions_eagerly(False)
# tf.compat.v1.disable_eager_execution()
def objective_patch(trial):
    """Objective function for Optuna optimization"""
    # tf.compat.v1.reset_default_graph()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
    # tf.config.run_functions_eagerly(False)
    # tf.compat.v1.disable_eager_execution()
    # tf.debugging.set_log_device_placement(True)
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
        from model.model_fn import build_DBI_Patch as build_DBI_TCN

    # pdb.set_trace()
    # Timing parameters remain the same
    # params['NO_TIMEPOINTS'] = trial.suggest_categorical('NO_TIMEPOINTS', [128, 196, 384])
    params['NO_TIMEPOINTS'] = 64#*3
    # params['NO_STRIDES'] = int(params['NO_TIMEPOINTS'] // 2)
    params['NO_STRIDES'] = 32#trial.suggest_int('NO_STRIDES', 16, 64*2, step=32)

    # Timing parameters remain the same
    params['HORIZON_MS'] = 1#trial.suggest_int('HORIZON_MS', 0, 10, step=2)
    params['SHIFT_MS'] = 0

    # params['WEIGHT_Recon'] = trial.suggest_float('WEIGHT_Recon', 0.000001, 10, log=True)
    # params['WEIGHT_Patch'] = trial.suggest_categorical('WEIGHT_Patch', [-1.0, 1.0])
    # params['WEIGHT_Class'] = trial.suggest_float('WEIGHT_Class', 0.01, 1000, log=True)
    params['LOSS_WEIGHT'] = trial.suggest_float('LOSS_WEIGHT', 0.000001, 10.0, log=True)

    # entropyLib = [0, 0.5, 1, 3]
    # entropy_ind = trial.suggest_categorical('HYPER_ENTROPY', [0,1,2,3])
    # params['HYPER_ENTROPY'] = entropyLib[entropy_ind]
    # params['HYPER_MONO'] = 0
    params['NO_KERNELS'] = trial.suggest_int('NO_KERNELS', 1, 3) # for kernels 2,3,4,5,6
    params['NO_DILATIONS'] = trial.suggest_int('NO_DILATIONS', 1, 3)
    params['NO_FILTERS'] = trial.suggest_categorical('NO_FILTERS', [32, 64, 128])

    ax = 25
    gx = 150
    # ax = trial.suggest_int('AX', 25, 75, step=25)
    # gx = trial.suggest_int('GX', 50, 200, step=50)
    params['focal_alpha'] = ax
    params['focal_gamma'] = gx
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

    # import matplotlib.pyplot as plt
    # for ib in range(200):
    #     aa = next(iter(train_dataset))
    #     for ii in range(64):
    #         plt.subplot(8,8,ii+1)
    #         plt.plot(aa[0][ii,:,4])
    #         plt.plot(aa[1][ii,:,4]*1.2)
    #         plt.plot(aa[1][ii,:,8]*np.max(aa[1][ii,:,4]*1.2))
    #         plt.ylim(-3,3)
    #     plt.show()
    # pdb.set_trace()

    patch_lib = [64,32,16,4,2]
    patches = []
    for ii in range(params['NO_DILATIONS']):
        patches.append(patch_lib[ii])
    print(patches)
    params['seq_length'] = params["NO_TIMEPOINTS"]*2 # Or however seq_length is determined
    params['input_channels'] = params["NO_CHANNELS"] # Or however input_channels is determined
    params['patch_sizes'] = patches # Or however patch_size is determined

    model = build_DBI_TCN(params=params) # Pass only the params dictionary
    # model = build_DBI_TCN(params["NO_TIMEPOINTS"],
    #                     input_chans=8,
    #                     patch_sizes=patches,
    #                     d_model=params['NO_FILTERS'],
    #                     num_layers=params['NO_KERNELS'],
    #                     params=params)
    model.summary()
    from model.training import TerminateOnNaN
    terminate_on_nan_callback = TerminateOnNaN()
    # tf.profiler.experimental.start('logdir')

    # Setup callbacks including the verifier
    callbacks = [cb.TensorBoard(log_dir=f"{study_dir}/",
                                      write_graph=True,
                                      write_images=True,
                                    #   profile_batch='10,15',
                                      update_freq='epoch'),
                cb.EarlyStopping(monitor='val_event_f1',  # Change monitor
                                patience=50,
                                mode='max',
                                verbose=1,
                                restore_best_weights=True),
                cb.ModelCheckpoint(f"{study_dir}/max.weights.h5",
                                    monitor='val_f1',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max'),
                terminate_on_nan_callback,
                                    # cb.ModelCheckpoint(
                                    # f"{study_dir}/robust.weights.h5",
                                    # monitor='val_robust_f1',  # Change monitor
                                    # verbose=1,
                                    # save_best_only=True,
                                    # save_weights_only=True,
                                    # mode='max'),
                cb.ModelCheckpoint(f"{study_dir}/event.weights.h5",
                                    monitor='val_event_f1',  # Change monitor
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max')
    ]

    # Train and evaluate
    history = model.fit(
        train_dataset,
        # steps_per_epoch=30,
        validation_data=val_dataset,
        epochs=params['NO_EPOCHS'],
        callbacks=callbacks,
        verbose=1
    )
    # tf.profiler.experimental.stop()
    val_accuracy = (max(history.history['val_event_f1'])+max(history.history['val_f1']))/2
    val_accuracy_mean = (np.mean(history.history['val_event_f1'])+np.mean(history.history['val_f1']))/2
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
    tf.config.run_functions_eagerly(False)

    # Start with base parameters
    params = {'BATCH_SIZE': 64, 'SHUFFLE_BUFFER_SIZE': 4096*2,
            'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-3, 'NO_EPOCHS': 300,
            'NO_TIMEPOINTS': 50, 'NO_CHANNELS': 8, 'SRATE': 2500,
            'EXP_DIR': '/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
            'mode': 'train'
            }

    # Dynamic learning rate range
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    # learning_rate = 3e-3
    params['LEARNING_RATE'] = learning_rate

    # Optional batch size tuning
    batch_size = 128#trial.suggest_categorical('batch_size', [32, 64, 128])
    # batch_size = 64
    params['BATCH_SIZE'] = batch_size
    # ...rest of existing objective function code...

    # Base parameters
    params['SRATE'] = 2500
    params['NO_EPOCHS'] = 500
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
        # from model.input_proto import rippleAI_load_dataset
        from model.input_proto_new import rippleAI_load_dataset
        from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN

        # Set a reasonable steps_per_epoch value - much smaller than 1500 for initial testing
        # params['steps_per_epoch'] = 1200  # Increase gradually if training works

        # Add debug lines
        # print("Setting up triplet dataset with steps_per_epoch:", params['steps_per_epoch'])
        # print("Batch size:", params['BATCH_SIZE'])

    # pdb.set_trace()
    # Timing parameters remain the same
    params['NO_TIMEPOINTS'] = 64#trial.suggest_categorical('NO_TIMEPOINTS', [32, 64, 128])
    # params['NO_TIMEPOINTS'] = 92
    params['NO_STRIDES'] = 32#int(params['NO_TIMEPOINTS'] // 4)

    # Timing parameters remain the same
    params['HORIZON_MS'] = 1#trial.suggest_int('HORIZON_MS', 1, 5)
    params['SHIFT_MS'] = 0

    params['LOSS_TupMPN'] = trial.suggest_float('LOSS_TupMPN', 1.0, 1000.0, log=True)
    params['LOSS_SupCon'] = trial.suggest_float('LOSS_SupCon', 100.0, 1000.0, log=True)
    params['LOSS_WEIGHT'] = 1.0#trial.suggest_float('LOSS_WEIGHT', 0.001, 100.0, log=True)
    params['LOSS_NEGATIVES'] = trial.suggest_float('LOSS_NEGATIVES', 100.0, 1000.0, log=True)
    # params['LOSS_WEIGHT'] = 7.5e-4

    ax = 25#trial.suggest_int('AX', 25, 85, step=20)
    gx = 200#350#trial.suggest_int('GX', 150, 400, step=100)

    # Removed duplicate TYPE_ARCH suggestion that was causing the error
    # params['TYPE_LOSS'] = 'FocalGapAx{:03d}Gx{:03d}'.format(ax, gx)
    params['TYPE_LOSS'] = 'FocalAx{:03d}Gx{:03d}'.format(ax, gx)

    # entropyLib = [0, 0.005, 0.5, 1]
    # entropy_ind = trial.suggest_categorical('HYPER_ENTROPY', [0,1,2,3])
    # if entropy_ind > 0:
    #     params['HYPER_ENTROPY'] = entropyLib[entropy_ind]
    params['HYPER_ENTROPY'] = trial.suggest_float('HYPER_ENTROPY', 0.001, 5.0, log=True)
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
    params['NO_FILTERS'] = 32 #trial.suggest_categorical('NO_FILTERS', [32, 64, 128])
    # params['NO_FILTERS'] = 64

    # Remove the hardcoded use_freq and derive it from tag instead

    params['TYPE_LOSS'] += tag
    print(params['TYPE_LOSS'])


    # init_lib = ['He', 'Glo']
    # par_init = init_lib[trial.suggest_int('IND_INIT', 0, len(init_lib)-1)]
    par_init = 'He'
    # norm_lib = ['LN','BN','GN','WN']
    # norm_lib = ['LN', 'WN']
    # par_norm = norm_lib[trial.suggest_int('IND_NORM', 0, len(norm_lib)-1)]
    # par_norm = norm_lib[trial.suggest_int('IND_NORM', 0, len(norm_lib)-1)]
    par_norm = 'LN'

    # act_lib = ['ELU', 'GELU'] # 'RELU',
    # par_act = act_lib[trial.suggest_int('IND_ACT', 0, len(act_lib)-1)]
    par_act = 'ELU'

    # opt_lib = ['Adam', 'AdamW', 'SGD']
    par_opt = 'Adam'
    # par_opt = opt_lib[trial.suggest_int('IND_OPT', 0, len(opt_lib)-1)]

    reg_lib = ['LOne',  'None'] #'LTwo',
    par_reg = reg_lib[trial.suggest_int('IND_REG', 0, len(reg_lib)-1)]
    # par_reg = 'LOne'
    # par_reg = 'None'  # No regularization for now

    params['TYPE_REG'] = (f"{par_init}"f"{par_norm}"f"{par_act}"f"{par_opt}"f"{par_reg}")
    # Build architecture string with timing parameters (adjust format)
    arch_str = (f"{params['TYPE_ARCH']}"  # Take first 4 chars: Hori/Dori/Cori
                f"{int(params['HORIZON_MS']):02d}")
    print(arch_str)
    params['TYPE_ARCH'] = arch_str


    # Use multiple binary flags for a combinatorial categorical parameter
    # params['USE_ZNorm'] = trial.suggest_categorical('USE_ZNorm', [True, False])
    # if params['USE_ZNorm']:
    # params['TYPE_ARCH'] += 'ZNorm'
    # params['USE_L2N'] = trial.suggest_categorical('USE_L2N', [True, False])
    # if params['USE_L2N']:
    #     params['TYPE_ARCH'] += 'L2N'
    params['TYPE_ARCH'] += 'L2N'


    params['USE_Aug'] = trial.suggest_categorical('USE_Aug', [True, False])
    if params['USE_Aug']:
        params['TYPE_ARCH'] += 'Aug'


    # params['USE_StopGrad'] = trial.suggest_categorical('USE_StopGrad', [True, False])
    # if params['USE_StopGrad']:
    #     print('Using Stop Gradient for Class. Branch')
    #     params['TYPE_ARCH'] += 'StopGrad'
    # params['TYPE_ARCH'] += 'StopGrad'

    # params['USE_Attention'] = trial.suggest_categorical('USE_Attention', [True, False])
    # if params['USE_Attention']:
    #     print('Using Attention')
    #     params['TYPE_ARCH'] += 'Att'
    params['TYPE_ARCH'] += 'Att'
        
    # params['USE_Attention'] = trial.suggest_categorical('USE_Online', [True, False])
    # if params['USE_Attention']:
    #     print('Using Attention')
    params['TYPE_ARCH'] += 'Online'        
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

    # drop_lib = [0, 5, 10, 20]
    # drop_ind = trial.suggest_categorical('Dropout', [0,1,2,3])
    # print('Dropout rate:', drop_lib[drop_ind])
    # if drop_ind > 0:
    #     params['TYPE_ARCH'] += f"Drop{drop_lib[drop_ind]:02d}"

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
            params['steps_per_epoch'] = 1000
            flag_online = 'Online' in params['TYPE_ARCH']
            train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=flag_online)
        else:
            train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)

    # for batch in next(iter(train_dataset)):
    #     x_batch, y_batch = batch
    #     x_batch = x_batch.numpy()
    #     y_batch = y_batch.numpy()
    #     # Optional: convert x-axis to milliseconds
    #     t = np.arange(x_batch.shape[1]) / args.sr * 1000.0
    #     plot_triplet_batch(x_batch, y_batch,
    #                        n_examples=args.n_examples,
    #                        channel=args.channel,
    #                        timepoints=t,
    #                        title_prefix=f"SR={args.sr}Hz")        
    #     pdb.set_trace()

    # # Ensure we're not in eager execution mode during training
    # tf.config.run_functions_eagerly(False)

    # # Set thread settings for optimal performance
    # if 'TF_NUM_INTEROP_THREADS' not in os.environ:
    #     os.environ['TF_NUM_INTEROP_THREADS'] = '4'
    # if 'TF_NUM_INTRAOP_THREADS' not in os.environ:
    #     os.environ['TF_NUM_INTRAOP_THREADS'] = '0'  # Let TF decide based on CPU count

    # # Set memory optimizations
    # if 'TF_GPU_ALLOCATOR' not in os.environ:
    #     os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # # # Disable JIT compilation logs to reduce noise
    # # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # # Enable XLA compilation for faster training
    # if params.get('ENABLE_XLA', True):
    #     tf.config.optimizer.set_jit(True)

    # # Enable tensor fusion for increased performance
    # tf.config.optimizer.set_experimental_options({
    #     'layout_optimizer': True,
    #     'constant_folding': True,
    #     'shape_optimization': True,
    #     'remapping': True,
    #     'arithmetic_optimization': True,
    #     'dependency_optimization': True,
    #     'loop_optimization': True,
    #     'function_optimization': True,
    #     'debug_stripper': True,
    #     'scoped_allocator_optimization': True,
    #     'pin_to_host_optimization': True,
    #     'implementation_selector': True,
    #     # 'auto_mixed_precision': True,
    #     'disable_meta_optimizer': False
    # })

    # if 'TripletOnly' in params['TYPE_ARCH']:
    #         train_dataset = transform_dataset_for_training(train_dataset)
    #         val_dataset = transform_dataset_for_training(val_dataset)
    # if params['TYPE_MODEL'] == 'SingleCh':
    #     model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params, input_chans=1)
    # else:
    # pdb.set_trace()
    model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    # Early stopping with tunable patience
    model.summary()
    # Early stopping parameters
    best_metric = float('-inf')
    best_metric2 = float('-inf')
    best_metric3 = float('inf')
    patience = 30
    min_delta = 0.0001
    patience_counter = 0

    # Create a list to collect history from each epoch
    history_list = []

    # val_event_pr_auc: 0.83 ‖ val_latency_weighted_f1: 0.71 ‖ val_tpr_at_fpmin: 0.88

   # Setup callbacks including the verifier
    callbacks = [cb.TensorBoard(log_dir=f"{study_dir}/",
                                      write_graph=True,
                                      write_images=True,
                                      update_freq='epoch'),
        cb.EarlyStopping(monitor='val_event_pr_auc',  # Change monitor
                        patience=patience,
                        mode='max',
                        verbose=1,
                        restore_best_weights=True),
        cb.ModelCheckpoint(f"{study_dir}/max.weights.h5",
                            monitor='val_f1',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='max'),                        
        # cb.ModelCheckpoint(f"{study_dir}/max.weights.h5",
        #                     monitor='val_latency_weighted_f1',
        #                     verbose=1,
        #                     save_best_only=True,
        #                     save_weights_only=True,
        #                     mode='max'),
                            # cb.ModelCheckpoint(
                            # f"{study_dir}/robust.weights.h5",
                            # monitor='val_robust_f1',  # Change monitor
                            # verbose=1,
                            # save_best_only=True,
                            # save_weights_only=True,
                            # mode='max'),
                            cb.ModelCheckpoint(
                            f"{study_dir}/event.weights.h5",
                            monitor='val_event_pr_auc',  # Change monitor
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='max')
    ]

    # # Setup callbacks including the verifier
    # callbacks = [cb.TensorBoard(log_dir=f"{study_dir}/",
    #                                   write_graph=True,
    #                                   write_images=True,
    #                                   update_freq='epoch'),
    #     cb.EarlyStopping(monitor='val_event_f1',  # Change monitor
    #                     patience=patience,
    #                     mode='max',
    #                     verbose=1,
    #                     restore_best_weights=True),
    #     cb.ModelCheckpoint(f"{study_dir}/max.weights.h5",
    #                         monitor='val_f1',
    #                         verbose=1,
    #                         save_best_only=True,
    #                         save_weights_only=True,
    #                         mode='max'),
    #                         # cb.ModelCheckpoint(
    #                         # f"{study_dir}/robust.weights.h5",
    #                         # monitor='val_robust_f1',  # Change monitor
    #                         # verbose=1,
    #                         # save_best_only=True,
    #                         # save_weights_only=True,
    #                         # mode='max'),
    #                         cb.ModelCheckpoint(
    #                         f"{study_dir}/event.weights.h5",
    #                         monitor='val_event_f1',  # Change monitor
    #                         verbose=1,
    #                         save_best_only=True,
    #                         save_weights_only=True,
    #                         mode='max')
    # ]

    # # Loop through epochs manually
    # n_epoch = params['NO_EPOCHS']
    # for epoch in range(n_epoch):
    #     print(f"\nEpoch {epoch+1}/{n_epoch}")
    #     if dataset_params is not None and 'triplet_regenerator' in dataset_params:
    #         regenerating_dataset = dataset_params['triplet_regenerator']
    #         print(f"Regenerating triplet samples for epoch {epoch+1}")

    #         if epoch > 0:
    #             regenerating_dataset.reinitialize()
    #         train_data = regenerating_dataset.dataset if hasattr(regenerating_dataset, 'dataset') else regenerating_dataset

    #         steps = 1000 #dataset_params.get('steps_per_epoch', 500)
    #         # pdb.set_trace()
    #         epoch_history = model.fit(train_data,
    #             steps_per_epoch=steps,
    #             initial_epoch=epoch,
    #             epochs=epoch+1,
    #             validation_data=test_dataset,
    #             callbacks=callbacks,
    #             verbose=1
    #         )

    #     # Collect history
    #     history_list.append(epoch_history.history)

    #     # Early stopping check after each epoch
    #     current_metric = epoch_history.history.get('val_f1', [float('-inf')])[0]
    #     current_metric2 = epoch_history.history.get('val_event_f1', [float('-inf')])[0]
    #     current_metric3 = epoch_history.history.get('val_event_fp_rate', [float('inf')])[0]

    #     if (current_metric > (best_metric + min_delta)) or (current_metric2 > (best_metric2 + min_delta)) or (current_metric3 < (best_metric3 - min_delta)):
    #         if current_metric > best_metric:
    #             print(f"New best metric: {current_metric}")
    #             best_metric = current_metric
    #         elif current_metric2 > best_metric2:
    #             best_metric2 = current_metric2
    #             print(f"New best metric2: {current_metric2}")
    #         elif current_metric3 < best_metric3:
    #             best_metric3 = current_metric3
    #             print(f"New best metric3: {current_metric3}")
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1

    #     if patience_counter >= patience:
    #         print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
    #         break
    val_steps = dataset_params['VAL_STEPS']
    # pdb.set_trace()    # Train and evaluate
    history = model.fit(
        train_dataset,
        steps_per_epoch=dataset_params['ESTIMATED_STEPS_PER_EPOCH'],
        validation_data=test_dataset,
        validation_steps=val_steps,  # Explicitly set to avoid the partial batch
        epochs=params['NO_EPOCHS'],
        callbacks=callbacks,
        verbose=1,
        max_queue_size=8)       # how many batches to keep ready
            # workers=4,               # >0  → background threads / procs
        # use_multiprocessing=True,# True → processes, False → threads

    # tf.profiler.experimental.stop()
    # val_event_pr_auc: 0.83 ‖ val_latency_weighted_f1: 0.71 ‖ val_tpr_at_fpmin: 0.88

    # val_accuracy = (max(history.history['val_event_pr_auc'])+max(history.history['val_latency_weighted_f1']))/2
    # val_accuracy_mean = (np.mean(history.history['val_event_pr_auc'])+np.mean(history.history['val_latency_weighted_f1']))/2
    val_accuracy = (max(history.history['val_event_pr_auc'])+max(history.history['val_f1']))/2
    val_accuracy_mean = (np.mean(history.history['val_event_pr_auc'])+np.mean(history.history['val_f1']))/2    
    val_accuracy = (val_accuracy + val_accuracy_mean)/2
    val_latency = np.mean(history.history['val_fp_per_min'])

    # val_accuracy = (max(history.history['val_event_f1'])+max(history.history['val_f1']))/2
    # val_accuracy_mean = (np.mean(history.history['val_event_f1'])+np.mean(history.history['val_f1']))/2
    # val_accuracy = (val_accuracy + val_accuracy_mean)/2
    # val_latency = np.mean(history.history['val_event_fp_rate'])
    # Log results
    logger.info(f"Trial {trial.number} finished with val_accuracy: {val_accuracy:.4f}, val_fprate: {val_latency:.4f}")

    # Save trial information
    trial_info = {
        'parameters': params,
        'metrics': {
        'val_f1_accuracy': val_accuracy,
        'val_fp_penalty': val_latency
        }
    }

    # # Combine histories from all epochs
    # combined_history = {}
    # for key in history_list[0].keys():
    #     combined_history[key] = []
    #     for h in history_list:
    #         combined_history[key].extend(h[key])

    # If the trial completes, compute the final metrics.
    # pdb.set_trace()
    # final_f1 = (np.mean(combined_history['val_robust_f1']) +
    #             max(combined_history['val_max_f1_metric_horizon'])) / 2
    # final_f1 = max(combined_history['val_event_f1_metric'])
    # final_latency = np.mean(combined_history['val_latency_metric'])

    # final_f1 = (np.mean(combined_history['val_f1']) +
    #             max(combined_history['val_f1']) +
    #             np.mean(combined_history['val_event_f1']) +
    #             max(combined_history['val_event_f1'])) / 4
    # # final_f1 = np.mean(combined_history['val_event_f1'])
    # final_fp_penalty = np.mean(combined_history['val_event_fp_rate'])  # Or your new FP-aware metric

    # trial_info = {
    #     'parameters': params,
    #     'metrics': {
    #         'val_f1_accuracy': final_f1,
    #         'val_fp_penalty': final_fp_penalty
    #         # 'val_latency': final_latency
    #     }
    # }
    with open(f"{study_dir}/trial_info.json", 'w') as f:
        json.dump(trial_info, f, indent=4)

    return val_accuracy, val_latency #, final_latency

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
    gx = 200

    # Removed duplicate TYPE_ARCH suggestion that was causing the error
    params['TYPE_LOSS'] = 'Samp{}FocalGapAx{:03d}Gx{:03d}Entropy'.format(params['SRATE'], ax, gx)

    # Remove the hardcoded use_freq and derive it from tag instead

    params['TYPE_LOSS'] += tag
    print(params['TYPE_LOSS'])
    # init_lib = ['He', 'Glo']
    # par_init = init_lib[trial.suggest_int('IND_INIT', 0, len(init_lib)-1)]
    par_init = 'Glo'
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
    pdb.set_trace()
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
    # tag = args.tag[0]
    # arch_ind = np.where([(arch.lower() in tag.lower()) for arch in arch_lib])[0][0]
    arch_ind = 0
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
    params['NO_TIMEPOINTS'] = 32#*3
    params['NO_STRIDES'] = int(params['NO_TIMEPOINTS'] // 4)
    # params['NO_STRIDES'] = trial.suggest_int('NO_STRIDES', 32, 160, step=32)

    # Timing parameters remain the same
    params['HORIZON_MS'] = 0#trial.suggest_int('HORIZON_MS', 1, 5, step=2)
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
    params['NO_FILTERS'] = trial.suggest_categorical('NO_FILTERS', [32, 64, 128])
    # params['NO_FILTERS'] = 64
    # ax = trial.suggest_int('AX', 25, 75, step=10)
    # gx = trial.suggest_int('GX', 50, 999, step=150)
    ax = 25
    gx = 200
    # Removed duplicate TYPE_ARCH suggestion that was causing the error
    params['TYPE_LOSS'] = 'FocalGapAx{:03d}Gx{:03d}Entropy'.format(ax, gx)

    # Remove the hardcoded use_freq and derive it from tag instead

    # params['TYPE_LOSS'] += tag
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
    # params['USE_ZNorm'] = False#trial.suggest_categorical('USE_ZNorm', [True, False])
    # if params['USE_ZNorm']:
    #     params['TYPE_ARCH'] += 'ZNorm'
    # params['USE_L2N'] = True#trial.suggest_categorical('USE_L2N', [True, False])
    # if params['USE_L2N']:
    #     params['TYPE_ARCH'] += 'L2N'


    # params['USE_Aug'] = trial.suggest_categorical('USE_Aug', [True, False])
    # if params['USE_Aug']:
    #     params['TYPE_ARCH'] += 'Aug'

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
    else:
        params['LOSS_WEIGHT'] = 1

    # get sampling rate # little dangerous assumes 4 digits
    if 'Samp' in params['TYPE_LOSS']:
        sind = params['TYPE_LOSS'].find('Samp')
        try:
            # First attempt with 5 digits
            params['SRATE'] = int(params['TYPE_LOSS'][sind+4:sind+9])
        except ValueError:
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
        # from model.input_proto import rippleAI_load_dataset
        from model.input_proto_new import rippleAI_load_dataset
        from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
    elif 'CADOnly' in params['TYPE_ARCH']:
        from model.model_fn import build_DBI_TCN_CADMixerOnly as build_DBI_TCN
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
    elif model_name.find('Patch') != -1:
        from model.model_fn import build_DBI_Patch as build_DBI_TCN
        from model.input_augment_weighted import rippleAI_load_dataset
    else:
        from model.model_fn import build_DBI_TCN
        from model.input_aug import rippleAI_load_dataset

    # input
    if 'Patch' in model_name:
        # tf.config.run_functions_eagerly(True)
        # model = build_DBI_TCN(params["NO_TIMEPOINTS"],
        #                       input_chans=8,
        #                       patch_sizes=[64,32],#params['NO_DILATIONS'],
        #                       d_model=params['NO_FILTERS'],
        #                       num_layers=params['NO_KERNELS'],
        #                       params=params)
        model = build_DBI_TCN(params)

    elif 'CADOnly' in model_name:
        pretrain_tag = 'params_mixerOnlyEvents2500'
        pretrain_num = 1414#958
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

        import importlib.util
        spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        build_DBI_TCN_Pretrained = model_module.build_DBI_TCN_MixerOnly

        pretrained_tcn = build_DBI_TCN_Pretrained(pretrained_params["NO_TIMEPOINTS"], params=pretrained_params)
        pretrained_tcn.load_weights(weight_file)
        pretrained_tcn.trainable = False
        pretrained_tcn.compile(optimizer='adam', loss='mse')
        model = build_DBI_TCN(pretrained_params["NO_TIMEPOINTS"], params=params, pretrained_tcn=pretrained_tcn)
    elif ('Proto' not in model_name) and ('Barlow' not in model_name):
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()

    # pdb.set_trace()
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
            flag_online = 'Online' in params['TYPE_ARCH']
            train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=flag_online)
        else:
            train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)
    # train_size = len(list(train_dataset))
    params['RIPPLE_RATIO'] = label_ratio

    # --- Preview one triplet batch (debug utility) ---
    DEBUG_PLOT_TRIPLET = True  # set False to disable
    # pdb.set_trace()
    if DEBUG_PLOT_TRIPLET:
        # Fetch first batch (supports tf.data.Dataset or Keras Sequence / custom Sequence)
        if isinstance(train_dataset, tf.data.Dataset):
            first_batch = next(iter(train_dataset))
        else:  # Sequence-like
            first_batch = train_dataset[0]

        # Unpack (x, y, *rest)
        if isinstance(first_batch, (list, tuple)):
            if len(first_batch) < 2:
                raise ValueError("First batch does not contain at least (x, y)")
            x_preview, y_preview = first_batch[0], first_batch[1]
        else:
            raise ValueError("Unsupported batch type returned by dataset")

        # Convert tensors to numpy
        if tf.is_tensor(x_preview):
            x_preview = x_preview.numpy()
        if tf.is_tensor(y_preview):
            y_preview = y_preview.numpy()

        # pdb.set_trace()
        n=0
        for ii in range(0, 10):
            n+=1
            plt.subplot(10,3,n)
            plt.plot(np.arange(128), x_preview[0+ii, :, :]*4+np.array([0, 5, 10, 15, 20, 25, 30, 35]))
            plt.plot(np.arange(64)+64, y_preview[0+ii, :]*50, 'r')
            n+=1
            plt.subplot(10,3,n)
            plt.plot(np.arange(128), x_preview[32+ii, :, :]*4+np.array([0, 5, 10, 15, 20, 25, 30, 35]))
            plt.plot(np.arange(64)+64, y_preview[32+ii, :]*50, 'r')
            n+=1
            plt.subplot(10,3,n)
            plt.plot(np.arange(128), x_preview[64+ii, :, :]*4+np.array([0, 5, 10, 15, 20, 25, 30, 35]))
            plt.plot(np.arange(64)+64, y_preview[64+ii, :]*50, 'r')
        plt.show()
    pdb.set_trace()
    # n = 0
    # pdb.set_trace()
    # for ii in range(2000):
    #     [x, y] = next(iter(train_dataset))
    #     n += 1
    #     plt.subplot(10, 6, n)
    #     plt.plot(x[ii, 128:, :]*4+np.array([0, 5, 10, 15, 20, 25, 30, 35]))
    #     plt.plot(y[ii, :]*50, 'r')
    #     n += 1
    #     plt.subplot(10, 6, n)
    #     plt.plot(x[ii+32, 128:, :]*4+np.array([0, 5, 10, 15, 20, 25, 30, 35]))
    #     plt.plot(y[ii+32, :]*50, 'r')
    #     n += 1
    #     plt.subplot(10, 6, n)
    #     plt.plot(x[ii+64, 128:, :]*4+np.array([0, 5, 10, 15, 20, 25, 30, 35]))
    #     plt.plot(y[ii+64, :]*50, 'r')
    #     if ii % 20 == 19:
    #         plt.show()
    #         n = 0

    # Calculate model FLOPs using TensorFlow Profiler
    @tf.function
    def get_flops(model, batch_size=1, params=params):
        concrete_func = tf.function(lambda x: model(x))
        if 'TripletOnly' in params['TYPE_ARCH']:
            tt = tf.TensorSpec([batch_size*3, params['NO_TIMEPOINTS'], params['NO_CHANNELS']], tf.float32)
            frozen_func = concrete_func.get_concrete_function(tt)
        elif 'CADOnly' in params['TYPE_ARCH']:
            tt = tf.TensorSpec([batch_size, 1104, params['NO_CHANNELS']], tf.float32)
            frozen_func = concrete_func.get_concrete_function(tt)
        elif 'Patch' in params['TYPE_ARCH']:
            tt = tf.TensorSpec([batch_size, params['NO_TIMEPOINTS']*2, params['NO_CHANNELS']], tf.float32)
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
        study_dirs = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/study_{study_num}_*')
        # study_dirs = glob.glob(f'studies_CHECK_SIGNALS/{param_dir}/study_{study_num}_*')
        if not study_dirs:
            raise ValueError(f"No study directory found for study number {study_num}")
        study_dir = study_dirs[0]  # Take the first matching directory

        # Load trial info to get parameters
        with open(f"{study_dir}/trial_info.json", 'r') as f:
            trial_info = json.load(f)
            params.update(trial_info['parameters'])

        params['mode'] = 'predict'
        # Import required modules
        if 'MixerOnly' in params['TYPE_ARCH']:
            from model.model_fn import build_DBI_TCN_MixerOnly as build_DBI_TCN
            # import importlib.util
            # spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
            # model_module = importlib.util.module_from_spec(spec)
            # spec.loader.exec_module(model_module)
            # build_DBI_TCN = model_module.build_DBI_TCN_MixerOnly
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
        elif 'Patch' in params['TYPE_ARCH']:
            tf.config.run_functions_eagerly(True)
            from model.input_augment_weighted import rippleAI_load_dataset
            from model.model_fn import build_DBI_Patch as build_DBI_TCN

        elif 'CADOnly' in params['TYPE_ARCH']:
            pretrain_tag = 'params_mixerOnlyEvents2500'
            pretrain_num = 1414#958
            pretrained_dir = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{pretrain_tag}/study_{pretrain_num}_*')
            if not pretrained_dir:
                raise ValueError(f"No study directory found for study number {pretrain_num}")
            pretrained_dir = pretrained_dir[0]  # Take the first matching directory

            pretrained_params = copy.deepcopy(params)
            # Load trial info to get parameters
            with open(f"{pretrained_dir}/trial_info.json", 'r') as f:
                trial_info = json.load(f)
                pretrained_params.update(trial_info['parameters'])

            # Check which weight file is most recent
            event_weights = f"{pretrained_dir}/event.weights.h5"
            max_weights = f"{pretrained_dir}/max.weights.h5"

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
                raise ValueError(f"Neither event.weights.h5 nor max.weights.h5 found in {pretrained_dir}")
            print(f"Loading weights from: {weight_file}")

            pretrained_params["WEIGHT_FILE"] = weight_file

            # load pretrained model
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_fn", f"{pretrained_dir}/model/model_fn.py")
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            build_DBI_TCN_Pretrained = model_module.build_DBI_TCN_MixerOnly
            pretrained_tcn = build_DBI_TCN_Pretrained(pretrained_params["NO_TIMEPOINTS"], params=pretrained_params)
            pretrained_tcn.load_weights(weight_file)
            pretrained_tcn.trainable = False
            pretrained_tcn.compile(optimizer='adam', loss='mse')
            from model.model_fn import build_DBI_TCN_CADMixerOnly as build_DBI_TCN


        from model.model_fn import CSDLayer
        from tcn import TCN
        from tensorflow.keras.models import load_model

        # Check which weight file is most recent
        event_weights = f"{study_dir}/event.weights.h5"
        max_weights = f"{study_dir}/max.weights.h5"
        # event_weights = f"{study_dir}/event.tuned.weights.h5"
        # max_weights = f"{study_dir}/max.tuned.weights.h5"
        # event_weights = f"{study_dir}/event.mpntuned.weights.h5"
        # max_weights = f"{study_dir}/max.mpntuned.weights.h5"
        # event_weights = f"{study_dir}/event.finetune.weights.h5"
        # max_weights = f"{study_dir}/max.finetune.weights.h5"
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
            tag += 'EvF1'
            print("Using event.weights.h5 (max.weights.h5 not found)")
        elif os.path.exists(max_weights):
            weight_file = max_weights
            tag += 'MaxF1'
            print("Using max.weights.h5 (event.weights.h5 not found)")
        else:
            raise ValueError(f"Neither event.weights.h5 nor max.weights.h5 found in {study_dir}")
        print(f"Loading weights from: {weight_file}")

        # load models
        if 'CADOnly' in params['TYPE_ARCH']:
            model = build_DBI_TCN(pretrained_params["NO_TIMEPOINTS"], params=params, pretrained_tcn=pretrained_tcn)
        elif 'Patch' in params['TYPE_ARCH']:
            model = build_DBI_TCN(params=params) # Pass only the params dictionary
        else:
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

    # inference parameters
    squence_stride = 1
    # params['BATCH_SIZE'] =1# 512*4*3
    params['NO_TIMEPOINTS'] = 44
    params["BATCH_SIZE"] = 1024*4
    # pdb.set_trace()
    # from model.input_augment_weighted import rippleAI_load_dataset
    from model.input_proto_new import rippleAI_load_dataset
    # import importlib.util
    # spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/input_augment_weighted.py")
    # model_module = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(model_module)
    # rippleAI_load_dataset = model_module.rippleAI_load_dataset

    # from model.input_aug import rippleAI_load_dataset
    # from model.input_fn import rippleAI_load_dataset
    flag_numpy = False
    if flag_numpy:
        print('Using numpy')
        # LFP = np.load('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_1_FlatBrain/FlatLFP_2500.npy')
        # LFP = np.load('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_1_FlatBrain/SexyLFP_2500.npy')
        # LFP = np.load('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/Mouse3_LFPSub_2500.npy')
        # LFP = np.load('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/Mouse3_LFP_2500.npy')
        # LFP = np.load('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/Mouse3_ModelInputs_2500.npy')
        LFP = np.load('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/Mouse3_LFP_ZSig_2500.npy')

        LFP = np.transpose(LFP, (1, 0))
        # pdb.set_trace()
        peak_chind = 12 # sexy peak
        # peak_chind = 15 # sexy peak
        # peak_chind = 28 # sexy cortex
        LFP = LFP[:,peak_chind-4:peak_chind+4]
        # LFP = (LFP - np.mean(LFP, axis=0)) / np.std(LFP, axis=0)
        labels = np.zeros(LFP.shape[0])
    else:
        preproc = False if model_name=='RippleNet' else True
        if 'NAME' not in params or not params['NAME']:
            params['NAME'] = model_name
        if 'FiltL' in params['NAME']:
            val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', use_band='low', preprocess=preproc)
        elif 'FiltH' in params['NAME']:
            val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', use_band='high', preprocess=preproc)
        elif 'FiltM' in params['NAME']:
            val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', use_band='muax', preprocess=preproc)
        else:
            if 'TripletOnly' in params['TYPE_ARCH']:
                flag_online = 'Online' in params['TYPE_ARCH']
                val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', preprocess=True, process_online=flag_online)
            else:
                val_datasets, val_labels = rippleAI_load_dataset(params, preprocess=preproc)
                
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
    from tensorflow.keras.utils import timeseries_dataset_from_array
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

        if flag_numpy:
                # pdb.set_trace()
                probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal))
                # np.save('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_1_FlatBrain/probs_{0}_{1}.npy'.format(study_num, tag), probs)
                pdb.set_trace()
                np.save('/mnt/hpc/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/probs_{0}_decimate_{1}.npy'.format(study_num, tag), probs)
                # np.save('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/probs_{0}_sub_{1}.npy'.format(study_num, tag), probs)
                sys.exit(0)
    else:
        sample_length = params['NO_TIMEPOINTS']
        if 'Patch' in params['TYPE_ARCH']:
            sample_length = params['seq_length']
            params['BATCH_SIZE'] = 512
            squence_stride = 1
        elif flag_numpy:
            print('Using numpy')
            sample_length = params['NO_TIMEPOINTS']
            squence_stride = 10
            # pdb.set_trace()
        train_x = timeseries_dataset_from_array(LFP, None, sequence_length=sample_length, sequence_stride=squence_stride, batch_size=params["BATCH_SIZE"])
        windowed_signal = np.squeeze(model.predict(train_x, verbose=1))
        if flag_numpy:
            # pdb.set_trace()
            probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal))
            # np.save('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_1_FlatBrain/probs_{0}_{1}.npy'.format(study_num, tag), probs)
            # pdb.set_trace()
            # np.save('/mnt/hpc/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/probs_{0}_decimate_{1}.npy'.format(study_num, tag), probs)
            # np.save('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/probs_{0}_sub_{1}.npy'.format(study_num, tag), probs)
            # np.save('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/probs_{0}_subZ_{1}.npy'.format(study_num, tag), probs)
            np.save('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/probs_{0}_sub_LFPzSig_{1}.npy'.format(study_num, tag), probs)
            # np.save('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/probs_{0}_sub_{1}.npy'.format(study_num, tag), probs)
            sys.exit(0)

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
            elif 'Patch' in params['TYPE_ARCH']:
                # pdb.set_trace()
                probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal[:,-1]))
                horizon = np.vstack((np.zeros((sample_length-1, 8)), windowed_signal[:, :-1]))
            else:
                probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal[:,-1]))
                horizon = np.vstack((np.zeros((sample_length-1, 8)), windowed_signal[:, :-1]))
        elif model_name.find('Proto') != -1:
            probs = np.hstack((windowed_signal[0,:-1], windowed_signal[:, -1]))
        elif model_name.find('Base_') != -1:
            probs = np.hstack((np.zeros((sample_length-1, 1)).flatten(), windowed_signal))
        else:
            probs = np.hstack((windowed_signal[0,:-1], windowed_signal[:, -1]))
    # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/preds_val{0}_{1}_sf{2}.npy'.format(val_id, model_name, params['SRATE']), probs)
    np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/preds_val{0}_{1}_{3}_sf{2}.npy'.format(val_id, model_name, params['SRATE'], tag), probs)

    if  model_name == 'CNN1D':
        print('no horizon')

    elif model_name.find('Hori') != -1 or model_name.find('Dori') != -1 or model_name.find('Cori') != -1 or model_name.startswith('Tune') != -1:
        if not ('Only' in params['TYPE_ARCH']):
            # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/horis_val{0}_{1}_sf{2}.npy'.format(val_id, model_name, params['SRATE']), horizon)
            np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/horis_val{0}_{1}_{3}_sf{2}.npy'.format(val_id, model_name, params['SRATE'], tag), horizon)

    # pdb.set_trace()
    for i,th in enumerate(th_arr):
        pred_val_events = get_predictions_index(probs,th)/samp_freq
        [precision[0,i], recall[0,i], F1_val[0,i], tmpTP, tmpFN, tmpIOU] = get_performance(pred_val_events,labels,verbose=False)
        TP[0,i] = tmpTP.sum()
        FN[0,i] = tmpFN.sum()
        IOU[0,i] = np.mean(tmpIOU.sum(axis=0))
    stats = np.stack((precision, recall, F1_val, TP, FN, IOU), axis=-1)
    # np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/stats_val{0}_{1}_sf{2}.npy'.format(val_id, model_name, params['SRATE']), stats)
    np.save('/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/stats_val{0}_{1}_{3}_sf{2}.npy'.format(val_id, model_name, params['SRATE'], tag), stats)

elif mode == 'fine_tune':
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
        study_dirs = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/study_{study_num}_*')
        # study_dirs = glob.glob(f'studies_CHECK_SIGNALS/{param_dir}/study_{study_num}_*')
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
            # import importlib.util
            # spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
            # model_module = importlib.util.module_from_spec(spec)
            # spec.loader.exec_module(model_module)
            # build_DBI_TCN = model_module.build_DBI_TCN_MixerOnly
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

            # spec_inp = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/input_proto_new.py")
            # inp_module = importlib.util.module_from_spec(spec_inp)
            # spec_inp.loader.exec_module(inp_module)
            # rippleAI_load_dataset = inp_module.rippleAI_load_dataset
            from model.input_proto_new import rippleAI_load_dataset
        elif 'Patch' in params['TYPE_ARCH']:
            tf.config.run_functions_eagerly(True)
            from model.input_augment_weighted import rippleAI_load_dataset
            from model.model_fn import build_DBI_Patch as build_DBI_TCN

        elif 'CADOnly' in params['TYPE_ARCH']:
            pretrain_tag = 'params_mixerOnlyEvents2500'
            pretrain_num = 1414#958
            pretrained_dir = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{pretrain_tag}/study_{pretrain_num}_*')
            if not pretrained_dir:
                raise ValueError(f"No study directory found for study number {pretrain_num}")
            pretrained_dir = pretrained_dir[0]  # Take the first matching directory

            pretrained_params = copy.deepcopy(params)
            # Load trial info to get parameters
            with open(f"{pretrained_dir}/trial_info.json", 'r') as f:
                trial_info = json.load(f)
                pretrained_params.update(trial_info['parameters'])

            # Check which weight file is most recent
            event_weights = f"{pretrained_dir}/event.weights.h5"
            max_weights = f"{pretrained_dir}/max.weights.h5"

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
                raise ValueError(f"Neither event.weights.h5 nor max.weights.h5 found in {pretrained_dir}")
            print(f"Loading weights from: {weight_file}")

            pretrained_params["WEIGHT_FILE"] = weight_file

            # load pretrained model
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_fn", f"{pretrained_dir}/model/model_fn.py")
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            build_DBI_TCN_Pretrained = model_module.build_DBI_TCN_MixerOnly
            pretrained_tcn = build_DBI_TCN_Pretrained(pretrained_params["NO_TIMEPOINTS"], params=pretrained_params)
            pretrained_tcn.load_weights(weight_file)
            pretrained_tcn.trainable = False
            pretrained_tcn.compile(optimizer='adam', loss='mse')
            from model.model_fn import build_DBI_TCN_CADMixerOnly as build_DBI_TCN


        from model.model_fn import CSDLayer
        from tcn import TCN
        from tensorflow.keras.models import load_model
        # Load weights
        params['mode'] = 'predict'

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
            tag += 'EvF1'
            print("Using event.weights.h5 (max.weights.h5 not found)")
        elif os.path.exists(max_weights):
            weight_file = max_weights
            tag += 'MaxF1'
            print("Using max.weights.h5 (event.weights.h5 not found)")
        else:
            raise ValueError(f"Neither event.weights.h5 nor max.weights.h5 found in {study_dir}")
        print(f"Loading weights from: {weight_file}")

        if 'CADOnly' in params['TYPE_ARCH']:
            model = build_DBI_TCN(pretrained_params["NO_TIMEPOINTS"], params=params, pretrained_tcn=pretrained_tcn)
        elif 'Patch' in params['TYPE_ARCH']:
            model = build_DBI_TCN(params=params) # Pass only the params dictionary
        else:
            params['LOSS_NEGATIVES'] = 30
            # params['LOSS_WEIGHT'] = 0.25
            params['LEARNING_RATE'] = 0.00003
            params['BATCH_SIZE'] = 32
            # params['HYPER_ENTROPY'] = 3.0
            # params['LOSS_TupMPN'] = 10.0
            params['mode'] = 'fine_tune'
            model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
        model.load_weights(weight_file)

        # get sampling rate # little dangerous assumes 4 digits
        if 'Samp' in params['TYPE_LOSS']:
            sind = params['TYPE_LOSS'].find('Samp')
            params['SRATE'] = int(params['TYPE_LOSS'][sind+4:sind+8])
        else:
            params['SRATE'] = 1250

    model.summary()
    params['BATCH_SIZE'] = 32
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
            flag_online = 'Online' in params['TYPE_ARCH']
            train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=flag_online)

        else:
            train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)
    # train_size = len(list(train_dataset))
    params['RIPPLE_RATIO'] = label_ratio


    # Setup callbacks including the verifier
    # callbacks = [cb.EarlyStopping(monitor='val_f1',  # Change monitor
    #                     patience=50,
    #                     mode='max',
    #                     verbose=1,
    #                     restore_best_weights=True),
    #             cb.ModelCheckpoint(f"{study_dir}/max.mpntuned.weights.h5",
    #                         monitor='val_f1',
    #                         verbose=1,
    #                         save_best_only=True,
    #                         save_weights_only=True,
    #                         mode='max'),
    #                         cb.ModelCheckpoint(
    #                         f"{study_dir}/event.mpntuned.weights.h5",
    #                         monitor='val_event_f1',  # Change monitor
    #                         verbose=1,
    #                         save_best_only=True,
    #                         save_weights_only=True,
    #                         mode='max')]
    patience = 50
    callbacks = [cb.EarlyStopping(monitor='val_event_pr_auc',  # Change monitor
                        patience=patience,
                        mode='max',
                        verbose=1,
                        restore_best_weights=True),
        cb.ModelCheckpoint(f"{study_dir}/max.finetune.weights.h5",
                            monitor='val_latency_weighted_f1',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='max'),
                            cb.ModelCheckpoint(
                            f"{study_dir}/event.finetune.weights.h5",
                            monitor='val_event_pr_auc',  # Change monitor
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='max')
    ]    
    val_steps = dataset_params['VAL_STEPS']

    # Train and evaluate
    history = model.fit(train_dataset,
        steps_per_epoch=dataset_params['ESTIMATED_STEPS_PER_EPOCH'],
        validation_data=test_dataset,
        validation_steps=val_steps,  # Explicitly set to avoid the partial batch
        epochs=params['NO_EPOCHS'],
        callbacks=callbacks,
        verbose=1)


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
    from tensorflow.keras.utils import timeseries_dataset_from_array

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
        # study_dirs = glob.glob(f'/mnt/hpc/projects/OWVinckSWR/DL/predSWR/experiments/studies/{param_dir}/study_{study_num}_*')
        study_dirs = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/study_{study_num}_*')
        # study_dirs = glob.glob(f'studies_CHECK_SIGNALS/{param_dir}/study_{study_num}_*')
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
        elif 'CADOnly' in params['TYPE_ARCH']:
            pretrain_tag = 'params_mixerOnlyEvents2500'
            pretrain_num = 1414#958
            pretrained_dir = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{pretrain_tag}/study_{pretrain_num}_*')
            if not pretrained_dir:
                raise ValueError(f"No study directory found for study number {pretrain_num}")
            pretrained_dir = pretrained_dir[0]  # Take the first matching directory

            pretrained_params = copy.deepcopy(params)
            # Load trial info to get parameters
            with open(f"{pretrained_dir}/trial_info.json", 'r') as f:
                trial_info = json.load(f)
                pretrained_params.update(trial_info['parameters'])

            # Check which weight file is most recent
            event_weights = f"{pretrained_dir}/event.finetune.weights.h5"
            # event_weights = f"{pretrained_dir}/event.weights.h5"
            max_weights = f"{pretrained_dir}/max.weights.h5"

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
                raise ValueError(f"Neither event.weights.h5 nor max.weights.h5 found in {pretrained_dir}")
            print(f"Loading weights from: {weight_file}")

            pretrained_params["WEIGHT_FILE"] = weight_file

            # load pretrained model
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_fn", f"{pretrained_dir}/model/model_fn.py")
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            build_DBI_TCN_Pretrained = model_module.build_DBI_TCN_MixerOnly
            pretrained_tcn = build_DBI_TCN_Pretrained(pretrained_params["NO_TIMEPOINTS"], params=pretrained_params)
            pretrained_tcn.load_weights(weight_file)
            pretrained_tcn.trainable = False
            pretrained_tcn.compile(optimizer='adam', loss='mse')


            spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            build_DBI_TCN = model_module.build_DBI_TCN_CADMixerOnly

        # weight_file = f"{study_dir}/robust.weights.h5"

        if 'CADOnly' in params['TYPE_ARCH']:
            from model.model_fn import CSDLayer
            from tcn import TCN
            from tensorflow.keras.models import load_model
            # Load weights
            params['mode'] = 'predict'

            # weight_file = f"{study_dir}/last.weights.h5"
            if 'Events' in tag:
                weight_file = f"{study_dir}/event.weights.h5"
                tag += 'EvF1'
            else:
                weight_file = f"{study_dir}/max.weights.h5"
                tag += 'MaxF1'

            # weight_file = f"{study_dir}/robust.weights.h5"
            print(f"Loading weights from: {weight_file}")
            if 'CADOnly' in params['TYPE_ARCH']:
                model = build_DBI_TCN(pretrained_params["NO_TIMEPOINTS"], params=params, pretrained_tcn=pretrained_tcn)
            elif 'Patch' in params['TYPE_ARCH']:
                model = build_DBI_TCN(params=params) # Pass only the params dictionary
            else:
                model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
            model.load_weights(weight_file)
        else:
            from model.model_fn import CSDLayer
            from tcn import TCN
            from tensorflow.keras.models import load_model
            # Load weights
            params['mode'] = 'predict'
            # weight_file = f"{study_dir}/last.weights.h5"
            if 'Events' in tag:
                weight_file = f"{study_dir}/event.weights.h5"
                tag += 'EvF1'
            else:
                weight_file = f"{study_dir}/max.weights.h5"
                tag += 'MaxF1'
            params['mode'] = 'predict'
            weight_file = f"{study_dir}/event.weights.h5"

            # weight_file = f"{study_dir}/max.mpntuned.weights.h5"
            # max_weights = f"{study_dir}/max.mpntuned.weights.h5"
            # # if 'MixerOnly' in params['TYPE_ARCH']:
            # weight_file = f"{study_dir}/max.weights.h5"
            # tag += 'MaxF1'
            # # else:
            #     weight_file = f"{study_dir}/event.weights.h5"
            #     tag += 'EvF1'

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

    model_converter = 'ONNX' # 'TF' 'TFLite'
    if model_converter == 'ONNX':

        import tf2onnx
        # Convert the model to ONNX format
        # spec = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="x")]
        # spec = [tf.TensorSpec([1,92,8], model.inputs[0].dtype, name="x")]
        # spec = [tf.TensorSpec([2,44,8], model.inputs[0].dtype, name="x")]
        spec = [tf.TensorSpec([1,44,8], model.inputs[0].dtype, name="x")]
        # pdb.set_trace()
        output_path = f"./frozen_models/{model_name}/model.onnx"
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path, opset=15)
        print(f"Model saved to {output_path}")

    elif model_converter == 'TFLite':
        # Convert the model to a TensorFlow Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # tflite_model = converter.convert()
        # # Save the model to disk
        # with open('model.tflite', 'wb') as f:
        #     f.write(tflite_model)
    elif model_converter == 'TF':
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        full_model = tf.function(lambda x: model(x))
        # pdb.set_trace()
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


        # os.mkdir('./frozen_models/{}'.format(model_name))
        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir="./frozen_models/{}".format(model_name),
                        name="simple_frozen_graph.pb",
                        as_text=False)
        
elif mode=='exportBranch':


    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers

    # modelname
    model = args.model[0]
    model_name = model
    import importlib

    # necessary !!!
    # tf.compat.v1.disable_eager_execution()


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
        # study_dirs = glob.glob(f'/mnt/hpc/projects/OWVinckSWR/DL/predSWR/experiments/studies/{param_dir}/study_{study_num}_*')
        # study_dirs = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/study_{study_num}_*')

        study_dirs = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/study_{study_num}_*')
        # study_dirs = glob.glob(f'studies_CHECK_SIGNALS/{param_dir}/study_{study_num}_*')
        if not study_dirs:
            raise ValueError(f"No study directory found for study number {study_num}")
        study_dir = study_dirs[0]  # Take the first matching directory

        # Load trial info to get parameters
        with open(f"{study_dir}/trial_info.json", 'r') as f:
            trial_info = json.load(f)
            params.update(trial_info['parameters'])


        if 'TripletOnly' in params['TYPE_ARCH']:
            # from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            build_DBI_TCN = model_module.build_DBI_TCN_TripletOnly
    
        from model.model_fn import CSDLayer
        from tcn import TCN
        from tensorflow.keras.models import load_model
        
        # Load weights
        params['mode'] = 'predict'
        weight_file = f"{study_dir}/event.finetune.weights.h5"
        print(f"Loading weights from: {weight_file}")
        model1 = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
        model1.load_weights(weight_file)
    model1.summary()
    
    
    # model 2
    param_dir_2 = 'params_tripletOnlyShort2500'
    study_num_2 = 295
    study_dirs = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir_2}/study_{study_num_2}_*')
    # study_dirs = glob.glob(f'studies_CHECK_SIGNALS/{param_dir}/study_{study_num}_*')
    if not study_dirs:
        raise ValueError(f"No study directory found for study number {study_num}")
    study_dir = study_dirs[0]  # Take the first matching directory

    # Load trial info to get parameters
    with open(f"{study_dir}/trial_info.json", 'r') as f:
        trial_info = json.load(f)
        params.update(trial_info['parameters'])


    if 'TripletOnly' in params['TYPE_ARCH']:
        # from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        build_DBI_TCN = model_module.build_DBI_TCN_TripletOnly

    from model.model_fn import CSDLayer
    from tcn import TCN
    from tensorflow.keras.models import load_model
    
    # Load weights
    params['mode'] = 'predict'
    # weight_file = f"{study_dir}/event.finetune.weights.h5"
    weight_file = f"{study_dir}/max.tuned.weights.h5"
    print(f"Loading weights from: {weight_file}")
    model2 = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model2.load_weights(weight_file)
    model2.summary()    

    T = 44
    C = 8
    inp = Input(shape=(T, C), dtype=tf.float32, name='x')
    out1 = model1(inp)
    out2 = model2(inp)
    # if you want a tuple of outputs instead of concat, just do outputs=[out1,out2]
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.models import Model
    merged = Concatenate(axis=-1, name='combined_output')([out1, out2])
    model = Model(inputs=inp, outputs=merged, name='TwoBranchModel')
    model.summary()
    
    
    import tf2onnx
    # Convert the model to ONNX format
    # spec = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="x")]
    # spec = [tf.TensorSpec([1,92,8], model.inputs[0].dtype, name="x")]
    # spec = [tf.TensorSpec([2,44,8], model.inputs[0].dtype, name="x")]
    spec = [tf.TensorSpec([1,44,8], model.inputs[0].dtype, name="x")]
    # pdb.set_trace()
    output_path = f"./frozen_models/{model_name}/model_combined.onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path, opset=15)
    print(f"Model saved to {output_path}")

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
        from tensorflow.keras.models import load_model
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
    from tensorflow.keras.utils import timeseries_dataset_from_array

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
    directions=["maximize", "minimize"],  # Maximize F1, minimize FP
    load_if_exists=True,
    sampler=optuna.samplers.GPSampler())

    # sampler=NSGAIISampler(
    #     # population_size=30,  # Number of parallel solutions evolved
    #     population_size=16,  # Number of parallel solutions evolved
    #     crossover_prob=0.9,  # Probability of crossover between solutions
    #     mutation_prob=0.3,   # Probability of mutating a solution
    #     seed=42
    # ),
    # pruner=optuna.pruners.PatientPruner(
    #     optuna.pruners.MedianPruner(
    #         n_startup_trials=20,
    #         n_warmup_steps=50,
    #         interval_steps=10,
    #     ),
    # patience=3,
    # min_delta=0.0
    # )
# )

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
        # from model.study_objectives import objective_triplet as objective
        objective = objective_triplet
    elif 'MixerOnly'.lower() in tag.lower():
        # from model.study_objectives import objective_only as objective
        objective = objective_only
    elif 'CADOnly'.lower() in tag.lower():
        # from model.study_objectives import objective_only_30k as objective
        objective = objective_only_30k
    elif 'Patch'.lower() in tag.lower():
        # from model.study_objectives import objective_patch as objective
        objective = objective_patch
    else:
        from model.study_objectives import objective_only as objective

    study.optimize(
        objective,
        n_trials=48,
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
    N = 80
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
# ...existing code...
elif mode == 'tune_viz_multi':
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json
    import os
    import optuna
    import numpy as np
    import sys

    tag = args.tag[0]
    param_dir = f'params_{tag}'
    viz_dir = f"studies/{param_dir}/visualizations"
    param_impact_dir = os.path.join(viz_dir, "param_impact")

    # Create visualization directories
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(param_impact_dir, exist_ok=True)

    # Load study for visualization
    try:
        storage_url = f"sqlite:///studies/{param_dir}/{param_dir}.db"
        study = optuna.load_study(
            study_name=param_dir,
            storage=storage_url,
        )
        print(f"Loaded study '{param_dir}' from {storage_url}")
    except Exception as e:
        print(f"Error loading study '{param_dir}': {e}")
        sys.exit(1) # Exit if study cannot be loaded

    # --- Standard Optuna Plots ---
    print("Generating standard Optuna plots...")
    try:
        # Plot optimization history for F1 Score (Objective 0)
        fig_hist_f1 = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[0], target_name="F1 Score")
        fig_hist_f1.write_html(os.path.join(viz_dir, "optimization_history_f1.html"))

        # Plot optimization history for Latency (Objective 1)
        fig_hist_latency = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[1], target_name="Latency")
        fig_hist_latency.write_html(os.path.join(viz_dir, "optimization_history_latency.html"))

        # Plot Pareto Front
        fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1 Score", "Latency"])
        fig_pareto.write_html(os.path.join(viz_dir, "pareto_front.html"))

        # Plot parameter importance for F1 Score
        fig_imp_f1 = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name="F1 Score")
        fig_imp_f1.write_html(os.path.join(viz_dir, "param_importances_f1.html"))

        # Plot parameter importance for Latency
        fig_imp_latency = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[1], target_name="Latency")
        fig_imp_latency.write_html(os.path.join(viz_dir, "param_importances_latency.html"))
        print("Standard plots generated.")
    except Exception as e:
        print(f"Warning: Error generating standard Optuna plots: {e}")

    # --- Data Preparation ---
    print("Preparing data for analysis...")
    all_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))

    if not all_trials:
        print("No completed trials found in the study. Exiting visualization.")
        sys.exit(0)

    all_trial_data = []
    for trial in all_trials:
        if trial.values is not None and len(trial.values) == 2:
            # Ensure values are floats, handle potential None or non-numeric values gracefully
            try:
                f1_score = float(trial.values[0]) if trial.values[0] is not None else np.nan
                latency = float(trial.values[1]) if trial.values[1] is not None else np.nan
            except (ValueError, TypeError):
                f1_score = np.nan
                latency = np.nan

            if not (np.isnan(f1_score) or np.isnan(latency)):
                 all_trial_data.append({
                    "trial_number": trial.number,
                    "f1_score": f1_score,
                    "latency": latency,
                    **trial.params
                })

    if not all_trial_data:
        print("No completed trials with valid F1 score and Latency found. Exiting visualization.")
        sys.exit(0)

    all_df = pd.DataFrame(all_trial_data)
    all_df.to_csv(os.path.join(viz_dir, "all_completed_trials_data.csv"), index=False)
    print(f"Saved data for {len(all_df)} completed trials.")

    # --- Calculate Quantile Ranges for Plotting ---
    f1_q05, f1_q95 = np.nanquantile(all_df['f1_score'], [0.05, 0.95]) if not all_df['f1_score'].isnull().all() else (0, 1)
    lat_q05, lat_q95 = np.nanquantile(all_df['latency'], [0.05, 0.95]) if not all_df['latency'].isnull().all() else (0, 1)
    # Add a small margin
    f1_margin = (f1_q95 - f1_q05) * 0.05
    lat_margin = (lat_q95 - lat_q05) * 0.05
    f1_ylim = (max(0, f1_q05 - f1_margin), min(1, f1_q95 + f1_margin))
    lat_ylim = (max(0, lat_q05 - lat_margin), lat_q95 + lat_margin)
    # Ensure min < max
    if f1_ylim[0] >= f1_ylim[1]: f1_ylim = (f1_ylim[0] - 0.1, f1_ylim[1] + 0.1)
    if lat_ylim[0] >= lat_ylim[1]: lat_ylim = (lat_ylim[0] - 0.1, lat_ylim[1] + 0.1)
    print(f"Using F1 Score Y-axis range: {f1_ylim}")
    print(f"Using Latency Y-axis range: {lat_ylim}")


    # --- Pareto Optimal Trials Analysis ---
    print("Analyzing Pareto optimal trials...")
    pareto_trials = study.best_trials # These are the trials on the Pareto front

    if not pareto_trials:
        print("No Pareto optimal trials found (study.best_trials is empty).")
        pareto_df = pd.DataFrame() # Create empty dataframe
    else:
        pareto_trial_numbers = [t.number for t in pareto_trials]
        pareto_df = all_df[all_df['trial_number'].isin(pareto_trial_numbers)].copy()

        # Optional: Calculate a combined score for ranking Pareto trials (adjust weighting as needed)
        # Normalization might be useful if scales differ greatly
        # For simplicity, using the previous formula, but apply only to Pareto front trials
        pareto_df['combined_score'] = (pareto_df['f1_score'] + (1 - pareto_df['latency'])) / 2
        pareto_df = pareto_df.sort_values('combined_score', ascending=False)

        # Save Pareto optimal trials data
        pareto_df.to_csv(os.path.join(viz_dir, "pareto_optimal_trials.csv"), index=False)

        # Generate HTML report for Pareto optimal trials
        html_content_pareto = f"""
        <html><head><title>Pareto Optimal Trials</title>
        <style> table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid black; padding: 8px; text-align: left; }} tr:nth-child(even) {{ background-color: #f2f2f2; }} th {{ background-color: #007bff; color: white; }} </style>
        </head><body>
        <h2>Pareto Optimal Trials ({len(pareto_df)} trials)</h2>
        <p>These trials represent the best trade-offs found between maximizing F1 Score and minimizing Latency.</p>
        <p>Sorted by combined score = (F1 Score + (1 - Latency)) / 2</p>
        {pareto_df.to_html(index=False)}
        <p><a href="hyperparameter_impact_analysis.html">View Hyperparameter Impact Analysis</a></p>
        </body></html>
        """
        with open(os.path.join(viz_dir, "pareto_optimal_trials.html"), "w") as f:
            f.write(html_content_pareto)

        print(f"Found {len(pareto_df)} Pareto optimal trials.")
        print(f"Pareto optimal trials saved to {os.path.join(viz_dir, 'pareto_optimal_trials.csv')} and .html")
        print("\nTop 10 Pareto Optimal Trials (ranked by combined_score):")
        print(pareto_df[['trial_number', 'f1_score', 'latency', 'combined_score']].head(10))

    # --- Hyperparameter Impact Analysis ---
    print("Generating hyperparameter impact plots...")
    hyperparams = [p for p in all_df.columns if p not in ['trial_number', 'f1_score', 'fp_per_min', 'combined_score']]
    impact_plot_files = []

    # Function to check if a parameter should use log scale
    def should_use_log_scale(series):
        if not pd.api.types.is_numeric_dtype(series): return False
        if series.min() <= 0: return False
        # Check if max is significantly larger than min (avoid division by zero)
        min_val = series.min()
        max_val = series.max()
        if min_val < 1e-9: # If min is very close to zero, don't use log scale
             return False
        return max_val / min_val > 100 # Use log if range spans > 2 orders of magnitude

    for param in hyperparams:
        if param not in all_df.columns or all_df[param].isnull().all() or all_df[param].nunique() <= 1:
            print(f"Skipping parameter '{param}' due to missing data or single value.")
            continue

        is_numeric = pd.api.types.is_numeric_dtype(all_df[param])
        plot_filename = os.path.join(param_impact_dir, f"{param}_impact.png")
        impact_plot_files.append({"name": param, "path": f"param_impact/{param}_impact.png"})

        try:
            fig, ax1 = plt.subplots(figsize=(12, 7))
            use_log = is_numeric and should_use_log_scale(all_df[param])

            # Plot all completed trials as background
            if is_numeric:
                sns.scatterplot(data=all_df, x=param, y='f1_score', ax=ax1, color='lightblue', alpha=0.3, label='_nolegend_')
            else: # Categorical: Use stripplot for background distribution
                 sns.stripplot(data=all_df, x=param, y='f1_score', ax=ax1, color='lightblue', alpha=0.3, order=sorted(all_df[param].unique()), label='_nolegend_')


            # Highlight Pareto optimal trials for F1 Score
            if not pareto_df.empty:
                if is_numeric:
                    sns.scatterplot(data=pareto_df, x=param, y='f1_score', ax=ax1, color='blue', s=100, marker='o', label='Pareto F1 Score', alpha=0.8)
                else: # Categorical: Overlay points
                    sns.stripplot(data=pareto_df, x=param, y='f1_score', ax=ax1, color='blue', s=8, marker='o', order=sorted(all_df[param].unique()), label='Pareto F1 Score', alpha=0.8, jitter=False)


            ax1.set_ylabel('F1 Score', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(f1_ylim) # Set Y-axis limits for F1 score
            if use_log:
                ax1.set_xscale('log')
                ax1.set_xlabel(f"{param} (log scale)")
            else:
                ax1.set_xlabel(param)
            if not is_numeric: # Rotate labels for categorical
                 plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")


            # Create second y-axis for Latency
            ax2 = ax1.twinx()
            if is_numeric:
                sns.scatterplot(data=all_df, x=param, y='latency', ax=ax2, color='lightcoral', alpha=0.3, label='_nolegend_')
            else: # Categorical
                 sns.stripplot(data=all_df, x=param, y='latency', ax=ax2, color='lightcoral', alpha=0.3, order=sorted(all_df[param].unique()), label='_nolegend_')

            if not pareto_df.empty:
                if is_numeric:
                    sns.scatterplot(data=pareto_df, x=param, y='latency', ax=ax2, color='red', s=100, marker='X', label='Pareto Latency', alpha=0.8)
                else: # Categorical
                    sns.stripplot(data=pareto_df, x=param, y='latency', ax=ax2, color='red', s=8, marker='X', order=sorted(all_df[param].unique()), label='Pareto Latency', alpha=0.8, jitter=False)

            ax2.set_ylabel('Latency', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(lat_ylim) # Set Y-axis limits for Latency

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # Filter out '_nolegend_' entries before combining
            valid_lines1 = [l for l, lbl in zip(lines1, labels1) if lbl != '_nolegend_']
            valid_labels1 = [lbl for lbl in labels1 if lbl != '_nolegend_']
            valid_lines2 = [l for l, lbl in zip(lines2, labels2) if lbl != '_nolegend_']
            valid_labels2 = [lbl for lbl in labels2 if lbl != '_nolegend_']
            ax2.legend(valid_lines1 + valid_lines2, valid_labels1 + valid_labels2, loc='best')
            ax1.get_legend().remove() # Remove ax1 legend as it's combined in ax2

            plt.title(f'Impact of {param} on F1 Score and Latency (Pareto Optimal Highlighted)')
            ax1.grid(True, axis='x', linestyle='--', alpha=0.6)
            fig.tight_layout()
            plt.savefig(plot_filename)
            plt.close(fig)

        except Exception as e:
            print(f"Error plotting impact for parameter '{param}': {e}")
            plt.close(fig) # Ensure figure is closed on error
            # Remove potentially corrupted file entry
            impact_plot_files = [f for f in impact_plot_files if f["name"] != param]


    # --- Correlation Heatmap ---
    print("Generating correlation heatmap...")
    numeric_df = all_df[hyperparams].select_dtypes(include=np.number)
    if not numeric_df.empty:
        try:
            correlation_matrix = numeric_df.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title('Hyperparameter Correlation Heatmap (Numeric Only)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            heatmap_filename = os.path.join(param_impact_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_filename)
            plt.close()
            print(f"Correlation heatmap saved to {heatmap_filename}")
            heatmap_rel_path = "param_impact/correlation_heatmap.png"
        except Exception as e:
            print(f"Error generating correlation heatmap: {e}")
            plt.close()
            heatmap_rel_path = None
    else:
        print("No numeric hyperparameters found for correlation heatmap.")
        heatmap_rel_path = None


    # --- Generate Summary HTML for Impact Plots ---
    print("Generating summary HTML for impact plots...")
    impact_html = """
    <!DOCTYPE html><html><head><title>Hyperparameter Impact Analysis</title>
    <style> body {{ font-family: sans-serif; margin: 20px; }} .gallery {{ display: flex; flex-wrap: wrap; gap: 20px; }}
           .param-card {{ border: 1px solid #ccc; padding: 15px; width: 400px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }}
           .param-card img {{ max-width: 100%; height: auto; }} h1, h2 {{ color: #333; }} </style>
    </head><body>
    <h1>Hyperparameter Impact on F1 Score and Latency</h1>
    <p>Plots show all completed trials (light points) and Pareto optimal trials (dark points/markers).</p>
    <p>Y-axis ranges are focused on the 5th-95th percentile of all completed trials.</p>
    <p><a href="../pareto_optimal_trials.html">View Pareto Optimal Trial Details</a></p>"""

    if heatmap_rel_path:
        impact_html += f"""
        <h2>Parameter Correlation Analysis (Numeric)</h2>
        <div class="param-card" style="width: 600px;">
            <a href="{heatmap_rel_path}" target="_blank"><img src="{heatmap_rel_path}" alt="Correlation Heatmap"></a>
        </div>"""

    impact_html += "<h2>Individual Parameter Analysis</h2><div class='gallery'>"

    for plot_info in impact_plot_files:
        impact_html += f"""
        <div class="param-card">
            <h3>{plot_info['name']}</h3>
            <a href="{plot_info['path']}" target="_blank">
                <img src="{plot_info['path']}" alt="Impact of {plot_info['name']}">
            </a>
        </div>"""

    impact_html += "</div></body></html>"

    with open(os.path.join(viz_dir, "hyperparameter_impact_analysis.html"), "w") as f:
        f.write(impact_html)
    print(f"Hyperparameter impact analysis summary saved to {os.path.join(viz_dir, 'hyperparameter_impact_analysis.html')}")


    # --- Study Statistics ---
    print("Saving study statistics...")
    stats = {
        "study_name": study.study_name,
        "n_total_trials": len(study.trials), # Includes non-completed trials
        "n_completed_trials": len(all_trials),
        "n_valid_data_trials": len(all_df), # Trials with valid F1 and Latency
        "n_pareto_trials": len(pareto_trials),
        "n_pruned_trials": len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])),
        "n_failed_trials": len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.FAIL]))
    }
    with open(os.path.join(viz_dir, "study_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    print("Study statistics saved.")
    print("\nStudy Statistics Summary:")
    print(json.dumps(stats, indent=2))

    print(f"\nVisualization generation complete. Results are in: {viz_dir}")

elif mode == 'tune_viz_multi_v2':
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json
    import os
    import optuna
    import numpy as np
    import sys

    tag = args.tag[0]
    param_dir = f'params_{tag}'
    viz_dir = f"studies/{param_dir}/visualizations"
    param_impact_dir = os.path.join(viz_dir, "param_impact")

    # Create visualization directories
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(param_impact_dir, exist_ok=True)

    # Load study for visualization
    try:
        storage_url = f"sqlite:///studies/{param_dir}/{param_dir}.db"
        study = optuna.load_study(
            study_name=param_dir,
            storage=storage_url,
        )
        print(f"Loaded study '{param_dir}' from {storage_url}")
    except Exception as e:
        print(f"Error loading study '{param_dir}': {e}")
        sys.exit(1) # Exit if study cannot be loaded

    # --- Standard Optuna Plots ---
    print("Generating standard Optuna plots...")
    try:
        # Plot optimization history for F1 Score (Objective 0)
        fig_hist_f1 = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[0] if t.values and len(t.values) > 0 else float('nan'), target_name="F1 Score")
        fig_hist_f1.write_html(os.path.join(viz_dir, "optimization_history_f1.html"))

        # Plot optimization history for FP Rate per Minute (Objective 1)
        fig_hist_fp = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[1] if t.values and len(t.values) > 1 else float('nan'), target_name="FP Rate per Minute")
        fig_hist_fp.write_html(os.path.join(viz_dir, "optimization_history_fp_per_min.html"))

        # Plot Pareto Front
        fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1 Score", "FP Rate per Minute"])
        fig_pareto.write_html(os.path.join(viz_dir, "pareto_front.html"))

        # Plot parameter importance for F1 Score
        fig_imp_f1 = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0] if t.values and len(t.values) > 0 else float('nan'), target_name="F1 Score")
        fig_imp_f1.write_html(os.path.join(viz_dir, "param_importances_f1.html"))

        # Plot parameter importance for FP Rate per Minute
        fig_imp_fp = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[1] if t.values and len(t.values) > 1 else float('nan'), target_name="FP Rate per Minute")
        fig_imp_fp.write_html(os.path.join(viz_dir, "param_importances_fp_per_min.html"))
        print("Standard plots generated.")
    except Exception as e:
        print(f"Warning: Error generating standard Optuna plots: {e}")

    # --- Data Preparation ---
    print("Preparing data for analysis...")
    all_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))

    if not all_trials:
        print("No completed trials found in the study. Exiting visualization.")
        sys.exit(0)

    all_trial_data = []
    for trial in all_trials:
        if trial.values is not None and len(trial.values) == 2:
            try:
                f1_score = float(trial.values[0]) if trial.values[0] is not None else np.nan
                fp_per_min = float(trial.values[1]) if trial.values[1] is not None else np.nan
            except (ValueError, TypeError):
                f1_score = np.nan
                fp_per_min = np.nan

            if not (np.isnan(f1_score) or np.isnan(fp_per_min)):
                 all_trial_data.append({
                    "trial_number": trial.number,
                    "f1_score": f1_score,
                    "fp_per_min": fp_per_min,
                    **trial.params
                })

    if not all_trial_data:
        print("No completed trials with valid F1 score and FP Rate per Minute found. Exiting visualization.")
        sys.exit(0)

    all_df = pd.DataFrame(all_trial_data)
    all_df.to_csv(os.path.join(viz_dir, "all_completed_trials_data.csv"), index=False)
    print(f"Saved data for {len(all_df)} completed trials.")

    # --- Calculate Quantile Ranges for Plotting ---
    f1_q05, f1_q95 = np.nanquantile(all_df['f1_score'], [0.05, 0.95]) if not all_df['f1_score'].isnull().all() else (0, 1)
    fp_q05, fp_q95 = np.nanquantile(all_df['fp_per_min'], [0.05, 0.95]) if not all_df['fp_per_min'].isnull().all() else (0, 1)
    f1_margin = (f1_q95 - f1_q05) * 0.05 if (f1_q95 - f1_q05) > 0 else 0.05
    fp_margin = (fp_q95 - fp_q05) * 0.05 if (fp_q95 - fp_q05) > 0 else 0.05
    f1_ylim = (max(0, f1_q05 - f1_margin), min(1, f1_q95 + f1_margin))
    fp_ylim = (max(0, fp_q05 - fp_margin), fp_q95 + fp_margin)
    if f1_ylim[0] >= f1_ylim[1]: f1_ylim = (max(0, f1_ylim[0] - 0.1), min(1, f1_ylim[1] + 0.1))
    if fp_ylim[0] >= fp_ylim[1]: fp_ylim = (max(0, fp_ylim[0] - 0.1), min(1, fp_ylim[1] + 0.1))
    print(f"Using F1 Score Y-axis range: {f1_ylim}")
    print(f"Using FP Rate per Minute Y-axis range: {fp_ylim}")
    # NOTE: fp_per_min may be overcounted due to stride/window overlap. Each event can be counted multiple times if it spans multiple windows. No deduplication is performed here.


    # --- Pareto Optimal Trials Analysis ---
    print("Analyzing Pareto optimal trials...")
    pareto_trials_optuna_objects = study.best_trials

    if not pareto_trials_optuna_objects:
        print("No Pareto optimal trials found (study.best_trials is empty).")
        pareto_df = pd.DataFrame()
    else:
        pareto_trial_numbers = [t.number for t in pareto_trials_optuna_objects]
        pareto_df = all_df[all_df['trial_number'].isin(pareto_trial_numbers)].copy()

        if not pareto_df.empty:
            pareto_df['combined_score'] = (pareto_df['f1_score'] + (1 - pareto_df['fp_per_min'])) / 2 # Lower fp_per_min is better
            pareto_df = pareto_df.sort_values('combined_score', ascending=False)
            pareto_df.to_csv(os.path.join(viz_dir, "pareto_optimal_trials.csv"), index=False)

            html_content_pareto = f"""
            <html><head><title>Pareto Optimal Trials: {study.study_name}</title>
            <style> table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid black; padding: 8px; text-align: left; }} tr:nth-child(even) {{ background-color: #f2f2f2; }} th {{ background-color: #007bff; color: white; }} </style>
            </head><body>
            <h2>Pareto Optimal Trials ({len(pareto_df)} trials)</h2>
            <p>These trials represent the best trade-offs found between maximizing F1 Score and minimizing FP Rate per Minute.</p>
            <p>Sorted by a simple combined score = (F1 Score + (1 - FP/min)) / 2. See combined_scores.html for robust variants.</p>
            {pareto_df.to_html(index=False)}
            </body></html>
            """
            with open(os.path.join(viz_dir, "pareto_optimal_trials.html"), "w") as f:
                f.write(html_content_pareto)
            print(f"Found {len(pareto_df)} Pareto optimal trials.")
            print(f"Pareto optimal trials saved to {os.path.join(viz_dir, 'pareto_optimal_trials.csv')} and .html")
            if not pareto_df.empty:
                 print("\nTop 10 Pareto Optimal Trials (ranked by combined_score):")
                 print(pareto_df[['trial_number', 'f1_score', 'fp_per_min', 'combined_score']].head(10))
        else:
            print("Pareto DataFrame is empty after filtering.")
            # Create an empty pareto_optimal_trials.html if no pareto trials
            with open(os.path.join(viz_dir, "pareto_optimal_trials.html"), "w") as f:
                f.write("<html><body><h2>No Pareto Optimal Trials Found</h2></body></html>")


    N_TOP_PARETO = min(10, len(pareto_df)) if not pareto_df.empty else 0
    top_pareto_df = pareto_df.head(N_TOP_PARETO) if not pareto_df.empty else pd.DataFrame()

    # --- Combined Score Strategies and Rankings ---
    print("Computing combined score strategies and consensus ranking...")
    df = all_df.copy()
    # Robust normalization for fp_per_min
    fp_q05, fp_q50, fp_q95 = np.nanquantile(df['fp_per_min'], [0.05, 0.50, 0.95]) if not df['fp_per_min'].isnull().all() else (0.0, 0.0, 1.0)
    # Min-max on 5-95% range with clipping
    denom = max(1e-9, fp_q95 - fp_q05)
    df['fp_norm_minmax'] = np.clip((df['fp_per_min'] - fp_q05) / denom, 0, 1)
    df['fp_score_minmax'] = 1 - df['fp_norm_minmax']
    # Rank-based percentiles
    n = len(df)
    if n > 1:
        df['f1_rank_score'] = (df['f1_score'].rank(method='min', ascending=True) - 1) / (n - 1)
        fp_rank_raw = (df['fp_per_min'].rank(method='min', ascending=True) - 1) / (n - 1)
        df['fp_rank_score'] = 1 - fp_rank_raw
    else:
        df['f1_rank_score'] = 0.5
        df['fp_rank_score'] = 0.5
    # Logistic transform for FP (lower is better). Midpoint at median.
    k = max(1e-9, fp_q50)
    df['fp_score_logistic'] = 1.0 / (1.0 + (df['fp_per_min'] / k))
    # Approximate deduplication based on window and stride; defaults: window=64 samples, stride=8 samples
    default_window = 64
    default_stride = 8
    try:
        window_samples = int(np.nanmedian(df['NO_TIMEPOINTS'])) if 'NO_TIMEPOINTS' in df.columns else default_window
    except Exception:
        window_samples = default_window
    # Try a few possible stride param names; fallback to default
    if 'VAL_STRIDE' in df.columns:
        try:
            approx_stride = int(np.nanmedian(df['VAL_STRIDE']))
        except Exception:
            approx_stride = default_stride
    elif 'SEQ_STRIDE' in df.columns:
        try:
            approx_stride = int(np.nanmedian(df['SEQ_STRIDE']))
        except Exception:
            approx_stride = default_stride
    else:
        approx_stride = default_stride
    dup_factor = max(1, int(round(window_samples / max(1, approx_stride))))
    df['fp_per_min_adj'] = df['fp_per_min'] / dup_factor
    # Recompute minmax/logistic on adjusted FP
    fp_adj_q05, fp_adj_q50, fp_adj_q95 = np.nanquantile(df['fp_per_min_adj'], [0.05, 0.50, 0.95]) if not df['fp_per_min_adj'].isnull().all() else (0.0, 0.0, 1.0)
    denom_adj = max(1e-9, fp_adj_q95 - fp_adj_q05)
    df['fp_adj_norm_minmax'] = np.clip((df['fp_per_min_adj'] - fp_adj_q05) / denom_adj, 0, 1)
    df['fp_adj_score_minmax'] = 1 - df['fp_adj_norm_minmax']
    k_adj = max(1e-9, fp_adj_q50)
    df['fp_adj_score_logistic'] = 1.0 / (1.0 + (df['fp_per_min_adj'] / k_adj))
    # Combined metrics
    eps = 1e-9
    df['combined_avg_minmax'] = (df['f1_score'] + df['fp_score_minmax']) / 2.0
    df['combined_hmean_minmax'] = 2 * df['f1_score'] * df['fp_score_minmax'] / np.clip(df['f1_score'] + df['fp_score_minmax'], eps, None)
    df['combined_avg_logistic'] = (df['f1_score'] + df['fp_score_logistic']) / 2.0
    df['combined_rank_avg'] = (df['f1_rank_score'] + df['fp_rank_score']) / 2.0
    # Using adjusted FP
    df['combined_adj_avg_minmax'] = (df['f1_score'] + df['fp_adj_score_minmax']) / 2.0
    df['combined_adj_hmean_minmax'] = 2 * df['f1_score'] * df['fp_adj_score_minmax'] / np.clip(df['f1_score'] + df['fp_adj_score_minmax'], eps, None)
    df['combined_adj_avg_logistic'] = (df['f1_score'] + df['fp_adj_score_logistic']) / 2.0

    # Rank each combined metric (higher is better)
    combined_cols = [
        'combined_avg_minmax','combined_hmean_minmax','combined_avg_logistic','combined_rank_avg',
        'combined_adj_avg_minmax','combined_adj_hmean_minmax','combined_adj_avg_logistic'
    ]
    for c in combined_cols:
        df[f'rank_{c}'] = df[c].rank(method='min', ascending=False)
    rank_cols = [f'rank_{c}' for c in combined_cols]
    df['consensus_rank_sum'] = df[rank_cols].sum(axis=1)
    df['consensus_rank'] = df['consensus_rank_sum'].rank(method='min', ascending=True)
    # Save outputs
    df_path = os.path.join(viz_dir, 'all_trials_with_scores.csv')
    df.to_csv(df_path, index=False)
    top_consensus = df.sort_values(['consensus_rank','trial_number']).head(20)
    top_consensus_path = os.path.join(viz_dir, 'top_consensus_trials.csv')
    top_consensus.to_csv(top_consensus_path, index=False)
    for c in combined_cols:
        out = df.sort_values(c, ascending=False).head(20)
        out.to_csv(os.path.join(viz_dir, f'top20_{c}.csv'), index=False)

    # Visualization: Overlay scatter with combined metrics and Pareto
    try:
        plt.figure(figsize=(10, 7))
        # All trials as background
        plt.scatter(df['fp_per_min'], df['f1_score'], c='lightgray', s=18, label='_nolegend_')

        # Pareto front trials
        pareto_nums = [t.number for t in pareto_trials_optuna_objects] if pareto_trials_optuna_objects else []
        df_pareto = df[df['trial_number'].isin(pareto_nums)]
        if not df_pareto.empty:
            plt.scatter(df_pareto['fp_per_min'], df_pareto['f1_score'], facecolors='none', edgecolors='black', s=60, linewidths=1.2, label='Optuna Pareto')

        # Select a few representative combined metrics to highlight
        highlight_metrics = [
            'combined_avg_minmax',
            'combined_hmean_minmax',
            'combined_adj_hmean_minmax',
            'combined_rank_avg'
        ]
        colors = {
            'combined_avg_minmax': 'tab:blue',
            'combined_hmean_minmax': 'tab:orange',
            'combined_adj_hmean_minmax': 'tab:green',
            'combined_rank_avg': 'tab:red'
        }
        markers = {
            'combined_avg_minmax': 'o',
            'combined_hmean_minmax': 's',
            'combined_adj_hmean_minmax': 'D',
            'combined_rank_avg': '^'
        }
        top_k = 20
        for m in highlight_metrics:
            if m in df.columns:
                dtop = df.nlargest(top_k, m)
                plt.scatter(dtop['fp_per_min'], dtop['f1_score'], c=colors[m], marker=markers[m], s=45, edgecolors='white', linewidths=0.6, label=f'Top {top_k} {m}')

        # Iso-score lines for combined_avg_minmax over fp range
        if 'fp_norm_minmax' in df.columns:
            x_min = float(df['fp_per_min'].min())
            x_max = float(df['fp_per_min'].max())
            xs = np.linspace(x_min, x_max, 100)
            # Using previously computed fp_q05, denom (from this scope)
            iso_scores = [0.6, 0.7, 0.8]
            for s in iso_scores:
                ys = 2*s - 1 + np.clip((xs - fp_q05) / denom, 0, 1)
                plt.plot(xs, ys, linestyle='--', linewidth=1.0, label=f'iso combined_avg_minmax={s}', alpha=0.6)

        plt.xlabel('FP Rate per Minute')
        plt.ylabel('F1 Score')
        plt.title('F1 vs FP/min with Pareto and Combined Metrics (Top-20)')
        # Build a clean legend (dedupe labels)
        handles, labels = plt.gca().get_legend_handles_labels()
        seen = set(); new_h = []; new_l = []
        for h, l in zip(handles, labels):
            if l == '_nolegend_' or l in seen: continue
            seen.add(l); new_h.append(h); new_l.append(l)
        plt.legend(new_h, new_l, loc='best', fontsize='small', frameon=True)
        plt.grid(True, linestyle='--', alpha=0.3)
        overlay_path = os.path.join(viz_dir, 'combined_metrics_pareto_overlay.png')
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=140)
        plt.close()
        print(f"Saved overlay scatter to {overlay_path}")
    except Exception as e:
        print(f"Warning: failed to create combined metrics overlay plot: {e}")
    # Create a small HTML report
    combined_html = f"""
    <html><head><title>Combined Score Rankings: {study.study_name}</title>
    <style> table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #ddd; padding: 6px; }} th {{ background:#007bff; color:white; }} tr:nth-child(even){{background:#f9f9f9}} body {{ font-family: Arial; margin:20px; }} </style>
    </head><body>
    <h2>Combined Score Strategies</h2>
    <p>This study uses F1 Score (higher is better) and FP Rate per Minute (lower is better). We compute several robust combined scores to rank trials, addressing scale issues and potential window/stride overcounting:</p>
    <ul>
      <li><b>combined_avg_minmax</b>: (F1 + (1 - minmax(FP/min))) / 2 over 5–95% range.</li>
      <li><b>combined_hmean_minmax</b>: Harmonic mean of F1 and (1 - minmax(FP/min)).</li>
      <li><b>combined_avg_logistic</b>: (F1 + 1/(1 + FP/min / median(FP/min))) / 2.</li>
      <li><b>combined_rank_avg</b>: Average of percentile ranks (F1 ascending, FP/min descending).</li>
      <li><b>combined_adj_* </b>: Same as above but with FP/min adjusted by an approximate duplication factor = round(window/stride) (window≈{window_samples}, stride≈{approx_stride}).</li>
    </ul>
    <p>CSV with all scores: <a href="all_trials_with_scores.csv" target="_blank">all_trials_with_scores.csv</a></p>
    <h3>Top 20 by Consensus Rank (average of ranks across strategies)</h3>
    {top_consensus[['trial_number','f1_score','fp_per_min','consensus_rank'] + combined_cols].to_html(index=False)}
    </body></html>
    """
    with open(os.path.join(viz_dir, 'combined_scores.html'), 'w') as fh:
        fh.write(combined_html)


    # --- Hyperparameter Impact Analysis ---
    print("Generating hyperparameter impact plots...")
    hyperparams = [p for p in all_df.columns if p not in ['trial_number', 'f1_score', 'fp_per_min', 'combined_score']]
    impact_plot_files = []

    def should_use_log_scale(series):
        if not pd.api.types.is_numeric_dtype(series): return False
        if series.isnull().all() or series.nunique() <=1: return False
        min_val, max_val = series.min(), series.max()
        if min_val <= 0: return False
        if min_val < 1e-9 : return False # Avoid log for extremely small values if range is also small
        return (max_val / min_val) > 100

    for param in hyperparams:
        if param not in all_df.columns or all_df[param].isnull().all() or all_df[param].nunique() <= 1:
            print(f"Skipping parameter '{param}' due to missing data or single value.")
            continue

        is_numeric = pd.api.types.is_numeric_dtype(all_df[param])
        plot_filename = os.path.join(param_impact_dir, f"{param}_impact.png")
        impact_plot_files.append({"name": param, "path": f"param_impact/{param}_impact.png"})

        try:
            fig, ax1 = plt.subplots(figsize=(12, 7))
            use_log = is_numeric and should_use_log_scale(all_df[param])

            # Plot all completed trials as background
            if is_numeric:
                sns.scatterplot(data=all_df, x=param, y='f1_score', ax=ax1, color='lightgray', alpha=0.3, label='_nolegend_', s=30)
            else:
                 sns.stripplot(data=all_df, x=param, y='f1_score', ax=ax1, color='lightgray', alpha=0.3, order=sorted(all_df[param].astype(str).unique()), label='_nolegend_', s=4)

            # Highlight Pareto optimal trials for F1 Score
            if not pareto_df.empty and param in pareto_df.columns:
                if is_numeric:
                    sns.scatterplot(data=pareto_df, x=param, y='f1_score', ax=ax1, color='skyblue', s=70, marker='o', label='Pareto F1 Score', alpha=0.7, edgecolor='blue')
                else:
                    sns.stripplot(data=pareto_df, x=param, y='f1_score', ax=ax1, color='skyblue', s=7, marker='o', order=sorted(all_df[param].astype(str).unique()), label='Pareto F1 Score', alpha=0.7, jitter=False, edgecolor='blue')

            # Highlight Top N Pareto optimal trials even more for F1 Score
            if not top_pareto_df.empty and param in top_pareto_df.columns:
                if is_numeric:
                    sns.scatterplot(data=top_pareto_df, x=param, y='f1_score', ax=ax1, color='gold', s=150, marker='*', edgecolor='black', label=f'Top {N_TOP_PARETO} Pareto F1', alpha=1.0, zorder=5)
                else:
                    sns.stripplot(data=top_pareto_df, x=param, y='f1_score', ax=ax1, color='gold', s=12, marker='*', order=sorted(all_df[param].astype(str).unique()), label=f'Top {N_TOP_PARETO} Pareto F1', alpha=1.0, jitter=False, zorder=5, edgecolor='black')

            ax1.set_ylabel('F1 Score', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(f1_ylim)
            if use_log:
                ax1.set_xscale('log')
                ax1.set_xlabel(f"{param} (log scale)")
            else:
                ax1.set_xlabel(param)
            if not is_numeric:
                 plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

            ax2 = ax1.twinx()
            if is_numeric:
                sns.scatterplot(data=all_df, x=param, y='fp_per_min', ax=ax2, color='lightpink', alpha=0.3, label='_nolegend_', s=30)
            else:
                 sns.stripplot(data=all_df, x=param, y='fp_per_min', ax=ax2, color='lightpink', alpha=0.3, order=sorted(all_df[param].astype(str).unique()), label='_nolegend_', s=4)

            if not pareto_df.empty and param in pareto_df.columns:
                if is_numeric:
                    sns.scatterplot(data=pareto_df, x=param, y='fp_per_min', ax=ax2, color='salmon', s=70, marker='X', label='Pareto FP/min', alpha=0.7, edgecolor='red')
                else:
                    sns.stripplot(data=pareto_df, x=param, y='fp_per_min', ax=ax2, color='salmon', s=7, marker='X', order=sorted(all_df[param].astype(str).unique()), label='Pareto FP/min', alpha=0.7, jitter=False, edgecolor='red')

            if not top_pareto_df.empty and param in top_pareto_df.columns:
                if is_numeric:
                    sns.scatterplot(data=top_pareto_df, x=param, y='fp_per_min', ax=ax2, color='orangered', s=150, marker='P', edgecolor='black', label=f'Top {N_TOP_PARETO} Pareto FP/min', alpha=1.0, zorder=5)
                else:
                    sns.stripplot(data=top_pareto_df, x=param, y='fp_per_min', ax=ax2, color='orangered', s=12, marker='P', order=sorted(all_df[param].astype(str).unique()), label=f'Top {N_TOP_PARETO} Pareto FP/min', alpha=1.0, jitter=False, zorder=5, edgecolor='black')

            ax2.set_ylabel('FP Rate per Minute', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(fp_ylim)

            # Combine legends
            legend_items = {}
            for ax_ in [ax1, ax2]:
                h, l = ax_.get_legend_handles_labels()
                for handle, label_text in zip(h, l):
                    if label_text != '_nolegend_' and label_text not in legend_items:
                        legend_items[label_text] = handle

            priority_order = [
                f'Top {N_TOP_PARETO} Pareto F1', f'Top {N_TOP_PARETO} Pareto FP/min',
                'Pareto F1 Score', 'Pareto FP/min'
            ]
            sorted_legend_items_tuples = []
            for label_text in priority_order:
                if label_text in legend_items:
                    sorted_legend_items_tuples.append((legend_items[label_text], label_text))
                    del legend_items[label_text]
            for label_text, handle in legend_items.items():
                sorted_legend_items_tuples.append((handle, label_text))

            if sorted_legend_items_tuples:
                final_handles, final_labels = zip(*sorted_legend_items_tuples)
                ax1.legend(final_handles, final_labels, loc='best', fontsize='small', frameon=True, facecolor='white', framealpha=0.7)

            if ax2.get_legend() is not None: ax2.get_legend().remove()


            plt.title(f'Impact of {param} on F1 Score & FP/min (Pareto & Top {N_TOP_PARETO} Highlighted)')
            ax1.grid(True, axis='x', linestyle='--', alpha=0.6)
            fig.tight_layout()
            plt.savefig(plot_filename)
            plt.close(fig)

        except Exception as e:
            print(f"Error plotting impact for parameter '{param}': {e}")
            if 'fig' in locals() and fig: plt.close(fig)
            impact_plot_files = [f for f in impact_plot_files if f["name"] != param]


    # --- Correlation Heatmap ---
    print("Generating correlation heatmap...")
    numeric_hyperparams = all_df[hyperparams].select_dtypes(include=np.number).columns.tolist()
    if numeric_hyperparams: # Check if there are any numeric hyperparameters
        numeric_df_for_corr = all_df[numeric_hyperparams]
        if not numeric_df_for_corr.empty and numeric_df_for_corr.shape[1] > 1:
            try:
                correlation_matrix = numeric_df_for_corr.corr()
                plt.figure(figsize=(max(10, len(numeric_hyperparams)*0.8), max(8, len(numeric_hyperparams)*0.6)))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
                plt.title('Hyperparameter Correlation Heatmap (Numeric Only)')
                plt.xticks(rotation=45, ha='right', fontsize=9)
                plt.yticks(rotation=0, fontsize=9)
                plt.tight_layout()
                heatmap_filename = os.path.join(param_impact_dir, "correlation_heatmap.png")
                plt.savefig(heatmap_filename)
                plt.close()
                print(f"Correlation heatmap saved to {heatmap_filename}")
                heatmap_rel_path = "param_impact/correlation_heatmap.png"
            except Exception as e:
                print(f"Error generating correlation heatmap: {e}")
                if 'plt' in locals() and plt.gcf().get_axes(): plt.close()
                heatmap_rel_path = None
        else:
            print("Not enough numeric hyperparameters for correlation heatmap.")
            heatmap_rel_path = None
    else:
        print("No numeric hyperparameters found for correlation heatmap.")
        heatmap_rel_path = None

    # --- Parallel Coordinate Plot for Pareto Front ---
    parallel_pareto_path_rel = None
    if not pareto_df.empty and pareto_trials_optuna_objects:
        print("Generating Parallel Coordinate plot for Pareto front...")
        try:
            # Filter hyperparams to those present in pareto_df to avoid errors if some params were conditional
            pareto_hyperparams = [p for p in hyperparams if p in pareto_df.columns]
            if pareto_hyperparams: # Ensure there are params to plot
                fig_parallel_pareto = optuna.visualization.plot_parallel_coordinate(
                    study,
                    params=pareto_hyperparams,
                    target=lambda t: (t.values[0], t.values[1]) if t.values and len(t.values)==2 else (float('nan'), float('nan')),
                    target_name=["F1 Score", "FP Rate per Minute"],
                    trials=pareto_trials_optuna_objects
                )
                fig_parallel_pareto.update_layout(title=f"Parallel Coordinate Plot (Pareto Optimal Trials - {len(pareto_trials_optuna_objects)} trials)")
                parallel_pareto_filename = "parallel_coordinate_pareto.html"
                parallel_pareto_path_abs = os.path.join(viz_dir, parallel_pareto_filename)
                fig_parallel_pareto.write_html(parallel_pareto_path_abs)
                parallel_pareto_path_rel = parallel_pareto_filename # Relative path for HTML linking
                print(f"Parallel Coordinate plot for Pareto front saved to {parallel_pareto_path_abs}")
            else:
                print("No common hyperparameters found in Pareto trials for parallel coordinate plot.")
        except Exception as e:
            print(f"Warning: Error generating Parallel Coordinate plot for Pareto front: {e}")


    # --- Generate Summary HTML for Impact Plots ---
    print("Generating summary HTML for impact plots...")

    html_parts = [f"""
    <!DOCTYPE html><html><head><title>Hyperparameter Impact Analysis: {study.study_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ max-width: 1200px; margin: auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .gallery {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-around; }}
        .param-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; }}
        .param-card img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        .param-card.individual-plot {{ width: calc(50% - 20px); /* Two columns */ min-width: 400px; }}
        .param-card.full-width-plot {{ width: 100%; }}
        .param-card iframe {{ width: 100%; height: 500px; border: 1px solid #ccc; border-radius: 4px; }}
        h1, h2 {{ color: #333; text-align: center; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h1 {{ font-size: 2em; }} h2 {{ font-size: 1.5em; margin-top: 30px; }}
        nav {{ text-align: center; margin-bottom: 30px; background-color: #007bff; padding: 10px; border-radius: 5px; }}
        nav a {{ margin: 0 15px; text-decoration: none; color: white; font-weight: bold; font-size: 1.1em; }}
        nav a:hover {{ text-decoration: underline; }}
        p {{ line-height: 1.6; }}
    </style>
    </head><body><div class="container">
    <h1>Hyperparameter Impact Analysis: {study.study_name}</h1>
    <nav>
    <a href="#pareto_details">Pareto Trial Details</a> |
        { '<a href="#correlation">Correlation Analysis</a> |' if heatmap_rel_path else ""}
        { '<a href="#parallel_coord_pareto">Parallel Coordinate (Pareto)</a> |' if parallel_pareto_path_rel else ""}
    <a href="#combined_scores">Combined Scores</a> |
    <a href="#individual_params">Individual Parameter Analysis</a>
    </nav>
    <p style="text-align:center; font-style:italic;">Plots show all completed trials (light gray), Pareto optimal trials (skyblue/salmon), and Top {N_TOP_PARETO} Pareto trials (gold star/orangered P).</p>
    <p style="text-align:center; font-style:italic;">Y-axis ranges for individual plots are focused on the 5th-95th percentile of all completed trials for better visibility.</p>
    """]

    html_parts.append(f"""
    <h2 id="pareto_details">Pareto Optimal Trial Details</h2>
    <div class="param-card full-width-plot">
    <p>The following table details the <strong>{len(pareto_df)}</strong> trials found on the Pareto front, representing the best trade-offs between F1 Score (higher is better) and FP Rate per Minute (lower is better). They are sorted by a combined score for ranking purposes.</p>
        <p><a href="pareto_optimal_trials.html" target="_blank" style="font-weight:bold; color: #007bff;">Open Pareto Optimal Trials Table in new tab</a> (Recommended for full view)</p>
        <iframe src="pareto_optimal_trials.html" title="Pareto Optimal Trials Details" style="height: 400px;"></iframe>
    </div>""")

    if heatmap_rel_path:
        html_parts.append(f"""
        <h2 id="correlation">Parameter Correlation Analysis (Numeric)</h2>
        <div class="param-card full-width-plot" style="text-align:center;">
            <p>This heatmap shows linear correlations between numeric hyperparameters. Values close to 1 or -1 indicate strong positive or negative correlation, respectively. Values close to 0 indicate weak linear correlation.</p>
            <a href="{heatmap_rel_path}" target="_blank"><img src="{heatmap_rel_path}" alt="Correlation Heatmap" style="max-width: 800px; display: inline-block;"></a>
        </div>""")

    if parallel_pareto_path_rel:
        html_parts.append(f"""
        <h2 id="parallel_coord_pareto">Parallel Coordinate Plot (Pareto Optimal Trials)</h2>
        <div class="param-card full-width-plot">
            <p>This plot shows the parameter values for trials on the Pareto front. Each line represents a trial. It helps visualize combinations of parameters that lead to optimal trade-offs for F1 Score and FP Rate per Minute.</p>
            <iframe src="{parallel_pareto_path_rel}" title="Parallel Coordinate Plot for Pareto Optimal Trials"></iframe>
        </div>""")
    # Combined overlay figure
    overlay_rel = 'combined_metrics_pareto_overlay.png'
    if os.path.exists(os.path.join(viz_dir, overlay_rel)):
        html_parts.append(f"""
        <h2 id="combined_overlay">F1 vs FP/min with Combined Metrics</h2>
        <div class="param-card full-width-plot" style="text-align:center;">
            <p>Scatter of all trials (gray), Optuna Pareto (black circles), and Top-20 per selected combined metrics (colors/markers). Dashed lines show iso-scores for combined_avg_minmax.</p>
            <a href="{overlay_rel}" target="_blank"><img src="{overlay_rel}" alt="Combined metrics overlay" style="max-width: 900px; display: inline-block;"></a>
        </div>
        """)
    # Combined score section
    html_parts.append(f"""
    <h2 id='combined_scores'>Combined Scores and Consensus Ranking</h2>
    <div class='param-card full-width-plot'>
        <p>We compute several combined-score variants to balance F1 and FP/min, including min-max, logistic, rank-based, and approximate de-duplication based on window/stride. See details in the linked report.</p>
        <p><a href="combined_scores.html" target="_blank" style="font-weight:bold; color: #007bff;">Open Combined Scores Report</a></p>
        <iframe src="combined_scores.html" title="Combined Scores"></iframe>
    </div>
    """)

    html_parts.append("<h2 id='individual_params'>Individual Parameter Analysis</h2><div class='gallery'>")
    for plot_info in impact_plot_files:
        html_parts.append(f"""
        <div class="param-card individual-plot">
            <h3 style="text-align:center;">{plot_info['name']}</h3>
            <a href="{plot_info['path']}" target="_blank">
                <img src="{plot_info['path']}" alt="Impact of {plot_info['name']}">
            </a>
        </div>""")
    html_parts.append("</div>")

    html_parts.append("</div></body></html>") # Close container and body
    final_html_content = "\n".join(html_parts)

    with open(os.path.join(viz_dir, "hyperparameter_impact_analysis.html"), "w") as f:
        f.write(final_html_content)
    print(f"Hyperparameter impact analysis summary saved to {os.path.join(viz_dir, 'hyperparameter_impact_analysis.html')}")


    # --- Study Statistics ---
    print("Saving study statistics...")
    stats = {
        "study_name": study.study_name,
        "n_total_trials_in_db": len(study.trials),
        "n_completed_trials_retrieved": len(all_trials),
        "n_completed_trials_with_valid_objectives": len(all_df),
        "n_pareto_trials": len(pareto_trials_optuna_objects),
        "n_top_pareto_highlighted": N_TOP_PARETO,
        "n_pruned_trials": len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])),
        "n_failed_trials": len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.FAIL]))
    }
    with open(os.path.join(viz_dir, "study_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    print("Study statistics saved.")
    print("\nStudy Statistics Summary:")
    print(json.dumps(stats, indent=2))

    print(f"\nVisualization generation complete. Results are in: {viz_dir}")
    print(f"Main report: {os.path.join(viz_dir, 'hyperparameter_impact_analysis.html')}")

elif mode == 'tune_viz_multi_v3':
    import argparse
    import os
    import sys
    import json
    import warnings
    import math

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import optuna
    from scipy.stats import ks_2samp

    # Optional imports – Plotly & SHAP
    try:
        import plotly.express as px
        HAVE_PLOTLY = True
    except ImportError:
        HAVE_PLOTLY = False

    try:
        import lightgbm as lgb, shap
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        HAVE_SHAP = True
    except ImportError:
        HAVE_SHAP = False

    # 2. Paths
    param_dir = f"params_{tag}"
    viz_dir = os.path.join("studies", param_dir, "visualizations")
    param_impact_dir = os.path.join(viz_dir, "param_impact")
    os.makedirs(param_impact_dir, exist_ok=True)

    # 3. Load study
    try:
        storage_url = f"sqlite:///studies/{param_dir}/{param_dir}.db"
        study = optuna.load_study(study_name=param_dir, storage=storage_url)
    except Exception as e:
        print(f"Error loading study '{param_dir}': {e}")
        sys.exit(1)

    # 4. Standard Optuna plots
    print("Generating standard Optuna plots...")
    try:
        hist_f1 = optuna.visualization.plot_optimization_history(
            study,
            target=lambda t: t.values[0] if t.values and len(t.values)>0 else float('nan'),
            target_name="F1 Score")
        hist_f1.write_html(os.path.join(viz_dir, "optimization_history_f1.html"))

        hist_lat = optuna.visualization.plot_optimization_history(
            study,
            target=lambda t: t.values[1] if t.values and len(t.values)>1 else float('nan'),
            target_name="Latency")
        hist_lat.write_html(os.path.join(viz_dir, "optimization_history_latency.html"))

        pareto = optuna.visualization.plot_pareto_front(
            study, target_names=["F1 Score", "Latency"])
        pareto.write_html(os.path.join(viz_dir, "pareto_front.html"))

        imp_f1 = optuna.visualization.plot_param_importances(
            study,
            target=lambda t: t.values[0] if t.values and len(t.values)>0 else float('nan'),
            target_name="F1 Score")
        imp_f1.write_html(os.path.join(viz_dir, "param_importances_f1.html"))

        imp_lat = optuna.visualization.plot_param_importances(
            study,
            target=lambda t: t.values[1] if t.values and len(t.values)>1 else float('nan'),
            target_name="Latency")
        imp_lat.write_html(os.path.join(viz_dir, "param_importances_latency.html"))

        print("Standard plots generated.")
    except Exception as e:
        print(f"Warning: Error generating standard plots: {e}")

    # 5. Data prep
    all_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    rows = []
    for t in all_trials:
        if t.values and len(t.values)==2:
            try:
                f1, lat = float(t.values[0]), float(t.values[1])
            except:
                continue
            if not (np.isnan(f1) or np.isnan(lat)):
                d = {"trial_number": t.number, "f1_score": f1, "latency": lat}
                d.update(t.params)
                rows.append(d)
    all_df = pd.DataFrame(rows)

    # 6. Axis quantiles
    f1_q05, f1_q95 = np.nanquantile(all_df['f1_score'], [0.05,0.95])
    lat_q05, lat_q95 = np.nanquantile(all_df['latency'], [0.05,0.95])
    f1_ylim = (max(0, f1_q05-(f1_q95-f1_q05)*0.05), min(1, f1_q95+(f1_q95-f1_q05)*0.05))
    lat_ylim = (max(0, lat_q05-(lat_q95-lat_q05)*0.05), lat_q95+(lat_q95-lat_q05)*0.05)

    # 7. Pareto analysis
    pareto_trials = study.best_trials
    pareto_df = all_df[all_df['trial_number'].isin([t.number for t in pareto_trials])].copy()
    pareto_df['combined_score'] = (pareto_df['f1_score'] + (1-pareto_df['latency']))/2
    pareto_df.sort_values('combined_score', ascending=False, inplace=True)
    N_TOP = min(10, len(pareto_df))

    # 8. Quantitative summaries
    hyperparams = [c for c in all_df.columns if c not in ['trial_number','f1_score','latency','combined_score']]
    def summarize_param(p):
        s_all = all_df[p].dropna()
        s_par = pareto_df[p].dropna()
        if pd.api.types.is_numeric_dtype(s_all) and not pd.api.types.is_bool_dtype(s_all):
            qs = (0.10, 0.50, 0.90)
            good = s_par
            bad  = s_all[~s_all.index.isin(good.index)]
            return {
                "pareto_q": np.quantile(good, qs).round(6).tolist() if len(good) else [None]*3,
                "other_q": np.quantile(bad, qs).round(6).tolist() if len(bad) else [None]*3,
                "ks_p": float(ks_2samp(good, bad).pvalue) if len(good) and len(bad) else None
            }
        else:
            return {
                "all_counts": s_all.value_counts().to_dict(),
                "pareto_counts": s_par.value_counts().to_dict()
            }


    with open(os.path.join(param_impact_dir, "param_ranges.json"), "w") as fp:
        json.dump({p: summarize_param(p) for p in hyperparams}, fp, indent=2)

    # 8c. Per‐param boxplots
    # Create Top-k subsets
    top_f1_df  = pareto_df.nlargest(N_TOP, 'f1_score')  if not pareto_df.empty else pd.DataFrame()
    top_lat_df = pareto_df.nsmallest(N_TOP, 'latency')    if not pareto_df.empty else pd.DataFrame()

    def make_bins(s, n=5):
        try: return pd.qcut(s, n, duplicates='drop')
        except: return s.astype(str)

    for p in hyperparams:
        # bin continuous
        vals = all_df[p].dropna()
        if pd.api.types.is_numeric_dtype(vals) and vals.nunique()>5:
            bcol = f"{p}_bin"
            all_df[bcol] = make_bins(all_df[p])
            bin_map = all_df.set_index('trial_number')[bcol].to_dict()
            pareto_df[bcol]  = pareto_df['trial_number'].map(bin_map)
            top_f1_df[bcol]  = top_f1_df['trial_number'].map(bin_map)
            top_lat_df[bcol] = top_lat_df['trial_number'].map(bin_map)
            xcol, xlabel = bcol, f"{p} (binned)"
        else:
            xcol, xlabel = p, p

        fig, ax1 = plt.subplots(figsize=(8,6))
        ax2 = ax1.twinx()

        # F1 box + Pareto + Top-F1
        sns.boxplot(x=xcol, y='f1_score', data=all_df, ax=ax1, palette='Blues', fliersize=2)
        sns.stripplot(x=xcol, y='f1_score', data=pareto_df, ax=ax1,
                      color='navy', marker='o', size=8, edgecolor='white', label='Pareto F1', jitter=False)
        sns.stripplot(x=xcol, y='f1_score', data=top_f1_df, ax=ax1,
                      color='gold', marker='*', size=12, edgecolor='darkorange', label=f'Top {N_TOP} F1', jitter=False)
        ax1.set_ylabel('F1 Score', color='navy'); ax1.tick_params(labelcolor='navy')
        ax1.set_xlabel(xlabel); plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

        # Latency box + Pareto + Top-Lat
        sns.boxplot(x=xcol, y='latency', data=all_df, ax=ax2, palette='Reds', fliersize=2)
        sns.stripplot(x=xcol, y='latency', data=pareto_df, ax=ax2,
                      color='darkred', marker='^', size=8, edgecolor='white', label='Pareto Lat', jitter=False)
        sns.stripplot(x=xcol, y='latency', data=top_lat_df, ax=ax2,
                      color='orangered', marker='s', size=10, edgecolor='firebrick', label=f'Top {N_TOP} Lat', jitter=False)
        ax2.set_ylabel('Latency', color='firebrick'); ax2.tick_params(labelcolor='firebrick')
        ax2.set_ylim(lat_ylim)

        # legend
        h1,l1 = ax1.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()
        lookup = dict(zip(l1+l2, h1+h2))
        order = ['Pareto F1','Pareto Lat',f'Top {N_TOP} F1',f'Top {N_TOP} Lat']
        labs   = [lbl for lbl in order if lbl in lookup]
        handles= [lookup[lbl] for lbl in labs]
        ax1.legend(handles, labs, loc='best', fontsize='small', frameon=True, facecolor='white', framealpha=0.85)

        plt.title(f'Impact of {p} on F1 & Latency')
        plt.tight_layout()
        out_fp = os.path.join(param_impact_dir, f"{p}_impact_boxplot.png")
        plt.savefig(out_fp, dpi=120)
        plt.close(fig)
        print(f"Saved {out_fp}")
        
elif mode == 'tune_viz_multi_v4':
    import seaborn as sns
    # Directories and study loading
    param_dir = f"params_{tag}"
    study_db = f"studies/{param_dir}/{param_dir}.db"
    viz_dir = f"studies/{param_dir}/visualizations/advanced"
    os.makedirs(viz_dir, exist_ok=True)

    # Load the Optuna study
    try:
        study = optuna.load_study(
            study_name=param_dir,
            storage=f"sqlite:///{study_db}",
        )
        print(f"Loaded study '{param_dir}' from {study_db}")
    except Exception as e:
        print(f"Error loading study '{param_dir}': {e}")
        sys.exit(1)

    # Gather completed trials
    trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not trials:
        print("No completed trials found. Exiting.")
        sys.exit(0)

    # Build DataFrame of trial results
    records = []
    for t in trials:
        if t.values and len(t.values) == 2:
            f1 = float(t.values[0])
            fpr = float(t.values[1])  # fprate per minute
            params = {k: v for k, v in t.params.items()}
            records.append({"trial_number": t.number, "f1_score": f1, "fprate_per_min": fpr, **params})

    all_df = pd.DataFrame(records)
    if all_df.empty:
        print("No valid trial data. Exiting.")
        sys.exit(0)
    print(f"Built DataFrame with {len(all_df)} trials.")

    # Identify Pareto-optimal trials
    pareto_trials = study.best_trials
    pareto_nums = [t.number for t in pareto_trials]
    pareto_df = all_df[all_df.trial_number.isin(pareto_nums)].copy()
    if pareto_df.empty:
        print("No Pareto-optimal trials.")
    else:
        print(f"Found {len(pareto_df)} Pareto-optimal trials.")

    # --- Advanced Visualizations ---
    print("Generating advanced multi-objective visualizations...")
    try:
        # 1. Hypervolume History
        max_fpr = all_df['fprate_per_min'].max() * 1.1
        ref_point = [0.0, max_fpr]
        fig_hv = optuna.visualization.plot_hypervolume_history(
            study,
            reference_point=ref_point
        )
        fig_hv.write_html(os.path.join(viz_dir, "hypervolume_history.html"))

        # 2. Empirical Distribution Functions (EDF)
        for col, name in [("f1_score", "F1 Score"), ("fprate_per_min", "FPRate/min")]:
            fig_edf = optuna.visualization.plot_edf(
                study,
                target=(lambda t: t.values[0]) if col == "f1_score" else (lambda t: t.values[1]),
                target_name=name
            )
            fig_edf.write_html(os.path.join(viz_dir, f"edf_{col}.html"))

        # 3. Pareto front scatter + marginals
        jp = sns.jointplot(
            data=all_df,
            x="fprate_per_min",
            y="f1_score",
            kind="scatter",
            marginal_kws={"bins": 20, "fill": True},
            space=0.2
        )
        jp.set_axis_labels("FPRate/min", "F1 Score")
        jp.fig.suptitle("Pareto Trade-off: FPRate/min vs. F1", y=1.02)
        jp.savefig(os.path.join(viz_dir, "pareto_marginals.png"))
        plt.close()

        # 4. Pareto front plot
        fig_pareto = optuna.visualization.plot_pareto_front(
            study,
            target_names=["F1 Score", "FPRate/min"]
        )
        fig_pareto.write_html(os.path.join(viz_dir, "pareto_front.html"))

        # 5. Rank plots for each objective
        params = [c for c in all_df.columns if c not in ["trial_number", "f1_score", "fprate_per_min"]]
        fig_rank_f1 = optuna.visualization.plot_rank(
            study,
            params=params,
            target=lambda t: t.values[0],
            target_name="F1 Score"
        )
        fig_rank_f1.write_html(os.path.join(viz_dir, "param_rank_f1.html"))
        fig_rank_fpr = optuna.visualization.plot_rank(
            study,
            params=params,
            target=lambda t: t.values[1],
            target_name="FPRate/min"
        )
        fig_rank_fpr.write_html(os.path.join(viz_dir, "param_rank_fpr.html"))

        # 6. Slice & Contour plots for each objective
        fig_slice_f1 = optuna.visualization.plot_slice(
            study,
            params=params,
            target=lambda t: t.values[0],
            target_name="F1 Score"
        )
        fig_slice_f1.write_html(os.path.join(viz_dir, "slice_f1.html"))
        fig_slice_fpr = optuna.visualization.plot_slice(
            study,
            params=params,
            target=lambda t: t.values[1],
            target_name="FPRate/min"
        )
        fig_slice_fpr.write_html(os.path.join(viz_dir, "slice_fpr.html"))

        numeric_params = [p for p in params if pd.api.types.is_numeric_dtype(all_df[p])]
        if len(numeric_params) >= 2:
            fig_contour_f1 = optuna.visualization.plot_contour(
                study,
                params=numeric_params[:2],
                target=lambda t: t.values[0],
                target_name="F1 Score"
            )
            fig_contour_f1.write_html(os.path.join(viz_dir, "contour_f1.html"))
            fig_contour_fpr = optuna.visualization.plot_contour(
                study,
                params=numeric_params[:2],
                target=lambda t: t.values[1],
                target_name="FPRate/min"
            )
            fig_contour_fpr.write_html(os.path.join(viz_dir, "contour_fpr.html"))

        # 7. Timeline plot
        fig_time = optuna.visualization.plot_timeline(study)
        fig_time.write_html(os.path.join(viz_dir, "timeline.html"))

        # 8. Hypervolume contributions (optional, requires pygmo)
        try:
            import pygmo as pg
            pts = np.array([[1 - t.values[0], t.values[1]] for t in study.best_trials])
            hv = pg.hypervolume(pts)
            contribs = hv.contributions(ref_point=[1, max_fpr])
            order = np.argsort(contribs)[::-1]
            labels = [f"#{study.best_trials[i].number}" for i in order]
            vals = contribs[order]
            plt.figure(figsize=(8, 4))
            plt.bar(labels, vals)
            plt.xlabel("Pareto Trial")
            plt.ylabel("HV Contribution")
            plt.title("Hypervolume Contribution per Pareto Trial")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "hv_contributions.png"))
            plt.close()
        except ImportError:
            print("pygmo not installed; skipping HV contributions")

        print("Advanced visualizations written to:", viz_dir)

    except Exception as err:
        print("Error generating advanced visualizations:", err)


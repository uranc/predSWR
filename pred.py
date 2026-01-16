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
params = {'BATCH_SIZE': 64, 'SHUFFLE_BUFFER_SIZE': 4096*2,
          'WEIGHT_FILE': '', 'LEARNING_RATE': 1e-3, 'NO_EPOCHS': 200,
          'NO_TIMEPOINTS': 64, 'NO_CHANNELS': 8, 'SRATE': 30000,
          'EXP_DIR': '/cs/projects/MWNaturalPredict/DL/predSWR/experiments/' + model_name,
          }
params['mode'] = mode

if mode == 'train':
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
        # from model.input_augment_weighted import rippleAI_load_dataset
        from model.input_fn_TTE import rippleAI_load_dataset
    elif 'TripletOnly' in params['TYPE_ARCH']:
        print('Using TripletOnly')
        from model.input_augment_weighted_transpose import rippleAI_load_dataset
        # from model.input_proto import rippleAI_load_dataset
        # from model.input_proto_new import rippleAI_load_dataset
        # from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
        from model.model_fn import build_DBI_TCN_TripletOnlyTranspose as build_DBI_TCN
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

    tf.config.run_functions_eagerly(True)
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
            params['steps_per_epoch'] = 2
            flag_online = 'Online' in params['TYPE_ARCH']
            # train_dataset, test_dataset, label_ratio, dataset_params = rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=flag_online)
            train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)
            dataset_params = params
        elif 'MixerOnly' in params['TYPE_ARCH']:
            flag_online = 'Online' in params['TYPE_ARCH']
            train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=flag_online)
        else:
            train_dataset, test_dataset, label_ratio = rippleAI_load_dataset(params, preprocess=preproc)
    # train_size = len(list(train_dataset))
    params['RIPPLE_RATIO'] = label_ratio

    # --- Preview one triplet batch (debug utility) ---
    DEBUG_PLOT_TRIPLET = False  # set False to disable
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

        # n=0
        # for istep in range(50):
        #     x_preview, y_preview = next(iter(train_dataset))
        #     x_preview = x_preview.numpy()
        #     y_preview = y_preview.numpy()
        #     pdb.set_trace()
        #     if np.sum(y_preview[:,:,0])==0:
        #         continue
        #     # pdb.set_trace()
        #     for ii in range(0, 25):
        #         n+=1
        #         plt.subplot(5,5,n)
        #         plt.plot(np.arange(512), x_preview[0+ii, :, :]*100+np.array([0, 5, 10, 15, 20, 25, 30, 35]))
        #         plt.plot(np.arange(256)+256, -20+20*y_preview[0+ii, :, 1], 'r')        
        #         plt.axis('off')
        #         if np.mod(n, 25) == 0:
        #             plt.show()
        #             n = 0
        # n=0
        # for ii in range(0, 10):
        #     n+=1
        #     plt.subplot(10,3,n)
        #     plt.plot(np.arange(128), x_preview[0+ii, :, :]*4+np.array([0, 5, 10, 15, 20, 25, 30, 35]))
        #     plt.plot(np.arange(64)+64, y_preview[0+ii, :]*50, 'r')
        #     n+=1
        #     plt.subplot(10,3,n)
        #     plt.plot(np.arange(128), x_preview[32+ii, :, :]*4+np.array([0, 5, 10, 15, 20, 25, 30, 35]))
        #     plt.plot(np.arange(64)+64, y_preview[32+ii, :]*50, 'r')
        #     n+=1
        #     plt.subplot(10,3,n)
        #     plt.plot(np.arange(128), x_preview[64+ii, :, :]*4+np.array([0, 5, 10, 15, 20, 25, 30, 35]))
        #     plt.plot(np.arange(64)+64, y_preview[64+ii, :]*50, 'r')
        # plt.show()
        
    # pdb.set_trace()
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
            hist = train_pred(model, train_dataset, test_dataset, params['NO_EPOCHS'], params['EXP_DIR'], dataset_params=dataset_params, steps_per_epoch=10)
        else:
            hist = train_pred(model, train_dataset, test_dataset, params['NO_EPOCHS'], params['EXP_DIR'])

elif mode == 'predict':
    # modelname
    model = args.model[0]
    model_name = model
    import importlib

    if model_name.startswith('Tune'):
        tf.keras.backend.clear_session()    
        gc.collect()
        tf.config.run_functions_eagerly(False)
        tf.random.set_seed(1337); np.random.seed(1337); random.seed(1337)        
        # Extract study number from model name (e.g., 'Tune_45_' -> '45')
        study_num = model_name.split('_')[1]
        print(f"Loading tuned model from study {study_num}")
        # pdb.set_trace()
        # params['SRATE'] = 2500
        # Find the study directory
        import glob
        # study_dirs = glob.glob(f'studies_1/study_{study_num}_*')
        tag = args.tag[0]
        param_dir = f"params_{tag}"
        study_dirs = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/study_{study_num}_*')
        base_dir = f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/'
        # study_dirs = glob.glob(f'studies_CHECK_SIGNALS/{param_dir}/study_{study_num}_*')
        if not study_dirs:
            raise ValueError(f"No study directory found for study number {study_num}")
        study_dir = study_dirs[0]  # Take the first matching directory
        # pdb.set_trace()
        # Load trial info to get parameters
        with open(f"{study_dir}/trial_info.json", 'r') as f:
            trial_info = json.load(f)
            params.update(trial_info['parameters'])
        # pdb.set_trace()
        params['mode'] = 'predict'

        # pdb.set_trace()
        # from tensorflow.keras.models import load_model
        # from tcn import TCN
        # from tensorflow.keras.regularizers import L1
        # from model.model_fn import WarmStableCool, mixed_latent_loss, MultiScaleCausalGate
        # from model.model_fn import SampleMaxF1, SampleMaxMCC, SamplePRAUC, LatencyScore, FPperMinMetric
        # with tf.keras.utils.custom_object_scope({'TCN': TCN, 
        #                                         'L1': L1,
        #                                         'WarmStableCool': WarmStableCool,
        #                                         'loss_fn': 'mse',
        #                                         'SampleMaxF1': SampleMaxF1,
        #                                         'SampleMaxMCC': SampleMaxMCC, 
        #                                         'SamplePRAUC': SamplePRAUC,
        #                                         'LatencyScore': LatencyScore,
        #                                         'FPperMinMetric': FPperMinMetric,
        #                                         'MultiScaleCausalGate': MultiScaleCausalGate
        #                                         }):
        #     trained_model = load_model('/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/params_tripletOnlyRemake2500/study_911_20251013_024414/mcc.weights.h5')
        # pdb.set_trace()
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
            # pdb.set_trace()
            # from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN
            # pdb.set_trace()
            import importlib.util
            # spec = glob.glob(f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/study_{2211}_*')[0]
            # spec = importlib.util.spec_from_file_location("model_fn", f"{tmp_dir}/model/model_fn.py")
            # spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn_BACKUP.py")
            spec = importlib.util.spec_from_file_location("model_fn", f"{base_dir}/base_model/model_fn.py")
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
            mcc_weights = f"{pretrained_dir}/mcc.weights.h5"

            
            if os.path.exists(event_weights) and os.path.exists(max_weights):
                # Both files exist, select the most recently modified one
                event_mtime = os.path.getmtime(event_weights)
                max_mtime = os.path.getmtime(max_weights)
                mcc_mtime = os.path.getmtime(mcc_weights)

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

        # pdb.set_trace()
        from model.model_fn import CSDLayer
        from tcn import TCN
        from tensorflow.keras.models import load_model

        # Check which weight file is most recent
        event_weights = f"{study_dir}/event.weights.h5"
        max_weights = f"{study_dir}/max.weights.h5"
        mcc_weights = f"{study_dir}/mcc.weights.h5"

        # event_weights = f"{study_dir}/event.finetune.weights.h5"
        # max_weights = f"{study_dir}/max.finetune.weights.h5"
        if os.path.exists(event_weights) and os.path.exists(max_weights):
            # Both files exist, select the most recently modified one
            event_mtime = os.path.getmtime(event_weights)
            max_mtime = os.path.getmtime(max_weights)
            
            if os.path.exists(mcc_weights):
                print('MCC weights also found')
                mcc_mtime = os.path.getmtime(mcc_weights)
                if (mcc_mtime > event_mtime) and (mcc_mtime > max_mtime):
                    weight_file = mcc_weights
                    tag += 'MCC'
                    print(f"Using mcc.weights.h5 (more recent, modified at {time.ctime(mcc_mtime)})")
                elif event_mtime > max_mtime:
                    weight_file = event_weights
                    tag += 'EvF1'
                    print(f"Using event.weights.h5 (more recent, modified at {time.ctime(event_mtime)})")
                else:
                    weight_file = max_weights
                    tag += 'MaxF1'
                    print(f"Using max.weights.h5 (more recent, modified at {time.ctime(max_mtime)})")
            else:
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
        # weight_file = max_weights
        # pdb.set_trace()
        print(f"Loading weights from: {weight_file}")

        # params["WEIGHT_FILE"] = weight_file
        # load models
        if 'CADOnly' in params['TYPE_ARCH']:
            model = build_DBI_TCN(pretrained_params["NO_TIMEPOINTS"], params=params, pretrained_tcn=pretrained_tcn)
        elif 'Patch' in params['TYPE_ARCH']:
            model = build_DBI_TCN(params=params) # Pass only the params dictionary
        else:
            params["WEIGHT_FILE"] = weight_file
            model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
        try:
            model.load_weights(weight_file)
            print('Loaded weights successfully')
        except:
            from tcn import TCN
            from tensorflow.keras.regularizers import L1
            from model.model_fn import WarmStableCool, mixed_latent_loss, MultiScaleCausalGate
            from model.model_fn import SampleMaxF1, SampleMaxMCC, SamplePRAUC, LatencyScore, FPperMinMetric
            with tf.keras.utils.custom_object_scope({'TCN': TCN, 
                                                    'L1': L1,
                                                    'WarmStableCool': WarmStableCool,
                                                    'loss_fn': 'mse',
                                                    'SampleMaxF1': SampleMaxF1,
                                                    'SampleMaxMCC': SampleMaxMCC, 
                                                    'SamplePRAUC': SamplePRAUC,
                                                    'LatencyScore': LatencyScore,
                                                    'FPperMinMetric': FPperMinMetric,
                                                    'MultiScaleCausalGate': MultiScaleCausalGate
                                                    }):
                trained_model = load_model(weight_file, compile=False)
                trained_model.save_weights(weight_file[:-2]+'weightsOnly.h5')
                model.load_weights(weight_file[:-2]+'weightsOnly.h5')
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

    # model.layers[1].get_layer('cls_logits').bias.assign(tf.constant(0.))[None,]
    # inference parameters
    squence_stride = 1
    # params['BATCH_SIZE'] = 512*4*3
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
        # LFP = np.load('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/Mouse3_LFP_ZSig_2500.npy')
        # LFP = np.load('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/MouseTest_LFP_ZSig_2500.npy')
        # LFP = np.load('/mnt/hpc/projects/OWVinckSWR/Dataset/ONIXData/Awake03/LFP_zSig_141125_suffix11_sf2500.npy')
        LFP = np.load('/cs/projects/OWVinckSWR/Cem/StateMachine_Session1_LFP_ZScore.npy')
        # LFP = np.load('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake03/download/lfp_sub12_slidingZ_CropTEST.npy')
        # pdb.set_trace()
        LFP = np.transpose(LFP, (1, 0)).astype(np.float32)
        # import matplotlib.pyplot as plt
        # LFP += np.array([0, 5, 10, 15, 20, 25, 30, 35])
        # for ii in range(LFP.shape[1]):
        #     plt.plot(LFP[:,ii])
        # plt.show()
        # pdb.set_trace()
        # peak_chind = 13 # sexy peak
        # peak_chind = 15 # sexy peak
        # peak_chind = 28 # sexy cortex
        # LFP = np.fliplr(LFP[::2,peak_chind-3:peak_chind+5])
        # LFP = LFP[:,peak_chind-3:peak_chind+5]
        # pdb.set_trace()
        # LFP = np.fliplr(LFP[:,:]) # necessary for onix data, deep is the last channel
        # LFP = np.fliplr(LFP[:,peak_chind-3:peak_chind+5])
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
                # val_datasets, val_labels = rippleAI_load_dataset(params, preprocess=preproc)
                flag_online = 'Online' in params['TYPE_ARCH']
                val_datasets, val_labels = rippleAI_load_dataset(params, mode='test', preprocess=True, process_online=flag_online)
                
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
                # np.save('/mnt/hpc/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/probs_{0}_decimate_{1}.npy'.format(study_num, tag), probs)
                np.save('/mnt/hpc/projects/OWVinckSWR/Dataset/ONIXData/Awake03/probs_{0}_decimate_{1}.npy'.format(study_num, tag), probs)
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
            sample_length = 44#params['NO_TIMEPOINTS']
            squence_stride = 2

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
            # np.save('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/probs_{0}_sub_LFPzSig_{1}.npy'.format(study_num, tag), probs)
            np.save('/cs/projects/OWVinckSWR/Dataset/ONIXData/Awake03/probs_{0}_sub_LFPzSig_{1}.npy'.format(study_num, tag), probs)
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

        params['SRATE'] = 2500
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
            # tf.config.run_functions_eagerly(True)
            from model.model_fn import build_DBI_TCN_TripletOnly as build_DBI_TCN

            # model_dir = f"studies/{param_dir}/base_model"
            # spec = importlib.util.spec_from_file_location("model_fn", f"{model_dir}/model_fn.py")
            # model_module = importlib.util.module_from_spec(spec)
            # spec.loader.exec_module(model_module)
            # build_DBI_TCN = model_module.build_DBI_TCN_TripletOnly            
            # import importlib.util
            # spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
            # model_module = importlib.util.module_from_spec(spec)
            # spec.loader.exec_module(model_module)
            # build_DBI_TCN = model_module.build_DBI_TCN_TripletOnly

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
        params['WEIGHT_FILE'] = weight_file
        if 'CADOnly' in params['TYPE_ARCH']:
            model = build_DBI_TCN(pretrained_params["NO_TIMEPOINTS"], params=params, pretrained_tcn=pretrained_tcn)
        elif 'Patch' in params['TYPE_ARCH']:
            model = build_DBI_TCN(params=params) # Pass only the params dictionary
        else:
            # pdb.set_trace()
            params.update({
                # 1. ARCHITECTURE & SPEED
                'TYPE_ARCH': params['TYPE_ARCH'].replace("StopGrad", ""),
                'NAME': params['NAME'].replace("StopGrad", ""),
                
                # change optimizer
                'TYPE_REG': params['TYPE_REG'].replace("AdamWA", "AdamMixer"),
                'NAME': params['NAME'].replace("AdamWA", "AdamMixer"),
                
                'USE_StopGrad': False,
                'USE_LR_SCHEDULE': False,
                'LEARNING_RATE': 1e-4,        # Very low (Protect the backbone)
                'BATCH_SIZE': 128,            # Maximize stability

                # 2. DISABLE GEOMETRY (Crucial)
                # The backbone structure is already good. Don't let Triplet/Circle
                # distract the classifier. Silence them.
                'LOSS_TupMPN': 50.0,
                # 'LOSS_Circle': 0.0,
                # 'LOSS_SupCon': 0.0,
                'MARGIN_WEAK': 0.05,
                # 3. FIX THE COLLAPSE (Rebalance)
                # The Ramp was killing you. Disable it.
                'NEG_RAMP_DELAY': 0,          # No warmup
                'NEG_RAMP_STEPS': 1,          # Instant on
                
                # Drastically reduce negative weight (was ~26.0 -> Now 1.0)
                'LOSS_NEGATIVES': 5.0,        
                'LOSS_NEGATIVES_MIN': 1.0,

                # Drastically INCREASE positive weight (was ~2.0 -> Now 5.0)
                # This forces the model to recover Recall immediately.
                'BCE_POS_ALPHA': 2.5,
                'DROP_RATE': 0.05,
                # 4. SHARPEN PREDICTIONS (Latency & Certainty)
                # Remove smoothing so the model can hit 1.0
                'LABEL_SMOOTHING': 0.1,
                'LOSS_TV': 0.3,
                'NO_EPOCHS': 200,
            })
            params.update({'mode': 'fine_tune'})
            model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
        # model.load_weights(weight_file, skip_mismatch=True)
        print(params)
        # get sampling rate # little dangerous assumes 4 digits
        if 'Samp' in params['TYPE_LOSS']:
            sind = params['TYPE_LOSS'].find('Samp')
            params['SRATE'] = int(params['TYPE_LOSS'][sind+4:sind+8])
        else:
            params['SRATE'] = 1250

    model.summary()
    # params['BATCH_SIZE'] = 128
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
        elif 'MixerOnly' in params['TYPE_ARCH']:
            flag_online = 'Online' in params['TYPE_ARCH']
            train_dataset, test_dataset = rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=flag_online)
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
    callbacks = [cb.EarlyStopping(monitor='val_sample_pr_auc',  # Change monitor
                        patience=20,
                        mode='max',
                        verbose=1,
                        restore_best_weights=True),
        cb.ModelCheckpoint(f"{study_dir}/max.finetune.weights.h5",
                            monitor='val_sample_max_f1',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='max'),
        cb.ModelCheckpoint(
                            f"{study_dir}/event.finetune.weights.h5",
                            monitor='val_sample_pr_auc',  # Change monitor
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

elif mode=='export':
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers

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
            # from model.model_fn import build_DBI_TCN_TripletOnlyTranspose as build_DBI_TCN
            import importlib.util
            base_dir = f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/'
            # study_dirs = glob.glob(f'studies_CHECK_SIGNALS/{param_dir}/study_{study_num}_*')
            spec = importlib.util.spec_from_file_location("model_fn", f"{base_dir}/base_model/model_fn.py")            
            # spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            build_DBI_TCN = model_module.build_DBI_TCN_TripletOnly
            # build_DBI_TCN = model_module.build_DBI_TCN_TripletOnlyTranspose
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
            # event_weights = f"{pretrained_dir}/event.finetune.weights.h5"
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
            # weight_file = f"{study_dir}/max.weights.h5"

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
            params['WEIGHT_FILE'] = weight_file
            model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
            # model.load_weights(weight_file)

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
        # a_model = importlib.import_module('experiments.{0}.model.model_fn'.format(model))
        a_model = importlib.import_module('model.model_fn')
        if model.find('CSD') != -1:
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_CSD')
        else:
            # build_DBI_TCN = getattr(a_model, 'build_DBI_TCN')
            # build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_TripletOnly')
            # build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_TripletOnlyTranspose')
            build_DBI_TCN = getattr(a_model, 'build_DBI_TCN_CADMixerOnly')
            params['LOSS_WEIGHT'] = 1.0
        # from model.model_fn import build_DBI_TCN

        params['WEIGHT_FILE'] = 'experiments/{0}/'.format(model_name)+'weights.last.h5'
        model = build_DBI_TCN(params["NO_TIMEPOINTS"], params=params)
    model.summary()

    model_converter = 'ONNX' # 'TF' 'TFLite'
    if model_converter == 'ONNX':
        os.environ["OMP_NUM_THREADS"] = "1" 
        # Prevents GPU conflicts during simplification
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

        import tf2onnx
        import onnx
        from onnxsim import simplify
        import tensorflow as tf
        # Convert the model to ONNX format
        # spec = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="x")]
        # spec = [tf.TensorSpec([1,92,8], model.inputs[0].dtype, name="x")]
        spec = [tf.TensorSpec([1,44,8],  tf.float32, name="x")]
        # spec = [tf.TensorSpec([1,44,8], model.inputs[0].dtype, name="x")]
        # spec = [tf.TensorSpec([1,560,8], model.inputs[0].dtype, name="x")]
        # spec = [tf.TensorSpec([1, 8, 560], model.inputs[0].dtype, name="x")]
        # pdb.set_trace()
        output_path = f"./frozen_models/{model_name}/model.onnx"
        output_simple = f"./frozen_models/{model_name}/model_simple.onnx"
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path, opset=15)
        print(f"Model saved to {output_path}")
                
        # # NEW: Simplify the graph immediately
        # print("Simplifying...")
            
        # try:
        #     # Load the base model object
        #     model_onnx = onnx.load(output_path)

        #     # Simplify parameters
        #     # skip_fuse_bn=False: Merges BatchNormalization into Convolutions (Speedup!)
        #     # dynamic_input_shape=False: We want fixed shapes for maximum TensorRT speed
        #     model_simp, check = simplify(
        #         model_onnx,
        #         check_n=0,  # Disable runtime check if it crashes, or set to 3 to verify
        #         skip_fuse_bn=False,
        #         input_shapes={'x': [1, 8, 560]} # Explicitly tell it the new shape
        #     )

        #     if check:
        #         print("✅ Simplification Validated.")
        #     else:
        #         print("⚠️ Simplification succeeded but validation was skipped/failed.")

        #     onnx.save(model_simp, output_simple)
        #     print(f"🚀 Optimized model saved to: {output_simple}")

        # except Exception as e:
        #     print(f"❌ Simplification Failed: {e}")
        #     print("Using Base model instead.")

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

elif mode == 'exportSimplify':
    output_path = '/cs/projects/MWNaturalPredict/DL/predSWR/frozen_models/Tune_217_/model_CAD_217.onnx'
    output_simple = '/cs/projects/MWNaturalPredict/DL/predSWR/frozen_models/Tune_217_/model_CAD_217_simplified.onnx'
    os.environ["OMP_NUM_THREADS"] = "1" 
    # Prevents GPU conflicts during simplification
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
    
    import tf2onnx
    import onnx
    from onnxsim import simplify
    import tensorflow as tf
    # Convert the model to ONNX format
    # spec = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="x")]
    # spec = [tf.TensorSpec([1,92,8], model.inputs[0].dtype, name="x")]
    # spec = [tf.TensorSpec([4,44,8], model.inputs[0].dtype, name="x")]
    # spec = [tf.TensorSpec([1,44,8], model.inputs[0].dtype, name="x")]
    # spec = [tf.TensorSpec([1,560,8], model.inputs[0].dtype, name="x")]
    # spec = [tf.TensorSpec([1, 8, 560], model.inputs[0].dtype, name="x")]
    # pdb.set_trace()
    # model = onnx.load(output_path)
    # # output_path = f"./frozen_models/{model_name}/model.onnx"
    # # output_simple = f"./frozen_models/{model_name}/model_simple.onnx"
    # # model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path, opset=15)
    # print(f"Model saved to {output_path}")
            
    # NEW: Simplify the graph immediately
    print("Simplifying...")
        
    try:
        # Load the base model object
        model_onnx = onnx.load(output_path)
        spec = [tf.TensorSpec([1,560,8], tf.float32, name="x")]
        # Simplify parameters
        # skip_fuse_bn=False: Merges BatchNormalization into Convolutions (Speedup!)
        # dynamic_input_shape=False: We want fixed shapes for maximum TensorRT speed
        model_simp, check = simplify(
            model_onnx,
            check_n=0,  # Disable runtime check if it crashes, or set to 3 to verify
            skip_fuse_bn=False,
            input_shapes={'x': [1, 560, 8]} # Explicitly tell it the new shape
            # input_shapes={'x': [1, 8, 560]} # Explicitly tell it the new shape
        )

        if check:
            print("✅ Simplification Validated.")
        else:
            print("⚠️ Simplification succeeded but validation was skipped/failed.")

        onnx.save(model_simp, output_simple)
        print(f"🚀 Optimized model saved to: {output_simple}")

    except Exception as e:
        print(f"❌ Simplification Failed: {e}")
        print("Using Base model instead.")
    
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
    from optuna.storages import RDBStorage
    from optuna.samplers import GPSampler, TPESampler

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    tag = args.tag[0]
    param_dir = f'params_{tag}'

    if not os.path.exists(f'studies/{param_dir}'):
        os.makedirs(f'studies/{param_dir}')
    else:
        print(f"Directory studies/{param_dir} already exists. Using existing directory.")
        pdb.set_trace()

    # Copy base model code to study directory for reproducibility
    model_dir = f"studies/{param_dir}/base_model"
    shutil.copytree('./model', model_dir)
    shutil.copy2('./pred.py', f"{model_dir}/pred.py")

    # robust SQLite storage
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///studies/{param_dir}/{param_dir}.db?timeout=300",  # move timeout to URL
        heartbeat_interval=30,
        grace_period=600,
        failed_trial_callback=lambda study_id, trial_id: True,
        engine_kwargs={
            "connect_args": {
                # keep it simple for SQLite; don't set isolation_level here
                # "check_same_thread": False,  # optional if you ever use threads
            }
        },
    )
    # sampler = TPESampler(
    #     multivariate=True,      # joint modeling across params
    #     group=True,             # handles conditional/subspace grouping
    #     constant_liar=True,     # parallel-friendly; avoids collisions
    #     n_startup_trials=40,    # pure random warmup (bump if your space is large)
    #     n_ei_candidates=24,     # more candidate draws for better proposals
    #     seed=1337,
    # )
    sampler = NSGAIISampler(
        population_size=36,   # High diversity to utilize 18 parallel streams
        mutation_prob=0.2,   # Slightly increased (default 0.1) to keep discovering new logic over weeks
        crossover_prob=0.9,   # Keep high to combine features of "Fast" and "Accurate" models
        seed=1337
    )
    # pruner = optuna.pruners.HyperbandPruner(
    #         min_resource=30,    # Give model 30 epochs before judging
    #         max_resource=500,   # Max epochs
    #         reduction_factor=3  # Check at 30 -> 90 -> 270...
    #     )    
    study = optuna.create_study(
        study_name=param_dir,
        storage=storage,
        directions=["maximize", "minimize"],  # PR-AUC up, FP/min down
        load_if_exists=True,
        sampler=sampler,
        # pruner=pruner,
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

    study = optuna.load_study(
        study_name=param_dir,
        storage=f"sqlite:///studies/{param_dir}/{param_dir}.db",
    )
    # Optimize for 1000 trials
    if 'TripletOnly'.lower() in tag.lower():
        # from model.study_objectives import objective_triplet as objective
        from model.study_objectives import objective_proxy as objective
        # objective = objective_triplet
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
        lambda trial: objective(trial, model_name, tag, logger),
        n_trials=80,
        gc_after_trial=True,
        show_progress_bar=True,
        catch=(tf.errors.ResourceExhaustedError,),
        callbacks=[lambda study, trial: logger.info(f"Trial {trial.number} finished")]
    )

elif mode == 'tune_viz_multi_v5':
    import os, sys, json, math, warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import optuna
    import glob

    # Optional: Plotly
    try:
        import plotly.express as px
        HAVE_PLOTLY = True
    except Exception:
        HAVE_PLOTLY = False

    # ---------- CONFIG ----------
    tag = args.tag[0]
    param_dir = f'params_{tag}'
    storage_url = f"sqlite:///studies/{param_dir}/{param_dir}.db"
    viz_dir = f"studies/{param_dir}/visualizations_v5"
    param_impact_dir = os.path.join(viz_dir, "param_impact")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(param_impact_dir, exist_ok=True)

    # Default mapping (objective_triplet returns [PR-AUC (max), FP/min (min), LatencyScore (max)])
    OBJECTIVE_INDEX = dict(pr_auc=0, fp_per_min=1, latency=2)

    print(f"Loading study '{param_dir}' from {storage_url}")
    try:
        study = optuna.load_study(study_name=param_dir, storage=storage_url)
    except Exception as e:
        print(f"Error loading study: {e}")
        sys.exit(1)

    # ---------- COLLECT COMPLETED TRIALS ----------
    trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not trials:
        print("No completed trials; nothing to visualize.")
        sys.exit(0)

    # Detect dimensionality
    nvals = max((len(t.values) if t.values else 0) for t in trials)
    if nvals < 3:
        print(f"Trials have only {nvals} objectives; need at least 3 (PR-AUC, FP/min, LatencyScore).")
        sys.exit(0)
    if nvals > 3:
        print(f"Detected {nvals} objectives; using OBJECTIVE_INDEX mapping {OBJECTIVE_INDEX}. Adjust if needed.")

    # Constraint feasibility check for hypervolume plotting
    def _get_constraints(tr):
        return tr.system_attrs.get("constraints") or tr.user_attrs.get("constraints")
    has_constraints = any(_get_constraints(t) is not None for t in trials)
    feasible_trials = [
        t for t in trials
        if (_get_constraints(t) is None) or all((c is not None) and (c <= 0) for c in _get_constraints(t))
    ]
    DO_HV = not (has_constraints and len(feasible_trials) == 0)
    if has_constraints and not DO_HV:
        print("No feasible trials under constraints; skipping hypervolume plot.")

    # Collect rows from trials
    rows = []
    for t in trials:
        if not t.values:
            continue
        try:
            pr = float(t.values[OBJECTIVE_INDEX['pr_auc']])
            fp = float(t.values[OBJECTIVE_INDEX['fp_per_min']])
            la = float(t.values[OBJECTIVE_INDEX['latency']])
        except Exception:
            continue
        bad = lambda x: (x is None) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
        if any(bad(x) for x in [pr, fp, la]):
            continue
        rec = {"trial_number": t.number,
               "val_sample_pr_auc": pr,
               "val_latency_score": la,
               "val_fp_per_min": fp}
        rec.update(t.params)
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid completed trials with 3 objectives.")
        sys.exit(0)

    # Auto-detect and correct swapped LatencyScore ↔ FP/min (LatencyScore must be in [0,1])
    mapping_notice = ""
    try:
        q95_lat = float(np.nanquantile(df["val_latency_score"], 0.95)) if not df["val_latency_score"].isnull().all() else np.nan
        q95_fp  = float(np.nanquantile(df["val_fp_per_min"], 0.95)) if not df["val_fp_per_min"].isnull().all() else np.nan
        if (not math.isnan(q95_lat) and q95_lat > 1.05) and (not math.isnan(q95_fp) and q95_fp <= 1.05):
            tmp = df["val_latency_score"].copy()
            df["val_latency_score"] = df["val_fp_per_min"]
            df["val_fp_per_min"] = tmp
            mapping_notice = "Auto-corrected objective labels: swapped FP/min and LatencyScore based on value ranges (LatencyScore must be in [0,1])."
            print(mapping_notice)
    except Exception as e:
        print(f"Objective range check warning: {e}")

    # Save all trials CSV
    all_csv = os.path.join(viz_dir, "all_completed_trials.csv")
    df.to_csv(all_csv, index=False)
    print(f"Saved {len(df)} trials → {all_csv}")

    # ---------- STANDARD OPTUNA VISUALS ----------
    print("Generating Optuna standard plots...")
    try:
        # History per objective
        fig_hist_pr = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'),
            target_name="Sample PR-AUC")
        fig_hist_pr.write_html(os.path.join(viz_dir, "history_pr_auc.html"))

        fig_hist_fp = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['fp_per_min']] if t.values else float('nan'),
            target_name="FP/min")
        fig_hist_fp.write_html(os.path.join(viz_dir, "history_fp_per_min.html"))

        fig_hist_lat = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['latency']] if t.values else float('nan'),
            target_name="LatencyScore")
        fig_hist_lat.write_html(os.path.join(viz_dir, "history_latency.html"))

        # Param importances per objective
        fig_imp_pr = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'),
            target_name="Sample PR-AUC")
        fig_imp_pr.write_html(os.path.join(viz_dir, "param_importances_pr_auc.html"))

        fig_imp_fp = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['fp_per_min']] if t.values else float('nan'),
            target_name="FP/min")
        fig_imp_fp.write_html(os.path.join(viz_dir, "param_importances_fp_per_min.html"))

        fig_imp_lat = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['latency']] if t.values else float('nan'),
            target_name="LatencyScore")
        fig_imp_lat.write_html(os.path.join(viz_dir, "param_importances_latency.html"))

        # Pareto front — target_names must match study value order
        names_by_index = [""] * nvals
        names_by_index[OBJECTIVE_INDEX['pr_auc']] = "Sample PR-AUC"
        names_by_index[OBJECTIVE_INDEX['fp_per_min']] = "FP/min"
        names_by_index[OBJECTIVE_INDEX['latency']] = "LatencyScore"
        fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=names_by_index)
        fig_pareto.write_html(os.path.join(viz_dir, "pareto_front_3obj.html"))
    except Exception as e:
        print(f"Standard plot warning: {e}")

    # ---------- EDFs ----------
    try:
        fig_edf_pr = optuna.visualization.plot_edf(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'),
            target_name="Sample PR-AUC (↑)")
        fig_edf_pr.write_html(os.path.join(viz_dir, "edf_pr_auc.html"))

        fig_edf_fp = optuna.visualization.plot_edf(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['fp_per_min']] if t.values else float('nan'),
            target_name="FP/min (↓)")
        fig_edf_fp.write_html(os.path.join(viz_dir, "edf_fp_per_min.html"))

        fig_edf_lat = optuna.visualization.plot_edf(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['latency']] if t.values else float('nan'),
            target_name="LatencyScore (↑)")
        fig_edf_lat.write_html(os.path.join(viz_dir, "edf_latency.html"))
    except Exception as e:
        print(f"EDF plot warning: {e}")

    # ---------- PARETO SET & CSV/HTML ----------
    pareto_trials = study.best_trials
    pareto_nums = [t.number for t in pareto_trials]
    pareto_df = df[df.trial_number.isin(pareto_nums)].copy()
    pareto_csv = os.path.join(viz_dir, "pareto_trials.csv")
    pareto_df.to_csv(pareto_csv, index=False)
    print(f"Pareto set size: {len(pareto_df)} → {pareto_csv}")

    pareto_snapshot_html = ""

    def _trial_link(trial_no: int) -> str:
        base = os.path.join("studies", param_dir)
        matches = glob.glob(os.path.join(base, f"study_{trial_no}_*"))
        if matches:
            rel = os.path.relpath(matches[0], viz_dir)
            return f'<a href="{rel}">study_{trial_no}</a>'
        return ""

    if not pareto_df.empty:
        pareto_view = pareto_df.copy()
        pareto_view["study_dir"] = pareto_view["trial_number"].apply(_trial_link)
        lead_cols = ["trial_number", "val_sample_pr_auc", "val_latency_score", "val_fp_per_min", "study_dir"]
        remaining = [c for c in pareto_view.columns if c not in lead_cols]
        pareto_html = pareto_view[lead_cols + remaining].to_html(escape=False, index=False)
        with open(os.path.join(viz_dir, "pareto_trials.html"), "w") as fh:
            fh.write(f"""<html><head><meta charset="utf-8"><title>Pareto Trials — {study.study_name}</title>
<style>body{{font-family:Arial;margin:20px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px}} th{{background:#f5f5f5}}</style>
</head><body><h2>Pareto Trials</h2>
<p>Higher is better: PR-AUC, LatencyScore. Lower is better: FP/min.</p>
{pareto_html}
</body></html>""")
        # Snapshot: top 15 by PR-AUC
        snapshot = pareto_view.sort_values("val_sample_pr_auc", ascending=False)[lead_cols].head(15)
        pareto_snapshot_html = snapshot.to_html(escape=False, index=False)

    # ---------- HYPERVOLUME HISTORY ----------
    try:
        if DO_HV:
            ref_point = [0.0] * nvals
            ref_point[OBJECTIVE_INDEX['pr_auc']] = float(df["val_sample_pr_auc"].min() - 1e-3)  # smaller worse
            ref_point[OBJECTIVE_INDEX['fp_per_min']] = float(df["val_fp_per_min"].max() + 1e-3) # larger worse
            ref_point[OBJECTIVE_INDEX['latency']] = float(df["val_latency_score"].min() - 1e-3) # smaller worse
            fig_hv = optuna.visualization.plot_hypervolume_history(study, reference_point=ref_point)
            fig_hv.write_html(os.path.join(viz_dir, "hypervolume_history.html"))
        else:
            print("Hypervolume plot skipped due to infeasible trials under constraints.")
    except Exception as e:
        print(f"Hypervolume plot skipped: {e}")

    # ---------- 3D PROJECTION ----------
    try:
        if HAVE_PLOTLY:
            fig3d = px.scatter_3d(
                df,
                x="val_sample_pr_auc", y="val_latency_score", z="val_fp_per_min",
                color=np.where(df.trial_number.isin(pareto_nums), "Pareto", "Other"),
                hover_name="trial_number", opacity=0.85
            )
            # Clamp axes for clarity
            x_range = [0, 1]
            y_range = [0, 1]
            z_min, z_max = float(df["val_fp_per_min"].min()), float(df["val_fp_per_min"].max())
            fig3d.update_layout(
                scene=dict(
                    xaxis_title="PR-AUC (↑)",
                    yaxis_title="LatencyScore (↑)",
                    zaxis_title="FP/min (↓)",
                    xaxis=dict(range=x_range),
                    yaxis=dict(range=y_range),
                    zaxis=dict(range=[z_min, z_max])
                ),
                legend_title_text="Trials"
            )
            fig3d.write_html(os.path.join(viz_dir, "scatter3d_all.html"))
    except Exception as e:
        print(f"Projection plot warning: {e}")

    # ---------- PARAMETER IMPACT (per-param quick plots) ----------
    def qylim(series, lo=0.05, hi=0.95, pad=0.05, clamp=None):
        if series.isnull().all():
            return (0, 1)
        qlo, qhi = np.nanquantile(series, [lo, hi])
        span = max(1e-9, qhi - qlo)
        lo_v = qlo - pad * span
        hi_v = qhi + pad * span
        if clamp:
            lo_v = max(clamp[0], lo_v); hi_v = min(clamp[1], hi_v)
            if lo_v >= hi_v: lo_v, hi_v = clamp[0], clamp[1]
        return (lo_v, hi_v)

    pr_ylim  = qylim(df["val_sample_pr_auc"], clamp=(0,1))
    lat_ylim = qylim(df["val_latency_score"], clamp=(0,1))
    fp_ylim  = qylim(df["val_fp_per_min"], clamp=None)

    hyperparams = [c for c in df.columns if c not in
                   ["trial_number","val_sample_pr_auc","val_latency_score","val_fp_per_min"]]

    for p in hyperparams:
        if df[p].isnull().all() or df[p].nunique() <= 1:
            continue
        is_num = pd.api.types.is_numeric_dtype(df[p])
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                 gridspec_kw={"height_ratios":[2,1]})
        ax_top, ax_bot = axes

        # Top: PR-AUC and LatencyScore on twin axes
        if is_num:
            sns.scatterplot(data=df, x=p, y="val_sample_pr_auc", ax=ax_top, color="tab:blue", s=18, alpha=0.5, label="PR-AUC")
        else:
            sns.stripplot(data=df, x=p, y="val_sample_pr_auc", ax=ax_top, color="tab:blue", size=4, alpha=0.7, jitter=True)
        ax_top.set_ylabel("PR-AUC (↑)", color="tab:blue"); ax_top.tick_params(axis='y', labelcolor="tab:blue"); ax_top.set_ylim(pr_ylim)

        ax2 = ax_top.twinx()
        if is_num:
            sns.scatterplot(data=df, x=p, y="val_latency_score", ax=ax2, color="tab:green", s=18, alpha=0.5, label="LatencyScore")
        else:
            sns.stripplot(data=df, x=p, y="val_latency_score", ax=ax2, color="tab:green", size=4, alpha=0.7, jitter=True)
        ax2.set_ylabel("LatencyScore (↑)", color="tab:green"); ax2.tick_params(axis='y', labelcolor="tab:green"); ax2.set_ylim(lat_ylim)

        # Bottom: FP/min
        if is_num:
            sns.scatterplot(data=df, x=p, y="val_fp_per_min", ax=ax_bot, color="tab:red", s=18, alpha=0.5)
        else:
            sns.stripplot(data=df, x=p, y="val_fp_per_min", ax=ax_bot, color="tab:red", size=4, alpha=0.7, jitter=True)
        ax_bot.set_ylabel("FP/min (↓)", color="tab:red"); ax_bot.tick_params(axis='y', labelcolor="tab:red"); ax_bot.set_ylim(fp_ylim)

        if not is_num:
            for ax in (ax_top, ax_bot):
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax_bot.set_xlabel(p)
        fig.tight_layout()
        outp = os.path.join(param_impact_dir, f"{p}_impact.png")
        fig.savefig(outp, dpi=130)
        plt.close(fig)

    # ---------- QUANTILE TREND PLOTS (StopGrad split on the same axes) ----------
    qt_dir = os.path.join(viz_dir, "quantile_trends_stopgrad")
    os.makedirs(qt_dir, exist_ok=True)

    def _infer_stopgrad_column(df_in: pd.DataFrame) -> pd.Series:
        if "USE_StopGrad" in df_in.columns:
            col = df_in["USE_StopGrad"]
            if pd.api.types.is_bool_dtype(col):
                return np.where(col, "SG=True", "SG=False")
            try:
                val = pd.to_numeric(col, errors="coerce")
                return np.where(val == 1, "SG=True",
                                np.where(val == 0, "SG=False", "SG=unknown"))
            except Exception:
                s = col.astype(str).str.lower()
                return np.where(s.isin(["1","true","yes","y"]), "SG=True",
                                np.where(s.isin(["0","false","no","n"]), "SG=False", "SG=unknown"))
        elif "TYPE_ARCH" in df_in.columns:
            s = df_in["TYPE_ARCH"].astype(str)
            return np.where(s.str.contains("StopGrad", case=False, na=False),
                            "SG=True", "SG=False")
        return ["SG=unknown"] * len(df_in)

    df["COHORT_StopGrad"] = _infer_stopgrad_column(df)

    def _quantile_bins(x: pd.Series, nbins=12):
        x = x.astype(float)
        x = x.dropna()
        if x.empty: 
            return None
        qs = np.linspace(0, 1, nbins + 1)
        edges = np.unique(np.nanquantile(x, qs))
        if len(edges) < 4:
            lo, hi = np.nanmin(x), np.nanmax(x)
            edges = np.linspace(lo, hi, 4)
        eps = 1e-12 * (edges[-1] - edges[0] + 1.0)
        edges[0] -= eps; edges[-1] += eps
        return edges

    def _trend_by_bins(dfin: pd.DataFrame, param: str, nbins=12):
        dfin = dfin.dropna(subset=[param])  # fix length mismatch
        edges = _quantile_bins(dfin[param], nbins=nbins)
        if edges is None:
            return pd.DataFrame()
        b = pd.cut(dfin[param].astype(float), bins=edges, include_lowest=True)
        mids = b.apply(lambda iv: np.mean([iv.left, iv.right]) if pd.notnull(iv) else np.nan)
        tmp = dfin.copy()
        tmp["_bin"] = b
        tmp["_mid"] = mids
        long = tmp.melt(
            id_vars=["_bin","_mid","COHORT_StopGrad"],
            value_vars=["val_sample_pr_auc","val_latency_score","val_fp_per_min"],
            var_name="metric", value_name="val"
        )
        agg = (long.dropna(subset=["_bin","_mid","val"])
                    .groupby(["_bin","_mid","COHORT_StopGrad","metric"], observed=True)
                    .agg(q10=("val", lambda s: np.nanquantile(s,0.10)),
                        median=("val","median"),
                        q90=("val", lambda s: np.nanquantile(s,0.90)),
                        count=("val","size"))
                    .reset_index())
        agg.rename(columns={"_mid":"bin_mid"}, inplace=True)
        return agg

    def _plot_trend_overlay(agg: pd.DataFrame, p: str, out_png: str):
        order = ["val_sample_pr_auc","val_latency_score","val_fp_per_min"]
        titles = {"val_sample_pr_auc":"PR-AUC (↑)",
                "val_latency_score":"LatencyScore (↑)",
                "val_fp_per_min":"FP/min (↓)"}
        colors = {"SG=False":"tab:blue","SG=True":"tab:orange","SG=unknown":"tab:gray"}
        fig, axes = plt.subplots(3,1,figsize=(9,10),sharex=True)
        for ax, m in zip(axes, order):
            d = agg[agg["metric"] == m]
            for g, dd in d.groupby("COHORT_StopGrad"):
                dd = dd.sort_values("bin_mid")
                c = colors.get(g,"tab:gray")
                ax.plot(dd["bin_mid"], dd["median"], label=g, color=c, lw=2)
                ax.fill_between(dd["bin_mid"], dd["q10"], dd["q90"], alpha=0.15, color=c)
            ax.set_ylabel(titles[m]); ax.grid(alpha=0.3)
        axes[0].legend(title="StopGrad")
        axes[-1].set_xlabel(p)
        fig.suptitle(f"Quantile trend (StopGrad overlay): {p}")
        fig.tight_layout(rect=[0,0,1,0.96])
        fig.savefig(out_png, dpi=140); plt.close(fig)

    num_params = [c for c in df.columns
                if c not in ["trial_number","val_sample_pr_auc","val_latency_score","val_fp_per_min","COHORT_StopGrad"]
                and pd.api.types.is_numeric_dtype(df[c])
                and df[c].nunique() > 8]

    for p in num_params:
        try:
            sub = df[["val_sample_pr_auc","val_latency_score","val_fp_per_min","COHORT_StopGrad",p]]
            agg = _trend_by_bins(sub, p, nbins=12)
            if not agg.empty and agg["bin_mid"].nunique() >= 3:
                outp = os.path.join(qt_dir, f"{p}_trend_stopgrad.png")
                _plot_trend_overlay(agg, p, outp)
        except Exception as e:
            print(f"[trend] Skipped {p}: {e}")

    # ---------- CORRELATIONS ----------
    # Spearman correlation: numeric hypers vs objectives
    num_cols = [c for c in hyperparams if pd.api.types.is_numeric_dtype(df[c])]
    heatmap_fp = None
    if num_cols:
        try:
            corr_df = df[num_cols + ["val_sample_pr_auc","val_latency_score","val_fp_per_min"]].corr(method='spearman')
            plt.figure(figsize=(max(8, 0.6*len(corr_df.columns)), max(6, 0.5*len(corr_df))))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=False)
            plt.title("Spearman correlation: numeric hyperparameters vs objectives")
            plt.tight_layout()
            heatmap_fp = os.path.join(viz_dir, "correlations_spearman.png")
            plt.savefig(heatmap_fp, dpi=130); plt.close()
        except Exception as e:
            print(f"Correlation heatmap warning: {e}")
            plt.close()

    obj_corr_fp = None
    try:
        obj_corr = df[["val_sample_pr_auc","val_latency_score","val_fp_per_min"]].corr(method='spearman')
        plt.figure(figsize=(4.8,4))
        sns.heatmap(obj_corr, annot=True, fmt=".2f", cmap="vlag", cbar=False,
                    xticklabels=["PR-AUC (↑)","LatencyScore (↑)","FP/min (↓)"],
                    yticklabels=["PR-AUC (↑)","LatencyScore (↑)","FP/min (↓)"])
        plt.tight_layout()
        obj_corr_fp = os.path.join(viz_dir, "objective_correlations.png")
        plt.savefig(obj_corr_fp, dpi=140); plt.close()
    except Exception as e:
        print(f"Objective correlation warning: {e}")
        plt.close()

    # ---------- SIMPLE COMBINED RANK (top-k) ----------
    def robust_minmax(x):
        q05, q95 = np.nanquantile(x, [0.05, 0.95])
        d = max(1e-9, q95 - q05); z = (x - q05) / d
        return np.clip(z, 0, 1)

    df["score_pr"] = df["val_sample_pr_auc"]          # higher better
    df["score_lat"] = df["val_latency_score"]         # higher better
    df["score_fp"] = 1 - robust_minmax(df["val_fp_per_min"])  # lower fp → higher score
    df["combined_avg"] = (df["score_pr"] + df["score_lat"] + df["score_fp"]) / 3.0
    top_combined = df.sort_values("combined_avg", ascending=False).head(25)
    top_combined.to_csv(os.path.join(viz_dir, "top25_combined_avg.csv"), index=False)

    # Top-by-objective HTML tables with links
    def _trial_link(trial_no: int) -> str:
        base = os.path.join("studies", param_dir)
        matches = glob.glob(os.path.join(base, f"study_{trial_no}_*"))
        if matches:
            rel = os.path.relpath(matches[0], viz_dir)
            return f'<a href="{rel}">study_{trial_no}</a>'
        return ""

    try:
        top_k = 25
        best_pr  = df.sort_values("val_sample_pr_auc", ascending=False).head(top_k).copy()
        best_lat = df.sort_values("val_latency_score", ascending=False).head(top_k).copy()
        best_fp  = df.sort_values("val_fp_per_min", ascending=True ).head(top_k).copy()
        for fname, d in [("top_by_pr_auc.html", best_pr),
                         ("top_by_latency.html", best_lat),
                         ("top_by_fpmin.html", best_fp),
                         ("top_by_combined.html", top_combined)]:
            dd = d.copy()
            dd["study_dir"] = dd["trial_number"].apply(_trial_link)
            lead = ["trial_number","val_sample_pr_auc","val_latency_score","val_fp_per_min","combined_avg","study_dir"]
            keep = [c for c in lead if c in dd.columns] + [c for c in dd.columns if c not in lead]
            html_tbl = dd[keep].to_html(escape=False, index=False)
            with open(os.path.join(viz_dir, fname), "w") as fh:
                fh.write(f"<html><head><meta charset='utf-8'><style>body{{font-family:Arial;margin:20px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px}} th{{background:#f5f5f5}}</style></head><body>{html_tbl}</body></html>")
    except Exception as e:
        print(f"Top-k HTML warning: {e}")

    # ---------- HTML REPORT ----------
    html = []
    html.append(f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Study Visualization (3-obj): {study.study_name}</title>
    <style>
    body{{font-family:Arial,Helvetica,sans-serif;margin:20px;}} h2{{margin-top:28px}}
    a{{color:#007bff;text-decoration:none}} a:hover{{text-decoration:underline}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
    .card{{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}}
    img{{max-width:100%}}
    .notice{{padding:10px;border:1px solid #ffc107;background:#fff3cd;border-radius:6px}}
    </style></head><body>
    <h1>Visualization — {study.study_name}</h1>
    <p>Objectives: <b>maximize</b> PR-AUC, <b>minimize</b> FP/min, <b>maximize</b> LatencyScore.</p>
    <p><b>Trials:</b> {len(df)} | <b>Pareto set:</b> {len(pareto_df)}</p>""")
    if mapping_notice:
        html.append(f'<p class="notice">{mapping_notice}</p>')
    if has_constraints and not DO_HV:
        html.append('<p class="notice">Note: No feasible trials under constraints; hypervolume history omitted.</p>')
    html.append(f"""
    <ul>
      <li><a href="all_completed_trials.csv">all_completed_trials.csv</a></li>
      <li><a href="pareto_trials.csv">pareto_trials.csv</a> &nbsp;|&nbsp; <a href="pareto_trials.html">Pareto table (HTML)</a></li>
      <li><a href="top25_combined_avg.csv">top25_combined_avg.csv</a></li>
      <li>Top-K (HTML): <a href="top_by_pr_auc.html">PR-AUC</a> · <a href="top_by_fpmin.html">FP/min</a> · <a href="top_by_latency.html">LatencyScore</a> · <a href="top_by_combined.html">Combined</a></li>
      <li>Objective EDFs: <a href="edf_pr_auc.html">PR-AUC</a> · <a href="edf_fp_per_min.html">FP/min</a> · <a href="edf_latency.html">LatencyScore</a></li>
    </ul>""")
    if pareto_snapshot_html:
        html.append("<h2>Pareto snapshot (top 15 by PR-AUC)</h2>")
        html.append(pareto_snapshot_html)
    html.append("""
    <h2>Standard Optuna Plots</h2>
    <div class="grid">
      <div class="card"><a href="history_pr_auc.html">Optimization History — PR-AUC</a></div>
      <div class="card"><a href="history_fp_per_min.html">Optimization History — FP/min</a></div>
      <div class="card"><a href="history_latency.html">Optimization History — LatencyScore</a></div>
      <div class="card"><a href="param_importances_pr_auc.html">Param Importances — PR-AUC</a></div>
      <div class="card"><a href="param_importances_fp_per_min.html">Param Importances — FP/min</a></div>
      <div class="card"><a href="param_importances_latency.html">Param Importances — LatencyScore</a></div>
      <div class="card"><a href="pareto_front_3obj.html">Pareto Front (interactive)</a></div>
      <div class="card"><a href="hypervolume_history.html">Hypervolume History</a></div>
    </div>
    <h2>Objective Projections</h2>
    <div class="grid">""")
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter3d_all.html")):
        html.append('<div class="card"><a href="scatter3d_all.html">3D Scatter (All trials, Pareto highlighted)</a></div>')
    html.append("</div>")

    if heatmap_fp or obj_corr_fp:
        html.append("<h2>Correlations</h2><div class='grid'>")
        if heatmap_fp:
            html.append(f'<div class="card"><img src="correlations_spearman.png" alt="Hyperparameter correlations"></div>')
        if obj_corr_fp and os.path.exists(obj_corr_fp):
            html.append(f'<div class="card"><img src="objective_correlations.png" alt="Objective correlations"></div>')
        html.append("</div>")

    html.append("<h2>Per-Parameter Impact</h2><div class='grid'>")
    for p in hyperparams:
        imgp = os.path.join("param_impact", f"{p}_impact.png")
        if os.path.exists(os.path.join(viz_dir, imgp)):
            html.append(f'<div class="card"><h3>{p}</h3><img src="{imgp}"></div>')
    html.append("</div></body></html>")

    with open(os.path.join(viz_dir, "index.html"), "w") as f:
        f.write("\n".join(html))

    # ---------- SUMMARY ----------
    stats = {
        "study_name": study.study_name,
        "n_completed_trials": int(len(df)),
        "n_pareto_trials": int(len(pareto_df)),
        "has_constraints": bool(has_constraints),
        "n_feasible_trials": int(len(feasible_trials)),
        "objectives": ["val_sample_pr_auc (max)", "val_fp_per_min (min)", "val_latency_score (max)"],
        "objective_index_map": OBJECTIVE_INDEX,
        "auto_swap_applied": bool(mapping_notice != "")
    }
    with open(os.path.join(viz_dir, "study_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Visualization complete → {viz_dir}")

    # =========================
    # EXTRA ANALYSIS ADD-ONS (robust)
    # =========================

    # integrate into HTML report
    html.append("<h2>Quantile Trends (StopGrad overlay)</h2><div class='grid'>")
    for p in num_params:
        imgp = os.path.join("quantile_trends_stopgrad", f"{p}_trend_stopgrad.png")
        if os.path.exists(os.path.join(viz_dir, imgp)):
            html.append(f'<div class="card"><h3>{p}</h3><img src="{imgp}"></div>')
    html.append("</div>")
    
    # Output folders
    extras_dir   = os.path.join(viz_dir, "extras");            os.makedirs(extras_dir, exist_ok=True)
    qplots_dir   = os.path.join(extras_dir, "quantile_trends"); os.makedirs(qplots_dir, exist_ok=True)
    heat2d_dir   = os.path.join(extras_dir, "heatmaps_2d");     os.makedirs(heat2d_dir, exist_ok=True)
    cohort_dir   = os.path.join(extras_dir, "cohorts");         os.makedirs(cohort_dir, exist_ok=True)
    recommend_dir= os.path.join(extras_dir, "range_recommendations"); os.makedirs(recommend_dir, exist_ok=True)
    recon_dir    = os.path.join(extras_dir, "importance_reconciliation"); os.makedirs(recon_dir, exist_ok=True)

    # Columns
    objective_cols = ["val_sample_pr_auc","val_latency_score","val_fp_per_min"]
    param_cols = [c for c in df.columns if c not in ["trial_number", *objective_cols,
                                                     "score_pr","score_lat","score_fp","combined_avg"]]
    num_params = [p for p in param_cols if pd.api.types.is_numeric_dtype(df[p])]
    cat_params = [p for p in param_cols if p not in num_params]

    # -------- StopGrad robust detector
    def _detect_stopgrad_series(dfx: pd.DataFrame) -> pd.Series:
        # Priority 1: explicit boolean/0-1 param
        if "USE_StopGrad" in dfx.columns:
            s = dfx["USE_StopGrad"]
            # Normalize to bool safely
            if s.dtype == bool:
                return s
            # Map strings/numbers to bool; NaNs -> False
            return s.map(lambda v: bool(v) if pd.notna(v) else False)
        # Priority 2: implied by TYPE_ARCH substring
        if "TYPE_ARCH" in dfx.columns:
            return dfx["TYPE_ARCH"].astype(str).str.lower().str.contains("stopgrad", na=False)
        return None  # not available

    stopgrad_series = _detect_stopgrad_series(df)

    # -------- Helpers
    def _safe_corr_heatmap(data: pd.DataFrame, name: str, out_png: str):
        try:
            cols = [*num_params, *objective_cols]
            cols = [c for c in cols if c in data.columns]
            if len(cols) < 3: return
            corr_df = data[cols].corr(method="spearman")
            plt.figure(figsize=(max(8, 0.6*len(corr_df.columns)), max(6, 0.5*len(corr_df))))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title(f"Spearman (numeric) — {name}")
            plt.tight_layout()
            plt.savefig(out_png, dpi=130); plt.close()
        except Exception:
            plt.close()

    def _param_summary_scatter(data: pd.DataFrame, name: str):
        # Per-objective, per-param quick summaries (robust to dtype/NaN)
        for obj, ylabel in [("val_sample_pr_auc","PR-AUC (↑)"),
                            ("val_latency_score","LatencyScore (↑)"),
                            ("val_fp_per_min","FP/min (↓)")]:
            if obj not in data.columns: continue
            plt.figure(figsize=(max(8, 0.4*len(param_cols)), 4))
            ax = plt.gca()
            x_idx, x_labs = [], []
            for i, p in enumerate(param_cols):
                if p not in data.columns or data[p].nunique(dropna=True) <= 1:
                    continue
                # Numeric → quantile means, Categorical → category means
                if pd.api.types.is_numeric_dtype(data[p]):
                    valid = data[[p, obj]].dropna()
                    if valid[p].nunique() < 3 or len(valid) < 8: 
                        continue
                    qbins = pd.qcut(valid[p], q=min(10, max(3, valid[p].nunique())),
                                    duplicates="drop")
                    means = valid.groupby(qbins, observed=True)[obj].mean().values
                    ax.scatter(np.full_like(means, len(x_idx)), means, s=26, alpha=0.7)
                else:
                    g = data.groupby(p, observed=True)[obj].mean().sort_values(ascending=False)
                    ax.scatter(np.full_like(g.values, len(x_idx)), g.values, s=26, alpha=0.7)
                x_idx.append(len(x_idx)); x_labs.append(p)
            ax.set_xticks(range(len(x_labs))); ax.set_xticklabels(x_labs, rotation=28, ha="right")
            ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} vs params — {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(cohort_dir, f"{obj}_vs_params_{name}.png"), dpi=120); plt.close()

    # -------- Cohort analysis (StopGrad ON/OFF)
    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)  # NaN → False
        d0 = df[~sg_mask].copy()  # StopGrad OFF
        d1 = df[ sg_mask].copy()  # StopGrad ON

        if len(d0) >= 30:
            _safe_corr_heatmap(d0, "StopGrad_OFF", os.path.join(cohort_dir, "corr_spearman_StopGrad_OFF.png"))
            _param_summary_scatter(d0, "StopGrad_OFF")
        if len(d1) >= 30:
            _safe_corr_heatmap(d1, "StopGrad_ON",  os.path.join(cohort_dir, "corr_spearman_StopGrad_ON.png"))
            _param_summary_scatter(d1, "StopGrad_ON")

    # -------- Quantile trend curves (robust)
    def quantile_trend(x: pd.Series, y: pd.Series, q=12):
        valid = x.notna() & y.notna()
        xv, yv = x[valid], y[valid]
        if len(xv) < 10 or xv.nunique() < 3: 
            return None
        bins = pd.qcut(xv, q=min(q, max(3, xv.nunique())), duplicates="drop")
        grp  = pd.DataFrame({"y": yv, "bin": bins, "x": xv}).groupby("bin", observed=True)
        mu = grp["y"].mean().values
        sd = grp["y"].std().values
        n  = grp["y"].size().values.astype(float)
        se = np.where(n>1, sd/np.sqrt(n), np.nan)
        xc = grp["x"].mean().values
        return xc, mu, se

    for p in num_params:
        fig, axs = plt.subplots(3,1,figsize=(8,9), sharex=True)
        ok = False
        for ax, obj, lab in zip(axs,
                                ["val_sample_pr_auc","val_latency_score","val_fp_per_min"],
                                ["PR-AUC (↑)", "LatencyScore (↑)","FP/min (↓)"]):
            if obj not in df.columns: 
                ax.set_ylabel(lab); continue
            out = quantile_trend(df[p], df[obj], q=12)
            if out is None:
                ax.set_ylabel(lab); continue
            x, mu, se = out
            ax.plot(x, mu, marker="o", linewidth=1.5)
            if np.isfinite(se).any():
                ax.fill_between(x, mu - 1.96*np.nan_to_num(se), mu + 1.96*np.nan_to_num(se), alpha=0.2)
            ax.set_ylabel(lab); ok = True
        axs[-1].set_xlabel(p)
        if ok:
            fig.suptitle(f"Quantile trend — {p}")
            fig.tight_layout(rect=[0,0,1,0.97])
            fig.savefig(os.path.join(qplots_dir, f"{p}_quantile_trends.png"), dpi=130)
        plt.close(fig)

    # -------- 2D interaction heatmaps (only if both axes exist and are informative)
    def heat2d(x: pd.Series, y: pd.Series, z: pd.Series, xq=12, yq=12):
        valid = x.notna() & y.notna() & z.notna()
        xv, yv, zv = x[valid], y[valid], z[valid]
        if xv.nunique() < 4 or yv.nunique() < 4 or len(zv) < 25:
            return None
        xb = pd.qcut(xv, q=min(xq, max(4, xv.nunique())), duplicates="drop")
        yb = pd.qcut(yv, q=min(yq, max(4, yv.nunique())), duplicates="drop")
        grid = pd.DataFrame({"xb": xb, "yb": yb, "z": zv}).groupby(["xb","yb"], observed=True)["z"].mean().unstack()
        return grid

    def _maybe_heatmap(xname, yname):
        if xname not in df.columns or yname not in df.columns: 
            return
        for obj, lab in [("val_sample_pr_auc","PR-AUC (↑)"),
                         ("val_latency_score","LatencyScore (↑)"),
                         ("val_fp_per_min","FP/min (↓)")]:
            if obj not in df.columns: continue
            H = heat2d(df[xname], df[yname], df[obj])
            if H is None: 
                continue
            plt.figure(figsize=(6.8,5.2))
            sns.heatmap(H, cmap="viridis", annot=False)
            plt.title(f"{lab} mean — {xname} × {yname}")
            plt.tight_layout()
            plt.savefig(os.path.join(heat2d_dir, f"{obj}_{xname}_x_{yname}.png"), dpi=140)
            plt.close()

    if "LOSS_SupCon" in df.columns and "LOSS_TupMPN" in df.columns:
        _maybe_heatmap("LOSS_SupCon", "LOSS_TupMPN")
    if "learning_rate" in df.columns and "LOSS_NEGATIVES" in df.columns:
        _maybe_heatmap("learning_rate", "LOSS_NEGATIVES")
    if "LOSS_TV" in df.columns and "LOSS_TupMPN" in df.columns:
        _maybe_heatmap("LOSS_TV", "LOSS_TupMPN")

    # -------- Range recommendations: top-quartile trial bands per objective
    def recommend_ranges(data: pd.DataFrame, name: str, top_frac=0.25):
        recs = {}
        for obj, asc, nice in [("val_sample_pr_auc", False, "PR-AUC (↑)"),
                               ("val_latency_score", False, "LatencyScore (↑)"),
                               ("val_fp_per_min", True,  "FP/min (↓)")]:
            if obj not in data.columns or data[obj].isnull().all():
                continue
            dsort = data.sort_values(obj, ascending=asc)
            top_n = max(20, int(len(dsort)*top_frac))
            top   = dsort.head(top_n)
            # numeric 20–80%
            rng = {}
            for p in num_params:
                if p not in top.columns: continue
                col = top[p].dropna()
                if col.nunique() < 2: 
                    continue
                try:
                    rng[p] = (float(np.nanquantile(col, 0.20)), float(np.nanquantile(col, 0.80)))
                except Exception:
                    pass
            # categorical modes
            cat = {}
            for p in cat_params:
                if p not in top.columns: continue
                vc = top[p].value_counts(normalize=True, dropna=False)
                if len(vc):
                    cat[p] = vc.head(3).to_dict()
            recs[nice] = {"num_quantile_20_80": rng, "cat_top3_props": cat, "n_top": int(len(top))}
        # write
        with open(os.path.join(recommend_dir, f"ranges_{name}.json"), "w") as fh:
            json.dump(recs, fh, indent=2)
        rows = []
        for obj, payload in recs.items():
            for p,(lo,hi) in payload["num_quantile_20_80"].items():
                rows.append({"objective": obj, "param": p, "q20": lo, "q80": hi})
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(recommend_dir, f"ranges_numeric_{name}.csv"), index=False)

    recommend_ranges(df, "ALL")
    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)
        if df[~sg_mask].shape[0] >= 40: recommend_ranges(df[~sg_mask], "StopGrad_OFF")
        if df[ sg_mask].shape[0] >= 40: recommend_ranges(df[ sg_mask], "StopGrad_ON")

    # -------- Importance reconciliation: absolute Spearman vs PR-AUC (global/cohort)
    imp_rows = []
    # Global
    for p in num_params:
        try:
            r = df[[p,"val_sample_pr_auc"]].dropna().corr(method="spearman").iloc[0,1]
            if pd.notna(r):
                imp_rows.append({"param": p, "source": "Spearman|rho| (global)", "value": abs(float(r))})
        except Exception:
            pass
    # Cohorts
    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)
        for name, dsub in [("StopGrad_OFF", df[~sg_mask]), ("StopGrad_ON", df[sg_mask])]:
            if len(dsub) < 30: 
                continue
            for p in num_params:
                try:
                    rr = dsub[[p,"val_sample_pr_auc"]].dropna().corr(method="spearman").iloc[0,1]
                    if pd.notna(rr):
                        imp_rows.append({"param": p, "source": f"Spearman|rho| ({name})", "value": abs(float(rr))})
                except Exception:
                    pass
    if imp_rows:
        imp_df = pd.DataFrame(imp_rows)
        piv = imp_df.pivot_table(index="param", columns="source", values="value", aggfunc="max").fillna(0.0)
        piv = piv.sort_values(by=list(piv.columns)[0], ascending=False)
        plt.figure(figsize=(max(8, 0.5*len(piv)), 5))
        piv.plot(kind="bar", figsize=(max(8, 0.55*len(piv)),5))
        plt.ylabel("|Spearman| vs PR-AUC"); plt.title("Correlation-based importances (compare with Optuna fANOVA)")
        plt.tight_layout()
        plt.savefig(os.path.join(recon_dir, "spearman_vs_pr_auc.png"), dpi=130)
        plt.close()

elif mode == 'tune_viz_multi_v6':
    import os, sys, json, math, warnings, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import optuna

    warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

    # Optional: Plotly
    try:
        import plotly.express as px
        HAVE_PLOTLY = True
    except Exception:
        HAVE_PLOTLY = False

    # ---------- CONFIG ----------
    tag = args.tag[0]
    param_dir = f'params_{tag}'
    storage_url = f"sqlite:///studies/{param_dir}/{param_dir}.db"
    viz_dir = f"studies/{param_dir}/visualizations_v5"
    os.makedirs(viz_dir, exist_ok=True)

    print(f"Loading study '{param_dir}' from {storage_url}")
    try:
        study = optuna.load_study(study_name=param_dir, storage=storage_url)
    except Exception as e:
        print(f"Error loading study: {e}")
        sys.exit(1)

    # ---------- COLLECT COMPLETED TRIALS ----------
    trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not trials:
        print("No completed trials; nothing to visualize.")
        sys.exit(0)

    # Detect dimensionality
    nvals = max((len(t.values) if t.values else 0) for t in trials)
    if nvals < 3:
        print(f"Trials have only {nvals} objectives; need at least 3 (PR-AUC, FP/min, LatencyScore).")
        sys.exit(0)

    # The objective order used when saving values
    OBJECTIVE_INDEX = dict(pr_auc=0, fp_per_min=1, latency=2)
    if nvals > 3:
        print(f"Detected {nvals} objectives; using OBJECTIVE_INDEX mapping {OBJECTIVE_INDEX}. Adjust if needed.")

    # Constraint feasibility check for hypervolume plotting
    def _get_constraints(tr):
        return tr.system_attrs.get("constraints") or tr.user_attrs.get("constraints")

    has_constraints = any(_get_constraints(t) is not None for t in trials)
    feasible_trials = [
        t for t in trials
        if (_get_constraints(t) is None) or all((c is not None) and (c <= 0) for c in _get_constraints(t))
    ]
    DO_HV = not (has_constraints and len(feasible_trials) == 0)
    if has_constraints and not DO_HV:
        print("No feasible trials under constraints; skipping hypervolume plot.")

    # Collect rows from trials
    rows = []
    for t in trials:
        if not t.values:
            continue
        try:
            pr = float(t.values[OBJECTIVE_INDEX['pr_auc']])
            fp = float(t.values[OBJECTIVE_INDEX['fp_per_min']])
            la = float(t.values[OBJECTIVE_INDEX['latency']])
        except Exception:
            continue
        bad = lambda x: (x is None) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
        if any(bad(x) for x in [pr, fp, la]):
            continue
        rec = {"trial_number": t.number,
               "val_sample_pr_auc": pr,
               "val_latency_score": la,
               "val_fp_per_min": fp}
        # pull params
        rec.update(t.params)
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid completed trials with 3 objectives.")
        sys.exit(0)

    # Auto-detect and correct swapped LatencyScore ↔ FP/min (LatencyScore must be in [0,1])
    mapping_notice = ""
    try:
        q95_lat = float(np.nanquantile(df["val_latency_score"], 0.95)) if "val_latency_score" in df else np.nan
        q95_fp  = float(np.nanquantile(df["val_fp_per_min"], 0.95)) if "val_fp_per_min" in df else np.nan
        if (not math.isnan(q95_lat) and q95_lat > 1.05) and (not math.isnan(q95_fp) and q95_fp <= 1.05):
            tmp = df["val_latency_score"].copy()
            df["val_latency_score"] = df["val_fp_per_min"]
            df["val_fp_per_min"] = tmp
            mapping_notice = "Auto-corrected objective labels: swapped FP/min and LatencyScore based on value ranges (LatencyScore must be in [0,1])."
            print(mapping_notice)
    except Exception as e:
        print(f"Objective range check warning: {e}")

    # Save all trials CSV
    all_csv = os.path.join(viz_dir, "all_completed_trials.csv")
    df.to_csv(all_csv, index=False)
    print(f"Saved {len(df)} trials → {all_csv}")

    # ---------- STANDARD OPTUNA VISUALS ----------
    print("Generating Optuna standard plots...")
    try:
        # History per objective
        fig_hist_pr = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'),
            target_name="Sample PR-AUC")
        fig_hist_pr.write_html(os.path.join(viz_dir, "history_pr_auc.html"))

        fig_hist_fp = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['fp_per_min']] if t.values else float('nan'),
            target_name="FP/min")
        fig_hist_fp.write_html(os.path.join(viz_dir, "history_fp_per_min.html"))

        fig_hist_lat = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['latency']] if t.values else float('nan'),
            target_name="LatencyScore")
        fig_hist_lat.write_html(os.path.join(viz_dir, "history_latency.html"))

        # Param importances per objective
        fig_imp_pr = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'),
            target_name="Sample PR-AUC")
        fig_imp_pr.write_html(os.path.join(viz_dir, "param_importances_pr_auc.html"))

        fig_imp_fp = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['fp_per_min']] if t.values else float('nan'),
            target_name="FP/min")
        fig_imp_fp.write_html(os.path.join(viz_dir, "param_importances_fp_per_min.html"))

        fig_imp_lat = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['latency']] if t.values else float('nan'),
            target_name="LatencyScore")
        fig_imp_lat.write_html(os.path.join(viz_dir, "param_importances_latency.html"))

        # Pareto front — target_names must match study value order
        names_by_index = [""] * nvals
        names_by_index[OBJECTIVE_INDEX['pr_auc']] = "Sample PR-AUC"
        names_by_index[OBJECTIVE_INDEX['fp_per_min']] = "FP/min"
        names_by_index[OBJECTIVE_INDEX['latency']] = "LatencyScore"
        fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=names_by_index)
        fig_pareto.write_html(os.path.join(viz_dir, "pareto_front_3obj.html"))
    except Exception as e:
        print(f"Standard plot warning: {e}")

    # ---------- EDFs ----------
    try:
        fig_edf_pr = optuna.visualization.plot_edf(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'),
            target_name="Sample PR-AUC (↑)")
        fig_edf_pr.write_html(os.path.join(viz_dir, "edf_pr_auc.html"))

        fig_edf_fp = optuna.visualization.plot_edf(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['fp_per_min']] if t.values else float('nan'),
            target_name="FP/min (↓)")
        fig_edf_fp.write_html(os.path.join(viz_dir, "edf_fp_per_min.html"))

        fig_edf_lat = optuna.visualization.plot_edf(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['latency']] if t.values else float('nan'),
            target_name="LatencyScore (↑)")
        fig_edf_lat.write_html(os.path.join(viz_dir, "edf_latency.html"))
    except Exception as e:
        print(f"EDF plot warning: {e}")

    # ---------- PARETO SET & CSV/HTML ----------
    pareto_trials = study.best_trials
    pareto_nums = [t.number for t in pareto_trials]
    pareto_df = df[df.trial_number.isin(pareto_nums)].copy()
    pareto_csv = os.path.join(viz_dir, "pareto_trials.csv")
    pareto_df.to_csv(pareto_csv, index=False)
    print(f"Pareto set size: {len(pareto_df)} → {pareto_csv}")

    pareto_snapshot_html = ""

    def _trial_link(trial_no: int) -> str:
        base = os.path.join("studies", param_dir)
        matches = glob.glob(os.path.join(base, f"study_{trial_no}_*"))
        if matches:
            rel = os.path.relpath(matches[0], viz_dir)
            return f'<a href="{rel}">study_{trial_no}</a>'
        return ""

    if not pareto_df.empty:
        _pareto_view = pareto_df.copy()
        _pareto_view["study_dir"] = _pareto_view["trial_number"].apply(_trial_link)
        lead_cols = ["trial_number", "val_sample_pr_auc", "val_latency_score", "val_fp_per_min", "study_dir"]
        remaining = [c for c in _pareto_view.columns if c not in lead_cols]
        pareto_html = _pareto_view[lead_cols + remaining].to_html(escape=False, index=False)
        with open(os.path.join(viz_dir, "pareto_trials.html"), "w") as fh:
            fh.write(f"""<html><head><meta charset="utf-8"><title>Pareto Trials — {study.study_name}</title>
<style>body{{font-family:Arial;margin:20px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px}} th{{background:#f5f5f5}}</style>
</head><body><h2>Pareto Trials</h2>
<p>Higher is better: PR-AUC, LatencyScore. Lower is better: FP/min.</p>
{pareto_html}
</body></html>""")
        snapshot = _pareto_view.sort_values("val_sample_pr_auc", ascending=False)[lead_cols].head(15)
        pareto_snapshot_html = snapshot.to_html(escape=False, index=False)

    # ---------- HYPERVOLUME HISTORY ----------
    try:
        if DO_HV:
            ref_point = [0.0] * nvals
            ref_point[OBJECTIVE_INDEX['pr_auc']] = float(df["val_sample_pr_auc"].min() - 1e-3)   # smaller worse
            ref_point[OBJECTIVE_INDEX['fp_per_min']] = float(df["val_fp_per_min"].max() + 1e-3)  # larger worse
            ref_point[OBJECTIVE_INDEX['latency']] = float(df["val_latency_score"].min() - 1e-3)  # smaller worse
            fig_hv = optuna.visualization.plot_hypervolume_history(study, reference_point=ref_point)
            fig_hv.write_html(os.path.join(viz_dir, "hypervolume_history.html"))
        else:
            print("Hypervolume plot skipped due to infeasible trials under constraints.")
    except Exception as e:
        print(f"Hypervolume plot skipped: {e}")

    # ---------- 3D PROJECTION ----------
    try:
        if HAVE_PLOTLY:
            fig3d = px.scatter_3d(
                df,
                x="val_sample_pr_auc", y="val_latency_score", z="val_fp_per_min",
                color=np.where(df.trial_number.isin(pareto_nums), "Pareto", "Other"),
                hover_name="trial_number", opacity=0.85
            )
            x_range = [0, 1]; y_range = [0, 1]
            z_min, z_max = float(df["val_fp_per_min"].min()), float(df["val_fp_per_min"].max())
            fig3d.update_layout(
                scene=dict(
                    xaxis_title="PR-AUC (↑)",
                    yaxis_title="LatencyScore (↑)",
                    zaxis_title="FP/min (↓)",
                    xaxis=dict(range=x_range),
                    yaxis=dict(range=y_range),
                    zaxis=dict(range=[z_min, z_max])
                ),
                legend_title_text="Trials"
            )
            fig3d.write_html(os.path.join(viz_dir, "scatter3d_all.html"))
    except Exception as e:
        print(f"Projection plot warning: {e}")

    # ---------- PER-PARAM IMPACT (quick plots) ----------
    param_impact_dir = os.path.join(viz_dir, "param_impact")
    os.makedirs(param_impact_dir, exist_ok=True)

    def qylim(series, lo=0.05, hi=0.95, pad=0.05, clamp=None):
        if series.isnull().all():
            return (0, 1)
        qlo, qhi = np.nanquantile(series, [lo, hi])
        span = max(1e-9, qhi - qlo)
        lo_v = qlo - pad * span
        hi_v = qhi + pad * span
        if clamp:
            lo_v = max(clamp[0], lo_v); hi_v = min(clamp[1], hi_v)
            if lo_v >= hi_v: lo_v, hi_v = clamp[0], clamp[1]
        return (lo_v, hi_v)

    pr_ylim  = qylim(df["val_sample_pr_auc"], clamp=(0,1))
    lat_ylim = qylim(df["val_latency_score"], clamp=(0,1))
    fp_ylim  = qylim(df["val_fp_per_min"], clamp=None)

    hyperparams = [c for c in df.columns if c not in
                   ["trial_number","val_sample_pr_auc","val_latency_score","val_fp_per_min"]]

    for p in hyperparams:
        if df[p].isnull().all() or df[p].nunique(dropna=True) <= 1:
            continue
        is_num = pd.api.types.is_numeric_dtype(df[p])
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                 gridspec_kw={"height_ratios":[2,1]})
        ax_top, ax_bot = axes

        # Top: PR-AUC and LatencyScore on twin axes
        if is_num:
            sns.scatterplot(data=df, x=p, y="val_sample_pr_auc", ax=ax_top, color="tab:blue", s=18, alpha=0.5, label="PR-AUC")
        else:
            sns.stripplot(data=df, x=p, y="val_sample_pr_auc", ax=ax_top, color="tab:blue", size=4, alpha=0.7, jitter=True)
        ax_top.set_ylabel("PR-AUC (↑)", color="tab:blue"); ax_top.tick_params(axis='y', labelcolor="tab:blue"); ax_top.set_ylim(pr_ylim)

        ax2 = ax_top.twinx()
        if is_num:
            sns.scatterplot(data=df, x=p, y="val_latency_score", ax=ax2, color="tab:green", s=18, alpha=0.5, label="LatencyScore")
        else:
            sns.stripplot(data=df, x=p, y="val_latency_score", ax=ax2, color="tab:green", size=4, alpha=0.7, jitter=True)
        ax2.set_ylabel("LatencyScore (↑)", color="tab:green"); ax2.tick_params(axis='y', labelcolor="tab:green"); ax2.set_ylim(lat_ylim)

        # Bottom: FP/min
        if is_num:
            sns.scatterplot(data=df, x=p, y="val_fp_per_min", ax=ax_bot, color="tab:red", s=18, alpha=0.5)
        else:
            sns.stripplot(data=df, x=p, y="val_fp_per_min", ax=ax_bot, color="tab:red", size=4, alpha=0.7, jitter=True)
        ax_bot.set_ylabel("FP/min (↓)", color="tab:red"); ax_bot.tick_params(axis='y', labelcolor="tab:red"); ax_bot.set_ylim(fp_ylim)

        if not is_num:
            for ax in (ax_top, ax_bot):
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax_bot.set_xlabel(p)
        fig.tight_layout()
        outp = os.path.join(param_impact_dir, f"{p}_impact.png")
        fig.savefig(outp, dpi=130)
        plt.close(fig)

    # ---------- STOPGRAD QUANTILE TRENDS (overlay + facets) ----------
    qt_dir = os.path.join(viz_dir, "quantile_trends_stopgrad")
    os.makedirs(qt_dir, exist_ok=True)

    def _infer_stopgrad_column(df_in: pd.DataFrame) -> pd.Series:
        if "USE_StopGrad" in df_in.columns:
            col = df_in["USE_StopGrad"]
            if pd.api.types.is_bool_dtype(col):
                return pd.Series(np.where(col, "SG=True", "SG=False"), index=col.index)
            try:
                val = pd.to_numeric(col, errors="coerce")
                return pd.Series(np.where(val == 1, "SG=True",
                                   np.where(val == 0, "SG=False", "SG=unknown")), index=col.index)
            except Exception:
                s = col.astype(str).str.lower()
                return pd.Series(np.where(s.isin(["1","true","yes","y"]), "SG=True",
                                   np.where(s.isin(["0","false","no","n"]), "SG=False", "SG=unknown")), index=col.index)
        if "TYPE_ARCH" in df_in.columns:
            s = df_in["TYPE_ARCH"].astype(str).str.lower()
            return pd.Series(np.where(s.str.contains("stopgrad", na=False), "SG=True", "SG=False"), index=df_in.index)
        return pd.Series(["SG=unknown"] * len(df_in), index=df_in.index)

    df["COHORT_StopGrad"] = _infer_stopgrad_column(df)

    def _quantile_edges(x: pd.Series, nbins=12):
        x = pd.to_numeric(x, errors="coerce").dropna()
        if x.nunique() < 3 or len(x) < 12:
            if len(x) == 0:
                return None
            lo, hi = float(x.min()), float(x.max())
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                return None
            return np.linspace(lo, hi, 4)
        qs = np.linspace(0, 1, nbins + 1)
        edges = np.unique(np.nanquantile(x, qs))
        if len(edges) < 4:
            lo, hi = float(x.min()), float(x.max())
            edges = np.linspace(lo, hi, 4)
        eps = 1e-12 * (edges[-1] - edges[0] + 1.0)
        edges[0] -= eps; edges[-1] += eps
        return edges

    def _trend_by_bins_cohort(dfin: pd.DataFrame, param: str, nbins=12):
        keep_cols = ["val_sample_pr_auc","val_latency_score","val_fp_per_min","COHORT_StopGrad", param]
        dfin = dfin[keep_cols].copy()
        dfin[param] = pd.to_numeric(dfin[param], errors="coerce")
        dfin = dfin.dropna(subset=[param])

        edges = _quantile_edges(dfin[param], nbins=nbins)
        if edges is None:
            return pd.DataFrame()

        b = pd.cut(dfin[param], bins=edges, include_lowest=True)
        mids = b.apply(lambda iv: np.mean([iv.left, iv.right]) if pd.notnull(iv) else np.nan)
        tmp = dfin.copy()
        tmp["_bin"] = b
        tmp["_mid"] = mids

        long = tmp.melt(
            id_vars=["_bin","_mid","COHORT_StopGrad"],
            value_vars=["val_sample_pr_auc","val_latency_score","val_fp_per_min"],
            var_name="metric", value_name="val"
        )

        agg = (long.dropna(subset=["_bin","_mid","val"])
                    .groupby(["_bin","_mid","COHORT_StopGrad","metric"], observed=True)
                    .agg(q10=("val", lambda s: np.nanquantile(s, 0.10)),
                         median=("val", "median"),
                         q90=("val", lambda s: np.nanquantile(s, 0.90)),
                         count=("val", "size"))
                    .reset_index())
        agg.rename(columns={"_mid": "bin_mid"}, inplace=True)
        return agg

    def _plot_trend_overlay(agg: pd.DataFrame, p: str, out_png: str):
        order  = ["val_sample_pr_auc","val_latency_score","val_fp_per_min"]
        titles = {"val_sample_pr_auc": "PR-AUC (↑)",
                  "val_latency_score": "LatencyScore (↑)",
                  "val_fp_per_min":    "FP/min (↓)"}
        colors = {"SG=False":"tab:blue","SG=True":"tab:orange","SG=unknown":"tab:gray"}

        if agg.empty or agg["bin_mid"].nunique() < 3:
            return False

        fig, axes = plt.subplots(3,1,figsize=(9,10), sharex=True)
        for ax, m in zip(axes, order):
            d = agg[agg["metric"] == m]
            ok_any = False
            for g, dd in d.groupby("COHORT_StopGrad"):
                dd = dd.sort_values("bin_mid")
                if dd["bin_mid"].nunique() < 3:
                    continue
                c = colors.get(g, "tab:gray")
                ax.plot(dd["bin_mid"], dd["median"], label=g, color=c, lw=2)
                ax.fill_between(dd["bin_mid"], dd["q10"], dd["q90"], alpha=0.18, color=c)
                for xm, cnt in zip(dd["bin_mid"], dd["count"]):
                    ax.text(xm, dd["median"].min(), f"n={int(cnt)}", fontsize=7, ha="center", va="top", alpha=0.35)
                ok_any = True
            ax.set_ylabel(titles[m]); ax.grid(alpha=0.3)
            if not ok_any:
                ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes, alpha=0.6)

        axes[0].legend(title="StopGrad", ncol=3)
        axes[-1].set_xlabel(p)
        fig.suptitle(f"Quantile trend (StopGrad overlay): {p}")
        fig.tight_layout(rect=[0,0,1,0.96])
        fig.savefig(out_png, dpi=140); plt.close(fig)
        return True

    def _plot_trend_facets(agg: pd.DataFrame, p: str, out_png: str):
        order  = ["val_sample_pr_auc","val_latency_score","val_fp_per_min"]
        titles = {"val_sample_pr_auc": "PR-AUC (↑)",
                  "val_latency_score": "LatencyScore (↑)",
                  "val_fp_per_min":    "FP/min (↓)"}
        cohorts = [c for c in ["SG=False","SG=True","SG=unknown"] if (agg["COHORT_StopGrad"] == c).any()]
        if agg.empty or len(cohorts) == 0 or agg["bin_mid"].nunique() < 3:
            return False

        fig, axes = plt.subplots(len(cohorts), 3, figsize=(12, 3.5 * len(cohorts)), sharex=True)
        if len(cohorts) == 1:
            axes = np.expand_dims(axes, 0)

        for row, cohort in enumerate(cohorts):
            for col, m in enumerate(order):
                ax = axes[row, col]
                dd = agg[(agg["COHORT_StopGrad"] == cohort) & (agg["metric"] == m)].sort_values("bin_mid")
                if dd["bin_mid"].nunique() >= 3:
                    ax.plot(dd["bin_mid"], dd["median"], lw=2)
                    ax.fill_between(dd["bin_mid"], dd["q10"], dd["q90"], alpha=0.2)
                    for xm, cnt in zip(dd["bin_mid"], dd["count"]):
                        ax.text(xm, dd["median"].min(), f"n={int(cnt)}", fontsize=7, ha="center", va="top", alpha=0.35)
                else:
                    ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes, alpha=0.6)
                if row == 0:
                    ax.set_title(titles[m])
                if col == 0:
                    ax.set_ylabel(cohort)
                ax.grid(alpha=0.3)
        axes[-1, -1].set_xlabel(p)
        fig.suptitle(f"Quantile trend (StopGrad facets): {p}")
        fig.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(out_png, dpi=140); plt.close(fig)
        return True

    # numeric params to trend; relax uniqueness (>=3)
    trend_params = [c for c in df.columns
                    if c not in ["trial_number","val_sample_pr_auc","val_latency_score","val_fp_per_min","COHORT_StopGrad"]
                    and pd.api.types.is_numeric_dtype(df[c])
                    and df[c].nunique(dropna=True) >= 3]

    for must in ["learning_rate","LOSS_SupCon","LOSS_TupMPN","LOSS_NEGATIVES","LOSS_TV"]:
        if must in df.columns and must not in trend_params and pd.api.types.is_numeric_dtype(df[must]) and df[must].nunique(dropna=True) >= 3:
            trend_params.append(must)

    _created_stopgrad_imgs = []
    for p in trend_params:
        try:
            sub = df[["val_sample_pr_auc","val_latency_score","val_fp_per_min","COHORT_StopGrad", p]].copy()
            agg = _trend_by_bins_cohort(sub, p, nbins=12)
            if agg.empty or agg["bin_mid"].nunique() < 3:
                print(f"[trend] Skipped {p}: insufficient data after binning")
                continue
            out1 = os.path.join(qt_dir, f"{p}_trend_stopgrad_overlay.png")
            out2 = os.path.join(qt_dir, f"{p}_trend_stopgrad_facets.png")
            ok1 = _plot_trend_overlay(agg, p, out1)
            ok2 = _plot_trend_facets(agg, p, out2)
            if ok1: _created_stopgrad_imgs.append(os.path.relpath(out1, viz_dir))
            if ok2: _created_stopgrad_imgs.append(os.path.relpath(out2, viz_dir))
        except Exception as e:
            print(f"[trend] Skipped {p}: {e}")

    # ---------- CORRELATIONS ----------
    heatmap_fp = None
    obj_corr_fp = None
    try:
        num_cols = [c for c in df.columns
                    if c not in ["trial_number","val_sample_pr_auc","val_latency_score","val_fp_per_min"]
                    and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            corr_df = df[num_cols + ["val_sample_pr_auc","val_latency_score","val_fp_per_min"]].corr(method='spearman')
            plt.figure(figsize=(max(8, 0.6*len(corr_df.columns)), max(6, 0.5*len(corr_df))))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=False)
            plt.title("Spearman correlation: numeric hyperparameters vs objectives")
            plt.tight_layout()
            heatmap_fp = os.path.join(viz_dir, "correlations_spearman.png")
            plt.savefig(heatmap_fp, dpi=130); plt.close()
    except Exception as e:
        print(f"Correlation heatmap warning: {e}"); plt.close()

    try:
        obj_corr = df[["val_sample_pr_auc","val_latency_score","val_fp_per_min"]].corr(method='spearman')
        plt.figure(figsize=(4.8,4))
        sns.heatmap(obj_corr, annot=True, fmt=".2f", cmap="vlag", cbar=False,
                    xticklabels=["PR-AUC (↑)","LatencyScore (↑)","FP/min (↓)"],
                    yticklabels=["PR-AUC (↑)","LatencyScore (↑)","FP/min (↓)"])
        plt.tight_layout()
        obj_corr_fp = os.path.join(viz_dir, "objective_correlations.png")
        plt.savefig(obj_corr_fp, dpi=140); plt.close()
    except Exception as e:
        print(f"Objective correlation warning: {e}"); plt.close()

    # ---------- SIMPLE COMBINED RANK (top-k) ----------
    def robust_minmax(x):
        q05, q95 = np.nanquantile(x, [0.05, 0.95])
        d = max(1e-9, q95 - q05); z = (x - q05) / d
        return np.clip(z, 0, 1)

    df["score_pr"] = df["val_sample_pr_auc"]          # higher better
    df["score_lat"] = df["val_latency_score"]         # higher better
    df["score_fp"] = 1 - robust_minmax(df["val_fp_per_min"])  # lower fp → higher score
    df["combined_avg"] = (df["score_pr"] + df["score_lat"] + df["score_fp"]) / 3.0
    top_combined = df.sort_values("combined_avg", ascending=False).head(25)
    top_combined.to_csv(os.path.join(viz_dir, "top25_combined_avg.csv"), index=False)

    # Top-by-objective HTML tables with links
    def _trial_link(trial_no: int) -> str:
        base = os.path.join("studies", param_dir)
        matches = glob.glob(os.path.join(base, f"study_{trial_no}_*"))
        if matches:
            rel = os.path.relpath(matches[0], viz_dir)
            return f'<a href="{rel}">study_{trial_no}</a>'
        return ""

    try:
        top_k = 25
        best_pr  = df.sort_values("val_sample_pr_auc", ascending=False).head(top_k).copy()
        best_lat = df.sort_values("val_latency_score", ascending=False).head(top_k).copy()
        best_fp  = df.sort_values("val_fp_per_min", ascending=True ).head(top_k).copy()
        for fname, ddd in [("top_by_pr_auc.html", best_pr),
                           ("top_by_latency.html", best_lat),
                           ("top_by_fpmin.html", best_fp),
                           ("top_by_combined.html", top_combined)]:
            dd = ddd.copy()
            dd["study_dir"] = dd["trial_number"].apply(_trial_link)
            lead = ["trial_number","val_sample_pr_auc","val_latency_score","val_fp_per_min","combined_avg","study_dir"]
            keep = [c for c in lead if c in dd.columns] + [c for c in dd.columns if c not in lead]
            html_tbl = dd[keep].to_html(escape=False, index=False)
            with open(os.path.join(viz_dir, fname), "w") as fh:
                fh.write(f"<html><head><meta charset='utf-8'><style>body{{font-family:Arial;margin:20px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px}} th{{background:#f5f5f5}}</style></head><body>{html_tbl}</body></html>")
    except Exception as e:
        print(f"Top-k HTML warning: {e}")

    # =========================
    # EXTRA ANALYSIS ADD-ONS
    # =========================
    extras_dir    = os.path.join(viz_dir, "extras");                os.makedirs(extras_dir, exist_ok=True)
    qplots_dir    = os.path.join(extras_dir, "quantile_trends");    os.makedirs(qplots_dir, exist_ok=True)
    heat2d_dir    = os.path.join(extras_dir, "heatmaps_2d");        os.makedirs(heat2d_dir, exist_ok=True)
    cohort_dir    = os.path.join(extras_dir, "cohorts");            os.makedirs(cohort_dir, exist_ok=True)
    recommend_dir = os.path.join(extras_dir, "range_recommendations"); os.makedirs(recommend_dir, exist_ok=True)
    recon_dir     = os.path.join(extras_dir, "importance_reconciliation"); os.makedirs(recon_dir, exist_ok=True)

    objective_cols = ["val_sample_pr_auc","val_latency_score","val_fp_per_min"]
    param_cols = [c for c in df.columns if c not in ["trial_number", *objective_cols,
                                                     "score_pr","score_lat","score_fp","combined_avg","COHORT_StopGrad"]]
    num_params_all = [p for p in param_cols if pd.api.types.is_numeric_dtype(df[p])]
    cat_params = [p for p in param_cols if p not in num_params_all]

    # Cohort analysis (StopGrad ON/OFF) — safe split
    def _detect_stopgrad_series(dfx: pd.DataFrame) -> pd.Series:
        if "USE_StopGrad" in dfx.columns:
            s = dfx["USE_StopGrad"]
            if s.dtype == bool:
                return s
            return s.map(lambda v: bool(v) if pd.notna(v) else False)
        if "TYPE_ARCH" in dfx.columns:
            return dfx["TYPE_ARCH"].astype(str).str.lower().str.contains("stopgrad", na=False)
        return None

    stopgrad_series = _detect_stopgrad_series(df)

    def _safe_corr_heatmap(data: pd.DataFrame, name: str, out_png: str):
        try:
            cols = [*num_params_all, *objective_cols]
            cols = [c for c in cols if c in data.columns]
            if len(cols) < 3: return
            corr_df = data[cols].corr(method="spearman")
            plt.figure(figsize=(max(8, 0.6*len(corr_df.columns)), max(6, 0.5*len(corr_df))))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title(f"Spearman (numeric) — {name}")
            plt.tight_layout()
            plt.savefig(out_png, dpi=130); plt.close()
        except Exception:
            plt.close()

    def _param_summary_scatter(data: pd.DataFrame, name: str):
        for obj, ylabel in [("val_sample_pr_auc","PR-AUC (↑)"),
                            ("val_latency_score","LatencyScore (↑)"),
                            ("val_fp_per_min","FP/min (↓)")]:
            if obj not in data.columns: continue
            plt.figure(figsize=(max(8, 0.4*len(param_cols)), 4))
            ax = plt.gca()
            x_idx, x_labs = [], []
            for i, p in enumerate(param_cols):
                if p not in data.columns or data[p].nunique(dropna=True) <= 1:
                    continue
                if pd.api.types.is_numeric_dtype(data[p]):
                    valid = data[[p, obj]].dropna()
                    if valid[p].nunique() < 3 or len(valid) < 8:
                        continue
                    qbins = pd.qcut(valid[p], q=min(10, max(3, valid[p].nunique())), duplicates="drop")
                    means = valid.groupby(qbins, observed=True)[obj].mean().values
                    ax.scatter(np.full_like(means, len(x_idx)), means, s=26, alpha=0.7)
                else:
                    g = data.groupby(p, observed=True)[obj].mean().sort_values(ascending=False)
                    ax.scatter(np.full_like(g.values, len(x_idx)), g.values, s=26, alpha=0.7)
                x_idx.append(len(x_idx)); x_labs.append(p)
            ax.set_xticks(range(len(x_labs))); ax.set_xticklabels(x_labs, rotation=28, ha="right")
            ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} vs params — {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(cohort_dir, f"{obj}_vs_params_{name}.png"), dpi=120); plt.close()

    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)
        d0 = df[~sg_mask].copy()  # StopGrad OFF
        d1 = df[ sg_mask].copy()  # StopGrad ON
        if len(d0) >= 30:
            _safe_corr_heatmap(d0, "StopGrad_OFF", os.path.join(cohort_dir, "corr_spearman_StopGrad_OFF.png"))
            _param_summary_scatter(d0, "StopGrad_OFF")
        if len(d1) >= 30:
            _safe_corr_heatmap(d1, "StopGrad_ON",  os.path.join(cohort_dir, "corr_spearman_StopGrad_ON.png"))
            _param_summary_scatter(d1, "StopGrad_ON")

    # Quantile trend curves (global simple ones)
    def quantile_trend(x: pd.Series, y: pd.Series, q=12):
        valid = x.notna() & y.notna()
        xv, yv = x[valid], y[valid]
        if len(xv) < 10 or xv.nunique() < 3:
            return None
        bins = pd.qcut(xv, q=min(q, max(3, xv.nunique())), duplicates="drop")
        grp  = pd.DataFrame({"y": yv, "bin": bins, "x": xv}).groupby("bin", observed=True)
        mu = grp["y"].mean().values
        sd = grp["y"].std().values
        n  = grp["y"].size().values.astype(float)
        se = np.where(n>1, sd/np.sqrt(n), np.nan)
        xc = grp["x"].mean().values
        return xc, mu, se

    num_params_simple = [c for c in df.columns
                         if c not in ["trial_number","val_sample_pr_auc","val_latency_score","val_fp_per_min","COHORT_StopGrad"]
                         and pd.api.types.is_numeric_dtype(df[c])
                         and df[c].nunique(dropna=True) >= 3]

    for p in num_params_simple:
        fig, axs = plt.subplots(3,1,figsize=(8,9), sharex=True)
        ok = False
        for ax, obj, lab in zip(axs,
                                ["val_sample_pr_auc","val_latency_score","val_fp_per_min"],
                                ["PR-AUC (↑)", "LatencyScore (↑)","FP/min (↓)"]):
            if obj not in df.columns:
                ax.set_ylabel(lab); continue
            out = quantile_trend(df[p], df[obj], q=12)
            if out is None:
                ax.set_ylabel(lab); continue
            x, mu, se = out
            ax.plot(x, mu, marker="o", linewidth=1.5)
            if np.isfinite(se).any():
                ax.fill_between(x, mu - 1.96*np.nan_to_num(se), mu + 1.96*np.nan_to_num(se), alpha=0.2)
            ax.set_ylabel(lab); ok = True
        axs[-1].set_xlabel(p)
        if ok:
            fig.suptitle(f"Quantile trend — {p}")
            fig.tight_layout(rect=[0,0,1,0.97])
            fig.savefig(os.path.join(qplots_dir, f"{p}_quantile_trends.png"), dpi=130)
        plt.close(fig)

    # 2D interaction heatmaps (if both axes are informative)
    def heat2d(x: pd.Series, y: pd.Series, z: pd.Series, xq=12, yq=12):
        valid = x.notna() & y.notna() & z.notna()
        xv, yv, zv = x[valid], y[valid], z[valid]
        if xv.nunique() < 4 or yv.nunique() < 4 or len(zv) < 25:
            return None
        xb = pd.qcut(xv, q=min(xq, max(4, xv.nunique())), duplicates="drop")
        yb = pd.qcut(yv, q=min(yq, max(4, yv.nunique())), duplicates="drop")
        grid = pd.DataFrame({"xb": xb, "yb": yb, "z": zv}).groupby(["xb","yb"], observed=True)["z"].mean().unstack()
        return grid

    heat2d_dir = os.path.join(extras_dir, "heatmaps_2d"); os.makedirs(heat2d_dir, exist_ok=True)
    def _maybe_heatmap(xname, yname):
        if xname not in df.columns or yname not in df.columns:
            return
        for obj, lab in [("val_sample_pr_auc","PR-AUC (↑)"),
                         ("val_latency_score","LatencyScore (↑)"),
                         ("val_fp_per_min","FP/min (↓)")]:
            if obj not in df.columns: continue
            H = heat2d(df[xname], df[yname], df[obj])
            if H is None:
                continue
            plt.figure(figsize=(6.8,5.2))
            sns.heatmap(H, cmap="viridis", annot=False)
            plt.title(f"{lab} mean — {xname} × {yname}")
            plt.tight_layout()
            plt.savefig(os.path.join(heat2d_dir, f"{obj}_{xname}_x_{yname}.png"), dpi=140)
            plt.close()

    if "LOSS_SupCon" in df.columns and "LOSS_TupMPN" in df.columns:
        _maybe_heatmap("LOSS_SupCon", "LOSS_TupMPN")
    if "learning_rate" in df.columns and "LOSS_NEGATIVES" in df.columns:
        _maybe_heatmap("learning_rate", "LOSS_NEGATIVES")
    if "LOSS_TV" in df.columns and "LOSS_TupMPN" in df.columns:
        _maybe_heatmap("LOSS_TV", "LOSS_TupMPN")

    # Range recommendations: top-quartile trial bands per objective
    def recommend_ranges(data: pd.DataFrame, name: str, top_frac=0.25):
        num_params_local = [p for p in data.columns
                            if p not in ["trial_number","val_sample_pr_auc","val_latency_score","val_fp_per_min",
                                         "score_pr","score_lat","score_fp","combined_avg","COHORT_StopGrad"]
                            and pd.api.types.is_numeric_dtype(data[p])]
        cat_params_local = [p for p in data.columns
                            if p not in ["trial_number","val_sample_pr_auc","val_latency_score","val_fp_per_min",
                                         "score_pr","score_lat","score_fp","combined_avg","COHORT_StopGrad"]
                            and p not in num_params_local]
        recs = {}
        for obj, asc, nice in [("val_sample_pr_auc", False, "PR-AUC (↑)"),
                               ("val_latency_score", False, "LatencyScore (↑)"),
                               ("val_fp_per_min", True,  "FP/min (↓)")]:
            if obj not in data.columns or data[obj].isnull().all():
                continue
            dsort = data.sort_values(obj, ascending=asc)
            top_n = max(20, int(len(dsort)*top_frac))
            top   = dsort.head(top_n)
            rng = {}
            for p in num_params_local:
                col = pd.to_numeric(top[p], errors="coerce").dropna()
                if col.nunique() < 2:
                    continue
                try:
                    rng[p] = (float(np.nanquantile(col, 0.20)), float(np.nanquantile(col, 0.80)))
                except Exception:
                    pass
            cat = {}
            for p in cat_params_local:
                vc = top[p].value_counts(normalize=True, dropna=False)
                if len(vc):
                    cat[p] = vc.head(3).to_dict()
            recs[nice] = {"num_quantile_20_80": rng, "cat_top3_props": cat, "n_top": int(len(top))}
        with open(os.path.join(recommend_dir, f"ranges_{name}.json"), "w") as fh:
            json.dump(recs, fh, indent=2)
        rows = []
        for obj, payload in recs.items():
            for p,(lo,hi) in payload["num_quantile_20_80"].items():
                rows.append({"objective": obj, "param": p, "q20": lo, "q80": hi})
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(recommend_dir, f"ranges_numeric_{name}.csv"), index=False)

    recommend_ranges(df, "ALL")
    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)
        if df[~sg_mask].shape[0] >= 40: recommend_ranges(df[~sg_mask], "StopGrad_OFF")
        if df[ sg_mask].shape[0] >= 40: recommend_ranges(df[ sg_mask], "StopGrad_ON")

    # Importance reconciliation: absolute Spearman vs PR-AUC (global/cohort)
    imp_rows = []
    for p in num_params_all:
        try:
            r = df[[p,"val_sample_pr_auc"]].dropna().corr(method="spearman").iloc[0,1]
            if pd.notna(r):
                imp_rows.append({"param": p, "source": "Spearman|rho| (global)", "value": abs(float(r))})
        except Exception:
            pass
    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)
        for name, dsub in [("StopGrad_OFF", df[~sg_mask]), ("StopGrad_ON", df[sg_mask])]:
            if len(dsub) < 30: continue
            for p in num_params_all:
                try:
                    rr = dsub[[p,"val_sample_pr_auc"]].dropna().corr(method="spearman").iloc[0,1]
                    if pd.notna(rr):
                        imp_rows.append({"param": p, "source": f"Spearman|rho| ({name})", "value": abs(float(rr))})
                except Exception:
                    pass
    if imp_rows:
        imp_df = pd.DataFrame(imp_rows)
        piv = imp_df.pivot_table(index="param", columns="source", values="value", aggfunc="max").fillna(0.0)
        piv = piv.sort_values(by=list(piv.columns)[0], ascending=False)
        # ensure consistent colors/legend
        ax = piv.plot(kind="bar", figsize=(max(9, 0.55*len(piv)),5))
        ax.set_ylabel("|Spearman| vs PR-AUC"); plt.title("Correlation-based importances (compare with Optuna fANOVA)")
        plt.tight_layout()
        plt.savefig(os.path.join(recon_dir, "spearman_vs_pr_auc.png"), dpi=130)
        plt.close()

    # ---------- HTML REPORT ----------
    html = []
    html.append(f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Study Visualization (3-obj): {study.study_name}</title>
    <style>
    body{{font-family:Arial,Helvetica,sans-serif;margin:20px;}} h2{{margin-top:28px}}
    a{{color:#007bff;text-decoration:none}} a:hover{{text-decoration:underline}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
    .card{{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}}
    img{{max-width:100%}}
    .notice{{padding:10px;border:1px solid #ffc107;background:#fff3cd;border-radius:6px}}
    </style></head><body>
    <h1>Visualization — {study.study_name}</h1>
    <p>Objectives: <b>maximize</b> PR-AUC, <b>minimize</b> FP/min, <b>maximize</b> LatencyScore.</p>
    <p><b>Trials:</b> {len(df)} | <b>Pareto set:</b> {len(pareto_df)}</p>""")
    if mapping_notice:
        html.append(f'<p class="notice">{mapping_notice}</p>')
    if has_constraints and not DO_HV:
        html.append('<p class="notice">Note: No feasible trials under constraints; hypervolume history omitted.</p>')
    html.append(f"""
    <ul>
      <li><a href="all_completed_trials.csv">all_completed_trials.csv</a></li>
      <li><a href="pareto_trials.csv">pareto_trials.csv</a> &nbsp;|&nbsp; <a href="pareto_trials.html">Pareto table (HTML)</a></li>
      <li><a href="top25_combined_avg.csv">top25_combined_avg.csv</a></li>
      <li>Top-K (HTML): <a href="top_by_pr_auc.html">PR-AUC</a> · <a href="top_by_fpmin.html">FP/min</a> · <a href="top_by_latency.html">LatencyScore</a> · <a href="top_by_combined.html">Combined</a></li>
      <li>Objective EDFs: <a href="edf_pr_auc.html">PR-AUC</a> · <a href="edf_fp_per_min.html">FP/min</a> · <a href="edf_latency.html">LatencyScore</a></li>
    </ul>""")
    if pareto_snapshot_html:
        html.append("<h2>Pareto snapshot (top 15 by PR-AUC)</h2>")
        html.append(pareto_snapshot_html)

    html.append("""
    <h2>Standard Optuna Plots</h2>
    <div class="grid">
      <div class="card"><a href="history_pr_auc.html">Optimization History — PR-AUC</a></div>
      <div class="card"><a href="history_fp_per_min.html">Optimization History — FP/min</a></div>
      <div class="card"><a href="history_latency.html">Optimization History — LatencyScore</a></div>
      <div class="card"><a href="param_importances_pr_auc.html">Param Importances — PR-AUC</a></div>
      <div class="card"><a href="param_importances_fp_per_min.html">Param Importances — FP/min</a></div>
      <div class="card"><a href="param_importances_latency.html">Param Importances — LatencyScore</a></div>
      <div class="card"><a href="pareto_front_3obj.html">Pareto Front (interactive)</a></div>
      <div class="card"><a href="hypervolume_history.html">Hypervolume History</a></div>
    </div>""")

    html.append("<h2>Objective Projections</h2><div class='grid'>")
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter3d_all.html")):
        html.append('<div class="card"><a href="scatter3d_all.html">3D Scatter (All trials, Pareto highlighted)</a></div>')
    html.append("</div>")

    # StopGrad trend plots (overlay + facets)
    if len(_created_stopgrad_imgs) > 0:
        html.append("<h2>Quantile Trends — StopGrad</h2><div class='grid'>")
        for imgp in _created_stopgrad_imgs:
            html.append(f'<div class="card"><img src="{imgp}"></div>')
        html.append("</div>")
    else:
        html.append("<h2>Quantile Trends — StopGrad</h2><p class='notice'>No StopGrad trend plots were created (insufficient data or parameters too discrete).</p>")

    # Correlations
    if heatmap_fp or obj_corr_fp:
        html.append("<h2>Correlations</h2><div class='grid'>")
        if heatmap_fp and os.path.exists(heatmap_fp):
            html.append(f'<div class="card"><img src="correlations_spearman.png" alt="Hyperparameter correlations"></div>')
        if obj_corr_fp and os.path.exists(obj_corr_fp):
            html.append(f'<div class="card"><img src="objective_correlations.png" alt="Objective correlations"></div>')
        html.append("</div>")

    # Per-Parameter Impact thumbnails
    html.append("<h2>Per-Parameter Impact</h2><div class='grid'>")
    for p in hyperparams:
        imgp = os.path.join("param_impact", f"{p}_impact.png")
        if os.path.exists(os.path.join(viz_dir, imgp)):
            html.append(f'<div class="card"><h3>{p}</h3><img src="{imgp}"></div>')
    html.append("</div>")

    # Extras sections links
    html.append("""
    <h2>Extras</h2>
    <ul>
      <li><b>Cohorts:</b> see PNGs in <code>extras/cohorts/</code></li>
      <li><b>Quantile Trends (global):</b> <code>extras/quantile_trends/</code></li>
      <li><b>2D Heatmaps:</b> <code>extras/heatmaps_2d/</code></li>
      <li><b>Range Recommendations:</b> JSON/CSV in <code>extras/range_recommendations/</code></li>
      <li><b>Importance Reconciliation:</b> <code>extras/importance_reconciliation/spearman_vs_pr_auc.png</code></li>
    </ul>
    </body></html>""")

    with open(os.path.join(viz_dir, "index.html"), "w") as f:
        f.write("\n".join(html))

    # ---------- SUMMARY ----------
    stats = {
        "study_name": study.study_name,
        "n_completed_trials": int(len(df)),
        "n_pareto_trials": int(len(pareto_df)),
        "has_constraints": bool(has_constraints),
        "n_feasible_trials": int(len(feasible_trials)),
        "objectives": ["val_sample_pr_auc (max)", "val_fp_per_min (min)", "val_latency_score (max)"],
        "objective_index_map": OBJECTIVE_INDEX,
        "auto_swap_applied": bool(mapping_notice != "")
    }
    with open(os.path.join(viz_dir, "study_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Visualization complete → {viz_dir}")

elif mode == 'tune_viz_multi_v7':
    import os, sys, json, math, warnings, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import optuna

    warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

    # Optional: Plotly
    try:
        import plotly.express as px
        HAVE_PLOTLY = True
    except Exception:
        HAVE_PLOTLY = False

    # ---------- CONFIG ----------
    tag = args.tag[0]
    param_dir = f'params_{tag}'
    storage_url = f"sqlite:///studies/{param_dir}/{param_dir}.db"
    viz_dir = f"studies/{param_dir}/visualizations_v7"
    os.makedirs(viz_dir, exist_ok=True)

    print(f"Loading study '{param_dir}' from {storage_url}")
    try:
        study = optuna.load_study(study_name=param_dir, storage=storage_url)
    except Exception as e:
        print(f"Error loading study: {e}")
        sys.exit(1)

    # ---------- HELPERS ----------
    def _ua(tr, key, default=np.nan):
        # Prefer user_attrs, then system_attrs
        try:
            if hasattr(tr, "user_attrs") and key in tr.user_attrs:
                return tr.user_attrs.get(key, default)
            if hasattr(tr, "system_attrs") and key in tr.system_attrs:
                return tr.system_attrs.get(key, default)
        except Exception:
            pass
        return default

    # ---------- COLLECT COMPLETED TRIALS ----------
    trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not trials:
        print("No completed trials; nothing to visualize.")
        sys.exit(0)

    # Expect 2 objectives (PR-AUC, FP/min)
    nvals = max((len(t.values) if t.values else 0) for t in trials)
    if nvals < 2:
        print(f"Trials have only {nvals} objectives; need 2 (PR-AUC, FP/min).")
        sys.exit(0)
    OBJECTIVE_INDEX = dict(pr_auc=0, fp_per_min=1)
    if nvals > 2:
        print(f"Detected {nvals} objectives; using first two as {OBJECTIVE_INDEX}. Adjust if needed.")

    # Constraints (optional)
    def _get_constraints(tr):
        return tr.system_attrs.get("constraints") or tr.user_attrs.get("constraints")

    has_constraints = any(_get_constraints(t) is not None for t in trials)
    feasible_trials = [
        t for t in trials
        if (_get_constraints(t) is None) or all((c is not None) and (c <= 0) for c in _get_constraints(t))
    ]
    DO_HV = not (has_constraints and len(feasible_trials) == 0)
    if has_constraints and not DO_HV:
        print("No feasible trials under constraints; skipping hypervolume plot.")

    # Collect rows from trials
    rows = []
    for t in trials:
        if not t.values:
            continue
        try:
            pr = float(t.values[OBJECTIVE_INDEX['pr_auc']])
            fp = float(t.values[OBJECTIVE_INDEX['fp_per_min']])
        except Exception:
            continue
        bad = lambda x: (x is None) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
        if any(bad(x) for x in [pr, fp]):
            continue
        rec = {
            "trial_number": t.number,
            "val_sample_pr_auc": pr,
            "val_fp_per_min": fp,
            # ---- NEW METRICS FROM user_attrs / system_attrs ----
            "val_latency_score":  _ua(t, "sel_latency_score"),
            "val_recall_at_0p7":  _ua(t, "sel_recall_at_0p7"),
            "val_sample_max_f1":  _ua(t, "sel_max_f1"),
            "val_sample_max_mcc": _ua(t, "sel_max_mcc"),
            "sel_epoch":          _ua(t, "sel_epoch")
        }
        rec.update(t.params)  # include hyperparams
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid completed trials with 2 objectives.")
        sys.exit(0)

    # Optional sanity: PR-AUC should be in [0,1]. If >1 frequently, user probably swapped.
    if (df["val_sample_pr_auc"] > 1.05).mean() > 0.5:
        print("Warning: Many PR-AUC values > 1. Did you swap objective order when saving?")

    # Save all trials CSV
    all_csv = os.path.join(viz_dir, "all_completed_trials.csv")
    df.to_csv(all_csv, index=False)
    print(f"Saved {len(df)} trials → {all_csv}")

    # ---------- STANDARD OPTUNA VISUALS ----------
    print("Generating Optuna standard plots...")
    try:
        # History per objective
        fig_hist_pr = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'),
            target_name="Sample PR-AUC")
        fig_hist_pr.write_html(os.path.join(viz_dir, "history_pr_auc.html"))

        fig_hist_fp = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['fp_per_min']] if t.values else float('nan'),
            target_name="FP/min")
        fig_hist_fp.write_html(os.path.join(viz_dir, "history_fp_per_min.html"))

        # ---- NEW: Histories for latency & recall ----
        try:
            fig_hist_lat = optuna.visualization.plot_optimization_history(
                study, target=lambda t: _ua(t, "sel_latency_score"),
                target_name="Latency score (↑)"
            )
            fig_hist_lat.write_html(os.path.join(viz_dir, "history_latency.html"))
        except Exception as e:
            print(f"Latency history warning: {e}")

        try:
            fig_hist_rec = optuna.visualization.plot_optimization_history(
                study, target=lambda t: _ua(t, "sel_recall_at_0p7"),
                target_name="Recall@0.7 (↑)"
            )
            fig_hist_rec.write_html(os.path.join(viz_dir, "history_recall.html"))
        except Exception as e:
            print(f"Recall history warning: {e}")

        # Param importances per objective
        fig_imp_pr = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'),
            target_name="Sample PR-AUC")
        fig_imp_pr.write_html(os.path.join(viz_dir, "param_importances_pr_auc.html"))

        fig_imp_fp = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['fp_per_min']] if t.values else float('nan'),
            target_name="FP/min")
        fig_imp_fp.write_html(os.path.join(viz_dir, "param_importances_fp_per_min.html"))

        # ---- NEW: Param importances for latency & recall ----
        try:
            fig_imp_lat = optuna.visualization.plot_param_importances(
                study, target=lambda t: _ua(t, "sel_latency_score"),
                target_name="Latency score (↑)"
            )
            fig_imp_lat.write_html(os.path.join(viz_dir, "param_importances_latency.html"))
        except Exception as e:
            print(f"Latency importances warning: {e}")

        try:
            fig_imp_rec = optuna.visualization.plot_param_importances(
                study, target=lambda t: _ua(t, "sel_recall_at_0p7"),
                target_name="Recall@0.7 (↑)"
            )
            fig_imp_rec.write_html(os.path.join(viz_dir, "param_importances_recall.html"))
        except Exception as e:
            print(f"Recall importances warning: {e}")

        # Pareto front (2D)
        names_by_index = [""] * nvals
        names_by_index[OBJECTIVE_INDEX['pr_auc']] = "Sample PR-AUC"
        names_by_index[OBJECTIVE_INDEX['fp_per_min']] = "FP/min"
        fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=names_by_index)
        fig_pareto.write_html(os.path.join(viz_dir, "pareto_front_2obj.html"))
    except Exception as e:
        print(f"Standard plot warning: {e}")

    # ---------- EDFs ----------
    try:
        fig_edf_pr = optuna.visualization.plot_edf(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'),
            target_name="Sample PR-AUC (↑)")
        fig_edf_pr.write_html(os.path.join(viz_dir, "edf_pr_auc.html"))

        fig_edf_fp = optuna.visualization.plot_edf(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['fp_per_min']] if t.values else float('nan'),
            target_name="FP/min (↓)")
        fig_edf_fp.write_html(os.path.join(viz_dir, "edf_fp_per_min.html"))

        # ---- NEW: EDFs for latency & recall ----
        try:
            fig_edf_lat = optuna.visualization.plot_edf(
                study, target=lambda t: _ua(t, "sel_latency_score"),
                target_name="Latency score (↑)")
            fig_edf_lat.write_html(os.path.join(viz_dir, "edf_latency.html"))
        except Exception as e:
            print(f"EDF latency warning: {e}")

        try:
            fig_edf_rec = optuna.visualization.plot_edf(
                study, target=lambda t: _ua(t, "sel_recall_at_0p7"),
                target_name="Recall@0.7 (↑)")
            fig_edf_rec.write_html(os.path.join(viz_dir, "edf_recall.html"))
        except Exception as e:
            print(f"EDF recall warning: {e}")

    except Exception as e:
        print(f"EDF plot warning: {e}")

    # ---------- PARETO SET & CSV/HTML ----------
    pareto_trials = study.best_trials
    pareto_nums = [t.number for t in pareto_trials]
    pareto_df = df[df.trial_number.isin(pareto_nums)].copy()
    pareto_csv = os.path.join(viz_dir, "pareto_trials.csv")
    pareto_df.to_csv(pareto_csv, index=False)
    print(f"Pareto set size: {len(pareto_df)} → {pareto_csv}")

    def _trial_link(trial_no: int) -> str:
        base = os.path.join("studies", param_dir)
        matches = glob.glob(os.path.join(base, f"study_{trial_no}_*"))
        if matches:
            rel = os.path.relpath(matches[0], viz_dir)
            return f'<a href="{rel}">study_{trial_no}</a>'
        return ""

    pareto_snapshot_html = ""
    if not pareto_df.empty:
        _pareto_view = pareto_df.copy()
        _pareto_view["study_dir"] = _pareto_view["trial_number"].apply(_trial_link)
        lead_cols = ["trial_number", "val_sample_pr_auc", "val_fp_per_min", "study_dir"]
        remaining = [c for c in _pareto_view.columns if c not in lead_cols]
        pareto_html = _pareto_view[lead_cols + remaining].to_html(escape=False, index=False)
        with open(os.path.join(viz_dir, "pareto_trials.html"), "w") as fh:
            fh.write(f"""<html><head><meta charset="utf-8"><title>Pareto Trials — {study.study_name}</title>
<style>body{{font-family:Arial;margin:20px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px}} th{{background:#f5f5f5}}</style>
</head><body><h2>Pareto Trials</h2>
<p>Higher is better: PR-AUC. Lower is better: FP/min.</p>
{pareto_html}
</body></html>""")
        snapshot = _pareto_view.sort_values("val_sample_pr_auc", ascending=False)[lead_cols].head(15)
        pareto_snapshot_html = snapshot.to_html(escape=False, index=False)

    # ---------- HYPERVOLUME HISTORY ----------
    try:
        if DO_HV:
            ref_point = [0.0] * nvals
            ref_point[OBJECTIVE_INDEX['pr_auc']] = float(df["val_sample_pr_auc"].min() - 1e-3)   # smaller worse
            ref_point[OBJECTIVE_INDEX['fp_per_min']] = float(df["val_fp_per_min"].max() + 1e-3)  # larger worse
            fig_hv = optuna.visualization.plot_hypervolume_history(study, reference_point=ref_point)
            fig_hv.write_html(os.path.join(viz_dir, "hypervolume_history.html"))
        else:
            print("Hypervolume plot skipped due to infeasible trials under constraints.")
    except Exception as e:
        print(f"Hypervolume plot skipped: {e}")

    # ---------- OBJECTIVE PROJECTION (Plotly) ----------
    try:
        if HAVE_PLOTLY:
            fig2d = px.scatter(
                df,
                x="val_sample_pr_auc", y="val_fp_per_min",
                color=np.where(df.trial_number.isin(pareto_nums), "Pareto", "Other"),
                hover_name="trial_number", opacity=0.9
            )
            fig2d.update_layout(
                xaxis_title="PR-AUC (↑)",
                yaxis_title="FP/min (↓)",
                legend_title_text="Trials"
            )
            fig2d.write_html(os.path.join(viz_dir, "scatter2d_all.html"))

            # ---- NEW: PR-AUC vs FP/min colored by Latency ----
            if "val_latency_score" in df.columns:
                fig2d_lat = px.scatter(
                    df, x="val_sample_pr_auc", y="val_fp_per_min",
                    color="val_latency_score", color_continuous_scale="Viridis",
                    hover_name="trial_number",
                    hover_data=[c for c in ["val_recall_at_0p7","val_sample_max_f1","val_sample_max_mcc","sel_epoch"]
                                if c in df.columns],
                    opacity=0.9, title="PR-AUC vs FP/min — color: Latency"
                )
                fig2d_lat.update_layout(xaxis_title="PR-AUC (↑)", yaxis_title="FP/min (↓)")
                fig2d_lat.write_html(os.path.join(viz_dir, "scatter2d_prauc_fp_color_latency.html"))

            # ---- NEW: PR-AUC vs FP/min colored by Recall ----
            if "val_recall_at_0p7" in df.columns:
                fig2d_rec = px.scatter(
                    df, x="val_sample_pr_auc", y="val_fp_per_min",
                    color="val_recall_at_0p7", color_continuous_scale="Plasma",
                    hover_name="trial_number",
                    hover_data=[c for c in ["val_latency_score","val_sample_max_f1","val_sample_max_mcc","sel_epoch"]
                                if c in df.columns],
                    opacity=0.9, title="PR-AUC vs FP/min — color: Recall@0.7"
                )
                fig2d_rec.update_layout(xaxis_title="PR-AUC (↑)", yaxis_title="FP/min (↓)")
                fig2d_rec.write_html(os.path.join(viz_dir, "scatter2d_prauc_fp_color_recall.html"))

            # ---- NEW: Pairwise with Latency ----
            if "val_latency_score" in df.columns:
                fig_p_l = px.scatter(
                    df, x="val_sample_pr_auc", y="val_latency_score",
                    color=np.where(df.trial_number.isin(pareto_nums), "Pareto(PR↔FP)", "Other"),
                    hover_name="trial_number", opacity=0.9,
                    title="PR-AUC vs Latency (↑/↑)"
                )
                fig_p_l.update_layout(xaxis_title="PR-AUC (↑)", yaxis_title="Latency score (↑)")
                fig_p_l.write_html(os.path.join(viz_dir, "scatter2d_prauc_latency.html"))

                fig_f_l = px.scatter(
                    df, x="val_fp_per_min", y="val_latency_score",
                    color=np.where(df.trial_number.isin(pareto_nums), "Pareto(PR↔FP)", "Other"),
                    hover_name="trial_number", opacity=0.9,
                    title="FP/min vs Latency (↓/↑)"
                )
                fig_f_l.update_layout(xaxis_title="FP/min (↓)", yaxis_title="Latency score (↑)")
                fig_f_l.write_html(os.path.join(viz_dir, "scatter2d_fpmin_latency.html"))
    except Exception as e:
        print(f"Projection plot warning: {e}")

    # ---------- PER-PARAM IMPACT (quick plots) ----------
    param_impact_dir = os.path.join(viz_dir, "param_impact")
    os.makedirs(param_impact_dir, exist_ok=True)

    def qylim(series, lo=0.05, hi=0.95, pad=0.05, clamp=None):
        if series.isnull().all():
            return (0, 1)
        qlo, qhi = np.nanquantile(series, [lo, hi])
        span = max(1e-9, qhi - qlo)
        lo_v = qlo - pad * span
        hi_v = qhi + pad * span
        if clamp:
            lo_v = max(clamp[0], lo_v); hi_v = min(clamp[1], hi_v)
            if lo_v >= hi_v: lo_v, hi_v = clamp[0], clamp[1]
        return (lo_v, hi_v)

    pr_ylim = qylim(df["val_sample_pr_auc"], clamp=(0,1))
    fp_ylim = qylim(df["val_fp_per_min"], clamp=None)

    hyperparams = [c for c in df.columns if c not in [
        "trial_number","val_sample_pr_auc","val_fp_per_min",
        "val_latency_score","val_recall_at_0p7","val_sample_max_f1","val_sample_max_mcc",
        "score_pr","score_fp","combined_avg","COHORT_StopGrad","sel_epoch"
    ]]

    targets = [("val_sample_pr_auc","PR-AUC (↑)"),
               ("val_fp_per_min","FP/min (↓)")]
    if "val_latency_score" in df.columns:   targets.append(("val_latency_score","Latency (↑)"))
    if "val_recall_at_0p7" in df.columns:   targets.append(("val_recall_at_0p7","Recall@0.7 (↑)"))

    for p in hyperparams:
        if df[p].isnull().all() or df[p].nunique(dropna=True) <= 1:
            continue
        is_num = pd.api.types.is_numeric_dtype(df[p])
        fig, axes = plt.subplots(len(targets), 1, figsize=(10, 3.2*len(targets)), sharex=True)
        if len(targets) == 1: axes = [axes]

        for ax, (obj, ylabel) in zip(axes, targets):
            ylim = None
            if obj == "val_sample_pr_auc": ylim = pr_ylim
            elif obj == "val_fp_per_min": ylim = fp_ylim
            if is_num:
                sns.scatterplot(data=df, x=p, y=obj, ax=ax, s=18, alpha=0.5)
            else:
                sns.stripplot(data=df, x=p, y=obj, ax=ax, size=4, alpha=0.7, jitter=True)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_ylabel(ylabel)
            if ylim is not None: ax.set_ylim(ylim)
        axes[-1].set_xlabel(p)
        fig.tight_layout()
        outp = os.path.join(param_impact_dir, f"{p}_impact.png")
        fig.savefig(outp, dpi=130)
        plt.close(fig)

    # ---------- STOPGRAD QUANTILE TRENDS (overlay + facets) ----------
    qt_dir = os.path.join(viz_dir, "quantile_trends_stopgrad")
    os.makedirs(qt_dir, exist_ok=True)

    def _infer_stopgrad_column(df_in: pd.DataFrame) -> pd.Series:
        if "USE_StopGrad" in df_in.columns:
            col = df_in["USE_StopGrad"]
            if pd.api.types.is_bool_dtype(col):
                return pd.Series(np.where(col, "SG=True", "SG=False"), index=col.index)
            try:
                val = pd.to_numeric(col, errors="coerce")
                return pd.Series(np.where(val == 1, "SG=True",
                                   np.where(val == 0, "SG=False", "SG=unknown")), index=col.index)
            except Exception:
                s = col.astype(str).str.lower()
                return pd.Series(np.where(s.isin(["1","true","yes","y"]), "SG=True",
                                   np.where(s.isin(["0","false","no","n"]), "SG=False", "SG=unknown")), index=col.index)
        if "TYPE_ARCH" in df_in.columns:
            s = df_in["TYPE_ARCH"].astype(str).str.lower()
            return pd.Series(np.where(s.str.contains("stopgrad", na=False), "SG=True", "SG=False"), index=df_in.index)
        return pd.Series(["SG=unknown"] * len(df_in), index=df_in.index)

    df["COHORT_StopGrad"] = _infer_stopgrad_column(df)

    def _quantile_edges(x: pd.Series, nbins=12):
        x = pd.to_numeric(x, errors="coerce").dropna()
        if x.nunique() < 3 or len(x) < 12:
            if len(x) == 0:
                return None
            lo, hi = float(x.min()), float(x.max())
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                return None
            return np.linspace(lo, hi, 4)
        qs = np.linspace(0, 1, nbins + 1)
        edges = np.unique(np.nanquantile(x, qs))
        if len(edges) < 4:
            lo, hi = float(x.min()), float(x.max())
            edges = np.linspace(lo, hi, 4)
        eps = 1e-12 * (edges[-1] - edges[0] + 1.0)
        edges[0] -= eps; edges[-1] += eps
        return edges

    def _trend_by_bins_cohort(dfin: pd.DataFrame, param: str, nbins=12):
        keep_cols = ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7","COHORT_StopGrad", param]
        keep_cols = [c for c in keep_cols if c in dfin.columns]
        dfin = dfin[keep_cols].copy()
        dfin[param] = pd.to_numeric(dfin[param], errors="coerce")
        dfin = dfin.dropna(subset=[param])

        edges = _quantile_edges(dfin[param], nbins=nbins)
        if edges is None:
            return pd.DataFrame()

        b = pd.cut(dfin[param], bins=edges, include_lowest=True)
        mids = b.apply(lambda iv: np.mean([iv.left, iv.right]) if pd.notnull(iv) else np.nan)
        tmp = dfin.copy()
        tmp["_bin"] = b
        tmp["_mid"] = mids

        value_vars = [c for c in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7"] if c in tmp.columns]
        long = tmp.melt(
            id_vars=["_bin","_mid","COHORT_StopGrad"],
            value_vars=value_vars,
            var_name="metric", value_name="val"
        )

        agg = (long.dropna(subset=["_bin","_mid","val"])
                    .groupby(["_bin","_mid","COHORT_StopGrad","metric"], observed=True)
                    .agg(q10=("val", lambda s: np.nanquantile(s, 0.10)),
                         median=("val", "median"),
                         q90=("val", lambda s: np.nanquantile(s, 0.90)),
                         count=("val", "size"))
                    .reset_index())
        agg.rename(columns={"_mid": "bin_mid"}, inplace=True)
        return agg

    def _plot_trend_overlay(agg: pd.DataFrame, p: str, out_png: str):
        order = [m for m in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7"]
                 if (agg["metric"] == m).any()]
        titles = {
            "val_sample_pr_auc": "PR-AUC (↑)",
            "val_fp_per_min":    "FP/min (↓)",
            "val_latency_score": "Latency (↑)",
            "val_recall_at_0p7": "Recall@0.7 (↑)"
        }
        colors = {"SG=False":"tab:blue","SG=True":"tab:orange","SG=unknown":"tab:gray"}

        if agg.empty or agg["bin_mid"].nunique() < 3 or len(order) == 0:
            return False

        fig, axes = plt.subplots(len(order), 1, figsize=(9, 3.3*len(order)), sharex=True)
        if len(order) == 1: axes = [axes]
        for ax, m in zip(axes, order):
            d = agg[agg["metric"] == m]
            ok_any = False
            for g, dd in d.groupby("COHORT_StopGrad"):
                dd = dd.sort_values("bin_mid")
                if dd["bin_mid"].nunique() < 3:
                    continue
                c = colors.get(g, "tab:gray")
                ax.plot(dd["bin_mid"], dd["median"], label=g, color=c, lw=2)
                ax.fill_between(dd["bin_mid"], dd["q10"], dd["q90"], alpha=0.18, color=c)
                ok_any = True
            ax.set_ylabel(titles[m]); ax.grid(alpha=0.3)
            if ok_any:
                ax.legend(title="StopGrad", ncol=3, fontsize=8)
        axes[-1].set_xlabel(p)
        fig.suptitle(f"Quantile trend (StopGrad overlay): {p}")
        fig.tight_layout(rect=[0,0,1,0.96])
        fig.savefig(out_png, dpi=140); plt.close(fig)
        return True

    def _plot_trend_facets(agg: pd.DataFrame, p: str, out_png: str):
        order  = [m for m in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7"]
                  if (agg["metric"] == m).any()]
        titles = {"val_sample_pr_auc": "PR-AUC (↑)",
                  "val_fp_per_min":    "FP/min (↓)",
                  "val_latency_score": "Latency (↑)",
                  "val_recall_at_0p7": "Recall@0.7 (↑)"}
        cohorts = [c for c in ["SG=False","SG=True","SG=unknown"] if (agg["COHORT_StopGrad"] == c).any()]
        if agg.empty or len(cohorts) == 0 or agg["bin_mid"].nunique() < 3 or len(order) == 0:
            return False

        fig, axes = plt.subplots(len(cohorts), len(order), figsize=(5.5*len(order), 3.5*len(cohorts)), sharex=True)
        if len(cohorts) == 1 and len(order) == 1:
            axes = np.array([[axes]])
        elif len(cohorts) == 1:
            axes = np.expand_dims(axes, 0)
        elif len(order) == 1:
            axes = np.expand_dims(axes, 1)

        for row, cohort in enumerate(cohorts):
            for col, m in enumerate(order):
                ax = axes[row, col]
                dd = agg[(agg["COHORT_StopGrad"] == cohort) & (agg["metric"] == m)].sort_values("bin_mid")
                if dd["bin_mid"].nunique() >= 3:
                    ax.plot(dd["bin_mid"], dd["median"], lw=2)
                    ax.fill_between(dd["bin_mid"], dd["q10"], dd["q90"], alpha=0.2)
                    for xm, cnt in zip(dd["bin_mid"], dd["count"]):
                        ax.text(xm, dd["median"].min(), f"n={int(cnt)}", fontsize=7, ha="center", va="top", alpha=0.35)
                else:
                    ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes, alpha=0.6)
                if row == 0:
                    ax.set_title(titles[m])
                if col == 0:
                    ax.set_ylabel(cohort)
                ax.grid(alpha=0.3)
        axes[-1, -1].set_xlabel(p)
        fig.suptitle(f"Quantile trend (StopGrad facets): {p}")
        fig.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(out_png, dpi=140); plt.close(fig)
        return True

    trend_params = [c for c in df.columns
                    if c not in ["trial_number","val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7",
                                 "val_sample_max_f1","val_sample_max_mcc","COHORT_StopGrad","score_pr","score_fp","combined_avg","sel_epoch"]
                    and pd.api.types.is_numeric_dtype(df[c])
                    and df[c].nunique(dropna=True) >= 3]

    for must in ["learning_rate","LOSS_SupCon","LOSS_TupMPN","LOSS_NEGATIVES","LOSS_TV"]:
        if must in df.columns and must not in trend_params and pd.api.types.is_numeric_dtype(df[must]) and df[must].nunique(dropna=True) >= 3:
            trend_params.append(must)

    _created_stopgrad_imgs = []
    for p in trend_params:
        try:
            sub = df[["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7","COHORT_StopGrad", p] \
                     if "val_latency_score" in df.columns and "val_recall_at_0p7" in df.columns else \
                     [c for c in ["val_sample_pr_auc","val_fp_per_min","COHORT_StopGrad", p, "val_latency_score","val_recall_at_0p7"] if c in df.columns]].copy()
            agg = _trend_by_bins_cohort(sub, p, nbins=12)
            if agg.empty or agg["bin_mid"].nunique() < 3:
                print(f"[trend] Skipped {p}: insufficient data after binning")
                continue
            out1 = os.path.join(qt_dir, f"{p}_trend_stopgrad_overlay.png")
            out2 = os.path.join(qt_dir, f"{p}_trend_stopgrad_facets.png")
            ok1 = _plot_trend_overlay(agg, p, out1)
            ok2 = _plot_trend_facets(agg, p, out2)
            if ok1: _created_stopgrad_imgs.append(os.path.relpath(out1, viz_dir))
            if ok2: _created_stopgrad_imgs.append(os.path.relpath(out2, viz_dir))
        except Exception as e:
            print(f"[trend] Skipped {p}: {e}")

    # ---------- CORRELATIONS ----------
    heatmap_fp = None
    obj_corr_fp = None
    try:
        # include extra objectives if present
        obj_cols_all = [c for c in ["val_sample_pr_auc","val_fp_per_min",
                                    "val_latency_score","val_recall_at_0p7",
                                    "val_sample_max_f1","val_sample_max_mcc"]
                        if c in df.columns]
        num_cols = [c for c in df.columns
                    if c not in ["trial_number", *obj_cols_all, "COHORT_StopGrad","score_pr","score_fp","combined_avg","sel_epoch"]
                    and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            corr_df = df[num_cols + obj_cols_all].corr(method='spearman')
            plt.figure(figsize=(max(9, 0.6*len(corr_df.columns)), max(6, 0.5*len(corr_df))))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=False)
            plt.title("Spearman correlation: numeric hyperparameters vs objectives")
            plt.tight_layout()
            heatmap_fp = os.path.join(viz_dir, "correlations_spearman.png")
            plt.savefig(heatmap_fp, dpi=130); plt.close()
    except Exception as e:
        print(f"Correlation heatmap warning: {e}"); plt.close()

    try:
        # Objective-to-objective correlations (whatever exists)
        obj_cols = [c for c in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7",
                                "val_sample_max_f1","val_sample_max_mcc"] if c in df.columns]
        if len(obj_cols) >= 2:
            obj_corr = df[obj_cols].corr(method='spearman')
            plt.figure(figsize=(max(3.8, 0.9*len(obj_cols)), max(3.6, 0.8*len(obj_cols))))
            sns.heatmap(obj_corr, annot=True, fmt=".2f", cmap="vlag", cbar=False,
                        xticklabels=obj_cols, yticklabels=obj_cols)
            plt.tight_layout()
            obj_corr_fp = os.path.join(viz_dir, "objective_correlations.png")
            plt.savefig(obj_corr_fp, dpi=140); plt.close()
    except Exception as e:
        print(f"Objective correlation warning: {e}"); plt.close()

    # ---------- SIMPLE COMBINED RANK (top-k) ----------
    def robust_minmax(x):
        q05, q95 = np.nanquantile(x, [0.05, 0.95])
        d = max(1e-9, q95 - q05); z = (x - q05) / d
        return np.clip(z, 0, 1)

    df["score_pr"] = df["val_sample_pr_auc"]                 # higher better
    df["score_fp"] = 1 - robust_minmax(df["val_fp_per_min"]) # lower fp → higher score
    df["combined_avg"] = (df["score_pr"] + df["score_fp"]) / 2.0
    top_combined = df.sort_values("combined_avg", ascending=False).head(25)
    top_combined.to_csv(os.path.join(viz_dir, "top25_combined_avg.csv"), index=False)

    # Top-by-objective HTML tables with links (extended)
    try:
        top_k = 25
        best_pr  = df.sort_values("val_sample_pr_auc", ascending=False).head(top_k).copy()
        best_fp  = df.sort_values("val_fp_per_min", ascending=True ).head(top_k).copy()
        best_lat = df.sort_values("val_latency_score", ascending=False).head(top_k).copy() if "val_latency_score" in df.columns else None
        best_rec = df.sort_values("val_recall_at_0p7", ascending=False).head(top_k).copy() if "val_recall_at_0p7" in df.columns else None

        def _write_top(fname, ddd):
            if ddd is None or ddd.empty: return
            dd = ddd.copy()
            dd["study_dir"] = dd["trial_number"].apply(_trial_link)
            lead = ["trial_number","val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7","combined_avg","study_dir"]
            keep = [c for c in lead if c in dd.columns] + [c for c in dd.columns if c not in lead]
            html_tbl = dd[keep].to_html(escape=False, index=False)
            with open(os.path.join(viz_dir, fname), "w") as fh:
                fh.write(f"<html><head><meta charset='utf-8'><style>body{{font-family:Arial;margin:20px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px}} th{{background:#f5f5f5}}</style></head><body>{html_tbl}</body></html>")

        for fname, ddd in [("top_by_pr_auc.html", best_pr),
                           ("top_by_fpmin.html", best_fp),
                           ("top_by_latency.html", best_lat),
                           ("top_by_recall.html",  best_rec),
                           ("top_by_combined.html", top_combined)]:
            _write_top(fname, ddd)

    except Exception as e:
        print(f"Top-k HTML warning: {e}")

    # =========================
    # EXTRA ANALYSIS ADD-ONS
    # =========================
    extras_dir    = os.path.join(viz_dir, "extras");                os.makedirs(extras_dir, exist_ok=True)
    qplots_dir    = os.path.join(extras_dir, "quantile_trends");    os.makedirs(qplots_dir, exist_ok=True)
    heat2d_dir    = os.path.join(extras_dir, "heatmaps_2d");        os.makedirs(heat2d_dir, exist_ok=True)
    cohort_dir    = os.path.join(extras_dir, "cohorts");            os.makedirs(cohort_dir, exist_ok=True)
    recommend_dir = os.path.join(extras_dir, "range_recommendations"); os.makedirs(recommend_dir, exist_ok=True)
    recon_dir     = os.path.join(extras_dir, "importance_reconciliation"); os.makedirs(recon_dir, exist_ok=True)

    objective_cols = [c for c in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7"]
                      if c in df.columns]
    param_cols = [c for c in df.columns if c not in ["trial_number", *objective_cols,
                                                     "score_pr","score_fp","combined_avg","COHORT_StopGrad","sel_epoch"]]
    num_params_all = [p for p in param_cols if pd.api.types.is_numeric_dtype(df[p])]
    cat_params = [p for p in param_cols if p not in num_params_all]

    # Cohort analysis (StopGrad ON/OFF)
    def _detect_stopgrad_series(dfx: pd.DataFrame) -> pd.Series:
        if "USE_StopGrad" in dfx.columns:
            s = dfx["USE_StopGrad"]
            if s.dtype == bool:
                return s
            return s.map(lambda v: bool(v) if pd.notna(v) else False)
        if "TYPE_ARCH" in dfx.columns:
            return dfx["TYPE_ARCH"].astype(str).str.lower().str.contains("stopgrad", na=False)
        return None

    stopgrad_series = _detect_stopgrad_series(df)

    def _safe_corr_heatmap(data: pd.DataFrame, name: str, out_png: str):
        try:
            cols = [*num_params_all, *objective_cols]
            cols = [c for c in cols if c in data.columns]
            if len(cols) < 3: return
            corr_df = data[cols].corr(method="spearman")
            plt.figure(figsize=(max(8, 0.6*len(corr_df.columns)), max(6, 0.5*len(corr_df))))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title(f"Spearman (numeric) — {name}")
            plt.tight_layout()
            plt.savefig(out_png, dpi=130); plt.close()
        except Exception:
            plt.close()

    def _param_summary_scatter(data: pd.DataFrame, name: str):
        for obj, ylabel in [(c, {"val_sample_pr_auc":"PR-AUC (↑)","val_fp_per_min":"FP/min (↓)",
                                 "val_latency_score":"Latency (↑)","val_recall_at_0p7":"Recall@0.7 (↑)"}[c]) for c in objective_cols]:
            plt.figure(figsize=(max(8, 0.4*len(param_cols)), 4))
            ax = plt.gca()
            x_idx, x_labs = [], []
            for i, p in enumerate(param_cols):
                if p not in data.columns or data[p].nunique(dropna=True) <= 1:
                    continue
                if pd.api.types.is_numeric_dtype(data[p]):
                    valid = data[[p, obj]].dropna()
                    if valid[p].nunique() < 3 or len(valid) < 8:
                        continue
                    qbins = pd.qcut(valid[p], q=min(10, max(3, valid[p].nunique())), duplicates="drop")
                    means = valid.groupby(qbins, observed=True)[obj].mean().values
                    ax.scatter(np.full_like(means, len(x_idx)), means, s=26, alpha=0.7)
                else:
                    g = data.groupby(p, observed=True)[obj].mean().sort_values(ascending=False)
                    ax.scatter(np.full_like(g.values, len(x_idx)), g.values, s=26, alpha=0.7)
                x_idx.append(len(x_idx)); x_labs.append(p)
            ax.set_xticks(range(len(x_labs))); ax.set_xticklabels(x_labs, rotation=28, ha="right")
            ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} vs params — {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(cohort_dir, f"{obj}_vs_params_{name}.png"), dpi=120); plt.close()

    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)
        d0 = df[~sg_mask].copy()  # StopGrad OFF
        d1 = df[ sg_mask].copy()  # StopGrad ON
        if len(d0) >= 30:
            _safe_corr_heatmap(d0, "StopGrad_OFF", os.path.join(cohort_dir, "corr_spearman_StopGrad_OFF.png"))
            _param_summary_scatter(d0, "StopGrad_OFF")
        if len(d1) >= 30:
            _safe_corr_heatmap(d1, "StopGrad_ON",  os.path.join(cohort_dir, "corr_spearman_StopGrad_ON.png"))
            _param_summary_scatter(d1, "StopGrad_ON")

    # Quantile trends (global)
    def quantile_trend(x: pd.Series, y: pd.Series, q=12):
        valid = x.notna() & y.notna()
        xv, yv = x[valid], y[valid]
        if len(xv) < 10 or xv.nunique() < 3:
            return None
        bins = pd.qcut(xv, q=min(q, max(3, xv.nunique())), duplicates="drop")
        grp  = pd.DataFrame({"y": yv, "bin": bins, "x": xv}).groupby("bin", observed=True)
        mu = grp["y"].mean().values
        sd = grp["y"].std().values
        n  = grp["y"].size().values.astype(float)
        se = np.where(n>1, sd/np.sqrt(n), np.nan)
        xc = grp["x"].mean().values
        return xc, mu, se

    num_params_simple = [c for c in df.columns
                         if c not in ["trial_number","val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7",
                                      "val_sample_max_f1","val_sample_max_mcc","COHORT_StopGrad","score_pr","score_fp","combined_avg","sel_epoch"]
                         and pd.api.types.is_numeric_dtype(df[c])
                         and df[c].nunique(dropna=True) >= 3]

    for p in num_params_simple:
        fig, axs = plt.subplots(len(objective_cols), 1, figsize=(8, 3.2*len(objective_cols)), sharex=True)
        if len(objective_cols) == 1: axs = [axs]
        ok_any = False
        for ax, obj in zip(axs, objective_cols):
            label_map = {"val_sample_pr_auc":"PR-AUC (↑)", "val_fp_per_min":"FP/min (↓)",
                         "val_latency_score":"Latency (↑)", "val_recall_at_0p7":"Recall@0.7 (↑)"}
            out = quantile_trend(df[p], df[obj], q=12)
            if out is None:
                ax.set_ylabel(label_map[obj]); continue
            x, mu, se = out
            ax.plot(x, mu, marker="o", linewidth=1.5)
            if np.isfinite(se).any():
                ax.fill_between(x, mu - 1.96*np.nan_to_num(se), mu + 1.96*np.nan_to_num(se), alpha=0.2)
            ax.set_ylabel(label_map[obj]); ok_any = True
        axs[-1].set_xlabel(p)
        if ok_any:
            fig.suptitle(f"Quantile trend — {p}")
            fig.tight_layout(rect=[0,0,1,0.97])
            fig.savefig(os.path.join(qplots_dir, f"{p}_quantile_trends.png"), dpi=130)
        plt.close(fig)

    # 2D interaction heatmaps for informative pairs (example axes)
    def heat2d(x: pd.Series, y: pd.Series, z: pd.Series, xq=12, yq=12):
        valid = x.notna() & y.notna() & z.notna()
        xv, yv, zv = x[valid], y[valid], z[valid]
        if xv.nunique() < 4 or yv.nunique() < 4 or len(zv) < 25:
            return None
        xb = pd.qcut(xv, q=min(xq, max(4, xv.nunique())), duplicates="drop")
        yb = pd.qcut(yv, q=min(yq, max(4, yv.nunique())), duplicates="drop")
        grid = pd.DataFrame({"xb": xb, "yb": yb, "z": zv}).groupby(["xb","yb"], observed=True)["z"].mean().unstack()
        return grid

    def _maybe_heatmap(xname, yname):
        if xname not in df.columns or yname not in df.columns:
            return
        for obj, lab in [("val_sample_pr_auc","PR-AUC (↑)"),
                         ("val_fp_per_min","FP/min (↓)"),
                         ("val_latency_score","Latency (↑)"),
                         ("val_recall_at_0p7","Recall@0.7 (↑)")]:
            if obj not in df.columns: continue
            H = heat2d(df[xname], df[yname], df[obj])
            if H is None:
                continue
            plt.figure(figsize=(6.8,5.2))
            sns.heatmap(H, cmap="viridis", annot=False)
            plt.title(f"{lab} mean — {xname} × {yname}")
            plt.tight_layout()
            plt.savefig(os.path.join(heat2d_dir, f"{obj}_{xname}_x_{yname}.png"), dpi=140)
            plt.close()

    if "LOSS_SupCon" in df.columns and "LOSS_TupMPN" in df.columns:
        _maybe_heatmap("LOSS_SupCon", "LOSS_TupMPN")
    if "learning_rate" in df.columns and "LOSS_NEGATIVES" in df.columns:
        _maybe_heatmap("learning_rate", "LOSS_NEGATIVES")
    if "LOSS_TV" in df.columns and "LOSS_TupMPN" in df.columns:
        _maybe_heatmap("LOSS_TV", "LOSS_TupMPN")

    # Range recommendations: top-quartile trial bands per objective (extended)
    def recommend_ranges(data: pd.DataFrame, name: str, top_frac=0.25):
        metric_list = [c for c in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7"]
                       if c in data.columns and not data[c].isnull().all()]
        nice_name = {"val_sample_pr_auc":"PR-AUC (↑)",
                     "val_fp_per_min":"FP/min (↓)",
                     "val_latency_score":"Latency (↑)",
                     "val_recall_at_0p7":"Recall@0.7 (↑)"}
        num_params_local = [p for p in data.columns
                            if p not in ["trial_number", *metric_list, "score_pr","score_fp","combined_avg","COHORT_StopGrad","sel_epoch"]
                            and pd.api.types.is_numeric_dtype(data[p])]
        cat_params_local = [p for p in data.columns
                            if p not in ["trial_number", *metric_list, "score_pr","score_fp","combined_avg","COHORT_StopGrad","sel_epoch"]
                            and p not in num_params_local]
        recs = {}
        for obj in metric_list:
            asc = (obj == "val_fp_per_min")  # only FP/min is minimize
            dsort = data.sort_values(obj, ascending=asc)
            top_n = max(20, int(len(dsort)*top_frac))
            top   = dsort.head(top_n)
            rng = {}
            for p in num_params_local:
                col = pd.to_numeric(top[p], errors="coerce").dropna()
                if col.nunique() < 2:
                    continue
                try:
                    rng[p] = (float(np.nanquantile(col, 0.20)), float(np.nanquantile(col, 0.80)))
                except Exception:
                    pass
            cat = {}
            for p in cat_params_local:
                vc = top[p].value_counts(normalize=True, dropna=False)
                if len(vc):
                    cat[p] = vc.head(3).to_dict()
            recs[nice_name[obj]] = {"num_quantile_20_80": rng, "cat_top3_props": cat, "n_top": int(len(top))}
        with open(os.path.join(recommend_dir, f"ranges_{name}.json"), "w") as fh:
            json.dump(recs, fh, indent=2)
        rows = []
        for obj, payload in recs.items():
            for p,(lo,hi) in payload["num_quantile_20_80"].items():
                rows.append({"objective": obj, "param": p, "q20": lo, "q80": hi})
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(recommend_dir, f"ranges_numeric_{name}.csv"), index=False)

    recommend_ranges(df, "ALL")
    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)
        if df[~sg_mask].shape[0] >= 40: recommend_ranges(df[~sg_mask], "StopGrad_OFF")
        if df[ sg_mask].shape[0] >= 40: recommend_ranges(df[ sg_mask], "StopGrad_ON")

    # Importance reconciliation: absolute Spearman vs PR-AUC (global/cohort) — keep as-is for PR-AUC
    imp_rows = []
    for p in num_params_all:
        try:
            r = df[[p,"val_sample_pr_auc"]].dropna().corr(method="spearman").iloc[0,1]
            if pd.notna(r):
                imp_rows.append({"param": p, "source": "Spearman|rho| (global)", "value": abs(float(r))})
        except Exception:
            pass
    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)
        for name, dsub in [("StopGrad_OFF", df[~sg_mask]), ("StopGrad_ON", df[sg_mask])]:
            if len(dsub) < 30: continue
            for p in num_params_all:
                try:
                    rr = dsub[[p,"val_sample_pr_auc"]].dropna().corr(method="spearman").iloc[0,1]
                    if pd.notna(rr):
                        imp_rows.append({"param": p, "source": f"Spearman|rho| ({name})", "value": abs(float(rr))})
                except Exception:
                    pass
    if imp_rows:
        imp_df = pd.DataFrame(imp_rows)
        piv = imp_df.pivot_table(index="param", columns="source", values="value", aggfunc="max").fillna(0.0)
        piv = piv.sort_values(by=list(piv.columns)[0], ascending=False)
        ax = piv.plot(kind="bar", figsize=(max(9, 0.55*len(piv)),5))
        ax.set_ylabel("|Spearman| vs PR-AUC"); plt.title("Correlation-based importances (compare with Optuna fANOVA)")
        plt.tight_layout()
        plt.savefig(os.path.join(recon_dir, "spearman_vs_pr_auc.png"), dpi=130)
        plt.close()

    # ---------- HTML REPORT ----------
    html = []
    html.append(f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Study Visualization (2-obj): {study.study_name}</title>
    <style>
    body{{font-family:Arial,Helvetica,sans-serif;margin:20px;}} h2{{margin-top:28px}}
    a{{color:#007bff;text-decoration:none}} a:hover{{text-decoration:underline}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
    .card{{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}}
    img{{max-width:100%}}
    .notice{{padding:10px;border:1px solid #ffc107;background:#fff3cd;border-radius:6px}}
    </style></head><body>
    <h1>Visualization — {study.study_name}</h1>
    <p>Objectives: <b>maximize</b> PR-AUC, <b>minimize</b> FP/min.</p>
    <p><b>Trials:</b> {len(df)} | <b>Pareto set:</b> {len(pareto_df)}</p>""")

    html.append(f"""
    <ul>
      <li><a href="all_completed_trials.csv">all_completed_trials.csv</a></li>
      <li><a href="pareto_trials.csv">pareto_trials.csv</a> &nbsp;|&nbsp; <a href="pareto_trials.html">Pareto table (HTML)</a></li>
      <li><a href="top25_combined_avg.csv">top25_combined_avg.csv</a></li>
      <li>Top-K (HTML): <a href="top_by_pr_auc.html">PR-AUC</a> · <a href="top_by_fpmin.html">FP/min</a> · <a href="top_by_latency.html">Latency</a> · <a href="top_by_recall.html">Recall</a> · <a href="top_by_combined.html">Combined</a></li>
      <li>Objective EDFs: <a href="edf_pr_auc.html">PR-AUC</a> · <a href="edf_fp_per_min.html">FP/min</a> · <a href="edf_latency.html">Latency</a> · <a href="edf_recall.html">Recall</a></li>
    </ul>""")

    if pareto_snapshot_html:
        html.append("<h2>Pareto snapshot (top 15 by PR-AUC)</h2>")
        html.append(pareto_snapshot_html)

    html.append("""
    <h2>Standard Optuna Plots</h2>
    <div class="grid">
      <div class="card"><a href="history_pr_auc.html">Optimization History — PR-AUC</a></div>
      <div class="card"><a href="history_fp_per_min.html">Optimization History — FP/min</a></div>
      <div class="card"><a href="param_importances_pr_auc.html">Param Importances — PR-AUC</a></div>
      <div class="card"><a href="param_importances_fp_per_min.html">Param Importances — FP/min</a></div>
      <div class="card"><a href="history_latency.html">Optimization History — Latency</a></div>
      <div class="card"><a href="history_recall.html">Optimization History — Recall@0.7</a></div>
      <div class="card"><a href="param_importances_latency.html">Param Importances — Latency</a></div>
      <div class="card"><a href="param_importances_recall.html">Param Importances — Recall@0.7</a></div>
      <div class="card"><a href="pareto_front_2obj.html">Pareto Front (interactive)</a></div>
      <div class="card"><a href="hypervolume_history.html">Hypervolume History</a></div>
    </div>""")

    html.append("<h2>Objective Projection</h2><div class='grid'>")
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter2d_all.html")):
        html.append('<div class="card"><a href="scatter2d_all.html">PR-AUC vs FP/min — Pareto highlighted</a></div>')
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter2d_prauc_fp_color_latency.html")):
        html.append('<div class="card"><a href="scatter2d_prauc_fp_color_latency.html">PR-AUC vs FP/min — color=Latency</a></div>')
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter2d_prauc_fp_color_recall.html")):
        html.append('<div class="card"><a href="scatter2d_prauc_fp_color_recall.html">PR-AUC vs FP/min — color=Recall@0.7</a></div>')
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter2d_prauc_latency.html")):
        html.append('<div class="card"><a href="scatter2d_prauc_latency.html">PR-AUC vs Latency</a></div>')
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter2d_fpmin_latency.html")):
        html.append('<div class="card"><a href="scatter2d_fpmin_latency.html">FP/min vs Latency</a></div>')
    html.append("</div>")

    # StopGrad trend plots (overlay + facets)
    if len(_created_stopgrad_imgs) > 0:
        html.append("<h2>Quantile Trends — StopGrad</h2><div class='grid'>")
        for imgp in _created_stopgrad_imgs:
            html.append(f'<div class="card"><img src="{imgp}"></div>')
        html.append("</div>")
    else:
        html.append("<h2>Quantile Trends — StopGrad</h2><p class='notice'>No StopGrad trend plots were created (insufficient data or parameters too discrete).</p>")

    # Correlations
    if heatmap_fp or obj_corr_fp:
        html.append("<h2>Correlations</h2><div class='grid'>")
        if heatmap_fp and os.path.exists(heatmap_fp):
            html.append(f'<div class="card"><img src="correlations_spearman.png" alt="Hyperparameter correlations"></div>')
        if obj_corr_fp and os.path.exists(obj_corr_fp):
            html.append(f'<div class="card"><img src="objective_correlations.png" alt="Objective correlations"></div>')
        html.append("</div>")

    # Per-Parameter Impact thumbnails
    html.append("<h2>Per-Parameter Impact</h2><div class='grid'>")
    for p in hyperparams:
        imgp = os.path.join("param_impact", f"{p}_impact.png")
        if os.path.exists(os.path.join(viz_dir, imgp)):
            html.append(f'<div class="card"><h3>{p}</h3><img src="{imgp}"></div>')
    html.append("</div>")

    # Extras links
    html.append("""
    <h2>Extras</h2>
    <ul>
      <li><b>Cohorts:</b> see PNGs in <code>extras/cohorts/</code></li>
      <li><b>Quantile Trends (global):</b> <code>extras/quantile_trends/</code></li>
      <li><b>2D Heatmaps:</b> <code>extras/heatmaps_2d/</code></li>
      <li><b>Range Recommendations:</b> JSON/CSV in <code>extras/range_recommendations/</code></li>
      <li><b>Importance Reconciliation:</b> <code>extras/importance_reconciliation/spearman_vs_pr_auc.png</code></li>
    </ul>
    </body></html>""")

    with open(os.path.join(viz_dir, "index.html"), "w") as f:
        f.write("\n".join(html))

    # ---------- SUMMARY ----------
    stats = {
        "study_name": study.study_name,
        "n_completed_trials": int(len(df)),
        "n_pareto_trials": int(len(pareto_df)),
        "has_constraints": bool(has_constraints),
        "n_feasible_trials": int(len(feasible_trials)),
        "objectives": ["val_sample_pr_auc (max)", "val_fp_per_min (min)"],
        "objective_index_map": OBJECTIVE_INDEX
    }
    with open(os.path.join(viz_dir, "study_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Visualization complete → {viz_dir}")


elif mode == 'tune_viz_multi_v8':
    import os, sys, json, math, warnings, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import optuna

    warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

    # Optional: Plotly
    try:
        import plotly.express as px
        HAVE_PLOTLY = True
    except Exception:
        HAVE_PLOTLY = False

    # ---------- CONFIG ----------
    tag = args.tag[0]
    param_dir = f'params_{tag}'
    storage_url = f"sqlite:///studies/{param_dir}/{param_dir}.db"
    viz_dir = f"studies/{param_dir}/visualizations_v7"
    os.makedirs(viz_dir, exist_ok=True)

    print(f"Loading study '{param_dir}' from {storage_url}")
    try:
        study = optuna.load_study(study_name=param_dir, storage=storage_url)
    except Exception as e:
        print(f"Error loading study: {e}")
        sys.exit(1)

    # ---------- HELPERS ----------
    def _ua(tr, key, default=np.nan):
        # Prefer user_attrs, then system_attrs
        try:
            if hasattr(tr, "user_attrs") and key in tr.user_attrs:
                return tr.user_attrs.get(key, default)
            if hasattr(tr, "system_attrs") and key in tr.system_attrs:
                return tr.system_attrs.get(key, default)
        except Exception:
            pass
        return default

    # ---------- COLLECT COMPLETED TRIALS ----------
    # ---------- COLLECT COMPLETED TRIALS ----------
    trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not trials:
        print("No completed trials; nothing to visualize.")
        sys.exit(0)

    # 1. Determine Objectives
    nvals = max((len(t.values) if t.values else 0) for t in trials)
    print(f"Detected {nvals} objectives per trial.")

    # Fix mapping: Index 0 is PR-AUC, Index 1 is Latency (if present)
    if nvals == 2:
        OBJECTIVE_INDEX = dict(pr_auc=0, latency=1)
        print("Mapping: Index 0 -> PR-AUC, Index 1 -> Latency")
    else:
        OBJECTIVE_INDEX = dict(pr_auc=0)

    # 2. Fix Constraints Logic (Restore missing variables)
    def _get_constraints(tr):
        return tr.system_attrs.get("constraints") or tr.user_attrs.get("constraints")

    has_constraints = any(_get_constraints(t) is not None for t in trials)

    feasible_trials = [
        t for t in trials 
        if (_get_constraints(t) is None) or all((c is not None) and (c <= 0) for c in _get_constraints(t))
    ]
    DO_HV = not (has_constraints and len(feasible_trials) == 0)

    if has_constraints and not DO_HV:
        print("No feasible trials under constraints; skipping hypervolume plot.")

    # 3. Collect Rows
    rows = []
    for t in trials:
        if not t.values: continue
        try:
            # Get Objectives from t.values
            pr = float(t.values[OBJECTIVE_INDEX['pr_auc']])
            lat_obj = float(t.values[OBJECTIVE_INDEX['latency']]) if 'latency' in OBJECTIVE_INDEX else np.nan
            
            # Get FP/min from Attributes (NOT t.values)
            real_fp = _ua(t, "sel_fp_per_min")
            
        except Exception:
            continue

        # Filter bad values
        bad = lambda x: (x is None) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
        if bad(pr): continue

        rec = {
            "trial_number": t.number,
            "val_sample_pr_auc": pr,
            "val_latency_score": lat_obj,
            "val_fp_per_min":    real_fp, # Now safely loaded from attributes
            
            # Other attributes
            "val_recall_at_0p7":  _ua(t, "sel_recall_at_0p7"),
            "val_sample_max_f1":  _ua(t, "sel_max_f1"),
            "val_sample_max_mcc": _ua(t, "sel_max_mcc"),
            "sel_epoch":          _ua(t, "sel_epoch")
        }
        rec.update(t.params)
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid completed trials.")
        sys.exit(0)

    # Save all trials CSV
    all_csv = os.path.join(viz_dir, "all_completed_trials.csv")
    df.to_csv(all_csv, index=False)
    print(f"Saved {len(df)} trials -> {all_csv}")

    # ---------- STANDARD OPTUNA VISUALS ----------
    # ---------- STANDARD OPTUNA VISUALS ----------
    print("Generating Optuna standard plots...")
    try:
        # 1. History - PR-AUC (Objective 0)
        fig_hist_pr = optuna.visualization.plot_optimization_history(
            study, 
            target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'), 
            target_name="Sample PR-AUC"
        )
        fig_hist_pr.write_html(os.path.join(viz_dir, "history_pr_auc.html"))

        # 2. History - Latency (Objective 1)
        if 'latency' in OBJECTIVE_INDEX:
            fig_hist_lat = optuna.visualization.plot_optimization_history(
                study, 
                target=lambda t: t.values[OBJECTIVE_INDEX['latency']] if t.values else float('nan'),
                target_name="Latency Score"
            )
            fig_hist_lat.write_html(os.path.join(viz_dir, "history_latency.html"))

        # 3. History - FP/min (Attribute - NOT Objective)
        # FIX: Use _ua helper, not t.values
        fig_hist_fp = optuna.visualization.plot_optimization_history(
            study, 
            target=lambda t: _ua(t, "sel_fp_per_min"), 
            target_name="FP/min"
        )
        fig_hist_fp.write_html(os.path.join(viz_dir, "history_fp_per_min.html"))

        # 4. Param Importances
        # PR-AUC
        fig_imp_pr = optuna.visualization.plot_param_importances(
            study, 
            target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'), 
            target_name="Sample PR-AUC"
        )
        fig_imp_pr.write_html(os.path.join(viz_dir, "param_importances_pr_auc.html"))
        
        # Latency
        if 'latency' in OBJECTIVE_INDEX:
            fig_imp_lat = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[OBJECTIVE_INDEX['latency']] if t.values else float('nan'),
                target_name="Latency Score"
            )
            fig_imp_lat.write_html(os.path.join(viz_dir, "param_importances_latency.html"))

        # FP/min (Attribute)
        fig_imp_fp = optuna.visualization.plot_param_importances(
            study, 
            target=lambda t: _ua(t, "sel_fp_per_min"), 
            target_name="FP/min"
        )
        fig_imp_fp.write_html(os.path.join(viz_dir, "param_importances_fp_per_min.html"))

        # 5. Pareto Front
        # Only plots the ACTUAL objectives (PR-AUC vs Latency)
        names_by_index = [""] * nvals
        names_by_index[OBJECTIVE_INDEX['pr_auc']] = "Sample PR-AUC"
        if 'latency' in OBJECTIVE_INDEX:
            names_by_index[OBJECTIVE_INDEX['latency']] = "Latency Score"
            
        fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=names_by_index)
        fig_pareto.write_html(os.path.join(viz_dir, "pareto_front_2obj.html"))

    except Exception as e:
        print(f"Standard plot warning: {e}")

    # ---------- EDFs ----------
    try:
        fig_edf_pr = optuna.visualization.plot_edf(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'),
            target_name="Sample PR-AUC (↑)")
        fig_edf_pr.write_html(os.path.join(viz_dir, "edf_pr_auc.html"))

        fig_edf_fp = optuna.visualization.plot_edf(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['fp_per_min']] if t.values else float('nan'),
            target_name="FP/min (↓)")
        fig_edf_fp.write_html(os.path.join(viz_dir, "edf_fp_per_min.html"))

        # ---- NEW: EDFs for latency & recall ----
        try:
            fig_edf_lat = optuna.visualization.plot_edf(
                study, target=lambda t: _ua(t, "sel_latency_score"),
                target_name="Latency score (↑)")
            fig_edf_lat.write_html(os.path.join(viz_dir, "edf_latency.html"))
        except Exception as e:
            print(f"EDF latency warning: {e}")

        try:
            fig_edf_rec = optuna.visualization.plot_edf(
                study, target=lambda t: _ua(t, "sel_recall_at_0p7"),
                target_name="Recall@0.7 (↑)")
            fig_edf_rec.write_html(os.path.join(viz_dir, "edf_recall.html"))
        except Exception as e:
            print(f"EDF recall warning: {e}")

    except Exception as e:
        print(f"EDF plot warning: {e}")

    # ---------- PARETO SET & CSV/HTML ----------
    pareto_trials = study.best_trials
    pareto_nums = [t.number for t in pareto_trials]
    pareto_df = df[df.trial_number.isin(pareto_nums)].copy()
    pareto_csv = os.path.join(viz_dir, "pareto_trials.csv")
    pareto_df.to_csv(pareto_csv, index=False)
    print(f"Pareto set size: {len(pareto_df)} → {pareto_csv}")

    def _trial_link(trial_no: int) -> str:
        base = os.path.join("studies", param_dir)
        matches = glob.glob(os.path.join(base, f"study_{trial_no}_*"))
        if matches:
            rel = os.path.relpath(matches[0], viz_dir)
            return f'<a href="{rel}">study_{trial_no}</a>'
        return ""

    pareto_snapshot_html = ""
    if not pareto_df.empty:
        _pareto_view = pareto_df.copy()
        _pareto_view["study_dir"] = _pareto_view["trial_number"].apply(_trial_link)
        lead_cols = ["trial_number", "val_sample_pr_auc", "val_fp_per_min", "study_dir"]
        remaining = [c for c in _pareto_view.columns if c not in lead_cols]
        pareto_html = _pareto_view[lead_cols + remaining].to_html(escape=False, index=False)
        with open(os.path.join(viz_dir, "pareto_trials.html"), "w") as fh:
            fh.write(f"""<html><head><meta charset="utf-8"><title>Pareto Trials — {study.study_name}</title>
<style>body{{font-family:Arial;margin:20px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px}} th{{background:#f5f5f5}}</style>
</head><body><h2>Pareto Trials</h2>
<p>Higher is better: PR-AUC. Lower is better: FP/min.</p>
{pareto_html}
</body></html>""")
        snapshot = _pareto_view.sort_values("val_sample_pr_auc", ascending=False)[lead_cols].head(15)
        pareto_snapshot_html = snapshot.to_html(escape=False, index=False)

    # ---------- HYPERVOLUME HISTORY ----------
    try:
        if DO_HV:
            ref_point = [0.0] * nvals
            ref_point[OBJECTIVE_INDEX['pr_auc']] = float(df["val_sample_pr_auc"].min() - 1e-3)   # smaller worse
            ref_point[OBJECTIVE_INDEX['fp_per_min']] = float(df["val_fp_per_min"].max() + 1e-3)  # larger worse
            fig_hv = optuna.visualization.plot_hypervolume_history(study, reference_point=ref_point)
            fig_hv.write_html(os.path.join(viz_dir, "hypervolume_history.html"))
        else:
            print("Hypervolume plot skipped due to infeasible trials under constraints.")
    except Exception as e:
        print(f"Hypervolume plot skipped: {e}")

    # ---------- OBJECTIVE PROJECTION (Plotly) ----------
    try:
        if HAVE_PLOTLY:
            fig2d = px.scatter(
                df,
                x="val_sample_pr_auc", y="val_fp_per_min",
                color=np.where(df.trial_number.isin(pareto_nums), "Pareto", "Other"),
                hover_name="trial_number", opacity=0.9
            )
            fig2d.update_layout(
                xaxis_title="PR-AUC (↑)",
                yaxis_title="FP/min (↓)",
                legend_title_text="Trials"
            )
            fig2d.write_html(os.path.join(viz_dir, "scatter2d_all.html"))

            # ---- NEW: PR-AUC vs FP/min colored by Latency ----
            if "val_latency_score" in df.columns:
                fig2d_lat = px.scatter(
                    df, x="val_sample_pr_auc", y="val_fp_per_min",
                    color="val_latency_score", color_continuous_scale="Viridis",
                    hover_name="trial_number",
                    hover_data=[c for c in ["val_recall_at_0p7","val_sample_max_f1","val_sample_max_mcc","sel_epoch"]
                                if c in df.columns],
                    opacity=0.9, title="PR-AUC vs FP/min — color: Latency"
                )
                fig2d_lat.update_layout(xaxis_title="PR-AUC (↑)", yaxis_title="FP/min (↓)")
                fig2d_lat.write_html(os.path.join(viz_dir, "scatter2d_prauc_fp_color_latency.html"))

            # ---- NEW: PR-AUC vs FP/min colored by Recall ----
            if "val_recall_at_0p7" in df.columns:
                fig2d_rec = px.scatter(
                    df, x="val_sample_pr_auc", y="val_fp_per_min",
                    color="val_recall_at_0p7", color_continuous_scale="Plasma",
                    hover_name="trial_number",
                    hover_data=[c for c in ["val_latency_score","val_sample_max_f1","val_sample_max_mcc","sel_epoch"]
                                if c in df.columns],
                    opacity=0.9, title="PR-AUC vs FP/min — color: Recall@0.7"
                )
                fig2d_rec.update_layout(xaxis_title="PR-AUC (↑)", yaxis_title="FP/min (↓)")
                fig2d_rec.write_html(os.path.join(viz_dir, "scatter2d_prauc_fp_color_recall.html"))

            # ---- NEW: Pairwise with Latency ----
            if "val_latency_score" in df.columns:
                fig_p_l = px.scatter(
                    df, x="val_sample_pr_auc", y="val_latency_score",
                    color=np.where(df.trial_number.isin(pareto_nums), "Pareto(PR↔FP)", "Other"),
                    hover_name="trial_number", opacity=0.9,
                    title="PR-AUC vs Latency (↑/↑)"
                )
                fig_p_l.update_layout(xaxis_title="PR-AUC (↑)", yaxis_title="Latency score (↑)")
                fig_p_l.write_html(os.path.join(viz_dir, "scatter2d_prauc_latency.html"))

                fig_f_l = px.scatter(
                    df, x="val_fp_per_min", y="val_latency_score",
                    color=np.where(df.trial_number.isin(pareto_nums), "Pareto(PR↔FP)", "Other"),
                    hover_name="trial_number", opacity=0.9,
                    title="FP/min vs Latency (↓/↑)"
                )
                fig_f_l.update_layout(xaxis_title="FP/min (↓)", yaxis_title="Latency score (↑)")
                fig_f_l.write_html(os.path.join(viz_dir, "scatter2d_fpmin_latency.html"))
    except Exception as e:
        print(f"Projection plot warning: {e}")

    # ---------- PER-PARAM IMPACT (quick plots) ----------
    param_impact_dir = os.path.join(viz_dir, "param_impact")
    os.makedirs(param_impact_dir, exist_ok=True)

    def qylim(series, lo=0.05, hi=0.95, pad=0.05, clamp=None):
        if series.isnull().all():
            return (0, 1)
        qlo, qhi = np.nanquantile(series, [lo, hi])
        span = max(1e-9, qhi - qlo)
        lo_v = qlo - pad * span
        hi_v = qhi + pad * span
        if clamp:
            lo_v = max(clamp[0], lo_v); hi_v = min(clamp[1], hi_v)
            if lo_v >= hi_v: lo_v, hi_v = clamp[0], clamp[1]
        return (lo_v, hi_v)

    pr_ylim = qylim(df["val_sample_pr_auc"], clamp=(0,1))
    fp_ylim = qylim(df["val_fp_per_min"], clamp=None)

    hyperparams = [c for c in df.columns if c not in [
        "trial_number","val_sample_pr_auc","val_fp_per_min",
        "val_latency_score","val_recall_at_0p7","val_sample_max_f1","val_sample_max_mcc",
        "score_pr","score_fp","combined_avg","COHORT_StopGrad","sel_epoch"
    ]]

    targets = [("val_sample_pr_auc","PR-AUC (↑)"),
               ("val_fp_per_min","FP/min (↓)")]
    if "val_latency_score" in df.columns:   targets.append(("val_latency_score","Latency (↑)"))
    if "val_recall_at_0p7" in df.columns:   targets.append(("val_recall_at_0p7","Recall@0.7 (↑)"))

    for p in hyperparams:
        if df[p].isnull().all() or df[p].nunique(dropna=True) <= 1:
            continue
        is_num = pd.api.types.is_numeric_dtype(df[p])
        fig, axes = plt.subplots(len(targets), 1, figsize=(10, 3.2*len(targets)), sharex=True)
        if len(targets) == 1: axes = [axes]

        for ax, (obj, ylabel) in zip(axes, targets):
            ylim = None
            if obj == "val_sample_pr_auc": ylim = pr_ylim
            elif obj == "val_fp_per_min": ylim = fp_ylim
            if is_num:
                sns.scatterplot(data=df, x=p, y=obj, ax=ax, s=18, alpha=0.5)
            else:
                sns.stripplot(data=df, x=p, y=obj, ax=ax, size=4, alpha=0.7, jitter=True)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_ylabel(ylabel)
            if ylim is not None: ax.set_ylim(ylim)
        axes[-1].set_xlabel(p)
        fig.tight_layout()
        outp = os.path.join(param_impact_dir, f"{p}_impact.png")
        fig.savefig(outp, dpi=130)
        plt.close(fig)

    # ---------- STOPGRAD QUANTILE TRENDS (overlay + facets) ----------
    qt_dir = os.path.join(viz_dir, "quantile_trends_stopgrad")
    os.makedirs(qt_dir, exist_ok=True)

    def _infer_stopgrad_column(df_in: pd.DataFrame) -> pd.Series:
        if "USE_StopGrad" in df_in.columns:
            col = df_in["USE_StopGrad"]
            if pd.api.types.is_bool_dtype(col):
                return pd.Series(np.where(col, "SG=True", "SG=False"), index=col.index)
            try:
                val = pd.to_numeric(col, errors="coerce")
                return pd.Series(np.where(val == 1, "SG=True",
                                   np.where(val == 0, "SG=False", "SG=unknown")), index=col.index)
            except Exception:
                s = col.astype(str).str.lower()
                return pd.Series(np.where(s.isin(["1","true","yes","y"]), "SG=True",
                                   np.where(s.isin(["0","false","no","n"]), "SG=False", "SG=unknown")), index=col.index)
        if "TYPE_ARCH" in df_in.columns:
            s = df_in["TYPE_ARCH"].astype(str).str.lower()
            return pd.Series(np.where(s.str.contains("stopgrad", na=False), "SG=True", "SG=False"), index=df_in.index)
        return pd.Series(["SG=unknown"] * len(df_in), index=df_in.index)

    df["COHORT_StopGrad"] = _infer_stopgrad_column(df)

    def _quantile_edges(x: pd.Series, nbins=12):
        x = pd.to_numeric(x, errors="coerce").dropna()
        if x.nunique() < 3 or len(x) < 12:
            if len(x) == 0:
                return None
            lo, hi = float(x.min()), float(x.max())
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                return None
            return np.linspace(lo, hi, 4)
        qs = np.linspace(0, 1, nbins + 1)
        edges = np.unique(np.nanquantile(x, qs))
        if len(edges) < 4:
            lo, hi = float(x.min()), float(x.max())
            edges = np.linspace(lo, hi, 4)
        eps = 1e-12 * (edges[-1] - edges[0] + 1.0)
        edges[0] -= eps; edges[-1] += eps
        return edges

    def _trend_by_bins_cohort(dfin: pd.DataFrame, param: str, nbins=12):
        keep_cols = ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7","COHORT_StopGrad", param]
        keep_cols = [c for c in keep_cols if c in dfin.columns]
        dfin = dfin[keep_cols].copy()
        dfin[param] = pd.to_numeric(dfin[param], errors="coerce")
        dfin = dfin.dropna(subset=[param])

        edges = _quantile_edges(dfin[param], nbins=nbins)
        if edges is None:
            return pd.DataFrame()

        b = pd.cut(dfin[param], bins=edges, include_lowest=True)
        mids = b.apply(lambda iv: np.mean([iv.left, iv.right]) if pd.notnull(iv) else np.nan)
        tmp = dfin.copy()
        tmp["_bin"] = b
        tmp["_mid"] = mids

        value_vars = [c for c in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7"] if c in tmp.columns]
        long = tmp.melt(
            id_vars=["_bin","_mid","COHORT_StopGrad"],
            value_vars=value_vars,
            var_name="metric", value_name="val"
        )

        agg = (long.dropna(subset=["_bin","_mid","val"])
                    .groupby(["_bin","_mid","COHORT_StopGrad","metric"], observed=True)
                    .agg(q10=("val", lambda s: np.nanquantile(s, 0.10)),
                         median=("val", "median"),
                         q90=("val", lambda s: np.nanquantile(s, 0.90)),
                         count=("val", "size"))
                    .reset_index())
        agg.rename(columns={"_mid": "bin_mid"}, inplace=True)
        return agg

    def _plot_trend_overlay(agg: pd.DataFrame, p: str, out_png: str):
        order = [m for m in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7"]
                 if (agg["metric"] == m).any()]
        titles = {
            "val_sample_pr_auc": "PR-AUC (↑)",
            "val_fp_per_min":    "FP/min (↓)",
            "val_latency_score": "Latency (↑)",
            "val_recall_at_0p7": "Recall@0.7 (↑)"
        }
        colors = {"SG=False":"tab:blue","SG=True":"tab:orange","SG=unknown":"tab:gray"}

        if agg.empty or agg["bin_mid"].nunique() < 3 or len(order) == 0:
            return False

        fig, axes = plt.subplots(len(order), 1, figsize=(9, 3.3*len(order)), sharex=True)
        if len(order) == 1: axes = [axes]
        for ax, m in zip(axes, order):
            d = agg[agg["metric"] == m]
            ok_any = False
            for g, dd in d.groupby("COHORT_StopGrad"):
                dd = dd.sort_values("bin_mid")
                if dd["bin_mid"].nunique() < 3:
                    continue
                c = colors.get(g, "tab:gray")
                ax.plot(dd["bin_mid"], dd["median"], label=g, color=c, lw=2)
                ax.fill_between(dd["bin_mid"], dd["q10"], dd["q90"], alpha=0.18, color=c)
                ok_any = True
            ax.set_ylabel(titles[m]); ax.grid(alpha=0.3)
            if ok_any:
                ax.legend(title="StopGrad", ncol=3, fontsize=8)
        axes[-1].set_xlabel(p)
        fig.suptitle(f"Quantile trend (StopGrad overlay): {p}")
        fig.tight_layout(rect=[0,0,1,0.96])
        fig.savefig(out_png, dpi=140); plt.close(fig)
        return True

    def _plot_trend_facets(agg: pd.DataFrame, p: str, out_png: str):
        order  = [m for m in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7"]
                  if (agg["metric"] == m).any()]
        titles = {"val_sample_pr_auc": "PR-AUC (↑)",
                  "val_fp_per_min":    "FP/min (↓)",
                  "val_latency_score": "Latency (↑)",
                  "val_recall_at_0p7": "Recall@0.7 (↑)"}
        cohorts = [c for c in ["SG=False","SG=True","SG=unknown"] if (agg["COHORT_StopGrad"] == c).any()]
        if agg.empty or len(cohorts) == 0 or agg["bin_mid"].nunique() < 3 or len(order) == 0:
            return False

        fig, axes = plt.subplots(len(cohorts), len(order), figsize=(5.5*len(order), 3.5*len(cohorts)), sharex=True)
        if len(cohorts) == 1 and len(order) == 1:
            axes = np.array([[axes]])
        elif len(cohorts) == 1:
            axes = np.expand_dims(axes, 0)
        elif len(order) == 1:
            axes = np.expand_dims(axes, 1)

        for row, cohort in enumerate(cohorts):
            for col, m in enumerate(order):
                ax = axes[row, col]
                dd = agg[(agg["COHORT_StopGrad"] == cohort) & (agg["metric"] == m)].sort_values("bin_mid")
                if dd["bin_mid"].nunique() >= 3:
                    ax.plot(dd["bin_mid"], dd["median"], lw=2)
                    ax.fill_between(dd["bin_mid"], dd["q10"], dd["q90"], alpha=0.2)
                    for xm, cnt in zip(dd["bin_mid"], dd["count"]):
                        ax.text(xm, dd["median"].min(), f"n={int(cnt)}", fontsize=7, ha="center", va="top", alpha=0.35)
                else:
                    ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes, alpha=0.6)
                if row == 0:
                    ax.set_title(titles[m])
                if col == 0:
                    ax.set_ylabel(cohort)
                ax.grid(alpha=0.3)
        axes[-1, -1].set_xlabel(p)
        fig.suptitle(f"Quantile trend (StopGrad facets): {p}")
        fig.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(out_png, dpi=140); plt.close(fig)
        return True

    trend_params = [c for c in df.columns
                    if c not in ["trial_number","val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7",
                                 "val_sample_max_f1","val_sample_max_mcc","COHORT_StopGrad","score_pr","score_fp","combined_avg","sel_epoch"]
                    and pd.api.types.is_numeric_dtype(df[c])
                    and df[c].nunique(dropna=True) >= 3]

    for must in ["learning_rate","LOSS_SupCon","LOSS_TupMPN","LOSS_NEGATIVES","LOSS_TV"]:
        if must in df.columns and must not in trend_params and pd.api.types.is_numeric_dtype(df[must]) and df[must].nunique(dropna=True) >= 3:
            trend_params.append(must)

    _created_stopgrad_imgs = []
    for p in trend_params:
        try:
            sub = df[["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7","COHORT_StopGrad", p] \
                     if "val_latency_score" in df.columns and "val_recall_at_0p7" in df.columns else \
                     [c for c in ["val_sample_pr_auc","val_fp_per_min","COHORT_StopGrad", p, "val_latency_score","val_recall_at_0p7"] if c in df.columns]].copy()
            agg = _trend_by_bins_cohort(sub, p, nbins=12)
            if agg.empty or agg["bin_mid"].nunique() < 3:
                print(f"[trend] Skipped {p}: insufficient data after binning")
                continue
            out1 = os.path.join(qt_dir, f"{p}_trend_stopgrad_overlay.png")
            out2 = os.path.join(qt_dir, f"{p}_trend_stopgrad_facets.png")
            ok1 = _plot_trend_overlay(agg, p, out1)
            ok2 = _plot_trend_facets(agg, p, out2)
            if ok1: _created_stopgrad_imgs.append(os.path.relpath(out1, viz_dir))
            if ok2: _created_stopgrad_imgs.append(os.path.relpath(out2, viz_dir))
        except Exception as e:
            print(f"[trend] Skipped {p}: {e}")

    # ---------- CORRELATIONS ----------
    heatmap_fp = None
    obj_corr_fp = None
    try:
        # include extra objectives if present
        obj_cols_all = [c for c in ["val_sample_pr_auc","val_fp_per_min",
                                    "val_latency_score","val_recall_at_0p7",
                                    "val_sample_max_f1","val_sample_max_mcc"]
                        if c in df.columns]
        num_cols = [c for c in df.columns
                    if c not in ["trial_number", *obj_cols_all, "COHORT_StopGrad","score_pr","score_fp","combined_avg","sel_epoch"]
                    and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            corr_df = df[num_cols + obj_cols_all].corr(method='spearman')
            plt.figure(figsize=(max(9, 0.6*len(corr_df.columns)), max(6, 0.5*len(corr_df))))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=False)
            plt.title("Spearman correlation: numeric hyperparameters vs objectives")
            plt.tight_layout()
            heatmap_fp = os.path.join(viz_dir, "correlations_spearman.png")
            plt.savefig(heatmap_fp, dpi=130); plt.close()
    except Exception as e:
        print(f"Correlation heatmap warning: {e}"); plt.close()

    try:
        # Objective-to-objective correlations (whatever exists)
        obj_cols = [c for c in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7",
                                "val_sample_max_f1","val_sample_max_mcc"] if c in df.columns]
        if len(obj_cols) >= 2:
            obj_corr = df[obj_cols].corr(method='spearman')
            plt.figure(figsize=(max(3.8, 0.9*len(obj_cols)), max(3.6, 0.8*len(obj_cols))))
            sns.heatmap(obj_corr, annot=True, fmt=".2f", cmap="vlag", cbar=False,
                        xticklabels=obj_cols, yticklabels=obj_cols)
            plt.tight_layout()
            obj_corr_fp = os.path.join(viz_dir, "objective_correlations.png")
            plt.savefig(obj_corr_fp, dpi=140); plt.close()
    except Exception as e:
        print(f"Objective correlation warning: {e}"); plt.close()

    # ---------- SIMPLE COMBINED RANK (top-k) ----------
    def robust_minmax(x):
        q05, q95 = np.nanquantile(x, [0.05, 0.95])
        d = max(1e-9, q95 - q05); z = (x - q05) / d
        return np.clip(z, 0, 1)

    df["score_pr"] = df["val_sample_pr_auc"]                 # higher better
    df["score_fp"] = 1 - robust_minmax(df["val_fp_per_min"]) # lower fp → higher score
    df["combined_avg"] = (df["score_pr"] + df["score_fp"]) / 2.0
    top_combined = df.sort_values("combined_avg", ascending=False).head(25)
    top_combined.to_csv(os.path.join(viz_dir, "top25_combined_avg.csv"), index=False)

    # Top-by-objective HTML tables with links (extended)
    try:
        top_k = 25
        best_pr  = df.sort_values("val_sample_pr_auc", ascending=False).head(top_k).copy()
        best_fp  = df.sort_values("val_fp_per_min", ascending=True ).head(top_k).copy()
        best_lat = df.sort_values("val_latency_score", ascending=False).head(top_k).copy() if "val_latency_score" in df.columns else None
        best_rec = df.sort_values("val_recall_at_0p7", ascending=False).head(top_k).copy() if "val_recall_at_0p7" in df.columns else None

        def _write_top(fname, ddd):
            if ddd is None or ddd.empty: return
            dd = ddd.copy()
            dd["study_dir"] = dd["trial_number"].apply(_trial_link)
            lead = ["trial_number","val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7","combined_avg","study_dir"]
            keep = [c for c in lead if c in dd.columns] + [c for c in dd.columns if c not in lead]
            html_tbl = dd[keep].to_html(escape=False, index=False)
            with open(os.path.join(viz_dir, fname), "w") as fh:
                fh.write(f"<html><head><meta charset='utf-8'><style>body{{font-family:Arial;margin:20px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px}} th{{background:#f5f5f5}}</style></head><body>{html_tbl}</body></html>")

        for fname, ddd in [("top_by_pr_auc.html", best_pr),
                           ("top_by_fpmin.html", best_fp),
                           ("top_by_latency.html", best_lat),
                           ("top_by_recall.html",  best_rec),
                           ("top_by_combined.html", top_combined)]:
            _write_top(fname, ddd)

    except Exception as e:
        print(f"Top-k HTML warning: {e}")

    # =========================
    # EXTRA ANALYSIS ADD-ONS
    # =========================
    extras_dir    = os.path.join(viz_dir, "extras");                os.makedirs(extras_dir, exist_ok=True)
    qplots_dir    = os.path.join(extras_dir, "quantile_trends");    os.makedirs(qplots_dir, exist_ok=True)
    heat2d_dir    = os.path.join(extras_dir, "heatmaps_2d");        os.makedirs(heat2d_dir, exist_ok=True)
    cohort_dir    = os.path.join(extras_dir, "cohorts");            os.makedirs(cohort_dir, exist_ok=True)
    recommend_dir = os.path.join(extras_dir, "range_recommendations"); os.makedirs(recommend_dir, exist_ok=True)
    recon_dir     = os.path.join(extras_dir, "importance_reconciliation"); os.makedirs(recon_dir, exist_ok=True)

    objective_cols = [c for c in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7"]
                      if c in df.columns]
    param_cols = [c for c in df.columns if c not in ["trial_number", *objective_cols,
                                                     "score_pr","score_fp","combined_avg","COHORT_StopGrad","sel_epoch"]]
    num_params_all = [p for p in param_cols if pd.api.types.is_numeric_dtype(df[p])]
    cat_params = [p for p in param_cols if p not in num_params_all]

    # Cohort analysis (StopGrad ON/OFF)
    def _detect_stopgrad_series(dfx: pd.DataFrame) -> pd.Series:
        if "USE_StopGrad" in dfx.columns:
            s = dfx["USE_StopGrad"]
            if s.dtype == bool:
                return s
            return s.map(lambda v: bool(v) if pd.notna(v) else False)
        if "TYPE_ARCH" in dfx.columns:
            return dfx["TYPE_ARCH"].astype(str).str.lower().str.contains("stopgrad", na=False)
        return None

    stopgrad_series = _detect_stopgrad_series(df)

    def _safe_corr_heatmap(data: pd.DataFrame, name: str, out_png: str):
        try:
            cols = [*num_params_all, *objective_cols]
            cols = [c for c in cols if c in data.columns]
            if len(cols) < 3: return
            corr_df = data[cols].corr(method="spearman")
            plt.figure(figsize=(max(8, 0.6*len(corr_df.columns)), max(6, 0.5*len(corr_df))))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title(f"Spearman (numeric) — {name}")
            plt.tight_layout()
            plt.savefig(out_png, dpi=130); plt.close()
        except Exception:
            plt.close()

    def _param_summary_scatter(data: pd.DataFrame, name: str):
        for obj, ylabel in [(c, {"val_sample_pr_auc":"PR-AUC (↑)","val_fp_per_min":"FP/min (↓)",
                                 "val_latency_score":"Latency (↑)","val_recall_at_0p7":"Recall@0.7 (↑)"}[c]) for c in objective_cols]:
            plt.figure(figsize=(max(8, 0.4*len(param_cols)), 4))
            ax = plt.gca()
            x_idx, x_labs = [], []
            for i, p in enumerate(param_cols):
                if p not in data.columns or data[p].nunique(dropna=True) <= 1:
                    continue
                if pd.api.types.is_numeric_dtype(data[p]):
                    valid = data[[p, obj]].dropna()
                    if valid[p].nunique() < 3 or len(valid) < 8:
                        continue
                    qbins = pd.qcut(valid[p], q=min(10, max(3, valid[p].nunique())), duplicates="drop")
                    means = valid.groupby(qbins, observed=True)[obj].mean().values
                    ax.scatter(np.full_like(means, len(x_idx)), means, s=26, alpha=0.7)
                else:
                    g = data.groupby(p, observed=True)[obj].mean().sort_values(ascending=False)
                    ax.scatter(np.full_like(g.values, len(x_idx)), g.values, s=26, alpha=0.7)
                x_idx.append(len(x_idx)); x_labs.append(p)
            ax.set_xticks(range(len(x_labs))); ax.set_xticklabels(x_labs, rotation=28, ha="right")
            ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} vs params — {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(cohort_dir, f"{obj}_vs_params_{name}.png"), dpi=120); plt.close()

    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)
        d0 = df[~sg_mask].copy()  # StopGrad OFF
        d1 = df[ sg_mask].copy()  # StopGrad ON
        if len(d0) >= 30:
            _safe_corr_heatmap(d0, "StopGrad_OFF", os.path.join(cohort_dir, "corr_spearman_StopGrad_OFF.png"))
            _param_summary_scatter(d0, "StopGrad_OFF")
        if len(d1) >= 30:
            _safe_corr_heatmap(d1, "StopGrad_ON",  os.path.join(cohort_dir, "corr_spearman_StopGrad_ON.png"))
            _param_summary_scatter(d1, "StopGrad_ON")

    # Quantile trends (global)
    def quantile_trend(x: pd.Series, y: pd.Series, q=12):
        valid = x.notna() & y.notna()
        xv, yv = x[valid], y[valid]
        if len(xv) < 10 or xv.nunique() < 3:
            return None
        bins = pd.qcut(xv, q=min(q, max(3, xv.nunique())), duplicates="drop")
        grp  = pd.DataFrame({"y": yv, "bin": bins, "x": xv}).groupby("bin", observed=True)
        mu = grp["y"].mean().values
        sd = grp["y"].std().values
        n  = grp["y"].size().values.astype(float)
        se = np.where(n>1, sd/np.sqrt(n), np.nan)
        xc = grp["x"].mean().values
        return xc, mu, se

    num_params_simple = [c for c in df.columns
                         if c not in ["trial_number","val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7",
                                      "val_sample_max_f1","val_sample_max_mcc","COHORT_StopGrad","score_pr","score_fp","combined_avg","sel_epoch"]
                         and pd.api.types.is_numeric_dtype(df[c])
                         and df[c].nunique(dropna=True) >= 3]

    for p in num_params_simple:
        fig, axs = plt.subplots(len(objective_cols), 1, figsize=(8, 3.2*len(objective_cols)), sharex=True)
        if len(objective_cols) == 1: axs = [axs]
        ok_any = False
        for ax, obj in zip(axs, objective_cols):
            label_map = {"val_sample_pr_auc":"PR-AUC (↑)", "val_fp_per_min":"FP/min (↓)",
                         "val_latency_score":"Latency (↑)", "val_recall_at_0p7":"Recall@0.7 (↑)"}
            out = quantile_trend(df[p], df[obj], q=12)
            if out is None:
                ax.set_ylabel(label_map[obj]); continue
            x, mu, se = out
            ax.plot(x, mu, marker="o", linewidth=1.5)
            if np.isfinite(se).any():
                ax.fill_between(x, mu - 1.96*np.nan_to_num(se), mu + 1.96*np.nan_to_num(se), alpha=0.2)
            ax.set_ylabel(label_map[obj]); ok_any = True
        axs[-1].set_xlabel(p)
        if ok_any:
            fig.suptitle(f"Quantile trend — {p}")
            fig.tight_layout(rect=[0,0,1,0.97])
            fig.savefig(os.path.join(qplots_dir, f"{p}_quantile_trends.png"), dpi=130)
        plt.close(fig)

    # 2D interaction heatmaps for informative pairs (example axes)
    def heat2d(x: pd.Series, y: pd.Series, z: pd.Series, xq=12, yq=12):
        valid = x.notna() & y.notna() & z.notna()
        xv, yv, zv = x[valid], y[valid], z[valid]
        if xv.nunique() < 4 or yv.nunique() < 4 or len(zv) < 25:
            return None
        xb = pd.qcut(xv, q=min(xq, max(4, xv.nunique())), duplicates="drop")
        yb = pd.qcut(yv, q=min(yq, max(4, yv.nunique())), duplicates="drop")
        grid = pd.DataFrame({"xb": xb, "yb": yb, "z": zv}).groupby(["xb","yb"], observed=True)["z"].mean().unstack()
        return grid

    def _maybe_heatmap(xname, yname):
        if xname not in df.columns or yname not in df.columns:
            return
        for obj, lab in [("val_sample_pr_auc","PR-AUC (↑)"),
                         ("val_fp_per_min","FP/min (↓)"),
                         ("val_latency_score","Latency (↑)"),
                         ("val_recall_at_0p7","Recall@0.7 (↑)")]:
            if obj not in df.columns: continue
            H = heat2d(df[xname], df[yname], df[obj])
            if H is None:
                continue
            plt.figure(figsize=(6.8,5.2))
            sns.heatmap(H, cmap="viridis", annot=False)
            plt.title(f"{lab} mean — {xname} × {yname}")
            plt.tight_layout()
            plt.savefig(os.path.join(heat2d_dir, f"{obj}_{xname}_x_{yname}.png"), dpi=140)
            plt.close()

    if "LOSS_SupCon" in df.columns and "LOSS_TupMPN" in df.columns:
        _maybe_heatmap("LOSS_SupCon", "LOSS_TupMPN")
    if "learning_rate" in df.columns and "LOSS_NEGATIVES" in df.columns:
        _maybe_heatmap("learning_rate", "LOSS_NEGATIVES")
    if "LOSS_TV" in df.columns and "LOSS_TupMPN" in df.columns:
        _maybe_heatmap("LOSS_TV", "LOSS_TupMPN")

    # Range recommendations: top-quartile trial bands per objective (extended)
    def recommend_ranges(data: pd.DataFrame, name: str, top_frac=0.25):
        metric_list = [c for c in ["val_sample_pr_auc","val_fp_per_min","val_latency_score","val_recall_at_0p7"]
                       if c in data.columns and not data[c].isnull().all()]
        nice_name = {"val_sample_pr_auc":"PR-AUC (↑)",
                     "val_fp_per_min":"FP/min (↓)",
                     "val_latency_score":"Latency (↑)",
                     "val_recall_at_0p7":"Recall@0.7 (↑)"}
        num_params_local = [p for p in data.columns
                            if p not in ["trial_number", *metric_list, "score_pr","score_fp","combined_avg","COHORT_StopGrad","sel_epoch"]
                            and pd.api.types.is_numeric_dtype(data[p])]
        cat_params_local = [p for p in data.columns
                            if p not in ["trial_number", *metric_list, "score_pr","score_fp","combined_avg","COHORT_StopGrad","sel_epoch"]
                            and p not in num_params_local]
        recs = {}
        for obj in metric_list:
            asc = (obj == "val_fp_per_min")  # only FP/min is minimize
            dsort = data.sort_values(obj, ascending=asc)
            top_n = max(20, int(len(dsort)*top_frac))
            top   = dsort.head(top_n)
            rng = {}
            for p in num_params_local:
                col = pd.to_numeric(top[p], errors="coerce").dropna()
                if col.nunique() < 2:
                    continue
                try:
                    rng[p] = (float(np.nanquantile(col, 0.20)), float(np.nanquantile(col, 0.80)))
                except Exception:
                    pass
            cat = {}
            for p in cat_params_local:
                vc = top[p].value_counts(normalize=True, dropna=False)
                if len(vc):
                    cat[p] = vc.head(3).to_dict()
            recs[nice_name[obj]] = {"num_quantile_20_80": rng, "cat_top3_props": cat, "n_top": int(len(top))}
        with open(os.path.join(recommend_dir, f"ranges_{name}.json"), "w") as fh:
            json.dump(recs, fh, indent=2)
        rows = []
        for obj, payload in recs.items():
            for p,(lo,hi) in payload["num_quantile_20_80"].items():
                rows.append({"objective": obj, "param": p, "q20": lo, "q80": hi})
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(recommend_dir, f"ranges_numeric_{name}.csv"), index=False)

    recommend_ranges(df, "ALL")
    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)
        if df[~sg_mask].shape[0] >= 40: recommend_ranges(df[~sg_mask], "StopGrad_OFF")
        if df[ sg_mask].shape[0] >= 40: recommend_ranges(df[ sg_mask], "StopGrad_ON")

    # Importance reconciliation: absolute Spearman vs PR-AUC (global/cohort) — keep as-is for PR-AUC
    imp_rows = []
    for p in num_params_all:
        try:
            r = df[[p,"val_sample_pr_auc"]].dropna().corr(method="spearman").iloc[0,1]
            if pd.notna(r):
                imp_rows.append({"param": p, "source": "Spearman|rho| (global)", "value": abs(float(r))})
        except Exception:
            pass
    if isinstance(stopgrad_series, pd.Series):
        sg_mask = stopgrad_series.fillna(False)
        for name, dsub in [("StopGrad_OFF", df[~sg_mask]), ("StopGrad_ON", df[sg_mask])]:
            if len(dsub) < 30: continue
            for p in num_params_all:
                try:
                    rr = dsub[[p,"val_sample_pr_auc"]].dropna().corr(method="spearman").iloc[0,1]
                    if pd.notna(rr):
                        imp_rows.append({"param": p, "source": f"Spearman|rho| ({name})", "value": abs(float(rr))})
                except Exception:
                    pass
    if imp_rows:
        imp_df = pd.DataFrame(imp_rows)
        piv = imp_df.pivot_table(index="param", columns="source", values="value", aggfunc="max").fillna(0.0)
        piv = piv.sort_values(by=list(piv.columns)[0], ascending=False)
        ax = piv.plot(kind="bar", figsize=(max(9, 0.55*len(piv)),5))
        ax.set_ylabel("|Spearman| vs PR-AUC"); plt.title("Correlation-based importances (compare with Optuna fANOVA)")
        plt.tight_layout()
        plt.savefig(os.path.join(recon_dir, "spearman_vs_pr_auc.png"), dpi=130)
        plt.close()

    # ---------- HTML REPORT ----------
    html = []
    html.append(f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Study Visualization (2-obj): {study.study_name}</title>
    <style>
    body{{font-family:Arial,Helvetica,sans-serif;margin:20px;}} h2{{margin-top:28px}}
    a{{color:#007bff;text-decoration:none}} a:hover{{text-decoration:underline}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
    .card{{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}}
    img{{max-width:100%}}
    .notice{{padding:10px;border:1px solid #ffc107;background:#fff3cd;border-radius:6px}}
    </style></head><body>
    <h1>Visualization — {study.study_name}</h1>
    <p>Objectives: <b>maximize</b> PR-AUC, <b>minimize</b> FP/min.</p>
    <p><b>Trials:</b> {len(df)} | <b>Pareto set:</b> {len(pareto_df)}</p>""")

    html.append(f"""
    <ul>
      <li><a href="all_completed_trials.csv">all_completed_trials.csv</a></li>
      <li><a href="pareto_trials.csv">pareto_trials.csv</a> &nbsp;|&nbsp; <a href="pareto_trials.html">Pareto table (HTML)</a></li>
      <li><a href="top25_combined_avg.csv">top25_combined_avg.csv</a></li>
      <li>Top-K (HTML): <a href="top_by_pr_auc.html">PR-AUC</a> · <a href="top_by_fpmin.html">FP/min</a> · <a href="top_by_latency.html">Latency</a> · <a href="top_by_recall.html">Recall</a> · <a href="top_by_combined.html">Combined</a></li>
      <li>Objective EDFs: <a href="edf_pr_auc.html">PR-AUC</a> · <a href="edf_fp_per_min.html">FP/min</a> · <a href="edf_latency.html">Latency</a> · <a href="edf_recall.html">Recall</a></li>
    </ul>""")

    if pareto_snapshot_html:
        html.append("<h2>Pareto snapshot (top 15 by PR-AUC)</h2>")
        html.append(pareto_snapshot_html)

    html.append("""
    <h2>Standard Optuna Plots</h2>
    <div class="grid">
      <div class="card"><a href="history_pr_auc.html">Optimization History — PR-AUC</a></div>
      <div class="card"><a href="history_fp_per_min.html">Optimization History — FP/min</a></div>
      <div class="card"><a href="param_importances_pr_auc.html">Param Importances — PR-AUC</a></div>
      <div class="card"><a href="param_importances_fp_per_min.html">Param Importances — FP/min</a></div>
      <div class="card"><a href="history_latency.html">Optimization History — Latency</a></div>
      <div class="card"><a href="history_recall.html">Optimization History — Recall@0.7</a></div>
      <div class="card"><a href="param_importances_latency.html">Param Importances — Latency</a></div>
      <div class="card"><a href="param_importances_recall.html">Param Importances — Recall@0.7</a></div>
      <div class="card"><a href="pareto_front_2obj.html">Pareto Front (interactive)</a></div>
      <div class="card"><a href="hypervolume_history.html">Hypervolume History</a></div>
    </div>""")

    html.append("<h2>Objective Projection</h2><div class='grid'>")
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter2d_all.html")):
        html.append('<div class="card"><a href="scatter2d_all.html">PR-AUC vs FP/min — Pareto highlighted</a></div>')
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter2d_prauc_fp_color_latency.html")):
        html.append('<div class="card"><a href="scatter2d_prauc_fp_color_latency.html">PR-AUC vs FP/min — color=Latency</a></div>')
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter2d_prauc_fp_color_recall.html")):
        html.append('<div class="card"><a href="scatter2d_prauc_fp_color_recall.html">PR-AUC vs FP/min — color=Recall@0.7</a></div>')
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter2d_prauc_latency.html")):
        html.append('<div class="card"><a href="scatter2d_prauc_latency.html">PR-AUC vs Latency</a></div>')
    if HAVE_PLOTLY and os.path.exists(os.path.join(viz_dir, "scatter2d_fpmin_latency.html")):
        html.append('<div class="card"><a href="scatter2d_fpmin_latency.html">FP/min vs Latency</a></div>')
    html.append("</div>")

    # StopGrad trend plots (overlay + facets)
    if len(_created_stopgrad_imgs) > 0:
        html.append("<h2>Quantile Trends — StopGrad</h2><div class='grid'>")
        for imgp in _created_stopgrad_imgs:
            html.append(f'<div class="card"><img src="{imgp}"></div>')
        html.append("</div>")
    else:
        html.append("<h2>Quantile Trends — StopGrad</h2><p class='notice'>No StopGrad trend plots were created (insufficient data or parameters too discrete).</p>")

    # Correlations
    if heatmap_fp or obj_corr_fp:
        html.append("<h2>Correlations</h2><div class='grid'>")
        if heatmap_fp and os.path.exists(heatmap_fp):
            html.append(f'<div class="card"><img src="correlations_spearman.png" alt="Hyperparameter correlations"></div>')
        if obj_corr_fp and os.path.exists(obj_corr_fp):
            html.append(f'<div class="card"><img src="objective_correlations.png" alt="Objective correlations"></div>')
        html.append("</div>")

    # Per-Parameter Impact thumbnails
    html.append("<h2>Per-Parameter Impact</h2><div class='grid'>")
    for p in hyperparams:
        imgp = os.path.join("param_impact", f"{p}_impact.png")
        if os.path.exists(os.path.join(viz_dir, imgp)):
            html.append(f'<div class="card"><h3>{p}</h3><img src="{imgp}"></div>')
    html.append("</div>")

    # Extras links
    html.append("""
    <h2>Extras</h2>
    <ul>
      <li><b>Cohorts:</b> see PNGs in <code>extras/cohorts/</code></li>
      <li><b>Quantile Trends (global):</b> <code>extras/quantile_trends/</code></li>
      <li><b>2D Heatmaps:</b> <code>extras/heatmaps_2d/</code></li>
      <li><b>Range Recommendations:</b> JSON/CSV in <code>extras/range_recommendations/</code></li>
      <li><b>Importance Reconciliation:</b> <code>extras/importance_reconciliation/spearman_vs_pr_auc.png</code></li>
    </ul>
    </body></html>""")

    with open(os.path.join(viz_dir, "index.html"), "w") as f:
        f.write("\n".join(html))

    # ---------- SUMMARY ----------
    stats = {
        "study_name": study.study_name,
        "n_completed_trials": int(len(df)),
        "n_pareto_trials": int(len(pareto_df)),
        "has_constraints": bool(has_constraints),
        "n_feasible_trials": int(len(feasible_trials)),
        "objectives": ["val_sample_pr_auc (max)", "val_fp_per_min (min)"],
        "objective_index_map": OBJECTIVE_INDEX
    }
    with open(os.path.join(viz_dir, "study_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Visualization complete → {viz_dir}")

elif mode == 'tune_viz_multi_v9':
    import os, sys, json, math, warnings, glob, re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import optuna

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Increase Plot Resolution
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['savefig.dpi'] = 120

    # Optional: Plotly
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        HAVE_PLOTLY = True
    except ImportError:
        HAVE_PLOTLY = False
        print("WARNING: Plotly not found. Interactive plots will be skipped.")

    # ---------- CONFIG ----------
    tag = args.tag[0]
    param_dir = f'params_{tag}'
    storage_url = f"sqlite:///studies/{param_dir}/{param_dir}.db"
    viz_dir = f"studies/{param_dir}/visualizations_v9_final"
    os.makedirs(viz_dir, exist_ok=True)

    print(f"Loading study '{param_dir}' from {storage_url}")
    try:
        study = optuna.load_study(study_name=param_dir, storage=storage_url)
    except Exception as e:
        print(f"Error loading study: {e}")
        sys.exit(1)

    # ---------- HELPERS ----------
    def _ua(tr, keys, default=np.nan):
        if isinstance(keys, str): keys = [keys]
        for key in keys:
            try:
                if hasattr(tr, "user_attrs") and key in tr.user_attrs:
                    val = tr.user_attrs.get(key)
                    if val is not None: return float(val)
                if hasattr(tr, "system_attrs") and key in tr.system_attrs:
                    val = tr.system_attrs.get(key)
                    if val is not None: return float(val)
            except: continue
        return default

    def _trial_link(trial_no: int) -> str:
        base = os.path.join("studies", param_dir)
        matches = glob.glob(os.path.join(base, f"study_{trial_no}_*"))
        if matches:
            rel = os.path.relpath(matches[0], viz_dir)
            return f'<a href="{rel}" target="_blank">study_{trial_no}</a>'
        return str(trial_no)

    def _clean_filename(s):
        return re.sub(r'[\\/*?:"<>|]', "_", str(s))

    # ---------- COLLECT DATA ----------
    trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not trials:
        print("No completed trials.")
        sys.exit(0)

    nvals = max((len(t.values) if t.values else 0) for t in trials)
    OBJECTIVE_INDEX = dict(pr_auc=0, fp_min=1) if nvals == 2 else dict(pr_auc=0)

    rows = []
    for t in trials:
        if not t.values: continue
        try:
            pr = float(t.values[OBJECTIVE_INDEX['pr_auc']])
            if 'fp_min' in OBJECTIVE_INDEX:
                real_fp = float(t.values[OBJECTIVE_INDEX['fp_min']])
            else:
                real_fp = _ua(t, ["sel_low_conf_fp", "sel_fp_per_min", "val_mean_low_conf_fp"])
            
            if math.isnan(pr) or math.isinf(pr): continue

            rec = {
                "trial_number": t.number,
                "val_sample_pr_auc": pr,      
                "val_fp_per_min":    real_fp,
                "val_latency_score":  _ua(t, ["sel_latency_range", "sel_latency_score", "val_latency_score_range"]), 
                "val_recall_at_0p7":  _ua(t, ["sel_high_conf_rec", "sel_recall_at_0p7", "val_mean_high_conf_recall"]),
                "val_sample_max_f1":  _ua(t, ["sel_max_f1", "val_sample_max_f1"]),
                "val_sample_max_mcc": _ua(t, ["sel_max_mcc", "val_sample_max_mcc"]),
                "sel_epoch":          _ua(t, ["sel_epoch", "selected_epoch", "best_epoch"]), 
            }
            rec.update(t.params)
            rows.append(rec)
        except Exception: continue

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(viz_dir, "all_trials_data.csv"), index=False)

    # --- CLASSIFY COLUMNS ---
    known_metrics = ["val_sample_pr_auc", "val_fp_per_min", "val_latency_score", 
                     "val_recall_at_0p7", "val_sample_max_mcc", "val_sample_max_f1", 
                     "sel_epoch"] 
    
    metric_cols = [c for c in df.columns if c in known_metrics or c.startswith("val_") or c.startswith("sel_") or c.startswith("score_")]
    metric_cols = list(set(metric_cols)) 
    
    exclude_cols = metric_cols + ["trial_number"]
    param_cols = sorted([c for c in df.columns if c not in exclude_cols])
    
    num_params = [p for p in param_cols if pd.api.types.is_numeric_dtype(df[p]) and df[p].nunique() > 1]
    cat_params = [p for p in param_cols if p not in num_params]

    # Pareto
    pareto_trials = study.best_trials
    pareto_nums = [t.number for t in pareto_trials]
    pareto_df = df[df.trial_number.isin(pareto_nums)].copy()
    pareto_df.to_csv(os.path.join(viz_dir, "pareto_trials.csv"), index=False)

    # --- HELPER FOR TABLES ---
    def get_display_cols(target_df):
        cols = ["trial_number", "study_dir"]
        if "sel_epoch" in target_df.columns: cols.append("sel_epoch")
        cols += [x for x in known_metrics if x in target_df.columns and x not in cols]
        cols += param_cols
        return [c for c in cols if c in target_df.columns]

    # Generate All Trials HTML
    all_trials_view = df.copy()
    all_trials_view["study_dir"] = all_trials_view["trial_number"].apply(_trial_link)
    all_cols = get_display_cols(all_trials_view)
    with open(os.path.join(viz_dir, "all_trials.html"), "w") as f:
        f.write("<html><head><link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'></head><body class='p-4'><h3>All Completed Trials</h3>")
        f.write(all_trials_view[all_cols].to_html(escape=False, index=False, classes='table table-sm table-hover table-striped'))
        f.write("</body></html>")

    # =========================================================
    # 1. OPTUNA STANDARD VISUALIZATIONS
    # =========================================================
    print("Generating Standard Optuna Plots...")
    
    # History
    try:
        fig_hist_pr = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[OBJECTIVE_INDEX['pr_auc']] if t.values else float('nan'), target_name="PR-AUC")
        fig_hist_pr.write_html(os.path.join(viz_dir, "history_pr_auc.html"))
    except: pass
    try:
        fig_hist_fp = optuna.visualization.plot_optimization_history(
            study, target=lambda t: _ua(t, ["sel_low_conf_fp", "sel_fp_per_min"]), target_name="FP/min")
        fig_hist_fp.write_html(os.path.join(viz_dir, "history_fp_per_min.html"))
    except: pass

    # Pareto Front
    names_by_index = [""] * nvals
    names_by_index[OBJECTIVE_INDEX['pr_auc']] = "PR-AUC"
    if 'fp_min' in OBJECTIVE_INDEX: names_by_index[OBJECTIVE_INDEX['fp_min']] = "FP/min"
    try:
        fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=names_by_index)
        fig_pareto.write_html(os.path.join(viz_dir, "pareto_front_optuna.html"))
    except: pass

    # Importance
    imp_dir = os.path.join(viz_dir, "importance_analysis")
    os.makedirs(imp_dir, exist_ok=True)
    valid_imp_metrics = [m for m in known_metrics if m in df.columns and df[m].nunique() > 1 and m != "sel_epoch"]
    for metric in valid_imp_metrics:
        try:
            t_map = df.set_index("trial_number")[metric].to_dict()
            fig_imp = optuna.visualization.plot_param_importances(study, target=lambda t: t_map.get(t.number, float('nan')), target_name=metric)
            fig_imp.write_html(os.path.join(imp_dir, f"importance_{metric}.html"))
        except: pass

    # =========================================================
    # 2. INTERACTIVE PLOTS
    # =========================================================
    if HAVE_PLOTLY:
        print("Generating Interactive Plots...")
        
        # 3D Plot
        z_axis = "val_latency_score" if "val_latency_score" in df.columns else "val_sample_max_mcc"
        c_col = "val_recall_at_0p7" if "val_recall_at_0p7" in df.columns else "val_sample_pr_auc"
        if z_axis in df.columns:
            try:
                fig_3d = px.scatter_3d(
                    df, x="val_fp_per_min", y="val_sample_pr_auc", z=z_axis,
                    color=c_col, color_continuous_scale="Plasma",
                    hover_name="trial_number", hover_data=param_cols[:6],
                    title=f"3D: FP vs PR vs {z_axis} (Color={c_col})"
                )
                fig_3d.update_layout(scene=dict(xaxis_title='FP/min', yaxis_title='PR-AUC', zaxis_title=z_axis))
                fig_3d.write_html(os.path.join(viz_dir, "3d_tradeoff.html"))
            except: pass

        # Main 2D Scatter
        fig_main = px.scatter(
            df, x="val_fp_per_min", y="val_sample_pr_auc",
            color="val_sample_max_mcc" if "val_sample_max_mcc" in df.columns else None,
            size="val_latency_score" if "val_latency_score" in df.columns else None,
            color_continuous_scale="Turbo",
            hover_name="trial_number", hover_data=param_cols[:6] + known_metrics,
            title="<b>Main Trade-off</b>: FP vs PR (Color=MCC, Size=Latency)"
        )
        if not pareto_df.empty:
            fig_main.add_trace(go.Scatter(x=pareto_df["val_fp_per_min"], y=pareto_df["val_sample_pr_auc"],
                mode='markers', marker=dict(symbol='star', size=12, color='black', line=dict(width=1, color='white')),
                name='Pareto'))
        fig_main.write_html(os.path.join(viz_dir, "main_interactive_scatter.html"))

        # X-Rays
        xray_dir = os.path.join(viz_dir, "xray_plots")
        os.makedirs(xray_dir, exist_ok=True)
        for p in param_cols:
            if df[p].nunique() <= 1: continue
            try:
                fig_x = px.scatter(df, x="val_fp_per_min", y="val_sample_pr_auc", color=p,
                    color_continuous_scale="Spectral_r" if pd.api.types.is_numeric_dtype(df[p]) else None,
                    title=f"Trade-off colored by: {p}", hover_name="trial_number")
                fig_x.write_html(os.path.join(xray_dir, f"xray_{_clean_filename(p)}.html"))
            except: pass
            
        # Parallel Coords
        try:
            tunable = list(study.best_trial.params.keys())
            valid_p = [p for p in tunable if p in df.columns]
            if valid_p:
                corr_pr = df[valid_p].corrwith(df["val_sample_pr_auc"], method="spearman").abs()
                top_p = corr_pr.sort_values(ascending=False).head(8).index.tolist()
                t_map_pr = df.set_index("trial_number")["val_sample_pr_auc"].to_dict()
                fig_par = optuna.visualization.plot_parallel_coordinate(
                    study, params=top_p, target=lambda t: t_map_pr.get(t.number, np.nan), target_name="PR-AUC")
                fig_par.write_html(os.path.join(viz_dir, "parallel_coords_pr_auc.html"))
        except: pass

    # =========================================================
    # 3. STATIC DIAGNOSTICS
    # =========================================================
    print("Generating Static Diagnostics...")
    diag_dir = os.path.join(viz_dir, "diagnostic_plots")
    os.makedirs(diag_dir, exist_ok=True)

    top_params = []
    if num_params:
        corr_pr = df[num_params].corrwith(df["val_sample_pr_auc"], method="spearman").abs()
        top_params = corr_pr.sort_values(ascending=False).head(5).index.tolist()
    if cat_params:
        top_params.extend(cat_params[:2])

    # A. Box Plots
    box_dir = os.path.join(viz_dir, "box_plots")
    os.makedirs(box_dir, exist_ok=True)
    for p in param_cols:
        if df[p].nunique() <= 1: continue
        try:
            plt.figure(figsize=(8, 5))
            plot_df = df.copy()
            if pd.api.types.is_numeric_dtype(df[p]) and df[p].nunique() > 8:
                try: plot_df[p] = pd.qcut(plot_df[p], q=4, duplicates='drop')
                except: pass 
            sns.boxplot(data=plot_df, x=p, y="val_sample_pr_auc", palette="Blues")
            plt.title(f"Impact of {p} on PR-AUC")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(box_dir, f"boxplot_{_clean_filename(p)}.png"))
            plt.close()
        except: plt.close()

    # B. Pairplot
    try:
        pair_cols = top_params[:6] + ["val_sample_pr_auc", "val_fp_per_min", "val_latency_score"]
        pair_cols = [c for c in pair_cols if c in df.columns]
        pp = sns.pairplot(df[pair_cols], diag_kind="kde", corner=True, 
                          plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': 'none'})
        pp.fig.suptitle("Pairwise Interactions", y=1.02)
        pp.savefig(os.path.join(diag_dir, "smart_pairplot.png"), dpi=100)
        plt.close()
    except: pass

    # C. Heatmap
    if num_params and known_metrics:
        try:
            valid_m = [m for m in known_metrics if m in df.columns and m != "sel_epoch"] 
            full_corr = df[num_params + valid_m].corr(method='spearman')
            sliced_corr = full_corr.loc[num_params, valid_m]
            plt.figure(figsize=(max(8, len(valid_m)*1.2), max(8, len(num_params)*0.4)))
            sns.heatmap(sliced_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, cbar_kws={"shrink": 0.5})
            plt.title("Correlation: Params vs Metrics")
            plt.tight_layout()
            plt.savefig(os.path.join(diag_dir, "global_correlation_heatmap.png"))
            plt.close()
        except: pass

    # D. Grid
    if top_params and known_metrics:
        grid_m = [m for m in known_metrics if m in df.columns and m != "sel_epoch"][:5]
        n_rows, n_cols = len(grid_m), len(top_params)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 2.5*n_rows), constrained_layout=True)
        if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
        elif n_rows == 1: axes = np.array([axes])
        elif n_cols == 1: axes = np.array([[ax] for ax in axes])
        for r, metric in enumerate(grid_m):
            for c, param in enumerate(top_params):
                ax = axes[r][c]
                if pd.api.types.is_numeric_dtype(df[param]):
                    sns.scatterplot(data=df, x=param, y=metric, ax=ax, alpha=0.5, s=15, color='#2b7bba')
                    try: sns.regplot(data=df, x=param, y=metric, ax=ax, scatter=False, color='red', line_kws={'alpha':0.5})
                    except: pass
                else:
                    sns.boxplot(data=df, x=param, y=metric, ax=ax, palette="light:b")
                if r == 0: ax.set_title(param, fontsize=9)
                if c == 0: ax.set_ylabel(metric, fontsize=8)
                else: ax.set_ylabel("")
                ax.set_xlabel("")
                ax.tick_params(labelsize=8)
        plt.savefig(os.path.join(diag_dir, "top_params_impact_grid.png"))
        plt.close()

    # =========================================================
    # 4. TABLES & REPORT
    # =========================================================
    table_htmls = {}
    sort_metrics = [("val_sample_pr_auc", False), ("val_fp_per_min", True)]
    if "val_latency_score" in df.columns: sort_metrics.append(("val_latency_score", False))
    if "val_sample_max_mcc" in df.columns: sort_metrics.append(("val_sample_max_mcc", False)) 
    if "val_recall_at_0p7" in df.columns: sort_metrics.append(("val_recall_at_0p7", False)) 

    # Generate Top-K Tables
    for m, ascending in sort_metrics:
        if m not in df.columns: continue
        top_df = df.sort_values(m, ascending=ascending).head(20).copy()
        top_df["study_dir"] = top_df["trial_number"].apply(_trial_link)
        d_cols = get_display_cols(top_df)
        table_htmls[m] = top_df[d_cols].to_html(escape=False, index=False, classes='table table-sm table-hover')

    # Generate Standalone Pareto HTML
    pareto_html_content = ""
    if not pareto_df.empty:
        p_view = pareto_df.copy()
        p_view["study_dir"] = p_view["trial_number"].apply(_trial_link)
        p_cols = get_display_cols(p_view)
        pareto_html_content = p_view[p_cols].to_html(escape=False, index=False, classes='table table-sm table-striped table-bordered')
        with open(os.path.join(viz_dir, "pareto_trials.html"), "w") as f:
            f.write(f"<html><head><link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'></head><body class='p-4'><h3>Pareto Trials</h3>{pareto_html_content}</body></html>")

    # HTML Assembly
    html = []
    html.append(f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Viz v9 (Final): {tag}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body{{background:#f8f9fa; font-family:'Segoe UI', Roboto, sans-serif;}}
        .card{{margin-bottom:20px; border:none; box-shadow:0 2px 5px rgba(0,0,0,0.05);}}
        .card-header{{background-color:#fff; border-bottom:1px solid #eee; font-weight:600; color:#495057;}}
        .nav-tabs .nav-link.active {{font-weight:bold; border-bottom:3px solid #0d6efd; color:#0d6efd;}}
        iframe{{width:100%; height:500px; border:none;}}
        img{{max-width:100%; height:auto;}}
    </style></head><body class="container-fluid py-4" style="max-width:1600px;">
    
    <div class="d-flex justify-content-between align-items-center mb-4 border-bottom pb-3">
        <div>
            <h1 class="h3 text-dark mb-0">Analysis: {tag}</h1>
            <small class="text-muted">Trials: {len(df)} | Pareto: {len(pareto_df)}</small>
        </div>
        <div>
            <a href="all_trials_data.csv" class="btn btn-sm btn-outline-secondary">Download CSV</a>
            <a href="all_trials.html" class="btn btn-sm btn-outline-primary">View All Trials HTML</a>
            <a href="pareto_trials.csv" class="btn btn-sm btn-success">Pareto CSV</a>
        </div>
    </div>

    <ul class="nav nav-tabs mb-4" id="myTab" role="tablist">
        <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-interactive">Interactive</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-diagnostics">Diagnostics</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-xray">X-Rays & History</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-boxplots">Box Plots</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-tables">Top Models</button></li>
    </ul>

    <div class="tab-content" id="myTabContent">
        
        <div class="tab-pane fade show active" id="tab-interactive">
            <div class="row">
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">Main Trade-off (2D)</div>
                        <div class="card-body p-0"><iframe src="main_interactive_scatter.html"></iframe></div>
                    </div>
                </div>
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">3D Exploration</div>
                        <div class="card-body p-0"><iframe src="3d_tradeoff.html"></iframe></div>
                    </div>
                </div>
            </div>
             <div class="row">
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">Importance (PR-AUC)</div>
                        <div class="card-body p-0"><iframe src="importance_analysis/importance_val_sample_pr_auc.html"></iframe></div>
                    </div>
                </div>
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">Parallel Coordinates</div>
                        <div class="card-body p-0"><iframe src="parallel_coords_pr_auc.html"></iframe></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="tab-pane fade" id="tab-diagnostics">
            <div class="row">
                <div class="col-lg-8">
                    <div class="card">
                        <div class="card-header">Global Correlation Heatmap</div>
                        <div class="card-body text-center"><img src="diagnostic_plots/global_correlation_heatmap.png"></div>
                    </div>
                </div>
                <div class="col-lg-4">
                    <div class="card">
                        <div class="card-header">Smart Scatter Matrix</div>
                        <div class="card-body text-center"><img src="diagnostic_plots/smart_pairplot.png"></div>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">Top Parameters vs Metrics Grid</div>
                <div class="card-body text-center"><img src="diagnostic_plots/top_params_impact_grid.png"></div>
            </div>
        </div>

        <div class="tab-pane fade" id="tab-xray">
            <h5 class="mt-2">Optimization History</h5>
            <div class="row">
                <div class="col-md-6"><div class="card"><div class="card-header">PR-AUC History</div><div class="card-body p-0"><iframe src="history_pr_auc.html" height="400"></iframe></div></div></div>
                <div class="col-md-6"><div class="card"><div class="card-header">FP/min History</div><div class="card-body p-0"><iframe src="history_fp_per_min.html" height="400"></iframe></div></div></div>
            </div>
            <h5 class="mt-4">Hyperparameter X-Rays (Colored Trade-offs)</h5>
            <div class="row">
    """)
    xray_files = sorted(glob.glob(os.path.join(xray_dir, "xray_*.html")))
    for xf in xray_files:
        rel = os.path.relpath(xf, viz_dir)
        name = os.path.basename(xf).replace("xray_", "").replace(".html", "")
        html.append(f'<div class="col-md-4"><div class="card"><div class="card-header">{name}</div><div class="card-body p-0"><iframe src="{rel}" height="350"></iframe></div></div></div>')
    html.append("""</div></div>""")

    # 4. BOX PLOTS TAB
    html.append("""<div class="tab-pane fade" id="tab-boxplots">
            <p class="text-muted ms-2">Visualizing impact of individual parameters on PR-AUC.</p>
            <div class="row">""")
    box_imgs = sorted(glob.glob(os.path.join(box_dir, "*.png")))
    for img_p in box_imgs:
        rel = os.path.relpath(img_p, viz_dir)
        p_name = os.path.basename(img_p).replace("boxplot_", "").replace(".png", "")
        html.append(f'<div class="col-md-4"><div class="card"><div class="card-header">{p_name}</div><div class="card-body p-1"><img src="{rel}"></div></div></div>')
    html.append("""</div></div>""")

    # 5. TABLES TAB
    html.append("""<div class="tab-pane fade" id="tab-tables">
            <div class="accordion" id="accTables">""")
    
    # Pareto Table
    if pareto_html_content:
        html.append(f"""<div class="accordion-item"><h2 class="accordion-header"><button class="accordion-button" data-bs-toggle="collapse" data-bs-target="#cPareto">Pareto Set ({len(pareto_df)})</button></h2><div id="cPareto" class="accordion-collapse collapse show" data-bs-parent="#accTables"><div class="accordion-body" style="overflow-x:auto;">{pareto_html_content}</div></div></div>""")

    for i, (m, tbl) in enumerate(table_htmls.items()):
        html.append(f"""<div class="accordion-item"><h2 class="accordion-header"><button class="accordion-button collapsed" data-bs-toggle="collapse" data-bs-target="#c{i}">Top 20 by {m}</button></h2><div id="c{i}" class="accordion-collapse collapse" data-bs-parent="#accTables"><div class="accordion-body" style="overflow-x:auto;">{tbl}</div></div></div>""")
        
    html.append("""</div></div></div><script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script></body></html>""")

    with open(os.path.join(viz_dir, "index.html"), "w") as f:
        f.write("\n".join(html))

    print(f"Viz v9 Final Complete → {viz_dir}")
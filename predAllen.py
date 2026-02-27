import os, pdb, time, copy, shutil, argparse, sys, glob, importlib, random
import joblib, json, logging, gc, ctypes, h5py, datetime
from os import path
from shutil import copyfile
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.utils import timeseries_dataset_from_array

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Fixed Parallel Inference')
parser.add_argument('--model', type=str, nargs=1, help='model name', default=['testSWR'])
parser.add_argument('--mode', type=str, nargs=1, help='mode predictGPU', default=['predictGPU'])
parser.add_argument('--val', type=str, nargs=1, help='val_id', default=['0'])
parser.add_argument('--tag', type=str, nargs=1, help='tag', default=['base'])
parser.add_argument('--start_ind', type=int, help='Channel window index', required=True)

args = parser.parse_args()
model_name = args.model[0]
val_id = int(args.val[0])
tag = args.tag[0]
start_ind = args.start_ind

# --- Config ---
LFP_ROOT = '/mnt/hpc/projects/OWVinckSWR/Cem/VisualBehavior/upsampled_ca1_cortical_2500Hz/'
SAVE_ROOT = f'/mnt/hpc/projects/OWVinckSWR/Cem/VisualBehavior/predictions_ca1_cortical_2500Hz/{model_name}/temp_windows/'
os.makedirs(SAVE_ROOT, exist_ok=True)

# --- 1. Parameters & Study Setup ---
study_num = model_name.split('_')[1]
param_dir = f"params_{tag}"
base_dir = f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/'
study_dirs = glob.glob(f'{base_dir}study_{study_num}_*')
if not study_dirs: raise ValueError(f"No study directory found for {study_num}")
study_dir = study_dirs[0]

params = {}
with open(f"{study_dir}/trial_info.json", 'r') as f:
    trial_info = json.load(f)
    params.update(trial_info['parameters'])
params['mode'] = 'predict'

# --- 2. Correct Model Function Selection (Restored Logic) ---
import importlib.util
if 'MixerOnly' in params['TYPE_ARCH']:
    spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    build_DBI_TCN = model_module.build_DBI_TCN_MixerOnly
elif 'TripletOnly' in params['TYPE_ARCH']:
    # Reference logic for TripletOnly selection
    if int(study_num) < 850:
        spec = importlib.util.spec_from_file_location("model_fn", f"{base_dir}/base_model_tr859/model_fn.py")
    else:
        spec = importlib.util.spec_from_file_location("model_fn", f"{base_dir}/base_model/model_fn.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    build_DBI_TCN = model_module.build_DBI_TCN_TripletOnly
else:
    spec = importlib.util.spec_from_file_location("model_fn", f"{base_dir}/base_model/model_fn.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    build_DBI_TCN = model_module.build_DBI_TCN

# --- 3. Weight Selection ---
event_weights = f"{study_dir}/event.weights.h5"
max_weights = f"{study_dir}/max.weights.h5"
mcc_weights = f"{study_dir}/mcc.weights.h5"

if os.path.exists(mcc_weights):
    weight_file = mcc_weights
elif os.path.exists(event_weights) and os.path.exists(max_weights):
    weight_file = event_weights if os.path.getmtime(event_weights) > os.path.getmtime(max_weights) else max_weights
else:
    weight_file = event_weights if os.path.exists(event_weights) else max_weights

params['WEIGHT_FILE'] = weight_file

# --- 4. Initialize Model with Original NO_TIMEPOINTS ---
# We MUST use the original 44 points here because of the internal tf.split logic
sample_length = params['NO_TIMEPOINTS'] 
model = build_DBI_TCN(sample_length, params=params)
# model.load_weights(weight_file) is handled inside the reference build_DBI_TCN if WEIGHT_FILE is in params

# --- 5. Data Processing (Sliding Window) ---
ses_list = sorted(glob.glob(os.path.join(LFP_ROOT, 'ses_*')))
lfp_raw = xr.open_dataarray(ses_list[val_id]).astype(np.float32).values
lfp_flipped = np.flip(lfp_raw, axis=1)

# Extract the 8-channel window
chan_slice = lfp_flipped[:, start_ind : start_ind + 8]

# Create the sliding window dataset (44 points, stride 1)
# BATCH_SIZE must be a multiple of 3 for the Triplet architecture!
predict_batch_size = 9000
train_x = timeseries_dataset_from_array(
    chan_slice, 
    None, 
    sequence_length=sample_length, 
    sequence_stride=1, 
    batch_size=predict_batch_size
)

# --- 6. Prediction ---
print(f"Predicting Window {start_ind}...")
windowed_signal = np.squeeze(model.predict(train_x, verbose=1))

# Padding the start to maintain n_timepoints length
probs = np.hstack((np.zeros(sample_length - 1), windowed_signal))

# --- 7. Save ---
session_name = os.path.basename(ses_list[val_id])
save_path = os.path.join(SAVE_ROOT, f'{session_name}_win{start_ind:02d}.npy')
np.save(save_path, probs)
print(f"Saved: {save_path}")
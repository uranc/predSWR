import os, argparse, sys, glob, json, re, time
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.utils import timeseries_dataset_from_array

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Fixed Parallel Inference')
parser.add_argument('--model', type=str, help='model name (e.g., 14500)', required=True)
parser.add_argument('--val', type=int, help='val_id index', required=True)
parser.add_argument('--tag', type=str, help='tag', required=True)
parser.add_argument('--start_ind', type=int, help='Channel window index', required=True)

args = parser.parse_args()
model_name = args.model
val_id = args.val
tag = args.tag
start_ind = args.start_ind

# --- Config ---
LFP_ROOT = '/mnt/hpc/projects/OWVinckSWR/Cem/VisualBehavior/upsampled_ca1_cortical_2500Hz/'
SAVE_ROOT = f'/mnt/hpc/projects/OWVinckSWR/Cem/VisualBehavior/predictions_ca1_cortical_2500Hz/{model_name}/temp_windows/'
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# --- 1. Parameters & Study Setup ---
# Robustly extract number: handles "14500" or "Tune_14500_"
study_num_match = re.search(r'(\d+)', model_name)
if not study_num_match:
    raise ValueError(f"Could not extract a numeric study ID from model name: {model_name}")
study_num = study_num_match.group(1)

param_dir = f"params_{tag}"
base_dir = f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/'

# Find the study folder
study_dirs = glob.glob(f'{base_dir}study_{study_num}_*')
if not study_dirs:
    study_dirs = glob.glob(f'{base_dir}study_{study_num}*')
    if not study_dirs:
        raise ValueError(f"No study directory found for {study_num} in {base_dir}")

study_dir = study_dirs[0]

with open(f"{study_dir}/trial_info.json", 'r') as f:
    trial_info = json.load(f)
    params = trial_info['parameters']
params['mode'] = 'predict'

# --- 2. Model Function Selection ---
import importlib.util
if 'MixerOnly' in params.get('TYPE_ARCH', ''):
    spec = importlib.util.spec_from_file_location("model_fn", f"{study_dir}/model/model_fn.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    build_DBI_TCN = model_module.build_DBI_TCN_MixerOnly
elif 'TripletOnly' in params.get('TYPE_ARCH', ''):
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
mcc_w, evt_w, max_w = f"{study_dir}/mcc.weights.h5", f"{study_dir}/event.weights.h5", f"{study_dir}/max.weights.h5"
weight_file = mcc_w if os.path.exists(mcc_w) else (evt_w if os.path.exists(evt_w) else max_w)
params['WEIGHT_FILE'] = weight_file

# --- 4. Initialize Model ---
sample_length = params['NO_TIMEPOINTS'] 
model = build_DBI_TCN(sample_length, params=params)

# --- 5. Data Processing & Timing ---
# Start overall timer
start_total = time.time()

ses_list = sorted(glob.glob(os.path.join(LFP_ROOT, 'ses_*.nc')))
with xr.open_dataarray(ses_list[val_id]) as ds:
    lfp_raw = ds.astype(np.float32).values
    chan_ids = ds.channel.values # Defined before closing file

# Window Anatomy Check
top_id = chan_ids[start_ind]
bot_id = chan_ids[start_ind + 7]
print(f"Processing Window {start_ind}: Top ID {top_id} (Superficial) -> Bottom ID {bot_id} (Deep)")

# Slice and Batching (9000 for TripletOnly compatibility)
chan_slice = lfp_raw[:, start_ind : start_ind + 8]
train_x = timeseries_dataset_from_array(
    chan_slice, None, sequence_length=sample_length, 
    sequence_stride=1, batch_size=900, shuffle=False
)

# --- 6. Prediction & Timing ---
print(f"Starting Prediction...")
start_pred = time.time()

windowed_signal = np.squeeze(model.predict(train_x, verbose=1))

end_pred = time.time()
pred_duration = end_pred - start_pred

# --- 7. Formatting & Saving ---
probs = np.hstack((np.zeros(sample_length - 1), windowed_signal))
save_path = os.path.join(SAVE_ROOT, f'{os.path.basename(ses_list[val_id])}_win{start_ind:02d}.npy')
np.save(save_path, probs)

end_total = time.time()
total_duration = end_total - start_total

# --- Final Output Report ---
print(f"\n{'='*40}")
print(f"SESSION: {os.path.basename(ses_list[val_id])}")
print(f"WINDOW START INDEX: {start_ind}")
print(f"PREDICTION TIME: {pred_duration:.2f} seconds")
print(f"TOTAL ELAPSED:   {total_duration:.2f} seconds")
print(f"{'='*40}\n")
import os, argparse, sys, glob, json, re, time
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.utils import timeseries_dataset_from_array

# ----------------------------- Utilities ------------------------------------
def sliding_window_zscore(x32, win=1250, eps=1e-8):
    """
    Per-channel running Z-score using cumulative sums.
    x32: (time, channels)
    win: window size in samples
    """
    x64 = x32.astype(np.float64)
    n, c = x32.shape
    cs   = np.cumsum(np.pad(x64, ((1,0),(0,0))), axis=0, dtype=np.float64)
    cs2  = np.cumsum(np.pad(x64**2, ((1,0),(0,0))), axis=0, dtype=np.float64)
    idx0 = np.clip(np.arange(n) - win + 1, 0, None)
    L    = (np.arange(n) - idx0 + 1).astype(np.float64)
    w_sum  = cs [1:] - cs [idx0]
    w_sum2 = cs2[1:] - cs2[idx0]
    mu  = w_sum / L[:, None]
    var = w_sum2 / L[:, None] - mu**2
    sig = np.sqrt(np.maximum(var, 0.0))
    return ((x64 - mu) / (sig + eps)).astype(np.float32)

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
study_num_match = re.search(r'(\d+)', model_name)
if not study_num_match:
    raise ValueError(f"Could not extract a numeric study ID from model name: {model_name}")
study_num = study_num_match.group(1)

param_dir = f"params_{tag}"
base_dir = f'/mnt/hpc/projects/MWNaturalPredict/DL/predSWR/studies/{param_dir}/'
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
    m_path = f"{base_dir}/base_model_tr859/model_fn.py" if int(study_num) < 850 else f"{base_dir}/base_model/model_fn.py"
    spec = importlib.util.spec_from_file_location("model_fn", m_path)
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
# Note: build_DBI_TCN usually loads weights if params['WEIGHT_FILE'] is set.

# --- 5. Data Processing & Normalization ---
start_total = time.time()
ses_list = sorted(glob.glob(os.path.join(LFP_ROOT, 'ses_*.nc')))

with xr.open_dataarray(ses_list[val_id]) as ds:
    # Optimized: Loading ONLY the 8 channels required
    lfp_8ch_raw = ds.isel(channel=slice(start_ind, start_ind + 8)).astype(np.float32).values
    chan_ids_all = ds.channel.values

# Anatomy Check
top_id = chan_ids_all[start_ind]
bot_id = chan_ids_all[start_ind + 7]
print(f"Processing Window {start_ind}: Top ID {top_id} (Superficial) -> Bottom ID {bot_id} (Deep)")

# --- MANDATORY Z-SCORE ---
# This normalizes the 8 channels so the model sees the correct signal scale.
chan_slice = sliding_window_zscore(lfp_8ch_raw, win=1250)

# Batch size 9000 for efficiency; shuffle must be False for time-series prediction
train_x = timeseries_dataset_from_array(
    chan_slice, None, sequence_length=sample_length, 
    sequence_stride=1, batch_size=9000, shuffle=False
)

# --- 6. Prediction & Timing ---
print(f"Starting Prediction...")
start_pred = time.time()
windowed_signal = np.squeeze(model.predict(train_x, verbose=1))
pred_duration = time.time() - start_pred

# --- 7. Formatting & Saving ---
# Pad start to maintain alignment with LFP timepoints
probs = np.hstack((np.zeros(sample_length - 1), windowed_signal))
save_path = os.path.join(SAVE_ROOT, f'{os.path.basename(ses_list[val_id])}_win{start_ind:02d}.npy')
np.save(save_path, probs)

total_duration = time.time() - start_total

print(f"\n{'='*40}")
print(f"SESSION: {os.path.basename(ses_list[val_id])}")
print(f"WINDOW START: {start_ind} | CH_RANGE: {top_id}-{bot_id}")
print(f"PREDICTION TIME: {pred_duration:.2f}s")
print(f"TOTAL ELAPSED:   {total_duration:.2f}s")
print(f"{'='*40}\n")
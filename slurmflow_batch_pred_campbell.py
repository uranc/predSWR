import os, glob, re, subprocess
from pathlib import Path
import numpy as np
import xarray as xr

# --- Configuration ---
EVENT_BASE = Path("/cs/projects/OWVinckSWR/Cem/CampbellMurphy2025_SWRs_data/allen_visbehave_swr_murphylab2024")
LFP_ROOT = Path("/mnt/hpc/projects/OWVinckSWR/Cem/VisualBehavior/upsampled_ca1_cortical_2500Hz/")
tag = 'tripletOnlyProxy2500' 
dry_run = False 

model_lib = [2402, 2403, 2404] 
ses_list = sorted(list(LFP_ROOT.glob("ses_*.nc")))

for iv, lfp_path in enumerate(ses_list):
    lfp_name = lfp_path.name
    s_match = re.search(r"ses_(\d+)", lfp_name)
    p_match = re.search(r"probe_(\d+)", lfp_name)
    if not s_match or not p_match: continue
    
    session_id, probe_id = s_match.group(1), p_match.group(1)
    session_event_dir = list(EVENT_BASE.glob(f"swrs_session_{session_id}"))
    if not session_event_dir: continue

    swr_csvs = list(session_event_dir[0].glob(f"*probe_{probe_id}_channel_*_putative_swr_events.csv.gz"))
    move_csvs = list(session_event_dir[0].glob(f"*probe_{probe_id}_*movement*.csv.gz"))
    if not (swr_csvs and move_csvs): continue
    
    swr_target_id = int(re.search(r"channel_(\d+)", swr_csvs[0].name).group(1))
    move_target_id = int(re.search(r"channel_(\d+)", move_csvs[0].name).group(1))

    try:
        with xr.open_dataarray(lfp_path) as ds:
            # CHECK ORIENTATION: We want Superficial (Higher ID) at Index 0
            raw_chans = ds.channel.values
            if raw_chans[0] < raw_chans[-1]:
                # It's Deep -> Superficial, we need to flip it
                channels = np.flip(raw_chans)
                needs_flip_in_pred = True
            else:
                # It's already Superficial -> Deep
                channels = raw_chans
                needs_flip_in_pred = False
            
            # Find indices in the "Superficial First" array
            swr_idx = np.where(channels == swr_target_id)[0][0]
            move_idx = np.where(channels == move_target_id)[0][0]
            
            # Anchor at index-3 to put target at window-offset 3
            swr_start_ind = max(0, min(swr_idx - 3, len(channels) - 8))
            move_start_ind = max(0, min(move_idx - 3, len(channels) - 8))

            for model_num in model_lib:
                model_name = f'Tune_{model_num}_'
                print(f"Ses {session_id} | SWR Start: {swr_start_ind} | Move Start: {move_start_ind}")
                
                if not dry_run:
                    # SWR Job
                    subprocess.call(['python', 'predAllen.py', '--model', model_name, '--val', str(iv), 
                                     '--start_ind', str(swr_start_ind), '--tag', tag])
                    # Movement Job
                    subprocess.call(['python', 'predAllen.py', '--model', model_name, '--val', str(iv), 
                                     '--start_ind', str(move_start_ind), '--tag', tag])
                    
    except Exception as e:
        print(f"Error {session_id}: {e}")
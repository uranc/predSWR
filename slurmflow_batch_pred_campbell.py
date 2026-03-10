import os, glob, re, subprocess, pdb, time
from pathlib import Path
import numpy as np
import xarray as xr

# --- Configuration ---
EVENT_BASE = Path("/cs/projects/OWVinckSWR/Cem/CampbellMurphy2025_SWRs_data/allen_visbehave_swr_murphylab2024")
LFP_ROOT = Path("/mnt/hpc/projects/OWVinckSWR/Cem/VisualBehavior/upsampled_ca1_cortical_2500Hz/")
tag = 'tripletOnlyProxy2500' 
# tag = 'tripletOnlyClean2500' 
dry_run = False 

model_lib = [2671,2672]#[2250]#2250]#1450 2250, 2346]#14500, 
ses_list = sorted(list(LFP_ROOT.glob("ses_*.nc")))

print(f"Scanning {len(ses_list)} upsampled LFP files...")

for iv, lfp_path in enumerate(ses_list):
    
    lfp_name = lfp_path.name
    s_match = re.search(r"ses_(\d+)", lfp_name)
    p_match = re.search(r"probe_(\d+)", lfp_name)
    if not s_match or not p_match: continue
    
    session_id, probe_id = s_match.group(1), p_match.group(1)
    session_event_dirs = list(EVENT_BASE.glob(f"swrs_session_{session_id}"))
    if not session_event_dirs: continue
    session_event_dir = session_event_dirs[0]

    # Find Target Channels from CSV filenames
    swr_csvs = list(session_event_dir.glob(f"*probe_{probe_id}_channel_*_putative_swr_events.csv.gz"))
    move_csvs = list(session_event_dir.glob(f"*probe_{probe_id}_*movement*.csv.gz"))
    
    if not (swr_csvs and move_csvs): 
        print(f"Skipping {session_id} probe {probe_id}: Missing target CSVs.")
        continue
    
    swr_target_id = int(re.search(r"channel_(\d+)", swr_csvs[0].name).group(1))
    move_target_id = int(re.search(r"channel_(\d+)", move_csvs[0].name).group(1))
    
    try:
        with xr.open_dataarray(lfp_path) as ds:
            channels = ds.channel.values 
            
            # Find index in raw orientation (Superficial -> Deep)
            swr_idx = np.where(channels == swr_target_id)[0][0]
            move_idx = np.where(channels == move_target_id)[0][0]
            
            # Anchor at index-3 to put target at relative index 3 of 8-ch window
            swr_start = max(0, min(swr_idx - 3, len(channels) - 8))
            move_start = max(0, min(move_idx - 3, len(channels) - 8))

            for model_num in model_lib:
                model_name = f'Tune_{model_num}_'
                
                # SAVE_ROOT matches the script's internal config
                save_root = Path(f'/mnt/hpc/projects/OWVinckSWR/Cem/VisualBehavior/predictions_ca1_cortical_2500Hz/{model_name}/temp_windows/')
                
                # Pattern: {original_lfp_name}_win{index:02d}.npy
                swr_out_file = save_root / f"{lfp_name}_win{swr_start:02d}.npy"
                move_out_file = save_root / f"{lfp_name}_win{move_start:02d}.npy"

                # Check and submit SWR
                if swr_out_file.exists():
                    print(f"Skipping SWR: {swr_out_file.name} exists.")
                else:
                    print(f"Submitting SWR | Ses {session_id} | Win {swr_start}")
                    if not dry_run:
                        subprocess.call(['sbatch', 'cpu_batch_16GB_campbell.sh', model_name, str(iv), str(swr_start), tag])
                        # subprocess.call(['sbatch', 'cpu_batch_8GB_campbell.sh', model_name, str(iv), str(swr_start), tag])
                    time.sleep(0.001)
                # pdb.set_trace()
                # Check and submit Move
                if move_out_file.exists():
                    print(f"Skipping Move: {move_out_file.name} exists.")
                else:
                    print(f"Submitting Move | Ses {session_id} | Win {move_start}")
                    if not dry_run:
                        subprocess.call(['sbatch', 'cpu_batch_16GB_campbell.sh', model_name, str(iv), str(move_start), tag])
                        # subprocess.call(['sbatch', 'cpu_batch_8GB_campbell.sh', model_name, str(iv), str(move_start), tag])
                    time.sleep(0.001)

    except Exception as e:
        print(f"Error processing session {session_id}: {e}")
    # pdb.set_trace()
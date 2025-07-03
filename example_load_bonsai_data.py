from load_bonsai_data import load_bonsai_data
import numpy as np
import pdb

# Block 1: Load LFP data
lfp_file_path = '/mnt/hpc/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/amplifier-data_1.raw'
lfp_data = load_bonsai_data(file_path=lfp_file_path, dtype=np.uint16, channels=32)
print(f"LFP data shape: {lfp_data.shape}")

# Block 2: Load analog data
analog_file_path = '/mnt/hpc/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/analog-data_1.raw'
analog_data = load_bonsai_data(file_path=analog_file_path, dtype='float32', channels=12)
print(f"Analog data shape: {analog_data.shape}")

# Block 3: Load probability data
prob_file_path = '/mnt/hpc/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/ripple-prob_1.raw'
prob_data = load_bonsai_data(file_path=prob_file_path, dtype='float32', channels=1)
print(f"Probability data shape: {prob_data.shape}")

# Block 4: Load input model data
inputs_file_path = '/mnt/hpc/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/input_model1.raw'
inputs_data = load_bonsai_data(file_path=inputs_file_path, dtype='float32', channels=8)
print(f"Inputs data shape: {inputs_data.shape}")



pdb.set_trace()  # 

# Block to write LFP data as int16
lfp_data_int16 = (lfp_data.astype(np.int32) - 32768).astype(np.int16) # Convert to int16
output_file_path = '/mnt/hpc/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/amplifier-data_1_int16.raw'
lfp_data_int16.tofile(output_file_path)
print(f"Converted LFP data saved to {output_file_path}")

# Convert LFP data to microvolts
lfp_data_microvolts = (lfp_data.astype(np.int32) - 32768).astype(np.float32) * 0.195 # Convert to int16
output_file_path = '/mnt/hpc/projects/OWVinckSWR/Dataset/ONIXData/Awake02_Test/amplifier-data_1_float32.raw'
lfp_data_int16.tofile(output_file_path)
print(f"Converted LFP data to microvolts saved to {output_file_path}")

import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 
import numpy as np

# model_lib = ['predSWR_TCN_BaseModel']
# model_lib = ['predSWR_TCN_BaseModel_k2']
# model_lib = ['predSWR_TCN_BaseModel_k3_t40']
# model_lib = ['predSWR_TCN_BaseModel_k3_t40_d8']
# model_lib = ['predSWR_TCN_BaseModel_k5_t40_d8']
# model_lib = ['predSWR_TCN_BaseModel_k5_t40_d8_n256']
# model_lib = ['predSWR_TCN_BaseModel_k5_t40_d8_n64_customFocalLoss']
# model_lib = ['predSWR_TCN_BaseModel_k5_t40_d8_n64_drop']
# model_lib = ['predSWR_TCN_BaseModel_k5_t40_d8_n64_glorot']
# model_lib = ['predSWR_TCN_BaseModel_k5_t40_d8_dilated_elu00']
# model_lib = ['predSWR_TCN_BaseModel_k5_t40_d8_dilated_relu']
# model_lib = ['predSWR_TCN_BaseModel_k5_t40_d8_dilated_elu00_dedilated']
# model_lib = ['predSWR_TCN_BaseModel_k2_t40_d8_dilated_elu00_dedilated']
# model_lib = ['predSWR_TCN_BaseModel_k2_t40_d8_dilated_elu00_dedilated']
# model_lib = ['predSWR_TCNGroup_k2_t40_d32_n128_e500_relu_dedilated_weightNorm_drop']
# model_lib = ['predSWR_TCNGroup_k3_t40_d16_n128_e500_ELU_dedilated_weightNorm']

# model_lib = ['predSWR_TCN_BaseModel_k2_t96_d32_n64_AnchorLoss']
# model_lib = ['predSWR_TCN_BaseModel_k3_t96_d32_n64_AnchorLoss']
# model_lib = ['predSWR_TCN_BaseModel_k3_t96_d32_n128_AnchorLoss']
# model_lib = ['predSWR_TCN_BaseModel_k3_t96_d32_n64_AnchorLossDrop05']

# model_lib = ['Base_k5_t96_d32_n64_AnchorLossFix']
# model_lib = ['Base_k3_t96_d32_n64_AnchorLossFix']
# model_lib = ['Base_K3_T96_D32_N128_B32_Le4_AnchorLossFix']
# model_lib = ['Base_K3_T96_D32_N128_B32_Le4_AnchorLossFixAdamW']
# model_lib = ['Base_K3_T96_D32_N256_B32_Le4_AnchorLossFix']
# model_lib = ['Base_K3_T96_D32_N128_B32_Le4_AnchorLossFixZERO']
# model_lib = ['Base_K3_T96_D32_N64_B32_Le4_AnchorLossFixNarrow']
# model_lib = ['Base_K3_T96_D32_N64_B32_Le4_AnchorLossFixNarrowNonZero']
# model_lib = ['Base_K2_T96_D32_N64_B32_Le4_AnchorLossFix']
# model_lib = ['Base_K5_T96_D32_N64_B32_Le4_AnchorLossFix']
# model_lib = ['Base_K4_T200_D64_N64_B16_Le4_AnchorLossFix']
model_lib = ['Base_K2_T200_D128_N64_B16_Le4_AnchorLossFix']
# model_lib = ['Average_K3_T100_D32_N128_B32_Le4_AnchorLossFix']
# model_lib = ['Average_K3_T100_D32_N64_B32_Le4_AnchorLossFix']
# model_lib = ['Average_K3_T100_D32_N64_B16_Le4_AnchorLossFix']
# model_lib = ['Average_K5_T200_D32_N64_B16_Le4_AnchorLossFix']
# model_lib = ['Average_K3_T200_D32_N64_B16_Le4_AnchorLossFix']
# model_lib = ['Average_K4_T200_D64_N64_B16_Le4_AnchorLossFix']

for model_name in model_lib:
    exp_dir = 'experiments/' + model_name
    pr = exp_dir + '/model/'
    if not path.exists(pr):
        print(exp_dir + '/model')
        shutil.copytree('model', exp_dir +'/model')
        shutil.copyfile('./pred.py', exp_dir +'/pred.py')
        time.sleep(0.1)
    subprocess.call(['sbatch', 'gpu_batch_107.sh', model_name])
    # if ijob < 3:
    #     subprocess.call(['sbatch', 'gpu_batch_power_GPUshort.sh', model_name])
    # else:
    #     subprocess.call(['sbatch', 'gpu_batch_gpu.sh', model_name])
        

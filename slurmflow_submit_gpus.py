import subprocess
import os.path
from os import path
import shutil
import time

# TCN-only models with receptive field < 64 pixels
model_lib = [
    # Base TCN models
    'Base_K3_T64_D4_N32_L3_E300_B32_S32_TCNOnly_GloLN_Loss012Shift00',
    'Base_K3_T64_D4_N64_L3_E300_B32_S32_TCNOnly_GloLN_Loss012Shift00',
    
    # TCN with different normalizations
    'Base_K3_T64_D4_N32_L3_E300_B32_S32_TCNOnly_GloWReg_Loss012Shift00',
    'Base_K3_T64_D4_N64_L3_E300_B32_S32_TCNOnly_GloWReg_Loss012Shift00',
    'Base_K3_T64_D4_N32_L3_E300_B32_S32_TCNOnly_GloBN_Loss012Shift00',
    'Base_K3_T64_D4_N32_L3_E300_B32_S32_TCNOnly_GloAdaNorm_Loss012Shift00',
    
    # TCN with different dilation patterns while keeping RF < 64
    'Base_K3_T64_D4_N32_L3_E300_B32_S32_TCNOnly_GloLN_LinearDilation_Loss012Shift00',
    'Base_K3_T64_D4_N32_L3_E300_B32_S32_TCNOnly_GloLN_PyramidDilation_Loss012Shift00'
]

# Rest of script remains unchanged
ijob = -1
for model_name in model_lib:
    exp_dir = 'experiments/' + model_name
    pr = exp_dir + '/model/'
    if not path.exists(pr):
        print(exp_dir + '/model')
        shutil.copytree('model', exp_dir +'/model')
        shutil.copyfile('./pred.py', exp_dir +'/pred.py')
        time.sleep(0.1)
    ijob += 1
    subprocess.call(['sbatch', 'gpu_batch_107test.sh', model_name])


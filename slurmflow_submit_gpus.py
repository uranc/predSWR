import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy
import numpy as np




model_lib = ['Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto150Focal_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto150FocalGap_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto150FocalAnchor_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto150FocalGapAnchor_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto054Focal_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto054FocalGap_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto054FocalAnchor_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto054FocalGapAnchor_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto004Focal_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto004FocalGap_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto004FocalAnchor_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto004FocalGapAnchor_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto150Tversky_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto150TverskyGap_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto150TverskyAnchor_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto150TverskyGapAnchor_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto054Tversky_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto054TverskyGap_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto054TverskyAnchor_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto054TverskyGapAnchor_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto004Tversky_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto004TverskyGap_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto004TverskyAnchor_GloWReg_Shift00',
             'Base_K4_T50_D3_N32_L3_E100_B32_S50_XProto004TverskyGapAnchor_GloWReg_Shift00',
             ]


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
    # subprocess.call(['sbatch', 'gpu_batch_107short.sh', model_name])
    # subprocess.call(['sbatch', 'gpu_batch_107long.sh', model_name])
    # if ijob % 10 < 4:
    #     # subprocess.call(['sbatch', 'gpu_batch_107test.sh', model_name])
    # # elif ijob % 10 < 6:
    # # #     print('short', ijob)
    #     # subprocess.call(['sbatch', 'gpu_batch_power_GPUshort.sh', model_name])
    #     subprocess.call(['sbatch', 'gpu_batch_107short.sh', model_name])
    #     # subprocess.call(['sbatch', 'gpu_batch_107long.sh', model_name])
    # else:
    #     print('long', ijob)
    #     subprocess.call(['sbatch', 'gpu_batch_107test.sh', model_name])
    #     # subprocess.call(['sbatch', 'gpu_batch_gpu.sh', model_name])
        # subprocess.call(['sbatch', 'gpu_batch_107short.sh', model_name])
        # subprocess.call(['sbatch', 'gpu_batch_107long.sh', model_name])


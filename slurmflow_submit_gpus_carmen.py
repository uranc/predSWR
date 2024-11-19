import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy
import numpy as np


# model_lib = ['Base_K4_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Shift00',
#              'Base_K4_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop10Shift00',
#              'Base_K4_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop20Shift00',
#              'Base_K4_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop05Shift00', 
#              'Base_K5_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop05Shift00', 
#              'Base_K4_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Shift00', 
#              'Base_K4_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop10Shift00', 
#              'Base_K4_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop20Shift00', 
#              'Base_K5_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop10Shift00',
#              'Base_K5_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop20Shift00', 
#              'Base_K5_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Shift00',
#              'Base_K5_T50_D4_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Shift00',
#              'Base_K4_T50_D4_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Shift00',
#              'Base_K4_T50_D4_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop10Shift00',
#              'Base_K7_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop10Shift00'
#     ]
# model_lib = ['',
#              '',
#              '',
#              '',
#              '',
#              '',
#              '',
#              '',
#              '',
#              '',
#              '',
#              '',
#             ]
model_lib = ['Base_K4_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiNormHori05Loss011Shift00',
            ]

ijob = -1
for model_name in model_lib:
    exp_dir = '/cs/projects/OWVinckSWR/DL/predSWR/experiments_carmen/' + model_name
    pr = exp_dir + '/model/'
    if not path.exists(pr):
        print(exp_dir + '/model')
        shutil.copytree('/cs/projects/OWVinckSWR/DL/predSWR/model', exp_dir +'/model')
        shutil.copyfile('/cs/projects/OWVinckSWR/DL/predSWR/pred_carmen.py', exp_dir +'/pred_carmen.py')
        time.sleep(0.1)
    ijob += 1
    subprocess.call(['sbatch', '/cs/projects/OWVinckSWR/DL/predSWR/gpu_batch_107test_carmen.sh', model_name])
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


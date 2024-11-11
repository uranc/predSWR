import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy

model_lib = ['Base_K4_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Shift00',
             'Base_K4_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop10Shift00',
             'Base_K4_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop20Shift00',
             'Base_K4_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop05Shift00', 
             'Base_K5_T50_D3_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop05Shift00', 
             'Base_K4_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Shift00', 
             'Base_K4_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop10Shift00', 
             'Base_K4_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop20Shift00', 
             'Base_K5_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop10Shift00',
             'Base_K5_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop20Shift00', 
             'Base_K5_T50_D2_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Shift00',
             'Base_K5_T50_D4_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Shift00',
             'Base_K4_T50_D4_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Shift00',
             'Base_K4_T50_D4_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop10Shift00',
             'Base_K5_T50_D4_N32_L3_E400_B32_S50_FocalGapAx025Gx200Margin_GloWReg_MMaskedmultiHori05Loss011Drop10Shift00',
    ]
for im, model in enumerate(model_lib):
    # subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model])
    subprocess.call(['sbatch', 'cpu_batch_16GBXS_carmen.sh', model, str(im)])
    # pr = '/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/horis_val{0}_{1}.npy'.format(1, model)
    # if not path.exists(pr):
    #     print('submitting job for model: ', model, im)
    #     # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
    #     # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
    # else:
    #     print('model already predicted: ', model)
    # # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
    # # subprocess.call(['python', 'pred.py', '--mode', 'predictPlot', '--model', model])
    # # subprocess.call(['sbatch', 'gpu_batch_107inference.sh', model])



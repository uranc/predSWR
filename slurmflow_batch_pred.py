import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy

model_lib = ['Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalAx025Gx200_GloWReg_Hori01Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalAx025Gx200_GloWReg_Hori05Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalAx025Gx200_GloWReg_Hori10Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalAx025Gx200_GloWReg_Hori20Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalGapAx025Gx200_GloWReg_Hori01Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalGapAx025Gx200_GloWReg_Hori05Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalGapAx025Gx200_GloWReg_Hori10Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalGapAx025Gx200_GloWReg_Hori20Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalGapAx025Gx100_GloWReg_Hori01Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalGapAx025Gx100_GloWReg_Hori05Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalGapAx025Gx100_GloWReg_Hori10Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_FocalGapAx025Gx100_GloWReg_Hori20Loss012Shift00']


for im, model in enumerate(model_lib):
    # subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model])
    subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
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



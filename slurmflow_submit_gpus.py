import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 
import numpy as np


# Average_K4_T50_D16_N16_B128_L2_E200_S5_FocalGap_WRegDense
# Average_K4_T50_D16_N16_B512_L3_E200_S10_AnchorGap_WReg_AvgBottle
# Average_K4_T50_D16_N16_B64_L3_E200_S10_AnchorGap_WReg_AvgBottle
# Average_K4_T50_D16_N64_B64_L3_E200_S10_AnchorGap_WReg_AvgBottle
# Average_K4_T50_D16_N64_B64_L3_E200_S5_AnchorGap_WReg_DenseBottle

# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S25_Focal_WReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S25_FocalGap_WReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGap_WReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGap_WReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L4_E200_S25_AnchorGap_WReg_AvgBottle'] # every other time sasmple

# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S25_Focal_GloWReg_AvgBottle'] # every other time sasmple
# model_lib = ['Base_K4_T50_D16_N64_B64_L3_E200_S25_Focal_GloWReg_AvgBottle'] # every other time sasmple
# model_lib = ['Base_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGapWide_GloWReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGapWide_GloWReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S25_FocalGap_GloWReg_AvgBottleDense'] # every other time sasmple

# model_lib = ['Average_K2_T50_D64_N32_B32_L3_E200_S25_FocalGap_GloWReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K2_T50_D64_N32_B32_L3_E200_S25_AnchorGap_GloWReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K2_T50_D64_N32_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K2_T50_D64_N32_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K2_T50_D64_N32_B64_L4_E200_S25_AnchorGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N32_B64_L4_E200_S25_AnchorGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N32_B64_L4_E200_S25_AnchorGapWide_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N32_B64_L3_E200_S25_AnchorGapWide_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N16_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottleDenseELU'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N16_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottleDenseELU'] # every other time sasmple

# model_lib = ['Average_K7_T50_D8_N16_B128_L3_E200_S50_FocalGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K7_T50_D8_N16_B64_L3_E200_S50_FocalGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K7_T50_D8_N32_B32_L3_E200_S50_FocalGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K7_T50_D8_N32_B32_L3_E200_S100_FocalGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K13_T50_D4_N32_B32_L3_E200_S100_FocalGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K13_T50_D4_N32_B32_L3_E200_S50_FocalGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K13_T50_D4_N32_B32_L3_E200_S25_FocalGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K13_T50_D4_N32_B32_L4_E200_S25_FocalGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K13_T50_D4_N32_B32_L3_E200_S25_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K13_T50_D4_N32_B32_L3_E200_S50_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K13_T50_D4_N32_B32_L3_E200_S100_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K13_T50_D4_N64_B32_L3_E200_S50_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K13_T50_D4_N128_B32_L3_E200_S50_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K3_T50_D16_N128_B32_L3_E200_S50_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Average_K3_T50_D16_N128_B32_L3_E200_S100_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Base_K3_T50_D16_N64_L3_E200_B64_S50_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Base_K3_T50_D16_N64_L3_E200_B64_S100_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Base_K7_T50_D8_N64_L3_E200_B64_S50_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Base_K13_T50_D4_N64_L3_E200_B64_S50_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Base_K13_T50_D4_N64_L3_E200_B64_S100_AnchorGap20_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Base_K13_T50_D4_N64_L3_E200_B64_S100_FocalGap_GloWReg_AvgBottleDense'] # every other time sasmple
# model_lib = ['Base_K13_T50_D4_N64_L3_E200_B64_S50_FocalGap_GloWReg_AvgBottleDense'] # every other time sasmple

# model_lib = ['Base_K7_T50_D8_N64_L3_E200_B64_S50_AnchorGap_GloWReg_AvgBottleDenseLBuff'] # every other time sasmple
# model_lib = ['Base_K7_T50_D8_N64_L3_E200_B64_S50_AnchorGap_GloWReg_AvgBottleDenseL1'] # every other time sasmple
# model_lib = ['Base_K7_T50_D8_N64_L3_E200_B64_S50_FocalSmooth_GloWReg_AvgBottleDenseLBuff'] # every other time sasmple
# model_lib = ['Average_K7_T50_D8_N64_L3_E200_B64_S50_FocalSmoothTV_GloWReg_AvgBottleDenseLBuff'] # every other time sasmple
# model_lib = ['Average_K7_T50_D8_N64_L3_E200_B16_S50_FocalSmoothTV_GloWReg_AvgBottleDenseLBuff'] # every other time sasmple
# model_lib = ['Average_K24_T50_D2_N64_L3_E200_B32_S50_FocalSmooth_GloWReg_AvgBottleDenseLBuff'] # every other time sasmple
# model_lib = ['Average_K24_T50_D2_N64_L3_E200_B32_S50_FocalSmoothTV_GloWReg_AvgBottleDenseLBuff'] # every other time sasmple

# model_lib = ['Average_K4_T50_D3_N64_L3_E200_B64_S50_Focal_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L3_E200_B64_S50_FocalGap_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L3_E200_B64_S50_FocalSmooth_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B64_S50_Focal_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B64_S50_FocalGap_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B64_S50_FocalSmooth_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L3_E200_B32_S50_Hinge_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B32_S50_Hinge_GloWReg_BottleDense']
# model_lib = ['Average_K4_T50_D3_N64_L3_E200_B32_S50_Tversky07_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B32_S50_Tversky07_GloWReg_BottleDense']
model_lib = ['Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowGap_GloWReg_BottleDense',
             'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorWiderGap_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowGap_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorWiderGap_GloWReg_BottleDense',
             'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowSmoothGap_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowSmoothGap_GloWReg_BottleDense'
             ]

# Average_K3_T50_D16_N128_B32_L3_E200_S100_AnchorGap20_GloWReg_AvgBottleDense
# Average_K13_T50_D4_N32_B32_L3_E200_S100_AnchorGap20_GloWReg_AvgBottleDense
# Average_K13_T50_D4_N32_B32_L3_E200_S25_AnchorGap20_GloWReg_AvgBottleDense
# Average_K13_T50_D4_N32_B32_L3_E200_S25_FocalGap_GloWReg_AvgBottleDense
# Average_K13_T50_D4_N32_B32_L4_E200_S25_FocalGap_GloWReg_AvgBottleDense
# Average_K7_T50_D8_N32_B32_L3_E200_S100_FocalGap_GloWReg_AvgBottleDense
# Average_K13_T50_D4_N32_B32_L3_E200_S100_FocalGap_GloWReg_AvgBottleDense

# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S10_AnchorGap_WReg_AvgBottleDStride'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N16_B64_L3_E200_S10_AnchorGap_L1Reg_HeInitELU_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N16_B64_L3_E200_S10_AnchorGap_L1Reg_HeInit_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N16_B64_L3_E200_S10_AnchorGap_L1Reg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N16_B512_L3_E200_S10_AnchorGap_WReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N16_B64_L3_E200_S10_AnchorGap_WReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S10_AnchorGap_WReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S5_AnchorGap_WReg_AvgBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S5_AnchorGap_WReg_DenseBottle'] # every other time sasmple
# model_lib = ['Average_K3_T36_D16_N64_B128_L2_E200_S5_AnchorGap_NoReg_DenseBottle'] # every other time sasmple
# model_lib = ['Average_K3_T36_D16_N64_B128_L2_E200_S5_FocalGap_NoReg_DenseBottle'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N16_B128_L2_E200_S5_FocalGap_WRegDense'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N16_B128_L2_E200_S5_AnchorGap_WRegDense'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N32_B64_L2_E200_S5_AnchorGap_WReg'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N32_B128_L2_E200_S2_AnchorGap_WReg'] # every other time sasmple
# model_lib = ['Average_K4_T50_D16_N32_B64_L3_E200_S2_AnchorGap_WReg'] # every other time sasmple
# model_lib = ['Base_K7_T100_D16_N32_B64_L2_E200_S4_AnchorGap_WReg'] # every other time sasmple
# model_lib = ['Base_K4_T50_D16_N32_B256_L2_E200_AnchorGap_WReg'] # every other time sasmple
# model_lib = ['Base_K4_T50_D16_N32_B256_L3_E200_AnchorGap_WReg'] # every other time sasmple
# model_lib = ['Base_K4_T50_D16_N32_B256_L4_E200_AnchorGap_WReg'] # every other time sasmple
# model_lib = ['Base_K4_T50_D16_N32_B64_L2_E200_AnchorGap_WReg'] # every other time sasmple
# model_lib = ['Base_K4_T50_D16_N32_B64_L3_E200_AnchorGap_WReg'] # every other time sasmple
# model_lib = ['Base_K4_T50_D16_N32_B64_L4_E200_AnchorGap_WReg'] # every other time sasmple

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
        

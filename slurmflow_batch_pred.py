import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 

# model_lib = ['Average_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottle',
#             'Average_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottleDense',
#             'Average_K4_T50_D16_N64_B64_L3_E200_S25_Focal_GloWReg_AvgBottle',
#             'Base_K4_T50_D16_N64_B64_L3_E200_S25_Focal_GloWReg_AvgBottle',
#             'Base_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGapWide_GloWReg_AvgBottle',
#             'Average_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGapWide_GloWReg_AvgBottle',
#             'Average_K4_T50_D16_N64_B64_L3_E200_S25_FocalGap_GloWReg_AvgBottleDense',
#             'Average_K2_T50_D64_N32_B32_L3_E200_S25_FocalGap_GloWReg_AvgBottle',
#             'Average_K2_T50_D64_N32_B32_L3_E200_S25_AnchorGap_GloWReg_AvgBottle',
#             'Average_K2_T50_D64_N32_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottle',
#             'Average_K2_T50_D64_N32_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottleDense',
#             'Average_K2_T50_D64_N32_B64_L4_E200_S25_AnchorGap_GloWReg_AvgBottleDense',
#             'Average_K4_T50_D16_N32_B64_L4_E200_S25_AnchorGap_GloWReg_AvgBottleDense',
#             'Average_K4_T50_D16_N32_B64_L4_E200_S25_AnchorGapWide_GloWReg_AvgBottleDense',
#             'Average_K4_T50_D16_N32_B64_L3_E200_S25_AnchorGapWide_GloWReg_AvgBottleDense',
#             'Average_K4_T50_D16_N16_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottleDense',
#             'Average_K4_T50_D16_N64_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottleDenseELU',
#             'Average_K4_T50_D16_N16_B64_L3_E200_S25_AnchorGap_GloWReg_AvgBottleDenseELU']

# model_lib = ['Average_K7_T50_D8_N16_B128_L3_E200_S50_FocalGap_GloWReg_AvgBottleDense',
#             'Average_K7_T50_D8_N16_B64_L3_E200_S50_FocalGap_GloWReg_AvgBottleDense',
#             'Average_K7_T50_D8_N32_B32_L3_E200_S50_FocalGap_GloWReg_AvgBottleDense',
#             'Average_K7_T50_D8_N32_B32_L3_E200_S100_FocalGap_GloWReg_AvgBottleDense',
#             'Average_K13_T50_D4_N32_B32_L3_E200_S100_FocalGap_GloWReg_AvgBottleDense',
#             'Average_K13_T50_D4_N32_B32_L3_E200_S50_FocalGap_GloWReg_AvgBottleDense',
#             'Average_K13_T50_D4_N32_B32_L3_E200_S25_AnchorGap20_GloWReg_AvgBottleDense',
#             'Average_K13_T50_D4_N32_B32_L3_E200_S50_AnchorGap20_GloWReg_AvgBottleDense',
#             'Average_K13_T50_D4_N32_B32_L3_E200_S100_AnchorGap20_GloWReg_AvgBottleDense',
#             'Average_K13_T50_D4_N64_B32_L3_E200_S50_AnchorGap20_GloWReg_AvgBottleDense',
#             'Average_K13_T50_D4_N128_B32_L3_E200_S50_AnchorGap20_GloWReg_AvgBottleDense',
#             'Average_K3_T50_D16_N128_B32_L3_E200_S50_AnchorGap20_GloWReg_AvgBottleDense',
#             'Average_K3_T50_D16_N128_B32_L3_E200_S100_AnchorGap20_GloWReg_AvgBottleDense',
#             'Base_K3_T50_D16_N64_L3_E200_B64_S50_AnchorGap20_GloWReg_AvgBottleDense',
#             'Base_K3_T50_D16_N64_L3_E200_B64_S100_AnchorGap20_GloWReg_AvgBottleDense',
#             'Base_K7_T50_D8_N64_L3_E200_B64_S50_AnchorGap20_GloWReg_AvgBottleDense',
#             'Base_K13_T50_D4_N64_L3_E200_B64_S50_AnchorGap20_GloWReg_AvgBottleDense',
#             'Base_K13_T50_D4_N64_L3_E200_B64_S100_AnchorGap20_GloWReg_AvgBottleDense',
#             'Base_K13_T50_D4_N64_L3_E200_B64_S100_FocalGap_GloWReg_AvgBottleDense',
#             'Base_K13_T50_D4_N64_L3_E200_B64_S50_FocalGap_GloWReg_AvgBottleDense']

# model_lib = ['Base_K7_T50_D8_N64_L3_E200_B64_S50_AnchorGap_GloWReg_AvgBottleDenseLBuff',
#             'Base_K7_T50_D8_N64_L3_E200_B64_S50_AnchorGap_GloWReg_AvgBottleDenseL1',
#             'Base_K7_T50_D8_N64_L3_E200_B64_S50_FocalSmooth_GloWReg_AvgBottleDenseLBuff',
#             'Average_K7_T50_D8_N64_L3_E200_B64_S50_FocalSmoothTV_GloWReg_AvgBottleDenseLBuff',
#             'Average_K7_T50_D8_N64_L3_E200_B16_S50_FocalSmoothTV_GloWReg_AvgBottleDenseLBuff']
model_lib = ['Average_K4_T50_D3_N64_L3_E200_B64_S50_Focal_GloWReg_BottleDense',
             'Average_K4_T50_D3_N64_L3_E200_B64_S50_FocalGap_GloWReg_BottleDense',
             'Average_K4_T50_D3_N64_L3_E200_B64_S50_FocalSmooth_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_Focal_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_FocalGap_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_FocalSmooth_GloWReg_BottleDense',
             'Average_K4_T50_D3_N64_L3_E200_B32_S50_Hinge_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B32_S50_Hinge_GloWReg_BottleDense']
# model_lib = ['Average_K24_T50_D2_N64_L3_E200_B32_S50_FocalSmooth_GloWReg_AvgBottleDenseLBuff',
#             'Average_K24_T50_D2_N64_L3_E200_B32_S50_FocalSmoothTV_GloWReg_AvgBottleDenseLBuff'] 
# model_lib = ['Average_K24_T50_D2_N64_L3_E200_B32_S50_FocalSmooth_GloWReg_AvgBottleDenseLBuff',
#             'Average_K24_T50_D2_N64_L3_E200_B32_S50_FocalSmoothTV_GloWReg_AvgBottleDenseLBuff'] 

for model in model_lib:
    subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model])


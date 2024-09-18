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
# model_lib = ['Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowGap_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorWiderGap_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowGap_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorWiderGap_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowSmoothGap_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowSmoothGap_GloWReg_BottleDense'
#              ]

# model_lib = ['Average_K4_T50_D3_N64_L4_E200_B32_S50_Focal_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_FocalGap_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_Focal_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalGap_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGap_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_AnchorWiderGap_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGap_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorWiderGap_GloWReg_BottleDense',
#              ]
# model_lib = ['Average_K4_T50_D3_N64_L4_E200_B32_S50_FocalTVFix_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_FocalGapTVFix_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalTVFix_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalGapTVFix_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGapTVFix_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_AnchorWiderGapTVFix_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGapTVFix_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorWiderGapTVFix_GloWReg_BottleDense',
#              ]
# model_lib = ['Average_K4_T50_D3_N64_L4_E200_B32_S50_FocalTV_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_FocalGapTV_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalTV_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalGapTV_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGapTV_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_AnchorWiderGapTV_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGapTV_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorWiderGapTV_GloWReg_BottleDense',
#              ]
# model_lib = ['Average_K4_T50_D3_N64_L4_E200_B32_S50_FocalTV3_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_FocalGapTV3_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalTV3_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalGapTV3_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGapTV3_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_AnchorWiderGapTV3_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGapTV3_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorWiderGapTV3_GloWReg_BottleDense',
#              ]
# model_lib = ['Average_K4_T50_D3_N64_L4_E200_B32_S50_Focal_GloWReg_BottleDenseELU',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_FocalGap_GloWReg_BottleDenseELU',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_Focal_GloWReg_BottleDenseELU',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalGap_GloWReg_BottleDenseELU',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGap_GloWReg_BottleDenseELU',
#              'Average_K4_T50_D3_N64_L4_E200_B32_S50_AnchorWiderGap_GloWReg_BottleDenseELU',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGap_GloWReg_BottleDenseELU',
#              'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorWiderGap_GloWReg_BottleDenseELU',
#              ]
# model_lib = ['Average_K4_T50_D3_N64_L4_E200_B64_S50_FocalMargin_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B64_S50_FocalGapMargin_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B64_S50_FocalMargin_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B64_S50_FocalGapMargin_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B64_S50_AnchorNarrowGapMargin_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B64_S50_AnchorWiderGapMargin_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B64_S50_AnchorNarrowGapMargin_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B64_S50_AnchorWiderGapMargin_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B64_S50_FocalMargin_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B64_S50_FocalGapMarginTV_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B64_S50_FocalMarginTV_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B64_S50_FocalGapMarginTV_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B64_S50_AnchorNarrowGapMarginTV_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L4_E200_B64_S50_AnchorWiderGapMarginTV_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B64_S50_AnchorNarrowGapMarginTV_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L4_E200_B64_S50_AnchorWiderGapMarginTV_GloWReg_BottleDense',
#              ]

# model_lib = ['Base_K4_T50_D3_N64_L3_E200_B64_S50_FocalGapMarginTV25_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B32_S50_FocalGapMarginTV25_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B64_S25_FocalGapMarginTV25_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N128_L3_E200_B32_S50_FocalGapMarginTV25_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N256_L3_E200_B32_S50_FocalGapMarginTV25_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N32_L3_E200_B32_S50_FocalGapMarginTV25_GloWReg_BottleDense']
#              'Base_K4_T50_D3_N64_L3_E200_B64_S50_FocalGapMarginTV25Drop05_GloWReg_BottleDense']


# model_lib = ['Base_K4_T50_D3_N32_L3_E200_B32_S50_FocalGapMarginTV14_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N32_L3_E200_B32_S50_FocalGapMarginTV14_GloWReg_BottleDenseDrop05',
#              'Base_K4_T50_D3_N32_L3_E200_B32_S50_AnchorNarrowGapMarginTV14_GloWReg_BottleDenseDrop05',
#              'Average_K4_T50_D3_N32_L3_E200_B32_S50_FocalGapMarginTV14_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N32_L3_E200_B32_S50_AnchorNarrowGapMarginTV14_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N32_L3_E200_B32_S50_AnchorNarrowGapMarginTV14_GloWReg_BottleDenseDrop05',
#              'Average_K4_T50_D3_N32_L3_E200_B64_S50_FocalGapMarginTV14_GloWReg_Bottle',
#              ]

# model_lib = ['Base_K4_T50_D3_N32_L3_E200_B32_S50_FocalGapMarginTV14_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N32_L3_E200_B32_S50_FocalGapMarginTV14_GloWReg_BottleDenseDrop05',
#              'Base_K4_T50_D3_N32_L3_E200_B32_S50_AnchorNarrowGapMarginTV14_GloWReg_BottleDenseDrop05',
#              'Base_K4_T50_D3_N32_L3_E200_B32_S50_AnchorNarrowGapMarginTV14_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N32_L3_E200_B32_S50_AnchorNarrowGapMarginTV14_GloWReg_BottleDenseDrop05',
#              ]
# model_lib = ['Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalGapSmoothMarginTV15_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowGapSmoothMarginTV15_GloWReg_BottleDense',
#              'Base_K9_T50_D2_N32_L3_E300_B32_S50_FocalGapSmoothMarginTV15_GloWReg_BottleDense',
#              'Base_K9_T50_D2_N32_L3_E300_B32_S50_AnchorNarrowGapSmoothMarginTV15_GloWReg_BottleDense',
#              ]
# model_lib = ['Base_K9_T50_D2_N32_L3_E200_B64_S50_FocalGapSmoothMarginL215_GloWReg_BottleDense',
#              'Base_K9_T50_D2_N32_L3_E200_B64_S50_AnchorNarrowGapSmoothMarginL215_GloWReg_BottleDense',
#              ]
# model_lib = ['Base_K9_T50_D2_N32_L3_E200_B64_S50_FocalGapSmoothMarginL215_GloWReg_BottleDenseShift',
#              'Base_K9_T50_D2_N32_L3_E200_B64_S50_AnchorNarrowGapSmoothMarginL215_GloWReg_BottleDenseShift',
#              ]
model_lib = [
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmoothMarginTV055_GloWReg_GapFixShift10',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapMarginTV055_GloWReg_GapFixShift10',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapSmoothMarginTV055_GloWReg_GapFixShift10',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmoothMarginTV055_GloWReg_GapFixShift10',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapMarginTV055_GloWReg_GapFixShift10',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapSmoothMarginTV055_GloWReg_GapFixShift10',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorWiderGapSmoothMarginTV055_GloWReg_GapFixShift10',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmoothMarginTV055_GloWReg_GapFixShift05',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapMarginTV055_GloWReg_GapFixShift05',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapSmoothMarginTV055_GloWReg_GapFixShift05',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmoothMarginTV055_GloWReg_GapFixShift05',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapMarginTV055_GloWReg_GapFixShift05',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapSmoothMarginTV055_GloWReg_GapFixShift05',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorWiderGapSmoothMarginTV055_GloWReg_GapFixShift05',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmoothMarginTV055_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapMarginTV055_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapSmoothMarginTV055_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmoothMarginTV055_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapMarginTV055_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapSmoothMarginTV055_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorWiderGapSmoothMarginTV055_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmoothMargin05L2_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapMargin05L2_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapSmoothMargin05L2_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05L2_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapMargin05L2_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapSmoothMargin05L2_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorWiderGapSmoothMargin05L2_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_Focal_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapMargin05_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapMargin05_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorWiderGapSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_Focal_GloWReg_GapFixShift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalGapMargin05_GloWReg_GapFixShift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalGapSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowGapMargin05_GloWReg_GapFixShift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowGapSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorWiderGapSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K3_T50_D4_N32_L4_E200_B32_S50_Focal_GloWReg_GapFixShift00',
            #  'Base_K3_T50_D4_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K3_T50_D4_N32_L4_E200_B32_S50_FocalGapMargin05_GloWReg_GapFixShift00',
            #  'Base_K3_T50_D4_N32_L4_E200_B32_S50_FocalGapSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K3_T50_D4_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K3_T50_D4_N32_L4_E200_B32_S50_AnchorNarrowGapMargin05_GloWReg_GapFixShift00',
            #  'Base_K3_T50_D4_N32_L4_E200_B32_S50_AnchorNarrowGapSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K3_T50_D4_N32_L4_E200_B32_S50_AnchorWiderGapSmoothMargin05_GloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmoothMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalGapSmoothMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmoothMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowGapSmoothMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorWiderGapSmoothMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_FocalSmoothMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_FocalGapMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_FocalGapSmoothMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_AnchorNarrowSmoothMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_AnchorNarrowGapMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_AnchorNarrowGapSmoothMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_AnchorWiderGapSmoothMarginTV055_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_FocalSmoothMargin05L2_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_FocalGapMargin05L2_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_FocalGapSmoothMargin05L2_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_AnchorNarrowSmoothMargin05L2_AdamWGloWReg_GapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_AnchorNarrowGapMargin05L2_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_AnchorNarrowGapSmoothMargin05L2_GloWReg_AdamWGapFixShift00',
            #  'Base_K9_T50_D2_N32_L5_E200_B16_S50_AnchorWiderGapSmoothMargin05L2_GloWReg_AdamWGapFixShift00',
            #
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalGapSmooth_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowGapSmooth_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalGapSmoothMargin05_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowGapSmoothMargin05_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalGapSmooth_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowGapSmooth_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalGapSmoothMargin05_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowGapSmoothMargin05_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalGapSmooth_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowGapSmooth_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalGapSmoothMargin05_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowGapSmoothMargin05_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift10',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift00',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift05',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift05',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift05',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift05',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift10',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift10',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift10',
            #  'Base_K9_T50_D2_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmooth_GloWReg_Drop05Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Drop05Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Drop05Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Drop05Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmooth_GloWReg_Drop05Shift05',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Drop05Shift05',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Drop05Shift05',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Drop05Shift05',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmooth_GloWReg_Drop05Shift10',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Drop05Shift10',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Drop05Shift10',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Drop05Shift10',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmoothMargin0000_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmoothMargin0000_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmoothMargin0000_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmoothMargin0000_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalSmoothMargin0000_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowSmoothMargin0000_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx025Gx200Margin0000_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx025Gx200Margin0000_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalGapAx025Gx200Margin0000_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGapAx025Gx200Margin0000_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalGapAx025Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowGapAx025Gx200_GloWReg_Shift00',

            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx025Gx050_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx025Gx050_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx025Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx025Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx025Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx025Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx025Gx400_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx025Gx400_GloWReg_Shift00',

            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx000Gx050_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx000Gx050_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx000Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx000Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx000Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx000Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx000Gx400_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx000Gx400_GloWReg_Shift00',

            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx075Gx050_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx075Gx050_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx075Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx075Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx075Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx075Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx075Gx400_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx075Gx400_GloWReg_Shift00',

            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx099Gx050_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx099Gx050_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx099Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx099Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx099Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx099Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_FocalAx099Gx400_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N64_L4_E200_B32_S50_AnchorNarrowAx099Gx400_GloWReg_Shift00',
            ##############################################
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalAx099Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowAx099Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalAx099Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowAx099Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalAx099Gx200_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowAx099Gx200_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalAx090Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowAx090Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalAx090Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowAx090Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalAx090Gx200_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowAx090Gx200_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalGapAx099Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowGapAx099Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalGapAx099Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowGapAx099Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalGapAx099Gx200_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowGapAx099Gx200_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalGapAx090Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowGapAx090Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalGapAx090Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowGapAx090Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalGapAx090Gx200_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_AnchorNarrowGapAx090Gx200_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalAx025Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowAx025Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalAx025Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowAx025Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_FocalAx025Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_AnchorNarrowAx025Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_FocalAx025Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_AnchorNarrowAx025Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalGapAx025Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowGapAx025Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalGapAx025Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowGapAx025Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_FocalGapAx025Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_AnchorNarrowGapAx025Gx100_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_FocalGapAx025Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_AnchorNarrowGapAx025Gx200_GloWReg_Shift00',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalAx025Gx100_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowAx025Gx100_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalAx025Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowAx025Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_FocalAx025Gx100_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_AnchorNarrowAx025Gx100_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_FocalAx025Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_AnchorNarrowAx025Gx200_GloWReg_Shift05',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalAx025Gx100_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowAx025Gx100_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_FocalAx025Gx200_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowAx025Gx200_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_FocalAx025Gx100_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_AnchorNarrowAx025Gx100_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_FocalAx025Gx200_GloWReg_Shift10',
            #  'Base_K4_T50_D3_N128_L4_E200_B32_S50_AnchorNarrowAx025Gx200_GloWReg_Shift10',
            ##########################################
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_Focal_GloWReg_Shift00',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrow_GloWReg_Shift00',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift05',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift05',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift10',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift10',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_FocalMargin05_GloWReg_Shift00',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowMargin05_GloWReg_Shift00',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift05',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift05',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift10',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift10',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_Focal_GloWReg_Shift00',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrow_GloWReg_Shift00',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift05',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift05',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift10',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift10',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_FocalMargin05_GloWReg_Shift00',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowMargin05_GloWReg_Shift00',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift05',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift05',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift10',
            #  'BaseAvg_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift10',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_Focal_GloWReg_Shift00DenseFirst',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrow_GloWReg_Shift00DenseFirst',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift05DenseFirst',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift05DenseFirst',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmooth_GloWReg_Shift10DenseFirst',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift10DenseFirst',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_FocalMargin05_GloWReg_Shift00DenseFirst',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowMargin05_GloWReg_Shift00DenseFirst',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift05DenseFirst',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift05DenseFirst',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_FocalSmoothMargin05_GloWReg_Shift10DenseFirst',
            #  'Average_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmoothMargin05_GloWReg_Shift10DenseFirst',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalAx090Gx200_GloNoW_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalAx099Gx200_GloNoW_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalAx025Gx200_GloNoW_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalAx099Gx200_GloLNorm_Shift00',
            #  'Base_K4_T50_D3_N32_L3_E300_B32_S50_FocalAx025Gx200_GloLNorm_Shift00',
            #  'Base_K5_T300_D5_N32_L3_E150_B32_S800_FocalAx099Gx200_GloNoW_Shift00',
            #  'Base_K5_T300_D5_N32_L3_E150_B32_S800_FocalAx025Gx200_GloNoW_Shift00',
            #  'Base_K7_T200_D4_N32_L3_E150_B32_S800_FocalAx099Gx200_GloNoW_Shift00',
            #  'Base_K7_T200_D4_N32_L3_E150_B32_S800_FocalAx025Gx200_GloNoW_Shift00',
            #  'Base_K5_T300_D5_N32_L3_E300_B64_S800_FocalAx099Gx200_GloNoW_Shift00',
            #  'Base_K5_T300_D5_N32_L3_E300_B64_S800_FocalAx025Gx200_GloNoW_Shift00',
            #  'Base_K7_T200_D4_N32_L3_E300_B64_S800_FocalAx099Gx200_GloNoW_Shift00',
            #  'Base_K7_T200_D4_N32_L3_E300_B64_S800_FocalAx025Gx200_GloNoW_Shift00',
            #  'Base_K5_T300_D5_N32_L3_E300_B64_S800_FocalAx099Gx100_GloNoW_Shift00',
            #  'Base_K5_T300_D5_N32_L3_E300_B64_S800_FocalAx025Gx100_GloNoW_Shift00',
            #  'Base_K7_T200_D4_N32_L3_E300_B64_S800_FocalAx099Gx100_GloNoW_Shift00',
            #  'Base_K7_T200_D4_N32_L3_E300_B64_S800_FocalAx025Gx100_GloNoW_Shift00',
            #  'Base_K5_T300_D5_N32_L3_E300_B64_S800_FocalSmooth_GloNoW_Shift00',
            #  'Base_K7_T200_D4_N32_L3_E300_B64_S800_FocalSmooth_GloNoW_Shift00',
             'Base_K5_T300_D5_N32_L3_E300_B64_S800_FocalAx099Gx200_GloNoW_Shift00ELU',
             'Base_K5_T300_D5_N32_L3_E300_B64_S800_FocalAx025Gx200_GloNoW_Shift00ELU',
             'Base_K7_T200_D4_N32_L3_E300_B64_S800_FocalAx099Gx200_GloNoW_Shift00ELU',
             'Base_K7_T200_D4_N32_L3_E300_B64_S800_FocalAx025Gx200_GloNoW_Shift00ELU',
             'Base_K5_T300_D5_N32_L3_E300_B64_S800_FocalAx099Gx100_GloNoW_Shift00ELU',
             'Base_K5_T300_D5_N32_L3_E300_B64_S800_FocalAx025Gx100_GloNoW_Shift00ELU',
             'Base_K7_T200_D4_N32_L3_E300_B64_S800_FocalAx099Gx100_GloNoW_Shift00ELU',
             'Base_K7_T200_D4_N32_L3_E300_B64_S800_FocalAx025Gx100_GloNoW_Shift00ELU',             
            #  'Base_K3_T600_D7_N32_L3_E150_B32_S600_FocalAx099Gx200_GloNoW_Shift00Drop05',
            #  'Base_K3_T600_D7_N32_L3_E150_B32_S600_FocalAx090Gx200_GloNoW_Shift00Drop05',
            #  'Base_K3_T600_D7_N32_L3_E150_B32_S600_FocalAx025Gx200_GloNoW_Shift00Drop05',
            #  'Base_K3_T600_D7_N32_L3_E150_B32_S600_FocalAx099Gx100_GloNoW_Shift00Drop05',
            #  'Base_K3_T600_D7_N32_L3_E150_B32_S600_FocalAx090Gx100_GloNoW_Shift00Drop05',
            #  'Base_K3_T600_D7_N32_L3_E150_B32_S600_FocalAx025Gx100_GloNoW_Shift00Drop05',
            #  'Base_K10_T600_D5_N32_L3_E150_B32_S600_FocalAx099Gx200_GloNoW_Shift00Drop05',
            #  'Base_K10_T600_D5_N32_L3_E150_B32_S600_FocalAx090Gx200_GloNoW_Shift00Drop05',
            #  'Base_K10_T600_D5_N32_L3_E150_B32_S600_FocalAx025Gx200_GloNoW_Shift00Drop05',
            #  'Base_K10_T600_D5_N32_L3_E150_B32_S600_FocalAx099Gx100_GloNoW_Shift00Drop05',
            #  'Base_K10_T600_D5_N32_L3_E150_B32_S600_FocalAx090Gx100_GloNoW_Shift00Drop05',
            #  'Base_K10_T600_D5_N32_L3_E150_B32_S600_FocalAx025Gx100_GloNoW_Shift00Drop05',
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


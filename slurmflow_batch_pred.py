import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy


# model_lib = ['Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250SigmoidFocGapAx025Gx100_GloWReg_MixerHori01Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250SigmoidFocGapAx025Gx100_GloWReg_MixerHori10Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250SigmoidFocGapAx025Gx100_GloL1Reg_MixerHori01Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250SigmoidFocGapAx025Gx100_GloL1Reg_MixerHori10Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250SigmoidFocGapAx025Gx100Entropy_GloWReg_MixerHori01Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250SigmoidFocGapAx025Gx100Entropy_GloWReg_MixerHori10Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250SigmoidFocGapAx025Gx100EntropyTMSE_GloWReg_MixerHori01Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250SigmoidFocGapAx025Gx100EntropyTMSE_GloWReg_MixerHori10Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx100_GloWReg_MixerHori01Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx100_GloWReg_MixerHori10Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx100_GloL1Reg_MixerHori01Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx100_GloL1Reg_MixerHori10Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx100Entropy_GloWReg_MixerHori01Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx100Entropy_GloWReg_MixerHori10Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx100EntropyTMSE_GloWReg_MixerHori01Loss012Shift00',
#              'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx100EntropyTMSE_GloWReg_MixerHori10Loss012Shift00',
# ]


model_lib = ['Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FixSigmoidFocGapAx025Gx100_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FixSigmoidFocGapAx025Gx100_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FixSigmoidFocGapAx025Gx100_GloL1Reg_MixerHori01Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FixSigmoidFocGapAx025Gx100_GloL1Reg_MixerHori10Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FixSigmoidFocGapAx025Gx100Entropy_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FixSigmoidFocGapAx025Gx100Entropy_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FixSigmoidFocGapAx025Gx100EntropyTMSE_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S50_Samp1250FixSigmoidFocGapAx025Gx100EntropyTMSE_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FixSigmoidFocGapAx025Gx100_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FixSigmoidFocGapAx025Gx100_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FixSigmoidFocGapAx025Gx100_GloL1Reg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FixSigmoidFocGapAx025Gx100_GloL1Reg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FixSigmoidFocGapAx025Gx100Entropy_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FixSigmoidFocGapAx025Gx100Entropy_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FixSigmoidFocGapAx025Gx100EntropyTMSE_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FixSigmoidFocGapAx025Gx100EntropyTMSE_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalGapAx025Gx200_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalGapAx025Gx200_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalGapAx025Gx200_GloL1Reg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalGapAx025Gx200_GloL1Reg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalGapAx025Gx200Entropy_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalGapAx025Gx200Entropy_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalGapAx025Gx200EntropyTMSE_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalGapAx025Gx200EntropyTMSE_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200_GloWReg_MixerHori05Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200_GloWReg_MixerHori20Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200_GloL1Reg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200_GloL1Reg_MixerHori05Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200_GloL1Reg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200_GloL1Reg_MixerHori20Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori05Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori20Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200EntropyTMSE_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200EntropyTMSE_GloWReg_MixerHori05Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200EntropyTMSE_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S25_Samp1250FocalAx025Gx200EntropyTMSE_GloWReg_MixerHori20Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx200Entropy_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx200Entropy_GloWReg_MixerHori05Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx200Entropy_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx200Entropy_GloWReg_MixerHori20Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori05Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori20Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalGapAx025Gx200Entropy_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalGapAx025Gx200Entropy_GloWReg_MixerHori05Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalGapAx025Gx200Entropy_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalGapAx025Gx200Entropy_GloWReg_MixerHori20Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori05Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalAx025Gx200Entropy_GloWReg_MixerHori20Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx200EntropyL1Reg_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx200EntropyL1Reg_GloWReg_MixerHori05Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx200EntropyL1Reg_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalGapAx025Gx200EntropyL1Reg_GloWReg_MixerHori20Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalAx025Gx200EntropyL1Reg_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalAx025Gx200EntropyL1Reg_GloWReg_MixerHori05Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalAx025Gx200EntropyL1Reg_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T50_D4_N32_L3_E300_B32_S50_Samp1250FocalAx025Gx200EntropyL1Reg_GloWReg_MixerHori20Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalGapAx025Gx200EntropyL1Reg_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalGapAx025Gx200EntropyL1Reg_GloWReg_MixerHori05Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalGapAx025Gx200EntropyL1Reg_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalGapAx025Gx200EntropyL1Reg_GloWReg_MixerHori20Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalAx025Gx200EntropyL1Reg_GloWReg_MixerHori01Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalAx025Gx200EntropyL1Reg_GloWReg_MixerHori05Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalAx025Gx200EntropyL1Reg_GloWReg_MixerHori10Loss012Shift00',
             'Base_K4_T100_D4_N32_L3_E300_B32_S100_Samp1250FocalAx025Gx200EntropyL1Reg_GloWReg_MixerHori20Loss012Shift00',
             ]

n=0
for im, model in enumerate(model_lib):
    # subprocess.call(['python', 'pred.py', '--mode', 'export', '--model', model])

    for iv in [0,1,2]:#range(3):#range(1,3):
        n+=1
        # print(n)
        # subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model, '--val', str(iv)])
        # subprocess.call(['python', 'pred.py', '--mode', 'export', '--model', model])
        subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(iv)])
        # pr = '/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/horis_val{0}_{1}_sf2500.npy'.format(iv, model)
        # # pdb.set_trace()
        # if not path.exists(pr):
        #     print('submitting job for model: ', model, im, iv)
        #     subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(iv)])
        #     # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
        #     # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
        # else:
        #     print('model already predicted: ', model, im, iv)
        # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
        # subprocess.call(['python', 'pred.py', '--mode', 'predictPlot', '--model', model])
        # subprocess.call(['sbatch', 'gpu_batch_107inference.sh', model])



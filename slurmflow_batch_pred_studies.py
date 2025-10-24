import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy


tag = 'tripletOnlyClean2500' # FiltL, FiltH, FiltM, SingleCh
# tag = 'tripletOnlyLatents2500' # FiltL, FiltH, FiltM, SingleCh
# tag = 'mixerOnly' # FiltL, FiltH, FiltM, SingleCh
# tag = 'mixerHori' # FiltL, FiltH, FiltM, SingleCh
# model_lib = [1,10]#,16,165,89,186,3,11,23,150,142,2,148
# model_lib = [3,2,4]# FiltL
# model_lib = [99,113,20,123,122] #FiltH
# model_lib = [94,81,44,71,95,89,66,34,50] #latency
# model_lib = [137,150,135,123,107,108]
# model_lib = [131, 63, 52, 128] #only
# model_lib = [127,159,5,169,147] #hori
# model_lib = [222,258,196,246,235]# hori
# model_lib = [222,258,196,246,235]# hori
# model_lib = []
# model_lib = [193,275,333,254] # only
# model_lib = [286,302,303,278] # hori
# model_lib = [254, 275, 283, 319, 428, 437, 443, 504, 521] # only

# model_lib = [1022, 426,717,829,867,1033, 623, 946, 554, 1194] # latents
# model_lib = [926, 398, 597, 377, 438] # latentsTuned
model_lib = [1383,1392,1417,1440,1450,1530,1550,1559,1594,1601,1624,1633,1723]
n=0
for im, model in enumerate(model_lib):
    # subprocess.call(['python', 'pred.py', '--mode', 'export', '--model', model])

    for iv in [0]:#,1,2]:#range(3):#range(1,3):
        n+=1
        # print(n)
        # subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model, '--val', str(iv)])
        # subprocess.call(['python', 'pred.py', '--mode', 'embedding', '--model', model, '--val', str(iv)])
        # subprocess.call(['python', 'pred.py', '--mode', 'export', '--model', model])
        # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(iv)])
        model_name = 'Tune_'+'{0}_'.format(model)
        print('submitting job for model: ', model_name, im, iv)
        # subprocess.call(['python', 'pred.py', '--mode', 'embedding', '--model', model_name, '--val', str(iv), '--tag', tag])
        subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model_name, '--val', str(iv), '--tag', tag])
        # subprocess.call(['python', 'pred.py', '--mode', 'export', '--model', model_name, '--val', str(iv), '--tag', tag])

        # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model_name, str(iv)])
    
        # # pr = '/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/horis_val{0}_{1}_sf1250.npy'.format(iv, model)
        # # # pdb.set_trace()
        # # if not path.exists(pr):
        # #     print('submitting job for model: ', model, im, iv)
        #     subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(iv)])
        #     # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
        #     # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
        # else:
        #     print('model already predicted: ', model, im, iv)
        # # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
        # # subprocess.call(['python', 'pred.py', '--mode', 'predictPlot', '--model', model])
        # # subprocess.call(['sbatch', 'gpu_batch_107inference.sh', model])



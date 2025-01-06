import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy


tag = 'FiltH' # FiltL, FiltH, FiltM
# model_lib = [1,10]#,16,165,89,186,3,11,23,150,142,2,148
# model_lib = [3,2,4]# FiltL
model_lib = [3,12,4] #FiltH
n=0
for im, model in enumerate(model_lib):
    # subprocess.call(['python', 'pred.py', '--mode', 'export', '--model', model])

    for iv in [0,1,2]:#,1,2]:#range(3):#range(1,3):
        n+=1
        # print(n)
        # subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model, '--val', str(iv)])
        # subprocess.call(['python', 'pred.py', '--mode', 'embedding', '--model', model, '--val', str(iv)])
        # subprocess.call(['python', 'pred.py', '--mode', 'export', '--model', model])
        # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(iv)])
        model_name = 'Tune_'+'{0}_'.format(model)
        print('submitting job for model: ', model_name, im, iv)
        subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model_name, '--val', str(iv), '--tag', tag])
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



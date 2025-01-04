import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy


# model_lib = [45, 221, 215, 204, 183, 247, 238, 263, 613, 578, 631, 125, 474, 423, 321, 293, 639, 580, 634, 458]
# model_lib = [27,32,34,31,15,26,73,19,8,23]
model_lib = [19,8,23,350,197,261,250,131,225,303,244,247,198,245,278,240,348,182,221,290,170,126]
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
        subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model_name, '--val', str(iv)])
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



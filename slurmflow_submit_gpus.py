import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 
import numpy as np

model_lib = ['predSWR_TCN_TestDilNorm']

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
        

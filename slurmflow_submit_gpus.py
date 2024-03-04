import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 
import numpy as np

model_lib = ['predSWRTest']

for model in model_lib:
    model_base = copy.deepcopy(model)
    for irep in [1]:#range(1,4):
        model_name = model_base[:-3]+'_{}_'.format(irep)
            if ijob < 3:
                subprocess.call(['sbatch', 'gpu_batch_power_GPUshort.sh', model_name])
            else:
                subprocess.call(['sbatch', 'gpu_batch_gpu.sh', model_name])
        
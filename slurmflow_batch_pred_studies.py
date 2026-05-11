import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy


tag = 'tripletOnlyV42500' # FiltL, FiltH, FiltM, SingleCh
# tag = 'tripletOnlyProxyCircleMask2500' # FiltL, FiltH, FiltM, SingleCh

model_lib = [18, 19, 20, 21, 22, 23, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 56]

n = 0

for im, model in enumerate(model_lib):
    for iv in [0]:#,1,2]:#range(3):#range(1,3):
        n+=1
        model_name = 'Tune_'+'{0}_'.format(model)
        pr_path = '/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/'
        pr = 'preds_val{0}_{1}_{2}'.format(iv, model_name, tag)
        files = [(pr in i) for i in os.listdir(pr_path)]
        if sum(files) == 0:
            print('submitting job for model: ', model_name, im, iv)
            subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model_name, '--val', str(iv), '--tag', tag])
            subprocess.call(['python', 'pred.py', '--mode', 'embedding', '--model', model_name, '--val', str(iv), '--tag', tag])
        else:
            print('model already predicted: ', model_name, im, iv)
import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 
import glob

prob_dir = '/cs/projects/OWVinckSWR/DL/predSWR/probs/'
prob_list = glob.glob(prob_dir+'*.npy')

# pdb.set_trace()

for model in prob_list:
    model_ind = model.find('preds_val0_')+len('preds_val0_')
    model_name = model[model_ind:-4]
    subprocess.call(['python', 'pred.py', '--mode', 'predictPlot', '--model', model_name])


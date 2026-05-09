import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 


for ijob in range(31885761, 31885761+60):
    subprocess.call(['scancel', str(ijob)])
    print(ijob)
        

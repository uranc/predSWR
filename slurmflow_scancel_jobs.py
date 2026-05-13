import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 


for ijob in range(32066650, 32066650+80):
    subprocess.call(['scancel', str(ijob)])
    print(ijob)
        

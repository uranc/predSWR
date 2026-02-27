import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 


for ijob in range(31406613, 31406613+50):
    subprocess.call(['scancel', str(ijob)])
    print(ijob)
        

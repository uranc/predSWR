import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 


for ijob in range(31609412, 31609412+25):
    subprocess.call(['scancel', str(ijob)])
    print(ijob)
        

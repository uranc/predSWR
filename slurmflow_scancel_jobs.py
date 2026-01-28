import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 


for ijob in range(30804609, 30804609 + 50):
    subprocess.call(['scancel', str(ijob)])
    print(ijob)
        

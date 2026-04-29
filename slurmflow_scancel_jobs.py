import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy 


for ijob in range(31877871, 31877871+40):
    subprocess.call(['scancel', str(ijob)])
    print(ijob)
        

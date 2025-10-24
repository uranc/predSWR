import subprocess
import os.path
from os import path
import shutil
import time, pdb

# tag = 'tripletOnlyShort2500'
# tag = 'tripletOnlyMPN2500'
# tag = 'tripletOnlyMixedGP2500'
# tag = 'tripletOnlyGPSeq2500'
# tag = 'tripletOnlyGPOpt2500'
# tag = 'tripletOnlyAug2500'
# tag = 'tripletOnlyAttAug2500'
tag = 'tripletOnlyRemake2500'

# Rest of script remains unchanged
ijob = -1
for model_name in range(4):
    # exp_dir = 'experiments/' + model_name
    # pr = exp_dir + '/model/'
    # if not path.exists(pr):
    #     print(exp_dir + '/model')
    #     shutil.copytree('model', exp_dir +'/model')
    #     shutil.copyfile('./pred.py', exp_dir +'/pred.py')
    #     time.sleep(0.1)
    ijob += 1
    # subprocess.call(['sbatch', 'gpu_batch_107_worker_power_test.sh', tag])
    subprocess.call(['sbatch', 'gpu_batch_107_worker_power.sh', tag])
    # subprocess.call(['sbatch', 'gpu_batch_107_worker_test.sh', tag])
    # subprocess.call(['sbatch', 'gpu_batch_107_worker_long.sh', tag])
    # subprocess.call(['sbatch', 'gpu_batch_107_worker_short.sh', tag])
    # subprocess.call(['sbatch', 'gpu_batch_107_worker_short.sh', tag])
    # subprocess.call(['sbatch', 'gpu_batch_103_worker_titan.sh', tag])
    # subprocess.call(['sbatch', 'gpu_batch_103_worker_vinck.sh', tag])
    # subprocess.call(['sbatch', 'gpu_batch_107_worker_short.sh', tag])
    # subprocess.call(['sbatch', 'gpu_batch_107_worker_test.sh', tag])
    # pdb.set_trace()
    # if ijob < 6:
    #     # subprocess.call(['sbatch', 'gpu_batch_107_worker_power.sh', tag])
    #     # subprocess.call(['sbatch', 'gpu_batch_107_worker_long.sh', tag])
    #     subprocess.call(['sbatch', 'gpu_batch_107_worker_short.sh', tag])
    #     # subprocess.call(['sbatch', 'gpu_batch_103_worker_vinck.sh', tag])
    # elif ijob < 14:
    #     # subprocess.call(['sbatch', 'gpu_batch_107_worker_power_test.sh', tag])
    #     subprocess.call(['sbatch', 'gpu_batch_107_worker_power.sh', tag])
    # elif ijob < 18:
    #     subprocess.call(['sbatch', 'gpu_batch_103_worker_vinck.sh', tag])
    # #     # subprocess.call(['sbatch', 'gpu_batch_107_worker_power_test.sh', tag])
    # # # elif ijob < 24:
    # # #     subprocess.call(['sbatch', 'gpu_batch_103_worker_titan.sh', tag])
    # else:
    #     subprocess.call(['sbatch', 'gpu_batch_103_worker_titan.sh', tag])
    # #     subprocess.call(['sbatch', 'gpu_batch_103_worker_vinck.sh', tag])
    # #     # subprocess.call(['sbatch', 'gpu_batch_107_worker_test.sh', tag])
    # #     # subprocess.call(['sbatch', 'gpu_batch_103_worker_vinck.sh', tag])
    # #     # subprocess.call(['sbatch', 'gpu_batch_107_worker_short.sh', tag])
    # #     # subprocess.call(['sbatch', 'gpu_batch_107_worker_long.sh', tag])
    # # # elif ijob < 20:
    # # #     subprocess.call(['sbatch', 'gpu_batch_103_worker_titan.sh', tag])
    # # # else:
    # # #     subprocess.call(['sbatch', 'gpu_batch_103_worker_vinck.sh', tag])
    # # # # #    # subprocess.call(['sbatch', 'gpu_batch_107_worker_long.sh', tag])
    # # # # #     # subprocess.call(['sbatch', 'gpu_batch_107_worker_short.sh', tag])
    # # #     # subprocess.call(['sbatch', 'gpu_batch_107_worker_test.sh', tag])
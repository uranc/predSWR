import subprocess
import os.path
from os import path
import pdb
import shutil
import time
import copy


tag = 'tripletOnlyProxy2500' # FiltL, FiltH, FiltM, SingleCh
# tag = 'tripletOnlyLatents2500' # FiltL, FiltH, FiltM, SingleCh
# tag = 'mixerOnly' # FiltL, FiltH, FiltM, SingleCh
# tag = 'mixerHori' # FiltL, FiltH, FiltM, SingleCh
# model_lib = [1,10]#,16,165,89,186,3,11,23,150,142,2,148
# model_lib = [3,2,4]# FiltL
# model_lib = [99,113,20,123,122] #FiltH
# model_lib = [94,81,44,71,95,89,66,34,50] #latency
# model_lib = [137,150,135,123,107,108]
# model_lib = [131, 63, 52, 128] #only
# model_lib = [127,159,5,169,147] #hori
# model_lib = [222,258,196,246,235]# hori
# model_lib = [222,258,196,246,235]# hori
# model_lib = []
# model_lib = [193,275,333,254] # only
# model_lib = [286,302,303,278] # hori
# model_lib = [254, 275, 283, 319, 428, 437, 443, 504, 521] # only

# model_lib = [1022, 426,717,829,867,1033, 623, 946, 554, 1194] # latents
# model_lib = [926, 398, 597, 377, 438] # latentsTuned
# model_lib = [1383,1392,1417,1440,1450,1530,1550,1559,1594,1601,1624,1633,1723]
# model_lib = [0, 2, 4, 7, 8, 16, 18, 22, 26, 28, 29, 32, 33, 35, 38, 39, 44, 45, 50, 55, 58, 59, 62, 65, 66, 68, 70, 74, 87, 88, 100, 105, 106, 107, 109, 111, 116, 119, 120, 121, 123, 124, 126, 136, 137, 142, 145, 147, 149, 153, 154, 155, 164, 167, 168, 206, 207, 208, 209, 210, 216, 226, 231, 234, 238]
# model_lib = [268, 272, 276, 284, 286, 292, 299, 303, 305, 306, 315, 333, 334, 338]
# model_lib = [362, 364, 368, 370, 382, 384, 385, 386, 387, 388, 389, 397, 400, 401, 403, 404, 405, 407, 415, 416, 430]
# model_lib = [ 443, 455, 457, 460, 467, 468, 474, 476, 479, 482, 489, 494]
# model_lib = [498, 503, 504, 508]
# model_lib = [2, 7, 16, 22, 26, 28, 32, 35, 44, 50, 55, 59, 88, 106, 119, 120, 121, 123, 124, 126, 142, 145, 154, 168, 206, 207, 209, 213, 216, 222, 228, 253, 254, 268, 286, 292, 299, 305, 315, 364, 368, 370, 382, 384, 385, 386, 389, 400, 403, 405, 415, 416, 430, 443, 448, 455, 457, 460, 468, 471, 474, 476, 479, 482, 487, 489, 490, 494, 496, 498, 502, 503, 511, 528, 529, 534]
model_lib = [ 668, 677, 678, 679, 680, 683, 684, 687, 690, 693, 695, 697, 701, 702, 707, 709]
n=0
for im, model in enumerate(model_lib):
    # subprocess.call(['python', 'pred.py', '--mode', 'export', '--model', model])

    for iv in [0]:#,1,2]:#range(3):#range(1,3):
        n+=1
        # print(n)
        # subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model, '--val', str(iv)])
        # subprocess.call(['python', 'pred.py', '--mode', 'embedding', '--model', model, '--val', str(iv)])
        # subprocess.call(['python', 'pred.py', '--mode', 'export', '--model', model])
        # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(iv)])
        model_name = 'Tune_'+'{0}_'.format(model)
        print('submitting job for model: ', model_name, im, iv)
        # subprocess.call(['python', 'pred.py', '--mode', 'embedding', '--model', model_name, '--val', str(iv), '--tag', tag])
        subprocess.call(['python', 'pred.py', '--mode', 'predict', '--model', model_name, '--val', str(iv), '--tag', tag])
        # subprocess.call(['python', 'pred.py', '--mode', 'export', '--model', model_name, '--val', str(iv), '--tag', tag])

        # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model_name, str(iv)])
    
        # # pr = '/mnt/hpc/projects/OWVinckSWR/DL/predSWR/probs/horis_val{0}_{1}_sf1250.npy'.format(iv, model)
        # # # pdb.set_trace()
        # # if not path.exists(pr):
        # #     print('submitting job for model: ', model, im, iv)
        #     subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(iv)])
        #     # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
        #     # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
        # else:
        #     print('model already predicted: ', model, im, iv)
        # # subprocess.call(['sbatch', 'cpu_batch_16GBXS.sh', model, str(im)])
        # # subprocess.call(['python', 'pred.py', '--mode', 'predictPlot', '--model', model])
        # # subprocess.call(['sbatch', 'gpu_batch_107inference.sh', model])



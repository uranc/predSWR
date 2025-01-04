#!/bin/bash
#SBATCH --job-name=tensorflow_gpu
#SBATCH --partition=24GB


source /mnt/hpc/slurm/uranc/anacond/bin/activate
conda activate tf_2_14
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export HDF5_USE_FILE_LOCKING=FALSE 
srun /mnt/hpc/slurm/uranc/anacond/envs/tf_2_14/bin/python /cs/projects/MWNaturalPredict/DL/predSWR/pred.py --mode predict --model $1 --val $2 

exit 0

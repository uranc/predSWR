#!/bin/bash
#SBATCH --job-name=tensorflow_gpu
#SBATCH --partition=48GBS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000


source /mnt/hpc/slurm/uranc/anacond/bin/activate
conda activate tfSWR
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export HDF5_USE_FILE_LOCKING=FALSE 
srun /mnt/hpc/slurm/uranc/anacond/envs/tfSWR/bin/python /cs/projects/MWNaturalPredict/DL/predSWR/pred.py --mode predict --model $1 --val $2 

exit 0

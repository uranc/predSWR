#!/bin/bash
   
#SBATCH --job-name=tensorflow_gpu
#SBATCH --partition=GPUshort
#SBATCH --gpus=rtxa6000:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=7000
#SBATCH --time=0-10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cem.uran@esi-frankfurt.de

source /mnt/hpc/slurm/uranc/anacond/bin/activate
conda activate tfSWR
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export HDF5_USE_FILE_LOCKING=FALSE 

srun /mnt/hpc/slurm/uranc/anacond/envs/tfSWR/bin/python /cs/projects/MWNaturalPredict/DL/predSWR/experiments/$1/pred.py --mode train --model $1

exit 0
